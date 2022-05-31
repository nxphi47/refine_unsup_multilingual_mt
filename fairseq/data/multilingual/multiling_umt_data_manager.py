
from enum import Enum
import itertools
import json
import logging
import math
import os
from collections import OrderedDict, defaultdict
from argparse import ArgumentError
from sys import prefix
import threading
from typing import List, Tuple, Union
import numpy as np
import copy
import sys

import torch
from torch.functional import Tensor
import torch.nn.functional as F
from torch.utils.data import dataset

from fairseq import utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    Dictionary,
    LanguagePairDataset,
    PrependTokenDataset,
    SampledMultiDataset,
    SampledMultiEpochDataset,
    StripTokenDataset,
    TransformEosLangPairDataset,
    TruncateDataset,
    data_utils,
    # dictionary,
    indexed_dataset,
)
from fairseq.data.fairseq_dataset import FairseqDataset
from fairseq.data.language_pair_weight_dataset import LanguagePairWeightDataset
from fairseq.data.multilingual.multilingual_utils import (
    EncoderLangtok,
    LangTokSpec,
    LangTokStyle,
    augment_dictionary,
    get_lang_from_lang_tok,
    get_lang_tok,
)
from fairseq.data.multilingual.sampled_multi_dataset import CollateFormat
from fairseq.data.round_robin_zip_datasets import RoundRobinZipDatasets
from fairseq.distributed.utils import get_global_rank, get_global_world_size
from fairseq.file_io import PathManager
from fairseq.sequence_generator import MultiPrefixSequenceGenerator
from fairseq.utils import FileContentsAction, csv_str_list, eval_str_dict, move_to_cpu, move_to_cuda
from fairseq.kmeans_db_utils import multi_layer_db_search_single, load_kmeans_database

from .multilingual_data_manager import MultilingualDatasetManager, _lang_id

try:
    import editdistance
    EDITDISTANCE_AVAILABLE = True
except Exception as e:
    print(f'ERROR IMPORTing editdistance: try `pip install editdistance`')
    EDITDISTANCE_AVAILABLE = False

logger = logging.getLogger(__name__)

SRC_DICT_NAME = 'src'
TGT_DICT_NAME = 'tgt'


class DataDomainSpec(Enum):
    main = "MMT"
    bt = "MBT"
    ct = "MCT"

    @classmethod
    def convert_key(cls, name):
        if name == cls.main.value:
            return "mt"
        elif name == cls.bt.value:
            return "bt"
        elif name == cls.ct.value:
            return "ct"
        else:
            raise ValueError


def para_scores_to_weights(args, scores):
    scores_to_weights = args.scores_to_weights
    params = [float(x) for x in args.scores_to_weights_params.split(",")]
    logger.info(f'scores_to_weights params: {params}')
    if scores_to_weights is None:
        return None
    if scores_to_weights == 'scale_min_max':
        logger.warning(
            f'WARNING: scale_min_max for positive similarity correlation (higher more similar), like cosine_sim, '
            f'for distance score, use --scores_to_weights neg_scale_min_max')
        _min = params[0] if len(params) >= 1 else 0.0
        _max = params[1] if len(params) >= 2 else 1.0
        scores = np.array(scores)
        weights = (scores - scores.min()) / (scores.max() - scores.min()) * (_max - _min) + _min
    elif scores_to_weights == 'neg_scale_min_max':
        logger.warning(
            f'WARNING: neg_scale_min_max for negative similarity correlation (higher more similar), like distances, '
            f'for cosine_sim score, use --scores_to_weights scale_min_max')
        scores = -np.array(scores)
        weights = (scores - scores.min()) / (scores.max() - scores.min())
    elif scores_to_weights == 'scale_min_max_old':
        scores = np.array(scores)
        weights = (scores - scores.min()) / (scores.max() - scores.min())
    elif scores_to_weights == 'ones':
        weights = np.ones(shape=(len(scores,)))
    elif scores_to_weights == "uniform_rank":
        _min = params[0] if len(params) >= 1 else 0.0
        _max = params[1] if len(params) >= 2 else 1.0
        incr = (_max - _min) / float(len(scores))
        weights = [0] * len(scores)
        scores = np.array(scores)
        for i, idx in enumerate(np.argsort(scores)):
            weights[idx] = _min + (i + 1) * incr
        weights = np.array(weights)
    else:
        raise ValueError(f'{scores_to_weights} invalid')
    return weights


class MultilingualUmtDatasetManager(MultilingualDatasetManager):
    """
    Specifying for unsupervised MT with online back-translation as well
    """
    def __init__(self, args, lang_pairs, langs, dicts, sampling_method):
        super().__init__(args, lang_pairs, langs, dicts, sampling_method)
        self._bt_langs = None
        self._bt_direction_dict = None
        self._ct_langs = None
        self._ct_direction_dict = None
        self.SHOW_SAMPLES_INTERVAL = args.show_interval
        # Start by showing samples
        self._show_samples_ctr = self.SHOW_SAMPLES_INTERVAL
        self.SHOW_SAMPLES_NUMBER = args.show_samples_number
        self.SHOW_SAMPLES_TOKEN_INDICES = args.show_samples_token_indices

        if isinstance(self.args.bt_directions, str):
            self.args.bt_directions = self.args.bt_directions.split(",")
            
        if isinstance(self.args.ct_directions, str):
            assert not self.args.no_main_data, f"{self.args.ct_directions=} but no_main_data"
            self.args.ct_directions = self.args.ct_directions.split(",")
        
    @classmethod
    def setup_data_manager(cls, args, lang_pairs, langs, dicts, sampling_method):
        return cls(
            args, lang_pairs, langs, dicts, sampling_method
        )
    
    @staticmethod
    def add_args(parser):
        MultilingualDatasetManager.add_args(parser)
        parser.add_argument(
            "--bt-data",
            help='path to bt data',
            type=str,
            default=None,
        )
        parser.add_argument(
            "--bt-lang-pairs",
            help='a dictionary of data name to the language pairs they serve, \
                            e.g. {"bt":  "en-en,fr-fr,de-de"}',
            type=lambda uf: eval_str_dict(uf, type=str),
            default=None,
        )
        parser.add_argument(
            '--bt-directions', default=None, metavar='PAIRS',
            help='comma-separated list of language directions for BT: \
                    e.g: en-de,de-en,en-fr,fr-en,de-fr,fr-de',
            action=FileContentsAction
        )
        parser.add_argument(
            '--ct-directions', default=None, metavar='PAIRS',
            help='comma-separated list of language directions for Cross-translation: \
                    e.g: en-de,de-en,en-fr,fr-en,de-fr,fr-de \
                    search from any target -> source language    ',
            action=FileContentsAction
        )
        parser.add_argument('--show-interval', default=1000, type=int,
                            help='update to show interval')
        parser.add_argument('--show-samples-number', default=10, type=int,
                            help='logs sample numbers show-samples-number')
        parser.add_argument('--show-samples-token-indices', action="store_true", default=False,
                            help='logs samples with token indices')

        parser.add_argument(
            "--no-main-data",
            action="store_true",
            help="use no main data",
        )
        
        parser.add_argument(
            "--use-main-weights",
            action="store_true",
            help="Use main weights, index file: index.ne_NP-en_XX.pth",
        )
        
        parser.add_argument(
            "--force-use-main-weights",
            action="store_true",
            help="Force use main weight, will raise eror",
        )

        parser.add_argument(
            "--weight-distance",
            type=str, default='dist_per_len',
            help="Force use main weight, will raise eror",
        )
        parser.add_argument('--scores-to-weights', type=str, 
                            # default='scale_min_max',
                            default=None,
                            help='type of scoring alignment scores of mined data to loss weights')

        parser.add_argument('--scores-to-weights-params', type=str, default='0,1',
                            help='params for scores2weights')

        parser.add_argument('--multi-bt-mandatory-lang', type=str, default=None,
                            help='mandatory-lang for multilingual BT')
    
    @property
    def lang_tok_replacing_bos_eos(self):
        return self.args.lang_tok_replacing_bos_eos

    @property
    def left_pad_source(self):
        return self.args.left_pad_source
    
    @property
    def multi_bt_mandatory_lang(self):
        return self.args.multi_bt_mandatory_lang
    
    def convert_generated_hypos_to_tokens(self, generated, get_src_sample=False, get_tgt_sample=False, 
        smp_example=None, to_cuda=False) -> Union[list, dict]:
        dictionary = self.main_dictionary
        left_pad_source = self.args.left_pad_source

        prefix_removal = int(not self.args.lang_tok_replacing_bos_eos)
        generated_tokens = [gn[0]['tokens'][prefix_removal:] for gn in generated]
        if not (get_src_sample or get_tgt_sample):
            if to_cuda:
                generated_tokens = move_to_cuda(generated_tokens)
            return generated_tokens
        else:
            assert not (get_src_sample and get_tgt_sample)
            assert get_src_sample, f'only currently support get_src_sample'
            if get_src_sample:
                assert smp_example is not None
                max_length = max(len(x) for x in generated_tokens)
                src_tokens = torch.empty(size=(len(generated_tokens), max_length), dtype=smp_example['net_input']["src_tokens"].dtype)
                src_lengths = torch.empty(len(generated_tokens), dtype=smp_example['net_input']["src_lengths"].dtype)
                for i, stok in enumerate(generated_tokens):
                    tok_size = stok.size(0)
                    padding_needed = max_length - stok.size(0)
                    if left_pad_source:
                        tokens = F.pad(stok, (padding_needed, 0), value=dictionary.pad())
                    else:
                        tokens = F.pad(stok, (0, padding_needed), value=dictionary.pad())
                    src_tokens[i] = tokens
                    src_lengths[i] = tok_size
                
                gen_smp = {
                    "net_input": {"src_tokens": src_tokens, 'src_lengths': src_lengths}
                }
            if to_cuda:
                gen_smp = move_to_cuda(gen_smp)
            return gen_smp
    
    @classmethod
    def load_all_dictionaries(cls, args, language_list, load_dictionary, training):
        dicts = OrderedDict()
        if args.source_dict is not None:
            dicts[SRC_DICT_NAME] = load_dictionary(args.source_dict)
        if args.target_dict is not None:
            dicts[TGT_DICT_NAME] = load_dictionary(args.target_dict)

        if training:
            extra_lang_pairs = (
                list(
                    {p for _, v in args.extra_lang_pairs.items() for p in v.split(",")}
                )
                if args.extra_lang_pairs
                else []
            )
            extra_lang_pairs = extra_lang_pairs + (
                list(
                    {p for _, v in args.bt_lang_pairs.items() for p in v.split(",")}
                )
                if args.bt_lang_pairs
                else []
            )
            src_langs_to_load_dicts = sorted(
                {p.split("-")[0] for p in (args.lang_pairs + extra_lang_pairs)}
            )
            tgt_langs_to_load_dicts = sorted(
                {p.split("-")[1] for p in (args.lang_pairs + extra_lang_pairs)}
            )
        else:
            src_langs_to_load_dicts = [args.source_lang]
            tgt_langs_to_load_dicts = [args.target_lang]

        if args.no_main_data:
            data_path = args.bt_data
        else:
            data_path = args.data
        paths = utils.split_paths(data_path)
        assert len(paths) > 0

        def load_dicts(langs_to_load_dicts):
            for lang in langs_to_load_dicts:
                dicts[lang] = load_dictionary(
                    os.path.join(paths[0], "dict.{}.txt".format(lang))
                )
            if len(dicts) > 0:
                dict0 = next(iter(dicts.values()))
                assert dicts[lang].pad() == dict0.pad()
                assert dicts[lang].eos() == dict0.eos()
                assert dicts[lang].unk() == dict0.unk()
            logger.info("[{}] dictionary: {} types".format(lang, len(dicts[lang])))

        if args.fixed_dictionary is not None:
            fixed_dict = load_dictionary(args.fixed_dictionary)
            dicts = {lang: fixed_dict for lang in src_langs_to_load_dicts + tgt_langs_to_load_dicts}
        else:
            if args.source_dict is None:
                load_dicts(src_langs_to_load_dicts)
            if args.target_dict is None:
                load_dicts(tgt_langs_to_load_dicts)
        return dicts
    
    @classmethod
    def prepare(cls, load_dictionary, args, **kargs):
        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)

        if not hasattr(args, "shuffle_instance"):
            args.shuffle_instance = False
        if args.langtoks is None:
            args.langtoks = {}
        if "main" not in args.langtoks:
            src_langtok_spec = args.encoder_langtok if args.encoder_langtok else None
            tgt_langtok_spec = "tgt" if args.decoder_langtok else None
            args.langtoks["main"] = (src_langtok_spec, tgt_langtok_spec)
            
        if "bt" not in args.langtoks:
            src_langtok_spec = args.encoder_langtok if args.encoder_langtok else None
            tgt_langtok_spec = "tgt" if args.decoder_langtok else None
            args.langtoks["bt"] = (src_langtok_spec, tgt_langtok_spec)

        def check_langs(langs, pairs):
            messages = []
            for src, tgt in pairs:
                if src not in langs or tgt not in langs:
                    messages.append(
                        f"language pair {src}-{tgt} contains languages "
                        "that are not in the language dictionary"
                    )
            if len(messages) > 0:
                raise ValueError(" ".join(messages) + f"; langs: {langs}")

        if args.lang_pairs is None:
            raise ValueError(
                "--lang-pairs is required. List all the language pairs in the training objective."
            )
        if isinstance(args.lang_pairs, str):
            args.lang_pairs = args.lang_pairs.split(",")
        if args.source_lang is not None or args.target_lang is not None:
            training = False
        else:
            training = True
        language_list = cls.load_langs(args, **kargs)
        check_langs(
            language_list,
            (
                [p.split("-") for p in args.lang_pairs]
                if training
                else [(args.source_lang, args.target_lang)]
            ),
        )

        def load_dictionary_and_postproc(path):
            d = load_dictionary(path)
            augment_dictionary(
                dictionary=d,
                language_list=language_list,
                lang_tok_style=args.lang_tok_style,
                langtoks_specs=args.langtoks_specs,
                extra_data=args.extra_data,
            )
            return d

        dicts = cls.load_all_dictionaries(args, language_list, load_dictionary_and_postproc, training)
        return language_list, dicts, training
    
    @property
    def main_dictionary(self):
        return self.dicts[self.lang_pairs[0].split('-')[0]]
    
    @property
    def bt_langs(self):
        if self._bt_langs is None:
            self._bt_langs = [x.split('-')[0] for x in self.args.bt_directions]
        return self._bt_langs
    
    @property
    def bt_direction_dict(self):
        if self._bt_direction_dict is None:
            self._bt_direction_dict = {}
            logger.info(f'{self.args.bt_directions=}')
            for pair in self.args.bt_directions:
                s, t = pair.split('-')
                # assert s != t, f'{s} == {t} invalid'
                if s == t:
                    logger.warning(f'WARNING ! WARNIGN !: {self.args.bt_directions=} same pair')
                if s not in self._bt_direction_dict:
                    self._bt_direction_dict[s] = [t]
                else:
                    self._bt_direction_dict[s].append(t)
            logger.info(f'{self._bt_direction_dict=}')
        return self._bt_direction_dict
    
    @property
    def ct_langs(self):
        if self._ct_langs is None and self.args.ct_directions is not None:
            self._ct_langs = [x.split('-')[0] for x in self.args.ct_directions]
        return self._ct_langs
    
    @property
    def ct_direction_dict(self):
        if self._ct_direction_dict is None and self.args.ct_directions is not None:
            self._ct_direction_dict = {}
            logger.info(f'{self.args.ct_directions=}')
            for pair in self.args.ct_directions:
                s, t = pair.split('-')
                assert s != t, f'{s} == {t} invalid'
                if t not in self._ct_direction_dict:
                    self._ct_direction_dict[t] = [s]
                else:
                    self._ct_direction_dict[t].append(s)
            logger.info(f'{self._ct_direction_dict=}')
        return self._ct_direction_dict

    def get_lang_tok(self, lang, spec=LangTokSpec.main.value):
        return get_lang_tok(lang, self.args.lang_tok_style, spec=spec)
    
    def get_other_lang(self, lang, direction_dict=None):
        # TODO: allow more complex mapping
        direction_dict = direction_dict or self.bt_direction_dict
        return direction_dict[lang][np.random.randint(0, len(direction_dict[lang]))]
    
    def get_other_unique_langs(self, num, lang, mandatory_lang: str = None, direction_dict=None):
        direction_dict = direction_dict or self.bt_direction_dict
        target_list = direction_dict[lang]
        assert len(target_list) >= num, f'{len(target_list)=}'
        if lang == mandatory_lang or mandatory_lang is None:
            target_langs = [target_list[i] for i in np.random.permutation(len(target_list))[:num]]
        else:
            fil_target_list = [x for x in target_list if x != mandatory_lang]
            target_langs = [mandatory_lang] + [fil_target_list[i] for i in np.random.permutation(len(fil_target_list))[:num - 1]]
            target_langs = [target_langs[i] for i in np.random.permutation(len(target_langs))]
        return target_langs
    
    def get_other_lang_idx(self, lang_idx, dictionary, spec=LangTokSpec.main.value, direction_dict=None):
        lang_tok = dictionary[lang_idx]
        lang = get_lang_from_lang_tok(lang_tok, lang_tok_style=self.args.lang_tok_style, spec=spec)
        other_lang = self.get_other_lang(lang, direction_dict)
        other_lang_tok = get_lang_tok(other_lang, lang_tok_style=self.args.lang_tok_style, spec=spec)
        other_lang_id = _lang_id(dictionary, other_lang_tok)
        return other_lang_id
    
    def get_other_lang_indices(self, num, lang_idx, dictionary, mandatory_lang_idx: int = None, spec=LangTokSpec.main.value, direction_dict=None):
        lang_tok = dictionary[lang_idx]
        lang = get_lang_from_lang_tok(lang_tok, lang_tok_style=self.args.lang_tok_style, spec=spec)
        mandatory_lang = get_lang_from_lang_tok(
            dictionary[mandatory_lang_idx], 
            lang_tok_style=self.args.lang_tok_style, spec=spec
        ) if mandatory_lang_idx is not None else None
        other_langs = self.get_other_unique_langs(
            num, lang, 
            mandatory_lang=mandatory_lang,
            direction_dict=direction_dict)
        other_lang_indices = [
            _lang_id(dictionary, get_lang_tok(l, lang_tok_style=self.args.lang_tok_style, spec=spec))
            for l in other_langs
        ]
        return other_lang_indices
    
    def load_mono_lang_dataset(
        self,
        data_path,
        split,
        src,
        src_dict,
        combine,
        dataset_impl,
        upsample_primary,
        max_source_positions,
        prepend_bos=False,
        load_alignments=False,
        truncate_source=False,
    ):
        src_datasets = []
        tgt = src

        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else "")

            # infer langcode
            if self.split_exists(split_k, src, tgt, src, data_path, dataset_impl):
                prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
            elif self.split_exists(split_k, tgt, src, src, data_path, dataset_impl):
                prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
            else:
                if k > 0:
                    break
                else:
                    logger.error(
                        f"Dataset not found: {data_path}, {split_k}, {src}, {tgt}"
                    )
                    raise FileNotFoundError(
                        "Dataset not found: {} ({})".format(split, data_path)
                    )

            src_dataset = self.load_data(prefix + src, src_dict, dataset_impl)
            if truncate_source:
                src_dataset = AppendTokenDataset(
                    TruncateDataset(
                        StripTokenDataset(src_dataset, src_dict.eos()),
                        max_source_positions - 1,
                    ),
                    src_dict.eos(),
                )
            src_datasets.append(src_dataset)

            logger.info(
                "{} {} {}-{} {} examples".format(
                    data_path, split_k, src, tgt, len(src_datasets[-1])
                )
            )

            if not combine:
                break

        if len(src_datasets) == 1:
            src_dataset = src_datasets[0]
        else:
            sample_ratios = [1] * len(src_datasets)
            sample_ratios[0] = upsample_primary
            src_dataset = ConcatDataset(src_datasets, sample_ratios)

        if prepend_bos:
            assert hasattr(src_dict, "bos_index")
            src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())

        return src_dataset

    def load_bt_langpair_dataset(
        self,
        data_path,
        split,
        src,
        src_dict,
        tgt,
        tgt_dict,
        combine,
        dataset_impl,
        upsample_primary,
        left_pad_source,
        left_pad_target,
        max_source_positions,
        max_target_positions,
        prepend_bos=False,
        load_alignments=False,
        truncate_source=False,
        src_dataset_transform_func=lambda dataset: dataset,
        tgt_dataset_transform_func=lambda dataset: dataset,
        src_lang_id=None,
        tgt_lang_id=None,
        langpairs_sharing_datasets=None,
    ):
        # tgt = src
        assert src == tgt
        norm_direction = "bt-" + "-".join(sorted([src, tgt]))
        if langpairs_sharing_datasets is not None:
            src_dataset = langpairs_sharing_datasets.get(
                (data_path, split, norm_direction, src), "NotInCache"
            )
            tgt_dataset = langpairs_sharing_datasets.get(
                (data_path, split, norm_direction, src, "tgt"), "NotInCache"
            )
        if (
            langpairs_sharing_datasets is None
            or src_dataset == "NotInCache"
            or split != getattr(self.args, "train_subset", None)
        ):
            # source and target datasets can be reused in reversed directions to save memory
            # reversed directions of valid and test data will not share source and target datasets
            src_dataset = self.load_mono_lang_dataset(
                data_path,
                split,
                src,
                src_dict,
                combine,
                dataset_impl,
                upsample_primary,
                max_source_positions=max_source_positions,
                prepend_bos=prepend_bos,
                load_alignments=load_alignments,
                truncate_source=truncate_source,
            )
            src_dataset = src_dataset_transform_func(src_dataset)
            tgt_dataset = tgt_dataset_transform_func(src_dataset)
            if langpairs_sharing_datasets is not None:
                langpairs_sharing_datasets[
                    (data_path, split, norm_direction, src)
                ] = src_dataset
                
                langpairs_sharing_datasets[
                    (data_path, split, norm_direction, src, "tgt")
                ] = tgt_dataset

        else:
            logger.info(
                f"Reusing source for BT datasets of [{split}] {tgt}-{src} for reversed direction: "
                f"[{split}] {src}-{tgt}: src length={len(src_dataset)}; "
            )
        
        # NOTE: prepare mono_dataset_bt
        mono_dataset_bt = LanguagePairDataset(
            src_dataset,
            src_dataset.sizes,
            src_dict,
            # tgt side
            tgt_dataset,
            tgt_dataset.sizes if tgt_dataset is not None else None,
            src_dict,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            # align_dataset=align_dataset,
            src_lang_id=src_lang_id,
            tgt_lang_id=tgt_lang_id,
        )
        return mono_dataset_bt
    
    def maybe_get_parallel_weights(self, src, tgt, data_path):
        use_main_weights = self.args.use_main_weights
        if not use_main_weights:
            return None
        
        force_use_main_weights = self.args.force_use_main_weights
        index_file = os.path.join(data_path, f'index.{src}-{tgt}.pth')
        rev_index_file = os.path.join(data_path, f'index.{tgt}-{src}.pth')
        if PathManager.exists(index_file) or PathManager.exists(rev_index_file):
            _index_f = index_file if PathManager.exists(index_file) else rev_index_file
            logger.info(f'Load parallel index {_index_f}')
            pth = torch.load(_index_f)
            if "scores" in pth:
                scores = pth['scores']
            else:
                # get scores from itable['info'][2]
                info = pth['info']
                src_lengths = info[:, 0]
                search_lengths = info[:, 1]
                search_distance = info[:, 2]

                if self.args.weight_distance == "dist_per_len":
                    scores = search_distance.float() / ((src_lengths + search_lengths).float() / 2)
                else:
                    scores = search_distance.float()

            weights = para_scores_to_weights(self.args, scores)
            return weights
        else:
            assert not force_use_main_weights, f'force-mode: index file {index_file}/{rev_index_file} not exists!'
            return None
    
    def load_langpair_dataset(
        self,
        data_path,
        split,
        src,
        src_dict,
        tgt,
        tgt_dict,
        combine,
        dataset_impl,
        upsample_primary,
        left_pad_source,
        left_pad_target,
        max_source_positions,
        max_target_positions,
        prepend_bos=False,
        load_alignments=False,
        truncate_source=False,
        src_dataset_transform_func=lambda dataset: dataset,
        tgt_dataset_transform_func=lambda dataset: dataset,
        src_lang_id=None,
        tgt_lang_id=None,
        langpairs_sharing_datasets=None,
        pair_score=False,
    ):
        norm_direction = "-".join(sorted([src, tgt]))
        if langpairs_sharing_datasets is not None:
            src_dataset = langpairs_sharing_datasets.get(
                (data_path, split, norm_direction, src), "NotInCache"
            )
            tgt_dataset = langpairs_sharing_datasets.get(
                (data_path, split, norm_direction, tgt), "NotInCache"
            )
            align_dataset = langpairs_sharing_datasets.get(
                (data_path, split, norm_direction, src, tgt), "NotInCache"
            )

        if split == getattr(self.args, "train_subset", None):
            weights = self.maybe_get_parallel_weights(src, tgt, data_path)
        else:
            weights = None
        # a hack: any one is not in cache, we need to reload them
        if (
            langpairs_sharing_datasets is None
            or src_dataset == "NotInCache"
            or tgt_dataset == "NotInCache"
            or align_dataset == "NotInCache"
            or split != getattr(self.args, "train_subset", None)
        ):
            # source and target datasets can be reused in reversed directions to save memory
            # reversed directions of valid and test data will not share source and target datasets
            src_dataset, tgt_dataset, align_dataset = self.load_lang_dataset(
                data_path,
                split,
                src,
                src_dict,
                tgt,
                tgt_dict,
                combine,
                dataset_impl,
                upsample_primary,
                max_source_positions=max_source_positions,
                prepend_bos=prepend_bos,
                load_alignments=load_alignments,
                truncate_source=truncate_source,
            )
            src_dataset = src_dataset_transform_func(src_dataset)
            tgt_dataset = tgt_dataset_transform_func(tgt_dataset)
            if langpairs_sharing_datasets is not None:
                langpairs_sharing_datasets[
                    (data_path, split, norm_direction, src)
                ] = src_dataset
                langpairs_sharing_datasets[
                    (data_path, split, norm_direction, tgt)
                ] = tgt_dataset
                langpairs_sharing_datasets[
                    (data_path, split, norm_direction, src, tgt)
                ] = align_dataset
                if align_dataset is None:
                    # no align data so flag the reverse direction as well in sharing
                    langpairs_sharing_datasets[
                        (data_path, split, norm_direction, tgt, src)
                    ] = align_dataset
        else:
            logger.info(
                f"Reusing source and target datasets of [{split}] {tgt}-{src} for reversed direction: "
                f"[{split}] {src}-{tgt}: src length={len(src_dataset)}; tgt length={len(tgt_dataset)}"
            )
        
        if weights is not None:
            return LanguagePairWeightDataset(
                weights,
                src_dataset,
                src_dataset.sizes,
                src_dict,
                tgt_dataset,
                tgt_dataset.sizes if tgt_dataset is not None else None,
                tgt_dict,
                left_pad_source=left_pad_source,
                left_pad_target=left_pad_target,
                align_dataset=align_dataset,
                src_lang_id=src_lang_id,
                tgt_lang_id=tgt_lang_id,
            )
        else:
            return LanguagePairDataset(
                src_dataset,
                src_dataset.sizes,
                src_dict,
                tgt_dataset,
                tgt_dataset.sizes if tgt_dataset is not None else None,
                tgt_dict,
                left_pad_source=left_pad_source,
                left_pad_target=left_pad_target,
                align_dataset=align_dataset,
                src_lang_id=src_lang_id,
                tgt_lang_id=tgt_lang_id,
            )
    
    def load_a_bt_dataset(
        self,
        split,
        data_path,
        src,
        src_dict,
        combine,
        prepend_bos=False,
        langpairs_sharing_datasets=None,
        data_category=None,
        **extra_kwargs,
    ):
        tgt = src
        tgt_dict = src_dict
        dataset_impl = self.args.dataset_impl
        upsample_primary = self.args.upsample_primary
        left_pad_source = self.args.left_pad_source
        left_pad_target = self.args.left_pad_target
        max_source_positions = self.args.max_source_positions
        max_target_positions = self.args.max_target_positions
        load_alignments = self.args.load_alignments
        truncate_source = self.args.truncate_source
        src_dataset_transform_func = self.src_dataset_tranform_func
        tgt_dataset_transform_func = self.tgt_dataset_tranform_func
        enable_lang_ids = self.args.enable_lang_ids
        lang_dictionary = self.lang_dict
        src_langtok_spec, tgt_langtok_spec = extra_kwargs["langtok_spec"]

        src_langtok = self.get_encoder_langtok(src, tgt, src_langtok_spec)
        tgt_langtok = self.get_decoder_langtok(tgt, tgt_langtok_spec)
        logger.info(
            f"{data_category}:{src}-{tgt} src_langtok: {src_langtok}; tgt_langtok: {tgt_langtok}"
        )

        langpair_ds = self.load_bt_langpair_dataset(
            data_path,
            split,
            src,
            src_dict,
            tgt,
            tgt_dict,
            combine,
            dataset_impl,
            upsample_primary,
            left_pad_source,
            left_pad_target,
            max_source_positions,
            max_target_positions,
            prepend_bos,
            load_alignments,
            truncate_source,
            src_dataset_transform_func=lambda dataset: src_dataset_transform_func(
                src, tgt, dataset, src_langtok_spec
            ),
            tgt_dataset_transform_func=lambda dataset: tgt_dataset_transform_func(
                src, tgt, dataset, tgt_langtok_spec
            ),
            src_lang_id=_lang_id(lang_dictionary, src)
            if enable_lang_ids and lang_dictionary is not None
            else None,
            tgt_lang_id=_lang_id(lang_dictionary, tgt)
            if enable_lang_ids and lang_dictionary is not None
            else None,
            langpairs_sharing_datasets=langpairs_sharing_datasets,
        )
        # TODO: handle modified lang toks for mined data and dae data
        if self.args.lang_tok_replacing_bos_eos:
            ds = self.alter_dataset_langtok(
                langpair_ds,
                src_eos=self.get_source_dictionary(src).eos() if src else self.get_target_dictionary(tgt).eos(),
                src_lang=src,
                tgt_eos=self.get_target_dictionary(tgt).eos(),
                tgt_lang=tgt,
                src_langtok_spec=src_langtok_spec,
                tgt_langtok_spec=tgt_langtok_spec,
            )
        else:
            ds = langpair_ds
        return ds
    
    def get_data_paths_and_lang_pairs(self, split, include_bt=False, is_bt=False, bt_lang_pairs=None):
        bt_lang_pairs = bt_lang_pairs or self.args.bt_lang_pairs
        if is_bt:
            # assert split == getattr(self.args, "train_subset", None)
            datapaths = {"bt": self.args.bt_data}
            # lang_pairs = self.args.bt_lang_pairs
            lang_pairs = {
                k: v.split(",") for k, v in bt_lang_pairs.items()
            }
            assert "bt" in datapaths
            assert "bt" in lang_pairs
            return datapaths, lang_pairs

        no_main_data = self.args.no_main_data
        is_train = split == getattr(self.args, "train_subset", None)
        datapaths = {"main": self.args.data}
        lang_pairs = {"main": self.lang_pairs}
        if is_train:
            # only training data can have extra data and extra language pairs
            if self.args.extra_data:
                extra_datapaths = self.args.extra_data
                datapaths.update(extra_datapaths)
            if self.args.extra_lang_pairs:
                extra_lang_pairs = {
                    k: v.split(",") for k, v in self.args.extra_lang_pairs.items()
                }
                lang_pairs.update(extra_lang_pairs)
            if include_bt:
                datapaths.update({"bt": self.args.bt_data})
                bt_lang_pairs = {
                    k: v.split(",") for k, v in bt_lang_pairs.items()
                }
                lang_pairs.update(bt_lang_pairs)
        return datapaths, lang_pairs
    
    def get_split_num_data_shards(self, split, bt_lang_pairs=None):
        if split in self._num_shards_dict:
            return self._num_shards_dict[split]

        num_shards_dict = {}
        data_paths, lang_pairs = self.get_data_paths_and_lang_pairs(split, include_bt=True, bt_lang_pairs=bt_lang_pairs)
        for data_category, paths in data_paths.items():
            if data_category not in lang_pairs:
                continue
            paths = utils.split_paths(paths)
            shards_dict = self._get_shard_num_dict(split, paths)
            lang_dirs = [
                lang_pair.split("-") for lang_pair in lang_pairs[data_category]
            ]
            lang_dirs = [x if len(x) > 1 else (x[0], x[0]) for x in lang_dirs]
            for src, tgt in lang_dirs:
                key = self.get_dataset_key(data_category, src, tgt)
                if "mono_" in data_category:
                    # monolingual data requires tgt only
                    assert src is None or src == tgt, (
                        f"error: src={src}, "
                        f"tgt={tgt} for data_category={data_category}"
                    )
                    num_shards_dict[key] = shards_dict[tgt]
                elif "bt" in data_category:
                    # monolingual data for back-translation, tgt=src
                    assert src == tgt, (
                        f"error: src={src}, "
                        f"tgt={tgt} for data_category={data_category}"
                    )
                    # num_shards_dict[key] = shards_dict[tgt]
                    num_shards_dict[key] = shards_dict[f"{src}-{src}"]
                else:
                    if f"{src}-{tgt}" in shards_dict:
                        num_shards_dict[key] = shards_dict[f"{src}-{tgt}"]
                    elif f"{tgt}-{src}" in shards_dict:
                        # follow the fairseq tradition to use reversed direction data if it is not available
                        num_shards_dict[key] = shards_dict[f"{tgt}-{src}"]
        self._num_shards_dict[split] = num_shards_dict
        logger.info(f"[{split}] num of shards: {num_shards_dict}")
        return num_shards_dict
    
    def get_split_data_param_list(self, split, epoch, shard_epoch=None, is_bt=False):
        # TODO: to extend with extra datasets and keys and loop over different shard data paths
        param_list = []
        data_paths, lang_pairs = self.get_data_paths_and_lang_pairs(split, is_bt=is_bt)
        logger.info(f"langtoks settings: {self.args.langtoks}, {is_bt=}")
        split_num_shards_dict = self.get_split_num_data_shards(split)
        for data_category, paths in data_paths.items():
            if data_category not in lang_pairs:
                continue
            paths = utils.split_paths(paths)
            assert len(paths) > 0
            if len(paths) > 1:
                self._has_sharded_data = True
            if split != getattr(self.args, "train_subset", None):
                # if not training data set, use the first shard for valid and test
                paths = paths[:1]

            if data_category in self.args.langtoks:
                lang_tok_spec = self.args.langtoks[data_category]
            else:
                # default to None
                lang_tok_spec = (None, None)

            # infer langcode
            lang_dirs = [
                lang_pair.split("-") for lang_pair in lang_pairs[data_category]
            ]
            lang_dirs = [x if len(x) > 1 else (x[0], x[0]) for x in lang_dirs]
            for src, tgt in lang_dirs:
                assert src is not None or data_category == "mono_dae", (
                    f"error: src={src}, tgt={tgt} for data_category={data_category}"
                )
                key = self.get_dataset_key(data_category, src, tgt)
                data_path = self.get_split_data_path(
                    paths, epoch, shard_epoch, split_num_shards_dict[key]
                )
                param_list.append(
                    {
                        "key": key,
                        "data_path": data_path,
                        "split": split,
                        "src": src,
                        "src_dict": self.get_source_dictionary(src)
                        if src and data_category != "mono_dae"
                        else None,
                        "tgt": tgt,
                        "tgt_dict": self.get_target_dictionary(tgt),
                        "data_category": data_category,
                        "langtok_spec": lang_tok_spec,
                    }
                )
        return param_list
    
    def load_split_bt_datasets(
        self, split, training, epoch=1, combine=False, shard_epoch=None, **kwargs
    ):
        data_param_list = self.get_split_data_param_list(
            split, epoch, shard_epoch=shard_epoch, is_bt=True
        )
        langpairs_sharing_datasets = (
            {} if self.args.enable_reservsed_directions_shared_datasets else None
        )
        datasets = [
            (
                param["key"],
                self.load_a_bt_dataset(
                    combine=combine,
                    langpairs_sharing_datasets=langpairs_sharing_datasets,
                    **param,
                ),
            )
            for param in data_param_list
        ]
        return datasets, data_param_list
    
    def load_sampled_multi_epoch_bt_dataset(
        self, split, training, epoch=0, combine=False, shard_epoch=None, **kwargs
    ):
        datasets, data_param_list = self.load_split_bt_datasets(
            split, training, epoch, combine, shard_epoch=shard_epoch, **kwargs
        )
        if training and split == getattr(self.args, "train_subset", None):
            sample_ratios = self.get_sampling_ratios(data_param_list, datasets, epoch)
            return SampledMultiEpochDataset(
                OrderedDict(datasets),
                epoch=epoch,
                shard_epoch=shard_epoch,
                # valid and test datasets will be degenerate to concating datasets:
                sampling_ratios=sample_ratios,
                eval_key=None,
                collate_format=CollateFormat.single,
                virtual_size=self.args.virtual_data_size,
                split=split,
                virtual_epoch_size=self.args.virtual_epoch_size,
                # if not using lang_tok altering, simplified to use the same collater
                shared_collater=self._shared_collater(),
            )
        else:
            return self.load_into_concat_dataset(split, datasets, data_param_list)
    
    def load_sampled_multi_bt_dataset(
        self, split, training, epoch=0, combine=False, shard_epoch=None, **kwargs
    ):
        datasets, data_param_list = self.load_split_bt_datasets(
            split, training, epoch, combine, shard_epoch=shard_epoch, **kwargs
        )
        if training and split == getattr(self.args, "train_subset", None):
            sample_ratios = self.get_sampling_ratios(data_param_list, datasets, epoch)
            return SampledMultiDataset(
                OrderedDict(datasets),
                epoch=epoch,
                # valid and test datasets will be degerate to concating datasets:
                sampling_ratios=sample_ratios,
                eval_key=None,
                collate_format=CollateFormat.single,
                virtual_size=self.args.virtual_data_size,
                split=split,
                # if not using lang_tok altering, simplified to use the same collater
                shared_collater=self._shared_collater(),
            )
        else:
            return self.load_into_concat_dataset(split, datasets, data_param_list)

    def load_dataset(
        self, split, training, epoch=0, combine=False, shard_epoch=None, **kwargs
    ):
        datasets = []
        no_main_data = self.args.no_main_data
        load_mmt = not (no_main_data and training and split == getattr(self.args, "train_subset", None))
        if load_mmt:
            if self.args.virtual_epoch_size is None:
                mt_dataset = self.load_sampled_multi_dataset(
                    split, training, epoch, combine, shard_epoch, **kwargs
                )
            else:
                mt_dataset = self.load_sampled_multi_epoch_dataset(
                    split, training, epoch, combine, shard_epoch, **kwargs
                )
            datasets.append((DataDomainSpec.main.value, mt_dataset))
            if self.ct_direction_dict is not None:
                logger.warning(f"Load main as cross-translaiton data")
                datasets.append((DataDomainSpec.ct.value, mt_dataset))
        
        # load BT dataset
        # FIXME: TODO: --------------
        if training and split == getattr(self.args, "train_subset", None):
            if self.args.virtual_epoch_size is None:
                bt_dataset = self.load_sampled_multi_bt_dataset(
                    split, training, epoch, combine, shard_epoch, **kwargs
                )
            else:
                bt_dataset = self.load_sampled_multi_epoch_bt_dataset(
                    split, training, epoch, combine, shard_epoch, **kwargs
                )
            datasets.append((DataDomainSpec.bt.value, bt_dataset))
        assert len(datasets) > 0
        final_dataset = RoundRobinZipDatasets(OrderedDict(datasets))
        return final_dataset
    
    def infer_bt_src_bos_toks(self, tgt_bos_toks: torch.Tensor, spec=LangTokSpec.main.value):
        # NOTE dictionary must be shared
        dictionary = self.main_dictionary
        src_bos_toks = [self.get_other_lang_idx(lid, dictionary, spec=spec) 
            for lid in tgt_bos_toks.cpu().tolist()]
        src_bos_toks_ = torch.tensor(src_bos_toks, dtype=tgt_bos_toks.dtype, device=tgt_bos_toks.device)
        return src_bos_toks_
    
    def infer_bt_multi_src_bos_toks(self, num: int, tgt_bos_toks: torch.Tensor, mandatory_lang_idx: int, spec=LangTokSpec.main.value) -> List[torch.Tensor]:
        dictionary = self.main_dictionary
        src_bos_toks_pack = [
            self.get_other_lang_indices(num, lid, dictionary, mandatory_lang_idx=mandatory_lang_idx, spec=spec)
            for lid in tgt_bos_toks.cpu().tolist()
        ]
        src_bos_toks_list = [
            torch.tensor(
                [src_bos_toks_pack[j][i] for j in range(len(src_bos_toks_pack))],
                dtype=tgt_bos_toks.dtype, device=tgt_bos_toks.device
            )
            for i in range(num)
        ]
        return src_bos_toks_list
    
    def infer_ct_src_bos_toks(self, tgt_bos_toks: torch.Tensor, spec=LangTokSpec.main.value):
        dictionary = self.main_dictionary
        src_bos_toks = [
            self.get_other_lang_idx(lid, dictionary, spec=spec,  direction_dict=self.ct_direction_dict)
            for lid in tgt_bos_toks.cpu().tolist()
        ]
        src_bos_toks_ = torch.tensor(src_bos_toks, dtype=tgt_bos_toks.dtype, device=tgt_bos_toks.device)
        return src_bos_toks_
    
    def _configure_generated_sample(self, generated, smp):
        dictionary = self.main_dictionary
        left_pad_source = self.args.left_pad_source

        max_length = max([gn[0]["tokens"].size(0) for gn in generated])
        net_input = smp["net_input"]
        # NOTE: not adding bos token to source, rm 1 to remove lang-tok (prefix)
        prefix_removal = int(not self.args.lang_tok_replacing_bos_eos)
        max_length -= prefix_removal
        n_src_tokens = torch.empty(
            size=(len(generated), max_length), dtype=net_input["src_tokens"].dtype
        )
        n_src_lengths = torch.empty(
            len(generated), dtype=net_input["src_lengths"].dtype
        )

        for i, gn in enumerate(generated):
            tokens = gn[0]["tokens"][prefix_removal:]
            tokens_size = tokens.size(0)
            padding_needed = max_length - tokens_size
            if left_pad_source:
                tokens = F.pad(tokens, (padding_needed, 0), value=dictionary.pad())
            else:
                tokens = F.pad(tokens, (0, padding_needed), value=dictionary.pad())
            n_src_tokens[i] = tokens
            n_src_lengths[i] = tokens_size

        device = net_input["src_tokens"].device
        # This seems to be important
        del net_input["src_tokens"]
        del net_input["src_lengths"]
        net_input["src_tokens"] = n_src_tokens.to(device)
        net_input["src_lengths"] = n_src_lengths.to(device)
        return smp
        
    def backtranslate_multi_sample(self, generator, smp) -> torch.Tensor:
        """
        Back-translation with data manaer
        # importance params
            lang_tok_replacing_bos_eos=False
            left_pad_source=True, ????
            left_pad_target=False, 
            lang_tok_style='mbart', 
            langs=None, 
            langtoks=None, 
            langtoks_specs='main', 
            bilm_add_bos=False, 
            enable_lang_ids=False, 
            decoder_langtok=True, 
            encoder_langtok=None, 
        # with above params, src not have lang-tok, tgt will have lang-tok prepended
        Expected input sample:
            - smp['net_input']['src_tokens']            hello world
            - smp["net_input"]["prev_output_tokens"]    </s> <langtok> hello world
            - smp["target"]                             <langtok> hello world </s>
        Expected output
            - smp['net_input']['src_tokens']            salut lume
            - smp["net_input"]["prev_output_tokens"]    </s> <langtok> hello world
            - smp["target"]                             <langtok> hello world </s>
        """
        # _target = smp["target"]
        assert not self.args.encoder_langtok
        assert self.args.decoder_langtok
        _prev_tokens = smp["net_input"]["prev_output_tokens"]
        tgt_bos_toks = _prev_tokens[:, int(not self.lang_tok_replacing_bos_eos)]

        # FIXME: check generate, consider to use task.inference_step
        src_bos_toks = self.infer_bt_src_bos_toks(tgt_bos_toks)
        prefix_tokens = src_bos_toks.unsqueeze(1)
        generated = generator.generate(
            models=[], 
            sample=smp, 
            prefix_tokens=prefix_tokens
        )

        self._configure_generated_sample(generated, smp)
        return src_bos_toks
    
    def multi_backtranslate_multi_sample(self, generator, smp, num: int = 2):
        """
        Back-translation with data manaer
        # with above params, src not have lang-tok, tgt will have lang-tok prepended
        Expected input sample:
            - smp['net_input']['src_tokens']            hello world
            - smp["net_input"]["prev_output_tokens"]    </s> <langtok> hello world
            - smp["target"]                             <langtok> hello world </s>
        Expected output
            - smp['net_input']['src_tokens']            salut lume
            - smp["net_input"]["prev_output_tokens"]    </s> <langtok> hello world
            - smp["target"]                             <langtok> hello world </s>
        """
        assert isinstance(generator, MultiPrefixSequenceGenerator)
        assert not self.args.encoder_langtok
        assert self.args.decoder_langtok
        _prev_tokens = smp["net_input"]["prev_output_tokens"]
        tgt_bos_toks = _prev_tokens[:, int(not self.lang_tok_replacing_bos_eos)]

        # FIXME: check generate, consider to use task.inference_step
        src_bos_toks_list = self.infer_bt_multi_src_bos_toks(
            num, tgt_bos_toks, 
            mandatory_lang_idx=_lang_id(
                self.main_dictionary, 
                get_lang_tok(self.args.multi_bt_mandatory_lang, lang_tok_style=self.args.lang_tok_style)
            ) if self.args.multi_bt_mandatory_lang is not None else None
        )

        generated_list = generator.generate(
            models=[], 
            sample=smp, 
            prefix_tokens=[x.unsqueeze(1) for x in src_bos_toks_list]
        )

        smp_list = [
            self._configure_generated_sample(generated, copy.deepcopy(smp))
            for generated in generated_list
        ]

        return src_bos_toks_list, smp_list
    
    def _triangulate_samples(self, dictionary, first_smp, second_smp, second_bos_toks):
        """
        x -> y (src_bos_toks_1) => y->x
        x -> z (src_bos_toks_2) => z->x
        pair:
        y->[lz,z]
        """
        # convert second_smp/src_tokens to first_smp['target'] and 'prev_output_tokens'

        assert "alignments" not in first_smp and "align_weights" not in first_smp
        assert "constraints" not in first_smp
        output = copy.deepcopy(first_smp)
        sec_src_tokens = second_smp['net_input']['src_tokens']
        sec_src_lengths = second_smp['net_input']['src_lengths']
        sec_target = second_smp['target']
        bsz = sec_src_lengths.size(0)

        max_tgt_length = sec_src_lengths.max() + 1
        target = torch.empty(size=(bsz, max_tgt_length), dtype=sec_target.dtype, device=sec_target.device)
        prev_output_tokens = torch.empty(size=(bsz, max_tgt_length), dtype=sec_src_tokens.dtype, device=sec_src_tokens.device)
        second_bos_toks = second_bos_toks.unsqueeze(1)

        for i in range(bsz):
            if self.left_pad_source:
                tokens = sec_src_tokens[i, -sec_src_lengths[i]:]
            else:
                tokens = sec_src_tokens[i, :sec_src_lengths[i]]

            prepended_tokens = torch.cat([second_bos_toks[i], tokens], 0)
            padding_needed = (0, max_tgt_length - prepended_tokens.size(0))
            target[i] = F.pad(prepended_tokens, padding_needed, value=dictionary.pad())
            prev_output_tokens[i] = F.pad(
                torch.cat([prepended_tokens[-1:], prepended_tokens[:-1]], 0), 
                padding_needed, value=dictionary.pad()
            )
            
        output['net_input']['prev_output_tokens'] = prev_output_tokens
        output['target'] = target
        return output
    
    def multi_triangle_smp_from_smp_list(self, src_bos_toks_list, smp_list):
        """Generated single smp from list of smp_list(size=2) for triangle bt
        batch = {
            "id": id,
            "nsentences": len(samples),
            "ntokens": ntokens,
            "net_input": {
                "src_tokens": src_tokens, 
                "src_lengths": src_lengths, 
                "prev_output_tokens": prev_output_tokens
            },
            "target": target,
        }
        NOTE: target is the same, src_tokens are each target language
            -> replace target/prev_output_tokens with src_tokens of second
        first:
            - smp['net_input']['src_tokens']            salut lume
            - smp["net_input"]["prev_output_tokens"]    </s> <langtok> hello world
            - smp["target"]                             <langtok> hello world </s>
        second:
            - smp['net_input']['src_tokens']            bon jour
            - smp["net_input"]["prev_output_tokens"]    </s> <langtok> hello world
            - smp["target"]                             <langtok> hello world </s>
        output:
            ??? does src_tokens has </s>
            - smp['net_input']['src_tokens']            salut lume
            - smp["net_input"]["prev_output_tokens"]    </s> <src_bos_tok> bon jour
            - smp["target"]                             <src_bos_tok> hello world </s>
        Args:
            src_bos_toks_list: list of bos toks
            smp_list (Dict): Dict of sample tensors
        """
        assert len(smp_list) == 2, f'{len(smp_list)=} wrong or not implemented'
        dictionary = self.main_dictionary
        output = self._triangulate_samples(dictionary, smp_list[0], smp_list[1], src_bos_toks_list[1])
        output_list = [output]
        in_bos_toks_list = [src_bos_toks_list[0]]
        out_bos_toks_list = [src_bos_toks_list[1]]
        return output_list, in_bos_toks_list, out_bos_toks_list
    
    def crosstranslate_multi_sample(self, generator, smp, src_lang_toks=None) -> None:
        """
        Cross-Translation
        Given (x->y) pair of (l_x, l_y), and l_z != l_x, l_y
        Compute loss: z(x)->y
        TODO: how to get source l_x if src_tokens does not have langtok???
            l_z != l_y
            maintain a dict[l_z] -> [l_1, l_2, l_3] as potential src langs
        Expected input sample:
            - smp['net_input']['src_tokens']            hello world
            - smp["net_input"]["prev_output_tokens"]    </s> [fr_XX] bonjour
            - smp["target"]                             [fr_XX] bonjour </s>
        Expected output, l_z = ro_RO
            - smp['net_input']['src_tokens']            salut lume
            - smp["net_input"]["prev_output_tokens"]    </s> [fr_XX] bonjour
            - smp["target"]                             [fr_XX] bonjour </s>
        """
        assert not self.args.encoder_langtok
        assert self.args.decoder_langtok
        _prev_tokens = smp["net_input"]["prev_output_tokens"]
        # dictionary = self.main_dictionary
        left_pad_source = self.args.left_pad_source
        if self.args.lang_tok_replacing_bos_eos:
            tgt_bos_toks = _prev_tokens[:, 0]
        else:
            tgt_bos_toks = _prev_tokens[:, 1]

        src_bos_toks = src_lang_toks or self.infer_ct_src_bos_toks(tgt_bos_toks)
        # same as back-translation, just different in terms of infer_ct_src_bos_toks
        generated = generator.generate(
            models=[], 
            sample=smp, 
            # bos_token=src_bos_toks,
            prefix_tokens=src_bos_toks.unsqueeze(1)
        )
        self._configure_generated_sample(generated, smp)
        return src_bos_toks

    def display_samples_once_in_a_while(self, smp, src_bos_toks=None, prefix=""):
        dictionary = self.main_dictionary
        if 1 < self._show_samples_ctr < self.SHOW_SAMPLES_INTERVAL:
            self._show_samples_ctr += 1
            return
        elif self._show_samples_ctr >= self.SHOW_SAMPLES_INTERVAL:
            self._show_samples_ctr = 0
        else:
            self._show_samples_ctr += 1

        ln = smp["net_input"]["src_tokens"].shape[0]

        logger.info(
            f"(r:{self.args.distributed_rank}) :"
            f"{prefix} generated by back-translation.) {ln} samples"
        )
        bpe_symbol = "sentencepiece"
        assert src_bos_toks is None or len(src_bos_toks) == smp["net_input"]["src_tokens"].size(0)
        tgt_bos_index = int(not self.lang_tok_replacing_bos_eos)
        tgt_bos_toks = smp["net_input"]["prev_output_tokens"][:, tgt_bos_index]

        for i in range(min(ln, self.SHOW_SAMPLES_NUMBER)):
            src_tokens = smp["net_input"]["src_tokens"][i]
            tgt_tokens = smp["target"][i]
            src_lang = dictionary[src_bos_toks[i]] if src_bos_toks is not None else None
            tgt_lang = dictionary[tgt_bos_toks[i]]

            src_str = dictionary.string(utils.strip_pad(src_tokens, dictionary.pad()), bpe_symbol)
            tgt_str = dictionary.string(utils.strip_pad(tgt_tokens, dictionary.pad()), bpe_symbol)
            src_tokens_str = f"\t\t[ src tokens]  {src_tokens}\n" if self.SHOW_SAMPLES_TOKEN_INDICES else ""
            logger.info(
                f"\n{i}\t\t[{src_lang} generated]  {src_str}\n"
                f"\t\t[{tgt_lang} original ]  {tgt_str}\n" + src_tokens_str
            )


class GpuSepMultilingualUmtDatasetManager(MultilingualUmtDatasetManager):
    @staticmethod
    def add_args(parser):
        # MultilingualUmtDatasetManager.add_args(parser)
        # NOTE: task must run MultilingualUmtDatasetManager.add_args(parser) first
        parser.add_argument(
            "--rank-to-langs",
            help='rank to langauges, \
                    e.g. {0:  "en_XX-en_XX,ne_NP-ne_NP", 1: "en_XX-en_XX,si_LK-si_LK"}',
            type=lambda uf: eval_str_dict(uf, type=str),
            default=None,
        )
    
    @classmethod
    def setup_data_manager(cls, args, lang_pairs, langs, dicts, sampling_method):
        return GpuSepMultilingualUmtDatasetManager(
            args, lang_pairs, langs, dicts, sampling_method
        )

    def get_data_paths_and_lang_pairs(self, split, include_bt=False, is_bt=False, rank_langs=None, bt_lang_pairs=None):
        bt_lang_pairs = bt_lang_pairs or self.args.bt_lang_pairs
        if is_bt:
            assert split == getattr(self.args, "train_subset", None)
            datapaths = {"bt": self.args.bt_data}
            lang_pairs = {
                k: v.split(",") for k, v in bt_lang_pairs.items()
            }
            if rank_langs is not None:
                assert isinstance(rank_langs, str)
                rank_lang_list = rank_langs.split(",")
                for k, v in lang_pairs.items():
                    lang_pairs[k] = [x for x in v if x in rank_lang_list]
                logger.warning(f'get_data_paths_and_lang_pairs: {is_bt=} -> rank_langs:[{get_global_rank()}] {lang_pairs=}')
            assert "bt" in datapaths
            assert "bt" in lang_pairs
            return datapaths, lang_pairs

        no_main_data = self.args.no_main_data
        is_train = split == getattr(self.args, "train_subset", None)
        datapaths = {"main": self.args.data}
        lang_pairs = {"main": self.lang_pairs}
        if is_train:
            if rank_langs is not None:
                assert isinstance(rank_langs, str)
                lang_pairs_rank_lang_list = [x.split("-")[0] for x in rank_langs.split(",")]
                for k, v in lang_pairs.items():
                    lang_pairs[k] = [x for x in v if (x.split("-")[0] in lang_pairs_rank_lang_list and x.split("-")[1] in lang_pairs_rank_lang_list)]

            # only training data can have extra data and extra language pairs
            if self.args.extra_data:
                extra_datapaths = self.args.extra_data
                datapaths.update(extra_datapaths)
            if self.args.extra_lang_pairs:
                extra_lang_pairs = {
                    k: v.split(",") for k, v in self.args.extra_lang_pairs.items()
                }
                lang_pairs.update(extra_lang_pairs)
            if include_bt:
                datapaths.update({"bt": self.args.bt_data})
                bt_lang_pairs = {
                    k: v.split(",") for k, v in bt_lang_pairs.items()
                }
                if rank_langs is not None:
                    assert isinstance(rank_langs, str)
                    rank_lang_list = rank_langs.split(",")
                    for k, v in bt_lang_pairs.items():
                        bt_lang_pairs[k] = [x for x in v if x in rank_lang_list]
                    logger.warning(f'get_data_paths_and_lang_pairs: {is_bt=}/{is_train=} -> rank_langs:[{get_global_rank()}] {lang_pairs=}')
                    
                lang_pairs.update(bt_lang_pairs)
        return datapaths, lang_pairs
    
    def get_split_data_param_list(self, split, epoch, shard_epoch=None, is_bt=False, rank_langs=None):
        # TODO: to extend with extra datasets and keys and loop over different shard data paths
        param_list = []
        data_paths, lang_pairs = self.get_data_paths_and_lang_pairs(split, is_bt=is_bt, rank_langs=rank_langs)
        logger.info(f"langtoks settings: {self.args.langtoks}, {is_bt=}")
        split_num_shards_dict = self.get_split_num_data_shards(split, rank_langs=rank_langs)
        for data_category, paths in data_paths.items():
            if data_category not in lang_pairs:
                continue
            paths = utils.split_paths(paths)
            assert len(paths) > 0
            if len(paths) > 1:
                self._has_sharded_data = True
            if split != getattr(self.args, "train_subset", None):
                # if not training data set, use the first shard for valid and test
                paths = paths[:1]

            if data_category in self.args.langtoks:
                lang_tok_spec = self.args.langtoks[data_category]
            else:
                # default to None
                lang_tok_spec = (None, None)

            # infer langcode
            lang_dirs = [
                lang_pair.split("-") for lang_pair in lang_pairs[data_category]
            ]
            lang_dirs = [x if len(x) > 1 else (x[0], x[0]) for x in lang_dirs]
            for src, tgt in lang_dirs:
                assert src is not None or data_category == "mono_dae", (
                    f"error: src={src}, " "tgt={tgt} for data_category={data_category}"
                )
                key = self.get_dataset_key(data_category, src, tgt)
                data_path = self.get_split_data_path(
                    paths, epoch, shard_epoch, split_num_shards_dict[key]
                )
                param_list.append(
                    {
                        "key": key,
                        "data_path": data_path,
                        "split": split,
                        "src": src,
                        "src_dict": self.get_source_dictionary(src)
                        if src and data_category != "mono_dae"
                        else None,
                        "tgt": tgt,
                        "tgt_dict": self.get_target_dictionary(tgt),
                        "data_category": data_category,
                        "langtok_spec": lang_tok_spec,
                    }
                )
        return param_list
    
    def get_split_num_data_shards(self, split, bt_lang_pairs=None, rank_langs=None):
        if split in self._num_shards_dict:
            return self._num_shards_dict[split]

        num_shards_dict = {}
        data_paths, lang_pairs = self.get_data_paths_and_lang_pairs(split, include_bt=True, rank_langs=rank_langs,bt_lang_pairs=bt_lang_pairs)
        for data_category, paths in data_paths.items():
            if data_category not in lang_pairs:
                continue
            paths = utils.split_paths(paths)
            shards_dict = self._get_shard_num_dict(split, paths)
            lang_dirs = [
                lang_pair.split("-") for lang_pair in lang_pairs[data_category]
            ]
            lang_dirs = [x if len(x) > 1 else (x[0], x[0]) for x in lang_dirs]
            for src, tgt in lang_dirs:
                key = self.get_dataset_key(data_category, src, tgt)
                if "mono_" in data_category:
                    # monolingual data requires tgt only
                    assert src is None or src == tgt, (
                        f"error: src={src}, "
                        f"tgt={tgt} for data_category={data_category}"
                    )
                    num_shards_dict[key] = shards_dict[tgt]
                elif "bt" in data_category:
                    # monolingual data for back-translation, tgt=src
                    assert src == tgt, (
                        f"error: src={src}, "
                        f"tgt={tgt} for data_category={data_category}"
                    )
                    # num_shards_dict[key] = shards_dict[tgt]
                    num_shards_dict[key] = shards_dict[f"{src}-{src}"]
                else:
                    if f"{src}-{tgt}" in shards_dict:
                        num_shards_dict[key] = shards_dict[f"{src}-{tgt}"]
                    elif f"{tgt}-{src}" in shards_dict:
                        # follow the fairseq tradition to use reversed direction data if it is not available
                        num_shards_dict[key] = shards_dict[f"{tgt}-{src}"]
        self._num_shards_dict[split] = num_shards_dict
        logger.info(f"[{split}] num of shards: {num_shards_dict}")
        return num_shards_dict
    
    def load_split_bt_datasets(
        self, split, training, epoch=1, combine=False, shard_epoch=None, rank_langs=None, **kwargs
    ):
        data_param_list = self.get_split_data_param_list(
            split, epoch, shard_epoch=shard_epoch, is_bt=True, rank_langs=rank_langs,
        )
        langpairs_sharing_datasets = (
            {} if self.args.enable_reservsed_directions_shared_datasets else None
        )
        datasets = [
            (
                param["key"],
                self.load_a_bt_dataset(
                    combine=combine,
                    langpairs_sharing_datasets=langpairs_sharing_datasets,
                    **param,
                ),
            )
            for param in data_param_list
        ]
        return datasets, data_param_list
    
    def load_split_datasets(
        self, split, training, epoch=1, combine=False, shard_epoch=None, rank_langs=None, **kwargs
    ):
        data_param_list = self.get_split_data_param_list(
            split, epoch, shard_epoch=shard_epoch, rank_langs=rank_langs
        )
        langpairs_sharing_datasets = (
            {} if self.args.enable_reservsed_directions_shared_datasets else None
        )
        datasets = [
            (
                param["key"],
                self.load_a_dataset(
                    combine=combine,
                    langpairs_sharing_datasets=langpairs_sharing_datasets,
                    **param,
                ),
            )
            for param in data_param_list
        ]
        return datasets, data_param_list
    
    def load_sampled_multi_epoch_dataset(
        self, split, training, epoch=0, combine=False, shard_epoch=None, rank_langs=None, **kwargs
    ):
        datasets, data_param_list = self.load_split_datasets(
            split, training, epoch, combine, shard_epoch=shard_epoch, rank_langs=rank_langs, **kwargs
        )
        if training and split == getattr(self.args, "train_subset", None):
            sample_ratios = self.get_sampling_ratios(data_param_list, datasets, epoch)
            logger.warning(f'load_sampled_multi_epoch_dataset: {get_global_rank()=}: {len(datasets)=}, {rank_langs=}')
            return SampledMultiEpochDataset(
                OrderedDict(datasets),
                epoch=epoch,
                shard_epoch=shard_epoch,
                # valid and test datasets will be degenerate to concating datasets:
                sampling_ratios=sample_ratios,
                eval_key=None,
                collate_format=CollateFormat.single,
                virtual_size=self.args.virtual_data_size,
                split=split,
                virtual_epoch_size=self.args.virtual_epoch_size,
                # if not using lang_tok altering, simplified to use the same collater
                shared_collater=self._shared_collater(),
            )
        else:
            return self.load_into_concat_dataset(split, datasets, data_param_list)

    def load_sampled_multi_dataset(
        self, split, training, epoch=0, combine=False, shard_epoch=None, rank_langs=None, **kwargs
    ):
        datasets, data_param_list = self.load_split_datasets(
            split, training, epoch, combine, shard_epoch=shard_epoch, rank_langs=rank_langs, **kwargs
        )
        if training and split == getattr(self.args, "train_subset", None):
            sample_ratios = self.get_sampling_ratios(data_param_list, datasets, epoch)
            logger.warning(f'load_sampled_multi_dataset: {get_global_rank()=}: {len(datasets)=}, {rank_langs=}')
            return SampledMultiDataset(
                OrderedDict(datasets),
                epoch=epoch,
                # valid and test datasets will be degerate to concating datasets:
                sampling_ratios=sample_ratios,
                eval_key=None,
                collate_format=CollateFormat.single,
                virtual_size=self.args.virtual_data_size,
                split=split,
                # if not using lang_tok altering, simplified to use the same collater
                shared_collater=self._shared_collater(),
            )
        else:
            return self.load_into_concat_dataset(split, datasets, data_param_list)
    
    def load_sampled_multi_epoch_bt_dataset(
        self, split, training, epoch=0, combine=False, shard_epoch=None, rank_langs=None, **kwargs
    ):
        datasets, data_param_list = self.load_split_bt_datasets(
            split, training, epoch, combine, shard_epoch=shard_epoch, rank_langs=rank_langs, **kwargs
        )
        if training and split == getattr(self.args, "train_subset", None):
            sample_ratios = self.get_sampling_ratios(data_param_list, datasets, epoch)
            logger.warning(f'load_sampled_multi_epoch_bt_dataset: {get_global_rank()=}: {len(datasets)=}, {rank_langs=}')
            return SampledMultiEpochDataset(
                OrderedDict(datasets),
                epoch=epoch,
                shard_epoch=shard_epoch,
                # valid and test datasets will be degenerate to concating datasets:
                sampling_ratios=sample_ratios,
                eval_key=None,
                collate_format=CollateFormat.single,
                virtual_size=self.args.virtual_data_size,
                split=split,
                virtual_epoch_size=self.args.virtual_epoch_size,
                # if not using lang_tok altering, simplified to use the same collater
                shared_collater=self._shared_collater(),
            )
        else:
            return self.load_into_concat_dataset(split, datasets, data_param_list)
    
    def load_sampled_multi_bt_dataset(
        self, split, training, epoch=0, combine=False, shard_epoch=None, rank_langs=None, **kwargs
    ):
        datasets, data_param_list = self.load_split_bt_datasets(
            split, training, epoch, combine, shard_epoch=shard_epoch, rank_langs=rank_langs, **kwargs
        )
        if training and split == getattr(self.args, "train_subset", None):
            sample_ratios = self.get_sampling_ratios(data_param_list, datasets, epoch)
            return SampledMultiDataset(
                OrderedDict(datasets),
                epoch=epoch,
                # valid and test datasets will be degerate to concating datasets:
                sampling_ratios=sample_ratios,
                eval_key=None,
                collate_format=CollateFormat.single,
                virtual_size=self.args.virtual_data_size,
                split=split,
                # if not using lang_tok altering, simplified to use the same collater
                shared_collater=self._shared_collater(),
            )
        else:
            return self.load_into_concat_dataset(split, datasets, data_param_list)

    def load_dataset(
        self, split, training, epoch=0, combine=False, shard_epoch=None, 
        world_size=None, rank=None,
        **kwargs
    ):
        world_size = world_size or get_global_world_size()
        rank = int(rank or get_global_rank())
        rank_to_langs = self.args.rank_to_langs
        rank_langs = rank_to_langs[str(rank)]

        datasets = []
        no_main_data = self.args.no_main_data
        load_mmt = not (no_main_data and training and split == getattr(self.args, "train_subset", None))
        logger.warning(f'Load dataset: {split=}, {no_main_data=}, {training=}: {load_mmt}')
        # FIXME: not sure why training hang at end of 1st virtual epoch size
        if load_mmt:
            if self.args.virtual_epoch_size is None:
                mt_dataset = self.load_sampled_multi_dataset(
                    split, training, epoch, combine, shard_epoch, rank_langs=rank_langs, **kwargs
                )
            else:
                mt_dataset = self.load_sampled_multi_epoch_dataset(
                    split, training, epoch, combine, shard_epoch, rank_langs=rank_langs, **kwargs
                )
            datasets.append((DataDomainSpec.main.value, mt_dataset))
            if self.ct_direction_dict is not None:
                logger.warning(f"Load main as cross-translaiton data")
                datasets.append((DataDomainSpec.ct.value, mt_dataset))
            # raise NotImplementedError(f'load_mmt not impl.')
        
        # load BT dataset
        if training and split == getattr(self.args, "train_subset", None):
            if self.args.virtual_epoch_size is None:
                bt_dataset = self.load_sampled_multi_bt_dataset(
                    split, training, epoch, combine, shard_epoch, rank_langs=rank_langs, **kwargs
                )
            else:
                bt_dataset = self.load_sampled_multi_epoch_bt_dataset(
                    split, training, epoch, combine, shard_epoch, rank_langs=rank_langs, **kwargs
                )
            datasets.append((DataDomainSpec.bt.value, bt_dataset))
        assert len(datasets) > 0
        final_dataset = RoundRobinZipDatasets(OrderedDict(datasets))
        return final_dataset
    
    def display_samples_once_in_a_while(self, smp, src_bos_toks=None, prefix=""):
        dictionary = self.main_dictionary
        if 1 < self._show_samples_ctr < self.SHOW_SAMPLES_INTERVAL:
            self._show_samples_ctr += 1
            return
        elif self._show_samples_ctr >= self.SHOW_SAMPLES_INTERVAL:
            self._show_samples_ctr = 0
        else:
            self._show_samples_ctr += 1

        ln = smp["net_input"]["src_tokens"].shape[0]
        rank = get_global_rank()
        # rank = self.args.distributed_rank
        log_string = f"(r:{rank}): {prefix} generated by back-translation.) {ln} samples"
        bpe_symbol = "sentencepiece"
        assert src_bos_toks is None or len(src_bos_toks) == smp["net_input"]["src_tokens"].size(0)
        tgt_bos_index = int(not self.args.lang_tok_replacing_bos_eos)
        tgt_bos_toks = smp["net_input"]["prev_output_tokens"][:, tgt_bos_index]

        for i in range(min(ln, self.SHOW_SAMPLES_NUMBER)):
            src_tokens = smp["net_input"]["src_tokens"][i]
            tgt_tokens = smp["target"][i]
            src_lang = None
            if src_bos_toks is not None:
                src_lang = dictionary[src_bos_toks[i]]
            tgt_lang = dictionary[tgt_bos_toks[i]]

            src_str = dictionary.string(utils.strip_pad(src_tokens, dictionary.pad()), bpe_symbol)
            tgt_str = dictionary.string(utils.strip_pad(tgt_tokens, dictionary.pad()), bpe_symbol)
            log_string += f"\n\t\t[{rank}:{i} - {src_lang} generated]  {src_str}\n"
            log_string += f"\t\t[{rank}:{i} - {tgt_lang} original ]  {tgt_str}\n"
            log_string += f"\t\t[{rank}:{i} - src tokens]  {src_tokens}\n" if self.SHOW_SAMPLES_TOKEN_INDICES else ""
        logger.warning(log_string)
    
    def infer_bt_src_bos_toks(self, tgt_bos_toks: torch.Tensor, spec=LangTokSpec.main.value):
        # NOTE dictionary must be shared
        dictionary = self.main_dictionary
        src_bos_toks = [self.get_other_lang_idx(lid, dictionary, spec=spec) 
            for lid in tgt_bos_toks.cpu().tolist()
        ]
        src_bos_toks_ = torch.tensor(src_bos_toks, dtype=tgt_bos_toks.dtype, device=tgt_bos_toks.device)
        return src_bos_toks_
    
    def get_other_lang(self, lang, direction_dict=None):
        # TODO: allow more complex mapping
        direction_dict = direction_dict or self.bt_direction_dict
        rank = get_global_rank()
        rank_to_langs = self.args.rank_to_langs
        rank_langs = [x.split("-")[0] for x in rank_to_langs[str(rank)].split(",")]
        tgt_lang_list = [x for x in direction_dict[lang] if x in rank_langs]
        assert len(tgt_lang_list) >= 1, f'{lang=}/{tgt_lang_list=},{rank=},{rank_to_langs=},{direction_dict[lang]=}'
        return tgt_lang_list[np.random.randint(0, len(tgt_lang_list))]
    
    def get_other_lang_idx(self, lang_idx, dictionary, spec=LangTokSpec.main.value, direction_dict=None):
        lang_tok = dictionary[lang_idx]
        lang = get_lang_from_lang_tok(lang_tok, lang_tok_style=self.args.lang_tok_style, spec=spec)
        other_lang = self.get_other_lang(lang, direction_dict)
        other_lang_tok = get_lang_tok(other_lang, lang_tok_style=self.args.lang_tok_style, spec=spec)
        other_lang_id = _lang_id(dictionary, other_lang_tok)
        return other_lang_id


def get_encoder_avg_pool(
    ens_model, sample, has_langtok=False
):
    # model = EnsembleModel([model])
    with torch.no_grad():
        model = ens_model

        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }

        # compute the encoder output for each beam
        encoder_outs = model.forward_encoder(encoder_input)
        # np_encoder_outs = encoder_outs[0].encoder_out.cpu().numpy().astype(np.float32)
        # encoder_mask = 1 - encoder_outs[0].encoder_padding_mask.cpu().numpy().astype(
        #     np.float32
        # )
        
        # np_encoder_outs = encoder_outs[0]["encoder_out"][0].cpu().numpy().astype(np.float32)
        # encoder_mask = 1 - encoder_outs[0]["encoder_padding_mask"][0].cpu().numpy().astype(
        #     np.float32
        # )
        _encoder_outs = encoder_outs[0]["encoder_out"][0].to(torch.float32)
        _encoder_mask = 1.0 - encoder_outs[0]["encoder_padding_mask"][0].to(torch.float32)

        # encoder_mask = np.expand_dims(encoder_mask.T, axis=2)
        encoder_mask = _encoder_mask.transpose(1, 0).contiguous()[:, :, None]
        if has_langtok:
            encoder_mask = encoder_mask[1:, :, :]
            _encoder_outs = _encoder_outs[1, :, :]
        masked_encoder_outs = encoder_mask * _encoder_outs
        avg_pool = (masked_encoder_outs / encoder_mask.sum(0)).sum(0)
        avg_pool = move_to_cpu(avg_pool)
    return avg_pool


class SearchDBMultilingualUmtDatasetManager(MultilingualUmtDatasetManager):
    """UMT manager with capability to search kmeans database from generated samples
    How it works:
        * store local reference to original datasets of each language
        * load kmeans database for each langauge (put args in here)
        * for each generated samples, search on the database for topk samples based on leveinstein distance
    How to do the searching:
    Option 1: vanilla, straight-forward
        * do it in current process
        * for each sample, search and return :(
    Option 2: multi-workers DataLoader with stream dataset
        * ?
    Option 3: async multiprocessing search while doing forward
        ## 3.1: multiprocessing, ## 3.2 multi-threading! (thread is little safer)
        * run BT
        * dispatch search to multiple cpu tasks
        * run forward
        * processes.join()
        ***** not sure if this work well DistributedDataParallel
    solutions: 
        * Combine 1, 3.1 3.2 into a single API
    """

    def __init__(self, args, lang_pairs, langs, dicts, sampling_method):
        super().__init__(args, lang_pairs, langs, dicts, sampling_method)
        assert EDITDISTANCE_AVAILABLE
        self._search_databases = None
        self._search_idx_databases = None
        self._searchable_datasets = None
        self._searchable_idx_datasets = None
        self._db_search_mode = self.args.db_search_mode
        self._db_search_type = self.args.db_search_type
        self._search_fn = self.build_search_fn()
    
    def build_search_fn(self):
        if self._db_search_type == "faiss":
            import faiss
            # assert self._db_search_mode == 0
            # current ly first synchronous search
            def search_fn(
                embeds: torch.Tensor,   # (bsz, dim)
                lang_ids: torch.Tensor, # (bsz)
                tokens: Union[torch.Tensor, list],
                top_k: int = 1,
                **kwargs
            ):
                # normalize embeds
                is_add_ref_indices = kwargs.get("is_add_ref_indices", False)
                assert len(embeds) == len(tokens) == len(lang_ids)
                embeds = embeds.cpu().numpy().astype("float32")
                faiss.normalize_L2(embeds)

                lang_ids = lang_ids.cpu()
                tokens = tokens.cpu() if isinstance(tokens, torch.Tensor) else [x.cpu() for x in tokens]
                # partition by lang_ids
                unique_lang_ids = torch.unique(lang_ids).tolist()
                best_s_items = [None] * len(embeds)
                best_s_dataset_indices = torch.zeros(len(embeds), dtype=torch.long)
                best_s_distances = torch.zeros(len(embeds), dtype=torch.float)
                for _, lid in enumerate(unique_lang_ids):
                    databases = self.search_idx_databases[lid]
                    ref_dataset = self.searchable_idx_datasets[lid]
                    _in_lang_id = lang_ids == lid
                    _arange_in_lang_id = torch.arange(len(tokens))[_in_lang_id]
                    _embeds = embeds[_in_lang_id.numpy()]
                    _tokens = [tokens[_j] for _j in _arange_in_lang_id]
                    _lid_best_s_items = []
                    _lid_best_s_dataset_indices = []
                    _lid_best_s_distances = []
                    for db in databases:
                        # db stuff is numpy
                        _db_search_indices = db['index'].search(_embeds, top_k)[1]
                        _db_s_dataset_indices = db['ref'][_db_search_indices]
                        _db_s_dataset_indices = torch.from_numpy(_db_s_dataset_indices)
                        # -----
                        _db_search_items = [
                            [ref_dataset[ref_idx]['source'] for ref_idx in v.tolist()]
                            for v in _db_s_dataset_indices
                        ]
                        _db_search_distances = torch.tensor([
                            [editdistance.eval(_tok.numpy(), _r_tok.numpy()) for _r_tok in _row]
                            for _tok, _row in zip(_tokens, _db_search_items)
                        ])
                        # ----
                        _db_best_topk_indices = _db_search_distances.argmin(dim=1)
                        _db_best_s_dataset_indices = _db_s_dataset_indices.gather(1, _db_best_topk_indices.unsqueeze(-1))
                        _lid_best_s_dataset_indices.append(_db_best_s_dataset_indices[:, 0].unsqueeze(0))
                        _lid_best_s_items.append([
                            _r[_j]
                            for (_j, _r) in zip(_db_best_topk_indices, _db_search_items)
                        ])
                        _lid_best_s_distances.append([
                            _d[_j]
                            for (_j, _d) in zip(_db_best_topk_indices, _db_search_distances)
                        ])
                    _lid_best_s_dataset_indices = torch.cat(_lid_best_s_dataset_indices, 0)
                    _lid_best_s_distances = torch.tensor(_lid_best_s_distances)
                    
                    _min_ref_distances, _best_distance_ids = _lid_best_s_distances.min(0)
                    _best_db_ref_indices_ref = _lid_best_s_dataset_indices.gather(0, _best_distance_ids.unsqueeze(0))[0]
                    _db_search_items = [_lid_best_s_items[_j][_i] for _i, _j in enumerate(_best_distance_ids)]
                    for _id, _item in zip(_arange_in_lang_id, _db_search_items):
                        best_s_items[_id] = _item
                    best_s_distances[_in_lang_id] = _min_ref_distances.to(best_s_distances)
                    best_s_dataset_indices[_in_lang_id] = _best_db_ref_indices_ref.to(best_s_dataset_indices)
                
                if is_add_ref_indices:
                    return best_s_items, best_s_distances, best_s_dataset_indices
                else:
                    return best_s_items, best_s_distances
            
            def search_fn_mode_0(
                embeds: torch.Tensor,   # (bsz, dim)
                lang_ids: torch.Tensor, # (bsz)
                tokens: Union[torch.Tensor, list],
                top_k: int = 1,
                **kwargs
            ):
                pack = search_fn(embeds, lang_ids, tokens, top_k, **kwargs)
                out_holder = [pack]
                callback = lambda : None
                return out_holder, callback
            
            def search_fn_mode_1(
                embeds: torch.Tensor,   # (bsz, dim)
                lang_ids: torch.Tensor, # (bsz)
                tokens: Union[torch.Tensor, list],
                top_k: int = 1,
                **kwargs
            ):
                out_holder = []
                def _run():
                    pack = search_fn(embeds, lang_ids, tokens, top_k, **kwargs)
                    out_holder.append(pack)
                p = threading.Thread(target=_run, args=(), daemon=True)
                p.start()
                callback = lambda : p.join()
                return out_holder, callback

            if self._db_search_mode == 0:
                return search_fn_mode_0
            elif self._db_search_mode == 1:
                return search_fn_mode_1
            else:
                raise ValueError(f'{self._db_search_mode=}')
        else:
            raise NotImplementedError(f"{self._db_search_type} not impl")

    @staticmethod
    def add_args(parser):
        # MultilingualUmtDatasetManager.add_args(parser)
        # NOTE: task must run MultilingualUmtDatasetManager.add_args(parser) first
        parser.add_argument('--db-search-type', type=str, default='faiss',
            help='search-type')

        parser.add_argument(
            "--db-map-paths",
            help='dict of comma separated db paths for each language, \
                    e.g. {"en_XX": "/projects/nmt/..../en_XX.db.npy"}',
            type=lambda uf: eval_str_dict(uf, type=str),
            default=None,
        )
        parser.add_argument(
            "--db-search-mode",
            default=0,
            type=int,
            help="search mode: 0: vanilla, 1: sub-thread async, ",
        )
        parser.add_argument(
            "--db-search-topk",
            default=1,
            type=int,
            help="search mode: 0: vanilla, 1: sub-thread async, ",
        )
        parser.add_argument(
            "--db-faiss-nprobe",
            default=1,
            type=int,
            help="search mode: 0: vanilla, 1: sub-thread async, ",
        )
        parser.add_argument('--edit-distance-weight-type', type=str, default='',
            help='Type of edit distance weighting')
        parser.add_argument('--edit-distance-weight-params', type=str, default='0,1',
            help='params for scores2weights')

        parser.add_argument('--db-inference-mode', default=False, action='store_true',
            help='only used for testing')
        parser.add_argument('--db-inference-mode-reverse', default=False, action='store_true',
            help='only used for testing')
    
    @property
    def db_inference_mode(self):
        return self.args.db_inference_mode
    
    def edit_distances_to_weights(self, search_tokens, original_tokens, distances):
        # FIXME: too much distance to make on-the-fly search possible
        edit_distance_weight_type = self.args.edit_distance_weight_type
        edit_distance_weight_params = self.args.edit_distance_weight_params
        if edit_distance_weight_type == "" or edit_distance_weight_type is None:
            return None
        elif edit_distance_weight_type == "len_ratio_":
            assert isinstance(search_tokens, list)
            dist_on_len = [
                y / len(x)
                for x, y in zip(search_tokens, distances)
            ]
            # high dist -> low score
            # return weights
            # TODO: -=-----
        else:
            raise ValueError(f'{edit_distance_weight_type=} not found')
    
    @classmethod
    def setup_data_manager(cls, args, lang_pairs, langs, dicts, sampling_method):
        return cls(
            args, lang_pairs, langs, dicts, sampling_method
        )
    
    def load_sampled_multi_bt_dataset(
        self, split, training, epoch=0, combine=False, shard_epoch=None, **kwargs
    ):
        datasets, data_param_list = self.load_split_bt_datasets(
            split, training, epoch, combine, shard_epoch=shard_epoch, **kwargs
        )
        if training and split == getattr(self.args, "train_subset", None):
            sample_ratios = self.get_sampling_ratios(data_param_list, datasets, epoch)
            return SampledMultiDataset(
                OrderedDict(datasets),
                epoch=epoch,
                # valid and test datasets will be degerate to concating datasets:
                sampling_ratios=sample_ratios,
                eval_key=None,
                collate_format=CollateFormat.single,
                virtual_size=self.args.virtual_data_size,
                split=split,
                # if not using lang_tok altering, simplified to use the same collater
                shared_collater=self._shared_collater(),
            )
        else:
            if self.db_inference_mode:
                logger.warning(f'WARNING!!! DB inference mode !, load searchable datasets and only concat data[:1]')
                self._load_searchable_datasets([d for _, d in datasets])
                return self.load_into_concat_dataset(split, datasets[:1], data_param_list[:1])
            else:
                return self.load_into_concat_dataset(split, datasets, data_param_list)
    
    def load_sampled_multi_epoch_bt_dataset(
        self, split, training, epoch=0, combine=False, shard_epoch=None, **kwargs
    ):
        datasets, data_param_list = self.load_split_bt_datasets(
            split, training, epoch, combine, shard_epoch=shard_epoch, **kwargs
        )
        if training and split == getattr(self.args, "train_subset", None):
            sample_ratios = self.get_sampling_ratios(data_param_list, datasets, epoch)
            return SampledMultiEpochDataset(
                OrderedDict(datasets),
                epoch=epoch,
                shard_epoch=shard_epoch,
                # valid and test datasets will be degenerate to concating datasets:
                sampling_ratios=sample_ratios,
                eval_key=None,
                collate_format=CollateFormat.single,
                virtual_size=self.args.virtual_data_size,
                split=split,
                virtual_epoch_size=self.args.virtual_epoch_size,
                # if not using lang_tok altering, simplified to use the same collater
                shared_collater=self._shared_collater(),
            )
        else:
            if self.db_inference_mode:
                logger.warning(f'WARNING!!! DB inference mode !, load searchable datasets and only concat data[:1]')
                self._load_searchable_datasets([d for _, d in datasets])
                return self.load_into_concat_dataset(split, datasets[:1], data_param_list[:1])
            else:
                return self.load_into_concat_dataset(split, datasets, data_param_list)
    
    def _load_searchable_datasets(self, packed_dataset, lang_pairs=None):
        if self._searchable_datasets is None:
            if isinstance(packed_dataset, SampledMultiDataset):
                self._searchable_datasets = {
                    k.split("-")[-1]: v
                    for k, v in zip(packed_dataset.keys, packed_dataset.datasets)
                }
            elif isinstance(packed_dataset, ConcatDataset):
                _datasets = packed_dataset.datasets
                lang_pairs = lang_pairs or {
                    k: v.split(",") for k, v in self.args.bt_lang_pairs.items()
                }['bt']
                self._searchable_datasets = {
                    ks.split("-")[0]: _datasets[i]
                    for i, ks in enumerate(lang_pairs)
                }
            elif isinstance(packed_dataset, (list, tuple)):
                lang_pairs = lang_pairs or {
                    k: v.split(",") for k, v in self.args.bt_lang_pairs.items()
                }['bt']
                assert isinstance(packed_dataset[0], FairseqDataset)
                self._searchable_datasets = {
                    ks.split("-")[0]: packed_dataset[i]
                    for i, ks in enumerate(lang_pairs)
                }
            else:
                raise ValueError(f'{type(packed_dataset)}')
            # logger.warning(f'Loaded searchable datasets: {self._searchable_datasets.keys()}')
    
    def _load_mmt_dataset(self, datasets, split, training, epoch, combine, shard_epoch, **kwargs):
        if self.args.virtual_epoch_size is None:
            mt_dataset = self.load_sampled_multi_dataset(
                split, training, epoch, combine, shard_epoch, **kwargs
            )
        else:
            mt_dataset = self.load_sampled_multi_epoch_dataset(
                split, training, epoch, combine, shard_epoch, **kwargs
            )
        datasets.append((DataDomainSpec.main.value, mt_dataset))
        # self._load_searchable_datasets(mt_dataset)
        if self.ct_direction_dict is not None:
            logger.warning(f"Load main as cross-translaiton data")
            datasets.append((DataDomainSpec.ct.value, mt_dataset))
    
    def _load_mbt_dataset(self, datasets, split, training, epoch, combine, shard_epoch, **kwargs):
        if self.args.virtual_epoch_size is None:
            bt_dataset = self.load_sampled_multi_bt_dataset(
                split, training, epoch, combine, shard_epoch, debug_load_km=True, **kwargs
            )
        else:
            bt_dataset = self.load_sampled_multi_epoch_bt_dataset(
                split, training, epoch, combine, shard_epoch, **kwargs
            )
        datasets.append((DataDomainSpec.bt.value, bt_dataset))
        self._load_searchable_datasets(bt_dataset)
        
    def load_dataset(
        self, split, training, epoch=0, combine=False, shard_epoch=None, **kwargs
    ):
        datasets = []
        no_main_data = self.args.no_main_data
        load_mmt = not (no_main_data and training and split == getattr(self.args, "train_subset", None))
        if load_mmt and not self.db_inference_mode:
            self._load_mmt_dataset(datasets, split, training, epoch, combine, shard_epoch, **kwargs)
        
        # load BT dataset
        if (training and split == getattr(self.args, "train_subset", None)) or self.db_inference_mode:
            self._load_mbt_dataset(datasets, split, training, epoch, combine, shard_epoch, **kwargs)
            
        assert len(datasets) > 0
        if load_mmt and not training and split != getattr(self.args, "train_subset", None):
            return datasets[0][1]
        final_dataset = RoundRobinZipDatasets(OrderedDict(datasets))
        return final_dataset
    
    def load_mbt_datasets_from_database(self, split, epoch=0, combine=False, shard_epoch=None, **kwargs):
        if self._searchable_datasets is not None:
            return
        # datasets, data_param_list = self.load_split_bt_datasets(
        #     split, training, epoch, combine, shard_epoch=shard_epoch, **kwargs
        # )
        # todo: load_split_bt_datasets
        # data_param_list = self.get_split_data_param_list(
        #     split, epoch, shard_epoch=shard_epoch, is_bt=True
        # )

        # TODO: get bt_lang_pairs
        bt_lang_pairs = {
            "bt": ','.join([f'{k}-{k}' for k in self.search_databases.keys()])
        }
        # ===============================
        # get_split_data_param_list
        param_list = []
        data_paths, lang_pairs = self.get_data_paths_and_lang_pairs(
            split, is_bt=True, bt_lang_pairs=bt_lang_pairs)
        logger.info(f"load_mbt_datasets_by_database: langtoks settings: {self.args.langtoks}")
        split_num_shards_dict = self.get_split_num_data_shards(split, bt_lang_pairs=bt_lang_pairs)
        for data_category, paths in data_paths.items():
            if data_category not in lang_pairs:
                continue
            paths = utils.split_paths(paths)
            assert len(paths) > 0
            if len(paths) > 1:
                self._has_sharded_data = True
            if split != getattr(self.args, "train_subset", None):
                # if not training data set, use the first shard for valid and test
                paths = paths[:1]

            if data_category in self.args.langtoks:
                lang_tok_spec = self.args.langtoks[data_category]
            else:
                # default to None
                lang_tok_spec = (None, None)

            # infer langcode
            lang_dirs = [
                lang_pair.split("-") for lang_pair in lang_pairs[data_category]
            ]
            lang_dirs = [x if len(x) > 1 else (x[0], x[0]) for x in lang_dirs]
            for src, tgt in lang_dirs:
                assert src is not None or data_category == "mono_dae", (
                    f"error: src={src}, tgt={tgt} for data_category={data_category}"
                )
                key = self.get_dataset_key(data_category, src, tgt)
                data_path = self.get_split_data_path(
                    paths, epoch, shard_epoch, split_num_shards_dict[key]
                )
                param_list.append(
                    {
                        "key": key,
                        "data_path": data_path,
                        "split": split,
                        "src": src,
                        "src_dict": self.get_source_dictionary(src)
                        if src and data_category != "mono_dae"
                        else None,
                        "tgt": tgt,
                        "tgt_dict": self.get_target_dictionary(tgt),
                        "data_category": data_category,
                        "langtok_spec": lang_tok_spec,
                    }
                )
        # ===============================

        langpairs_sharing_datasets = (
            {} if self.args.enable_reservsed_directions_shared_datasets else None
        )
        datasets = [
            (
                param["key"],
                self.load_a_bt_dataset(
                    combine=combine,
                    langpairs_sharing_datasets=langpairs_sharing_datasets,
                    **param,
                ),
            )
            for param in param_list
        ]
        _datasets = [d for _, d in datasets]
        # logger.warning(f'{len(_datasets)} from {param_list=}')
        self._load_searchable_datasets(_datasets, bt_lang_pairs['bt'].split(","))

    @classmethod
    def load_search_database(cls, path, db_type, args):
        if db_type == "faiss":
            import faiss
            from fairseq.file_io import PathManager
            assert PathManager.exists(f'{path}.index.bin'), f'{path}.index.bin not found'
            assert PathManager.exists(f'{path}.index.data_indices.npy'), f'{path}.index.data_indices.npy not found'
            index = faiss.read_index(f'{path}.index.bin')
            with open(f'{path}.index.data_indices.npy', 'rb') as f:
                data_indices = np.load(f)
            index.nprobe = args.db_faiss_nprobe
            db = {
                "index": index,
                "ref": data_indices
            }
            return db
        elif db_type == "km_db":
            # load_kmeans_database
            raise NotImplementedError(f'{db_type=} not impl')
        else:
            raise NotImplementedError(f'{db_type=} not impl')
    
    @property
    def db_search_mode(self):
        return self._db_search_mode
    
    @property
    def search_databases(self):
        if self._search_databases is None:
            db_map_paths = self.args.db_map_paths
            self._search_databases = {
                k: [
                    self.__class__.load_search_database(x, self._db_search_type, self.args) 
                    for x in v.split(",")]
                for k, v in db_map_paths.items()
            }
        return self._search_databases

    @property
    def search_idx_databases(self):
        if self._search_idx_databases is None:
            self._search_idx_databases = {
                _lang_id(
                    self.main_dictionary, 
                    get_lang_tok(
                        k, 
                        lang_tok_style=self.args.lang_tok_style,
                        spec=LangTokSpec.main.value
                    )
                ): v
                for k, v in self.search_databases.items()
            }
        return self._search_idx_databases
    
    @property
    def searchable_idx_datasets(self):
        if self._searchable_idx_datasets is None:
            assert isinstance(self._searchable_datasets, dict), f'km_datasets probably not set yet!'
            self._searchable_idx_datasets = {
                _lang_id(
                    self.main_dictionary, 
                    get_lang_tok(
                        k, 
                        lang_tok_style=self.args.lang_tok_style,
                        spec=LangTokSpec.main.value
                    )
                ): v
                for k, v in self._searchable_datasets.items()
            }
        return self._searchable_idx_datasets
    
    def _search_km_db_single(self, emb, lid, tok):
        # search over multiple databases, if possible
        best_refs = []
        db_best_distances = []
        for db in self.search_idx_databases[lid]:
            ref_indices = multi_layer_db_search_single(emb, db)
            ref_dataset = self.searchable_idx_datasets[lid]
            # f"{data_category}:{src}-{tgt}"
            try:
                ref_items = [ref_dataset[j]['source'] for j in ref_indices.tolist()]
            except Exception as e:
                print(f'{len(ref_dataset)=}')
                print(f'{ref_indices.tolist()}, max={ref_indices.max()}, min={ref_indices.min()}')
                print(f'{ref_dataset[ref_indices[0]]=}')
                raise e
            ref_distances = np.array([
                editdistance.eval(tok, r_tok)
                for r_tok in ref_items
            ])
            best_index = np.argmin(ref_distances)
            best_refs.append(ref_items[best_index])
            db_best_distances.append(ref_distances[best_index])
        if len(best_refs) == 1:
            best_distance = db_best_distances[0]
            best_ref = best_refs[0]
        else:
            best_distances = [editdistance.eval(tok, x) for x in best_refs]
            best_ref_index = np.argmin(np.array(best_distances))
            best_ref = best_refs[best_ref_index]
            best_distance = best_distances[best_ref_index]
        return best_ref, best_distance
    
    def _search_km_db_on_embeds_mode_0(
        self, 
        embeds: torch.Tensor,   # (bsz, dim)
        lang_ids: torch.Tensor, # (bsz)
        tokens: Union[torch.Tensor, list],
        top_k: int = 1
    ):
        import faiss
        # linear search!
        assert top_k == 1
        embeds = embeds.cpu().numpy().astype("float32")
        faiss.normalize_L2(embeds)

        lang_ids = lang_ids.cpu().numpy()
        tokens = tokens.cpu().numpy() if isinstance(tokens, torch.Tensor) else [x.cpu().numpy() for x in tokens]
        out_tokens = []
        out_dists = []
        for i, (emb, lid, tok) in enumerate(zip(embeds, lang_ids, tokens)):
            best_out, best_dist = self._search_km_db_single(emb, lid, tok)
            out_tokens.append(best_out)
            out_dists.append(best_dist)
        callback = lambda : None
        out_tokens_list = [(out_tokens, out_dists)]
        return out_tokens_list, callback
    
    def _search_km_db_on_embeds_mode_1(
        self, 
        embeds: torch.Tensor,   # (bsz, dim)
        lang_ids: torch.Tensor, # (bsz)
        tokens: Union[torch.Tensor, list],
        top_k: int = 1
    ):
        import faiss

        assert top_k == 1
        out_tokens_holder_list = []
        def _run(_embeds, _lang_ids, _tokens):
            _embeds = _embeds.cpu().numpy().astype("float32")
            faiss.normalize_L2(embeds)

            _lang_ids = _lang_ids.cpu().numpy()
            _tokens = _tokens.cpu().numpy() if isinstance(_tokens, torch.Tensor) else [x.cpu().numpy() for x in _tokens]
            out_tokens = []
            out_dists = []
            for i, (emb, lid, tok) in enumerate(zip(_embeds, _lang_ids, _tokens)):
                # search over multiple databases, if possible
                best_out, best_dist = self._search_km_db_single(emb, lid, tok)
                out_tokens.append(best_out)
                out_dists.append(best_dist)
            out_tokens_holder_list.append((out_tokens, out_dists))
        p = threading.Thread(target=_run, args=(embeds, lang_ids, tokens), daemon=True)
        p.start()
        callback = lambda : p.join()
        return out_tokens_holder_list, callback

    def search_db_on_embeds(
        self, 
        embeds: torch.Tensor,   # (bsz, dim)    embeds is array
        lang_ids: torch.Tensor, # (bsz)
        tokens: Union[torch.Tensor, list],
        top_k: int = 1,
        mode=0,
        **kwargs
    ):
        return self._search_fn(embeds, lang_ids, tokens, top_k, **kwargs)
    
    def generate_encoder_avg_pool(self, generator, generated, smp):
        dictionary = self.main_dictionary
        left_pad_source = self.args.left_pad_source

        # prefix_removal = 1  # FIXME: this one is hard-coded
        # generated_tokens = [gn[0]['tokens'][prefix_removal:] for gn in generated]
        
        # max_length = max(len(x) for x in generated_tokens)
        # src_tokens = torch.empty(size=(len(generated_tokens), max_length), dtype=smp['net_input']["src_tokens"].dtype)
        # src_lengths = torch.empty(len(generated_tokens), dtype=smp['net_input']["src_lengths"].dtype)
        # for i, stok in enumerate(generated_tokens):
        #     tok_size = stok.size(0)
        #     padding_needed = max_length - stok.size(0)
        #     if left_pad_source:
        #         tokens = F.pad(stok, (padding_needed, 0), value=dictionary.pad())
        #     else:
        #         tokens = F.pad(stok, (0, padding_needed), value=dictionary.pad())
        #     src_tokens[i] = tokens
        #     src_lengths[i] = tok_size
        
        # gen_smp = {
        #     "net_input": {"src_tokens": src_tokens, 'src_lengths': src_lengths}
        # }
        # gen_smp = move_to_cuda(gen_smp)
        gen_smp = self.convert_generated_hypos_to_tokens(
            generated, get_src_sample=True, smp_example=smp, to_cuda=True
        )
        
        # forward
        avg_pool = get_encoder_avg_pool(
            generator.model,
            gen_smp,
            has_langtok=self.args.encoder_langtok is not None,
        )
        return avg_pool
    
    def build_search_smp_from_smp(self, search_tokens, search_distances, smp, is_source=True):
        # search_tokens
        dictionary = self.main_dictionary
        left_pad_source = self.args.left_pad_source
        assert is_source, f'{is_source=} must be True for now'
        search_smp = copy.deepcopy(smp)

        net_input = search_smp['net_input']
        src_max_length = max(len(x) for x in search_tokens)
        src_tokens = torch.empty(size=(len(search_tokens), src_max_length), dtype=net_input["src_tokens"].dtype)
        src_lengths = torch.empty(len(search_tokens), dtype=net_input["src_lengths"].dtype)
        for i, stok in enumerate(search_tokens):
            tok_size = stok.size(0)
            padding_needed = src_max_length - stok.size(0)
            if left_pad_source:
                tokens = F.pad(stok, (padding_needed, 0), value=dictionary.pad())
            else:
                tokens = F.pad(stok, (0, padding_needed), value=dictionary.pad())
            src_tokens[i] = tokens
            src_lengths[i] = tok_size
        device = net_input["src_tokens"].device
        # This seems to be important
        del net_input["src_tokens"]
        del net_input["src_lengths"]
        net_input["src_tokens"] = src_tokens.to(device)
        net_input["src_lengths"] = src_lengths.to(device)
        return search_smp
    
    def backtranslate_multi_sample_km_search(
        self, generator, smp, topk=1
    ) -> Tuple:
        """
        Back-translation and then search for best alternative from database
        
        generated_encoder_fn[torch.no_grad()](generated, **kwargs) -> embeddings[bsz, dim]

        # with above params, src not have lang-tok, tgt will have lang-tok prepended
        Expected input sample:
            - smp['net_input']['src_tokens']            hello world
            - smp["net_input"]["prev_output_tokens"]    </s> <langtok> hello world
            - smp["target"]                             <langtok> hello world </s>
        Expected output
            - smp['net_input']['src_tokens']            salut lume
            - smp["net_input"]["prev_output_tokens"]    </s> <langtok> hello world
            - smp["target"]                             <langtok> hello world </s>
        """
        # _target = smp["target"]
        assert topk == 1, f'{topk=} not supported'
        assert not self.args.encoder_langtok
        assert self.args.decoder_langtok
        search_mode = self.db_search_mode

        _prev_tokens = smp["net_input"]["prev_output_tokens"]
        is_gpu = _prev_tokens.is_cuda
        tgt_bos_toks = _prev_tokens[:, int(not self.lang_tok_replacing_bos_eos)]

        # FIXME: check generate, consider to use task.inference_step
        src_bos_toks = self.infer_bt_src_bos_toks(tgt_bos_toks)
        prefix_tokens = src_bos_toks.unsqueeze(1)
        generated = generator.generate(
            models=[], 
            sample=smp, 
            prefix_tokens=prefix_tokens
        )
        self._configure_generated_sample(generated, smp)

        # search km database
        generated_tokens = self.convert_generated_hypos_to_tokens(generated)
        generated_embeds = self.generate_encoder_avg_pool(generator, generated, smp)
        generated_lang_ids = src_bos_toks

        generated_tokens, generated_embeds, generated_lang_ids = move_to_cpu(
            (generated_tokens, generated_embeds, generated_lang_ids)
        )

        search_tokens_list, search_callback = self.search_db_on_embeds(
            generated_embeds, generated_lang_ids, generated_tokens,
            top_k=self.args.db_search_topk,
            mode=search_mode
        )
        # handle asynchronous processing
        smp_list = [smp]
        src_bos_toks_list = [src_bos_toks]
        def callback():
            search_callback()
            search_tokens, search_distances = search_tokens_list[0]
            search_smp = self.build_search_smp_from_smp(search_tokens, search_distances, smp, is_source=True)
            if is_gpu:
                search_smp = move_to_cuda(search_smp)
            smp_list.append(search_smp)
            src_bos_toks_list.append(src_bos_toks)
        return src_bos_toks_list, smp_list, callback
    
    def display_searched_samples_once_in_a_while(self, ori_smp, smp, src_bos_toks=None, prefix=""):
        dictionary = self.main_dictionary
        if 1 < self._show_samples_ctr < self.SHOW_SAMPLES_INTERVAL:
            self._show_samples_ctr += 1
            return
        elif self._show_samples_ctr >= self.SHOW_SAMPLES_INTERVAL:
            self._show_samples_ctr = 0
        else:
            self._show_samples_ctr += 1

        ln = smp["net_input"]["src_tokens"].shape[0]

        logger.info(
            f"(r:{self.args.distributed_rank}) :"
            f"{prefix} generated by back-translation.) {ln} samples"
        )
        bpe_symbol = "sentencepiece"
        assert src_bos_toks is None or len(src_bos_toks) == smp["net_input"]["src_tokens"].size(0)
        tgt_bos_index = int(not self.lang_tok_replacing_bos_eos)
        tgt_bos_toks = smp["net_input"]["prev_output_tokens"][:, tgt_bos_index]

        for i in range(min(ln, self.SHOW_SAMPLES_NUMBER)):
            ori_src_tokens = ori_smp["net_input"]["src_tokens"][i]
            src_tokens = smp["net_input"]["src_tokens"][i]
            tgt_tokens = smp["target"][i]
            src_lang = dictionary[src_bos_toks[i]] if src_bos_toks is not None else None
            tgt_lang = dictionary[tgt_bos_toks[i]]

            ori_src_str = dictionary.string(utils.strip_pad(ori_src_tokens, dictionary.pad()), bpe_symbol)
            src_str = dictionary.string(utils.strip_pad(src_tokens, dictionary.pad()), bpe_symbol)
            tgt_str = dictionary.string(utils.strip_pad(tgt_tokens, dictionary.pad()), bpe_symbol)
            src_tokens_str = f"\t\t[ src tokens]  {src_tokens}\n" if self.SHOW_SAMPLES_TOKEN_INDICES else ""
            logger.info(
                f"\n{i}\t\t[{src_lang} generated]  {ori_src_str}"
                f"\n{i}\t\t[{src_lang} searched]  {src_str}\n"
                f"\t\t[{tgt_lang} original ]  {tgt_str}\n" + src_tokens_str
            )