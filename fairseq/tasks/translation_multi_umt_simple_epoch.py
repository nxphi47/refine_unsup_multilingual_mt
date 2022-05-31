from argparse import Namespace
from collections import defaultdict
import datetime
import itertools
import json
import logging
from multiprocessing import Value
import time
from typing import Dict
import numpy as np

import torch
from fairseq import checkpoint_utils, models, utils
from fairseq.data import (
    FairseqDataset,
    LanguagePairDataset,
    ListDataset,
    data_utils,
    encoders,
    iterators,
)
from fairseq.data.multilingual.multilingual_data_manager import MultilingualDatasetManager
from fairseq.data.multilingual.multiling_umt_data_manager import DataDomainSpec, GpuSepMultilingualUmtDatasetManager, SearchDBMultilingualUmtDatasetManager, MultilingualUmtDatasetManager
from fairseq.data.multilingual.sampling_method import SamplingMethod
from fairseq.file_io import PathManager
from fairseq.logging import metrics
from fairseq.sequence_generator import MultiDecPassEnsembleModel, MultiPrefixSequenceGenerator, OBTmBARTSequenceGenerator, SequenceGenerator
from fairseq.tasks import register_task
from fairseq.tasks.online_backtranslation import PiecewiseLinearFn
from fairseq.tasks.translation import EVAL_BLEU_ORDER, TranslationTask
from fairseq.utils import FileContentsAction, apply_to_sample, eval_str_dict

from fairseq.optim.amp_optimizer import AMPOptimizer

from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask
from fairseq.tasks.translation_multi_simple_epoch import get_time_gap
from fairseq.trainer import _catalog_shared_params, _get_module_by_path, _set_module_by_path

from fairseq.distributed import utils as distributed_utils

import copy

logger = logging.getLogger(__name__)


@register_task("translation_multi_umt_simple_epoch")
class TranslationMultiUmtSimpleEpochTask(TranslationMultiSimpleEpochTask):
    """
    Apart from parallel data, do BT
    For each batch of multi-lingual data
        for each src sentence, select tgt lang in bt-infer-lang-pairs:
        do back-translation with tgt_lang_id be correspond bos tokens
        create tgt-src data
        train on BT task
    
    """
    def __init__(self, args, langs, dicts, training, freq_dicts=None):
        super(TranslationMultiSimpleEpochTask, self).__init__(args)
        
        self.langs = langs
        self.dicts = dicts
        self.training = training
        if training:
            self.lang_pairs = args.lang_pairs
        else:
            self.lang_pairs = ["{}-{}".format(args.source_lang, args.target_lang)]
        self.dictionary = self.dicts[self.lang_pairs[0].split('-')[0]]
        # eval_lang_pairs for multilingual translation is usually all of the
        # lang_pairs. However for other multitask settings or when we want to
        # optimize for certain languages we want to use a different subset. Thus
        # the eval_lang_pairs class variable is provided for classes that extend
        # this class.
        self.eval_lang_pairs = self.lang_pairs
        # model_lang_pairs will be used to build encoder-decoder model pairs in
        # models.build_model(). This allows multitask type of sub-class can
        # build models other than the input lang_pairs
        self.model_lang_pairs = self.lang_pairs
        self.source_langs = [d.split("-")[0] for d in self.lang_pairs]
        self.target_langs = [d.split("-")[1] for d in self.lang_pairs]
        self.check_dicts(self.dicts, self.source_langs, self.target_langs)

        self.sampling_method = SamplingMethod.build_sampler(args, self)
        self.data_manager = self.setup_data_manager(args, langs, dicts)
        self._bt_langs = None
        self._bt_direction_dict = None
        self.freq_dicts = freq_dicts
        
        # lambda
        self.lambda_bt = PiecewiseLinearFn.from_string(args.lambda_bt)
        self.lambda_ct = PiecewiseLinearFn.from_string(args.lambda_ct)
        self.lambda_main = PiecewiseLinearFn.from_string(args.lambda_main)
    
    def setup_data_manager(self, args, langs, dicts):
        return MultilingualUmtDatasetManager.setup_data_manager(
            args, self.lang_pairs, langs, dicts, self.sampling_method
        )
    
    @classmethod
    def build_frequency_dicts(cls, args, langs, dicts, training):
        # freq_dicts = {}
        # for mono_lang in mono_langs:
        #     freq_dict_path = os.path.join(paths[0], f"dict.freq.{mono_lang}.txt")
        #     if os.path.exists(freq_dict_path):
        #         try:
        #             freq_dict = cls.load_dictionary(freq_dict_path)
        #             for lang in languages:
        #                 freq_dict.add_symbol(_mbart_lang_token(lang))
        #             freq_dict.add_symbol("<mask>")
        #             freq_dicts[mono_lang] = freq_dict
        #             logger.info(f'Load Top freq dictionary dict {mono_lang}')
        #         except Exception as e:
        #             logger.warning(f'Error Frequency dict {mono_lang}: {e}')
        #             freq_dicts = {}
        #             break
        #     else:
        #         logger.warning(f'Frequency dict {mono_lang} not found {freq_dict_path}, not using this --top-frequency')
        #         freq_dicts = {}
        #         break
        # TODO: frequency dicts
        return None

    @classmethod
    def setup_task(cls, args, **kwargs):
        langs, dicts, training = MultilingualUmtDatasetManager.prepare(
           cls.load_dictionary, args, **kwargs
        )
        freq_dicts = cls.build_frequency_dicts(args, langs, dicts, training)
        return cls(args, langs, dicts, training, freq_dicts=freq_dicts)
    
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='inference source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='inference target language')
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr',
                            action=FileContentsAction)
        
        parser.add_argument('--keep-inference-langtok', action='store_true',
                            help='keep language tokens in inference output (e.g. for analysis or debugging)')

        SamplingMethod.add_arguments(parser)
        MultilingualUmtDatasetManager.add_args(parser)

        parser.add_argument('--lambda-bt', default="1.0", type=str, metavar='N',
                            help='back-translation weight')
        
        parser.add_argument('--lambda-ct', default="0.0", type=str, metavar='N',
                            help='cross-translation weight')
        
        parser.add_argument('--lambda-main', default="1.0", type=str, metavar='N',
                            help='main weight')
        # disable target tokens according to top-frequency of weights
        parser.add_argument('--top-frequency', default=1, type=float, metavar='N',
                            help='During BT, only generate tokens appearing > top-freq in target corpus.'
                                 'Data path must contains dict dict.freq.<lang>.txt specifying the frequencies of all tokens'
                                 ' for that language corpus'
                            ),
        parser.add_argument('--no-top-freq-after', default=1000, type=int, metavar='N',
                            help='Stop top-frequency inferencing after number of updates'),
        # valid bleu
        parser.add_argument(
            "--eval-bleu",
            action="store_true",
            help="evaluation with BLEU scores",
        )
        parser.add_argument(
            "--eval-bleu-args",
            help='generation args for BLUE scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string',
            type=lambda uf: eval_str_dict(uf, type=str),
            default='{}',
        )
        parser.add_argument(
            '--eval-bleu-detok', default="space", type=str,
            help="detokenize before computing BLEU (e.g., 'moses'); required if using --eval-bleu; "
                "use 'space' to disable detokenization; see fairseq.data.encoders for other options"
        )
        parser.add_argument(
            "--eval-bleu-detok-args",
            help="args for building the tokenizer, if needed, as JSON string",
            type=lambda uf: eval_str_dict(uf, type=str),
            default='{}',
        )
        parser.add_argument(
            "--eval-tokenized-bleu",
            action="store_true",
            help="compute tokenized BLEU instead of sacrebleu",
        )
        parser.add_argument(
            '--eval-bleu-remove-bpe', default=None, type=str,
            help="remove BPE before computing BLEU"
        )
        parser.add_argument(
            "--eval-bleu-print-samples",
            action="store_true",
            help="compute tokenized BLEU instead of sacrebleu",
        )
        # secondary models as translation
        parser.add_argument(
            "--cache-model-path",
            default=None,
            type=str,
            help="Secondary model path, same params as main model",
        )
        
        parser.add_argument(
            "--cache-overwrite-sharded-path",
            default=None,
            type=str,
            help="Secondary model path, same params as main model",
        )

        
    def build_model(self, args):
        model = super().build_model(args)

        cache_path = str(getattr(args, "cache_model_path", None))
        if PathManager.isfile(cache_path):
            # FIXME: previous version of translation_multi_umt_cachedgen_simple_epoch
            #   cast TranslationMultiUmtSimpleEpochTask super -> build_model
            #       which means it is standard TranslationMultiSimpleEpochTask
            cache_model = super(TranslationMultiSimpleEpochTask, self).build_model(args)
            for param in cache_model.parameters():
                param.requires_grad = False
            logger.info(f"Use Secondary model cached from {cache_path} ; no gradient here!")
        else:
            cache_model = model

        self.bt_sequence_generator = SequenceGenerator(
            [cache_model],
            tgt_dict=self.dictionary,
            beam_size=1,
            max_len_a=1.3,
            max_len_b=5,
            min_len=5,
            # keep 1 to be able to prepend bos
            max_len=model.max_decoder_positions() - 1,
        )
        if getattr(args, "eval_bleu", False):
            assert getattr(args, "eval_bleu_detok", None) is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(getattr(args, "eval_bleu_detok_args", "{}") or "{}")
            self.tokenizer = encoders.build_tokenizer(
                Namespace(
                    tokenizer=getattr(args, "eval_bleu_detok", None), **detok_args
                )
            )

            gen_args = json.loads(getattr(args, "eval_bleu_args", "{}") or "{}")
            self.bleu_sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )

        return model
    
    def setup_cache_model(self, trainer):
        # move cache model to designated device
        # NOTE to be ran be trainer.py
        if PathManager.isfile(str(self.args.cache_model_path)):
            logger.info(f"Setting up Cached Model at {self.args.cache_model_path}")
            assert self.bt_sequence_generator.model.models_size == 1
            single_model = self.bt_sequence_generator.model.single_model 

            cfg = trainer.cfg
            shared_params = _catalog_shared_params(single_model)

            if not trainer.is_fsdp:
                if cfg.common.fp16:
                    assert not cfg.common.amp, "Cannot use fp16 and AMP together"
                    single_model = single_model.half()
                elif cfg.common.bf16:
                    single_model = single_model.to(dtype=torch.bfloat16)
            if (
                not cfg.distributed_training.pipeline_model_parallel
                # the DistributedFairseqModel wrapper will handle moving to device,
                # so only handle cases which don't use the wrapper
                and not trainer.use_distributed_wrapper
            ):
                single_model = single_model.to(device=trainer.device)

            # check that shared parameters are preserved after device transfer
            for shared_param in shared_params:
                ref = _get_module_by_path(single_model, shared_param[0])
                for path in shared_param[1:]:
                    logger.info(
                        "detected shared parameter: {} <- {}".format(shared_param[0], path)
                    )
                    _set_module_by_path(single_model, path, ref)
            
            # use_distributed_wrapper
            if trainer.use_distributed_wrapper:
                assert cfg.distributed_training.ddp_backend in {"c10d", "pytorch_ddp"}
                # logger.warning(f"USE {trainer.use_distributed_wrapper=}") --> Expect True
                single_model.to(trainer.device)

            # logger.warning(f"{next(single_model.parameters()).device=}")
            self.load_cache_checkpoint(trainer, single_model)
        else:
            logger.warning(f"NOT setup cached model as {self.args.cache_model_path=} not found")
    
    def load_cache_checkpoint(self, trainer, cache_model):
        assert PathManager.isfile(str(self.args.cache_model_path))
        # Load checkpoint
        cache_path = self.args.cache_model_path
        cache_overwrite_path = self.args.cache_overwrite_sharded_path
        logger.info(f"Preparing to load cache checkpoint {cache_path=}")
        is_distributed = trainer.data_parallel_world_size > 1

        load_on_all_ranks = (
            trainer.cfg.checkpoint.load_checkpoint_on_all_dp_ranks
            # TPUs don't support broadcast yet, so load checkpoints
            # on every worker for now
            or trainer.tpu
            # FSDP requires loading checkpoint shards on all ranks
            or (trainer.is_fsdp and trainer.cfg.distributed_training.use_sharded_state)
            or getattr(trainer.cfg.model, "base_layers", 0) > 0
            # or (trainer.cfg.distributed_training.save_sharded_state
            #     and cache_overwrite_path is not None and cache_overwrite_path.endswith("r$.pt")
            # )
        )

        if load_on_all_ranks or trainer.data_parallel_rank == 0:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                cache_path, load_on_all_ranks=load_on_all_ranks
            )
            # last_optim_state = state.get("last_optimizer_state", None)

            # If doing zero_sharding, do not broadcast global optimizer
            # state. Later we will broadcast sharded states to each rank
            # to avoid memory from exploding.
            if (
                not load_on_all_ranks
                and trainer.cfg.distributed_training.zero_sharding == "os"
                and "last_optimizer_state" in state
                and is_distributed
            ):
                state["last_optimizer_state"] = "SHARDED"
        else:
            # last_optim_state = None
            state = None

        if is_distributed and not load_on_all_ranks:
            state = distributed_utils.broadcast_object(
                state,
                src_rank=0,
                group=trainer.data_parallel_process_group,
                dist_device=trainer.device,
            )
            # if trainer.data_parallel_rank > 0:
            #     last_optim_state = state.get("last_optimizer_state", None)
        
        if cache_overwrite_path is not None and cache_overwrite_path != "":
            rank = trainer.data_parallel_rank
            if PathManager.isfile(cache_overwrite_path):
                _cache_overwrite_path = cache_overwrite_path
            else:
                _cache_overwrite_path = cache_overwrite_path.replace("r$.pt", f'r{rank}.pt')
            if PathManager.isfile(_cache_overwrite_path):
                with open(_cache_overwrite_path, 'rb') as shard_s_f:
                    sharded_states = torch.load(shard_s_f, map_location=torch.device("cpu"))['sharded_states']
                    logger.warning(f'{rank}: Load cached sharded states at {_cache_overwrite_path}')
                    utils.overwrite_sharded_states(sharded_states, state['model'])

        # load model parameters
        try:
            cache_model.load_state_dict(
                state["model"], strict=True, model_cfg=trainer.cfg.model
            )
            # save memory for later steps
            del state
        except Exception:
            raise Exception(
                "Cannot load model parameters from checkpoint {}; "
                "please ensure that the architectures match.".format(cache_model)
            )
    
    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)
    
    def build_generator(self, models, args, eos=None, lang=None, seq_gen_cls=None, extra_gen_cls_kwargs=None):
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                eos=self.tgt_dict.index("[{}]".format(self.args.target_lang)),
            )
        else:
            # FIXME: top_frequency generator not use!

            # return OBTmBARTSequenceGenerator(
            #     models,
            #     self.target_dictionary,
            #     beam_size=getattr(args, "beam", 5),
            #     max_len_a=getattr(args, "max_len_a", 0),
            #     max_len_b=getattr(args, "max_len_b", 200),
            #     min_len=getattr(args, "min_len", 1),
            #     normalize_scores=(not getattr(args, "unnormalized", False)),
            #     len_penalty=getattr(args, "lenpen", 1),
            #     unk_penalty=getattr(args, "unkpen", 0),
            #     temperature=getattr(args, "temperature", 1.0),
            #     match_source_len=getattr(args, "match_source_len", False),
            #     no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            #     # eos=eos if eos is not None else self.tgt_dict.index("[{}]".format(self.args.target_lang)),
            #     eos=eos,
            #     top_frequency=getattr(args, "top_frequency", 1),
            #     top_freq_dict=self.freq_dicts[lang] if self.freq_dicts is not None and lang in self.freq_dicts else None,
            # )
            # normal
            # if not getattr(args, "keep_inference_langtok", False):
            #     _, tgt_langtok_spec = self.args.langtoks["main"]
            #     if tgt_langtok_spec:
            #         tgt_lang_tok = self.data_manager.get_decoder_langtok(
            #             self.args.target_lang, tgt_langtok_spec
            #         )
            #         extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
            #         extra_gen_cls_kwargs["symbols_to_strip_from_output"] = {tgt_lang_tok}

            return super().build_generator(
                models, args, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
            )

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None, infer_from_smp=False, get_prefix_tokens=False
    ):
        with torch.no_grad():
            if DataDomainSpec.main.value in sample:
                # convert sample -> sample[main]
                sample = sample[DataDomainSpec.main.value]
            _, tgt_langtok_spec = self.args.langtoks["main"]
            if not self.args.lang_tok_replacing_bos_eos:
                if prefix_tokens is None:
                    if infer_from_smp:
                        prefix_tokens = sample["net_input"]["prev_output_tokens"][:, 1:2]
                    elif tgt_langtok_spec:
                        tgt_lang_tok = self.data_manager.get_decoder_langtok(
                            self.args.target_lang, tgt_langtok_spec
                        )
                        src_tokens = sample["net_input"]["src_tokens"]
                        bsz = src_tokens.size(0)
                        prefix_tokens = (
                            torch.LongTensor([[tgt_lang_tok]]).expand(bsz, 1).to(src_tokens)
                        )
                generated = generator.generate(
                    models,
                    sample,
                    prefix_tokens=prefix_tokens,
                    constraints=constraints,
                )
                if get_prefix_tokens:
                    return generated, prefix_tokens
                else:
                    return generated
            else:
                assert not infer_from_smp, f'{infer_from_smp=} not impl'
                bos_token = self.data_manager.get_decoder_langtok(
                    self.args.target_lang, tgt_langtok_spec
                )
                generated = generator.generate(
                    models,
                    sample,
                    prefix_tokens=prefix_tokens,
                    bos_token=bos_token
                    if tgt_langtok_spec
                    else self.target_dictionary.eos(),
                )
                if get_prefix_tokens:
                    return generated, bos_token
                else:
                    return generated
            
    def _inference_with_bleu(self, generator, sample, model, eval_tokenized_bleu=None):
        import sacrebleu
        langs = self.langs
        ignore_lang_indices = [self.dictionary.index(self.data_manager.get_lang_tok(x)) for x in langs]
        extra_symbols_to_ignore = [generator.eos] + ignore_lang_indices
        
        def decode(toks, escape_unk=False):
            s = self.dictionary.string(
                toks.int().cpu(),
                getattr(self.args, "eval_bleu_remove_bpe", None),
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
                extra_symbols_to_ignore=extra_symbols_to_ignore,
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out, prefix_tokens = self.inference_step(
            generator, [model], sample, 
            prefix_tokens=None,
            infer_from_smp=True,
            get_prefix_tokens=True
        )
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.dictionary.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )

        if getattr(self.args, "eval_bleu_print_samples", False):
            # lang = self.dictionary[generator.eos]
            bos_lang_toks = prefix_tokens.squeeze(1)
            lang = self.dictionary[bos_lang_toks[0]]
            logger.info(f"example hypothesis {lang}: " + hyps[0])
            logger.info(f"example reference  {lang}: " + refs[0])

        eval_tokenized_bleu = getattr(self.args, "eval_tokenized_bleu", False) if eval_tokenized_bleu is None else eval_tokenized_bleu
        if eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])
    
    def valid_step(self, sample, model, criterion):
        # super(TranslationTask, self) to bypass TranslationTask valid_step
        # logger.warning(f'{sample.keys()}')
        sample = sample[DataDomainSpec.main.value]
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        # NOTE forward blue, temporarily disabled
        if getattr(self.args, "eval_bleu", False):
            # forward mt
            # mono_langs = self.mono_langs
            bleu = self._inference_with_bleu(self.bleu_sequence_generator, sample, model)
            # logging_output["_bleu_sys_len"] = bleu.sys_len
            # logging_output["_bleu_ref_len"] = bleu.ref_len
            logging_output["_bleu_sys_len"] = int(bleu.sys_len)
            logging_output["_bleu_ref_len"] = int(bleu.ref_len)

            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]

            # NOTE backward blue, temporarily disabled
            # if self.valid_mt_rev:
            #     bwd_sample = self.sample_switch_srctgt(sample)
            #     bwd_bleu = self._inference_with_bleu(
            #         self.bleu_sequence_generator[mono_langs[1]], bwd_sample, model,
            #         eval_tokenized_bleu=getattr(self.args, "bwd_eval_tokenized_bleu", False),
            #     )
            #     logging_output["_bwd_bleu_sys_len"] = bwd_bleu.sys_len
            #     logging_output["_bwd_bleu_ref_len"] = bwd_bleu.ref_len
            #     # we split counts into separate entries so that they can be
            #     # summed efficiently across workers using fast-stat-sync
            #     assert len(bwd_bleu.counts) == EVAL_BLEU_ORDER
            #     for i in range(EVAL_BLEU_ORDER):
            #         logging_output["_bwd_bleu_counts_" + str(i)] = bwd_bleu.counts[i]
            #         logging_output["_bwd_bleu_totals_" + str(i)] = bwd_bleu.totals[i]
        return loss, sample_size, logging_output
    
    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if getattr(self.args, "eval_bleu", False):
            from sacrebleu.metrics import BLEU
            def sum_logs(key):
                import torch
                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu

                    fn_sig = inspect.getfullargspec(BLEU.compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = BLEU.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=int(meters["_bleu_sys_len"].sum),
                        ref_len=int(meters["_bleu_ref_len"].sum),
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)

        agg_loss, agg_sample_size = 0.0, 0.0
        agg_logging_output: Dict[str, float] = defaultdict(float)

        weights = {
            DataDomainSpec.main.value: self.lambda_main(update_num),
            DataDomainSpec.bt.value: self.lambda_bt(update_num),
            DataDomainSpec.ct.value: self.lambda_ct(update_num),
        }
        log_keys = {
            DataDomainSpec.main.value: f"{DataDomainSpec.convert_key(DataDomainSpec.main.value)}_",
            DataDomainSpec.bt.value: f"{DataDomainSpec.convert_key(DataDomainSpec.bt.value)}_",
            DataDomainSpec.ct.value: f"{DataDomainSpec.convert_key(DataDomainSpec.ct.value)}_",
        }
        smp_keys = sample.keys()
        for subtype in smp_keys:
            smp = sample[subtype]
            if weights[subtype] == 0:
                continue

            if subtype == DataDomainSpec.bt.value:
                with torch.no_grad():
                    model.eval()
                    src_bos_toks = self.data_manager.backtranslate_multi_sample(self.bt_sequence_generator, smp)
                    self.data_manager.display_samples_once_in_a_while(smp, src_bos_toks=src_bos_toks, prefix="BT")
                    model.train()
            
            if subtype == DataDomainSpec.ct.value:
                with torch.no_grad():
                    model.eval()
                    src_bos_toks = self.data_manager.crosstranslate_multi_sample(self.bt_sequence_generator, smp)
                    self.data_manager.display_samples_once_in_a_while(smp, src_bos_toks=src_bos_toks, prefix="CT")
                    model.train()

            with torch.autograd.profiler.record_function("forward"):
                with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                    loss, sample_size, logging_output = criterion(model, smp)
                    loss *= weights[subtype]
            if ignore_grad:
                loss *= 0
            with torch.autograd.profiler.record_function("backward"):
                optimizer.backward(loss)
            agg_loss += loss.item()
            agg_sample_size += sample_size
            for k in logging_output:
                agg_logging_output[log_keys[subtype] + k] += logging_output[k]
                agg_logging_output[k] += logging_output[k]
        return agg_loss, agg_sample_size, agg_logging_output


@register_task("translation_multi_triangle_umt_simple_epoch")
class TranslationMultiTriangleUmtSimpleEpochTask(TranslationMultiUmtSimpleEpochTask):

    def build_model(self, args):
        model = super(TranslationMultiUmtSimpleEpochTask, self).build_model(args)

        cache_path = str(getattr(args, "cache_model_path", None))
        if PathManager.isfile(cache_path):
            # FIXME: previous version of translation_multi_umt_cachedgen_simple_epoch
            #   cast TranslationMultiUmtSimpleEpochTask super -> build_model
            #       which means it is standard TranslationMultiSimpleEpochTask
            cache_model = super(TranslationMultiSimpleEpochTask, self).build_model(args)
            for param in cache_model.parameters():
                param.requires_grad = False
            logger.info(f"Use Secondary model cached from {cache_path}...., no gradient here.")
        else:
            cache_model = model

        self.bt_sequence_generator = MultiPrefixSequenceGenerator(
            [cache_model],
            tgt_dict=self.dictionary,
            beam_size=1,
            max_len_a=1.3,
            max_len_b=5,
            min_len=5,
            # keep 1 to be able to prepend bos
            max_len=model.max_decoder_positions() - 1,
        )
        if getattr(args, "eval_bleu", False):
            assert getattr(args, "eval_bleu_detok", None) is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(getattr(args, "eval_bleu_detok_args", "{}") or "{}")
            self.tokenizer = encoders.build_tokenizer(
                Namespace(
                    tokenizer=getattr(args, "eval_bleu_detok", None), **detok_args
                )
            )

            gen_args = json.loads(getattr(args, "eval_bleu_args", "{}") or "{}")
            self.bleu_sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )

        return model

    def _single_train_step(self, model, smp, weight, subtype, agg_loss, agg_sample_size, 
        optimizer, criterion, ignore_grad, agg_logging_output, log_keys):
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, smp)
                loss *= weight
                
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        agg_loss += loss.item()
        agg_sample_size += sample_size
        for k in logging_output:
            agg_logging_output[log_keys[subtype] + k] += logging_output[k]
            agg_logging_output[k] += logging_output[k]

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)

        agg_loss, agg_sample_size = 0.0, 0.0
        agg_logging_output: Dict[str, float] = defaultdict(float)

        weights = {
            DataDomainSpec.main.value: self.lambda_main(update_num),
            DataDomainSpec.bt.value: self.lambda_bt(update_num),
            DataDomainSpec.ct.value: self.lambda_ct(update_num),
        }
        log_keys = {
            DataDomainSpec.main.value: f"{DataDomainSpec.convert_key(DataDomainSpec.main.value)}_",
            DataDomainSpec.bt.value: f"{DataDomainSpec.convert_key(DataDomainSpec.bt.value)}_",
            DataDomainSpec.ct.value: f"{DataDomainSpec.convert_key(DataDomainSpec.ct.value)}_",
        }
        smp_keys = sample.keys()

        for subtype in smp_keys:
            smp = sample[subtype]
            if weights[subtype] == 0:
                continue

            if subtype == DataDomainSpec.bt.value:
                with torch.no_grad():
                    model.eval()
                    # src_bos_toks = self.data_manager.backtranslate_multi_sample(self.bt_sequence_generator, smp)
                    # self.data_manager.display_samples_once_in_a_while(smp, src_bos_toks=src_bos_toks, prefix="BT")
                    src_bos_toks_list, smp_list = self.data_manager.multi_backtranslate_multi_sample(
                        self.bt_sequence_generator, smp, num=2)
                    tri_smp_list, tri_in_bos_toks_list, tri_out_bos_toks_list = self.data_manager.multi_triangle_smp_from_smp_list(
                        src_bos_toks_list, smp_list)
                        
                    smp_list.extend(tri_smp_list)
                    src_bos_toks_list.extend(tri_in_bos_toks_list)
                    for i, (smp, src_bos_toks) in enumerate(zip(smp_list, src_bos_toks_list)):
                        self.data_manager.display_samples_once_in_a_while(smp, src_bos_toks=src_bos_toks, prefix=f"BT-{i}")
                    model.train()
                
                for i, smp in enumerate(smp_list):
                    self._single_train_step(model, smp, weights[subtype], subtype, agg_loss, agg_sample_size,
                        optimizer, criterion, ignore_grad, agg_logging_output, log_keys
                    )
            
            if subtype == DataDomainSpec.ct.value:
                with torch.no_grad():
                    model.eval()
                    src_bos_toks = self.data_manager.crosstranslate_multi_sample(self.bt_sequence_generator, smp)
                    self.data_manager.display_samples_once_in_a_while(smp, src_bos_toks=src_bos_toks, prefix="CT")
                    model.train()
                self._single_train_step(model, smp, weights[subtype], subtype, agg_loss, agg_sample_size,
                    optimizer, criterion, ignore_grad, agg_logging_output, log_keys
                )

        return agg_loss, agg_sample_size, agg_logging_output


@register_task("search_db_translation_multi_umt_simple_epoch")
class SearchDBTranslationMultiUmtSimpleEpochTask(TranslationMultiUmtSimpleEpochTask):
    def setup_data_manager(self, args, langs, dicts):
        return SearchDBMultilingualUmtDatasetManager.setup_data_manager(
            args, self.lang_pairs, langs, dicts, self.sampling_method
        )
    
    @classmethod
    def setup_task(cls, args, **kwargs):
        langs, dicts, training = SearchDBMultilingualUmtDatasetManager.prepare(
           cls.load_dictionary, args, **kwargs
        )
        freq_dicts = cls.build_frequency_dicts(args, langs, dicts, training)
        return cls(args, langs, dicts, training, freq_dicts=freq_dicts)
    
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationMultiUmtSimpleEpochTask.add_args(parser)
        SearchDBMultilingualUmtDatasetManager.add_args(parser)
        parser.add_argument(
            "--gen-search-alternative",
            type=str,
            default='no',
            choices=['no', 'replace', 'logprobs'],
            help="During inference, search for alternative in the mono data",
        )
        parser.add_argument
    
    @property
    def gen_search_alternative(self):
        return self.args.gen_search_alternative
    
    def _search_replace_generated(self, generator, generated, sample):
        prefix_removal = 1  # FIXME: this one is hard-coded
        generated_tokens = [gn[0]['tokens'][prefix_removal:] for gn in generated]
        _src_tokens = sample['net_input']['src_tokens']
        tgt_lang_tok = self.data_manager.get_decoder_langtok(
            self.args.target_lang, self.args.langtoks["main"][1]
        )
        src_bos_toks = (
            torch.LongTensor([tgt_lang_tok]).expand(_src_tokens.size(0)).to(_src_tokens)
        )
        generated_embeds = self.data_manager.generate_encoder_avg_pool(generator, generated, sample)
        generated_lang_ids = src_bos_toks
        generated_tokens, generated_embeds, generated_lang_ids = utils.move_to_cpu(
            (generated_tokens, generated_embeds, generated_lang_ids)
        )
        search_tokens_list, search_callback = self.data_manager.search_db_on_embeds(
            generated_embeds, generated_lang_ids, generated_tokens,
            top_k=self.data_manager.args.db_search_topk, 
            mode=0,
            # is_add_ref_indices=True,
        )
        search_callback()
        # search_tokens, search_distances, search_dataset_indices = search_tokens_list[0]
        search_tokens, search_distances = search_tokens_list[0]

        new_generated = copy.deepcopy(generated)
        lang_ids = generated_lang_ids.unsqueeze(1)
        for gn, ngnin, stok, lang_id in zip(generated, new_generated, search_tokens, lang_ids):
            ngnin[0]['tokens'] = torch.cat([lang_id, stok], 0)
            ngnin[0]['original_tokens'] = gn[0]['tokens']
        return new_generated
    
    def find_alternative(self, generator, models, sample, generated):
        if self.gen_search_alternative == "no":
            return generated
        elif self.gen_search_alternative == 'replace':
            self.data_manager.load_mbt_datasets_from_database("train")
            generated = self._search_replace_generated(generator, generated, sample)
            return generated
        else:
            raise ValueError
    
    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None, infer_from_smp=False, get_prefix_tokens=False
    ):
        with torch.no_grad():
            if DataDomainSpec.main.value in sample:
                # convert sample -> sample[main]
                sample = sample[DataDomainSpec.main.value]
            _, tgt_langtok_spec = self.args.langtoks["main"]
            if not self.args.lang_tok_replacing_bos_eos:
                if prefix_tokens is None:
                    if infer_from_smp:
                        prefix_tokens = sample["net_input"]["prev_output_tokens"][:, 1:2]
                    elif tgt_langtok_spec:
                        tgt_lang_tok = self.data_manager.get_decoder_langtok(
                            self.args.target_lang, tgt_langtok_spec
                        )
                        src_tokens = sample["net_input"]["src_tokens"]
                        bsz = src_tokens.size(0)
                        prefix_tokens = (
                            torch.LongTensor([[tgt_lang_tok]]).expand(bsz, 1).to(src_tokens)
                        )
                generated = generator.generate(
                    models,
                    sample,
                    prefix_tokens=prefix_tokens,
                    constraints=constraints,
                )
                generated = self.find_alternative(generator, models, sample, generated)
                if get_prefix_tokens:
                    return generated, prefix_tokens
                else:
                    return generated
            else:
                # assert not infer_from_smp, f'{infer_from_smp=} not impl'
                # bos_token = self.data_manager.get_decoder_langtok(
                #     self.args.target_lang, tgt_langtok_spec
                # )
                # generated = generator.generate(
                #     models,
                #     sample,
                #     prefix_tokens=prefix_tokens,
                #     bos_token=bos_token
                #     if tgt_langtok_spec
                #     else self.target_dictionary.eos(),
                # )
                # if get_prefix_tokens:
                #     return generated, bos_token
                # else:
                #     return generated
                raise NotImplementedError
    
    # =========================================================================

    def _single_train_step(self, model, smp, weight, subtype, agg_loss, agg_sample_size, 
        optimizer, criterion, ignore_grad, agg_logging_output, log_keys):
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, smp)
                loss *= weight
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        agg_loss += loss.item()
        agg_sample_size += sample_size
        for k in logging_output:
            agg_logging_output[log_keys[subtype] + k] += logging_output[k]
            agg_logging_output[k] += logging_output[k]
    
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)

        agg_loss, agg_sample_size = 0.0, 0.0
        agg_logging_output: Dict[str, float] = defaultdict(float)

        weights = {
            DataDomainSpec.main.value: self.lambda_main(update_num),
            DataDomainSpec.bt.value: self.lambda_bt(update_num),
            DataDomainSpec.ct.value: self.lambda_ct(update_num),
        }
        log_keys = {
            DataDomainSpec.main.value: f"{DataDomainSpec.convert_key(DataDomainSpec.main.value)}_",
            DataDomainSpec.bt.value: f"{DataDomainSpec.convert_key(DataDomainSpec.bt.value)}_",
            DataDomainSpec.ct.value: f"{DataDomainSpec.convert_key(DataDomainSpec.ct.value)}_",
        }
        smp_keys = sample.keys()

        for subtype in smp_keys:
            smp = sample[subtype]
            if weights[subtype] == 0:
                continue

            if subtype == DataDomainSpec.bt.value:
                with torch.no_grad():
                    model.eval()
                    src_bos_toks_list, smp_list, callback = self.data_manager.backtranslate_multi_sample_km_search(
                        self.bt_sequence_generator, smp, topk=1
                    )
                    self.data_manager.display_samples_once_in_a_while(smp_list[0], src_bos_toks=src_bos_toks_list[0], prefix=f"BT")
                
                    model.train()
                
                self._single_train_step(model, smp_list[0], weights[subtype], subtype, agg_loss, agg_sample_size,
                    optimizer, criterion, ignore_grad, agg_logging_output, log_keys
                )
                callback()
                assert len(smp_list) > 1
                self.data_manager.display_searched_samples_once_in_a_while(
                    smp_list[0], smp_list[1], src_bos_toks=src_bos_toks_list[1], prefix=f"BT-search")

                self._single_train_step(model, smp_list[1], weights[subtype], subtype, agg_loss, agg_sample_size,
                    optimizer, criterion, ignore_grad, agg_logging_output, log_keys
                )
            
            if subtype == DataDomainSpec.ct.value:
                with torch.no_grad():
                    model.eval()
                    src_bos_toks = self.data_manager.crosstranslate_multi_sample(self.bt_sequence_generator, smp)
                    self.data_manager.display_samples_once_in_a_while(smp, src_bos_toks=src_bos_toks, prefix="CT")
                    model.train()
                self._single_train_step(model, smp, weights[subtype], subtype, agg_loss, agg_sample_size,
                    optimizer, criterion, ignore_grad, agg_logging_output, log_keys
                )

        return agg_loss, agg_sample_size, agg_logging_output


def aggregate_smp_list(smp_list):
    if len(smp_list) == 1:
        return smp_list[0]
    if len(smp_list) == 0:
        raise ValueError(f'len smp_list empty')

    def _aggregate_item(items):
        first = items[0]
        if torch.is_tensor(first):
            return torch.cat(items, 0)
        elif isinstance(first, (int, float)):
            return sum(items)
        elif isinstance(first, dict):
            return {k: _aggregate_item([x[k] for x in items]) for k in first.keys()}
        elif isinstance(first, (list, tuple)):
            first_len = len(first)
            return type(first)([_aggregate_item([x[i] for x in items]) for i in range(first_len)])
        else:
            raise ValueError(f'unexpected type {type(first)} to aggregate')

    keys = list(smp_list[0].keys())
    agg_smp = {
        k: _aggregate_item([smp_list[i] for i in range(len(smp_list))])
        for k in keys
    }
    return agg_smp


def dist_reroute_samples(smp_list, id_list, rank_to_id, group=None, recompute_smp_data=None, original_device=None):
    # reroute items from total smp_list with identifier to specific rank according to
    group = group or distributed_utils.get_global_group()
    cur_rank = distributed_utils.get_rank(group=group)
    world_size = distributed_utils.get_world_size(group=group)
    smp_list = smp_list if isinstance(smp_list, list) else [smp_list]
    id_list = id_list if isinstance(id_list, list) else [id_list]
    assert isinstance(rank_to_id, dict)
    id_to_keep = rank_to_id[cur_rank]
    smp_list_size = len(smp_list)

    # get original devices
    # FIXME: this is hardcoded
    original_device = original_device or smp_list[0]['net_input']['src_tokens'].device
    from fairseq.utils import move_to_cuda, move_to_cpu

    # itertools
    agg_smp = aggregate_smp_list(smp_list)
    agg_ids = torch.cat(id_list, 0)

    # distributed gather
    dist_agg_smp_list = distributed_utils.all_gather_list(agg_smp, group=group)
    dist_agg_smp = aggregate_smp_list(dist_agg_smp_list)
    dist_agg_ids = torch.cat(distributed_utils.all_gather(agg_ids, group=group), 0)

    filtered_indices = torch.arange(len(dist_agg_ids)).to(dist_agg_ids)[dist_agg_ids == id_to_keep]
    # FIXME: check for possibility of empty tensors
    filtered_ids = dist_agg_ids[filtered_indices]
    filtered_smp = apply_to_sample(lambda x: x[filtered_indices], dist_agg_smp)
    if callable(recompute_smp_data):
        filtered_smp = recompute_smp_data(filtered_smp)
    
    if "cuda" in original_device.type:
        filtered_smp = move_to_cuda(filtered_smp)
    
    # truncate smp_list_size (equally)
    if smp_list_size == 1:
        return [filtered_ids], [filtered_smp]
    else:
        chunk_size = filtered_ids.size(0) // smp_list_size
        filtered_ids_list = [
            filtered_ids[i * chunk_size:(i + 1) * chunk_size] if i < smp_list_size - 1 else
                filtered_ids[i * chunk_size:]
            for i in range(smp_list_size)
        ]
        filtered_smp_list = [
            apply_to_sample(
                (lambda x: x[i * chunk_size:(i + 1) * chunk_size]) if i < smp_list_size - 1 else
                    (lambda x: x[i * chunk_size:]),
                filtered_smp
            )
            for i in range(smp_list_size)
        ]
        if callable(recompute_smp_data):
            filtered_smp_list = [recompute_smp_data(x) for x in filtered_smp_list]
        return filtered_ids_list, filtered_smp_list


# dist_reroute_samples
def langpair_dist_reroute_samples(smp_list, id_list, rank_to_id, group=None):
    # recompute_smp_data=None, original_device=None
    smp_list = smp_list if isinstance(smp_list, list) else [smp_list]
    id_list = id_list if isinstance(id_list, list) else [id_list]
    original_device = smp_list[0]['net_input']['src_tokens'].device
    
    def recompute_smp_data(_smp):
        nsentences = _smp['target'].size(0)
        ntokens = _smp['net_input']['src_lengths'].sum().item()
        _smp['nsentences'] = nsentences
        _smp['ntokens'] = ntokens
        return _smp

    return dist_reroute_samples(
        smp_list, id_list, rank_to_id, group, recompute_smp_data, original_device
    )



@register_task("translation_multi_umt_cachedgen_simple_epoch")
class TranslationMultiUmtCachedGenSimpleEpochTask(TranslationMultiUmtSimpleEpochTask):
    pass


@register_task("translation_multi_gennp_simple_epoch")
class TranslationMultiGenNpassSimpleEpochTask(TranslationMultiSimpleEpochTask):
    
    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        models = MultiDecPassEnsembleModel(models, n_pass=getattr(args, "n_pass", 2), seed=getattr(args, "seed", 1))
        return super().build_generator(
            models, args, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )
    

@register_task("gpusep_translation_multi_umt_simple_epoch")
class GpuSepTranslationMultiUmtSimpleEpochTask(TranslationMultiUmtSimpleEpochTask):
    
    def setup_data_manager(self, args, langs, dicts):
        return GpuSepMultilingualUmtDatasetManager.setup_data_manager(
            args, self.lang_pairs, langs, dicts, self.sampling_method
        )
    
    @classmethod
    def setup_task(cls, args, **kwargs):
        langs, dicts, training = GpuSepMultilingualUmtDatasetManager.prepare(
           cls.load_dictionary, args, **kwargs
        )
        freq_dicts = cls.build_frequency_dicts(args, langs, dicts, training)
        return cls(args, langs, dicts, training, freq_dicts=freq_dicts)
    
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationMultiUmtSimpleEpochTask.add_args(parser)
        GpuSepMultilingualUmtDatasetManager.add_args(parser)
    

@register_task("gpusep_ffnencdec_translation_multi_umt_simple_epoch")
class GpuSepFfnEncDecTranslationMultiUmtSimpleEpochTask(GpuSepTranslationMultiUmtSimpleEpochTask):
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)

        agg_loss, agg_sample_size = 0.0, 0.0
        agg_logging_output: Dict[str, float] = defaultdict(float)

        weights = {
            DataDomainSpec.main.value: self.lambda_main(update_num),
            DataDomainSpec.bt.value: self.lambda_bt(update_num),
            DataDomainSpec.ct.value: self.lambda_ct(update_num),
        }
        log_keys = {
            DataDomainSpec.main.value: f"{DataDomainSpec.convert_key(DataDomainSpec.main.value)}_",
            DataDomainSpec.bt.value: f"{DataDomainSpec.convert_key(DataDomainSpec.bt.value)}_",
            DataDomainSpec.ct.value: f"{DataDomainSpec.convert_key(DataDomainSpec.ct.value)}_",
        }
        smp_keys = sample.keys()
        for subtype in smp_keys:
            smp = sample[subtype]
            if weights[subtype] == 0:
                continue

            if subtype == DataDomainSpec.bt.value:
                with torch.no_grad():
                    model.eval()
                    smp['net_input']['bt_prev_output_tokens'] = smp['net_input']['prev_output_tokens']
                    src_bos_toks = self.data_manager.backtranslate_multi_sample(self.bt_sequence_generator, smp)
                    self.data_manager.display_samples_once_in_a_while(smp, src_bos_toks=src_bos_toks, prefix="BT")
                    model.train()
                    smp['net_input'].pop('bt_prev_output_tokens')
            
            if subtype == DataDomainSpec.ct.value:
                with torch.no_grad():
                    model.eval()
                    src_bos_toks = self.data_manager.crosstranslate_multi_sample(self.bt_sequence_generator, smp)
                    self.data_manager.display_samples_once_in_a_while(smp, src_bos_toks=src_bos_toks, prefix="CT")
                    model.train()

            with torch.autograd.profiler.record_function("forward"):
                with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                    loss, sample_size, logging_output = criterion(model, smp)
                    loss *= weights[subtype]
            if ignore_grad:
                loss *= 0
            with torch.autograd.profiler.record_function("backward"):
                optimizer.backward(loss)
            agg_loss += loss.item()
            agg_sample_size += sample_size
            for k in logging_output:
                agg_logging_output[log_keys[subtype] + k] += logging_output[k]
                agg_logging_output[k] += logging_output[k]
        return agg_loss, agg_sample_size, agg_logging_output