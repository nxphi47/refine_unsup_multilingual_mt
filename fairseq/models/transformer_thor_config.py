import re
from dataclasses import dataclass, field, fields
from typing import Any, List, Optional

from omegaconf import II

from fairseq import utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models.bart.hub_interface import BARTHubInterface
from fairseq.models.bart.model import BARTClassificationHead
from fairseq.models.transformer.transformer_legacy import base_architecture

from fairseq.models.transformer import (
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
    TransformerConfig
)

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import transformer_layer
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor
from fairseq.models.transformer import (
    TransformerConfig,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params

import logging

logger = logging.getLogger(__name__)


@dataclass
class ThorTransformerConfig(TransformerConfig):
    num_experts: int = field(
        default=2,
        metadata={"help": "Number of experts"},
    )
    inference_level: int = field(
        default=0,
        metadata={"help": "0 for token level, 1 for sentence level"},
    )
    
    expert_increment: int = field(
        default=2,
        metadata={"help": "Increment to use ffn expert"},
    )
    
    no_expert_layer: str = field(
        default=None,
        metadata={"help": "comma separated where no expert is allowed"},
    )
    


@dataclass
class LangFFNListTransformerConfig(ThorTransformerConfig):
    langs_for_experts: str = field(
        default="en_XX",
        metadata={"help": "comma separated langs "},
    )


@dataclass
class GpuFFNSharedLangThorTransformerConfig(ThorTransformerConfig):
    shared_lang: str = field(
        default="en_XX",
        metadata={"help": "Shared lang"},
    )


@dataclass
class GpuAttSharedLangTransformerConfig(ThorTransformerConfig):
    shared_lang: str = field(
        default="en_XX",
        metadata={"help": "Shared lang"},
    )
    no_fc_expert: bool = field(
        default=False,
        metadata={"help": "disable fc expert"},
    )
    self_att_expert: bool = field(
        default=False,
        metadata={"help": "enable expert"},
    )
    enc_att_expert: bool = field(
        default=False,
        metadata={"help": "enable expert"},
    )


def convert_transformer_base_to_transformer(config_class, base_architecture, assign_fns=None):
    if assign_fns is not None:
        assign_fns = assign_fns if isinstance(assign_fns, list) else [assign_fns]
    else:
        assign_fns = []

    def _convertor(root_cls):
        class _transformer(root_cls):
            def __init__(self, args, encoder, decoder):
                cfg = config_class.from_namespace(args)
                super().__init__(cfg, encoder, decoder)
                self.args = args
            
            @staticmethod
            def add_args(parser):
                """Add model-specific arguments to the parser."""
                # we want to build the args recursively in this case.
                gen_parser_from_dataclass(
                    parser, config_class(), 
                    # NEED TO SET delete_default, 
                    # otherwise it will cause default to be written first and model_architectures below won't work
                    delete_default=True, 
                    with_prefix=""
                )

            @classmethod
            def add_args(cls, parser):
                """Add model-specific arguments to the parser."""
                # we want to build the args recursively in this case.
                # do not set defaults so that settings defaults from various architectures still works
                gen_parser_from_dataclass(
                    parser, config_class(), delete_default=True, with_prefix=""
                )
            
            @classmethod
            def build_model(cls, args, task):
                """Build a new model instance."""

                # make sure all arguments are present in older models
                base_architecture(args)

                if args.encoder_layers_to_keep:
                    args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
                if args.decoder_layers_to_keep:
                    args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

                if getattr(args, "max_source_positions", None) is None:
                    args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
                if getattr(args, "max_target_positions", None) is None:
                    args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

                src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

                if args.share_all_embeddings:
                    if src_dict != tgt_dict:
                        raise ValueError("--share-all-embeddings requires a joined dictionary")
                    if args.encoder_embed_dim != args.decoder_embed_dim:
                        raise ValueError(
                            "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                        )
                    if args.decoder_embed_path and (
                        args.decoder_embed_path != args.encoder_embed_path
                    ):
                        raise ValueError(
                            "--share-all-embeddings not compatible with --decoder-embed-path"
                        )
                    args.share_decoder_input_output_embed = True

                if getattr(args, "offload_activations", False):
                    args.checkpoint_activations = True  # offloading implies checkpointing

                if not args.share_all_embeddings:
                    args.min_params_to_wrap = getattr(
                        args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
                    )
                cfg = config_class.from_namespace(args)
                model = super().build_model(cfg, task)
                if hasattr(model, "setup_ddp_ignore_params"):
                    # try:
                    model.setup_ddp_ignore_params()
                    # except AttributeError as e:
                    #     logger.warning(f'model.setup_ddp_ignore_params not found, skip this step')
                return model

            @classmethod
            def build_embedding(cls, args, dictionary, embed_dim, path=None):
                return super().build_embedding(
                    config_class.from_namespace(args), dictionary, embed_dim, path
                )

            @classmethod
            def build_encoder(cls, args, src_dict, embed_tokens):
                return super().build_encoder(
                    config_class.from_namespace(args), src_dict, embed_tokens
                )

            @classmethod
            def build_decoder(cls, args, tgt_dict, embed_tokens):
                return super().build_decoder(
                    config_class.from_namespace(args), tgt_dict, embed_tokens
                )
        
        _transformer.__name__ = root_cls.__name__
        for _fn in assign_fns:
            setattr(_transformer, _fn.__name__, _fn)
        return _transformer
    
    return _convertor


def convert_transformer_to_bart_cls(cus_upgrade_state_dict_named=None, assign_fns=None):
    if assign_fns is not None:
        assign_fns = assign_fns if isinstance(assign_fns, list) else [assign_fns]
    else:
        assign_fns = []
    def _convertor(root_cls):
        class _bartCls(root_cls):
            __jit_unused_properties__ = ["supported_targets"]
            def __init__(self, args, encoder, decoder):
                super().__init__(args, encoder, decoder)

                # We follow BERT's random weight initialization
                self.apply(init_bert_params)

                self.classification_heads = nn.ModuleDict()
                if hasattr(self.encoder, "dictionary"):
                    self.eos: int = self.encoder.dictionary.eos()

            @staticmethod
            def add_args(parser):
                super(root_cls, root_cls).add_args(parser)
                parser.add_argument(
                    "--pooler-dropout",
                    type=float,
                    metavar="D",
                    help="dropout probability in the masked_lm pooler layers",
                )
                parser.add_argument(
                    "--pooler-activation-fn",
                    choices=utils.get_available_activation_fns(),
                    help="activation function to use for pooler layer",
                )
                parser.add_argument(
                    "--spectral-norm-classification-head",
                    action="store_true",
                    help="Apply spectral normalization on the classification head",
                )
            
            @property
            def supported_targets(self):
                return {"self"}
            
            def forward(
                self,
                src_tokens,
                src_lengths,
                prev_output_tokens,
                features_only: bool = False,
                classification_head_name: Optional[str] = None,
                token_embeddings: Optional[torch.Tensor] = None,
                return_all_hiddens: bool = True,
                alignment_layer: Optional[int] = None,
                alignment_heads: Optional[int] = None,
            ):
                if classification_head_name is not None:
                    features_only = True

                encoder_out = self.encoder(
                    src_tokens,
                    src_lengths=src_lengths,
                    token_embeddings=token_embeddings,
                    return_all_hiddens=return_all_hiddens
                )
                x, extra = self.decoder(
                    prev_output_tokens,
                    encoder_out=encoder_out,
                    features_only=features_only,
                    alignment_layer=alignment_layer,
                    alignment_heads=alignment_heads,
                    src_lengths=src_lengths,
                    return_all_hiddens=return_all_hiddens,
                )
                eos: int = self.eos
                if classification_head_name is not None:
                    sentence_representation = x[
                        src_tokens.eq(eos), :
                    ].view(x.size(0), -1, x.size(-1))[:, -1, :]
                    for k, head in self.classification_heads.items():
                        # for torch script only supports iteration
                        if k == classification_head_name:
                            x = head(sentence_representation)
                            break
                return x, extra
            
            @classmethod
            def from_pretrained(
                cls,
                model_name_or_path,
                checkpoint_file="model.pt",
                data_name_or_path=".",
                bpe="gpt2",
                sample_break_mode="eos",
                **kwargs,
            ):
                from fairseq import hub_utils

                x = hub_utils.from_pretrained(
                    model_name_or_path,
                    checkpoint_file,
                    data_name_or_path,
                    archive_map=cls.hub_models(),
                    bpe=bpe,
                    load_checkpoint_heads=True,
                    sample_break_mode=sample_break_mode,
                    **kwargs,
                )
                return BARTHubInterface(x["args"], x["task"], x["models"][0])

            def register_classification_head(
                self, name, num_classes=None, inner_dim=None, **kwargs
            ):
                """Register a classification head."""
                logger.info("Registering classification head: {0}".format(name))
                if name in self.classification_heads:
                    prev_num_classes = self.classification_heads[name].out_proj.out_features
                    prev_inner_dim = self.classification_heads[name].dense.out_features
                    if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                        logger.warning(
                            're-registering head "{}" with num_classes {} (prev: {}) '
                            "and inner_dim {} (prev: {})".format(
                                name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                            )
                        )
                self.classification_heads[name] = BARTClassificationHead(
                    input_dim=self.args.encoder_embed_dim,
                    inner_dim=inner_dim or self.args.encoder_embed_dim,
                    num_classes=num_classes,
                    activation_fn=self.args.pooler_activation_fn,
                    pooler_dropout=self.args.pooler_dropout,
                    do_spectral_norm=getattr(
                        self.args, "spectral_norm_classification_head", False
                    ),
                )

            def upgrade_state_dict_named(self, state_dict, name):
                super().upgrade_state_dict_named(state_dict, name)

                prefix = name + "." if name != "" else ""
                current_head_names = (
                    []
                    if not hasattr(self, "classification_heads")
                    else self.classification_heads.keys()
                )

                # TODO: handle multiple fc1 and fc2
                #   if multiple mismatch -> assign single->multiple
                #   decoder.layers.X.fc1.0.linear
                if callable(cus_upgrade_state_dict_named):
                    cus_upgrade_state_dict_named(self, state_dict)

                # Handle new classification heads present in the state dict.
                keys_to_delete = []
                for k in state_dict.keys():
                    if not k.startswith(prefix + "classification_heads."):
                        continue

                    head_name = k[len(prefix + "classification_heads.") :].split(".")[0]
                    num_classes = state_dict[
                        prefix + "classification_heads." + head_name + ".out_proj.weight"
                    ].size(0)
                    inner_dim = state_dict[
                        prefix + "classification_heads." + head_name + ".dense.weight"
                    ].size(0)

                    if getattr(self.args, "load_checkpoint_heads", False):
                        if head_name not in current_head_names:
                            self.register_classification_head(head_name, num_classes, inner_dim)
                    else:
                        if head_name not in current_head_names:
                            logger.warning(
                                "deleting classification head ({}) from checkpoint "
                                "not present in current model: {}".format(head_name, k)
                            )
                            keys_to_delete.append(k)
                        elif (
                            num_classes
                            != self.classification_heads[head_name].out_proj.out_features
                            or inner_dim
                            != self.classification_heads[head_name].dense.out_features
                        ):
                            logger.warning(
                                "deleting classification head ({}) from checkpoint "
                                "with different dimensions than current model: {}".format(
                                    head_name, k
                                )
                            )
                            keys_to_delete.append(k)
                for k in keys_to_delete:
                    del state_dict[k]

                def truncate_emb(key):
                    if key in state_dict:
                        state_dict[key] = state_dict[key][:-1, :]

                # When finetuning on translation task, remove last row of
                # embedding matrix that corresponds to mask_idx token.
                loaded_dict_size = state_dict["encoder.embed_tokens.weight"].size(0)
                if (
                    loaded_dict_size == len(self.encoder.dictionary) + 1
                    and "<mask>" not in self.encoder.dictionary
                ):
                    truncate_emb("encoder.embed_tokens.weight")
                    truncate_emb("decoder.embed_tokens.weight")
                    truncate_emb("encoder.output_projection.weight")
                    truncate_emb("decoder.output_projection.weight")

                # When continued pretraining on new set of languages for mbart,
                # add extra lang embeddings at the end of embed_tokens.
                # Note: newly added languages are assumed to have been added at the end.
                if self.args.task == "multilingual_denoising" and loaded_dict_size < len(
                    self.encoder.dictionary
                ):
                    logger.info(
                        "Adding extra language embeddings not found in pretrained model for "
                        "continued pretraining of MBART on new set of languages."
                    )
                    loaded_mask_token_embedding = state_dict["encoder.embed_tokens.weight"][
                        -1, :
                    ]

                    num_langids_to_add = len(self.encoder.dictionary) - loaded_dict_size
                    embed_dim = state_dict["encoder.embed_tokens.weight"].size(1)

                    new_lang_embed_to_add = torch.zeros(num_langids_to_add, embed_dim)
                    nn.init.normal_(new_lang_embed_to_add, mean=0, std=embed_dim ** -0.5)
                    new_lang_embed_to_add = new_lang_embed_to_add.to(
                        dtype=state_dict["encoder.embed_tokens.weight"].dtype,
                    )

                    state_dict["encoder.embed_tokens.weight"] = torch.cat(
                        [
                            state_dict["encoder.embed_tokens.weight"][
                                : loaded_dict_size - 1, :
                            ],
                            new_lang_embed_to_add,
                            loaded_mask_token_embedding.unsqueeze(0),
                        ]
                    )
                    state_dict["decoder.embed_tokens.weight"] = torch.cat(
                        [
                            state_dict["decoder.embed_tokens.weight"][
                                : loaded_dict_size - 1, :
                            ],
                            new_lang_embed_to_add,
                            loaded_mask_token_embedding.unsqueeze(0),
                        ]
                    )

                # Copy any newly-added classification heads into the state dict
                # with their current weights.
                if hasattr(self, "classification_heads"):
                    cur_state = self.classification_heads.state_dict()
                    for k, v in cur_state.items():
                        if prefix + "classification_heads." + k not in state_dict:
                            logger.info("Overwriting " + prefix + "classification_heads." + k)
                            state_dict[prefix + "classification_heads." + k] = v

        _bartCls.__name__ = root_cls.__name__
        for _fn in assign_fns:
            if callable(_fn):
                setattr(_bartCls, _fn.__name__, _fn)
            else:
                assert isinstance(_fn, tuple)
                assert isinstance(_fn[0], str) and callable(_fn[1])
                setattr(_bartCls, _fn[0], _fn[1])

        return _bartCls
    return _convertor
    