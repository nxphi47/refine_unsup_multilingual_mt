
import logging
import re
from dataclasses import dataclass, field, fields
from typing import Any, List, Optional

from omegaconf import II

from fairseq import utils
from fairseq.data.multilingual import multilingual_data_manager
from fairseq.data.multilingual.multilingual_utils import LangTokSpec, LangTokStyle, get_lang_tok
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.distributed.utils import get_global_rank
from fairseq.models.bart.model import BARTClassificationHead, bart_large_architecture
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
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoder, register_model, register_model_architecture
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)
from fairseq.modules import transformer_layer
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor
from fairseq.models.transformer import (
    TransformerConfig,
    TransformerModelBase,
    TransformerEncoderBase,
    TransformerDecoderBase,
)
from fairseq.modules.transformer_thor_layer import (
    ColumnParallelLinear,
    GpuAttTransformerDecoderLayerBase,
    GpuFFNSeparateSharedLangThorTransformerDecoderLayerBase,
    GpuFFNSharedLangThorTransformerDecoderLayerBase,
    GpuFFNSharedLangThorTransformerEncoderLayerBase,
    LangFFNListTransformerDecoderLayerBase,
    ThorTransformerDecoderLayerBase,
    ThorTransformerEncoderLayerBase,
    GpuFFNThorTransformerDecoderLayerBase
)
from fairseq.models.transformer_thor_config import GpuAttSharedLangTransformerConfig, GpuFFNSharedLangThorTransformerConfig, LangFFNListTransformerConfig, ThorTransformerConfig, convert_transformer_base_to_transformer, convert_transformer_to_bart_cls

from fairseq.modules.transformer_sentence_encoder import init_bert_params

from fairseq.models.bart.hub_interface import BARTHubInterface

logger = logging.getLogger(__name__)

# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == 'ThorTransformerEncoderBase':
        return 'ThorTransformerEncoder'
    else:
        return module_name


class ThorTransformerEncoderBase(TransformerEncoderBase):
    def __init__(self, cfg, dictionary, embed_tokens):
        self.layer_idx = 0  # for building encoder layers
        super().__init__(cfg, dictionary, embed_tokens)
        assert self.layer_idx == cfg.encoder.layers

    def build_encoder_layer(self, cfg):
        layer = ThorTransformerEncoderLayerBase(cfg, layer_idx=self.layer_idx)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        self.layer_idx += 1
        return layer

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        expert_num: Optional[int] = None,
        return_expert: Optional[bool] = False,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(
            src_tokens, src_lengths, return_all_hiddens, token_embeddings,
            expert_num, return_expert
        )
    
    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        expert_num: Optional[int] = None,
        return_expert: Optional[bool] = False,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        selection = []
        for layer in self.layers:
            xout = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None,
                expert_num=expert_num, return_expert=return_expert
            )
            if return_expert:
                x, selection_layer = xout
                selection.append(selection_layer)
            else:
                x = xout
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = src_tokens.ne(self.padding_idx).sum(dim=1, dtype=torch.int32).reshape(-1, 1).contiguous()
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }


class ThorTransformerEncoder(ThorTransformerEncoderBase):
    def __init__(self, args, dictionary, embed_tokens):
        self.args = args
        super().__init__(
            ThorTransformerConfig.from_namespace(args),
            dictionary,
            embed_tokens,
        )

    def build_encoder_layer(self, args):
        return super().build_encoder_layer(
            ThorTransformerConfig.from_namespace(args),
        )


class ThorTransformerDecoderBase(TransformerDecoderBase):
    def __init__(self, cfg, dictionary, embed_tokens, no_encoder_attn=False, output_projection=None):
        self.layer_idx = 0  # for building decoder layers
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn, output_projection=output_projection)
        assert self.layer_idx == cfg.decoder.layers
    
    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = ThorTransformerDecoderLayerBase(cfg, no_encoder_attn, layer_idx=self.layer_idx)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        self.layer_idx += 1

        return layer
    
    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        expert_num: Optional[int] = None,
        return_expert: Optional[bool] = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            expert_num=expert_num,
            return_expert=return_expert,
        )

        if not features_only:
            x = self.output_layer(x)
        return x, extra
    
    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        expert_num: Optional[int] = None,
        return_expert: Optional[bool] = False,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            expert_num=expert_num,
            return_expert=return_expert
        )
    
    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        expert_num: Optional[int] = None,
        return_expert: Optional[bool] = False,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        selection = []
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            xout = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
                expert_num=expert_num,
                return_expert=return_expert,
            )
            if return_expert:
                x, layer_attn, _, selection_layer = xout
                if selection_layer is not None:
                    selection.append(selection_layer)
            else:
                x, layer_attn, _ = xout

            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if return_expert:
            return x, {"attn": [attn], "inner_states": inner_states, "selection": selection}
        else:
            return x, {"attn": [attn], "inner_states": inner_states}


class ThorTransformerDecoder(ThorTransformerDecoderBase):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.args = args
        super().__init__(
            ThorTransformerConfig.from_namespace(args),
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )

    def build_output_projection(self, args, dictionary, embed_tokens):
        super().build_output_projection(
            ThorTransformerConfig.from_namespace(args), dictionary, embed_tokens
        )

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return super().build_decoder_layer(
            ThorTransformerConfig.from_namespace(args), no_encoder_attn=no_encoder_attn
        )


class GpuFFNThorTransformerDecoderBase(TransformerDecoderBase):
    def __init__(self, cfg, dictionary, embed_tokens, no_encoder_attn=False, output_projection=None):
        self.layer_idx = 0  # for building decoder layers
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn, output_projection=output_projection)
        assert self.layer_idx == cfg.decoder.layers
    
    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = GpuFFNThorTransformerDecoderLayerBase(cfg, no_encoder_attn, layer_idx=self.layer_idx,
            # ?????
            world_size=None, rank=None,
        )
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        self.layer_idx += 1

        return layer


class GpuAttTransformerDecoderBase(TransformerDecoderBase):
    def __init__(self, cfg, dictionary, embed_tokens, no_encoder_attn=False, output_projection=None):
        self.layer_idx = 0  # for building decoder layers
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn, output_projection=output_projection)
        assert self.layer_idx == cfg.decoder.layers
    
    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = GpuAttTransformerDecoderLayerBase(cfg, no_encoder_attn, layer_idx=self.layer_idx,
            # ?????
            world_size=None, rank=None,
        )
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        self.layer_idx += 1

        return layer


class GpuFFNSharedLangThorTransformerEncoderBase(TransformerEncoderBase):
    def __init__(self, cfg, dictionary, embed_tokens):
        self.layer_idx = 0  # for building encoder layers
        super().__init__(cfg, dictionary, embed_tokens)
        assert self.layer_idx == cfg.encoder.layers
        self.shared_lang = cfg.shared_lang
        self.shared_lang_index = dictionary.index(f'[{self.shared_lang}]')
        assert self.shared_lang_index != dictionary.unk_index

    def get_shared_mask(self, prev_output_tokens):
        # prev_output_tokens: (bs, slen)
        # return: shared_mask: (bs)
        # NOTE: prev_output_tokens: [<eos> <en_XX> ....]
        #   on generation [<en_XX>]
        dictionary = self.dictionary
        if prev_output_tokens.size(1) == 1:
            first_tok = prev_output_tokens[:, 0]
        else:
            first_tok = prev_output_tokens[:, 1]
            assert not (first_tok == dictionary.eos()).any(), f'first_tok has eos, {prev_output_tokens.size()}'
            assert not (first_tok == dictionary.pad()).any(), f'first_tok has pad, {prev_output_tokens.size()}'
            assert not (first_tok == dictionary.bos()).any(), f'first_tok has bos, {prev_output_tokens.size()}'
        shared_mask = (first_tok == self.shared_lang_index)
        return shared_mask
    
    def build_encoder_layer(self, cfg):
        layer = GpuFFNSharedLangThorTransformerEncoderLayerBase(cfg, layer_idx=self.layer_idx,
            world_size=None, rank=None,
        )
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        self.layer_idx += 1
        return layer

    def forward(
        self,
        src_tokens,
        shared_mask: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(
            src_tokens, shared_mask, src_lengths, return_all_hiddens, token_embeddings,
        )
    
    def forward_scriptable(
        self,
        src_tokens,
        shared_mask: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        selection = []
        for layer in self.layers:
            x = layer(
                x, shared_mask, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = src_tokens.ne(self.padding_idx).sum(dim=1, dtype=torch.int32).reshape(-1, 1).contiguous()
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }
    
    def forward_torchscript(self, net_input: Dict[str, Tensor]):
        """A TorchScript-compatible version of forward.

        Encoders which use additional arguments may want to override
        this method for TorchScript compatibility.
        """
        if torch.jit.is_scripting():
            # return self.forward(
            #     src_tokens=net_input["src_tokens"],
            #     src_lengths=net_input["src_lengths"],
            # )
            raise ValueError(f'scripting not supported!')
        else:
            return self.forward_non_torchscript(net_input)

    @torch.jit.unused
    def forward_non_torchscript(self, net_input: Dict[str, Tensor]):
        # expect net_input to have prev_output_tokens when backtranslate
        encoder_input = {
            k: v for k, v in net_input.items() if k != "prev_output_tokens" and k != "bt_prev_output_tokens"
        }
        if "bt_prev_output_tokens" in net_input:
            encoder_input['shared_mask'] = self.get_shared_mask(net_input['bt_prev_output_tokens'])
        else:
            encoder_input['shared_mask'] = ~self.get_shared_mask(net_input['prev_output_tokens'])
        return self.forward(**encoder_input)


class GpuFFNSharedLangThorTransformerDecoderBase(TransformerDecoderBase):
    def __init__(self, cfg, dictionary, embed_tokens, no_encoder_attn=False, output_projection=None):
        self.layer_idx = 0  # for building decoder layers
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn, output_projection=output_projection)
        assert self.layer_idx == cfg.decoder.layers
        self.shared_lang = cfg.shared_lang
        self.shared_lang_index = dictionary.index(f'[{self.shared_lang}]')
        assert self.shared_lang_index != dictionary.unk_index
    
    def get_shared_mask(self, prev_output_tokens):
        # prev_output_tokens: (bs, slen)
        # return: shared_mask: (bs)
        # NOTE: prev_output_tokens: [<eos> <en_XX> ....]
        #   on generation [<en_XX>]
        dictionary = self.dictionary
        slen = prev_output_tokens.size(1)
        if slen == 1:
            # only when generate at first step, the output must be lang_tok, this first tok is <eos>
            first_tok = prev_output_tokens[:, 0]
        else:
            # FIXME: slen == 2, only the primary first_tok has lang_tok, how to deal with this?
            # FIXME: slen >  2, all first_tok must be lang_tok
            first_tok = prev_output_tokens[:, 1]
            _any = any([(first_tok == dictionary.eos()).any(), (first_tok == dictionary.bos()).any(), (first_tok == dictionary.pad()).any()])
            assert slen == 2 or not _any, f'{slen=}, first_tok has eos/bos/pad, {prev_output_tokens.size()=}, \n{prev_output_tokens=}'
            # FIXME: when generating, if beam > 1, the expanded batch only contain lang_tok in the 1st instance
        shared_mask = (first_tok == self.shared_lang_index)
        return shared_mask
    
    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = GpuFFNSharedLangThorTransformerDecoderLayerBase(cfg, no_encoder_attn, layer_idx=self.layer_idx,
            # ?????
            world_size=None, rank=None,
        )
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        self.layer_idx += 1

        return layer
    
    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]
        
        shared_mask = self.get_shared_mask(prev_output_tokens)

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                shared_mask,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}


class GpuFFNSeparateSharedLangThorTransformerDecoderBase(GpuFFNSharedLangThorTransformerDecoderBase):
    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = GpuFFNSeparateSharedLangThorTransformerDecoderLayerBase(cfg, no_encoder_attn, layer_idx=self.layer_idx,
            # ?????
            world_size=None, rank=None,
        )
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        self.layer_idx += 1

        return layer


class LangFFNListTransformerDecoderBase(TransformerDecoderBase):
    def __init__(self, cfg, dictionary, embed_tokens, no_encoder_attn=False, output_projection=None):
        self.layer_idx = 0  # for building decoder layers
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn, output_projection=output_projection)
        assert self.layer_idx == cfg.decoder.layers
        # self.shared_lang = cfg.shared_lang
        # self.shared_lang_index = dictionary.index(f'[{self.shared_lang}]')
        # assert self.shared_lang_index != dictionary.unk_index
        self.langs_for_experts = cfg.langs_for_experts.split(",")
        assert len(self.langs_for_experts) == cfg.num_experts
        self.lang_id_to_experts = {
            multilingual_data_manager._lang_id(
                dictionary,
                get_lang_tok(
                    k, 
                    lang_tok_style=LangTokStyle.mbart.value,
                    spec=LangTokSpec.main.value
                )
            ): i
            for i, k in enumerate(self.langs_for_experts)
        }
    
    def get_select_experts(self, prev_output_tokens):
        first_tok = prev_output_tokens[:, int(not prev_output_tokens.size(1) == 1)]
        select_experts = first_tok.clone().detach()
        for k, v in self.lang_id_to_experts.items():
            select_experts[first_tok == k] = v
        return select_experts
    
    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = LangFFNListTransformerDecoderLayerBase(cfg, no_encoder_attn, layer_idx=self.layer_idx)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        self.layer_idx += 1

        return layer
    
    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]
        
        select_experts = self.get_select_experts(prev_output_tokens)

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                select_experts,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}




class ThorTransformerModelBase(TransformerModelBase):
    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, ThorTransformerConfig(), delete_default=False, with_prefix=""
        )
    
    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return ThorTransformerEncoderBase(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return ThorTransformerDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )


class ThorDecTransformerModelBase(TransformerModelBase):
    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, ThorTransformerConfig(), delete_default=False, with_prefix=""
        )
    
    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return TransformerEncoderBase(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return ThorTransformerDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )
    
    @staticmethod
    def custom_upgrade_state_dict(model, state_dict):
        for i, dec_layer in enumerate(model.decoder.layers):
            is_expert = dec_layer.use_expert
            is_state_expert = f"decoder.layers.{i}.fc1.0.weight" in state_dict
            if is_expert and not is_state_expert:
                num_experts = dec_layer.num_experts
                for k in ['fc1', 'fc2']:
                    assert f"decoder.layers.{i}.{k}.weight" in state_dict
                    logger.warning(f'reassign decoder.layers.{i}.{k}.weight -> decoder.layers.{i}.{k}.0-{num_experts - 1}.weight')
                    for j in range(num_experts):
                        state_dict[f"decoder.layers.{i}.{k}.{j}.weight"] = state_dict[f"decoder.layers.{i}.{k}.weight"]
                    del state_dict[f"decoder.layers.{i}.{k}.weight"]
                    if f"decoder.layers.{i}.{k}.bias" in state_dict:
                        logger.warning(f'reassign decoder.layers.{i}.{k}.bias -> decoder.layers.{i}.{k}.0-{num_experts - 1}.bias')
                        for j in range(num_experts):
                            state_dict[f"decoder.layers.{i}.{k}.{j}.bias"] = state_dict[f"decoder.layers.{i}.{k}.bias"]
                        del state_dict[f"decoder.layers.{i}.{k}.bias"]


class GpuFFNThorDecTransformerModelBase(TransformerModelBase):
    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, ThorTransformerConfig(), delete_default=False, with_prefix=""
        )
    
    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return TransformerEncoderBase(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return GpuFFNThorTransformerDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )


class GpuAttDecTransformerModelBase(TransformerModelBase):
    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, GpuAttSharedLangTransformerConfig(), delete_default=False, with_prefix=""
        )
    
    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return TransformerEncoderBase(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return GpuAttTransformerDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )


class GpuFFNSharedLangThorDecTransformerModelBase(TransformerModelBase):
    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, GpuFFNSharedLangThorTransformerConfig(), delete_default=False, with_prefix=""
        )
    
    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return TransformerEncoderBase(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return GpuFFNSharedLangThorTransformerDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )
    
    @staticmethod
    def custom_upgrade_state_dict(model, state_dict):
        for i, dec_layer in enumerate(model.decoder.layers):
            is_expert = dec_layer.use_expert
            is_state_expert = f"decoder.layers.{i}.fc1.fc.weight" in state_dict and f"decoder.layers.{i}.fc1.shfc.weight" in state_dict
            if is_expert and not is_state_expert:
                for k in ['fc1', 'fc2']:
                    assert f"decoder.layers.{i}.{k}.weight" in state_dict
                    logger.warning(f'reassign decoder.layers.{i}.{k}.weight -> decoder.layers.{i}.{k}.fc/shfc.weight')
                    state_dict[f"decoder.layers.{i}.{k}.fc.weight"] = state_dict[f"decoder.layers.{i}.{k}.weight"]
                    state_dict[f"decoder.layers.{i}.{k}.shfc.weight"] = state_dict[f"decoder.layers.{i}.{k}.weight"]
                    del state_dict[f"decoder.layers.{i}.{k}.weight"]

                    if f"decoder.layers.{i}.{k}.bias" in state_dict:
                        logger.warning(f'reassign decoder.layers.{i}.{k}.bias -> decoder.layers.{i}.{k}.fc/shfc.bias')
                        state_dict[f"decoder.layers.{i}.{k}.fc.bias"] = state_dict[f"decoder.layers.{i}.{k}.bias"]
                        state_dict[f"decoder.layers.{i}.{k}.shfc.bias"] = state_dict[f"decoder.layers.{i}.{k}.bias"]
                        del state_dict[f"decoder.layers.{i}.{k}.bias"]


class GpuFFNSeparateSharedLangThorDecTransformerModelBase(TransformerModelBase):
    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, GpuFFNSharedLangThorTransformerConfig(), delete_default=False, with_prefix=""
        )
    
    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return TransformerEncoderBase(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return GpuFFNSeparateSharedLangThorTransformerDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )
    
    @staticmethod
    def custom_upgrade_state_dict(model, state_dict):
        for i, dec_layer in enumerate(model.decoder.layers):
            is_expert = dec_layer.use_expert
            is_separate_shared = dec_layer.use_separate_shared
            is_state_expert = f"decoder.layers.{i}.fc1.fc.weight" in state_dict and f"decoder.layers.{i}.fc1.shfc.weight" in state_dict
            is_state_separate_shared = f"decoder.layers.{i}.fc1.fc.weight" in state_dict and f"decoder.layers.{i}.fc1.shfc.weight" in state_dict
            if (is_expert and not is_state_expert) or (is_separate_shared and not is_state_separate_shared):
                logger.warning(
                    f'{is_expert=}/{is_state_expert=}/{is_separate_shared=}/{is_state_separate_shared=}'
                    f'reassign decoder.layers.{i}.{k}.weight/bias -> decoder.layers.{i}.{k}.fc/shfc.weight/bias'
                )
                for k in ['fc1', 'fc2']:
                    assert f"decoder.layers.{i}.{k}.weight" in state_dict
                    for n in ['fc', 'shfc']:
                        state_dict[f"decoder.layers.{i}.{k}.{n}.weight"] = state_dict[f"decoder.layers.{i}.{k}.weight"]
                    del state_dict[f"decoder.layers.{i}.{k}.weight"]
                    if f"decoder.layers.{i}.{k}.bias" in state_dict:
                        for n in ['fc', 'shfc']:
                            state_dict[f"decoder.layers.{i}.{k}.{n}.bias"] = state_dict[f"decoder.layers.{i}.{k}.bias"]
                        del state_dict[f"decoder.layers.{i}.{k}.bias"]


class LangFFNListDecTransformerModelBase(TransformerModelBase):
    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, LangFFNListTransformerConfig(), delete_default=False, with_prefix=""
        )
    
    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return TransformerEncoderBase(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return LangFFNListTransformerDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )
    
    @staticmethod
    def custom_upgrade_state_dict(model, state_dict):
        for i, dec_layer in enumerate(model.decoder.layers):
            is_expert = dec_layer.use_expert
            is_state_expert = f"decoder.layers.{i}.fc1.0.weight" in state_dict
            if is_expert and not is_state_expert:
                num_experts = dec_layer.num_experts
                for k in ['fc1', 'fc2']:
                    assert f"decoder.layers.{i}.{k}.weight" in state_dict
                    logger.warning(f'reassign decoder.layers.{i}.{k}.weight -> decoder.layers.{i}.{k}.0-{num_experts - 1}.weight')
                    for j in range(num_experts):
                        state_dict[f"decoder.layers.{i}.{k}.{j}.weight"] = state_dict[f"decoder.layers.{i}.{k}.weight"]
                    del state_dict[f"decoder.layers.{i}.{k}.weight"]
                    if f"decoder.layers.{i}.{k}.bias" in state_dict:
                        logger.warning(f'reassign decoder.layers.{i}.{k}.bias -> decoder.layers.{i}.{k}.0-{num_experts - 1}.bias')
                        for j in range(num_experts):
                            state_dict[f"decoder.layers.{i}.{k}.{j}.bias"] = state_dict[f"decoder.layers.{i}.{k}.bias"]
                        del state_dict[f"decoder.layers.{i}.{k}.bias"]



class GpuFFNSharedLangThorEncDecTransformerModelBase(GpuFFNSharedLangThorDecTransformerModelBase):
    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return GpuFFNSharedLangThorTransformerEncoderBase(cfg, src_dict, embed_tokens)
    
    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.

        # FIXME: limitation: enc_shared_mask = ~dec_shared_mask
        #   hence, we can only support 1 shared language
        #   all other languages must be translated to/from the shared language
        """
        enc_shared_mask = ~self.encoder.get_shared_mask(prev_output_tokens)
        encoder_out = self.encoder(
            src_tokens, enc_shared_mask, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out
    
    @staticmethod
    def custom_upgrade_state_dict(model, state_dict):
        def reassign(ed_module, ed_name):
            for i, layer in enumerate(ed_module.layers):
                is_expert = layer.use_expert
                is_state_expert = f"{ed_name}.layers.{i}.fc1.fc.weight" in state_dict and f"{ed_name}.layers.{i}.fc1.shfc.weight" in state_dict
                if is_expert and not is_state_expert:
                    for k in ['fc1', 'fc2']:
                        assert f"{ed_name}.layers.{i}.{k}.weight" in state_dict
                        logger.warning(f'reassign {ed_name}.layers.{i}.{k}.weight -> {ed_name}.layers.{i}.{k}.fc/shfc.weight')
                        state_dict[f"{ed_name}.layers.{i}.{k}.fc.weight"] = state_dict[f"{ed_name}.layers.{i}.{k}.weight"]
                        state_dict[f"{ed_name}.layers.{i}.{k}.shfc.weight"] = state_dict[f"{ed_name}.layers.{i}.{k}.weight"]
                        del state_dict[f"{ed_name}.layers.{i}.{k}.weight"]

                        if f"{ed_name}.layers.{i}.{k}.bias" in state_dict:
                            logger.warning(f'reassign {ed_name}.layers.{i}.{k}.bias -> {ed_name}.layers.{i}.{k}.fc/shfc.bias')
                            state_dict[f"{ed_name}.layers.{i}.{k}.fc.bias"] = state_dict[f"{ed_name}.layers.{i}.{k}.bias"]
                            state_dict[f"{ed_name}.layers.{i}.{k}.shfc.bias"] = state_dict[f"{ed_name}.layers.{i}.{k}.bias"]
                            del state_dict[f"{ed_name}.layers.{i}.{k}.bias"]
        reassign(model.decoder, "decoder")
        reassign(model.encoder, "encoder")


def setup_ddp_ignore_params(model):
    params_to_ignore = []
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if isinstance(module, ColumnParallelLinear):
                # Create expected format
                fqn = f"{module_name}.{param_name}"
                params_to_ignore.append(fqn)
    # logger.warning(f'{get_global_rank()}:{model.__class__.__name__}:setup_ddp_ignore_params: {params_to_ignore=}')
    torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(model, params_to_ignore)


@register_model("thor_dec_transformer")
@convert_transformer_base_to_transformer(
    ThorTransformerConfig, base_architecture, setup_ddp_ignore_params)
class ThorDecTransformer(ThorDecTransformerModelBase):
    pass


@register_model("langffnlst_dec_transformer")
@convert_transformer_base_to_transformer(
    LangFFNListTransformerConfig, base_architecture)
class LangFFNListDecTransformer(LangFFNListDecTransformerModelBase):
    pass


@register_model("gpuffn_thor_dec_transformer")
@convert_transformer_base_to_transformer(
    ThorTransformerConfig, base_architecture, setup_ddp_ignore_params)
class GpuFFNThorDecTransformer(GpuFFNThorDecTransformerModelBase):
    pass


@register_model("gpu_att_dec_transformer")
@convert_transformer_base_to_transformer(
    GpuAttSharedLangTransformerConfig, base_architecture, setup_ddp_ignore_params)
class GpuAttDecTransformer(GpuAttDecTransformerModelBase):
    pass


@register_model("gpuffn_sharedlang_thor_dec_transformer_backup")
class GpuFFNSharedLangThorDecTransformer_Backup(GpuFFNSharedLangThorDecTransformerModelBase):
    def __init__(self, args, encoder, decoder):
        cfg = GpuFFNSharedLangThorTransformerConfig.from_namespace(args)
        super().__init__(cfg, encoder, decoder)
        self.args = args
    
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, GpuFFNSharedLangThorTransformerConfig(), 
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
            parser, GpuFFNSharedLangThorTransformerConfig(), delete_default=True, with_prefix=""
        )
    
    def setup_ddp_ignore_params(self):
        params_to_ignore = []
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if isinstance(module, ColumnParallelLinear):
                    # Create expected format
                    fqn = f"{module_name}.{param_name}"
                    params_to_ignore.append(fqn)
        logger.warning(f'{get_global_rank()}:{self.__class__.__name__}:setup_ddp_ignore_params: {params_to_ignore=}')
        torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(self, params_to_ignore)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        thor_base_architecture(args)

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
        cfg = GpuFFNSharedLangThorTransformerConfig.from_namespace(args)
        model = super().build_model(cfg, task)
        model.setup_ddp_ignore_params()
        return model

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        return super().build_embedding(
            GpuFFNSharedLangThorTransformerConfig.from_namespace(args), dictionary, embed_dim, path
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return super().build_encoder(
            GpuFFNSharedLangThorTransformerConfig.from_namespace(args), src_dict, embed_tokens
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return super().build_decoder(
            GpuFFNSharedLangThorTransformerConfig.from_namespace(args), tgt_dict, embed_tokens
        )


@register_model("gpuffn_sharedlang_thor_dec_transformer")
@convert_transformer_base_to_transformer(
    GpuFFNSharedLangThorTransformerConfig, base_architecture, setup_ddp_ignore_params)
class GpuFFNSharedLangThorDecTransformer(GpuFFNSharedLangThorDecTransformerModelBase):
    pass


@register_model("gpuffn_sep_sharedlang_thor_dec_transformer")
@convert_transformer_base_to_transformer(
    GpuFFNSharedLangThorTransformerConfig, base_architecture, setup_ddp_ignore_params)
class GpuFFNSeparateSharedLangThorDecTransformer(GpuFFNSeparateSharedLangThorDecTransformerModelBase):
    pass


@register_model("gpuffn_sharedlang_thor_encdec_transformer")
@convert_transformer_base_to_transformer(
    GpuFFNSharedLangThorTransformerConfig, base_architecture, setup_ddp_ignore_params)
class GpuFFNSharedLangThorEncDecTransformer(GpuFFNSharedLangThorEncDecTransformerModelBase):
    pass


def thor_dec_bart_upgrade_state_dict(model, state_dict):
    for i, dec_layer in enumerate(model.decoder.layers):
        # encoder.layers.1.fc1.weight
        # is_expert = isinstance(dec_layer.fc1, nn.ModuleList)
        is_expert = dec_layer.use_expert
        is_state_expert = f"decoder.layers.{i}.fc1.0.weight" in state_dict
        if is_expert and not is_state_expert:
            num_experts = dec_layer.num_experts
            for k in ['fc1', 'fc2']:
                assert f"decoder.layers.{i}.{k}.weight" in state_dict
                logger.warning(f'reassign decoder.layers.{i}.{k}.weight -> decoder.layers.{i}.{k}.0-{num_experts - 1}.weight')
                for j in range(num_experts):
                    state_dict[f"decoder.layers.{i}.{k}.{j}.weight"] = state_dict[f"decoder.layers.{i}.{k}.weight"]
                del state_dict[f"decoder.layers.{i}.{k}.weight"]
                if f"decoder.layers.{i}.{k}.bias" in state_dict:
                    logger.warning(f'reassign decoder.layers.{i}.{k}.bias -> decoder.layers.{i}.{k}.0-{num_experts - 1}.bias')
                    for j in range(num_experts):
                        state_dict[f"decoder.layers.{i}.{k}.{j}.bias"] = state_dict[f"decoder.layers.{i}.{k}.bias"]
                    del state_dict[f"decoder.layers.{i}.{k}.bias"]


@register_model("thor_dec_bart")
@convert_transformer_to_bart_cls(ThorDecTransformerModelBase.custom_upgrade_state_dict)
class ThorDecBARTModel(ThorDecTransformer):
    pass


@register_model("langffnlst_dec_bart")
@convert_transformer_to_bart_cls(LangFFNListDecTransformerModelBase.custom_upgrade_state_dict)
class LangFFNListDecBARTModel(LangFFNListDecTransformer):
    pass


@register_model("gpuffn_thor_dec_bart")
@convert_transformer_to_bart_cls()
class GpuFFNThorDecBARTModel(GpuFFNThorDecTransformer):
    pass


@register_model("gpu_att_dec_bart")
@convert_transformer_to_bart_cls()
class GpuAttDecBARTModel(GpuAttDecTransformer):
    pass


def gpuffn_sharedlang_upgrade_state_dict(model, state_dict):
    for i, dec_layer in enumerate(model.decoder.layers):
        is_expert = dec_layer.use_expert
        is_state_expert = f"decoder.layers.{i}.fc1.fc.weight" in state_dict and f"decoder.layers.{i}.fc1.shfc.weight" in state_dict
        if is_expert and not is_state_expert:
            for k in ['fc1', 'fc2']:
                assert f"decoder.layers.{i}.{k}.weight" in state_dict
                logger.warning(f'reassign decoder.layers.{i}.{k}.weight -> decoder.layers.{i}.{k}.fc/shfc.weight')
                state_dict[f"decoder.layers.{i}.{k}.fc.weight"] = state_dict[f"decoder.layers.{i}.{k}.weight"]
                state_dict[f"decoder.layers.{i}.{k}.shfc.weight"] = state_dict[f"decoder.layers.{i}.{k}.weight"]
                del state_dict[f"decoder.layers.{i}.{k}.weight"]

                if f"decoder.layers.{i}.{k}.bias" in state_dict:
                    logger.warning(f'reassign decoder.layers.{i}.{k}.bias -> decoder.layers.{i}.{k}.fc/shfc.bias')
                    state_dict[f"decoder.layers.{i}.{k}.fc.bias"] = state_dict[f"decoder.layers.{i}.{k}.bias"]
                    state_dict[f"decoder.layers.{i}.{k}.shfc.bias"] = state_dict[f"decoder.layers.{i}.{k}.bias"]
                    del state_dict[f"decoder.layers.{i}.{k}.bias"]


@register_model("gpuffn_sharedlang_thor_dec_bart")
@convert_transformer_to_bart_cls(GpuFFNSharedLangThorDecTransformerModelBase.custom_upgrade_state_dict)
class GpuFFNSharedLangThorDecBARTModel(GpuFFNSharedLangThorDecTransformer):
    pass


@register_model("gpuffn_sep_sharedlang_thor_dec_bart")
@convert_transformer_to_bart_cls(GpuFFNSeparateSharedLangThorDecTransformerModelBase.custom_upgrade_state_dict)
class GpuFFNSeparateSharedLangThorDecBARTModel(GpuFFNSeparateSharedLangThorDecTransformer):
    pass


def gpu_ffn_shared_lang_thor_encdec_bart_forward(
    model,
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
    enc_shared_mask = ~model.encoder.get_shared_mask(prev_output_tokens)
    encoder_out = model.encoder(
        src_tokens,
        shared_mask=enc_shared_mask,
        src_lengths=src_lengths,
        token_embeddings=token_embeddings,
        return_all_hiddens=return_all_hiddens
    )
    x, extra = model.decoder(
        prev_output_tokens,
        encoder_out=encoder_out,
        features_only=features_only,
        alignment_layer=alignment_layer,
        alignment_heads=alignment_heads,
        src_lengths=src_lengths,
        return_all_hiddens=return_all_hiddens,
    )
    eos: int = model.eos
    if classification_head_name is not None:
        sentence_representation = x[
            src_tokens.eq(eos), :
        ].view(x.size(0), -1, x.size(-1))[:, -1, :]
        for k, head in model.classification_heads.items():
            # for torch script only supports iteration
            if k == classification_head_name:
                x = head(sentence_representation)
                break
    return x, extra


@register_model("gpuffn_sharedlang_thor_encdec_bart")
@convert_transformer_to_bart_cls(
    GpuFFNSharedLangThorEncDecTransformerModelBase.custom_upgrade_state_dict,
    assign_fns=[('forward', gpu_ffn_shared_lang_thor_encdec_bart_forward)])
class GpuFFNSharedLangThorEncDecBARTModel(GpuFFNSharedLangThorEncDecTransformer):
    pass


@register_model_architecture("thor_dec_transformer", "thor_dec_transformer")
def thor_base_architecture(args):
    base_architecture(args)


@register_model_architecture("thor_dec_bart", "thor_dec_bart_large")
def thor_dec_bart_large_architecture(args):
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    bart_large_architecture(args)


@register_model_architecture("langffnlst_dec_bart", "langffnlst_dec_mbart_large")
def langffnlst_dec_mbart_large_architecture(args):
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    bart_large_architecture(args)


@register_model_architecture("thor_dec_bart", "thor_dec_mbart_large")
def thor_dec_mbart_large_architecture(args):
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    bart_large_architecture(args)


@register_model_architecture("gpuffn_thor_dec_bart", "gpuffn_thor_dec_bart_large")
def gpuffn_thor_dec_bart_large_architecture(args):
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    bart_large_architecture(args)


@register_model_architecture("gpuffn_thor_dec_bart", "gpuffn_thor_dec_mbart_large")
def gpuffn_thor_dec_mbart_large_architecture(args):
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    bart_large_architecture(args)


@register_model_architecture("gpu_att_dec_bart", "gpu_att_dec_mbart_large")
def gpu_att_dec_mbart_large_architecture(args):
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    bart_large_architecture(args)


@register_model_architecture("gpuffn_sharedlang_thor_dec_bart", "gpuffn_sharedlang_thor_dec_mbart_large")
def gpuffn_sharedlang_thor_dec_mbart_large_architecture(args):
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    bart_large_architecture(args)


@register_model_architecture("gpuffn_sep_sharedlang_thor_dec_bart", "gpuffn_sep_sharedlang_thor_dec_mbart_large")
def gpuffn_sep_sharedlang_thor_dec_mbart_large_architecture(args):
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    bart_large_architecture(args)


@register_model_architecture("gpuffn_sharedlang_thor_encdec_bart", "gpuffn_sharedlang_thor_encdec_mbart_large")
def gpuffn_sharedlang_thor_encdec_mbart_large_architecture(args):
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    bart_large_architecture(args)

# @register_model_architecture("gpuffn_sharedlang_thor_dec_bart_v2", "gpuffn_sharedlang_thor_dec_mbart_large_v2")
# def gpuffn_sharedlang_thor_dec_mbart_large_v2_architecture(args):
#     args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
#     bart_large_architecture(args)
