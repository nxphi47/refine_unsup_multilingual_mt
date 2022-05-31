# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.multihead_attention import CollumParallelMultiheadAttention
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor
from fairseq.models.transformer import (
    TransformerConfig,
)

import math
import torch.nn.init as init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
# from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm


from fairseq.modules.transformer_layer import (
    TransformerConfig,
    TransformerDecoderLayer,
    TransformerDecoderLayerBase,
    TransformerEncoderLayer,
    TransformerEncoderLayerBase
)

# from fairseq.models.transformer_thor import (
#     ThorTransformerConfig,
# )
from fairseq.distributed.utils import (
    get_global_rank,
    get_global_world_size
)
from fairseq.models.transformer_thor_config import GpuAttSharedLangTransformerConfig, GpuFFNSharedLangThorTransformerConfig, LangFFNListTransformerConfig, ThorTransformerConfig

import logging
logger = logging.getLogger(__name__)


def use_expert(layer_idx, step=2):
    return layer_idx % step == 0


class ThorTransformerEncoderLayerBase(TransformerEncoderLayerBase):
    def __init__(self, cfg, layer_idx=-1):
        self.num_experts = cfg.num_experts
        self.inference_level = cfg.inference_level
        self.expert_increment = cfg.expert_increment
        self.use_expert = use_expert(layer_idx, self.expert_increment)
        super().__init__(cfg)
    
    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        if self.use_expert:
            return nn.ModuleList(
                [
                    quant_noise(
                        nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
                    )
                    for _ in range(self.num_experts)
                ]
            )
        else:
            return quant_noise(
                nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
            )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        if self.use_expert:
            return nn.ModuleList(
                [
                    quant_noise(
                        nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
                    )
                    for _ in range(self.num_experts)
                ]
            )
        else:
            return quant_noise(
                nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
            )

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
        expert_num: Optional[int] = None,
        return_expert: Optional[bool] = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(
                attn_mask.to(torch.bool),
                -1e8 if x.dtype == torch.float32 else -1e4
            )

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        expert  = None
        if self.use_expert:
            if self.training:
                if expert_num is None:
                    expert = torch.randint(low=0, high=self.num_experts, size=(1,)).item()  # selected expert
                else:
                    expert = expert_num
                x = self.activation_fn(self.fc1[expert](x))
                x = self.activation_dropout_module(x)
                x = self.fc2[expert](x)
            else:
                result = []
                for expert in range(self.num_experts):
                    temp = self.activation_fn(self.fc1[expert](x))
                    temp = self.activation_dropout_module(temp)
                    temp = self.fc2[expert](temp)
                    result.append(temp)
                result = torch.stack(result, dim=0)
                if self.inference_level == 0:  # token level
                    mask = torch.randint(0, self.num_experts,
                                         size=(result.size(1), result.size(2)), device=result.device)
                    for i in range(self.num_experts):
                        expert_mask = mask.eq(i)
                        result[i] *= expert_mask.unsqueeze(-1)
                    x = result.sum(0)
                elif self.inference_level == 1:  # sentence level
                    mask = torch.randint(0, self.num_experts,
                                         size=(result.size(1),), device=result.device)
                    for i in range(self.num_experts):
                        expert_mask = mask.eq(i)
                        result[i] *= expert_mask.unsqueeze(-1).unsqueeze(-1)
                    x = result.sum(0)
        else:
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        if return_expert:
            return x, expert
        else:
            return x


# backward compatible with the legacy argparse format
class ThorTransformerEncoderLayer(ThorTransformerEncoderLayerBase):
    def __init__(self, args):
        super().__init__(ThorTransformerConfig.from_namespace(args))
        self.args = args

    def build_self_attention(self, embed_dim, args):
        return super().build_self_attention(
            embed_dim, ThorTransformerConfig.from_namespace(args)
        )


class ThorTransformerDecoderLayerBase(TransformerDecoderLayerBase):
    def __init__(self, cfg, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False, layer_idx=-1):
        self.num_experts = cfg.num_experts
        self.inference_level = cfg.inference_level
        self.expert_increment = cfg.expert_increment
        self.use_expert = use_expert(layer_idx, self.expert_increment)
        super().__init__(cfg, no_encoder_attn=no_encoder_attn, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn)
    
    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        if self.use_expert:
            return nn.ModuleList(
                [
                    quant_noise(
                        nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
                    )
                    for _ in range(self.num_experts)
                ]
            )
        else:
            return quant_noise(
                nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
            )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        if self.use_expert:
            return nn.ModuleList(
                [
                    quant_noise(
                        nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
                    )
                    for _ in range(self.num_experts)
                ]
            )
        else:
            return quant_noise(
                nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
            )
    
    def _dummy_ffn(self):
        x = 0
        for expert in range(self.num_experts):
            x += self.fc1[expert].weight.sum() * 0
            x += self.fc2[expert].weight.sum() * 0
            
            x += self.fc1[expert].bias.sum() * 0
            x += self.fc2[expert].bias.sum() * 0
        return x

    def _ffn12(self, expert, x):
        if self.use_expert:
            x = self.activation_fn(self.fc1[expert](x))
            x = self.activation_dropout_module(x)
            # if self.ffn_layernorm is not None:
            #     x = self.ffn_layernorm(x)
            x = self.fc2[expert](x)
        else:
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            # if self.ffn_layernorm is not None:
            #     x = self.ffn_layernorm(x)
            x = self.fc2(x)
        return x
    
    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        expert_num: Optional[int] = None,
        return_expert: Optional[bool] = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        expert = None
        if self.use_expert:
            if self.training:
                if expert_num is None:
                    expert = torch.randint(low=0, high=self.num_experts, size=(1,)).item()  # selected expert
                else:
                    expert = expert_num
                x = self._ffn12(expert, x)
                x = x + self._dummy_ffn()
            else:
                result = []
                for expert in range(self.num_experts):
                    temp = self._ffn12(expert, x)
                    result.append(temp)
                result = torch.stack(result, dim=0)
                if self.inference_level == 0:  # token level
                    mask = torch.randint(0, self.num_experts,
                                         size=(result.size(1), result.size(2)), device=result.device)
                    for i in range(self.num_experts):
                        expert_mask = mask.eq(i)
                        result[i] *= expert_mask.unsqueeze(-1)
                    x = result.sum(0)
                    # FIXME: big error when using level 0
                elif self.inference_level == 1:  # sentence level
                    mask = torch.randint(0, self.num_experts,
                                         size=(result.size(1),), device=result.device)
                    for i in range(self.num_experts):
                        expert_mask = mask.eq(i)
                        result[i] *= expert_mask.unsqueeze(-1).unsqueeze(-1)
                    x = result.sum(0)
                else:
                    raise ValueError(f'{self.inference_level=} wrong!')
        else:
            # x = self.activation_fn(self.fc1(x))
            # x = self.activation_dropout_module(x)
            # # if self.ffn_layernorm is not None:
            # #     x = self.ffn_layernorm(x)
            # x = self.fc2(x)
            x = self._ffn12(None, x)

        # NOTE old code
        # x = self.activation_fn(self.fc1(x))
        # x = self.activation_dropout_module(x)
        # if self.ffn_layernorm is not None:
        #     x = self.ffn_layernorm(x)
        # x = self.fc2(x)

        x = self.dropout_module(x)
        # NOTE new code main
        # if self.w_resid is not None:
        #     residual = torch.mul(self.w_resid, residual)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state

        if return_expert:
            return x, attn, None, expert
        else:
            return x, attn, None


# backward compatible with the legacy argparse format
class ThorTransformerDecoderLayer(ThorTransformerDecoderLayerBase):
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(
            ThorTransformerConfig.from_namespace(args),
            no_encoder_attn=no_encoder_attn,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.args = args

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return super().build_self_attention(
            embed_dim,
            ThorTransformerConfig.from_namespace(args),
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

    def build_encoder_attention(self, embed_dim, args):
        return super().build_encoder_attention(
            embed_dim,
            ThorTransformerConfig.from_namespace(args),
        )


# distributed collumn parallel from megatron


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def _initialize_affine_weight(
    weight, output_size, input_size,
    per_partition_size, partition_dim, init_method,
    stride=1, return_master_weight=False,
    world_size=None, rank=None,
    ):
    """Initialize affine weight for model parallel.
    Build the master weight on all processes and scatter
    the relevant chunk."""
    # If we only use 1 process for model parallelism, bypass scatter.
    world_size = world_size or get_global_world_size()
    rank = rank or get_global_rank()
    # logger.warning(f'_initialize_affine_weight({output_size}, {input_size}), rank={rank} / {world_size}')
    if world_size == 1:
        init_method(weight)
        if return_master_weight:
            return weight
        return None

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size,
                                dtype=weight.dtype,
                                requires_grad=False)
    init_method(master_weight)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size,
                              dim=partition_dim)
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.
    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """
    def __init__(self, input_size, output_size, bias=True,
                 init_method=init.xavier_normal_, 
                 stride=1,
                 keep_master_weight_for_test=False,
                 world_size=None, rank=None,
                 ):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        # Divide the weight matrix along the last dimension.
        world_size = world_size or get_global_world_size()
        rank = rank or get_global_rank()
        self.total_output_size = output_size * world_size

        self.output_size_per_partition = divide(self.total_output_size, world_size)
        assert self.output_size_per_partition == self.output_size

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(torch.Tensor(self.output_size_per_partition,
                                             self.input_size))
        self.weight.model_parallel = True
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size_per_partition))
            self.bias.model_parallel = True
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

        # Initialize weight.
        self.master_weight = _initialize_affine_weight(
            self.weight, self.total_output_size, self.input_size,
            self.output_size_per_partition, 0, init_method,
            stride=stride, return_master_weight=keep_master_weight_for_test,
            world_size=world_size, rank=rank
        )
    
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, total_out_features={}'.format(
            self.input_size, self.output_size, self.bias is not None, self.total_output_size
        )

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)


class GpuFFNThorTransformerDecoderLayerBase(TransformerDecoderLayerBase):
    def __init__(self, cfg, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False, layer_idx=-1, world_size=None, rank=None):
        self.num_experts = cfg.num_experts
        self.inference_level = cfg.inference_level
        self.expert_increment = cfg.expert_increment

        self.world_size = world_size or self.num_experts
        self.rank = rank
        self.use_expert = use_expert(layer_idx, self.expert_increment)
        super().__init__(cfg, no_encoder_attn=no_encoder_attn, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        if self.use_expert:
            assert q_noise <= 0
            linear = ColumnParallelLinear(
                input_dim, output_dim, world_size=self.world_size, rank=self.rank
            )
        else:
            linear = quant_noise(
                nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
            )
        return linear

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        if self.use_expert:
            assert q_noise <= 0
            linear = ColumnParallelLinear(
                input_dim, output_dim, world_size=self.world_size, rank=self.rank
            )
        else:
            linear = quant_noise(
                nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
            )
        return linear
    
    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None


class GpuAttTransformerDecoderLayerBase(GpuFFNThorTransformerDecoderLayerBase):
    def __init__(self, cfg, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False, layer_idx=-1, world_size=None, rank=None):
        self.cfg = cfg
        self.fc_expert = not cfg.no_fc_expert
        self.self_att_expert = cfg.self_att_expert
        self.enc_att_expert = cfg.enc_att_expert
        super().__init__(
            cfg, no_encoder_attn=no_encoder_attn, add_bias_kv=add_bias_kv, 
            add_zero_attn=add_zero_attn, layer_idx=layer_idx, world_size=world_size, rank=rank)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        if self.use_expert and self.fc_expert:
            assert q_noise <= 0
            linear = ColumnParallelLinear(
                input_dim, output_dim, world_size=self.world_size, rank=self.rank
            )
        else:
            linear = quant_noise(
                nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
            )
        return linear

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        if self.use_expert and self.fc_expert:
            assert q_noise <= 0
            linear = ColumnParallelLinear(
                input_dim, output_dim, world_size=self.world_size, rank=self.rank
            )
        else:
            linear = quant_noise(
                nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
            )
        return linear
    
    def build_self_attention(
        self, embed_dim, cfg, add_bias_kv=False, add_zero_attn=False
    ):
        if self.use_expert and self.self_att_expert:
            return CollumParallelMultiheadAttention(
                embed_dim,
                cfg.decoder.attention_heads,
                dropout=cfg.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=not cfg.cross_self_attention,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
                world_size=self.world_size, rank=self.rank
            )
        else:
            return MultiheadAttention(
                embed_dim,
                cfg.decoder.attention_heads,
                dropout=cfg.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=not cfg.cross_self_attention,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )

    def build_encoder_attention(self, embed_dim, cfg):
        if self.use_expert and self.enc_att_expert:
            return CollumParallelMultiheadAttention(
                embed_dim,
                cfg.decoder.attention_heads,
                kdim=cfg.encoder.embed_dim,
                vdim=cfg.encoder.embed_dim,
                dropout=cfg.attention_dropout,
                encoder_decoder_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
                world_size=self.world_size, rank=self.rank
            )
        else:
            return MultiheadAttention(
                embed_dim,
                cfg.decoder.attention_heads,
                kdim=cfg.encoder.embed_dim,
                vdim=cfg.encoder.embed_dim,
                dropout=cfg.attention_dropout,
                encoder_decoder_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )


# backward compatible with the legacy argparse format
class GpuFFNThorTransformerDecoderLayer(GpuFFNThorTransformerDecoderLayerBase):
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(
            ThorTransformerConfig.from_namespace(args),
            no_encoder_attn=no_encoder_attn,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.args = args

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return super().build_self_attention(
            embed_dim,
            ThorTransformerConfig.from_namespace(args),
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

    def build_encoder_attention(self, embed_dim, args):
        return super().build_encoder_attention(
            embed_dim,
            ThorTransformerConfig.from_namespace(args),
        )

# backward compatible with the legacy argparse format
class GpuAttTransformerDecoderLayer(GpuAttTransformerDecoderLayerBase):
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(
            GpuAttSharedLangTransformerConfig.from_namespace(args),
            no_encoder_attn=no_encoder_attn,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.args = args

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return super().build_self_attention(
            embed_dim,
            GpuAttSharedLangTransformerConfig.from_namespace(args),
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

    def build_encoder_attention(self, embed_dim, args):
        return super().build_encoder_attention(
            embed_dim,
            GpuAttSharedLangTransformerConfig.from_namespace(args),
        )


class GpuFFNSharedLangThorTransformerEncoderLayerBase(TransformerEncoderLayerBase):
    def __init__(self, cfg, layer_idx=-1, world_size=None, rank=None):
        self.num_experts = cfg.num_experts
        self.inference_level = cfg.inference_level
        self.expert_increment = cfg.expert_increment

        self.world_size = world_size or self.num_experts
        self.rank = rank
        self.use_expert = use_expert(layer_idx, self.expert_increment)
        super().__init__(cfg)
    
    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        if self.use_expert:
            assert q_noise <= 0
            linear = ColumnParallelLinear(
                input_dim, output_dim, world_size=self.world_size, rank=self.rank
            )
            shared_linear = quant_noise(
                nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
            )
            linear = nn.ModuleDict({
                "fc": linear,
                "shfc": shared_linear
            })
        else:
            linear = quant_noise(
                nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
            )
        return linear

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        if self.use_expert:
            assert q_noise <= 0
            linear = ColumnParallelLinear(
                input_dim, output_dim, world_size=self.world_size, rank=self.rank
            )
            shared_linear = quant_noise(
                nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
            )
            linear = nn.ModuleDict({
                "fc": linear,
                "shfc": shared_linear
            })
        else:
            linear = quant_noise(
                nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
            )
        return linear
    
    def _forward_ffn(self, ffn, x, shared_mask):
        if not self.use_expert:
            return ffn(x)
        # if shared_mask.dim() == 1:
        assert shared_mask.dim() == 1
        assert shared_mask.size(0) == x.size(1), f'{shared_mask.size()=} != {x.size()=}'
        # shared_mask = shared_mask.unsqueeze(0).unsqueeze(-1)
        shared_mask = shared_mask[None, :, None]
        outx = torch.where(shared_mask, ffn['shfc'](x), ffn['fc'](x))
        return outx
    
    def forward(
        self,
        x,
        shared_mask: torch.Tensor,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
    ):
        # FIXME: PROBLEM: in encoder, x does not have language specifier
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self._forward_ffn(self.fc1, x, shared_mask=shared_mask))
        x = self.activation_dropout_module(x)
        x = self._forward_ffn(self.fc2, x, shared_mask=shared_mask)

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


# backward compatible with the legacy argparse format
class GpuFFNSharedLangThorTransformerEncoderLayer(GpuFFNSharedLangThorTransformerEncoderLayerBase):
    def __init__(self, args):
        super().__init__(GpuFFNSharedLangThorTransformerConfig.from_namespace(args))
        self.args = args

    def build_self_attention(self, embed_dim, args):
        return super().build_self_attention(
            embed_dim, GpuFFNSharedLangThorTransformerConfig.from_namespace(args)
        )


class GpuFFNSharedLangThorTransformerDecoderLayerBase(TransformerDecoderLayerBase):
    def __init__(self, cfg, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False, layer_idx=-1, world_size=None, rank=None):
        self.num_experts = cfg.num_experts
        self.inference_level = cfg.inference_level
        self.expert_increment = cfg.expert_increment

        self.world_size = world_size or self.num_experts
        self.rank = rank
        self.use_expert = use_expert(layer_idx, self.expert_increment)
        super().__init__(cfg, no_encoder_attn=no_encoder_attn, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        if self.use_expert:
            assert q_noise <= 0
            linear = ColumnParallelLinear(
                input_dim, output_dim, world_size=self.world_size, rank=self.rank
            )
            shared_linear = quant_noise(
                nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
            )
            linear = nn.ModuleDict({
                "fc": linear,
                "shfc": shared_linear
            })
        else:
            linear = quant_noise(
                nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
            )
        return linear

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        if self.use_expert:
            assert q_noise <= 0
            linear = ColumnParallelLinear(
                input_dim, output_dim, world_size=self.world_size, rank=self.rank
            )
            shared_linear = quant_noise(
                nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
            )
            linear = nn.ModuleDict({
                "fc": linear,
                "shfc": shared_linear
            })
        else:
            linear = quant_noise(
                nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
            )
        return linear
    
    def _forward_ffn(self, ffn, x, shared_mask):
        if not self.use_expert:
            return ffn(x)
        # if shared_mask.dim() == 1:
        assert shared_mask.dim() == 1
        assert shared_mask.size(0) == x.size(1), f'{shared_mask.size()=} != {x.size()=}'
        # shared_mask = shared_mask.unsqueeze(0).unsqueeze(-1)
        # shared_mask = shared_mask[None, :, None]
        # outx = torch.where(shared_mask, ffn['shfc'](x), ffn['fc'](x))
        # TODO: better implementatioon
        outx = torch.empty(tuple(list(x.size())[:-1] + [ffn['shfc'].out_features]), dtype=x.dtype, device=x.device)
        outx[:, shared_mask] = ffn['shfc'](x[:, shared_mask])
        outx[:, ~shared_mask] = ffn['fc'](x[:, ~shared_mask])
        return outx
    
    def forward(
        self,
        x,
        shared_mask: torch.Tensor,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            shared_mask (Tensor): true if x to use shared ffn `(batch)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self._forward_ffn(self.fc1, x, shared_mask=shared_mask))
        x = self.activation_dropout_module(x)
        x = self._forward_ffn(self.fc2, x, shared_mask=shared_mask)

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None


class GpuFFNSeparateSharedLangThorTransformerDecoderLayerBase(GpuFFNSharedLangThorTransformerDecoderLayerBase):
    def __init__(self, cfg, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False, layer_idx=-1, world_size=None, rank=None):
        self.use_separate_shared = self.get_use_separate_shared(layer_idx, cfg)
        super().__init__(cfg, no_encoder_attn=no_encoder_attn, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn,
            layer_idx=layer_idx, world_size=world_size, rank=rank
        )
    
    def get_use_separate_shared(self, layer_idx, cfg):
        return layer_idx % cfg.expert_increment != 0
    
    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        if self.use_expert:
            assert q_noise <= 0
            linear = ColumnParallelLinear(
                input_dim, output_dim, world_size=self.world_size, rank=self.rank
            )
            shared_linear = quant_noise(
                nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
            )
            linear = nn.ModuleDict({
                "fc": linear,
                "shfc": shared_linear
            })
        elif self.use_separate_shared:
            linear = nn.ModuleDict({
                "fc": quant_noise(
                    nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
                ),
                "shfc": quant_noise(
                    nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
                )
            })
        else:
            linear = quant_noise(
                nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
            )
        return linear

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        if self.use_expert:
            assert q_noise <= 0
            linear = ColumnParallelLinear(
                input_dim, output_dim, world_size=self.world_size, rank=self.rank
            )
            shared_linear = quant_noise(
                nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
            )
            linear = nn.ModuleDict({
                "fc": linear,
                "shfc": shared_linear
            })
        elif self.use_separate_shared:
            linear = nn.ModuleDict({
                "fc": quant_noise(
                    nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
                ),
                "shfc": quant_noise(
                    nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
                )
            })
        else:
            linear = quant_noise(
                nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
            )
        return linear
    
    def _forward_ffn(self, ffn, x, shared_mask):
        if not (self.use_expert or self.use_separate_shared):
            return ffn(x)
        assert shared_mask.dim() == 1
        assert shared_mask.size(0) == x.size(1), f'{shared_mask.size()=} != {x.size()=}'
        # TODO: this avoid unsused params but has computation and memory pressure
        # shared_mask = shared_mask[None, :, None]
        # # shared_mask: [slen, bsz, dim]
        # outx = torch.where(shared_mask, ffn['shfc'](x), ffn['fc'](x))
        
        # TODO: better implementatioon
        outx = torch.empty(tuple(list(x.size())[:-1] + [ffn['shfc'].out_features]), dtype=x.dtype, device=x.device)
        outx[:, shared_mask] = ffn['shfc'](x[:, shared_mask])
        outx[:, ~shared_mask] = ffn['fc'](x[:, ~shared_mask])
        return outx


class LangFFNTransformerDecoderLayerBase(TransformerDecoderLayerBase):
    def __init__(self, cfg, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False, layer_idx=-1, world_size=None, rank=None):
        self.num_experts = cfg.num_experts
        self.inference_level = cfg.inference_level
        self.expert_increment = cfg.expert_increment

        self.world_size = world_size or self.num_experts
        self.rank = rank
        self.use_expert = use_expert(layer_idx, self.expert_increment)
        super().__init__(cfg, no_encoder_attn=no_encoder_attn, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn)
    
    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        linear = quant_noise(
            nn.Linear(
                input_dim, 
                output_dim * (self.num_experts if self.use_expert else 1)
            ), p=q_noise, block_size=qn_block_size
        )
        return linear

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        linear = quant_noise(
            nn.Linear(
                input_dim, 
                output_dim * (self.num_experts if self.use_expert else 1)
            ), p=q_noise, block_size=qn_block_size
        )
        return linear
    
    def _forward_ffn(self, ffn, x, select_experts):
        # x: (slen, bsz, dim)
        # select_indices: (bsz)
        # out: (slen, bsz, experts, dim)
        # gather_indices (slen, bsz, 1, dim)
        if not self.use_expert:
            return ffn(x)
        # outx = torch.where(shared_mask, ffn['shfc'](x), ffn['fc'](x))
        slen, bsz, dim = x.size()
        out = ffn(x)
        out_dim = out.size(-1) // self.num_experts
        assert out.dim() == 3, f'{out.dim()=} wrong!'
        out = out.view(*(list(out.size()[:-1]) + [self.num_experts, out_dim]))
        gather_indices = select_experts[None, :, None, None].expand(slen, -1, -1, dim)
        outx = out.gather(index=gather_indices, dim=2)
        return outx
    
    def forward(
        self,
        x,
        select_experts: torch.Tensor,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            select_experts (Tensor): select experts for each item (batch)
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self._forward_ffn(self.fc1, x, select_experts=select_experts))
        x = self.activation_dropout_module(x)
        x = self._forward_ffn(self.fc2, x, select_experts=select_experts)

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None


class LangFFNListTransformerDecoderLayerBase(TransformerDecoderLayerBase):
    def __init__(self, cfg, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False, layer_idx=-1, world_size=None, rank=None):
        self.num_experts = cfg.num_experts
        self.inference_level = cfg.inference_level
        self.expert_increment = cfg.expert_increment

        self.world_size = world_size or self.num_experts
        self.rank = rank
        self.use_expert = use_expert(layer_idx, self.expert_increment)
        super().__init__(cfg, no_encoder_attn=no_encoder_attn, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn)
    
    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        if self.use_expert:
            return nn.ModuleList(
                [
                    quant_noise(
                        nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
                    )
                    for _ in range(self.num_experts)
                ]
            )
        else:
            return quant_noise(
                nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
            )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        if self.use_expert:
            return nn.ModuleList(
                [
                    quant_noise(
                        nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
                    )
                    for _ in range(self.num_experts)
                ]
            )
        else:
            return quant_noise(
                nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
            )
    
    def _forward_ffn(self, ffn, x, select_experts):
        # x: (slen, bsz, dim)
        # select_indices: (bsz)
        # out: (slen, bsz, experts, dim)
        # gather_indices (slen, bsz, 1, dim)
        if not self.use_expert or not isinstance(ffn, nn.ModuleList):
            return ffn(x)
        # outx = torch.where(shared_mask, ffn['shfc'](x), ffn['fc'](x))
        slen, bsz, dim = x.size()

        output = torch.empty(size=(slen, bsz, ffn[0].out_features), dtype=x.dtype, device=x.device)
        for i in range(self.num_experts):
            be_expert = select_experts == i
            _input = x[:, be_expert]
            _out = ffn[i](_input)
            output[:, be_expert] = _out

        return output
    
    def forward(
        self,
        x,
        select_experts: torch.Tensor,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            select_experts (Tensor): select experts for each item (batch)
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self._forward_ffn(self.fc1, x, select_experts=select_experts))
        x = self.activation_dropout_module(x)
        x = self._forward_ffn(self.fc2, x, select_experts=select_experts)

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None


# backward compatible with the legacy argparse format
class LangFFNListTransformerDecoderLayer(LangFFNListTransformerDecoderLayerBase):
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(
            LangFFNListTransformerConfig.from_namespace(args),
            no_encoder_attn=no_encoder_attn,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.args = args

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return super().build_self_attention(
            embed_dim,
            LangFFNListTransformerConfig.from_namespace(args),
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

    def build_encoder_attention(self, embed_dim, args):
        return super().build_encoder_attention(
            embed_dim,
            LangFFNListTransformerConfig.from_namespace(args),
        )