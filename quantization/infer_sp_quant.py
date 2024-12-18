import argparse
import math
import pickle
import subprocess
from contextlib import nullcontext
from typing import Any, List, Optional

import einops
import torch
import torch.distributed as dist
import torch.nn.functional as F
from attnq import attnq_pv, attnq_qk, softmaxq

# from apex.normalization.fused_layer_norm import MixedFusedLayerNorm
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from torch.nn.parameter import Parameter

from megatron import core
from megatron.core.utils import divide

# from megatron.model import LayerNorm
# from megatron.model.enums import AttnMaskType
from megatron.model.fused_bias_gelu import bias_gelu_impl
from zutils import net as znet

GENERATION_SERVER_IP = "192.168.0.2"
GENERATION_SERVER_PORT = 43211
MODEL_IPS = [GENERATION_SERVER_IP, "192.168.0.3"]
MODEL_GPUS = [[0], [0]]
CMD_SERVER_PORT = 34119
NCCL_MASTER_PORT = 43214
CODE_NAME_FOR_SHELL = "infer_sp_quant.py"
CODE_PATH_FOR_SHELL = "/home/zeyu/Megatron-DeepSpeed/quantization"


# Inference config
CONFIG = {}
CONFIG["param_path"] = "/home/zeyu/large_files/gpt_params.pkl"
CONFIG["tokenizer_path"] = "/home/zeyu/Megatron-DeepSpeed/gpt3_infer/gpt_tokenizer_kernel.pkl"


CONTEXT_LEN = 2048
HIDDEN_SIZE = 2048
NUM_LAYERS = 24
NUM_HEADS = 16
MLP_HIDDEN_SIZE = 3072
VOCAB_SIZE = 50304
assert HIDDEN_SIZE % NUM_HEADS == 0


SYNTHETIC_PARAMS = True
FLASH_ATTENTION = False


class _VocabEmbedding(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weight = Parameter(
            torch.empty(
                VOCAB_SIZE,
                HIDDEN_SIZE,
                dtype=torch.float16,
            )
        )
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input_):
        masked_input = input_
        output = F.embedding(masked_input, self.weight, None, None, 2.0, False, False)
        return output


class Embedding(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.word_embeddings = _VocabEmbedding()
        self.position_embeddings = torch.nn.Embedding(CONTEXT_LEN, HIDDEN_SIZE, dtype=torch.float16)
        self.embedding_dropout = torch.nn.Dropout(0.1)

    def forward(self, input_ids, position_ids):
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings
        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        embeddings = embeddings.transpose(0, 1).contiguous()
        # Dropout
        embeddings = self.embedding_dropout(embeddings)
        return embeddings


class _QKVLinear(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weight = Parameter(
            torch.empty(
                3 * HIDDEN_SIZE,
                HIDDEN_SIZE,
                dtype=torch.float16,
            )
        )
        self.bias = Parameter(torch.empty(3 * HIDDEN_SIZE, dtype=torch.float16))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input_):
        output = torch.matmul(input_, self.weight.t())
        return output + self.bias, None


class _CoreAttention(torch.nn.Module):
    def __init__(self, layer_num, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_num = layer_num
        self.norm_factor = math.sqrt(HIDDEN_SIZE / NUM_HEADS)
        coeff = self.layer_num
        self.norm_factor *= coeff
        self.attention_dropout = torch.nn.Dropout(0.1)

    def forward(self, query_layer, key_layer, value_layer, quant_factors, causal):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================
        qmin, qscale, kmin, kscale, vmin, vscale = quant_factors

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = einops.rearrange(query_layer, "s b h d -> b h s d").to(torch.int8)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = einops.rearrange(key_layer, "s b h d -> b h s d").to(torch.int8)

        qmin = einops.rearrange(qmin.squeeze(3), "s b h -> b h s")
        qscale = einops.rearrange(qscale.squeeze(3), "s b h -> b h s")
        kmin = einops.rearrange(kmin.squeeze(3), "s b h -> b h s")
        kscale = einops.rearrange(kscale.squeeze(3), "s b h -> b h s")

        attention_scores = attnq_qk(query_layer, key_layer, qscale, qmin, kscale, kmin)

        quant_prob, pmin, pscale = softmaxq(attention_scores, 1 / math.sqrt(128), causal)

        value_layer = einops.rearrange(value_layer, "s b h d -> b h s d").to(torch.int8)
        vmin = einops.rearrange(vmin.squeeze(4), "g b h d -> b h g d")
        vscale = einops.rearrange(vscale.squeeze(4), "g b h d -> b h g d")
        attn_out = attnq_pv(quant_prob, value_layer, pscale, pmin, vscale, vmin)

        context_layer = einops.rearrange(attn_out, "b h s d -> s b (h d)")

        return context_layer


class _CoreFlashAttention(torch.nn.Module):
    def __init__(self, layer_num, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_num = layer_num

    def forward(self, query_layer, key_layer, value_layer, causal):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================
        batch_size = query_layer.size(1)
        seqlen_q = query_layer.size(0)
        seqlen_k = key_layer.size(0)
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=query_layer.device)
        cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=query_layer.device)
        query_layer = einops.rearrange(query_layer, "s b h d -> (s b) h d")
        key_layer = einops.rearrange(key_layer, "s b h d -> (s b) h d")
        value_layer = einops.rearrange(value_layer, "s b h d -> (s b) h d")
        output = flash_attn_varlen_func(
            query_layer,
            key_layer,
            value_layer,
            cu_seqlens_q,
            cu_seqlens_k,
            seqlen_q,
            seqlen_k,
            0.1,
            causal=causal,
        )
        output = einops.rearrange(output, "(s b) h d -> s b (h d)", b=batch_size, s=seqlen_q)
        return output


class _SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, input: torch.Tensor, scatter_idx: int, gather_idx: int, seq_len_list, is_first_seq_a2a=True
    ) -> torch.Tensor:
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        seq_world_size = dist.get_world_size()

        if is_first_seq_a2a:
            input_list = [t.contiguous() for t in torch.tensor_split(input, seq_world_size, scatter_idx)]
            shape = torch.tensor(input_list[0].shape)
            output_list = []
            for sl in seq_len_list:
                shape[gather_idx] = sl
                output_list.append(torch.empty([x.item() for x in shape], dtype=input.dtype, device=input.device))
        else:
            split_indices = []
            for sl in seq_len_list:
                if len(split_indices) == 0:
                    split_indices.append(sl)
                else:
                    split_indices.append(split_indices[-1] + sl)
            input_list = [t.contiguous() for t in torch.tensor_split(input, split_indices, scatter_idx)]
            shape = torch.tensor(input_list[0].shape)
            shape[scatter_idx] = seq_len_list[dist.get_rank()]
            output_list = [
                torch.empty([x.item() for x in shape], dtype=input.dtype, device=input.device)
                for _ in range(dist.get_world_size())
            ]

        # TODO Use all_to_all_single instead
        dist.all_to_all(output_list, input_list)

        return torch.cat(output_list, dim=gather_idx).contiguous()


class _DistributedAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        core_attention: torch.nn.Module,
        scatter_idx: int = 2,
        gather_idx: int = 0,
    ) -> None:
        super(_DistributedAttention, self).__init__()
        self.core_attn = core_attention
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args: Any) -> torch.Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """

        is_prompt = args[0]
        current_seq_len = args[1]

        quant_factors = args[2]
        qmin = quant_factors[0]
        qscale = quant_factors[1]
        kmin = quant_factors[2]
        kscale = quant_factors[3]
        vmin = quant_factors[4]
        vscale = quant_factors[5]
        q_min_scale = torch.cat((qmin, qscale), dim=-1)
        kv_min_scale = torch.cat((kmin, kscale, vmin, vscale), dim=-1)

        num_tokens = query.size(0)
        num_tokens_tensor = torch.tensor(num_tokens, device=query.device)
        num_tokens_list = [torch.empty_like(num_tokens_tensor, device=query.device) for _ in range(dist.get_world_size())]
        dist.all_gather(num_tokens_list, num_tokens_tensor)
        num_tokens_list = [t.item() for t in num_tokens_list]

        if is_prompt:
            seq_len_list = num_tokens_list
        else:
            seq_len_tensor = torch.tensor(current_seq_len, device=query.device)
            seq_len_list = [torch.empty_like(seq_len_tensor, device=query.device) for _ in range(dist.get_world_size())]
            dist.all_gather(seq_len_list, seq_len_tensor)
            seq_len_list = [t.item() for t in seq_len_list]

        # TODO Merge three alltoall calls into one
        # in shape : e.g.,  [s/p:h:]
        # The current shape is [seq, batch, head, head_dim].
        query_layer = _SeqAllToAll.apply(query, self.scatter_idx, self.gather_idx, num_tokens_list)
        key_layer = _SeqAllToAll.apply(key, self.scatter_idx, self.gather_idx, seq_len_list)
        value_layer = _SeqAllToAll.apply(value, self.scatter_idx, self.gather_idx, seq_len_list)

        q_min_scale = _SeqAllToAll.apply(q_min_scale, self.scatter_idx, self.gather_idx, num_tokens_list)
        kv_min_scale = _SeqAllToAll.apply(kv_min_scale, self.scatter_idx, self.gather_idx, seq_len_list)
        qmin, qscale = torch.tensor_split(q_min_scale, 2, dim=-1)
        kmin, kscale, vmin, vscale = torch.tensor_split(kv_min_scale, 4, dim=-1)

        q1 = query_layer >> 4
        q2 = query_layer << 4 >> 4
        query_layer = torch.cat([q1, q2], dim=-1)
        k1 = key_layer >> 4
        k2 = key_layer << 4 >> 4
        key_layer = torch.cat([k1, k2], dim=-1)
        v1 = value_layer >> 4
        v2 = value_layer << 4 >> 4
        value_layer = torch.cat([v1, v2], dim=-1)

        # change value_layer quantization form to another one
        value_layer = value_layer.to(torch.float16) * vscale + vmin
        v_window = value_layer.unfold(0, 128, 128)
        v_max = v_window.max(dim=4, keepdim=True).values
        v_min = v_window.min(dim=4, keepdim=True).values
        v_scale = (v_max - v_min) / 16
        v_quant = ((v_window - v_min) / v_scale).to(torch.uint8)
        v_quant = einops.rearrange(v_quant, "g b h d s -> (g s) b h d")
        if value_layer.size(0) % 128 != 0:
            remaining = value_layer[-value_layer.size(0) % 128 :, ...]
            remaining = einops.rearrange(remaining, "s b h d -> 1 b h d s")
            max_values_remaining = remaining.max(dim=4, keepdim=True).values
            min_values_remaining = remaining.min(dim=4, keepdim=True).values
            scale_remaining = (max_values_remaining - min_values_remaining) / 16
            v_quant_remaining = ((remaining - min_values_remaining) / scale_remaining).to(torch.uint8)
            v_quant_remaining = einops.rearrange(v_quant_remaining, "g b h d s -> (g s) b h d")
        vmin = torch.cat((v_min, min_values_remaining), dim=0)
        vscale = torch.cat((v_scale, scale_remaining), dim=0)
        value_layer = torch.cat((v_quant, v_quant_remaining), dim=0)

        # out shape : e.g., [s:h/p:]
        context_layer = self.core_attn(
            query_layer, key_layer, value_layer, (qmin, qscale, kmin, kscale, vmin, vscale), is_prompt
        )

        output = _SeqAllToAll.apply(context_layer, self.gather_idx, self.scatter_idx, num_tokens_list, False)

        # out e.g., [s/p::h]
        return output


class _PostAttnLinear(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weight = Parameter(
            torch.empty(
                HIDDEN_SIZE,
                HIDDEN_SIZE,
                dtype=torch.float16,
            )
        )
        self.bias = Parameter(torch.empty(HIDDEN_SIZE, dtype=torch.float16))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input_):
        output = torch.matmul(input_, self.weight.t())
        return output, self.bias


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class Attention(torch.nn.Module):
    def __init__(self, layer_num, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_num = layer_num
        self.kv_channels = HIDDEN_SIZE // NUM_HEADS
        self.query_key_value = _QKVLinear()
        if FLASH_ATTENTION:
            self.core_attn = _CoreFlashAttention(self.layer_num)
        else:
            self.core_attn = _CoreAttention(self.layer_num)
        self.dist_attn = _DistributedAttention(self.core_attn)
        self.dense = _PostAttnLinear()

        # Inference key-value memory
        self.inference_key_memory = None
        self.inference_value_memory = None
        self.inference_current_sequence_len = 0

        # Quantization
        self.quant_kmin = None
        self.quant_kscaling = None
        self.quant_vmin = None
        self.quant_vscaling = None

    def forward(self, hidden_states, set_inference_key_value_memory=False, inference_max_sequence_len=None):
        # hidden_states: [sq, b, h]
        if inference_max_sequence_len is not None and inference_max_sequence_len == 0:
            assert hidden_states.size(0) == 0

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        if set_inference_key_value_memory:
            assert inference_max_sequence_len is not None and inference_max_sequence_len >= 0
            self.inference_key_memory = self._allocate_memory(
                inference_max_sequence_len, hidden_states.size(1), torch.uint8, hidden_states.device
            )
            self.inference_value_memory = self._allocate_memory(
                inference_max_sequence_len, hidden_states.size(1), torch.uint8, hidden_states.device
            )
            self.inference_current_sequence_len = 0

        # Some consistency check.
        if inference_max_sequence_len is not None:
            assert self.inference_current_sequence_len <= self.inference_key_memory.size(0)
            assert inference_max_sequence_len == self.inference_key_memory.size(0)
        # This is added for safety. In case inference_max_sequence_len
        # is not provided, make sure there is no potential memory left
        # from previous inference.
        if inference_max_sequence_len is None:
            self.inference_key_memory = None
            self.inference_value_memory = None

            self.quant_kmin = None
            self.quant_kscaling = None
            self.quant_vmin = None
            self.quant_vscaling = None

        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            NUM_HEADS,
            3 * HIDDEN_SIZE // NUM_HEADS,
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # Quantization
        quant_qmax = torch.max(query_layer, dim=3, keepdim=True).values
        quant_qmin = torch.min(query_layer, dim=3, keepdim=True).values
        quant_qscaling = (quant_qmax - quant_qmin) / 16
        quant_kmax = torch.max(key_layer, dim=3, keepdim=True).values
        quant_kmin = torch.min(key_layer, dim=3, keepdim=True).values
        quant_kscaling = (quant_kmax - quant_kmin) / 16
        quant_vmax = torch.max(value_layer, dim=3, keepdim=True).values
        quant_vmin = torch.min(value_layer, dim=3, keepdim=True).values
        quant_vscaling = (quant_vmax - quant_vmin) / 16
        if self.quant_kmin is None:
            self.quant_kmin = quant_kmin
            self.quant_kscaling = quant_kscaling
            self.quant_vmin = quant_vmin
            self.quant_vscaling = quant_vscaling
        else:
            self.quant_kmin = torch.cat((self.quant_kmin, quant_kmin), dim=0)
            self.quant_kscaling = torch.cat((self.quant_kscaling, quant_kscaling), dim=0)
            self.quant_vmin = torch.cat((self.quant_vmin, quant_vmin), dim=0)
            self.quant_vscaling = torch.cat((self.quant_vscaling, quant_vscaling), dim=0)
        query_layer = ((query_layer - quant_qmin) / quant_qscaling).to(torch.uint8)
        key_layer = ((key_layer - quant_kmin) / quant_kscaling).to(torch.uint8)
        value_layer = ((value_layer - quant_vmin) / quant_vscaling).to(torch.uint8)

        # Pack two uint4 into one uint8
        q1 = query_layer[..., 0:64] << 4
        q2 = query_layer[..., 64:128]
        query_layer = q1 + q2
        k1 = key_layer[..., 0:64] << 4
        k2 = key_layer[..., 64:128]
        key_layer = k1 + k2
        v1 = value_layer[..., 0:64] << 4
        v2 = value_layer[..., 64:128]
        value_layer = v1 + v2

        # ===================================================
        # Adjust key, value, and attention mask for inference
        # ===================================================
        if inference_max_sequence_len:
            # Adjust the range variables.
            start = self.inference_current_sequence_len
            self.inference_current_sequence_len += key_layer.size(0)
            end = self.inference_current_sequence_len
            # Copy key and values.
            self.inference_key_memory[start:end, ...] = key_layer
            self.inference_value_memory[start:end, ...] = value_layer
            key_layer = self.inference_key_memory[:end, ...]
            value_layer = self.inference_value_memory[:end, ...]

        context_layer = self.dist_attn(
            query_layer,
            key_layer,
            value_layer,
            set_inference_key_value_memory,
            self.inference_current_sequence_len,
            (quant_qmin, quant_qscaling, self.quant_kmin, self.quant_kscaling, self.quant_vmin, self.quant_vscaling),
        )
        output, bias = self.dense(context_layer)

        return output, bias

    def _allocate_memory(self, inference_max_sequence_len, batch_size, dtype, device):
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            NUM_HEADS,
            HIDDEN_SIZE // NUM_HEADS // 2,
            dtype=dtype,
            device=device,
        )


class _MLPLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, skip_bias_add=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.skip_bias_add = skip_bias_add
        self.weight = Parameter(
            torch.empty(
                output_size,
                input_size,
                dtype=torch.float16,
            )
        )
        self.bias = Parameter(torch.empty(output_size, dtype=torch.float16))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None
        output = torch.matmul(input_, self.weight.t())
        if bias is not None:
            output += bias
            output_bias = None
        else:
            output_bias = self.bias
        return output, output_bias


class MLP(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dense_h_to_xh = _MLPLinear(HIDDEN_SIZE, MLP_HIDDEN_SIZE, skip_bias_add=True)
        self.bias_gelu_fusion = True
        self.activation_func = F.gelu
        self.dense_xh_to_h = _MLPLinear(MLP_HIDDEN_SIZE, HIDDEN_SIZE)

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate, bias = self.dense_h_to_xh(hidden_states)

        if intermediate.size(0) == 0:
            pass
        else:
            if self.bias_gelu_fusion:
                intermediate = bias_gelu_impl(intermediate, bias)
            else:
                if bias is not None:
                    intermediate = intermediate + bias
                intermediate = self.activation_func(intermediate)

        # [s, b, h]
        output, output_bias = self.dense_xh_to_h(intermediate)
        return output, output_bias


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Optional[Tensor], Tensor, float, bool) -> Tensor
    if bias is not None:
        x = x + bias
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    out = residual + out
    return out


@torch.jit.script
def bias_dropout_add_fused_inference(
    x: torch.Tensor, bias: Optional[torch.Tensor], residual: torch.Tensor, prob: float
) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, False)


class TransformerLayer(torch.nn.Module):
    def __init__(self, layer_num, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_num = layer_num
        self.input_layernorm = torch.nn.LayerNorm(HIDDEN_SIZE, 1e-5)
        torch.nn.init.constant_(self.input_layernorm.weight, 0.5)
        # Self attention.
        self.self_attention = Attention(layer_num)

        self.hidden_dropout = 0.1

        self.post_attention_layernorm = torch.nn.LayerNorm(HIDDEN_SIZE, 1e-5)
        torch.nn.init.constant_(self.post_attention_layernorm.weight, 0.5)

        # MLP
        self.mlp = MLP()

        # Set bias+dropout+add fusion grad_enable execution handler.
        TORCH_MAJOR = int(torch.__version__.split(".")[0])
        TORCH_MINOR = int(torch.__version__.split(".")[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad

    def forward(self, hidden_states, set_inference_key_value_memory=False, inference_max_sequence_len=None):
        layernorm_output = self.input_layernorm(hidden_states)
        layernorm_output = torch.where(torch.isnan(layernorm_output), torch.tensor(0.1), layernorm_output)
        attention_output, attention_bias = self.self_attention(
            layernorm_output, set_inference_key_value_memory, inference_max_sequence_len
        )
        residual = hidden_states
        bias_dropout_add_func = bias_dropout_add_fused_inference
        if attention_bias is not None:
            attention_bias = attention_bias.expand_as(residual)
        with self.bias_dropout_add_exec_handler():
            layernorm_input = bias_dropout_add_func(attention_output, attention_bias, residual, self.hidden_dropout)
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        layernorm_output = torch.where(torch.isnan(layernorm_output), torch.tensor(0.1), layernorm_output)

        # MLP.
        mlp_bias = torch.tensor(0.0, device=layernorm_output.device, dtype=layernorm_output.dtype)

        mlp_output, mlp_bias = self.mlp(layernorm_output)

        residual = layernorm_input

        if mlp_bias is not None:
            mlp_bias = mlp_bias.expand_as(residual)
        with self.bias_dropout_add_exec_handler():
            output = bias_dropout_add_func(mlp_output, mlp_bias, residual, self.hidden_dropout)
        output = core.utils.make_viewless_tensor(inp=output, requires_grad=output.requires_grad, keep_graph=True)

        return output


class Transformer(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        def _build_layer(layer_num):
            return TransformerLayer(layer_num)

        self.layers = []
        for i in range(NUM_LAYERS):
            layer_num = i + 1
            self.layers.append(_build_layer(layer_num))
        self.layers = torch.nn.ModuleList(self.layers)

        self.final_layernorm = torch.nn.LayerNorm(HIDDEN_SIZE, 1e-5)
        torch.nn.init.constant_(self.final_layernorm.weight, 0.5)

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(self, hidden_states, set_inference_key_value_memory=False, inference_max_sequence_len=None):
        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = core.utils.make_viewless_tensor(
            inp=hidden_states,
            requires_grad=hidden_states.requires_grad,
            keep_graph=True,
        )

        def custom(start, end):
            def custom_forward(*args, **kwargs):
                x_, *args = args
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, *args, **kwargs)
                return x_

            return custom_forward

        for ly in range(NUM_LAYERS):
            hidden_states = custom(ly, ly + 1)(hidden_states, set_inference_key_value_memory, inference_max_sequence_len)

        hidden_states = self.final_layernorm(hidden_states)
        hidden_states = torch.where(torch.isnan(hidden_states), torch.tensor(0.1), hidden_states)

        return hidden_states


class GPTTokenizer:
    def __init__(self, tokenizer_pickle_path) -> None:
        self.tokenizer = None
        with open(tokenizer_pickle_path, "rb") as file:
            self.tokenizer = pickle.load(file)

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    def text_to_tokens(self, text):
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def tokens_to_text(self, tokens):
        text = self.tokenizer.convert_tokens_to_string(tokens)
        return text

    def token_to_id(self, token):
        return self.tokens_to_ids([token])[0]

    def tokens_to_ids(self, tokens):
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids

    def ids_to_tokens(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return tokens

    def text_to_ids(self, text):
        tokens = self.text_to_tokens(text)
        ids = self.tokens_to_ids(tokens)
        return ids

    def ids_to_text(self, ids):
        tokens = self.ids_to_tokens(ids)
        tokens_clean = [t for t in tokens if t not in self.tokenizer.all_special_tokens]
        text = self.tokens_to_text(tokens_clean)
        return text

    @property
    def vocab(self):
        id2vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        return [id2vocab[i] for i in range(len(id2vocab))]

    @property
    def pad_id(self):
        return self.tokens_to_ids([None])[0]

    @property
    def bos_id(self):
        return self.tokens_to_ids(["<|endoftext|>"])[0]

    @property
    def eos_id(self):
        return self.tokens_to_ids(["<|endoftext|>"])[0]

    @property
    def sep_id(self):
        return self.tokens_to_ids(["<|endoftext|>"])[0]

    @property
    def cls_id(self):
        return self.tokens_to_ids(["<|endoftext|>"])[0]

    @property
    def unk_id(self):
        return self.tokens_to_ids(["<|endoftext|>"])[0]

    @property
    def mask_id(self):
        return self.tokens_to_ids([None])[0]

    @property
    def name(self):
        return type(self.tokenizer).__name__


class GPTModel(torch.nn.Module):
    def __init__(self, param_pickle_path=None, device: str = "cuda", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.enc_total_context_len = CONTEXT_LEN

        self.device = torch.device(device)

        torch.cuda.set_device(self.device)

        self.embedding = Embedding()
        self.encoder = Transformer()

        self.half()

        if param_pickle_path:
            self._init_model(param_pickle_path)

        self.to(self.device)
        self.eval()

    def forward(self, inputs):
        (enc_input_ids, enc_position_ids, set_inference_key_value_memory, inference_max_sequence_len) = inputs
        enc_input_ids = enc_input_ids.to(self.device)
        enc_position_ids = enc_position_ids.to(self.device)
        encoder_input = self.embedding(enc_input_ids, enc_position_ids)
        encoder_output = self.encoder(encoder_input, set_inference_key_value_memory, inference_max_sequence_len)
        word_embeddings_weight = self.embedding.word_embeddings.weight
        logits = torch.matmul(encoder_output, word_embeddings_weight.t())
        return logits.transpose(0, 1).contiguous()

    def _init_model(self, param_pickle_path):
        if not SYNTHETIC_PARAMS:
            with open(param_pickle_path, "rb") as file:
                params = pickle.load(file)
            for (key, _), (_, param) in zip(self.named_parameters(), params):
                self.state_dict()[key].copy_(param.half())


class DistributedGPTModelWorker:
    def __init__(self) -> None:
        num_wrks = 0
        for node in MODEL_GPUS:
            for _ in node:
                num_wrks += 1
        self.num_wrks = num_wrks

        self.generation_server_conn = None

    def run(self, rank):
        world_size = self.num_wrks

        gpus_idx = 0
        gpu_id = None
        for node in MODEL_GPUS:
            for gid in node:
                if gpus_idx == rank:
                    gpu_id = gid
                    break
                gpus_idx += 1
            if gpu_id is not None:
                break
        assert gpu_id is not None

        self.generation_server_conn = znet.SocketMsger.tcp_connect(GENERATION_SERVER_IP, GENERATION_SERVER_PORT)
        self.generation_server_conn.send(rank)
        model = GPTModel(CONFIG["param_path"], f"cuda:{gpu_id}")
        # get cmd "START"
        self.generation_server_conn.recv()
        dist.init_process_group(
            "nccl", init_method=f"tcp://{MODEL_IPS[0]}:{NCCL_MASTER_PORT}", rank=rank, world_size=world_size
        )

        while True:
            inputs = self.generation_server_conn.recv()
            if isinstance(inputs, str) and inputs == "EXIT":
                self.generation_server_conn.close()
                return
            with torch.no_grad():
                output = model(inputs)
            self.generation_server_conn.send(output.cpu())


class TextGeneration:
    def __init__(self) -> None:
        num_wrks = 0
        for node in MODEL_GPUS:
            for _ in node:
                num_wrks += 1
        self.num_wrks = num_wrks
        self.encoder_seq_length = self.num_wrks * CONTEXT_LEN

        self.model_instance_conns = {}

    def run(self, inputs, max_gen_len):
        gen_listener = znet.SocketMsger.tcp_listener("0.0.0.0", GENERATION_SERVER_PORT)

        while True:
            conn, _ = gen_listener.accept()
            rank = conn.recv()
            self.model_instance_conns[rank] = conn
            if len(self.model_instance_conns) == self.num_wrks:
                break

        for i in range(self.num_wrks):
            self.model_instance_conns[i].send("START")

        resp_sentences, resp_sentences_seg = self._generate(inputs, max_gen_len)

        print(resp_sentences, resp_sentences_seg)

    def _generate(self, inputs, max_gen_len):
        tokenizer = GPTTokenizer(CONFIG["tokenizer_path"])

        token_pool, prompt_lengths = self._tokenize_batch(tokenizer, inputs, max_gen_len)
        infer_cursor = prompt_lengths.min().item()
        position_ids = self._get_position_ids(token_pool)

        eod_id = tokenizer.eos_id
        kvc_allocated = False

        batch_size = token_pool.size(0)
        is_done = torch.zeros([batch_size], device=token_pool.device).byte()
        output_logits = None
        all_generated_indices = None  # used to track all generated indices
        max_len = max_gen_len + prompt_lengths.max().item()

        if max_len > self.encoder_seq_length + 1:
            max_len = self.encoder_seq_length + 1

        while infer_cursor < max_len:
            # types2use = None
            if not kvc_allocated:
                # Allocate memory for the entire context.
                set_inference_key_value_memory = True
                tokens2use = token_pool[:, :infer_cursor]
                positions2use = position_ids[:, :infer_cursor]
            else:
                # Set this to false so the memory is not reallocated.
                set_inference_key_value_memory = False
                tokens2use = token_pool[:, infer_cursor - 1].view(batch_size, -1)
                positions2use = position_ids[:, infer_cursor - 1].view(batch_size, -1)

            if not kvc_allocated:
                tokens2use_list = None
                positions2use_list = None
                split_indices = []
                remaining_len = infer_cursor
                for i in range(self.num_wrks):
                    if remaining_len >= CONTEXT_LEN:
                        remaining_len -= CONTEXT_LEN
                        split_indices.append((i + 1) * CONTEXT_LEN)
                    else:
                        split_indices.append(i * CONTEXT_LEN + remaining_len)
                        break
                tokens2use_list = [t.contiguous() for t in torch.tensor_split(tokens2use, split_indices, 1)]
                if len(tokens2use_list) > self.num_wrks:
                    tokens2use_list.pop()
                positions2use_list = [t.contiguous() for t in torch.tensor_split(positions2use, split_indices, 1)]
                if len(positions2use_list) > self.num_wrks:
                    positions2use_list.pop()
                if len(tokens2use_list) < self.num_wrks:
                    for _ in range(self.num_wrks - len(tokens2use_list)):
                        tokens2use_list.append(torch.empty([batch_size, 0], dtype=tokens2use.dtype))
                        positions2use_list.append(torch.empty([batch_size, 0], dtype=positions2use.dtype))
            else:
                wrk_id = int(infer_cursor / CONTEXT_LEN)
                tokens2use_list = [torch.empty([batch_size, 0], dtype=tokens2use.dtype) for _ in range(self.num_wrks)]
                positions2use_list = [torch.empty([batch_size, 0], dtype=positions2use.dtype) for _ in range(self.num_wrks)]
                tokens2use_list[wrk_id] = tokens2use
                positions2use_list[wrk_id] = positions2use

            max_infer_len_list = []
            remaining_len = max_len - 1
            for _ in range(self.num_wrks):
                if remaining_len >= CONTEXT_LEN:
                    remaining_len -= CONTEXT_LEN
                    max_infer_len_list.append(CONTEXT_LEN)
                elif remaining_len > 0:
                    max_infer_len_list.append(remaining_len)
                    remaining_len = 0
                else:
                    max_infer_len_list.append(0)

            for i in range(self.num_wrks):
                tokens2use = tokens2use_list[i]
                positions2use = positions2use_list[i]
                max_infer_len = max_infer_len_list[i]
                inputs = [tokens2use, positions2use, set_inference_key_value_memory, max_infer_len]
                model_conn = self.model_instance_conns[i]
                model_conn.send(inputs)

            all_outputs = []
            for i in range(self.num_wrks):
                model_conn = self.model_instance_conns[i]
                all_outputs.append(model_conn.recv())
            output = torch.cat(all_outputs, 1)

            assert output is not None
            output = output.float()
            logits = output[:, -1].view(batch_size, -1).contiguous()

            # make sure it won't sample outside the vocab_size range
            logits[:, tokenizer.vocab_size :] = -float("Inf")

            logits /= 1.0  # temperature_value
            # handle repetition penality
            logits = self._repetition_penalty(logits, 1.2, all_generated_indices)
            logits = self._top_k_logits(logits, top_k=0, top_p=0.9)
            log_probs = F.softmax(logits, dim=-1)
            prev = torch.multinomial(log_probs, num_samples=1).view(-1)

            started = prompt_lengths <= infer_cursor

            # Clamp the predicted out of vocabulary tokens
            prev = torch.clamp(prev, max=tokenizer.vocab_size - 1)
            new_tokens = self._switch(token_pool[:, infer_cursor].view(-1), prev, started)

            # Replace sampled tokens w/ done token if EOD has already been sampled
            new_tokens = self._switch(new_tokens, eod_id, is_done)

            # Insert either new predicted or next prompt token
            token_pool[:, infer_cursor] = new_tokens

            if output_logits is None:
                output = F.log_softmax(output[:, :infer_cursor, :], 2)
                indices = torch.unsqueeze(token_pool[:, 1 : infer_cursor + 1], 2)
                output_logits = torch.gather(output, 2, indices).squeeze(2)
                all_generated_indices = indices[:, :, 0]
            else:
                output = F.log_softmax(output, 2)
                indices = torch.unsqueeze(new_tokens, 1).unsqueeze(2)
                new_output_logits = torch.gather(output, 2, indices).squeeze(2)

                # TODO(rprenger) we're copying output_logits every time.  Should pre-allocate
                output_logits = torch.cat([output_logits, new_output_logits], 1)
                all_generated_indices = torch.cat([all_generated_indices, indices[:, :, 0]], 1)

            done_token = (prev == eod_id).byte() & started.byte()
            is_done = is_done | done_token

            done = torch.all(is_done)

            infer_cursor += 1
            if not kvc_allocated:
                kvc_allocated = True
            if done:
                break

        # token_pool and output_logits can be used after the while loop.
        resp_sentences = []
        resp_sentences_seg = []

        for i in range(self.num_wrks):
            self.model_instance_conns[i].send("EXIT")
            self.model_instance_conns[i].close()

        decode_tokens = token_pool[:, :infer_cursor]
        decode_tokens = decode_tokens.numpy().tolist()
        for decode_token in decode_tokens:
            sentence = tokenizer.ids_to_text(decode_token)
            resp_sentences.append(sentence)

            words = []
            for token in decode_token:
                # Skip any soft prompt pseudo tokens
                if token not in tokenizer.tokenizer.decoder:
                    continue
                word = tokenizer.tokenizer.decoder[token]
                word = bytearray([tokenizer.tokenizer.byte_decoder[c] for c in word]).decode("utf-8", errors="replace")
                words.append(word)
            resp_sentences_seg.append(words)

        return resp_sentences, resp_sentences_seg

    def _pad_batch(self, batch, pad_id, max_len):
        context_lengths = []
        max_context_length = max([len(tokens) for tokens in batch])
        for tokens in batch:
            context_length = len(tokens)
            if context_length < max_context_length + max_len:
                tokens.extend([pad_id] * (max_context_length + max_len - context_length))
            context_lengths.append(context_length)
        return batch, context_lengths

    def _tokenize_batch(self, tokenizer, sentences, max_len, add_BOS=True):
        if add_BOS:
            context_tokens = [[tokenizer.eos_id] + tokenizer.text_to_ids(s) for s in sentences]
        else:
            context_tokens = [tokenizer.text_to_ids(s) for s in sentences]
        context_tokens, context_lengths = self._pad_batch(context_tokens, tokenizer.eos_id, max_len)
        context_tokens_tensor = torch.LongTensor(context_tokens)
        context_length_tensor = torch.LongTensor(context_lengths)
        return context_tokens_tensor, context_length_tensor

    def _get_position_ids(self, tokens):
        batch_size, seq_length = tokens.size()
        pos_ids = [i % CONTEXT_LEN for i in range(seq_length)]
        position_ids = torch.tensor(pos_ids, dtype=torch.long, device=tokens.device)
        position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)
        return position_ids

    def _repetition_penalty(self, logits, repetition_penalty, used_tokens):
        """Implement the repetition penalty, check paper
        https://arxiv.org/pdf/1909.05858.pdf
        """
        if used_tokens is not None and repetition_penalty != 1.0:
            logits_update = torch.gather(logits, 1, used_tokens)
            logits = torch.scatter(logits, 1, used_tokens, logits_update / repetition_penalty)
        return logits

    def _top_k_logits(self, logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
        """This function has been mostly taken from huggingface conversational
        ai code at
            https://medium.com/huggingface/how-to-build-a-state-of-the-art-
                conversational-ai-with-transfer-learning-2d818ac26313"""

        if top_k > 0:
            # Remove all tokens with a probability less than the
            # last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            # Cconvert to 1D
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token
            # above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            for i in range(sorted_indices.size(0)):
                indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                logits[i][indices_to_remove] = filter_value

        return logits

    def _switch(self, val1, val2, boolean):
        boolean = boolean.type_as(val1)
        return (1 - boolean) * val1 + boolean * val2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--model_rank", default=0, type=int)
    parser.add_argument("-s", "--cmd_server", action="store_true")
    parser.add_argument("-w", "--model_worker", action="store_true")
    args = parser.parse_args()

    if args.cmd_server:
        listener = znet.SocketMsger.tcp_listener("0.0.0.0", CMD_SERVER_PORT)
        while True:
            req_conn, _ = listener.accept()
            node_cmds = req_conn.recv()
            for cmd in node_cmds:
                subprocess.call(cmd, shell=True)
            req_conn.close()
    else:
        if args.model_worker:  # now support model_rank == 0 and run it as a worker via cmd_server
            # assert args.model_rank > 0
            model_worker = DistributedGPTModelWorker()
            model_worker.run(args.model_rank)
        # The generation server
        else:
            remote_model_rank = 0
            for node_idx in range(len(MODEL_IPS)):
                node_ip = MODEL_IPS[node_idx]
                node_gpus = MODEL_GPUS[node_idx]
                node_conn = znet.SocketMsger.tcp_connect(node_ip, CMD_SERVER_PORT)
                node_cmds = []
                for _ in node_gpus:
                    node_cmds.append(f"python {CODE_PATH_FOR_SHELL}/{CODE_NAME_FOR_SHELL} -r {remote_model_rank} -w")
                    remote_model_rank += 1
                node_conn.send(node_cmds)
                node_conn.close()
            text_generation = TextGeneration()
            # text_generation.run(["How big is the universe?"], 100)
            text = "how "
            for _ in range(3900):
                text += "how "
            text_generation.run([text], 3)