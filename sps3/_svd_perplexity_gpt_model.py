#! /usr/bin/env python3


import math
import pickle
from contextlib import nullcontext
from typing import List, Optional

import einops
import numpy as np
import torch
import torch.nn.functional as F
from apex.normalization.fused_layer_norm import MixedFusedLayerNorm
from torch.nn.parameter import Parameter

from megatron import core
from megatron.core.utils import divide

# from megatron.model import LayerNorm
from megatron.model.enums import AttnMaskType
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.fused_softmax import FusedScaleMaskSoftmax

IS_TRAINING = False
CONTEXT_LEN = 2048
HIDDEN_SIZE = 2048
FFN_HIDDEN_SIZE = 3072
NUM_HEADS = 16
VOCAB_SIZE = 50304
NUM_LAYERS = 24
assert HIDDEN_SIZE % NUM_HEADS == 0


INF_TARGET_TENSOR = None


with open("/u/qxc4fh/zeyu_workspace/svd_usvh_v_for_qk.pkl", "rb") as f:
    svd_usvh_v_for_qk = pickle.load(f)
with open("/u/qxc4fh/zeyu_workspace/svd_usvh_v_for_vlinear.pkl", "rb") as f:
    svd_usvh_v_for_vlinear = pickle.load(f)
with open("/u/qxc4fh/zeyu_workspace/linear_for_svd.pkl", "rb") as f:
    linear_for_svd = pickle.load(f)
with open("/u/qxc4fh/zeyu_workspace/svd_usvh_v_for_token.pkl", "rb") as f:
    svd_usvh_v_for_token = pickle.load(f)

vlinear_drop = 0
qk_drop = 0
token_stat_drop = 0

compute_matrix_mse_vlinear = False
compute_matrix_mse_qk = False
compute_matrix_mse_token = False
matrix_mse_vlinear = []
matrix_mse_qk = []
matrix_mse_token = []


class VocabEmbedding(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weight = Parameter(
            torch.empty(
                VOCAB_SIZE,
                HIDDEN_SIZE,
                dtype=torch.float32,
            )
        )

    def forward(self, input_):
        masked_input = input_
        output = F.embedding(masked_input, self.weight, None, None, 2.0, False, False)
        return output


class Embedding(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.word_embeddings = VocabEmbedding()
        self.position_embeddings = torch.nn.Embedding(CONTEXT_LEN, HIDDEN_SIZE)
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


# token_for_svd = [[None for _ in range(17)] for _ in range(24)]


class QKVLinear(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weight = Parameter(
            torch.empty(
                3 * HIDDEN_SIZE,
                HIDDEN_SIZE,
                dtype=torch.float32,
            )
        )
        self.bias = Parameter(torch.empty(3 * HIDDEN_SIZE, dtype=torch.float32))

    def forward(self, input_, layernum):
        # token_for_svd[layernum][0] = input_[:, 0, :].clone().detach().cpu()
        # for head in range(16):
        #     tsr = self.weight.t()[:, head * 128 * 3 : (head + 1) * 128 * 3].clone().detach().cpu()
        #     token_for_svd[layernum][head + 1] = tsr

        svd = svd_usvh_v_for_token[layernum].to(input_.device)
        input_tmp = torch.matmul(input_[:, 0, :], svd).view(input_.size(0), 1, input_.size(2))
        if compute_matrix_mse_token:
            input_tmp_origin = input_tmp.clone().detach()
        input_tmp[:, 0, int((1 - token_stat_drop) * 2048) : 2048] = 0
        weight_tmp = torch.matmul(self.weight, svd)
        if compute_matrix_mse_token:
            weight_tmp_origin = weight_tmp.clone().detach()
        weight_tmp[:, int((1 - token_stat_drop) * 2048) : 2048] = 0
        # output = torch.matmul(input_, self.weight.t())
        if compute_matrix_mse_token:
            output_origin = torch.matmul(input_tmp_origin, weight_tmp_origin.t())
        output = torch.matmul(input_tmp, weight_tmp.t())
        if compute_matrix_mse_token:
            diff = output_origin - output
            mse = torch.pow(diff, 2).sum() / diff.numel()
            matrix_mse_token.append(mse.item())

        return output + self.bias, None


def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


# tkimp_unimp_tkids = []  # [ids_in_a_layer, ids_in_a_layer, ...]

# attn_dist = 1150
# dist_mask = torch.ones([2048, 2048], dtype=torch.int64, device=torch.device("cuda:0")) < 0
# for qid in range(2048):
#     end = qid - attn_dist
#     for kid in range(0, end):
#         dist_mask[qid, kid] = True
# dist_mask = einops.repeat(dist_mask, "c d -> a b c d", a=1, b=16)

# import pickle

# with open("svd_usvh_v.pkl", "rb") as f:
#     svd_v = pickle.load(f)

# v_for_svd = [[[] for _ in range(16)] for _ in range(24)]


class CoreAttention(torch.nn.Module):
    def __init__(self, layer_num, precision, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_num = max(1, layer_num)
        self.precision = precision
        self.norm_factor = math.sqrt(HIDDEN_SIZE / NUM_HEADS)
        coeff = self.layer_num
        self.norm_factor *= coeff
        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.precision == 16,
            False,
            AttnMaskType.causal,
            True,
            attention_mask_func,
            True,
            coeff,
        )
        self.attention_dropout = torch.nn.Dropout(0.1)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # for head_id in range(16):
        #     q_tmp = query_layer[:, head_id, :].contiguous().cpu()
        #     k_tmp = key_layer[:, head_id, :].contiguous().cpu()
        #     qk_list[self.layer_num - 1][head_id].extend([q_tmp, k_tmp])

        # print(value_layer.shape)
        # my_v = value_layer[:, 0, :, :]
        # for head_id in range(16):
        #     v_tmp = my_v[:, head_id, :].contiguous().cpu()
        #     v_for_svd[self.layer_num - 1][head_id].append(v_tmp)

        h_drop = qk_drop
        h_mask = torch.ones(2048, 128, dtype=torch.long, device=query_layer.device)
        view = h_mask[:, int((1 - h_drop) * 128) : 128]
        view.fill_(0)
        h_mask = h_mask < 0.5
        # # print(h_mask)
        # # exit()
        qs = []
        ks = []
        qs_origin = []
        ks_origin = []
        for head in range(16):
            Q = query_layer[:, head, :]
            K = key_layer[:, head, :]
            v = svd_usvh_v_for_qk[self.layer_num - 1][head].to(Q.device)
            Q = torch.matmul(Q, v)
            if compute_matrix_mse_qk:
                Q_origin = Q.clone().detach()
                qs_origin.append(Q_origin)
            Q.masked_fill_(h_mask, 0)
            qs.append(Q)
            K = torch.matmul(K, v)
            if compute_matrix_mse_qk:
                K_origin = K.clone().detach()
                ks_origin.append(K_origin)
            K.masked_fill_(h_mask, 0)
            ks.append(K)
        query_layer = torch.concat(qs, dim=1).view(2048, 16, 128)
        key_layer = torch.concat(ks, dim=1).view(2048, 16, 128)
        if compute_matrix_mse_qk:
            query_layer_origin = torch.concat(qs_origin, dim=1).view(2048, 16, 128)
            key_layer_origin = torch.concat(ks_origin, dim=1).view(2048, 16, 128)
        # # print(key_layer.shape)
        # # exit()

        # preallocting input tensor: [b * np, sq, sk]
        # TODO: improve the performance
        # matmul_input_buffer = torch.empty(
        #     output_size[0] * output_size[1], output_size[2], output_size[3], dtype=query_layer.dtype
        # )
        matmul_input_buffer = torch.empty(1, device=query_layer.device)

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )
        if compute_matrix_mse_qk:
            matmul_result_origin = torch.baddbmm(
                matmul_input_buffer,
                query_layer_origin.transpose(0, 1),  # [b * np, sq, hn]
                key_layer_origin.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0,
                alpha=(1.0 / self.norm_factor),
            )
            diff = matmul_result_origin - matmul_result
            mse = torch.pow(diff, 2).sum() / diff.numel()
            matrix_mse_qk.append(mse.item())

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # unimp_percent = 0.1
        # ids = [[] for _ in range(16)]
        # for head in range(16):
        #     tsr = attention_scores[0, head, :, :]
        #     avg_values = torch.mean(tsr, 1)
        #     avg_values = [(x.item(), idx) for idx, x in enumerate(avg_values)]
        #     avg_values.sort(key=lambda x: x[0])
        #     ids_per_head = [i[1] for i in avg_values[: int(unimp_percent * len(avg_values))]]
        #     ids[head] = ids_per_head
        # global tkimp_unimp_tkids
        # unimp_percent = 0.3
        # tsr = attention_scores[0, :, :, :]
        # avg_values = torch.mean(tsr, [0, 2])
        # avg_values = [(x.item(), idx) for idx, x in enumerate(avg_values)]
        # avg_values.sort(key=lambda x: x[0])
        # ids = [i[1] for i in avg_values[: int(unimp_percent * len(avg_values))]]
        # tkimp_unimp_tkids.append(ids)
        # fill_tsr = torch.tensor(
        #     [-10000.0 for _ in range(16)], dtype=attention_scores.dtype, device=attention_scores.device
        # ).reshape((16, 1, 1))
        # if len(tkimp_unimp_tkids) >= 1:
        #     for k in ids:
        #         for q in range(2048):
        #             if k != q:
        #                 for head in range(16):
        #                     attention_scores[0, head, q, k] = -10000.0
        #     # print(attention_scores)
        #     # exit()

        attention_scores.masked_fill_(attention_mask, -10000.0)

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        # print(attention_probs.shape)
        # print(value_layer.shape)
        vs = []
        vs_origin = []
        for head in range(16):
            partv = value_layer[:, head, :]
            svd = svd_usvh_v_for_vlinear[self.layer_num - 1][head]
            new_v = torch.matmul(partv, svd.to(partv.device))
            if compute_matrix_mse_vlinear:
                new_v_origin = new_v.clone().detach()
                vs_origin.append(new_v_origin)
            new_v[:, int((1 - vlinear_drop) * 128) : 128] = 0
            vs.append(new_v)
        value_layer = torch.concat(vs, dim=1).view(2048, 16, 128)
        if compute_matrix_mse_vlinear:
            value_layer_origin = torch.concat(vs_origin, dim=1).view(2048, 16, 128)
            context_layer_origin = torch.bmm(attention_probs, value_layer_origin.transpose(0, 1))
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        if compute_matrix_mse_vlinear:
            context_layer_origin = context_layer_origin.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
        if compute_matrix_mse_vlinear:
            context_layer_origin = context_layer_origin.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (HIDDEN_SIZE,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if compute_matrix_mse_vlinear:
            context_layer_origin = context_layer_origin.view(*new_context_layer_shape)
            return context_layer, context_layer_origin

        return context_layer


class AttnOutputLinear(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weight = Parameter(
            torch.empty(
                HIDDEN_SIZE,
                HIDDEN_SIZE,
                dtype=torch.float32,
            )
        )
        self.bias = Parameter(torch.empty(HIDDEN_SIZE, dtype=torch.float32))

    def forward(self, input_, layer_num, input_origin):
        ps = []
        ps_origin = []
        for head in range(16):
            param = linear_for_svd[layer_num][head][0].to(input_.device).t()
            svd = svd_usvh_v_for_vlinear[layer_num][head].to(input_.device)
            new_param = torch.matmul(param, svd).t()
            if compute_matrix_mse_vlinear:
                new_param_origin = new_param.clone().detach()
                ps_origin.append(new_param_origin)
            new_param[int((1 - vlinear_drop) * 128) : 128, :] = 0
            ps.append(new_param)
        tsr = torch.concat(ps, dim=0)
        if compute_matrix_mse_vlinear:
            tsr_origin = torch.concat(ps_origin, dim=0)
            output_origin = torch.matmul(input_origin, tsr_origin)
        output = torch.matmul(input_, tsr)
        if compute_matrix_mse_vlinear:
            diff = output - output_origin
            mse = torch.pow(diff, 2).sum() / diff.numel()
            matrix_mse_vlinear.append(mse.item())
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
    def __init__(self, layer_num, precision, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_num = layer_num
        self.precision = precision
        self.kv_channels = HIDDEN_SIZE // NUM_HEADS
        self.query_key_value = QKVLinear()
        self.core_attention = CoreAttention(self.layer_num, self.precision)
        self.dense = AttnOutputLinear()

        # Inference key-value memory
        self.inference_key_memory = None
        self.inference_value_memory = None
        self.inference_current_sequence_len = 0

    def forward(self, hidden_states, attention_mask, set_inference_key_value_memory=False, inference_max_sequence_len=None):
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        if set_inference_key_value_memory:
            assert inference_max_sequence_len and inference_max_sequence_len > 0
            self.inference_key_memory = self._allocate_memory(
                inference_max_sequence_len, hidden_states.size(1), hidden_states.dtype, hidden_states.device
            )
            self.inference_value_memory = self._allocate_memory(
                inference_max_sequence_len, hidden_states.size(1), hidden_states.dtype, hidden_states.device
            )
            self.inference_current_sequence_len = 0

        # Some consistency check.
        if inference_max_sequence_len:
            assert self.inference_current_sequence_len < self.inference_key_memory.size(0)
            assert inference_max_sequence_len == self.inference_key_memory.size(0)
        # This is added for safety. In case inference_max_sequence_len
        # is not provided, make sure there is no potential memory left
        # from previous inference.
        if not inference_max_sequence_len:
            self.inference_key_memory = None
            self.inference_value_memory = None

        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states, self.layer_num - 1)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            NUM_HEADS,
            3 * HIDDEN_SIZE // NUM_HEADS,
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

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
            # Adjust attention mask
            attention_mask = attention_mask[..., start:end, :end]

        if compute_matrix_mse_vlinear:
            context_layer, context_layer_origin = self.core_attention(query_layer, key_layer, value_layer, attention_mask)
        else:
            context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)

        if compute_matrix_mse_vlinear:
            output, bias = self.dense(context_layer, self.layer_num - 1, context_layer_origin)
        else:
            output, bias = self.dense(context_layer, self.layer_num - 1, None)

        return output, bias

    def _allocate_memory(self, inference_max_sequence_len, batch_size, dtype, device):
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            NUM_HEADS,
            HIDDEN_SIZE // NUM_HEADS,
            dtype=dtype,
            device=device,
        )


class MLPLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, skip_bias_add=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.skip_bias_add = skip_bias_add
        self.weight = Parameter(
            torch.empty(
                output_size,
                input_size,
                dtype=torch.float32,
            )
        )
        self.bias = Parameter(torch.empty(output_size, dtype=torch.float32))

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
        self.dense_h_to_4h = MLPLinear(HIDDEN_SIZE, FFN_HIDDEN_SIZE, skip_bias_add=True)
        self.bias_gelu_fusion = True
        self.activation_func = F.gelu
        self.dense_4h_to_h = MLPLinear(FFN_HIDDEN_SIZE, HIDDEN_SIZE)

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate, bias = self.dense_h_to_4h(hidden_states)

        if self.bias_gelu_fusion:
            intermediate = bias_gelu_impl(intermediate, bias)
        else:
            if bias is not None:
                intermediate = intermediate + bias
            intermediate = self.activation_func(intermediate)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate)
        return output, output_bias


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Optional[Tensor], Tensor, float, bool) -> Tensor
    if bias is not None:
        x = x + bias
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    out = residual + out
    return out


@torch.jit.script
def bias_dropout_add_fused_train(
    x: torch.Tensor, bias: Optional[torch.Tensor], residual: torch.Tensor, prob: float
) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(
    x: torch.Tensor, bias: Optional[torch.Tensor], residual: torch.Tensor, prob: float
) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, False)


class TransformerLayer(torch.nn.Module):
    def __init__(self, layer_num, precision, drop_path_rate, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_num = layer_num
        self.precision = precision
        self.drop_path_rate = drop_path_rate
        # self.input_layernorm = LayerNorm(
        #     HIDDEN_SIZE,
        #     eps=1e-5,
        #     no_persist_layer_norm=False,
        #     sequence_parallel=False,
        #     apply_layernorm_1p=False,
        # )
        self.input_layernorm = MixedFusedLayerNorm(HIDDEN_SIZE, 1e-5, sequence_parallel_enbaled=False)
        # Self attention.
        self.self_attention = Attention(layer_num, self.precision)

        self.hidden_dropout = 0.1
        self.bias_dropout_fusion = True
        self.drop_path = None

        # self.post_attention_layernorm = LayerNorm(
        #     HIDDEN_SIZE,
        #     eps=1e-5,
        #     no_persist_layer_norm=False,
        #     sequence_parallel=False,
        #     apply_layernorm_1p=False,
        # )
        self.post_attention_layernorm = MixedFusedLayerNorm(HIDDEN_SIZE, 1e-5, sequence_parallel_enbaled=False)

        # MLP
        self.mlp = MLP()

        # Set bias+dropout+add fusion grad_enable execution handler.
        TORCH_MAJOR = int(torch.__version__.split(".")[0])
        TORCH_MINOR = int(torch.__version__.split(".")[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad

    def forward(self, hidden_states, attention_mask, set_inference_key_value_memory=False, inference_max_sequence_len=None):
        layernorm_output = self.input_layernorm(hidden_states)
        attention_output, attention_bias = self.self_attention(
            layernorm_output, attention_mask, set_inference_key_value_memory, inference_max_sequence_len
        )
        residual = hidden_states
        if IS_TRAINING:
            bias_dropout_add_func = bias_dropout_add_fused_train
        else:
            bias_dropout_add_func = bias_dropout_add_fused_inference
        if attention_bias is not None:
            attention_bias = attention_bias.expand_as(residual)
        with self.bias_dropout_add_exec_handler():
            layernorm_input = bias_dropout_add_func(attention_output, attention_bias, residual, self.hidden_dropout)
        layernorm_output = self.post_attention_layernorm(layernorm_input)

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
    def __init__(self, precision, drop_path_rate=0.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.precision = precision
        self.drop_path_rate = drop_path_rate
        self.drop_path_rates = [rate.item() for rate in torch.linspace(0, self.drop_path_rate, NUM_LAYERS)]

        def build_layer(layer_num):
            return TransformerLayer(layer_num, self.precision, drop_path_rate=self.drop_path_rates[layer_num - 1])

        self.layers = []
        for i in range(NUM_LAYERS):
            layer_num = i + 1
            self.layers.append(build_layer(layer_num))
        self.layers = torch.nn.ModuleList(self.layers)

        # self.final_layernorm = LayerNorm(
        #     HIDDEN_SIZE,
        #     eps=1e-5,
        #     no_persist_layer_norm=False,
        #     sequence_parallel=False,
        #     apply_layernorm_1p=False,
        # )
        self.final_layernorm = MixedFusedLayerNorm(HIDDEN_SIZE, 1e-5, sequence_parallel_enbaled=False)

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(self, hidden_states, attention_mask, set_inference_key_value_memory=False, inference_max_sequence_len=None):
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
            hidden_states,
            requires_grad=True,
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

        if self.precision == 16:
            for ly in range(NUM_LAYERS):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    hidden_states = custom(ly, ly + 1)(
                        hidden_states, attention_mask, set_inference_key_value_memory, inference_max_sequence_len
                    )
        else:
            for ly in range(NUM_LAYERS):
                hidden_states = custom(ly, ly + 1)(
                    hidden_states, attention_mask, set_inference_key_value_memory, inference_max_sequence_len
                )

        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


class GPTModel(torch.nn.Module):
    def __init__(self, param_pickle_path, precision=32, device="cuda", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.precision = precision
        self.device = torch.device(device)
        if self.precision == 16:
            assert self.device.type == "cuda"

        torch.cuda.set_device(self.device)

        self.embedding = Embedding()
        self.encoder = Transformer(self.precision)

        self.encoder_seq_length = CONTEXT_LEN

        self._init_model(param_pickle_path)

    def forward(self, batch, inf_target_tensor):
        global INF_TARGET_TENSOR
        INF_TARGET_TENSOR = inf_target_tensor

        global tkimp_unimp_tkids
        tkimp_unimp_tkids = []

        (enc_input_ids, enc_attn_mask, enc_position_ids, set_inference_key_value_memory, inference_max_sequence_len) = batch
        enc_input_ids = enc_input_ids.to(self.device)
        enc_attn_mask = enc_attn_mask.to(self.device)
        enc_position_ids = enc_position_ids.to(self.device)
        encoder_input = self.embedding(enc_input_ids, enc_position_ids)
        encoder_output = self.encoder(
            encoder_input, enc_attn_mask, set_inference_key_value_memory[0].item(), inference_max_sequence_len[0].item()
        )
        word_embeddings_weight = self.embedding.word_embeddings.weight
        logits = torch.matmul(encoder_output, word_embeddings_weight.t())

        def cal_overlap(a, b):
            a = set(a)
            b = set(b)
            c = a.intersection(b)
            return len(c) / len(a)

        # for a in range(23):
        #     for b in range(a + 1, 23):
        #         print(cal_overlap(tkimp_unimp_tkids[a], tkimp_unimp_tkids[b]))
        #     print("=========================")
        # exit()

        # global qk_list
        # with open("qk_for_svd.pkl", "wb") as f:
        #     import pickle

        #     pickle.dump(qk_list, f)

        # global v_for_svd
        # with open("v_for_svd.pkl", "wb") as f:
        #     import pickle

        #     pickle.dump(v_for_svd, f)

        # global token_for_svd
        # with open("token_for_svd.pkl", "wb") as f:
        #     pickle.dump(token_for_svd, f)

        if compute_matrix_mse_vlinear:
            print(np.average(matrix_mse_vlinear))
        if compute_matrix_mse_qk:
            print(np.average(matrix_mse_qk))
        if compute_matrix_mse_token:
            print(np.average(matrix_mse_token))

        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits.transpose(0, 1)[0, :, :].contiguous(), INF_TARGET_TENSOR).item()
        perplexity = math.exp(loss)
        print(perplexity)
        exit()

        return logits.transpose(0, 1).contiguous()

    def _init_model(self, param_pickle_path):
        # linear_for_svd = [[[] for _ in range(16)] for _ in range(24)]
        with open(param_pickle_path, "rb") as file:
            params = pickle.load(file)
        for (key, _), (_, param) in zip(self.named_parameters(), params):
            # if "self_attention.dense.weight" in key:
            #     strs = key.split(".")
            #     layer = int(strs[2])
            #     param_t = param.transpose(0, 1)
            #     for head in range(16):
            #         part = param_t[head * 128 : (head + 1) * 128, :].contiguous().cpu()
            #         linear_for_svd[layer][head].append(part)
            self.state_dict()[key].copy_(param)

        # with open("linear_for_svd.pkl", "wb") as f:
        #     pickle.dump(linear_for_svd, f)

        # exit()
        self.to(self.device)


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
