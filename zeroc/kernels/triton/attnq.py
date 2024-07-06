import torch
import triton
import triton.language as tl

qk_configs = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BM in [64, 128]
    for BN in [32, 64]
    for s in ([3, 4, 7])
    for w in [4, 8]
]


def qk_conf_filter(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(list(filter(qk_conf_filter, qk_configs)), key=["M", "N", "HEAD_DIM"])
@triton.jit
def attn_qk(
    Q,
    K,
    S,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_sz,
    stride_sh,
    stride_sm,
    stride_sn,  #
    Z,
    H,
    M,
    N,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    s_offset = off_z.to(tl.int64) * stride_sz + off_h.to(tl.int64) * stride_sh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(M, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, N),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    S_block_ptr = tl.make_block_ptr(
        base=S + s_offset,
        shape=(M, N),
        strides=(stride_sm, stride_sn),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)

    # loop over k and update result
    for start_n in range(0, N, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr)
        s = tl.dot(q, k)
        s = s.to(tl.float16)
        tl.store(S_block_ptr, s)
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        S_block_ptr = tl.advance(S_block_ptr, (0, BLOCK_N))


def attention(q, k):
    Z = q.shape[0]
    H = q.shape[1]
    M = q.shape[2]
    N = k.shape[2]
    HEAD_DIM = q.shape[3]
    attn_score = torch.empty([Z, H, M, N], dtype=torch.float16, device=q.device)
    qk_grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
    attn_qk[qk_grid](
        q,
        k,
        attn_score,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),  #
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        attn_score.stride(0),
        attn_score.stride(1),
        attn_score.stride(2),
        attn_score.stride(3),
        Z,
        H,
        M,
        N,
        HEAD_DIM=HEAD_DIM,
    )
    return attn_score


q = torch.ones([1, 1, 64000, 128], dtype=torch.float16, device="cuda")
k = torch.ones([1, 1, 64000, 128], dtype=torch.float16, device="cuda")

print(attention(q, k))
