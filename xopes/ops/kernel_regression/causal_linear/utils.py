import triton
import triton.language as tl

from xopes.utils import generate_configs


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_D": [64, 128],
        }
    ),
    key=[
        "B",
        "N",
        "H",
        "D",
    ],
)
@triton.jit
def _krcl_parallel_inverse(
    Q,  # B N H D
    K,  # B N H D
    INV,  # B H NUM_BLOCK_N BLOCK_N BLOCK_N
    LOG_DECAY,  # B N H
    ALPHA,  # B N H
    BETA,  # B N H
    CU_SEQLENS,
    USE_Q: tl.constexpr,  # bool
    USE_ALPHA: tl.constexpr,  # bool
    USE_BETA: tl.constexpr,  # bool
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    REVERSE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_BLOCK_N: tl.constexpr,
):
    NUM_BLOCK_D = tl.cdiv(D, BLOCK_D)
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    off_block_n = tl.program_id(2)

    offset_block_n = off_block_n * BLOCK_N
    offset_qk = off_b * N * H * D + off_h * D
    offset_ld = off_b * N * H + off_h
    offset_inv = (
        off_b * H * NUM_BLOCK_N * BLOCK_N * BLOCK_N
        + off_h * NUM_BLOCK_N * BLOCK_N * BLOCK_N
        + off_block_n * BLOCK_N * BLOCK_N
    )
    offset_block_n * H * D
    offset_block_ld = offset_block_n * H

    a = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
    array = tl.arange(0, BLOCK_N)
    if USE_Q:
        q_block_ptr = tl.make_block_ptr(
            base=Q + offset_qk,
            shape=(N, D),
            strides=(H * D, 1),
            offsets=(offset_block_n, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )

    k_block_ptr = tl.make_block_ptr(
        base=K + offset_qk,
        shape=(N, D),
        strides=(H * D, 1),
        offsets=(offset_block_n, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    inv_block_ptr = tl.make_block_ptr(
        base=INV + offset_inv,
        shape=(BLOCK_N, BLOCK_N),
        strides=(BLOCK_N, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_N),
        order=(1, 0),
    )

    if USE_ALPHA:
        alpha_block_ptr = tl.make_block_ptr(
            base=ALPHA + offset_ld,
            shape=(N, 1),
            strides=(H, 1),
            offsets=(offset_block_n, 0),
            block_shape=(BLOCK_N, 1),
            order=(1, 0),
        )
        alpha = tl.load(alpha_block_ptr, boundary_check=(0, 1)).to(tl.float32)

    if USE_BETA:
        beta_trans_block_ptr = tl.make_block_ptr(
            base=BETA + offset_ld,
            shape=(N, 1),
            strides=(H, 1),
            offsets=(offset_block_n, 0),
            block_shape=(BLOCK_N, 1),
            order=(1, 0),
        )
        beta = tl.load(beta_trans_block_ptr, boundary_check=(0, 1)).to(tl.float32)

    for i in range(NUM_BLOCK_D):
        k = tl.load(k_block_ptr, boundary_check=(0, 1))
        if USE_Q:
            q = tl.load(q_block_ptr, boundary_check=(0, 1))
        else:
            q = k
        if USE_ALPHA:
            q = q * alpha
        if USE_BETA:
            k = k * beta

        k_trans = tl.trans(k)

        a += tl.dot(q, k_trans)

        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_D))
        if USE_Q:
            q_block_ptr = tl.advance(q_block_ptr, (0, BLOCK_D))

    # add decay
    ld_sum_block_ptr = LOG_DECAY + offset_ld + offset_block_ld + array * H
    mask = (offset_block_n + array) < N
    ld = tl.load(ld_sum_block_ptr, mask=mask, other=0.0).to(tl.float32)
    diff = ld[:, None] - ld[None, :]
    if REVERSE:  # triu
        diff = tl.where(array[:, None] < array[None, :], diff, -float("inf"))
    else:  # tril
        diff = tl.where(array[:, None] > array[None, :], diff, -float("inf"))
    a *= tl.exp(diff)

    a_inv = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
    eye = array[:, None] == array[None, :]
    for i in range(BLOCK_N):
        # C = (I + A) ^ -1, Ck = -A * Ck + I
        a_inv = -tl.dot(a, a_inv) + eye

    tl.store(inv_block_ptr, a_inv.to(inv_block_ptr.dtype.element_ty))
