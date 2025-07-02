import triton
import triton.language as tl

from xopes.utils import generate_configs
from xopes.utils.constant import XOPES_DEBUG

BLOCK_D_INV_LIST = [128]

if XOPES_DEBUG:
    BLOCK_D_LIST = [64]
    BLOCK_E_LIST = [32]
else:
    BLOCK_D_LIST = [64, 128]
    BLOCK_E_LIST = [64, 128]


@triton.heuristics(
    {
        "MAX_BLOCK_N": lambda args: triton.next_power_of_2(args["N"]),
    }
)
@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_D": BLOCK_D_INV_LIST,
            "BLOCK_NUM": [1, 2, 4],
        }
    ),
    key=[
        "B",
        "MAX_BLOCK_N",
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
    BLOCK_N1: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_BLOCK_N: tl.constexpr,
    MAX_BLOCK_N: tl.constexpr,
    BLOCK_NUM: tl.constexpr,
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
    offset_block_ld = offset_block_n * H

    if BLOCK_NUM == 1:
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
            alpha = tl.load(
                alpha_block_ptr, boundary_check=(0, 1), padding_option="zero"
            )

        if USE_BETA:
            beta_trans_block_ptr = tl.make_block_ptr(
                base=BETA + offset_ld,
                shape=(N, 1),
                strides=(H, 1),
                offsets=(offset_block_n, 0),
                block_shape=(BLOCK_N, 1),
                order=(1, 0),
            )
            beta = tl.load(
                beta_trans_block_ptr, boundary_check=(0, 1), padding_option="zero"
            )

        for i in range(NUM_BLOCK_D):
            k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
            if USE_Q:
                q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
            else:
                q = k
            if USE_ALPHA:
                q = (q * alpha).to(q.dtype)
            if USE_BETA:
                k = (k * beta).to(k.dtype)

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
        eye = (array[:, None] == array[None, :]).to(a.dtype)
        for i in tl.static_range(BLOCK_N):
            # C = (I + A) ^ -1, Ck = -A * Ck + I
            a_inv = -tl.dot(a_inv, a) + eye

        tl.store(
            inv_block_ptr,
            a_inv.to(inv_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )
    elif BLOCK_NUM == 2:
        a11 = tl.zeros((BLOCK_N1, BLOCK_N1), dtype=tl.float32)
        a21 = tl.zeros((BLOCK_N1, BLOCK_N1), dtype=tl.float32)
        a22 = tl.zeros((BLOCK_N1, BLOCK_N1), dtype=tl.float32)
        if USE_Q:
            q1_block_ptr = tl.make_block_ptr(
                base=Q + offset_qk,
                shape=(N, D),
                strides=(H * D, 1),
                offsets=(offset_block_n + BLOCK_N1, 0)
                if REVERSE
                else (offset_block_n, 0),
                block_shape=(BLOCK_N1, BLOCK_D),
                order=(1, 0),
            )

            q2_block_ptr = tl.make_block_ptr(
                base=Q + offset_qk,
                shape=(N, D),
                strides=(H * D, 1),
                offsets=(offset_block_n, 0)
                if REVERSE
                else (offset_block_n + BLOCK_N1, 0),
                block_shape=(BLOCK_N1, BLOCK_D),
                order=(1, 0),
            )

        k1_block_ptr = tl.make_block_ptr(
            base=K + offset_qk,
            shape=(N, D),
            strides=(H * D, 1),
            offsets=(offset_block_n + BLOCK_N1, 0) if REVERSE else (offset_block_n, 0),
            block_shape=(BLOCK_N1, BLOCK_D),
            order=(1, 0),
        )

        k2_block_ptr = tl.make_block_ptr(
            base=K + offset_qk,
            shape=(N, D),
            strides=(H * D, 1),
            offsets=(offset_block_n, 0) if REVERSE else (offset_block_n + BLOCK_N1, 0),
            block_shape=(BLOCK_N1, BLOCK_D),
            order=(1, 0),
        )

        inv11_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(BLOCK_N1, BLOCK_N1) if REVERSE else (0, 0),
            block_shape=(BLOCK_N1, BLOCK_N1),
            order=(1, 0),
        )

        inv12_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(BLOCK_N1, 0) if REVERSE else (0, BLOCK_N1),
            block_shape=(BLOCK_N1, BLOCK_N1),
            order=(1, 0),
        )

        inv21_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(0, BLOCK_N1) if REVERSE else (BLOCK_N1, 0),
            block_shape=(BLOCK_N1, BLOCK_N1),
            order=(1, 0),
        )

        inv22_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(0, 0) if REVERSE else (BLOCK_N1, BLOCK_N1),
            block_shape=(BLOCK_N1, BLOCK_N1),
            order=(1, 0),
        )

        if USE_ALPHA:
            alpha1_block_ptr = tl.make_block_ptr(
                base=ALPHA + offset_ld,
                shape=(N, 1),
                strides=(H, 1),
                offsets=(offset_block_n + BLOCK_N1, 0)
                if REVERSE
                else (offset_block_n, 0),
                block_shape=(BLOCK_N1, 1),
                order=(1, 0),
            )

            alpha2_block_ptr = tl.make_block_ptr(
                base=ALPHA + offset_ld,
                shape=(N, 1),
                strides=(H, 1),
                offsets=(offset_block_n, 0)
                if REVERSE
                else (offset_block_n + BLOCK_N1, 0),
                block_shape=(BLOCK_N1, 1),
                order=(1, 0),
            )

            alpha1 = tl.load(
                alpha1_block_ptr, boundary_check=(0, 1), padding_option="zero"
            )

            alpha2 = tl.load(
                alpha2_block_ptr, boundary_check=(0, 1), padding_option="zero"
            )

        if USE_BETA:
            beta1_block_ptr = tl.make_block_ptr(
                base=BETA + offset_ld,
                shape=(N, 1),
                strides=(H, 1),
                offsets=(offset_block_n + BLOCK_N1, 0)
                if REVERSE
                else (offset_block_n, 0),
                block_shape=(BLOCK_N1, 1),
                order=(1, 0),
            )

            beta2_block_ptr = tl.make_block_ptr(
                base=BETA + offset_ld,
                shape=(N, 1),
                strides=(H, 1),
                offsets=(offset_block_n, 0)
                if REVERSE
                else (offset_block_n + BLOCK_N1, 0),
                block_shape=(BLOCK_N1, 1),
                order=(1, 0),
            )

            beta1 = tl.load(
                beta1_block_ptr, boundary_check=(0, 1), padding_option="zero"
            )

            beta2 = tl.load(
                beta2_block_ptr, boundary_check=(0, 1), padding_option="zero"
            )

        for i in range(NUM_BLOCK_D):
            k1 = tl.load(k1_block_ptr, boundary_check=(0, 1), padding_option="zero")
            k2 = tl.load(k2_block_ptr, boundary_check=(0, 1), padding_option="zero")
            if USE_Q:
                q1 = tl.load(q1_block_ptr, boundary_check=(0, 1), padding_option="zero")
                q2 = tl.load(q2_block_ptr, boundary_check=(0, 1), padding_option="zero")
            else:
                q1 = k1
                q2 = k2

            if USE_ALPHA:
                q1 = (q1 * alpha1).to(q1.dtype)
                q2 = (q2 * alpha2).to(q2.dtype)

            if USE_BETA:
                k1 = (k1 * beta1).to(k1.dtype)
                k2 = (k2 * beta2).to(k2.dtype)

            k1_trans = tl.trans(k1)
            k2_trans = tl.trans(k2)

            a11 += tl.dot(q1, k1_trans)
            a21 += tl.dot(q2, k1_trans)
            a22 += tl.dot(q2, k2_trans)

            k1_block_ptr = tl.advance(k1_block_ptr, (0, BLOCK_D))
            k2_block_ptr = tl.advance(k2_block_ptr, (0, BLOCK_D))
            if USE_Q:
                q1_block_ptr = tl.advance(q1_block_ptr, (0, BLOCK_D))
                q2_block_ptr = tl.advance(q2_block_ptr, (0, BLOCK_D))

        # add decay
        if REVERSE:
            array1 = tl.arange(0, BLOCK_N1) + BLOCK_N1
            array2 = tl.arange(0, BLOCK_N1)
        else:
            array1 = tl.arange(0, BLOCK_N1)
            array2 = tl.arange(0, BLOCK_N1) + BLOCK_N1
        ld1_sum_block_ptr = LOG_DECAY + offset_ld + offset_block_ld + array1 * H
        ld2_sum_block_ptr = LOG_DECAY + offset_ld + offset_block_ld + array2 * H
        mask1 = (offset_block_n + array1) < N
        mask2 = (offset_block_n + array2) < N
        ld1 = tl.load(ld1_sum_block_ptr, mask=mask1, other=0.0).to(tl.float32)
        ld2 = tl.load(ld2_sum_block_ptr, mask=mask2, other=0.0).to(tl.float32)
        diff11 = ld1[:, None] - ld1[None, :]
        diff21 = ld2[:, None] - ld1[None, :]
        diff22 = ld2[:, None] - ld2[None, :]
        if REVERSE:  # triu
            diff11 = tl.where(array1[:, None] < array1[None, :], diff11, -float("inf"))
            diff21 = tl.where(array2[:, None] < array1[None, :], diff21, -float("inf"))
            diff22 = tl.where(array2[:, None] < array2[None, :], diff22, -float("inf"))
        else:  # tril
            diff11 = tl.where(array1[:, None] > array1[None, :], diff11, -float("inf"))
            diff21 = tl.where(array2[:, None] > array1[None, :], diff21, -float("inf"))
            diff22 = tl.where(array2[:, None] > array2[None, :], diff22, -float("inf"))
        a11 *= tl.exp(diff11)
        a21 *= tl.exp(diff21)
        a22 *= tl.exp(diff22)

        zero = tl.zeros((BLOCK_N1, BLOCK_N1), dtype=tl.float32)
        a11_inv = zero
        a12_inv = zero
        a22_inv = zero
        eye = (array1[:, None] == array1[None, :]).to(a11.dtype)
        for i in tl.static_range(BLOCK_N1):
            # C = (I + A) ^ -1, Ck = -A * Ck + I
            a11_inv = -tl.dot(a11_inv, a11) + eye
            a22_inv = -tl.dot(a22_inv, a22) + eye

        a21_inv = -tl.dot(a22_inv, tl.dot(a21, a11_inv))

        tl.store(
            inv11_block_ptr,
            a11_inv.to(inv11_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            inv12_block_ptr,
            a12_inv.to(inv12_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            inv21_block_ptr,
            a21_inv.to(inv21_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            inv22_block_ptr,
            a22_inv.to(inv22_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )
    elif BLOCK_NUM == 4:
        a11 = tl.zeros((BLOCK_N2, BLOCK_N2), dtype=tl.float32)
        a21 = tl.zeros((BLOCK_N2, BLOCK_N2), dtype=tl.float32)
        a22 = tl.zeros((BLOCK_N2, BLOCK_N2), dtype=tl.float32)
        a31 = tl.zeros((BLOCK_N2, BLOCK_N2), dtype=tl.float32)
        a32 = tl.zeros((BLOCK_N2, BLOCK_N2), dtype=tl.float32)
        a33 = tl.zeros((BLOCK_N2, BLOCK_N2), dtype=tl.float32)
        a41 = tl.zeros((BLOCK_N2, BLOCK_N2), dtype=tl.float32)
        a42 = tl.zeros((BLOCK_N2, BLOCK_N2), dtype=tl.float32)
        a43 = tl.zeros((BLOCK_N2, BLOCK_N2), dtype=tl.float32)
        a44 = tl.zeros((BLOCK_N2, BLOCK_N2), dtype=tl.float32)

        if USE_Q:
            q1_block_ptr = tl.make_block_ptr(
                base=Q + offset_qk,
                shape=(N, D),
                strides=(H * D, 1),
                offsets=(offset_block_n + 3 * BLOCK_N2, 0)
                if REVERSE
                else (offset_block_n, 0),
                block_shape=(BLOCK_N2, BLOCK_D),
                order=(1, 0),
            )

            q2_block_ptr = tl.make_block_ptr(
                base=Q + offset_qk,
                shape=(N, D),
                strides=(H * D, 1),
                offsets=(offset_block_n + 2 * BLOCK_N2, 0)
                if REVERSE
                else (offset_block_n + BLOCK_N2, 0),
                block_shape=(BLOCK_N2, BLOCK_D),
                order=(1, 0),
            )

            q3_block_ptr = tl.make_block_ptr(
                base=Q + offset_qk,
                shape=(N, D),
                strides=(H * D, 1),
                offsets=(offset_block_n + BLOCK_N2, 0)
                if REVERSE
                else (offset_block_n + 2 * BLOCK_N2, 0),
                block_shape=(BLOCK_N2, BLOCK_D),
                order=(1, 0),
            )

            q4_block_ptr = tl.make_block_ptr(
                base=Q + offset_qk,
                shape=(N, D),
                strides=(H * D, 1),
                offsets=(offset_block_n, 0)
                if REVERSE
                else (offset_block_n + 3 * BLOCK_N2, 0),
                block_shape=(BLOCK_N2, BLOCK_D),
                order=(1, 0),
            )

        k1_block_ptr = tl.make_block_ptr(
            base=K + offset_qk,
            shape=(N, D),
            strides=(H * D, 1),
            offsets=(offset_block_n + 3 * BLOCK_N2, 0)
            if REVERSE
            else (offset_block_n, 0),
            block_shape=(BLOCK_N2, BLOCK_D),
            order=(1, 0),
        )

        k2_block_ptr = tl.make_block_ptr(
            base=K + offset_qk,
            shape=(N, D),
            strides=(H * D, 1),
            offsets=(offset_block_n + 2 * BLOCK_N2, 0)
            if REVERSE
            else (offset_block_n + BLOCK_N2, 0),
            block_shape=(BLOCK_N2, BLOCK_D),
            order=(1, 0),
        )

        k3_block_ptr = tl.make_block_ptr(
            base=K + offset_qk,
            shape=(N, D),
            strides=(H * D, 1),
            offsets=(offset_block_n + BLOCK_N2, 0)
            if REVERSE
            else (offset_block_n + 2 * BLOCK_N2, 0),
            block_shape=(BLOCK_N2, BLOCK_D),
            order=(1, 0),
        )

        k4_block_ptr = tl.make_block_ptr(
            base=K + offset_qk,
            shape=(N, D),
            strides=(H * D, 1),
            offsets=(offset_block_n, 0)
            if REVERSE
            else (offset_block_n + 3 * BLOCK_N2, 0),
            block_shape=(BLOCK_N2, BLOCK_D),
            order=(1, 0),
        )

        inv11_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(3 * BLOCK_N2, 3 * BLOCK_N2) if REVERSE else (0, 0),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        inv12_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(3 * BLOCK_N2, 2 * BLOCK_N2) if REVERSE else (0, BLOCK_N2),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        inv13_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(3 * BLOCK_N2, 1 * BLOCK_N2) if REVERSE else (0, 2 * BLOCK_N2),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        inv14_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(3 * BLOCK_N2, 0) if REVERSE else (0, 3 * BLOCK_N2),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        inv21_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(2 * BLOCK_N2, 3 * BLOCK_N2) if REVERSE else (BLOCK_N2, 0),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        inv22_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(2 * BLOCK_N2, 2 * BLOCK_N2) if REVERSE else (BLOCK_N2, BLOCK_N2),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        inv23_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(2 * BLOCK_N2, 1 * BLOCK_N2)
            if REVERSE
            else (BLOCK_N2, 2 * BLOCK_N2),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        inv24_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(2 * BLOCK_N2, 0) if REVERSE else (BLOCK_N2, 3 * BLOCK_N2),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        inv31_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(1 * BLOCK_N2, 3 * BLOCK_N2) if REVERSE else (2 * BLOCK_N2, 0),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        inv32_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(1 * BLOCK_N2, 2 * BLOCK_N2)
            if REVERSE
            else (2 * BLOCK_N2, BLOCK_N2),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        inv33_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(1 * BLOCK_N2, 1 * BLOCK_N2)
            if REVERSE
            else (2 * BLOCK_N2, 2 * BLOCK_N2),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        inv34_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(1 * BLOCK_N2, 0) if REVERSE else (2 * BLOCK_N2, 3 * BLOCK_N2),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        inv41_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(0, 3 * BLOCK_N2) if REVERSE else (3 * BLOCK_N2, 0),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        inv42_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(0, 2 * BLOCK_N2) if REVERSE else (3 * BLOCK_N2, BLOCK_N2),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        inv43_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(0, 1 * BLOCK_N2) if REVERSE else (3 * BLOCK_N2, 2 * BLOCK_N2),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        inv44_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(0, 0) if REVERSE else (3 * BLOCK_N2, 3 * BLOCK_N2),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        if USE_ALPHA:
            alpha1_block_ptr = tl.make_block_ptr(
                base=ALPHA + offset_ld,
                shape=(N, 1),
                strides=(H, 1),
                offsets=(offset_block_n + 3 * BLOCK_N2, 0)
                if REVERSE
                else (offset_block_n, 0),
                block_shape=(BLOCK_N2, 1),
                order=(1, 0),
            )

            alpha2_block_ptr = tl.make_block_ptr(
                base=ALPHA + offset_ld,
                shape=(N, 1),
                strides=(H, 1),
                offsets=(offset_block_n + 2 * BLOCK_N2, 0)
                if REVERSE
                else (offset_block_n + BLOCK_N2, 0),
                block_shape=(BLOCK_N2, 1),
                order=(1, 0),
            )

            alpha3_block_ptr = tl.make_block_ptr(
                base=ALPHA + offset_ld,
                shape=(N, 1),
                strides=(H, 1),
                offsets=(offset_block_n + 1 * BLOCK_N2, 0)
                if REVERSE
                else (offset_block_n + 2 * BLOCK_N2, 0),
                block_shape=(BLOCK_N2, 1),
                order=(1, 0),
            )

            alpha4_block_ptr = tl.make_block_ptr(
                base=ALPHA + offset_ld,
                shape=(N, 1),
                strides=(H, 1),
                offsets=(offset_block_n + 0 * BLOCK_N2, 0)
                if REVERSE
                else (offset_block_n + 3 * BLOCK_N2, 0),
                block_shape=(BLOCK_N2, 1),
                order=(1, 0),
            )

            alpha1 = tl.load(
                alpha1_block_ptr, boundary_check=(0, 1), padding_option="zero"
            )

            alpha2 = tl.load(
                alpha2_block_ptr, boundary_check=(0, 1), padding_option="zero"
            )

            alpha3 = tl.load(
                alpha3_block_ptr, boundary_check=(0, 1), padding_option="zero"
            )

            alpha4 = tl.load(
                alpha4_block_ptr, boundary_check=(0, 1), padding_option="zero"
            )

        if USE_BETA:
            beta1_block_ptr = tl.make_block_ptr(
                base=BETA + offset_ld,
                shape=(N, 1),
                strides=(H, 1),
                offsets=(offset_block_n + 3 * BLOCK_N2, 0)
                if REVERSE
                else (offset_block_n, 0),
                block_shape=(BLOCK_N2, 1),
                order=(1, 0),
            )

            beta2_block_ptr = tl.make_block_ptr(
                base=BETA + offset_ld,
                shape=(N, 1),
                strides=(H, 1),
                offsets=(offset_block_n + 2 * BLOCK_N2, 0)
                if REVERSE
                else (offset_block_n + BLOCK_N2, 0),
                block_shape=(BLOCK_N2, 1),
                order=(1, 0),
            )

            beta3_block_ptr = tl.make_block_ptr(
                base=BETA + offset_ld,
                shape=(N, 1),
                strides=(H, 1),
                offsets=(offset_block_n + 1 * BLOCK_N2, 0)
                if REVERSE
                else (offset_block_n + 2 * BLOCK_N2, 0),
                block_shape=(BLOCK_N2, 1),
                order=(1, 0),
            )

            beta4_block_ptr = tl.make_block_ptr(
                base=BETA + offset_ld,
                shape=(N, 1),
                strides=(H, 1),
                offsets=(offset_block_n + 0 * BLOCK_N2, 0)
                if REVERSE
                else (offset_block_n + 3 * BLOCK_N2, 0),
                block_shape=(BLOCK_N2, 1),
                order=(1, 0),
            )

            beta1 = tl.load(
                beta1_block_ptr, boundary_check=(0, 1), padding_option="zero"
            )

            beta2 = tl.load(
                beta2_block_ptr, boundary_check=(0, 1), padding_option="zero"
            )

            beta3 = tl.load(
                beta3_block_ptr, boundary_check=(0, 1), padding_option="zero"
            )

            beta4 = tl.load(
                beta4_block_ptr, boundary_check=(0, 1), padding_option="zero"
            )

        for i in range(NUM_BLOCK_D):
            k1 = tl.load(k1_block_ptr, boundary_check=(0, 1), padding_option="zero")
            k2 = tl.load(k2_block_ptr, boundary_check=(0, 1), padding_option="zero")
            k3 = tl.load(k3_block_ptr, boundary_check=(0, 1), padding_option="zero")
            k4 = tl.load(k4_block_ptr, boundary_check=(0, 1), padding_option="zero")
            if USE_Q:
                q1 = tl.load(q1_block_ptr, boundary_check=(0, 1), padding_option="zero")
                q2 = tl.load(q2_block_ptr, boundary_check=(0, 1), padding_option="zero")
                q3 = tl.load(q3_block_ptr, boundary_check=(0, 1), padding_option="zero")
                q4 = tl.load(q4_block_ptr, boundary_check=(0, 1), padding_option="zero")
            else:
                q1 = k1
                q2 = k2
                q3 = k3
                q4 = k4

            if USE_ALPHA:
                q1 = (q1 * alpha1).to(q1.dtype)
                q2 = (q2 * alpha2).to(q2.dtype)
                q3 = (q3 * alpha3).to(q3.dtype)
                q4 = (q4 * alpha4).to(q4.dtype)

            if USE_BETA:
                k1 = (k1 * beta1).to(k1.dtype)
                k2 = (k2 * beta2).to(k2.dtype)
                k3 = (k3 * beta3).to(k3.dtype)
                k4 = (k4 * beta4).to(k4.dtype)

            k1_trans = tl.trans(k1)
            k2_trans = tl.trans(k2)
            k3_trans = tl.trans(k3)
            k4_trans = tl.trans(k4)

            a11 += tl.dot(q1, k1_trans)
            a21 += tl.dot(q2, k1_trans)
            a22 += tl.dot(q2, k2_trans)
            a31 += tl.dot(q3, k1_trans)
            a32 += tl.dot(q3, k2_trans)
            a33 += tl.dot(q3, k3_trans)
            a41 += tl.dot(q4, k1_trans)
            a42 += tl.dot(q4, k2_trans)
            a43 += tl.dot(q4, k3_trans)
            a44 += tl.dot(q4, k4_trans)

            k1_block_ptr = tl.advance(k1_block_ptr, (0, BLOCK_D))
            k2_block_ptr = tl.advance(k2_block_ptr, (0, BLOCK_D))
            k3_block_ptr = tl.advance(k3_block_ptr, (0, BLOCK_D))
            k4_block_ptr = tl.advance(k4_block_ptr, (0, BLOCK_D))

            if USE_Q:
                q1_block_ptr = tl.advance(q1_block_ptr, (0, BLOCK_D))
                q2_block_ptr = tl.advance(q2_block_ptr, (0, BLOCK_D))
                q3_block_ptr = tl.advance(q3_block_ptr, (0, BLOCK_D))
                q4_block_ptr = tl.advance(q4_block_ptr, (0, BLOCK_D))

        # add decay
        if REVERSE:
            array1 = tl.arange(0, BLOCK_N2) + 3 * BLOCK_N2
            array2 = tl.arange(0, BLOCK_N2) + 2 * BLOCK_N2
            array3 = tl.arange(0, BLOCK_N2) + 1 * BLOCK_N2
            array4 = tl.arange(0, BLOCK_N2)
        else:
            array1 = tl.arange(0, BLOCK_N2)
            array2 = tl.arange(0, BLOCK_N2) + BLOCK_N2
            array3 = tl.arange(0, BLOCK_N2) + 2 * BLOCK_N2
            array4 = tl.arange(0, BLOCK_N2) + 3 * BLOCK_N2
        ld1_sum_block_ptr = LOG_DECAY + offset_ld + offset_block_ld + array1 * H
        ld2_sum_block_ptr = LOG_DECAY + offset_ld + offset_block_ld + array2 * H
        ld3_sum_block_ptr = LOG_DECAY + offset_ld + offset_block_ld + array3 * H
        ld4_sum_block_ptr = LOG_DECAY + offset_ld + offset_block_ld + array4 * H
        mask1 = (offset_block_n + array1) < N
        mask2 = (offset_block_n + array2) < N
        mask3 = (offset_block_n + array3) < N
        mask4 = (offset_block_n + array4) < N
        ld1 = tl.load(ld1_sum_block_ptr, mask=mask1, other=0.0).to(tl.float32)
        ld2 = tl.load(ld2_sum_block_ptr, mask=mask2, other=0.0).to(tl.float32)
        ld3 = tl.load(ld3_sum_block_ptr, mask=mask3, other=0.0).to(tl.float32)
        ld4 = tl.load(ld4_sum_block_ptr, mask=mask4, other=0.0).to(tl.float32)
        diff11 = ld1[:, None] - ld1[None, :]
        diff21 = ld2[:, None] - ld1[None, :]
        diff22 = ld2[:, None] - ld2[None, :]
        diff31 = ld3[:, None] - ld1[None, :]
        diff32 = ld3[:, None] - ld2[None, :]
        diff33 = ld3[:, None] - ld3[None, :]
        diff41 = ld4[:, None] - ld1[None, :]
        diff42 = ld4[:, None] - ld2[None, :]
        diff43 = ld4[:, None] - ld3[None, :]
        diff44 = ld4[:, None] - ld4[None, :]

        if REVERSE:  # triu
            diff11 = tl.where(array1[:, None] < array1[None, :], diff11, -float("inf"))
            diff21 = tl.where(array2[:, None] < array1[None, :], diff21, -float("inf"))
            diff22 = tl.where(array2[:, None] < array2[None, :], diff22, -float("inf"))
            diff31 = tl.where(array3[:, None] < array1[None, :], diff31, -float("inf"))
            diff32 = tl.where(array3[:, None] < array2[None, :], diff32, -float("inf"))
            diff33 = tl.where(array3[:, None] < array3[None, :], diff33, -float("inf"))
            diff41 = tl.where(array4[:, None] < array1[None, :], diff41, -float("inf"))
            diff42 = tl.where(array4[:, None] < array2[None, :], diff42, -float("inf"))
            diff43 = tl.where(array4[:, None] < array3[None, :], diff43, -float("inf"))
            diff44 = tl.where(array4[:, None] < array4[None, :], diff44, -float("inf"))
        else:  # tril
            diff11 = tl.where(array1[:, None] > array1[None, :], diff11, -float("inf"))
            diff21 = tl.where(array2[:, None] > array1[None, :], diff21, -float("inf"))
            diff22 = tl.where(array2[:, None] > array2[None, :], diff22, -float("inf"))
            diff31 = tl.where(array3[:, None] > array1[None, :], diff31, -float("inf"))
            diff32 = tl.where(array3[:, None] > array2[None, :], diff32, -float("inf"))
            diff33 = tl.where(array3[:, None] > array3[None, :], diff33, -float("inf"))
            diff41 = tl.where(array4[:, None] > array1[None, :], diff41, -float("inf"))
            diff42 = tl.where(array4[:, None] > array2[None, :], diff42, -float("inf"))
            diff43 = tl.where(array4[:, None] > array3[None, :], diff43, -float("inf"))
            diff44 = tl.where(array4[:, None] > array4[None, :], diff44, -float("inf"))

        a11 *= tl.exp(diff11)
        a21 *= tl.exp(diff21)
        a22 *= tl.exp(diff22)
        a31 *= tl.exp(diff31)
        a32 *= tl.exp(diff32)
        a33 *= tl.exp(diff33)
        a41 *= tl.exp(diff41)
        a42 *= tl.exp(diff42)
        a43 *= tl.exp(diff43)
        a44 *= tl.exp(diff44)

        zero = tl.zeros((BLOCK_N2, BLOCK_N2), dtype=tl.float32)
        a11_inv = zero
        a22_inv = zero
        a33_inv = zero
        a44_inv = zero

        eye = (array1[:, None] == array1[None, :]).to(a11.dtype)
        for i in tl.static_range(BLOCK_N1):
            # C = (I + A) ^ -1, Ck = -A * Ck + I
            a11_inv = -tl.dot(a11_inv, a11) + eye
            a22_inv = -tl.dot(a22_inv, a22) + eye
            a33_inv = -tl.dot(a33_inv, a33) + eye
            a44_inv = -tl.dot(a44_inv, a44) + eye

        a21_inv = -tl.dot(a22_inv, tl.dot(a21, a11_inv))
        a32_inv = -tl.dot(a33_inv, tl.dot(a32, a22_inv))
        a31_inv = -tl.dot(a33_inv, tl.dot(a31, a11_inv) + tl.dot(a32, a21_inv))
        a43_inv = -tl.dot(a44_inv, tl.dot(a43, a33_inv))
        a42_inv = -tl.dot(a44_inv, tl.dot(a42, a22_inv) + tl.dot(a43, a32_inv))
        a41_inv = -tl.dot(
            a44_inv, tl.dot(a41, a11_inv) + tl.dot(a42, a21_inv) + tl.dot(a43, a31_inv)
        )

        a12_inv = zero
        a13_inv = zero
        a14_inv = zero
        a23_inv = zero
        a24_inv = zero
        a34_inv = zero

        tl.store(
            inv11_block_ptr,
            a11_inv.to(inv11_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            inv12_block_ptr,
            a12_inv.to(inv12_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            inv13_block_ptr,
            a13_inv.to(inv13_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            inv14_block_ptr,
            a14_inv.to(inv14_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )

        tl.store(
            inv21_block_ptr,
            a21_inv.to(inv21_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            inv22_block_ptr,
            a22_inv.to(inv22_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            inv23_block_ptr,
            a23_inv.to(inv23_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            inv24_block_ptr,
            a24_inv.to(inv24_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )

        tl.store(
            inv31_block_ptr,
            a31_inv.to(inv31_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            inv32_block_ptr,
            a32_inv.to(inv32_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            inv33_block_ptr,
            a33_inv.to(inv33_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            inv34_block_ptr,
            a34_inv.to(inv34_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )

        tl.store(
            inv41_block_ptr,
            a41_inv.to(inv41_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            inv42_block_ptr,
            a42_inv.to(inv42_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            inv43_block_ptr,
            a43_inv.to(inv43_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            inv44_block_ptr,
            a44_inv.to(inv44_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )


@triton.heuristics(
    {
        "MAX_BLOCK_N": lambda args: triton.next_power_of_2(args["N"]),
    }
)
@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_D": BLOCK_D_INV_LIST,
        }
    ),
    key=[
        "B",
        "MAX_BLOCK_N",
        "H",
        "D",
    ],
)
@triton.jit
def _krcl_parallel_inverse_diag(
    Q,  # B N H D
    K,  # B N H D
    ATTENTION,  # B H NUM_BLOCK_N BLOCK_N BLOCK_N
    INV,  # B H NUM_BLOCK_N BLOCK_N BLOCK_N
    LOG_DECAY,  # B N H
    ALPHA,  # B N H
    BETA,  # B N H
    CU_SEQLENS,
    USE_Q: tl.constexpr,  # bool
    USE_LD: tl.constexpr,  # bool
    USE_ALPHA: tl.constexpr,  # bool
    USE_BETA: tl.constexpr,  # bool
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    REVERSE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_LOOP_M: tl.constexpr,
    NUM_BLOCK_N: tl.constexpr,
    MAX_BLOCK_N: tl.constexpr,
    USE_ATTENTION: tl.constexpr,
    USE_PAD: tl.constexpr,
):
    NUM_BLOCK_D = tl.cdiv(D, BLOCK_D)
    off_bh = tl.program_id(0)
    off_b = off_bh // H
    off_h = off_bh % H
    off_block_n = tl.program_id(1)
    off_diag = tl.program_id(2)

    offset_block_m = off_diag * BLOCK_M
    offset_block_n = off_block_n * BLOCK_N + offset_block_m
    offset_qk = off_b * N * H * D + off_h * D
    offset_ld = off_b * N * H + off_h
    offset_inv = (
        off_b * H * NUM_BLOCK_N * BLOCK_N * BLOCK_N
        + off_h * NUM_BLOCK_N * BLOCK_N * BLOCK_N
        + off_block_n * BLOCK_N * BLOCK_N
    )
    offset_block_ld = offset_block_n * H

    inv_block_ptr = tl.make_block_ptr(
        base=INV + offset_inv,
        shape=(BLOCK_N, BLOCK_N),
        strides=(BLOCK_N, 1),
        offsets=(offset_block_m, offset_block_m),
        block_shape=(BLOCK_M, BLOCK_M),
        order=(1, 0),
    )

    array = tl.arange(0, BLOCK_M)
    if not USE_ATTENTION:
        a = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)
        if USE_Q:
            q_block_ptr = tl.make_block_ptr(
                base=Q + offset_qk,
                shape=(N, D),
                strides=(H * D, 1),
                offsets=(offset_block_n, 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )

        k_block_ptr = tl.make_block_ptr(
            base=K + offset_qk,
            shape=(N, D),
            strides=(H * D, 1),
            offsets=(offset_block_n, 0),
            block_shape=(BLOCK_M, BLOCK_D),
            order=(1, 0),
        )

        if USE_ALPHA:
            alpha_block_ptr = tl.make_block_ptr(
                base=ALPHA + offset_ld,
                shape=(N, 1),
                strides=(H, 1),
                offsets=(offset_block_n, 0),
                block_shape=(BLOCK_M, 1),
                order=(1, 0),
            )
            alpha = tl.load(
                alpha_block_ptr, boundary_check=(0, 1), padding_option="zero"
            )

        if USE_BETA:
            beta_trans_block_ptr = tl.make_block_ptr(
                base=BETA + offset_ld,
                shape=(N, 1),
                strides=(H, 1),
                offsets=(offset_block_n, 0),
                block_shape=(BLOCK_M, 1),
                order=(1, 0),
            )
            beta = tl.load(
                beta_trans_block_ptr, boundary_check=(0, 1), padding_option="zero"
            )

        for i in range(NUM_BLOCK_D):
            k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
            if USE_Q:
                q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
            else:
                q = k
            if USE_ALPHA:
                q = (q * alpha).to(q.dtype)
            if USE_BETA:
                k = (k * beta).to(k.dtype)

            k_trans = tl.trans(k)

            a += tl.dot(q, k_trans)

            k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_D))
            if USE_Q:
                q_block_ptr = tl.advance(q_block_ptr, (0, BLOCK_D))

        if USE_LD:
            # add decay
            ld_sum_block_ptr = LOG_DECAY + offset_ld + offset_block_ld + array * H
            mask = (offset_block_n + array) < N
            ld = tl.load(ld_sum_block_ptr, mask=mask, other=0.0).to(tl.float32)
            diff = ld[:, None] - ld[None, :]
            if REVERSE:  # triu
                diff = tl.where(array[:, None] < array[None, :], diff, -float("inf"))
            else:  # tril
                diff = tl.where(array[:, None] > array[None, :], diff, -float("inf"))
            attn_mask = tl.exp(diff)
        else:
            if REVERSE:  # triu
                attn_mask = tl.where(array[:, None] < array[None, :], 1, 0)
            else:  # tril
                attn_mask = tl.where(array[:, None] > array[None, :], 1, 0)
        a *= attn_mask
    else:
        a_block_ptr = tl.make_block_ptr(
            base=ATTENTION + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(offset_block_m, offset_block_m),
            block_shape=(BLOCK_M, BLOCK_M),
            order=(1, 0),
        )
        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero").to(
            tl.float32
        )

    # # todo: no use, for reference only
    # # 1 + x + ... + x ^ (2 ^ n) = prod (1 + x ^ (2 ^ i)), i = 1, ... , n - 1
    # eye = (array[:, None] == array[None, :]).to(a.dtype)
    # a_inv = eye - a
    # for i in range(NUM_LOOP_M - 1):
    #     # a_inv = (1 + a ^ (2 ^ i)) * a_inv
    #     a = tl.dot(a, a)
    #     a_inv += tl.dot(a_inv, a)

    # tl.store(
    #     inv_block_ptr, a_inv.to(inv_block_ptr.dtype.element_ty), boundary_check=(0, 1)
    # )

    eye = (array[:, None] == array[None, :]).to(a.dtype)
    # todo: update BLOCK_M
    if offset_block_n >= N:
        L = 0
    elif offset_block_n + BLOCK_M >= N:
        L = N - offset_block_n
    else:
        L = BLOCK_M

    # for i in range(BLOCK_M):
    for i in range(L):
        if REVERSE:
            j = BLOCK_M - 1 - i
        else:
            j = i

        if USE_ATTENTION:
            # n 1
            index_i = array == j
            ai_block_ptr = (
                ATTENTION
                + offset_inv
                + offset_block_m * BLOCK_N
                + offset_block_m
                + j * BLOCK_N
                + array
            )
            ai = tl.load(ai_block_ptr).to(tl.float32)
        else:
            # n 1
            index_i = array == j
            # n
            ai = tl.sum(tl.where(index_i[:, None], a, 0), axis=0)

        # compute
        if REVERSE:
            index_i_ = array > j
        else:
            index_i_ = array < j
        a_ = tl.where(index_i_[:, None], a, 0)
        ai_ = index_i - tl.sum(ai[:, None] * a_, axis=0)
        a = tl.where(index_i[:, None], ai_, a)

    tl.store(inv_block_ptr, a.to(inv_block_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "MAX_BLOCK_N": lambda args: triton.next_power_of_2(args["N"]),
    }
)
@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_D": BLOCK_D_INV_LIST,
        }
    ),
    key=[
        "B",
        "MAX_BLOCK_N",
        "H",
        "D",
    ],
)
@triton.jit
def _krcl_parallel_inverse_merge(
    Q,  # B N H D
    K,  # B N H D
    ATTENTION,  # B H NUM_BLOCK_N BLOCK_N BLOCK_N
    INV,  # B H NUM_BLOCK_N BLOCK_N BLOCK_N
    LOG_DECAY,  # B N H
    ALPHA,  # B N H
    BETA,  # B N H
    CU_SEQLENS,
    USE_Q: tl.constexpr,  # bool
    USE_LD: tl.constexpr,  # bool
    USE_ALPHA: tl.constexpr,  # bool
    USE_BETA: tl.constexpr,  # bool
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    REVERSE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_BLOCK_N: tl.constexpr,
    NUM_BLOCK_M: tl.constexpr,
    MAX_BLOCK_N: tl.constexpr,
    USE_ATTENTION: tl.constexpr,
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
    offset_block_ld = offset_block_n * H

    if NUM_BLOCK_M == 2:
        inv11_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(BLOCK_M, BLOCK_M) if REVERSE else (0, 0),
            block_shape=(BLOCK_M, BLOCK_M),
            order=(1, 0),
        )

        inv12_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(BLOCK_M, 0) if REVERSE else (0, BLOCK_M),
            block_shape=(BLOCK_M, BLOCK_M),
            order=(1, 0),
        )

        inv21_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(0, BLOCK_M) if REVERSE else (BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_M),
            order=(1, 0),
        )

        inv22_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(0, 0) if REVERSE else (BLOCK_M, BLOCK_M),
            block_shape=(BLOCK_M, BLOCK_M),
            order=(1, 0),
        )

        if not USE_ATTENTION:
            a11 = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)
            a21 = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)
            a22 = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)
            if USE_Q:
                q1_block_ptr = tl.make_block_ptr(
                    base=Q + offset_qk,
                    shape=(N, D),
                    strides=(H * D, 1),
                    offsets=(offset_block_n + BLOCK_M, 0)
                    if REVERSE
                    else (offset_block_n, 0),
                    block_shape=(BLOCK_M, BLOCK_D),
                    order=(1, 0),
                )

                q2_block_ptr = tl.make_block_ptr(
                    base=Q + offset_qk,
                    shape=(N, D),
                    strides=(H * D, 1),
                    offsets=(offset_block_n, 0)
                    if REVERSE
                    else (offset_block_n + BLOCK_M, 0),
                    block_shape=(BLOCK_M, BLOCK_D),
                    order=(1, 0),
                )

            k1_block_ptr = tl.make_block_ptr(
                base=K + offset_qk,
                shape=(N, D),
                strides=(H * D, 1),
                offsets=(offset_block_n + BLOCK_M, 0)
                if REVERSE
                else (offset_block_n, 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )

            k2_block_ptr = tl.make_block_ptr(
                base=K + offset_qk,
                shape=(N, D),
                strides=(H * D, 1),
                offsets=(offset_block_n, 0)
                if REVERSE
                else (offset_block_n + BLOCK_M, 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )

            if USE_ALPHA:
                alpha1_block_ptr = tl.make_block_ptr(
                    base=ALPHA + offset_ld,
                    shape=(N, 1),
                    strides=(H, 1),
                    offsets=(offset_block_n + BLOCK_M, 0)
                    if REVERSE
                    else (offset_block_n, 0),
                    block_shape=(BLOCK_M, 1),
                    order=(1, 0),
                )

                alpha2_block_ptr = tl.make_block_ptr(
                    base=ALPHA + offset_ld,
                    shape=(N, 1),
                    strides=(H, 1),
                    offsets=(offset_block_n, 0)
                    if REVERSE
                    else (offset_block_n + BLOCK_M, 0),
                    block_shape=(BLOCK_M, 1),
                    order=(1, 0),
                )

                alpha1 = tl.load(
                    alpha1_block_ptr, boundary_check=(0, 1), padding_option="zero"
                ).to(tl.float32)

                alpha2 = tl.load(
                    alpha2_block_ptr, boundary_check=(0, 1), padding_option="zero"
                ).to(tl.float32)

            if USE_BETA:
                beta1_block_ptr = tl.make_block_ptr(
                    base=BETA + offset_ld,
                    shape=(N, 1),
                    strides=(H, 1),
                    offsets=(offset_block_n + BLOCK_M, 0)
                    if REVERSE
                    else (offset_block_n, 0),
                    block_shape=(BLOCK_M, 1),
                    order=(1, 0),
                )

                beta2_block_ptr = tl.make_block_ptr(
                    base=BETA + offset_ld,
                    shape=(N, 1),
                    strides=(H, 1),
                    offsets=(offset_block_n, 0)
                    if REVERSE
                    else (offset_block_n + BLOCK_M, 0),
                    block_shape=(BLOCK_M, 1),
                    order=(1, 0),
                )

                beta1 = tl.load(
                    beta1_block_ptr, boundary_check=(0, 1), padding_option="zero"
                )

                beta2 = tl.load(
                    beta2_block_ptr, boundary_check=(0, 1), padding_option="zero"
                )

            for i in range(NUM_BLOCK_D):
                k1 = tl.load(k1_block_ptr, boundary_check=(0, 1), padding_option="zero")
                if USE_Q:
                    q2 = tl.load(
                        q2_block_ptr, boundary_check=(0, 1), padding_option="zero"
                    )
                else:
                    k2 = tl.load(
                        k2_block_ptr, boundary_check=(0, 1), padding_option="zero"
                    )
                    q2 = k2

                if USE_ALPHA:
                    q2 = (q2 * alpha2).to(q2.dtype)

                if USE_BETA:
                    k1 = (k1 * beta1).to(k1.dtype)

                k1_trans = tl.trans(k1)

                a21 += tl.dot(q2, k1_trans)

                k1_block_ptr = tl.advance(k1_block_ptr, (0, BLOCK_D))

                if USE_Q:
                    q2_block_ptr = tl.advance(q2_block_ptr, (0, BLOCK_D))
                else:
                    k2_block_ptr = tl.advance(k2_block_ptr, (0, BLOCK_D))

            # add decay
            if REVERSE:
                array1 = tl.arange(0, BLOCK_M) + BLOCK_M
                array2 = tl.arange(0, BLOCK_M)
            else:
                array1 = tl.arange(0, BLOCK_M)
                array2 = tl.arange(0, BLOCK_M) + BLOCK_M

            if USE_LD:
                ld1_sum_block_ptr = LOG_DECAY + offset_ld + offset_block_ld + array1 * H
                ld2_sum_block_ptr = LOG_DECAY + offset_ld + offset_block_ld + array2 * H
                mask1 = (offset_block_n + array1) < N
                mask2 = (offset_block_n + array2) < N
                ld1 = tl.load(ld1_sum_block_ptr, mask=mask1, other=0.0).to(tl.float32)
                ld2 = tl.load(ld2_sum_block_ptr, mask=mask2, other=0.0).to(tl.float32)
                diff21 = ld2[:, None] - ld1[None, :]
                if REVERSE:  # triu
                    diff21 = tl.where(
                        array2[:, None] < array1[None, :], diff21, -float("inf")
                    )
                else:  # tril
                    diff21 = tl.where(
                        array2[:, None] > array1[None, :], diff21, -float("inf")
                    )
                attn_mask21 = tl.exp(diff21)
            else:
                if REVERSE:  # triu
                    attn_mask21 = tl.where(array2[:, None] < array1[None, :], 1, 0)
                else:  # tril
                    attn_mask21 = tl.where(array2[:, None] > array1[None, :], 1, 0)

            a21 *= attn_mask21
        else:
            a21_block_ptr = tl.make_block_ptr(
                base=ATTENTION + offset_inv,
                shape=(BLOCK_N, BLOCK_N),
                strides=(BLOCK_N, 1),
                offsets=(0, BLOCK_M) if REVERSE else (BLOCK_M, 0),
                block_shape=(BLOCK_M, BLOCK_M),
                order=(1, 0),
            )

            a21 = tl.load(
                a21_block_ptr, boundary_check=(0, 1), padding_option="zero"
            ).to(tl.float32)

        a11_inv = tl.load(
            inv11_block_ptr, boundary_check=(0, 1), padding_option="zero"
        ).to(tl.float32)
        a22_inv = tl.load(
            inv22_block_ptr, boundary_check=(0, 1), padding_option="zero"
        ).to(tl.float32)

        a21_inv = -tl.dot(a22_inv, tl.dot(a21, a11_inv))

        tl.store(
            inv21_block_ptr,
            a21_inv.to(inv21_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )
    else:
        inv11_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(3 * BLOCK_M, 3 * BLOCK_M) if REVERSE else (0, 0),
            block_shape=(BLOCK_M, BLOCK_M),
            order=(1, 0),
        )

        inv21_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(2 * BLOCK_M, 3 * BLOCK_M) if REVERSE else (BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_M),
            order=(1, 0),
        )

        inv22_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(2 * BLOCK_M, 2 * BLOCK_M) if REVERSE else (BLOCK_M, BLOCK_M),
            block_shape=(BLOCK_M, BLOCK_M),
            order=(1, 0),
        )

        inv31_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(1 * BLOCK_M, 3 * BLOCK_M) if REVERSE else (2 * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_M),
            order=(1, 0),
        )

        inv32_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(1 * BLOCK_M, 2 * BLOCK_M) if REVERSE else (2 * BLOCK_M, BLOCK_M),
            block_shape=(BLOCK_M, BLOCK_M),
            order=(1, 0),
        )

        inv33_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(1 * BLOCK_M, 1 * BLOCK_M)
            if REVERSE
            else (2 * BLOCK_M, 2 * BLOCK_M),
            block_shape=(BLOCK_M, BLOCK_M),
            order=(1, 0),
        )

        inv41_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(0, 3 * BLOCK_M) if REVERSE else (3 * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_M),
            order=(1, 0),
        )

        inv42_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(0, 2 * BLOCK_M) if REVERSE else (3 * BLOCK_M, BLOCK_M),
            block_shape=(BLOCK_M, BLOCK_M),
            order=(1, 0),
        )

        inv43_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(0, 1 * BLOCK_M) if REVERSE else (3 * BLOCK_M, 2 * BLOCK_M),
            block_shape=(BLOCK_M, BLOCK_M),
            order=(1, 0),
        )

        inv44_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(0, 0) if REVERSE else (3 * BLOCK_M, 3 * BLOCK_M),
            block_shape=(BLOCK_M, BLOCK_M),
            order=(1, 0),
        )

        if not USE_ATTENTION:
            a21 = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)
            a31 = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)
            a32 = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)
            a41 = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)
            a42 = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)
            a43 = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)

            if USE_Q:
                q1_block_ptr = tl.make_block_ptr(
                    base=Q + offset_qk,
                    shape=(N, D),
                    strides=(H * D, 1),
                    offsets=(offset_block_n + 3 * BLOCK_M, 0)
                    if REVERSE
                    else (offset_block_n, 0),
                    block_shape=(BLOCK_M, BLOCK_D),
                    order=(1, 0),
                )

                q2_block_ptr = tl.make_block_ptr(
                    base=Q + offset_qk,
                    shape=(N, D),
                    strides=(H * D, 1),
                    offsets=(offset_block_n + 2 * BLOCK_M, 0)
                    if REVERSE
                    else (offset_block_n + BLOCK_M, 0),
                    block_shape=(BLOCK_M, BLOCK_D),
                    order=(1, 0),
                )

                q3_block_ptr = tl.make_block_ptr(
                    base=Q + offset_qk,
                    shape=(N, D),
                    strides=(H * D, 1),
                    offsets=(offset_block_n + BLOCK_M, 0)
                    if REVERSE
                    else (offset_block_n + 2 * BLOCK_M, 0),
                    block_shape=(BLOCK_M, BLOCK_D),
                    order=(1, 0),
                )

                q4_block_ptr = tl.make_block_ptr(
                    base=Q + offset_qk,
                    shape=(N, D),
                    strides=(H * D, 1),
                    offsets=(offset_block_n, 0)
                    if REVERSE
                    else (offset_block_n + 3 * BLOCK_M, 0),
                    block_shape=(BLOCK_M, BLOCK_D),
                    order=(1, 0),
                )

            k1_block_ptr = tl.make_block_ptr(
                base=K + offset_qk,
                shape=(N, D),
                strides=(H * D, 1),
                offsets=(offset_block_n + 3 * BLOCK_M, 0)
                if REVERSE
                else (offset_block_n, 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )

            k2_block_ptr = tl.make_block_ptr(
                base=K + offset_qk,
                shape=(N, D),
                strides=(H * D, 1),
                offsets=(offset_block_n + 2 * BLOCK_M, 0)
                if REVERSE
                else (offset_block_n + BLOCK_M, 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )

            k3_block_ptr = tl.make_block_ptr(
                base=K + offset_qk,
                shape=(N, D),
                strides=(H * D, 1),
                offsets=(offset_block_n + BLOCK_M, 0)
                if REVERSE
                else (offset_block_n + 2 * BLOCK_M, 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )

            k4_block_ptr = tl.make_block_ptr(
                base=K + offset_qk,
                shape=(N, D),
                strides=(H * D, 1),
                offsets=(offset_block_n, 0)
                if REVERSE
                else (offset_block_n + 3 * BLOCK_M, 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )

            if USE_ALPHA:
                alpha1_block_ptr = tl.make_block_ptr(
                    base=ALPHA + offset_ld,
                    shape=(N, 1),
                    strides=(H, 1),
                    offsets=(offset_block_n + 3 * BLOCK_M, 0)
                    if REVERSE
                    else (offset_block_n, 0),
                    block_shape=(BLOCK_M, 1),
                    order=(1, 0),
                )

                alpha2_block_ptr = tl.make_block_ptr(
                    base=ALPHA + offset_ld,
                    shape=(N, 1),
                    strides=(H, 1),
                    offsets=(offset_block_n + 2 * BLOCK_M, 0)
                    if REVERSE
                    else (offset_block_n + BLOCK_M, 0),
                    block_shape=(BLOCK_M, 1),
                    order=(1, 0),
                )

                alpha3_block_ptr = tl.make_block_ptr(
                    base=ALPHA + offset_ld,
                    shape=(N, 1),
                    strides=(H, 1),
                    offsets=(offset_block_n + 1 * BLOCK_M, 0)
                    if REVERSE
                    else (offset_block_n + 2 * BLOCK_M, 0),
                    block_shape=(BLOCK_M, 1),
                    order=(1, 0),
                )

                alpha4_block_ptr = tl.make_block_ptr(
                    base=ALPHA + offset_ld,
                    shape=(N, 1),
                    strides=(H, 1),
                    offsets=(offset_block_n + 0 * BLOCK_M, 0)
                    if REVERSE
                    else (offset_block_n + 3 * BLOCK_M, 0),
                    block_shape=(BLOCK_M, 1),
                    order=(1, 0),
                )

                alpha1 = tl.load(
                    alpha1_block_ptr, boundary_check=(0, 1), padding_option="zero"
                )

                alpha2 = tl.load(
                    alpha2_block_ptr, boundary_check=(0, 1), padding_option="zero"
                )

                alpha3 = tl.load(
                    alpha3_block_ptr, boundary_check=(0, 1), padding_option="zero"
                )

                alpha4 = tl.load(
                    alpha4_block_ptr, boundary_check=(0, 1), padding_option="zero"
                )

            if USE_BETA:
                beta1_block_ptr = tl.make_block_ptr(
                    base=BETA + offset_ld,
                    shape=(N, 1),
                    strides=(H, 1),
                    offsets=(offset_block_n + 3 * BLOCK_M, 0)
                    if REVERSE
                    else (offset_block_n, 0),
                    block_shape=(BLOCK_M, 1),
                    order=(1, 0),
                )

                beta2_block_ptr = tl.make_block_ptr(
                    base=BETA + offset_ld,
                    shape=(N, 1),
                    strides=(H, 1),
                    offsets=(offset_block_n + 2 * BLOCK_M, 0)
                    if REVERSE
                    else (offset_block_n + BLOCK_M, 0),
                    block_shape=(BLOCK_M, 1),
                    order=(1, 0),
                )

                beta3_block_ptr = tl.make_block_ptr(
                    base=BETA + offset_ld,
                    shape=(N, 1),
                    strides=(H, 1),
                    offsets=(offset_block_n + 1 * BLOCK_M, 0)
                    if REVERSE
                    else (offset_block_n + 2 * BLOCK_M, 0),
                    block_shape=(BLOCK_M, 1),
                    order=(1, 0),
                )

                beta4_block_ptr = tl.make_block_ptr(
                    base=BETA + offset_ld,
                    shape=(N, 1),
                    strides=(H, 1),
                    offsets=(offset_block_n + 0 * BLOCK_M, 0)
                    if REVERSE
                    else (offset_block_n + 3 * BLOCK_M, 0),
                    block_shape=(BLOCK_M, 1),
                    order=(1, 0),
                )

                beta1 = tl.load(
                    beta1_block_ptr, boundary_check=(0, 1), padding_option="zero"
                )

                beta2 = tl.load(
                    beta2_block_ptr, boundary_check=(0, 1), padding_option="zero"
                )

                beta3 = tl.load(
                    beta3_block_ptr, boundary_check=(0, 1), padding_option="zero"
                )

                beta4 = tl.load(
                    beta4_block_ptr, boundary_check=(0, 1), padding_option="zero"
                )

            for i in range(NUM_BLOCK_D):
                k1 = tl.load(k1_block_ptr, boundary_check=(0, 1), padding_option="zero")
                k2 = tl.load(k2_block_ptr, boundary_check=(0, 1), padding_option="zero")
                k3 = tl.load(k3_block_ptr, boundary_check=(0, 1), padding_option="zero")
                k4 = tl.load(k4_block_ptr, boundary_check=(0, 1), padding_option="zero")

                if USE_Q:
                    q1 = tl.load(
                        q1_block_ptr, boundary_check=(0, 1), padding_option="zero"
                    )
                    q2 = tl.load(
                        q2_block_ptr, boundary_check=(0, 1), padding_option="zero"
                    )
                    q3 = tl.load(
                        q3_block_ptr, boundary_check=(0, 1), padding_option="zero"
                    )
                    q4 = tl.load(
                        q4_block_ptr, boundary_check=(0, 1), padding_option="zero"
                    )
                else:
                    q1 = k1
                    q2 = k2
                    q3 = k3
                    q4 = k4

                if USE_ALPHA:
                    q1 = (q1 * alpha1).to(q1.dtype)
                    q2 = (q2 * alpha2).to(q2.dtype)
                    q3 = (q3 * alpha3).to(q3.dtype)
                    q4 = (q4 * alpha4).to(q4.dtype)

                if USE_BETA:
                    k1 = (k1 * beta1).to(k1.dtype)
                    k2 = (k2 * beta2).to(k2.dtype)
                    k3 = (k3 * beta3).to(k3.dtype)
                    k4 = (k4 * beta4).to(k4.dtype)

                k1_trans = tl.trans(k1)
                k2_trans = tl.trans(k2)
                k3_trans = tl.trans(k3)

                a21 += tl.dot(q2, k1_trans)
                a31 += tl.dot(q3, k1_trans)
                a32 += tl.dot(q3, k2_trans)
                a41 += tl.dot(q4, k1_trans)
                a42 += tl.dot(q4, k2_trans)
                a43 += tl.dot(q4, k3_trans)

                k1_block_ptr = tl.advance(k1_block_ptr, (0, BLOCK_D))
                k2_block_ptr = tl.advance(k2_block_ptr, (0, BLOCK_D))
                k3_block_ptr = tl.advance(k3_block_ptr, (0, BLOCK_D))
                k4_block_ptr = tl.advance(k4_block_ptr, (0, BLOCK_D))

                if USE_Q:
                    q1_block_ptr = tl.advance(q1_block_ptr, (0, BLOCK_D))
                    q2_block_ptr = tl.advance(q2_block_ptr, (0, BLOCK_D))
                    q3_block_ptr = tl.advance(q3_block_ptr, (0, BLOCK_D))
                    q4_block_ptr = tl.advance(q4_block_ptr, (0, BLOCK_D))

            # add decay
            if REVERSE:
                array1 = tl.arange(0, BLOCK_M) + 3 * BLOCK_M
                array2 = tl.arange(0, BLOCK_M) + 2 * BLOCK_M
                array3 = tl.arange(0, BLOCK_M) + 1 * BLOCK_M
                array4 = tl.arange(0, BLOCK_M)
            else:
                array1 = tl.arange(0, BLOCK_M)
                array2 = tl.arange(0, BLOCK_M) + BLOCK_M
                array3 = tl.arange(0, BLOCK_M) + 2 * BLOCK_M
                array4 = tl.arange(0, BLOCK_M) + 3 * BLOCK_M
            ld1_sum_block_ptr = LOG_DECAY + offset_ld + offset_block_ld + array1 * H
            ld2_sum_block_ptr = LOG_DECAY + offset_ld + offset_block_ld + array2 * H
            ld3_sum_block_ptr = LOG_DECAY + offset_ld + offset_block_ld + array3 * H
            ld4_sum_block_ptr = LOG_DECAY + offset_ld + offset_block_ld + array4 * H

            if USE_LD:
                mask1 = (offset_block_n + array1) < N
                mask2 = (offset_block_n + array2) < N
                mask3 = (offset_block_n + array3) < N
                mask4 = (offset_block_n + array4) < N
                ld1 = tl.load(ld1_sum_block_ptr, mask=mask1, other=0.0).to(tl.float32)
                ld2 = tl.load(ld2_sum_block_ptr, mask=mask2, other=0.0).to(tl.float32)
                ld3 = tl.load(ld3_sum_block_ptr, mask=mask3, other=0.0).to(tl.float32)
                ld4 = tl.load(ld4_sum_block_ptr, mask=mask4, other=0.0).to(tl.float32)
                diff21 = ld2[:, None] - ld1[None, :]
                diff31 = ld3[:, None] - ld1[None, :]
                diff32 = ld3[:, None] - ld2[None, :]
                diff41 = ld4[:, None] - ld1[None, :]
                diff42 = ld4[:, None] - ld2[None, :]
                diff43 = ld4[:, None] - ld3[None, :]

                if REVERSE:  # triu
                    diff21 = tl.where(
                        array2[:, None] < array1[None, :], diff21, -float("inf")
                    )
                    diff31 = tl.where(
                        array3[:, None] < array1[None, :], diff31, -float("inf")
                    )
                    diff32 = tl.where(
                        array3[:, None] < array2[None, :], diff32, -float("inf")
                    )
                    diff41 = tl.where(
                        array4[:, None] < array1[None, :], diff41, -float("inf")
                    )
                    diff42 = tl.where(
                        array4[:, None] < array2[None, :], diff42, -float("inf")
                    )
                    diff43 = tl.where(
                        array4[:, None] < array3[None, :], diff43, -float("inf")
                    )
                else:  # tril
                    diff21 = tl.where(
                        array2[:, None] > array1[None, :], diff21, -float("inf")
                    )
                    diff31 = tl.where(
                        array3[:, None] > array1[None, :], diff31, -float("inf")
                    )
                    diff32 = tl.where(
                        array3[:, None] > array2[None, :], diff32, -float("inf")
                    )
                    diff41 = tl.where(
                        array4[:, None] > array1[None, :], diff41, -float("inf")
                    )
                    diff42 = tl.where(
                        array4[:, None] > array2[None, :], diff42, -float("inf")
                    )
                    diff43 = tl.where(
                        array4[:, None] > array3[None, :], diff43, -float("inf")
                    )

                attn_mask21 = tl.exp(diff21)
                attn_mask31 = tl.exp(diff31)
                attn_mask32 = tl.exp(diff32)
                attn_mask41 = tl.exp(diff41)
                attn_mask42 = tl.exp(diff42)
                attn_mask43 = tl.exp(diff43)
            else:
                if REVERSE:  # triu
                    attn_mask21 = tl.where(array2[:, None] < array1[None, :], 1, 0)
                    attn_mask31 = tl.where(array3[:, None] < array1[None, :], 1, 0)
                    attn_mask32 = tl.where(array3[:, None] < array2[None, :], 1, 0)
                    attn_mask41 = tl.where(array4[:, None] < array1[None, :], 1, 0)
                    attn_mask42 = tl.where(array4[:, None] < array2[None, :], 1, 0)
                    attn_mask43 = tl.where(array4[:, None] < array3[None, :], 1, 0)
                else:  # tril
                    attn_mask21 = tl.where(array2[:, None] > array1[None, :], 1, 0)
                    attn_mask31 = tl.where(array3[:, None] > array1[None, :], 1, 0)
                    attn_mask32 = tl.where(array3[:, None] > array2[None, :], 1, 0)
                    attn_mask41 = tl.where(array4[:, None] > array1[None, :], 1, 0)
                    attn_mask42 = tl.where(array4[:, None] > array2[None, :], 1, 0)
                    attn_mask43 = tl.where(array4[:, None] > array3[None, :], 1, 0)

            a21 *= attn_mask21
            a31 *= attn_mask31
            a32 *= attn_mask32
            a41 *= attn_mask41
            a42 *= attn_mask42
            a43 *= attn_mask43
        else:
            a21_block_ptr = tl.make_block_ptr(
                base=ATTENTION + offset_inv,
                shape=(BLOCK_N, BLOCK_N),
                strides=(BLOCK_N, 1),
                offsets=(2 * BLOCK_M, 3 * BLOCK_M) if REVERSE else (BLOCK_M, 0),
                block_shape=(BLOCK_M, BLOCK_M),
                order=(1, 0),
            )

            a31_block_ptr = tl.make_block_ptr(
                base=ATTENTION + offset_inv,
                shape=(BLOCK_N, BLOCK_N),
                strides=(BLOCK_N, 1),
                offsets=(1 * BLOCK_M, 3 * BLOCK_M) if REVERSE else (2 * BLOCK_M, 0),
                block_shape=(BLOCK_M, BLOCK_M),
                order=(1, 0),
            )

            a32_block_ptr = tl.make_block_ptr(
                base=ATTENTION + offset_inv,
                shape=(BLOCK_N, BLOCK_N),
                strides=(BLOCK_N, 1),
                offsets=(1 * BLOCK_M, 2 * BLOCK_M)
                if REVERSE
                else (2 * BLOCK_M, BLOCK_M),
                block_shape=(BLOCK_M, BLOCK_M),
                order=(1, 0),
            )

            a41_block_ptr = tl.make_block_ptr(
                base=ATTENTION + offset_inv,
                shape=(BLOCK_N, BLOCK_N),
                strides=(BLOCK_N, 1),
                offsets=(0, 3 * BLOCK_M) if REVERSE else (3 * BLOCK_M, 0),
                block_shape=(BLOCK_M, BLOCK_M),
                order=(1, 0),
            )

            a42_block_ptr = tl.make_block_ptr(
                base=ATTENTION + offset_inv,
                shape=(BLOCK_N, BLOCK_N),
                strides=(BLOCK_N, 1),
                offsets=(0, 2 * BLOCK_M) if REVERSE else (3 * BLOCK_M, BLOCK_M),
                block_shape=(BLOCK_M, BLOCK_M),
                order=(1, 0),
            )

            a43_block_ptr = tl.make_block_ptr(
                base=ATTENTION + offset_inv,
                shape=(BLOCK_N, BLOCK_N),
                strides=(BLOCK_N, 1),
                offsets=(0, 1 * BLOCK_M) if REVERSE else (3 * BLOCK_M, 2 * BLOCK_M),
                block_shape=(BLOCK_M, BLOCK_M),
                order=(1, 0),
            )

            a21 = tl.load(
                a21_block_ptr, boundary_check=(0, 1), padding_option="zero"
            ).to(tl.float32)
            a31 = tl.load(
                a31_block_ptr, boundary_check=(0, 1), padding_option="zero"
            ).to(tl.float32)
            a32 = tl.load(
                a32_block_ptr, boundary_check=(0, 1), padding_option="zero"
            ).to(tl.float32)
            a41 = tl.load(
                a41_block_ptr, boundary_check=(0, 1), padding_option="zero"
            ).to(tl.float32)
            a42 = tl.load(
                a42_block_ptr, boundary_check=(0, 1), padding_option="zero"
            ).to(tl.float32)
            a43 = tl.load(
                a43_block_ptr, boundary_check=(0, 1), padding_option="zero"
            ).to(tl.float32)

        a11_inv = tl.load(
            inv11_block_ptr, boundary_check=(0, 1), padding_option="zero"
        ).to(tl.float32)
        a22_inv = tl.load(
            inv22_block_ptr, boundary_check=(0, 1), padding_option="zero"
        ).to(tl.float32)
        a33_inv = tl.load(
            inv33_block_ptr, boundary_check=(0, 1), padding_option="zero"
        ).to(tl.float32)
        a44_inv = tl.load(
            inv44_block_ptr, boundary_check=(0, 1), padding_option="zero"
        ).to(tl.float32)

        a21_inv = -tl.dot(a22_inv, tl.dot(a21, a11_inv))
        a32_inv = -tl.dot(a33_inv, tl.dot(a32, a22_inv))
        a31_inv = -tl.dot(a33_inv, tl.dot(a31, a11_inv) + tl.dot(a32, a21_inv))
        a43_inv = -tl.dot(a44_inv, tl.dot(a43, a33_inv))
        a42_inv = -tl.dot(a44_inv, tl.dot(a42, a22_inv) + tl.dot(a43, a32_inv))
        a41_inv = -tl.dot(
            a44_inv, tl.dot(a41, a11_inv) + tl.dot(a42, a21_inv) + tl.dot(a43, a31_inv)
        )

        tl.store(
            inv21_block_ptr,
            a21_inv.to(inv21_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )

        tl.store(
            inv31_block_ptr,
            a31_inv.to(inv31_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            inv32_block_ptr,
            a32_inv.to(inv32_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )

        tl.store(
            inv41_block_ptr,
            a41_inv.to(inv41_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            inv42_block_ptr,
            a42_inv.to(inv42_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            inv43_block_ptr,
            a43_inv.to(inv43_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )


@triton.heuristics(
    {
        "MAX_BLOCK_N": lambda args: triton.next_power_of_2(args["N"]),
    }
)
@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "num_stages": [2, 3, 4],
            "BLOCK_D": BLOCK_D_LIST,
        }
    ),
    key=[
        "B",
        "MAX_BLOCK_N",
        "H",
        "D",
    ],
)
@triton.jit
def _krcl_parallel_inverse_attention(
    Q,  # B N H D
    K,  # B N H D
    ATTENTION,  # B H NUM_BLOCK_N BLOCK_N BLOCK_N
    LOG_DECAY,  # B N H
    ALPHA,  # B N H
    BETA,  # B N H
    CU_SEQLENS,
    USE_Q: tl.constexpr,  # bool
    USE_LD: tl.constexpr,  # bool
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
    MAX_BLOCK_N: tl.constexpr,
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

    a_block_ptr = tl.make_block_ptr(
        base=ATTENTION + offset_inv,
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
        alpha = tl.load(alpha_block_ptr, boundary_check=(0, 1), padding_option="zero")

    if USE_BETA:
        beta_trans_block_ptr = tl.make_block_ptr(
            base=BETA + offset_ld,
            shape=(N, 1),
            strides=(H, 1),
            offsets=(offset_block_n, 0),
            block_shape=(BLOCK_N, 1),
            order=(1, 0),
        )
        beta = tl.load(
            beta_trans_block_ptr, boundary_check=(0, 1), padding_option="zero"
        )

    for i in range(NUM_BLOCK_D):
        k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        if USE_Q:
            q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        else:
            q = k
        if USE_ALPHA:
            q = (q * alpha).to(q.dtype)
        if USE_BETA:
            k = (k * beta).to(k.dtype)

        k_trans = tl.trans(k)

        a += tl.dot(q, k_trans)

        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_D))
        if USE_Q:
            q_block_ptr = tl.advance(q_block_ptr, (0, BLOCK_D))

    # add decay
    if USE_LD:
        ld_sum_block_ptr = LOG_DECAY + offset_ld + offset_block_ld + array * H
        mask = (offset_block_n + array) < N
        ld = tl.load(ld_sum_block_ptr, mask=mask, other=0.0).to(tl.float32)
        diff = ld[:, None] - ld[None, :]
        if REVERSE:  # triu
            diff = tl.where(array[:, None] < array[None, :], diff, -float("inf"))
        else:  # tril
            diff = tl.where(array[:, None] > array[None, :], diff, -float("inf"))
        attn_mask = tl.exp(diff)
    else:
        if REVERSE:  # triu
            attn_mask = tl.where(array[:, None] < array[None, :], 1, 0)
        else:  # tril
            attn_mask = tl.where(array[:, None] > array[None, :], 1, 0)

    a *= attn_mask

    tl.store(a_block_ptr, a.to(a_block_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "MAX_BLOCK_N": lambda args: triton.next_power_of_2(args["N"]),
    }
)
@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_D": BLOCK_D_LIST,
            "BLOCK_E": BLOCK_E_LIST,
        }
    ),
    key=[
        "B",
        "MAX_BLOCK_N",
        "H",
        "D",
        "E",
    ],
)
@triton.jit
def _krcl_parallel_chunk_loop(
    Q,  # B N H D
    K,  # B N H D
    V,  # B N H E
    O,  # B N H E NUM_BLOCK_D
    INV,  # B H NUM_BLOCK_N BLOCK_N BLOCK_N
    LOG_DECAY,  # B N H
    ALPHA,  # B N H
    BETA,  # B N H
    INITIAL_STATE,  # B H D E
    FINAL_STATE,  # B H D E
    STATES,  # B H (NUM_BLOCK_N // STATE_STRIDE) D E
    USE_Q: tl.constexpr,
    USE_LD: tl.constexpr,
    USE_ALPHA: tl.constexpr,
    USE_BETA: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_PAD: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    REVERSE: tl.constexpr,
    SAVE_STATES: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
    NUM_BLOCK_N: tl.constexpr,
    NUM_STATES: tl.constexpr,
    MAX_BLOCK_N: tl.constexpr,
):
    tl.cdiv(D, BLOCK_D)
    off_bh = tl.program_id(0)
    off_block_e = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H

    # compute offset
    offset_qk = off_b * N * H * D + off_h * D
    offset_vo = off_b * N * H * E + off_h * E
    offset_ld = off_b * N * H + off_h
    offset_inv = (
        off_b * H * NUM_BLOCK_N * BLOCK_N * BLOCK_N
        + off_h * NUM_BLOCK_N * BLOCK_N * BLOCK_N
    )
    offset_state = off_b * H * D * E + off_h * D * E
    offset_states = off_bh * NUM_STATES * D * E
    offset_block_e = off_block_e * BLOCK_E

    if REVERSE:
        off_block_n = NUM_BLOCK_N - 1
        off_state = NUM_STATES - 1
        stride = -1
        offset_ld_sum = (NUM_BLOCK_N - 1) * BLOCK_N
    else:
        off_block_n = 0
        off_state = 0
        stride = 1
        # last block
        # offset of sum of local log decay, when not reverse, the offset is the last position of the block
        offset_ld_sum = BLOCK_N - 1
    offset_block_n = off_block_n * BLOCK_N

    # compute block ptr
    offset_block_d = 0
    if USE_Q:
        q_block_ptr = tl.make_block_ptr(
            base=Q + offset_qk,
            shape=(N, D),
            strides=(H * D, 1),
            offsets=(offset_block_n, offset_block_d),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )

    k_block_ptr = tl.make_block_ptr(
        base=K + offset_qk,
        shape=(N, D),
        strides=(H * D, 1),
        offsets=(offset_block_n, offset_block_d),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    if USE_INITIAL_STATE:
        state_block_ptr = tl.make_block_ptr(
            base=INITIAL_STATE + offset_state,
            shape=(D, E),
            strides=(E, 1),
            offsets=(offset_block_d, offset_block_e),
            block_shape=(BLOCK_D, BLOCK_E),
            order=(1, 0),
        )
        state = tl.load(
            state_block_ptr, boundary_check=(0, 1), padding_option="zero"
        ).to(tl.float32)
    else:
        state = tl.zeros((BLOCK_D, BLOCK_E), dtype=tl.float32)

    final_state_block_ptr = tl.make_block_ptr(
        base=FINAL_STATE + offset_state,
        shape=(D, E),
        strides=(E, 1),
        offsets=(offset_block_d, offset_block_e),
        block_shape=(BLOCK_D, BLOCK_E),
        order=(1, 0),
    )

    if D > BLOCK_D:
        offset_block_d = BLOCK_D
        if USE_Q:
            q1_block_ptr = tl.make_block_ptr(
                base=Q + offset_qk,
                shape=(N, D),
                strides=(H * D, 1),
                offsets=(offset_block_n, offset_block_d),
                block_shape=(BLOCK_N, BLOCK_D),
                order=(1, 0),
            )

        k1_block_ptr = tl.make_block_ptr(
            base=K + offset_qk,
            shape=(N, D),
            strides=(H * D, 1),
            offsets=(offset_block_n, offset_block_d),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )

        if USE_INITIAL_STATE:
            state1_block_ptr = tl.make_block_ptr(
                base=INITIAL_STATE + offset_state,
                shape=(D, E),
                strides=(E, 1),
                offsets=(offset_block_d, offset_block_e),
                block_shape=(BLOCK_D, BLOCK_E),
                order=(1, 0),
            )
            state1 = tl.load(
                state1_block_ptr, boundary_check=(0, 1), padding_option="zero"
            ).to(tl.float32)
        else:
            state1 = tl.zeros((BLOCK_D, BLOCK_E), dtype=tl.float32)

        final_state1_block_ptr = tl.make_block_ptr(
            base=FINAL_STATE + offset_state,
            shape=(D, E),
            strides=(E, 1),
            offsets=(offset_block_d, offset_block_e),
            block_shape=(BLOCK_D, BLOCK_E),
            order=(1, 0),
        )

    v_block_ptr = tl.make_block_ptr(
        base=V + offset_vo,
        shape=(N, E),
        strides=(H * E, 1),
        offsets=(offset_block_n, offset_block_e),
        block_shape=(BLOCK_N, BLOCK_E),
        order=(1, 0),
    )

    o_block_ptr = tl.make_block_ptr(
        base=O + offset_vo,
        shape=(N, E),
        strides=(H * E, 1),
        offsets=(offset_block_n, offset_block_e),
        block_shape=(BLOCK_N, BLOCK_E),
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

    if USE_BETA:
        beta_trans_block_ptr = tl.make_block_ptr(
            base=BETA + offset_ld,
            shape=(N, 1),
            strides=(H, 1),
            offsets=(offset_block_n, 0),
            block_shape=(BLOCK_N, 1),
            order=(1, 0),
        )

    array = offset_block_n + tl.arange(0, BLOCK_N)
    if USE_LD:
        ld_block_ptr = LOG_DECAY + offset_ld

    for j in range(NUM_BLOCK_N):
        # todo
        if REVERSE:
            inv_block_ptr = tl.make_block_ptr(
                base=INV + offset_inv + (off_block_n + j * stride) * BLOCK_N * BLOCK_N,
                shape=(BLOCK_N, BLOCK_N),
                strides=(1, BLOCK_N),
                offsets=(0, 0),
                block_shape=(BLOCK_N, BLOCK_N),
                order=(0, 1),
            )
        else:
            inv_block_ptr = tl.make_block_ptr(
                base=INV + offset_inv + (off_block_n + j * stride) * BLOCK_N * BLOCK_N,
                shape=(BLOCK_N, BLOCK_N),
                strides=(BLOCK_N, 1),
                offsets=(0, 0),
                block_shape=(BLOCK_N, BLOCK_N),
                order=(1, 0),
            )

        if SAVE_STATES:
            if REVERSE:
                if (j == 0) or (j % STATE_STRIDE == NUM_BLOCK_N % STATE_STRIDE):
                    flag = True
                else:
                    flag = False

                if NUM_BLOCK_N % STATE_STRIDE == 0:
                    m = 0
                else:
                    m = STATE_STRIDE - NUM_BLOCK_N % STATE_STRIDE

                k = (j + m) // STATE_STRIDE
            else:
                if j % STATE_STRIDE == 0:
                    flag = True
                else:
                    flag = False

                k = j // STATE_STRIDE

            if flag:
                states_block_ptr = tl.make_block_ptr(
                    base=STATES + offset_states + (off_state + k * stride) * D * E,
                    shape=(D, E),
                    strides=(E, 1),
                    offsets=(0, offset_block_e),
                    block_shape=(BLOCK_D, BLOCK_E),
                    order=(1, 0),
                )

                tl.store(
                    states_block_ptr,
                    state.to(states_block_ptr.dtype.element_ty),
                    boundary_check=(0, 1),
                )

                if D > BLOCK_D:
                    states1_block_ptr = tl.make_block_ptr(
                        base=STATES + offset_states + (off_state + k * stride) * D * E,
                        shape=(D, E),
                        strides=(E, 1),
                        offsets=(BLOCK_D, offset_block_e),
                        block_shape=(BLOCK_D, BLOCK_E),
                        order=(1, 0),
                    )

                    tl.store(
                        states1_block_ptr,
                        state1.to(states1_block_ptr.dtype.element_ty),
                        boundary_check=(0, 1),
                    )
                # tl.debug_barrier()

        k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")

        if USE_Q:
            q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        else:
            q = k

        if USE_ALPHA:
            alpha = tl.load(
                alpha_block_ptr, boundary_check=(0, 1), padding_option="zero"
            )
            q = q * alpha

        if USE_BETA:
            beta = tl.load(
                beta_trans_block_ptr, boundary_check=(0, 1), padding_option="zero"
            )
            k = k * beta

        inv = tl.load(inv_block_ptr, boundary_check=(0, 1))

        if USE_LD:
            mask = (array < N) & (array >= 0)
            log_decay = tl.load(ld_block_ptr + array * H, mask=mask, other=0.0).to(
                tl.float32
            )
            if REVERSE:
                offset_ld_sum = max(0, offset_ld_sum)
            else:
                offset_ld_sum = min(offset_ld_sum, N - 1)

            log_decay_sum = tl.load(ld_block_ptr + offset_ld_sum * H).to(tl.float32)
            log_k_decay = log_decay_sum - log_decay

            q = (q * tl.exp(log_decay[:, None])).to(q.dtype)
            k = (k * tl.exp(log_k_decay[:, None])).to(k.dtype)

        k_trans = tl.trans(k)
        p = tl.dot(q, state.to(q.dtype))

        if D > BLOCK_D:
            k1 = tl.load(k1_block_ptr, boundary_check=(0, 1), padding_option="zero")
            if USE_Q:
                q1 = tl.load(q1_block_ptr, boundary_check=(0, 1), padding_option="zero")
            else:
                q1 = k1

            if USE_ALPHA:
                q1 = q1 * alpha

            if USE_BETA:
                k1 = k1 * beta

            if USE_LD:
                q1 = (q1 * tl.exp(log_decay[:, None])).to(q1.dtype)
                k1 = (k1 * tl.exp(log_k_decay[:, None])).to(k1.dtype)
            k1_trans = tl.trans(k1)

            p1 = tl.dot(q1, state1.to(q1.dtype))
            p += p1

        o = tl.dot(inv, (v - p).to(inv.dtype)).to(q.dtype)

        if USE_LD:
            state *= tl.exp(log_decay_sum)
        state += tl.dot(k_trans, o)

        if D > BLOCK_D:
            if USE_LD:
                state1 *= tl.exp(log_decay_sum)
            state1 += tl.dot(k1_trans, o)

        tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), boundary_check=(0, 1))

        # update block ptr
        k_block_ptr = tl.advance(k_block_ptr, (BLOCK_N * stride, 0))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_N * stride, 0))
        o_block_ptr = tl.advance(o_block_ptr, (BLOCK_N * stride, 0))
        if USE_Q:
            q_block_ptr = tl.advance(q_block_ptr, (BLOCK_N * stride, 0))
        if USE_ALPHA:
            alpha_block_ptr = tl.advance(alpha_block_ptr, (BLOCK_N * stride, 0))
        if USE_BETA:
            beta_trans_block_ptr = tl.advance(
                beta_trans_block_ptr, (BLOCK_N * stride, 0)
            )
        array += stride * BLOCK_N
        offset_ld_sum += stride * BLOCK_N
        offset_block_n += stride * BLOCK_N

        if D > BLOCK_D:
            k1_block_ptr = tl.advance(k1_block_ptr, (BLOCK_N * stride, 0))
            if USE_Q:
                q1_block_ptr = tl.advance(q1_block_ptr, (BLOCK_N * stride, 0))

    tl.store(
        final_state_block_ptr,
        state.to(final_state_block_ptr.dtype.element_ty),
        boundary_check=(0, 1),
    )

    if D > BLOCK_D:
        tl.store(
            final_state1_block_ptr,
            state1.to(final_state1_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )


@triton.heuristics(
    {
        "MAX_BLOCK_N": lambda args: triton.next_power_of_2(args["N"]),
    }
)
@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
        }
    ),
    key=[
        "B",
        "MAX_BLOCK_N",
        "H",
        "D",
        "E",
        "USE_CU_SEQLENS",
    ],
)
@triton.jit
def _krcl_parallel_intra_inter(
    Q,  # B N H D
    K,  # B N H D
    V,  # B N H E
    O,  # B N H E
    ALPHA,  # B N H
    BETA,  # B N H
    STATES,  # B H L D E if not trans_states, B H L E D if trans_states
    LOG_DECAY,  # B N H
    LOG_DECAY_REVERSE,  # B N H
    X,  # B N H E
    DLOG_DECAY,  # B N H NUM_BLOCK_E
    CU_SEQLENS,  # M
    USE_Q: tl.constexpr,
    USE_LD: tl.constexpr,
    USE_ALPHA: tl.constexpr,
    USE_BETA: tl.constexpr,
    COMPUTE_DQ: tl.constexpr,
    SHARE_QK: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    REVERSE: tl.constexpr,
    TRANS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
    NUM_BLOCK_N: tl.constexpr,
    NUM_BLOCK_D: tl.constexpr,
    NUM_BLOCK_E: tl.constexpr,
    MAX_BLOCK_N: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_b = off_bh // H
    off_h = off_bh % H
    off_block_n = tl.program_id(1)
    off_block_e = tl.program_id(2)

    # compute offset
    offset_qk = off_b * N * H * D + off_h * D
    offset_vo = off_b * N * H * E + off_h * E
    offset_ld = off_b * N * H + off_h
    offset_block_n = off_block_n * BLOCK_N
    offset_block_n * H * D
    offset_block_n * H * E
    offset_block_ld = offset_block_n * H
    offset_block_e = off_block_e * BLOCK_E

    offset_state = off_bh * NUM_BLOCK_N * D * E
    offset_block_state = off_block_n * D * E

    # compute block ptr and mask
    q_block_ptr = tl.make_block_ptr(
        base=Q + offset_qk,
        shape=(N, D),
        strides=(H * D, 1),
        offsets=(offset_block_n, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    k_trans_block_ptr = tl.make_block_ptr(
        base=K + offset_qk,
        shape=(D, N),
        strides=(1, H * D),
        offsets=(0, offset_block_n),
        block_shape=(BLOCK_D, BLOCK_N),
        order=(0, 1),
    )

    v_block_ptr = tl.make_block_ptr(
        base=V + offset_vo,
        shape=(N, E),
        strides=(H * E, 1),
        offsets=(offset_block_n, offset_block_e),
        block_shape=(BLOCK_N, BLOCK_E),
        order=(1, 0),
    )

    o_block_ptr = tl.make_block_ptr(
        base=O + offset_vo,
        shape=(N, E),
        strides=(H * E, 1),
        offsets=(offset_block_n, offset_block_e),
        block_shape=(BLOCK_N, BLOCK_E),
        order=(1, 0),
    )

    if TRANS:
        state_block_ptr = tl.make_block_ptr(
            base=STATES + offset_state + offset_block_state,
            shape=(D, E),
            strides=(1, D),
            offsets=(0, offset_block_e),
            block_shape=(BLOCK_D, BLOCK_E),
            order=(0, 1),
        )
    else:
        state_block_ptr = tl.make_block_ptr(
            base=STATES + offset_state + offset_block_state,
            shape=(D, E),
            strides=(E, 1),
            offsets=(0, offset_block_e),
            block_shape=(BLOCK_D, BLOCK_E),
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

    if USE_BETA:
        beta_trans_block_ptr = tl.make_block_ptr(
            base=BETA + offset_ld,
            shape=(N, 1),
            strides=(H, 1),
            offsets=(offset_block_n, 0),
            block_shape=(BLOCK_N, 1),
            order=(1, 0),
        )

    # init
    v_ = tl.load(
        v_block_ptr, boundary_check=(0, 1), padding_option="zero"
    )  # for dalpha
    v = v_
    o = tl.zeros((BLOCK_N, BLOCK_E), dtype=tl.float32)
    if USE_ALPHA:
        alpha = tl.load(alpha_block_ptr, boundary_check=(0, 1), padding_option="zero")
    if USE_BETA:
        beta = tl.load(
            beta_trans_block_ptr, boundary_check=(0, 1), padding_option="zero"
        )

    if COMPUTE_DQ:
        if USE_BETA:
            v *= beta
    else:
        if USE_ALPHA:
            v *= alpha

    # setup decay
    array = tl.arange(0, BLOCK_N)
    mask = (offset_block_n + array) < N

    if USE_LD:
        if REVERSE:
            ld_sum_block_ptr = (
                LOG_DECAY_REVERSE + offset_ld + offset_block_ld + array * H
            )
        else:
            ld_sum_block_ptr = LOG_DECAY + offset_ld + offset_block_ld + array * H
        ld = tl.load(ld_sum_block_ptr, mask=mask, other=0.0).to(tl.float32)
        diff = ld[:, None] - ld[None, :]

        if REVERSE:  # triu
            diff = tl.where(array[:, None] < array[None, :], diff, -float("inf"))
        else:  # tril
            diff = tl.where(array[:, None] > array[None, :], diff, -float("inf"))
    else:
        if REVERSE:  # triu
            diff = tl.where(array[:, None] < array[None, :], 0, -float("inf"))
        else:  # tril
            diff = tl.where(array[:, None] > array[None, :], 0, -float("inf"))

    decay = tl.exp(diff)

    if USE_LD:
        q_decay = tl.exp(ld)

    for i in range(NUM_BLOCK_D):
        q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        k_trans = tl.load(
            k_trans_block_ptr, boundary_check=(0, 1), padding_option="zero"
        )

        score = tl.dot(q, k_trans)
        score *= decay
        o -= tl.dot(score.to(v.dtype), v)
        ##### inter start #####
        state = tl.load(state_block_ptr, boundary_check=(0, 1), padding_option="zero")
        o_ = tl.dot(q, state)
        if USE_LD:
            o_ *= q_decay[:, None]
        o -= o_
        ##### inter end #####

        q_block_ptr = tl.advance(q_block_ptr, (0, BLOCK_D))
        k_trans_block_ptr = tl.advance(k_trans_block_ptr, (BLOCK_D, 0))
        state_block_ptr = tl.advance(state_block_ptr, (BLOCK_D, 0))

    if USE_LD:
        # compute dld
        if USE_Q:
            x_block_ptr = tl.make_block_ptr(
                base=X + offset_vo,
                shape=(N, E),
                strides=(H * E, 1),
                offsets=(offset_block_n, offset_block_e),
                block_shape=(BLOCK_N, BLOCK_E),
                order=(1, 0),
            )
            x = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        else:
            x = v_

        if COMPUTE_DQ:
            if USE_ALPHA:
                x *= alpha
        else:
            if USE_BETA:
                x *= beta

        # N D -> N
        dld = tl.sum(x * o, axis=-1)

        offset_dld = off_b * N * H * NUM_BLOCK_E + off_h * NUM_BLOCK_E
        offset_block_dld = offset_block_n * H * NUM_BLOCK_E
        dld_block_ptr = (
            DLOG_DECAY
            + offset_dld
            + offset_block_dld
            + array * H * NUM_BLOCK_E
            + off_block_e
        )

        tl.store(dld_block_ptr, dld.to(dld_block_ptr.dtype.element_ty), mask=mask)

    # save o
    if COMPUTE_DQ:
        if USE_ALPHA:
            o *= alpha
    else:
        if USE_BETA:
            o *= beta

    if SHARE_QK:
        o_ = tl.load(o_block_ptr, boundary_check=(0, 1), padding_option="zero")
        o += o_

    tl.store(
        o_block_ptr,
        o.to(o_block_ptr.dtype.element_ty),
        boundary_check=(0, 1),
    )


##### dld, dalpha, dbeta #####
@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_N": [32, 64, 128],
        }
    ),
    key=["B", "N", "H", "D", "E", "USE_FINAL_STATE"],
)
@triton.jit
def _compute_dld_cumsum_kernel(
    DLD_Q,  # B N H F
    DLD_K,  # B N H F
    DLD,  # B N H F or B N H
    FINAL_STATE,  # B H D E
    DFINAL_STATE,  # B H D E
    DLD_STATE,  # B H or B H F
    ALPHA,  # B H
    BETA,  # B H
    DALPHA,  # B H
    DBETA,  # B H
    CU_SEQLENS,  # M
    SUM_OPTION: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    F: tl.constexpr,
    USE_FINAL_STATE: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_LD: tl.constexpr,
    USE_ALPHA: tl.constexpr,
    USE_BETA: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_F: tl.constexpr,
    NUM_BLOCK_D: tl.constexpr,
    NUM_BLOCK_E: tl.constexpr,
):
    NUM_BLOCK_N = triton.cdiv(N, BLOCK_N)
    # Calculate program ID and offsets
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)

    offset_b = off_b * N * H * F
    offset_h = off_h * F
    if SUM_OPTION == -1:
        offset_b_dld = off_b * N * H
        offset_h_dld = off_h
    offset_b_alpha_beta = off_b * N * H
    offset_h_alpha_beta = off_h

    # Calculate pointers for DLD_Q and DLD_K
    array_n = N - 1 - tl.arange(0, BLOCK_N)
    array_f = tl.arange(0, BLOCK_F)
    mask_f = array_f < F

    # If using final_state, calculate additional term
    if USE_FINAL_STATE:
        if SUM_OPTION == -1:
            dld_state = tl.load(DLD_STATE + off_b * H + offset_h_dld).to(tl.float32)
        else:
            dld_state = tl.load(
                DLD_STATE + off_b * H * F + offset_h + tl.arange(0, BLOCK_F),
                mask=mask_f,
            ).to(tl.float32)

    if SUM_OPTION == -1:
        dld_cumsum = tl.zeros((1,), dtype=tl.float32)
    else:
        dld_cumsum = tl.zeros((BLOCK_F,), dtype=tl.float32)

    for i in range(NUM_BLOCK_N):
        dld_q_block_ptr = (
            DLD_Q + offset_b + array_n[:, None] * H * F + offset_h + array_f[None, :]
        )
        dld_k_block_ptr = (
            DLD_K + offset_b + array_n[:, None] * H * F + offset_h + array_f[None, :]
        )
        if USE_LD:
            if SUM_OPTION == -1:
                dld_block_ptr = DLD + offset_b_dld + array_n[:, None] * H + offset_h_dld
            else:
                dld_block_ptr = (
                    DLD
                    + offset_b
                    + array_n[:, None] * H * F
                    + offset_h
                    + array_f[None, :]
                )
        mask_n = array_n >= 0
        mask = mask_n[:, None] & mask_f[None, :]

        # Load values from DLD_Q and DLD_K
        dld_q = tl.load(dld_q_block_ptr, mask=mask, other=0.0).to(tl.float32)
        dld_k = tl.load(dld_k_block_ptr, mask=mask, other=0.0).to(tl.float32)
        dld = dld_q - dld_k

        if USE_ALPHA:
            alpha_block_ptr = (
                ALPHA + offset_b_alpha_beta + array_n * H + offset_h_alpha_beta
            )
            alpha = tl.load(alpha_block_ptr, mask=mask_n, other=0.0).to(tl.float32)

            dalpha_block_ptr = (
                DALPHA + offset_b_alpha_beta + array_n * H + offset_h_alpha_beta
            )
            dalpha = tl.sum(dld_q / alpha[:, None], axis=-1)
            tl.store(
                dalpha_block_ptr,
                dalpha.to(dalpha_block_ptr.dtype.element_ty),
                mask=mask_n,
            )

        if USE_BETA:
            beta_block_ptr = (
                BETA + offset_b_alpha_beta + array_n * H + offset_h_alpha_beta
            )
            beta = tl.load(beta_block_ptr, mask=mask_n, other=0.0).to(tl.float32)

            dbeta_block_ptr = (
                DBETA + offset_b_alpha_beta + array_n * H + offset_h_alpha_beta
            )
            dbeta = tl.sum(dld_k / beta[:, None], axis=-1)
            tl.store(
                dbeta_block_ptr, dbeta.to(dbeta_block_ptr.dtype.element_ty), mask=mask_n
            )

        if USE_LD:
            if SUM_OPTION == -1:
                # BLOCK_N, BLOCK_F -> BLOCK_N, 1
                dld_ = tl.sum(dld, axis=-1, keep_dims=True)
                # BLOCK_N, 1 -> BLOCK_N, 1
                dld__ = tl.cumsum(dld_, axis=0) + dld_cumsum
                # BLOCK_N, 1 -> 1
                dld_cumsum += tl.sum(dld_, axis=0)

                if USE_FINAL_STATE:
                    dld__ += dld_state

                # Store result
                tl.store(
                    dld_block_ptr,
                    dld__.to(dld_block_ptr.dtype.element_ty),
                    mask=mask_n[:, None],
                )
            else:
                dld_ = tl.cumsum(dld, axis=0) + dld_cumsum
                dld_cumsum += tl.sum(dld, axis=0)

                if USE_FINAL_STATE:
                    dld_ += dld_state

                # Store result
                tl.store(
                    dld_block_ptr, dld_.to(dld_block_ptr.dtype.element_ty), mask=mask
                )

        array_n -= BLOCK_N
