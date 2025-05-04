import torch
import triton
import triton.language as tl

from xopes.utils import MIN_BLOCK, generate_configs, prod


@triton.autotune(
    generate_configs(
        {
            "num_warps": [2, 4, 8, 16, 32],
        }
    ),
    key=["B", "N"],
)
@triton.jit
def _inverse_jacobian_naive_triton(
    A,
    A_inv,
    B: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_b = tl.program_id(0)
    offset_b = off_b * N * N
    array_n = tl.arange(0, BLOCK_N)
    mask_n = array_n < N

    a_block_ptr = tl.make_block_ptr(
        base=A + offset_b,
        shape=(N, N),
        strides=(N, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_N),
        order=(1, 0),
    )

    # A[..., i, i]
    a_diag_block_ptr = A + offset_b + (N + 1) * tl.arange(0, BLOCK_N)

    a_inv_block_ptr = tl.make_block_ptr(
        base=A_inv + offset_b,
        shape=(N, N),
        strides=(N, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_N),
        order=(1, 0),
    )

    a = tl.load(a_block_ptr, boundary_check=(0, 1)).to(tl.float32)
    a_diag = tl.load(a_diag_block_ptr, mask=mask_n, other=1).to(tl.float32)[:, None]
    eye = array_n[:, None] == array_n[None, :]

    a_inv = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)

    for i in range(N):
        # a_inv = (-tril(a) * a_inv + eye) / diag = (-a * a_inv + a_diag * a_inv + eye) / diag = (-a * a_inv + eye) / diag + a_inv
        a_inv = (-tl.dot(a, a_inv) + eye) / a_diag + a_inv

    tl.store(a_inv_block_ptr, a_inv, boundary_check=(0, 1))


@triton.heuristics(
    {
        "BLOCK_N1": lambda args: max(MIN_BLOCK, args["BLOCK_N"] // 2),
        "BLOCK_N2": lambda args: max(MIN_BLOCK, args["BLOCK_N"] // 4),
    }
)
@triton.autotune(
    generate_configs(
        {
            "num_warps": [2, 4, 8, 16, 32],
            "BLOCK_NUM": [1, 2, 4],
        }
    ),
    key=["B", "N"],
)
@triton.jit
def _inverse_jacobian_triton(
    A,
    A_inv,
    B: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLOCK_NUM: tl.constexpr,
):
    off_b = tl.program_id(0)
    offset_b = off_b * N * N

    if BLOCK_NUM == 1:
        array_n = tl.arange(0, BLOCK_N)
        mask_n = array_n < N

        a_block_ptr = tl.make_block_ptr(
            base=A + offset_b,
            shape=(N, N),
            strides=(N, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_N),
            order=(1, 0),
        )

        # A[..., i, i]
        a_diag_block_ptr = A + offset_b + (N + 1) * tl.arange(0, BLOCK_N)

        a_inv_block_ptr = tl.make_block_ptr(
            base=A_inv + offset_b,
            shape=(N, N),
            strides=(N, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_N),
            order=(1, 0),
        )

        a = tl.load(a_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        a_diag = tl.load(a_diag_block_ptr, mask=mask_n, other=1).to(tl.float32)[:, None]
        eye = array_n[:, None] == array_n[None, :]

        a_inv = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)

        for i in range(N):
            # a_inv = (-tril(a) * a_inv + eye) / diag = (-a * a_inv + a_diag * a_inv + eye) / diag = (-a * a_inv + eye) / diag + a_inv
            a_inv = (-tl.dot(a, a_inv) + eye) / a_diag + a_inv

        tl.store(a_inv_block_ptr, a_inv, boundary_check=(0, 1))
    elif BLOCK_NUM == 2:
        array_n = tl.arange(0, BLOCK_N1)
        mask_n1 = array_n < N
        mask_n2 = (BLOCK_N1 + array_n) < N

        a11_block_ptr = tl.make_block_ptr(
            base=A + offset_b,
            shape=(N, N),
            strides=(N, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_N1, BLOCK_N1),
            order=(1, 0),
        )

        a21_block_ptr = tl.make_block_ptr(
            base=A + offset_b,
            shape=(N, N),
            strides=(N, 1),
            offsets=(BLOCK_N1, 0),
            block_shape=(BLOCK_N1, BLOCK_N1),
            order=(1, 0),
        )

        a22_block_ptr = tl.make_block_ptr(
            base=A + offset_b,
            shape=(N, N),
            strides=(N, 1),
            offsets=(BLOCK_N1, BLOCK_N1),
            block_shape=(BLOCK_N1, BLOCK_N1),
            order=(1, 0),
        )

        # A[..., i, i]
        a11_diag_block_ptr = A + offset_b + (N + 1) * tl.arange(0, BLOCK_N1)
        a22_diag_block_ptr = (
            A + offset_b + (N + 1) * (BLOCK_N1 + tl.arange(0, BLOCK_N1))
        )

        a11_inv_block_ptr = tl.make_block_ptr(
            base=A_inv + offset_b,
            shape=(N, N),
            strides=(N, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_N1, BLOCK_N1),
            order=(1, 0),
        )

        a21_inv_block_ptr = tl.make_block_ptr(
            base=A_inv + offset_b,
            shape=(N, N),
            strides=(N, 1),
            offsets=(BLOCK_N1, 0),
            block_shape=(BLOCK_N1, BLOCK_N1),
            order=(1, 0),
        )

        a22_inv_block_ptr = tl.make_block_ptr(
            base=A_inv + offset_b,
            shape=(N, N),
            strides=(N, 1),
            offsets=(BLOCK_N1, BLOCK_N1),
            block_shape=(BLOCK_N1, BLOCK_N1),
            order=(1, 0),
        )

        a11 = tl.load(a11_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        a21 = tl.load(a21_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        a22 = tl.load(a22_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        a11_diag = tl.load(a11_diag_block_ptr, mask=mask_n1, other=1).to(tl.float32)[
            :, None
        ]
        a22_diag = tl.load(a22_diag_block_ptr, mask=mask_n2, other=1).to(tl.float32)[
            :, None
        ]
        eye = array_n[:, None] == array_n[None, :]

        a11_inv = tl.zeros((BLOCK_N1, BLOCK_N1), dtype=tl.float32)
        a22_inv = tl.zeros((BLOCK_N1, BLOCK_N1), dtype=tl.float32)

        for i in range(BLOCK_N1):
            # a_inv = (-tril(a) * a_inv + eye) / diag = (-a * a_inv + a_diag * a_inv + eye) / diag = (-a * a_inv + eye) / diag + a_inv
            a11_inv = (-tl.dot(a11, a11_inv) + eye) / a11_diag + a11_inv
            a22_inv = (-tl.dot(a22, a22_inv) + eye) / a22_diag + a22_inv

        a21_inv = -tl.dot(a22_inv, tl.dot(a21, a11_inv))

        tl.store(a11_inv_block_ptr, a11_inv, boundary_check=(0, 1))
        tl.store(a21_inv_block_ptr, a21_inv, boundary_check=(0, 1))
        tl.store(a22_inv_block_ptr, a22_inv, boundary_check=(0, 1))
    else:
        array_n = tl.arange(0, BLOCK_N2)
        mask_n1 = array_n < N
        mask_n2 = (BLOCK_N2 + array_n) < N
        mask_n3 = (2 * BLOCK_N2 + array_n) < N
        mask_n4 = (3 * BLOCK_N2 + array_n) < N

        a11_block_ptr = tl.make_block_ptr(
            base=A + offset_b,
            shape=(N, N),
            strides=(N, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        a21_block_ptr = tl.make_block_ptr(
            base=A + offset_b,
            shape=(N, N),
            strides=(N, 1),
            offsets=(BLOCK_N2, 0),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        a22_block_ptr = tl.make_block_ptr(
            base=A + offset_b,
            shape=(N, N),
            strides=(N, 1),
            offsets=(BLOCK_N2, BLOCK_N2),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        a31_block_ptr = tl.make_block_ptr(
            base=A + offset_b,
            shape=(N, N),
            strides=(N, 1),
            offsets=(2 * BLOCK_N2, 0),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        a32_block_ptr = tl.make_block_ptr(
            base=A + offset_b,
            shape=(N, N),
            strides=(N, 1),
            offsets=(2 * BLOCK_N2, BLOCK_N2),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        a33_block_ptr = tl.make_block_ptr(
            base=A + offset_b,
            shape=(N, N),
            strides=(N, 1),
            offsets=(2 * BLOCK_N2, 2 * BLOCK_N2),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        a41_block_ptr = tl.make_block_ptr(
            base=A + offset_b,
            shape=(N, N),
            strides=(N, 1),
            offsets=(3 * BLOCK_N2, 0),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        a42_block_ptr = tl.make_block_ptr(
            base=A + offset_b,
            shape=(N, N),
            strides=(N, 1),
            offsets=(3 * BLOCK_N2, BLOCK_N2),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        a43_block_ptr = tl.make_block_ptr(
            base=A + offset_b,
            shape=(N, N),
            strides=(N, 1),
            offsets=(3 * BLOCK_N2, 2 * BLOCK_N2),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        a44_block_ptr = tl.make_block_ptr(
            base=A + offset_b,
            shape=(N, N),
            strides=(N, 1),
            offsets=(3 * BLOCK_N2, 3 * BLOCK_N2),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        # A[..., i, i]
        a11_diag_block_ptr = A + offset_b + (N + 1) * tl.arange(0, BLOCK_N2)
        a22_diag_block_ptr = (
            A + offset_b + (N + 1) * (BLOCK_N2 + tl.arange(0, BLOCK_N2))
        )
        a33_diag_block_ptr = (
            A + offset_b + (N + 1) * (2 * BLOCK_N2 + tl.arange(0, BLOCK_N2))
        )
        a44_diag_block_ptr = (
            A + offset_b + (N + 1) * (3 * BLOCK_N2 + tl.arange(0, BLOCK_N2))
        )

        a11_inv_block_ptr = tl.make_block_ptr(
            base=A_inv + offset_b,
            shape=(N, N),
            strides=(N, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        a21_inv_block_ptr = tl.make_block_ptr(
            base=A_inv + offset_b,
            shape=(N, N),
            strides=(N, 1),
            offsets=(BLOCK_N2, 0),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        a22_inv_block_ptr = tl.make_block_ptr(
            base=A_inv + offset_b,
            shape=(N, N),
            strides=(N, 1),
            offsets=(BLOCK_N2, BLOCK_N2),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        a31_inv_block_ptr = tl.make_block_ptr(
            base=A_inv + offset_b,
            shape=(N, N),
            strides=(N, 1),
            offsets=(2 * BLOCK_N2, 0),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        a32_inv_block_ptr = tl.make_block_ptr(
            base=A_inv + offset_b,
            shape=(N, N),
            strides=(N, 1),
            offsets=(2 * BLOCK_N2, BLOCK_N2),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        a33_inv_block_ptr = tl.make_block_ptr(
            base=A_inv + offset_b,
            shape=(N, N),
            strides=(N, 1),
            offsets=(2 * BLOCK_N2, 2 * BLOCK_N2),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        a41_inv_block_ptr = tl.make_block_ptr(
            base=A_inv + offset_b,
            shape=(N, N),
            strides=(N, 1),
            offsets=(3 * BLOCK_N2, 0),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        a42_inv_block_ptr = tl.make_block_ptr(
            base=A_inv + offset_b,
            shape=(N, N),
            strides=(N, 1),
            offsets=(3 * BLOCK_N2, BLOCK_N2),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        a43_inv_block_ptr = tl.make_block_ptr(
            base=A_inv + offset_b,
            shape=(N, N),
            strides=(N, 1),
            offsets=(3 * BLOCK_N2, 2 * BLOCK_N2),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        a44_inv_block_ptr = tl.make_block_ptr(
            base=A_inv + offset_b,
            shape=(N, N),
            strides=(N, 1),
            offsets=(3 * BLOCK_N2, 3 * BLOCK_N2),
            block_shape=(BLOCK_N2, BLOCK_N2),
            order=(1, 0),
        )

        a11 = tl.load(a11_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        a21 = tl.load(a21_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        a22 = tl.load(a22_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        a31 = tl.load(a31_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        a32 = tl.load(a32_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        a33 = tl.load(a33_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        a41 = tl.load(a41_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        a42 = tl.load(a42_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        a43 = tl.load(a43_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        a44 = tl.load(a44_block_ptr, boundary_check=(0, 1)).to(tl.float32)

        a11_diag = tl.load(a11_diag_block_ptr, mask=mask_n1, other=1).to(tl.float32)[
            :, None
        ]
        a22_diag = tl.load(a22_diag_block_ptr, mask=mask_n2, other=1).to(tl.float32)[
            :, None
        ]
        a33_diag = tl.load(a33_diag_block_ptr, mask=mask_n3, other=1).to(tl.float32)[
            :, None
        ]
        a44_diag = tl.load(a44_diag_block_ptr, mask=mask_n4, other=1).to(tl.float32)[
            :, None
        ]

        eye = array_n[:, None] == array_n[None, :]

        a11_inv = tl.zeros((BLOCK_N2, BLOCK_N2), dtype=tl.float32)
        a22_inv = tl.zeros((BLOCK_N2, BLOCK_N2), dtype=tl.float32)
        a33_inv = tl.zeros((BLOCK_N2, BLOCK_N2), dtype=tl.float32)
        a44_inv = tl.zeros((BLOCK_N2, BLOCK_N2), dtype=tl.float32)

        for i in range(BLOCK_N2):
            # a_inv = (-tril(a) * a_inv + eye) / diag = (-a * a_inv + a_diag * a_inv + eye) / diag = (-a * a_inv + eye) / diag + a_inv
            a11_inv = (-tl.dot(a11, a11_inv) + eye) / a11_diag + a11_inv
            a22_inv = (-tl.dot(a22, a22_inv) + eye) / a22_diag + a22_inv
            a33_inv = (-tl.dot(a33, a33_inv) + eye) / a33_diag + a33_inv
            a44_inv = (-tl.dot(a44, a44_inv) + eye) / a44_diag + a44_inv

        a21_inv = -tl.dot(a22_inv, tl.dot(a21, a11_inv))
        a32_inv = -tl.dot(a33_inv, tl.dot(a32, a22_inv))
        a31_inv = -tl.dot(a33_inv, tl.dot(a31, a11_inv) + tl.dot(a32, a21_inv))
        a43_inv = -tl.dot(a44_inv, tl.dot(a43, a33_inv))
        a42_inv = -tl.dot(a44_inv, tl.dot(a42, a22_inv) + tl.dot(a43, a32_inv))
        a41_inv = -tl.dot(
            a44_inv, tl.dot(a41, a11_inv) + tl.dot(a42, a21_inv) + tl.dot(a43, a31_inv)
        )

        tl.store(a11_inv_block_ptr, a11_inv, boundary_check=(0, 1))
        tl.store(a21_inv_block_ptr, a21_inv, boundary_check=(0, 1))
        tl.store(a22_inv_block_ptr, a22_inv, boundary_check=(0, 1))
        tl.store(a31_inv_block_ptr, a31_inv, boundary_check=(0, 1))
        tl.store(a32_inv_block_ptr, a32_inv, boundary_check=(0, 1))
        tl.store(a33_inv_block_ptr, a33_inv, boundary_check=(0, 1))
        tl.store(a41_inv_block_ptr, a41_inv, boundary_check=(0, 1))
        tl.store(a42_inv_block_ptr, a42_inv, boundary_check=(0, 1))
        tl.store(a43_inv_block_ptr, a43_inv, boundary_check=(0, 1))
        tl.store(a44_inv_block_ptr, a44_inv, boundary_check=(0, 1))


def inverse_jacobian_triton(A: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse of a lower triangular matrix using jacobian method.

    Args:
        A: lower triangular matrix of shape (*, n, n)

    Returns:
        A_inv: inverse of A of shape (*, n, n)
    """
    n = A.shape[-1]
    b = prod(A.shape[:-2])
    A_inv = torch.zeros_like(A, dtype=torch.float32)

    grid = (b,)
    BLOCK_N = max(MIN_BLOCK, triton.next_power_of_2(n))

    _inverse_jacobian_triton[grid](
        A=A,
        A_inv=A_inv,
        B=b,
        N=n,
        BLOCK_N=BLOCK_N,
    )

    return A_inv
