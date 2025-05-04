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
def _inverse_jacobian_triton(
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


def inverse_jacobian_triton(A: torch.Tensor, op_type: int = 0) -> torch.Tensor:
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
