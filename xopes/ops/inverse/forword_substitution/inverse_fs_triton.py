import torch
import triton
import triton.language as tl

from xopes.utils import generate_configs, prod


@triton.autotune(
    generate_configs(
        {
            "num_warps": [2, 4, 8, 16, 32],
        }
    ),
    key=["B", "N"],
)
@triton.jit
def _inverse_fs_naive_triton(
    A,
    A_inv,
    B: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_b = tl.program_id(0)
    offset_b = off_b * N * N
    array_n = tl.arange(0, BLOCK_N)

    a_block_ptr = tl.make_block_ptr(
        base=A + offset_b,
        shape=(N, N),
        strides=(N, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_N),
        order=(1, 0),
    )

    # A[..., i, i]
    a_diag_block_ptr = A + offset_b

    a_inv_block_ptr = tl.make_block_ptr(
        base=A_inv + offset_b,
        shape=(N, N),
        strides=(N, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_N),
        order=(1, 0),
    )

    a = tl.load(a_block_ptr, boundary_check=(0, 1)).to(tl.float32)

    for i in range(N):
        # n 1
        index_i = array_n == i
        # n
        ai = tl.sum(tl.where(index_i[:, None], a, 0), axis=0)
        aii = tl.load(a_diag_block_ptr).to(tl.float32)

        # compute
        index_i_ = array_n < i
        a_ = tl.where(index_i_[:, None], a, 0)
        ai_ = (index_i - tl.sum(ai[:, None] * a_, axis=0)) / aii
        a = tl.where(index_i[:, None], ai_, a)

        # update block ptr
        a_diag_block_ptr += N + 1

    tl.store(a_inv_block_ptr, a.to(A.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    generate_configs(
        {
            "num_warps": [2, 4, 8, 16, 32],
        }
    ),
    key=["B", "N"],
)
@triton.jit
def _inverse_fs_loop_triton(
    A,
    A_inv,
    B: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_b = tl.program_id(0)
    offset_b = off_b * N * N
    array_n = tl.arange(0, BLOCK_N)

    a_block_ptr = tl.make_block_ptr(
        base=A + offset_b,
        shape=(N, N),
        strides=(N, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_N),
        order=(1, 0),
    )

    # A[..., i, i]
    a_diag_block_ptr = A + offset_b

    a_inv_block_ptr = tl.make_block_ptr(
        base=A_inv + offset_b,
        shape=(N, N),
        strides=(N, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_N),
        order=(1, 0),
    )

    a = tl.load(a_block_ptr, boundary_check=(0, 1)).to(tl.float32)

    for i in range(N):
        # n 1
        index_i = array_n == i
        # n
        ai = tl.sum(tl.where(index_i[:, None], a, 0), axis=0)
        aii = tl.load(a_diag_block_ptr).to(tl.float32)

        ai_ = (index_i).to(tl.float32)
        for j in range(i):
            # compute
            index_j = array_n == j
            aj = tl.sum(tl.where(index_j[:, None], a, 0), axis=0)
            coef = tl.sum(ai * index_j)
            ai_ -= coef * aj
        ai_ /= aii

        a = tl.where(index_i[:, None], ai_, a)

        # update block ptr
        a_diag_block_ptr += N + 1

    tl.store(a_inv_block_ptr, a.to(A.dtype.element_ty), boundary_check=(0, 1))


def inverse_fs_triton(A: torch.Tensor, op_type: int = 0) -> torch.Tensor:
    """
    Compute the inverse of a lower triangular matrix using forward substitution.

    Args:
        A: lower triangular matrix of shape (*, n, n)
        op_type: 0 for naive, 1 for optimized

    Returns:
        A_inv: inverse of A of shape (*, n, n)
    """
    n = A.shape[-1]
    b = prod(A.shape[:-2])
    A_inv = torch.zeros_like(A, dtype=torch.float32)

    grid = (b,)
    BLOCK_N = triton.next_power_of_2(n)

    if op_type == 0:
        fn = _inverse_fs_naive_triton
    elif op_type == 1:
        fn = _inverse_fs_loop_triton
    else:
        raise ValueError(f"op_type {op_type} not supported")

    fn[grid](
        A=A,
        A_inv=A_inv,
        B=b,
        N=n,
        BLOCK_N=BLOCK_N,
    )

    return A_inv


if __name__ == "__main__":
    from xopes.ops.inverse.utils import construct_lower_triangular_matrix

    A = construct_lower_triangular_matrix((4, 16))
    print(A)
    print(inverse_fs_triton(A))
