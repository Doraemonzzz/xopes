import pytest
import torch

from xopes.ops.inverse.forword_substitution.inverse_fs_torch import inverse_fs_torch
from xopes.ops.inverse.forword_substitution.inverse_fs_triton import inverse_fs_triton
from xopes.ops.inverse.jacobian.inverse_jacobian_torch import inverse_jacobian_torch
from xopes.ops.inverse.utils import construct_lower_triangular_matrix
from xopes.utils import get_threshold


def get_params():
    # Test with different matrix sizes
    shapes = [
        (4, 16, 32, 128),
        (4, 16, 31, 128),
        (4, 16, 33, 128),
        (4, 16, 4, 128),
    ]
    return shapes


def check_result(
    A: torch.Tensor, A_inv: torch.Tensor, atol: float, rtol: float, name: str
):
    """
    Check the result of the inverse of a lower triangular matrix.
    """
    o = torch.einsum("...ij,...jk->...ik", A, A_inv)
    identity = torch.eye(A.shape[-1], device=A.device)
    if o.shape != identity.shape:
        identity = identity.expand(*o.shape[:-2], *identity.shape[-2:])

    diff = o - identity
    print(f"Ainv diff max (Vs {name}): {diff.max().item()}")
    print(f"Ainv diff norm (Vs {name}): {diff.norm().item()}")

    assert torch.allclose(o, identity, atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("dtype", [torch.float32])
def test(shape, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")

    # Create a lower triangular matrix
    A_original = construct_lower_triangular_matrix(shape, dtype=dtype, device=device)

    # Compute inverse
    A_inv_fs = inverse_fs_torch(A_original.clone())
    A_inv_fs_naive_triton = inverse_fs_triton(A_original.clone(), op_type=0)
    A_inv_jac = inverse_jacobian_torch(A_original.clone())

    print(A_original[0, 0])
    print(A_inv_fs[0, 0])
    print(A_inv_jac[0, 0])
    print(A_inv_fs_naive_triton[0, 0])

    # Get tolerance thresholds based on data type
    atol, rtol = get_threshold(dtype)

    # Check result
    check_result(A_original, A_inv_fs, atol, rtol, "torch forward substitution")
    check_result(A_original, A_inv_jac, atol, rtol, "torch jacobian")
    check_result(A_original, A_inv_fs_naive_triton, atol, rtol, "naive triton")
