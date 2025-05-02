import torch


def inverse_jacobian_torch(A: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse of a lower triangular matrix using jacobian method.
    """
    n = A.shape[-1]
    array = torch.arange(n, device=A.device)
    diag = A[..., array, array].unsqueeze(-1)
    L = -torch.tril(A, diagonal=-1) / diag
    eye = torch.eye(n, device=A.device)
    b1 = eye / diag
    A_inv = b1.clone()
    for _ in range(n - 1):
        A_inv = torch.einsum("...ij,...jk->...ik", L, A_inv) + b1

    return A_inv


if __name__ == "__main__":
    from xopes.ops.inverse.utils import construct_lower_triangular_matrix

    A = construct_lower_triangular_matrix((4, 16))
    print(inverse_jacobian_torch(A))
