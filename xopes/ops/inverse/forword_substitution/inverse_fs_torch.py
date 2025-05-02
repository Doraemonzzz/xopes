import torch


def inverse_fs_torch(A: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse of a lower triangular matrix using forward substitution.

    Args:
        A: lower triangular matrix of shape (*, n, n)

    Returns:
        A_inv: inverse of A of shape (*, n, n)
    """
    n = A.shape[-1]
    l = len(A.shape) - 2
    array = torch.arange(n, device=A.device)
    aii = A[..., 0, 0].unsqueeze(-1)
    for _ in range(l):
        array = array.unsqueeze(0)
    one_hot = (array == 0).to(A.dtype)
    A[..., 0, :] = one_hot / aii
    for i in range(1, n):
        aii = A[..., i, i].unsqueeze(-1)
        # *, i - 1
        ai = A[..., i, :i]
        one_hot = (array == i).to(A.dtype)
        A[..., i, :] = (
            one_hot - torch.einsum("...i,...ij->...j", ai, A[..., :i, :])
        ) / aii

    return A


if __name__ == "__main__":
    from xopes.ops.inverse.utils import construct_lower_triangular_matrix

    A = construct_lower_triangular_matrix((4, 16))
    print(inverse_fs_torch(A))
