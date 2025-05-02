import torch
import torch.nn.functional as F


def construct_lower_triangular_matrix(
    shape: tuple,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """
    Construct a lower triangular matrix (I + tril(AA^T, -1))

    Args:
        shape: the shape of the matrix A: (*, n, d)

    Returns:
        A lower triangular matrix of shape (*, n, n)
    """
    x = torch.randn(shape, dtype=dtype, device=device)
    x = F.normalize(x, p=2, dim=-1)
    A = torch.einsum("...id,...jd->...ij", x, x)
    # since A is l2 norm, (I + tril(AA^T, -1)) = tril(AA^T, 0)
    A = torch.tril(A, diagonal=0)
    return A


if __name__ == "__main__":
    shape = (4, 16)
    A = construct_lower_triangular_matrix(shape)
    print(A)
