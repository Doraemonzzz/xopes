import torch
import torch.nn.functional as F


def construct_lower_triangular_matrix(
    shape: tuple,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """
    Construct a lower triangular matrix tril(diag(B) * A * A ^ T, 0)

    Args:
        shape: the shape of the matrix A: (*, n, d)

    Returns:
        A lower triangular matrix of shape (*, n, n)
    """
    x = torch.randn(shape, dtype=dtype, device=device)
    x = F.normalize(x, p=2, dim=-1)
    A = torch.einsum("...id,...jd->...ij", x, x)
    B = torch.randn(list(shape[:-1]) + [1], dtype=dtype, device=device)
    A = B * A
    A = torch.tril(A, diagonal=0)
    return A


if __name__ == "__main__":
    shape = (4, 16)
    A = construct_lower_triangular_matrix(shape)
    print(A)
