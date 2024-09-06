import torch
import torch.nn.functional as F


def tpe_torch(x, b, log_lambda):
    # x: b, n, d
    # b: d, e
    # log_lambda_: e
    b, h, n, d = x.shape
    index = torch.arange(n, device=x.device, dtype=torch.int64).unsqueeze(-1).unsqueeze(-1)

    # n, 1, e
    lambda_ = torch.exp(
        index * log_lambda.float()
    )
    # n, d
    coef = torch.einsum("... e, ... e -> ...", lambda_, .b.float())

    x_fft = torch.fft.rfft(x.float(), 2 * n, dim=dim)
    coef_fft = torch.fft.rfft(coef.float(), 2 * n, dim=dim)
    y_fft = x_fft * coef_fft
    y = torch.fft.irfft(y_fft, 2 * n, dim=self.dim)
    y = y.type_as(x)

    return y
