import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs


@triton.autotune(
    generate_configs(
        {
            "num_warps": [1, 2, 4, 8],
        }
    ),
    key=["D"],
)
@triton.jit
def _householder_fwd(
    X,  # B D
    Y,  # B D
    O,  # B D
    SIGMA,  # B
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    B: tl.constexpr,
    D: tl.constexpr,
):
    off_b = tl.program_id(0)

    # compute offset
    offset = off_b * D

    # mask
    array_d = tl.arange(0, BLOCK_D)
    mask = array_d < D

    # compute block ptr
    x_block_ptr = X + offset + array_d
    y_block_ptr = Y + offset + array_d
    o_block_ptr = O + offset + array_d
    sigma_block_ptr = SIGMA + off_b

    # load
    x = tl.load(x_block_ptr, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_block_ptr, mask=mask, other=0.0).to(tl.float32)

    # normalize y
    sigma = tl.sqrt(tl.sum(y * y, axis=0) + EPS)
    y_ = y / sigma

    # compute o = x - 2 * c * y
    c = tl.sum(x * y_, axis=0)
    o = x - 2.0 * c * y_

    # store
    tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=mask)
    tl.store(sigma_block_ptr, sigma.to(sigma_block_ptr.dtype.element_ty))


@triton.jit
def _householder_bwd(
    X,  # B D
    Y,  # B D
    SIGMA,  # B
    DO,  # B D
    DX,  # B D
    DY,  # B D
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    B: tl.constexpr,
    D: tl.constexpr,
):
    off_b = tl.program_id(0)

    # compute offset
    offset = off_b * D

    # mask
    array_d = tl.arange(0, BLOCK_D)
    mask = array_d < D

    # compute block ptr
    do_block_ptr = DO + offset + array_d
    x_block_ptr = X + offset + array_d
    y_block_ptr = Y + offset + array_d
    sigma_block_ptr = SIGMA + off_b
    dx_block_ptr = DX + offset + array_d
    dy_block_ptr = DY + offset + array_d

    # load
    x = tl.load(x_block_ptr, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_block_ptr, mask=mask, other=0.0).to(tl.float32)
    sigma = tl.load(sigma_block_ptr)
    do = tl.load(do_block_ptr, mask=mask, other=0.0).to(tl.float32)

    # normalize y
    y_ = y / sigma

    # compute gradients
    c1 = tl.sum(do * y_, axis=0)
    dx = do - 2 * c1 * y_

    c2 = tl.sum(do * y_, axis=0)
    c3 = tl.sum(x * y_, axis=0)
    dy_ = -2 * c2 * x - 2 * c3 * do
    dy = 1 / sigma * (dy_ - tl.sum(dy_ * y_, axis=0) * y_)

    # store
    tl.store(dx_block_ptr, dx.to(dx_block_ptr.dtype.element_ty), mask=mask)
    tl.store(dy_block_ptr, dy.to(dy_block_ptr.dtype.element_ty), mask=mask)


class HouseholderTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, y, eps=1e-5):
        # allocate output
        o = torch.empty_like(x)
        sigma = torch.empty((x.shape[0]), dtype=torch.float32, device=x.device)

        # catch eps being too small if the tensors are fp16
        if x.dtype == torch.float16:
            eps = max(eps, 1.6e-5)

        # reshape input data into 2D tensor
        x_ = x.reshape(-1, x.shape[-1])
        y_ = y.reshape(-1, y.shape[-1])
        b, d = x_.shape

        # Less than 64KB per feature
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_D = min(MAX_FUSED_SIZE, triton.next_power_of_2(d))
        if d > BLOCK_D:
            raise RuntimeError("Householder doesn't support feature dim >= 64KB.")

        grid = (b,)
        _householder_fwd[grid](
            X=x_,
            Y=y_,
            O=o,
            SIGMA=sigma,
            EPS=eps,
            BLOCK_D=BLOCK_D,
            B=b,
            D=d,
        )

        ctx.save_for_backward(x, y, sigma)
        ctx.eps = eps

        return o.reshape_as(x)

    @staticmethod
    @contiguous
    def backward(ctx, do):
        x, y, sigma = ctx.saved_tensors
        eps = ctx.eps

        # allocate gradient tensors
        dx = torch.empty_like(x)
        dy = torch.empty_like(y)

        # reshape tensors
        do_ = do.reshape(-1, do.shape[-1])
        x_ = x.reshape(-1, x.shape[-1])
        y_ = y.reshape(-1, y.shape[-1])
        b, d = x_.shape

        # Less than 64KB per feature
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_D = min(MAX_FUSED_SIZE, triton.next_power_of_2(d))
        if d > BLOCK_D:
            raise RuntimeError("Householder doesn't support feature dim >= 64KB.")

        grid = (b,)
        _householder_bwd[grid](
            X=x_,
            Y=y_,
            SIGMA=sigma,
            DO=do_,
            DX=dx,
            DY=dy,
            EPS=eps,
            BLOCK_D=BLOCK_D,
            B=b,
            D=d,
        )

        return dx.reshape_as(x), dy.reshape_as(y), None


def householder_triton(x, y, eps=1e-5):
    """
    Applies Householder transformation using Triton.

    Args:
        x: Input tensor
        y: Direction vector

    Returns:
        Transformed tensor
    """
    return HouseholderTriton.apply(x, y, eps)


if __name__ == "__main__":
    # Test code
    b, n = 2, 512
    dtype = torch.float32
    x = torch.randn((b, n), dtype=dtype).cuda()
    y = torch.randn((b, n), dtype=dtype).cuda()
    o = householder_triton(x, y)
    print(o.shape)
