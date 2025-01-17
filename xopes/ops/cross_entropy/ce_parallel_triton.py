import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs


@triton.autotune(
    generate_configs(
        {
            "num_warps": [1, 2, 4, 8, 16, 32],
        }
    ),
    key=["B", "V"],
)
@triton.jit
def _ce_fwd_parallel(
    Z,  # B V
    Y,  # B
    LSE,  # B G
    S,  # B G
    ZK,  # B
    IGNORE_INDEX: tl.constexpr,
    LABEL_SMOOTHING: tl.constexpr,
    USE_LABEL_SMOOTHING: tl.constexpr,
    B: tl.constexpr,
    V: tl.constexpr,
    G: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_g = tl.program_id(1)
    # compute offset
    offset_z = off_b * V + off_g * BLOCK_V
    offset_ls = off_b * G + off_g
    offset_y = off_b
    # compute block ptr
    zy_block_ptr = Z + offset_z
    z_block_ptr = zy_block_ptr + tl.arange(0, BLOCK_V)
    y_block_ptr = (
        Y + offset_y
    )  # since we need to use y as a scalar, we don't need to use block_ptr
    lse_block_ptr = LSE + offset_ls + tl.arange(0, 1)
    if USE_LABEL_SMOOTHING:
        s_block_ptr = S + offset_ls + tl.arange(0, 1)
    zk_block_ptr = ZK + off_b
    array = tl.arange(0, BLOCK_V)
    # mask
    mask = (off_g * BLOCK_V + array) < V

    # get label
    y = tl.load(y_block_ptr)

    s = tl.full([1], 0, dtype=tl.float32)

    if y == IGNORE_INDEX:
        lse = tl.full([1], -float("inf"), dtype=tl.float32)
        tl.store(lse_block_ptr, lse.to(lse_block_ptr.dtype.element_ty))

        if USE_LABEL_SMOOTHING:
            tl.store(s_block_ptr, s.to(s_block_ptr.dtype.element_ty))
    else:
        z = tl.load(z_block_ptr, mask=mask, other=-float("inf"))
        m = tl.max(z)
        lse = tl.log(tl.sum(tl.exp(z - m), keep_dims=True)) + m
        if USE_LABEL_SMOOTHING:
            z_ = tl.where(mask, z, 0.0).to(z.dtype)
            s += tl.sum(z_)
            tl.store(s_block_ptr, s.to(s_block_ptr.dtype.element_ty))

        tl.store(lse_block_ptr, lse.to(lse_block_ptr.dtype.element_ty))

        if off_g == 0:
            # !!! important: y is the index of the target, so we need to offset it by the global index
            y_offset = y - off_g * BLOCK_V
            zk = tl.load(zy_block_ptr + y_offset)
            tl.store(zk_block_ptr, zk.to(zk_block_ptr.dtype.element_ty))


@triton.autotune(
    generate_configs(
        {
            "num_warps": [1, 2, 4, 8, 16, 32],
        }
    ),
    key=["B", "V"],
)
@triton.jit
def _ce_fwd_reduce(
    O,  # B
    ZK,  # B
    LSE,  # B G
    S,  # B G
    LABEL_SMOOTHING: tl.constexpr,
    USE_LABEL_SMOOTHING: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    V: tl.constexpr,
    G: tl.constexpr,
):
    off_b = tl.program_id(0)
    # compute offset
    offset_ls = off_b * G
    offset_zo = off_b
    # compute block ptr
    lse_block_ptr = LSE + offset_ls + tl.arange(0, G)
    lse_output_block_ptr = LSE + offset_ls
    zk_block_ptr = ZK + offset_zo + tl.arange(0, 1)
    o_block_ptr = O + offset_zo + tl.arange(0, 1)
    # load
    lse = tl.load(lse_block_ptr)
    zk = tl.load(zk_block_ptr)

    # compute global lse and sum
    m = tl.max(lse)
    lse = tl.log(tl.sum(tl.exp(lse - m))) + m

    if USE_LABEL_SMOOTHING:
        s_block_ptr = S + offset_ls + tl.arange(0, G)
        s_ = tl.load(s_block_ptr)
        s = tl.sum(s_)
    else:
        s = 0.0

    o = (-(1 - LABEL_SMOOTHING) * zk + lse - (LABEL_SMOOTHING / V) * s) / N
    tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty))
    tl.store(lse_output_block_ptr, lse.to(lse_output_block_ptr.dtype.element_ty))


@triton.autotune(
    generate_configs(
        {
            "num_warps": [1, 2, 4, 8, 16, 32],
        }
    ),
    key=["B", "V"],
)
@triton.jit
def _ce_bwd(
    Z,  # B V
    Y,  # B
    LSE,  # B
    DO,  # B
    DZ,  # B V
    IGNORE_INDEX: tl.constexpr,
    LABEL_SMOOTHING: tl.constexpr,
    USE_LABEL_SMOOTHING: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    V: tl.constexpr,
    G: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_g = tl.program_id(1)
    # compute offset
    offset_z = off_b * V
    offset_ylse = off_b
    # compute block ptr
    array = off_g * BLOCK_V + tl.arange(0, BLOCK_V)
    z_block_ptr = Z + offset_z + array
    y_block_ptr = Y + offset_ylse
    dz_block_ptr = DZ + offset_z + array
    lse_block_ptr = LSE + offset_ylse
    do_block_ptr = DO + offset_ylse
    # mask
    mask = array < V

    do = tl.load(do_block_ptr)
    z = tl.load(z_block_ptr, mask=mask, other=-float("inf"))
    y = tl.load(y_block_ptr)
    lse = tl.load(lse_block_ptr)
    p = tl.exp(z - lse)
    c = -LABEL_SMOOTHING / V
    dz = tl.where(array == y, -1 + LABEL_SMOOTHING + p + c, p + c) * do / N
    tl.store(dz_block_ptr, dz.to(dz_block_ptr.dtype.element_ty), mask=mask)


class CrossEntropyParallelTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, z, y, ignore_index=-100, reduction="mean", label_smoothing=0.0):
        b, v = z.shape

        if reduction == "mean":
            n = y.ne(ignore_index).sum().item()
        else:
            n = 1

        # TODO: tune the parameters
        MAX_BLOCK_SIZE = 65536
        BLOCK_V = min(triton.next_power_of_2(v), MAX_BLOCK_SIZE)

        g = triton.cdiv(v, BLOCK_V)
        lse = torch.empty((b, g), dtype=torch.float32, device=z.device)
        zk = torch.empty((b), dtype=torch.float32, device=z.device)
        use_label_smoothing = label_smoothing > 0
        if use_label_smoothing:
            s = torch.empty((b, g), dtype=torch.float32, device=z.device)
        else:
            s = None

        grid = (b, g)

        _ce_fwd_parallel[grid](
            Z=z,
            Y=y,
            LSE=lse,
            S=s,
            ZK=zk,
            IGNORE_INDEX=ignore_index,
            LABEL_SMOOTHING=label_smoothing,
            USE_LABEL_SMOOTHING=use_label_smoothing,
            B=b,
            V=v,
            G=g,
            BLOCK_V=BLOCK_V,
        )

        grid = (b,)
        _ce_fwd_reduce[grid](
            O=zk,  # use inplace operation
            ZK=zk,
            LSE=lse,
            S=s,
            LABEL_SMOOTHING=label_smoothing,
            USE_LABEL_SMOOTHING=use_label_smoothing,
            N=n,
            B=b,
            V=v,
            G=g,
        )

        # Important!!! Should use contiguous() to save the tensor
        ctx.save_for_backward(z, y, lse[:, 0].contiguous())
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        ctx.label_smoothing = label_smoothing
        ctx.n = n
        ctx.MAX_BLOCK_SIZE = MAX_BLOCK_SIZE

        del lse
        if use_label_smoothing:
            del s

        return zk

    @staticmethod
    @contiguous
    def backward(ctx, do):
        z, y, lse = ctx.saved_tensors
        ignore_index = ctx.ignore_index
        ctx.reduction
        label_smoothing = ctx.label_smoothing
        n = ctx.n
        MAX_BLOCK_SIZE = ctx.MAX_BLOCK_SIZE

        b, v = z.shape
        BLOCK_V = min(triton.next_power_of_2(v), MAX_BLOCK_SIZE)
        g = triton.cdiv(v, BLOCK_V)

        # init
        use_label_smoothing = label_smoothing > 0

        grid = (b, g)

        _ce_bwd[grid](
            Z=z,
            Y=y,
            LSE=lse,
            DO=do,
            DZ=z,  # use inplace operation
            IGNORE_INDEX=ignore_index,
            LABEL_SMOOTHING=label_smoothing,
            USE_LABEL_SMOOTHING=use_label_smoothing,
            N=n,
            B=b,
            V=v,
            G=g,
            BLOCK_V=BLOCK_V,
        )

        return z, None, None, None, None


def cross_entropy_parallel_triton(
    z: torch.Tensor,  # (b v)
    y: torch.Tensor,  # (b)
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Applies cross entropy loss using Triton, parallelized over the vocab dimension.

    Args:
        z: Input logits tensor of shape (B, V)
        y: Target indices tensor of shape (B)
        ignore_index: Index to ignore in loss calculation
        reduction: Reduction method
        label_smoothing: Label smoothing factor

    Returns:
        Cross entropy loss tensor
    """
    assert reduction in ["mean", "sum", "none"], f"Unsupported reduction: {reduction}"
    o = CrossEntropyParallelTriton.apply(z, y, ignore_index, reduction, label_smoothing)
    if reduction in ["mean", "sum"]:
        o = o.sum()
    return o


if __name__ == "__main__":
    # Test code
    b, v = 2, 1000
    dtype = torch.float32
    z = torch.randn((b, v), dtype=dtype).cuda().requires_grad_(True)
    y = torch.randint(0, v, (b,)).cuda()
    o = cross_entropy_parallel_triton(z, y)
    print(o.shape)
    (o.sum()).backward()
