import torch
import triton
import triton.language as tl

from xopes.ops.element_wise_binary_op import ewbo_fwd_fn
from xopes.utils import contiguous, generate_configs

MAX_BLOCK_SIZE = 64 * 1024


@triton.autotune(
    generate_configs(
        {
            "num_warps": [1, 2, 4, 8, 16, 32],
        }
    ),
    key=["B", "V"],
)
@triton.jit
def _ce_fwd(
    Z,  # B V
    Y,  # B
    O,  # B
    DZ,  # B V
    IGNORE_INDEX: tl.constexpr,
    LABEL_SMOOTHING: tl.constexpr,
    USE_LABEL_SMOOTHING: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    off_b = tl.program_id(0)
    # compute offset
    offset_z = off_b * V
    offset_oyn = off_b
    # compute block ptr
    zy_block_ptr = Z + offset_z
    z_block_ptr = zy_block_ptr + tl.arange(0, BLOCK_V)
    y_block_ptr = (
        Y + offset_oyn
    )  # since we need to use y as a scalar, we don't need to use block_ptr
    o_block_ptr = O + offset_oyn + tl.arange(0, 1)
    dz_block_ptr = DZ + offset_z + tl.arange(0, BLOCK_V)
    array = tl.arange(0, BLOCK_V)
    NUM_BLOCKS = tl.cdiv(V, BLOCK_V)

    # get label
    y = tl.load(y_block_ptr)

    if y == IGNORE_INDEX:
        o = tl.full([1], 0, dtype=tl.float32)
        dz = tl.full([BLOCK_V], 0, dtype=tl.float32)
        tl.store(o_block_ptr, o.to(O.dtype.element_ty))
        for i in range(NUM_BLOCKS):
            mask = array < V
            tl.store(dz_block_ptr, dz.to(DZ.dtype.element_ty), mask=mask)
            dz_block_ptr += BLOCK_V
            array += BLOCK_V
    else:
        zk = tl.load(zy_block_ptr + y + tl.arange(0, 1))
        # initialize
        m = tl.full([1], -float("inf"), dtype=tl.float32)
        sse = tl.full([1], 0, dtype=tl.float32)
        s = tl.full([1], 0, dtype=tl.float32)

        for i in range(NUM_BLOCKS):
            mask = array < V
            z = tl.load(z_block_ptr, mask=mask, other=-float("inf"))
            mi = tl.max(z)
            m_ = tl.maximum(m, mi)
            sse = tl.exp(m - m_) * sse + tl.sum(tl.exp(z - m_))
            m = m_
            if USE_LABEL_SMOOTHING:
                if i == NUM_BLOCKS - 1:
                    # update the masked value to 0
                    z_ = tl.where(mask, z, 0.0).to(z.dtype)
                else:
                    z_ = z
                s += tl.sum(z_)

            z_block_ptr += BLOCK_V
            array += BLOCK_V

        lse = tl.log(sse) + m
        o = (-(1 - LABEL_SMOOTHING) * zk + lse - (LABEL_SMOOTHING / V) * s) / N
        tl.store(o_block_ptr, o.to(O.dtype.element_ty))

        # compute gradient
        # refresh
        z_block_ptr = zy_block_ptr + tl.arange(0, BLOCK_V)
        array = tl.arange(0, BLOCK_V)
        for i in range(NUM_BLOCKS):
            mask = array < V
            z = tl.load(z_block_ptr, mask=mask, other=-float("inf"))
            p = tl.exp(z - lse)
            c = -LABEL_SMOOTHING / V
            dz = tl.where(array == y, -1 + LABEL_SMOOTHING + p + c, p + c) / N
            tl.store(dz_block_ptr, dz.to(DZ.dtype.element_ty), mask=mask)

            z_block_ptr += BLOCK_V
            dz_block_ptr += BLOCK_V
            array += BLOCK_V


class CrossEntropyTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, z, y, ignore_index=-100, reduction="mean", label_smoothing=0.0):
        b, v = z.shape

        if reduction == "mean":
            n = y.ne(ignore_index).sum().item()
        else:
            n = 1

        o = torch.empty((b,), dtype=torch.float32, device=z.device)

        MAX_BLOCK_SIZE = 64 * 1024
        BLOCK_V = min(triton.next_power_of_2(v), MAX_BLOCK_SIZE)

        grid = (b,)

        _ce_fwd[grid](
            Z=z,
            Y=y,
            O=o,
            DZ=z,  # use inplace operation
            IGNORE_INDEX=ignore_index,
            LABEL_SMOOTHING=label_smoothing,
            USE_LABEL_SMOOTHING=label_smoothing > 0,
            N=n,
            B=b,
            V=v,
            BLOCK_V=BLOCK_V,
        )

        if reduction in ["mean", "sum"]:
            o = o.sum()

        ctx.save_for_backward(z)

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        dz = ctx.saved_tensors[0]
        dz = ewbo_fwd_fn(dz, do, op="mul")

        return dz, None, None, None, None


def cross_entropy_triton(
    z: torch.Tensor,
    y: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Applies cross entropy loss using Triton.

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
    return CrossEntropyTriton.apply(z, y, ignore_index, reduction, label_smoothing)


if __name__ == "__main__":
    # Test code
    b, v = 2, 1000
    dtype = torch.float32
    z = torch.randn((b, v), dtype=dtype).cuda().requires_grad_(True)
    y = torch.randint(0, v, (b,)).cuda()
    o = cross_entropy_triton(z, y)
    print(o.shape)
    (o.sum()).backward()
