import os

import numpy as np
import torch
import triton

from xopes.ops.lrpe.cosine._1d import (
    lrpe_cosine_1d_bp_triton,
    lrpe_cosine_1d_sp_triton,
    lrpe_cosine_1d_torch,
)
from xopes.utils import get_memory

b, h, n, d = 12, 12, 8192, 128
# b, h, n, d = 12, 12, 8192, 64
device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

module_map = {
    "triton_sp": lrpe_cosine_1d_sp_triton,
    "triton_bp": lrpe_cosine_1d_bp_triton,
    "torch": lrpe_cosine_1d_torch,
    "torch_compile": torch.compile(lrpe_cosine_1d_torch),
}

configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(8, 16)],
        xlabel="Sequence Length",
        ylabel="Execution Time(ms)",
        line_arg="provider",
        line_vals=[
            "triton_sp",
            "triton_bp",
            "torch",
            "torch_compile",
        ],
        line_names=["Triton Sp", "Triton Bp", "Torch", "Torch C"],
        styles=[
            ("red", "-"),
            ("orange", "-"),
            ("green", "-"),
            ("blue", "-"),
            ("black", "-"),
        ],
        plot_name=f"lrpe_cosine-{bench_type}-{mode}-batch{b}-head{h}-dim{d}-act_{act}-dim_{dim}-{dtype_name}"
        if dim is not None
        else f"lrpe_cosine-{bench_type}-{mode}-batch{b}-head{h}-dim{d}-act_{act}-{dtype_name}",
        args={
            "b": b,
            "h": h,
            "d": d,
            "act": act,
            "dim": dim,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
            "bench_type": bench_type,
        },
    )
    for mode in ["fwd", "bwd"]
    for dtype_name in ["bf16"]
    for bench_type in ["speed", "memory"]
    # witout dim
    for act in ["silu", "none"]
    for dim in [None]
    # with dim
    # for act in ["softmax"]
    # for dim in [-1, -2]
]


@triton.testing.perf_report(configs)
def benchmark(b, h, n, d, act, dim, dtype, device, mode, provider, bench_type="speed"):
    torch.manual_seed(2024)
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    x = torch.randn((b, h, n, d), dtype=dtype, device=device).requires_grad_()
    theta = torch.randn((h, d), dtype=dtype, device=device)

    module = module_map[provider]

    fn = lambda: module(x, theta, act=act, dim=dim)
    if mode == "bwd":
        o = fn()
        do = torch.randn((b, h, n, 2 * d), dtype=dtype, device=device)
        fn = lambda: o.backward(do, retain_graph=True)

    if bench_type == "speed":
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

        return ms
    else:
        rep = 20
        try:
            torch.cuda.reset_peak_memory_stats(device)
            mb_arr = []
            for _ in range(rep):
                fn()
                mb_arr.append(get_memory(device))
            mb = np.mean(mb_arr)
        except:
            mb = -1

        return mb


save_path = "stat/lrpe_fa_cosine"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
