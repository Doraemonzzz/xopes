import os

import numpy as np
import torch
import triton

from xopes.ops import (
    md_lrpe_cosine_cache_triton,
    md_lrpe_cosine_parallel_triton,
    md_lrpe_cosine_torch,
    md_lrpe_cosine_triton,
)
from xopes.utils import get_memory, next_power_of_two

b, h, n, d = 12, 12, 8192, 128
# b, h, n, d = 1, 12, 8192, 128
# dim = 1
# dim = 2
dim = 3
device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

module_map = {
    "triton": md_lrpe_cosine_triton,
    "triton_cache": md_lrpe_cosine_cache_triton,
    "triton_parallel": md_lrpe_cosine_parallel_triton,
    "torch": md_lrpe_cosine_torch,
    "torch_compile": torch.compile(md_lrpe_cosine_torch),
}

x_vals_map = {
    1: [2**i for i in range(8, 16)],
    2: [2**i for i in range(4, 8)],
    3: [2**i for i in range(2, 6)],
}

configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=x_vals_map[dim],
        xlabel="Sequence Length",
        ylabel="Execution Time(ms)",
        line_arg="provider",
        line_vals=[
            "triton",
            "triton_cache",
            "triton_parallel",
            "torch",
            "torch_compile",
        ],
        line_names=["Tri", "Tri Ca", "Tri P", "Tor", "Tor C"],
        styles=[
            ("red", "-"),
            ("orange", "-"),
            ("green", "-"),
            ("blue", "-"),
            ("black", "-"),
        ],
        plot_name=f"md_lrpe_cosine-{bench_type}-{mode}-batch{b}-head{h}-dim{d}-{dtype_name}-{dim}d",
        args={
            "b": b,
            "h": h,
            "d": d,
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
]


@triton.testing.perf_report(configs)
def benchmark(b, h, n, d, dim, dtype, device, mode, provider, bench_type="speed"):
    torch.manual_seed(2024)
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100

    shape = tuple([b, h] + [n] * dim + [d])
    h = shape[1]
    d = shape[-1]
    m = len(shape) - 3
    e = next_power_of_two((d + m - 1) // m)
    x = (torch.randn(shape, dtype=dtype, device=device)).requires_grad_()
    theta = torch.randn((h, e), dtype=dtype, device=device)
    shape = shape[:-1] + (shape[-1] * 2,)

    module = module_map[provider]

    try:
        fn = lambda: module(x, theta)
    except:
        fn = None

    if mode == "bwd":
        try:
            y = fn()
            dy = torch.randn(shape, dtype=dtype, device=device)
            fn = lambda: y.backward(dy, retain_graph=True)
        except:
            fn = None

    if bench_type == "speed":
        try:
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        except:
            ms = -1

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


save_path = "stat/md_lrpe_cosine"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
