import os

import numpy as np
import torch
import triton

from xopes.ops.householder import householder_torch, householder_triton
from xopes.utils import get_memory

device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

module_map = {
    "triton": householder_triton,
    "torch": householder_torch,
    "torch_compile": torch.compile(householder_torch),
}

configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(10, 16)],
        xlabel="Sequence Length",
        ylabel="Execution Time(ms)",
        line_arg="provider",
        line_vals=[
            "triton",
            "torch",
            "torch_compile",
        ],
        line_names=["tr", "to", "toc"],
        styles=[
            ("red", "-"),
            ("orange", "-"),
            ("green", "-"),
        ],
        plot_name=f"householder-{bench_type}-{mode}-batch{b}-dim{d}-{dtype_name}",
        args={
            "b": b,
            "d": d,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
            "bench_type": bench_type,
        },
    )
    for bench_type in ["speed", "memory"]
    for mode in ["fwd", "bwd"]
    for dtype_name in ["bf16"]
    for b in [4]
    for d in [2048]
]


@triton.testing.perf_report(configs)
def benchmark(
    b,
    n,
    d,
    dtype,
    device,
    mode,
    provider,
    bench_type="speed",
):
    torch.manual_seed(2024)
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100

    shape = (b, n, d)
    x = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    y = torch.randn(shape, dtype=dtype, device=device).requires_grad_()

    module = module_map[provider]

    try:
        fn = lambda: module(x, y)
    except:
        fn = None

    if mode == "bwd":
        try:
            o = fn()
            do = torch.randn(shape, dtype=dtype, device=device)
            fn = lambda: o.backward(do, retain_graph=True)
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


save_path = "stat/householder"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
