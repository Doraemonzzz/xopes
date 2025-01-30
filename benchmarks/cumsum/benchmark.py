import os

import numpy as np
import torch
import triton

from xopes.ops.cumsum import cumsum_torch, cumsum_triton
from xopes.utils import get_memory

device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

module_map = {
    "triton": cumsum_triton,
    "torch": cumsum_torch,
    "torch_compile": torch.compile(cumsum_torch),
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
        plot_name=f"cumsum-{bench_type}-{mode}-batch{b}-reverse_{reverse}-{dtype_name}",
        args={
            "b": b,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
            "bench_type": bench_type,
            "reverse": reverse,
        },
    )
    for reverse in [True, False]
    for bench_type in ["speed", "memory"]
    for mode in ["fwd", "bwd"]
    for dtype_name in ["bf16"]
    for b in [4096]
]


@triton.testing.perf_report(configs)
def benchmark(
    b,
    n,
    dtype,
    device,
    mode,
    provider,
    bench_type="speed",
    reverse=False,
):
    torch.manual_seed(2024)
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100

    shape = (b, n)
    x = torch.randn(shape, dtype=dtype, device=device).requires_grad_()

    module = module_map[provider]

    try:
        fn = lambda: module(x, reverse=reverse)
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


save_path = "stat/cumsum"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)