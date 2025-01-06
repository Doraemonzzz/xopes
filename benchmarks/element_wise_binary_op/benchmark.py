import os

import numpy as np
import torch
import triton

from xopes.ops.element_wise_binary_op import ewbo_torch, ewbo_triton
from xopes.utils import get_memory

device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

module_map = {
    "triton": ewbo_triton,
    "torch": ewbo_torch,
    "torch_compile": torch.compile(ewbo_torch),
}

configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        # x_vals=[2**i for i in range(8, 16)],  # From 4096 to 32768
        x_vals=[2**i for i in range(8, 14)],  # From 4096 to 32768
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
        plot_name=f"ewbo-{bench_type}-{mode}-{op}-batch{b}-{dtype_name}",
        args={
            "b": b,
            "d": 1024,  # Fixed dimension
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
            "bench_type": bench_type,
            "op": op,
        },
    )
    for bench_type in ["speed", "memory"]
    for mode in ["fwd", "bwd"]
    # for mode in ["fwd"]
    for dtype_name in ["bf16"]
    for b in [128]
    for op in [
        "add",
    ]
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
    op,
    bench_type="speed",
):
    torch.manual_seed(2024)
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100

    # Generate input tensors similar to test.py
    x_shape = (b, n, d)
    y_shape = (b, n)  # Broadcasting dimension
    x = torch.randn(x_shape, dtype=dtype, device=device).requires_grad_()
    y = torch.randn(y_shape, dtype=dtype, device=device)
    y.requires_grad_()

    module = module_map[provider]

    try:
        fn = lambda: module(x, y, op)
    except:
        fn = None

    if mode == "bwd":
        try:
            o = fn()
            do = torch.randn(x_shape, dtype=dtype, device=device)
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


save_path = "stat/element_wise_binary_op"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
