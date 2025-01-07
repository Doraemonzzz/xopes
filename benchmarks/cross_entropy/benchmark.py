import os

import numpy as np
import torch
import triton

from xopes.ops.cross_entropy import (
    cross_entropy_fla_wrapper,
    cross_entropy_torch,
    cross_entropy_triton,
)
from xopes.utils import get_memory

device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

module_map = {
    "triton": cross_entropy_triton,
    "fla": cross_entropy_fla_wrapper,
    "torch": cross_entropy_torch,
    "torch_compile": torch.compile(cross_entropy_torch),
}

configs = [
    triton.testing.Benchmark(
        x_names=["v"],  # vocabulary size
        x_vals=[2**i for i in range(10, 18)],
        xlabel="Vocabulary Size",
        ylabel="Execution Time(ms)",
        line_arg="provider",
        line_vals=[
            "triton",
            "fla",
            "torch",
            "torch_compile",
        ],
        line_names=["tr", "fla", "to", "toc"],
        styles=[
            ("red", "-"),
            ("orange", "-"),
            ("green", "-"),
            ("blue", "-"),
        ],
        plot_name=f"cross_entropy-{bench_type}-{mode}-batch{b}-{dtype_name}-ls{label_smoothing}-{reduction}",
        args={
            "b": b,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
            "bench_type": bench_type,
            "label_smoothing": label_smoothing,
            "reduction": reduction,
        },
    )
    for bench_type in ["speed", "memory"]
    for mode in ["fwd", "bwd"]
    for dtype_name in ["bf16"]
    for b in [4096]
    for label_smoothing in [0.0]
    for reduction in ["mean"]
]


@triton.testing.perf_report(configs)
def benchmark(
    b,
    v,
    dtype,
    device,
    mode,
    provider,
    label_smoothing,
    reduction,
    bench_type="speed",
):
    torch.manual_seed(2024)
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100

    # Generate input tensors
    z = torch.randn((b, v), dtype=dtype, device=device).requires_grad_()
    y = torch.randint(0, v, (b,), device=device)

    module = module_map[provider]

    try:
        fn = lambda: module(
            z,
            y,
            reduction=reduction,
            label_smoothing=label_smoothing,
            ignore_index=-100,
        )
    except:
        fn = None

    if mode == "bwd":
        try:
            o = fn()
            do = torch.rand_like(o)
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


save_path = "stat/cross_entropy"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
