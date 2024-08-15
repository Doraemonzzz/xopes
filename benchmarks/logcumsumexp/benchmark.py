import os

import numpy as np
import torch
import triton
from lightning_attn.utils import get_memory

from xopes.ops import (
    logcumsumexp_block_parallel_triton,
    logcumsumexp_block_recurrence_triton,
    logcumsumexp_recurrence_triton,
    logcumsumexp_torch,
)

b, n, d = 12, 8192, 2048
b, n, d = 1, 8192, 2048
device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

module_map = {
    "recurrence_triton": logcumsumexp_recurrence_triton,
    "block_recurrence_triton": logcumsumexp_block_recurrence_triton,
    "block_parallel_triton": logcumsumexp_block_parallel_triton,
    "torch": logcumsumexp_torch,
}

configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(9, 16)],
        xlabel="Sequence Length",
        ylabel="Execution Time(ms)",
        line_arg="provider",
        line_vals=[
            "recurrence_triton",
            "block_recurrence_triton",
            "block_parallel_triton",
            "torch",
        ],
        line_names=[
            "R_Triton",
            "BR_Triton",
            "BP_Triton",
            "Torch",
        ],
        styles=[
            ("red", "-"),
            ("orange", "-"),
            ("green", "-"),
            ("blue", "-"),
            ("black", "-"),
        ],
        plot_name=f"logcumsumexp-{bench_type}-{mode}-batch{b}-dim{d}-{dtype_name}",
        args={
            "b": b,
            "d": d,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
            "bench_type": bench_type,
        },
    )
    for mode in [
        "fwd",
    ]
    for dtype_name in ["bf16"]
    for bench_type in ["speed", "memory"]
]


@triton.testing.perf_report(configs)
def benchmark(b, n, d, dtype, device, mode, provider, dim=-2, bench_type="speed"):
    torch.manual_seed(2024)
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    x = torch.randn((b, n, d), dtype=dtype, device=device).requires_grad_()

    module = module_map[provider]

    fn = lambda: module(x)
    if mode == "bwd":
        y = fn()
        dy = torch.randn((b, n, d), dtype=dtype, device=device)
        fn = lambda: y.backward(dy, retain_graph=True)

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


save_path = "stat/logcumsumexp"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
