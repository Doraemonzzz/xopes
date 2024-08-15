import os

import numpy as np
import torch
import triton
from lightning_attn.utils import get_memory

from xopes.ops import (
    logcumsumexp_block_recurrence_triton,
    logcumsumexp_recurrence_triton,
    logcumsumexp_torch,
)

b, n, d = 12, 8192, 2048
device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

##### speed benchmark

speed_configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(9, 16)],
        xlabel="Sequence Length",
        ylabel="Execution Time(ms)",
        line_arg="provider",
        line_vals=["recurrence_triton", "block_recurrence_triton", "torch"],
        line_names=["Recurrence_Triton", "Block_Recurrence_Triton", "Torch"],
        styles=[
            ("red", "-"),
            ("orange", "-"),
            ("green", "-"),
            ("blue", "-"),
            ("black", "-"),
        ],
        plot_name=f"logcumsumexp-speed_{mode}-batch{b}-dim{d}-dtype_{dtype_name}",
        args={
            "b": b,
            "d": d,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
        },
    )
    for mode in [
        "fwd",
    ]
    for dtype_name in ["bf16"]
]


@triton.testing.perf_report(speed_configs)
def bench_speed(b, n, d, dtype, device, mode, provider, dim=-2):
    torch.manual_seed(2024)
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    x = torch.randn((b, n, d), dtype=dtype, device=device).requires_grad_()

    if provider == "recurrence_triton":
        module = logcumsumexp_recurrence_triton
    elif provider == "block_recurrence_triton":
        module = logcumsumexp_block_recurrence_triton
    else:
        module = logcumsumexp_torch

    fn = lambda: module(x)
    if mode == "bwd":
        y = fn()
        dy = torch.randn((b, n, d), dtype=dtype, device=device)
        fn = lambda: y.backward(dy, retain_graph=True)

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    return ms


##### memory benchmark
memory_configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(9, 16)],
        xlabel="Sequence Length",
        ylabel="Memory(mb)",
        line_arg="provider",
        line_vals=["recurrence_triton", "block_recurrence_triton", "torch"],
        line_names=["Recurrence_Triton", "Block_Recurrence_Triton", "Torch"],
        styles=[
            ("red", "-"),
            ("orange", "-"),
            ("green", "-"),
            ("blue", "-"),
            ("black", "-"),
        ],
        plot_name=f"logcumsumexp-memory_{mode}-batch{b}-dim{d}-dtype_{dtype_name}",
        args={
            "b": b,
            "d": d,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
        },
    )
    for mode in [
        "fwd",
    ]
    for dtype_name in ["bf16"]
]


@triton.testing.perf_report(memory_configs)
def bench_memory(b, n, d, dtype, device, mode, provider):
    torch.manual_seed(2024)
    assert mode in ["fwd", "bwd"]
    rep = 20
    x = torch.randn((b, n, d), dtype=dtype, device=device).requires_grad_()

    if provider == "recurrence_triton":
        module = logcumsumexp_recurrence_triton
    elif provider == "block_recurrence_triton":
        module = logcumsumexp_block_recurrence_triton
    else:
        module = logcumsumexp_torch

    fn = lambda: module(x)
    if mode == "bwd":
        y = fn()
        dy = torch.randn((b, n, d), dtype=dtype, device=device)
        fn = lambda: y.backward(dy, retain_graph=True)

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
bench_speed.run(save_path=save_path, print_data=True)
bench_memory.run(save_path=save_path, print_data=True)
