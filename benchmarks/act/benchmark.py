import os

import numpy as np
import torch
import triton

from xopes.ops.act import act_torch, act_triton
from xopes.utils import get_memory

b, h, n, d = 12, 12, 8192, 128
dim = None
device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

module_map = {
    "triton": act_triton,
    "torch": act_torch,
}

configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(8, 16)],
        xlabel="Sequence Length",
        ylabel="Execution Time(ms)",
        line_arg="provider",
        line_vals=[
            "triton",
            "torch",
        ],
        line_names=[
            "Triton",
            "Torch",
        ],
        styles=[
            ("red", "-"),
            ("orange", "-"),
            ("green", "-"),
            ("blue", "-"),
            ("black", "-"),
        ],
        plot_name=f"act-{bench_type}-{mode}-batch{b}-head{h}-dim{d}-act_{act}-{dtype_name}",
        args={
            "b": b,
            "h": h,
            "d": d,
            "act": act,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
            "bench_type": bench_type,
        },
    )
    for mode in ["fwd", "bwd"]
    for dtype_name in ["bf16"]
    for bench_type in ["speed", "memory"]
    for act in ["relu", "sigmoid", "silu", "none"]
]


@triton.testing.perf_report(configs)
def benchmark(b, h, n, d, act, dtype, device, mode, provider, bench_type="speed"):
    torch.manual_seed(2024)
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    x = torch.randn((b, h, n, d), dtype=dtype, device=device).requires_grad_()

    module = module_map[provider]

    fn = lambda: module(x, act, dim)
    if mode == "bwd":
        o = fn()
        do = torch.randn((b, h, n, d), dtype=dtype, device=device)
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


save_path = "stat/act"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
