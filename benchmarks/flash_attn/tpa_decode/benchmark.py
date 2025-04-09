import os
import time

import numpy as np
import torch
import triton

from xopes.ops.flash_attn.tpa.tpa_decode_torch import (
    tpa_decode_naive_torch,
    tpa_decode_torch,
)
from xopes.ops.flash_attn.tpa.tpa_decode_triton import (
    tpa_decode_parallel_b_triton,
    tpa_decode_parallel_bh_triton,
    tpa_decode_parallel_bn_triton,
)
from xopes.utils import get_memory

# Default parameters for the benchmark
device = torch.device("cuda")

# Map of PyTorch datatypes
dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

# Map of implementations to benchmark
module_map = {
    "triton_b": tpa_decode_parallel_b_triton,
    "triton_bh": tpa_decode_parallel_bh_triton,
    "triton_bn": tpa_decode_parallel_bn_triton,
    "torch": tpa_decode_torch,
    "torch_compile": torch.compile(tpa_decode_torch),
    "torch_naive": tpa_decode_naive_torch,
    "torch_naive_compile": torch.compile(tpa_decode_naive_torch),
}

# Define benchmark configurations
configs = [
    triton.testing.Benchmark(
        x_names=["m"],
        x_vals=[2**i for i in range(8, 16)],
        xlabel="Sequence Length",
        ylabel="Execution Time(ms)" if bench_type == "speed" else "Memory Usage(MB)",
        line_arg="provider",
        line_vals=[
            "triton_b",
            "triton_bh",
            "triton_bn",
            "torch",
            "torch_compile",
            "torch_naive",
            "torch_naive_compile",
        ],
        line_names=[
            "trb",
            "trbh",
            "trbn",
            "to",
            "toc",
            "ton",
            "tonc",
        ],
        styles=[
            ("red", "-"),
            ("orange", "-"),
            ("blue", "-"),
            ("green", "-"),
            ("purple", "-"),
            ("yellow", "-"),
            ("cyan", "-"),
        ],
        plot_name=f"tpa_decode-{bench_type}-{mode}-batch{b}-head{h}-dim{d}-rank{r}-{dtype_name}",
        args={
            "b": b,
            "n": n,
            "h": h,
            "r": r,
            "d": d,
            "e": e,
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
    for b in [8, 32]
    for n in [1]
    for h in [32]
    for r in [16]
    for d in [128]
    for e in [128]
]


@triton.testing.perf_report(configs)
def benchmark(b, n, m, h, r, d, e, dtype, device, mode, provider, bench_type="speed"):
    torch.manual_seed(2024)
    assert mode in [
        "fwd",
    ]
    assert n == 1, "n must be 1 for decoding"

    warmup = 25
    rep = 100

    # Create tensors for benchmarking
    aq = torch.randn((b, n, h, r), dtype=dtype, device=device).requires_grad_()
    ak = torch.randn((b, m, h), dtype=dtype, device=device).requires_grad_()
    av = torch.randn((b, m, h), dtype=dtype, device=device).requires_grad_()
    bq = torch.randn((b, n, r, d), dtype=dtype, device=device).requires_grad_()
    bk = torch.randn((b, m, d), dtype=dtype, device=device).requires_grad_()
    bv = torch.randn((b, m, e), dtype=dtype, device=device).requires_grad_()

    module = module_map[provider]

    fn = lambda: module(aq, ak, av, bq, bk, bv)

    if bench_type == "speed":
        # Measure execution time
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms
    else:
        # Measure memory usage
        rep = 20
        try:
            torch.cuda.reset_peak_memory_stats(device)
            mb_arr = []
            for _ in range(rep):
                fn()
                mb_arr.append(get_memory(device))
            mb = np.mean(mb_arr)
        except Exception as e:
            print(f"Error setting up {provider}: {e}")
            mb = -1

        return mb


start_time = time.time()
save_path = "stat/tpa_decode"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
end_time = time.time()
total_time = end_time - start_time
print(f"Total time: {total_time} seconds, {total_time/60} minutes")
