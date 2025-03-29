import os
import time

import numpy as np
import torch
import triton

from xopes.ops.cumsum.baseline import chunk_local_cumsum_wrapper
from xopes.ops.cumsum.chunk_cumsum import chunk_cumsum_torch
from xopes.ops.cumsum.chunk_cumsum_decay import chunk_cumsum_decay_triton
from xopes.ops.cumsum.chunk_reverse_cumsum import chunk_reverse_cumsum_torch
from xopes.utils import get_memory

device = torch.device("cuda")

# Map string names to torch dtypes
dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

# Define implementation mapping
module_map = {
    "triton": chunk_cumsum_decay_triton,
    "torch": chunk_cumsum_torch,
    "torch_compile": torch.compile(chunk_cumsum_torch),
    "torch_reverse": chunk_reverse_cumsum_torch,
    "torch_compile_reverse": torch.compile(chunk_reverse_cumsum_torch),
    "fla": chunk_local_cumsum_wrapper,
}

# Benchmark configurations
configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        # Use a wider range for detailed benchmarking
        x_vals=[2**i for i in range(10, 16)],
        # x_vals=[2**i for i in range(10, 11)],
        xlabel="Sequence Length",
        ylabel="Execution Time(ms)" if bench_type == "speed" else "Memory Usage(MB)",
        line_arg="provider",
        line_vals=[
            "triton",
            "torch",
            "torch_compile",
            "fla",
        ],
        line_names=["triton", "torch", "torch_compile", "fla"],
        styles=[
            ("red", "-"),
            ("blue", "-"),
            ("green", "-"),
            ("orange", "-"),
        ],
        plot_name=f"chunk_cumsum_decay-{bench_type}-b{b}-h{h}-d{d}-reverse_{reverse}-{dtype_name}",
        args={
            "b": b,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "bench_type": bench_type,
            "reverse": reverse,
            "chunk_size": chunk_size,
            "h": h,
            "d": d,
        },
    )
    for reverse in [False, True]
    for bench_type in ["speed", "memory"]
    for dtype_name in [
        "bf16",
    ]
    for b in [4, 16]
    # for b in [16]
    for chunk_size in [128]
    for h in [16]
    for d in [1, 128]
    # for d in [128]
]


@triton.testing.perf_report(configs)
def benchmark(
    b,
    n,
    dtype,
    device,
    provider,
    bench_type="speed",
    reverse=False,
    chunk_size=128,
    h=16,
    d=128,
):
    # Set seed for reproducibility
    torch.manual_seed(2024)
    warmup = 25
    rep = 100

    # Create input tensor
    if d == 1:
        shape = (b, n, h)
    else:
        shape = (b, n, h, d)
    x = torch.randn(shape, dtype=dtype, device=device).requires_grad_()

    # Get the appropriate module
    if reverse and "torch" in provider:
        provider = provider + "_reverse"
    module = module_map[provider]

    # Set up the function to benchmark
    try:
        fn = lambda: module(x, reverse=reverse, chunk_size=chunk_size, dim=-2)
    except Exception as e:
        print(f"Error setting up {provider}: {e}")
        return -1

    # Speed benchmark
    if bench_type == "speed":
        try:
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        except Exception as e:
            print(f"Error benchmarking {provider}: {e}")
            ms = -1

        return ms
    # Memory benchmark
    else:
        rep = 20
        try:
            torch.cuda.reset_peak_memory_stats(device)
            mb_arr = []
            for _ in range(rep):
                fn()
                mb_arr.append(get_memory(device))
            mb = np.mean(mb_arr)
        except Exception as e:
            print(f"Error measuring memory for {provider}: {e}")
            mb = -1

        return mb


# Run benchmarks
start_time = time.time()
save_path = "stat/chunk_cumsum_decay"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
end_time = time.time()
total_time = end_time - start_time
print(f"Total time: {total_time} seconds ({total_time/60:.2f} minutes)")
