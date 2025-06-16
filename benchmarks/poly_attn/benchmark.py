import os
import time

import numpy as np
import torch
import triton

from xopes.ops.lightning_attn.baseline import flash_attn_func
from xopes.ops.poly_attn import poly_attn_chunk, poly_attn_log_torch
from xopes.utils import get_memory

device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

module_map = {
    "poly_log": poly_attn_log_torch,
    "poly_chunk": poly_attn_chunk,
    "poly_log_compile": torch.compile(poly_attn_log_torch),
    "poly_chunk_compile": torch.compile(poly_attn_chunk),
    "flash_attn": flash_attn_func,
}

# Generate configs for benchmark
configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(8, 11)],
        xlabel="Sequence Length",
        ylabel="Execution Time(ms)" if bench_type == "speed" else "Memory Usage(MB)",
        line_arg="provider",
        line_vals=[
            "poly_log",
            "poly_chunk",
            "poly_log_compile",
            "poly_chunk_compile",
            "flash_attn",
        ],
        line_names=["log", "chunk", "log_compile", "chunk_compile", "flash_attn"],
        # line_names=["log", "chunk"],
        styles=[
            ("red", "-"),
            ("blue", "-"),
            ("orange", "-"),
            ("green", "-"),
            ("purple", "-"),
        ],
        plot_name=f"poly_attn-{bench_type}-{mode}-batch{b}-head{h}-dim{d}-causal{causal}-{dtype_name}",
        args={
            "b": b,
            "h": h,
            "d": d,
            "p": 2,  # Fixed polynomial order
            "causal": causal,
            "chunk_size": 128,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
            "bench_type": bench_type,
        },
    )
    for bench_type in [
        "speed",
    ]
    for mode in ["fwd", "bwd"]
    for dtype_name in ["bf16"]
    for b in [2]
    for h in [16]
    for d in [64]
    for causal in [
        True,
    ]
]


@triton.testing.perf_report(configs)
def benchmark(
    b,
    n,
    h,
    d,
    p,
    causal,
    chunk_size,
    dtype,
    device,
    mode,
    provider,
    bench_type="speed",
):
    """
    Benchmark polynomial attention implementations.

    Args:
        b: Batch size
        n: Sequence length
        h: Number of heads
        d: Head dimension
        p: Polynomial order (fixed)
        causal: Whether to use causal attention
        chunk_size: Chunk size for chunked implementation
        dtype: Data type
        device: Device to run on
        mode: 'fwd' for forward pass, 'bwd' for backward pass
        provider: Implementation to benchmark
        bench_type: 'speed' or 'memory'
    """
    torch.manual_seed(2024)
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100

    # Set up tensors with correct shape (b, n, h, d)
    shape = (b, n, h, d)
    q = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    k = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    v = torch.randn(shape, dtype=dtype, device=device).requires_grad_()

    module = module_map[provider]

    # Define function based on provider
    try:
        if provider == "flash_attn":
            fn = lambda: flash_attn_func(q, k, v, causal=causal)
        elif provider.startswith("poly_log"):
            fn = lambda: module(q, k, v, p=p, causal=causal)
        else:  # poly_chunk implementations
            fn = lambda: module(q, k, v, p=p, chunk_size=chunk_size, causal=causal)
    except Exception as e:
        print(f"Error setting up {provider}: {e}")
        fn = None

    # Modify function for backward pass benchmarking
    if mode == "bwd":
        try:
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        except Exception as e:
            print(f"Error in backward setup for {provider}: {e}")
            fn = None

    # Benchmark speed
    if bench_type == "speed":
        try:
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        except Exception as e:
            print(f"Error in speed benchmark for {provider}: {e}")
            ms = -1

        return ms
    # Benchmark memory
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
            print(f"Error in memory benchmark for {provider}: {e}")
            mb = -1

        return mb


# Run benchmarks and save results
start_time = time.time()
save_path = "stat/poly_attn"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
end_time = time.time()
total_time = end_time - start_time
print(f"Total time: {total_time} seconds")
