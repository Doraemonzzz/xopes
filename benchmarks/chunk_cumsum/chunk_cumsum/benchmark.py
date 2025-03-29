import os
import time

import numpy as np
import torch
import triton

from xopes.ops.cumsum.baseline import chunk_local_cumsum_wrapper
from xopes.ops.cumsum.chunk_cumsum import chunk_cumsum_torch, chunk_cumsum_triton
from xopes.utils import get_memory

device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

module_map = {
    "triton": chunk_cumsum_triton,
    "torch": chunk_cumsum_torch,
    "torch_compile": torch.compile(chunk_cumsum_torch),
    "fla": chunk_local_cumsum_wrapper,
}

configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        # x_vals=[2**i for i in range(10, 16)],
        x_vals=[2**i for i in range(10, 11)],
        xlabel="Sequence Length",
        ylabel="Execution Time(ms)" if bench_type == "speed" else "Memory Usage(MB)",
        line_arg="provider",
        line_vals=[
            "triton",
            "torch",
            "torch_compile",
            "fla",
        ],
        line_names=["tr", "to", "toc", "fla"],
        styles=[
            ("red", "-"),
            ("orange", "-"),
            ("green", "-"),
            ("blue", "-"),
        ],
        plot_name=f"chunk_cumsum-{bench_type}-{mode}-batch{b}-reverse_{reverse}-{dtype_name}",
        args={
            "b": b,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
            "bench_type": bench_type,
            "reverse": reverse,
            "chunk_size": chunk_size,
            "dim": dim,
            "h": h,
        },
    )
    for reverse in [False]
    # for bench_type in ["speed", "memory"]
    for bench_type in ["speed"]
    for mode in ["fwd", "bwd"]
    for dtype_name in ["bf16"]
    for b in [4]
    for chunk_size in [128]
    for dim in [-2]
    for h in [8]
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
    chunk_size=128,
    dim=-2,
    h=8,
):
    torch.manual_seed(2024)
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100

    shape = (b, n, h)
    x = torch.randn(shape, dtype=dtype, device=device).requires_grad_()

    module = module_map[provider]

    try:
        fn = lambda: module(x, dim=dim, reverse=reverse, chunk_size=chunk_size)
    except Exception as e:
        print(f"Error setting up {provider}: {e}")
        fn = None

    if mode == "bwd":
        try:
            o = fn()
            do = torch.randn(shape, dtype=dtype, device=device)
            fn = lambda: o.backward(do, retain_graph=True)
        except Exception as e:
            print(f"Error setting up {provider}: {e}")
            fn = None

    if bench_type == "speed":
        try:
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        except Exception as e:
            print(f"Error setting up {provider}: {e}")
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
        except Exception as e:
            print(f"Error setting up {provider}: {e}")
            mb = -1

        return mb


start_time = time.time()
save_path = "stat/chunk_cumsum"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
end_time = time.time()
total_time = end_time - start_time
print(f"Total time: {total_time} seconds")
