import os
import time

import numpy as np
import torch
import triton

from xopes.ops.logsumexp import lse_parallel_triton, lse_recurrence_triton, lse_torch
from xopes.utils import get_memory

device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

module_map = {
    "parallel_triton": lse_parallel_triton,
    "recurrence_triton": lse_recurrence_triton,
    "torch": lse_torch,
    "torch_compile": torch.compile(lse_torch),
}

configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(10, 18)],  # Sequence lengths from 1024 to 65536
        xlabel="Sequence Length",
        ylabel="Execution Time(ms)" if bench_type == "speed" else "Memory Usage(MB)",
        line_arg="provider",
        line_vals=[
            "parallel_triton",
            "recurrence_triton",
            "torch",
            "torch_compile",
        ],
        line_names=["ptr", "rtr", "to", "toc"],
        styles=[
            ("red", "-"),
            ("blue", "-"),
            ("orange", "-"),
            ("green", "-"),
        ],
        plot_name=f"logsumexp-{bench_type}-{mode}-batch{b}-dim{d}-{dtype_name}",
        args={
            "b": b,
            "d": d,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
            "bench_type": bench_type,
        },
    )
    for bench_type in ["speed", "memory"]
    for mode in ["fwd", "bwd"]
    for dtype_name in ["bf16"]
    for b in [4]
    for d in [1024]
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
    bench_type="speed",
):
    torch.manual_seed(2024)
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100

    shape = (b, d, n)
    x = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    dim = -1  # Using last dimension as in the test
    keepdim = True

    module = module_map[provider]

    try:
        fn = lambda: module(x, dim=dim, keepdim=keepdim)
    except:
        fn = None

    if mode == "bwd":
        try:
            o = fn()
            do = torch.randn_like(o)
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
save_path = "stat/logsumexp"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
end_time = time.time()
total_time = end_time - start_time
print(f"Total time: {total_time} seconds")
