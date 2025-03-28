import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import triton

from xopes.ops.out_product_linear_recurrence.data_dependent_decay import (
    oplr_ddd_torch,
    oplr_ddd_triton,
)
from xopes.utils import get_memory

device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

module_map = {
    "triton": oplr_ddd_triton,
    "torch": oplr_ddd_torch,
    "torch_compile": torch.compile(oplr_ddd_torch),
}

configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(8, 12)],
        xlabel="Sequence Length",
        ylabel="Execution Time(ms)",
        line_arg="provider",
        line_vals=[
            "triton",
            "torch_compile",
        ],
        line_names=["tr", "toc"],
        styles=[
            ("orange", "-"),
            ("green", "-"),
        ],
        plot_name=f"oplr_ddd-{bench_type}-{mode}-batch{b}-dim{d}-dim{e}-{dtype_name}",
        args={
            "b": b,
            "d": d,
            "e": e,
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
    for d in [128]
    for e in [128]
]


@triton.testing.perf_report(configs)
def benchmark(
    b,
    n,
    d,
    e,
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

    xk = torch.randn((b, n, d), dtype=dtype, device=device).requires_grad_()
    xv = torch.randn((b, n, e), dtype=dtype, device=device).requires_grad_()
    log_decay = F.logsigmoid(
        torch.randn((b, n, d), dtype=dtype, device=device)
    ).requires_grad_()

    module = module_map[provider]

    try:
        fn = lambda: module(xk, xv, log_decay)
    except Exception as e:
        print(f"Error setting up {provider}: {e}")
        fn = None

    if mode == "bwd":
        try:
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        except Exception as e:
            print(f"Error in bwd benchmark for {provider}: {e}")
            fn = None

    if bench_type == "speed":
        try:
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        except Exception as e:
            print(f"Error in speed benchmark for {provider}: {e}")
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
            print(f"Error in memory benchmark for {provider}: {e}")
            mb = -1

        return mb


start_time = time.time()
save_path = "stat/oplr_data_dependent_decay"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
end_time = time.time()
total_time = end_time - start_time
print(f"Total time: {total_time} seconds")
