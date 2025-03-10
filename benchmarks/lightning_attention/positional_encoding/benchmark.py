import os

import numpy as np
import torch
import torch.nn.functional as F
import triton

from xopes.ops.lightning_attn.positional_encoding import (
    lape_parallel_triton,
    lape_recurrence_triton,
)
from xopes.utils import get_memory

device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def lape_recurrence(q, k, v, ld):
    return lape_recurrence_triton(q, k, v, ld=ld)[0]


def lape_parallel(q, k, v, ld):
    return lape_parallel_triton(q, k, v, ld=ld)[0]


module_map = {
    "lape_r": lape_recurrence,
    "lape_p": lape_parallel,
}

configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(8, 16)],
        xlabel="Sequence Length",
        ylabel="Execution Time(ms)" if bench_type == "speed" else "Memory Usage(MB)",
        line_arg="provider",
        line_vals=[
            "lape_r",
            "lape_p",
        ],
        line_names=[
            "LAPE_Recurrence",
            "LAPE_Parallel",
        ],
        styles=[
            ("red", "-"),
            ("blue", "-"),
            ("green", "-"),
        ],
        plot_name=f"lape-{bench_type}-{mode}-batch{b}-head{h}-dim{d}-{dtype_name}",
        args={
            "b": b,
            "h": h,
            "d": d,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
            "bench_type": bench_type,
        },
    )
    for bench_type in [
        "speed",
        "memory",
    ]
    for mode in [
        "fwd",
        "bwd",
    ]
    for dtype_name in ["bf16"]
    for b, h, d in [[4, 32, 128]]
]


@triton.testing.perf_report(configs)
def benchmark(
    b,
    n,
    h,
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

    # Generate input tensors
    q = torch.randn((h, d), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((h, d), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()
    ld = F.logsigmoid(torch.randn(h, dtype=dtype, device=device)).requires_grad_()

    module = module_map[provider]

    try:
        fn = lambda: module(q, k, v, ld=ld)
    except Exception as e:
        print(f"Error setting up {provider}: {e}")
        fn = None

    if mode == "bwd":
        try:
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        except Exception as e:
            print(f"Error in backward benchmark for {provider}: {e}")
            fn = None

    if bench_type == "speed":
        try:
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        except Exception as e:
            print(f"Error in speed benchmark for {provider}: {e}")
            ms = -1

        return ms
    else:  # memory benchmark
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


save_path = "stat/lape"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
