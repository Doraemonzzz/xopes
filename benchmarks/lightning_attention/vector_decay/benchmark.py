import os

import numpy as np
import torch
import torch.nn.functional as F
import triton

from xopes.ops.lightning_attn.baseline import chunk_gla_wrapper, flash_attn_wrapper
from xopes.ops.lightning_attn.vector_decay.lavd_chunk_parallel_triton import (
    lavd_chunk_parallel_triton,
)
from xopes.utils import get_memory

device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def lavd_dk(q, k, v, ldk, ldv):
    return lavd_chunk_parallel_triton(q, k, v, ldk=ldk)[0]


def lavd_dv(q, k, v, ldk, ldv):
    return lavd_chunk_parallel_triton(q, k, v, ldv=ldv)[0]


def lavd_dk_dv(q, k, v, ldk, ldv):
    return lavd_chunk_parallel_triton(q, k, v, ldk=ldk, ldv=ldv)[0]


module_map = {
    "lavd_dk": lavd_dk,
    "lavd_dv": lavd_dv,
    "lavd_dk_dv": lavd_dk_dv,
    "flash": flash_attn_wrapper,
    "gla": chunk_gla_wrapper,
}

configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        # x_vals=[2**i for i in range(8, 14)],  # Sequence lengths from 256 to 8192
        x_vals=[2**i for i in range(8, 9)],
        xlabel="Sequence Length",
        ylabel="Execution Time(ms)",
        line_arg="provider",
        line_vals=[
            "lavd_dk",
            "lavd_dv",
            "lavd_dk_dv",
            "flash",
            "gla",
        ],
        line_names=["LAVD_DK", "LAVD_DV", "LAVD_DK_DV", "Flash", "GLA"],
        styles=[
            ("red", "-"),
            ("blue", "-"),
            ("green", "-"),
            ("orange", "-"),
            ("purple", "-"),
        ],
        plot_name=f"lavd-{bench_type}-{mode}-batch{b}-head{h}-dim{d}-{dtype_name}",
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
    for bench_type in ["speed", "memory"]
    for mode in [
        "fwd",
    ]
    for dtype_name in ["bf16"]
    for b in [4]
    for h in [32]
    for d in [128]
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

    shape = (b, n, h, d)
    q = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    k = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    v = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    ldk = F.sigmoid(torch.randn(shape, dtype=dtype, device=device)).requires_grad_()
    ldv = F.sigmoid(torch.randn(shape, dtype=dtype, device=device)).requires_grad_()

    module = module_map[provider]

    try:
        fn = lambda: module(q, k, v, ldk=ldk, ldv=ldv)
    except Exception as e:
        print(f"Error setting up {provider}: {e}")
        fn = None

    if mode == "bwd":
        try:
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        except Exception as e:
            print(f"Error in speed benchmark for {provider}: {e}")
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
            print(f"Error in memory benchmark for {provider}: {e}")
            mb = -1

        return mb


save_path = "stat/lavd"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
