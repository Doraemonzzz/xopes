import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import triton

from xopes.ops.lightning_attn.vector_decay.lavd_parallel_triton import (
    lavd_parallel_intra,
    lavd_parallel_sub_intra,
    lavd_parallel_sub_intra_sep,
)
from xopes.utils import get_memory

device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def lavd_sub_intra_dk(q, k, v, ldk, ldv):
    return lavd_parallel_sub_intra(q, k, v, ldk=ldk)


def lavd_sub_intra_sep_dk(q, k, v, ldk, ldv):
    return lavd_parallel_sub_intra_sep(q, k, v, ldk=ldk)


def lavd_intra_dk(q, k, v, ldk, ldv):
    return lavd_parallel_intra(q, k, v, ldk=ldk)


def lavd_sub_intra_dk_dv(q, k, v, ldk, ldv):
    return lavd_parallel_sub_intra(q, k, v, ldk=ldk, ldv=ldv)


def lavd_sub_intra_sep_dk_dv(q, k, v, ldk, ldv):
    return lavd_parallel_sub_intra_sep(q, k, v, ldk=ldk, ldv=ldv)


def lavd_intra_dk_dv(q, k, v, ldk, ldv):
    return lavd_parallel_intra(q, k, v, ldk=ldk, ldv=ldv)


module_map = {
    "lavd_sub_intra_dk": lavd_sub_intra_dk,
    "lavd_sub_intra_sep_dk": lavd_sub_intra_sep_dk,
    "lavd_intra_dk": lavd_intra_dk,
    "lavd_sub_intra_dk_dv": lavd_sub_intra_dk_dv,
    "lavd_sub_intra_sep_dk_dv": lavd_sub_intra_sep_dk_dv,
    "lavd_intra_dk_dv": lavd_intra_dk_dv,
}

configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(8, 14)],  # Sequence lengths from 256 to 8192
        xlabel="Sequence Length",
        ylabel="Execution Time(ms)",
        line_arg="provider",
        line_vals=[
            "lavd_sub_intra_dk",
            "lavd_sub_intra_sep_dk",
            "lavd_intra_dk",
            "lavd_sub_intra_dk_dv",
            "lavd_sub_intra_sep_dk_dv",
            "lavd_intra_dk_dv",
        ],
        line_names=[
            "LAVD_S1_DK",
            "LAVD_S2_DK",
            "LAVD_DK",
            "LAVD_S1_DK_DV",
            "LAVD_S2_DK_DV",
            "LAVD_DK_DV",
        ],
        styles=[
            ("red", "-"),
            ("blue", "-"),
            ("green", "-"),
            ("orange", "-"),
            ("purple", "-"),
            ("yellow", "-"),
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
    for bench_type in [
        "speed",
    ]
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


start_time = time.time()
save_path = "stat/lavd/sub_intra"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
end_time = time.time()
total_time = end_time - start_time
print(f"Total time: {total_time} seconds")
