import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import triton
from einops import rearrange

from xopes.ops.lightning_attn.baseline import (
    chunk_gla_wrapper,
    chunk_simple_gla_wrapper,
    flash_attn_wrapper,
    lightning_attn_wrapper,
)
from xopes.ops.lightning_attn.constant_decay import (
    lacd_parallel_triton,
    lacd_recurrence_triton,
)
from xopes.ops.lightning_attn.scalar_decay import lasd_parallel_triton
from xopes.ops.lightning_attn.simple_recurrence import lasr_recurrence_triton
from xopes.ops.lightning_attn.vector_decay import lavd_parallel_triton
from xopes.utils import get_memory

device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def chunk_gla_k(q, k, v, **kwargs):
    return chunk_gla_wrapper(q, k, v, ldk=kwargs["ldk"])[0]


def chunk_simple_gla_k(q, k, v, **kwargs):
    return chunk_simple_gla_wrapper(q, k, v, ld3=kwargs["ld3"])[0]


def lacd_recurrence(q, k, v, **kwargs):
    return lacd_recurrence_triton(q, k, v, ld=kwargs["ld"])[0]


def lacd_parallel(q, k, v, **kwargs):
    return lacd_parallel_triton(q, k, v, ld=kwargs["ld"])[0]


def land_parallel(q, k, v, **kwargs):
    return lacd_parallel_triton(q, k, v)[0]


def lasd_parallel(q, k, v, **kwargs):
    return lasd_parallel_triton(q, k, v, ld=kwargs["ld3"])[0]


def lavd_k_parallel(q, k, v, **kwargs):
    return lavd_parallel_triton(q, k, v, ldk=kwargs["ldk"])[0]


def lavd_kv_parallel(q, k, v, **kwargs):
    return lavd_parallel_triton(q, k, v, ldk=kwargs["ldk"], ldv=kwargs["ldv"])[0]


def lasr_recurrence(q, k, v, **kwargs):
    return lasr_recurrence_triton(q, k, v, ld=kwargs["ldk"])[0]


def lightning_parallel(q, k, v, **kwargs):
    return lightning_attn_wrapper(q, k, v, ld=kwargs["ld"], variant="parallel")


def lightning_chunk_loop(q, k, v, **kwargs):
    return lightning_attn_wrapper(q, k, v, ld=kwargs["ld"], variant="chunk_loop")


module_map = {
    "lacd_r": lacd_recurrence,
    "lacd_p": lacd_parallel,
    "land_p": land_parallel,
    "lacd_pl": lacd_parallel,
    "lasd_p": lasd_parallel,
    "lavd_k_p": lavd_k_parallel,
    "lavd_kv_p": lavd_kv_parallel,
    "lasr_r": lasr_recurrence,
    "flash": flash_attn_wrapper,
    "lightning_p": lightning_parallel,
    "lightning_c": lightning_chunk_loop,
    "gla_k": chunk_gla_k,
    "gla_s_k": chunk_simple_gla_k,
}

configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(8, 16)],
        xlabel="Sequence Length",
        ylabel="Execution Time(ms)",
        line_arg="provider",
        line_vals=[
            "lacd_r",
            "lacd_p",
            "land_p",
            "lacd_pl",
            "lasd_p",
            "lavd_k_p",
            "lavd_kv_p",
            "lasr_r",
            "flash",
            "lightning_p",
            "lightning_c",
            "gla_k",
            "gla_s_k",
        ],
        line_names=[
            "LACD_R",
            "LACD_P",
            "LAND_P",
            "LACD_PL",
            "LASD_P",
            "LAVD_K_P",
            "LAVD_KV_P",
            "LASR_R",
            "Flash",
            "LP",
            "LC",
            "GLA_K",
            "GLA_S_K",
        ],
        styles=[
            ("red", "-"),
            ("blue", "-"),
            ("green", "-"),
            ("orange", "-"),
            ("purple", "-"),
            ("pink", "-"),
            ("yellow", "-"),
            ("cyan", "-"),
            ("brown", "-"),
            ("magenta", "-"),
            ("gray", "-"),
            ("lime", "-"),
            ("olive", "-"),
            ("teal", "-"),
            ("black", "-"),
        ],
        plot_name=f"la-{bench_type}-{mode}-batch{b}-head{h}-dim{d}-{dtype_name}",
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
        # "memory",
    ]
    for mode in [
        "fwd",
        "bwd",
    ]
    for dtype_name in ["bf16"]
    # for b, h, d in [[4, 32, 128], [1, 16, 128]]
    for b, h, d in [[4, 32, 128]]
    # for b, h, d in [[1, 16, 128]]
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
    if provider == "lacd_pl":
        ld = F.logsigmoid(torch.randn(h, dtype=dtype, device=device)).requires_grad_()
    else:
        ld = F.logsigmoid(torch.randn(h, dtype=dtype, device=device))
    ld3 = F.logsigmoid(
        torch.randn((b, n, h), dtype=dtype, device=device)
    ).requires_grad_()
    ldk = F.logsigmoid(
        torch.randn((b, n, h, d), dtype=dtype, device=device)
    ).requires_grad_()
    ldv = F.logsigmoid(
        torch.randn((b, n, h, d), dtype=dtype, device=device)
    ).requires_grad_()

    module = module_map[provider]

    if provider == "lasr_r":
        q, k, v, ldk = map(
            lambda x: rearrange(x, "... h d -> ... (h d)"), (q, k, v, ldk)
        )

    try:
        fn = lambda: module(q, k, v, ld=ld, ld3=ld3, ldk=ldk, ldv=ldv)
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
        rep = 5
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
save_path = "stat/la"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
end_time = time.time()
total_time = end_time - start_time
print(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
