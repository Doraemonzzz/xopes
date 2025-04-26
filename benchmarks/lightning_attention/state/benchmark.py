import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import triton

from xopes.ops.lightning_attn.baseline import state_fla_wrapper
from xopes.ops.lightning_attn.vector_decay.lavd_parallel_triton import (
    lavd_parallel_state_parallel_reduce,
    lavd_parallel_state_parallel_reduce_sep,
)
from xopes.utils import get_memory

device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def state_gla_k(k, v, **kwargs):
    return state_fla_wrapper(
        k=k,
        v=v,
        ldk=kwargs["ldk"],
        chunk_size=kwargs["chunk_size"],
    )[0]


def lavd_k_state_chunk_loop(k, v, **kwargs):
    b, n, h, d = k.shape
    e = v.shape[-1]
    return lavd_parallel_state_parallel_reduce(
        b=b,
        n=n,
        h=h,
        d=d,
        e=e,
        k=k,
        v=v,
        ldk=kwargs["ldk"],
        BLOCK_N=kwargs["chunk_size"],
    )


def lavd_kv_state_chunk_loop(k, v, **kwargs):
    b, n, h, d = k.shape
    e = v.shape[-1]
    return lavd_parallel_state_parallel_reduce(
        b=b,
        n=n,
        h=h,
        d=d,
        e=e,
        k=k,
        v=v,
        ldk=kwargs["ldk"],
        ldv=kwargs["ldv"],
        BLOCK_N=kwargs["chunk_size"],
    )


def lavd_k_state_chunk_parallel(k, v, **kwargs):
    b, n, h, d = k.shape
    e = v.shape[-1]
    return lavd_parallel_state_parallel_reduce_sep(
        b=b,
        n=n,
        h=h,
        d=d,
        e=e,
        k=k,
        v=v,
        ldk=kwargs["ldk"],
        BLOCK_N=kwargs["chunk_size"],
    )


def lavd_kv_state_chunk_parallel(k, v, **kwargs):
    b, n, h, d = k.shape
    e = v.shape[-1]
    return lavd_parallel_state_parallel_reduce_sep(
        b=b,
        n=n,
        h=h,
        d=d,
        e=e,
        k=k,
        v=v,
        ldk=kwargs["ldk"],
        ldv=kwargs["ldv"],
        BLOCK_N=kwargs["chunk_size"],
    )


module_map = {
    "gla": state_gla_k,
    "lavd_k": lavd_k_state_chunk_loop,
    "lavd_kv": lavd_kv_state_chunk_loop,
    "lavd_k_p": lavd_k_state_chunk_parallel,
    "lavd_kv_p": lavd_kv_state_chunk_parallel,
}

configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(8, 16)],
        xlabel="Sequence Length",
        ylabel="Execution Time(ms)",
        line_arg="provider",
        line_vals=[
            "gla",
            "lavd_k",
            "lavd_kv",
            "lavd_k_p",
            "lavd_kv_p",
        ],
        line_names=[
            "GLA",
            "LAVD_K",
            "LAVD_KV",
            "LAVD_K_P",
            "LAVD_KV_P",
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
        plot_name=f"state-{bench_type}-{mode}-batch{b}-head{h}-dim{d}-{dtype_name}",
        args={
            "b": b,
            "h": h,
            "d": d,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
            "bench_type": bench_type,
            "chunk_size": chunk_size,
        },
    )
    for bench_type in [
        "speed",
    ]
    for mode in [
        "fwd",
    ]
    for dtype_name in ["bf16"]
    for b, h, d in [[4, 32, 128], [1, 16, 128]]
    # for b, h, d in [[4, 32, 128]]
    # for b, h, d in [[1, 16, 128]]
    for chunk_size in [128]
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
    chunk_size=128,
    bench_type="speed",
):
    torch.manual_seed(2024)
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100

    shape = (b, n, h, d)
    k = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    v = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    ldk = F.logsigmoid(
        torch.randn((b, n, h, d), dtype=dtype, device=device)
    ).requires_grad_()
    ldv = F.logsigmoid(
        torch.randn((b, n, h, d), dtype=dtype, device=device)
    ).requires_grad_()

    module = module_map[provider]

    try:
        fn = lambda: module(k, v, ldk=ldk, ldv=ldv, chunk_size=chunk_size)
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
save_path = "stat/state"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
end_time = time.time()
total_time = end_time - start_time
print(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
