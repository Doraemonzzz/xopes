import os
import time

import numpy as np
import torch
import triton

from xopes.ops.flash_attn.tpa.tpa_decode_triton import tpa_decode_parallel_bn_triton
from xopes.utils import get_memory

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None

try:
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata
except ImportError:
    flash_mla_with_kvcache = None
    get_mla_metadata = None


def flash_attn_decode_fn(q, k, v):
    return flash_attn_with_kvcache(q, k, v)


def flash_mla_decode_fn(
    q_mla,
    blocked_k,
    block_table,
    cache_seqlens,
    e,
    tile_scheduler_metadata,
    num_splits,
    causal=True,
):
    return flash_mla_with_kvcache(
        q_mla,
        blocked_k,
        block_table,
        cache_seqlens,
        e,
        tile_scheduler_metadata,
        num_splits,
        causal=causal,
    )[0]


# Default parameters for the benchmark
device = torch.device("cuda")

# Map of PyTorch datatypes
dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

# Map of implementations to benchmark
module_map = {
    "tpa": tpa_decode_parallel_bn_triton,
    "mha": flash_attn_decode_fn,
    "gqa": flash_attn_decode_fn,
    "mqa": flash_attn_decode_fn,
    "mla": flash_mla_decode_fn,
}

# Define benchmark configurations
configs = [
    triton.testing.Benchmark(
        x_names=["m"],
        x_vals=[2**i for i in range(8, 16)],
        # x_vals=[2**i for i in range(8, 9)],
        xlabel="Sequence Length",
        ylabel="Execution Time(ms)" if bench_type == "speed" else "Memory Usage(MB)",
        line_arg="provider",
        line_vals=[
            "tpa",
            "mha",
            "gqa",
            "mqa",
            "mla",
        ],
        line_names=[
            "tpa",
            "mha",
            "gqa",
            "mqa",
            "mla",
        ],
        styles=[
            ("red", "-"),
            ("orange", "-"),
            ("blue", "-"),
            ("green", "-"),
            ("purple", "-"),
            ("yellow", "-"),
            ("cyan", "-"),
        ],
        plot_name=f"tpa_decode-{bench_type}-{mode}-batch{b}-head{h}-dim{d}-rank{r}-group{g}-{dtype_name}",
        args={
            "b": b,
            "n": n,
            "h": h,
            "r": r,
            "g": g,
            "d": d,
            "e": e,
            # for mla
            "h_kv": h_kv,
            "d_qk": d_qk,
            "dv": dv,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
            "bench_type": bench_type,
        },
    )
    for mode in [
        "fwd",
    ]
    for dtype_name in ["bf16"]
    for bench_type in ["speed"]
    for b in [8, 32]
    for n in [1]
    for h in [32]
    for g in [8]
    for r in [16]
    for d in [128]
    for e in [128]
    # for mla
    for h_kv in [1]
    for d_qk in [192]  # if have bug, change this to 576
    for dv in [512]
]


@triton.testing.perf_report(configs)
def benchmark(
    b,
    n,
    m,
    h,
    r,
    g,
    d,
    e,
    h_kv,
    d_qk,
    dv,
    dtype,
    device,
    mode,
    provider,
    bench_type="speed",
):
    torch.manual_seed(2024)
    assert mode in [
        "fwd",
    ]
    assert n == 1, "n must be 1 for decoding"

    warmup = 25
    rep = 100

    # Create tensors for benchmarking
    aq = torch.randn((b, n, h, r), dtype=dtype, device=device)
    ak = torch.randn((b, m, h), dtype=dtype, device=device)
    av = torch.randn((b, m, h), dtype=dtype, device=device)
    bq = torch.randn((b, n, r, d), dtype=dtype, device=device)
    bk = torch.randn((b, m, d), dtype=dtype, device=device)
    bv = torch.randn((b, m, e), dtype=dtype, device=device)

    q = torch.randn((b, n, h, d), dtype=dtype, device=device)

    # for mha
    k_mha = torch.randn((b, m, h, d), dtype=dtype, device=device)
    v_mha = torch.randn((b, m, h, e), dtype=dtype, device=device)

    # for gqa
    k_gqa = torch.randn((b, m, g, d), dtype=dtype, device=device)
    v_gqa = torch.randn((b, m, g, e), dtype=dtype, device=device)

    # for mqa
    k_mqa = torch.randn((b, m, 1, d), dtype=dtype, device=device)
    v_mqa = torch.randn((b, m, 1, e), dtype=dtype, device=device)

    # for mla
    h_q = h
    q_mla = torch.randn((b, n, h_q, d_qk), dtype=dtype, device=device)
    cache_seqlens = torch.tensor(
        [m for i in range(b)], dtype=torch.int32, device=device
    )
    cache_seqlens.sum().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
    block_size = 64
    block_table = (
        torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32)
        .view(b, max_seqlen_pad // block_size)
        .to(device)
    )
    blocked_k = torch.randn(
        block_table.numel(), block_size, h_kv, d_qk, dtype=dtype, device=device
    )
    for i in range(b):
        blocked_k.view(b, max_seqlen_pad, h_kv, d_qk)[
            i, cache_seqlens[i].item() :
        ] = float("nan")
    blocked_v = blocked_k[..., :dv]

    tile_scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens, n * h // h_kv, h_kv
    )

    module = module_map[provider]

    if provider == "mha":
        fn = lambda: module(q, k_mha, v_mha)
    elif provider == "gqa":
        fn = lambda: module(q, k_gqa, v_gqa)
    elif provider == "mqa":
        fn = lambda: module(q, k_mqa, v_mqa)
    elif provider == "mla":
        fn = lambda: module(
            q_mla,
            blocked_k,
            block_table,
            cache_seqlens,
            e,
            tile_scheduler_metadata,
            num_splits,
            causal=True,
        )
    else:
        fn = lambda: module(aq, ak, av, bq, bk, bv)

    if bench_type == "speed":
        # Measure execution time
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms
    else:
        # Measure memory usage
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
save_path = "stat/decode"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
end_time = time.time()
total_time = end_time - start_time
print(f"Total time: {total_time} seconds, {total_time/60} minutes")
