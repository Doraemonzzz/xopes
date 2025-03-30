import os
import time
from typing import Optional

import numpy as np
import torch
import triton

from xopes.ops.cumsum import cumsum_fn
from xopes.ops.lightning_attn.log_decay import compute_dld_fn
from xopes.ops.lightning_attn.log_decay.log_decay_with_cumsum import (
    compute_dld_with_cumsum_triton,
)
from xopes.utils import get_memory

device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def compute_dld_with_cumsum_triton_sep(
    dld_q: torch.Tensor,  # B N H D
    dld_k: torch.Tensor,  # B N H D
    final_state: Optional[torch.Tensor] = None,  # B H D E
    dfinal_state: Optional[torch.Tensor] = None,  # B H D E
    cu_seqlens: Optional[torch.Tensor] = None,  # M
    sum_option: Optional[int] = -1,
):
    # B N H NUM_BLOCK_E -> B N H
    if dld_q.shape[-1] == 1:
        dld_q = dld_q.squeeze(-1)
    else:
        dld_q = dld_q.sum(-1)

    dld_q = cumsum_fn(dld_q, dim=1, reverse=True)

    if dld_k.shape[-1] == 1:
        dld_k = dld_k.squeeze(-1)
    else:
        dld_k = dld_k.sum(-1)

    dld_k = cumsum_fn(dld_k, dim=1, reverse=True)

    dld = compute_dld_fn(
        dld_q=dld_q,
        dld_k=dld_k,
        final_state=final_state,
        dfinal_state=dfinal_state,
    )

    return dld


module_map = {
    "triton_sep": compute_dld_with_cumsum_triton_sep,
    "triton": compute_dld_with_cumsum_triton,
}


# Define the benchmark configurations
configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(8, 16)],
        xlabel="Sequence Length",
        ylabel="Execution Time(ms)",
        line_arg="provider",
        line_vals=["triton_sep", "triton"],
        line_names=["Triton Sep", "Triton"],
        styles=[("red", "-"), ("blue", "-")],
        plot_name=f"dld-with-cumsum-benchmark-{bench_type}-{mode}-batch{b}-head{h}-dim{d}-dim{e}-{use_final_state}-{sum_option}-{dtype_name}",
        args={
            "b": b,
            "h": h,
            "d": d,
            "e": e,
            "f": f,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
            "bench_type": bench_type,
            "use_final_state": use_final_state,
            "sum_option": sum_option,
        },
    )
    for bench_type in ["speed", "memory"]
    # for bench_type in ["speed"]
    for mode in [
        "fwd",
    ]
    for dtype_name in ["bf16"]
    for sum_option in [-1]
    for b, h, d, e, use_final_state in [
        [4, 32, 128, 128, True],
    ]
    for f in [4]
]


@triton.testing.perf_report(configs)
def benchmark_dld_with_cumsum(
    b,
    n,
    h,
    d,
    e,
    f,
    dtype,
    device,
    mode,
    provider,
    bench_type="speed",
    use_final_state=False,
    sum_option=-1,
):
    torch.manual_seed(2024)
    assert mode in ["fwd"]
    warmup = 25
    rep = 100
    cu_seqlens = None

    # Generate input tensors based on sum_option
    dld_q = torch.randn((b, n, h, f), dtype=dtype, device=device).requires_grad_()
    dld_k = torch.randn((b, n, h, f), dtype=dtype, device=device).requires_grad_()

    # Generate final state tensors if needed
    if use_final_state:
        final_state = torch.randn(
            (b, h, d, e), dtype=dtype, device=device
        ).requires_grad_()
        dfinal_state = torch.randn(
            (b, h, d, e), dtype=dtype, device=device
        ).requires_grad_()
    else:
        final_state = None
        dfinal_state = None

    module = module_map[provider]

    # Define the function to benchmark based on provider
    try:
        fn = lambda: module(
            dld_q, dld_k, final_state, dfinal_state, cu_seqlens, sum_option
        )
    except Exception as e:
        print(f"Error setting up {provider}: {e}")
        fn = None

    # Benchmark speed
    if bench_type == "speed":
        try:
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        except Exception as e:
            print(f"Error setting up {provider}: {e}")
            ms = -1

        return ms
    # Benchmark memory usage
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
save_path = "stat/dld_with_cumsum"
os.makedirs(save_path, exist_ok=True)
benchmark_dld_with_cumsum.run(save_path=save_path, print_data=True)
end_time = time.time()
total_time = end_time - start_time
print(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
