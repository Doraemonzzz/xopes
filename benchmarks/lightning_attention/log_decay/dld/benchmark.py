import os
import time

import numpy as np
import torch
import triton

from xopes.ops.lightning_attn.log_decay import compute_dld_torch, compute_dld_triton
from xopes.utils import get_memory

device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


# Define the benchmark configurations
configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(8, 16)],
        xlabel="Sequence Length",
        ylabel="Execution Time(ms)",
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["Torch", "Triton"],
        styles=[("red", "-"), ("blue", "-")],
        plot_name=f"dld-benchmark-{bench_type}-{mode}-batch{b}-head{h}-dim{d}-dim{e}-{use_final_state}-{dtype_name}",
        args={
            "b": b,
            "h": h,
            "d": d,
            "e": e,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
            "bench_type": bench_type,
            "use_final_state": use_final_state,
        },
    )
    for bench_type in ["speed", "memory"]
    for mode in [
        "fwd",
    ]
    for dtype_name in ["bf16"]
    for b, h, d, e, use_final_state in [
        [4, 32, 128, 128, True],
        [4, 32, 128, 128, False],
    ]
]


@triton.testing.perf_report(configs)
def benchmark_dld(
    b,
    n,
    h,
    d,
    e,
    dtype,
    device,
    mode,
    provider,
    bench_type="speed",
    use_final_state=False,
):
    torch.manual_seed(2024)
    assert mode in [
        "fwd",
    ]
    warmup = 25
    rep = 100

    (b, n, h, d)
    dld_q = torch.randn((b, n, h), dtype=dtype, device=device).requires_grad_()
    dld_k = torch.randn((b, n, h), dtype=dtype, device=device).requires_grad_()
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

    if provider == "torch":
        fn = lambda: torch.compile(compute_dld_torch)(
            dld_q, dld_k, final_state, dfinal_state
        )
    elif provider == "triton":
        fn = lambda: compute_dld_triton(dld_q, dld_k, final_state, dfinal_state)
    else:
        raise ValueError(f"Unknown provider: {provider}")

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


# Run the benchmark
start_time = time.time()
save_path = "stat/dld"
os.makedirs(save_path, exist_ok=True)
benchmark_dld.run(save_path=save_path, print_data=True)
end_time = time.time()
total_time = end_time - start_time
print(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
