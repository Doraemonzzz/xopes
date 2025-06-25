import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import triton

from xopes.ops.kernel_regression.causal_linear.krcl_parallel_triton import (
    krcl_parallel_inverse,
)

try:
    from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    from fla.ops.utils.solve_tril import solve_tril
except:
    chunk_scaled_dot_kkt_fwd = None
    solve_tril = None

from xopes.utils import get_memory

device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def fla_inv(q, k, ld, ld_cumsum, alpha, beta, **kwargs):
    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g_cumsum=ld_cumsum, output_dtype=torch.float32
    )

    A = solve_tril(A=A, output_dtype=k.dtype)

    return A


module_map = {
    "krcl": krcl_parallel_inverse,
    "fla": fla_inv,
}

configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(8, 17)],
        xlabel="Sequence Length",
        ylabel="Execution Time(ms)",
        line_arg="provider",
        line_vals=["krcl", "fla"],
        line_names=["KRCL", "FLA"],
        styles=[
            ("red", "-"),
            ("blue", "-"),
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
        "memory",
    ]
    for mode in [
        "fwd",
    ]
    for dtype_name in ["bf16"]
    # for b, h, d in [[4, 32, 128], [1, 16, 128]]
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

    if "laer" not in provider:
        shape = (b, n, h, d)
    else:
        shape = (b, n, h * d)
    k = F.normalize(
        torch.randn(shape, dtype=dtype, device=device), dim=-1
    ).requires_grad_()
    ld = F.logsigmoid(torch.randn((b, n, h), dtype=torch.float32, device=device))
    ld_cumsum = torch.cumsum(ld, dim=1)
    ld.requires_grad_()
    ld_cumsum.requires_grad_()
    beta = torch.randn(b, n, h, dtype=dtype, device=device).requires_grad_()

    module = module_map[provider]

    try:
        fn = lambda: module(
            q=None, k=k, ld=ld, ld_cumsum=ld_cumsum, alpha=None, beta=beta, BLOCK_N=64
        )
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
save_path = "stat/kernel_regression"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
end_time = time.time()
total_time = end_time - start_time
print(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
