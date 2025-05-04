import os
import time

import numpy as np
import torch
import triton

from xopes.ops.inverse.forword_substitution.inverse_fs_torch import inverse_fs_torch
from xopes.ops.inverse.forword_substitution.inverse_fs_triton import inverse_fs_triton
from xopes.ops.inverse.jacobian.inverse_jacobian_torch import inverse_jacobian_torch
from xopes.ops.inverse.jacobian.inverse_jacobian_triton import inverse_jacobian_triton
from xopes.ops.inverse.utils import construct_lower_triangular_matrix
from xopes.utils import get_memory

device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

# Define all implementations for benchmarking
module_map = {
    "fs_torch": inverse_fs_torch,
    "fs_triton_naive": lambda x: inverse_fs_triton(x, op_type=0),
    "fs_triton_loop": lambda x: inverse_fs_triton(x, op_type=1),
    "jac_torch": inverse_jacobian_torch,
    "jac_triton": lambda x: inverse_jacobian_triton(x),
    "fs_torch_compile": torch.compile(inverse_fs_torch),
    "jac_torch_compile": torch.compile(inverse_jacobian_torch),
}

configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(4, 8)],
        xlabel="Matrix Size (n x n)",
        ylabel="Execution Time(ms)" if bench_type == "speed" else "Memory Usage(MB)",
        line_arg="provider",
        line_vals=[
            # "fs_torch",
            "fs_triton_naive",
            # "fs_triton_loop",
            # "jac_torch",
            "jac_triton",
            # "fs_torch_compile",
            # "jac_torch_compile",
        ],
        line_names=[
            # "FS-Torch",
            "FS-Triton-Naive",
            # "FS-Triton-Loop",
            # "Jac-Torch",
            "Jac-Triton",
            # "FS-Torch-Compile",
            # "Jac-Torch-Compile",
        ],
        styles=[
            ("red", "-"),
            ("orange", "-"),
            ("blue", "-"),
            ("green", "-"),
            ("purple", "-"),
            ("cyan", "-"),
            ("magenta", "-"),
        ],
        plot_name=f"inverse-{bench_type}-{mode}-batch{b}-{dtype_name}",
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
    for dtype_name in ["fp32"]
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
    assert mode in [
        "fwd",
    ]
    warmup = 25
    rep = 100

    # Create a lower triangular matrix for inversion
    shape = (b, h, n, d)
    A = construct_lower_triangular_matrix(shape, dtype=dtype, device=device)

    module = module_map[provider]

    try:
        fn = lambda: module(A)
    except Exception as e:
        print(f"Error setting up {provider}: {e}")
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
save_path = "stat/inverse"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
end_time = time.time()
total_time = end_time - start_time
print(f"Total time: {total_time} seconds")
