import os
import time

import numpy as np
import torch
import triton

from xopes.ops.linear_cross_entropy import (
    linear_cross_entropy_torch,
    linear_cross_entropy_triton,
)
from xopes.ops.linear_cross_entropy.baseline import (
    linear_cross_entropy_cut_wrapper,
    linear_cross_entropy_fla_wrapper,
    linear_cross_entropy_jg_wrapper,
    linear_cross_entropy_liger_wrapper,
    linear_cross_entropy_xopes_wrapper,
)
from xopes.utils import get_memory

device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

module_map = {
    "triton_jg": linear_cross_entropy_jg_wrapper,
    "triton_cut": linear_cross_entropy_cut_wrapper,
    "triton_liger": linear_cross_entropy_liger_wrapper,
    "triton_fla": linear_cross_entropy_fla_wrapper,
    "triton": linear_cross_entropy_triton,
    "triton_linear_ce": linear_cross_entropy_xopes_wrapper,
    "torch": linear_cross_entropy_torch,
    "torch_compile": torch.compile(linear_cross_entropy_torch),
}

configs = [
    triton.testing.Benchmark(
        x_names=["b"],
        x_vals=[2**i for i in range(9, 15)],
        xlabel="Vocab Size",
        ylabel="Execution Time(ms)" if bench_type == "speed" else "Memory Usage(MB)",
        line_arg="provider",
        line_vals=[
            "triton_jg",
            "triton_cut",
            "triton_liger",
            "triton_fla",
            "triton",
            "triton_linear_ce",
            "torch",
            "torch_compile",
        ],
        line_names=[
            "tr_jg",
            "tr_cut",
            "tr_liger",
            "tr_fla",
            "tr",
            "tr_lce",
            "to",
            "toc",
        ],
        styles=[
            ("red", "-"),
            ("orange", "-"),
            ("green", "-"),
            ("blue", "-"),
            ("purple", "-"),
            ("yellow", "-"),
            ("pink", "-"),
            ("black", "-"),
        ],
        plot_name=f"lce-{bench_type}-{mode}-dim{d}-v{v}-{dtype_name}",
        args={
            "d": d,
            "v": v,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
            "bench_type": bench_type,
        },
    )
    for bench_type in ["speed", "memory"]
    for mode in ["fwd+bwd"]
    for dtype_name in ["bf16"]
    for d in [4096]
    for v in [131072]
]


@triton.testing.perf_report(configs)
def benchmark(
    b,
    d,
    v,
    dtype,
    device,
    mode,
    provider,
    bench_type,
):
    torch.manual_seed(2024)
    torch.cuda.empty_cache()
    assert mode in ["fwd", "bwd", "fwd+bwd"]
    warmup = 25
    rep = 100 if bench_type == "speed" else 20

    x = torch.randn((b, d), dtype=dtype, device=device).requires_grad_()
    y = torch.randint(0, v, (b,), device=device)
    W = torch.randn((v, d), dtype=dtype, device=device).requires_grad_()
    do = torch.randn((), dtype=dtype, device=device)

    module = module_map[provider]

    try:
        fn = lambda: module(x, y, W)
    except Exception as e:
        print(f"Error setting up {provider}: {e}")
        fn = None

    if mode == "bwd":
        try:
            o = fn()
            fn = lambda: o.backward(do, retain_graph=True)
        except Exception as e:
            print(f"Error setting up {provider}: {e}")
            fn = None
    elif mode == "fwd+bwd":
        try:

            def fn():
                o = module(x, y, W)
                return o.backward(do, retain_graph=True)

        except Exception as e:
            print(f"Error setting up {provider}: {e}")
            fn = None

    # Run benchmark
    if bench_type == "speed":
        try:
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        except Exception as e:
            print(f"Error in speed benchmark for {provider}: {e}")
            ms = -1
        return ms
    else:  # memory benchmark
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
save_path = "stat/linear_cross_entropy"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
end_time = time.time()
total_time = end_time - start_time
print(f"Total time: {total_time} seconds")
