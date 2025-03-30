import os
import time

import numpy as np
import torch
import triton

from xopes.ops.cumsum.cumsum import (
    cumsum_chunk_loop_triton,
    cumsum_no_reshape_triton,
    cumsum_torch,
    cumsum_triton,
)
from xopes.utils import get_memory

device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

module_map = {
    "triton": cumsum_triton,
    "triton_chunk_loop": cumsum_chunk_loop_triton,
    "triton_no_reshape": cumsum_no_reshape_triton,
    "torch": cumsum_torch,
    "torch_compile": torch.compile(cumsum_torch),
}

configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(10, 16)],
        xlabel="Sequence Length",
        ylabel="Execution Time(ms)",
        line_arg="provider",
        line_vals=[
            "triton",
            "triton_chunk_loop",
            "triton_no_reshape",
            "torch",
            "torch_compile",
        ],
        line_names=[
            "tr",
            "trc",
            "trnr",
            "to",
            "toc",
        ],
        styles=[
            ("red", "-"),
            ("orange", "-"),
            ("green", "-"),
            ("blue", "-"),
            ("purple", "-"),
        ],
        plot_name=f"cumsum-{bench_type}-{mode}-b{b}-h{h}-reverse_{reverse}-{dtype_name}",
        args={
            "b": b,
            "h": h,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
            "bench_type": bench_type,
            "reverse": reverse,
        },
    )
    for reverse in [True, False]
    for bench_type in ["speed", "memory"]
    for mode in [
        "fwd",
    ]
    for dtype_name in ["bf16"]
    for b in [12]
    for h in [1, 128]
]


@triton.testing.perf_report(configs)
def benchmark(
    b,
    n,
    h,
    dtype,
    device,
    mode,
    provider,
    bench_type="speed",
    reverse=False,
):
    torch.manual_seed(2024)
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100

    if h == 1:
        shape = (b, n)
    else:
        shape = (b, n, h)
    x = torch.randn(shape, dtype=dtype, device=device).requires_grad_()

    module = module_map[provider]

    try:
        fn = lambda: module(x, dim=1, reverse=reverse)
    except Exception as e:
        print(f"Error setting up {provider}: {e}")
        fn = None

    if mode == "bwd":
        try:
            o = fn()
            do = torch.randn(shape, dtype=dtype, device=device)
            fn = lambda: o.backward(do, retain_graph=True)
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
            print(f"Error setting up {provider}: {e}")
            mb = -1

        return mb


start_time = time.time()
save_path = "stat/cumsum"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
end_time = time.time()
total_time = end_time - start_time
print(f"Total time: {total_time} seconds ({(total_time/60):.2f} minutes)")
