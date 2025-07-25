import os
import time

import numpy as np
import torch
import triton

from xopes.ops.lrpe.cosine._1d import lrpe_cosine_1d_sp_triton, lrpe_cosine_1d_torch
from xopes.utils import get_memory

b, h, n, d = 12, 12, 8192, 128
h_t, d_t = -1, -1
# h_t, d_t = -1, d
# h_t, d_t = h, -1
device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

module_map = {
    "triton": lrpe_cosine_1d_sp_triton,
    "torch": lrpe_cosine_1d_torch,
    "torch_compile": torch.compile(lrpe_cosine_1d_torch),
}

configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(8, 16)],
        xlabel="Sequence Length",
        ylabel="Execution Time(ms)",
        line_arg="provider",
        line_vals=[
            "triton",
            "torch",
            "torch_compile",
        ],
        line_names=["tr", "to", "toc"],
        styles=[
            ("red", "-"),
            ("orange", "-"),
            ("green", "-"),
        ],
        plot_name=f"lrpe_cosine-{bench_type}-{mode}-batch{b}-head{h}-dim{d}-act_{act}-dim_{dim}-{dtype_name}"
        if dim is not None
        else f"lrpe_cosine-{bench_type}-{mode}-batch{b}-head{h}-dim{d}-act_{act}-{dtype_name}",
        args={
            "b": b,
            "h": h,
            "d": d,
            "h_t": h_t,
            "d_t": d_t,
            "act": act,
            "dim": dim,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
            "bench_type": bench_type,
        },
    )
    for mode in ["fwd", "bwd"]
    for dtype_name in ["bf16"]
    for bench_type in ["speed", "memory"]
    # witout dim
    for act in ["none"]
    # for act in ["silu"]
    for dim in [None]
    # with dim
    # for act in ["softmax"]
    # for dim in [-1]
    # for dim in [1]
]


@triton.testing.perf_report(configs)
def benchmark(
    b, h, n, d, h_t, d_t, act, dim, dtype, device, mode, provider, bench_type="speed"
):
    torch.manual_seed(2024)
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    h_t, d_t = h_t, d_t
    if h_t == -1:
        h_t = h
    if d_t == -1:
        d_t = d
    x = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()
    theta = torch.randn((h_t, d_t), dtype=dtype, device=device)

    module = module_map[provider]

    fn = lambda: module(x, theta, act=act, dim=dim)
    if mode == "bwd":
        o = fn()
        do = torch.randn((b, n, h, 2 * d), dtype=dtype, device=device)
        fn = lambda: o.backward(do, retain_graph=True)

    if bench_type == "speed":
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

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
save_path = "stat/lrpe_cosine"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
end_time = time.time()
total_time = end_time - start_time
print(f"Total time: {total_time} seconds")
