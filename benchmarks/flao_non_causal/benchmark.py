import os

import numpy as np
import torch
import triton

from xopes.ops import flao_non_causal_torch, lao_non_causal_torch
from xopes.utils import get_memory

b, h, n, m, d, e = 12, 12, 1024, 1024, 128, 128

device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

module_map = {
    "lao_torch": lao_non_causal_torch,
    "lao_torch_complie": torch.compile(lao_non_causal_torch),
    "flao_torch": flao_non_causal_torch,
    "flao_torch_complie": torch.compile(flao_non_causal_torch),
}

configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(8, 16)],
        xlabel="Sequence Length",
        ylabel="Execution Time(ms)",
        line_arg="provider",
        line_vals=[
            "lao_torch",
            "lao_torch_complie",
            "flao_torch",
            "flao_torch_complie",
        ],
        line_names=[
            "Lao Torch",
            "Lao Torch C",
            "Flao Torch",
            "Flao Torch C",
        ],
        styles=[
            ("red", "-"),
            ("orange", "-"),
            ("green", "-"),
            ("blue", "-"),
            ("black", "-"),
        ],
        plot_name=f"flao-{bench_type}-{mode}-batch{b}-head{h}-dim{d}-dim{e}-{dtype_name}",
        args={
            "b": b,
            "h": h,
            "m": m,
            "d": d,
            "e": e,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
            "bench_type": bench_type,
        },
    )
    for mode in ["fwd", "bwd"]
    for dtype_name in ["bf16"]
    for bench_type in ["speed", "memory"]
]


@triton.testing.perf_report(configs)
def benchmark(b, h, n, m, d, e, dtype, device, mode, provider, bench_type="speed"):
    torch.manual_seed(2024)
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    m = n

    q = torch.randn((b, h, n, d), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((b, h, m, d), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((b, h, m, e), dtype=dtype, device=device).requires_grad_()
    g = torch.randn((b, h, n, e), dtype=dtype, device=device).requires_grad_()
    do = torch.randn((b, h, n, e), dtype=dtype, device=device)

    module = module_map[provider]

    fn = lambda: module(q, k, v, g)
    if mode == "bwd":
        o = fn()
        do = torch.randn((b, h, n, e), dtype=dtype, device=device)
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
        except:
            mb = -1

        return mb


save_path = "stat/flao"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
