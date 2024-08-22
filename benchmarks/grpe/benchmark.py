import os

import numpy as np
import torch
import torch.nn.functional as F
import triton
from lightning_attn.utils import get_memory

from xopes.ops import grpe_recurrence_torch, grpe_recurrence_triton

b, h, n, d = 4, 64, 8192, 64
e = 64
device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

module_map = {
    "recurrence_triton": grpe_recurrence_triton,
    # "block_recurrence_triton": logcumsumexp_block_recurrence_triton,
    # "block_parallel_triton": logcumsumexp_block_parallel_triton,
    # "block_recurrence_torch": grpe_block_recurrence_torch,
    "torch": grpe_recurrence_torch,
}

configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(9, 11)],
        xlabel="Sequence Length",
        ylabel="Execution Time(ms)",
        line_arg="provider",
        line_vals=[
            "recurrence_triton",
            # "block_recurrence_torch",
            "torch",
        ],
        line_names=[
            "BR_Torch",
            "Torch",
        ],
        styles=[
            ("red", "-"),
            ("orange", "-"),
            ("green", "-"),
            ("blue", "-"),
            ("black", "-"),
        ],
        plot_name=f"grpe-{bench_type}-{mode}-batch{b}-dim{d}-{dtype_name}",
        args={
            "b": b,
            "d": d,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
            "bench_type": bench_type,
        },
    )
    for mode in [
        "fwd",
        # "bwd"
    ]
    for dtype_name in ["bf16"]
    for bench_type in ["speed", "bwd"]
]


@triton.testing.perf_report(configs)
def benchmark(b, n, d, dtype, device, mode, provider, dim=-2, bench_type="speed"):
    torch.manual_seed(2024)
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    q = (torch.randn((b, h, n, d), dtype=dtype, device=device)).requires_grad_()
    k = (torch.randn((b, h, n, d), dtype=dtype, device=device)).requires_grad_()
    v = (torch.randn((b, h, n, e), dtype=dtype, device=device)).requires_grad_()
    alpha = torch.log(
        0.95
        + (1 - 0.95) * F.sigmoid(torch.randn((b, h, n, d), dtype=dtype, device=device))
    ).requires_grad_()
    beta = torch.log(
        0.95
        + (1 - 0.95) * F.sigmoid(torch.randn((b, h, n), dtype=dtype, device=device))
    ).requires_grad_()
    gamma = F.normalize(
        torch.randn((b, h, n, d), dtype=dtype, device=device)
    ).requires_grad_()
    do = torch.randn((b, h, n, e), dtype=dtype, device=device)

    module = module_map[provider]

    fn = lambda: module(q, k, v, alpha, beta, gamma)
    if mode == "bwd":
        y = fn()[0]
        dy = torch.randn((b, h, n, e), dtype=dtype, device=device)
        fn = lambda: y.backward(dy, retain_graph=True)

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


save_path = "stat/grpe"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
