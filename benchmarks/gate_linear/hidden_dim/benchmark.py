import os

import numpy as np
import torch
import triton

from xopes.ops.gate_linear import gate_linear_torch, gate_linear_triton
from xopes.utils import get_memory

torch._functorch.config.donated_buffer = False

device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

module_map = {
    "triton": gate_linear_triton,
    "torch": gate_linear_torch,
    "torch_compile": torch.compile(gate_linear_torch),
}

configs = [
    triton.testing.Benchmark(
        x_names=["d1"],
        x_vals=[2**i for i in range(8, 14)],
        xlabel="Hidden Dimension",
        ylabel="Execution Time(ms)" if bench_type == "speed" else "Memory (MB)",
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
        plot_name=f"gate_linear-{bench_type}-{mode}-batch{b}-{act}-bias_{use_bias}-residual_{use_residual}-{dtype_name}",
        args={
            "b": b,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
            "bench_type": bench_type,
            "act": act,
            "use_bias": use_bias,
            "use_residual": use_residual,
        },
    )
    for use_residual in [True, False]
    for use_bias in [False]
    for act in ["silu"]
    for bench_type in ["speed", "memory"]
    for mode in ["fwd", "bwd"]
    for dtype_name in ["bf16"]
    for b in [4096]
]


@triton.testing.perf_report(configs)
def benchmark(
    b,
    d1,
    dtype,
    device,
    mode,
    provider,
    bench_type="speed",
    act="none",
    use_bias=True,
    use_residual=False,
):
    torch.manual_seed(2024)
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100

    shape = (b, d1)
    x1 = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    x2 = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    d2 = d1 // 2
    weight = torch.randn((d2, d1), dtype=dtype, device=device).requires_grad_()

    bias = None
    if use_bias:
        bias = torch.randn(d2, dtype=dtype, device=device).requires_grad_()

    residual = None
    if use_residual:
        residual_shape = (b, d2)
        residual = torch.randn(
            residual_shape, dtype=dtype, device=device
        ).requires_grad_()

    module = module_map[provider]

    try:
        fn = lambda: module(x1, x2, weight, bias, residual, act)
    except Exception as e:
        print(e)
        fn = None

    if mode == "bwd":
        try:
            o = fn()
            do = torch.randn((b, d2), dtype=dtype, device=device)
            fn = lambda: o.backward(do, retain_graph=True)
        except Exception as e:
            print(e)
            fn = None

    if bench_type == "speed":
        try:
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        except Exception:
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
            print(e)
            mb = -1
        return mb


save_path = "stat/gate_linear"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
