import os
import time

import numpy as np
import torch
import triton
import triton.testing

from xopes.ops.normalize import normalize_torch, normalize_triton
from xopes.utils import get_memory

torch._functorch.config.donated_buffer = False  # noqa


def get_torch_compile_fn(fn):
    return torch.compile(fn)


module_map = {
    "torch": normalize_torch,
    "triton": normalize_triton,
    "torch_compile": get_torch_compile_fn(normalize_torch),
}

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
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
            "torch",
            "torch_compile",
        ],
        line_names=["tr", "to", "toc"],
        styles=[
            ("red", "-"),
            ("orange", "-"),
            ("green", "-"),
            ("blue", "-"),
            ("black", "-"),
        ],
        plot_name=f"normalize_gate-{bench_type}-{mode}-batch{b}-dim{d}-gate_act_{gate_act}-gate_pos_{gate_pos}-{dtype_name}",
        args={
            "b": b,
            "d": d,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
            "bench_type": bench_type,
            "num_groups": num_groups,
            "use_mean": use_mean,
            "use_weight": use_weight,
            "use_bias": use_bias,
            "gate_act": gate_act,
            "gate_pos": gate_pos,
        },
    )
    for bench_type in ["speed", "memory"]
    for gate_pos in ["pre", "post"]
    for gate_act in [
        "sigmoid",
    ]
    for mode in ["fwd", "bwd"]
    for dtype_name in ["bf16"]
    for use_mean in [False]
    for use_weight in [True]
    for use_bias in [False]
    for num_groups in [1]
    for b in [4]
    for d in [2048]
    for device in ["cuda"]
]


@triton.testing.perf_report(configs)
def benchmark(
    b,
    n,
    d,
    dtype,
    device,
    mode,
    provider,
    num_groups=1,
    use_mean=False,
    use_weight=False,
    use_bias=False,
    gate_act="sigmoid",
    gate_pos="pre",
    c=1,
    eps=1e-5,
    bench_type="speed",
):
    """
    Benchmark normalize with gate operations

    Args:
        b: batch size
        n: sequence length
        d: hidden dimension
        dtype: data type
        device: device to run on
        mode: forward or backward
        provider: implementation provider (torch, triton, torch_compile)
        num_groups: number of groups for group normalization
        use_mean: whether to use mean in normalization
        use_weight: whether to use weight parameter
        use_bias: whether to use bias parameter
        gate_act: gate activation function (sigmoid, relu, silu)
        gate_pos: gate position (pre, post)
        c: scaling factor
        eps: epsilon for numerical stability
        bench_type: speed or memory benchmark
    """
    torch.manual_seed(2024)
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100

    shape = (b, n, d)
    x = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    # Create gate tensor with same shape as input
    gate = torch.randn(shape, dtype=dtype, device=device).requires_grad_()

    if use_weight:
        weight = torch.randn((d,), dtype=dtype, device=device).requires_grad_()
    else:
        weight = None

    if use_bias:
        bias = torch.randn((d,), dtype=dtype, device=device).requires_grad_()
    else:
        bias = None

    module = module_map[provider]

    try:
        fn = lambda: module(
            x,
            weight=weight,
            bias=bias,
            residual=None,
            gate=gate,
            gate_act=gate_act,
            gate_pos=gate_pos,
            c=c,
            eps=eps,
            num_groups=num_groups,
            use_mean=use_mean,
            return_residual=False,
        )
    except Exception as e:
        print(f"Error setting up {provider}: {e}")
        fn = None

    if mode == "bwd":
        try:
            o = fn()
            if isinstance(o, tuple):
                o = o[0]
            do = torch.randn(shape, dtype=dtype, device=device)
            fn = lambda: o.backward(do, retain_graph=True)
        except Exception as e:
            print(f"Error in speed benchmark for {provider}: {e}")
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
save_path = "stat/normalize_gate"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
end_time = time.time()
total_time = end_time - start_time
print(f"Total time: {total_time} seconds")
