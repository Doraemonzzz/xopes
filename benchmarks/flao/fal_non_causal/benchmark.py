import os

import numpy as np
import torch
import torch._dynamo
import triton

from xopes.ops.flao.fal_non_causal import (
    flao_al_non_causal_torch,
    flao_fal_non_causal_torch,
)
from xopes.utils import get_memory

torch._dynamo.config.suppress_errors = True

b, h, n, m, d, e = 12, 12, 1024, 1024, 128, 128
# act
q_act = "silu"
q_act_dim = None
# q_act = "softmax"
# q_act_dim = -1
k_act = "silu"
k_act_dim = None
v_act = "none"
v_act_dim = None
g_act = "silu"
g_act_dim = None
# lrpe
use_lrpe = True
shape = None
lrpe_type = "cosine"
offset = 0
l = 0

device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

module_map = {
    "flao_al_torch": flao_al_non_causal_torch,
    "flao_al_torch_complie": torch.compile(flao_al_non_causal_torch),
    "flao_fal_torch": flao_fal_non_causal_torch,
    "flao_fal_torch_complie": torch.compile(flao_fal_non_causal_torch),
}

configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(8, 16)],
        xlabel="Sequence Length",
        ylabel="Execution Time(ms)",
        line_arg="provider",
        line_vals=[
            "flao_al_torch",
            "flao_al_torch_complie",
            "flao_fal_torch",
            "flao_fal_torch_complie",
        ],
        line_names=["Flao Tor", "Flao Tor C", "Flao F Tor", "Flao F Tor C"],
        styles=[
            ("red", "-"),
            ("orange", "-"),
            ("green", "-"),
            ("blue", "-"),
            ("black", "-"),
        ],
        plot_name=f"flao_fal-{bench_type}-{mode}-batch{b}-head{h}-dim{d}-dim{e}-qact_{q_act}_{q_act_dim}-kact_{k_act}_{k_act_dim}-vact_{v_act}_{v_act_dim}-gact_{g_act}_{g_act_dim}-{dtype_name}"
        if use_lrpe
        else f"flao_al-{bench_type}-{mode}-batch{b}-head{h}-dim{d}-dim{e}-qact_{q_act}_{q_act_dim}-kact_{k_act}_{k_act_dim}-vact_{v_act}_{v_act_dim}-gact_{g_act}_{g_act_dim}-{lrpe_type}-{dtype_name}",
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
    if use_lrpe:
        theta = torch.randn((h, d), dtype=dtype, device=device)
    else:
        theta = None

    module = module_map[provider]

    fn = lambda: module(
        q,
        k,
        v,
        g,
        q_act,
        q_act_dim,
        k_act,
        k_act_dim,
        v_act,
        v_act_dim,
        g_act,
        g_act_dim,
        theta,
        shape,
        lrpe_type,
        offset,
        l,
    )
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


save_path = "stat/fuse_act_lrpe_non_causal"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
