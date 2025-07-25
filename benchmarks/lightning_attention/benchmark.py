import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import triton
from einops import rearrange

from xopes.ops.kernel_regression.causal_linear import krcl_parallel_triton
from xopes.ops.lightning_attn.baseline import (
    chunk_gla_wrapper,
    chunk_hgrn_fla_wrapper,
    chunk_linear_attn_wrapper,
    chunk_simple_gla_wrapper,
    delta_rule_wrapper,
    flash_attn_wrapper,
    gated_delta_rule_wrapper,
    lightning_attn_no_decay_wrapper,
    lightning_attn_wrapper,
    mamba2_wrapper,
    mlstm_wrapper,
)
from xopes.ops.lightning_attn.constant_decay import (
    lacd_parallel_triton,
    lacd_recurrence_triton,
)
from xopes.ops.lightning_attn.element_recurrence import (
    laer_parallel_triton,
    laer_recurrence_triton,
)
from xopes.ops.lightning_attn.scalar_decay import lasd_parallel_triton
from xopes.ops.lightning_attn.vector_decay import lavd_parallel_triton
from xopes.utils import get_memory

device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def chunk_gla_k(q, k, v, **kwargs):
    return chunk_gla_wrapper(q, k, v, ldk=kwargs["ldk"])[0]


def chunk_simple_gla_k(q, k, v, **kwargs):
    return chunk_simple_gla_wrapper(q, k, v, ld3=kwargs["ld3"])[0]


def chunk_linear_attn(q, k, v, **kwargs):
    return chunk_linear_attn_wrapper(q, k, v, ldk=kwargs["ldk"])[0]


def lacd_recurrence(q, k, v, **kwargs):
    return lacd_recurrence_triton(q, k, v, ld=kwargs["ld"])[0]


def lacd_parallel(q, k, v, **kwargs):
    return lacd_parallel_triton(q, k, v, ld=kwargs["ld"])[0]


def land_parallel(q, k, v, **kwargs):
    return lacd_parallel_triton(q, k, v)[0]


def lasd_parallel(q, k, v, **kwargs):
    return lasd_parallel_triton(q, k, v, ld=kwargs["ld3"])[0]


def lavd_k_parallel(q, k, v, **kwargs):
    return lavd_parallel_triton(q, k, v, ldk=kwargs["ldk"])[0]


def lavd_kv_parallel(q, k, v, **kwargs):
    return lavd_parallel_triton(q, k, v, ldk=kwargs["ldk"], ldv=kwargs["ldv"])[0]


def laer_recurrence(q, k, v, **kwargs):
    return laer_recurrence_triton(q, k, v, ld=kwargs["ldk"])[0]


def laer_parallel(q, k, v, **kwargs):
    return laer_parallel_triton(q, k, v, ld=kwargs["ldk"])[0]


def lightning_parallel(q, k, v, **kwargs):
    return lightning_attn_wrapper(q, k, v, ld=kwargs["ld"], variant="parallel")


def lightning_no_decay_parallel(q, k, v, **kwargs):
    return lightning_attn_no_decay_wrapper(q, k, v, variant="parallel")


def lightning_no_decay_chunk_loop(q, k, v, **kwargs):
    return lightning_attn_no_decay_wrapper(q, k, v, variant="chunk_loop")


def chunk_hgrn_fla(q, k, v, **kwargs):
    return chunk_hgrn_fla_wrapper(q, k, v, ldk=kwargs["ldk"])[0]


def mamba2(q, k, v, **kwargs):
    return mamba2_wrapper(
        x=v,
        dt=kwargs["dt"],
        A=kwargs["A"],
        B=k,
        C=q,
    )


def mlstm(q, k, v, **kwargs):
    return mlstm_wrapper(q, k, v, i=kwargs["i"], ld3=kwargs["ld3"])[0]


def delta_rule(q, k, v, **kwargs):
    return delta_rule_wrapper(q, k, v, beta=kwargs["beta"])[0]


def gated_delta_rule(q, k, v, **kwargs):
    return gated_delta_rule_wrapper(q, k, v, ld3=kwargs["ld3"], beta=kwargs["beta"])[0]


def krcl(q, k, v, **kwargs):
    return krcl_parallel_triton(
        q=None,
        k=k,
        v=v,
        ld=kwargs["ld3"],
        alpha=kwargs["alpha"],
        beta=kwargs["beta"],
        BLOCK_N=64,
    )[0]


module_map = {
    "lacd_r": lacd_recurrence,
    "lacd_p": lacd_parallel,
    "land_p": land_parallel,
    "lacd_pl": lacd_parallel,
    "lasd_p": lasd_parallel,
    "lavd_k_p": lavd_k_parallel,
    "lavd_kv_p": lavd_kv_parallel,
    "laer_r": laer_recurrence,
    "laer_p": laer_parallel,
    "flash": flash_attn_wrapper,
    "lightning_p": lightning_parallel,
    "lightning_nd_p": lightning_no_decay_parallel,
    "lightning_nd_c": lightning_no_decay_chunk_loop,
    "gla_k": chunk_gla_k,
    "gla_s_k": chunk_simple_gla_k,
    "fla_laer": chunk_hgrn_fla,
    "fla_linear": chunk_linear_attn,
    "mamba2": mamba2,
    "mlstm": mlstm,
    "krcl": krcl,
    "delta_rule": delta_rule,
    "gated_delta_rule": gated_delta_rule,
}

configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        # x_vals=[2**i for i in range(8, 17)],
        # x_vals=[2**i for i in range(8, 11)],
        # x_vals=[2**i for i in range(9, 10)],
        # x_vals=[2**i for i in range(10, 13)],
        x_vals=[2**i for i in range(10, 11)],
        # x_vals=[2**i for i in range(8, 18)],
        # x_vals=[2**i for i in range(17, 18)],
        xlabel="Sequence Length",
        ylabel="Execution Time(ms)",
        line_arg="provider",
        line_vals=[
            # "lacd_r",
            # "lacd_p",
            # "land_p",
            # "lacd_pl",
            "lasd_p",
            # "lavd_k_p",
            # "lavd_kv_p",
            # "laer_r",
            # "laer_p",
            # "flash",
            # "lightning_p",
            # "lightning_c",
            # "gla_k",
            # "gla_s_k",
            # "fla_laer",
            # "mamba2",
            # "mlstm",
            "krcl",
            "delta_rule",
            "gated_delta_rule",
        ],
        line_names=[
            # "LACD_R",
            # "LACD_P",
            # "LAND_P",
            # "LACD_PL",
            "LASD_P",
            # "LAVD_K_P",
            # "LAVD_KV_P",
            # "LAER_R",
            # "LAER_P",
            # "Flash",
            # "LP",
            # "LC",
            # "GLA_K",
            # "GLA_S_K",
            # "FLA_LAER",
            # "MAMBA2",
            # "MLSTM",
            "KRCL",
            "DR",
            "GDR",
        ],
        styles=[
            ("red", "-"),
            ("blue", "-"),
            ("green", "-"),
            ("orange", "-"),
            ("purple", "-"),
            ("pink", "-"),
            ("yellow", "-"),
            ("cyan", "-"),
            ("brown", "-"),
            ("magenta", "-"),
            ("gray", "-"),
            ("lime", "-"),
            ("olive", "-"),
            ("teal", "-"),
            ("black", "-"),
            ("red", "--"),
            ("blue", "--"),
            ("green", "--"),
            ("orange", "--"),
            ("purple", "--"),
            ("pink", "--"),
            ("yellow", "--"),
            ("cyan", "--"),
            ("brown", "--"),
            ("magenta", "--"),
            ("gray", "--"),
            ("lime", "--"),
            ("olive", "--"),
            ("teal", "--"),
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
        # "memory",
    ]
    for mode in [
        "fwd",
        "bwd",
    ]
    for dtype_name in ["bf16"]
    # for b, h, d in [[4, 32, 128], [1, 16, 128]]
    for b, h, d in [[4, 32, 128]]
    # for b, h, d in [[1, 16, 128]]
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
    q = F.normalize(
        torch.randn(shape, dtype=dtype, device=device), dim=-1
    ).requires_grad_()
    k = F.normalize(
        torch.randn(shape, dtype=dtype, device=device), dim=-1
    ).requires_grad_()
    v = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    if provider == "lacd_pl":
        ld = F.logsigmoid(torch.randn(h, dtype=dtype, device=device)).requires_grad_()
    else:
        ld = F.logsigmoid(torch.randn(h, dtype=dtype, device=device))
    ld3 = F.logsigmoid(
        torch.randn((b, n, h), dtype=dtype, device=device)
    ).requires_grad_()
    i = F.logsigmoid(
        torch.randn((b, n, h), dtype=dtype, device=device)
    ).requires_grad_()
    ldk = F.logsigmoid(torch.randn(shape, dtype=dtype, device=device)).requires_grad_()
    ldv = F.logsigmoid(torch.randn(shape, dtype=dtype, device=device)).requires_grad_()
    dt = -F.logsigmoid(
        torch.randn(b, n, h, dtype=dtype, device=device)
    ).requires_grad_()
    A = torch.randn(h, dtype=dtype, device=device).requires_grad_()
    # for kernel regression
    alpha = torch.randn(b, n, h, dtype=dtype, device=device).requires_grad_()
    beta = torch.randn(b, n, h, dtype=dtype, device=device).requires_grad_()

    module = module_map[provider]

    if provider == "lasr_r":
        q, k, v = map(lambda x: rearrange(x, "... h d -> ... (h d)"), (q, k, v, ld3, i))

    if provider == "mlstm":
        q, k, v = map(lambda x: rearrange(x, "b n h d -> b h n d"), (q, k, v))
        i, ld3 = map(lambda x: rearrange(x, "b n h -> b h n"), (i, ld3))

    try:
        fn = lambda: module(
            q,
            k,
            v,
            ld=ld,
            ld3=ld3,
            ldk=ldk,
            ldv=ldv,
            dt=dt,
            A=A,
            i=i,
            alpha=alpha,
            beta=beta,
        )
    except Exception as e:
        print(f"Error setting up {provider}: {e}")
        fn = None

    if mode == "bwd":
        try:
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        except Exception as e:
            print(f"Error in speed benchmark for {provider}: {e}")
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
save_path = "stat/la"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
end_time = time.time()
total_time = end_time - start_time
print(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
