import os
import time

import numpy as np
import torch
import triton

from xopes.ops import (
    multinomial_torch,
    online_multinomial_torch,
    online_multinomial_triton,
    online_with_cache_multinomial_torch,
    parallel_gumbel_multinomial_triton,
    parallel_multinomial_triton,
)
from xopes.utils import get_memory

torch._dynamo.config.suppress_errors = True

# b, d = 12 * 2048, 4096
b, d = 12, 4096
num_samples = 1
# num_samples = 2048
# num_samples = 16
device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

module_map = {
    "torch": multinomial_torch,
    "owc_t": online_with_cache_multinomial_torch,
    "owc_t_c": torch.compile(online_with_cache_multinomial_torch),
    "o_t": online_multinomial_torch,
    "o_t_c": torch.compile(online_multinomial_torch),
    "om_tr": online_multinomial_triton,
    "pm_tr": parallel_multinomial_triton,
    "pgm_tr": parallel_gumbel_multinomial_triton,
}

configs = [
    triton.testing.Benchmark(
        x_names=["V"],
        x_vals=[2**i for i in range(10, 20)],
        # x_vals=[2**i for i in range(10, 14)],
        xlabel="Vocab Size",
        ylabel="Execution Time(ms)",
        line_arg="provider",
        # line_vals=["torch", "owc_t", "owc_t_c", "o_t", "o_t_c", "triton"],
        # line_names=["Torch", "Owc_T", "Owc_T_C", "O_T", "O_T_C", "Triton"],
        # line_vals=["torch", "owc_t", "o_t", "om_tr", "pm_tr"],
        # line_names=["Torch", "Owc_T", "O_T", "Om_Tr", "Pm_Tr"],
        line_vals=["torch", "pm_tr", "pgm_tr"],
        line_names=["Torch", "Pm_Tr", "Pgm_Tr"],
        styles=[
            ("red", "-"),
            ("orange", "-"),
            ("green", "-"),
            ("blue", "-"),
            ("black", "-"),
            ("purple", "-"),
        ],
        plot_name=f"multinomial-{bench_type}-{mode}-batch{b}-dim{d}-num{num_samples}-{dtype_name}",
        args={
            "b": b,
            "d": d,
            "num_samples": num_samples,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
            "bench_type": bench_type,
        },
    )
    for mode in [
        "fwd",
    ]
    for dtype_name in ["bf16"]
    for bench_type in ["speed", "memory"]
]


@triton.testing.perf_report(configs)
def benchmark(
    b, d, V, num_samples, dtype, device, mode, provider, dim=-2, bench_type="speed"
):
    torch.manual_seed(2024)
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    x = torch.randn((b, d), dtype=dtype, device=device)
    weight = torch.randn((d, V), dtype=dtype, device=device)

    module = module_map[provider]

    fn = lambda: module(x, weight, num_samples)
    if mode == "bwd":
        y = fn()
        dy = torch.randn((b, n, d), dtype=dtype, device=device)
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
        except Exception as e:
            print(f"Error setting up {provider}: {e}")
            mb = -1

        return mb


start_time = time.time()
save_path = "stat/multinomial"
os.makedirs(save_path, exist_ok=True)
benchmark.run(save_path=save_path, print_data=True)
end_time = time.time()
total_time = end_time - start_time
print(f"Total time: {total_time} seconds")
