from einops import rearrange

try:
    from flash_attn import flash_attn_func
except:
    flash_attn_func = lambda x: None

try:
    from fla.ops.gla import chunk_gla
except:
    chunk_gla = lambda x: None

try:
    from fla.ops.simple_gla import chunk_simple_gla
except:
    chunk_simple_gla = lambda x: None

try:
    from fla.ops.linear_attn import chunk_linear_attn
except:
    chunk_linear_attn = lambda x: None

try:
    from lightning_attn.ops import lightning_attn_func
except:
    lightning_attn_func = lambda x: None

try:
    from fla.ops.common.chunk_h import chunk_fwd_h
except:
    chunk_fwd_h = lambda x: None

try:
    from fla.ops.hgrn import chunk_hgrn
except:
    chunk_hgrn = lambda x: None

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except:
    mamba_chunk_scan_combined = lambda x: None

try:
    from mlstm_kernels.torch.chunkwise.triton_xl_chunk import mlstm_chunkwise__xl_chunk
except:
    mlstm_chunkwise__xl_chunk = lambda x: None

from xopes.ops.cumsum import chunk_cumsum_decay_fn


def flash_attn_wrapper(q, k, v, **kwargs):
    o = flash_attn_func(
        q,
        k,
        v,
        causal=True,
    )

    return o


def chunk_gla_wrapper(q, k, v, **kwargs):
    o = chunk_gla(
        q,
        k,
        v,
        g=kwargs["ldk"],
        head_first=False,
        output_final_state=True,
    )
    return o


def chunk_simple_gla_wrapper(q, k, v, **kwargs):
    o = chunk_simple_gla(
        q,
        k,
        v,
        g=kwargs["ld3"],
        head_first=False,
        output_final_state=True,
    )
    return o


def chunk_linear_attn_wrapper(q, k, v, **kwargs):
    o = chunk_linear_attn(
        q,
        k,
        v,
        head_first=False,
        output_final_state=True,
    )
    return o


def lightning_attn_wrapper(q, k, v, **kwargs):
    q, k, v = map(lambda x: rearrange(x, "b n h d -> b h n d"), (q, k, v))
    o = lightning_attn_func(
        q,
        k,
        v,
        s=-kwargs["ld"],
        variant=kwargs["variant"],
    )
    o = rearrange(o, "b h n d -> b n h d")
    return o


def lightning_attn_no_decay_wrapper(q, k, v, **kwargs):
    q, k, v = map(lambda x: rearrange(x, "b n h d -> b h n d"), (q, k, v))
    o = lightning_attn_func(
        q,
        k,
        v,
        variant=kwargs["variant"],
    )
    o = rearrange(o, "b h n d -> b n h d")
    return o


def state_fla_wrapper(k, v, ldk=None, ldv=None, chunk_size=128, **kwargs):
    if ldk is not None:
        ldk = chunk_cumsum_decay_fn(k, chunk_size=chunk_size, **kwargs)
    if ldv is not None:
        ldv = chunk_cumsum_decay_fn(v, chunk_size=chunk_size, **kwargs)

    o = chunk_fwd_h(
        k=k,
        v=v,
        g=None,
        gk=ldk,
        gv=ldv,
        h0=None,
        output_final_state=True,
        chunk_size=chunk_size,
        states_in_fp32=False,
        head_first=False,
    )

    return o


def chunk_hgrn_fla_wrapper(q, k, v, **kwargs):
    ld = kwargs["ldk"]
    x = k * v
    o, state = chunk_hgrn(
        x=x,
        g=ld,
        output_final_state=True,
    )
    o = q * o

    return o, state


def mamba2_wrapper(x, dt, A, B, C, **kwargs):
    o = mamba_chunk_scan_combined(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        chunk_size=64,
    )

    return o


def mlstm_wrapper(q, k, v, **kwargs):
    o = mlstm_chunkwise__xl_chunk(
        q=q,
        k=k,
        v=v,
        i=kwargs["i"],
        f=kwargs["ld3"],
        return_last_states=False,
        chunk_size=256,
    )

    return o
