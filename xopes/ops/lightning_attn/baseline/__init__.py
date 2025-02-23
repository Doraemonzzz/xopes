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
    from lightning_attn.ops import lightning_attn_func
except:
    lightning_attn_func = lambda x: None


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
