from .cosine import lrpe_cosine_bwd_fn, lrpe_cosine_fn, lrpe_cosine_fwd_fn
from .rotate import lrpe_rotate_bwd_fn, lrpe_rotate_fn, lrpe_rotate_fwd_fn

LRPE_DICT = {
    "cosine": lrpe_cosine_fn,
    "rotate": lrpe_rotate_fn,
}

LRPE_FWD_DICT = {
    "cosine": lrpe_cosine_fwd_fn,
    "rotate": lrpe_rotate_fwd_fn,
}

LRPE_BWD_DICT = {
    "cosine": lrpe_cosine_bwd_fn,
    "rotate": lrpe_rotate_bwd_fn,
}


def lrpe_fn(x, theta, offset=0, act="none", dim=None, lrpe_type="cosine", **kwargs):
    fn = LRPE_DICT[lrpe_type]
    return fn(x=x, theta=theta, offset=offset, act=act, dim=dim)


def lrpe_fwd_fn(x, theta, offset=0, act="none", dim=None, lrpe_type="cosine", **kwargs):
    fn = LRPE_FWD_DICT[lrpe_type]
    return fn(x=x, theta=theta, offset=offset, act=act, dim=dim)


def lrpe_bwd_fn(
    x, theta, do, offset=0, act="none", dim=None, lrpe_type="cosine", **kwargs
):
    fn = LRPE_BWD_DICT[lrpe_type]
    return fn(x=x, theta=theta, do=do, offset=offset, act=act, dim=dim)
