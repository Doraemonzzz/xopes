# from .cosine import md_lrpe_cosine_bwd, md_lrpe_cosine_fn, md_lrpe_cosine_fwd

# MD_LRPE_DICT = {
#     "cosine": md_lrpe_cosine_fn,
# }

# MD_LRPE_FWD_DICT = {
#     "cosine": md_lrpe_cosine_fwd,
# }

# MD_LRPE_BWD_DICT = {
#     "cosine": md_lrpe_cosine_bwd,
# }


# def md_lrpe_fn(x, theta, shape, l=0, lrpe_type="cosine", **kwargs):
#     fn = MD_LRPE_DICT[lrpe_type]
#     return fn(x, theta, shape, l)


# def md_lrpe_fwd(x, theta, shape, l=0, lrpe_type="cosine", **kwargs):
#     fn = MD_LRPE_FWD_DICT[lrpe_type]
#     return fn(x, theta, shape, l)


# def md_lrpe_bwd(x, theta, do, shape, l=0, lrpe_type="cosine", **kwargs):
#     fn = MD_LRPE_BWD_DICT[lrpe_type]
#     return fn(x, theta, do, shape, l)
