from .group_norm import group_norm_torch, group_norm_triton
from .group_rms_norm import group_rms_norm_torch, group_rms_norm_triton
from .group_srms_norm import group_srms_norm_torch, group_srms_norm_triton
from .layer_norm import layer_norm_torch, layer_norm_triton
from .normalize import normalize_torch, normalize_triton
from .rms_norm import rms_norm_torch, rms_norm_triton
from .srms_norm import srms_norm_torch, srms_norm_triton

normalize_fn = normalize_triton
layer_norm_fn = layer_norm_triton
rms_norm_fn = rms_norm_triton
srms_norm_fn = srms_norm_triton
group_norm_fn = group_norm_triton
group_rms_norm_fn = group_rms_norm_triton
group_srms_norm_fn = group_srms_norm_triton
