from .groupnorm import groupnorm_torch, groupnorm_triton
from .grouprmsnorm import grouprmsnorm_torch, grouprmsnorm_triton
from .groupsrmsnorm import groupsrmsnorm_torch, groupsrmsnorm_triton
from .layernorm import layernorm_torch, layernorm_triton
from .normalize import normalize_torch, normalize_triton
from .rmsnorm import rmsnorm_torch, rmsnorm_triton
from .srmsnorm import srmsnorm_torch, srmsnorm_triton

normalize_fn = normalize_triton
layernorm_fn = layernorm_triton
rmsnorm_fn = rmsnorm_triton
srmsnorm_fn = srmsnorm_triton
groupnorm_fn = groupnorm_triton
grouprmsnorm_fn = grouprmsnorm_triton
groupsrmsnorm_fn = groupsrmsnorm_triton
