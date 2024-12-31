from .normalize_torch import normalize_torch
from .normalize_triton import normalize_triton
from .rmsnorm_torch import rmsnorm_torch
from .rmsnorm_triton import rmsnorm_triton
from .srmsnorm_torch import srmsnorm_torch
from .srmsnorm_triton import srmsnorm_triton

normalize_fn = normalize_triton
srmsnorm_fn = srmsnorm_triton
rmsnorm_fn = rmsnorm_triton
