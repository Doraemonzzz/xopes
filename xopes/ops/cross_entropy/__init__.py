from .baseline import cross_entropy_fla_wrapper
from .ce_parallel_triton import cross_entropy_parallel_triton
from .ce_torch import cross_entropy_torch
from .ce_triton import cross_entropy_triton

cross_entropy_fn = cross_entropy_parallel_triton
