from typing import List

import torch
import torch._dynamo as dynamo
from xmixers.modules.normalizations.rms_norm import RMSNorm

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True


def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward  # return a python callable


device = torch.device("cuda")


b = 4
d = 128

module = RMSNorm(d).to(device)
x = torch.randn(b, d, device=device)

explanation = dynamo.explain(module)(x)
print(explanation)

fn = torch.compile(module)

o = fn(x)
print("=====output=====", o.mean())
