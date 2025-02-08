import torch
import torch._dynamo as dynamo
from xmixers.modules import GLU

device = torch.device("cuda")

b = 4
d = 128
act = "silu"

module = GLU(d, d, act, use_gate_linear=True).to(device)
x = torch.randn(b, d, device=device)

explanation = dynamo.explain(module)(x)
print(explanation)

fn = torch.compile(module)

o = fn(x)
print("=====output=====", o.mean())
