import torch
import torch._dynamo as dynamo
from xmixers.modules.normalizations.rms_norm import RMSNorm

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
