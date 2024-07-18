import pytest
import torch

from xopes.ops import base_rule_recurrence_torch, base_rule_recurrence_triton


def get_params():
    array = [
        (6, 8, 128, 128, 64),
        # (6, 8, 1, 128, 128),
        # (6, 8, 512, 128, 64),
        # (6, 8, 1024, 128, 64),
        # (6, 8, 2048, 128, 64),
        # (6, 8, 4096, 128, 64),
        # (6, 8, 8192, 128, 64),
        # (6, 8, 2048, 32, 64),
        # (6, 8, 2048, 64, 64),
        # (6, 12, 2048, 128, 64),
        # (6, 16, 2048, 128, 64),
        # (6, 20, 2048, 128, 64),
        # (1, 8, 2048, 128, 64),
        # (2, 8, 2048, 128, 64),
        # (3, 8, 2048, 128, 64),
        # (6, 8, 913, 128, 64),
        # (6, 8, 513, 128, 64),
        # (6, 8, 1213, 128, 64),
        # (6, 8, 2048, 16, 64),
        # (1, 32, 55296, 128, 128),
    ]

    return array


@pytest.mark.parametrize("b, h, n, d, e", get_params())
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_lightning2(b, h, n, d, e, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    q = (torch.randn((b, h, n, d), dtype=dtype, device=device)).requires_grad_()
    k = (torch.randn((b, h, n, d), dtype=dtype, device=device)).requires_grad_()
    v = (torch.randn((b, h, n, e), dtype=dtype, device=device)).requires_grad_()
    do = torch.randn((b, h, n, e), dtype=dtype, device=device)

    # forward
    o_recurrence_torch = base_rule_recurrence_torch(q, k, v)

    o_recurrence_triton = base_rule_recurrence_triton(q, k, v)

    print("here", torch.mean(o_recurrence_torch), torch.mean(o_recurrence_triton))

    # # backward
    # o_ref.backward(do, retain_graph=True)
    # dq_ref, q.grad = q.grad.clone(), None
    # dk_ref, k.grad = k.grad.clone(), None
    # dv_ref, v.grad = v.grad.clone(), None

    # o.backward(do, retain_graph=True)
    # dq, q.grad = q.grad.clone(), None
    # dk, k.grad = k.grad.clone(), None
    # dv, v.grad = v.grad.clone(), None

    print(torch.norm(o_recurrence_torch - o_recurrence_triton))
    # print(torch.norm(dq - dq_ref))
    # print(torch.norm(dk - dk_ref))
    # print(torch.norm(dv - dv_ref))
    assert False
