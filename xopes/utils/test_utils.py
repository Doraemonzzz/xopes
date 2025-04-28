import torch


def get_abs_err(x, y):
    return (x.detach() - y.detach()).abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach() - y.detach()).norm(p=2).item()
    base = (x.detach()).norm(p=2).item()
    return err / (base + 1e-8)


def assert_close(ref, input, atol, rtol):
    abs_atol = get_abs_err(ref, input)
    error_rate = get_err_ratio(ref, input)

    if abs_atol <= atol:
        return
    if error_rate <= rtol:
        return
    assert (
        False
    ), f"abs_atol: {abs_atol} > atol: {atol} and error_rate: {error_rate} > rtol: {rtol}"


def print_diff(o1, o2, n, BLOCK_C=16):
    l = (n + BLOCK_C - 1) // BLOCK_C
    for i in range(l):
        start = i * BLOCK_C
        end = min(start + BLOCK_C, n)
        print(
            start,
            end,
            torch.norm(o1[:, start:end, :, :] - o2[:, start:end, :, :]).item(),
        )
