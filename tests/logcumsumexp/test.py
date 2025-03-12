import pytest
import torch

from xopes.ops.logcumsumexp import lcse_recurrence_triton, lcse_torch
from xopes.utils import get_threshold


def get_params():
    shapes = [
        (512,),
        (6, 128),
        (6, 129),
        (4, 4, 255),
        (4, 8, 256),
    ]

    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("use_initial_state", [True, False])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
    ],
)
def test(shape, dim, use_initial_state, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    scale = 5
    if len(shape) == 1:
        dim = 0

    # Generate input tensor
    x = (torch.randn(shape, dtype=dtype, device=device) * scale).requires_grad_()

    # Generate initial state if needed
    if use_initial_state:
        # Create initial state with shape matching x except for dim dimension
        initial_shape = list(shape)
        initial_shape[dim] = 1
        initial_state = (
            torch.randn(initial_shape, dtype=dtype, device=device) * scale
        ).requires_grad_()
    else:
        initial_state = None

    # Forward pass
    o_torch, state_torch = lcse_torch(x, dim=dim, initial_state=initial_state)
    o_torch_sum = o_torch + state_torch.sum()
    o_triton_recurrence, state_triton_recurrence = lcse_recurrence_triton(
        x, dim=dim, initial_state=initial_state
    )
    o_triton_recurrence_sum = o_triton_recurrence + state_triton_recurrence.sum()
    # Generate gradients for backward pass
    do = torch.randn_like(o_torch)

    # Backward pass for torch implementation
    o_torch_sum.backward(do, retain_graph=True)
    dx_torch, x.grad = x.grad.clone(), None
    if use_initial_state:
        dinitial_state_torch, initial_state.grad = initial_state.grad.clone(), None

    # Backward pass for triton implementation
    o_triton_recurrence_sum.backward(do, retain_graph=True)
    dx_triton_recurrence, x.grad = x.grad.clone(), None
    if use_initial_state:
        dinitial_state_triton_recurrence, initial_state.grad = (
            initial_state.grad.clone(),
            None,
        )

    # Get tolerance thresholds based on dtype
    atol, rtol = get_threshold(dtype)

    # Forward check for output
    print(
        "o diff max (Vs recurrence triton): ",
        torch.abs(o_torch - o_triton_recurrence).max().item(),
    )
    print(
        "o diff norm (Vs recurrence triton): ",
        torch.norm(o_torch - o_triton_recurrence).item(),
    )
    assert torch.allclose(o_torch, o_triton_recurrence, atol=atol, rtol=rtol)

    # Forward check for state
    print(
        "state diff max (Vs recurrence triton): ",
        torch.abs(state_torch - state_triton_recurrence).max().item(),
    )
    print(
        "state diff norm (Vs recurrence triton): ",
        torch.norm(state_torch - state_triton_recurrence).item(),
    )
    assert torch.allclose(state_torch, state_triton_recurrence, atol=atol, rtol=rtol)

    # Backward check for input gradients
    print(
        "dx diff max (Vs recurrence triton): ",
        torch.abs(dx_torch - dx_triton_recurrence).max().item(),
    )
    print(
        "dx diff norm (Vs recurrence triton): ",
        torch.norm(dx_torch - dx_triton_recurrence).item(),
    )
    assert torch.allclose(dx_torch, dx_triton_recurrence, atol=atol, rtol=rtol)

    # Backward check for initial state gradients if used
    if use_initial_state:
        print(
            "dinitial_state diff max (Vs recurrence triton): ",
            torch.abs(dinitial_state_torch - dinitial_state_triton_recurrence)
            .max()
            .item(),
        )
        print(
            "dinitial_state diff norm (Vs recurrence triton): ",
            torch.norm(dinitial_state_torch - dinitial_state_triton_recurrence).item(),
        )
        assert torch.allclose(
            dinitial_state_torch, dinitial_state_triton_recurrence, atol=atol, rtol=rtol
        )
