"""Demo: one NumPy transformer encoder layer vs its torch twin.

The script has two parts:

1. forward/backward parity for a single layer (output, input gradient and all
   parameter gradients) on several configurations and masks;
2. synchronized SGD training of the NumPy layer and the torch layer, with the
   same data, initial weights and updates.

Run from ``part_2_transformer``::

    ../.venv/bin/python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.nptorch import TransformerEncoderLayer


def build_pair(
    d_model: int = 8,
    nhead: int = 2,
    dim_feedforward: int = 16,
    *,
    seed: int = 0,
    dropout: float = 0.0,
    activation: str = "relu",
    batch_first: bool = False,
    norm_first: bool = False,
    bias: bool = True,
) -> tuple[TransformerEncoderLayer, torch.nn.TransformerEncoderLayer]:
    """Create a NumPy layer and a torch layer with identical parameters."""
    torch.manual_seed(seed)
    torch_layer = torch.nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        batch_first=batch_first,
        norm_first=norm_first,
        bias=bias,
        dtype=torch.float64,
    ).eval()
    numpy_layer = TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        batch_first=batch_first,
        norm_first=norm_first,
        bias=bias,
    ).eval()
    numpy_layer.load_state_dict(torch_layer.state_dict())
    return numpy_layer, torch_layer


def maxdiff(actual: np.ndarray, expected: np.ndarray) -> float:
    """Maximum absolute elementwise difference."""
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def make_data(
    seed: int = 7,
    seq_len: int = 6,
    batch_size: int = 8,
    d_model: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a smooth token-wise regression target for the training demo."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(seq_len, batch_size, d_model))
    projection = rng.normal(size=(d_model, d_model)) / np.sqrt(d_model)
    target = np.tanh(x @ projection + 0.2)
    return x, target


def gradient_report(
    numpy_layer: TransformerEncoderLayer,
    torch_layer: torch.nn.TransformerEncoderLayer,
    x: np.ndarray,
    *,
    src_mask=None,
    src_key_padding_mask=None,
    is_causal: bool = False,
    seed: int = 1,
) -> dict[str, float]:
    """Compare output, input gradient and all parameter gradients."""
    numpy_out = numpy_layer(
        x,
        src_mask=src_mask,
        src_key_padding_mask=src_key_padding_mask,
        is_causal=is_causal,
    )
    torch_input = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    torch_out = torch_layer(
        torch_input,
        src_mask=torch.as_tensor(src_mask) if src_mask is not None else None,
        src_key_padding_mask=torch.as_tensor(src_key_padding_mask)
        if src_key_padding_mask is not None
        else None,
        is_causal=is_causal,
    )
    forward_diff = maxdiff(numpy_out, torch_out.detach().numpy())

    cotangent = np.random.default_rng(seed).normal(size=numpy_out.shape)
    numpy_layer.zero_grad()
    numpy_dx = numpy_layer.backward(cotangent)
    torch_out.backward(torch.tensor(cotangent, dtype=torch.float64))

    input_grad_diff = maxdiff(numpy_dx, torch_input.grad.numpy())
    parameter_grad_diff = 0.0
    for name, parameter in torch_layer.named_parameters():
        parameter_grad_diff = max(
            parameter_grad_diff,
            maxdiff(numpy_layer.grads[name], parameter.grad.numpy()),
        )
    return {
        "forward": forward_diff,
        "d_src": input_grad_diff,
        "d_params": parameter_grad_diff,
    }


def train_twin(
    numpy_layer: TransformerEncoderLayer,
    torch_layer: torch.nn.TransformerEncoderLayer,
    x: np.ndarray,
    target: np.ndarray,
    *,
    steps: int = 80,
    learning_rate: float = 0.1,
    report_every: int = 10,
) -> tuple[list[tuple[int, float, float]], dict[str, float]]:
    """Train both layers with synchronized plain SGD.

    The scalar loss is mean squared error.  The NumPy gradient comes from the
    hand-written ``backward``; the torch gradient comes from autograd.  Both
    sides then apply exactly the same SGD update.
    """
    torch_target = torch.tensor(target, dtype=torch.float64)
    torch_x = torch.tensor(x, dtype=torch.float64)
    history: list[tuple[int, float, float]] = []
    max_grad_diff = 0.0
    max_parameter_diff = 0.0

    for step in range(steps):
        output = numpy_layer(x)
        difference = output - target
        loss = float(np.mean(difference * difference))
        dout = 2.0 * difference / difference.size
        numpy_layer.zero_grad()
        numpy_layer.backward(dout)

        torch_output = torch_layer(torch_x)
        torch_loss = torch.nn.functional.mse_loss(torch_output, torch_target)
        torch_layer.zero_grad(set_to_none=True)
        torch_loss.backward()

        torch_parameters = dict(torch_layer.named_parameters())
        for name, parameter in torch_parameters.items():
            max_grad_diff = max(
                max_grad_diff,
                maxdiff(numpy_layer.grads[name], parameter.grad.numpy()),
            )

        with torch.no_grad():
            for parameter in torch_parameters.values():
                parameter -= learning_rate * parameter.grad
        for name, parameter in numpy_layer.params.items():
            parameter -= learning_rate * numpy_layer.grads[name]

        for name, parameter in torch_parameters.items():
            max_parameter_diff = max(
                max_parameter_diff,
                maxdiff(numpy_layer.params[name], parameter.detach().numpy()),
            )

        history.append((step, loss, float(torch_loss.detach())))
        if step % report_every == 0 or step == steps - 1:
            print(
                f"step {step:3d}  numpy_loss={loss:.6f}  "
                f"torch_loss={float(torch_loss.detach()):.6f}"
            )
        numpy_layer.zero_grad()

    return history, {
        "max_grad_diff": max_grad_diff,
        "max_parameter_diff": max_parameter_diff,
    }


def parity_demo() -> None:
    """Print forward/backward parity for several layer configurations."""
    print("\n1. Forward / backward parity with torch")
    print("-" * 78)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 3, 8))

    cases = [
        ("post-norm", {}),
        ("pre-norm", {"norm_first": True}),
        ("gelu", {"activation": "gelu"}),
        ("no bias", {"bias": False}),
    ]
    print(f"{'configuration':<14} {'forward':>10} {'d_src':>10} {'d_params':>10}")
    for label, options in cases:
        numpy_layer, torch_layer = build_pair(**options)
        report = gradient_report(numpy_layer, torch_layer, x)
        print(
            f"{label:<14} {report['forward']:>10.2e} "
            f"{report['d_src']:>10.2e} {report['d_params']:>10.2e}"
        )

    numpy_layer, torch_layer = build_pair()
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 1] = True
    mask[2, 3] = True
    key_padding = np.zeros((3, 4), dtype=bool)
    key_padding[1, 0] = True
    report = gradient_report(
        numpy_layer,
        torch_layer,
        x,
        src_mask=mask,
        src_key_padding_mask=key_padding,
    )
    print(
        f"{'masks':<14} {report['forward']:>10.2e} "
        f"{report['d_src']:>10.2e} {report['d_params']:>10.2e}"
    )

    causal = np.triu(np.ones((4, 4), dtype=bool), 1)
    numpy_layer, torch_layer = build_pair(batch_first=True)
    x_batch_first = x.transpose(1, 0, 2)
    report = gradient_report(
        numpy_layer,
        torch_layer,
        x_batch_first,
        src_mask=causal,
        is_causal=True,
    )
    print(
        f"{'causal':<14} {report['forward']:>10.2e} "
        f"{report['d_src']:>10.2e} {report['d_params']:>10.2e}"
    )


def training_demo() -> None:
    """Train one layer on both implementations and compare the trajectories."""
    print("\n2. Synchronized training of one layer")
    print("-" * 78)
    numpy_layer, torch_layer = build_pair()
    x, target = make_data()
    history, summary = train_twin(numpy_layer, torch_layer, x, target)
    initial = history[0][1]
    final = history[-1][1]
    print(
        f"\nloss: {initial:.6f} -> {final:.6f} "
        f"({100.0 * final / initial:.1f}% of the initial value)"
    )
    print(f"max |gradient diff| during training: {summary['max_grad_diff']:.2e}")
    print(f"max |parameter diff| after updates:  {summary['max_parameter_diff']:.2e}")

    state_keys_match = set(numpy_layer.state_dict()) == set(
        torch_layer.state_dict()
    )
    print(f"state_dict keys match torch: {state_keys_match}")


def main() -> None:
    print(f"numpy {np.__version__}, torch {torch.__version__}")
    print(
        "layer: d_model=8, nhead=2, dim_feedforward=16, "
        "dropout=0.0, float64"
    )
    parity_demo()
    training_demo()


if __name__ == "__main__":
    main()
