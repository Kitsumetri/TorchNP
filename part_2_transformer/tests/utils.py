"""Shared helpers for the NumPy-vs-torch gradient tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.nptorch import TransformerEncoderLayer


def assert_close(
    testcase,
    actual,
    expected,
    atol: float = 2e-12,
    rtol: float = 1e-10,
    msg: str = "",
) -> float:
    """Assert closeness and return the maximum absolute difference."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    testcase.assertEqual(
        actual.shape,
        expected.shape,
        f"{msg}: shape {actual.shape} != {expected.shape}",
    )
    if actual.size == 0:
        return 0.0
    diff = float(np.max(np.abs(actual - expected)))
    np.testing.assert_allclose(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
        err_msg=f"{msg}: max|diff|={diff:.3e}",
    )
    return diff


def make_pair(
    d_model: int,
    nhead: int,
    dim_feedforward: int,
    *,
    dropout: float = 0.0,
    activation: str = "relu",
    layer_norm_eps: float = 1e-5,
    batch_first: bool = False,
    norm_first: bool = False,
    bias: bool = True,
) -> tuple[TransformerEncoderLayer, torch.nn.TransformerEncoderLayer]:
    """Build a NumPy layer and a torch layer with identical parameters."""
    torch_layer = torch.nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        layer_norm_eps=layer_norm_eps,
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
        layer_norm_eps=layer_norm_eps,
        batch_first=batch_first,
        norm_first=norm_first,
        bias=bias,
    ).eval()
    numpy_layer.load_state_dict(torch_layer.state_dict())
    return numpy_layer, torch_layer


def parity_check(
    testcase,
    *,
    x_seq: np.ndarray,
    d_model: int,
    nhead: int,
    dim_feedforward: int,
    src_mask=None,
    src_key_padding_mask=None,
    is_causal: bool = False,
    batch_first: bool = False,
    norm_first: bool = False,
    bias: bool = True,
    activation: str = "relu",
    layer_norm_eps: float = 1e-5,
    seed: int = 0,
    atol: float = 2e-12,
    rtol: float = 1e-10,
) -> dict[str, float]:
    """Compare forward output, input gradient and every parameter gradient.

    ``x_seq`` always has the canonical sequence-first shape ``(L, N, E)``;
    the helper transposes it for ``batch_first`` and squeezes ``N`` for
    unbatched 2-D input.
    """
    numpy_layer, torch_layer = make_pair(
        d_model,
        nhead,
        dim_feedforward,
        dropout=0.0,
        activation=activation,
        layer_norm_eps=layer_norm_eps,
        batch_first=batch_first,
        norm_first=norm_first,
        bias=bias,
    )

    unbatched = x_seq.ndim == 2
    if unbatched:
        x_seq = x_seq[:, None, :]
    raw = torch.tensor(x_seq, dtype=torch.float64, requires_grad=True)
    if unbatched:
        torch_input = raw[:, 0, :]
        numpy_input = x_seq[:, 0, :]
    elif batch_first:
        torch_input = raw.transpose(0, 1)
        numpy_input = x_seq.transpose(1, 0, 2)
    else:
        torch_input = raw
        numpy_input = x_seq

    numpy_out = numpy_layer(
        numpy_input,
        src_mask=src_mask,
        src_key_padding_mask=src_key_padding_mask,
        is_causal=is_causal,
    )
    torch_out = torch_layer(
        torch_input,
        src_mask=torch.as_tensor(src_mask)
        if src_mask is not None
        else None,
        src_key_padding_mask=torch.as_tensor(src_key_padding_mask)
        if src_key_padding_mask is not None
        else None,
        is_causal=is_causal,
    )

    rng = np.random.default_rng(seed)
    dout = rng.normal(size=numpy_out.shape)
    numpy_dx = numpy_layer.backward(dout)
    torch_out.backward(torch.tensor(dout, dtype=torch.float64))

    expected_dx = raw.grad.numpy()
    if unbatched:
        expected_dx = expected_dx[:, 0, :]
    elif batch_first:
        expected_dx = expected_dx.transpose(1, 0, 2)

    result = {
        "forward": assert_close(
            testcase, numpy_out, torch_out.detach().numpy(), atol=atol, rtol=rtol,
            msg="forward",
        ),
        "d_src": assert_close(
            testcase, numpy_dx, expected_dx, atol=atol, rtol=rtol, msg="d_src"
        ),
    }

    torch_params = dict(torch_layer.named_parameters())
    testcase.assertEqual(
        set(numpy_layer.grads), set(torch_params), "parameter-name mismatch"
    )
    worst = 0.0
    for name, parameter in torch_params.items():
        worst = max(
            worst,
            assert_close(
                testcase,
                numpy_layer.grads[name],
                parameter.grad.numpy(),
                atol=atol,
                rtol=rtol,
                msg=f"d[{name}]",
            ),
        )
    result["param_grads"] = worst
    return result


def scalar_loss(
    layer: TransformerEncoderLayer,
    x: np.ndarray,
    dout: np.ndarray,
    *,
    src_mask=None,
    src_key_padding_mask=None,
    is_causal: bool = False,
) -> float:
    """Return ``sum(forward(x) * dout)``; its gradient w.r.t. output is dout."""
    out = layer.forward(
        x,
        src_mask=src_mask,
        src_key_padding_mask=src_key_padding_mask,
        is_causal=is_causal,
    )
    return float(np.sum(out * dout))
