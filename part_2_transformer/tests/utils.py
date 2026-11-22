"""Shared helpers for the NumPy-vs-torch parity tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.nptorch import Tensor


def assert_close(
    testcase,
    actual,
    expected,
    atol: float = 1e-10,
    rtol: float = 1e-8,
    msg: str = "",
) -> float:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    prefix = f"{msg}: " if msg else ""
    testcase.assertEqual(
        actual.shape, expected.shape, f"{prefix}shape {actual.shape} != {expected.shape}"
    )
    if actual.size == 0:
        return 0.0
    if not np.all(np.isfinite(actual)):
        testcase.fail(f"{prefix}non-finite values in actual result")
    if not np.all(np.isfinite(expected)):
        testcase.fail(f"{prefix}non-finite values in expected result")
    abs_err = np.abs(actual - expected)
    tol = atol + rtol * np.abs(expected)
    max_err = float(abs_err.max())
    if np.any(abs_err > tol):
        flat = int(np.argmax(abs_err - tol))
        index = np.unravel_index(flat, actual.shape)
        testcase.fail(
            f"{prefix}max abs error {max_err:.3e} at {index}: "
            f"actual={actual[index]!r}, expected={expected[index]!r}, "
            f"atol={atol}, rtol={rtol}"
        )
    return max_err


def torch_tensor(array: np.ndarray, requires_grad: bool = True) -> torch.Tensor:
    return torch.tensor(array.copy(), requires_grad=requires_grad)


def copy_weights(torch_module, np_module) -> None:
    state = {
        key: value.detach().cpu().numpy()
        for key, value in torch_module.state_dict().items()
    }
    np_module.load_state_dict(state)


def np_param_grads(module) -> dict[str, np.ndarray | None]:
    result = {}
    for name, param in module.named_parameters():
        result[name] = None if param._grad is None else param._grad.copy()
    return result


def torch_param_grads(module) -> dict[str, np.ndarray | None]:
    result = {}
    for name, param in module.named_parameters():
        result[name] = None if param.grad is None else param.grad.detach().numpy().copy()
    return result


def assert_param_grads_close(
    testcase,
    torch_module,
    np_module,
    atol: float,
    rtol: float,
    msg: str = "",
) -> float:
    torch_grads = torch_param_grads(torch_module)
    np_grads = np_param_grads(np_module)
    testcase.assertEqual(sorted(torch_grads), sorted(np_grads), "parameter names differ")
    worst = 0.0
    for name in torch_grads:
        tg, ng = torch_grads[name], np_grads[name]
        if tg is None or ng is None:
            testcase.assertEqual(tg is None, ng is None, f"grad None mismatch for {name}")
            continue
        worst = max(
            worst,
            assert_close(
                testcase,
                ng,
                tg,
                atol=atol,
                rtol=rtol,
                msg=f"{msg} grad[{name}]",
            ),
        )
    return worst


def mha_parity(
    testcase,
    torch_module: torch.nn.MultiheadAttention,
    np_module,
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    *,
    need_weights: bool = False,
    attn_mask=None,
    key_padding_mask=None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
    atol: float = 1e-10,
    rtol: float = 1e-8,
) -> tuple[float, float]:
    """Run forward+backward in both frameworks and compare outputs/grads."""
    torch_module.eval()
    np_module.eval()
    torch_module.zero_grad(set_to_none=True)
    np_module.zero_grad(set_to_none=True)
    copy_weights(torch_module, np_module)

    def to_torch_mask(mask):
        return None if mask is None else torch.tensor(mask)

    tq = torch_tensor(query)
    tk = torch_tensor(key)
    tv = torch_tensor(value)
    nq = Tensor(query.copy(), requires_grad=True)
    nk = Tensor(key.copy(), requires_grad=True)
    nv = Tensor(value.copy(), requires_grad=True)

    out_t, weights_t = torch_module(
        tq,
        tk,
        tv,
        key_padding_mask=to_torch_mask(key_padding_mask),
        need_weights=need_weights,
        attn_mask=to_torch_mask(attn_mask),
        average_attn_weights=average_attn_weights,
        is_causal=is_causal,
    )
    out_n, weights_n = np_module(
        nq,
        nk,
        nv,
        key_padding_mask=key_padding_mask,
        need_weights=need_weights,
        attn_mask=attn_mask,
        average_attn_weights=average_attn_weights,
        is_causal=is_causal,
    )

    worst_out = assert_close(testcase, out_n.data, out_t.detach().numpy(), atol, rtol, "mha output")
    if need_weights:
        testcase.assertIsNotNone(weights_n)
        worst_out = max(
            worst_out,
            assert_close(
                testcase,
                weights_n.data,
                weights_t.detach().numpy(),
                atol,
                rtol,
                "mha weights",
            ),
        )
    else:
        testcase.assertIsNone(weights_n)

    np_dtype = np.float64 if out_t.dtype == torch.float64 else np.float32
    coeff = np.random.default_rng(123).standard_normal(out_t.shape).astype(np_dtype)
    coeff_t = torch.tensor(coeff, dtype=out_t.dtype)
    (out_t * coeff_t).sum().backward()
    (out_n * Tensor(coeff)).sum().backward()

    worst_grad = assert_close(
        testcase, nq.grad.data, tq.grad.detach().numpy(), atol, rtol, "mha grad query"
    )
    worst_grad = max(
        worst_grad,
        assert_close(
            testcase, nk.grad.data, tk.grad.detach().numpy(), atol, rtol, "mha grad key"
        ),
    )
    worst_grad = max(
        worst_grad,
        assert_close(
            testcase, nv.grad.data, tv.grad.detach().numpy(), atol, rtol, "mha grad value"
        ),
    )
    worst_grad = max(
        worst_grad,
        assert_param_grads_close(testcase, torch_module, np_module, atol, rtol, "mha"),
    )
    return worst_out, worst_grad
