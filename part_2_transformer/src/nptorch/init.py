"""Parameter initialization with the same distributions as ``torch.nn.init``.

Only what this project needs is implemented, but the formulas match torch
(xavier/kaiming/uniform/normal/constant).
"""

from __future__ import annotations

import math
import numpy as np

from .tensor import Tensor, get_rng


def _prod(seq) -> int:
    result = 1
    for value in seq:
        result *= int(value)
    return result


def calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
    shape = tensor.shape
    if len(shape) < 2:
        raise ValueError("fan_in/fan_out require a tensor with at least 2 dims")
    receptive_field_size = _prod(shape[2:]) if len(shape) > 2 else 1
    fan_in = shape[1] * receptive_field_size
    fan_out = shape[0] * receptive_field_size
    return fan_in, fan_out


def calculate_gain(nonlinearity: str, param: float | None = None) -> float:
    linear_fns = [
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    ]
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1.0
    if nonlinearity == "tanh":
        return 5.0 / 3.0
    if nonlinearity == "relu":
        return math.sqrt(2.0)
    if nonlinearity == "leaky_relu":
        negative_slope = 0.01 if param is None else float(param)
        return math.sqrt(2.0 / (1 + negative_slope**2))
    if nonlinearity == "selu":
        return 3.0 / 4.0
    raise ValueError(f"calculate_gain: unknown nonlinearity {nonlinearity!r}")


def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> Tensor:
    if tensor.dtype.kind != "f":
        raise ValueError("uniform_ only works with float tensors")
    tensor.data[...] = get_rng().uniform(a, b, size=tensor.shape).astype(tensor.dtype)
    return tensor


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    if tensor.dtype.kind != "f":
        raise ValueError("normal_ only works with float tensors")
    tensor.data[...] = get_rng().normal(mean, std, size=tensor.shape).astype(tensor.dtype)
    return tensor


def constant_(tensor: Tensor, val: float) -> Tensor:
    tensor.data[...] = np.asarray(val, dtype=tensor.dtype)
    return tensor


def zeros_(tensor: Tensor) -> Tensor:
    tensor.data[...] = 0
    return tensor


def ones_(tensor: Tensor) -> Tensor:
    tensor.data[...] = 1
    return tensor


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound)


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return normal_(tensor, 0.0, std)


def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> Tensor:
    if mode not in ("fan_in", "fan_out"):
        raise ValueError("kaiming_uniform_: mode must be 'fan_in' or 'fan_out'")
    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound)


def kaiming_normal_(
    tensor: Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> Tensor:
    if mode not in ("fan_in", "fan_out"):
        raise ValueError("kaiming_normal_: mode must be 'fan_in' or 'fan_out'")
    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return normal_(tensor, 0.0, std)
