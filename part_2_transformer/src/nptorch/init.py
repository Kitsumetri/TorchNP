"""In-place initialization helpers with torch-compatible formulas.

The formulas match ``torch.nn.init`` for the argument values used by
``TransformerEncoderLayer``.  Arrays are float64 and are modified in place.
"""

from __future__ import annotations

import math

import numpy as np


def _check_array(array: np.ndarray, name: str = "tensor") -> np.ndarray:
    array = np.asarray(array)
    if array.ndim < 2:
        raise ValueError(f"{name} must have at least 2 dimensions")
    return array


def calculate_fan_in_and_fan_out(array: np.ndarray) -> tuple[int, int]:
    """Return ``(fan_in, fan_out)`` using torch's convention.

    For a 2-D weight of shape ``(out_features, in_features)`` the fan-in is
    ``in_features`` and the fan-out is ``out_features``.  Dimensions after the
    first two are treated as receptive-field dimensions.
    """
    array = _check_array(array)
    num_input_fmaps = array.shape[1]
    num_output_fmaps = array.shape[0]
    receptive_field_size = int(np.prod(array.shape[2:], dtype=np.int64))
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def uniform_(array: np.ndarray, low: float, high: float) -> np.ndarray:
    """Fill ``array`` with samples from ``U(low, high)``."""
    array[...] = np.random.uniform(low, high, size=array.shape)
    return array


def constant_(array: np.ndarray, value: float) -> np.ndarray:
    """Fill ``array`` with ``value``."""
    array[...] = value
    return array


def zeros_(array: np.ndarray) -> np.ndarray:
    """Fill ``array`` with zeros."""
    array[...] = 0.0
    return array


def ones_(array: np.ndarray) -> np.ndarray:
    """Fill ``array`` with ones."""
    array[...] = 1.0
    return array


def xavier_uniform_(array: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Xavier/Glorot uniform initialization.

    ``bound = gain * sqrt(6 / (fan_in + fan_out))`` and every element is drawn
    from ``U(-bound, bound)``.
    """
    fan_in, fan_out = calculate_fan_in_and_fan_out(array)
    bound = gain * math.sqrt(6.0 / float(fan_in + fan_out))
    return uniform_(array, -bound, bound)


def kaiming_uniform_(array: np.ndarray, a: float = math.sqrt(5)) -> np.ndarray:
    """Kaiming/He uniform initialization for a leaky-ReLU slope ``a``.

    ``bound = sqrt(6 / ((1 + a**2) * fan_in))`` and every element is drawn
    from ``U(-bound, bound)``.  With the torch default ``a = sqrt(5)`` this
    becomes ``1 / sqrt(fan_in)``.
    """
    fan_in, _ = calculate_fan_in_and_fan_out(array)
    bound = math.sqrt(6.0 / ((1.0 + a * a) * float(fan_in)))
    return uniform_(array, -bound, bound)
