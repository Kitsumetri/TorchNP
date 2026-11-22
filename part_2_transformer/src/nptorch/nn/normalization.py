"""LayerNorm with torch-compatible state dict and biased variance."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .. import functional as F
from ..tensor import Parameter
from .module import Module, resolve_dtype


class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape: int | Sequence[int],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: Any | None = None,
        dtype: Any | None = None,
    ):
        super().__init__()
        if device is not None and str(device) not in ("cpu", "None"):
            raise ValueError("nptorch is CPU-only")
        if isinstance(normalized_shape, (int, np.integer)):
            normalized_shape = (int(normalized_shape),)
        self.normalized_shape = tuple(int(s) for s in normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        param_dtype = resolve_dtype(dtype)
        if self.elementwise_affine:
            self.weight = Parameter(np.ones(self.normalized_shape, dtype=param_dtype))
            if bias:
                self.bias = Parameter(np.zeros(self.normalized_shape, dtype=param_dtype))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, input):
        return F.layer_norm(
            input,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        )

    def extra_repr(self) -> str:
        return (
            f"normalized_shape={self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )
