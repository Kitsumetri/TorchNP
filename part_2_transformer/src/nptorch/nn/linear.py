"""Linear layer with torch-compatible parameter names and initialization."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .. import functional as F
from .. import init
from ..tensor import Parameter
from .module import Module, resolve_dtype


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Any | None = None,
        dtype: Any | None = None,
    ):
        super().__init__()
        if device is not None and str(device) not in ("cpu", "None"):
            raise ValueError("nptorch is CPU-only")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        param_dtype = resolve_dtype(dtype)
        self.weight = Parameter(
            np.empty((self.out_features, self.in_features), dtype=param_dtype)
        )
        if bias:
            self.bias = Parameter(np.empty(self.out_features, dtype=param_dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Same as torch.nn.Linear: kaiming_uniform_(a=sqrt(5)) for the weight,
        # which is equivalent to U(-1/sqrt(in_features), 1/sqrt(in_features)).
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features) if self.in_features > 0 else 0.0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
