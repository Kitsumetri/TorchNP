"""Activation modules."""

from __future__ import annotations

from .. import functional as F
from .module import Module


class ReLU(Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        if inplace:
            raise NotImplementedError("inplace ReLU is not supported")
        self.inplace = False

    def forward(self, input):
        return F.relu(input)

    def extra_repr(self) -> str:
        return f"inplace={self.inplace}"


class GELU(Module):
    def __init__(self, approximate: str = "none"):
        super().__init__()
        if approximate not in ("none", "tanh"):
            raise ValueError("GELU: approximate must be 'none' or 'tanh'")
        self.approximate = approximate

    def forward(self, input):
        return F.gelu(input, approximate=self.approximate)

    def extra_repr(self) -> str:
        return f"approximate={self.approximate!r}"
