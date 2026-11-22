"""Dropout module (inverted dropout, disabled in eval mode)."""

from __future__ import annotations

from .. import functional as F
from .module import Module


class Dropout(Module):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        if p < 0.0 or p > 1.0:
            raise ValueError(f"Dropout: p={p}, expected 0 <= p <= 1")
        if inplace:
            raise NotImplementedError("inplace dropout is not supported")
        self.p = float(p)
        self.inplace = False

    def forward(self, input):
        return F.dropout(input, self.p, self.training)

    def extra_repr(self) -> str:
        return f"p={self.p}, inplace={self.inplace}"
