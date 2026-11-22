"""Loss functions (MSE is used by the parity demo/tests)."""

from __future__ import annotations

from .module import Module


class MSELoss(Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("MSELoss: reduction must be 'none', 'mean' or 'sum'")
        self.reduction = reduction

    def forward(self, input, target):
        diff = input - target
        sq = diff * diff
        if self.reduction == "mean":
            return sq.mean()
        if self.reduction == "sum":
            return sq.sum()
        return sq

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"
