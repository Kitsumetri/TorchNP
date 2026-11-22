"""Optimizer parity tests against torch.optim (one and several steps)."""

from __future__ import annotations

import unittest

import numpy as np
import torch

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.nptorch import Tensor, nn, optim
from tests.utils import assert_close, copy_weights


class TestOptimizers(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(23)

    def _run(self, torch_optimizer_cls, np_optimizer_cls, *, steps=3, **kwargs):
        torch.manual_seed(0)
        np.random.seed(0)
        torch_linear = torch.nn.Linear(5, 3, dtype=torch.float64)
        np_linear = nn.Linear(5, 3, dtype=np.float64)
        copy_weights(torch_linear, np_linear)
        x = self.rng.standard_normal((4, 5))
        coeff = self.rng.standard_normal((4, 3))
        torch_opt = torch_optimizer_cls(torch_linear.parameters(), **kwargs)
        np_opt = np_optimizer_cls(np_linear.parameters(), **kwargs)
        for step in range(steps):
            torch_opt.zero_grad()
            np_opt.zero_grad()
            out_t = torch_linear(torch.tensor(x))
            out_n = np_linear(Tensor(x.copy()))
            (out_t * torch.tensor(coeff)).sum().backward()
            (out_n * Tensor(coeff)).sum().backward()
            torch_opt.step()
            np_opt.step()
            for name, param in np_linear.named_parameters():
                assert_close(
                    self,
                    param.data,
                    dict(torch_linear.named_parameters())[name].detach().numpy(),
                    1e-12,
                    1e-10,
                    f"step {step} {name}",
                )

    def test_sgd(self):
        self._run(torch.optim.SGD, optim.SGD, lr=0.1)

    def test_sgd_momentum(self):
        self._run(torch.optim.SGD, optim.SGD, lr=0.1, momentum=0.9)

    def test_sgd_nesterov_weight_decay(self):
        self._run(
            torch.optim.SGD,
            optim.SGD,
            lr=0.05,
            momentum=0.8,
            nesterov=True,
            weight_decay=1e-3,
        )

    def test_adam(self):
        self._run(torch.optim.Adam, optim.Adam, lr=1e-2)

    def test_adamw(self):
        self._run(torch.optim.AdamW, optim.AdamW, lr=1e-2)


if __name__ == "__main__":
    unittest.main()
