"""Autograd/op tests against torch and central finite differences."""

from __future__ import annotations

import unittest

import numpy as np
import torch

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.nptorch import Tensor, enable_grad, no_grad
from src.nptorch import functional as F
from tests.utils import assert_close


class TestAutogradOps(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def _check(self, np_fn, torch_fn, arrays, atol=1e-10, rtol=1e-9):
        np_args = [Tensor(a.copy(), requires_grad=True) for a in arrays]
        torch_args = [torch.tensor(a.copy(), requires_grad=True) for a in arrays]
        np_out = np_fn(*np_args)
        torch_out = torch_fn(*torch_args)
        assert_close(self, np_out.data, torch_out.detach().numpy(), atol, rtol, "output")
        coeff = self.rng.standard_normal(np_out.shape)
        np_out.backward(coeff)
        torch_out.backward(torch.tensor(coeff))
        for index, (np_arg, torch_arg) in enumerate(zip(np_args, torch_args)):
            self.assertIsNotNone(np_arg.grad, f"input {index} has no grad")
            assert_close(
                self,
                np_arg.grad.data,
                torch_arg.grad.detach().numpy(),
                atol,
                rtol,
                f"input {index} grad",
            )

    def test_arithmetic_and_broadcasting(self):
        a = self.rng.standard_normal((2, 1, 4))
        b = self.rng.standard_normal((3, 4))
        self._check(lambda x, y: x + y, lambda x, y: x + y, [a, b])
        self._check(lambda x, y: x - y, lambda x, y: x - y, [a, b])
        self._check(lambda x, y: x * y, lambda x, y: x * y, [a, b])
        self._check(lambda x, y: x / y, lambda x, y: x / y, [a, b])

    def test_scalar_pow_and_unary(self):
        x = self.rng.standard_normal((3, 4))
        self._check(lambda t: t**3, lambda t: t**3, [x])
        self._check(lambda t: (t * t + 2.0 * t - 1.5).mean(), lambda t: (t * t + 2.0 * t - 1.5).mean(), [x])
        positive = np.abs(x) + 0.5
        self._check(lambda t: t.exp(), lambda t: t.exp(), [x])
        self._check(lambda t: t.log(), lambda t: t.log(), [positive])
        self._check(lambda t: t.sqrt(), lambda t: t.sqrt(), [positive])
        self._check(lambda t: abs(t), lambda t: abs(t), [x])

    def test_matmul_shapes(self):
        pairs = [
            (self.rng.standard_normal((3,)), self.rng.standard_normal((3,))),
            (self.rng.standard_normal((2, 3)), self.rng.standard_normal((3,))),
            (self.rng.standard_normal((3,)), self.rng.standard_normal((3, 4))),
            (self.rng.standard_normal((2, 4, 3)), self.rng.standard_normal((3, 5))),
            (
                self.rng.standard_normal((3,)),
                self.rng.standard_normal((2, 5, 3, 4)),
            ),
            (
                self.rng.standard_normal((2, 5, 3, 4)),
                self.rng.standard_normal((4,)),
            ),
            (
                self.rng.standard_normal((2, 1, 4, 3)),
                self.rng.standard_normal((2, 5, 3, 6)),
            ),
        ]
        for a, b in pairs:
            with self.subTest(a=a.shape, b=b.shape):
                self._check(lambda x, y: x @ y, lambda x, y: x @ y, [a, b])

    def test_shape_ops(self):
        x = self.rng.standard_normal((2, 3, 4))
        self._check(lambda t: t.transpose(0, 2), lambda t: t.transpose(0, 2), [x])
        self._check(lambda t: t.permute(2, 0, 1), lambda t: t.permute(2, 0, 1), [x])
        self._check(lambda t: t.reshape(6, 4), lambda t: t.reshape(6, 4), [x])
        self._check(lambda t: t.unsqueeze(1), lambda t: t.unsqueeze(1), [x])
        self._check(lambda t: t[:, 1:3], lambda t: t[:, 1:3], [x])
        y = self.rng.standard_normal((2, 2, 4))
        self._check(
            lambda a, b: F.cat([a, b], dim=1),
            lambda a, b: torch.cat([a, b], dim=1),
            [x, y],
        )

    def test_squeeze(self):
        x = self.rng.standard_normal((2, 1, 4))
        self._check(lambda t: t.squeeze(1), lambda t: t.squeeze(1), [x])

    def test_reductions(self):
        x = self.rng.standard_normal((2, 3, 4))
        self._check(lambda t: t.sum(), lambda t: t.sum(), [x])
        self._check(lambda t: t.sum(dim=1), lambda t: t.sum(dim=1), [x])
        self._check(
            lambda t: t.sum(dim=(1, 2), keepdim=True),
            lambda t: t.sum(dim=(1, 2), keepdim=True),
            [x],
        )
        self._check(lambda t: t.mean(dim=-1), lambda t: t.mean(dim=-1), [x])
        self._check(
            lambda t: t.mean(dim=(0, 2), keepdim=True),
            lambda t: t.mean(dim=(0, 2), keepdim=True),
            [x],
        )
        self._check(
            lambda t: t.sum(dim=None, keepdim=True),
            lambda t: t.sum(dim=None, keepdim=True),
            [x],
        )
        self._check(lambda t: t.sum(dim=()), lambda t: t.sum(dim=()), [x])
        self._check(
            lambda t: t.mean(dim=(), keepdim=True),
            lambda t: t.mean(dim=(), keepdim=True),
            [x],
        )

    def test_activations(self):
        x = self.rng.standard_normal((3, 5)) * 2.0
        self._check(lambda t: F.relu(t), lambda t: torch.relu(t), [x])
        self._check(
            lambda t: F.gelu(t), lambda t: torch.nn.functional.gelu(t), [x]
        )
        self._check(
            lambda t: F.gelu(t, approximate="tanh"),
            lambda t: torch.nn.functional.gelu(t, approximate="tanh"),
            [x],
        )

    def test_softmax_and_log_softmax(self):
        x = self.rng.standard_normal((2, 4, 5))
        self._check(lambda t: F.softmax(t, dim=1), lambda t: torch.softmax(t, dim=1), [x])
        self._check(
            lambda t: F.log_softmax(t, dim=-1),
            lambda t: torch.log_softmax(t, dim=-1),
            [x],
        )

    def test_layer_norm(self):
        x = self.rng.standard_normal((2, 3, 4))
        weight = self.rng.standard_normal((4,))
        bias = self.rng.standard_normal((4,))
        self._check(
            lambda t, w, b: F.layer_norm(t, (4,), w, b, eps=1e-5),
            lambda t, w, b: torch.nn.functional.layer_norm(
                t, (4,), w, b, eps=1e-5
            ),
            [x, weight, bias],
        )

    def test_masked_fill(self):
        x = self.rng.standard_normal((3, 2, 4))
        mask = self.rng.random((2, 4)) > 0.5
        self._check(
            lambda t: t.masked_fill(mask, -1e9),
            lambda t: t.masked_fill(torch.tensor(mask), -1e9),
            [x],
        )

    def test_reused_tensor_accumulates(self):
        x_np = Tensor(self.rng.standard_normal((3, 4)), requires_grad=True)
        x_t = torch.tensor(x_np.data, requires_grad=True)
        out_np = x_np * x_np + x_np
        out_t = x_t * x_t + x_t
        coeff = self.rng.standard_normal(out_np.shape)
        out_np.backward(coeff)
        out_t.backward(torch.tensor(coeff))
        assert_close(self, out_np.data, out_t.detach().numpy())
        assert_close(self, x_np.grad.data, x_t.grad.numpy())

    def test_no_grad_and_detach(self):
        x = Tensor(self.rng.standard_normal((2, 3)), requires_grad=True)
        with no_grad():
            y = x * 2.0
            self.assertFalse(y.requires_grad)
            with enable_grad():
                z = x * 3.0
                self.assertTrue(z.requires_grad)
        self.assertIsNone(y.grad)
        self.assertIsNotNone(x.detach())
        self.assertFalse(x.detach().requires_grad)

    def test_no_grad_decorator(self):
        @no_grad()
        def double(x):
            return x * 2.0

        x = Tensor(np.ones((2, 2)), requires_grad=True)
        self.assertFalse(double(x).requires_grad)

    def test_unary_integer_dtype(self):
        values = np.array([1, 2, 3], dtype=np.int64)
        for np_fn, torch_fn in [
            (F.exp, torch.exp),
            (F.log, torch.log),
            (F.sqrt, torch.sqrt),
        ]:
            out = np_fn(Tensor(values.copy()))
            expected = torch_fn(torch.tensor(values))
            self.assertEqual(out.dtype, np.float32)
            assert_close(self, out.data, expected.numpy(), 1e-6, 1e-6)

    def test_cat_dtype_promotion(self):
        cases = [
            (np.float32, np.int64),
            (np.float64, np.float32),
            (np.int32, np.int64),
            (bool, np.int64),
        ]
        for left, right in cases:
            with self.subTest(left=left, right=right):
                a = Tensor(np.ones(2, dtype=left))
                b = Tensor(np.ones(2, dtype=right))
                expected = torch.cat(
                    [torch.ones(2, dtype=getattr(torch, str(np.dtype(left)))),
                     torch.ones(2, dtype=getattr(torch, str(np.dtype(right))))]
                )
                self.assertEqual(
                    F.cat([a, b]).dtype,
                    np.dtype(str(expected.dtype).replace("torch.", "")),
                )
                assert_close(self, F.cat([a, b]).data, expected.numpy(), 0.0, 0.0)

    def test_non_leaf_autograd_semantics(self):
        x = Tensor(np.ones(2), requires_grad=True)
        middle = x * 2.0
        with self.assertRaises(RuntimeError):
            middle.requires_grad_()
        middle.sum().backward()
        self.assertIsNone(middle.grad)
        self.assertTrue(np.array_equal(x.grad.data, [2.0, 2.0]))

    def test_float64_scalar_precision(self):
        # Python scalars must be converted directly to float64, without an
        # intermediate float32 rounding (regression test for MHA scale).
        base = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        coeff = np.array([0.3, -0.2, 0.7], dtype=np.float64)
        cases = [
            (lambda t: t * 0.1, lambda t: t * 0.1),
            (lambda t: t + 0.1, lambda t: t + 0.1),
            (lambda t: t / 0.1, lambda t: t / 0.1),
            (lambda t: t**0.1, lambda t: t**0.1),
        ]
        for np_fn, torch_fn in cases:
            x_np = Tensor(base.copy(), requires_grad=True)
            x_t = torch.tensor(base.copy(), requires_grad=True)
            out_np = np_fn(x_np)
            out_t = torch_fn(x_t)
            assert_close(self, out_np.data, out_t.detach().numpy(), 0.0, 0.0)
            out_np.backward(coeff)
            out_t.backward(torch.tensor(coeff))
            assert_close(self, x_np.grad.data, x_t.grad.numpy(), 0.0, 0.0)

    def test_type_promotion(self):
        cases = [
            (np.array([1, 2, 3]), 0.5),
            (np.array([1, 2, 3]), 2),
            (np.array([1, 2, 3]), np.ones(3, dtype=np.float32)),
            (np.array([1, 2, 3]), Tensor(np.ones(3, dtype=np.float32))),
        ]
        for array, other in cases:
            torch_other = other
            np_other = other
            if isinstance(other, np.ndarray):
                torch_other = torch.tensor(other)
            elif isinstance(other, Tensor):
                torch_other = torch.tensor(other.data)
            left = torch.tensor(array)
            expected = left + torch_other
            actual = Tensor(array.copy()) + np_other
            assert_close(self, actual.data, expected.numpy(), 0.0, 0.0)
            self.assertEqual(actual.dtype, np.dtype(str(expected.dtype).replace("torch.", "")))
        expected_div = torch.tensor([1, 2, 3]) / 2
        actual_div = Tensor(np.array([1, 2, 3])) / 2
        assert_close(self, actual_div.data, expected_div.numpy(), 0.0, 0.0)
        self.assertEqual(
            actual_div.dtype, np.dtype(str(expected_div.dtype).replace("torch.", ""))
        )

    def test_backward_seed_rules(self):
        x = Tensor(np.ones((2, 2)), requires_grad=True)
        with self.assertRaises(RuntimeError):
            (x * 2.0).backward()
        (x * 2.0).backward(np.ones((2, 2)))
        assert_close(self, x.grad.data, np.full((2, 2), 2.0))

    def test_zero_grad_and_accumulation(self):
        x = Tensor(np.ones((2, 3)), requires_grad=True)
        y = (x * x).sum()
        y.backward()
        first = x.grad.data.copy()
        y.backward()
        assert_close(self, x.grad.data, 2.0 * first)
        x.zero_grad()
        self.assertIsNone(x.grad)
        x.zero_grad(set_to_none=False)
        assert_close(self, x.grad.data, np.zeros((2, 3)))

    def test_float32_dtype_is_preserved(self):
        x = Tensor(np.ones((2, 3), dtype=np.float32), requires_grad=True)
        for y in (x * 2, x + 1, x / 3, x**2, x.sum(), F.softmax(x)):
            self.assertEqual(y.dtype, np.float32, f"dtype changed: {y.dtype}")
        loss = (x * 2).sum()
        loss.backward()
        self.assertEqual(x.grad.dtype, np.float32)

    def test_finite_differences_composite(self):
        rng = np.random.default_rng(11)
        x0 = rng.standard_normal((2, 3))
        w0 = rng.standard_normal((4, 3))
        b0 = rng.standard_normal((4,))
        gamma0 = rng.standard_normal((4,))
        beta0 = rng.standard_normal((4,))
        coef = rng.standard_normal((2, 4))

        def loss_fn(x, w, b, gamma, beta):
            h = F.gelu(F.linear(x, w, b))
            h = F.layer_norm(h, (4,), gamma, beta)
            s = F.softmax(h, dim=-1)
            return (s * coef).sum() + (h * h).mean()

        arrays = [x0, w0, b0, gamma0, beta0]
        tensors = [Tensor(a.copy(), requires_grad=True) for a in arrays]
        loss = loss_fn(*tensors)
        loss.backward()
        analytic = [t.grad.data.copy() for t in tensors]

        def numeric_value():
            with no_grad():
                value = loss_fn(*[Tensor(a.copy()) for a in arrays])
            return float(value.item())

        eps = 1e-6
        numeric = []
        for array in arrays:
            grad = np.zeros_like(array)
            it = np.nditer(array, flags=["multi_index"])
            for _ in it:
                index = it.multi_index
                old = array[index]
                array[index] = old + eps
                plus = numeric_value()
                array[index] = old - eps
                minus = numeric_value()
                array[index] = old
                grad[index] = (plus - minus) / (2.0 * eps)
            numeric.append(grad)

        names = ["x", "W", "b", "gamma", "beta"]
        for name, an, num in zip(names, analytic, numeric):
            assert_close(self, an, num, atol=2e-6, rtol=2e-6, msg=f"fd {name}")


if __name__ == "__main__":
    unittest.main()
