"""Unit tests for the hand-written numerical kernels in ``ops``."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from src.nptorch import init
from src.nptorch.nn import ops
from tests.utils import assert_close


def tensor(array: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(np.asarray(array), dtype=torch.float64).requires_grad_(
        requires_grad
    )


class LinearTest(unittest.TestCase):
    def _check(self, shape: tuple[int, ...], bias: bool) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=shape)
        weight = rng.normal(size=(6, shape[-1]))
        b = rng.normal(size=6) if bias else None

        out, cache = ops.linear_forward(x, weight, b)
        x_t = tensor(x, True)
        w_t = tensor(weight, True)
        b_t = tensor(b, True) if bias else None
        torch_out = torch.nn.functional.linear(x_t, w_t, b_t)
        assert_close(self, out, torch_out.detach().numpy(), msg="linear forward")

        dout = rng.normal(size=out.shape)
        dx, dw, db = ops.linear_backward(dout, cache)
        torch_out.backward(tensor(dout))
        assert_close(self, dx, x_t.grad.numpy(), msg="dX")
        assert_close(self, dw, w_t.grad.numpy(), msg="dW")
        if bias:
            assert_close(self, db, b_t.grad.numpy(), msg="db")

    def test_linear_2d_with_bias(self) -> None:
        self._check((4, 5), True)

    def test_linear_3d_with_bias(self) -> None:
        self._check((2, 4, 5), True)

    def test_linear_without_bias(self) -> None:
        self._check((3, 2, 5), False)


class LayerNormTest(unittest.TestCase):
    def _check(
        self, shape: tuple[int, ...], beta: bool, eps: float = 1e-5
    ) -> None:
        rng = np.random.default_rng(1)
        features = shape[-1]
        x = rng.normal(size=shape) * 3.0 + 0.5
        gamma = rng.normal(size=features)
        b = rng.normal(size=features) if beta else None

        out, cache = ops.layer_norm_forward(x, gamma, b, eps)
        x_t = tensor(x, True)
        g_t = tensor(gamma, True)
        b_t = tensor(b, True) if beta else None
        torch_out = torch.nn.functional.layer_norm(
            x_t, (features,), g_t, b_t, eps
        )
        assert_close(self, out, torch_out.detach().numpy(), msg="LN forward")

        dout = rng.normal(size=out.shape)
        dx, dgamma, dbeta = ops.layer_norm_backward(dout, cache)
        torch_out.backward(tensor(dout))
        assert_close(self, dx, x_t.grad.numpy(), msg="LN dX")
        assert_close(self, dgamma, g_t.grad.numpy(), msg="LN dgamma")
        if beta:
            assert_close(self, dbeta, b_t.grad.numpy(), msg="LN dbeta")

    def test_2d(self) -> None:
        self._check((6, 8), True)

    def test_3d(self) -> None:
        self._check((3, 4, 7), True)

    def test_without_bias(self) -> None:
        self._check((5, 4), False)

    def test_eps(self) -> None:
        self._check((2, 3, 6), True, eps=1e-3)


class ActivationTest(unittest.TestCase):
    def test_relu(self) -> None:
        rng = np.random.default_rng(2)
        x = rng.normal(size=(4, 5))
        out, cache = ops.relu_forward(x)
        x_t = tensor(x, True)
        torch_out = torch.nn.functional.relu(x_t)
        assert_close(self, out, torch_out.detach().numpy(), msg="relu forward")
        dout = rng.normal(size=out.shape)
        dx = ops.relu_backward(dout, cache)
        torch_out.backward(tensor(dout))
        assert_close(self, dx, x_t.grad.numpy(), msg="relu backward")
        self.assertTrue(np.all(dx[x <= 0.0] == 0.0))

    def test_gelu(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.normal(size=(4, 5)) * 2.0
        out, cache = ops.gelu_forward(x)
        x_t = tensor(x, True)
        torch_out = torch.nn.functional.gelu(x_t)
        assert_close(self, out, torch_out.detach().numpy(), msg="gelu forward")
        dout = rng.normal(size=out.shape)
        dx = ops.gelu_backward(dout, cache)
        torch_out.backward(tensor(dout))
        assert_close(self, dx, x_t.grad.numpy(), msg="gelu backward")


class SoftmaxTest(unittest.TestCase):
    def test_matches_torch(self) -> None:
        rng = np.random.default_rng(4)
        scores = rng.normal(size=(3, 2, 5)) * 2.0
        scores[0, 0, 2] = -np.inf
        scores[1, 1, 1] = -np.inf
        probs = ops.softmax_forward(scores)
        s_t = tensor(scores, True)
        torch_probs = torch.softmax(s_t, dim=-1)
        assert_close(self, probs, torch_probs.detach().numpy(), msg="softmax")

        dprobs = rng.normal(size=probs.shape)
        dscores = ops.softmax_backward(dprobs, probs)
        torch_probs.backward(tensor(dprobs))
        assert_close(self, dscores, s_t.grad.numpy(), msg="softmax backward")

    def test_fully_masked_row_is_zero(self) -> None:
        scores = np.array(
            [
                [1.0, 2.0, 3.0],
                [-np.inf, -np.inf, -np.inf],
            ]
        )
        probs = ops.softmax_forward(scores)
        self.assertTrue(np.all(probs[1] == 0.0))
        dprobs = np.ones_like(scores)
        dscores = ops.softmax_backward(dprobs, probs)
        assert_close(self, dscores[1], np.zeros(3), msg="masked dscore")

    def test_masked_entries_get_zero_gradient(self) -> None:
        scores = np.array([[0.5, -np.inf, 0.5]])
        probs = ops.softmax_forward(scores)
        dscores = ops.softmax_backward(np.ones_like(scores), probs)
        self.assertEqual(dscores[0, 1], 0.0)


class DropoutTest(unittest.TestCase):
    def test_explicit_keep_mask(self) -> None:
        rng = np.random.default_rng(5)
        x = rng.normal(size=(3, 4))
        keep = rng.random(x.shape) >= 0.25
        p = 0.25
        out, cache = ops.dropout_forward(x, p, True, mask=keep)
        expected = np.where(keep, x / (1.0 - p), 0.0)
        assert_close(self, out, expected, msg="dropout forward")

        dout = rng.normal(size=x.shape)
        dx = ops.dropout_backward(dout, cache)
        expected_dx = np.where(keep, dout / (1.0 - p), 0.0)
        assert_close(self, dx, expected_dx, msg="dropout backward")

    def test_eval_mode_is_identity(self) -> None:
        x = np.arange(12.0).reshape(3, 4)
        out, cache = ops.dropout_forward(x, 0.5, False)
        assert_close(self, out, x)
        self.assertIsNone(cache)
        dout = np.ones_like(x)
        assert_close(self, ops.dropout_backward(dout, cache), dout)

    def test_p_zero_is_identity(self) -> None:
        x = np.arange(6.0).reshape(2, 3)
        out, cache = ops.dropout_forward(x, 0.0, True)
        assert_close(self, out, x)
        self.assertIsNone(cache)

    def test_p_one_returns_zeros(self) -> None:
        x = np.arange(6.0).reshape(2, 3)
        out, cache = ops.dropout_forward(x, 1.0, True)
        assert_close(self, out, np.zeros_like(x))
        assert_close(
            self, ops.dropout_backward(np.ones_like(x), cache), np.zeros_like(x)
        )

    def test_random_mask_scales_kept_values(self) -> None:
        rng = np.random.default_rng(6)
        x = np.ones((20, 20))
        out, cache = ops.dropout_forward(x, 0.4, True, rng=rng)
        keep, scale = cache
        assert_close(self, out, np.where(keep, x / (1.0 - 0.4), 0.0))
        self.assertAlmostEqual(keep.mean(), 0.6, delta=0.05)

    def test_mask_shape_validation(self) -> None:
        with self.assertRaises(ValueError):
            ops.dropout_forward(
                np.zeros((2, 3)), 0.5, True, mask=np.ones((3, 2), dtype=bool)
            )


class InitializationTest(unittest.TestCase):
    def test_fan_in_fan_out(self) -> None:
        self.assertEqual(init.calculate_fan_in_and_fan_out(np.zeros((4, 3))), (3, 4))
        self.assertEqual(
            init.calculate_fan_in_and_fan_out(np.zeros((4, 3, 2, 2))), (12, 16)
        )
        with self.assertRaises(ValueError):
            init.calculate_fan_in_and_fan_out(np.zeros(4))

    def test_xavier_bounds(self) -> None:
        array = np.empty((4, 6))
        init.xavier_uniform_(array, gain=2.0)
        bound = 2.0 * math.sqrt(6.0 / (6 + 4))
        self.assertLessEqual(np.max(np.abs(array)), bound)
        self.assertGreater(np.std(array), 0.0)

    def test_kaiming_bounds(self) -> None:
        array = np.empty((5, 20))
        init.kaiming_uniform_(array)
        bound = math.sqrt(6.0 / (6.0 * 20.0))
        self.assertLessEqual(np.max(np.abs(array)), bound)

    def test_fill_helpers(self) -> None:
        array = np.empty((2, 3))
        self.assertIs(init.constant_(array, 2.5), array)
        assert_close(self, array, np.full((2, 3), 2.5))

        init.zeros_(array)
        assert_close(self, array, np.zeros((2, 3)))

        init.ones_(array)
        assert_close(self, array, np.ones((2, 3)))

        init.uniform_(array, -2.0, 3.0)
        self.assertTrue(np.all(array >= -2.0) and np.all(array <= 3.0))


if __name__ == "__main__":
    unittest.main()
