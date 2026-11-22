"""Tests for the torch-like nn modules and multi-head attention."""

from __future__ import annotations

import unittest

import numpy as np
import torch

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.nptorch import Tensor, manual_seed, nn
from src.nptorch import functional as F
from tests.utils import (
    assert_close,
    assert_param_grads_close,
    copy_weights,
    mha_parity,
)


class TestBasicModules(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)

    def test_linear_forward_backward(self):
        torch_linear = torch.nn.Linear(5, 3, dtype=torch.float64).eval()
        np_linear = nn.Linear(5, 3, dtype=np.float64).eval()
        copy_weights(torch_linear, np_linear)
        x = self.rng.standard_normal((4, 5))
        xt = torch.tensor(x, requires_grad=True)
        xn = Tensor(x.copy(), requires_grad=True)
        out_t = torch_linear(xt)
        out_n = np_linear(xn)
        coeff = self.rng.standard_normal(out_t.shape)
        out_t.backward(torch.tensor(coeff))
        out_n.backward(coeff)
        assert_close(self, out_n.data, out_t.detach().numpy())
        assert_close(self, xn.grad.data, xt.grad.numpy())
        assert_param_grads_close(self, torch_linear, np_linear, 1e-12, 1e-10)

    def test_linear_without_bias(self):
        torch_linear = torch.nn.Linear(4, 2, bias=False, dtype=torch.float64)
        np_linear = nn.Linear(4, 2, bias=False, dtype=np.float64)
        self.assertEqual(sorted(np_linear.state_dict()), ["weight"])
        copy_weights(torch_linear, np_linear)
        x = self.rng.standard_normal((3, 4))
        xt = torch.tensor(x, requires_grad=True)
        xn = Tensor(x.copy(), requires_grad=True)
        out_t = torch_linear(xt)
        out_n = np_linear(xn)
        coeff = self.rng.standard_normal(out_t.shape)
        (out_t * torch.tensor(coeff)).sum().backward()
        (out_n * Tensor(coeff)).sum().backward()
        assert_close(self, out_n.data, out_t.detach().numpy())
        assert_param_grads_close(self, torch_linear, np_linear, 1e-12, 1e-10)

    def test_linear_input_ndim(self):
        for shape in [(5,), (2, 3, 5)]:
            with self.subTest(shape=shape):
                torch_linear = torch.nn.Linear(5, 3, dtype=torch.float64).eval()
                np_linear = nn.Linear(5, 3, dtype=np.float64).eval()
                copy_weights(torch_linear, np_linear)
                x = self.rng.standard_normal(shape)
                xt = torch.tensor(x, requires_grad=True)
                xn = Tensor(x.copy(), requires_grad=True)
                out_t = torch_linear(xt)
                out_n = np_linear(xn)
                coeff = self.rng.standard_normal(out_t.shape)
                out_t.backward(torch.tensor(coeff))
                out_n.backward(coeff)
                assert_close(self, out_n.data, out_t.detach().numpy())
                assert_close(self, xn.grad.data, xt.grad.numpy())
                assert_param_grads_close(self, torch_linear, np_linear, 1e-12, 1e-10)

    def test_initialization_bounds(self):
        linear = nn.Linear(100, 50)
        self.assertTrue(np.all(np.abs(linear.weight.data) <= 0.1 + 1e-6))
        self.assertTrue(np.all(np.abs(linear.bias.data) <= 0.1 + 1e-6))

        mha = nn.MultiheadAttention(64, 8)
        xavier_bound = np.sqrt(6.0 / (64 + 3 * 64))
        self.assertTrue(
            np.all(np.abs(mha.in_proj_weight.data) <= xavier_bound + 1e-6)
        )
        assert_close(self, mha.in_proj_bias.data, np.zeros(3 * 64))
        assert_close(self, mha.out_proj.bias.data, np.zeros(64))

    def test_layer_norm_forward_backward(self):
        torch_norm = torch.nn.LayerNorm(4, eps=1e-5, dtype=torch.float64).eval()
        np_norm = nn.LayerNorm(4, eps=1e-5, dtype=np.float64).eval()
        copy_weights(torch_norm, np_norm)
        x = self.rng.standard_normal((2, 3, 4))
        xt = torch.tensor(x, requires_grad=True)
        xn = Tensor(x.copy(), requires_grad=True)
        out_t = torch_norm(xt)
        out_n = np_norm(xn)
        coeff = self.rng.standard_normal(out_t.shape)
        out_t.backward(torch.tensor(coeff))
        out_n.backward(coeff)
        assert_close(self, out_n.data, out_t.detach().numpy())
        assert_close(self, xn.grad.data, xt.grad.numpy())
        assert_param_grads_close(self, torch_norm, np_norm, 1e-12, 1e-10)

    def test_layer_norm_multi_dim_and_no_affine(self):
        for affine in (True, False):
            with self.subTest(affine=affine):
                torch_norm = torch.nn.LayerNorm(
                    (2, 3), eps=1e-5, elementwise_affine=affine, dtype=torch.float64
                ).eval()
                np_norm = nn.LayerNorm(
                    (2, 3), eps=1e-5, elementwise_affine=affine, dtype=np.float64
                ).eval()
                copy_weights(torch_norm, np_norm)
                x = self.rng.standard_normal((4, 2, 3))
                xt = torch.tensor(x, requires_grad=True)
                xn = Tensor(x.copy(), requires_grad=True)
                out_t = torch_norm(xt)
                out_n = np_norm(xn)
                coeff = self.rng.standard_normal(out_t.shape)
                out_t.backward(torch.tensor(coeff))
                out_n.backward(coeff)
                assert_close(self, out_n.data, out_t.detach().numpy())
                assert_close(self, xn.grad.data, xt.grad.numpy())
                assert_param_grads_close(self, torch_norm, np_norm, 1e-12, 1e-10)

    def test_activations_against_torch(self):
        x = self.rng.standard_normal((4, 6)) * 2.0
        for np_module, torch_module in [
            (nn.ReLU(), torch.nn.ReLU()),
            (nn.GELU(), torch.nn.GELU()),
            (nn.GELU(approximate="tanh"), torch.nn.GELU(approximate="tanh")),
        ]:
            with self.subTest(module=type(torch_module).__name__):
                xt = torch.tensor(x, requires_grad=True)
                xn = Tensor(x.copy(), requires_grad=True)
                out_t = torch_module(xt)
                out_n = np_module(xn)
                coeff = self.rng.standard_normal(out_t.shape)
                out_t.backward(torch.tensor(coeff))
                out_n.backward(coeff)
                assert_close(self, out_n.data, out_t.detach().numpy())
                assert_close(self, xn.grad.data, xt.grad.numpy())

    def test_dropout_with_explicit_mask(self):
        manual_seed(0)
        x = self.rng.standard_normal((5, 7))
        p = 0.3
        mask = self.rng.random(x.shape) > p
        x_np = Tensor(x.copy(), requires_grad=True)
        out = F.dropout(x_np, p=p, training=True, mask=mask)
        expected = x * mask / (1.0 - p)
        assert_close(self, out.data, expected)
        coeff = self.rng.standard_normal(out.shape)
        out.backward(coeff)
        assert_close(self, x_np.grad.data, coeff * mask / (1.0 - p))

    def test_dropout_statistics_and_modes(self):
        manual_seed(123)
        x = np.ones(200_000, dtype=np.float64)
        out = F.dropout(Tensor(x), p=0.25, training=True)
        self.assertAlmostEqual(float(np.mean(out.data != 0)), 0.75, delta=0.01)
        self.assertAlmostEqual(float(np.mean(out.data)), 1.0, delta=0.01)
        identity = F.dropout(Tensor(x), p=0.0, training=True)
        assert_close(self, identity.data, x)
        zero = F.dropout(Tensor(x), p=1.0, training=True)
        assert_close(self, zero.data, np.zeros_like(x))
        with self.assertRaises(ValueError):
            F.dropout(Tensor(x), p=1.5, training=True)

        module = nn.Dropout(0.5)
        module.eval()
        # In eval mode dropout is identity and returns the same tensor.
        x_tensor = Tensor(x)
        self.assertIs(module(x_tensor), x_tensor)

    def test_mse_loss(self):
        x = self.rng.standard_normal((3, 4))
        y = self.rng.standard_normal((3, 4))
        xt = torch.tensor(x)
        yt = torch.tensor(y)
        x_np = Tensor(x.copy())
        y_np = Tensor(y)
        torch_loss = torch.nn.MSELoss()(xt, yt)
        np_loss = nn.MSELoss()(x_np, y_np)
        assert_close(self, np_loss.data, torch_loss.item())
        assert_close(
            self,
            nn.MSELoss(reduction="sum")(x_np, y_np).data,
            torch.nn.MSELoss(reduction="sum")(xt, yt).item(),
        )

    def test_module_train_eval_propagation(self):
        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2)
                self.dropout = nn.Dropout(0.5)

            def forward(self, x):
                return self.dropout(self.linear(x))

        wrapper = Wrapper()
        self.assertTrue(wrapper.training)
        self.assertTrue(wrapper.linear.training)
        self.assertTrue(wrapper.dropout.training)
        wrapper.eval()
        self.assertFalse(wrapper.training)
        self.assertFalse(wrapper.linear.training)
        self.assertFalse(wrapper.dropout.training)


class TestMultiheadAttention(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(5)

    def _make_pair(self, embed_dim=16, num_heads=4, **kwargs):
        torch_mha = torch.nn.MultiheadAttention(
            embed_dim, num_heads, dtype=torch.float64, **kwargs
        )
        np_mha = nn.MultiheadAttention(
            embed_dim, num_heads, dtype=np.float64, **kwargs
        )
        return torch_mha, np_mha

    def test_self_attention_forward_backward(self):
        torch_mha, np_mha = self._make_pair(dropout=0.1)
        x = self.rng.standard_normal((5, 3, 16))
        out_err, grad_err = mha_parity(
            self, torch_mha, np_mha, x, x, x, need_weights=False
        )
        self.assertLess(out_err, 1e-12)
        self.assertLess(grad_err, 1e-12)

    def test_attention_weights(self):
        for average in (True, False):
            with self.subTest(average=average):
                torch_mha, np_mha = self._make_pair(dropout=0.1)
                x = self.rng.standard_normal((5, 3, 16))
                mha_parity(
                    self,
                    torch_mha,
                    np_mha,
                    x,
                    x,
                    x,
                    need_weights=True,
                    average_attn_weights=average,
                )

    def test_cross_attention_unbatched(self):
        torch_mha, np_mha = self._make_pair(dropout=0.1)
        query = self.rng.standard_normal((5, 16))
        key = self.rng.standard_normal((7, 16))
        value = self.rng.standard_normal((7, 16))
        mha_parity(
            self,
            torch_mha,
            np_mha,
            query,
            key,
            value,
            need_weights=True,
        )

    def test_cross_attention_unbatched_key_padding_mask(self):
        torch_mha, np_mha = self._make_pair(dropout=0.1)
        query = self.rng.standard_normal((5, 16))
        key = self.rng.standard_normal((7, 16))
        value = self.rng.standard_normal((7, 16))
        key_padding_mask = np.zeros((7,), dtype=bool)
        key_padding_mask[-2:] = True
        mha_parity(
            self,
            torch_mha,
            np_mha,
            query,
            key,
            value,
            need_weights=True,
            key_padding_mask=key_padding_mask,
        )

    def test_batch_first_with_masks(self):
        torch_mha, np_mha = self._make_pair(dropout=0.1, batch_first=True)
        x = self.rng.standard_normal((3, 5, 16))
        attn_mask = np.triu(np.ones((5, 5), dtype=bool), k=1)
        key_padding_mask = np.zeros((3, 5), dtype=bool)
        key_padding_mask[0, -2:] = True
        mha_parity(
            self,
            torch_mha,
            np_mha,
            x,
            x,
            x,
            need_weights=False,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )

    def test_float_masks(self):
        torch_mha, np_mha = self._make_pair(dropout=0.1)
        x = self.rng.standard_normal((3, 2, 16))
        attn_mask = np.zeros((3, 3), dtype=np.float64)
        attn_mask[:, -1] = -1e9
        key_padding_mask = np.zeros((2, 3), dtype=np.float64)
        key_padding_mask[0, -1] = -1e9
        mha_parity(
            self,
            torch_mha,
            np_mha,
            x,
            x,
            x,
            need_weights=False,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )

    def test_3d_attention_mask(self):
        torch_mha, np_mha = self._make_pair(dropout=0.1)
        x = self.rng.standard_normal((3, 2, 16))
        attn_mask = np.zeros((2 * 4, 3, 3), dtype=bool)
        attn_mask[:, :, -1] = True
        mha_parity(
            self,
            torch_mha,
            np_mha,
            x,
            x,
            x,
            need_weights=False,
            attn_mask=attn_mask,
        )

    def test_causal_mask(self):
        torch_mha, np_mha = self._make_pair(dropout=0.1)
        x = self.rng.standard_normal((5, 2, 16))
        causal = np.triu(np.ones((5, 5), dtype=bool), k=1)
        mha_parity(
            self,
            torch_mha,
            np_mha,
            x,
            x,
            x,
            need_weights=False,
            attn_mask=causal,
            is_causal=True,
        )

    def test_is_causal_hint_with_non_causal_mask(self):
        # torch treats is_causal as a hint: with need_weights=True it uses the
        # supplied attn_mask; with need_weights=False it uses the causal flag
        # and ignores the mask when there is no key padding mask.
        mask = np.zeros((5, 5), dtype=bool)
        x = self.rng.standard_normal((5, 2, 16))
        for need_weights in (True, False):
            with self.subTest(need_weights=need_weights):
                torch_mha, np_mha = self._make_pair(dropout=0.1)
                mha_parity(
                    self,
                    torch_mha,
                    np_mha,
                    x,
                    x,
                    x,
                    need_weights=need_weights,
                    attn_mask=mask,
                    is_causal=True,
                )

    def test_state_dict_matches_torch(self):
        torch_mha, np_mha = self._make_pair(dropout=0.1)
        self.assertEqual(
            sorted(torch_mha.state_dict().keys()), sorted(np_mha.state_dict().keys())
        )

    def test_is_causal_requires_mask(self):
        _, np_mha = self._make_pair(dropout=0.1)
        x = Tensor(self.rng.standard_normal((3, 2, 16)))
        with self.assertRaises(RuntimeError):
            np_mha(x, x, x, need_weights=False, is_causal=True)

    def test_unsupported_options_raise(self):
        with self.assertRaises(NotImplementedError):
            nn.MultiheadAttention(16, 4, add_bias_kv=True)
        with self.assertRaises(NotImplementedError):
            nn.MultiheadAttention(16, 4, kdim=8)


if __name__ == "__main__":
    unittest.main()
