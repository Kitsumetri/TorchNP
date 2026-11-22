"""End-to-end parity tests for TransformerEncoderLayer forward and backward."""

from __future__ import annotations

import unittest

import numpy as np
import torch

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.nptorch import Tensor, nn, no_grad, optim
from tests.utils import (
    assert_close,
    assert_param_grads_close,
    copy_weights,
)


class TestTransformerEncoderLayer(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(17)

    def _make_pair(self, d_model, nhead, dtype=np.float64, **kwargs):
        np_dtype = np.dtype(dtype)
        torch_dtype = torch.float64 if np_dtype == np.float64 else torch.float32
        torch_layer = torch.nn.TransformerEncoderLayer(
            d_model, nhead, dtype=torch_dtype, **kwargs
        )
        np_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dtype=np_dtype, **kwargs
        )
        return torch_layer, np_layer

    def _parity(
        self,
        torch_layer,
        np_layer,
        x,
        *,
        src_mask=None,
        src_key_padding_mask=None,
        is_causal: bool = False,
        training: bool = False,
        atol: float = 1e-12,
        rtol: float = 1e-10,
        label: str = "",
    ):
        if training:
            torch_layer.train()
            np_layer.train()
        else:
            torch_layer.eval()
            np_layer.eval()
        torch_layer.zero_grad(set_to_none=True)
        np_layer.zero_grad(set_to_none=True)
        copy_weights(torch_layer, np_layer)

        def mask_tensor(mask):
            return None if mask is None else torch.tensor(mask)

        xt = torch.tensor(x, requires_grad=True)
        xn = Tensor(x.copy(), requires_grad=True)
        out_t = torch_layer(
            xt,
            src_mask=mask_tensor(src_mask),
            src_key_padding_mask=mask_tensor(src_key_padding_mask),
            is_causal=is_causal,
        )
        out_n = np_layer(
            xn,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
        )
        assert_close(
            self, out_n.data, out_t.detach().numpy(), atol, rtol, f"{label} output"
        )

        coeff = self.rng.standard_normal(out_t.shape).astype(
            np.float64 if out_t.dtype == torch.float64 else np.float32
        )
        loss_t = (out_t * torch.tensor(coeff, dtype=out_t.dtype)).sum()
        loss_n = (out_n * Tensor(coeff)).sum()
        loss_t.backward()
        loss_n.backward()
        assert_close(
            self,
            np.asarray(loss_n.item()),
            np.asarray(loss_t.item()),
            atol,
            rtol,
            f"{label} loss",
        )
        assert_close(
            self, xn.grad.data, xt.grad.detach().numpy(), atol, rtol, f"{label} input grad"
        )
        assert_param_grads_close(
            self, torch_layer, np_layer, atol, rtol, f"{label}"
        )

    # ------------------------------------------------------------------
    # Defaults and tight numerical parity
    # ------------------------------------------------------------------

    def test_default_constructor_matches_torch_defaults(self):
        torch_layer = torch.nn.TransformerEncoderLayer(512, 8)
        np_layer = nn.TransformerEncoderLayer(512, 8)
        self.assertEqual(np_layer.dim_feedforward, 2048)
        self.assertAlmostEqual(np_layer.dropout.p, 0.1)
        self.assertAlmostEqual(np_layer.norm1.eps, 1e-5)
        self.assertFalse(np_layer.norm_first)
        self.assertFalse(np_layer.self_attn.batch_first)
        self.assertEqual(np_layer.self_attn.num_heads, 8)
        self.assertEqual(np_layer.self_attn.head_dim, 64)
        self.assertTrue(np_layer.linear1.bias is not None)
        self.assertEqual(np_layer.activation_relu_or_gelu, 1)
        self.assertEqual(
            sorted(torch_layer.state_dict().keys()),
            sorted(np_layer.state_dict().keys()),
        )
        for name, param in np_layer.named_parameters():
            self.assertEqual(
                tuple(torch_layer.state_dict()[name].shape),
                param.shape,
                f"shape mismatch for {name}",
            )

    def test_default_float32_forward_backward(self):
        torch_layer, np_layer = self._make_pair(512, 8, dtype=np.float32)
        x = self.rng.standard_normal((4, 2, 512)).astype(np.float32)
        self._parity(torch_layer, np_layer, x, atol=2e-5, rtol=2e-4, label="default")

    def test_float64_tight_forward_backward(self):
        torch_layer, np_layer = self._make_pair(16, 4, dim_feedforward=32)
        x = self.rng.standard_normal((5, 3, 16))
        self._parity(torch_layer, np_layer, x, label="tight")

    def test_float64_head_dim_eight(self):
        # head_dim=8 makes the attention scale 1/sqrt(8) inexact in float32;
        # this guards against scalar double-rounding in float64.
        torch_layer, np_layer = self._make_pair(32, 4, dim_feedforward=64)
        x = self.rng.standard_normal((7, 3, 32))
        self._parity(torch_layer, np_layer, x, label="head_dim=8")

    def test_masks_float64(self):
        torch_layer, np_layer = self._make_pair(16, 4, dim_feedforward=32)
        x = self.rng.standard_normal((5, 3, 16))
        bool_mask = np.triu(np.ones((5, 5), dtype=bool), k=1)
        kpm = np.zeros((3, 5), dtype=bool)
        kpm[0, 3:] = True
        float_mask = np.zeros((5, 5), dtype=np.float64)
        float_mask[:, -2:] = -1e9
        cases = [
            ("bool src_mask", bool_mask, None),
            ("bool key_padding_mask", None, kpm),
            ("both", bool_mask, kpm),
            ("float src_mask", float_mask, None),
            ("fully masked rows", np.ones((5, 5), dtype=bool), None),
        ]
        for label, mask, padding in cases:
            with self.subTest(case=label):
                self._parity(
                    torch_layer,
                    np_layer,
                    x,
                    src_mask=mask,
                    src_key_padding_mask=padding,
                    label=label,
                )

    def test_3d_src_mask(self):
        torch_layer, np_layer = self._make_pair(16, 4, dim_feedforward=32)
        x = self.rng.standard_normal((4, 2, 16))
        mask = np.zeros((2 * 4, 4, 4), dtype=bool)
        mask[:, :, -1] = True
        self._parity(torch_layer, np_layer, x, src_mask=mask, label="3d mask")

    def test_batch_first(self):
        torch_layer, np_layer = self._make_pair(
            16, 4, dim_feedforward=32, batch_first=True
        )
        x = self.rng.standard_normal((3, 5, 16))
        mask = np.triu(np.ones((5, 5), dtype=bool), k=1)
        kpm = np.zeros((3, 5), dtype=bool)
        kpm[1, -1] = True
        self._parity(
            torch_layer,
            np_layer,
            x,
            src_mask=mask,
            src_key_padding_mask=kpm,
            label="batch_first",
        )

    def test_pre_layer_norm_gelu_no_bias(self):
        torch_layer, np_layer = self._make_pair(
            16,
            4,
            dim_feedforward=32,
            norm_first=True,
            activation="gelu",
            bias=False,
            layer_norm_eps=1e-6,
            batch_first=True,
        )
        x = self.rng.standard_normal((3, 5, 16))
        self._parity(torch_layer, np_layer, x, label="preln gelu nobias")

    def test_causal_hint(self):
        torch_layer, np_layer = self._make_pair(16, 4, dim_feedforward=32)
        x = self.rng.standard_normal((5, 2, 16))
        causal = np.triu(np.ones((5, 5), dtype=bool), k=1)
        self._parity(
            torch_layer, np_layer, x, src_mask=causal, is_causal=True, label="causal"
        )

    # ------------------------------------------------------------------
    # Training mode / dropout
    # ------------------------------------------------------------------

    def test_training_mode_with_dropout_zero(self):
        torch_layer, np_layer = self._make_pair(
            16, 4, dim_feedforward=32, dropout=0.0
        )
        x = self.rng.standard_normal((5, 3, 16))
        self._parity(torch_layer, np_layer, x, training=True, label="train p=0")

    def test_training_mode_dropout_backward_is_finite(self):
        _, np_layer = self._make_pair(16, 4, dim_feedforward=32, dropout=0.5)
        np_layer.train()
        x = Tensor(self.rng.standard_normal((5, 3, 16)), requires_grad=True)
        out1 = np_layer(x)
        out2 = np_layer(x)
        self.assertTrue(np.all(np.isfinite(out1.data)))
        self.assertFalse(np.allclose(out1.data, out2.data))
        loss = (out1 * out1).mean()
        loss.backward()
        self.assertTrue(np.all(np.isfinite(x.grad.data)))
        for name, param in np_layer.named_parameters():
            self.assertIsNotNone(param.grad, f"missing grad for {name}")
            self.assertTrue(np.all(np.isfinite(param.grad.data)), f"non-finite grad {name}")

    # ------------------------------------------------------------------
    # Independent gradient check and optimizer parity
    # ------------------------------------------------------------------

    def test_finite_differences_for_all_parameters(self):
        _, layer = self._make_pair(
            4,
            2,
            dim_feedforward=8,
            dropout=0.0,
            activation="gelu",
            layer_norm_eps=1e-5,
        )
        layer.eval()
        x0 = self.rng.standard_normal((3, 2, 4))
        coef = self.rng.standard_normal((3, 2, 4))

        def compute_loss():
            with no_grad():
                out = layer(Tensor(x0.copy()))
                loss = (out * Tensor(coef)).sum()
            return float(loss.item())

        x_tensor = Tensor(x0.copy(), requires_grad=True)
        out = layer(x_tensor)
        loss = (out * Tensor(coef)).sum()
        loss.backward()
        analytic_x = x_tensor.grad.data.copy()
        analytic = {
            name: param.grad.data.copy() for name, param in layer.named_parameters()
        }

        eps = 1e-6
        numeric_x = np.zeros_like(x0)
        for index in np.ndindex(x0.shape):
            old = x0[index]
            x0[index] = old + eps
            plus = compute_loss()
            x0[index] = old - eps
            minus = compute_loss()
            x0[index] = old
            numeric_x[index] = (plus - minus) / (2.0 * eps)
        assert_close(self, analytic_x, numeric_x, atol=2e-6, rtol=2e-6, msg="fd input")

        for name, param in layer.named_parameters():
            numeric = np.zeros_like(param.data)
            for index in np.ndindex(param.shape):
                old = param.data[index]
                param.data[index] = old + eps
                plus = compute_loss()
                param.data[index] = old - eps
                minus = compute_loss()
                param.data[index] = old
                numeric[index] = (plus - minus) / (2.0 * eps)
            assert_close(
                self,
                analytic[name],
                numeric,
                atol=2e-6,
                rtol=2e-6,
                msg=f"fd {name}",
            )

    def test_sgd_step_matches_torch(self):
        torch_layer, np_layer = self._make_pair(16, 4, dim_feedforward=32)
        torch_layer.eval()
        np_layer.eval()
        copy_weights(torch_layer, np_layer)
        x = self.rng.standard_normal((4, 3, 16))
        target = self.rng.standard_normal((4, 3, 16))
        xt = torch.tensor(x)
        xn = Tensor(x.copy())
        torch_loss_fn = torch.nn.MSELoss()
        np_loss_fn = nn.MSELoss()
        torch_opt = torch.optim.SGD(torch_layer.parameters(), lr=0.05)
        np_opt = optim.SGD(np_layer.parameters(), lr=0.05)
        for step in range(3):
            torch_opt.zero_grad()
            np_opt.zero_grad()
            loss_t = torch_loss_fn(torch_layer(xt), torch.tensor(target))
            loss_n = np_loss_fn(np_layer(xn), Tensor(target))
            assert_close(
                self,
                np.asarray(loss_n.item()),
                np.asarray(loss_t.item()),
                1e-12,
                1e-10,
                f"step {step} loss",
            )
            loss_t.backward()
            loss_n.backward()
            torch_opt.step()
            np_opt.step()
            for name, param in np_layer.named_parameters():
                assert_close(
                    self,
                    param.data,
                    dict(torch_layer.named_parameters())[name].detach().numpy(),
                    1e-12,
                    1e-10,
                    f"step {step} param {name}",
                )

    def test_inference_no_grad_batch_first_float32(self):
        torch_layer, np_layer = self._make_pair(
            64, 4, dim_feedforward=128, batch_first=True, dtype=np.float32
        )
        torch_layer.eval()
        np_layer.eval()
        copy_weights(torch_layer, np_layer)
        x = self.rng.standard_normal((4, 16, 64)).astype(np.float32)
        with torch.no_grad():
            out_t = torch_layer(torch.tensor(x))
        with no_grad():
            out_n = np_layer(Tensor(x.copy()))
        assert_close(
            self,
            out_n.data,
            out_t.numpy(),
            2e-5,
            1e-4,
            "no_grad inference",
        )

    def test_state_dict_roundtrip(self):
        torch_layer, np_layer = self._make_pair(16, 4, dim_feedforward=32)
        copy_weights(torch_layer, np_layer)
        state = np_layer.state_dict()
        clone = nn.TransformerEncoderLayer(16, 4, dim_feedforward=32, dtype=np.float64)
        result = clone.load_state_dict(state)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        x = self.rng.standard_normal((3, 2, 16))
        out1 = np_layer.eval()(Tensor(x))
        out2 = clone.eval()(Tensor(x))
        assert_close(self, out1.data, out2.data)
        with self.assertRaises(RuntimeError):
            clone.load_state_dict({"bogus": np.zeros(1)})


if __name__ == "__main__":
    unittest.main()
