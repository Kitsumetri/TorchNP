"""Parity tests for torch.nn.TransformerEncoder."""

from __future__ import annotations

import unittest

import numpy as np
import torch

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nptorch import Tensor, nn, no_grad
from tests.utils import (
    assert_close,
    assert_param_grads_close,
    copy_weights,
)


class TestTransformerEncoder(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(31)

    def _make_pair(
        self,
        num_layers: int = 3,
        d_model: int = 16,
        nhead: int = 4,
        dim_feedforward: int = 32,
        dtype=np.float64,
        norm: bool = False,
        batch_first: bool = False,
        enable_nested_tensor: bool = False,
        **layer_kwargs,
    ):
        np_dtype = np.dtype(dtype)
        torch_dtype = torch.float64 if np_dtype == np.float64 else torch.float32
        torch_layer = torch.nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=dim_feedforward,
            batch_first=batch_first,
            dtype=torch_dtype,
            **layer_kwargs,
        )
        np_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=dim_feedforward,
            batch_first=batch_first,
            dtype=np_dtype,
            **layer_kwargs,
        )
        torch_norm = (
            torch.nn.LayerNorm(d_model, dtype=torch_dtype) if norm else None
        )
        np_norm = nn.LayerNorm(d_model, dtype=np_dtype) if norm else None
        torch_encoder = torch.nn.TransformerEncoder(
            torch_layer,
            num_layers,
            norm=torch_norm,
            enable_nested_tensor=enable_nested_tensor,
        )
        np_encoder = nn.TransformerEncoder(
            np_layer,
            num_layers,
            norm=np_norm,
            enable_nested_tensor=enable_nested_tensor,
        )
        return torch_encoder, np_encoder

    def _parity(
        self,
        torch_encoder,
        np_encoder,
        x: np.ndarray,
        *,
        mask=None,
        src_key_padding_mask=None,
        is_causal=None,
        training: bool = False,
        atol: float = 1e-12,
        rtol: float = 1e-10,
        label: str = "",
    ):
        if training:
            torch_encoder.train()
            np_encoder.train()
        else:
            torch_encoder.eval()
            np_encoder.eval()
        torch_encoder.zero_grad(set_to_none=True)
        np_encoder.zero_grad(set_to_none=True)
        copy_weights(torch_encoder, np_encoder)

        def to_torch(value):
            return None if value is None else torch.tensor(value)

        xt = torch.tensor(x, requires_grad=True)
        xn = Tensor(x.copy(), requires_grad=True)
        out_t = torch_encoder(
            xt,
            mask=to_torch(mask),
            src_key_padding_mask=to_torch(src_key_padding_mask),
            is_causal=is_causal,
        )
        out_n = np_encoder(
            xn,
            mask=mask,
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
            self,
            xn.grad.data,
            xt.grad.detach().numpy(),
            atol,
            rtol,
            f"{label} input grad",
        )
        assert_param_grads_close(self, torch_encoder, np_encoder, atol, rtol, label)

    # ------------------------------------------------------------------
    # Construction / state dict
    # ------------------------------------------------------------------

    def test_state_dict_and_clones(self):
        torch_encoder, np_encoder = self._make_pair(num_layers=3, norm=True)
        self.assertEqual(
            list(torch_encoder.state_dict().keys()),
            list(np_encoder.state_dict().keys()),
        )
        self.assertEqual(
            [name for name, _ in torch_encoder.named_parameters()],
            [name for name, _ in np_encoder.named_parameters()],
        )
        self.assertEqual(len(np_encoder.layers), 3)
        self.assertEqual(np_encoder.num_layers, 3)
        self.assertFalse(np_encoder.use_nested_tensor)
        self.assertTrue(np_encoder.mask_check)
        self.assertIsNot(np_encoder.layers[0], np_encoder.layers[1])
        self.assertIsNot(
            np_encoder.layers[0].linear1.weight,
            np_encoder.layers[1].linear1.weight,
        )
        self.assertFalse(
            np.shares_memory(
                np_encoder.layers[0].linear1.weight.data,
                np_encoder.layers[1].linear1.weight.data,
            )
        )

    def test_zero_layers(self):
        torch_encoder, np_encoder = self._make_pair(num_layers=0)
        self.assertEqual(len(np_encoder.layers), 0)
        self.assertEqual(len(torch_encoder.layers), 0)
        self.assertEqual(list(np_encoder.state_dict()), [])
        self.assertEqual(list(torch_encoder.state_dict()), [])

    def test_state_dict_roundtrip(self):
        torch_encoder, np_encoder = self._make_pair(num_layers=2, norm=True)
        copy_weights(torch_encoder, np_encoder)
        state = np_encoder.state_dict()
        clone = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                16, 4, dim_feedforward=32, dtype=np.float64
            ),
            2,
            norm=nn.LayerNorm(16, dtype=np.float64),
        )
        result = clone.load_state_dict(state)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        x = self.rng.standard_normal((3, 2, 16))
        assert_close(
            self,
            np_encoder.eval()(Tensor(x)).data,
            clone.eval()(Tensor(x)).data,
            0.0,
            0.0,
        )

    # ------------------------------------------------------------------
    # Forward / backward parity
    # ------------------------------------------------------------------

    def test_three_layers_float64(self):
        torch_encoder, np_encoder = self._make_pair(num_layers=3)
        x = self.rng.standard_normal((5, 3, 16))
        self._parity(torch_encoder, np_encoder, x, label="3 layers")

    def test_two_layers_float32(self):
        torch_encoder, np_encoder = self._make_pair(
            num_layers=2,
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            dtype=np.float32,
        )
        x = self.rng.standard_normal((4, 2, 32)).astype(np.float32)
        self._parity(
            torch_encoder,
            np_encoder,
            x,
            atol=2e-5,
            rtol=2e-4,
            label="float32",
        )

    def test_final_norm(self):
        torch_encoder, np_encoder = self._make_pair(num_layers=2, norm=True)
        x = self.rng.standard_normal((5, 3, 16))
        self._parity(torch_encoder, np_encoder, x, label="norm")

    def test_batch_first_with_masks(self):
        torch_encoder, np_encoder = self._make_pair(
            num_layers=2, batch_first=True
        )
        x = self.rng.standard_normal((3, 5, 16))
        mask = np.triu(np.ones((5, 5), dtype=bool), k=1)
        key_padding_mask = np.zeros((3, 5), dtype=bool)
        key_padding_mask[0, -2:] = True
        self._parity(
            torch_encoder,
            np_encoder,
            x,
            mask=mask,
            src_key_padding_mask=key_padding_mask,
            label="batch_first masks",
        )

    def test_float_masks_are_passed_through(self):
        torch_encoder, np_encoder = self._make_pair(num_layers=2)
        x = self.rng.standard_normal((3, 2, 16))
        mask = np.zeros((3, 3), dtype=np.float64)
        mask[:, -1] = -1e9
        key_padding_mask = np.zeros((2, 3), dtype=np.float64)
        key_padding_mask[0, -1] = -1e9
        self._parity(
            torch_encoder,
            np_encoder,
            x,
            mask=mask,
            src_key_padding_mask=key_padding_mask,
            label="float masks",
        )

    def test_is_causal_auto_and_explicit(self):
        causal = np.triu(np.ones((5, 5), dtype=bool), k=1)
        x = self.rng.standard_normal((5, 2, 16))
        for hint in (None, True, False):
            with self.subTest(is_causal=hint):
                torch_encoder, np_encoder = self._make_pair(num_layers=2)
                self._parity(
                    torch_encoder,
                    np_encoder,
                    x,
                    mask=causal,
                    is_causal=hint,
                    label=f"causal={hint}",
                )

    def test_is_causal_detection_rejects_non_causal_mask(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[:, -1] = True  # not a causal mask
        x = self.rng.standard_normal((5, 2, 16))
        torch_encoder, np_encoder = self._make_pair(num_layers=2)
        self._parity(
            torch_encoder,
            np_encoder,
            x,
            mask=mask,
            is_causal=None,
            label="non-causal",
        )

    def test_unbatched_2d_input(self):
        torch_encoder, np_encoder = self._make_pair(num_layers=2)
        x = self.rng.standard_normal((5, 16))
        self._parity(torch_encoder, np_encoder, x, label="2d")

    def test_training_mode_with_dropout_zero(self):
        torch_encoder, np_encoder = self._make_pair(
            num_layers=2, dropout=0.0
        )
        x = self.rng.standard_normal((5, 3, 16))
        self._parity(torch_encoder, np_encoder, x, training=True, label="train")

    def test_mask_type_validation(self):
        _, np_encoder = self._make_pair(num_layers=1)
        x = Tensor(self.rng.standard_normal((3, 2, 16)))
        with self.assertRaises(AssertionError):
            np_encoder(x, mask=np.ones((3, 3), dtype=np.int64))
        with self.assertRaises(AssertionError):
            np_encoder(x, src_key_padding_mask=np.ones((2, 3), dtype=np.int64))

    # ------------------------------------------------------------------
    # Inference / nested-tensor fallback
    # ------------------------------------------------------------------

    def test_no_grad_batch_first_inference(self):
        torch_encoder, np_encoder = self._make_pair(
            num_layers=2,
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            dtype=np.float32,
            batch_first=True,
            enable_nested_tensor=False,
        )
        copy_weights(torch_encoder, np_encoder)
        torch_encoder.eval()
        np_encoder.eval()
        x = self.rng.standard_normal((3, 8, 32)).astype(np.float32)
        key_padding_mask = np.zeros((3, 8), dtype=bool)
        key_padding_mask[0, -3:] = True
        key_padding_mask[1, -1:] = True
        with torch.no_grad():
            out_t = torch_encoder(
                torch.tensor(x),
                src_key_padding_mask=torch.tensor(key_padding_mask),
            )
        with no_grad():
            out_n = np_encoder(
                Tensor(x.copy()),
                src_key_padding_mask=key_padding_mask,
            )
        assert_close(
            self, out_n.data, out_t.numpy(), 2e-5, 1e-4, "inference"
        )

    def test_nested_tensor_flag_is_disabled(self):
        _, np_encoder = self._make_pair(
            num_layers=2,
            batch_first=True,
            enable_nested_tensor=True,
        )
        self.assertTrue(np_encoder.enable_nested_tensor)
        self.assertFalse(np_encoder.use_nested_tensor)
        self.assertTrue(np_encoder.mask_check)


if __name__ == "__main__":
    unittest.main()
