"""Parity, finite-difference and API tests for ``TransformerEncoderLayer``."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from src.nptorch import TransformerEncoderLayer
from src.nptorch.nn import ops
from tests.utils import assert_close, make_pair, parity_check


class ForwardBackwardParityTest(unittest.TestCase):
    def _run(
        self,
        *,
        d_model: int = 8,
        nhead: int = 2,
        dim_feedforward: int = 12,
        seq_len: int = 4,
        batch_size: int = 3,
        seed: int = 0,
        **kwargs,
    ) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        x_seq = rng.normal(size=(seq_len, batch_size, d_model))
        return parity_check(
            self,
            x_seq=x_seq,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            seed=seed,
            **kwargs,
        )

    def test_default_post_norm(self) -> None:
        result = self._run()
        self.assertLess(result["forward"], 2e-13)
        self.assertLess(result["param_grads"], 2e-12)

    def test_batch_first(self) -> None:
        result = self._run(batch_first=True)
        self.assertLess(result["forward"], 2e-13)
        self.assertLess(result["param_grads"], 2e-12)

    def test_pre_norm(self) -> None:
        result = self._run(norm_first=True)
        self.assertLess(result["forward"], 2e-13)
        self.assertLess(result["param_grads"], 2e-12)

    def test_no_bias(self) -> None:
        result = self._run(bias=False)
        self.assertLess(result["forward"], 2e-13)
        self.assertLess(result["param_grads"], 2e-12)

    def test_gelu(self) -> None:
        result = self._run(activation="gelu")
        self.assertLess(result["forward"], 2e-13)
        self.assertLess(result["param_grads"], 2e-12)

    def test_gelu_pre_norm_batch_first(self) -> None:
        result = self._run(
            activation="gelu", norm_first=True, batch_first=True
        )
        self.assertLess(result["forward"], 2e-13)
        self.assertLess(result["param_grads"], 2e-12)

    def test_other_dimensions(self) -> None:
        result = self._run(
            d_model=12, nhead=3, dim_feedforward=20, seq_len=5, batch_size=2
        )
        self.assertLess(result["forward"], 2e-13)
        self.assertLess(result["param_grads"], 2e-12)

    def test_custom_layer_norm_eps(self) -> None:
        result = self._run(layer_norm_eps=1e-3)
        self.assertLess(result["forward"], 2e-13)
        self.assertLess(result["param_grads"], 2e-12)

    def test_unbatched_2d_input(self) -> None:
        rng = np.random.default_rng(7)
        x_seq = rng.normal(size=(4, 8))
        result = parity_check(
            self,
            x_seq=x_seq,
            d_model=8,
            nhead=2,
            dim_feedforward=12,
        )
        self.assertLess(result["forward"], 2e-13)
        self.assertLess(result["param_grads"], 2e-12)

    def test_batch_first_two_dimensional_input(self) -> None:
        rng = np.random.default_rng(8)
        x_seq = rng.normal(size=(5, 8))
        result = parity_check(
            self,
            x_seq=x_seq,
            d_model=8,
            nhead=2,
            dim_feedforward=12,
            batch_first=True,
        )
        self.assertLess(result["forward"], 2e-13)
        self.assertLess(result["param_grads"], 2e-12)

    def test_pre_norm_unbatched(self) -> None:
        rng = np.random.default_rng(9)
        x_seq = rng.normal(size=(4, 8))
        result = parity_check(
            self,
            x_seq=x_seq,
            d_model=8,
            nhead=2,
            dim_feedforward=12,
            norm_first=True,
        )
        self.assertLess(result["forward"], 2e-13)
        self.assertLess(result["param_grads"], 2e-12)


class MaskParityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.seq_len = 4
        self.batch_size = 3
        self.d_model = 8
        self.nhead = 2
        self.rng = np.random.default_rng(11)
        self.x_seq = self.rng.normal(
            size=(self.seq_len, self.batch_size, self.d_model)
        )
        self.bool_mask = np.zeros((self.seq_len, self.seq_len), dtype=bool)
        self.bool_mask[0, 1] = True
        self.bool_mask[2, 3] = True
        self.float_mask = np.zeros((self.seq_len, self.seq_len))
        self.float_mask[0, 1] = -np.inf
        self.float_mask[2, 3] = -np.inf

    def _run(self, **kwargs) -> dict[str, float]:
        return parity_check(
            self,
            x_seq=self.x_seq,
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=12,
            **kwargs,
        )

    def test_bool_src_mask(self) -> None:
        self.assertLess(self._run(src_mask=self.bool_mask)["param_grads"], 2e-12)

    def test_float_src_mask(self) -> None:
        self.assertLess(
            self._run(src_mask=self.float_mask)["param_grads"], 2e-12
        )

    def test_3d_src_mask(self) -> None:
        mask = np.zeros(
            (self.batch_size * self.nhead, self.seq_len, self.seq_len),
            dtype=bool,
        )
        mask[:, 1, 2] = True
        mask[0, 3, 0] = True
        self.assertLess(self._run(src_mask=mask)["param_grads"], 2e-12)

    def test_3d_float_src_mask(self) -> None:
        mask = np.zeros(
            (self.batch_size * self.nhead, self.seq_len, self.seq_len)
        )
        mask[:, 1, 2] = -np.inf
        mask[0, 3, 0] = -np.inf
        result = self._run(src_mask=mask)
        self.assertLess(result["param_grads"], 2e-12)

    def test_3d_mask_with_key_padding(self) -> None:
        mask = np.zeros(
            (self.batch_size * self.nhead, self.seq_len, self.seq_len),
            dtype=bool,
        )
        mask[:, 1, 2] = True
        kpm = np.zeros((self.batch_size, self.seq_len), dtype=bool)
        kpm[1, 0] = True
        result = self._run(src_mask=mask, src_key_padding_mask=kpm)
        self.assertLess(result["param_grads"], 2e-12)

    def test_3d_mask_on_unbatched_input(self) -> None:
        x_2d = self.x_seq[:, 0, :]
        mask = np.zeros((self.nhead, self.seq_len, self.seq_len), dtype=bool)
        mask[:, 1, 3] = True
        result = parity_check(
            self,
            x_seq=x_2d,
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=12,
            src_mask=mask,
        )
        self.assertLess(result["param_grads"], 2e-12)

    def test_key_padding_mask_bool(self) -> None:
        kpm = np.zeros((self.batch_size, self.seq_len), dtype=bool)
        kpm[1, 0] = True
        result = self._run(src_key_padding_mask=kpm)
        self.assertLess(result["param_grads"], 2e-12)

    def test_key_padding_mask_float_inf(self) -> None:
        kpm = np.zeros((self.batch_size, self.seq_len))
        kpm[1, 0] = -np.inf
        result = self._run(src_key_padding_mask=kpm)
        self.assertLess(result["param_grads"], 2e-12)

    def test_key_padding_mask_float_finite(self) -> None:
        kpm = np.zeros((self.batch_size, self.seq_len))
        kpm[1, 0] = -1e9
        result = self._run(src_key_padding_mask=kpm)
        self.assertLess(result["param_grads"], 2e-12)

    def test_combined_masks(self) -> None:
        kpm = np.zeros((self.batch_size, self.seq_len), dtype=bool)
        kpm[1, 0] = True
        result = self._run(
            src_mask=self.bool_mask, src_key_padding_mask=kpm
        )
        self.assertLess(result["param_grads"], 2e-12)

    def test_pre_norm_combined_masks(self) -> None:
        kpm = np.zeros((self.batch_size, self.seq_len), dtype=bool)
        kpm[1, 0] = True
        result = self._run(
            src_mask=self.bool_mask,
            src_key_padding_mask=kpm,
            norm_first=True,
        )
        self.assertLess(result["param_grads"], 2e-12)

    def test_batch_first_with_mask(self) -> None:
        kpm = np.zeros((self.batch_size, self.seq_len), dtype=bool)
        kpm[1, 0] = True
        result = self._run(
            src_mask=self.bool_mask,
            src_key_padding_mask=kpm,
            batch_first=True,
        )
        self.assertLess(result["param_grads"], 2e-12)

    def test_unbatched_with_masks(self) -> None:
        x_2d = self.x_seq[:, 0, :]
        kpm = np.zeros((self.seq_len,), dtype=bool)
        kpm[0] = True
        result = parity_check(
            self,
            x_seq=x_2d,
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=12,
            src_mask=self.bool_mask,
            src_key_padding_mask=kpm,
        )
        self.assertLess(result["param_grads"], 2e-12)

    def test_causal_hint(self) -> None:
        causal = np.triu(np.ones((self.seq_len, self.seq_len), dtype=bool), 1)
        result = self._run(src_mask=causal, is_causal=True)
        self.assertLess(result["param_grads"], 2e-12)
        self.assertLess(result["forward"], 2e-13)

    def test_causal_hint_ignores_non_causal_mask_without_padding(self) -> None:
        non_causal = ~np.eye(self.seq_len, dtype=bool)
        result = self._run(src_mask=non_causal, is_causal=True)
        self.assertLess(result["param_grads"], 2e-12)

    def test_causal_hint_with_key_padding_uses_mask(self) -> None:
        # Deliberately non-causal src_mask: with key padding present, torch
        # uses src_mask and ignores the causal hint.  If the implementation
        # instead applied a causal mask, this test would diverge.
        mask = np.zeros((self.seq_len, self.seq_len), dtype=bool)
        mask[:, -1] = True
        kpm = np.zeros((self.batch_size, self.seq_len), dtype=bool)
        kpm[1, 0] = True
        result = self._run(
            src_mask=mask, src_key_padding_mask=kpm, is_causal=True
        )
        self.assertLess(result["param_grads"], 2e-12)
        self.assertLess(result["forward"], 2e-13)

    def test_fully_masked_row_matches_torch(self) -> None:
        mask = np.zeros((self.seq_len, self.seq_len), dtype=bool)
        mask[0, :] = True
        result = self._run(src_mask=mask)
        self.assertLess(result["forward"], 2e-13)
        self.assertLess(result["param_grads"], 2e-12)

    def test_fully_masked_row_is_finite(self) -> None:
        mask = np.zeros((self.seq_len, self.seq_len), dtype=bool)
        mask[0, :] = True
        layer = TransformerEncoderLayer(
            self.d_model, self.nhead, dim_feedforward=12, dropout=0.0
        ).eval()
        out = layer(self.x_seq, src_mask=mask)
        self.assertTrue(np.all(np.isfinite(out)))
        rng = np.random.default_rng(0)
        dout = rng.normal(size=out.shape)
        dx = layer.backward(dout)
        self.assertTrue(np.all(np.isfinite(dx)))
        for gradient in layer.grads.values():
            self.assertTrue(np.all(np.isfinite(gradient)))

    def test_fully_masked_sample_with_key_padding(self) -> None:
        kpm = np.zeros((self.batch_size, self.seq_len), dtype=bool)
        kpm[1, :] = True
        layer = TransformerEncoderLayer(
            self.d_model, self.nhead, dim_feedforward=12, dropout=0.0
        ).eval()
        out = layer(self.x_seq, src_key_padding_mask=kpm)
        self.assertTrue(np.all(np.isfinite(out)))


class FiniteDifferenceTest(unittest.TestCase):
    """Checks the analytic gradients without using torch autograd."""

    def _numeric_gradients(
        self, layer: TransformerEncoderLayer, x: np.ndarray, dout: np.ndarray
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        eps = 1e-6

        def loss() -> float:
            out = layer.forward(x)
            return float(np.sum(out * dout))

        numeric_x = np.zeros_like(x)
        for index in np.ndindex(x.shape):
            original = x[index]
            x[index] = original + eps
            plus = loss()
            x[index] = original - eps
            minus = loss()
            x[index] = original
            numeric_x[index] = (plus - minus) / (2.0 * eps)

        numeric_params: dict[str, np.ndarray] = {}
        for name, parameter in layer.params.items():
            gradient = np.zeros_like(parameter)
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + eps
                plus = loss()
                parameter[index] = original - eps
                minus = loss()
                parameter[index] = original
                gradient[index] = (plus - minus) / (2.0 * eps)
            numeric_params[name] = gradient
        return numeric_x, numeric_params

    def _check(self, norm_first: bool) -> None:
        layer, _ = make_pair(
            4, 2, 6, norm_first=norm_first, dropout=0.0
        )
        layer.eval()
        rng = np.random.default_rng(21)
        x = rng.normal(size=(3, 2, 4))
        dout = rng.normal(size=(3, 2, 4))

        layer.zero_grad()
        layer.forward(x)
        analytic_x = layer.backward(dout)
        analytic_params = {name: value.copy() for name, value in layer.grads.items()}
        numeric_x, numeric_params = self._numeric_gradients(layer, x, dout)

        np.testing.assert_allclose(
            analytic_x, numeric_x, atol=2e-7, rtol=1e-5, err_msg="finite-diff dX"
        )
        self.assertEqual(set(analytic_params), set(numeric_params))
        for name in analytic_params:
            np.testing.assert_allclose(
                analytic_params[name],
                numeric_params[name],
                atol=2e-7,
                rtol=1e-5,
                err_msg=f"finite-diff {name}",
            )

    def test_post_norm(self) -> None:
        self._check(False)

    def test_pre_norm(self) -> None:
        self._check(True)


class DropoutBehaviorTest(unittest.TestCase):
    def test_eval_mode_disables_dropout(self) -> None:
        rng = np.random.default_rng(31)
        x = rng.normal(size=(4, 2, 8))
        torch_layer = torch.nn.TransformerEncoderLayer(
            8, 2, dim_feedforward=12, dropout=0.0, dtype=torch.float64
        ).eval()
        numpy_layer = TransformerEncoderLayer(
            8, 2, dim_feedforward=12, dropout=0.4
        ).eval()
        numpy_layer.load_state_dict(torch_layer.state_dict())
        out = numpy_layer(x)
        expected = torch_layer(torch.tensor(x)).detach().numpy()
        assert_close(self, out, expected, msg="eval dropout")

    def test_train_mode_with_p_one(self) -> None:
        layer, _ = make_pair(4, 2, 6, dropout=1.0)
        layer.train()
        x = np.arange(24.0).reshape(3, 2, 4) / 10.0
        out = layer(x)
        self.assertTrue(np.all(np.isfinite(out)))
        # Both sublayer outputs are zeroed, so the result is LN2(LN1(x)).
        gamma1 = layer.params["norm1.weight"]
        gamma2 = layer.params["norm2.weight"]
        beta1 = layer.params["norm1.bias"]
        beta2 = layer.params["norm2.bias"]
        z, _ = ops.layer_norm_forward(x, gamma1, beta1, layer.layer_norm_eps)
        expected, _ = ops.layer_norm_forward(
            z, gamma2, beta2, layer.layer_norm_eps
        )
        assert_close(self, out, expected, msg="p=1 forward")
        dx = layer.backward(np.ones_like(out))
        self.assertTrue(np.all(np.isfinite(dx)))
        for gradient in layer.grads.values():
            self.assertTrue(np.all(np.isfinite(gradient)))

    def test_seed_makes_training_forward_reproducible(self) -> None:
        layer = TransformerEncoderLayer(8, 2, dim_feedforward=12, dropout=0.5)
        layer.train()
        x = np.random.default_rng(1).normal(size=(4, 2, 8))
        np.random.seed(123)
        first = layer(x)
        np.random.seed(123)
        second = layer(x)
        assert_close(self, first, second, msg="seeded dropout")
        np.random.seed(124)
        third = layer(x)
        self.assertFalse(np.allclose(first, third))

    def test_finite_difference_with_dropout(self) -> None:
        layer, _ = make_pair(4, 2, 6, dropout=0.25)
        layer.train()
        rng = np.random.default_rng(41)
        x = rng.normal(size=(3, 2, 4))
        dout = rng.normal(size=(3, 2, 4))

        def loss() -> float:
            return float(np.sum(layer.forward(x) * dout))

        np.random.seed(2024)
        layer.zero_grad()
        layer.forward(x)
        analytic_x = layer.backward(dout)
        analytic_params = {
            name: value.copy() for name, value in layer.grads.items()
        }

        eps = 1e-6
        numeric_x = np.zeros_like(x)
        for index in np.ndindex(x.shape):
            original = x[index]
            np.random.seed(2024)
            x[index] = original + eps
            plus = loss()
            np.random.seed(2024)
            x[index] = original - eps
            minus = loss()
            x[index] = original
            numeric_x[index] = (plus - minus) / (2.0 * eps)
        np.testing.assert_allclose(
            analytic_x,
            numeric_x,
            atol=2e-7,
            rtol=1e-5,
            err_msg="dropout dX",
        )

        numeric_params = {}
        for name, parameter in layer.params.items():
            gradient = np.zeros_like(parameter)
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                np.random.seed(2024)
                parameter[index] = original + eps
                plus = loss()
                np.random.seed(2024)
                parameter[index] = original - eps
                minus = loss()
                parameter[index] = original
                gradient[index] = (plus - minus) / (2.0 * eps)
            numeric_params[name] = gradient
        self.assertEqual(set(analytic_params), set(numeric_params))
        for name in analytic_params:
            np.testing.assert_allclose(
                analytic_params[name],
                numeric_params[name],
                atol=2e-7,
                rtol=1e-5,
                err_msg=f"dropout {name}",
            )

class GradientAccumulationTest(unittest.TestCase):
    def test_backward_accumulates(self) -> None:
        layer, _ = make_pair(4, 2, 6, dropout=0.0)
        x = np.random.default_rng(51).normal(size=(3, 2, 4))
        dout = np.random.default_rng(52).normal(size=(3, 2, 4))

        layer.zero_grad()
        layer.forward(x)
        first_dx = layer.backward(dout)
        first = {name: value.copy() for name, value in layer.grads.items()}
        second_dx = layer.backward(dout)
        assert_close(self, second_dx, first_dx, msg="second dX")
        for name, value in first.items():
            assert_close(
                self, layer.grads[name], 2.0 * value, msg=f"accumulated {name}"
            )

        layer.zero_grad()
        self.assertEqual(layer.grads, {})


class InitializationTest(unittest.TestCase):
    def test_fresh_layer_uses_torch_initialization_ranges(self) -> None:
        d_model, ffn = 8, 12
        layer = TransformerEncoderLayer(d_model, 2, dim_feedforward=ffn)

        for name in (
            "self_attn.in_proj_bias",
            "self_attn.out_proj.bias",
            "norm1.bias",
            "norm2.bias",
        ):
            assert_close(
                self, layer.params[name], np.zeros_like(layer.params[name])
            )
        for name in ("norm1.weight", "norm2.weight"):
            assert_close(
                self, layer.params[name], np.ones_like(layer.params[name])
            )

        # torch.nn.Linear bias: U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
        self.assertLessEqual(
            float(np.max(np.abs(layer.params["linear1.bias"]))),
            1.0 / math.sqrt(d_model) + 1e-15,
        )
        self.assertLessEqual(
            float(np.max(np.abs(layer.params["linear2.bias"]))),
            1.0 / math.sqrt(ffn) + 1e-15,
        )
        self.assertGreater(
            float(np.max(np.abs(layer.params["linear1.bias"]))), 0.0
        )
        self.assertGreater(
            float(np.max(np.abs(layer.params["linear2.bias"]))), 0.0
        )

        # Xavier for in-projection; Kaiming (a=sqrt(5)) for the rest.
        xavier = math.sqrt(6.0 / (d_model + 3 * d_model))
        self.assertLessEqual(
            float(np.max(np.abs(layer.params["self_attn.in_proj_weight"]))),
            xavier + 1e-15,
        )
        self.assertLessEqual(
            float(np.max(np.abs(layer.params["self_attn.out_proj.weight"]))),
            1.0 / math.sqrt(d_model) + 1e-15,
        )
        self.assertLessEqual(
            float(np.max(np.abs(layer.params["linear1.weight"]))),
            1.0 / math.sqrt(d_model) + 1e-15,
        )
        self.assertLessEqual(
            float(np.max(np.abs(layer.params["linear2.weight"]))),
            1.0 / math.sqrt(ffn) + 1e-15,
        )


class StateDictTest(unittest.TestCase):
    def test_keys_match_torch(self) -> None:
        for bias in (True, False):
            layer, torch_layer = make_pair(8, 2, 12, bias=bias)
            self.assertEqual(
                list(layer.params), list(dict(torch_layer.named_parameters()))
            )
            self.assertEqual(
                set(layer.state_dict()), set(torch_layer.state_dict())
            )

    def test_round_trip_and_copy_semantics(self) -> None:
        layer, _ = make_pair(8, 2, 12)
        state = layer.state_dict()
        original = layer.params["linear1.weight"].copy()
        state["linear1.weight"][...] = 0.0
        assert_close(
            self, layer.params["linear1.weight"], original, msg="state_dict copy"
        )

        other = TransformerEncoderLayer(8, 2, dim_feedforward=12)
        other.load_state_dict(layer.state_dict())
        for name in layer.params:
            assert_close(
                self, other.params[name], layer.params[name], msg=f"load {name}"
            )

    def test_load_torch_tensors(self) -> None:
        layer, torch_layer = make_pair(8, 2, 12)
        other = TransformerEncoderLayer(8, 2, dim_feedforward=12)
        other.load_state_dict(torch_layer.state_dict())
        for name, parameter in torch_layer.named_parameters():
            assert_close(
                self, other.params[name], parameter.detach().numpy()
            )

    def test_strict_errors(self) -> None:
        layer = TransformerEncoderLayer(8, 2, dim_feedforward=12)
        state = layer.state_dict()
        missing = dict(state)
        del missing["linear1.weight"]
        with self.assertRaises(RuntimeError):
            layer.load_state_dict(missing)

        unexpected = dict(state)
        unexpected["extra"] = np.zeros(1)
        with self.assertRaises(RuntimeError):
            layer.load_state_dict(unexpected)

        wrong_shape = dict(state)
        wrong_shape["linear1.weight"] = np.zeros((3, 3))
        with self.assertRaises(RuntimeError):
            layer.load_state_dict(wrong_shape)

    def test_non_strict_ignores_extra_keys(self) -> None:
        layer = TransformerEncoderLayer(8, 2, dim_feedforward=12)
        state = layer.state_dict()
        state["extra"] = np.zeros(1)
        layer.load_state_dict(state, strict=False)

    def test_load_clears_gradients(self) -> None:
        layer, _ = make_pair(4, 2, 6)
        x = np.zeros((3, 2, 4))
        layer.forward(x)
        layer.backward(np.ones((3, 2, 4)))
        self.assertTrue(layer.grads)
        layer.load_state_dict(layer.state_dict())
        self.assertEqual(layer.grads, {})


class ApiTest(unittest.TestCase):
    def test_parameter_accessors(self) -> None:
        layer = TransformerEncoderLayer(8, 2, dim_feedforward=12)
        named = list(layer.named_parameters())
        self.assertEqual([name for name, _ in named], list(layer.params))
        for (_, value), parameter in zip(named, layer.parameters()):
            self.assertIs(value, parameter)

    def test_call_and_repr(self) -> None:
        layer = TransformerEncoderLayer(8, 2, dim_feedforward=12).eval()
        x = np.zeros((3, 2, 8))
        assert_close(self, layer(x), layer.forward(x), msg="__call__")
        representation = repr(layer)
        self.assertIn("d_model=8", representation)
        self.assertIn("nhead=2", representation)
        self.assertIn("norm_first=False", representation)

    def test_train_eval(self) -> None:
        layer = TransformerEncoderLayer(8, 2, dim_feedforward=12)
        self.assertIs(layer.eval(), layer)
        self.assertFalse(layer.training)
        self.assertIs(layer.train(), layer)
        self.assertTrue(layer.training)

    def test_activation_aliases(self) -> None:
        layer = TransformerEncoderLayer(
            8, 2, dim_feedforward=12, activation=torch.nn.functional.relu
        )
        self.assertEqual(layer.activation, "relu")
        layer = TransformerEncoderLayer(
            8, 2, dim_feedforward=12, activation=torch.nn.functional.gelu
        )
        self.assertEqual(layer.activation, "gelu")
        with self.assertRaises(NotImplementedError):
            TransformerEncoderLayer(8, 2, activation=lambda x: x)
        with self.assertRaises(NotImplementedError):
            TransformerEncoderLayer(8, 2, activation="silu")
        with self.assertRaises(TypeError):
            TransformerEncoderLayer(8, 2, activation=3)

    def test_constructor_validation(self) -> None:
        with self.assertRaises(ValueError):
            TransformerEncoderLayer(0, 2)
        with self.assertRaises(ValueError):
            TransformerEncoderLayer(8, 0)
        with self.assertRaises(ValueError):
            TransformerEncoderLayer(8, 3)
        with self.assertRaises(ValueError):
            TransformerEncoderLayer(8, 2, dim_feedforward=0)
        with self.assertRaises(ValueError):
            TransformerEncoderLayer(8, 2, dropout=-0.1)
        with self.assertRaises(ValueError):
            TransformerEncoderLayer(8, 2, dropout=1.1)
        with self.assertRaises(ValueError):
            TransformerEncoderLayer(8, 2, layer_norm_eps=0.0)
        with self.assertRaises(TypeError):
            TransformerEncoderLayer(8.0, 2)
        with self.assertRaises(TypeError):
            TransformerEncoderLayer(8, 2, dim_feedforward=2.0)

    def test_input_validation(self) -> None:
        layer = TransformerEncoderLayer(8, 2, dim_feedforward=12)
        with self.assertRaises(RuntimeError):
            layer(np.zeros(8))
        with self.assertRaises(RuntimeError):
            layer(np.zeros((2, 2, 2, 8)))
        with self.assertRaises(RuntimeError):
            layer(np.zeros((4, 2, 7)))
        with self.assertRaises(TypeError):
            layer(np.array([["not", "numeric"]]))

    def test_mask_validation(self) -> None:
        layer = TransformerEncoderLayer(8, 2, dim_feedforward=12)
        x = np.zeros((4, 2, 8))
        with self.assertRaises(TypeError):
            layer(x, src_mask=np.zeros((4, 4), dtype=np.int64))
        with self.assertRaises(RuntimeError):
            layer(x, src_mask=np.zeros((3, 4), dtype=bool))
        with self.assertRaises(RuntimeError):
            layer(x, src_mask=np.zeros(4, dtype=bool))
        with self.assertRaises(RuntimeError):
            layer(x, src_mask=np.zeros((1, 4, 4), dtype=bool))
        with self.assertRaises(RuntimeError):
            layer(x, src_mask=np.zeros((3, 4, 4), dtype=bool))
        with self.assertRaises(RuntimeError):
            layer(x, src_key_padding_mask=np.zeros((2, 3), dtype=bool))
        with self.assertRaises(RuntimeError):
            layer(x, src_key_padding_mask=np.zeros((2, 4, 1), dtype=bool))
        with self.assertRaises(TypeError):
            layer(x, src_key_padding_mask=np.zeros((2, 4), dtype=np.int64))

    def test_unbatched_mask_validation(self) -> None:
        layer = TransformerEncoderLayer(8, 2, dim_feedforward=12)
        x = np.zeros((4, 8))
        with self.assertRaises(RuntimeError):
            layer(x, src_key_padding_mask=np.zeros((2, 4), dtype=bool))
        with self.assertRaises(RuntimeError):
            layer(x, src_key_padding_mask=np.zeros((4, 1, 1), dtype=bool))

    def test_is_causal_requires_mask(self) -> None:
        layer = TransformerEncoderLayer(8, 2, dim_feedforward=12)
        x = np.zeros((4, 2, 8))
        with self.assertRaises(RuntimeError):
            layer(x, is_causal=True)

    def test_backward_requires_forward(self) -> None:
        layer = TransformerEncoderLayer(8, 2, dim_feedforward=12)
        with self.assertRaises(RuntimeError):
            layer.backward(np.zeros((3, 2, 8)))

    def test_backward_shape_validation(self) -> None:
        layer = TransformerEncoderLayer(8, 2, dim_feedforward=12)
        layer.forward(np.zeros((3, 2, 8)))
        with self.assertRaises(RuntimeError):
            layer.backward(np.zeros((2, 3, 8)))

    def test_float32_input_is_promoted(self) -> None:
        layer = TransformerEncoderLayer(8, 2, dim_feedforward=12)
        out = layer(np.zeros((3, 2, 8), dtype=np.float32))
        self.assertEqual(out.dtype, np.float64)


if __name__ == "__main__":
    unittest.main()
