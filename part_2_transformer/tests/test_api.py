"""Public-API and error-path tests that complement the parity suites."""

from __future__ import annotations

import unittest

import numpy as np
import torch

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nptorch import Parameter, Tensor, init, nn, no_grad, optim
from src.nptorch import functional as F


class TestTensorApi(unittest.TestCase):
    def test_properties_and_conversions(self):
        x = Tensor(np.arange(6, dtype=np.float32).reshape(2, 3), requires_grad=True)
        self.assertEqual(x.shape, (2, 3))
        self.assertEqual(x.size(), (2, 3))
        self.assertEqual(x.size(0), 2)
        self.assertEqual(x.dim(), 2)
        self.assertEqual(x.ndim, 2)
        self.assertEqual(x.numel(), 6)
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(x.device, "cpu")
        self.assertFalse(x.is_cuda)
        self.assertTrue(x.is_leaf)
        self.assertIs(x.numpy(), x.data)
        self.assertEqual(x.tolist(), [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        self.assertEqual(x.T.shape, (3, 2))
        self.assertIs(x.contiguous(), x)
        self.assertEqual(len(x), 2)
        self.assertEqual(float(Tensor(2.5)), 2.5)
        self.assertEqual(Tensor(2.5).item(), 2.5)
        with self.assertRaises(RuntimeError):
            x.item()
        with self.assertRaises(ValueError):
            bool(x)

    def test_dtype_and_grad_management(self):
        x = Tensor(np.ones((2, 2), dtype=np.float32), requires_grad=True)
        self.assertIs(x.float(), x)
        y = x.double()
        self.assertEqual(y.dtype, np.float64)
        self.assertIs(y.to(np.float64), y)
        self.assertTrue(y.requires_grad)
        clone = x.clone()
        self.assertFalse(np.shares_memory(clone.data, x.data))
        self.assertTrue(clone.requires_grad)
        detached = x.detach()
        self.assertFalse(detached.requires_grad)
        self.assertFalse(x.requires_grad_(False).requires_grad)
        self.assertTrue(x.requires_grad_().requires_grad)
        with self.assertRaises(RuntimeError):
            Tensor(np.ones(2, dtype=np.int64)).requires_grad_()

        (x * 2.0).sum().backward()
        self.assertIsNotNone(x.grad)
        x.zero_grad(set_to_none=False)
        self.assertTrue(np.array_equal(x.grad.data, np.zeros((2, 2))))
        x.zero_grad()
        self.assertIsNone(x.grad)
        x.grad = Tensor(np.ones((2, 2)))
        self.assertTrue(np.array_equal(x.grad.data, np.ones((2, 2))))
        x.grad = None
        self.assertIsNone(x.grad)

    def test_backward_validation_and_seed(self):
        with self.assertRaises(RuntimeError):
            Tensor(np.ones(2)).backward()
        x = Tensor(np.ones((2, 2)), requires_grad=True)
        with self.assertRaises(RuntimeError):
            (x * 2).backward()
        with self.assertRaises(RuntimeError):
            (x * 2).backward(np.ones((3, 2)))
        (x * 2).backward(Tensor(np.ones((2, 2))))
        self.assertTrue(np.array_equal(x.grad.data, np.full((2, 2), 2.0)))

    def test_reflected_operators(self):
        x = Tensor(np.array([2.0, 4.0]), requires_grad=True)
        self.assertTrue(np.array_equal((1.0 + x).data, [3.0, 5.0]))
        self.assertTrue(np.array_equal((1.0 - x).data, [-1.0, -3.0]))
        self.assertTrue(np.array_equal((1.0 / x).data, [0.5, 0.25]))
        self.assertTrue(np.allclose((2.0**x).data, [4.0, 16.0]))
        m = np.ones((2, 2))
        self.assertEqual((m @ x).shape, (2,))
        self.assertEqual((x @ m).shape, (2,))

    def test_indexing_variants(self):
        x = Tensor(np.arange(6, dtype=np.float64).reshape(2, 3), requires_grad=True)
        self.assertEqual(x[..., None].shape, (2, 3, 1))
        self.assertEqual(x[None, :, 1].shape, (1, 2))
        self.assertEqual(x[:, ::-1].shape, (2, 3))
        y = x[[0, 0, 1]]
        y.sum().backward()
        self.assertTrue(np.array_equal(x.grad.data, [[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]]))

    def test_default_dtype_and_input_forms(self):
        original = init.default_dtype() if hasattr(init, "default_dtype") else None
        from src.nptorch import default_dtype, is_grad_enabled, set_default_dtype

        self.assertTrue(is_grad_enabled())
        old = default_dtype()
        try:
            set_default_dtype(np.float64)
            self.assertEqual(default_dtype(), np.dtype(np.float64))
            self.assertEqual(Tensor([1.0, 2.0]).dtype, np.float64)
        finally:
            set_default_dtype(old)
        self.assertEqual(Tensor([1, 2, 3]).dtype, np.int64)
        self.assertEqual(Tensor(np.float64(1.5)).dtype, np.float64)
        source = Tensor(np.ones(2, dtype=np.float32))
        self.assertEqual(Tensor(source, dtype=np.float64).dtype, np.float64)
        with self.assertRaises(RuntimeError):
            Tensor(np.ones(2, dtype=np.int64), requires_grad=True)
        x = Tensor(np.ones((2, 2), dtype=np.float64), requires_grad=True)
        x.grad = np.ones((2, 2))
        self.assertTrue(np.array_equal(x.grad.data, np.ones((2, 2))))
        x.reshape((2, 2)).sum().backward()
        self.assertEqual(Tensor(np.ones((2, 2))).view(4).shape, (4,))
        self.assertEqual(Tensor(np.ones((2, 2))).permute((1, 0)).shape, (2, 2))
        self.assertEqual(Tensor(np.ones((2, 2))).softmax().shape, (2, 2))
        self.assertEqual(Tensor(np.ones((2, 2))).log_softmax().shape, (2, 2))
        self.assertEqual(Tensor(np.array([-1.0, 2.0])).abs().data.tolist(), [1.0, 2.0])
        self.assertIn("Tensor", repr(Tensor(np.ones(2))))

    def test_double_backward_accumulates_on_leaves(self):
        x = Tensor(np.ones(2, dtype=np.float64), requires_grad=True)
        y = (x * 2.0).sum()
        y.backward()
        y.backward()
        self.assertTrue(np.array_equal(x.grad.data, [4.0, 4.0]))


class TestModuleApi(unittest.TestCase):
    def test_registration_and_traversal(self):
        class Empty(nn.Module):
            pass

        module = Empty()
        parameter = Parameter(np.ones(2, dtype=np.float64))
        module.register_parameter("p", parameter)
        self.assertIs(module.p, parameter)
        self.assertEqual([name for name, _ in module.named_parameters()], ["p"])
        with self.assertRaises(TypeError):
            module.register_parameter("bad", np.ones(2))
        with self.assertRaises(TypeError):
            module.register_parameter(1, parameter)
        with self.assertRaises(KeyError):
            module.register_parameter("a.b", parameter)

        child = nn.Linear(2, 2, dtype=np.float64)
        module.register_module("child", child)
        self.assertIs(module.child, child)
        self.assertEqual(
            [name for name, _ in module.named_modules()], ["", "child"]
        )
        self.assertEqual(len(list(module.modules())), 2)
        with self.assertRaises(TypeError):
            module.register_module("bad", object())
        with self.assertRaises(TypeError):
            module.register_module(1, child)
        module.register_parameter("p", None)
        self.assertIsNone(module.p)
        module.register_module("child", None)
        self.assertIsNone(module.child)

        module.p = Parameter(np.zeros(3, dtype=np.float64))
        self.assertIn("p", dict(module.named_parameters()))
        module.p = 5
        self.assertNotIn("p", dict(module.named_parameters()))

    def test_module_helpers(self):
        layer = nn.Linear(3, 2, dtype=np.float64)
        x = Tensor(np.ones((2, 3), dtype=np.float64), requires_grad=True)
        (layer(x) * 2.0).sum().backward()
        layer.zero_grad(set_to_none=False)
        self.assertTrue(np.all(layer.weight.grad.data == 0))
        layer.zero_grad()
        self.assertIsNone(layer.weight.grad)
        layer.requires_grad_(False)
        self.assertFalse(layer.weight.requires_grad)
        layer.requires_grad_(True)
        layer.double()
        self.assertEqual(layer.weight.dtype, np.float64)
        layer.float()
        self.assertEqual(layer.weight.dtype, np.float32)
        self.assertIs(layer.cpu(), layer)
        self.assertIs(layer.to("cpu"), layer)
        self.assertEqual(layer.to(np.float64).weight.dtype, np.float64)
        with self.assertRaises(TypeError):
            layer.to(np.int64)
        with self.assertRaises(ValueError):
            layer.to(device="cuda")
        self.assertIn("Linear", repr(layer))
        with self.assertRaises(NotImplementedError):
            nn.Module().forward()

    def test_state_dict_variants(self):
        layer = nn.Linear(3, 2, dtype=np.float64)
        torch_layer = torch.nn.Linear(3, 2)
        layer.load_state_dict(torch_layer.state_dict())
        state = layer.state_dict()
        self.assertTrue(np.array_equal(layer.weight.data, state["weight"]))
        destination = {}
        layer.state_dict(destination=destination, prefix="prefix.")
        self.assertIn("prefix.weight", destination)
        result = layer.load_state_dict({"weight": state["weight"]}, strict=False)
        self.assertEqual(result.missing_keys, ["bias"])
        self.assertEqual(result.unexpected_keys, [])
        with self.assertRaises(RuntimeError):
            layer.load_state_dict({"weight": np.zeros((3, 3))})
        with self.assertRaises(RuntimeError):
            layer.load_state_dict({"weight": state["weight"]})

    def test_module_list_repr_and_inplace_add(self):
        empty = nn.ModuleList()
        self.assertEqual(repr(empty), "ModuleList()")
        modules = nn.ModuleList([nn.Linear(2, 2)])
        self.assertIn("ModuleList", repr(modules))
        modules += [nn.Linear(2, 2)]
        self.assertEqual(len(modules), 2)
        self.assertIs(nn.ModuleList.__iadd__(modules, []), modules)

    def test_device_errors_and_repr(self):
        with self.assertRaises(ValueError):
            nn.Linear(2, 2, device="cuda")
        with self.assertRaises(ValueError):
            nn.LayerNorm(2, device="cuda")
        self.assertIn("Linear", repr(nn.Linear(2, 3)))
        self.assertIn("LayerNorm", repr(nn.LayerNorm(2)))
        self.assertIn("Dropout", repr(nn.Dropout(0.5)))
        self.assertIn("ReLU", repr(nn.ReLU()))
        self.assertIn("GELU", repr(nn.GELU()))
        self.assertIn("MSELoss", repr(nn.MSELoss()))
        with self.assertRaises(NotImplementedError):
            nn.ReLU(inplace=True)
        with self.assertRaises(NotImplementedError):
            nn.Dropout(0.5, inplace=True)
        with self.assertRaises(ValueError):
            nn.GELU(approximate="bad")
        with self.assertRaises(ValueError):
            nn.Dropout(1.5)
        with self.assertRaises(ValueError):
            nn.MSELoss(reduction="bad")

    def test_loss_reductions(self):
        x = Tensor(np.array([[1.0, 2.0]], dtype=np.float64))
        y = Tensor(np.zeros((1, 2), dtype=np.float64))
        self.assertEqual(nn.MSELoss(reduction="none")(x, y).shape, (1, 2))
        self.assertAlmostEqual(float(nn.MSELoss(reduction="sum")(x, y).item()), 5.0)
        self.assertAlmostEqual(float(nn.MSELoss()(x, y).item()), 2.5)


class TestInitApi(unittest.TestCase):
    def test_fan_and_gain(self):
        tensor = Tensor(np.empty((2, 3, 4, 5), dtype=np.float64))
        self.assertEqual(init.calculate_fan_in_and_fan_out(tensor), (60, 40))
        with self.assertRaises(ValueError):
            init.calculate_fan_in_and_fan_out(Tensor(np.empty(3)))
        self.assertEqual(init.calculate_gain("linear"), 1.0)
        self.assertEqual(init.calculate_gain("sigmoid"), 1.0)
        self.assertAlmostEqual(init.calculate_gain("tanh"), 5.0 / 3.0)
        self.assertAlmostEqual(init.calculate_gain("relu"), np.sqrt(2.0))
        self.assertAlmostEqual(
            init.calculate_gain("leaky_relu"), np.sqrt(2.0 / 1.0001)
        )
        self.assertAlmostEqual(init.calculate_gain("selu"), 0.75)
        with self.assertRaises(ValueError):
            init.calculate_gain("unknown")

    def test_initializers(self):
        tensor = Tensor(np.empty((100, 20), dtype=np.float64))
        init.zeros_(tensor)
        self.assertTrue(np.all(tensor.data == 0))
        init.ones_(tensor)
        self.assertTrue(np.all(tensor.data == 1))
        init.constant_(tensor, 3.0)
        self.assertTrue(np.all(tensor.data == 3.0))
        init.uniform_(tensor, -0.1, 0.1)
        self.assertTrue(np.all(np.abs(tensor.data) <= 0.1))
        init.normal_(tensor, 0.0, 1.0)
        self.assertLess(abs(float(tensor.data.mean())), 0.1)
        init.xavier_uniform_(tensor)
        bound = np.sqrt(6.0 / (20 + 100))
        self.assertTrue(np.all(np.abs(tensor.data) <= bound + 1e-12))
        init.xavier_normal_(tensor)
        init.kaiming_uniform_(tensor, a=0.0)
        init.kaiming_normal_(tensor, mode="fan_out")
        with self.assertRaises(ValueError):
            init.kaiming_uniform_(tensor, mode="bad")
        with self.assertRaises(ValueError):
            init.kaiming_normal_(tensor, mode="bad")
        int_tensor = Tensor(np.empty((2, 2), dtype=np.int64))
        with self.assertRaises(ValueError):
            init.uniform_(int_tensor)
        with self.assertRaises(ValueError):
            init.normal_(int_tensor)


class TestOptimizerApi(unittest.TestCase):
    def _loss(self, layer):
        x = np.ones((3, 3), dtype=np.float64)
        target = np.zeros((3, 2), dtype=np.float64)
        return ((layer(Tensor(x)) - Tensor(target)) ** 2).mean()

    def test_sgd_edges(self):
        base = optim.Optimizer(nn.Linear(2, 2, dtype=np.float64).parameters(), {})
        with self.assertRaises(NotImplementedError):
            base.step()
        with self.assertRaises(ValueError):
            optim.SGD([], lr=-1.0)
        with self.assertRaises(ValueError):
            optim.SGD([], lr=0.1, momentum=-1.0)
        with self.assertRaises(ValueError):
            optim.SGD([], lr=0.1, weight_decay=-1.0)
        with self.assertRaises(ValueError):
            optim.SGD([], lr=0.1, momentum=0.5, nesterov=True, dampening=0.1)
        with self.assertRaises(NotImplementedError):
            optim.SGD([], lr=0.1, maximize=True)
        with self.assertRaises(TypeError):
            optim.SGD(Tensor(np.ones(2)), lr=0.1)

        layer = nn.Linear(3, 2, dtype=np.float64)
        optimizer = optim.SGD(layer, lr=0.1, momentum=0.9, weight_decay=1e-3)
        loss = self._loss(layer)
        loss.backward()
        optimizer.zero_grad(set_to_none=False)
        self.assertTrue(np.all(layer.weight.grad.data == 0))
        loss = self._loss(layer)
        loss.backward()
        result = optimizer.step(closure=lambda: 1.0)
        self.assertEqual(result, 1.0)
        self.assertTrue(np.all(np.isfinite(layer.weight.data)))

    def test_adam_edges(self):
        with self.assertRaises(ValueError):
            optim.Adam([], lr=-1.0)
        with self.assertRaises(ValueError):
            optim.Adam([], betas=(1.0, 0.999))
        with self.assertRaises(ValueError):
            optim.Adam([], eps=-1.0)
        with self.assertRaises(NotImplementedError):
            optim.Adam([], amsgrad=True)
        with self.assertRaises(NotImplementedError):
            optim.Adam([], maximize=True)

        layer = nn.Linear(3, 2, dtype=np.float64)
        optimizer = optim.Adam(layer, lr=1e-2, weight_decay=1e-3)
        loss = self._loss(layer)
        loss.backward()
        result = optimizer.step(closure=lambda: 1.0)
        self.assertEqual(result, 1.0)
        self.assertTrue(np.all(np.isfinite(layer.weight.data)))
        self.assertIn(layer.weight, optimizer.state)


class TestFunctionalEdges(unittest.TestCase):
    def test_clone_cast_neg(self):
        x = Tensor(np.array([1.0, -2.0]), requires_grad=True)
        clone = F.clone(x)
        self.assertFalse(np.shares_memory(clone.data, x.data))
        cast = F.cast(x, np.float64)
        self.assertEqual(cast.dtype, np.float64)
        neg = F.neg(x)
        (clone.sum() + cast.sum() + neg.sum()).backward()
        self.assertTrue(np.allclose(x.grad.data, [1.0, 1.0]))

    def test_pow_with_tensor_exponent(self):
        base = Tensor(np.array([1.5, 2.0]), requires_grad=True)
        exponent = Tensor(np.array([2.0, 3.0]), requires_grad=True)
        out = F.pow(base, exponent)
        out.sum().backward()
        self.assertTrue(np.all(np.isfinite(base.grad.data)))
        self.assertTrue(np.all(np.isfinite(exponent.grad.data)))

    def test_squeeze_forms(self):
        x = Tensor(np.ones((1, 2, 1), dtype=np.float64), requires_grad=True)
        out = x.squeeze()
        self.assertEqual(out.shape, (2,))
        out.sum().backward()
        self.assertTrue(np.array_equal(x.grad.data, np.ones((1, 2, 1))))
        self.assertIs(x.squeeze(dim=1), x)

    def test_getitem_advanced(self):
        x = Tensor(np.arange(6, dtype=np.float64).reshape(2, 3), requires_grad=True)
        x[[1, 1]].sum().backward()
        self.assertTrue(np.array_equal(x.grad.data, [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]))

    def test_error_paths(self):
        x = Tensor(np.ones((2, 3), dtype=np.float64))
        with self.assertRaises(RuntimeError):
            F.matmul(Tensor(1.0), Tensor(1.0))
        with self.assertRaises(RuntimeError):
            F.linear(x, Tensor(np.ones(3)))
        with self.assertRaises(RuntimeError):
            F.linear(x, Tensor(np.ones((2, 2))))
        with self.assertRaises(RuntimeError):
            F.permute(x, (0, 0))
        with self.assertRaises(IndexError):
            F.unsqueeze(x, 3)
        with self.assertRaises(IndexError):
            F.sum(x, dim=2)
        with self.assertRaises(IndexError):
            F.mean(x, dim=3)
        with self.assertRaises(RuntimeError):
            F.cat([])
        with self.assertRaises(ValueError):
            F.gelu(x, approximate="bad")
        with self.assertRaises(RuntimeError):
            F.layer_norm(x, (4,))
        with self.assertRaises(TypeError):
            F.masked_fill(x, np.ones((2, 3), dtype=np.int64), 0.0)
        with self.assertRaises(ValueError):
            F.dropout(x, p=1.5)
        with self.assertRaises(ValueError):
            F.dropout(x, p=0.5, mask=np.ones((3, 3), dtype=bool))

    def test_dropout_p_one_backward(self):
        x = Tensor(np.ones((2, 2), dtype=np.float64), requires_grad=True)
        out = F.dropout(x, p=1.0, training=True)
        out.sum().backward()
        self.assertTrue(np.array_equal(out.data, np.zeros((2, 2))))
        self.assertTrue(np.array_equal(x.grad.data, np.zeros((2, 2))))

    def test_no_grad_reduction_and_scalar_branches(self):
        x = Tensor(np.ones((2, 3), dtype=np.float64))
        self.assertEqual(float(F.sum(x).item()), 6.0)
        self.assertEqual(float(F.mean(x).item()), 1.0)
        self.assertEqual(F.neg(x).data.tolist()[0], [-1.0, -1.0, -1.0])
        self.assertTrue(np.array_equal(F.dropout(x, p=1.0).data, np.zeros((2, 3))))
        self.assertEqual(F.layer_norm(x, 3).shape, (2, 3))
        self.assertEqual((True * x).dtype, np.float64)
        self.assertEqual((np.float64(2.0) * x).dtype, np.float64)
        self.assertEqual((np.int64(2) * x).dtype, np.float64)
        self.assertEqual((Tensor(np.zeros(3)) + x).dtype, np.float64)
        self.assertEqual((Tensor(np.zeros(3, dtype=np.float64)) + x).dtype, np.float64)
        self.assertEqual((Tensor(np.zeros(3, dtype=np.int64)) + x).dtype, np.float64)


class TestAttentionErrors(unittest.TestCase):
    def test_constructor_errors(self):
        with self.assertRaises(ValueError):
            nn.MultiheadAttention(16, 4, device="cuda")
        with self.assertRaises(ValueError):
            nn.MultiheadAttention(0, 4)
        with self.assertRaises(ValueError):
            nn.MultiheadAttention(16, 0)
        with self.assertRaises(AssertionError):
            nn.MultiheadAttention(15, 4)
        with self.assertRaises(NotImplementedError):
            nn.MultiheadAttention(16, 4, add_bias_kv=True)
        with self.assertRaises(NotImplementedError):
            nn.MultiheadAttention(16, 4, add_zero_attn=True)
        with self.assertRaises(NotImplementedError):
            nn.MultiheadAttention(16, 4, kdim=8)
        with self.assertRaises(NotImplementedError):
            nn.MultiheadAttention(16, 4, vdim=8)

    def test_forward_shape_and_mask_errors(self):
        mha = nn.MultiheadAttention(16, 4)
        x = Tensor(np.ones((3, 2, 16)))
        with self.assertRaises(RuntimeError):
            mha(Tensor(np.ones(16)), x, x)
        with self.assertRaises(RuntimeError):
            mha(x, Tensor(np.ones((3, 16))), Tensor(np.ones((3, 16))))
        with self.assertRaises(RuntimeError):
            mha(x, Tensor(np.ones((3, 2, 8))), Tensor(np.ones((3, 2, 8))))
        with self.assertRaises(RuntimeError):
            mha(x, x, Tensor(np.ones((3, 2, 8))))
        with self.assertRaises(RuntimeError):
            mha(x, x, x, attn_mask=np.ones((4, 4), dtype=bool))
        with self.assertRaises(RuntimeError):
            mha(x, x, x, attn_mask=np.ones((1, 2, 3, 4), dtype=bool))
        with self.assertRaises(RuntimeError):
            mha(x, x, x, key_padding_mask=np.ones((2, 4), dtype=bool))
        with self.assertRaises(TypeError):
            mha(x, x, x, attn_mask=np.ones((3, 3), dtype=np.int64))
        with self.assertRaises(RuntimeError):
            mha(x, x, x, is_causal=True)
        mha.eval()
        out, weights = mha(x, x, x, need_weights=True)
        self.assertEqual(out.shape, (3, 2, 16))
        self.assertEqual(weights.shape, (2, 3, 3))


class TestTransformerApiEdges(unittest.TestCase):
    def test_activation_validation_and_flags(self):
        with self.assertRaises(RuntimeError):
            nn.TransformerEncoderLayer(16, 4, activation="silu")
        layer = nn.TransformerEncoderLayer(16, 4, activation=nn.ReLU())
        self.assertEqual(layer.activation_relu_or_gelu, 1)
        layer = nn.TransformerEncoderLayer(16, 4, activation=nn.GELU())
        self.assertEqual(layer.activation_relu_or_gelu, 2)
        layer = nn.TransformerEncoderLayer(16, 4, activation=lambda x: F.relu(x))
        self.assertEqual(layer.activation_relu_or_gelu, 0)
        self.assertIn("TransformerEncoderLayer", repr(layer))

    def test_encoder_helpers_and_errors(self):
        encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(16, 4, dtype=np.float64), 2
        )
        self.assertIn("TransformerEncoder", repr(encoder))
        self.assertIn("num_layers=2", encoder.extra_repr())
        x = Tensor(np.ones((3, 2, 16), dtype=np.float64), requires_grad=True)
        with self.assertRaises(AssertionError):
            encoder(x, mask=np.ones((3, 3), dtype=np.int64))
        with self.assertRaises(RuntimeError):
            encoder(x, is_causal=True)
        empty = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(16, 4, dtype=np.float64), 0
        )
        with self.assertRaises(IndexError):
            empty(x)
        self.assertFalse(empty.use_nested_tensor)
        self.assertTrue(empty.mask_check)


if __name__ == "__main__":
    unittest.main()
