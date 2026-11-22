"""End-to-end training tests: one NumPy layer and one torch twin."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from tests.utils import assert_close, make_pair


def make_data(seed: int = 71):
    rng = np.random.default_rng(seed)
    seq_len, batch_size, d_model = 6, 8, 8
    x = rng.normal(size=(seq_len, batch_size, d_model))
    projection = rng.normal(size=(d_model, d_model)) / np.sqrt(d_model)
    target = np.tanh(x @ projection + 0.2)
    return x, target


class TwinTrainingTest(unittest.TestCase):
    def test_sgd_trajectories_match_torch(self) -> None:
        numpy_layer, torch_layer = make_pair(
            8, 2, 16, dropout=0.0, norm_first=False
        )
        numpy_layer.train(False)
        torch_layer.eval()
        x, target = make_data()
        target_t = torch.tensor(target, dtype=torch.float64)
        learning_rate = 0.1
        initial_loss = None
        previous_loss = None

        for step in range(40):
            out = numpy_layer(x)
            difference = out - target
            loss = float(np.mean(difference * difference))
            dout = 2.0 * difference / difference.size
            numpy_layer.zero_grad()
            d_src = numpy_layer.backward(dout)

            torch_out = torch_layer(torch.tensor(x, dtype=torch.float64))
            torch_loss = torch.nn.functional.mse_loss(torch_out, target_t)
            torch_loss.backward()

            self.assertAlmostEqual(
                loss, float(torch_loss.detach()), places=12
            )
            if initial_loss is None:
                initial_loss = loss
            previous_loss = loss

            self.assertEqual(d_src.shape, x.shape)
            torch_params = dict(torch_layer.named_parameters())
            for name, parameter in torch_params.items():
                assert_close(
                    self,
                    numpy_layer.grads[name],
                    parameter.grad.numpy(),
                    atol=2e-12,
                    rtol=1e-10,
                    msg=f"step {step} d[{name}]",
                )

            with torch.no_grad():
                for name, parameter in torch_params.items():
                    parameter -= learning_rate * parameter.grad
            for name, parameter in numpy_layer.params.items():
                parameter -= learning_rate * numpy_layer.grads[name]
            for name, parameter in torch_params.items():
                assert_close(
                    self,
                    numpy_layer.params[name],
                    parameter.detach().numpy(),
                    atol=2e-12,
                    rtol=1e-10,
                    msg=f"step {step} param[{name}]",
                )
            numpy_layer.zero_grad()
            torch_layer.zero_grad(set_to_none=True)

        self.assertIsNotNone(initial_loss)
        self.assertLess(previous_loss, 0.6 * initial_loss)

    def test_pre_norm_twin_training(self) -> None:
        numpy_layer, torch_layer = make_pair(
            8, 2, 16, dropout=0.0, norm_first=True, batch_first=True
        )
        x_seq, target_seq = make_data(seed=72)
        x = x_seq.transpose(1, 0, 2)
        target = target_seq.transpose(1, 0, 2)
        out = numpy_layer(x)
        self.assertEqual(out.shape, x.shape)

        difference = out - target
        loss = float(np.mean(difference * difference))
        dout = 2.0 * difference / difference.size
        numpy_layer.zero_grad()
        numpy_layer.backward(dout)

        torch_x = torch.tensor(x, dtype=torch.float64)
        torch_out = torch_layer(torch_x)
        torch_loss = torch.nn.functional.mse_loss(
            torch_out, torch.tensor(target, dtype=torch.float64)
        )
        torch_loss.backward()
        self.assertAlmostEqual(
            loss, float(torch_loss.detach()), places=12
        )
        for name, parameter in torch_layer.named_parameters():
            assert_close(
                self,
                numpy_layer.grads[name],
                parameter.grad.numpy(),
                msg=f"pre-norm d[{name}]",
            )


if __name__ == "__main__":
    unittest.main()
