"""Demo: NumPy TransformerEncoderLayer parity with torch and a tiny training run.

Run from part_2_transformer:

    ../.venv/bin/python main.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.nptorch import Tensor, nn, optim


def copy_weights(torch_module, np_module) -> None:
    np_module.load_state_dict(
        {key: value.detach().numpy() for key, value in torch_module.state_dict().items()}
    )


def maxdiff(actual, expected) -> float:
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def parity_demo(
    label: str,
    torch_layer: torch.nn.TransformerEncoderLayer,
    np_layer,
    x: np.ndarray,
    dtype: np.dtype,
) -> None:
    torch_layer.eval()
    np_layer.eval()
    copy_weights(torch_layer, np_layer)
    xt = torch.tensor(x.copy(), requires_grad=True)
    xn = Tensor(x.copy(), requires_grad=True)
    out_t = torch_layer(xt)
    out_n = np_layer(xn)
    coeff = np.random.default_rng(0).standard_normal(out_t.shape).astype(dtype)
    loss_t = (out_t * torch.tensor(coeff)).sum()
    loss_n = (out_n * Tensor(coeff)).sum()
    loss_t.backward()
    loss_n.backward()
    param_diff = 0.0
    for name, param in np_layer.named_parameters():
        torch_param = dict(torch_layer.named_parameters())[name]
        param_diff = max(param_diff, maxdiff(param.grad.data, torch_param.grad.numpy()))
    loss_abs = abs(loss_n.item() - loss_t.item())
    loss_rel = loss_abs / max(1.0, abs(loss_t.item()))
    print(
        f"{label:10s} forward={maxdiff(out_n.data, out_t.detach().numpy()):.2e}  "
        f"loss={loss_abs:.2e} (rel {loss_rel:.1e})  "
        f"input_grad={maxdiff(xn.grad.data, xt.grad.numpy()):.2e}  "
        f"param_grad={param_diff:.2e}"
    )


def encoder_demo() -> None:
    d_model, nhead, ffn, num_layers = 16, 4, 32, 3
    torch_layer = torch.nn.TransformerEncoderLayer(
        d_model, nhead, dim_feedforward=ffn, dtype=torch.float64
    )
    np_layer = nn.TransformerEncoderLayer(
        d_model, nhead, dim_feedforward=ffn, dtype=np.float64
    )
    torch_encoder = torch.nn.TransformerEncoder(
        torch_layer,
        num_layers,
        norm=torch.nn.LayerNorm(d_model, dtype=torch.float64),
        enable_nested_tensor=False,
    )
    np_encoder = nn.TransformerEncoder(
        np_layer,
        num_layers,
        norm=nn.LayerNorm(d_model, dtype=np.float64),
        enable_nested_tensor=False,
    )
    copy_weights(torch_encoder, np_encoder)
    torch_encoder.eval()
    np_encoder.eval()
    rng = np.random.default_rng(5)
    x = rng.standard_normal((5, 3, d_model))
    mask = np.triu(np.ones((5, 5), dtype=bool), k=1)
    key_padding_mask = np.zeros((3, 5), dtype=bool)
    key_padding_mask[0, -2:] = True
    xt = torch.tensor(x, requires_grad=True)
    xn = Tensor(x.copy(), requires_grad=True)
    out_t = torch_encoder(
        xt,
        mask=torch.tensor(mask),
        src_key_padding_mask=torch.tensor(key_padding_mask),
    )
    out_n = np_encoder(
        xn,
        mask=mask,
        src_key_padding_mask=key_padding_mask,
    )
    coeff = rng.standard_normal(out_t.shape)
    (out_t * torch.tensor(coeff)).sum().backward()
    (out_n * Tensor(coeff)).sum().backward()
    param_diff = max(
        maxdiff(param.grad.data, dict(torch_encoder.named_parameters())[name].grad.numpy())
        for name, param in np_encoder.named_parameters()
    )
    print(
        f"encoder    layers={num_layers}, norm: "
        f"forward={maxdiff(out_n.data, out_t.detach().numpy()):.2e}  "
        f"input_grad={maxdiff(xn.grad.data, xt.grad.numpy()):.2e}  "
        f"param_grad={param_diff:.2e}"
    )


def benchmark() -> None:
    d_model, nhead, seq, batch = 512, 8, 16, 4
    torch_layer = torch.nn.TransformerEncoderLayer(d_model, nhead).eval()
    np_layer = nn.TransformerEncoderLayer(d_model, nhead).eval()
    copy_weights(torch_layer, np_layer)
    x = np.random.default_rng(1).standard_normal((seq, batch, d_model)).astype(np.float32)
    xt = torch.tensor(x)
    xn = Tensor(x.copy())

    def timeit(fn, repeats: int = 5) -> float:
        fn()
        fn()
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        return (time.perf_counter() - start) / repeats

    torch_fwd = timeit(lambda: torch_layer(xt))
    np_fwd = timeit(lambda: np_layer(xn))
    torch_bwd = timeit(lambda: torch_layer(xt).sum().backward())
    np_bwd = timeit(lambda: np_layer(xn).sum().backward())
    print(
        f"default layer: torch fwd {torch_fwd*1e3:.1f} ms, numpy fwd {np_fwd*1e3:.1f} ms; "
        f"torch fwd+bwd {torch_bwd*1e3:.1f} ms, numpy fwd+bwd {np_bwd*1e3:.1f} ms"
    )


def training_demo() -> None:
    d_model, nhead, ffn, seq, batch = 32, 4, 64, 8, 4
    torch.manual_seed(0)
    torch_layer = torch.nn.TransformerEncoderLayer(
        d_model, nhead, dim_feedforward=ffn, dtype=torch.float64
    ).eval()
    np_layer = nn.TransformerEncoderLayer(
        d_model, nhead, dim_feedforward=ffn, dtype=np.float64
    ).eval()
    copy_weights(torch_layer, np_layer)
    rng = np.random.default_rng(2)
    x = rng.standard_normal((seq, batch, d_model))
    target = rng.standard_normal((seq, batch, d_model))
    torch_opt = torch.optim.AdamW(torch_layer.parameters(), lr=1e-3)
    np_opt = optim.AdamW(np_layer.parameters(), lr=1e-3)
    torch_loss_fn = torch.nn.MSELoss()
    np_loss_fn = nn.MSELoss()
    print("training with AdamW (eval mode, same initial weights):")
    for step in range(1, 11):
        torch_opt.zero_grad()
        np_opt.zero_grad()
        loss_t = torch_loss_fn(torch_layer(torch.tensor(x)), torch.tensor(target))
        loss_n = np_loss_fn(np_layer(Tensor(x.copy())), Tensor(target))
        loss_t.backward()
        loss_n.backward()
        torch_opt.step()
        np_opt.step()
        if step % 5 == 0:
            print(f"  step {step:2d}: torch={loss_t.item():.6f} numpy={loss_n.item():.6f}")
    param_diff = max(
        maxdiff(param.data, dict(torch_layer.named_parameters())[name].detach().numpy())
        for name, param in np_layer.named_parameters()
    )
    print(f"  final parameter maxdiff: {param_diff:.2e}")


def main() -> None:
    print(f"torch {torch.__version__}, numpy {np.__version__}")
    print("TransformerEncoderLayer parity (all parameter and input grads):")
    parity_demo(
        "float64",
        torch.nn.TransformerEncoderLayer(16, 4, dim_feedforward=32, dtype=torch.float64),
        nn.TransformerEncoderLayer(16, 4, dim_feedforward=32, dtype=np.float64),
        np.random.default_rng(3).standard_normal((5, 3, 16)),
        np.float64,
    )
    parity_demo(
        "default32",
        torch.nn.TransformerEncoderLayer(512, 8),
        nn.TransformerEncoderLayer(512, 8),
        np.random.default_rng(4).standard_normal((4, 2, 512)).astype(np.float32),
        np.float32,
    )
    encoder_demo()
    benchmark()
    training_demo()


if __name__ == "__main__":
    main()
