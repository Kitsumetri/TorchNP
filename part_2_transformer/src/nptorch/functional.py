"""Functional tensor ops with hand-written backward formulas.

Every function accepts and returns ``Tensor``. Gradients are verified in the
test suite against ``torch.autograd`` and central finite differences. As in
torch, operations preserve the input dtype (Python scalars do not silently
promote float32 to float64).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .tensor import Tensor, _add_grad, _make_result, default_dtype, get_rng

try:  # scipy is present via scikit-learn, but it is not required
    from scipy.special import erf as _erf
except Exception:  # pragma: no cover - fallback
    _erf = np.vectorize(math.erf, otypes=[np.float64])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_tensor(x: Any) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


def _is_scalar_operand(value: Any) -> bool:
    return isinstance(value, (bool, int, float, np.integer, np.floating))


def _coerce_scalar(value: Any, other: Tensor) -> Tensor:
    """Create a 0-dim tensor from a Python/NumPy scalar using torch rules."""
    if isinstance(value, (bool, np.bool_)):
        dtype = other.dtype
    elif isinstance(value, (int, np.integer)):
        dtype = np.dtype(np.int64) if other.dtype.kind == "b" else other.dtype
    elif isinstance(value, (float, np.floating)):
        dtype = other.dtype if other.dtype.kind == "f" else default_dtype()
    else:
        return _as_tensor(value)
    return Tensor(value, dtype=dtype)


def _binary_operands(a: Any, b: Any) -> tuple[Tensor, Tensor]:
    """Convert operands, making Python scalars exact in the other dtype.

    A Python float must be cast to float64 *before* it becomes a tensor;
    otherwise the default float32 dtype would round it and later promotion
    could not recover the lost bits.
    """
    a_is_tensor, b_is_tensor = isinstance(a, Tensor), isinstance(b, Tensor)
    if a_is_tensor and b_is_tensor:
        return a, b
    if a_is_tensor:
        if _is_scalar_operand(b):
            return a, _coerce_scalar(b, a)
        return a, _as_tensor(b)
    if b_is_tensor:
        if _is_scalar_operand(a):
            return _coerce_scalar(a, b), b
        return _as_tensor(a), b
    return _as_tensor(a), _as_tensor(b)


def _dtype_category(dtype: np.dtype) -> int:
    if dtype.kind == "b":
        return 0
    if dtype.kind in "iu":
        return 1
    if dtype.kind == "f":
        return 2
    return 3


def _promote_dtype(a: Tensor, b: Tensor) -> np.dtype:
    """Torch-like type promotion (including wrapped 0-dim numbers)."""
    ad, bd = a.dtype, b.dtype
    if ad == bd:
        return ad
    ca, cb = _dtype_category(ad), _dtype_category(bd)
    if ca != cb:
        # bool -> int -> float; the higher category wins.
        return ad if ca > cb else bd
    if ca == 2:  # both floating point
        if a.ndim == 0 and b.ndim > 0:
            return bd
        if b.ndim == 0 and a.ndim > 0:
            return ad
        return np.promote_types(ad, bd)
    if ca == 1:  # both integer
        if a.ndim == 0 and b.ndim > 0:
            return bd
        if b.ndim == 0 and a.ndim > 0:
            return ad
        return np.promote_types(ad, bd)
    return np.promote_types(ad, bd)


def _binary_data(a: Tensor, b: Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Operand data cast to the torch-promoted dtype."""
    dtype = _promote_dtype(a, b)
    ad, bd = a.data, b.data
    if ad.dtype != dtype:
        ad = ad.astype(dtype, copy=False)
    if bd.dtype != dtype:
        bd = bd.astype(dtype, copy=False)
    return ad, bd


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum a broadcast gradient back to its original shape."""
    g = np.asarray(grad)
    if g.shape == shape:
        return g
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and g.shape[axis] != 1:
            g = g.sum(axis=axis, keepdims=True)
    if g.shape != shape:
        g = g.reshape(shape)
    return g


def _norm_dim(dim: int, ndim: int) -> int:
    d = dim + ndim if dim < 0 else dim
    if not 0 <= d < ndim:
        raise IndexError(f"dim={dim} out of range for ndim={ndim}")
    return d


def _norm_dims(dim: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    if isinstance(dim, (int, np.integer)):
        dims = (int(dim),)
    else:
        dims = tuple(int(d) for d in dim)
    return tuple(sorted({_norm_dim(d, ndim) for d in dims}))


def _reduce_count(shape: tuple[int, ...], dims: tuple[int, ...]) -> int:
    count = 1
    for d in dims:
        count *= shape[d]
    return count


def _is_basic_index(index: Any) -> bool:
    """True for slices/ints/Ellipsis/newaxis (no duplicate indices)."""
    items = index if isinstance(index, tuple) else (index,)
    for item in items:
        if isinstance(item, (slice, int, np.integer)):
            continue
        if item is Ellipsis or item is None:
            continue
        return False
    return True


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def add(a: Any, b: Any, alpha: float = 1.0) -> Tensor:
    a, b = _binary_operands(a, b)
    ad, bd = _binary_data(a, b)

    def backward() -> None:
        g = out._grad
        assert g is not None
        if a.requires_grad:
            _add_grad(a, _unbroadcast(g, a.shape))
        if b.requires_grad:
            _add_grad(b, _unbroadcast(alpha * g, b.shape))

    out = _make_result(
        ad + bd if alpha == 1.0 else ad + alpha * bd, (a, b), backward, "add"
    )
    return out


def sub(a: Any, b: Any, alpha: float = 1.0) -> Tensor:
    a, b = _binary_operands(a, b)
    ad, bd = _binary_data(a, b)

    def backward() -> None:
        g = out._grad
        assert g is not None
        if a.requires_grad:
            _add_grad(a, _unbroadcast(g, a.shape))
        if b.requires_grad:
            _add_grad(b, _unbroadcast(-alpha * g, b.shape))

    out = _make_result(
        ad - bd if alpha == 1.0 else ad - alpha * bd, (a, b), backward, "sub"
    )
    return out


def mul(a: Any, b: Any) -> Tensor:
    a, b = _binary_operands(a, b)
    ad, bd = _binary_data(a, b)

    def backward() -> None:
        g = out._grad
        assert g is not None
        if a.requires_grad:
            _add_grad(a, _unbroadcast(g * bd, a.shape))
        if b.requires_grad:
            _add_grad(b, _unbroadcast(g * ad, b.shape))

    out = _make_result(ad * bd, (a, b), backward, "mul")
    return out


def div(a: Any, b: Any) -> Tensor:
    a, b = _binary_operands(a, b)
    ad, bd = _binary_data(a, b)
    if ad.dtype.kind != "f" and bd.dtype.kind != "f":
        # torch true division of integer tensors produces the default float dtype
        float_dtype = default_dtype()
        ad = ad.astype(float_dtype, copy=False)
        bd = bd.astype(float_dtype, copy=False)

    def backward() -> None:
        g = out._grad
        assert g is not None
        if a.requires_grad:
            _add_grad(a, _unbroadcast(g / bd, a.shape))
        if b.requires_grad:
            _add_grad(b, _unbroadcast(-g * ad / (bd * bd), b.shape))

    out = _make_result(ad / bd, (a, b), backward, "div")
    return out


def neg(a: Any) -> Tensor:
    a = _as_tensor(a)

    def backward() -> None:
        g = out._grad
        assert g is not None
        if a.requires_grad:
            _add_grad(a, -g)

    out = _make_result(-a.data, (a,), backward, "neg")
    return out


def pow(a: Any, exponent: Any) -> Tensor:
    a, exponent = _binary_operands(a, exponent)
    ad, exp = _binary_data(a, exponent)
    if exp.ndim == 0:
        # As in torch, a scalar exponent does not change the base dtype.
        exp = exp.astype(ad.dtype, copy=False)

    def backward() -> None:
        g = out._grad
        assert g is not None
        if a.requires_grad:
            _add_grad(a, _unbroadcast(g * exp * (ad ** (exp - 1)), a.shape))
        if exponent.requires_grad:
            with np.errstate(divide="ignore", invalid="ignore"):
                grad_exp = g * out.data * np.log(ad)
            _add_grad(exponent, _unbroadcast(grad_exp, exponent.shape))

    out = _make_result(ad**exp, (a, exponent), backward, "pow")
    return out


def _unary_float_data(a: Tensor) -> np.ndarray:
    # torch promotes integer inputs of exp/log/sqrt to the default float dtype.
    if a.dtype.kind == "f":
        return a.data
    return a.data.astype(default_dtype())


def exp(a: Any) -> Tensor:
    a = _as_tensor(a)
    x = _unary_float_data(a)

    def backward() -> None:
        g = out._grad
        assert g is not None
        if a.requires_grad:
            _add_grad(a, g * out.data)

    out = _make_result(np.exp(x), (a,), backward, "exp")
    return out


def log(a: Any) -> Tensor:
    a = _as_tensor(a)
    x = _unary_float_data(a)

    def backward() -> None:
        g = out._grad
        assert g is not None
        if a.requires_grad:
            _add_grad(a, g / x)

    out = _make_result(np.log(x), (a,), backward, "log")
    return out


def sqrt(a: Any) -> Tensor:
    a = _as_tensor(a)
    x = _unary_float_data(a)

    def backward() -> None:
        g = out._grad
        assert g is not None
        if a.requires_grad:
            _add_grad(a, g * 0.5 / out.data)

    out = _make_result(np.sqrt(x), (a,), backward, "sqrt")
    return out


def abs(a: Any) -> Tensor:  # noqa: A001 - torch-compatible name
    a = _as_tensor(a)

    def backward() -> None:
        g = out._grad
        assert g is not None
        if a.requires_grad:
            _add_grad(a, g * np.sign(a.data))

    out = _make_result(np.abs(a.data), (a,), backward, "abs")
    return out


def matmul(a: Any, b: Any) -> Tensor:
    a, b = _as_tensor(a), _as_tensor(b)
    A, B = a.data, b.data
    if A.ndim == 0 or B.ndim == 0:
        raise RuntimeError("matmul does not support 0-dim tensors, same as torch")

    def backward() -> None:
        g = out._grad
        assert g is not None
        # Promote 1-D operands to matrices so the same formula works everywhere.
        A_ = A[None, :] if A.ndim == 1 else A
        B_ = B[:, None] if B.ndim == 1 else B
        if A.ndim == 1 and B.ndim == 1:
            g_ = np.asarray(g).reshape(1, 1)
        elif A.ndim == 1:
            g_ = np.expand_dims(g, axis=-2)
        elif B.ndim == 1:
            g_ = np.expand_dims(g, axis=-1)
        else:
            g_ = g
        dA_ = np.matmul(g_, np.swapaxes(B_, -1, -2))
        dB_ = np.matmul(np.swapaxes(A_, -1, -2), g_)
        if A.ndim == 1:
            dA_ = np.squeeze(dA_, axis=-2)
        if B.ndim == 1:
            dB_ = np.squeeze(dB_, axis=-1)
        if a.requires_grad:
            _add_grad(a, _unbroadcast(dA_, A.shape))
        if b.requires_grad:
            _add_grad(b, _unbroadcast(dB_, B.shape))

    out = _make_result(np.matmul(A, B), (a, b), backward, "matmul")
    return out


def linear(input: Any, weight: Any, bias: Any | None = None) -> Tensor:
    """``input @ weight.T + bias``, like ``torch.nn.functional.linear``.

    Leading dimensions are flattened into one 2-D GEMM. This matters for
    numpy: a batched ``(L, N, E) @ (E, O)`` matmul becomes many small BLAS
    calls, while ``(L*N, E) @ (E, O)`` is a single large one.
    """
    input, weight = _as_tensor(input), _as_tensor(weight)
    if weight.ndim != 2:
        raise RuntimeError(f"weight must be 2-D, got shape={weight.shape}")
    out_features, in_features = weight.shape
    if input.ndim == 0 or input.shape[-1] != in_features:
        got = "scalar" if input.ndim == 0 else input.shape[-1]
        raise RuntimeError(
            f"linear: input last dim {got} != in_features {in_features}"
        )
    if input.ndim == 2:
        out = matmul(input, transpose(weight, -1, -2))
    else:
        flat = reshape(input, (-1, in_features))
        out = matmul(flat, transpose(weight, -1, -2))
        out = reshape(out, (*input.shape[:-1], out_features))
    if bias is not None:
        out = out + _as_tensor(bias)
    return out


# ---------------------------------------------------------------------------
# Shape ops
# ---------------------------------------------------------------------------


def reshape(a: Any, shape: Sequence[int]) -> Tensor:
    a = _as_tensor(a)
    new_shape = tuple(int(s) for s in shape)

    def backward() -> None:
        g = out._grad
        assert g is not None
        if a.requires_grad:
            _add_grad(a, g.reshape(a.shape))

    out = _make_result(a.data.reshape(new_shape), (a,), backward, "reshape")
    return out


def transpose(a: Any, dim0: int, dim1: int) -> Tensor:
    a = _as_tensor(a)
    d0, d1 = _norm_dim(dim0, a.ndim), _norm_dim(dim1, a.ndim)

    def backward() -> None:
        g = out._grad
        assert g is not None
        if a.requires_grad:
            _add_grad(a, np.swapaxes(g, d0, d1))

    out = _make_result(np.swapaxes(a.data, d0, d1), (a,), backward, "transpose")
    return out


def permute(a: Any, dims: Sequence[int]) -> Tensor:
    a = _as_tensor(a)
    dims = tuple(_norm_dim(d, a.ndim) for d in dims)
    if sorted(dims) != list(range(a.ndim)):
        raise RuntimeError(f"permute: invalid permutation {dims}")
    inverse = tuple(int(i) for i in np.argsort(dims))

    def backward() -> None:
        g = out._grad
        assert g is not None
        if a.requires_grad:
            _add_grad(a, np.transpose(g, inverse))

    out = _make_result(np.transpose(a.data, dims), (a,), backward, "permute")
    return out


def unsqueeze(a: Any, dim: int) -> Tensor:
    a = _as_tensor(a)
    d = dim + a.ndim + 1 if dim < 0 else dim
    if not 0 <= d <= a.ndim:
        raise IndexError(f"unsqueeze: dim={dim} out of range")

    def backward() -> None:
        g = out._grad
        assert g is not None
        if a.requires_grad:
            _add_grad(a, np.squeeze(g, axis=d))

    out = _make_result(np.expand_dims(a.data, d), (a,), backward, "unsqueeze")
    return out


def squeeze(a: Any, dim: int | None = None) -> Tensor:
    a = _as_tensor(a)
    if dim is None:
        out_data = np.squeeze(a.data)
    else:
        d = _norm_dim(dim, a.ndim)
        if a.shape[d] != 1:
            return a
        out_data = np.squeeze(a.data, axis=d)

    def backward() -> None:
        g = out._grad
        assert g is not None
        if a.requires_grad:
            _add_grad(a, g.reshape(a.shape))

    out = _make_result(out_data, (a,), backward, "squeeze")
    return out


def getitem(a: Any, index: Any) -> Tensor:
    a = _as_tensor(a)

    def backward() -> None:
        g = out._grad
        assert g is not None
        if a.requires_grad:
            gfull = np.zeros_like(a.data)
            if _is_basic_index(index):
                gfull[index] += g
            else:
                np.add.at(gfull, index, g)
            _add_grad(a, gfull)

    out = _make_result(a.data[index], (a,), backward, "getitem")
    return out


def cat(tensors: Sequence[Any], dim: int = 0) -> Tensor:
    ts = [_as_tensor(t) for t in tensors]
    if not ts:
        raise RuntimeError("cat: empty tensor list")
    ndim = ts[0].ndim
    d = _norm_dim(dim, ndim)
    sizes = [t.shape[d] for t in ts]

    # torch promotion: any float wins over integers; among floats promote.
    float_dtypes = [t.dtype for t in ts if t.dtype.kind == "f"]
    if float_dtypes:
        result_dtype = float_dtypes[0]
        for dtype in float_dtypes[1:]:
            result_dtype = np.promote_types(result_dtype, dtype)
    else:
        result_dtype = ts[0].dtype
        for t in ts[1:]:
            result_dtype = np.promote_types(result_dtype, t.dtype)
    data = [t.data.astype(result_dtype, copy=False) for t in ts]

    def backward() -> None:
        g = out._grad
        assert g is not None
        start = 0
        for t, size in zip(ts, sizes):
            if t.requires_grad:
                index = [slice(None)] * ndim
                index[d] = slice(start, start + size)
                _add_grad(t, g[tuple(index)])
            start += size

    out = _make_result(np.concatenate(data, axis=d), tuple(ts), backward, "cat")
    return out


def clone(a: Any) -> Tensor:
    a = _as_tensor(a)

    def backward() -> None:
        g = out._grad
        assert g is not None
        if a.requires_grad:
            _add_grad(a, g)

    out = _make_result(a.data.copy(), (a,), backward, "clone")
    return out


def cast(a: Any, dtype: Any) -> Tensor:
    a = _as_tensor(a)
    dtype = np.dtype(dtype)

    def backward() -> None:
        g = out._grad
        assert g is not None
        if a.requires_grad:
            _add_grad(a, g.astype(a.dtype))

    out = _make_result(a.data.astype(dtype), (a,), backward, "cast")
    return out


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


def _is_full_reduction(dim: Any) -> bool:
    if dim is None:
        return True
    if isinstance(dim, (int, np.integer)):
        return False
    return len(tuple(dim)) == 0


def sum(
    a: Any,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> Tensor:
    a = _as_tensor(a)
    full = _is_full_reduction(dim)
    if full:
        dims = tuple(range(a.ndim))
    else:
        dims = _norm_dims(dim, a.ndim)
    if full and not keepdim:
        out_data = np.sum(a.data)
    else:
        out_data = np.sum(a.data, axis=dims, keepdims=keepdim)

    def backward() -> None:
        g = out._grad
        assert g is not None
        if not a.requires_grad:
            return
        if full:
            _add_grad(a, np.broadcast_to(g, a.shape))
            return
        if not keepdim:
            g = np.expand_dims(g, axis=dims)
        _add_grad(a, np.broadcast_to(g, a.shape))

    out = _make_result(out_data, (a,), backward, "sum")
    return out


def mean(
    a: Any,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> Tensor:
    a = _as_tensor(a)
    full = _is_full_reduction(dim)
    if full:
        dims = tuple(range(a.ndim))
    else:
        dims = _norm_dims(dim, a.ndim)
    if full and not keepdim:
        out_data = np.mean(a.data)
    else:
        out_data = np.mean(a.data, axis=dims, keepdims=keepdim)
    count = int(a.data.size) if full else _reduce_count(a.shape, dims)

    def backward() -> None:
        g = out._grad
        assert g is not None
        if not a.requires_grad:
            return
        if full:
            _add_grad(a, np.broadcast_to(g / count, a.shape))
            return
        if not keepdim:
            g = np.expand_dims(g, axis=dims)
        _add_grad(a, np.broadcast_to(g / count, a.shape))

    out = _make_result(out_data, (a,), backward, "mean")
    return out


# ---------------------------------------------------------------------------
# Activations / normalization
# ---------------------------------------------------------------------------


def relu(a: Any) -> Tensor:
    a = _as_tensor(a)

    def backward() -> None:
        g = out._grad
        assert g is not None
        if a.requires_grad:
            _add_grad(a, g * (a.data > 0))

    out = _make_result(np.maximum(a.data, 0), (a,), backward, "relu")
    return out


def gelu(a: Any, approximate: str = "none") -> Tensor:
    """GELU as ``torch.nn.functional.gelu`` (exact erf by default)."""
    a = _as_tensor(a)
    x = a.data
    if approximate == "none":
        cdf = 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))
        cdf = np.asarray(cdf, dtype=x.dtype)
        out_data = x * cdf
        local_grad = cdf + x * np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    elif approximate == "tanh":
        c = math.sqrt(2.0 / math.pi)
        inner = c * (x + 0.044715 * x**3)
        tanh_inner = np.tanh(inner)
        out_data = 0.5 * x * (1.0 + tanh_inner)
        local_grad = 0.5 * (1.0 + tanh_inner) + 0.5 * x * (
            1.0 - tanh_inner * tanh_inner
        ) * c * (1.0 + 3.0 * 0.044715 * x * x)
    else:
        raise ValueError(f"gelu: approximate={approximate!r}, expected 'none' or 'tanh'")
    local_grad = np.asarray(local_grad, dtype=out_data.dtype)

    def backward() -> None:
        g = out._grad
        assert g is not None
        if a.requires_grad:
            _add_grad(a, g * local_grad)

    out = _make_result(out_data, (a,), backward, "gelu")
    return out


def softmax(a: Any, dim: int = -1) -> Tensor:
    a = _as_tensor(a)
    d = _norm_dim(dim, a.ndim)
    x = a.data
    shifted = x - np.max(x, axis=d, keepdims=True)
    exp_x = np.exp(shifted)
    out_data = exp_x / np.sum(exp_x, axis=d, keepdims=True)

    def backward() -> None:
        g = out._grad
        assert g is not None
        if a.requires_grad:
            dx = out_data * (g - np.sum(g * out_data, axis=d, keepdims=True))
            _add_grad(a, dx)

    out = _make_result(out_data, (a,), backward, "softmax")
    return out


def log_softmax(a: Any, dim: int = -1) -> Tensor:
    a = _as_tensor(a)
    d = _norm_dim(dim, a.ndim)
    x = a.data
    shifted = x - np.max(x, axis=d, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=d, keepdims=True))
    out_data = shifted - logsumexp

    def backward() -> None:
        g = out._grad
        assert g is not None
        if a.requires_grad:
            _add_grad(a, g - np.exp(out_data) * np.sum(g, axis=d, keepdims=True))

    out = _make_result(out_data, (a,), backward, "log_softmax")
    return out


def layer_norm(
    a: Any,
    normalized_shape: Sequence[int] | int,
    weight: Any | None = None,
    bias: Any | None = None,
    eps: float = 1e-5,
) -> Tensor:
    """LayerNorm with biased variance, like ``torch.nn.functional.layer_norm``."""
    a = _as_tensor(a)
    if isinstance(normalized_shape, (int, np.integer)):
        normalized_shape = (int(normalized_shape),)
    normalized_shape = tuple(int(s) for s in normalized_shape)
    nd = len(normalized_shape)
    if nd == 0 or a.shape[-nd:] != normalized_shape:
        raise RuntimeError(
            f"layer_norm: input shape tail {a.shape} does not match "
            f"normalized_shape {normalized_shape}"
        )
    axes = tuple(range(a.ndim - nd, a.ndim))
    x = a.data
    mu = np.mean(x, axis=axes, keepdims=True)
    xmu = x - mu
    var = np.mean(xmu * xmu, axis=axes, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = xmu * rstd
    out_data = xhat
    if weight is not None:
        weight = _as_tensor(weight)
        out_data = out_data * weight.data
    if bias is not None:
        bias = _as_tensor(bias)
        out_data = out_data + bias.data
    parents = tuple(p for p in (a, weight, bias) if isinstance(p, Tensor))

    def backward() -> None:
        g = out._grad
        assert g is not None
        if weight is not None:
            gxhat = g * weight.data
            if weight.requires_grad:
                _add_grad(weight, _unbroadcast(g * xhat, weight.shape))
        else:
            gxhat = g
        if bias is not None and bias.requires_grad:
            _add_grad(bias, _unbroadcast(g, bias.shape))
        if not a.requires_grad:
            return
        n = int(np.prod(normalized_shape))
        dvar = np.sum(gxhat * xmu * (-0.5) * (rstd**3), axis=axes, keepdims=True)
        dmu = np.sum(-gxhat * rstd, axis=axes, keepdims=True) + dvar * np.mean(
            -2.0 * xmu, axis=axes, keepdims=True
        )
        dx = gxhat * rstd + dvar * (2.0 * xmu / n) + dmu / n
        _add_grad(a, dx)

    out = _make_result(out_data, parents, backward, "layer_norm")
    return out


# ---------------------------------------------------------------------------
# Dropout and masks
# ---------------------------------------------------------------------------


def dropout(
    a: Any,
    p: float = 0.5,
    training: bool = True,
    mask: Any | None = None,
) -> Tensor:
    """Inverted dropout.

    ``mask`` is an optional extension used by tests: a boolean array where
    True means "keep". It allows comparing against torch with an identical
    mask even though the RNGs differ.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout: p={p}, expected 0 <= p <= 1")
    a = _as_tensor(a)
    if not training or p == 0.0:
        return a
    if p == 1.0:
        def backward() -> None:
            g = out._grad
            assert g is not None
            if a.requires_grad:
                _add_grad(a, np.zeros_like(a.data))

        out = _make_result(np.zeros_like(a.data), (a,), backward, "dropout")
        return out
    if mask is None:
        keep = get_rng().random(a.shape) >= p
    else:
        keep = np.asarray(mask.data if isinstance(mask, Tensor) else mask, dtype=bool)
        try:
            keep = np.broadcast_to(keep, a.shape)
        except ValueError as exc:
            raise ValueError(
                f"dropout: mask {keep.shape} does not broadcast to {a.shape}"
            ) from exc
    scale = 1.0 / (1.0 - p)
    out_data = a.data * keep * scale

    def backward() -> None:
        g = out._grad
        assert g is not None
        if a.requires_grad:
            _add_grad(a, g * keep * scale)

    out = _make_result(out_data, (a,), backward, "dropout")
    return out


def masked_fill(a: Any, mask: Any, value: float) -> Tensor:
    a = _as_tensor(a)
    raw = mask.data if isinstance(mask, Tensor) else np.asarray(mask)
    if raw.dtype != bool:
        raise TypeError(f"masked_fill: mask must be bool, got {raw.dtype}")
    mask_b = np.broadcast_to(raw, a.shape)
    fill = np.asarray(value, dtype=a.dtype)

    def backward() -> None:
        g = out._grad
        assert g is not None
        if a.requires_grad:
            _add_grad(a, np.where(mask_b, np.zeros((), dtype=a.dtype), g))

    out = _make_result(np.where(mask_b, fill, a.data), (a,), backward, "masked_fill")
    return out
