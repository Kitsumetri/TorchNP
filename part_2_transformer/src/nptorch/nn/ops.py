"""Numerical kernels with explicit forward and backward passes.

Every kernel caches exactly the values needed by its backward pass.  There is
no automatic differentiation: each backward function below evaluates the
analytic gradient and its docstring contains the complete derivation.  The
same derivations, expanded with intermediate steps, are collected in
``math_sources.md`` at the project root.
"""

from __future__ import annotations

import math

import numpy as np

try:  # SciPy is optional; math.erf is used as a fallback.
    from scipy.special import erf as _erf
except ImportError:  # pragma: no cover - exercised only without SciPy
    _erf = np.vectorize(math.erf, otypes=[np.float64])


def _leading_axes(array: np.ndarray) -> tuple[int, ...]:
    """Return all axes except the last one."""
    return tuple(range(array.ndim - 1))


# ---------------------------------------------------------------------------
# Affine map
# ---------------------------------------------------------------------------


def linear_forward(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    r"""Compute :math:`Y = X W^{\top} + b`.

    ``X`` has shape ``(..., D_in)``, ``W`` has shape ``(D_out, D_in)`` and
    ``b`` has shape ``(D_out,)``.  The leading dimensions of ``X`` are
    arbitrary (they are flattened by the backward pass).
    """
    y = x @ weight.T
    if bias is not None:
        y = y + bias
    return y, (x, weight)


def linear_backward(
    dy: np.ndarray,
    cache: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Backward pass of :math:`Y = X W^{\top} + b`.

    Let :math:`\bar Y = \partial L/\partial Y`.  With ``D_in`` input features,
    ``D_out`` output features and the leading index :math:`r` (all axes of
    ``X`` except the last one, flattened):

    .. math::

        \bar X_{\ldots i} = \sum_{o} \bar Y_{\ldots o} W_{o i},

        \bar W_{o i} = \sum_{r} \bar Y_{r o} X_{r i},

        \bar b_o = \sum_{r} \bar Y_{r o}.

    In array form: ``dX = dY @ W``, ``dW = dY_flat.T @ X_flat``,
    ``db = dY_flat.sum(axis=0)``.  The bias gradient is always computed;
    callers discard it when the layer has no bias.
    """
    x, weight = cache
    dx = dy @ weight
    dy_flat = dy.reshape(-1, dy.shape[-1])
    x_flat = x.reshape(-1, x.shape[-1])
    dw = dy_flat.T @ x_flat
    db = dy_flat.sum(axis=0)
    return dx, dw, db


# ---------------------------------------------------------------------------
# Layer normalization
# ---------------------------------------------------------------------------


def layer_norm_forward(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray | None,
    eps: float,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    r"""Layer normalization over the last axis.

    For :math:`D` features and one row :math:`x \in \mathbb{R}^{D}`:

    .. math::

        \mu = \frac{1}{D}\sum_{i=1}^{D} x_i, \qquad
        \sigma = \sqrt{\frac{1}{D}\sum_{i=1}^{D}(x_i-\mu)^2 + \varepsilon},

        \hat x_i = \frac{x_i-\mu}{\sigma}, \qquad
        y_i = \gamma_i \hat x_i + \beta_i.

    The cache stores ``(x_hat, rstd, gamma)`` with ``rstd = 1 / sigma``.
    """
    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    var = np.mean(centered * centered, axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    x_hat = centered * rstd
    y = x_hat * gamma
    if beta is not None:
        y = y + beta
    return y, (x_hat, rstd, gamma)


def layer_norm_backward(
    dy: np.ndarray,
    cache: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Backward pass of layer normalization.

    Write :math:`g_i = \bar y_i \gamma_i`, where
    :math:`\bar y = \partial L/\partial y`, and let :math:`D` be the number of
    normalized features.  Differentiating :math:`\hat x_j` gives

    .. math::

        \frac{\partial \hat x_j}{\partial x_i}
        = \frac{\delta_{ij}}{\sigma}
          - \frac{1}{D\sigma}
          - \frac{\hat x_i \hat x_j}{D\sigma}.

    Therefore

    .. math::

        \bar x_i = \sum_j g_j \frac{\partial \hat x_j}{\partial x_i}
        = \frac{1}{\sigma}\left[
            g_i - \frac{1}{D}\sum_j g_j
            - \frac{\hat x_i}{D}\sum_j g_j \hat x_j
          \right],

        \bar \gamma_i = \sum_{r} \bar y_{r i} \hat x_{r i}, \qquad
        \bar \beta_i = \sum_{r} \bar y_{r i},

    where :math:`r` enumerates all leading (non-normalized) indices.  The
    returned :math:`\bar\beta` is ignored by callers when ``beta`` is absent.
    """
    x_hat, rstd, gamma = cache
    g = dy * gamma
    num_features = x_hat.shape[-1]
    sum_g = g.sum(axis=-1, keepdims=True)
    sum_g_xhat = (g * x_hat).sum(axis=-1, keepdims=True)
    dx = (rstd / num_features) * (
        num_features * g - sum_g - x_hat * sum_g_xhat
    )
    axes = _leading_axes(x_hat)
    dgamma = (dy * x_hat).sum(axis=axes)
    dbeta = dy.sum(axis=axes)
    return dx, dgamma, dbeta


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------


def dropout_forward(
    x: np.ndarray,
    p: float,
    training: bool,
    mask: np.ndarray | None = None,
    rng: object | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, float] | None]:
    r"""Inverted dropout.

    For independent Bernoulli variables :math:`m_i \sim \mathrm{Bernoulli}(1-p)`:

    .. math::

        y_i = \frac{m_i x_i}{1-p}, \qquad
        \mathbb{E}[y_i] = x_i.

    ``mask`` is the boolean *keep* mask.  Supplying it makes the operation
    deterministic and is used by tests and finite-difference checks; otherwise
    it is sampled from ``rng`` (``numpy.random`` by default).  ``p == 1``
    returns zeros with an all-false mask.  In evaluation mode, or for
    ``p == 0``, the input is returned unchanged and the cache is ``None``.
    """
    if not training or p == 0.0:
        return x, None
    if p == 1.0:
        return np.zeros_like(x), (np.zeros(x.shape, dtype=bool), 0.0)
    if mask is None:
        source = np.random if rng is None else rng
        mask = source.random(x.shape) >= p
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != x.shape:
        raise ValueError(
            f"dropout mask shape {mask.shape} does not match input shape {x.shape}"
        )
    scale = 1.0 / (1.0 - p)
    return np.where(mask, x * scale, 0.0), (mask, scale)


def dropout_backward(
    dy: np.ndarray,
    cache: tuple[np.ndarray, float] | None,
) -> np.ndarray:
    r"""Backward pass of inverted dropout.

    Since :math:`y_i = m_i x_i/(1-p)` and the mask is constant during the
    backward pass,

    .. math::

        \bar x_i = \frac{m_i}{1-p}\bar y_i.

    For ``p == 1`` the cached scale is zero, which gives
    :math:`\bar x_i = 0` as required for a constant function.  With
    ``cache is None`` (dropout disabled) this is the identity.
    """
    if cache is None:
        return dy
    mask, scale = cache
    return dy * mask * scale


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------


def softmax_forward(scores: np.ndarray) -> np.ndarray:
    r"""Numerically stable softmax over the last axis.

    .. math::

        p_i = \frac{\exp(s_i - \max_j s_j)}
                   {\sum_k \exp(s_k - \max_j s_j)}.

    A row whose scores are all ``-inf`` (every key masked) is assigned an
    all-zero probability row instead of producing ``NaN``.  Its backward pass
    is consequently zero, matching the safe SDPA convention.
    """
    all_masked = np.all(np.isneginf(scores), axis=-1, keepdims=True)
    safe_scores = np.where(all_masked, 0.0, scores)
    shifted = safe_scores - safe_scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    return np.where(all_masked, 0.0, probs)


def softmax_backward(dprobs: np.ndarray, probs: np.ndarray) -> np.ndarray:
    r"""Backward pass of softmax.

    From :math:`p_j = e^{s_j}/\sum_k e^{s_k}` it follows that

    .. math::

        \frac{\partial p_j}{\partial s_i} = p_j(\delta_{ji} - p_i).

    Hence, for :math:`\bar p = \partial L/\partial p`,

    .. math::

        \bar s_i = \sum_j \bar p_j \frac{\partial p_j}{\partial s_i}
        = p_i\left(\bar p_i - \sum_j p_j \bar p_j\right).

    Masked positions have :math:`p_i = 0`, so their score gradient is exactly
    zero and the ``-inf`` mask never receives a gradient.  Fully masked rows
    also yield zero.
    """
    return probs * (dprobs - (dprobs * probs).sum(axis=-1, keepdims=True))


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


def relu_forward(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""ReLU: :math:`f(x) = \max(0, x)`; the cache is the positive mask."""
    return np.maximum(x, 0.0), x > 0.0


def relu_backward(dy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    r"""Backward pass of ReLU.

    With :math:`\mathbf{1}[x_i > 0]` evaluated at the forward input,

    .. math::

        \bar x_i = \bar y_i \, \mathbf{1}[x_i > 0].

    The derivative at :math:`x_i = 0` is defined as zero, as in PyTorch.
    """
    return dy * mask


def gelu_forward(x: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    r"""Exact (erf-based) GELU.

    Let :math:`\Phi(x) = \tfrac12\left(1+\operatorname{erf}(x/\sqrt2)\right)`
    be the standard normal CDF.  Then

    .. math::

        f(x) = x \Phi(x).

    The cache stores ``(x, Phi(x))``.
    """
    cdf = 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))
    return x * cdf, (x, cdf)


def gelu_backward(
    dy: np.ndarray,
    cache: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    r"""Backward pass of exact GELU.

    With the standard normal PDF
    :math:`\varphi(x) = e^{-x^2/2}/\sqrt{2\pi}`,

    .. math::

        f'(x) = \Phi(x) + x\varphi(x),

        \bar x_i = \bar y_i \left(\Phi(x_i) + x_i \varphi(x_i)\right).
    """
    x, cdf = cache
    pdf = np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    return dy * (cdf + x * pdf)
