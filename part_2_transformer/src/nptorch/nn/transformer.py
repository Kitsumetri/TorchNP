"""A NumPy transformer encoder layer with hand-written backpropagation.

This module implements a single ``TransformerEncoderLayer`` equivalent to
``torch.nn.TransformerEncoderLayer``.  There is no automatic differentiation
and no tensor abstraction: the layer works with plain ``numpy.ndarray``
objects, ``forward`` stores the intermediate values, and ``backward`` applies
the chain rule explicitly.

The parameter names in :attr:`TransformerEncoderLayer.params` match
``torch.nn.TransformerEncoderLayer.state_dict()`` exactly, which makes copying
weights and comparing gradients with PyTorch straightforward.  The complete
mathematical derivations are also collected in ``math_sources.md``.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Iterator

import numpy as np

from .. import init
from . import ops


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a torch tensor, NumPy array or array-like to float64."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def _float_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise TypeError(f"{name} must be numeric, got dtype={array.dtype}")
    if array.dtype != np.float64:
        array = array.astype(np.float64)
    return array


def _canonical_mask(mask: Any, name: str) -> np.ndarray | None:
    """Convert bool masks to additive ``-inf`` masks, keep float masks."""
    if mask is None:
        return None
    array = np.asarray(mask)
    if array.dtype == bool:
        return np.where(array, -np.inf, 0.0)
    if array.dtype.kind == "f":
        return array.astype(np.float64, copy=False)
    raise TypeError(f"{name} must be bool or floating point, got {array.dtype}")


class TransformerEncoderLayer:
    r"""One transformer encoder layer with explicit forward/backward passes.

    The layer matches the default PyTorch configuration: post-layer
    normalization, ReLU feed-forward activation, dropout ``0.1``, bias
    enabled and sequence-first input.  Supported options are ``norm_first``,
    ``batch_first``, ``bias=False``, ``activation="gelu"`` and arbitrary
    ``layer_norm_eps``.

    Parameters
    ----------
    d_model:
        Model width :math:`E`.
    nhead:
        Number of attention heads :math:`H`; must divide ``d_model``.
    dim_feedforward:
        Hidden width :math:`F` of the position-wise feed-forward network.
    dropout:
        Dropout probability.  As in PyTorch it is applied to the attention
        weights, the attention-block output, the activation between the two
        feed-forward linears, and the feed-forward output.
    activation:
        ``"relu"`` or ``"gelu"`` (exact erf-based GELU).  Callables whose
        ``__name__`` is ``"relu"`` or ``"gelu"`` are also accepted.
    layer_norm_eps:
        Numerical stabilization :math:`\varepsilon` inside layer norm.

    Forward equations
    -----------------
    Let the input be :math:`X \in \mathbb{R}^{L \times N \times E}` (the
    internal sequence-first layout).  With :math:`d_h = E/H`, the
    self-attention block is

    .. math::

        Q = XW_q^{\top}+b_q,\quad
        K = XW_k^{\top}+b_k,\quad
        V = XW_v^{\top}+b_v,

        Q_h = \operatorname{heads}(Q),\quad
        K_h = \operatorname{heads}(K),\quad
        V_h = \operatorname{heads}(V),

        S_h = \frac{Q_hK_h^{\top}}{\sqrt{d_h}} + M,
        \qquad
        A_h = \operatorname{softmax}(S_h),

        A_h^{d} = \operatorname{dropout}(A_h),\qquad
        O = \operatorname{concat}_h(A_h^{d}V_h),\qquad
        \operatorname{MHA}(X) = OW_o^{\top}+b_o,

        \operatorname{SA}(X) = \operatorname{dropout}(\operatorname{MHA}(X)).

    The position-wise feed-forward block is

    .. math::

        \operatorname{FF}(X)
        = \operatorname{dropout}\!\left(
            \operatorname{Linear}_2\!\left(
              \operatorname{dropout}\!\left(
                \sigma(\operatorname{Linear}_1(X))\right)\right)\right),

    where :math:`\sigma` is ReLU or exact GELU.  The two layer-norm
    arrangements are

    .. math::

        \text{post-norm:}\quad
        Z = \operatorname{LN}_1(X+\operatorname{SA}(X)),\quad
        Y = \operatorname{LN}_2(Z+\operatorname{FF}(Z)),

        \text{pre-norm:}\quad
        Z = X+\operatorname{SA}(\operatorname{LN}_1(X)),\quad
        Y = Z+\operatorname{FF}(\operatorname{LN}_2(Z)).

    Masks
    -----
    ``src_mask`` is an additive mask of shape ``(L, S)`` or
    ``(N*H, L, S)``; boolean entries mean "masked" and are converted to
    ``-inf``.  ``src_key_padding_mask`` has shape ``(N, S)`` (or ``(S,)``
    for unbatched input; ``(1, S)`` is also accepted there as a convenience,
    while torch requires the 1-D form); boolean ``True`` means that the key
    is padding.  ``is_causal`` follows PyTorch's MHA hint semantics: a mask
    is still required, and when ``src_key_padding_mask`` is absent the
    causal mask is used instead of ``src_mask``.  A query row with every key
    masked gets zero attention weights and a zero score-row gradient rather
    than ``NaN``; its input gradient can still be nonzero because its value
    vector is used by other queries.

    Gradients
    ---------
    :meth:`backward` accumulates parameter gradients into :attr:`grads`,
    keyed by the same names as :attr:`params`, and returns the gradient with
    respect to ``src``.  Call :meth:`zero_grad` between optimization steps.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Any = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
    ) -> None:
        self.d_model = self._positive_int(d_model, "d_model")
        self.nhead = self._positive_int(nhead, "nhead")
        self.dim_feedforward = self._positive_int(
            dim_feedforward, "dim_feedforward"
        )
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.head_dim = self.d_model // self.nhead
        self.dropout = float(dropout)
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")
        self.layer_norm_eps = float(layer_norm_eps)
        if self.layer_norm_eps <= 0.0:
            raise ValueError("layer_norm_eps must be positive")
        self.batch_first = bool(batch_first)
        self.norm_first = bool(norm_first)
        self.bias = bool(bias)
        self.activation = self._resolve_activation(activation)

        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}
        self._cache: dict[str, Any] | None = None
        self.training = True
        self._build_parameters()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return value

    @staticmethod
    def _resolve_activation(activation: Any) -> str:
        if isinstance(activation, str):
            name = activation
        elif callable(activation):
            name = getattr(activation, "__name__", "")
        else:
            raise TypeError("activation must be 'relu', 'gelu' or a callable")
        if name not in ("relu", "gelu"):
            raise NotImplementedError(
                f"activation must be 'relu' or 'gelu', got {activation!r}"
            )
        return name

    def _add_param(self, name: str, shape: tuple[int, ...], initializer) -> None:
        array = np.empty(shape, dtype=np.float64)
        initializer(array)
        self.params[name] = array

    def _build_parameters(self) -> None:
        d_model = self.d_model
        ffn = self.dim_feedforward

        # MHA in-projection: torch stores q, k, v concatenated row-wise.
        self._add_param(
            "self_attn.in_proj_weight",
            (3 * d_model, d_model),
            init.xavier_uniform_,
        )
        if self.bias:
            self._add_param(
                "self_attn.in_proj_bias",
                (3 * d_model,),
                lambda a: init.constant_(a, 0.0),
            )

        # MHA output projection.
        self._add_param(
            "self_attn.out_proj.weight",
            (d_model, d_model),
            init.kaiming_uniform_,
        )
        if self.bias:
            self._add_param(
                "self_attn.out_proj.bias",
                (d_model,),
                lambda a: init.constant_(a, 0.0),
            )

        # Feed-forward network.  torch.nn.Linear initializes its bias from
        # U(-1/sqrt(fan_in), 1/sqrt(fan_in)), unlike the attention and
        # layer-norm biases which are zero.
        self._add_param("linear1.weight", (ffn, d_model), init.kaiming_uniform_)
        if self.bias:
            bound1 = 1.0 / math.sqrt(float(d_model))
            self._add_param(
                "linear1.bias",
                (ffn,),
                lambda a: init.uniform_(a, -bound1, bound1),
            )
        self._add_param("linear2.weight", (d_model, ffn), init.kaiming_uniform_)
        if self.bias:
            bound2 = 1.0 / math.sqrt(float(ffn))
            self._add_param(
                "linear2.bias",
                (d_model,),
                lambda a: init.uniform_(a, -bound2, bound2),
            )

        # Layer norms.  With bias=False torch LayerNorm keeps weight only.
        for name in ("norm1", "norm2"):
            self._add_param(
                f"{name}.weight", (d_model,), lambda a: init.ones_(a)
            )
            if self.bias:
                self._add_param(
                    f"{name}.bias", (d_model,), lambda a: init.constant_(a, 0.0)
                )

    # ------------------------------------------------------------------
    # Public parameter / state API
    # ------------------------------------------------------------------

    def named_parameters(self) -> Iterator[tuple[str, np.ndarray]]:
        """Yield ``(name, array)`` pairs in torch ``state_dict`` order."""
        yield from self.params.items()

    def parameters(self) -> list[np.ndarray]:
        """Return the trainable arrays (live references, not copies)."""
        return list(self.params.values())

    def state_dict(self) -> dict[str, np.ndarray]:
        """Return a copy of the parameters using torch-compatible names."""
        return {name: value.copy() for name, value in self.params.items()}

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        strict: bool = True,
    ) -> "TransformerEncoderLayer":
        """Copy parameters from a state dict.

        Values may be NumPy arrays, array-likes or torch tensors (they are
        moved to the CPU).  With ``strict=True`` the key set and every shape
        must match exactly.  Loading clears accumulated gradients and the
        forward cache.
        """
        self._cache = None
        incoming = {name: _to_numpy(value) for name, value in state_dict.items()}
        missing = sorted(set(self.params) - set(incoming))
        unexpected = sorted(set(incoming) - set(self.params))
        if strict and (missing or unexpected):
            parts = []
            if missing:
                parts.append(f"missing keys: {missing}")
            if unexpected:
                parts.append(f"unexpected keys: {unexpected}")
            raise RuntimeError("Error(s) in loading state_dict: " + "; ".join(parts))
        for name, value in incoming.items():
            if name not in self.params:
                continue
            if value.shape != self.params[name].shape:
                raise RuntimeError(
                    f"shape mismatch for {name}: expected "
                    f"{self.params[name].shape}, got {value.shape}"
                )
            self.params[name][...] = value
        self.zero_grad()
        return self

    def zero_grad(self) -> None:
        """Drop all accumulated parameter gradients."""
        self.grads.clear()

    def train(self, mode: bool = True) -> "TransformerEncoderLayer":
        """Set training mode (dropout active when ``mode=True``)."""
        self.training = bool(mode)
        return self

    def eval(self) -> "TransformerEncoderLayer":
        """Set evaluation mode (dropout disabled)."""
        return self.train(False)

    # ------------------------------------------------------------------
    # Parameter gradient helpers
    # ------------------------------------------------------------------

    def _accumulate(self, name: str, gradient: np.ndarray) -> None:
        if name not in self.params:
            return
        if name in self.grads:
            self.grads[name] += gradient
        else:
            self.grads[name] = gradient.copy()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        src: Any,
        src_mask: Any = None,
        src_key_padding_mask: Any = None,
        is_causal: bool = False,
    ) -> np.ndarray:
        r"""Run the layer forward.

        ``src`` may be ``(L, N, E)``, ``(N, L, E)`` when ``batch_first=True``,
        or unbatched ``(L, E)``.  The result has the same layout as ``src``.
        All computations are performed in float64.

        The forward pass caches every intermediate needed by
        :meth:`backward`; calling ``forward`` again overwrites that cache.
        Do not modify ``src`` or :attr:`params` in place between ``forward``
        and ``backward``: the cache intentionally stores references to them.
        """
        self._cache = None
        x = _float_array(src, "src")
        unbatched = x.ndim == 2
        if unbatched:
            x_layout = x[:, None, :]
        elif x.ndim == 3:
            x_layout = x.transpose(1, 0, 2) if self.batch_first else x
        else:
            raise RuntimeError(
                f"src must be 2-D (L, E) or 3-D, got shape {x.shape}"
            )

        seq_len, batch_size, embed_dim = x_layout.shape
        if embed_dim != self.d_model:
            raise RuntimeError(
                f"src embedding dimension {embed_dim} != d_model {self.d_model}"
            )

        attn_mask = _canonical_mask(src_mask, "src_mask")
        key_padding_mask = _canonical_mask(
            src_key_padding_mask, "src_key_padding_mask"
        )
        self._check_mask_shapes(
            attn_mask, key_padding_mask, seq_len, batch_size, unbatched
        )

        if is_causal and src_mask is None:
            raise RuntimeError(
                "Need attn_mask if specifying the is_causal hint. "
                "Pass a (L, S) mask (a causal mask is the usual choice)."
            )

        # Match torch's slow-path is_causal handling.  The flag is a hint for
        # SDPA: with no key padding it replaces src_mask by a causal mask;
        # with key padding the explicit src_mask is used.
        use_mask = attn_mask
        apply_causal = False
        if is_causal:
            if key_padding_mask is None:
                use_mask = None
                apply_causal = True
            else:
                use_mask = attn_mask

        if self.norm_first:
            ln1_out, ln1_cache = self._ln_forward(x_layout, "norm1")
            sa_out, sa_cache = self._sa_forward(
                ln1_out, use_mask, key_padding_mask, apply_causal
            )
            hidden = x_layout + sa_out
            ln2_out, ln2_cache = self._ln_forward(hidden, "norm2")
            ff_out, ff_cache = self._ff_forward(ln2_out)
            out = hidden + ff_out
            caches = {
                "ln1": ln1_cache,
                "sa": sa_cache,
                "ln2": ln2_cache,
                "ff": ff_cache,
            }
        else:
            sa_out, sa_cache = self._sa_forward(
                x_layout, use_mask, key_padding_mask, apply_causal
            )
            hidden1 = x_layout + sa_out
            ln1_out, ln1_cache = self._ln_forward(hidden1, "norm1")
            ff_out, ff_cache = self._ff_forward(ln1_out)
            hidden2 = ln1_out + ff_out
            out, ln2_cache = self._ln_forward(hidden2, "norm2")
            caches = {
                "sa": sa_cache,
                "ln1": ln1_cache,
                "ff": ff_cache,
                "ln2": ln2_cache,
            }

        if unbatched:
            out = out[:, 0, :]
        elif self.batch_first:
            out = out.transpose(1, 0, 2)

        self._cache = {
            "caches": caches,
            "unbatched": unbatched,
            "batch_first": self.batch_first,
            "norm_first": self.norm_first,
            "output_shape": out.shape,
            "input_ndim": x.ndim,
            "seq_len": seq_len,
            "batch_size": batch_size,
        }
        return out

    def _check_mask_shapes(
        self,
        attn_mask: np.ndarray | None,
        key_padding_mask: np.ndarray | None,
        seq_len: int,
        batch_size: int,
        unbatched: bool,
    ) -> None:
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                if attn_mask.shape != (seq_len, seq_len):
                    raise RuntimeError(
                        f"2-D src_mask must have shape {(seq_len, seq_len)}, "
                        f"got {attn_mask.shape}"
                    )
            elif attn_mask.ndim == 3:
                expected = (batch_size * self.nhead, seq_len, seq_len)
                if attn_mask.shape != expected:
                    raise RuntimeError(
                        f"3-D src_mask must have shape {expected}, "
                        f"got {attn_mask.shape}"
                    )
            else:
                raise RuntimeError(
                    f"src_mask must be 2-D or 3-D, got {attn_mask.ndim}-D"
                )

        if key_padding_mask is not None:
            if unbatched:
                if key_padding_mask.ndim == 1:
                    expected = (seq_len,)
                elif key_padding_mask.ndim == 2:
                    expected = (1, seq_len)
                else:
                    expected = None
            else:
                expected = (batch_size, seq_len)
            if key_padding_mask.shape != expected:
                raise RuntimeError(
                    "src_key_padding_mask must have shape "
                    f"{expected}, got {key_padding_mask.shape}"
                )

    # ------------------------------------------------------------------
    # Layer norm / feed-forward / attention blocks
    # ------------------------------------------------------------------

    def _ln_forward(
        self, x: np.ndarray, name: str
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        gamma = self.params[f"{name}.weight"]
        beta = self.params.get(f"{name}.bias")
        return ops.layer_norm_forward(x, gamma, beta, self.layer_norm_eps)

    def _ln_backward(
        self,
        dy: np.ndarray,
        cache: tuple[np.ndarray, np.ndarray, np.ndarray],
        name: str,
    ) -> np.ndarray:
        r"""Backward pass of a layer-norm sublayer.

        With :math:`\hat x = (x-\mu)/\sigma`, :math:`g_i=\bar y_i\gamma_i`
        and :math:`D` normalized features:

        .. math::

            \bar x_i = \frac{1}{\sigma}\left[
                g_i - \frac{1}{D}\sum_j g_j
                - \frac{\hat x_i}{D}\sum_j g_j\hat x_j\right],

            \bar\gamma_i = \sum_r \bar y_{ri}\hat x_{ri},\qquad
            \bar\beta_i = \sum_r \bar y_{ri},

        where :math:`r` enumerates the leading batch/sequence indices.  The
        :math:`\gamma` and :math:`\beta` gradients are accumulated under
        ``name.weight`` and ``name.bias``; the method returns
        :math:`\bar x`.  If the layer was built with ``bias=False``, the
        ``.bias`` entry does not exist and its gradient is discarded.
        """
        dx, dgamma, dbeta = ops.layer_norm_backward(dy, cache)
        self._accumulate(f"{name}.weight", dgamma)
        self._accumulate(f"{name}.bias", dbeta)
        return dx

    def _activation_forward(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.activation == "relu":
            out, payload = ops.relu_forward(x)
        else:
            out, payload = ops.gelu_forward(x)
        return out, {"name": self.activation, "payload": payload}

    @staticmethod
    def _activation_backward(dy: np.ndarray, cache: dict[str, Any]) -> np.ndarray:
        r"""Backward pass of the configured activation.

        For ReLU, with the mask :math:`\mathbf 1[x_i>0]` cached in the forward
        pass,

        .. math::

            \bar x_i = \bar y_i\,\mathbf 1[x_i>0].

        For exact GELU, with
        :math:`\Phi(x)=\tfrac12(1+\operatorname{erf}(x/\sqrt2))` and
        :math:`\varphi(x)=e^{-x^2/2}/\sqrt{2\pi}`,

        .. math::

            \bar x_i = \bar y_i\left(\Phi(x_i)+x_i\varphi(x_i)\right).
        """
        if cache["name"] == "relu":
            return ops.relu_backward(dy, cache["payload"])
        return ops.gelu_backward(dy, cache["payload"])

    def _ff_forward(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        hidden, linear1_cache = ops.linear_forward(
            x, self.params["linear1.weight"], self.params.get("linear1.bias")
        )
        activated, activation_cache = self._activation_forward(hidden)
        dropped1, dropout1_cache = ops.dropout_forward(
            activated, self.dropout, self.training
        )
        projected, linear2_cache = ops.linear_forward(
            dropped1,
            self.params["linear2.weight"],
            self.params.get("linear2.bias"),
        )
        out, dropout2_cache = ops.dropout_forward(
            projected, self.dropout, self.training
        )
        return out, {
            "linear1": linear1_cache,
            "activation": activation_cache,
            "dropout1": dropout1_cache,
            "linear2": linear2_cache,
            "dropout2": dropout2_cache,
        }

    def _ff_backward(self, dout: np.ndarray, cache: dict[str, Any]) -> np.ndarray:
        r"""Backward pass of the position-wise feed-forward block.

        With :math:`H=\operatorname{Linear}_1(X)`,
        :math:`A=\sigma(H)`, :math:`D_1=\operatorname{dropout}(A)`,
        :math:`P=\operatorname{Linear}_2(D_1)` and
        :math:`Y=\operatorname{dropout}(P)`, the reverse chain is

        .. math::

            \bar P &= \operatorname{dropout}'(\bar Y),\\
            \bar D_1 &= \bar P W_2, \qquad
            \bar W_2 = \bar P^{\top} D_1, \qquad
            \bar b_2 = \sum_r \bar P_r,\\
            \bar A &= \operatorname{dropout}'(\bar D_1),\\
            \bar H &= \bar A \odot \sigma'(H),\\
            \bar X &= \bar H W_1, \qquad
            \bar W_1 = \bar H^{\top} X, \qquad
            \bar b_1 = \sum_r \bar H_r,

        where :math:`r` runs over all leading dimensions.  ``dropout'`` is
        the inverted-dropout backward map from :func:`ops.dropout_backward`
        and :math:`\sigma'` is :func:`ops.relu_backward` or
        :func:`ops.gelu_backward`.
        """
        dprojected = ops.dropout_backward(dout, cache["dropout2"])
        ddropped1, dweight2, dbias2 = ops.linear_backward(
            dprojected, cache["linear2"]
        )
        self._accumulate("linear2.weight", dweight2)
        self._accumulate("linear2.bias", dbias2)

        dactivated = ops.dropout_backward(ddropped1, cache["dropout1"])
        dhidden = self._activation_backward(dactivated, cache["activation"])
        dx, dweight1, dbias1 = ops.linear_backward(dhidden, cache["linear1"])
        self._accumulate("linear1.weight", dweight1)
        self._accumulate("linear1.bias", dbias1)
        return dx

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """(L, N, E) -> (N*H, L, head_dim)."""
        seq_len, batch_size, _ = x.shape
        return (
            x.reshape(seq_len, batch_size, self.nhead, self.head_dim)
            .transpose(1, 2, 0, 3)
            .reshape(batch_size * self.nhead, seq_len, self.head_dim)
        )

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """(N*H, L, head_dim) -> (L, N, E)."""
        heads = x.shape[0]
        batch_size = heads // self.nhead
        seq_len = x.shape[1]
        return (
            x.reshape(batch_size, self.nhead, seq_len, self.head_dim)
            .transpose(2, 0, 1, 3)
            .reshape(seq_len, batch_size, self.d_model)
        )

    def _sa_forward(
        self,
        x: np.ndarray,
        attn_mask: np.ndarray | None,
        key_padding_mask: np.ndarray | None,
        apply_causal: bool,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        seq_len, batch_size, _ = x.shape
        d_model = self.d_model
        in_proj = self.params["self_attn.in_proj_weight"]
        in_bias = self.params.get("self_attn.in_proj_bias")
        if in_bias is None:
            bias_q = bias_k = bias_v = None
        else:
            bias_q = in_bias[:d_model]
            bias_k = in_bias[d_model : 2 * d_model]
            bias_v = in_bias[2 * d_model :]

        query, q_cache = ops.linear_forward(
            x, in_proj[:d_model], bias_q
        )
        key, k_cache = ops.linear_forward(
            x, in_proj[d_model : 2 * d_model], bias_k
        )
        value, v_cache = ops.linear_forward(
            x, in_proj[2 * d_model :], bias_v
        )

        q_heads = self._split_heads(query)
        k_heads = self._split_heads(key)
        v_heads = self._split_heads(value)

        scale = 1.0 / math.sqrt(float(self.head_dim))
        scores = (q_heads @ k_heads.transpose(0, 2, 1)) * scale

        if attn_mask is not None:
            scores = scores + attn_mask
        if key_padding_mask is not None:
            kpm = key_padding_mask.reshape(batch_size, 1, 1, seq_len)
            kpm = np.broadcast_to(
                kpm, (batch_size, self.nhead, seq_len, seq_len)
            ).reshape(batch_size * self.nhead, seq_len, seq_len)
            scores = scores + kpm
        if apply_causal:
            causal = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
            scores = np.where(causal, -np.inf, scores)

        attn = ops.softmax_forward(scores)
        dropped_attn, attn_dropout_cache = ops.dropout_forward(
            attn, self.dropout, self.training
        )
        attended = dropped_attn @ v_heads
        merged = self._merge_heads(attended)
        out, out_proj_cache = ops.linear_forward(
            merged,
            self.params["self_attn.out_proj.weight"],
            self.params.get("self_attn.out_proj.bias"),
        )
        out, output_dropout_cache = ops.dropout_forward(
            out, self.dropout, self.training
        )
        return out, {
            "q": q_cache,
            "k": k_cache,
            "v": v_cache,
            "q_heads": q_heads,
            "k_heads": k_heads,
            "v_heads": v_heads,
            "attn": attn,
            "dropped_attn": dropped_attn,
            "attn_dropout": attn_dropout_cache,
            "attended": attended,
            "out_proj": out_proj_cache,
            "output_dropout": output_dropout_cache,
            "scale": scale,
            "shapes": (seq_len, batch_size),
        }

    def _sa_backward(self, dout: np.ndarray, cache: dict[str, Any]) -> np.ndarray:
        r"""Backward pass of self-attention (including both dropouts).

        Write :math:`\bar Y=\partial L/\partial\,\operatorname{SA}(X)` and
        :math:`d_h=E/H`.  Let
        :math:`A_h^{d}=\operatorname{dropout}(A_h)`,
        :math:`O=\operatorname{concat}_h(A_h^{d}V_h)` and
        :math:`\text{out}=OW_o^{\top}+b_o`.  Inverting the output dropout
        and the output projection gives

        .. math::

            \bar Y_d = \operatorname{dropout}'(\bar Y),\qquad
            \bar O = \bar Y_dW_o,\qquad
            \bar W_o = \bar Y_d^{\top}O,\qquad
            \bar b_o = \sum_r \bar Y_{d,r}.

        Splitting :math:`\bar O` back into per-head matrices
        :math:`\bar O_h`, and writing
        :math:`\bar A_h^{d}=\partial L/\partial A_h^{d}`:

        .. math::

            \bar A_h^{d} = \bar O_h V_h^{\top},\qquad
            \bar V_h = (A_h^{d})^{\top}\bar O_h.

        The attention dropout and softmax backward passes are then

        .. math::

            \bar A_h = \operatorname{dropout}'(\bar A_h^{d}),

            \bar S_h = A_h \odot \left(
                \bar A_h
                - \left(\sum_j A_{h,ij}\bar A_{h,ij}\right)\mathbf 1^{\top}
            \right),

            \bar Q_h = \frac{1}{\sqrt{d_h}}\bar S_h K_h,\qquad
            \bar K_h = \frac{1}{\sqrt{d_h}}\bar S_h^{\top} Q_h.

        Concatenating the head gradients gives :math:`\bar Q,\bar K,\bar V`;
        because all three projections share the input :math:`X`,

        .. math::

            \bar X = \bar Q W_q + \bar K W_k + \bar V W_v,

            \bar W_q = \bar Q^{\top} X,\quad
            \bar W_k = \bar K^{\top} X,\quad
            \bar W_v = \bar V^{\top} X,

            \bar b_q = \sum_r \bar Q_r,\quad
            \bar b_k = \sum_r \bar K_r,\quad
            \bar b_v = \sum_r \bar V_r,

        and the three weight gradients are stacked to match the
        ``(3E, E)`` in-projection layout.  Masked entries have
        :math:`A_{h,ij}=0`, so the softmax backward pass gives them zero
        score gradient automatically.
        """
        seq_len, batch_size = cache["shapes"]
        d_model = self.d_model

        dmerged = ops.dropout_backward(dout, cache["output_dropout"])
        dattended, dweight_out, dbias_out = ops.linear_backward(
            dmerged, cache["out_proj"]
        )
        self._accumulate("self_attn.out_proj.weight", dweight_out)
        self._accumulate("self_attn.out_proj.bias", dbias_out)

        dattended_heads = (
            dattended.reshape(
                seq_len, batch_size, self.nhead, self.head_dim
            )
            .transpose(1, 2, 0, 3)
            .reshape(batch_size * self.nhead, seq_len, self.head_dim)
        )
        value_heads = cache["v_heads"]
        attention = cache["attn"]
        dropped_attention = cache["dropped_attn"]

        dattention = dattended_heads @ value_heads.transpose(0, 2, 1)
        dvalue_heads = dropped_attention.transpose(0, 2, 1) @ dattended_heads

        dattention = ops.dropout_backward(
            dattention, cache["attn_dropout"]
        )
        dscores = ops.softmax_backward(dattention, attention)
        scale = cache["scale"]
        dquery_heads = scale * (dscores @ cache["k_heads"])
        dkey_heads = scale * (dscores.transpose(0, 2, 1) @ cache["q_heads"])

        dquery = self._merge_heads(dquery_heads)
        dkey = self._merge_heads(dkey_heads)
        dvalue = self._merge_heads(dvalue_heads)

        dx_q, dweight_q, dbias_q = ops.linear_backward(dquery, cache["q"])
        dx_k, dweight_k, dbias_k = ops.linear_backward(dkey, cache["k"])
        dx_v, dweight_v, dbias_v = ops.linear_backward(dvalue, cache["v"])

        self._accumulate(
            "self_attn.in_proj_weight",
            np.concatenate([dweight_q, dweight_k, dweight_v], axis=0),
        )
        if "self_attn.in_proj_bias" in self.params:
            self._accumulate(
                "self_attn.in_proj_bias",
                np.concatenate([dbias_q, dbias_k, dbias_v], axis=0),
            )
        return dx_q + dx_k + dx_v

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, grad_output: Any) -> np.ndarray:
        r"""Backpropagate through the layer.

        Parameters
        ----------
        grad_output:
            :math:`\bar Y = \partial L/\partial Y`, shaped like the array
            returned by :meth:`forward`.

        Returns
        -------
        numpy.ndarray
            :math:`\bar X = \partial L/\partial X`, shaped like ``src``.
            Parameter gradients are accumulated in :attr:`grads` under the
            same keys as :attr:`params` (call :meth:`zero_grad` first when a
            fresh gradient is needed).

        Full reverse chain
        ------------------
        Let the post-norm intermediates be

        .. math::

            U_1 = X + \operatorname{SA}(X),\quad Z=\operatorname{LN}_1(U_1),

            U_2 = Z + \operatorname{FF}(Z),\quad Y=\operatorname{LN}_2(U_2),

        and the pre-norm intermediates be

        .. math::

            A_1=\operatorname{LN}_1(X),\quad
            U_1=X+\operatorname{SA}(A_1),

            A_2=\operatorname{LN}_2(U_1),\quad
            Y=U_1+\operatorname{FF}(A_2).

        For post-norm, with :math:`\bar U_2=\bar Y`:

        .. math::

            \bar U_2 = \operatorname{LN}_2'(\bar Y),\qquad
            \bar Z = \bar U_2 + \operatorname{FF}'(\bar U_2),

            \bar U_1 = \operatorname{LN}_1'(\bar Z),\qquad
            \bar X = \bar U_1 + \operatorname{SA}'(\bar U_1).

        For pre-norm:

        .. math::

            \bar U_1 = \bar Y
              + \operatorname{LN}_2'\!\left(\operatorname{FF}'(\bar Y)\right),

            \bar X = \bar U_1
              + \operatorname{LN}_1'\!\left(\operatorname{SA}'(\bar U_1)\right).

        Here :math:`\operatorname{LN}'`, :math:`\operatorname{SA}'` and
        :math:`\operatorname{FF}'` are the block backward passes documented
        in :meth:`_ln_backward`, :meth:`_sa_backward` and
        :meth:`_ff_backward`; each also accumulates its own parameter
        gradients.  The layer-norm input gradient is
        :func:`ops.layer_norm_backward`, whose docstring derives

        .. math::

            \bar x_i = \frac{1}{\sigma}\left[
                g_i-\frac1D\sum_j g_j
                -\frac{\hat x_i}{D}\sum_j g_j\hat x_j\right],
            \qquad g_i=\bar y_i\gamma_i.

        If ``batch_first`` is enabled the upstream gradient is transposed
        back to sequence-first before the chain starts; unbatched input is
        temporarily treated as ``N=1`` and squeezed at the end.
        """
        if self._cache is None:
            raise RuntimeError("backward() called before forward()")
        dy = _float_array(grad_output, "grad_output")
        output_shape = self._cache["output_shape"]
        if dy.shape != output_shape:
            raise RuntimeError(
                f"grad_output shape {dy.shape} != forward output shape "
                f"{output_shape}"
            )

        if self._cache["unbatched"]:
            dx = dy[:, None, :]
        elif self._cache["batch_first"]:
            dx = dy.transpose(1, 0, 2)
        else:
            dx = dy

        caches = self._cache["caches"]
        if self._cache["norm_first"]:
            d_hidden = dx + self._ln_backward(
                self._ff_backward(dx, caches["ff"]), caches["ln2"], "norm2"
            )
            dx = d_hidden + self._ln_backward(
                self._sa_backward(d_hidden, caches["sa"]),
                caches["ln1"],
                "norm1",
            )
        else:
            dx = self._ln_backward(dx, caches["ln2"], "norm2")
            dx = dx + self._ff_backward(dx, caches["ff"])
            dx = self._ln_backward(dx, caches["ln1"], "norm1")
            dx = dx + self._sa_backward(dx, caches["sa"])

        if self._cache["unbatched"]:
            return dx[:, 0, :]
        if self._cache["batch_first"]:
            return dx.transpose(1, 0, 2)
        return dx

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __call__(
        self,
        src: Any,
        src_mask: Any = None,
        src_key_padding_mask: Any = None,
        is_causal: bool = False,
    ) -> np.ndarray:
        return self.forward(src, src_mask, src_key_padding_mask, is_causal)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, nhead={self.nhead}, "
            f"dim_feedforward={self.dim_feedforward}, dropout={self.dropout}, "
            f"activation={self.activation!r}, "
            f"layer_norm_eps={self.layer_norm_eps}, "
            f"batch_first={self.batch_first}, norm_first={self.norm_first}, "
            f"bias={self.bias}"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.extra_repr()})"
