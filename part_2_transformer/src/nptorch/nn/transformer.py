"""TransformerEncoderLayer and TransformerEncoder with torch-compatible APIs.

Default layer configuration matches
``torch.nn.TransformerEncoderLayer(d_model, nhead)``:
post-layer-norm, ReLU, dropout=0.1, bias=True, batch_first=False.

``TransformerEncoder`` mirrors ``torch.nn.TransformerEncoder``: cloned layers,
optional final norm, mask canonicalization and ``is_causal`` auto-detection.
Nested tensors do not exist in NumPy, so ``use_nested_tensor`` is always False.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any, Callable

import numpy as np

from .. import functional as F
from ..tensor import Tensor
from .activation import GELU, ReLU
from .attention import MultiheadAttention
from .container import ModuleList
from .dropout import Dropout
from .linear import Linear
from .module import Module
from .normalization import LayerNorm


def _mask_dtype(mask) -> np.dtype | None:
    if mask is None:
        return None
    return (mask.data if isinstance(mask, Tensor) else np.asarray(mask)).dtype


def _canonical_mask(
    mask,
    mask_name: str,
    target_dtype: np.dtype,
    other_type: np.dtype | None = None,
    check_other: bool = True,
):
    """Mirror ``torch.nn.functional._canonical_mask``.

    Boolean masks become additive masks with -inf at masked positions. Floating
    masks are kept as-is. Integer masks are rejected, as in torch.
    """
    if mask is None:
        return None
    array = mask.data if isinstance(mask, Tensor) else np.asarray(mask)
    is_float = array.dtype.kind == "f"
    if array.dtype != bool and not is_float:
        raise AssertionError(
            f"only bool and floating types of {mask_name} are supported"
        )
    if check_other and other_type is not None and array.dtype != np.dtype(other_type):
        warnings.warn(
            f"Support for mismatched {mask_name} and other mask types is "
            "deprecated. Use the same type for both instead.",
            stacklevel=2,
        )
    if not is_float:
        array = np.where(array, -np.inf, 0.0).astype(target_dtype, copy=False)
    return array


def _detect_is_causal_mask(mask, is_causal, size: int | None) -> bool:
    """Return whether ``mask`` is the standard upper-triangular causal mask."""
    if is_causal is not None:
        return is_causal is True
    if mask is None or size is None:
        return False
    array = mask.data if isinstance(mask, Tensor) else np.asarray(mask)
    if array.ndim != 2 or array.shape != (size, size):
        return False
    if array.dtype == bool:
        causal = np.triu(np.ones((size, size), dtype=bool), k=1)
    else:
        causal = np.triu(np.full((size, size), -np.inf, dtype=array.dtype), k=1)
    return bool(np.array_equal(array, causal))


def _get_seq_len(src: Tensor, batch_first: bool) -> int | None:
    if src.ndim == 2:
        return src.shape[0]
    if src.ndim >= 3:
        return src.shape[1] if batch_first else src.shape[0]
    return None


def _get_activation_fn(activation: str) -> Callable:
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu or a callable, not {activation!r}")



class TransformerEncoderLayer(Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device: Any | None = None,
        dtype: Any | None = None,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.dim_feedforward = int(dim_feedforward)
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
        # Feed-forward block.
        self.linear1 = Linear(
            d_model, dim_feedforward, bias=bias, device=device, dtype=dtype
        )
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(
            dim_feedforward, d_model, bias=bias, device=device, dtype=dtype
        )

        self.norm_first = bool(norm_first)
        self.norm1 = LayerNorm(
            d_model,
            eps=layer_norm_eps,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.norm2 = LayerNorm(
            d_model,
            eps=layer_norm_eps,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # Flags kept for parity with torch's fast-path checks.
        if activation is F.relu or isinstance(activation, ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    # ------------------------------------------------------------------
    # Blocks
    # ------------------------------------------------------------------

    def _sa_block(
        self,
        x: Tensor,
        attn_mask,
        key_padding_mask,
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
        is_causal: bool = False,
    ) -> Tensor:
        src = src if isinstance(src, Tensor) else Tensor(src)
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(
                    x, src_mask, src_key_padding_mask, is_causal=is_causal
                )
            )
            x = self.norm2(x + self._ff_block(x))
        return x

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, nhead={self.nhead}, "
            f"dim_feedforward={self.dim_feedforward}, dropout={self.dropout.p}, "
            f"activation={getattr(self.activation, '__name__', self.activation)}, "
            f"layer_norm_eps={self.norm1.eps}, batch_first={self.self_attn.batch_first}, "
            f"norm_first={self.norm_first}, bias={self.linear1.bias is not None}"
        )


class TransformerEncoder(Module):
    """Stack of cloned ``TransformerEncoderLayer`` modules.

    Mirrors ``torch.nn.TransformerEncoder``. Layers are deep-copied, so all of
    them start with identical parameters (torch does the same via
    ``copy.deepcopy``). ``enable_nested_tensor`` and ``mask_check`` are accepted
    for API compatibility; nested tensors are not implemented, therefore
    ``use_nested_tensor`` is always False.
    """

    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: Module | None = None,
        enable_nested_tensor: bool = True,
        mask_check: bool = True,
    ):
        super().__init__()
        self.layers = ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = bool(enable_nested_tensor)
        self.use_nested_tensor = False
        self.mask_check = bool(mask_check)

    def forward(
        self,
        src,
        mask=None,
        src_key_padding_mask=None,
        is_causal: bool | None = None,
    ) -> Tensor:
        src = src if isinstance(src, Tensor) else Tensor(src)

        # torch canonicalizes bool masks to additive -inf masks before the
        # layers see them; do the same so mask handling is identical.
        mask_type = _mask_dtype(mask)
        src_key_padding_mask = _canonical_mask(
            src_key_padding_mask,
            "src_key_padding_mask",
            src.dtype,
            other_type=mask_type,
        )
        mask = _canonical_mask(
            mask,
            "mask",
            src.dtype,
            other_type=None,
            check_other=False,
        )

        # torch reads layers[0] unconditionally, so num_layers=0 raises here.
        first_layer = self.layers[0]
        attention = getattr(first_layer, "self_attn", None)
        batch_first = bool(getattr(attention, "batch_first", False))
        seq_len = _get_seq_len(src, batch_first)
        make_causal = _detect_is_causal_mask(mask, is_causal, seq_len)

        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=make_causal,
            )

        if self.norm is not None:
            output = self.norm(output)
        return output

    def extra_repr(self) -> str:
        return (
            f"num_layers={self.num_layers}, "
            f"enable_nested_tensor={self.enable_nested_tensor}, "
            f"mask_check={self.mask_check}"
        )

