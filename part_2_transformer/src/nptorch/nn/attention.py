"""Multi-head attention with torch-compatible parameters and semantics.

The slow (manual) path is implemented deliberately: it matches
``torch.nn.functional.multi_head_attention_forward`` and gives us a fully
hand-written backward graph, including masks and attention-weight outputs.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .. import functional as F
from .. import init
from ..tensor import Parameter, Tensor
from .linear import Linear
from .module import Module, resolve_dtype


def _mask_array(mask: Any) -> np.ndarray:
    return mask.data if isinstance(mask, Tensor) else np.asarray(mask)


def _check_mask(mask: np.ndarray, name: str) -> np.ndarray:
    if mask.dtype == bool:
        return mask
    if mask.dtype.kind == "f":
        return mask
    raise TypeError(f"{name} must be bool or floating point, got dtype={mask.dtype}")


def _apply_bool_or_float_mask(scores: Tensor, mask: np.ndarray) -> Tensor:
    if mask.dtype == bool:
        return F.masked_fill(scores, np.broadcast_to(mask, scores.shape), -math.inf)
    additive = mask.astype(scores.dtype, copy=False)
    return scores + Tensor(additive)


class MultiheadAttention(Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int | None = None,
        vdim: int | None = None,
        batch_first: bool = False,
        device: Any | None = None,
        dtype: Any | None = None,
    ):
        super().__init__()
        if device is not None and str(device) not in ("cpu", "None"):
            raise ValueError("nptorch is CPU-only")
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                "embed_dim and num_heads must be greater than 0, "
                f"got embed_dim={embed_dim} and num_heads={num_heads}"
            )
        if embed_dim % num_heads != 0:
            raise AssertionError("embed_dim must be divisible by num_heads")
        if add_bias_kv:
            raise NotImplementedError("add_bias_kv=True is not implemented")
        if add_zero_attn:
            raise NotImplementedError("add_zero_attn=True is not implemented")
        if kdim is not None and kdim != embed_dim:
            raise NotImplementedError("kdim != embed_dim is not implemented")
        if vdim is not None and vdim != embed_dim:
            raise NotImplementedError("vdim != embed_dim is not implemented")

        self.embed_dim = int(embed_dim)
        self.kdim = self.embed_dim if kdim is None else int(kdim)
        self.vdim = self.embed_dim if vdim is None else int(vdim)
        self._qkv_same_embed_dim = True
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.batch_first = bool(batch_first)
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise AssertionError("embed_dim must be divisible by num_heads")

        param_dtype = resolve_dtype(dtype)
        self.in_proj_weight = Parameter(
            np.empty((3 * self.embed_dim, self.embed_dim), dtype=param_dtype)
        )
        if bias:
            self.in_proj_bias = Parameter(
                np.empty(3 * self.embed_dim, dtype=param_dtype)
            )
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = Linear(
            self.embed_dim, self.embed_dim, bias=bias, dtype=param_dtype
        )
        self.bias_k = None
        self.bias_v = None
        self.add_zero_attn = False
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Matches torch.nn.MultiheadAttention._reset_parameters for the
        # qkv-same-embed-dim case.
        init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            init.constant_(self.in_proj_bias, 0.0)
            init.constant_(self.out_proj.bias, 0.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights: bool = True,
        attn_mask=None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        query = query if isinstance(query, Tensor) else Tensor(query)
        key = key if isinstance(key, Tensor) else Tensor(key)
        value = value if isinstance(value, Tensor) else Tensor(value)

        if is_causal and attn_mask is None:
            raise RuntimeError(
                "Need attn_mask if specifying the is_causal hint. "
                "Use torch.nn.Transformer.generate_square_subsequent_mask-like "
                "mask (an upper-triangular boolean mask) as a hint."
            )

        unbatched = query.ndim == 2
        if unbatched:
            if key.ndim != 2 or value.ndim != 2:
                raise RuntimeError(
                    "unbatched query requires unbatched key and value"
                )
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
        elif query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise RuntimeError(
                "query/key/value must be 2-D (L, E) or 3-D (L, N, E) / (N, L, E)"
            )

        if self.batch_first and not unbatched:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        tgt_len, bsz, embed_dim = query.shape
        src_len, key_bsz, _ = key.shape
        if embed_dim != self.embed_dim:
            raise RuntimeError(
                f"query embedding dim {embed_dim} != embed_dim {self.embed_dim}"
            )
        if key.shape != value.shape:
            raise RuntimeError(
                f"key shape {key.shape} does not match value shape {value.shape}"
            )
        if key_bsz != bsz:
            raise RuntimeError(
                f"key batch size {key_bsz} does not match query batch size {bsz}"
            )

        E = self.embed_dim
        H = self.num_heads
        head_dim = self.head_dim
        W = self.in_proj_weight
        b = self.in_proj_bias
        if b is None:
            bq = bk = bv = None
        else:
            bq, bk, bv = b[0:E], b[E : 2 * E], b[2 * E : 3 * E]
        q = F.linear(query, W[0:E], bq)
        k = F.linear(key, W[E : 2 * E], bk)
        v = F.linear(value, W[2 * E : 3 * E], bv)

        # (L, N, E) -> (N*H, L, head_dim)
        q = q.reshape(tgt_len, bsz * H, head_dim).transpose(0, 1)
        k = k.reshape(src_len, bsz * H, head_dim).transpose(0, 1)
        v = v.reshape(src_len, bsz * H, head_dim).transpose(0, 1)

        scale = math.sqrt(1.0 / float(head_dim))
        scores = F.matmul(q * scale, k.transpose(-2, -1))  # (N*H, L, S)

        # Match torch's slow-path handling of is_causal as a *hint*:
        # - need_weights=True: attn_mask is authoritative, is_causal is ignored;
        # - need_weights=False + no key padding: SDPA ignores attn_mask and uses
        #   the causal flag;
        # - need_weights=False + key padding: masks are merged, causal ignored.
        use_attn_mask = attn_mask
        use_causal = False
        if is_causal:
            if need_weights:
                use_causal = False
            elif key_padding_mask is None:
                use_attn_mask = None
                use_causal = True
            else:
                use_causal = False

        if use_attn_mask is not None:
            mask = _check_mask(_mask_array(use_attn_mask), "attn_mask")
            if mask.ndim == 2:
                expected = (tgt_len, src_len)
                if mask.shape != expected:
                    raise RuntimeError(
                        f"2-D attn_mask shape {mask.shape} != {expected}"
                    )
            elif mask.ndim == 3:
                if mask.shape != (bsz * H, tgt_len, src_len):
                    raise RuntimeError(
                        "3-D attn_mask must have shape "
                        f"({bsz * H}, {tgt_len}, {src_len}), got {mask.shape}"
                    )
            else:
                raise RuntimeError(
                    f"attn_mask must be 2-D or 3-D, got {mask.ndim}-D"
                )
            scores = _apply_bool_or_float_mask(scores, mask)

        if key_padding_mask is not None:
            kpm = _check_mask(_mask_array(key_padding_mask), "key_padding_mask")
            if unbatched:
                if kpm.ndim != 1 or kpm.shape != (src_len,):
                    raise RuntimeError(
                        f"key_padding_mask shape {kpm.shape} != {(src_len,)}"
                    )
                kpm = kpm[None, :]
            else:
                if kpm.ndim != 2 or kpm.shape != (bsz, src_len):
                    raise RuntimeError(
                        f"key_padding_mask shape {kpm.shape} != {(bsz, src_len)}"
                    )
            kpm_expanded = np.broadcast_to(
                kpm.reshape(bsz, 1, 1, src_len), (bsz, H, tgt_len, src_len)
            ).reshape(bsz * H, tgt_len, src_len)
            scores = _apply_bool_or_float_mask(scores, kpm_expanded)

        if use_causal:
            causal = np.triu(
                np.ones((tgt_len, src_len), dtype=bool), k=1
            )
            scores = F.masked_fill(scores, causal, -math.inf)

        # SDPA (used by torch when need_weights=False) returns zero attention
        # for rows whose keys are all masked instead of NaN. Mirror that.
        all_masked = np.all(np.isneginf(scores.data), axis=-1, keepdims=True)
        if not need_weights and np.any(all_masked):
            scores = F.masked_fill(scores, all_masked, 0.0)

        attn = F.softmax(scores, dim=-1)
        if not need_weights and np.any(all_masked):
            attn = F.masked_fill(attn, all_masked, 0.0)
        if self.training and self.dropout > 0.0:
            attn = F.dropout(attn, p=self.dropout, training=True)

        out = F.matmul(attn, v)  # (N*H, L, head_dim)
        out = out.transpose(0, 1).reshape(tgt_len, bsz, E)
        out = F.linear(out, self.out_proj.weight, self.out_proj.bias)

        if self.batch_first and not unbatched:
            out = out.transpose(0, 1)
        if unbatched:
            out = out.squeeze(1)

        if not need_weights:
            return out, None

        weights = attn.reshape(bsz, H, tgt_len, src_len)
        if average_attn_weights:
            weights = weights.mean(dim=1)
        if unbatched:
            weights = weights.squeeze(0)
        return out, weights

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"dropout={self.dropout}, batch_first={self.batch_first}"
        )
