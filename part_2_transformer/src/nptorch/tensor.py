"""Tensor and a minimal reverse-mode autograd engine.

A ``Tensor`` stores a ``numpy`` array and a ``_backward`` closure that
accumulates gradients into its parents. ``Tensor.backward()`` topologically
sorts the graph and runs the closures in reverse order. This is enough for
every operation used by the transformer.

The public surface mirrors ``torch.Tensor`` where practical: ``requires_grad``,
``grad``, ``backward()``, ``detach()``, ``no_grad()`` and the usual operators.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Global settings (torch-like default dtype and RNG)
# ---------------------------------------------------------------------------

_DEFAULT_DTYPE: np.dtype = np.dtype(np.float32)
_rng: np.random.Generator = np.random.default_rng()
_grad_enabled: bool = True


def default_dtype() -> np.dtype:
    """Current default dtype (like ``torch.get_default_dtype()``)."""
    return _DEFAULT_DTYPE


def set_default_dtype(dtype: Any) -> None:
    """Set the dtype used for new tensors and parameters."""
    global _DEFAULT_DTYPE
    _DEFAULT_DTYPE = np.dtype(dtype)


def manual_seed(seed: int) -> None:
    """Seed the RNG shared by initializers and dropout."""
    global _rng
    _rng = np.random.default_rng(seed)


def get_rng() -> np.random.Generator:
    """Internal access to the RNG (used by init/dropout)."""
    return _rng


class _GradMode:
    """Base context manager that toggles graph construction."""

    def __init__(self, enabled: bool):
        self._enabled = enabled
        self._prev: bool | None = None

    def __enter__(self):
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = self._enabled
        return self

    def __exit__(self, exc_type, exc, tb):
        global _grad_enabled
        assert self._prev is not None
        _grad_enabled = self._prev
        self._prev = None
        return False

    def __call__(self, func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with _GradMode(self._enabled):
                return func(*args, **kwargs)

        return wrapper


class no_grad(_GradMode):
    """Disable graph construction: ``with no_grad(): ...``."""

    def __init__(self):
        super().__init__(False)


class enable_grad(_GradMode):
    """Enable graph construction (also inside an outer ``no_grad``)."""

    def __init__(self):
        super().__init__(True)


def is_grad_enabled() -> bool:
    return _grad_enabled


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_numpy(data: Any, dtype: Any | None = None) -> np.ndarray:
    """Convert input to ndarray with torch-like rules for lists/scalars."""
    if isinstance(data, Tensor):
        arr = data.data
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr
    if isinstance(data, (np.ndarray, np.generic)):
        arr = np.asarray(data)
        return arr if dtype is None else arr.astype(dtype, copy=False)
    arr = np.asarray(data)
    if dtype is not None:
        return arr.astype(dtype, copy=False)
    # Python floats and float lists use the default dtype (torch: float32).
    if arr.dtype.kind == "f":
        return arr.astype(_DEFAULT_DTYPE, copy=False)
    return arr


def _make_result(
    data: Any,
    parents: Sequence["Tensor"],
    backward: Callable[[], None],
    op: str,
) -> "Tensor":
    """Build an op result and attach it to the graph when needed."""
    out = Tensor(np.asarray(data))
    if _grad_enabled:
        needs_grad = [p for p in parents if p.requires_grad]
        if needs_grad:
            out.requires_grad = True
            out._prev = tuple(needs_grad)
            out._backward = backward
            out._is_leaf = False
            out._op = op
    return out


def _add_grad(tensor: "Tensor", grad: Any) -> None:
    """Accumulate a gradient into ``tensor._grad`` safely."""
    g = np.asarray(grad)
    if g.shape != tensor.data.shape:
        raise RuntimeError(
            f"grad shape {g.shape} does not match tensor shape "
            f"{tensor.data.shape} for op {tensor._op!r}"
        )
    if g.dtype != tensor.data.dtype:
        g = g.astype(tensor.data.dtype, copy=False)
    if tensor._grad is None:
        tensor._grad = np.array(g, copy=True)
    else:
        tensor._grad = tensor._grad + g


class Tensor:
    """Numpy tensor with reverse-mode autodiff.

    ``grad`` is exposed as a ``Tensor`` (like in torch) but stored internally
    as an ndarray.
    """

    __array_priority__ = 1000

    def __init__(
        self,
        data: Any,
        requires_grad: bool = False,
        dtype: Any | None = None,
    ):
        arr = _to_numpy(data, dtype)
        if requires_grad and arr.dtype.kind != "f":
            raise RuntimeError(
                "requires_grad=True is only supported for floating point "
                f"tensors, got dtype={arr.dtype}"
            )
        self.data: np.ndarray = arr
        self.requires_grad: bool = bool(requires_grad)
        self._grad: np.ndarray | None = None
        self._prev: tuple[Tensor, ...] = ()
        self._backward: Callable[[], None] = lambda: None
        self._is_leaf: bool = True
        self._op: str = "leaf"

    # -- properties -------------------------------------------------------

    @property
    def grad(self) -> "Tensor | None":
        if self._grad is None:
            return None
        return Tensor(self._grad)

    @grad.setter
    def grad(self, value: Any) -> None:
        if value is None:
            self._grad = None
        elif isinstance(value, Tensor):
            self._grad = value.data
        else:
            self._grad = np.asarray(value)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def is_leaf(self) -> bool:
        return self._is_leaf

    @property
    def is_cuda(self) -> bool:
        return False

    @property
    def T(self) -> "Tensor":
        return self.permute(*reversed(range(self.ndim)))

    # -- conversion / copying ---------------------------------------------

    def dim(self) -> int:
        return self.ndim

    def size(self, dim: int | None = None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    def numel(self) -> int:
        return int(self.data.size)

    def item(self) -> float:
        if self.data.size != 1:
            raise RuntimeError("item() is only defined for tensors with one element")
        return self.data.item()

    def numpy(self) -> np.ndarray:
        return self.data

    def tolist(self) -> Any:
        return self.data.tolist()

    def detach(self) -> "Tensor":
        return Tensor(self.data, requires_grad=False)

    def clone(self) -> "Tensor":
        from . import functional as F

        return F.clone(self)

    def to(self, dtype: Any) -> "Tensor":
        from . import functional as F

        dtype = np.dtype(dtype)
        if dtype == self.dtype:
            return self
        return F.cast(self, dtype)

    def float(self) -> "Tensor":
        return self.to(np.float32)

    def double(self) -> "Tensor":
        return self.to(np.float64)

    def contiguous(self) -> "Tensor":
        # Numpy arrays do not require contiguous storage for indexing.
        return self

    def requires_grad_(self, requires_grad: bool = True) -> "Tensor":
        if not self._is_leaf:
            raise RuntimeError(
                "you can only change requires_grad flags of leaf variables"
            )
        if requires_grad and self.dtype.kind != "f":
            raise RuntimeError("requires_grad=True is only valid for float tensors")
        self.requires_grad = bool(requires_grad)
        return self

    # -- autograd ---------------------------------------------------------

    def zero_grad(self, set_to_none: bool = True) -> None:
        if set_to_none:
            self._grad = None
        elif self._grad is None:
            self._grad = np.zeros_like(self.data)
        else:
            self._grad[...] = 0

    def backward(self, gradient: Any | None = None) -> None:
        """Run the backward pass starting from this tensor."""
        if not self.requires_grad:
            raise RuntimeError("backward() called on a tensor without requires_grad")
        if gradient is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "grad can be omitted only for a scalar output; "
                    f"got shape={self.shape}"
                )
            seed = np.ones_like(self.data)
        elif isinstance(gradient, Tensor):
            seed = gradient.data
        else:
            seed = np.asarray(gradient)
        if seed.shape != self.data.shape:
            raise RuntimeError(
                f"grad shape {seed.shape} does not match output shape {self.shape}"
            )
        seed = seed.astype(self.data.dtype, copy=False)

        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: Tensor) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node._prev:
                build(parent)
            topo.append(node)

        build(self)
        # Only leaves accumulate grads across backward() calls, like torch.
        # Intermediate nodes are recomputed from scratch on every pass.
        for node in topo:
            if not node._is_leaf:
                node._grad = None
        if self._grad is None:
            self._grad = np.array(seed, copy=True)
        else:
            self._grad = self._grad + seed

        for node in reversed(topo):
            if node._grad is not None:
                node._backward()
        # torch keeps .grad only on leaves (unless retain_grad is used).
        for node in topo:
            if not node._is_leaf:
                node._grad = None

    # -- operations (implemented in functional.py) ------------------------

    def sum(self, dim: int | Sequence[int] | None = None, keepdim: bool = False):
        from . import functional as F

        return F.sum(self, dim=dim, keepdim=keepdim)

    def mean(self, dim: int | Sequence[int] | None = None, keepdim: bool = False):
        from . import functional as F

        return F.mean(self, dim=dim, keepdim=keepdim)

    def reshape(self, *shape: Any):
        from . import functional as F

        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return F.reshape(self, tuple(shape))

    def view(self, *shape: Any):
        return self.reshape(*shape)

    def transpose(self, dim0: int, dim1: int):
        from . import functional as F

        return F.transpose(self, dim0, dim1)

    def permute(self, *dims: int):
        from . import functional as F

        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        return F.permute(self, tuple(dims))

    def unsqueeze(self, dim: int):
        from . import functional as F

        return F.unsqueeze(self, dim)

    def squeeze(self, dim: int | None = None):
        from . import functional as F

        return F.squeeze(self, dim)

    def softmax(self, dim: int = -1):
        from . import functional as F

        return F.softmax(self, dim=dim)

    def log_softmax(self, dim: int = -1):
        from . import functional as F

        return F.log_softmax(self, dim=dim)

    def masked_fill(self, mask: Any, value: float):
        from . import functional as F

        return F.masked_fill(self, mask, value)

    def exp(self):
        from . import functional as F

        return F.exp(self)

    def log(self):
        from . import functional as F

        return F.log(self)

    def sqrt(self):
        from . import functional as F

        return F.sqrt(self)

    def abs(self):
        from . import functional as F

        return F.abs(self)

    # -- dunder operators -------------------------------------------------

    def __add__(self, other):
        from . import functional as F

        return F.add(self, other)

    def __radd__(self, other):
        from . import functional as F

        return F.add(other, self)

    def __sub__(self, other):
        from . import functional as F

        return F.sub(self, other)

    def __rsub__(self, other):
        from . import functional as F

        return F.sub(other, self)

    def __mul__(self, other):
        from . import functional as F

        return F.mul(self, other)

    def __rmul__(self, other):
        from . import functional as F

        return F.mul(other, self)

    def __truediv__(self, other):
        from . import functional as F

        return F.div(self, other)

    def __rtruediv__(self, other):
        from . import functional as F

        return F.div(other, self)

    def __neg__(self):
        from . import functional as F

        return F.neg(self)

    def __abs__(self):
        from . import functional as F

        return F.abs(self)

    def __pow__(self, other):
        from . import functional as F

        return F.pow(self, other)

    def __rpow__(self, other):
        from . import functional as F

        return F.pow(other, self)

    def __matmul__(self, other):
        from . import functional as F

        return F.matmul(self, other)

    def __rmatmul__(self, other):
        from . import functional as F

        return F.matmul(other, self)

    def __getitem__(self, index):
        from . import functional as F

        return F.getitem(self, index)

    def __len__(self) -> int:
        return len(self.data)

    def __float__(self) -> float:
        return float(self.item())

    def __bool__(self) -> bool:
        raise ValueError("bool(Tensor) is ambiguous, same as in torch")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(shape={self.shape}, dtype={self.dtype}, "
            f"requires_grad={self.requires_grad})"
        )


class Parameter(Tensor):
    """Trainable tensor (analog of ``torch.nn.Parameter``)."""

    def __init__(self, data: Any, requires_grad: bool = True):
        super().__init__(data, requires_grad=requires_grad)
