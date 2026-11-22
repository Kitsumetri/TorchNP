"""``Module`` — a lightweight analog of ``torch.nn.Module``.

Parameter/submodule registration, ``train/eval``, ``state_dict`` /
``load_state_dict``, ``zero_grad`` and dtype conversion. This is enough to
build torch-like layers and to load weights from torch.
"""

from __future__ import annotations

from collections import OrderedDict, namedtuple
from typing import Any, Callable, Iterator

import numpy as np

from ..tensor import Parameter, Tensor, default_dtype

_IncompatibleKeys = namedtuple("_IncompatibleKeys", ["missing_keys", "unexpected_keys"])


def resolve_dtype(dtype: Any | None) -> np.dtype:
    return np.dtype(dtype) if dtype is not None else default_dtype()


def _value_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.data
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return np.asarray(value)
    # torch.Tensor or a similar object
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class Module:
    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "training", True)

    # -- registration -----------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        params = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")
        if params is not None:
            if isinstance(value, Parameter):
                params[name] = value
            elif name in params:
                del params[name]
        if modules is not None:
            if isinstance(value, Module):
                modules[name] = value
            elif name in modules:
                del modules[name]
        object.__setattr__(self, name, value)

    def register_parameter(self, name: str, param: Parameter | None) -> None:
        if not isinstance(name, str):
            raise TypeError("parameter name must be a string")
        if "." in name:
            raise KeyError("parameter name cannot contain '.'")
        if param is None:
            self._parameters.pop(name, None)
            object.__setattr__(self, name, None)
            return
        if not isinstance(param, Parameter):
            raise TypeError("register_parameter expects Parameter or None")
        setattr(self, name, param)

    def register_module(self, name: str, module: "Module | None") -> None:
        if not isinstance(name, str):
            raise TypeError("module name must be a string")
        if module is None:
            self._modules.pop(name, None)
            object.__setattr__(self, name, None)
            return
        if not isinstance(module, Module):
            raise TypeError("register_module expects Module or None")
        setattr(self, name, module)

    add_module = register_module

    # -- traversal --------------------------------------------------------

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[tuple[str, Parameter]]:
        memo: set[int] = set()

        def walk(module: "Module", pref: str) -> Iterator[tuple[str, Parameter]]:
            for name, param in module._parameters.items():
                if param is None or id(param) in memo:
                    continue
                memo.add(id(param))
                yield pref + name, param
            if recurse:
                for name, child in module._modules.items():
                    yield from walk(child, pref + name + ".")

        yield from walk(self, prefix)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, "Module"]]:
        yield prefix, self
        for name, child in self._modules.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from child.named_modules(child_prefix)

    def modules(self) -> Iterator["Module"]:
        for _, module in self.named_modules():
            yield module

    # -- mode -------------------------------------------------------------

    def train(self, mode: bool = True) -> "Module":
        self.training = bool(mode)
        for child in self._modules.values():
            child.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def requires_grad_(self, requires_grad: bool = True) -> "Module":
        for param in self.parameters():
            param.requires_grad_(requires_grad)
        return self

    # -- gradients --------------------------------------------------------

    def zero_grad(self, set_to_none: bool = True) -> None:
        for param in self.parameters():
            param.zero_grad(set_to_none=set_to_none)

    # -- dtype ------------------------------------------------------------

    def _apply(self, fn: Callable[[np.ndarray], np.ndarray]) -> "Module":
        for param in self.parameters():
            param.data = fn(param.data)
            if param._grad is not None:
                param._grad = fn(param._grad)
        return self

    def to(self, *args, dtype: Any | None = None, device: Any | None = None) -> "Module":
        """Move/cast like ``Module.to`` (device first, dtype second)."""
        if len(args) > 2:
            raise TypeError("to() takes at most two positional arguments")
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, str) and arg.startswith("torch."):
                dtype = arg
            elif isinstance(arg, str) or (
                hasattr(arg, "type") and not isinstance(arg, (np.dtype, type))
            ):
                device = arg
            else:
                dtype = arg
        elif len(args) == 2:
            device, dtype = args

        if device is not None:
            device_name = getattr(device, "type", None) or str(device)
            if str(device_name).split(":")[0] != "cpu":
                raise ValueError("nptorch is CPU-only")
        if dtype is not None:
            if str(dtype).startswith("torch."):
                dtype = np.dtype(str(dtype).split(".")[-1])
            dt = np.dtype(dtype)
            if dt.kind not in "fc":
                raise TypeError(
                    f"to() only supports floating point dtypes, got {dt}"
                )
            self._apply(lambda arr: arr.astype(dt, copy=False))
        return self

    def float(self) -> "Module":
        return self.to(np.float32)

    def double(self) -> "Module":
        return self.to(np.float64)

    def cpu(self) -> "Module":
        return self

    # -- state dict -------------------------------------------------------

    def state_dict(
        self,
        destination: OrderedDict | None = None,
        prefix: str = "",
    ) -> OrderedDict:
        if destination is None:
            destination = OrderedDict()
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param.data.copy()
        for name, child in self._modules.items():
            child.state_dict(destination, prefix + name + ".")
        return destination

    def load_state_dict(
        self, state_dict: dict[str, Any], strict: bool = True
    ) -> _IncompatibleKeys:
        missing: list[str] = []
        unexpected = set(state_dict.keys())
        for name, param in self.named_parameters():
            if name not in state_dict:
                missing.append(name)
                continue
            array = _value_to_numpy(state_dict[name])
            if tuple(array.shape) != param.shape:
                raise RuntimeError(
                    f"size mismatch for {name}: state_dict has {tuple(array.shape)}, "
                    f"module has {param.shape}"
                )
            param.data = np.array(array, dtype=param.dtype, copy=True)
            unexpected.discard(name)
        if strict and (missing or unexpected):
            raise RuntimeError(
                "load_state_dict: "
                f"missing_keys={missing}, unexpected_keys={sorted(unexpected)}"
            )
        return _IncompatibleKeys(missing, sorted(unexpected))

    # -- call -------------------------------------------------------------

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        extra = self.extra_repr()
        lines = [f"{self.__class__.__name__}({extra})"]
        for name, child in self._modules.items():
            child_repr = repr(child).replace("\n", "\n  ")
            lines.append(f"  ({name}): {child_repr}")
        return "\n".join(lines)
