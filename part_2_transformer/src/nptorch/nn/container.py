"""Torch-like ``ModuleList`` container.

Modules are registered under string indices, so ``state_dict`` keys look like
``layers.0.self_attn.in_proj_weight`` exactly as in torch.
"""

from __future__ import annotations

import operator
from collections import OrderedDict
from collections.abc import Iterable as IterableABC
from itertools import chain
from typing import Iterable, Iterator, overload

from .module import Module


class ModuleList(Module):
    def __init__(self, modules: Iterable[Module] | None = None):
        super().__init__()
        if modules is not None:
            self.extend(modules)

    # -- helpers ----------------------------------------------------------

    def _get_abs_string_index(self, idx: int) -> str:
        idx = operator.index(idx)
        if not -len(self) <= idx < len(self):
            raise IndexError(f"index {idx} is out of range")
        if idx < 0:
            idx += len(self)
        return str(idx)

    def _rebuild(self, modules: Iterable[Module]) -> None:
        items = list(modules)
        for key in list(self.__dict__):
            if key.isdigit():
                object.__delattr__(self, key)
        object.__setattr__(
            self, "_modules", OrderedDict((str(i), m) for i, m in enumerate(items))
        )
        for i, module in enumerate(items):
            object.__setattr__(self, str(i), module)

    def _remove(self, key: str) -> None:
        self._modules.pop(key, None)
        if key in self.__dict__:
            object.__delattr__(self, key)

    # -- sequence protocol ------------------------------------------------

    @overload
    def __getitem__(self, idx: slice) -> "ModuleList": ...

    @overload
    def __getitem__(self, idx: int) -> Module: ...

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        return self._modules[self._get_abs_string_index(idx)]

    def __setitem__(self, idx: int, module: Module) -> None:
        if not isinstance(module, Module):
            raise TypeError(
                f"cannot assign {type(module).__name__} as a child module; "
                "expected Module"
            )
        idx = self._get_abs_string_index(idx)
        setattr(self, idx, module)

    def __delitem__(self, idx: int | slice) -> None:
        if isinstance(idx, slice):
            removed = set(range(len(self))[idx])
            keep = [
                module
                for i, module in enumerate(self._modules.values())
                if i not in removed
            ]
            self._rebuild(keep)
            return
        key = self._get_abs_string_index(idx)
        self._remove(key)
        self._rebuild(list(self._modules.values()))

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __iadd__(self, modules: Iterable[Module]) -> "ModuleList":
        return self.extend(modules)

    def __add__(self, other: Iterable[Module]) -> "ModuleList":
        combined = ModuleList()
        for module in chain(self, other):
            combined.append(module)
        return combined

    # -- mutation ---------------------------------------------------------

    def append(self, module: Module) -> "ModuleList":
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules: Iterable[Module]) -> "ModuleList":
        if not isinstance(modules, IterableABC):
            raise TypeError(
                "ModuleList.extend should be called with an iterable, "
                f"but got {type(modules).__name__}"
            )
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self

    def insert(self, index: int, module: Module) -> None:
        index = operator.index(index)
        n = len(self)
        if index < 0:
            index = max(0, n + index)
        if index > n:
            index = n
        items = list(self._modules.values())
        items.insert(index, module)
        self._rebuild(items)

    def pop(self, key: int | slice) -> Module:
        value = self[key]
        del self[key]
        return value

    def __repr__(self) -> str:
        if not self._modules:
            return "ModuleList()"
        lines = [f"  ({name}): {module!r}" for name, module in self._modules.items()]
        return "ModuleList(\n" + "\n".join(lines) + "\n)"
