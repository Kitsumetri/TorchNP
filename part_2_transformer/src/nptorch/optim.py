"""Optimizers with torch-like ``zero_grad`` / ``step`` interfaces."""

from __future__ import annotations

from typing import Any

import numpy as np

from .nn.module import Module
from .tensor import Parameter, Tensor


def _as_param_list(params: Any) -> list[Parameter]:
    if isinstance(params, Module):
        return list(params.parameters())
    if isinstance(params, Tensor):
        raise TypeError(
            "params argument should be an iterable of Tensors, not a single Tensor"
        )
    return list(params)


class Optimizer:
    def __init__(self, params: Any, defaults: dict[str, Any]):
        self.defaults = dict(defaults)
        param_list = _as_param_list(params)
        self.param_groups = [{"params": param_list, **self.defaults}]
        self.state: dict[Parameter, dict[str, Any]] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                if param._grad is not None:
                    if set_to_none:
                        param._grad = None
                    else:
                        param._grad[...] = 0

    def step(self, closure=None):
        raise NotImplementedError

    def _add_state(self, param: Parameter) -> dict[str, Any]:
        if param not in self.state:
            self.state[param] = {}
        return self.state[param]


class SGD(Optimizer):
    def __init__(
        self,
        params: Any,
        lr: float,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        maximize: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("nesterov requires momentum > 0 and dampening == 0")
        if maximize:
            raise NotImplementedError("maximize=True is not implemented")
        super().__init__(
            params,
            dict(
                lr=float(lr),
                momentum=float(momentum),
                dampening=float(dampening),
                weight_decay=float(weight_decay),
                nesterov=bool(nesterov),
            ),
        )

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            for param in group["params"]:
                if param._grad is None:
                    continue
                grad = param._grad.copy()
                if weight_decay != 0:
                    grad = grad + weight_decay * param.data
                if momentum != 0:
                    state = self._add_state(param)
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = grad.copy()
                        state["momentum_buffer"] = buf
                    else:
                        buf = momentum * buf + (1.0 - dampening) * grad
                        state["momentum_buffer"] = buf
                    grad = grad + momentum * buf if nesterov else buf
                param.data -= lr * grad
        return loss


class Adam(Optimizer):
    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        maximize: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid beta values: {betas}")
        if eps < 0.0:
            raise ValueError(f"invalid epsilon value: {eps}")
        if amsgrad:
            raise NotImplementedError("amsgrad=True is not implemented")
        if maximize:
            raise NotImplementedError("maximize=True is not implemented")
        super().__init__(
            params,
            dict(
                lr=float(lr),
                betas=(float(betas[0]), float(betas[1])),
                eps=float(eps),
                weight_decay=float(weight_decay),
            ),
        )

    def _decoupled_weight_decay(self) -> bool:
        return False

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param._grad is None:
                    continue
                grad = param._grad.copy()
                if weight_decay != 0:
                    if self._decoupled_weight_decay():
                        param.data -= lr * weight_decay * param.data
                    else:
                        grad = grad + weight_decay * param.data
                state = self._add_state(param)
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = np.zeros_like(param.data)
                    state["exp_avg_sq"] = np.zeros_like(param.data)
                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad
                exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad
                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq
                step = state["step"]
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                step_size = lr / bias_correction1
                denom = np.sqrt(exp_avg_sq) / np.sqrt(bias_correction2) + eps
                param.data -= step_size * exp_avg / denom
        return loss


class AdamW(Adam):
    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        maximize: bool = False,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
        )

    def _decoupled_weight_decay(self) -> bool:
        return True
