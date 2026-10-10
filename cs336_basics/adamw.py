import torch
from _collections_abc import Callable, Iterable
from typing import Optional
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, weight_decay: float, betas: tuple, eps: float,):
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups: # self.param_groups 是一个列表，里面每个元素是一个字典，代表一个"参数组"。
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            eps = group["eps"] 
            for p in group["params"] :
                if p.grad is None:
                    continue

                state = self.state[p]
                grad = p.grad
                if len(state) == 0: # 第一次见到这个参数时初始化 m, v（与 p 同形状）
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                t = state["t"]
                m = state["m"]
                v = state["v"]

                lr_t = lr * (math.sqrt(1 - betas[1] ** t)) / (1 - betas[0] ** t)
                p.mul_(1 - lr * weight_decay) # 解耦的权重衰减：θ ← θ - lr·λ·θ
                m.mul_(betas[0]).add_(grad, alpha=1 - betas[0]) # 原地更新，state 中的张量随之改变
                v.mul_(betas[1]).addcmul_(grad, grad, value=1 - betas[1])
                state["t"] = t + 1

                p.addcdiv_(m, v.sqrt().add_(eps), value=-lr_t) # θ ← θ - lr_t · m / (√v + eps)

        return loss