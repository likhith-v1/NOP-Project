"""
Lipschitz-Bounded Momentum (LBM) Optimizer

Update rule:
    L_t  = σ_max(∇²f(θ_t))               via secant approximation
    β_t  = clip(1 - 1/√(L_t + ε), β_min, β_max)
    lr_t = α / (1 + λ_t)                  Levenberg-Marquardt damping
    y_t  = θ_t + β_t(θ_t - θ_{t-1})      Nesterov lookahead
    θ_{t+1} = y_t - lr_t · ∇f(y_t)
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import List, Optional, Dict, Callable
import math


class LipschitzMomentumOptimizer(Optimizer):
    """
    Lipschitz-Bounded Momentum Optimizer.
    Dynamic β_t via Hessian spectral norm + Levenberg-Marquardt damping + Nesterov update.
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        beta_min: float = 0.85,
        beta_max: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        power_iter_steps: int = 5,
        hvp_epsilon: float = 1e-3,
        per_layer: bool = True,
        lm_lambda_init: float = 1e-3,
        lm_lambda_up: float = 10.0,
        lm_lambda_down: float = 3.0,
        lm_lambda_min: float = 1e-6,
        lm_lambda_max: float = 1e2,
    ):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= beta_min < beta_max <= 1.0):
            raise ValueError(f"Invalid beta bounds: [{beta_min}, {beta_max}]")

        defaults = dict(
            lr=lr,
            beta_min=beta_min,
            beta_max=beta_max,
            eps=eps,
            weight_decay=weight_decay,
            power_iter_steps=power_iter_steps,
            hvp_epsilon=hvp_epsilon,
            per_layer=per_layer,
            lm_lambda_init=lm_lambda_init,
            lm_lambda_up=lm_lambda_up,
            lm_lambda_down=lm_lambda_down,
            lm_lambda_min=lm_lambda_min,
            lm_lambda_max=lm_lambda_max,
        )
        super().__init__(params, defaults)

        # Global LM damping state (shared across param groups)
        self._lm_lambda    = lm_lambda_init
        self._prev_loss    = None
        self._step_count   = 0

        # Trajectory logging (for report plots)
        self.beta_log: List[float]       = []   # mean β_t per step
        self.lipschitz_log: List[float]  = []   # mean L_t per step
        self.lambda_log: List[float]     = []   # λ_t per step

    # ── Hessian-vector product via double backprop ──────────────────────

    @staticmethod
    def _hvp(
        loss: torch.Tensor,
        params: List[torch.Tensor],
        vector: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Computes H·v = ∇(∇f · v) via double backprop."""
        grads = torch.autograd.grad(
            loss, params, create_graph=True, retain_graph=True, allow_unused=True
        )
        grads = [g if g is not None else torch.zeros_like(p)
                 for g, p in zip(grads, params)]

        grad_dot_v = sum((g * v).sum() for g, v in zip(grads, vector))

        hvp = torch.autograd.grad(
            grad_dot_v, params, retain_graph=True, allow_unused=True
        )
        return [h.detach() if h is not None else torch.zeros_like(p)
                for h, p in zip(hvp, params)]

    # ── Spectral norm via power iteration ──────────────────────────────

    def _estimate_spectral_norm(
        self,
        loss: torch.Tensor,
        params: List[torch.Tensor],
        num_iters: int,
    ) -> float:
        """Estimates σ_max(H) via Rayleigh quotient power iteration."""
        v = [torch.randn_like(p) for p in params]
        v_norm = math.sqrt(sum((vi ** 2).sum().item() for vi in v))
        v = [vi / (v_norm + 1e-10) for vi in v]

        for _ in range(num_iters):
            hv = self._hvp(loss, params, v)
            hv_norm = math.sqrt(sum((h ** 2).sum().item() for h in hv))
            if hv_norm < 1e-10:
                return 1.0
            v = [h / hv_norm for h in hv]

        hv = self._hvp(loss, params, v)
        rayleigh = sum((vi * hi).sum().item() for vi, hi in zip(v, hv))
        return max(abs(rayleigh), 1e-8)

    # ── Dynamic β from Lipschitz constant ──────────────────────────────

    @staticmethod
    def _compute_beta(L_t: float, beta_min: float, beta_max: float, eps: float) -> float:
        """β_t = clip(1 - 1/√(L_t + ε), β_min, β_max)"""
        raw = 1.0 - 1.0 / math.sqrt(L_t + eps)
        return float(max(beta_min, min(beta_max, raw)))

    # ── Levenberg-Marquardt damping ─────────────────────────────────────

    def _update_lm_damping(self, current_loss: float, group: dict) -> float:
        """Adjusts λ based on loss change. Returns effective_lr = α / (1 + λ)."""
        if self._prev_loss is not None:
            if current_loss > self._prev_loss:
                self._lm_lambda = min(self._lm_lambda * group["lm_lambda_up"],
                                      group["lm_lambda_max"])
            else:
                self._lm_lambda = max(self._lm_lambda / group["lm_lambda_down"],
                                      group["lm_lambda_min"])
        return group["lr"] / (1.0 + self._lm_lambda)

    # ── Main step ───────────────────────────────────────────────────────

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable] = None,
        current_loss: Optional[float] = None,
    ) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        loss_val = current_loss if current_loss is not None else (
            loss.item() if loss is not None else None
        )

        step_betas, step_lipschitz = [], []

        for group in self.param_groups:
            params_with_grad = [p for p in group["params"]
                                if p.grad is not None and p.requires_grad]
            if not params_with_grad:
                continue

            effective_lr = (self._update_lm_damping(loss_val, group)
                            if loss_val is not None else group["lr"])

            L_t    = self._estimate_lipschitz_from_grads(params_with_grad, group)
            beta_t = self._compute_beta(L_t, group["beta_min"],
                                        group["beta_max"], group["eps"])
            step_betas.append(beta_t)
            step_lipschitz.append(L_t)

            for p in params_with_grad:
                d_p = p.grad
                if group["weight_decay"] != 0:
                    d_p = d_p.add(p, alpha=group["weight_decay"])

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state["prev_p"]          = p.clone()

                buf = state["momentum_buffer"]
                buf.mul_(beta_t).add_(d_p)
                p.add_(buf, alpha=-effective_lr * beta_t)
                p.add_(d_p,  alpha=-effective_lr)

        if step_betas:
            self.beta_log.append(sum(step_betas) / len(step_betas))
        if step_lipschitz:
            self.lipschitz_log.append(sum(step_lipschitz) / len(step_lipschitz))
        self.lambda_log.append(self._lm_lambda)

        if loss_val is not None:
            self._prev_loss = loss_val

        return loss

    # ── Secant-based Lipschitz estimate (fast, per-step default) ────────

    def _estimate_lipschitz_from_grads(
        self,
        params: List[torch.Tensor],
        group: dict,
    ) -> float:
        """L_t ≈ ||∇f(θ_t) - ∇f(θ_{t-1})|| / ||θ_t - θ_{t-1}|| (secant approx)."""
        current_grad_norm_sq = sum(
            (p.grad ** 2).sum().item() for p in params if p.grad is not None
        )

        if self._step_count <= 1:
            return max(math.sqrt(current_grad_norm_sq), 1.0)

        grad_diff_norm_sq  = 0.0
        param_diff_norm_sq = 0.0

        for p in params:
            if p.grad is None:
                continue
            state = self.state[p]
            if "prev_grad" in state and "prev_p" in state:
                grad_diff_norm_sq  += ((p.grad - state["prev_grad"]) ** 2).sum().item()
                param_diff_norm_sq += ((p.data  - state["prev_p"])   ** 2).sum().item()
            state["prev_grad"] = p.grad.clone()
            state["prev_p"]    = p.data.clone()

        if param_diff_norm_sq < 1e-12:
            return max(math.sqrt(current_grad_norm_sq), 1.0)

        return max(math.sqrt(grad_diff_norm_sq / param_diff_norm_sq), 1e-4)

    # ── Full HVP step (exact, call every N steps from train.py) ────────

    def step_with_hvp(self, loss: torch.Tensor, model: nn.Module) -> None:
        """Exact spectral norm via power iteration. ~5x slower, use every N steps."""
        for group in self.param_groups:
            params_with_grad = [p for p in group["params"] if p.requires_grad]
            if not params_with_grad:
                continue

            with torch.enable_grad():
                L_t = self._estimate_spectral_norm(
                    loss, params_with_grad, group["power_iter_steps"]
                )

            beta_t       = self._compute_beta(L_t, group["beta_min"],
                                              group["beta_max"], group["eps"])
            loss_val     = loss.item()
            effective_lr = self._update_lm_damping(loss_val, group)

            with torch.no_grad():
                for p in params_with_grad:
                    if p.grad is None:
                        continue
                    d_p = p.grad.clone()
                    if group["weight_decay"] != 0:
                        d_p.add_(p, alpha=group["weight_decay"])
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    buf.mul_(beta_t).add_(d_p)
                    p.add_(buf, alpha=-effective_lr * beta_t)
                    p.add_(d_p, alpha=-effective_lr)

        self.beta_log.append(beta_t)
        self.lipschitz_log.append(L_t)
        self.lambda_log.append(self._lm_lambda)
        self._prev_loss  = loss_val
        self._step_count += 1

    # ── Introspection ───────────────────────────────────────────────────

    def get_current_beta(self) -> float:
        return self.beta_log[-1] if self.beta_log else float("nan")

    def get_current_lipschitz(self) -> float:
        return self.lipschitz_log[-1] if self.lipschitz_log else float("nan")

    def get_current_lambda(self) -> float:
        return self._lm_lambda

    def print_state(self) -> None:
        print(f"  [LBM] step={self._step_count:>6} | "
              f"β_t={self.get_current_beta():.4f} | "
              f"L_t={self.get_current_lipschitz():.4f} | "
              f"λ={self.get_current_lambda():.2e}")