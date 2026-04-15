"""Exponential Moving Average of model weights."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMA:
    """Maintains an EMA copy of a model's parameters.

    Call ema.update(model) after every optimizer step.
    Use ema.ema_model for validation and inference.

    Args:
        model: the live model whose weights to shadow.
        beta:  EMA decay (default 0.9999 — very slow, standard for diffusion).
        update_every: number of steps between EMA updates.
    """

    def __init__(self, model: nn.Module, beta: float = 0.9999, update_every: int = 1) -> None:
        self.beta = beta
        self.update_every = update_every
        self._step = 0
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA weights from the current model state."""
        self._step += 1
        if self._step % self.update_every != 0:
            return

        for ema_p, live_p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.lerp_(live_p.data, 1.0 - self.beta)

        for ema_b, live_b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.copy_(live_b)

    def state_dict(self) -> dict:
        return {
            "ema_model": self.ema_model.state_dict(),
            "step": self._step,
            "beta": self.beta,
        }

    def load_state_dict(self, state: dict) -> None:
        self.ema_model.load_state_dict(state["ema_model"])
        self._step = state.get("step", 0)
        self.beta = state.get("beta", self.beta)
