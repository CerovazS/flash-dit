"""Muon optimizer (inline Newton-Schulz) + AdamW hybrid builder.

Muon is applied only to hidden 2-D projection matrices inside transformer
blocks. AdamW handles embeddings, norms, biases, AdaLN control tensors, final
head, and other non-Muon parameters.

Reference:
  Keller Jordan, "Muon: An optimizer for hidden layers in neural networks"
  https://kellerjordan.github.io/posts/muon/
"""
from __future__ import annotations

import torch
from torch.optim import AdamW


class Muon(torch.optim.Optimizer):
    """MomentUm Orthogonalized by Newton-Schulz.

    Applies SGD momentum to 2-D weight matrices and orthogonalises
    the update via 5 Newton-Schulz iterations before applying.

    Args:
        params:       iterable of 2-D parameters (or param dicts).
        lr:           learning rate (default 0.02 — typically higher than AdamW).
        momentum:     momentum coefficient (default 0.95).
        ns_steps:     Newton-Schulz iteration count (5 is sufficient).
        weight_decay: L2 regularisation applied before momentum (default 0).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @staticmethod
    def _newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
        """Compute an approximate orthogonalisation of G via Newton-Schulz."""
        assert G.ndim == 2
        a, b, c = 3.4445, -4.7750, 2.0315

        X = G.to(torch.bfloat16)
        X = X / (X.norm() + 1e-7)

        if X.size(0) > X.size(1):
            X = X.T

        for _ in range(steps):
            A = X @ X.T
            X = a * X + (b * A + c * (A @ A)) @ X

        if G.size(0) > G.size(1):
            X = X.T

        return X.to(G.dtype)

    @torch.no_grad()
    def step(self, closure=None) -> None:  # type: ignore[override]
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data

                if wd != 0.0:
                    g = g.add(p.data, alpha=wd)

                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = g.clone()
                else:
                    state["buf"].mul_(momentum).add_(g, alpha=1.0 - momentum)

                g = state["buf"]

                if g.ndim == 2:
                    g_orth = self._newtonschulz5(g, ns_steps)
                    # Muon reference scale correction: rows > cols need larger
                    # updates after orthogonalization; wide matrices do not.
                    scale = float(max(1.0, g.size(0) / g.size(1))) ** 0.5
                    p.data.add_(g_orth, alpha=-lr * scale)
                else:
                    # Fallback for non-2D params (should not reach here if split correctly)
                    p.data.add_(g, alpha=-lr)


def _is_adamw_no_decay(name: str, p: torch.nn.Parameter) -> bool:
    """Return True for params that should not receive AdamW weight decay."""
    return p.ndim <= 1 or name.endswith(".bias") or "norm" in name or "embedding" in name


def _is_muon_matrix_param(name: str, p: torch.nn.Parameter) -> bool:
    """Return True for hidden projection matrices that are safe for Muon."""
    if p.ndim != 2:
        return False

    muon_suffixes = (
        ".attn.q_proj.weight",
        ".attn.k_proj.weight",
        ".attn.v_proj.weight",
        ".attn.out_proj.weight",
        ".attn.qkv.weight",
        ".mlp.gate.weight",
        ".mlp.up.weight",
        ".mlp.down.weight",
    )
    return name.startswith("blocks.") and name.endswith(muon_suffixes)


def build_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = "muon_adamw",
    lr_muon: float = 0.02,
    lr_adamw: float = 3e-4,
    weight_decay: float = 0.01,
    momentum: float = 0.95,
    ns_steps: int = 5,
    betas_adamw: tuple[float, float] = (0.9, 0.999),
) -> list[torch.optim.Optimizer]:
    """Split model parameters and return optimizer list.

    Returns:
        [muon, adamw] when optimizer_type == 'muon_adamw',
        [adamw]       when optimizer_type == 'adamw'.
    """
    if optimizer_type == "adamw":
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if _is_adamw_no_decay(name, p):
                no_decay.append(p)
            else:
                decay.append(p)
        return [AdamW(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=lr_adamw, betas=betas_adamw,
        )]

    # Separate hidden projection matrices (Muon) from everything else (AdamW).
    # In particular, keep embeddings, AdaLN-Zero control tensors, timestep MLPs,
    # input/output projections, norms, and biases on AdamW.
    muon_params, adamw_decay, adamw_no_decay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if _is_muon_matrix_param(name, p):
            muon_params.append(p)
        elif _is_adamw_no_decay(name, p):
            adamw_no_decay.append(p)
        else:
            adamw_decay.append(p)

    return [
        Muon(muon_params, lr=lr_muon, momentum=momentum, ns_steps=ns_steps, weight_decay=0.0),
        AdamW(
            [{"params": adamw_decay, "weight_decay": weight_decay},
             {"params": adamw_no_decay, "weight_decay": 0.0}],
            lr=lr_adamw,
            betas=betas_adamw,
        ),
    ]
