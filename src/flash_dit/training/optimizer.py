"""Muon optimizer (inline Newton-Schulz) + AdamW hybrid builder.

Muon is applied to all 2-D projection matrices in the transformer.
AdamW handles scalar/vector parameters (embeddings, norms, biases, final head).

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
    def step(self) -> None:  # type: ignore[override]
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
                    # Scale so that effective step size is lr
                    scale = float(max(g.size(0), g.size(1))) ** 0.5
                    p.data.add_(g_orth, alpha=-lr * scale)
                else:
                    # Fallback for non-2D params (should not reach here if split correctly)
                    p.data.add_(g, alpha=-lr)


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
        return [AdamW(model.parameters(), lr=lr_adamw, weight_decay=weight_decay, betas=betas_adamw)]

    # Separate 2-D matrices (Muon) from everything else (AdamW)
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    return [
        Muon(muon_params, lr=lr_muon, momentum=momentum, ns_steps=ns_steps, weight_decay=0.0),
        AdamW(adamw_params, lr=lr_adamw, weight_decay=weight_decay, betas=betas_adamw),
    ]
