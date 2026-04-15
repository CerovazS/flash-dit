"""MLP variants: SwiGLU and ReLU²."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: FFN(x) = SiLU(W1(x)) ⊙ W2(x), then W3.

    Uses expansion ratio 8/3 ≈ 2.667 (standard Llama configuration).
    """

    def __init__(self, d_model: int, mlp_mult: float = 8 / 3) -> None:
        super().__init__()
        hidden = int(d_model * mlp_mult)
        # Round to nearest multiple of 64 for hardware efficiency
        hidden = (hidden + 63) // 64 * 64
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class ReLUSquared(nn.Module):
    """ReLU² feed-forward: FFN(x) = W2(ReLU(W1(x))²).

    Sparse activation pattern, better parameter efficiency at scale.
    Reference: "Primer: Searching for Efficient Transformers" (So et al., 2021).
    """

    def __init__(self, d_model: int, mlp_mult: float = 4.0) -> None:
        super().__init__()
        hidden = int(d_model * mlp_mult)
        hidden = (hidden + 63) // 64 * 64
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.relu(self.up(x)).square())


def build_mlp(d_model: int, mlp_mult: float, mlp_type: str) -> nn.Module:
    """Factory: returns the correct MLP given mlp_type."""
    if mlp_type == "swiglu":
        return SwiGLU(d_model, mlp_mult)
    if mlp_type == "relu2":
        return ReLUSquared(d_model, mlp_mult)
    raise ValueError(f"Unknown mlp_type: {mlp_type!r}. Choose 'swiglu' or 'relu2'.")
