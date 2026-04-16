import torch
import torch.nn as nn

from flash_dit.training.optimizer import Muon, build_optimizer


class _DummyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 2, bias=False)
        self.v_proj = nn.Linear(4, 2, bias=False)
        self.out_proj = nn.Linear(4, 4, bias=False)


class _DummyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(4, 8, bias=False)
        self.up = nn.Linear(4, 8, bias=False)
        self.down = nn.Linear(8, 4, bias=False)


class _DummyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _DummyAttention()
        self.mlp = _DummyMLP()
        self.adaLN = nn.Linear(4, 24)
        self.norm = nn.LayerNorm(4)


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_DummyBlock()])
        self.x_embedder = nn.Linear(4, 4, bias=False)
        self.t_embedder = nn.Sequential(nn.Linear(4, 4), nn.SiLU(), nn.Linear(4, 4))
        self.y_embedder = nn.Embedding(8, 4)
        self.final_layer = nn.Linear(4, 4)


def _optimizer_param_names(model: nn.Module, optimizer: torch.optim.Optimizer) -> set[str]:
    names_by_param = {p: name for name, p in model.named_parameters()}
    return {
        names_by_param[p]
        for group in optimizer.param_groups
        for p in group["params"]
    }


def test_muon_only_gets_hidden_projection_matrices() -> None:
    model = _DummyModel()

    muon, adamw = build_optimizer(model, optimizer_type="muon_adamw")

    muon_names = _optimizer_param_names(model, muon)
    adamw_names = _optimizer_param_names(model, adamw)

    assert muon_names == {
        "blocks.0.attn.q_proj.weight",
        "blocks.0.attn.k_proj.weight",
        "blocks.0.attn.v_proj.weight",
        "blocks.0.attn.out_proj.weight",
        "blocks.0.mlp.gate.weight",
        "blocks.0.mlp.up.weight",
        "blocks.0.mlp.down.weight",
    }
    assert "blocks.0.adaLN.weight" in adamw_names
    assert "final_layer.weight" in adamw_names
    assert "x_embedder.weight" in adamw_names
    assert "y_embedder.weight" in adamw_names
    assert muon_names.isdisjoint(adamw_names)
    assert muon_names | adamw_names == {name for name, _ in model.named_parameters()}


def test_muon_scale_uses_row_column_ratio(monkeypatch) -> None:
    wide = nn.Parameter(torch.zeros(2, 8))
    tall = nn.Parameter(torch.zeros(8, 2))
    wide.grad = torch.ones_like(wide)
    tall.grad = torch.ones_like(tall)

    monkeypatch.setattr(Muon, "_newtonschulz5", staticmethod(lambda g, steps: torch.ones_like(g)))

    opt = Muon([wide, tall], lr=0.1, momentum=0.0, ns_steps=1)
    opt.step()

    assert torch.allclose(wide, torch.full_like(wide, -0.1))
    assert torch.allclose(tall, torch.full_like(tall, -0.2))
