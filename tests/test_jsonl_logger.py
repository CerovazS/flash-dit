import torch

from flash_dit.training.jsonl_logger import JSONLMetricsCallback, _to_json_scalar


def test_to_json_scalar_accepts_scalar_tensors_and_primitives() -> None:
    assert _to_json_scalar(torch.tensor(1.5)) == 1.5
    assert _to_json_scalar(torch.tensor(2)) == 2
    assert _to_json_scalar(True) is True
    assert _to_json_scalar("ok") == "ok"


def test_to_json_scalar_skips_non_scalar_and_non_finite_values() -> None:
    assert _to_json_scalar(torch.ones(2)) is None
    assert _to_json_scalar(float("nan")) is None
    assert _to_json_scalar(object()) is None


def test_collect_lrs_matches_lightning_lr_names() -> None:
    p1 = torch.nn.Parameter(torch.zeros(1))
    p2 = torch.nn.Parameter(torch.zeros(1))
    adamw = torch.optim.AdamW(
        [{"params": [p1], "lr": 1e-4}, {"params": [p2], "lr": 2e-4}],
        lr=3e-4,
    )
    sgd = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)

    trainer = type("TrainerStub", (), {"optimizers": [adamw, sgd]})()

    assert JSONLMetricsCallback._collect_lrs(trainer) == {
        "lr-AdamW/pg1": 0.0001,
        "lr-AdamW/pg2": 0.0002,
        "lr-SGD": 0.1,
    }
