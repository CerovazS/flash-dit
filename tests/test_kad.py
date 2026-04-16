"""Tests for the KAD wrapper.

The real PANNs backbone is ~300 MB and requires a GPU, so we stub it with
a ``FakeModelLoader`` that returns deterministic synthetic embeddings.
That covers the real integration points: EmbeddingLoader's on-disk caching,
the (N_total, D) concatenation, the MMD call, and the JSON output — without
downloading any weights or requiring CUDA.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from flash_dit.evaluation.kad import KADResult, compute_kad
from flash_dit.evaluation.kadtk_vendored.model_loader import ModelLoader


class FakeModelLoader(ModelLoader):
    """In-memory embedding model used for end-to-end unit tests.

    Each call returns ``n_frames`` rows of ``num_features`` dims, drawn from
    a Gaussian whose mean is derived from a hash of the input audio's first
    samples — so the same audio always produces the same embedding while
    different audio produces different ones.
    """

    def __init__(
        self,
        num_features: int = 16,
        sr: int = 16000,
        n_frames: int = 3,
        mean_shift: float = 0.0,
        name: str = "fake-model",
    ) -> None:
        super().__init__(name=name, num_features=num_features, sr=sr)
        self.device = torch.device("cpu")
        self.n_frames = n_frames
        self.mean_shift = mean_shift

    def load_model(self) -> None:
        self.model = torch.nn.Identity()

    def _get_embedding(self, audio: np.ndarray) -> torch.Tensor:
        seed = int(abs(hash(audio[:16].tobytes())) % (2**31))
        g = np.random.default_rng(seed)
        emb = g.normal(
            loc=self.mean_shift, scale=1.0,
            size=(self.n_frames, self.num_features),
        ).astype(np.float32)
        return torch.from_numpy(emb)

    def load_wav(self, wav_file: Path) -> np.ndarray:
        wav, _ = sf.read(str(wav_file), dtype="int16")
        return wav.astype(np.float32) / 32768.0


def _make_wav_dir(path: Path, count: int, sr: int, seed: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(count):
        data = (rng.normal(scale=0.3, size=sr) * 32767).astype(np.int16)
        sf.write(str(path / f"clip_{i:03d}.wav"), data, samplerate=sr)


def test_compute_kad_identical_distributions_is_near_zero(tmp_path):
    sr = 16000
    ref = tmp_path / "ref"
    gen = tmp_path / "gen"
    _make_wav_dir(ref, count=8, sr=sr, seed=0)
    _make_wav_dir(gen, count=8, sr=sr, seed=1)  # same distribution, different samples

    model = FakeModelLoader(num_features=8, sr=sr, n_frames=4, mean_shift=0.0)

    result = compute_kad(
        generated_dir=gen,
        reference_dir=ref,
        device="cpu",
        workers=1,
        scale_factor=1.0,
        model=model,
    )

    assert isinstance(result, KADResult)
    assert result.n_generated == 8 and result.n_reference == 8
    assert result.n_generated_embeddings == 8 * 4
    assert result.n_reference_embeddings == 8 * 4
    # Same synthetic distribution → raw MMD² should be small in magnitude.
    assert abs(result.mmd2) < 0.25
    assert result.bandwidth > 0.0
    assert result.bandwidth_source == "reference"
    assert result.model_name == "fake-model"


def test_compute_kad_shifted_distribution_gives_larger_score(tmp_path):
    sr = 16000
    ref = tmp_path / "ref"
    gen_close = tmp_path / "gen_close"
    gen_far = tmp_path / "gen_far"
    _make_wav_dir(ref, count=12, sr=sr, seed=0)
    _make_wav_dir(gen_close, count=12, sr=sr, seed=1)
    _make_wav_dir(gen_far, count=12, sr=sr, seed=2)

    ref_model = FakeModelLoader(num_features=8, sr=sr, n_frames=4, mean_shift=0.0)
    far_model = FakeModelLoader(num_features=8, sr=sr, n_frames=4, mean_shift=3.0)

    close = compute_kad(
        generated_dir=gen_close,
        reference_dir=ref,
        device="cpu", workers=1, scale_factor=1.0,
        model=ref_model,
    )
    far = compute_kad(
        generated_dir=gen_far,
        reference_dir=ref,
        device="cpu", workers=1, scale_factor=1.0,
        model=far_model,
    )
    assert far.mmd2 > close.mmd2
    assert far.mmd2 > 0.05


def test_compute_kad_writes_json(tmp_path):
    sr = 16000
    ref = tmp_path / "ref"
    gen = tmp_path / "gen"
    _make_wav_dir(ref, count=4, sr=sr, seed=0)
    _make_wav_dir(gen, count=4, sr=sr, seed=1)

    out_json = tmp_path / "metrics" / "kad.json"
    result = compute_kad(
        generated_dir=gen,
        reference_dir=ref,
        device="cpu", workers=1, scale_factor=100.0,
        model=FakeModelLoader(num_features=8, sr=sr, n_frames=4),
        output_json=out_json,
    )

    assert out_json.exists()
    data = json.loads(out_json.read_text())
    for k in [
        "score", "mmd2", "scale_factor", "bandwidth", "bandwidth_source",
        "model_name", "generated_dir", "reference_dir",
        "n_generated", "n_reference", "n_generated_embeddings",
        "n_reference_embeddings", "device", "workers", "timestamp",
    ]:
        assert k in data, f"missing key: {k}"
    assert data["score"] == pytest.approx(result.score)


def test_compute_kad_scale_factor_applied(tmp_path):
    sr = 16000
    ref = tmp_path / "ref"
    gen = tmp_path / "gen"
    _make_wav_dir(ref, count=6, sr=sr, seed=0)
    _make_wav_dir(gen, count=6, sr=sr, seed=1)

    model = FakeModelLoader(num_features=8, sr=sr, n_frames=3)
    raw = compute_kad(
        generated_dir=gen, reference_dir=ref,
        device="cpu", workers=1, scale_factor=1.0, model=model,
    )
    # Re-use same caches; ensure scale_factor is honored on the MMD output only.
    # Force re-extraction to avoid stale caches from previous scale-factor run
    # polluting embeddings (though in this path it shouldn't — scaling happens post-MMD).
    scaled = compute_kad(
        generated_dir=gen, reference_dir=ref,
        device="cpu", workers=1, scale_factor=100.0, model=model,
    )
    assert scaled.mmd2 == pytest.approx(raw.mmd2, rel=1e-6, abs=1e-8)
    assert scaled.score == pytest.approx(raw.mmd2 * 100.0, rel=1e-6, abs=1e-6)


def test_compute_kad_rejects_missing_directories(tmp_path):
    with pytest.raises(FileNotFoundError):
        compute_kad(
            generated_dir=tmp_path / "does_not_exist",
            reference_dir=tmp_path,
            device="cpu", workers=1,
            model=FakeModelLoader(),
        )


def test_eval_kad_cli_has_help():
    import subprocess

    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "scripts" / "eval_kad.py"
    assert script.exists()
    result = subprocess.run(
        ["python", str(script), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    for flag in ["--generated", "--reference", "--model-name", "--device", "--bandwidth-source"]:
        assert flag in result.stdout
