"""Tests for ML denoising pipeline.

Covers: DenoisingTrainingConfig, DenoisingLoss, DenoisingDataset,
DenoisingTrainer, PatternDenoiser, and pair extraction.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

# Same cause as tests/test_ml_phase_recognition.py: `.gitignore`'s `data/` rule
# excludes Ai_Ml/ebsd_ai/data/*.py, which ebsd_ai/__init__.py imports, so a
# fresh clone cannot import any ebsd_ai submodule — 19 red tests here that say
# nothing about the code. Fix pending as ad7ffbc2 on the unmerged branch
# feature/ml-sht-simulation-training. No-op where the files are on disk.
#
# try/except, not importorskip: the FIRST ml test module to attempt this
# import leaves a half-built `ebsd_ai` behind (the package entry is gone from
# sys.modules while a submodule still holds its path object), and the next
# importorskip then dies with `KeyError: 'ebsd_ai'` — a collection ERROR, which
# is exactly the red this skip exists to remove. Measured: running
# test_ml_phase_recognition.py and this file together.
try:                                            # pragma: no cover - import probe
    import ebsd_ai.data.dataset                 # noqa: F401
except Exception:                               # ModuleNotFoundError, KeyError
    pytest.skip(
        "Ai_Ml/ebsd_ai/data/*.py is excluded by the 'data/' rule in "
        ".gitignore, so a fresh clone has no ebsd_ai.data.dataset; fix "
        "pending as ad7ffbc2 on feature/ml-sht-simulation-training",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Task 1: DenoisingTrainingConfig
# ---------------------------------------------------------------------------


def test_denoising_training_config_defaults():
    from ebsd_ai.config import DenoisingTrainingConfig

    cfg = DenoisingTrainingConfig()
    assert cfg.learning_rate == 1e-3
    assert cfg.batch_size == 32
    assert cfg.epochs == 100
    assert cfg.early_stopping_patience == 15
    assert cfg.val_fraction == 0.15
    assert cfg.l1_weight == 1.0
    assert cfg.ssim_weight == 0.5
    assert cfg.min_training_pairs == 500
    assert cfg.residual_learning is True
    assert cfg.weight_decay == 1e-4
    assert cfg.mixed_precision is True


def test_denoising_training_config_serialization():
    from ebsd_ai.config import DenoisingTrainingConfig

    cfg = DenoisingTrainingConfig(learning_rate=5e-4, batch_size=16)
    d = cfg.to_dict()
    cfg2 = DenoisingTrainingConfig.from_dict(d)
    assert cfg2.learning_rate == 5e-4
    assert cfg2.batch_size == 16


def test_denoising_training_config_from_dict_ignores_extra_keys():
    from ebsd_ai.config import DenoisingTrainingConfig

    d = {"learning_rate": 2e-3, "unknown_key": 42}
    cfg = DenoisingTrainingConfig.from_dict(d)
    assert cfg.learning_rate == 2e-3


# ---------------------------------------------------------------------------
# Task 1: DenoisingLoss
# ---------------------------------------------------------------------------


def test_denoising_loss_output_shape():
    from ebsd_ai.models.denoising_loss import DenoisingLoss

    loss_fn = DenoisingLoss(l1_weight=1.0, ssim_weight=0.5)
    pred = torch.rand(4, 1, 128, 128)
    target = torch.rand(4, 1, 128, 128)
    loss = loss_fn(pred, target)
    assert loss.shape == ()  # scalar
    assert loss.item() > 0


def test_denoising_loss_identical_is_near_zero():
    from ebsd_ai.models.denoising_loss import DenoisingLoss

    loss_fn = DenoisingLoss()
    x = torch.rand(2, 1, 128, 128)
    loss = loss_fn(x, x)
    assert loss.item() < 0.01  # L1=0, SSIM near 1 so loss near 0


def test_denoising_loss_gradient_flows():
    from ebsd_ai.models.denoising_loss import DenoisingLoss

    loss_fn = DenoisingLoss()
    pred = torch.rand(2, 1, 64, 64, requires_grad=True)
    target = torch.rand(2, 1, 64, 64)
    loss = loss_fn(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.shape == pred.shape


def test_ssim_range():
    from ebsd_ai.models.denoising_loss import compute_ssim

    x = torch.rand(2, 1, 64, 64)
    y = torch.rand(2, 1, 64, 64)
    ssim_val = compute_ssim(x, y)
    assert -1.0 <= ssim_val.item() <= 1.0


def test_ssim_identical_is_one():
    from ebsd_ai.models.denoising_loss import compute_ssim

    x = torch.rand(2, 1, 64, 64)
    ssim_val = compute_ssim(x, x)
    assert ssim_val.item() > 0.99


# ---------------------------------------------------------------------------
# Task 2: TrainingStore Schema v2 + Denoising Pairs
# ---------------------------------------------------------------------------


def test_training_store_add_denoising_pair(tmp_path):
    from ebsd_ai.data.training_store import TrainingStore
    from ebsd_ai.config import DetectorInfo

    store = TrainingStore(str(tmp_path / "test_store"))
    experimental = np.random.rand(128, 128).astype(np.float32)
    simulated = np.random.rand(128, 128).astype(np.float32)
    detector = DetectorInfo(pc=(0.5, 0.5, 0.5), kv=20.0)

    store.add_denoising_pair(
        experimental_pattern=experimental,
        simulated_pattern=simulated,
        confirmed_phase="Ferrit",
        detector_info=detector,
        confidence_score=0.85,
        source_file="test.h5",
    )
    assert len(store) == 1
    sample = store.get_sample(0)
    assert sample["pair_type"] == "denoising"
    assert sample["pattern_target"] is not None
    assert sample["pattern_target"].shape == (128, 128)


def test_training_store_backward_compatible(tmp_path):
    from ebsd_ai.data.training_store import TrainingStore
    from ebsd_ai.config import DetectorInfo

    store = TrainingStore(str(tmp_path / "test_store"))
    detector = DetectorInfo(pc=(0.5, 0.5, 0.5), kv=20.0)
    pattern = np.random.rand(128, 128).astype(np.float32)
    store.add_sample(
        pattern=pattern,
        confirmed_phase="Ferrit",
        detector_info=detector,
        confidence_score=0.9,
    )
    sample = store.get_sample(0)
    assert sample.get("pair_type", "classification") == "classification"
    assert sample.get("pattern_target") is None


def test_training_store_denoising_pair_count(tmp_path):
    from ebsd_ai.data.training_store import TrainingStore
    from ebsd_ai.config import DetectorInfo

    store = TrainingStore(str(tmp_path / "test_store"))
    detector = DetectorInfo(pc=(0.5, 0.5, 0.5), kv=20.0)

    # Add 1 classification + 2 denoising
    store.add_sample(
        pattern=np.random.rand(128, 128).astype(np.float32),
        confirmed_phase="Ferrit",
        detector_info=detector,
    )
    for _ in range(2):
        store.add_denoising_pair(
            experimental_pattern=np.random.rand(128, 128).astype(np.float32),
            simulated_pattern=np.random.rand(128, 128).astype(np.float32),
            confirmed_phase="Ferrit",
            detector_info=detector,
        )
    assert len(store) == 3
    assert store.count_denoising_pairs() == 2


# ---------------------------------------------------------------------------
# Task 3: DenoisingDataset + Residual Learning
# ---------------------------------------------------------------------------


def test_denoising_dataset_filters_pairs(tmp_path):
    """DenoisingDataset only yields denoising pairs, not classification."""
    from ebsd_ai.data.training_store import TrainingStore
    from ebsd_ai.data.denoising_dataset import DenoisingDataset
    from ebsd_ai.config import DetectorInfo

    store = TrainingStore(str(tmp_path / "store"))
    det = DetectorInfo(pc=(0.5, 0.5, 0.5), kv=20.0)

    # 1 classification sample
    store.add_sample(
        pattern=np.random.rand(128, 128).astype(np.float32),
        confirmed_phase="Ferrit",
        detector_info=det,
    )
    # 2 denoising pairs
    for _ in range(2):
        store.add_denoising_pair(
            experimental_pattern=np.random.rand(128, 128).astype(np.float32),
            simulated_pattern=np.random.rand(128, 128).astype(np.float32),
            confirmed_phase="Ferrit",
            detector_info=det,
        )

    ds = DenoisingDataset(store)
    assert len(ds) == 2  # only denoising pairs


def test_denoising_dataset_returns_correct_shapes(tmp_path):
    """Each item has input (1,128,128), target (1,128,128), detector (15,)."""
    from ebsd_ai.data.training_store import TrainingStore
    from ebsd_ai.data.denoising_dataset import DenoisingDataset
    from ebsd_ai.config import DetectorInfo

    store = TrainingStore(str(tmp_path / "store"))
    det = DetectorInfo(pc=(0.5, 0.5, 0.5), kv=20.0)
    store.add_denoising_pair(
        experimental_pattern=np.random.rand(128, 128).astype(np.float32),
        simulated_pattern=np.random.rand(128, 128).astype(np.float32),
        confirmed_phase="Ferrit",
        detector_info=det,
    )

    ds = DenoisingDataset(store)
    item = ds[0]
    assert isinstance(item, dict)
    assert item["input"].shape == (1, 128, 128)
    assert item["target"].shape == (1, 128, 128)
    assert item["detector_encoding"].shape == (DetectorInfo.encoded_length(),)


def test_pattern_enhancer_residual_learning():
    """PatternEnhancer with residual_learning adds input to output."""
    from ebsd_ai.models.pattern_enhancer import PatternEnhancer, EnhancerConfig

    cfg = EnhancerConfig(use_film=False)
    model = PatternEnhancer(config=cfg, residual_learning=True)
    x = torch.rand(1, 1, 128, 128)
    out = model(x)
    assert out.shape == x.shape
    # With residual learning, output should differ from non-residual
    model_no_res = PatternEnhancer(config=cfg, residual_learning=False)
    model_no_res.load_state_dict(model.state_dict())
    out_no_res = model_no_res(x)
    # They should differ because residual adds x
    assert not torch.allclose(out, out_no_res, atol=1e-6)


# ---------------------------------------------------------------------------
# Task 4: DenoisingTrainer
# ---------------------------------------------------------------------------


def test_denoising_trainer_one_epoch(tmp_path):
    """DenoisingTrainer can complete 1 epoch and return a result."""
    from ebsd_ai.data.training_store import TrainingStore
    from ebsd_ai.training.denoising_trainer import DenoisingTrainer, DenoisingTrainingResult
    from ebsd_ai.config import DetectorInfo, DenoisingTrainingConfig

    store = TrainingStore(str(tmp_path / "store"))
    det = DetectorInfo(pc=(0.5, 0.5, 0.5), kv=20.0)

    # Add enough denoising pairs for a small train/val split
    for _ in range(10):
        store.add_denoising_pair(
            experimental_pattern=np.random.rand(128, 128).astype(np.float32),
            simulated_pattern=np.random.rand(128, 128).astype(np.float32),
            confirmed_phase="Ferrit",
            detector_info=det,
        )

    cfg = DenoisingTrainingConfig(
        epochs=1,
        batch_size=4,
        min_training_pairs=1,
        mixed_precision=False,
    )
    trainer = DenoisingTrainer(
        config=cfg,
        output_dir=str(tmp_path / "checkpoints"),
        device="cpu",
    )
    result = trainer.train(store)
    assert isinstance(result, DenoisingTrainingResult)
    assert result.epochs_completed == 1
    assert len(result.train_losses) == 1
    assert len(result.val_losses) == 1
    assert result.train_losses[0] > 0


def test_denoising_trainer_saves_checkpoint(tmp_path):
    """DenoisingTrainer saves best.pt checkpoint."""
    from ebsd_ai.data.training_store import TrainingStore
    from ebsd_ai.training.denoising_trainer import DenoisingTrainer
    from ebsd_ai.config import DetectorInfo, DenoisingTrainingConfig

    store = TrainingStore(str(tmp_path / "store"))
    det = DetectorInfo(pc=(0.5, 0.5, 0.5), kv=20.0)

    for _ in range(10):
        store.add_denoising_pair(
            experimental_pattern=np.random.rand(128, 128).astype(np.float32),
            simulated_pattern=np.random.rand(128, 128).astype(np.float32),
            confirmed_phase="Ferrit",
            detector_info=det,
        )

    ckpt_dir = tmp_path / "checkpoints"
    cfg = DenoisingTrainingConfig(
        epochs=2, batch_size=4, min_training_pairs=1, mixed_precision=False,
    )
    trainer = DenoisingTrainer(config=cfg, output_dir=str(ckpt_dir), device="cpu")
    trainer.train(store)
    assert (ckpt_dir / "best.pt").exists()


# ---------------------------------------------------------------------------
# Task 5: PatternDenoiser Inference API
# ---------------------------------------------------------------------------


def test_pattern_denoiser_no_model_passthrough():
    """Without a loaded model, PatternDenoiser returns normalized input."""
    from ebsd_ai.inference.pattern_denoiser import PatternDenoiser

    denoiser = PatternDenoiser()  # no model loaded
    pattern = np.random.rand(128, 128).astype(np.float32)
    result = denoiser.enhance(pattern)
    assert result.shape == (128, 128)
    assert result.dtype == np.float32


def test_pattern_denoiser_with_model():
    """PatternDenoiser with a model produces output of correct shape."""
    from ebsd_ai.inference.pattern_denoiser import PatternDenoiser
    from ebsd_ai.models.pattern_enhancer import PatternEnhancer

    model = PatternEnhancer(residual_learning=True)
    denoiser = PatternDenoiser(model=model, device="cpu")
    pattern = np.random.rand(128, 128).astype(np.float32)
    result = denoiser.enhance(pattern)
    assert result.shape == (128, 128)
    assert result.dtype == np.float32


def test_pattern_denoiser_enhance_batch():
    """PatternDenoiser.enhance_batch processes multiple patterns."""
    from ebsd_ai.inference.pattern_denoiser import PatternDenoiser
    from ebsd_ai.models.pattern_enhancer import PatternEnhancer

    model = PatternEnhancer(residual_learning=True)
    denoiser = PatternDenoiser(model=model, device="cpu")
    patterns = np.random.rand(6, 128, 128).astype(np.float32)
    result = denoiser.enhance_batch(patterns, batch_size=4)
    assert result.shape == (6, 128, 128)


def test_pattern_denoiser_from_checkpoint(tmp_path):
    """PatternDenoiser can load from a checkpoint file."""
    from ebsd_ai.inference.pattern_denoiser import PatternDenoiser
    from ebsd_ai.models.pattern_enhancer import PatternEnhancer

    model = PatternEnhancer(residual_learning=True)
    ckpt_path = tmp_path / "best.pt"
    torch.save(model.get_save_dict(), ckpt_path)

    denoiser = PatternDenoiser.from_checkpoint(str(ckpt_path), device="cpu")
    pattern = np.random.rand(128, 128).astype(np.float32)
    result = denoiser.enhance(pattern)
    assert result.shape == (128, 128)


# ---------------------------------------------------------------------------
# Task 6: Denoising Pair Extraction
# ---------------------------------------------------------------------------


def test_extract_denoising_pairs_basic():
    """extract_denoising_pairs filters by CI threshold and returns pairs."""
    import sys
    sys.path.insert(0, "Ai_Ml")
    from ml_controller import extract_denoising_pairs

    n = 10
    experimental = np.random.rand(n, 128, 128).astype(np.float32)
    simulated = np.random.rand(n, 128, 128).astype(np.float32)
    confidence = np.array([0.1, 0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    phase_ids = np.zeros(n, dtype=int)
    phase_names = ["Ferrit"]

    pairs = extract_denoising_pairs(
        experimental_patterns=experimental,
        simulated_patterns=simulated,
        confidence_scores=confidence,
        phase_ids=phase_ids,
        phase_names=phase_names,
        ci_threshold=0.3,
    )
    # Only patterns with CI >= 0.3 should be included (indices 2-9 = 8 pairs)
    assert len(pairs) == 8
    assert pairs[0]["confidence"] >= 0.3
    assert pairs[0]["experimental"].shape == (128, 128)
    assert pairs[0]["simulated"].shape == (128, 128)
    assert pairs[0]["phase"] == "Ferrit"


def test_add_denoising_pairs_to_store(tmp_path):
    """add_denoising_pairs_to_store adds pairs to the TrainingStore."""
    import sys
    sys.path.insert(0, "Ai_Ml")
    from ml_controller import add_denoising_pairs_to_store
    from ebsd_ai.data.training_store import TrainingStore
    from ebsd_ai.config import DetectorInfo

    store = TrainingStore(str(tmp_path / "store"))
    det = DetectorInfo(pc=(0.5, 0.5, 0.5), kv=20.0)

    pairs = []
    for i in range(5):
        pairs.append({
            "experimental": np.random.rand(128, 128).astype(np.float32),
            "simulated": np.random.rand(128, 128).astype(np.float32),
            "confidence": 0.8,
            "phase": "Ferrit",
        })

    count = add_denoising_pairs_to_store(pairs, store=store, detector_info=det)
    assert count == 5
    assert store.count_denoising_pairs() == 5


# ---------------------------------------------------------------------------
# Task 7: GUI Integration
# ---------------------------------------------------------------------------


def test_ml_hub_gui_has_denoising_section():
    """MLHubPage has denoising training controls."""
    try:
        from gui.ml_hub_gui import MLHubPage, _DenoisingTrainingWorker
        # Just verify the classes are importable
        assert MLHubPage is not None
        assert _DenoisingTrainingWorker is not None
    except ImportError:
        pytest.skip("PyQt5 not available")
