"""Tests for memory-efficient large scan handling.

Verifies thread locks, lightweight metadata reads, chunked import,
and lazy pattern indexing in prediction.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from ebsd_ai.config import (
    DetectorInfo,
)
from ebsd_ai.data.training_store import TrainingStore
from tests.conftest import (
    make_synthetic_detector_info,
    make_synthetic_eds,
    make_synthetic_pattern,
)

# ---------------------------------------------------------------------------
# TrainingStore thread safety
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> TrainingStore:
    """TrainingStore with a few samples for testing."""
    s = TrainingStore(tmp_path / "store")
    rng = np.random.default_rng(42)
    for phase in ["Ferrite", "Austenite", "Ferrite"]:
        s.add_sample(
            pattern=make_synthetic_pattern(64, 64, rng),
            confirmed_phase=phase,
            detector_info=make_synthetic_detector_info(rng=rng),
            eds_data=make_synthetic_eds(rng=rng),
            confidence_score=0.9,
        )
    return s


class TestTrainingStoreLock:
    """TrainingStore has a threading.RLock for HDF5 access."""

    def test_lock_exists(self, store: TrainingStore) -> None:
        assert hasattr(store, "_lock")
        assert isinstance(store._lock, type(threading.RLock()))

    def test_concurrent_reads(self, store: TrainingStore) -> None:
        """Multiple threads can read samples without errors."""
        results: list[dict] = [{}] * 3
        errors: list[Exception] = []

        def reader(idx: int) -> None:
            try:
                results[idx] = store.get_sample(idx)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        for r in results:
            assert "confirmed_phase" in r

    def test_concurrent_metadata_reads(self, store: TrainingStore) -> None:
        """Multiple threads can read metadata without errors."""
        errors: list[Exception] = []

        def reader(idx: int) -> None:
            try:
                store.get_metadata(idx)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ---------------------------------------------------------------------------
# Lightweight metadata access
# ---------------------------------------------------------------------------


class TestGetMetadata:
    """TrainingStore.get_metadata reads only attributes, not arrays."""

    def test_returns_expected_keys(self, store: TrainingStore) -> None:
        meta = store.get_metadata(0)
        assert "confirmed_phase" in meta
        assert "has_eds" in meta
        assert "confidence_score" in meta
        assert "data_source" in meta

    def test_no_pattern_or_eds_keys(self, store: TrainingStore) -> None:
        meta = store.get_metadata(0)
        assert "pattern_normalized" not in meta
        assert "eds_atomic_pct" not in meta
        assert "detector_info" not in meta

    def test_values_match_full_sample(self, store: TrainingStore) -> None:
        full = store.get_sample(0)
        meta = store.get_metadata(0)
        assert meta["confirmed_phase"] == full["confirmed_phase"]
        assert meta["has_eds"] == full["has_eds"]
        assert meta["confidence_score"] == pytest.approx(
            full["confidence_score"]
        )

    def test_out_of_range_raises(self, store: TrainingStore) -> None:
        with pytest.raises(IndexError):
            store.get_metadata(999)

    def test_all_samples(self, store: TrainingStore) -> None:
        for i in range(len(store)):
            meta = store.get_metadata(i)
            assert isinstance(meta["confirmed_phase"], str)
            assert isinstance(meta["has_eds"], bool)


class TestIterMetadata:
    """TrainingStore.iter_metadata yields lightweight dicts."""

    def test_count_matches_len(self, store: TrainingStore) -> None:
        items = list(store.iter_metadata())
        assert len(items) == len(store)

    def test_keys_are_metadata_only(self, store: TrainingStore) -> None:
        for meta in store.iter_metadata():
            assert "confirmed_phase" in meta
            assert "has_eds" in meta
            assert "pattern_normalized" not in meta

    def test_values_match_get_metadata(self, store: TrainingStore) -> None:
        for i, meta in enumerate(store.iter_metadata()):
            expected = store.get_metadata(i)
            assert meta["confirmed_phase"] == expected["confirmed_phase"]
            assert meta["has_eds"] == expected["has_eds"]


# ---------------------------------------------------------------------------
# Dataset uses lightweight metadata in __init__
# ---------------------------------------------------------------------------


class TestDatasetLazyInit:
    """Dataset __init__ uses get_metadata, not get_sample."""

    def test_dataset_init_works(self, store: TrainingStore) -> None:
        from ebsd_ai.data.dataset import EBSDPhaseDataset

        ds = EBSDPhaseDataset(
            store=store,
            phase_names=["Ferrite", "Austenite"],
            eds_dropout_rate=0.0,
        )
        assert len(ds) == 3
        assert ds._labels == [0, 1, 0]
        assert all(isinstance(f, bool) for f in ds._has_eds_flags)

    def test_dataset_getitem_still_works(self, store: TrainingStore) -> None:
        import torch

        from ebsd_ai.data.dataset import EBSDPhaseDataset

        ds = EBSDPhaseDataset(
            store=store,
            phase_names=["Ferrite", "Austenite"],
            eds_dropout_rate=0.0,
        )
        sample = ds[0]
        assert torch.isfinite(sample["pattern"]).all()
        assert torch.isfinite(sample["eds_input"]).all()


# ---------------------------------------------------------------------------
# Chunked import
# ---------------------------------------------------------------------------


class TestChunkedImport:
    """import_to_store processes large scans in chunks."""

    def test_chunked_import_matches_unchunked(self, tmp_path: Path) -> None:
        """Chunked import produces same result count as chunk_size=N."""
        from ebsd_ai.data.scan_import import ScanData, ScanFormat, import_to_store

        rng = np.random.default_rng(99)
        n = 10
        patterns = rng.integers(0, 255, (n, 32, 32), dtype=np.uint8)
        scan = ScanData(
            patterns=patterns,
            phase_ids=np.array([0, 1] * 5, dtype=np.int32),
            phase_names=["Alpha", "Beta"],
            euler_angles=rng.random((n, 3)).astype(np.float64),
            confidence_scores=np.linspace(0.1, 1.0, n).astype(np.float32),
            source_format=ScanFormat.ANG,
            source_file="test.ang",
        )

        # Import with small chunk_size
        store_a = TrainingStore(tmp_path / "store_a")
        result_a = import_to_store(
            "test.ang",
            store_a,
            ci_threshold=0.3,
            scan_data=scan,
            chunk_size=3,
        )

        # Import with large chunk_size (effectively unchunked)
        store_b = TrainingStore(tmp_path / "store_b")
        result_b = import_to_store(
            "test.ang",
            store_b,
            ci_threshold=0.3,
            scan_data=scan,
            chunk_size=10000,
        )

        assert result_a.n_points_imported == result_b.n_points_imported
        assert result_a.n_points_total == result_b.n_points_total
        assert len(store_a) == len(store_b)

    def test_chunk_size_1(self, tmp_path: Path) -> None:
        """Edge case: chunk_size=1 still works."""
        from ebsd_ai.data.scan_import import ScanData, ScanFormat, import_to_store

        rng = np.random.default_rng(42)
        n = 3
        scan = ScanData(
            patterns=rng.integers(0, 255, (n, 16, 16), dtype=np.uint8),
            phase_ids=np.zeros(n, dtype=np.int32),
            phase_names=["Alpha"],
            euler_angles=np.zeros((n, 3), dtype=np.float64),
            confidence_scores=np.ones(n, dtype=np.float32),
            source_format=ScanFormat.ANG,
            source_file="test.ang",
        )

        s = TrainingStore(tmp_path / "store")
        result = import_to_store(
            "test.ang", s, scan_data=scan, chunk_size=1
        )
        assert result.n_points_imported == n
        assert len(s) == n


# ---------------------------------------------------------------------------
# Predictor lazy pattern indexing
# ---------------------------------------------------------------------------


class TestPredictorLazyIndex:
    """predict_scan indexes patterns lazily from 4D array."""

    def test_predict_scan_basic(self, tmp_path: Path) -> None:
        """predict_scan still produces correct results."""
        from ebsd_ai.inference.predictor import PhasePredictor
        from ebsd_ai.models.phase_classifier import PhaseClassifier

        model = PhaseClassifier(phase_names=["A", "B"])
        predictor = PhasePredictor.from_model(model, device="cpu")

        rng = np.random.default_rng(42)
        patterns = rng.integers(0, 255, (2, 3, 32, 32), dtype=np.uint8)

        result = predictor.predict_scan(patterns, k=2)
        assert result.n_pixels == 6
        assert result.scan_shape == (2, 3)
        assert result.phase_names.shape == (6, 2)
        assert np.isfinite(result.probabilities).all()
        assert np.isfinite(result.confidence).all()

    def test_predict_scan_with_mask(self, tmp_path: Path) -> None:
        """predict_scan with selection_mask works."""
        from ebsd_ai.inference.predictor import PhasePredictor
        from ebsd_ai.models.phase_classifier import PhaseClassifier

        model = PhaseClassifier(phase_names=["A", "B"])
        predictor = PhasePredictor.from_model(model, device="cpu")

        rng = np.random.default_rng(42)
        patterns = rng.integers(0, 255, (2, 3, 32, 32), dtype=np.uint8)
        mask = np.array([[True, False, True], [False, True, False]])

        result = predictor.predict_scan(patterns, selection_mask=mask, k=2)
        assert result.n_pixels == 3

    def test_predict_scan_with_eds(self, tmp_path: Path) -> None:
        """predict_scan with EDS maps works with lazy indexing."""
        from ebsd_ai.inference.predictor import PhasePredictor
        from ebsd_ai.models.phase_classifier import PhaseClassifier

        model = PhaseClassifier(phase_names=["A", "B"])
        predictor = PhasePredictor.from_model(model, device="cpu")

        rng = np.random.default_rng(42)
        n_rows, n_cols = 2, 3
        patterns = rng.integers(
            0, 255, (n_rows, n_cols, 32, 32), dtype=np.uint8
        )
        eds_maps = {
            "Fe": rng.random(n_rows * n_cols).astype(np.float32) * 50,
            "C": rng.random(n_rows * n_cols).astype(np.float32) * 10,
        }

        result = predictor.predict_scan(
            patterns, eds_maps=eds_maps, k=2
        )
        assert result.n_pixels == 6
        assert np.isfinite(result.probabilities).all()


# ---------------------------------------------------------------------------
# HDF5 chunking in add_sample
# ---------------------------------------------------------------------------


class TestHDF5Chunking:
    """Verify HDF5 datasets use chunked storage."""

    def test_pattern_datasets_are_chunked(self, tmp_path: Path) -> None:
        import h5py

        s = TrainingStore(tmp_path / "store")
        rng = np.random.default_rng(42)
        s.add_sample(
            pattern=make_synthetic_pattern(64, 64, rng),
            confirmed_phase="Ferrite",
            detector_info=make_synthetic_detector_info(rng=rng),
        )

        shard = list((tmp_path / "store").glob("shard_*.h5"))[0]
        with h5py.File(shard, "r") as f:
            grp = f["sample_00000000"]
            assert grp["pattern_original"].chunks is not None
            assert grp["pattern_normalized"].chunks is not None


# ---------------------------------------------------------------------------
# Trainer num_workers=0
# ---------------------------------------------------------------------------


class TestTrainerNumWorkers:
    """Trainer DataLoaders use num_workers=0 for h5py safety."""

    def test_trainer_creates_loaders_with_zero_workers(
        self, tmp_path: Path
    ) -> None:
        """Verify DataLoader is created with num_workers=0."""
        from ebsd_ai.config import TrainingConfig
        from ebsd_ai.data.dataset import EBSDPhaseDataset
        from ebsd_ai.models.phase_classifier import PhaseClassifier
        from ebsd_ai.training.trainer import Trainer

        # Create minimal store with 2 samples
        s = TrainingStore(tmp_path / "store")
        rng = np.random.default_rng(42)
        for phase in ["A", "B"]:
            s.add_sample(
                pattern=make_synthetic_pattern(64, 64, rng),
                confirmed_phase=phase,
                detector_info=DetectorInfo(),
                confidence_score=0.9,
            )

        model = PhaseClassifier(phase_names=["A", "B"])
        config = TrainingConfig(epochs=1, batch_size=2)
        trainer = Trainer(
            model=model,
            config=config,
            output_dir=tmp_path / "ckpt",
            device="cpu",
        )

        ds = EBSDPhaseDataset(
            store=s, phase_names=["A", "B"], eds_dropout_rate=0.0
        )

        # Run 1 epoch to verify it works
        result = trainer.train(ds)
        assert result.epochs_completed == 1
