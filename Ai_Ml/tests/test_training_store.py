"""Tests for ebsd_ai.data.training_store module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ebsd_ai.config import (
    NUM_ELEMENTS,
    DataSource,
    DetectorConvention,
    DetectorInfo,
    DetectorManufacturer,
)
from ebsd_ai.data.training_store import TrainingStore
from tests.conftest import (
    make_synthetic_batch,
    make_synthetic_detector_info,
    make_synthetic_eds,
    make_synthetic_pattern,
)


@pytest.fixture
def store(tmp_path: Path) -> TrainingStore:
    """Empty TrainingStore in a temp directory."""
    return TrainingStore(tmp_path / "store")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


class TestAddSample:
    def test_single_with_eds(
        self, store: TrainingStore, rng: np.random.Generator
    ) -> None:
        pat = make_synthetic_pattern(80, 60, rng)
        eds = make_synthetic_eds(rng=rng)
        det = make_synthetic_detector_info(rng=rng)

        store.add_sample(
            pattern=pat,
            confirmed_phase="Ferrite",
            detector_info=det,
            eds_data=eds,
        )
        assert len(store) == 1

    def test_single_without_eds(
        self, store: TrainingStore, rng: np.random.Generator
    ) -> None:
        pat = make_synthetic_pattern(120, 120, rng)
        det = make_synthetic_detector_info(rng=rng)

        store.add_sample(
            pattern=pat,
            confirmed_phase="Austenite",
            detector_info=det,
            eds_data=None,
        )
        assert len(store) == 1
        sample = store.get_sample(0)
        assert sample["has_eds"] is False
        assert np.all(sample["eds_atomic_pct"] == 0)

    def test_roundtrip_read(
        self, store: TrainingStore, rng: np.random.Generator
    ) -> None:
        pat = make_synthetic_pattern(100, 100, rng)
        eds = {"Fe": 65.0, "C": 8.0}
        det = DetectorInfo(
            manufacturer=DetectorManufacturer.OXFORD,
            pc=(0.5, 0.3, 0.6),
            pc_convention=DetectorConvention.OXFORD,
            kv=20.0,
            working_distance=15.0,
            sample_tilt=70.0,
        )
        store.add_sample(
            pattern=pat,
            confirmed_phase="Ferrite",
            detector_info=det,
            eds_data=eds,
            confidence_score=0.85,
            data_source=DataSource.USER_CONFIRMED,
        )
        sample = store.get_sample(0)
        assert sample["confirmed_phase"] == "Ferrite"
        assert sample["has_eds"] is True
        assert sample["confidence_score"] == pytest.approx(0.85, abs=0.01)
        assert sample["data_source"] == "USER_CONFIRMED"
        assert sample["pattern_normalized"].shape == (128, 128)
        assert sample["eds_atomic_pct"].shape == (NUM_ELEMENTS,)
        assert sample["detector_info"].manufacturer == DetectorManufacturer.OXFORD


class TestAddBatch:
    def test_batch_with_eds(
        self, store: TrainingStore, rng: np.random.Generator
    ) -> None:
        batch = make_synthetic_batch(n_samples=10, n_phases=3, rng=rng)
        det = batch["detector_info"]
        phases = [batch["phase_names"][pid] for pid in batch["phase_ids"]]

        count = store.add_batch(
            patterns=batch["patterns"],
            confirmed_phases=phases,
            detector_info=det,
            orientations=batch["orientations"],
            confidence_scores=batch["confidence_scores"],
            eds_data=batch["eds_data"],
        )
        assert count == 10
        assert len(store) == 10

    def test_batch_without_eds(
        self, store: TrainingStore, rng: np.random.Generator
    ) -> None:
        batch = make_synthetic_batch(
            n_samples=5, n_phases=2, include_eds=False, rng=rng
        )
        phases = [batch["phase_names"][pid] for pid in batch["phase_ids"]]

        count = store.add_batch(
            patterns=batch["patterns"],
            confirmed_phases=phases,
            detector_info=batch["detector_info"],
            eds_data=None,
        )
        assert count == 5

        for sample in store.iter_samples():
            assert sample["has_eds"] is False


class TestAddFromIndexing:
    def test_ci_filtering(
        self, store: TrainingStore, rng: np.random.Generator
    ) -> None:
        n = 20
        batch = make_synthetic_batch(n_samples=n, n_phases=3, rng=rng)
        # Set half to high CI, half to low CI
        ci = np.zeros(n, dtype=np.float32)
        ci[:10] = 0.8  # Above threshold
        ci[10:] = 0.1  # Below threshold

        count = store.add_from_indexing(
            patterns=batch["patterns"],
            phase_ids=batch["phase_ids"],
            phase_names=batch["phase_names"],
            orientations=batch["orientations"],
            confidence_scores=ci,
            detector_info=batch["detector_info"],
            ci_threshold=0.3,
            eds_data=batch["eds_data"],
        )
        assert count == 10
        assert len(store) == 10

    def test_all_below_threshold(
        self, store: TrainingStore, rng: np.random.Generator
    ) -> None:
        batch = make_synthetic_batch(n_samples=5, n_phases=2, rng=rng)
        ci = np.full(5, 0.1, dtype=np.float32)

        count = store.add_from_indexing(
            patterns=batch["patterns"],
            phase_ids=batch["phase_ids"],
            phase_names=batch["phase_names"],
            orientations=batch["orientations"],
            confidence_scores=ci,
            detector_info=batch["detector_info"],
            ci_threshold=0.3,
        )
        assert count == 0
        assert len(store) == 0

    def test_auto_indexed_source(
        self, store: TrainingStore, rng: np.random.Generator
    ) -> None:
        batch = make_synthetic_batch(n_samples=5, n_phases=2, rng=rng)
        ci = np.full(5, 0.9, dtype=np.float32)

        store.add_from_indexing(
            patterns=batch["patterns"],
            phase_ids=batch["phase_ids"],
            phase_names=batch["phase_names"],
            orientations=batch["orientations"],
            confidence_scores=ci,
            detector_info=batch["detector_info"],
        )
        sample = store.get_sample(0)
        assert sample["data_source"] == "AUTO_INDEXED"


class TestIterAndStats:
    def test_iter_all_samples(
        self, store: TrainingStore, rng: np.random.Generator
    ) -> None:
        batch = make_synthetic_batch(n_samples=8, n_phases=2, rng=rng)
        phases = [batch["phase_names"][pid] for pid in batch["phase_ids"]]
        store.add_batch(
            patterns=batch["patterns"],
            confirmed_phases=phases,
            detector_info=batch["detector_info"],
            eds_data=batch["eds_data"],
        )
        samples = list(store.iter_samples())
        assert len(samples) == 8

    def test_get_sample_out_of_range(self, store: TrainingStore) -> None:
        with pytest.raises(IndexError):
            store.get_sample(0)

    def test_stats_with_mixed_eds(
        self, store: TrainingStore, rng: np.random.Generator
    ) -> None:
        # Add samples with EDS
        batch_eds = make_synthetic_batch(
            n_samples=5, n_phases=2, include_eds=True, rng=rng
        )
        phases_eds = [
            batch_eds["phase_names"][pid] for pid in batch_eds["phase_ids"]
        ]
        store.add_batch(
            patterns=batch_eds["patterns"],
            confirmed_phases=phases_eds,
            detector_info=batch_eds["detector_info"],
            eds_data=batch_eds["eds_data"],
        )

        # Add samples without EDS
        batch_no = make_synthetic_batch(
            n_samples=3, n_phases=2, include_eds=False, rng=rng
        )
        phases_no = [
            batch_no["phase_names"][pid] for pid in batch_no["phase_ids"]
        ]
        store.add_batch(
            patterns=batch_no["patterns"],
            confirmed_phases=phases_no,
            detector_info=batch_no["detector_info"],
            eds_data=None,
        )

        stats = store.get_dataset_stats()
        assert stats["total_samples"] == 8
        assert stats["samples_with_eds"] == 5
        assert stats["samples_without_eds"] == 3

    def test_phase_names(
        self, store: TrainingStore, rng: np.random.Generator
    ) -> None:
        det = make_synthetic_detector_info(rng=rng)
        pat = make_synthetic_pattern(60, 60, rng)

        store.add_sample(pat, "Ferrite", det)
        store.add_sample(pat, "Austenite", det)
        store.add_sample(pat, "Ferrite", det)

        names = store.get_phase_names()
        assert names == ["Austenite", "Ferrite"]

    def test_empty_store_stats(self, store: TrainingStore) -> None:
        stats = store.get_dataset_stats()
        assert stats["total_samples"] == 0
        assert stats["mean_confidence"] == 0.0


class TestSharding:
    def test_shard_creation(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        store = TrainingStore(tmp_path / "sharded", samples_per_shard=5)
        det = make_synthetic_detector_info(rng=rng)

        for i in range(12):
            pat = make_synthetic_pattern(60, 60, rng)
            store.add_sample(pat, f"Phase_{i % 3}", det)

        assert len(store) == 12
        shards = store._shard_files()
        assert len(shards) == 3  # 5 + 5 + 2
