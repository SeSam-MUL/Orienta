"""Shared fixtures and synthetic data generators for tests.

All test modules use these fixtures to create synthetic EBSD data.
Since real EBSD data is not included in the repo, tests verify that
the pipeline works correctly with random data of the right shapes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ebsd_ai.config import (
    ELEMENT_SYMBOLS,
    NUM_ELEMENTS,
    DataSource,
    DetectorConvention,
    DetectorInfo,
    DetectorManufacturer,
    eds_dict_to_vector,
)


# ---------------------------------------------------------------------------
# Pattern fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(42)


def make_synthetic_pattern(
    height: int = 120,
    width: int = 160,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a random uint8 grayscale pattern.

    Parameters
    ----------
    height, width : int
        Pattern dimensions.
    rng : np.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    np.ndarray
        uint8 array of shape ``(height, width)``.
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.integers(0, 256, size=(height, width), dtype=np.uint8)


@pytest.fixture
def synthetic_pattern(rng: np.random.Generator) -> np.ndarray:
    """A single 120x160 synthetic pattern."""
    return make_synthetic_pattern(120, 160, rng)


@pytest.fixture(params=[(60, 60), (80, 60), (120, 120), (160, 120), (320, 240)])
def varied_pattern(request: pytest.FixtureRequest, rng: np.random.Generator) -> np.ndarray:
    """Synthetic patterns of various sizes found in real EBSD detectors."""
    h, w = request.param
    return make_synthetic_pattern(h, w, rng)


# ---------------------------------------------------------------------------
# EDS fixtures
# ---------------------------------------------------------------------------


def make_synthetic_eds(
    elements: list[str] | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Generate random EDS concentrations for given elements.

    Parameters
    ----------
    elements : list[str], optional
        Element symbols. Defaults to a steel-like composition.
    rng : np.random.Generator, optional
        Random generator.

    Returns
    -------
    dict[str, float]
        Element symbol → atomic percent.
    """
    if rng is None:
        rng = np.random.default_rng()
    if elements is None:
        elements = ["Fe", "C", "Cr", "Ni", "Mn"]
    concentrations = rng.dirichlet(np.ones(len(elements))) * 100.0
    return {el: float(c) for el, c in zip(elements, concentrations)}


@pytest.fixture
def synthetic_eds(rng: np.random.Generator) -> dict[str, float]:
    """Random steel-like EDS composition."""
    return make_synthetic_eds(rng=rng)


@pytest.fixture
def synthetic_eds_vector(synthetic_eds: dict[str, float]) -> np.ndarray:
    """EDS composition as a 92-element vector."""
    return eds_dict_to_vector(synthetic_eds)


# ---------------------------------------------------------------------------
# Detector info fixtures
# ---------------------------------------------------------------------------


def make_synthetic_detector_info(
    convention: DetectorConvention = DetectorConvention.OXFORD,
    rng: np.random.Generator | None = None,
) -> DetectorInfo:
    """Generate a plausible DetectorInfo with random PC values.

    Parameters
    ----------
    convention : DetectorConvention
        PC convention to use.
    rng : np.random.Generator, optional
        Random generator.

    Returns
    -------
    DetectorInfo
    """
    if rng is None:
        rng = np.random.default_rng()

    manufacturers = list(DetectorManufacturer)
    return DetectorInfo(
        manufacturer=manufacturers[rng.integers(0, len(manufacturers))],
        pc=(
            float(rng.uniform(0.3, 0.7)),
            float(rng.uniform(0.1, 0.5)),
            float(rng.uniform(0.4, 0.8)),
        ),
        pc_convention=convention,
        kv=float(rng.uniform(10, 25)),
        working_distance=float(rng.uniform(8, 20)),
        sample_tilt=float(rng.uniform(60, 75)),
    )


@pytest.fixture
def synthetic_detector_info(rng: np.random.Generator) -> DetectorInfo:
    """A single synthetic DetectorInfo (Oxford convention)."""
    return make_synthetic_detector_info(DetectorConvention.OXFORD, rng)


@pytest.fixture(params=list(DetectorConvention))
def detector_all_conventions(
    request: pytest.FixtureRequest, rng: np.random.Generator
) -> DetectorInfo:
    """DetectorInfo for every PC convention."""
    return make_synthetic_detector_info(request.param, rng)


# ---------------------------------------------------------------------------
# Batch / dataset fixtures
# ---------------------------------------------------------------------------


def make_synthetic_batch(
    n_samples: int = 32,
    n_phases: int = 3,
    pattern_h: int = 120,
    pattern_w: int = 160,
    include_eds: bool = True,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Generate a complete batch of synthetic training data.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_phases : int
        Number of distinct phases.
    pattern_h, pattern_w : int
        Pattern dimensions.
    include_eds : bool
        Whether to include EDS data. If False, eds_data is None and
        has_eds flags are all False.
    rng : np.random.Generator, optional
        Random generator.

    Returns
    -------
    dict
        Keys: ``patterns``, ``phase_ids``, ``phase_names``,
        ``orientations``, ``confidence_scores``, ``eds_data``,
        ``detector_info``, ``has_eds``.
    """
    if rng is None:
        rng = np.random.default_rng()

    phase_names = [f"Phase_{i}" for i in range(n_phases)]
    phase_ids = rng.integers(0, n_phases, size=n_samples)

    patterns = rng.integers(
        0, 256, size=(n_samples, pattern_h, pattern_w), dtype=np.uint8
    )
    orientations = rng.random((n_samples, 4)).astype(np.float32)
    # Normalize quaternions
    norms = np.linalg.norm(orientations, axis=1, keepdims=True)
    orientations = orientations / norms

    confidence_scores = rng.uniform(0.1, 0.95, size=n_samples).astype(np.float32)

    detector_info = make_synthetic_detector_info(rng=rng)

    if include_eds:
        elements = ["Fe", "C", "Cr", "Ni", "Mn", "Si", "Mo"]
        eds_data: dict[str, np.ndarray] | None = {}
        for el in elements:
            eds_data[el] = rng.uniform(0, 30, size=n_samples).astype(np.float32)
        has_eds_flags = np.ones(n_samples, dtype=bool)
    else:
        eds_data = None
        has_eds_flags = np.zeros(n_samples, dtype=bool)

    return {
        "patterns": patterns,
        "phase_ids": phase_ids,
        "phase_names": phase_names,
        "orientations": orientations,
        "confidence_scores": confidence_scores,
        "eds_data": eds_data,
        "detector_info": detector_info,
        "has_eds": has_eds_flags,
    }


@pytest.fixture
def synthetic_batch(rng: np.random.Generator) -> dict[str, Any]:
    """Batch of 32 samples with 3 phases and EDS."""
    return make_synthetic_batch(n_samples=32, n_phases=3, rng=rng)


@pytest.fixture
def synthetic_batch_no_eds(rng: np.random.Generator) -> dict[str, Any]:
    """Batch of 32 samples with 3 phases and NO EDS."""
    return make_synthetic_batch(
        n_samples=32, n_phases=3, include_eds=False, rng=rng
    )


# ---------------------------------------------------------------------------
# Training store fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_store_path(tmp_path: Path) -> Path:
    """Temporary directory for a TrainingStore."""
    store_dir = tmp_path / "training_data"
    store_dir.mkdir()
    return store_dir
