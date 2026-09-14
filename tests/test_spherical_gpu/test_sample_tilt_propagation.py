"""Regression tests for the ``sample_tilt`` propagation fix (2026-05-22).

Historical bug
--------------
``backend/api/routes/indexing.py`` puts the calibration's experimental
sample tilt into ``det_params["sample_tilt"]`` (line ~615). But
``Tier1Indexer.__init__``'s previous priority order was

    explicit sample_tilt_deg arg > master.primary_tilt_deg > 70.0

— it silently fell through to ``master.primary_tilt_deg`` (the MC
simulation tilt metadata, typically 70 deg) whenever the caller did not
pass an explicit ``sample_tilt_deg`` to ``PhaseConfig``. The production
``spherical_gpu_index_patterns`` did NOT pass it. Result: on samples
captured at a tilt other than 70 deg (LoGainNi: 75.7 deg) the recovered
orientation had a fixed rotation offset equal to the gap.

Empirical magnitude verified by ``tasks/_kikuchipy_oracle_roundtrip.py``:
median symmetry-reduced disorientation jumped from 5.80 deg (before fix)
to 1.39 deg (after fix), with the best pixels at 0.11 deg — the sub-bin
quantization floor at L=88.

Fix
---
1. ``DetectorGeometry`` gained an optional ``sample_tilt_deg`` field,
   populated by ``DetectorGeometry.from_params`` from
   ``params["sample_tilt"]``.
2. ``Tier1Indexer.__init__`` priority order changed to

    explicit sample_tilt_deg arg
    > geom.sample_tilt_deg     (= det_params["sample_tilt"])
    > master.primary_tilt_deg  (last-resort fallback for tests that
                                don't propagate the calibration)
    > 70.0

These tests pin the new behaviour so a future "cleanup" can't
accidentally re-introduce the silent fallback.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from backend.spherical_gpu.pipeline.detector import DetectorGeometry
from backend.spherical_gpu.pipeline.indexer import Tier1Indexer
from backend.spherical_gpu.pipeline.sht_io import read_sht_master


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _base_det_params(**overrides) -> dict:
    """Minimal det_params dict accepted by DetectorGeometry.from_params."""
    p = dict(
        pc_x=0.5, pc_y=0.5, pc_z=0.6,
        pat_width=60, pat_height=60,
        pixel_size=55.0, tilt=10.0, binning=1,
        vendor="Bruker",
    )
    p.update(overrides)
    return p


# -----------------------------------------------------------------------------
# Unit tests on DetectorGeometry
# -----------------------------------------------------------------------------

def test_detector_geometry_reads_sample_tilt():
    """DetectorGeometry.from_params picks up params['sample_tilt']."""
    params = _base_det_params(sample_tilt=75.7)
    geom = DetectorGeometry.from_params(params, device=_device())
    assert geom.sample_tilt_deg == pytest.approx(75.7, abs=1e-6)


def test_detector_geometry_sample_tilt_missing():
    """When params has no 'sample_tilt', geom.sample_tilt_deg is None.

    Preserves backward compatibility with callers that don't propagate
    the calibration (older tests, debug scripts).
    """
    params = _base_det_params()  # no sample_tilt key
    geom = DetectorGeometry.from_params(params, device=_device())
    assert geom.sample_tilt_deg is None


def test_detector_geometry_accepts_float_or_int():
    """sample_tilt can be either int or float; geom stores as float."""
    params = _base_det_params(sample_tilt=70)  # int
    geom = DetectorGeometry.from_params(params, device=_device())
    assert isinstance(geom.sample_tilt_deg, float)
    assert geom.sample_tilt_deg == 70.0


# -----------------------------------------------------------------------------
# Tier1Indexer priority-order tests (use the oracle's embedded SHT)
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def master_loaded(oracle, tmp_path_factory):
    """Materialise the embedded SHT bytes and load the SHT master once."""
    tmp = tmp_path_factory.mktemp("sht_for_sample_tilt_tests")
    sht_filename = oracle["embedded_sht_filename"][()]
    if isinstance(sht_filename, bytes):
        sht_filename = sht_filename.decode("utf-8")
    target = tmp / sht_filename
    target.write_bytes(bytes(oracle["embedded_sht_bytes"][()]))
    return read_sht_master(str(target), device=_device())


def _build_indexer(
    *, master, sample_tilt_in_params=None, sample_tilt_explicit=None,
):
    """Build a minimal Tier1Indexer for priority-order assertions."""
    params = _base_det_params()
    if sample_tilt_in_params is not None:
        params["sample_tilt"] = sample_tilt_in_params
    geom = DetectorGeometry.from_params(params, device=_device())
    return Tier1Indexer(
        geom=geom,
        master=master,
        device=_device(),
        bandwidth=32,                          # small; cheap to construct
        sample_tilt_deg=sample_tilt_explicit,
    )


def test_priority_explicit_arg_wins(master_loaded):
    """Priority 1: explicit sample_tilt_deg arg wins over geom + master."""
    ix = _build_indexer(
        master=master_loaded,
        sample_tilt_in_params=80.0,
        sample_tilt_explicit=65.0,
    )
    assert ix.sample_tilt_deg == pytest.approx(65.0, abs=1e-6)


def test_priority_geom_wins_over_master(master_loaded):
    """Priority 2 (the bug fix): geom.sample_tilt_deg wins over master.

    With the OLD priority order, this would have returned
    master_loaded.primary_tilt_deg (typically 70 deg on EMsoft-generated
    masters), not 75.7. The 5.7-degree gap was the production bug.
    """
    master_tilt = master_loaded.primary_tilt_deg
    custom = master_tilt + 5.7  # something that demonstrably differs
    ix = _build_indexer(
        master=master_loaded,
        sample_tilt_in_params=custom,
        sample_tilt_explicit=None,
    )
    assert ix.sample_tilt_deg == pytest.approx(custom, abs=1e-6)
    assert ix.sample_tilt_deg != pytest.approx(master_tilt, abs=1e-6), (
        "FIX-REGRESSION: Tier1Indexer fell through to master.primary_tilt_deg "
        "when geom.sample_tilt_deg was set."
    )


def test_priority_master_fallback_preserved(master_loaded):
    """Priority 3: master.primary_tilt_deg used when nothing else is set.

    Preserves backward compatibility for unit tests + debug scripts that
    build a det_params dict without the 'sample_tilt' key.
    """
    ix = _build_indexer(
        master=master_loaded,
        sample_tilt_in_params=None,            # no det_params['sample_tilt']
        sample_tilt_explicit=None,             # no explicit override
    )
    assert ix.sample_tilt_deg == pytest.approx(
        master_loaded.primary_tilt_deg, abs=1e-6,
    )


def test_priority_geom_overrides_master_when_equal_value_chosen(master_loaded):
    """When sample_tilt happens to equal master.primary_tilt_deg, both
    sources resolve to the same number — but the chosen sigma_deg must
    still come from geom (the explicit calibration), not the master."""
    master_tilt = master_loaded.primary_tilt_deg
    ix = _build_indexer(
        master=master_loaded,
        sample_tilt_in_params=master_tilt,
        sample_tilt_explicit=None,
    )
    # Same numerical answer either way, but the geom path is the canonical
    # source. We can't observe the code path directly, only the result.
    assert ix.sample_tilt_deg == pytest.approx(master_tilt, abs=1e-6)
