"""Cross-file Band Contrast: the phase map must be able to read BC from a
result's OWN source file via a private handle, without disturbing the shared
EDS h5_session.

Covers ``virtual_images._read_native_band_contrast_from`` (2026-06-03 fix for
"No Band Contrast available ... no source H5OINA open" when viewing a result
whose source isn't the currently-open file).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TEST_DATA = ROOT / "Test_data"
OXFORD = TEST_DATA / "1.h5oina"          # Oxford → has native Band Contrast
EDAX = TEST_DATA / "HiGainNi.h5"          # EDAX → no Band Contrast dataset


def test_reader_returns_none_on_missing_path():
    from backend.api.routes.virtual_images import _read_native_band_contrast_from
    assert _read_native_band_contrast_from(None) is None
    assert _read_native_band_contrast_from("") is None
    assert _read_native_band_contrast_from(str(TEST_DATA / "does_not_exist.h5oina")) is None


@pytest.mark.skipif(not EDAX.exists(), reason="EDAX test file not present")
def test_reader_returns_none_for_edax_without_bc():
    """An EDAX .h5 with no Band Contrast dataset must return None (not crash) —
    this is the user's actual failing case; it correctly has no BC."""
    from backend.api.routes.virtual_images import _read_native_band_contrast_from
    assert _read_native_band_contrast_from(str(EDAX)) is None


@pytest.mark.skipif(not OXFORD.exists(), reason="Oxford h5oina test file not present")
def test_reader_reads_bc_from_oxford_independently():
    """Reads native BC from an Oxford file with a PRIVATE handle, and does NOT
    open/leave the shared EDS h5_session pointing at it."""
    from backend.api.routes.virtual_images import _read_native_band_contrast_from
    from backend.api.services import h5_session

    was_open = h5_session.is_open()
    prev_path = h5_session.get_current_path() if was_open else None

    bc = _read_native_band_contrast_from(str(OXFORD))
    assert bc is not None, "Oxford file should carry native Band Contrast"
    assert bc.ndim == 2 and bc.size > 0
    assert np.isfinite(bc).any()

    # The shared session must be untouched by the private read.
    assert h5_session.is_open() == was_open
    if was_open:
        assert h5_session.get_current_path() == prev_path


def test_band_contrast_computed_label():
    """When no native BC exists, the endpoint labels the computed metric
    honestly via the central quality service (not the old kikuchipy string)."""
    from backend.api.services.pattern_quality import LABEL_COMPUTED
    assert LABEL_COMPUTED == "Pattern Quality (computed)"
