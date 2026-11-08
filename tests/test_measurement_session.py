"""The open HDF5 session reads the measurement the viewer holds: EDS maps, electron images,
Euler/Phase, headers, static backgrounds, Band Contrast, the pattern a fallback reads, the beam
energy and the step size. The shared session itself is left to the HDF5 viewer page.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from backend.api.services.load_choices import LoadChoiceError, LoadVariant
from tests.two_measurements import *  # noqa: F401,F403  (constants and fixtures)
from tests.two_measurements import _grid, _group, _xmap  # noqa: F401


@pytest.mark.parametrize("variant,g", BOTH, ids=BOTH_IDS)
def test_the_beam_energy_is_the_measurements(two, variant, g):
    from backend.api.routes.ebsd_viewer import _extract_beam_energy
    assert _extract_beam_energy(two, None, variant) == KV[g]


def test_the_step_size_fallback_reads_the_measurements_header(viewer, two):
    from backend.api.routes.ebsd_viewer import _extract_step_size
    viewer(V2)
    assert _extract_step_size(None, two) == {"x": STEP[2], "y": STEP[2], "units": "um"}
    viewer(None)
    assert _extract_step_size(None, two) == {"x": STEP[1], "y": STEP[1], "units": "um"}


@pytest.mark.parametrize("variant,g", [(None, 1), (V1, 1), (V2, 2)], ids=BOTH_IDS)
def test_the_sessions_extractor_reads_the_loaded_measurement(viewer, variant, g):
    """EDS maps, electron images, Euler/Phase, headers, static backgrounds and Band Contrast."""
    session = viewer(variant)
    ext = session.get_active_extractor()
    assert ext.root_key == str(g)
    np.testing.assert_array_equal(ext.get_element_map_2d("Fe"), _grid(FE[g]))
    np.testing.assert_array_equal(ext.get_electron_image("SE/Image 1"), np.array(SE[g]))
    np.testing.assert_array_equal(ext.get_phase_map(), np.full((NY, NX), PHASE[g]))
    np.testing.assert_array_equal(ext.get_band_contrast_map(), _grid(BC[g], np.float32))
    np.testing.assert_array_equal(ext.get_static_background("processed"), np.full((SY, SX), 3 + g))
    assert ext.get_eds_header()["beam_voltage_kV"] == KV[g]
    assert ext.get_pixel_sizes()["ebsd"]["x"] == STEP[g]


def test_the_default_dataset_gets_the_very_object_the_session_made(viewer):
    session = viewer(None)
    assert session.get_active_extractor() is session.get_extractor()
    session = viewer(V1)                       # the default measurement named in full: the same
    assert session.get_active_extractor() is session.get_extractor()


def test_the_other_measurement_does_not_change_the_shared_sessions_extractor(viewer):
    session = viewer(V2)
    shared = session.get_extractor()
    assert shared.root_key == "1"              # the HDF5 viewer page keeps what it opened
    assert session.get_active_extractor().root_key == "2"
    assert session.get_active_extractor() is session.get_active_extractor()      # made once


def test_a_session_on_another_file_is_not_given_the_viewers_measurement(viewer, tmp_path):
    session = viewer(V2)
    other = tmp_path / "other.h5oina"
    with h5py.File(other, "w") as f:
        f.create_dataset("Manufacturer", data=np.bytes_("Oxford Instruments"))
        _group(f, 1)
    session.open_file(str(other))              # the HDF5 viewer page opened some other file
    assert session.get_active_extractor().root_key == "1"


def test_a_dataset_whose_measurement_the_file_lost_is_refused_with_the_code(viewer):
    session = viewer(V7)
    with pytest.raises(LoadChoiceError) as e:
        session.get_active_extractor()
    assert e.value.code == "variantNotSupported"


@pytest.mark.parametrize("variant,expected", [
    (None, (10, 10)),                       # processed of measurement 1: all 10
    (V2, (20, 20)),                         # processed of measurement 2: all 20
    (V2_RAW, (1300, 1303)),                 # unprocessed of measurement 2: 300 + 1000 + 3k
], ids=["default", "measurement-2", "measurement-2-unprocessed"])
def test_the_pattern_a_fallback_reads_is_the_loaded_source(viewer, variant, expected):
    session = viewer(variant)
    pattern = session.get_loaded_pattern(0, "processed")      # the caller's own choice is the old default
    assert (int(pattern[0, 0]), int(pattern[0, 1])) == expected


def test_crystal_hints_raw_fallback_reads_the_loaded_source(viewer, monkeypatch):
    from backend.api.routes import crystal_hint
    viewer(V2_RAW)
    monkeypatch.setattr(crystal_hint, "_fetch_active_pattern", lambda idx: None)
    p = crystal_hint._region_pattern(1, "processed")
    assert (int(p[0, 0]), int(p[0, 1])) == (1000 + 300 + 3 * SY * SX, 1000 + 300 + 3 * SY * SX + 3)
    neighbours = crystal_hint._collect_neighbour_patterns(1, 0, "processed")
    assert int(neighbours[0][0, 0]) == int(p[0, 0])


def test_crystal_hints_detector_geometry_states_the_loaded_measurements_voltage(viewer):
    from backend.api.routes.crystal_hint import _gather_detector_geom
    det = SimpleNamespace(pc=np.array([0.5, 0.5, 0.7]))
    signal = SimpleNamespace(axes_manager=SimpleNamespace(signal_shape=(SX, SY)), detector=det)
    for variant, g in ((None, 1), (V2, 2)):
        viewer(variant)
        with patch("backend.api.routes.ebsd_viewer._get_active_signal", return_value=signal), \
             patch("backend.api.services.calibration_store.calibration_store.get_detector",
                   return_value=det):
            assert _gather_detector_geom()["voltage_kv"] == KV[g], g
