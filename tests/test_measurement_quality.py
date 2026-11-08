"""Band Contrast and the image-quality channel follow the measurement that was loaded, or the one a
result is stamped with: the quality readers, the Phase Map layers, the overview, the analysis
bridge, the add-on context. For a second measurement with the SAME grid nothing complained before,
and measurement 1's map was drawn as measurement 2's.
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
def test_native_band_contrast_is_the_measurements(two, variant, g):
    from backend.api.services.pattern_quality import read_native_band_contrast
    got = read_native_band_contrast(two, NY, NX, variant)
    np.testing.assert_array_equal(got, _grid(BC[g]))


@pytest.mark.parametrize("variant,g", BOTH, ids=BOTH_IDS)
def test_native_image_quality_is_the_measurements(two, variant, g):
    from backend.api.services.pattern_quality import read_native_image_quality
    got = read_native_image_quality(two, NY, NX, variant)
    np.testing.assert_allclose(got, _grid(IQ[g]), rtol=1e-6)


@pytest.mark.parametrize("variant,g", BOTH, ids=BOTH_IDS)
def test_the_quality_map_is_the_measurements_and_says_native(two, variant, g):
    from backend.api.services.pattern_quality import LABEL_NATIVE, get_quality_map
    qm = get_quality_map(NY, NX, source_file=two, variant=variant)
    assert qm.label == LABEL_NATIVE and qm.source == "native"
    np.testing.assert_array_equal(qm.array, _grid(BC[g]))
    assert qm.value_range == (0, 255)


def test_a_measurement_without_band_contrast_has_none_not_the_first_ones(tmp_path):
    from backend.api.services.pattern_quality import read_native_band_contrast
    path = tmp_path / "half.h5oina"
    with h5py.File(path, "w") as f:
        _group(f, 1)
        _group(f, 2)
        del f["2/EBSD/Data/Band Contrast"]
    assert read_native_band_contrast(str(path), NY, NX, V2) is None
    assert read_native_band_contrast(str(path), NY, NX, V1) is not None


@pytest.mark.parametrize("reader", ["read_native_band_contrast", "read_native_image_quality"])
def test_a_measurement_the_file_does_not_hold_is_refused_with_the_code(two, reader):
    from backend.api.services import pattern_quality
    with pytest.raises(LoadChoiceError) as e:
        getattr(pattern_quality, reader)(two, NY, NX, V7)
    assert e.value.code == "variantNotSupported"
    assert "7" in str(e.value) and "two.h5oina" in str(e.value)


@pytest.mark.parametrize("variant,g", BOTH, ids=BOTH_IDS)
def test_the_open_files_band_contrast_is_the_measurements(session, variant, g):
    from backend.api.routes.virtual_images import _read_native_band_contrast
    np.testing.assert_array_equal(_read_native_band_contrast(variant), _grid(BC[g]))


@pytest.mark.parametrize("variant,g", BOTH, ids=BOTH_IDS)
def test_band_contrast_read_from_a_path_is_the_measurements(two, variant, g):
    from backend.api.routes.virtual_images import _read_native_band_contrast_from
    np.testing.assert_array_equal(_read_native_band_contrast_from(two, variant), _grid(BC[g]))


@pytest.mark.parametrize("variant,g", BOTH, ids=BOTH_IDS)
def test_the_stored_overview_map_is_the_measurements(two, variant, g):
    from backend.api.routes.ebsd_viewer import _read_stored_quality_map
    np.testing.assert_array_equal(_read_stored_quality_map(two, "bc", NY, NX, variant), _grid(BC[g]))


def test_the_quality_layer_overlay_of_a_stamped_result_reads_its_measurement(session):
    """The Phase Map BC layer: the result's own stamp decides, with the shared session as fallback."""
    from unittest.mock import patch
    from backend.api.routes.phase_map import _compute_layer_rgba
    from tests.test_phase_map_layer_endpoint import _make_fake_result
    for stamp, g in ((None, 1), ({"measurement": "2", "pattern_source": "processed"}, 2)):
        fake = _make_fake_result(n_rows=NY, n_cols=NX, with_bc=False)
        fake.metadata = {"source_file": None, **({} if stamp is None else {"load_variant": stamp})}
        scale = {}
        with patch("backend.api.routes.phase_map.get_last_indexing_result", return_value=fake), \
             patch("backend.api.routes.phase_map.get_analysis_dataset", return_value=None):
            _compute_layer_rgba(kind="bc", scale_out=scale)
        assert (scale["min"], scale["max"]) == (min(BC[g]), max(BC[g])), g


def test_the_quality_layer_of_a_stamped_result_reads_its_own_source_file(two):
    from unittest.mock import patch
    from backend.api.routes.phase_map import _compute_layer_rgba
    from tests.test_phase_map_layer_endpoint import _make_fake_result
    for stamp, g in ((None, 1), ({"measurement": "2", "pattern_source": "processed"}, 2),
                     # a result imported from an export says so under its own key
                     ({"measurement": "2", "pattern_source": "unprocessed"}, 2)):
        fake = _make_fake_result(n_rows=NY, n_cols=NX, with_bc=False)
        fake.metadata = {"source_file": two}
        if stamp is not None:
            fake.metadata["indexed_variant"] = stamp
        scale = {}
        with patch("backend.api.routes.phase_map.get_last_indexing_result", return_value=fake), \
             patch("backend.api.routes.phase_map.get_analysis_dataset", return_value=None):
            _compute_layer_rgba(kind="bc", scale_out=scale)
        assert (scale["min"], scale["max"]) == (min(BC[g]), max(BC[g])), g


def test_the_scalar_probe_of_a_stamped_result_reads_its_measurement(session):
    from unittest.mock import patch
    from backend.api.routes.phase_map import _get_scalar_array_2d
    from tests.test_phase_map_layer_endpoint import _make_fake_result
    for stamp, g in ((None, 1), ({"measurement": "2", "pattern_source": "processed"}, 2)):
        fake = _make_fake_result(n_rows=NY, n_cols=NX, with_bc=False)
        fake.metadata = {} if stamp is None else {"load_variant": stamp}
        with patch("backend.api.routes.phase_map.get_last_indexing_result", return_value=fake), \
             patch("backend.api.routes.phase_map.get_analysis_dataset", return_value=None):
            got = _get_scalar_array_2d("bc")
        np.testing.assert_array_equal(got, _grid(BC[g])), g


def test_the_band_contrast_route_follows_the_viewers_dataset(viewer):
    """GET /band-contrast: the open file's native Band Contrast of the dataset the viewer holds."""
    from backend.api.routes.virtual_images import _native_band_contrast_for_active_dataset
    viewer(V2)
    np.testing.assert_array_equal(_native_band_contrast_for_active_dataset(), _grid(BC[2]))
    viewer(None)
    np.testing.assert_array_equal(_native_band_contrast_for_active_dataset(), _grid(BC[1]))


def test_the_eds_probes_band_contrast_follows_the_viewers_dataset(viewer):
    from backend.api.routes.eds import _read_native_bc_safe
    viewer(V2)
    np.testing.assert_array_equal(_read_native_bc_safe(), _grid(BC[2]))


def test_the_addon_context_quality_is_the_results_measurement(two):
    from backend.api.routes.addons import _context_extras
    for meta, g in (({"source_file": two}, 1),
                    ({"source_file": two, "load_variant": V2.to_dict()}, 2)):
        result = SimpleNamespace(original_shape=(NY, NX), metadata=meta, xmap=None)
        extras = _context_extras(result)
        np.testing.assert_array_equal(extras["quality"], _grid(BC[g])), g
        assert extras["quality_source"] == "Band Contrast (native)"


def test_the_analysis_load_bridges_the_results_measurements_band_contrast(two, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.api.main import app
    import backend.api.routes.analysis as ar
    import backend.api.routes.indexing as ix
    from tests.test_analysis_routes_fail_loud import _FakeResult, _make_bcless_xmap

    saved = ar._dataset
    try:
        for stamp, g in ((None, 1), (V2.to_dict(), 2)):
            result = _FakeResult(_make_bcless_xmap(NY, NX), source_file=two)
            if stamp is not None:
                result.metadata["load_variant"] = stamp
            monkeypatch.setattr(ix, "get_last_indexing_result", lambda r=result: r)
            ar._dataset = None
            resp = TestClient(app).post("/api/analysis/load", json={"xmap_path": "__from_indexing__"})
            assert resp.status_code == 200, resp.text
            np.testing.assert_array_equal(ar._dataset.xmap.prop["bc"], np.array(BC[g], np.float32))
    finally:
        ar._dataset = saved
