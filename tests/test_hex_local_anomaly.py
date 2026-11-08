"""The local-anomaly map (ncc minus the median of its neighbourhood) on a hexagonal scan."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import forward_diagnostics as fd  # noqa: E402
from backend.api.services.forward_diagnostics import compute_local_anomaly_map  # noqa: E402
from backend.api.services.scan_geometry import ScanGeometry  # noqa: E402
from backend.api.services.scan_neighbours import Topology  # noqa: E402
from tests.test_det_tilt_propagation import _batch_renderer, _fake_result, _pattern  # noqa: E402

SIGNS = (0.5, -0.5)
N_ROWS, N_COLS = 8, 7


def _distances(shift):
    g = ScanGeometry.hex(N_ROWS, N_COLS, 1.0, odd_row_shift=shift)
    x, y = g.positions()
    pts = np.stack([x.ravel(), y.ravel()], axis=1)
    return np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])


def _oracle(ncc, shift, radius):
    """ncc minus the median over the points within ``radius`` steps on the sample (NaN ignored, border clipped)."""
    dist = _distances(shift)
    flat = ncc.ravel()
    out = np.full(flat.size, np.nan)
    for p in range(flat.size):
        window = flat[dist[p] <= radius + 1e-6]
        window = window[~np.isnan(window)]
        if window.size:
            out[p] = flat[p] - np.median(window)
    return out.reshape(ncc.shape)


@pytest.mark.parametrize("window, radius", [(5, 2.0), (3, 1.0)])
@pytest.mark.parametrize("shift", SIGNS)
def test_the_median_is_taken_over_the_points_within_the_disc_on_the_sample(shift, window, radius):
    rng = np.random.default_rng(1)
    ncc = rng.random((N_ROWS, N_COLS)).astype(np.float32)
    ncc[rng.random(ncc.shape) < 0.15] = np.nan                  # unindexed points
    got = compute_local_anomaly_map(ncc, window=window, topology=Topology("hex", N_ROWS, N_COLS, shift))
    want = _oracle(ncc, shift, radius)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, want, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("block", [1, 5, 56, 57])
def test_the_blocks_the_median_is_taken_in_do_not_change_a_number(monkeypatch, block):
    # 56 points: a block of 5 leaves a last block of one point, 56 and 57 are one block (the oracle is by hand)
    rng = np.random.default_rng(1)
    ncc = rng.random((N_ROWS, N_COLS)).astype(np.float32)
    ncc[rng.random(ncc.shape) < 0.15] = np.nan
    ncc[0, 0] = ncc[-1, -1] = 0.25                  # the first and the last point of the first and last block count
    monkeypatch.setattr(fd, "_LOCAL_ANOMALY_BLOCK", block)
    got = compute_local_anomaly_map(ncc, topology=Topology("hex", N_ROWS, N_COLS, 0.5))
    np.testing.assert_allclose(got, _oracle(ncc, 0.5, 2.0), atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("shift", SIGNS)
def test_a_window_of_five_is_the_nineteen_points_within_two_steps_and_no_other(shift):
    # Centre 0; its six neighbours 1; of the twelve points in the second ring nine are 0 and three are 1; every
    # point outside is 1. The median of the nineteen is 0 (ten zeros, nine ones). A window of one ring sees six
    # ones and a zero (median 1), a window of three rings sees mostly ones (median 1): only two rings give 0.
    dist = _distances(shift)
    centre = 3 * N_COLS + 3
    ring1 = [p for p in range(N_ROWS * N_COLS) if 1e-9 < dist[centre, p] <= 1 + 1e-6]
    ring2 = [p for p in range(N_ROWS * N_COLS) if 1 + 1e-6 < dist[centre, p] <= 2 + 1e-6]
    assert (len(ring1), len(ring2)) == (6, 12)
    ncc = np.ones(N_ROWS * N_COLS, dtype=np.float32)
    ncc[centre] = 0.0
    ncc[ring2[:9]] = 0.0
    got = compute_local_anomaly_map(ncc.reshape(N_ROWS, N_COLS), window=5,
                                    topology=Topology("hex", N_ROWS, N_COLS, shift))
    assert got.ravel()[centre] == 0.0


def test_a_lone_outlier_is_anomalous_and_a_uniform_field_is_not():
    topology = Topology("hex", N_ROWS, N_COLS, 0.5)
    flat = np.full((N_ROWS, N_COLS), 0.7, dtype=np.float32)
    np.testing.assert_allclose(compute_local_anomaly_map(flat, topology=topology), 0.0, atol=1e-6)
    flat[4, 3] = 0.0
    assert compute_local_anomaly_map(flat, topology=topology)[4, 3] == pytest.approx(-0.7, abs=1e-6)


def test_a_nan_point_and_an_all_nan_window_stay_nan_without_a_warning(recwarn):
    topology = Topology("hex", N_ROWS, N_COLS, 0.5)
    ncc = np.full((N_ROWS, N_COLS), np.nan, dtype=np.float32)
    ncc[0, 0] = 0.5
    got = compute_local_anomaly_map(ncc, topology=topology)
    assert got[0, 0] == 0.0                                      # its own window holds only itself
    assert np.isnan(got[0, 1])                                   # a NaN point stays NaN
    assert np.isnan(got[7, 6])                                   # nothing indexed within two steps
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_a_map_smaller_than_the_window_is_zeros_as_before():
    got = compute_local_anomaly_map(np.full((3, 3), 0.5, dtype=np.float32), topology=Topology("hex", 3, 3, 0.5))
    np.testing.assert_array_equal(got, 0.0)


def test_a_square_topology_agrees_with_the_square_filter_away_from_the_border():
    # scipy's filter reflects at the border and the topology clips; the two meet where the window is complete
    ncc = np.random.default_rng(2).random((9, 8)).astype(np.float32)
    old = compute_local_anomaly_map(ncc)
    new = compute_local_anomaly_map(ncc, topology=Topology("square", 9, 8))
    np.testing.assert_allclose(new[2:-2, 2:-2], old[2:-2, 2:-2], atol=1e-6)


# ----------------------------------------------- the pass hands the result's topology on

@pytest.mark.parametrize("stamp, hexagonal", [
    (ScanGeometry.hex(2, 2, 0.5).to_dict(), True),
    (ScanGeometry.square(2, 2, 0.5).to_dict(), False),
    (None, False),
], ids=["hex", "square", "unknown"])
def test_the_diagnostics_pass_computes_the_anomaly_on_the_results_topology(monkeypatch, stamp, hexagonal):
    from unittest.mock import MagicMock, patch
    result = _fake_result()
    if stamp is not None:
        result.metadata["scan_geometry"] = stamp
    seen = []
    original = fd.compute_local_anomaly_map

    def spy(ncc, window=5, topology=None):
        seen.append(topology)
        return original(ncc, window=window, topology=topology)

    monkeypatch.setattr(fd, "compute_local_anomaly_map", spy)
    with patch.object(fd, "load_or_get_phase", return_value=MagicMock()), \
         patch.object(fd, "get_renderer", return_value=_batch_renderer()), \
         patch.object(fd, "_get_experimental_pattern", return_value=_pattern(0)):
        fd.compute_full_diagnostics(result, max_bandwidth=64)
    assert len(seen) == 1
    if hexagonal:
        assert (seen[0].kind, seen[0].n_rows, seen[0].n_cols, seen[0].odd_row_shift) == ("hex", 2, 2, 0.5)
    else:
        assert seen[0] is None


def test_a_stamp_of_another_grid_is_refused_before_any_rendering():
    from unittest.mock import MagicMock, patch
    result = _fake_result()
    result.metadata["scan_geometry"] = ScanGeometry.hex(3, 3, 0.5).to_dict()
    renderer = _batch_renderer()
    with patch.object(fd, "load_or_get_phase", return_value=MagicMock()), \
         patch.object(fd, "get_renderer", return_value=renderer):
        with pytest.raises(ValueError, match="3 x 3"):
            fd.compute_full_diagnostics(result, max_bandwidth=64)
    renderer.render_batch.assert_not_called()


@pytest.mark.parametrize("stamp", [{"kind": "hex"}, {"kind": "hex", "n_rows": "x", "n_cols": 2}],
                         ids=["fields-missing", "not-a-number"])
def test_an_unreadable_hex_stamp_stops_the_pass_and_leaves_the_result_untouched(stamp):
    # The job reports the ValueError (routes/forward_diagnostics._worker); no value may come from the four
    # neighbours of the index grid instead, and nothing is written to the result.
    from unittest.mock import MagicMock, patch
    result = _fake_result()
    result.metadata["scan_geometry"] = stamp
    renderer = _batch_renderer()
    with patch.object(fd, "load_or_get_phase", return_value=MagicMock()), \
         patch.object(fd, "get_renderer", return_value=renderer):
        with pytest.raises(ValueError):
            fd.compute_full_diagnostics(result, max_bandwidth=64)
    renderer.render_batch.assert_not_called()
    assert "forward_diagnostics" not in result.metadata


# ----------------- the layer is the result's: the analysis dataset next to it does not decide how it is drawn

@pytest.mark.parametrize("result_stamp, dataset_stamp, hexagonal", [
    (ScanGeometry.hex(2, 2, 0.5).to_dict(), ScanGeometry.square(2, 2, 0.5).to_dict(), True),
    (ScanGeometry.square(2, 2, 0.5).to_dict(), ScanGeometry.hex(2, 2, 0.5).to_dict(), False),
], ids=["hex-result-square-dataset", "square-result-hex-dataset"])
def test_the_local_anomaly_layer_is_drawn_for_the_scan_of_the_result_not_of_the_dataset(
        monkeypatch, result_stamp, dataset_stamp, hexagonal):
    # `local_anomaly` is a diagnostics kind: `_compute_layer_rgba` reads it from the result alone, and the numbers
    # were computed on the result's own stamp (the test above). The gate it loses here read the geometry the
    # layer route's basis names, which for a result WITHOUT a crystal map is the dataset's; the drawing must
    # follow the result, so that a hexagonal result never meets a square raster and the other way round.
    from types import SimpleNamespace
    from backend.api.routes import phase_map
    result = SimpleNamespace(xmap=None, metadata={"scan_geometry": result_stamp})
    dataset = SimpleNamespace(xmap=object(), shape=(2, 2), scan_geometry=dataset_stamp, per_phase_data={})
    monkeypatch.setattr("backend.api.routes.phase_map.get_last_indexing_result", lambda: result)
    monkeypatch.setattr("backend.api.routes.phase_map.get_analysis_dataset", lambda: dataset)
    assert (phase_map._hex_layer_context("local_anomaly") is not None) is hexagonal


@pytest.mark.parametrize("small", [False, True])
def test_a_topology_of_another_grid_is_refused_not_applied_quietly(small):
    # 9 x 7 on a 7 x 9 map has the same number of points; the small map would otherwise be answered with zeros
    shape = (3, 2) if small else (N_COLS, N_ROWS + 1)
    ncc = np.full(shape, 0.5, dtype=np.float32)
    with pytest.raises(ValueError, match="grid"):
        compute_local_anomaly_map(ncc, topology=Topology("hex", N_ROWS + 1, N_COLS, 0.5))
