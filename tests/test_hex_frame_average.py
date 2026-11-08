"""Frame averaging on a hexagonal scan: the average over the six neighbours, held against kikuchipy and the lattice."""
from __future__ import annotations

import dataclasses
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.routes import ebsd_viewer  # noqa: E402
from backend.api.services import frame_average, scan_geometry  # noqa: E402
from backend.api.services.frame_average import average_neighbour_patterns  # noqa: E402
from backend.api.services.scan_geometry import ScanGeometry  # noqa: E402
from backend.api.services.scan_neighbours import Topology  # noqa: E402

SIGNS = (0.5, -0.5)
N_ROWS, N_COLS = 6, 5


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app, raise_server_exceptions=False)


def _oracle(data, shift, rings):
    """Mean over the points within ``rings`` steps on the sample (clipped at the border), rescaled per pattern.

    Written from the positions of the lattice; ``rings`` <= 3, where the points of the next ring are further
    than ``rings`` steps away (the nearest point of ring 4 is 4 * sqrt(3) / 2 = 3.46 steps from the centre)."""
    n_rows, n_cols = data.shape[:2]
    g = ScanGeometry.hex(n_rows, n_cols, 1.0, odd_row_shift=shift)
    x, y = g.positions()
    pts = np.stack([x.ravel(), y.ravel()], axis=1)
    dist = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    flat = data.reshape(n_rows * n_cols, *data.shape[2:]).astype(np.float64)
    out = np.empty_like(flat)
    for p in range(flat.shape[0]):
        mean = flat[dist[p] <= rings + 1e-6].mean(axis=0)
        out[p] = (mean - mean.min()) / (mean.max() - mean.min())
    return out.reshape(data.shape)                                # in [0, 1]


@pytest.mark.parametrize("rings", [1, 2, 3])
@pytest.mark.parametrize("shift", SIGNS)
def test_the_average_is_the_mean_over_the_points_within_the_disc_on_the_sample(shift, rings):
    data = np.random.default_rng(1).normal(size=(N_ROWS, N_COLS, 6, 7)).astype(np.float32)
    got = average_neighbour_patterns(data, Topology("hex", N_ROWS, N_COLS, shift), rings)
    want = _oracle(data, shift, rings) * 2.0 - 1.0                # float patterns fill [-1, 1]
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, want, atol=1e-5)


@pytest.mark.parametrize("shift", SIGNS)
def test_integer_patterns_fill_their_range_individually(shift):
    data = np.random.default_rng(2).integers(0, 200, size=(N_ROWS, N_COLS, 6, 7), dtype=np.uint8)
    got = average_neighbour_patterns(data, Topology("hex", N_ROWS, N_COLS, shift), 1)
    want = _oracle(data, shift, 1) * 255.0
    assert got.dtype == np.uint8
    assert np.abs(got.astype(float) - want).max() <= 1.0 + 1e-9    # to a grey level (truncation)
    assert (got.reshape(-1, 42).min(axis=1) == 0).all() and (got.reshape(-1, 42).max(axis=1) == 255).all()


def _three_by_three(shift):
    """Hand-made 3 x 3 scan of patterns with three pixels ``(x, y, 0)``.

    The two points (0, 0) and (2, 0) hold (0, 1), every other point (1, 0). With the odd rows shifted by +0.5 the
    six neighbours of (1, 1) are (1, 0) (1, 2) (0, 1) (0, 2) (2, 1) (2, 2), none of the two odd ones: the mean over
    the seven is (1, 0, 0), which fills the range as (255, 0, 0). With -0.5 they are (1, 0) (1, 2) (0, 0) (0, 1)
    (2, 0) (2, 1): the mean over the seven is (5/7, 2/7, 0) and fills the range as (255, 2/5 * 255 = 102, 0)."""
    data = np.zeros((3, 3, 1, 3), dtype=np.uint8)
    data[..., 0] = 1
    data[0, 0, 0] = (0, 1, 0)
    data[2, 0, 0] = (0, 1, 0)
    return data


@pytest.mark.parametrize("shift, middle", [(0.5, 0), (-0.5, 102)])
def test_the_six_neighbours_of_a_point_are_the_ones_the_shift_says_by_hand(shift, middle):
    got = average_neighbour_patterns(_three_by_three(shift), Topology("hex", 3, 3, shift), 1)
    assert np.abs(got[1, 1, 0].astype(int) - np.array([255, middle, 0])).max() <= 1


def test_a_point_that_was_not_measured_counts_nowhere_and_keeps_its_pattern():
    data = _three_by_three(0.5)
    data[0, 1, 0] = (0, 1, 0)                          # a neighbour of (1, 1), with a pattern that is not data
    valid = np.ones((3, 3), dtype=bool)
    valid[0, 1] = False
    topology = Topology("hex", 3, 3, 0.5)
    got = average_neighbour_patterns(data, topology, 1, valid)
    assert np.abs(got[1, 1, 0].astype(int) - np.array([255, 0, 0])).max() <= 1       # six neighbours minus one: (1, 0, 0)
    assert got[0, 1, 0].tolist() == [0, 1, 0]                                         # copied, not rescaled to 255
    without = average_neighbour_patterns(data, topology, 1)
    assert np.abs(without[1, 1, 0].astype(int) - np.array([255, 42, 0])).max() <= 1  # all 7: (6/7, 1/7, 0)


def test_a_mask_of_another_grid_is_refused():
    with pytest.raises(ValueError, match="measured points"):
        average_neighbour_patterns(np.zeros((3, 3, 2, 2), dtype=np.uint8), Topology("hex", 3, 3, 0.5), 1,
                                   np.ones((3, 4), dtype=bool))


@pytest.mark.parametrize("size", [3, 5])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_a_square_topology_averages_like_kikuchipy(size, dtype):
    import kikuchipy as kp
    rng = np.random.default_rng(3)
    data = (rng.integers(0, 255, size=(7, 8, 6, 6), dtype=np.uint8) if dtype is np.uint8
            else rng.normal(size=(7, 8, 6, 6)).astype(np.float32))
    signal = kp.signals.EBSD(data.copy())
    signal.average_neighbour_patterns(window="rectangular", window_shape=(size, size), show_progressbar=False)
    got = average_neighbour_patterns(data, Topology("square", 7, 8), (size - 1) // 2)
    tolerance = 1.0 if dtype is np.uint8 else 1e-4
    assert np.abs(got.astype(float) - np.asarray(signal.data).astype(float)).max() <= tolerance


def test_a_constant_pattern_is_set_to_the_lowest_value_not_to_nan():
    data = np.full((N_ROWS, N_COLS, 4, 4), 7, dtype=np.uint8)
    got = average_neighbour_patterns(data, Topology("hex", N_ROWS, N_COLS, 0.5), 1)
    assert (got == 0).all()


def test_a_topology_of_another_grid_is_refused():
    with pytest.raises(ValueError, match="grid"):
        average_neighbour_patterns(np.zeros((4, 4, 3, 3), dtype=np.uint8), Topology("hex", 3, 4, 0.5), 1)


def test_the_result_does_not_depend_on_the_band_size(monkeypatch):
    data = np.random.default_rng(4).integers(0, 255, size=(N_ROWS, N_COLS, 6, 7), dtype=np.uint8)
    topology = Topology("hex", N_ROWS, N_COLS, 0.5)
    whole = average_neighbour_patterns(data, topology, 2)
    monkeypatch.setattr(frame_average, "BAND_BYTES", 6 * 7 * 4 * 3)          # three patterns per step
    np.testing.assert_array_equal(average_neighbour_patterns(data, topology, 2), whole)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("order", ["F", "transposed view"])
def test_data_that_is_not_c_ordered_gives_the_bytes_of_the_c_ordered_result(dtype, order):
    rng = np.random.default_rng(11)
    c_data = (rng.integers(0, 255, size=(N_ROWS, N_COLS, 6, 7)).astype(dtype))
    if order == "F":
        other = np.asfortranarray(c_data)
    else:                                                          # the same values, laid out with the patterns first
        other = np.ascontiguousarray(c_data.transpose(2, 3, 0, 1)).transpose(2, 3, 0, 1)
    assert not other.flags.c_contiguous and np.array_equal(other, c_data)
    topology = Topology("hex", N_ROWS, N_COLS, 0.5)
    want = average_neighbour_patterns(c_data, topology, 2)
    got = average_neighbour_patterns(other, topology, 2)
    assert got.tobytes() == want.tobytes()
    assert (got.reshape(-1, 42).max(axis=1) > got.reshape(-1, 42).min(axis=1)).all()    # real data, not zeros


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_the_working_arrays_stay_within_the_bytes_the_constant_names(monkeypatch, dtype):
    """Peak of numpy's allocations above the result array, for three bands per scan, against ``working_bytes``."""
    monkeypatch.setattr(frame_average, "BAND_BYTES", 4 * 1024 * 1024)
    data = (np.random.default_rng(12).random((20, 20, 60, 60)) * 200).astype(dtype)
    topology = Topology("hex", 20, 20, 0.5)
    average_neighbour_patterns(data, topology, 1)                              # warm the imports
    tracemalloc.start()
    try:
        out = average_neighbour_patterns(data, topology, 1)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert out.nbytes == data.nbytes
    assert 3 * frame_average.BAND_BYTES < peak - out.nbytes <= frame_average.working_bytes(), peak - out.nbytes


def test_the_memory_does_not_grow_with_the_window():
    """A table of every point's neighbours would be 8 * 3600 * 169 = 4.9 MB for rings = 7 on this 60 x 60 scan; the
    function holds the result (3600 patterns of 2 x 2 = 14 KB), a few bands, and no such table."""
    topology = Topology("hex", 60, 60, 0.5)
    data = np.random.default_rng(5).integers(0, 255, size=(60, 60, 2, 2), dtype=np.uint8)
    average_neighbour_patterns(data, topology, 1)                              # warm the imports
    tracemalloc.start()
    try:
        average_neighbour_patterns(data, topology, 7)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 1_000_000, peak


# ------------------------------------------------------------------------ the route

def _hold(monkeypatch, signal, geometry):
    scan_geometry.clear_all()
    if geometry is not None:
        scan_geometry.set_geometry("ds", geometry)
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "ds")
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda: signal)
    monkeypatch.setattr(ebsd_viewer, "_mark_active_dirty", lambda: None)


@pytest.mark.parametrize("size", [3, 5])
@pytest.mark.parametrize("lazy", [False, True], ids=["in memory", "lazy"])
@pytest.mark.parametrize("shift", SIGNS)
def test_the_route_averages_a_hexagonal_dataset_on_its_six_neighbours(client, monkeypatch, shift, lazy, size):
    import kikuchipy as kp
    data = np.random.default_rng(5).integers(0, 255, size=(N_ROWS, N_COLS, 6, 6), dtype=np.uint8)
    signal = kp.signals.EBSD(data.copy())
    if lazy:
        signal = signal.as_lazy()
    _hold(monkeypatch, signal, ScanGeometry.hex(N_ROWS, N_COLS, 0.5, odd_row_shift=shift))
    r = client.post("/api/ebsd/frame-average", json={"window_size": size})
    assert r.status_code == 200, r.text
    assert r.json() == {"success": True, "window_size": size}
    assert isinstance(signal.data, np.ndarray)                        # a lazy signal is held in memory afterwards
    want = _oracle(data, shift, (size - 1) // 2) * 255.0              # 3 -> the point and its 6 neighbours, 5 -> 19
    assert np.abs(np.asarray(signal.data).astype(float) - want).max() <= 1.0 + 1e-9
    scan_geometry.clear_all()


def test_a_window_of_one_changes_nothing_on_a_hexagonal_dataset(client, monkeypatch):
    import kikuchipy as kp
    data = np.random.default_rng(6).integers(0, 255, size=(N_ROWS, N_COLS, 6, 6), dtype=np.uint8)
    signal = kp.signals.EBSD(data.copy())
    _hold(monkeypatch, signal, ScanGeometry.hex(N_ROWS, N_COLS, 0.5))
    assert client.post("/api/ebsd/frame-average", json={"window_size": 1}).status_code == 200
    np.testing.assert_array_equal(np.asarray(signal.data), data)
    scan_geometry.clear_all()


@pytest.mark.parametrize("size", [2, 4, 10])
def test_an_even_window_is_refused_on_a_hexagonal_dataset_and_names_the_odd_ones(client, monkeypatch, size):
    import kikuchipy as kp
    data = np.random.default_rng(7).integers(0, 255, size=(N_ROWS, N_COLS, 6, 6), dtype=np.uint8)
    signal = kp.signals.EBSD(data.copy())
    _hold(monkeypatch, signal, ScanGeometry.hex(N_ROWS, N_COLS, 0.5))
    r = client.post("/api/ebsd/frame-average", json={"window_size": size})
    assert r.status_code == 400 and f"window of {size}" in r.json()["detail"] and "odd" in r.json()["detail"]
    np.testing.assert_array_equal(np.asarray(signal.data), data)
    scan_geometry.clear_all()


def test_the_route_leaves_a_point_that_was_not_measured_alone(client, monkeypatch):
    import kikuchipy as kp
    data = _three_by_three(0.5)
    data[0, 1, 0] = (0, 1, 0)
    valid = np.ones((3, 3), dtype=bool)
    valid[0, 1] = False
    signal = kp.signals.EBSD(data.copy())
    _hold(monkeypatch, signal, dataclasses.replace(ScanGeometry.hex(3, 3, 0.5), valid_mask=valid))
    assert client.post("/api/ebsd/frame-average", json={"window_size": 3}).status_code == 200
    assert np.asarray(signal.data)[0, 1, 0].tolist() == [0, 1, 0]
    assert np.abs(np.asarray(signal.data)[1, 1, 0].astype(int) - np.array([255, 0, 0])).max() <= 1
    scan_geometry.clear_all()


def test_a_dataset_too_big_to_hold_twice_is_refused_with_the_way_out(client, monkeypatch):
    import kikuchipy as kp
    data = np.zeros((N_ROWS, N_COLS, 6, 6), dtype=np.uint8)
    signal = kp.signals.EBSD(data.copy())
    _hold(monkeypatch, signal, ScanGeometry.hex(N_ROWS, N_COLS, 0.5))
    monkeypatch.setattr(frame_average, "HEX_MAX_BYTES", data.nbytes - 1)
    r = client.post("/api/ebsd/frame-average", json={"window_size": 3})
    assert r.status_code == 413 and "crop" in r.json()["detail"].lower()
    np.testing.assert_array_equal(np.asarray(signal.data), data)
    scan_geometry.clear_all()


def test_the_limit_is_the_documented_two_gibibytes_and_a_dataset_at_it_is_not_refused():
    assert frame_average.HEX_MAX_BYTES == 2 * 1024 ** 3
    assert frame_average.HEX_MAX_BYTES * 2 == ebsd_viewer._STASH_MATERIALIZE_MAX_BYTES


def test_a_scan_of_1000_by_1200_points_with_60_by_60_patterns_is_refused_before_any_work(client, monkeypatch):
    """1000 * 1200 * 60 * 60 = 4.32e9 bytes = 4.02 GiB, twice the limit's half. The patterns are a broadcast of one
    zero (no memory); the signal has no method to average with, so any work would fail with another answer."""
    n_rows, n_cols = 1000, 1200

    class Axes:
        navigation_shape = (n_cols, n_rows)

    class Signal:
        data = np.broadcast_to(np.uint8(0), (n_rows, n_cols, 60, 60))
        axes_manager = Axes()

    _hold(monkeypatch, Signal(), ScanGeometry.hex(n_rows, n_cols, 0.5))
    r = client.post("/api/ebsd/frame-average", json={"window_size": 3})
    assert r.status_code == 413, r.text
    assert "4.02 GiB" in r.json()["detail"] and "2.00 GiB" in r.json()["detail"]
    assert not ebsd_viewer._frame_average_hex_lock.locked()
    scan_geometry.clear_all()


def test_a_dataset_just_over_the_limit_is_not_named_as_equal_to_it(client, monkeypatch):
    """2.03 GiB against 2 GiB: with one decimal both read 2.0."""
    n_rows, n_cols = 100, 100
    size = int(2.03 * 1024 ** 3) // (n_rows * n_cols)                  # pixels (= bytes, uint8) of one pattern
    assert 2.025 < n_rows * n_cols * size / 1024 ** 3 < 2.035

    class Axes:
        navigation_shape = (n_cols, n_rows)

    class Signal:
        data = np.broadcast_to(np.uint8(0), (n_rows, n_cols, size, 1))
        axes_manager = Axes()

    _hold(monkeypatch, Signal(), ScanGeometry.hex(n_rows, n_cols, 0.5))
    r = client.post("/api/ebsd/frame-average", json={"window_size": 3})
    assert r.status_code == 413, r.text
    assert "2.03 GiB" in r.json()["detail"] and "2.00 GiB" in r.json()["detail"]
    scan_geometry.clear_all()


@pytest.mark.parametrize("dtype, per_value", [(np.uint8, 1), (np.uint16, 2), (np.float32, 4)])
def test_the_limit_counts_bytes_and_a_dataset_exactly_at_it_goes_through(client, monkeypatch, dtype, per_value):
    import kikuchipy as kp
    data = np.random.default_rng(9).integers(0, 200, size=(N_ROWS, N_COLS, 6, 6)).astype(dtype)
    assert data.nbytes == N_ROWS * N_COLS * 36 * per_value                  # by hand: 1080 values times the item size
    signal = kp.signals.EBSD(data.copy())
    _hold(monkeypatch, signal, ScanGeometry.hex(N_ROWS, N_COLS, 0.5))
    monkeypatch.setattr(frame_average, "HEX_MAX_BYTES", N_ROWS * N_COLS * 36 * per_value)
    assert client.post("/api/ebsd/frame-average", json={"window_size": 3}).status_code == 200
    # one byte less, and (for the wider types) the limit that would hold if every value were one byte
    for limit in {N_ROWS * N_COLS * 36 * per_value - 1, N_ROWS * N_COLS * 36} - {N_ROWS * N_COLS * 36 * per_value}:
        fresh = kp.signals.EBSD(data.copy())
        _hold(monkeypatch, fresh, ScanGeometry.hex(N_ROWS, N_COLS, 0.5))
        monkeypatch.setattr(frame_average, "HEX_MAX_BYTES", limit)
        r = client.post("/api/ebsd/frame-average", json={"window_size": 3})
        assert r.status_code == 413, (limit, r.text)
        np.testing.assert_array_equal(np.asarray(fresh.data), data)
    scan_geometry.clear_all()


def test_a_window_of_one_is_not_held_to_the_memory_limit(client, monkeypatch):
    import kikuchipy as kp
    data = np.zeros((N_ROWS, N_COLS, 6, 6), dtype=np.uint8)
    _hold(monkeypatch, kp.signals.EBSD(data.copy()), ScanGeometry.hex(N_ROWS, N_COLS, 0.5))
    monkeypatch.setattr(frame_average, "HEX_MAX_BYTES", 1)
    assert client.post("/api/ebsd/frame-average", json={"window_size": 1}).status_code == 200
    scan_geometry.clear_all()


@pytest.mark.parametrize("nav", [(30,), (5, 6, 2)], ids=["1-d", "3-d"])
def test_a_navigation_that_is_not_two_dimensional_is_named_as_the_navigation(client, monkeypatch, nav):
    class Axes:
        navigation_shape = nav

    class Signal:
        data = np.zeros((N_ROWS, N_COLS, 6, 6), dtype=np.uint8)
        axes_manager = Axes()

    _hold(monkeypatch, Signal(), ScanGeometry.hex(N_ROWS, N_COLS, 0.5))
    r = client.post("/api/ebsd/frame-average", json={"window_size": 3})
    detail = r.json()["detail"]
    assert r.status_code == 409 and f"navigation is {len(nav)}-dimensional" in detail
    assert "patterns are" not in detail
    scan_geometry.clear_all()


def test_the_largest_window_on_a_hexagonal_scan_is_49_and_a_larger_one_is_refused_before_any_work(client, monkeypatch):
    import kikuchipy as kp
    data = np.random.default_rng(10).integers(0, 255, size=(N_ROWS, N_COLS, 6, 6), dtype=np.uint8)
    signal = kp.signals.EBSD(data.copy())
    _hold(monkeypatch, signal, ScanGeometry.hex(N_ROWS, N_COLS, 0.5))
    assert client.post("/api/ebsd/frame-average", json={"window_size": 49}).status_code == 200
    # 24 rings reach every point of this 6 x 5 scan, so every pattern is the mean of all thirty, filling the range
    mean = data.astype(np.float64).mean(axis=(0, 1))
    want = (mean - mean.min()) / (mean.max() - mean.min()) * 255.0
    assert np.abs(np.asarray(signal.data).astype(float) - want).max() <= 1.0 + 1e-9
    assert not ebsd_viewer._frame_average_hex_lock.locked()

    calls = []
    monkeypatch.setattr(ebsd_viewer.frame_average_service, "average_neighbour_patterns",
                        lambda *a, **k: calls.append(a))
    for size in (51, 50, 10 ** 6):
        fresh = kp.signals.EBSD(data.copy())
        _hold(monkeypatch, fresh, ScanGeometry.hex(N_ROWS, N_COLS, 0.5))
        r = client.post("/api/ebsd/frame-average", json={"window_size": size})
        assert r.status_code == 400 and "largest is 49" in r.json()["detail"], (size, r.text)
        np.testing.assert_array_equal(np.asarray(fresh.data), data)
    assert calls == []
    assert not ebsd_viewer._frame_average_hex_lock.locked()
    scan_geometry.clear_all()


def test_the_window_limit_does_not_apply_to_a_square_scan(client, monkeypatch):
    calls = []

    class Signal:
        data = np.zeros((N_ROWS, N_COLS, 6, 6), dtype=np.uint8)

        def average_neighbour_patterns(self, **kwargs):
            calls.append(kwargs)

    _hold(monkeypatch, Signal(), ScanGeometry.square(N_ROWS, N_COLS, 0.5))
    assert client.post("/api/ebsd/frame-average", json={"window_size": 51}).status_code == 200
    assert calls[0]["window_shape"] == (51, 51)
    scan_geometry.clear_all()


def test_a_second_run_while_one_is_going_is_refused(client, monkeypatch):
    import kikuchipy as kp
    data = np.zeros((N_ROWS, N_COLS, 6, 6), dtype=np.uint8)
    signal = kp.signals.EBSD(data.copy())
    _hold(monkeypatch, signal, ScanGeometry.hex(N_ROWS, N_COLS, 0.5))
    assert ebsd_viewer._frame_average_hex_lock.acquire(blocking=False)
    try:
        r = client.post("/api/ebsd/frame-average", json={"window_size": 3})
    finally:
        ebsd_viewer._frame_average_hex_lock.release()
    assert r.status_code == 409 and "still running" in r.json()["detail"]
    scan_geometry.clear_all()


def test_the_lock_is_released_after_a_run_and_after_a_failure(client, monkeypatch):
    import kikuchipy as kp
    data = np.random.default_rng(8).integers(0, 255, size=(N_ROWS, N_COLS, 6, 6), dtype=np.uint8)
    signal = kp.signals.EBSD(data.copy())
    _hold(monkeypatch, signal, ScanGeometry.hex(N_ROWS, N_COLS, 0.5))
    assert client.post("/api/ebsd/frame-average", json={"window_size": 3}).status_code == 200
    assert not ebsd_viewer._frame_average_hex_lock.locked()

    def broken(*_a, **_k):
        raise RuntimeError("boom")
    monkeypatch.setattr(ebsd_viewer.frame_average_service, "average_neighbour_patterns", broken)
    r = client.post("/api/ebsd/frame-average", json={"window_size": 3})
    assert r.status_code == 500 and "boom" in r.json()["detail"]
    assert not ebsd_viewer._frame_average_hex_lock.locked()
    scan_geometry.clear_all()


def test_a_stamp_of_another_grid_is_refused(client, monkeypatch):
    import kikuchipy as kp
    signal = kp.signals.EBSD(np.zeros((N_ROWS, N_COLS, 6, 6), dtype=np.uint8))
    _hold(monkeypatch, signal, ScanGeometry.hex(4, 4, 0.5))
    r = client.post("/api/ebsd/frame-average", json={"window_size": 3})
    assert r.status_code == 409 and "4 x 4" in r.json()["detail"]
    scan_geometry.clear_all()


@pytest.mark.parametrize("geometry", [ScanGeometry.square(N_ROWS, N_COLS, 0.5), None], ids=["square", "unknown"])
@pytest.mark.parametrize("size", [5, 4])
def test_a_square_or_unknown_dataset_goes_to_kikuchipy_as_before(client, monkeypatch, geometry, size):
    calls = []

    class Signal:
        data = np.zeros((N_ROWS, N_COLS, 6, 6), dtype=np.uint8)

        def average_neighbour_patterns(self, **kwargs):
            calls.append(kwargs)

    _hold(monkeypatch, Signal(), geometry)
    r = client.post("/api/ebsd/frame-average", json={"window_size": size})
    assert r.status_code == 200 and calls == [
        {"window": "rectangular", "window_shape": (size, size), "show_progressbar": False}]
    scan_geometry.clear_all()
