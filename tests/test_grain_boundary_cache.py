"""The angle cache of the grain-boundary layer follows the content of the map, not its name.

Unify, apply-to-grain, undo, reassign and clean-up change one ``xmap`` in place. A key made of
``id(xmap)`` or of a result id then returns the angles of the map as it was: the layer keeps drawing
the old boundaries. Every expectation below is derived by hand from the arrangement it builds.
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import grain_boundaries as gb  # noqa: E402
from backend.api.services.scan_geometry import ScanGeometry  # noqa: E402
from backend.api.services.scan_neighbours import Topology  # noqa: E402
from tests.test_phase_map_layer_endpoint import _make_fake_result  # noqa: E402

N_ROWS, N_COLS, STEP = 5, 5, 2.0
BANDS = [{"id": "hagb", "min": 15.0, "max": None, "color": "#ffffff", "width": 2, "on": True}]
HEX = Topology("hex", N_ROWS, N_COLS, 0.5)


def _qz(deg):
    a = np.radians(deg) / 2.0
    return np.array([np.cos(a), 0.0, 0.0, np.sin(a)])


def _quats(degrees_by_column):
    """One quaternion per point of the 5 x 5 map: the rotation about z of its column, in degrees."""
    return np.stack([_qz(degrees_by_column[c]) for _ in range(N_ROWS) for c in range(N_COLS)])


#: columns 0-1 at 0 degrees, columns 2-4 at 30 degrees: one boundary, between columns 1 and 2
STEP_AT_2 = [0, 0, 30, 30, 30]
#: columns 0-2 at 0 degrees, columns 3-4 at 30 degrees: the boundary has moved to columns 2 and 3
STEP_AT_3 = [0, 0, 0, 30, 30]
#: no boundary anywhere
FLAT = [0, 0, 0, 0, 0]


def _xmap(degrees_by_column, phase_id=None):
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation
    x, y = np.meshgrid(np.arange(N_COLS, dtype=float), np.arange(N_ROWS, dtype=float))
    return CrystalMap(
        rotations=Rotation(_quats(degrees_by_column)),
        phase_id=np.zeros(N_ROWS * N_COLS, int) if phase_id is None else phase_id,
        x=x.ravel(), y=y.ravel(),
        phase_list=PhaseList([Phase(name="a", point_group="m-3m"), Phase(name="b", point_group="m-3m")]))


def _rotate_in_place(xmap, degrees_by_column):
    """What unify, apply-to-grain, undo, reassign and the refinement do: assign ``xmap._rotations``."""
    from orix.quaternion import Rotation
    xmap._rotations = Rotation(_quats(degrees_by_column))


def _finite_pairs(angles):
    """How many edges carry an angle of at least 15 degrees, whatever the topology."""
    if "pairs" in angles:
        return int((np.nan_to_num(angles["angle"]) >= 15).sum())
    return int((np.nan_to_num(angles["h"]) >= 15).sum() + (np.nan_to_num(angles["v"]) >= 15).sum())


@pytest.fixture(autouse=True)
def _empty_cache():
    gb._ANGLE_CACHE.clear()
    yield
    gb._ANGLE_CACHE.clear()


# ------------------------------------------------------------------ the cache itself

@pytest.mark.parametrize("topology", [None, HEX], ids=["square", "hex"])
def test_an_unchanged_map_is_a_hit(topology):
    xm = _xmap(STEP_AT_2)
    first = gb.cached_boundary_angles(xm, N_ROWS, N_COLS, topology=topology)
    assert gb.cached_boundary_angles(xm, N_ROWS, N_COLS, topology=topology) is first


@pytest.mark.parametrize("topology", [None, HEX], ids=["square", "hex"])
def test_rotations_changed_in_place_give_new_angles(topology):
    xm = _xmap(STEP_AT_2)
    before = gb.cached_boundary_angles(xm, N_ROWS, N_COLS, topology=topology)
    # a step of 30 degrees between columns 1 and 2. Square: one horizontal edge per row, 5. Hexagonal:
    # the same 5 along the rows, and between two neighbouring rows one slanted pair (an even row's
    # point in column 2 touches the odd row's point in column 1; an odd row's point in column 1 touches
    # the even row's point in column 2), 4 pairs of rows, so 5 + 4 = 9
    assert _finite_pairs(before) == (5 if topology is None else 9)
    _rotate_in_place(xm, FLAT)
    after = gb.cached_boundary_angles(xm, N_ROWS, N_COLS, topology=topology)
    assert after is not before
    assert _finite_pairs(after) == 0


def test_the_square_angles_after_a_change_are_the_ones_computed_without_the_cache():
    xm = _xmap(STEP_AT_2)
    gb.cached_boundary_angles(xm, N_ROWS, N_COLS)
    _rotate_in_place(xm, STEP_AT_3)
    got = gb.cached_boundary_angles(xm, N_ROWS, N_COLS)
    want = gb.boundary_angles(xm, N_ROWS, N_COLS)
    np.testing.assert_array_equal(got["h"], want["h"])
    np.testing.assert_array_equal(got["v"], want["v"])
    # by hand: the boundary is now between columns 2 and 3 on every row
    assert np.isfinite(got["h"][:, 2]).all() and got["h"][:, 2] == pytest.approx(30.0, abs=1e-3)
    assert got["h"][:, 1] == pytest.approx(0.0, abs=1e-3)


def test_a_phase_changed_in_place_gives_new_angles():
    xm = _xmap(STEP_AT_2)
    before = gb.cached_boundary_angles(xm, N_ROWS, N_COLS)
    assert np.isfinite(before["h"][:, 1]).all()
    # what reassign and the clean-up do: write into the stored phase ids
    xm._phase_id[np.tile(np.arange(N_COLS) >= 2, N_ROWS)] = 1   # the 30-degree block becomes the other phase
    after = gb.cached_boundary_angles(xm, N_ROWS, N_COLS)
    assert np.isnan(after["h"][:, 1]).all()                  # across two phases there is no angle


def test_a_second_map_of_the_same_shape_with_the_same_id_does_not_get_the_first_ones_angles(monkeypatch):
    # a map built anew can get the memory address of one that was freed: force it
    monkeypatch.setattr(gb, "id", lambda obj: 1, raising=False)
    first = _xmap(STEP_AT_2)
    second = _xmap(STEP_AT_3)
    a = gb.cached_boundary_angles(first, N_ROWS, N_COLS)
    b = gb.cached_boundary_angles(second, N_ROWS, N_COLS)
    assert a["h"][:, 1] == pytest.approx(30.0, abs=1e-3) and a["h"][:, 2] == pytest.approx(0.0, abs=1e-3)
    assert b["h"][:, 2] == pytest.approx(30.0, abs=1e-3) and b["h"][:, 1] == pytest.approx(0.0, abs=1e-3)


def test_a_result_id_does_not_stand_for_the_content():
    class Result:
        result_id = "r1"
    xm = _xmap(STEP_AT_2)
    gb.cached_boundary_angles(xm, N_ROWS, N_COLS, result=Result())
    _rotate_in_place(xm, STEP_AT_3)
    got = gb.cached_boundary_angles(xm, N_ROWS, N_COLS, result=Result())
    assert got["h"][:, 2] == pytest.approx(30.0, abs=1e-3)


def test_a_mask_of_the_same_size_in_another_place_gives_new_angles():
    n = N_ROWS * N_COLS
    first = np.zeros(n, dtype=bool)
    first[:20] = True
    second = np.zeros(n, dtype=bool)
    second[5:25] = True
    # a result indexed on a sub-region carries only the selected points: 20 of the 25, which the
    # mask puts back where they were measured
    sub = _xmap(STEP_AT_2)[first]
    a = gb.cached_boundary_angles(sub, N_ROWS, N_COLS, first)
    b = gb.cached_boundary_angles(sub, N_ROWS, N_COLS, second)
    # the first mask leaves out the last row (points 20-24): no horizontal edge in it has an angle
    assert np.isnan(a["h"][4]).all()
    # the second one leaves out the first row (points 0-4) and puts the 20 points on rows 1-4: the
    # last row then holds the points 15-19 of the sub map, columns 0-4, with the step at 1 | 2
    assert np.isnan(b["h"][0]).all()
    assert b["h"][4, 1] == pytest.approx(30.0, abs=1e-3)


def test_the_cache_stays_one_entry_big():
    xm = _xmap(STEP_AT_2)
    for degrees in (STEP_AT_2, STEP_AT_3, FLAT, STEP_AT_2):
        _rotate_in_place(xm, degrees)
        gb.cached_boundary_angles(xm, N_ROWS, N_COLS)
        gb.cached_boundary_angles(xm, N_ROWS, N_COLS, topology=HEX)
    assert len(gb._ANGLE_CACHE) == 1


def _map_of_points(degrees_by_point, point_group="m-3m"):
    """A 5 x 5 map whose point k is rotated about z by ``degrees_by_point[k]``; both phases have ``point_group``."""
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation
    x, y = np.meshgrid(np.arange(N_COLS, dtype=float), np.arange(N_ROWS, dtype=float))
    return CrystalMap(
        rotations=Rotation(np.stack([_qz(d) for d in degrees_by_point])),
        phase_id=np.zeros(N_ROWS * N_COLS, int), x=x.ravel(), y=y.ravel(),
        phase_list=PhaseList([Phase(name="a", point_group=point_group), Phase(name="b", point_group=point_group)]))


def test_maps_that_differ_only_in_which_points_are_in_the_data_give_different_angles():
    # Same stored rotations, same phase ids of the points in the data (20 zeros), same shape: only the flags differ.
    # Point k is at 30 degrees for k < 5 (row 0) and at 0 degrees otherwise.
    degrees = [30.0] * 5 + [0.0] * 20
    first, second = _map_of_points(degrees), _map_of_points(degrees)
    first.is_in_data[20:] = False       # points 0-19: row 0 at 30 degrees, rows 1-3 at 0, row 4 empty
    second.is_in_data[:5] = False       # points 5-24 become the first 20: all at 0 degrees
    a = gb.cached_boundary_angles(first, N_ROWS, N_COLS)
    b = gb.cached_boundary_angles(second, N_ROWS, N_COLS)
    # by hand: the vertical edges between row 0 and row 1 cross from 30 to 0 degrees in the first map and
    # from 0 to 0 in the second; rows 3 | 4 have no angle in either, the last row being empty
    assert a["v"][0] == pytest.approx(30.0, abs=1e-3)
    assert b["v"][0] == pytest.approx(0.0, abs=1e-3)
    assert np.isnan(a["v"][3]).all() and np.isnan(b["v"][3]).all()


def test_maps_that_differ_only_in_the_point_group_of_a_phase_give_different_angles():
    # 60 degrees about z: under m-3m the four-fold axis along z takes it to 90 - 60 = 30 degrees,
    # under the point group 1 (no symmetry) it stays 60
    degrees = [0.0] * 2 + [60.0] * 3
    cubic = _map_of_points(degrees * N_ROWS, "m-3m")
    triclinic = _map_of_points(degrees * N_ROWS, "1")
    a = gb.cached_boundary_angles(cubic, N_ROWS, N_COLS)
    b = gb.cached_boundary_angles(triclinic, N_ROWS, N_COLS)
    assert a["h"][:, 1] == pytest.approx(30.0, abs=1e-3)
    assert b["h"][:, 1] == pytest.approx(60.0, abs=1e-3)


def test_the_three_axis_variants_of_2_over_m_share_a_name_and_do_not_share_angles(monkeypatch):
    # A turn of 180 degrees about z is the two-fold axis of the c-unique group (disorientation 0) and no
    # element of the a-unique group, whose two-fold axis is x (disorientation 180)
    from crystal_symmetry import frame_symmetry
    c_unique = frame_symmetry("2/m", lattice=(5, 6, 7, 90, 90, 100))
    a_unique = frame_symmetry("2/m", lattice=(5, 6, 7, 100, 90, 90))
    assert c_unique.name == a_unique.name == "2/m"
    xm = _map_of_points(([0.0] * 2 + [180.0] * 3) * N_ROWS)
    monkeypatch.setattr(gb, "phase_point_group", lambda phase: c_unique)
    a = gb.cached_boundary_angles(xm, N_ROWS, N_COLS)
    monkeypatch.setattr(gb, "phase_point_group", lambda phase: a_unique)
    b = gb.cached_boundary_angles(xm, N_ROWS, N_COLS)
    assert a["h"][:, 1] == pytest.approx(0.0, abs=1e-3)
    assert b["h"][:, 1] == pytest.approx(180.0, abs=1e-3)


# --------------------------------------------------------------------- the layer route

@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app, raise_server_exceptions=False)


def _result(stamp, degrees):
    fake = _make_fake_result(n_rows=N_ROWS, n_cols=N_COLS)
    fake.xmap = _xmap(degrees)
    fake.metadata = {"scan_geometry": stamp}
    return fake


def _layer(client):
    r = client.get("/api/phasemap/layer", params={"kind": "grain-boundaries", "gb_bands": json.dumps(BANDS)})
    assert r.status_code == 200, r.text
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(r.json()["image"]))))


@pytest.mark.parametrize("stamp", [ScanGeometry.square(N_ROWS, N_COLS, STEP).to_dict(),
                                   ScanGeometry.hex(N_ROWS, N_COLS, STEP).to_dict()], ids=["square", "hex"])
def test_the_layer_after_an_in_place_change_is_a_different_picture(client, monkeypatch, stamp):
    result = _result(stamp, STEP_AT_2)
    monkeypatch.setattr("backend.api.routes.phase_map.get_last_indexing_result", lambda: result)
    monkeypatch.setattr("backend.api.routes.phase_map.get_analysis_dataset", lambda: None)
    drawn = _layer(client)
    assert (drawn[..., 3] > 0).any()                         # a boundary is drawn
    _rotate_in_place(result.xmap, FLAT)
    flat = _layer(client)
    assert not (flat[..., 3] > 0).any()                      # one orientation: nothing is drawn
    _rotate_in_place(result.xmap, STEP_AT_3)
    moved = _layer(client)
    assert (moved[..., 3] > 0).any() and not np.array_equal(moved, drawn)
    _rotate_in_place(result.xmap, STEP_AT_2)
    np.testing.assert_array_equal(_layer(client), drawn)     # back to the first arrangement: the first picture
