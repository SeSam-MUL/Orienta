"""A joint R+PC refinement stores a second result; everything that lists or
reads results has to cope with it.

The refinement used to register a duck-typed stand-in (``_RefinedResult``) that
had ``xmap``, ``original_shape``, ``selection_mask`` and ``metadata`` and
nothing else. Every route that read ``result.method`` or
``result.confidence_scores`` -- the results gallery first of all -- answered
500 as soon as the stand-in was in the registry, and its Euler angles were the
PARENT's (``to_euler`` was borrowed from the parent's rotations).

These tests drive the real routes (TestClient without ``with``, so no startup
reaper) on a real ``IndexingResult`` with a real orix ``CrystalMap``. Only the
two Levenberg-Marquardt solvers and the experimental-pattern reader are
replaced; they are given a hand-written, known answer so the expected
orientations in this file are written down, not computed by the code under
test.

Hand-derived orientations (orix's Bunge ZXZ Euler angles, degrees): orix reads
a quaternion as the passive rotation, so the quaternion of a turn by ``-a``
about the x axis has Euler angles ``(0, a, 0)`` (checked once by hand:
``Rz(180) Rx(a) Rz(180) = Rx(-a)``, which is why the +a turn comes out as
``(180, a, 180)``). The parent's pixels sit at ``a = 40`` (first half of the
scan) and ``a = 25`` (second half); the fake solver turns the first half by a
further 3 degrees the same way (the second half is left alone), so the refined
pixels sit at ``a = 43`` and ``a = 25``.
"""
from __future__ import annotations

import base64
import contextlib
import io
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

# The app is imported while the real registry is still in place. Some route
# modules bind ``indexing._result_registry`` by name when they are first
# imported; if that first import happened inside a test, under the private
# registry the autouse fixture below installs, the module would keep the
# throw-away dict for the rest of the process and every later test file that
# reads the registry through those routes would answer 404.
import backend.api.main  # noqa: F401,E402

N_ROWS, N_COLS = 4, 4
N_PIX = N_ROWS * N_COLS
HALF = N_PIX // 2            # flat pixels below this index are the ones the fake solver turns
A_FIRST_PARENT, A_SECOND_PARENT = 40.0, 25.0   # Euler Phi in degrees, written down by hand
A_FIRST_REFINED = 43.0                          # 40 + the fake solver's 3 degrees
TURN_DEG = 3.0


@pytest.fixture
def client():
    from backend.api.main import app
    # Answer the way a user's browser sees it (a 500), not by re-raising.
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    """A private result registry and no active result, restored afterwards."""
    from backend.api.routes import indexing
    monkeypatch.setattr(indexing, "_result_registry", OrderedDict())
    monkeypatch.setattr(indexing, "_active_result_id", None)
    monkeypatch.setattr("backend.api.routes.ebsd_viewer._ebsd_file_path",
                        None, raising=False)
    yield


def _euler_0_a_0(phi_deg):
    """Quaternion (w, x, y, z) whose Bunge Euler angles are (0, phi, 0): a
    turn by -phi about x (see the module docstring)."""
    half = np.deg2rad(phi_deg) / 2.0
    return [np.cos(half), -np.sin(half), 0.0, 0.0]


def _real_parent(selection_mask=None, unindexed_flat=(), method="spherical",
                 source_file=None):
    """A real IndexingResult on a real CrystalMap, one cubic phase (id 1, as
    the spherical indexer numbers it), every indexed pixel a pure x rotation.

    ``selection_mask`` makes a region-of-interest result (the map has one row
    per selected pixel). ``unindexed_flat`` lists flat pixels that stay in a
    full-grid map but carry phase id -1 (not indexed). ``method`` is the
    indexing method name ("spherical" or "dictionary"; the refinement accepts
    those two). ``source_file`` tags the result with the scan it came from."""
    from diffpy.structure import Lattice, Structure
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation
    from indexing_controller import IndexingMethod, IndexingResult

    if selection_mask is None:
        selection_mask = np.ones((N_ROWS, N_COLS), dtype=bool)
    flat = selection_mask.ravel()
    rows = np.where(flat)[0]
    quats = np.array([_euler_0_a_0(A_FIRST_PARENT if g < HALF else A_SECOND_PARENT)
                      for g in rows], dtype=np.float64)
    ys, xs = np.divmod(rows, N_COLS)
    phase_id = np.ones(rows.size, dtype=np.int32)
    phase_id[np.isin(rows, list(unindexed_flat))] = -1
    xmap = CrystalMap(
        rotations=Rotation(quats),
        phase_id=phase_id,
        x=xs.astype(np.float64), y=ys.astype(np.float64),
        phase_list=PhaseList(phases=[
            Phase(name="Al", space_group=225, point_group="m-3m",
                  structure=Structure(
                      lattice=Lattice(4.0495, 4.0495, 4.0495, 90, 90, 90)))],
            ids=[1]),
        # the spherical indexer stores its score under both names
        prop={"ci": np.linspace(0.2, 0.6, rows.size).astype(np.float32),
              "scores": np.linspace(0.2, 0.6, rows.size).astype(np.float32)},
    )
    ci = np.linspace(0.2, 0.6, rows.size).astype(np.float32)
    metadata = {} if source_file is None else {"source_file": source_file}
    return IndexingResult(
        xmap=xmap,
        selection_mask=selection_mask.copy(),
        original_shape=(N_ROWS, N_COLS),
        method=IndexingMethod(method),
        confidence_scores=ci,
        metadata={
            **metadata,
            "indexing_method": method,
            "sht_paths_by_phase": {1: "/fake.sht"},
            "detector_geometry": {
                "pat_height": 8, "pat_width": 8,
                "pc_x": 0.5, "pc_y": 0.5, "pc_z": 0.6,
                "vendor": "Bruker", "pixel_size": 70.0,
                "sample_tilt": 70.0, "binning": 1,
            },
        },
    )


#: How the stage-3 fake solver reports the end of each pixel when asked for a
#: mix: flat pixels 0-1 failed (flag 0), 2-5 hit the iteration limit (flag 2),
#: the rest converged (flag 1) -- so 10 converged, 4 at the limit, 2 failed.
MIXED_FLAGS = {"failed": 2, "max_iter": 4, "converged": 10}


def _fake_solvers(jp=0.0, mixed_flags=False):
    """Replacements for the two LM solvers with a known answer.

    ``jp`` is the Jacobian column of the pattern-centre sensitivity the fake
    stage-1 solver reports. At 0 (the default) the refinement's degeneracy
    guard skips the smoothness solve, as it does in the real app when the
    autograd cannot flow through the pattern centre; a nonzero value lets the
    solve run.

    Pixels whose marker (the flat index, planted by the fake pattern reader)
    is below HALF are turned by TURN_DEG (in the sense that makes the Euler Phi
    grow); the others are not. The NCC is 0.9 / 0.5 so a refined-NCC map is
    told apart from anything else.
    """
    theta = float(np.deg2rad(TURN_DEG))

    def per_pixel(*, renderer, grid, I_exp, q0, pc0, **kw):
        marker = I_exp[:, 0, 0]
        omega = torch.zeros(q0.shape[0], 3, dtype=torch.float64)
        omega[marker < HALF, 0] = -theta
        return {
            "omega": omega,
            "dpc": torch.zeros(q0.shape[0], 3, dtype=torch.float64),
            "ncc_final": torch.where(marker < HALF, 0.9, 0.5).to(torch.float64),
            "J_p": torch.full((q0.shape[0], 3), float(jp), dtype=torch.float64),
            "convergence_flag": torch.ones(q0.shape[0], dtype=torch.int64),
            "iter_used": torch.ones(q0.shape[0], dtype=torch.int64),
        }

    def r_only(*, renderer, grid, I_exp, q0, pc_fixed, **kw):
        marker = I_exp[:, 0, 0]
        return {
            "omega": torch.zeros(q0.shape[0], 3, dtype=torch.float64),
            "ncc_final": torch.where(marker < HALF, 0.9, 0.5).to(torch.float64),
            "convergence_flag": (
                torch.where(marker < 2, 0, torch.where(marker < 6, 2, 1)).to(torch.int64)
                if mixed_flags else torch.ones(q0.shape[0], dtype=torch.int64)),
        }

    return per_pixel, r_only


@contextlib.contextmanager
def _fake_solvers_installed(jp=0.0, mixed_flags=False):
    """The two LM solvers, the SHT loader, the renderer and the pattern reader
    replaced for the duration (module attributes, so a worker thread started
    inside the block sees them too)."""
    from backend.api.services import refinement as rf

    per_pixel, r_only = _fake_solvers(jp, mixed_flags)
    n_cols = N_COLS
    with patch.object(rf, "load_or_get_phase", return_value=MagicMock()), \
         patch.object(rf, "get_renderer", return_value=MagicMock()), \
         patch.object(rf, "refine_per_pixel", per_pixel), \
         patch.object(rf, "refine_r_only", r_only), \
         patch.object(rf, "_get_experimental_pattern",
                      side_effect=lambda res, r, c: np.full(
                          (8, 8), r * n_cols + c, dtype=np.float32)):
        yield


def _run_refinement(parent, jp=0.0, mixed_flags=False, **kwargs):
    """Run the orchestrator on an already stored ``parent`` with the fake
    solvers. Returns the id of the refined result it registered."""
    from backend.api.services import refinement as rf

    with _fake_solvers_installed(jp, mixed_flags):
        rf.compute_full_refinement(parent, smoothness_lambda=0.01, **kwargs)
    return parent.metadata["refinement"]["refined_result_id"]


def _refine(parent, jp=0.0, mixed_flags=False):
    """Store ``parent`` and refine it with the fake solvers. Returns
    (parent_id, refined_id)."""
    from backend.api.routes import indexing

    parent_id = indexing._store_result(parent, "spherical")
    return parent_id, _run_refinement(parent, jp=jp, mixed_flags=mixed_flags)


def _euler_deg(result):
    return np.rad2deg(np.asarray(result.xmap.rotations.to_euler()))


def _angular_gap(a, b):
    """Smallest difference between angles in degrees, element-wise."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % 360.0
    return np.minimum(d, 360.0 - d)


def _decode_png(b64):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA"))


# --------------------------------------------------------------------------
# The listing
# --------------------------------------------------------------------------

def test_results_listing_answers_200_and_lists_parent_and_refined(client):
    parent_id, refined_id = _refine(_real_parent())

    resp = client.get("/api/indexing/results")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    by_id = {e["id"]: e for e in body["results"]}
    assert set(by_id) == {parent_id, refined_id}
    assert by_id[parent_id]["method"] == "spherical"
    assert by_id[refined_id]["method"] == "spherical"
    # The refinement covers the parent's pixels, no more and no fewer.
    assert by_id[refined_id]["n_indexed"] == by_id[parent_id]["n_indexed"] == N_PIX
    assert by_id[refined_id]["phases"] == [{"id": 1, "name": "Al"}]


def test_the_listing_takes_the_method_from_the_parent_not_a_fixed_value(client):
    parent_id, refined_id = _refine(_real_parent(method="dictionary"))

    by_id = {e["id"]: e for e in
             client.get("/api/indexing/results").json()["results"]}

    assert by_id[parent_id]["method"] == "dictionary"
    assert by_id[refined_id]["method"] == "dictionary"


def test_after_a_refinement_the_original_is_still_the_active_result(client):
    """The refined result is added next to the original;
    the original stays active until the user switches the view."""
    parent_id, refined_id = _refine(_real_parent())

    body = client.get("/api/indexing/results").json()

    assert body["active_id"] == parent_id
    by_id = {e["id"]: e for e in body["results"]}
    assert by_id[parent_id]["is_active"] is True
    assert by_id[refined_id]["is_active"] is False


def test_the_listing_marks_the_refined_result_and_names_its_parent(client):
    parent_id, refined_id = _refine(_real_parent())

    by_id = {e["id"]: e for e in
             client.get("/api/indexing/results").json()["results"]}

    assert by_id[refined_id]["is_refined"] is True
    assert by_id[refined_id]["refinement_of"] == parent_id
    assert by_id[parent_id]["is_refined"] is False
    assert by_id[parent_id]["refinement_of"] is None


def test_coming_back_to_a_file_re_activates_the_original_not_the_refined_result(
        client, tmp_path):
    """reactivate_result_for_source takes the LAST registry entry for a file;
    after a refinement that last entry is the refined result."""
    from backend.api.routes import indexing
    scan = str(tmp_path / "scan.h5oina")
    parent_id, refined_id = _refine(_real_parent(source_file=scan))
    # the refined result belongs to the same file as its parent
    assert (indexing._result_registry[refined_id].metadata["source_file"]
            == indexing._result_registry[parent_id].metadata["source_file"])

    picked = indexing.reactivate_result_for_source(scan)

    assert picked == parent_id
    assert indexing._active_result_id == parent_id


def test_refined_result_reports_the_parents_confidence_index(client):
    """The CI is the indexer's confidence; a refinement does not re-estimate
    it, so the refined entry carries the parent's, not the refined NCC."""
    parent_id, refined_id = _refine(_real_parent())

    by_id = {e["id"]: e for e in client.get("/api/indexing/results").json()["results"]}

    # mean of linspace(0.2, 0.6, 16) is 0.4; the refined NCC averages 0.7
    assert by_id[parent_id]["mean_ci"] == pytest.approx(0.4, abs=1e-4)
    assert by_id[refined_id]["mean_ci"] == by_id[parent_id]["mean_ci"]


def test_listing_with_a_region_of_interest_result(client):
    """An ROI result has fewer xmap rows than grid pixels; the refined result
    must keep that layout (the grid placement of every consumer relies on it)."""
    mask = np.zeros((N_ROWS, N_COLS), dtype=bool)
    mask[1:3, :] = True                      # 8 of 16 pixels
    parent = _real_parent(selection_mask=mask)
    parent_id, refined_id = _refine(parent)

    body = client.get("/api/indexing/results").json()
    by_id = {e["id"]: e for e in body["results"]}

    assert by_id[refined_id]["n_indexed"] == by_id[parent_id]["n_indexed"] == 8
    assert by_id[refined_id]["roi_bbox"] == by_id[parent_id]["roi_bbox"]
    from backend.api.routes import indexing
    refined = indexing._result_registry[refined_id]
    assert refined.xmap.rotations.size == 8


# --------------------------------------------------------------------------
# Activation
# --------------------------------------------------------------------------

def test_the_refined_result_can_be_activated_and_the_parent_re_activated(client):
    parent_id, refined_id = _refine(_real_parent())
    assert client.get("/api/indexing/results").json()["active_id"] == parent_id

    resp = client.post(f"/api/indexing/results/activate/{refined_id}")

    assert resp.status_code == 200, resp.text
    assert resp.json()["active_id"] == refined_id
    assert client.get("/api/indexing/results").json()["active_id"] == refined_id

    back = client.post(f"/api/indexing/results/activate/{parent_id}")

    assert back.status_code == 200, back.text
    assert client.get("/api/indexing/results").json()["active_id"] == parent_id


# --------------------------------------------------------------------------
# The orientations
# --------------------------------------------------------------------------

def test_the_refined_result_carries_the_refined_euler_angles_not_the_parents():
    from backend.api.routes import indexing
    parent = _real_parent()
    parent_id, refined_id = _refine(parent)
    refined = indexing._result_registry[refined_id]

    eulers = _euler_deg(refined)

    expected = np.array([[0.0, A_FIRST_REFINED if g < HALF else A_SECOND_PARENT, 0.0]
                         for g in range(N_PIX)])
    assert _angular_gap(eulers, expected).max() < 0.01
    # ... and the parent is untouched by the refinement
    parent_expected = np.array([[0.0, A_FIRST_PARENT if g < HALF else A_SECOND_PARENT, 0.0]
                                for g in range(N_PIX)])
    assert _angular_gap(_euler_deg(parent), parent_expected).max() < 0.01


def test_the_refined_xmap_is_a_real_crystal_map_with_the_parents_layout():
    from orix.crystal_map import CrystalMap
    from backend.api.routes import indexing
    parent = _real_parent()
    _, refined_id = _refine(parent)
    refined = indexing._result_registry[refined_id]

    assert isinstance(refined.xmap, CrystalMap)
    assert refined.xmap is not parent.xmap
    assert refined.xmap.shape == parent.xmap.shape
    np.testing.assert_array_equal(refined.xmap.phase_id, parent.xmap.phase_id)
    np.testing.assert_array_equal(refined.xmap.x, parent.xmap.x)
    np.testing.assert_array_equal(refined.xmap.y, parent.xmap.y)
    assert refined.xmap.phases.names == parent.xmap.phases.names
    # the parent's properties are still there, untouched
    np.testing.assert_array_equal(refined.xmap.prop["ci"], parent.xmap.prop["ci"])


def test_the_refined_ncc_lives_on_the_xmap_per_row_and_nan_where_nothing_was_refined():
    from backend.api.routes import indexing
    mask = np.zeros((N_ROWS, N_COLS), dtype=bool)
    mask[1:3, :] = True
    parent = _real_parent(selection_mask=mask)
    _, refined_id = _refine(parent)
    refined = indexing._result_registry[refined_id]

    ncc = np.asarray(refined.xmap.prop["refined_ncc"])

    # rows are the 8 indexed pixels in flat order: flat indices 4..11, the fake
    # solver scores flat indices < 8 with 0.9 and the rest with 0.5
    expected = np.array([0.9, 0.9, 0.9, 0.9, 0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(ncc, expected, atol=1e-6)


def test_the_refined_result_does_not_share_arrays_with_the_parent():
    from backend.api.routes import indexing
    parent = _real_parent()
    _, refined_id = _refine(parent)
    refined = indexing._result_registry[refined_id]

    refined.selection_mask[:] = False
    refined.confidence_scores[:] = 0.0
    refined.xmap.prop["ci"] = np.zeros(N_PIX)      # writes in place when the key exists
    refined.xmap.phase_id[:] = 7

    assert parent.selection_mask.all()
    assert np.all(parent.confidence_scores > 0.1)
    assert np.all(parent.xmap.prop["ci"] > 0.1)
    assert np.all(parent.xmap.phase_id == 1)


def test_the_refined_result_keeps_the_parents_eds_data_and_a_missing_mask():
    from backend.api.routes import indexing
    parent = _real_parent()
    parent.eds_counts = {"counts": {"Al": np.arange(N_PIX)}, "n_rows": N_ROWS,
                         "n_cols": N_COLS}
    parent.phase_compositions = {"Al": {"Al": 100.0}}
    parent.selection_mask = None
    _, refined_id = _refine(parent)
    refined = indexing._result_registry[refined_id]

    assert refined.eds_counts == parent.eds_counts
    assert refined.phase_compositions == {"Al": {"Al": 100.0}}
    assert refined.selection_mask is None


def test_unindexed_rows_of_a_full_grid_map_are_left_alone():
    """A full-grid map has one row per pixel, indexed or not. The refinement
    must put its numbers on the rows of the pixels it refined, and leave the
    rows of the unindexed ones exactly as the parent had them."""
    from backend.api.routes import indexing
    parent = _real_parent(unindexed_flat=[0, 1, 2, 3])   # the whole first row
    _, refined_id = _refine(parent)
    refined = indexing._result_registry[refined_id]

    eulers = _euler_deg(refined)
    ncc = np.asarray(refined.xmap.prop["refined_ncc"])

    # Rows 0..3: unindexed, still the parent's 40 degrees (flat < HALF) -- not 43.
    # Rows 4..7: refined, 40 -> 43.  Rows 8..15: refined but not turned, 25.
    expected_phi = np.array([40.0] * 4 + [43.0] * 4 + [25.0] * 8)
    np.testing.assert_allclose(eulers[:, 1], expected_phi, atol=0.01)
    # NCC: NaN where nothing was refined; 0.9 for flat < 8, 0.5 beyond.
    assert np.isnan(ncc[:4]).all()
    np.testing.assert_allclose(ncc[4:], [0.9] * 4 + [0.5] * 8, atol=1e-6)
    np.testing.assert_array_equal(refined.xmap.phase_id,
                                  [-1] * 4 + [1] * 12)


def test_a_region_of_interest_result_gets_the_refined_euler_angles_on_the_right_rows():
    """The 8 selected pixels are flat 4..11 and sit on xmap rows 0..7. Flat
    4..7 are turned (40 -> 43 degrees), flat 8..11 are not (25). A reversed or
    shifted row order puts 25 where 43 belongs and the other way round."""
    from backend.api.routes import indexing
    mask = np.zeros((N_ROWS, N_COLS), dtype=bool)
    mask[1:3, :] = True
    parent = _real_parent(selection_mask=mask)
    _, refined_id = _refine(parent)
    refined = indexing._result_registry[refined_id]

    eulers = _euler_deg(refined)

    np.testing.assert_allclose(eulers[:, 1], [43.0] * 4 + [25.0] * 4, atol=0.01)


def test_unindexed_pixels_inside_a_region_of_interest_do_not_shift_the_rows():
    """A region-of-interest result has one xmap row per SELECTED pixel,
    including a selected pixel that stayed unindexed. Selected: flat 4..15
    (xmap rows 0..11); flat 5 (row 1) is unindexed. Expected, written down by
    hand: Phi per row = flat 4, 5, 6, 7 -> 43, 40, 43, 43 (flat 5 is left as
    the parent had it), flat 8..15 -> 25 for all eight."""
    from backend.api.routes import indexing
    mask = np.zeros((N_ROWS, N_COLS), dtype=bool)
    mask[1:, :] = True
    parent = _real_parent(selection_mask=mask, unindexed_flat=[5])
    _, refined_id = _refine(parent)
    refined = indexing._result_registry[refined_id]

    eulers = _euler_deg(refined)
    ncc = np.asarray(refined.xmap.prop["refined_ncc"])

    np.testing.assert_allclose(
        eulers[:, 1], [43.0, 40.0, 43.0, 43.0] + [25.0] * 8, atol=0.01)
    # NCC: 0.9 for the refined flat 4, 6, 7; NaN for the unindexed flat 5;
    # 0.5 for flat 8..15
    np.testing.assert_allclose(
        ncc, [0.9, np.nan, 0.9, 0.9] + [0.5] * 8, atol=1e-6)
    np.testing.assert_array_equal(refined.xmap.phase_id, [1, -1] + [1] * 10)


def test_the_refined_orientations_go_into_the_rows_that_are_in_the_data():
    """``_refined_crystal_map`` on a map with rows outside the data (what an
    orix slice leaves behind): the in-data rows take the new quaternions, the
    others keep the parent's."""
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation
    from backend.api.services.refinement import _refined_crystal_map

    parent_q = np.array([_euler_0_a_0(a) for a in (10.0, 20.0, 30.0, 50.0)])
    xmap = CrystalMap(
        rotations=Rotation(parent_q),
        phase_id=np.zeros(4, dtype=np.int32),
        x=np.arange(4, dtype=np.float64), y=np.zeros(4),
        phase_list=PhaseList(phases=[Phase(name="Al", point_group="m-3m")]),
        is_in_data=np.array([True, False, True, False]),
    )
    new_q = np.array([_euler_0_a_0(a) for a in (11.0, 31.0)])   # for rows 0 and 2

    refined = _refined_crystal_map(xmap, new_q, np.array([0.8, 0.6]))

    full_phi = np.rad2deg(np.asarray(refined._rotations.to_euler()))[:, 1]
    np.testing.assert_allclose(full_phi, [11.0, 20.0, 31.0, 50.0], atol=0.01)
    np.testing.assert_allclose(refined.prop["refined_ncc"], [0.8, 0.6], atol=1e-6)
    # ... and the rows outside the data hold NaN ("nothing was refined here"),
    # not the 0.0 an unset orix property array starts with (a 0.0 reads as a
    # refined pixel whose pattern does not match at all)
    stored = np.asarray(dict.__getitem__(refined.prop, "refined_ncc"))
    np.testing.assert_allclose(stored, [0.8, np.nan, 0.6, np.nan], atol=1e-6)
    # the parent keeps its own
    np.testing.assert_allclose(
        np.rad2deg(np.asarray(xmap._rotations.to_euler()))[:, 1],
        [10.0, 20.0, 30.0, 50.0], atol=0.01)


# --------------------------------------------------------------------------
# A map drawn from the refined result
# --------------------------------------------------------------------------

def test_the_ipf_z_layer_of_the_refined_result_renders_and_differs_where_it_was_turned(client):
    parent_id, refined_id = _refine(_real_parent())

    assert client.post(f"/api/indexing/results/activate/{parent_id}").status_code == 200
    r_parent = client.get("/api/phasemap/layer", params={"kind": "ipf-z"})
    assert client.post(f"/api/indexing/results/activate/{refined_id}").status_code == 200
    r_refined = client.get("/api/phasemap/layer", params={"kind": "ipf-z"})

    assert r_parent.status_code == 200, r_parent.text
    assert r_refined.status_code == 200, r_refined.text
    img_p = _decode_png(r_parent.json()["image"]).astype(int)
    img_r = _decode_png(r_refined.json()["image"]).astype(int)
    assert img_p.shape == img_r.shape == (N_ROWS, N_COLS, 4)
    changed = np.abs(img_p - img_r).sum(axis=-1)
    flat_changed = changed.ravel()
    # first half was turned by 3 degrees: its colour must have moved
    assert (flat_changed[:HALF] > 0).all(), flat_changed
    # second half was not: bit-identical colour
    assert (flat_changed[HALF:] == 0).all(), flat_changed


def test_the_refinement_layers_render_right_after_a_refinement(client):
    """The six refinement layers read the refinement block of the ACTIVE
    result. That block sits on the parent, so the parent has to be the active
    one after a compute -- with the refined result active they answered 404
    while the layer picker offered them."""
    _refine(_real_parent())

    resp = client.get("/api/phasemap/layer", params={"kind": "refined_ncc"})

    assert resp.status_code == 200, resp.text


def test_the_confidence_overlay_of_the_refined_result_shows_the_parents_score_like_an_ordinary_one(client):
    """The PNG confidence overlay reads ``xmap.scores``: for an ordinary
    spherical result that is the indexer's score (the same numbers as the CI),
    and the refined result carries the parent's, since a refinement does not
    re-estimate it. It must not turn into the refined NCC (0.9 / 0.5 here) --
    a different quantity that has a layer of its own."""
    parent_id, refined_id = _refine(_real_parent())

    def render(result_id, overlay):
        assert client.post(f"/api/indexing/results/activate/{result_id}").status_code == 200
        r = client.get("/api/phasemap/render",
                       params={"direction": "phase", "confidence_overlay": overlay,
                               "show_ipf_keys": False})
        assert r.status_code == 200, r.text
        return _decode_png(r.json()["image"]).astype(int)

    plain = render(refined_id, False)
    refined_with = render(refined_id, True)
    parent_with = render(parent_id, True)

    assert np.abs(refined_with - plain).sum() > 0        # the overlay is drawn at all
    np.testing.assert_array_equal(refined_with, parent_with)


# --------------------------------------------------------------------------
# Routes that read the active result's method
# --------------------------------------------------------------------------

def test_exporting_the_refined_result_as_ang_writes_the_refined_orientations(
        client, tmp_path, monkeypatch):
    # A freshly indexed result has no step size of its own; the export reads
    # it off the loaded scan. No scan is loaded here, so stand in for it.
    from backend.api.routes import indexing
    monkeypatch.setattr(indexing, "_resolve_step_size_um", lambda active: 0.5)
    parent_id, refined_id = _refine(_real_parent())
    assert client.post(f"/api/indexing/results/activate/{refined_id}").status_code == 200

    out = tmp_path / "refined.ang"
    resp = client.post("/api/indexing/export",
                       json={"format": "ang", "filename": str(out)})

    assert resp.status_code == 200, resp.text
    rows = [ln.split() for ln in out.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]
    eulers_deg = np.rad2deg(np.array([[float(v) for v in r[:3]] for r in rows]))
    # .ang is written row by row over the grid; the first row of the grid is flat 0..3
    assert len(rows) == N_PIX
    # Phi (second Euler angle) of the first four pixels: 43 degrees, not the parent's 40
    np.testing.assert_allclose(eulers_deg[:4, 1], A_FIRST_REFINED, atol=0.01)
    np.testing.assert_allclose(eulers_deg[HALF:, 1], A_SECOND_PARENT, atol=0.01)


# --------------------------------------------------------------------------
# No orphaned refined results
# --------------------------------------------------------------------------

def test_a_new_refinement_replaces_the_previous_refined_result(client):
    """Re-refine (full recompute) and re-smooth (stage-1 cache) each used to
    add another refined result: unreachable, and counted towards the cap."""
    from backend.api.routes import indexing
    parent = _real_parent()
    parent_id, first = _refine(parent)

    second = _run_refinement(parent, force_full_recompute=True)    # re-refine
    third = _run_refinement(parent)                                # re-smooth

    assert len({first, second, third}) == 3
    assert set(indexing._result_registry) == {parent_id, third}
    assert parent.metadata["refinement"]["refined_result_id"] == third
    listed = {e["id"] for e in client.get("/api/indexing/results").json()["results"]}
    assert listed == {parent_id, third}


def test_replacing_the_refined_result_you_are_looking_at_keeps_you_on_a_refined_one(client):
    from backend.api.routes import indexing
    parent = _real_parent()
    parent_id, first = _refine(parent)
    assert client.post(f"/api/indexing/results/activate/{first}").status_code == 200

    second = _run_refinement(parent)

    assert first not in indexing._result_registry
    assert indexing._active_result_id == second


def test_a_refinement_of_one_parent_leaves_the_refined_result_of_another_alone():
    from backend.api.routes import indexing
    parent_a = _real_parent()
    parent_b = _real_parent()
    id_a, refined_a = _refine(parent_a)
    id_b, refined_b = _refine(parent_b)

    new_a = _run_refinement(parent_a)

    assert set(indexing._result_registry) == {id_a, id_b, refined_b, new_a}


def test_a_failed_refinement_leaves_no_refined_result_and_no_block():
    from backend.api.routes import indexing
    from backend.api.services import refinement as rf
    parent = _real_parent()
    parent_id = indexing._store_result(parent, "spherical")

    # the summary statistics run after the refined result is built
    with patch.object(rf, "summary_stats", side_effect=RuntimeError("boom")), \
         pytest.raises(RuntimeError, match="boom"):
        _run_refinement(parent)

    assert set(indexing._result_registry) == {parent_id}
    assert "refinement" not in parent.metadata
    assert indexing._active_result_id == parent_id


class _MetadataFailingOn(dict):
    """A metadata dict that refuses to take one key."""
    def __init__(self, data, refused):
        super().__init__(data)
        self._refused = refused

    def __setitem__(self, key, value):
        if key == self._refused:
            raise RuntimeError(f"cannot write {key}")
        super().__setitem__(key, value)


def test_a_failure_while_writing_the_parents_block_removes_the_new_refined_result():
    """The refined result is registered right before the parent's block is
    written. If the write fails, the parent keeps what it had -- including
    the previous refinement and its refined result."""
    from backend.api.routes import indexing
    parent = _real_parent()
    parent_id, first = _refine(parent)
    old_block = parent.metadata["refinement"]
    parent.metadata = _MetadataFailingOn(parent.metadata, "_refinement_stage1_cache")

    with pytest.raises(RuntimeError, match="cannot write"):
        _run_refinement(parent)

    assert set(indexing._result_registry) == {parent_id, first}
    assert parent.metadata["refinement"] is old_block
    assert parent.metadata["refinement"]["refined_result_id"] == first


def _run_compute_route(client, parent_id, *, after_compute):
    """POST /api/refinement/compute and wait for the worker thread. The real
    orchestrator runs (fake solvers); ``after_compute(job)`` runs right after
    it returns, which is where a cancel or a replaced parent lands."""
    from backend.api.services import refinement as rf

    real = rf.compute_full_refinement
    seen = {}

    def compute_then(result, smoothness_lambda, **kw):
        real(result, smoothness_lambda, **kw)
        after_compute(kw["cancel_event"])

    with _fake_solvers_installed(), \
         patch.object(rf, "compute_full_refinement", compute_then):
        resp = client.post("/api/refinement/compute",
                           json={"result_id": parent_id, "smoothness_lambda": 0.01})
        assert resp.status_code == 202, resp.text
        job = rf.JOB_REGISTRY[resp.json()["job_id"]]
        job.thread.join(timeout=60)
        assert not job.thread.is_alive()
        seen["state"] = job.state
    return seen["state"]


def test_cancelling_a_finished_compute_removes_the_refined_result_it_made(client):
    from backend.api.routes import indexing
    parent = _real_parent()
    parent_id = indexing._store_result(parent, "spherical")

    state = _run_compute_route(client, parent_id,
                               after_compute=lambda cancel_event: cancel_event.set())

    assert state == "cancelled"
    assert set(indexing._result_registry) == {parent_id}
    assert "refinement" not in parent.metadata
    assert indexing._active_result_id == parent_id


def test_a_parent_replaced_during_the_compute_does_not_leave_its_refined_result(client):
    from backend.api.routes import indexing
    parent = _real_parent()
    parent_id = indexing._store_result(parent, "spherical")

    def replace_parent(_cancel_event):
        indexing._result_registry[parent_id] = _real_parent()

    state = _run_compute_route(client, parent_id, after_compute=replace_parent)

    assert state == "failed"
    assert set(indexing._result_registry) == {parent_id}


# --------------------------------------------------------------------------
# A user's own result is never taken for a refinement's
# --------------------------------------------------------------------------

def test_a_new_refinement_only_replaces_the_refined_result_it_made_itself():
    """A result that merely carries the refined flags -- what the grain-phase
    route stored before it stopped copying them -- is not the parent's refined
    result: re-smoothing the parent must leave it registered."""
    import copy
    from backend.api.routes import indexing
    parent = _real_parent()
    parent_id, first = _refine(parent)
    lookalike = copy.deepcopy(indexing._result_registry[first])
    assert lookalike.metadata["is_refined"] is True
    assert lookalike.metadata["refinement_of"] == parent_id
    lookalike_id = indexing._store_result(lookalike, "grain", activate=False)

    second = _run_refinement(parent)

    assert first not in indexing._result_registry          # it made this one: replaced
    assert set(indexing._result_registry) == {parent_id, second, lookalike_id}


def test_refining_a_copy_of_a_refined_parent_does_not_delete_the_originals_refined_result():
    """A result derived from the parent by deep copy carries the parent's
    refinement block, so its ``refined_result_id`` names the PARENT's refined
    result. Refining the copy must not take that result away."""
    import copy
    from backend.api.routes import indexing
    parent = _real_parent()
    parent_id, refined_id = _refine(parent)
    derived = copy.deepcopy(parent)
    assert derived.metadata["refinement"]["refined_result_id"] == refined_id
    derived_id = indexing._store_result(derived, "grain", activate=False)

    new_refined = _run_refinement(derived, force_full_recompute=True)

    assert refined_id in indexing._result_registry
    assert set(indexing._result_registry) == {
        parent_id, refined_id, derived_id, new_refined}
    assert parent.metadata["refinement"]["refined_result_id"] == refined_id
    assert derived.metadata["refinement"]["refined_result_id"] == new_refined


def test_a_result_derived_from_the_refined_view_is_an_ordinary_result(client, tmp_path):
    """Made while the refined result is shown, a derived result (the
    grain-phase route's) must not be labelled "Refined", must be a candidate
    when the file is loaded again, and must survive the parent's next
    refinement."""
    import copy
    from backend.api.routes import indexing
    scan = str(tmp_path / "scan.h5oina")
    parent = _real_parent(source_file=scan)
    parent_id, refined_id = _refine(parent)
    derived = copy.deepcopy(indexing._result_registry[refined_id])
    derived.metadata = indexing._metadata_of_derived_result(derived.metadata)
    derived_id = indexing._store_result(derived, "grain")

    by_id = {e["id"]: e for e in
             client.get("/api/indexing/results").json()["results"]}
    assert by_id[derived_id]["is_refined"] is False
    assert by_id[derived_id]["refinement_of"] is None
    assert by_id[refined_id]["is_refined"] is True
    assert indexing.reactivate_result_for_source(scan) == derived_id

    second = _run_refinement(parent)

    assert set(indexing._result_registry) == {parent_id, derived_id, second}


# --------------------------------------------------------------------------
# What the refined result keeps of the indexing run it comes from
# --------------------------------------------------------------------------

def test_the_per_phase_ci_layer_of_the_refined_result_equals_the_parents(client):
    """Same pixels, same indexing run: the per-phase confidence maps are the
    parent's. The refined metadata used to be a new dict without them, so the
    layer was an error (a blank layer) in the refined view."""
    parent = _real_parent()
    # the array form the layer reads (what a loaded result file provides)
    parent.metadata["per_phase_data"] = {
        "Al": np.linspace(0.05, 0.95, N_PIX).astype(np.float32)}
    parent_id, refined_id = _refine(parent)

    def layer(result_id):
        assert client.post(f"/api/indexing/results/activate/{result_id}").status_code == 200
        r = client.get("/api/phasemap/layer", params={"kind": "ci_Al"})
        assert r.status_code == 200, r.text
        return _decode_png(r.json()["image"]).astype(int)

    img_parent = layer(parent_id)
    img_refined = layer(refined_id)

    assert img_parent[..., 3].min() > 0          # drawn: every pixel has a value
    np.testing.assert_array_equal(img_refined, img_parent)


def test_the_refined_result_keeps_the_dictionary_and_the_run_description_of_its_parent():
    from backend.api.routes import indexing
    parent = _real_parent(method="dictionary")
    dictionary, signal = object(), object()
    per_phase = {"Al": {"ci": np.arange(N_PIX, dtype=np.float32), "ci_mean": 0.4}}
    parent.metadata.update({
        "dictionary": dictionary, "dict_path": "dict_Al.h5", "signal": signal,
        "metric": "ncc", "keep_n": 3, "per_phase_data": per_phase,
        "step_size_um": 0.5,
    })
    _, refined_id = _refine(parent)
    md = indexing._result_registry[refined_id].metadata

    assert md["dictionary"] is dictionary
    assert md["signal"] is signal
    assert md["dict_path"] == "dict_Al.h5"
    assert md["metric"] == "ncc"
    assert md["keep_n"] == 3
    assert md["step_size_um"] == 0.5
    assert md["per_phase_data"] is per_phase
    # ... but what the refinement itself says about the refined result stays its own
    assert md["is_refined"] is True
    assert md["indexing_method"] == "dictionary"
    assert md["pc_per_pixel"] is not None


def test_the_refined_result_does_not_inherit_the_parents_diagnostics_or_refinement_block():
    """The forward diagnostics and the refinement block describe the PARENT's
    orientations; they must not turn up on the refined result."""
    from backend.api.routes import indexing
    parent = _real_parent()
    parent.metadata["forward_diagnostics"] = {"bandwidth": 128}
    parent_id, first = _refine(parent)
    second = _run_refinement(parent, force_full_recompute=True)
    md = indexing._result_registry[second].metadata

    for key in ("forward_diagnostics", "refinement", "_refinement_stage1_cache",
                "assignment_source", "cleaned_phase_id"):
        assert key not in md, key


# --------------------------------------------------------------------------
# The citation trail of the refined result
# --------------------------------------------------------------------------

def _parent_with_an_indexing_step():
    from backend.api.services.citations.provenance import record_step
    parent = _real_parent()
    record_step(parent, "indexing.spherical",
                {"bandwidth": 88, "orienta_version": "0.4.7"})
    return parent


def test_the_refined_result_carries_the_parents_trail_plus_one_refinement_step():
    """In the refined view the Citations panel reads the refined result's
    trail. It used to be empty: the metadata was built from scratch and the
    refinement recorded nothing."""
    import copy
    from backend.api.routes import indexing
    from backend.api.services.citations.provenance import get_steps
    parent = _parent_with_an_indexing_step()
    parent_before = copy.deepcopy(get_steps(parent))
    parent_id, refined_id = _refine(parent)

    steps = get_steps(indexing._result_registry[refined_id])

    assert steps[:1] == parent_before
    assert [s["key"] for s in steps] == ["indexing.spherical", "refinement.joint_r_pc"]
    params = steps[1]["params"]
    assert params["smoothness_lambda"] == 0.01
    assert params["n_pixels_refined"] == N_PIX
    assert params["bandwidth"] == 256            # the bandwidth the refinement renders at
    assert isinstance(params["outer_iterations"], int) and params["outer_iterations"] >= 1
    assert params["outer_iterations"] == parent.metadata["refinement"]["outer_iterations"]
    assert isinstance(params["orienta_version"], str)
    # the bounds that cap the result (hand-written: 5 deg of total orientation
    # change, +-10 px of pattern-centre shift, +-1000 um of distance change)
    assert params["orientation_bound_deg"] == 5.0
    assert params["pc_bound_px"] == 10.0
    assert params["distance_bound_um"] == 1000.0
    # what the summary of this run holds: the fake solvers converge every pixel
    assert params["n_converged"] == N_PIX
    assert params["n_max_iter"] == 0
    assert params["n_failed"] == 0
    # the fake solver reports no pattern-centre sensitivity: the smoothness
    # solve was skipped, and the block and the step both say so
    assert params["smoothness_applied"] is False
    assert parent.metadata["refinement"]["smoothness_applied"] is False
    # the parent's own trail is not changed by being refined
    assert get_steps(parent) == parent_before


def test_a_new_refinement_of_the_same_parent_still_records_the_step_once():
    from backend.api.routes import indexing
    from backend.api.services.citations.provenance import get_steps
    parent = _parent_with_an_indexing_step()
    _refine(parent)
    second = _run_refinement(parent, force_full_recompute=True)

    keys = [s["key"] for s in get_steps(indexing._result_registry[second])]

    assert keys == ["indexing.spherical", "refinement.joint_r_pc"]
    assert [s["key"] for s in get_steps(parent)] == ["indexing.spherical"]


def test_the_methods_paragraph_of_the_refined_result_names_the_run_and_the_refinement(client):
    parent_id, refined_id = _refine(_parent_with_an_indexing_step())

    body = client.get(f"/api/citations/result/{refined_id}").json()
    methods = body["methods"]
    parent_methods = client.get(f"/api/citations/result/{parent_id}").json()["methods"]

    assert "spherical-harmonic-transform indexing at bandwidth 88" in methods
    assert "refined" in methods
    assert f"{N_PIX} pixels" in methods
    assert body["undeclared"] == []
    # the parent's paragraph says nothing about a refinement it was not part of
    assert "refined" not in parent_methods
    assert "spherical-harmonic-transform indexing at bandwidth 88" in parent_methods


def test_the_methods_sentence_names_the_bounds_and_the_counts(client):
    """The bounds cap the result in practice (orientation change up to 5 deg,
    many pixels at the limit); a methods paragraph that left them out would
    present a clipped result as a free fit."""
    _, refined_id = _refine(_parent_with_an_indexing_step())

    methods = client.get(f"/api/citations/result/{refined_id}").json()["methods"]

    assert "total orientation change was limited to 5 deg" in methods
    assert "pattern-centre shift to 10 px in x and y" in methods
    assert "sample-to-detector distance to 1000 um" in methods
    assert (f"Of the {N_PIX} pixels refined, {N_PIX} converged, "
            "0 stopped at the iteration limit and 0 failed") in methods


def test_the_sentence_reports_how_the_pixels_ended_by_kind(client):
    """10 converged, 4 stopped at the iteration limit, 2 failed (hand-written
    in MIXED_FLAGS): each count lands in its own slot."""
    _, refined_id = _refine(_parent_with_an_indexing_step(), mixed_flags=True)

    methods = client.get(f"/api/citations/result/{refined_id}").json()["methods"]

    assert (f"Of the {N_PIX} pixels refined, {MIXED_FLAGS['converged']} converged, "
            f"{MIXED_FLAGS['max_iter']} stopped at the iteration limit and "
            f"{MIXED_FLAGS['failed']} failed") in methods


def test_when_the_smoothness_step_ran_the_sentence_says_so(client):
    parent = _parent_with_an_indexing_step()
    _, refined_id = _refine(parent, jp=1.0)

    methods = client.get(f"/api/citations/result/{refined_id}").json()["methods"]

    assert parent.metadata["refinement"]["smoothness_applied"] is True
    assert ("with a spatial smoothness constraint on the pattern-centre "
            "correction (smoothness weight 0.01)") in methods
    assert "without smoothing" not in methods


def test_when_the_smoothness_step_was_skipped_the_sentence_does_not_claim_it(client):
    """The degeneracy guard skips the smoothness solve when the pattern-centre
    sensitivity is zero. The sentence must not describe a step that did not run."""
    parent = _parent_with_an_indexing_step()
    _, refined_id = _refine(parent, jp=0.0)

    methods = client.get(f"/api/citations/result/{refined_id}").json()["methods"]

    assert parent.metadata["refinement"]["smoothness_applied"] is False
    assert "smoothness constraint" not in methods
    assert "smoothness weight" not in methods
    assert "without smoothing of the pattern-centre correction" in methods


# --------------------------------------------------------------------------
# The refinement summary is valid JSON
# --------------------------------------------------------------------------

def _strict_json(text):
    """Parse like a browser would: NaN / Infinity are not JSON."""
    def refuse(constant):
        raise ValueError(f"{constant} is not valid JSON")
    import json
    return json.loads(text, parse_constant=refuse)


def test_the_summary_after_a_refinement_without_forward_diagnostics_is_valid_json(client):
    """Nobody has to compute the forward diagnostics before refining. Without
    them there is no earlier NCC to compare with: the improvement is "not
    available" (null), not a block of NaNs -- the route answered 500 on those
    ("Out of range float values are not JSON compliant"), and the panel then
    reported a refinement that had worked as failed."""
    parent_id, _ = _refine(_real_parent())

    resp = client.get("/api/refinement/summary", params={"result_id": parent_id})

    assert resp.status_code == 200, resp.text
    summary = _strict_json(resp.text)["summary"]
    assert summary["ncc_improvement"] is None
    assert summary["n_pixels_refined"] == N_PIX
    # everything else in the summary is a number: the fake solver moves no pixel
    # centre, so the pixel-centre deltas are 0 for every pixel
    assert summary["pc_delta_x_px"]["mean"] == 0.0
    assert summary["pc_delta_x_px"]["max"] == 0.0


def test_the_summary_reports_the_ncc_improvement_when_the_diagnostics_were_computed(client):
    """Prior NCC 0.3 everywhere; the fake solver refines flat pixels 0..7 to
    0.9 and 8..15 to 0.5, so the improvements are 0.6 (eight pixels) and
    0.2 (eight pixels): mean 0.4, min 0.2, max 0.6 (written down by hand)."""
    parent = _real_parent()
    parent.metadata["forward_diagnostics"] = {
        "ncc_map": np.full((N_ROWS, N_COLS), 0.3, dtype=np.float32)}
    parent_id, _ = _refine(parent)

    resp = client.get("/api/refinement/summary", params={"result_id": parent_id})

    assert resp.status_code == 200, resp.text
    improvement = _strict_json(resp.text)["summary"]["ncc_improvement"]
    assert improvement["mean"] == pytest.approx(0.4, abs=1e-5)
    assert improvement["min"] == pytest.approx(0.2, abs=1e-5)
    assert improvement["max"] == pytest.approx(0.6, abs=1e-5)


def test_a_summary_map_that_is_all_nan_cannot_make_the_route_answer_500(client):
    """Every pixel failed, say: summary_stats of an all-NaN array has NaN for
    every scalar. The route turns whatever non-finite number the block holds
    into null instead of failing to serialise it."""
    from backend.api.routes import indexing
    parent = _real_parent()
    parent_id, _ = _refine(parent)
    nan = float("nan")
    all_nan = {"mean": nan, "std": nan, "min": nan, "max": float("inf"),
               "p05": nan, "p95": -float("inf"),
               "histogram": {"bins": [0.0, 1.0], "counts": [0]}}
    indexing._result_registry[parent_id].metadata["refinement"]["summary"][
        "orientation_delta_deg"] = all_nan

    resp = client.get("/api/refinement/summary", params={"result_id": parent_id})

    assert resp.status_code == 200, resp.text
    got = _strict_json(resp.text)["summary"]["orientation_delta_deg"]
    assert got["mean"] is None and got["max"] is None and got["p95"] is None
    assert got["histogram"] == {"bins": [0.0, 1.0], "counts": [0]}


# --------------------------------------------------------------------------
# The result cap never takes the parent away from its refinement
# --------------------------------------------------------------------------

def _fillers(n):
    """``n`` throw-away results, stored without becoming the active one."""
    from types import SimpleNamespace
    from backend.api.routes import indexing
    return [indexing._store_result(SimpleNamespace(xmap=None, metadata={}),
                                   "filler", activate=False)
            for _ in range(n)]


def test_storing_the_refined_result_at_the_cap_does_not_evict_its_parent():
    """12 results registered, the parent the oldest, a refined result active.
    A re-smooth stores one more: the cap evicts the oldest entry that is not
    the active one -- the parent -- and the refinement's own bookkeeping then
    forgets the previous refined result as well. Parent, old and new refined
    result all gone."""
    from backend.api.routes import indexing
    parent = _real_parent()
    parent_id, first = _refine(parent)               # parent is the oldest entry
    fillers = _fillers(indexing.MAX_STORED_RESULTS - 2)
    assert len(indexing._result_registry) == indexing.MAX_STORED_RESULTS
    assert next(iter(indexing._result_registry)) == parent_id
    indexing._active_result_id = first               # the user is looking at the refined view

    second = _run_refinement(parent)

    assert parent_id in indexing._result_registry
    assert second in indexing._result_registry
    assert parent.metadata["refinement"]["refined_result_id"] == second
    # what the user did not ask for is still there ...
    assert set(fillers) <= set(indexing._result_registry)
    # ... and only the replaced refined result is gone
    assert first not in indexing._result_registry
    assert set(indexing._result_registry) == {parent_id, second} | set(fillers)
    assert indexing._active_result_id == second


def test_the_refinement_stops_when_its_parent_left_the_registry_while_it_was_stored():
    """Nothing the refinement replaces may be forgotten for a parent that is
    no longer registered: the previous refined result and the parent's
    previous block stay as they were, and the new refined result goes."""
    from backend.api.routes import indexing
    parent = _real_parent()
    parent_id, first = _refine(parent)
    old_block = parent.metadata["refinement"]
    real_store = indexing._store_result

    def store_then_lose_the_parent(result, method_name, **kwargs):
        rid = real_store(result, method_name, **kwargs)
        indexing._result_registry.pop(parent_id)
        return rid

    with patch.object(indexing, "_store_result", store_then_lose_the_parent), \
         pytest.raises(RuntimeError, match="left the registry"):
        _run_refinement(parent)

    assert set(indexing._result_registry) == {first}
    assert parent.metadata["refinement"] is old_block


# --------------------------------------------------------------------------
# _store_result: the cap, with and without protected results
# --------------------------------------------------------------------------

def test_the_cap_evicts_the_oldest_result_that_is_not_the_active_one():
    """The behaviour every caller but the refinement relies on."""
    from backend.api.routes import indexing
    ids = _fillers(indexing.MAX_STORED_RESULTS)
    assert len(indexing._result_registry) == indexing.MAX_STORED_RESULTS

    newest = _fillers(1)[0]

    assert ids[0] not in indexing._result_registry
    assert set(indexing._result_registry) == set(ids[1:]) | {newest}


def test_the_cap_never_goes_past_the_active_result():
    """An oldest entry that is the active result ends the eviction: the cap is
    exceeded rather than the active result (or anything newer) dropped."""
    from backend.api.routes import indexing
    ids = _fillers(indexing.MAX_STORED_RESULTS)
    indexing._active_result_id = ids[0]

    newest = _fillers(1)[0]

    assert set(indexing._result_registry) == set(ids) | {newest}
    assert indexing._active_result_id == ids[0]


def test_a_kept_result_is_skipped_by_the_cap_and_the_next_oldest_goes():
    from types import SimpleNamespace
    from backend.api.routes import indexing
    ids = _fillers(indexing.MAX_STORED_RESULTS)

    newest = indexing._store_result(SimpleNamespace(xmap=None, metadata={}),
                                    "filler", activate=False, keep=(ids[0],))

    assert ids[0] in indexing._result_registry
    assert ids[1] not in indexing._result_registry
    assert set(indexing._result_registry) == {ids[0]} | set(ids[2:]) | {newest}
