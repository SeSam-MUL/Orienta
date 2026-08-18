"""Tests for the ``format='h5_light'`` branch of ``POST /api/indexing/export``.

The light branch was added 2026-05-26 to give users a MATLAB-compatible
alternative to the rich H5 export — the rich one inherits superblock v2
from the source h5oina (≥27 GB on our real datasets) and trips MATLAB
R2019b's h5info with "H5Fget_obj_count not a file id".

What we verify:

* The endpoint accepts ``format='h5_light'`` (rejected before this change).
* The output is a real HDF5 file with superblock **v0** — that is the
  on-disk property MATLAB actually depends on, independent of file size.
* The output is small: no source patterns copied, just the indexing
  payload. The test fabricates a no-source result and asserts the file
  is well under 1 MB (real-world expectation: 5–50 MB on a 2k×2k scan).
* Core groups are present (``/Indexing``, ``/SourceReference``,
  ``/Documentation``) with the 1-based phase_id convention preserved on
  ``/Indexing/phase_id`` so the Phase Map reader doesn't drop phase-0.
* Bogus ``format`` values still 400.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Rotation

from backend.api.main import app
from backend.api.routes import indexing as indexing_mod
from indexing_controller import IndexingMethod, IndexingResult


# ---------------------------------------------------------------------------
# Fixture: a minimal single-phase IndexingResult injected into the registry
# ---------------------------------------------------------------------------

@pytest.fixture
def active_single_phase_result(monkeypatch):
    """Wire a 4×3 single-phase synthetic result into the indexing registry.

    Twelve distinct Euler triples so any pixel-reordering bug in the
    writer surfaces as a mismatched euler_angles dataset on readback.
    """
    n_rows, n_cols = 4, 3
    n_pix = n_rows * n_cols

    eulers = np.deg2rad(np.array(
        [[i * 11.0, 20.0 + i, 30.0 + 2 * i] for i in range(n_pix)],
        dtype=np.float64,
    ))
    rotations = Rotation.from_euler(eulers)

    al = Phase(name="Al", space_group=225, point_group="m-3m")
    phase_list = PhaseList(phases=[al])

    xs = np.tile(np.arange(n_cols, dtype=np.float32), n_rows)
    ys = np.repeat(np.arange(n_rows, dtype=np.float32), n_cols)
    phase_id = np.zeros(n_pix, dtype=np.int32)  # all "Al" (orix id 0)

    xmap = CrystalMap(
        rotations=rotations,
        phase_id=phase_id,
        x=xs, y=ys,
        phase_list=phase_list,
    )

    result = IndexingResult(
        xmap=xmap,
        selection_mask=np.ones(n_pix, dtype=bool),
        original_shape=(n_rows, n_cols),
        method=IndexingMethod.SPHERICAL,
        confidence_scores=np.full(n_pix, 0.8, dtype=np.float32),
        # step_size_um carries the µm/pixel scale. Production resolves it from
        # the loaded EBSD signal's nav-axis scale; tests inject it here so the
        # writer has a deterministic source without a real signal.
        metadata={"step_size_um": 0.5},
    )

    rid = "test_h5_light"
    monkeypatch.setitem(indexing_mod._result_registry, rid, result)
    monkeypatch.setattr(indexing_mod, "_active_result_id", rid)
    # Ensure the SourceReference attrs end up empty / harmless rather than
    # accidentally pointing at a stale path from another test.
    monkeypatch.setattr(
        "backend.api.routes.ebsd_viewer._ebsd_file_path", None, raising=False,
    )
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _file_signature(p: Path) -> bytes:
    return p.read_bytes()[:16]


def test_format_h5_light_writes_superblock_v0(
    active_single_phase_result, tmp_path
):
    """The on-disk file must use HDF5 superblock v0.

    This is the actual MATLAB-compatibility property: old HDF5 1.8
    libraries (MATLAB ≤ R2019b) refuse v2 superblocks. Without
    ``libver='earliest'`` h5py 3.x writes v2 by default on Windows.
    """
    out = tmp_path / "result_synth_light.h5"
    client = TestClient(app)
    resp = client.post(
        "/api/indexing/export",
        json={"format": "h5_light", "filename": str(out)},
    )
    assert resp.status_code == 200, resp.text
    assert out.is_file(), "endpoint claimed success but no file was written"

    sig = _file_signature(out)
    assert sig[:8] == b"\x89HDF\r\n\x1a\n", "not a valid HDF5 magic header"
    assert sig[8] == 0, (
        f"expected superblock version 0 for MATLAB compatibility, "
        f"got version {sig[8]}. The light export must keep libver='earliest'."
    )


def test_format_h5_light_skips_source_copy(
    active_single_phase_result, tmp_path
):
    """No /1/EBSD/Patterns — that group only appears if the rich branch
    copied the source h5oina, which the light branch must NOT do."""
    out = tmp_path / "result_synth_light.h5"
    client = TestClient(app)
    resp = client.post(
        "/api/indexing/export",
        json={"format": "h5_light", "filename": str(out)},
    )
    assert resp.status_code == 200, resp.text

    # Far under 1 MB for a 4×3 grid; real scans land at 5–50 MB. A
    # multi-GB file here would mean shutil.copy2 of the source crept in.
    assert out.stat().st_size < 1_000_000, (
        f"light file is suspiciously large ({out.stat().st_size} bytes) — "
        "did the source-copy slip into the light branch?"
    )

    with h5py.File(out, "r") as f:
        assert "1" not in f, (
            "/1/ group is present — the light branch must NOT copy the "
            "h5oina source layout"
        )
        assert "Indexing" in f
        assert "SourceReference" in f
        assert "Documentation" in f


def test_format_h5_light_writes_1_based_phase_id(
    active_single_phase_result, tmp_path
):
    """On disk phase_id must be 1-based with 0=unindexed.

    The orix-native phase_id is 0-based (0 = first real phase).
    If the writer skipped the +1 shift, Phase Map would interpret
    every Al pixel as "unindexed" on reload — that's a real bug
    we hit in the rich branch and want to keep fixed in the light branch.
    """
    out = tmp_path / "result_synth_light.h5"
    client = TestClient(app)
    resp = client.post(
        "/api/indexing/export",
        json={"format": "h5_light", "filename": str(out)},
    )
    assert resp.status_code == 200, resp.text

    with h5py.File(out, "r") as f:
        # Single-phase path → flat /Indexing/phase_id
        pid = np.array(f["Indexing/phase_id"])
        unique = sorted(np.unique(pid).tolist())
        assert unique == [1], (
            f"expected all pixels at phase_id=1 (1-based Al), got {unique}. "
            "If you see 0 here the +1 shift was lost — phase-0 pixels will "
            "show as unindexed in Phase Map."
        )

        # Phase table for the Analysis loader
        assert "Indexing/Phases/1" in f
        assert f["Indexing/Phases/1"].attrs["name"] == "Al"


def test_format_h5_light_writes_step_size_and_coords(
    active_single_phase_result, tmp_path
):
    """The light file must carry step size + per-pixel X/Y µm coordinates.

    MTEX (and any generic HDF5 reader) cannot place orientations on a grid
    without these. The interactive Save as… → Light .h5 path used to write
    only grid_shape — no step_size_um, no coordinates — forcing the user to
    pull step size from the original h5oina by hand (Irmi's 2026-06-02
    feedback). Convention matches the .ang writer: X varies with column,
    Y with row, both = index · step_size_um.
    """
    out = tmp_path / "result_synth_light.h5"
    client = TestClient(app)
    resp = client.post(
        "/api/indexing/export",
        json={"format": "h5_light", "filename": str(out)},
    )
    assert resp.status_code == 200, resp.text

    n_rows, n_cols = 4, 3
    step = 0.5
    with h5py.File(out, "r") as f:
        assert f["Indexing"].attrs["step_size_um"] == pytest.approx(step), (
            "step_size_um attr missing/wrong on /Indexing — MTEX can't scale "
            "the map without it"
        )

        X = np.array(f["Indexing/X"])
        Y = np.array(f["Indexing/Y"])
        assert X.shape == (n_rows, n_cols), f"X shape {X.shape} != grid"
        assert Y.shape == (n_rows, n_cols), f"Y shape {Y.shape} != grid"

        cc, rr = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
        np.testing.assert_allclose(X, cc * step, err_msg="X must be col·step")
        np.testing.assert_allclose(Y, rr * step, err_msg="Y must be row·step")


def test_format_h5_light_fails_loud_without_step_size(monkeypatch, tmp_path):
    """When step size cannot be resolved, the export must 400 with an
    actionable message — never silently write a file with no/zero step
    (that's exactly the bug Irmi reported). Per the fail-loud rule we refuse
    rather than guess a default."""
    n_rows, n_cols = 3, 2
    n_pix = n_rows * n_cols
    rotations = Rotation.from_euler(np.zeros((n_pix, 3)))
    al = Phase(name="Al", space_group=225, point_group="m-3m")
    xmap = CrystalMap(
        rotations=rotations,
        phase_id=np.zeros(n_pix, dtype=np.int32),
        x=np.tile(np.arange(n_cols, dtype=np.float32), n_rows),
        y=np.repeat(np.arange(n_rows, dtype=np.float32), n_cols),
        phase_list=PhaseList(phases=[al]),
    )
    result = IndexingResult(
        xmap=xmap,
        selection_mask=np.ones(n_pix, dtype=bool),
        original_shape=(n_rows, n_cols),
        method=IndexingMethod.SPHERICAL,
        confidence_scores=np.full(n_pix, 0.8, dtype=np.float32),
        metadata={},  # no step size
    )
    rid = "test_h5_light_nostep"
    monkeypatch.setitem(indexing_mod._result_registry, rid, result)
    monkeypatch.setattr(indexing_mod, "_active_result_id", rid)
    monkeypatch.setattr(
        "backend.api.routes.ebsd_viewer._ebsd_file_path", None, raising=False,
    )
    # No active signal → no nav-axis scale to fall back on.
    monkeypatch.setattr(
        "backend.api.routes.ebsd_viewer._get_active_signal",
        lambda: None, raising=False,
    )

    out = tmp_path / "nostep_light.h5"
    client = TestClient(app)
    resp = client.post(
        "/api/indexing/export",
        json={"format": "h5_light", "filename": str(out)},
    )
    assert resp.status_code == 400, resp.text
    assert "step size" in resp.json()["detail"].lower()


@pytest.fixture
def active_oxford_result(monkeypatch):
    """Single-phase Oxford-sourced result with 12 distinct orientations."""
    n_rows, n_cols = 4, 3
    n = n_rows * n_cols
    eulers = np.deg2rad(np.array(
        [[i * 11.0, 20.0 + i, 30.0 + 2 * i] for i in range(n)], dtype=np.float64))
    xmap = CrystalMap(
        rotations=Rotation.from_euler(eulers),
        phase_id=np.zeros(n, dtype=np.int32),
        x=np.tile(np.arange(n_cols, dtype=np.float32), n_rows),
        y=np.repeat(np.arange(n_rows, dtype=np.float32), n_cols),
        phase_list=PhaseList(phases=[Phase(name="Al", space_group=225, point_group="m-3m")]),
    )
    result = IndexingResult(
        xmap=xmap, selection_mask=np.ones(n, dtype=bool),
        original_shape=(n_rows, n_cols), method=IndexingMethod.SPHERICAL,
        confidence_scores=np.full(n, 0.8, dtype=np.float32),
        metadata={"step_size_um": 0.5, "source_vendor": "oxford"},
    )
    rid = "test_oxford"
    monkeypatch.setitem(indexing_mod._result_registry, rid, result)
    monkeypatch.setattr(indexing_mod, "_active_result_id", rid)
    monkeypatch.setattr(
        "backend.api.routes.ebsd_viewer._ebsd_file_path", None, raising=False)
    return result


def _max_diso_deg(a, b):
    return float(np.rad2deg(np.max((a * ~b).angle)))


def test_light_export_rotates_euler_to_oxford_frame(active_oxford_result, tmp_path):
    """For Oxford-sourced results the stored Euler must be the Aztec/MTEX frame
    = our native rotations right-multiplied by Rz(+90deg). The file must tag
    source_vendor so the reader can invert it."""
    from orix.vector import Vector3d
    out = tmp_path / "ox_light.h5"
    resp = TestClient(app).post(
        "/api/indexing/export", json={"format": "h5_light", "filename": str(out)})
    assert resp.status_code == 200, resp.text

    orig = active_oxford_result.xmap.rotations
    expected = orig * Rotation.from_axes_angles(Vector3d.zvector(), np.deg2rad(90.0))
    with h5py.File(out, "r") as f:
        assert f["Indexing"].attrs["source_vendor"] == "oxford"
        eul = np.array(f["Indexing/euler_angles"]).reshape(-1, 3)
    got = Rotation.from_euler(eul)
    assert _max_diso_deg(got, expected) < 0.1, (
        "stored Euler is not in the Oxford (Aztec) frame")


def test_light_export_roundtrip_restores_native(active_oxford_result, tmp_path):
    """Re-importing our own Oxford export must return native orientations
    (renderer-safe), i.e. the reader inverts the export-frame rotation."""
    from backend.api.routes.analysis import _load_kikuchipy_rich_h5
    out = tmp_path / "ox_light.h5"
    TestClient(app).post(
        "/api/indexing/export", json={"format": "h5_light", "filename": str(out)})
    xmap = _load_kikuchipy_rich_h5(str(out))
    orig = active_oxford_result.xmap.rotations
    assert _max_diso_deg(xmap.rotations, orig) < 0.1, (
        "re-imported orientations did not return to the native frame")


def test_format_h5_light_rejects_bogus_format(active_single_phase_result):
    """Unknown format values still surface a clear 400."""
    client = TestClient(app)
    resp = client.post(
        "/api/indexing/export",
        json={"format": "parquet"},
    )
    assert resp.status_code == 400
    assert "Unsupported format" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Reader-side: the loader must accept the single-phase FLAT layout that the
# rich H5 writer produces. Writer and reader used to disagree — rich-h5
# single-phase puts datasets directly under /Indexing/, but the reader hard-
# required /Indexing/Assignment/ and threw "Assignment is missing". This
# locked users out of their own 27 GB rich exports.
# ---------------------------------------------------------------------------

def test_rich_h5_loader_accepts_flat_single_phase_layout(tmp_path):
    """A file with /Indexing/{phase_id,euler_angles,confidence_index} flat
    (no /Indexing/Assignment) must load — that's the layout the rich
    H5 single-phase branch in indexing.py writes."""
    import h5py
    from backend.api.routes.analysis import _load_kikuchipy_rich_h5

    n_rows, n_cols = 3, 4
    n_pix = n_rows * n_cols
    fpath = tmp_path / "flat_single_phase.h5"
    with h5py.File(fpath, "w", libver="earliest") as f:
        idx = f.create_group("Indexing")
        idx.attrs["step_size_um"] = 0.5
        # 1-based phase_id (1 = the one real phase, 0 would be unindexed)
        idx.create_dataset(
            "phase_id",
            data=np.ones((n_rows, n_cols), dtype=np.uint8),
        )
        idx.create_dataset(
            "euler_angles",
            data=np.zeros((n_rows, n_cols, 3), dtype=np.float32),
        )
        idx.create_dataset(
            "confidence_index",
            data=np.full((n_rows, n_cols), 0.7, dtype=np.float32),
        )
        phases = idx.create_group("Phases")
        pg = phases.create_group("1")
        pg.attrs["name"] = "Al"
        pg.attrs["space_group"] = 225
        pg.attrs["point_group"] = "m-3m"

    xmap = _load_kikuchipy_rich_h5(str(fpath))
    assert xmap.size == n_pix
    # 1-based on disk → 0-based on orix, all real Al pixels at id 0.
    assert int(xmap.phase_id.min()) == 0
    assert int(xmap.phase_id.max()) == 0
    # Phase name + symmetry round-tripped from /Indexing/Phases/1
    phase_names = [p.name for _, p in xmap.phases]
    assert "Al" in phase_names


def test_import_h5_route_registers_result_and_reports_capabilities(tmp_path):
    """End-to-end: roundtrip a light export, then re-import it via the new
    /api/indexing/import-h5 route. The route must return a real result_id,
    register the result in the indexing registry, and report which
    capabilities the file carries — minimum: indexing=True for a light
    file, with patterns/eds/electron_image=False because the light writer
    deliberately skips those."""
    import h5py
    from fastapi.testclient import TestClient
    from backend.api.main import app
    from backend.api.routes import indexing as imod

    # 1. Build a small light file by hand. The light writer's own output
    # would also work, but writing it directly keeps this test independent
    # of the export route's wiring (which other tests already cover).
    n_rows, n_cols = 3, 4
    fpath = tmp_path / "synth_light.h5"
    with h5py.File(fpath, "w", libver="earliest") as f:
        idx = f.create_group("Indexing")
        idx.attrs["method"] = "spherical"
        idx.attrs["grid_shape"] = [n_rows, n_cols]
        idx.create_dataset("phase_id", data=np.ones((n_rows, n_cols), dtype=np.uint8))
        idx.create_dataset("euler_angles", data=np.zeros((n_rows, n_cols, 3), dtype=np.float32))
        idx.create_dataset("confidence_index", data=np.full((n_rows, n_cols), 0.65, dtype=np.float32))
        idx.create_dataset("selection_mask", data=np.ones((n_rows, n_cols), dtype=np.uint8))
        phases = idx.create_group("Phases")
        pg = phases.create_group("1")
        pg.attrs["name"] = "Al"
        pg.attrs["space_group"] = 225
        pg.attrs["point_group"] = "m-3m"

    # 2. Import it
    client = TestClient(app)
    resp = client.post("/api/indexing/import-h5", json={"path": str(fpath)})
    assert resp.status_code == 200, resp.text
    d = resp.json()

    # 3. Real result_id (not None) → Save as… will work
    assert d["result_id"] and d["result_id"] in imod._result_registry, (
        f"import-h5 returned result_id={d['result_id']} but it's not in "
        f"the registry — Save as… would still fail."
    )
    assert imod._active_result_id == d["result_id"], (
        "imported result must be the active one so Phase Map renders it"
    )

    # 4. Capability dict reflects what's actually in the file
    caps = d["capabilities"]
    assert caps["indexing"] is True
    assert caps["patterns"] is False, "light files must not claim to have patterns"
    assert caps["eds"] is False
    assert caps["electron_image"] is False

    # 5. Summary fields populated correctly
    assert d["shape"] == [n_rows, n_cols]
    assert d["phases"] == ["Al"]
    assert abs(d["ci_mean"] - 0.65) < 1e-5
    assert d["n_pixels"] == n_rows * n_cols
    assert d["ebsd_signal_loaded"] is False, (
        "no /1/EBSD in this synthetic light file — wiring should not have been attempted"
    )


def test_import_h5_route_rejects_non_gui_file(tmp_path):
    """A bare .h5 with no /Indexing must surface a clear 400."""
    import h5py
    from fastapi.testclient import TestClient
    from backend.api.main import app

    fpath = tmp_path / "random.h5"
    with h5py.File(fpath, "w") as f:
        f.create_dataset("unrelated", data=np.zeros(10))

    client = TestClient(app)
    resp = client.post("/api/indexing/import-h5", json={"path": str(fpath)})
    assert resp.status_code == 400
    assert "/Indexing" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Persisted render geometry must survive import (bug 2026-08-17)
# ---------------------------------------------------------------------------
# The export persists detector_geometry + sht_paths_by_phase — the values the
# run ACTUALLY used. The import restored them, then unconditionally overwrote
# the geometry with a reconstruction from the freshly loaded signal, whose
# detector loses the run's calibration (real export: tilt 0.0 instead of
# 4.31°, px_size 55 instead of 70 µm). The SHT forward render then drew the
# simulated pattern ~8 px off and the Pattern Match dialog reported
# R = -0.09 ("Poor match") on a correctly indexed pixel.
#
# The persisted sht map is keyed by the LIVE run's 1-based phase ids while
# the imported xmap is re-keyed 0-based — so it must come back as per-phase
# hints (shifted -1), never verbatim.

_PERSISTED_GEOMETRY = {
    "pc_x": 0.5026, "pc_y": 0.3265, "pc_z": 0.8461,
    "pat_width": 156, "pat_height": 128,
    "n_cols": 4, "n_rows": 3,
    "pixel_size": 70.0, "tilt": 4.3107, "binning": 1,
    "step_x": 0.15, "step_y": 0.15,
    "vendor": "Bruker", "sample_tilt": 69.9916, "source_vendor": "oxford",
}


class _FakeNavAxis:
    scale = 0.5


class _FakeAxesManager:
    signal_shape = (156, 128)
    navigation_shape = (4, 3)
    navigation_axes = [_FakeNavAxis(), _FakeNavAxis()]


class _FakeDetector:
    """What a reloaded export's signal reports: calibration lost."""
    pc = np.array([[0.5026, 0.3265, 0.8461]])
    shape = (128, 156)
    px_size = 55.0
    tilt = 0.0
    sample_tilt = 70.0
    binning = 1


class _FakeSignal:
    axes_manager = _FakeAxesManager()
    detector = _FakeDetector()


def _write_light_file_with_patterns(fpath, persist_geometry: bool,
                                    persisted_sht: str | None = None):
    import json
    n_rows, n_cols = 3, 4
    with h5py.File(fpath, "w", libver="earliest") as f:
        idx = f.create_group("Indexing")
        idx.attrs["method"] = "spherical"
        idx.attrs["grid_shape"] = [n_rows, n_cols]
        idx.create_dataset("phase_id", data=np.ones((n_rows, n_cols), dtype=np.uint8))
        idx.create_dataset("euler_angles", data=np.zeros((n_rows, n_cols, 3), dtype=np.float32))
        idx.create_dataset("confidence_index", data=np.full((n_rows, n_cols), 0.65, dtype=np.float32))
        idx.create_dataset("selection_mask", data=np.ones((n_rows, n_cols), dtype=np.uint8))
        phases = idx.create_group("Phases")
        pg = phases.create_group("1")
        pg.attrs["name"] = "Al"
        pg.attrs["space_group"] = 225
        pg.attrs["point_group"] = "m-3m"
        if persist_geometry:
            idx.attrs["detector_geometry"] = json.dumps(_PERSISTED_GEOMETRY)
            # Keyed "1" like a real export: the live spherical xmap's 1-based id.
            idx.attrs["sht_paths_by_phase"] = json.dumps({"1": persisted_sht})
        # Presence of a pattern stack flips capabilities["patterns"], which is
        # what arms the signal-based geometry reconstruction on import.
        f.create_dataset(
            "1/EBSD/Data/Processed Patterns",
            data=np.zeros((n_rows * n_cols, 8, 8), dtype=np.uint8),
        )


def _patch_signal_side_of_import(monkeypatch, fpath):
    """Make the import believe the file's signal loaded, with a detector
    that has LOST the run's calibration (tilt 0, px 55)."""
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.services import h5_session as h5s
    monkeypatch.setattr(ev, "load_ebsd_file", lambda p: True)
    monkeypatch.setattr(ev, "_get_active_signal", lambda: _FakeSignal())
    # Keep the h5_session wiring inert — this test is about metadata.
    monkeypatch.setattr(h5s, "is_open", lambda: True)
    monkeypatch.setattr(h5s, "get_current_path", lambda: str(fpath))


def test_import_h5_keeps_persisted_render_geometry(tmp_path, monkeypatch):
    """The persisted detector_geometry must win over the reconstruction from
    the reloaded signal, and the persisted (1-based) sht map must come back
    re-keyed to the imported xmap's 0-based ids via the hint path."""
    from pathlib import Path
    from backend.api.routes import indexing as imod

    # The persisted SHT must EXIST for the exact-path hint to be honoured
    # (missing files fall through to name matching by design).
    persisted_sht = tmp_path / "Al (run used this).sht"
    persisted_sht.write_bytes(b"not a real sht - existence is all that counts")

    fpath = tmp_path / "with_geometry.h5"
    _write_light_file_with_patterns(
        fpath, persist_geometry=True, persisted_sht=str(persisted_sht))
    _patch_signal_side_of_import(monkeypatch, fpath)

    client = TestClient(app)
    resp = client.post("/api/indexing/import-h5", json={"path": str(fpath)})
    assert resp.status_code == 200, resp.text
    md = imod._result_registry[resp.json()["result_id"]].metadata

    dg = md["detector_geometry"]
    assert dg["tilt"] == pytest.approx(4.3107), (
        "reconstruction (tilt=0) overwrote the persisted detector tilt — "
        "this is exactly the 8-px render shift / R=-0.09 bug"
    )
    assert dg["pixel_size"] == pytest.approx(70.0)

    sht = md["sht_paths_by_phase"]
    # Written under key "1" (live 1-based id) → must come back under 0
    # (imported orix id). A verbatim key 1 would resolve every pixel to the
    # WRONG phase's SHT on multi-phase results.
    assert sht.get(0) == str(persisted_sht.resolve())
    assert 1 not in sht


def test_import_h5_reconstructs_geometry_when_not_persisted(tmp_path, monkeypatch):
    """Older exports without the persisted attrs must still get the
    signal-based fallback (the pre-fix behaviour)."""
    from backend.api.routes import indexing as imod

    fpath = tmp_path / "no_geometry.h5"
    _write_light_file_with_patterns(fpath, persist_geometry=False)
    _patch_signal_side_of_import(monkeypatch, fpath)
    monkeypatch.setattr(
        imod, "_match_library_shts_for_xmap",
        lambda xmap, hints=None: {0: "Y:/library/matched.sht"},
    )

    client = TestClient(app)
    resp = client.post("/api/indexing/import-h5", json={"path": str(fpath)})
    assert resp.status_code == 200, resp.text
    md = imod._result_registry[resp.json()["result_id"]].metadata

    dg = md.get("detector_geometry")
    assert dg is not None, "fallback reconstruction must still run"
    assert dg["tilt"] == pytest.approx(0.0)
    assert md.get("sht_paths_by_phase", {}).get(0) == "Y:/library/matched.sht"


def test_rich_h5_loader_still_rejects_empty_indexing(tmp_path):
    """When /Indexing has neither Assignment nor flat datasets, the loader
    must still fail loud rather than guess. We changed the failure path,
    not the failure mode."""
    import h5py
    from backend.api.routes.analysis import _load_kikuchipy_rich_h5

    fpath = tmp_path / "empty_indexing.h5"
    with h5py.File(fpath, "w") as f:
        # /Indexing exists but holds only an unrelated Phases group — no
        # phase_id / euler_angles anywhere.
        idx = f.create_group("Indexing")
        idx.create_group("Phases")

    with pytest.raises(ValueError, match="phase_id"):
        _load_kikuchipy_rich_h5(str(fpath))
