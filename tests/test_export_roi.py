"""ROI-export regression.

The /api/indexing/export rich + light branches reshaped per-row result
arrays blindly onto the full grid — every ROI export died with
"cannot reshape array of size N into shape (rows, cols, 3)". And the blind
``raw_pid + 1`` shift assumed 0-based xmap ids, silently shifting SPHERICAL
results (explicit 1-based PhaseList ids) onto the WRONG phase names on
re-import.

These tests run the real route against a fake ROI spherical result and
verify the ROUNDTRIP through the app's own reader
(`analysis._load_kikuchipy_rich_h5`): placement via selection mask, phase
NAMES per pixel (not just ids), orientation equality, and full-grid
behaviour unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

NR, NC = 6, 8


def _fake_roi_result():
    """Spherical-style ROI result: non-rectangular mask (26 px of 48),
    1-based PhaseList ids [1, 2] — exactly the convention the spherical
    pipeline produces."""
    from types import SimpleNamespace
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation

    mask = np.zeros((NR, NC), bool)
    mask[1:5, 1:7] = True          # 4x6 block
    mask[0, 0] = True              # plus two stragglers → non-rectangular
    mask[5, 7] = True
    n_roi = int(mask.sum())

    rng = np.random.default_rng(42)
    eulers = np.column_stack([
        rng.uniform(0, 2 * np.pi, n_roi),
        rng.uniform(0.1, np.pi - 0.1, n_roi),
        rng.uniform(0, 2 * np.pi, n_roi),
    ])
    pid = np.where(np.arange(n_roi) % 2 == 0, 1, 2)   # 1-based ids!
    yy, xx = np.nonzero(mask)
    xmap = CrystalMap(
        rotations=Rotation.from_euler(eulers),
        phase_id=pid,
        x=xx.astype(float), y=yy.astype(float),
        phase_list=PhaseList(names=["Al", "alpha"],
                             point_groups=["m-3m", "m-3"], ids=[1, 2]),
    )
    ci = rng.uniform(0.1, 0.5, n_roi).astype(np.float32)
    result = SimpleNamespace(
        xmap=xmap,
        original_shape=(NR, NC),
        selection_mask=mask,
        confidence_scores=ci,
        method=SimpleNamespace(value="spherical"),
        metadata={},
    )
    return result, mask, eulers, pid, ci


def _patch_route_seams(monkeypatch, result):
    import backend.api.routes.indexing as ix
    import backend.api.routes.ebsd_viewer as ev
    import backend.api.services.reference_frame_state as rfs
    monkeypatch.setattr(ix, "_get_result", lambda *_a, **_k: result)
    monkeypatch.setattr(ix, "_resolve_source_vendor", lambda *_a, **_k: "unknown")
    monkeypatch.setattr(ix, "_resolve_step_size_um", lambda *_a, **_k: 0.25)
    monkeypatch.setattr(ev, "_ebsd_file_path", None, raising=False)
    # No pattern source: whatever signal an earlier test left in the viewer
    # would make the export carry patterns, and the import would then open the
    # file through h5_session and hold it open.
    monkeypatch.setattr(ev, "_ebsd_signal", None, raising=False)
    monkeypatch.setattr(ev, "_active_dataset", "", raising=False)
    monkeypatch.setattr(ev, "_raw_signals", {}, raising=False)
    monkeypatch.setattr(rfs, "get_frame", lambda *_a, **_k: {"apply_to_export": False})
    return ix


def _release_h5_session():
    """Let go of every file the app's h5 session holds, so the test may reopen
    one of them for writing. ``close_file`` only parks the handle, so the
    parked ones are closed here too."""
    from backend.api.services import h5_session
    h5_session.close_file()
    with h5_session._lock:
        parked = list(h5_session._pending_close)
        h5_session._pending_close.clear()
    for handle in parked:
        try:
            handle.close()
        except Exception:
            pass


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def _roundtrip_checks(path, mask, eulers, pid, ci):
    from backend.api.routes.analysis import _load_kikuchipy_rich_h5
    from backend.spherical_gpu.pseudosym import same_orientation_angle_deg
    from orix.quaternion import Rotation

    xmap = _load_kikuchipy_rich_h5(str(path))
    got_pid = np.asarray(xmap.phase_id).reshape(NR, NC)
    got_q = np.asarray(xmap.rotations.data).reshape(NR, NC, 4)
    got_ci = np.asarray(xmap.prop["ci"]).reshape(NR, NC)

    # outside the ROI: unindexed
    assert (got_pid[~mask] == -1).all()
    # inside: PHASE NAMES survive (the +1-shift bug swapped them)
    names = {int(p): ph.name for p, ph in xmap.phases if int(p) >= 0}
    yy, xx = np.nonzero(mask)
    for i, (r, c) in enumerate(zip(yy, xx)):
        expected = "Al" if pid[i] == 1 else "alpha"
        assert names[int(got_pid[r, c])] == expected, (r, c)
        # orientation roundtrip (float32 storage → ~1e-4 deg tolerance)
        qa = np.asarray(Rotation.from_euler(eulers[i][None, :]).data).reshape(4)
        ang = float(same_orientation_angle_deg(
            got_q[r, c][None, :], qa, "1")[0])
        assert ang < 0.01, (r, c, ang)
        assert abs(float(got_ci[r, c]) - float(ci[i])) < 1e-5


def test_rich_h5_export_roundtrips_roi_result(monkeypatch, tmp_path):
    result, mask, eulers, pid, ci = _fake_roi_result()
    ix = _patch_route_seams(monkeypatch, result)
    out = tmp_path / "roi_rich.h5"
    _run(ix.export_indexing_result(ix.ExportRequest(
        format="h5", filename=str(out), include_eds=False,
        include_detector=False)))
    assert out.is_file()
    _roundtrip_checks(out, mask, eulers, pid, ci)


def test_light_h5_export_roundtrips_roi_result(monkeypatch, tmp_path):
    result, mask, eulers, pid, ci = _fake_roi_result()
    ix = _patch_route_seams(monkeypatch, result)
    out = tmp_path / "roi_light.h5"
    _run(ix.export_indexing_result(ix.ExportRequest(
        format="h5_light", filename=str(out), include_eds=False,
        include_detector=False)))
    assert out.is_file()
    _roundtrip_checks(out, mask, eulers, pid, ci)


def test_rich_h5_multiphase_assignment_branch_roi(monkeypatch, tmp_path):
    """per_phase_data present → the Assignment sub-group branch must place
    ROI rows too (it had the identical blind reshape)."""
    result, mask, eulers, pid, ci = _fake_roi_result()
    result.metadata["per_phase_data"] = {
        "Al":    {"euler": np.zeros((NR, NC, 3), np.float32),
                  "ci": np.zeros((NR, NC), np.float32), "ci_mean": 0.3},
        "alpha": {"euler": np.zeros((NR, NC, 3), np.float32),
                  "ci": np.zeros((NR, NC), np.float32), "ci_mean": 0.2},
    }
    ix = _patch_route_seams(monkeypatch, result)
    out = tmp_path / "roi_rich_mp.h5"
    _run(ix.export_indexing_result(ix.ExportRequest(
        format="h5", filename=str(out), include_eds=False,
        include_detector=False)))
    _roundtrip_checks(out, mask, eulers, pid, ci)


def test_full_grid_export_unchanged(monkeypatch, tmp_path):
    """Full-coverage results take the plain-reshape path — behaviour must be
    identical to before the fix (incl. correct names for 1-based ids)."""
    from types import SimpleNamespace
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation
    rng = np.random.default_rng(1)
    n = NR * NC
    eulers = np.column_stack([
        rng.uniform(0, 2 * np.pi, n),
        rng.uniform(0.1, np.pi - 0.1, n),
        rng.uniform(0, 2 * np.pi, n),
    ])
    pid = np.where(np.arange(n) % 3 == 0, 1, 2)
    yy, xx = np.mgrid[0:NR, 0:NC]
    xmap = CrystalMap(
        rotations=Rotation.from_euler(eulers), phase_id=pid,
        x=xx.ravel().astype(float), y=yy.ravel().astype(float),
        phase_list=PhaseList(names=["Al", "alpha"],
                             point_groups=["m-3m", "m-3"], ids=[1, 2]),
    )
    ci = rng.uniform(0.1, 0.5, n).astype(np.float32)
    result = SimpleNamespace(
        xmap=xmap, original_shape=(NR, NC),
        selection_mask=np.ones((NR, NC), bool),
        confidence_scores=ci,
        method=SimpleNamespace(value="spherical"), metadata={},
    )
    ix = _patch_route_seams(monkeypatch, result)
    out = tmp_path / "full_rich.h5"
    _run(ix.export_indexing_result(ix.ExportRequest(
        format="h5", filename=str(out), include_eds=False,
        include_detector=False)))
    mask = np.ones((NR, NC), bool)
    _roundtrip_checks(out, mask, eulers, pid, ci)


# ---------------------------------------------------------------------------
# Import-side session restoration: EDAX pattern detection +
# SHT library matching for re-imported rich/light files.
# ---------------------------------------------------------------------------

def test_h5_any_pattern_dataset_detects_edax_layout(tmp_path):
    import h5py
    from backend.api.routes.indexing import _h5_any_pattern_dataset
    p = tmp_path / "edax_like.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("Scan1/EBSD/Data/Pattern",
                         data=np.zeros((4, 8, 8), np.uint8))
    with h5py.File(p, "r") as f:
        assert _h5_any_pattern_dataset(f) is True
    p2 = tmp_path / "no_patterns.h5"
    with h5py.File(p2, "w") as f:
        f.create_dataset("Indexing/phase_id", data=np.zeros((4, 4), np.uint8))
    with h5py.File(p2, "r") as f:
        assert _h5_any_pattern_dataset(f) is False


def test_match_library_shts_by_phase_name(monkeypatch):
    import backend.api.routes.indexing as ix
    import indexing_controller as ic
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation

    fake_files = {"files": [
        {"path": r"X:\lib\Al (Al) [cF4] {20kV}.sht"},
        {"path": r"X:\lib\Al7FeCu2 (Al7FeCu2) [tP40] {20kV}.sht"},
        {"path": r"X:\lib\SomethingElse (X) [cP1] {20kV}.sht"},
        {"path": r"X:\lib\NotAnSht.h5"},
    ]}
    monkeypatch.setattr(ic, "discover_files_for_method",
                        lambda *_a, **_k: fake_files)

    n = 4
    xmap = CrystalMap(
        rotations=Rotation.identity((n,)),
        phase_id=np.array([1, 1, 2, 2]),
        x=np.arange(n, dtype=float), y=np.zeros(n),
        phase_list=PhaseList(names=["Al", "Al7FeCu2"],
                             point_groups=["m-3m", "4/mmm"], ids=[1, 2]),
    )
    out = ix._match_library_shts_for_xmap(xmap)
    assert set(out.keys()) == {1, 2}
    assert out[1].endswith("Al (Al) [cF4] {20kV}.sht")
    assert out[2].endswith("Al7FeCu2 (Al7FeCu2) [tP40] {20kV}.sht")


# ---------------------------------------------------------------------------
# SHT matching, layer 2 (only Al had simulations after
# re-import): pearson-less filenames, formula-ratio fallback, stored hints.
# ---------------------------------------------------------------------------

def test_formula_ratio_key_matches_renormalised_formulas():
    from backend.api.routes.indexing import _formula_ratio_key
    # The alpha phase: CIF name vs SHT stem = same stoichiometry ×5.31.
    k_cif = _formula_ratio_key("(Al19 Fe4 Mn Si2)5.31")
    k_sht = _formula_ratio_key("Al100.89Fe21.24Mn5.31Si10.62")
    assert k_cif is not None and k_cif == k_sht
    # Plain integer formulas work and implicit count = 1.
    assert _formula_ratio_key("Al7FeCu2") == (
        ("Al", 7.0), ("Cu", 2.0), ("Fe", 1.0))
    # Free-text names must NEVER be treated as formulas.
    assert _formula_ratio_key("alpha-AlFeMnSi_ICSD-52623") is None
    assert _formula_ratio_key("T-phase_Mg32(AlZn)49_Bergman") is None
    assert _formula_ratio_key("") is None
    # Different stoichiometry ≠ match.
    assert _formula_ratio_key("Al2Cu") != _formula_ratio_key("AlCu")


def _mini_xmap(names, point_groups, ids):
    from orix.crystal_map import CrystalMap, PhaseList
    from orix.quaternion import Rotation
    n = 2 * len(ids)
    pid = np.repeat(np.asarray(ids), 2)
    return CrystalMap(
        rotations=Rotation.identity((n,)),
        phase_id=pid,
        x=np.arange(n, dtype=float), y=np.zeros(n),
        phase_list=PhaseList(names=names, point_groups=point_groups, ids=ids),
    )


def test_match_library_shts_pearsonless_and_formula(monkeypatch):
    """The real-world trio: Al (name), Al7FeCu2 (pearson-less filename),
    alpha '(Al19 Fe4 Mn Si2)5.31' (formula-ratio) must ALL resolve."""
    import backend.api.routes.indexing as ix
    import indexing_controller as ic
    fake_files = {"files": [
        {"path": r"X:\lib\Al (Al) [cF4] {20kV}.sht"},
        {"path": r"X:\lib\Al7FeCu2 (Al7FeCu2) {20kV}.sht"},
        {"path": r"X:\lib\Al100.89Fe21.24Mn5.31Si10.62 (alpha-AlFeMnSi_ICSD-52623) [cP138] {20kV}.sht"},
    ]}
    monkeypatch.setattr(ic, "discover_files_for_method",
                        lambda *_a, **_k: fake_files)
    xmap = _mini_xmap(["Al", "Al7FeCu2", "(Al19 Fe4 Mn Si2)5.31"],
                      ["m-3m", "4/mmm", "m-3"], [1, 2, 3])
    out = ix._match_library_shts_for_xmap(xmap)
    assert set(out.keys()) == {1, 2, 3}
    assert out[2].endswith("Al7FeCu2 (Al7FeCu2) {20kV}.sht")
    assert "alpha-AlFeMnSi" in out[3]


def test_match_library_shts_stored_hints_win(monkeypatch, tmp_path):
    """sht_path (existing file) beats everything; sht_file basename beats
    name heuristics; dead hints fall through to name matching."""
    import backend.api.routes.indexing as ix
    import indexing_controller as ic
    exact = tmp_path / "Exact (X) [cP1] {20kV}.sht"
    exact.write_bytes(b"x")
    fake_files = {"files": [
        {"path": r"X:\lib\Al (Al) [cF4] {20kV}.sht"},
        {"path": r"X:\lib\Renamed Phase (Y) [cP1] {20kV}.sht"},
    ]}
    monkeypatch.setattr(ic, "discover_files_for_method",
                        lambda *_a, **_k: fake_files)
    xmap = _mini_xmap(["Al", "SomethingUnmatchable"],
                      ["m-3m", "m-3"], [1, 2])
    # Phase 1: dead sht_path → falls through to name match.
    # Phase 2: no name match, but sht_file basename is in the library.
    hints = {
        1: {"sht_path": str(tmp_path / "gone.sht")},
        2: {"sht_file": "Renamed Phase (Y) [cP1] {20kV}.sht"},
    }
    out = ix._match_library_shts_for_xmap(xmap, hints)
    assert out[1].endswith("Al (Al) [cF4] {20kV}.sht")
    assert out[2].endswith("Renamed Phase (Y) [cP1] {20kV}.sht")
    # Existing sht_path wins outright, even over a name match.
    out2 = ix._match_library_shts_for_xmap(
        xmap, {1: {"sht_path": str(exact)}})
    assert out2[1] == str(exact.resolve())


def _import_capture(monkeypatch, ix):
    """Neutralise the import route's side effects and capture the result."""
    captured = {}
    monkeypatch.setattr(ix, "_match_library_shts_for_xmap",
                        lambda *_a, **_k: {})
    monkeypatch.setattr(
        ix, "_store_result",
        lambda res, *_a, **_k: captured.update(res=res) or "imported_1")
    return captured


def test_import_h5_restores_roi_result(monkeypatch, tmp_path):
    """ROI export → import must come back as a ROI result (xmap subset +
    2D mask), NOT as a full grid padded with zero-score pixels — that
    padding rendered as a huge red frame in the NCC map. Covers BOTH mask sources: stored dataset and the
    derived-from-phase_id fallback for older files."""
    import h5py
    result, mask, eulers, pid, ci = _fake_roi_result()
    ix = _patch_route_seams(monkeypatch, result)
    out = tmp_path / "roi_rich.h5"
    _run(ix.export_indexing_result(ix.ExportRequest(
        format="h5", filename=str(out), include_eds=False,
        include_detector=False)))

    captured = _import_capture(monkeypatch, ix)
    n_roi = int(mask.sum())
    for drop_stored_mask in (False, True):
        if drop_stored_mask:
            _release_h5_session()      # the first import may have opened it
            with h5py.File(out, "a") as f:
                del f["Indexing/selection_mask"]
        resp = _run(ix.import_h5_result(ix.ImportH5Request(path=str(out))))
        res = captured["res"]
        assert int(res.xmap.size) == n_roi, drop_stored_mask
        assert res.selection_mask.shape == (NR, NC)
        assert (np.asarray(res.selection_mask, bool) == mask).all()
        assert res.confidence_scores.size == n_roi
        assert resp["n_pixels"] == n_roi
        assert resp["ci_mean"] > 0.0
        # The NCC/score map must embed via the mask: NaN outside the ROI
        # (that's what makes the heatmap crop instead of painting red).
        from tools.pattern_comparison import get_ncc_score_map
        sm = get_ncc_score_map(res)
        assert sm.shape == (NR, NC)
        assert np.isnan(sm[~mask]).all()
        assert np.isfinite(sm[mask]).all()


def test_import_h5_full_grid_stays_full_grid(monkeypatch, tmp_path):
    """Full-coverage exports keep the full-grid shape on import."""
    from types import SimpleNamespace
    from orix.crystal_map import CrystalMap, PhaseList
    from orix.quaternion import Rotation
    rng = np.random.default_rng(3)
    n = NR * NC
    xmap = CrystalMap(
        rotations=Rotation.from_euler(np.column_stack([
            rng.uniform(0, 2 * np.pi, n),
            rng.uniform(0.1, np.pi - 0.1, n),
            rng.uniform(0, 2 * np.pi, n)])),
        phase_id=np.ones(n, dtype=int),
        x=np.tile(np.arange(NC, dtype=float), NR),
        y=np.repeat(np.arange(NR, dtype=float), NC),
        phase_list=PhaseList(names=["Al"], point_groups=["m-3m"], ids=[1]),
    )
    result = SimpleNamespace(
        xmap=xmap, original_shape=(NR, NC),
        selection_mask=np.ones((NR, NC), bool),
        confidence_scores=rng.uniform(0.1, 0.5, n).astype(np.float32),
        method=SimpleNamespace(value="spherical"), metadata={},
    )
    ix = _patch_route_seams(monkeypatch, result)
    out = tmp_path / "full_rich.h5"
    _run(ix.export_indexing_result(ix.ExportRequest(
        format="h5", filename=str(out), include_eds=False,
        include_detector=False)))
    captured = _import_capture(monkeypatch, ix)
    resp = _run(ix.import_h5_result(ix.ImportH5Request(path=str(out))))
    res = captured["res"]
    assert int(res.xmap.size) == n
    assert res.selection_mask.all()
    assert resp["n_pixels"] == n


def test_rich_export_writes_sht_provenance(monkeypatch, tmp_path):
    """Exports stamp sht_file/sht_path per phase so re-imports restore the
    exact simulation files instead of guessing by name."""
    import h5py
    result, mask, eulers, pid, ci = _fake_roi_result()
    result.metadata["sht_paths_by_phase"] = {
        1: r"X:\lib\Al (Al) [cF4] {20kV}.sht",
        # string key + missing entry for phase 2 must both be tolerated
    }
    ix = _patch_route_seams(monkeypatch, result)
    out = tmp_path / "prov.h5"
    _run(ix.export_indexing_result(ix.ExportRequest(
        format="h5", filename=str(out), include_eds=False,
        include_detector=False)))
    with h5py.File(out, "r") as f:
        ph = f["Indexing/Phases"]
        assert ph["1"].attrs["sht_file"] == "Al (Al) [cF4] {20kV}.sht"
        assert ph["1"].attrs["sht_path"].endswith("Al (Al) [cF4] {20kV}.sht")
        assert "sht_file" not in ph["2"].attrs
