"""Render-verified phase check + reassignment (Stage A/B).

Chemically degenerate phases (cubic m-3 approximant on an Al m-3m matrix)
can "steal" pixels of another phase because per-phase indexing scores are
not comparable across phases. The check scores the STORED phase at its
stored orientation vs every candidate phase at a Hough-anchored orientation
via render-NCC (injected here as fakes), grain by grain; the reassignment
flips only grains where a candidate wins by a clear margin — whole grain or
nothing, with undo.
"""
from __future__ import annotations
import numpy as np
import pytest
from unittest.mock import MagicMock

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# Shared scenario: 8×8 grid, identity quats.
#   phase 2 ("alpha", m-3):  cols 0-2 = grain W (WRONG — really phase 1)
#                            cols 5-7 = grain C (correct alpha)
#   phase 1 ("Al", m-3m):    cols 3-4 (healthy)
# Scores: alpha renders 0.15 on W / 0.45 on C; Al renders 0.32 on W.
# ---------------------------------------------------------------------------

NR = NC = 8
IDENT = np.array([1.0, 0.0, 0.0, 0.0])


def _grid():
    full_q = np.tile(IDENT, (NR * NC, 1)).astype(float)
    pf = np.zeros(NR * NC, dtype=np.int64)
    for r in range(NR):
        for c in range(NC):
            pf[r * NC + c] = 2 if (c < 3 or c >= 5) else 1
    return full_q, pf


def _score_fns():
    calls = {"alpha": [], "al": []}

    def alpha_fn(flats, quats):
        flats = np.asarray(flats).reshape(-1)
        calls["alpha"].append(flats)
        return np.where(flats % NC < 3, 0.15, 0.45)

    def al_fn(flats, quats):
        flats = np.asarray(flats).reshape(-1)
        calls["al"].append(flats)
        return np.where((flats % NC >= 3) & (flats % NC < 5), 0.50, 0.32)

    return {1: al_fn, 2: alpha_fn}, calls


def _hough_fn_factory(fail_flats=(), fail_all=False):
    calls = []

    def hough(cand_pid, flats):
        flats = np.asarray(flats).reshape(-1)
        calls.append((int(cand_pid), flats.copy()))
        out = np.tile(IDENT, (flats.size, 1)).astype(float)
        if fail_all:
            out[:] = np.nan
        else:
            for i, f in enumerate(flats):
                if int(f) in fail_flats:
                    out[i] = np.nan
        return out

    hough.calls = calls
    return hough


PHASES = {1: "m-3m", 2: "m-3"}


# ---------------------------------------------------------------------------
# Stage A — check_map
# ---------------------------------------------------------------------------

def test_check_map_flags_wrong_phase_grain():
    from backend.spherical_gpu.pipeline.phase_reassignment import check_map
    full_q, pf = _grid()
    fns, _calls = _score_fns()
    hough = _hough_fn_factory()
    report, margin = check_map(full_q, pf, NR, NC, PHASES, fns, hough)

    by_dec = {}
    for e in report["grains"]:
        by_dec.setdefault(e["decision"], []).append(e)
    assert report["n_grains"] == 3
    assert report["n_reassign"] == 1
    w = by_dec["reassign"][0]
    assert w["phase_id"] == 2
    assert w["best_alt_phase"] == 1
    assert w["stored_score"] == pytest.approx(0.15, abs=1e-6)
    assert w["best_alt_score"] == pytest.approx(0.32, abs=1e-6)
    assert w["margin"] == pytest.approx(-0.17, abs=1e-6)
    # correct alpha grain + Al grain pass the floor untouched
    assert {e["phase_id"] for e in by_dec["ok"]} == {1, 2}


def test_check_map_fast_path_never_calls_hough_for_healthy_grains():
    """Healthy grains (stored score ≥ floor) must not pay candidate scoring."""
    from backend.spherical_gpu.pipeline.phase_reassignment import check_map
    full_q, pf = _grid()
    fns, _ = _score_fns()
    hough = _hough_fn_factory()
    check_map(full_q, pf, NR, NC, PHASES, fns, hough)
    # only the wrong grain (cols 0-2) is a suspect → hough only saw its pixels
    assert hough.calls, "candidates must be scored for the suspect grain"
    for _pid, flats in hough.calls:
        assert (flats % NC < 3).all()


def test_check_map_margin_map_semantics():
    """Margin map: NaN for healthy/unindexed, stored−best for suspects."""
    from backend.spherical_gpu.pipeline.phase_reassignment import check_map
    full_q, pf = _grid()
    fns, _ = _score_fns()
    _report, margin = check_map(full_q, pf, NR, NC, PHASES, fns,
                                _hough_fn_factory())
    m2 = margin.reshape(NR, NC)
    assert np.allclose(m2[:, :3], -0.17, atol=1e-6)      # suspect grain
    assert np.isnan(m2[:, 3:]).all()                     # healthy grains


def test_check_map_ambiguous_margin_keeps_grain():
    """An alternative that wins by less than the hysteresis must NOT trigger
    a reassignment — 'keep', conservatively."""
    from backend.spherical_gpu.pipeline.phase_reassignment import check_map
    full_q, pf = _grid()

    def alpha_fn(flats, quats):
        flats = np.asarray(flats).reshape(-1)
        return np.where(flats % NC < 3, 0.20, 0.45)

    def al_fn(flats, quats):
        flats = np.asarray(flats).reshape(-1)
        # healthy on its own grain; +0.03 < 0.05 hysteresis as a candidate
        return np.where((flats % NC >= 3) & (flats % NC < 5), 0.50, 0.23)

    report, _ = check_map(full_q, pf, NR, NC, PHASES,
                          {1: al_fn, 2: alpha_fn}, _hough_fn_factory())
    assert report["n_reassign"] == 0
    keeps = [e for e in report["grains"] if e["decision"] == "keep"]
    assert len(keeps) == 1 and keeps[0]["margin"] == pytest.approx(-0.03, abs=1e-6)


def test_check_map_no_sht_phase_is_reported_not_scored():
    from backend.spherical_gpu.pipeline.phase_reassignment import check_map
    full_q, pf = _grid()
    fns, _ = _score_fns()
    del fns[2]                                            # alpha has no scorer
    report, margin = check_map(full_q, pf, NR, NC, PHASES, fns,
                               _hough_fn_factory())
    no_sht = [e for e in report["grains"] if e["decision"] == "no-sht"]
    assert {e["phase_id"] for e in no_sht} == {2}
    assert np.isnan(margin).all()


# ---------------------------------------------------------------------------
# Stage B — apply_reassignment
# ---------------------------------------------------------------------------

def _checked():
    from backend.spherical_gpu.pipeline.phase_reassignment import check_map
    full_q, pf = _grid()
    fns, _ = _score_fns()
    report, _m = check_map(full_q, pf, NR, NC, PHASES, fns, _hough_fn_factory())
    return full_q, pf, report


def test_apply_flips_whole_grain_and_only_that_grain():
    from backend.spherical_gpu.pipeline.phase_reassignment import apply_reassignment
    full_q, pf, report = _checked()
    new_pf, new_q, applied, skipped = apply_reassignment(
        full_q, pf, NR, NC, report, PHASES, _hough_fn_factory())
    assert len(applied) == 1 and not skipped
    pf2 = new_pf.reshape(NR, NC)
    assert (pf2[:, :3] == 1).all()                        # W reassigned to Al
    assert (pf2[:, 3:5] == 1).all() and (pf2[:, 5:] == 2).all()
    # untouched pixels bit-identical
    keep = pf.reshape(NR, NC)[:, 3:] == new_pf.reshape(NR, NC)[:, 3:]
    assert keep.all()
    assert np.array_equal(new_q.reshape(NR, NC, 4)[:, 3:],
                          full_q.reshape(NR, NC, 4)[:, 3:])


def test_apply_nearest_fill_for_hough_failed_pixels():
    from backend.spherical_gpu.pipeline.phase_reassignment import apply_reassignment
    full_q, pf, report = _checked()
    fail = {0, 1}                                         # two W pixels fail
    new_pf, new_q, applied, skipped = apply_reassignment(
        full_q, pf, NR, NC, report, PHASES,
        _hough_fn_factory(fail_flats=fail))
    assert len(applied) == 1
    assert applied[0]["n_hough_filled"] == 2
    assert np.isfinite(new_q[list(fail)]).all()           # filled, not NaN
    assert (new_pf.reshape(NR, NC)[:, :3] == 1).all()


def test_apply_total_hough_failure_leaves_grain_untouched():
    from backend.spherical_gpu.pipeline.phase_reassignment import apply_reassignment
    full_q, pf, report = _checked()
    new_pf, new_q, applied, skipped = apply_reassignment(
        full_q, pf, NR, NC, report, PHASES,
        _hough_fn_factory(fail_all=True))
    assert not applied and len(skipped) == 1
    assert np.array_equal(new_pf, pf)
    assert np.array_equal(new_q, full_q)


# ---------------------------------------------------------------------------
# Endpoints — /phase-check → /phase-reassign → /phase-reassign/undo
# ---------------------------------------------------------------------------

def _fake_two_phase_result():
    from types import SimpleNamespace
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation
    full_q, pf = _grid()
    x, y = np.meshgrid(np.arange(NC, dtype=float), np.arange(NR, dtype=float))
    xmap = CrystalMap(
        rotations=Rotation(full_q), phase_id=pf.copy(),
        x=x.ravel(), y=y.ravel(),
        phase_list=PhaseList(
            names=["Al", "alpha"], point_groups=["m-3m", "m-3"], ids=[1, 2]),
    )
    return SimpleNamespace(
        xmap=xmap,
        original_shape=(NR, NC),
        selection_mask=np.ones((NR, NC), bool),
        metadata={
            "indexing_method": "spherical",
            "sht_paths_by_phase": {1: "al.sht", 2: "alpha.sht"},
            "detector_geometry": {"pat_width": 8, "pat_height": 8,
                                  "pc_x": 0.5, "pc_y": 0.5, "pc_z": 0.5},
        },
    )



def _fake_result_with_island(row: int, col: int):
    """The shared two-phase map, plus ONE phase-1 pixel inside the phase-2 field.

    That single pixel is the whole point: it is smaller than MIN_GRAIN_PX, so
    the per-grain check never sees it.
    """
    from types import SimpleNamespace
    from orix.crystal_map import CrystalMap, PhaseList
    from orix.quaternion import Rotation
    full_q, pf = _grid()
    assert pf[row * NC + col] == 2, "pick a pixel that is phase 2 to begin with"
    pf[row * NC + col] = 1
    x, y = np.meshgrid(np.arange(NC, dtype=float), np.arange(NR, dtype=float))
    xmap = CrystalMap(
        rotations=Rotation(full_q), phase_id=pf,
        x=x.ravel(), y=y.ravel(),
        phase_list=PhaseList(
            names=["Al", "alpha"], point_groups=["m-3m", "m-3"], ids=[1, 2]),
    )
    return SimpleNamespace(
        xmap=xmap,
        original_shape=(NR, NC),
        selection_mask=np.ones((NR, NC), bool),
        metadata={
            "indexing_method": "spherical",
            "sht_paths_by_phase": {1: "al.sht", 2: "alpha.sht"},
            "detector_geometry": {"pat_width": 8, "pat_height": 8,
                                  "pc_x": 0.5, "pc_y": 0.5, "pc_z": 0.5},
        },
    )


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def _patch_endpoint_seams(monkeypatch, result):
    """Wire the fake scorers / Hough / CIF resolution into the routes."""
    import backend.api.routes.indexing as ix
    import backend.spherical_gpu.pipeline.variant_unification as vu
    import indexing_controller as ic
    import tools.pattern_comparison as pcmp

    monkeypatch.setattr(ix, "_get_result", lambda *_a, **_k: result)
    monkeypatch.setattr(ix, "_free_interactive_gpu_caches", lambda: None)
    monkeypatch.setattr(ic, "_resolve_cif_for_sht", lambda s: "dummy.cif")
    monkeypatch.setattr(pcmp, "get_experimental_pattern",
                        lambda *_a, **_k: np.zeros((8, 8), np.float32))

    fns, _calls = _score_fns()

    def fake_build(sht_path, det, get_pattern, max_bandwidth=128):
        return fns[2] if "alpha" in str(sht_path) else fns[1]

    monkeypatch.setattr(vu, "build_render_score_fn", fake_build)
    # `**_kw` so the double keeps working as the real signature grows — it took
    # `err_out` in 2026-09 (so a failed RUN can be told apart from rejected
    # DATA) and this seam then rejected the call, which surfaced as a 409
    # carrying a TypeError instead of a test failure at the seam.
    monkeypatch.setattr(
        ix, "_hough_quats_for_phase_batch",
        lambda pats, cif, det, **_kw: np.tile(IDENT, (np.asarray(pats).shape[0], 1)))
    return ix


def test_endpoint_check_reassign_undo_roundtrip(monkeypatch):
    result = _fake_two_phase_result()
    ix = _patch_endpoint_seams(monkeypatch, result)

    out = _run(ix.phase_check(ix.PhaseCheckRequest()))
    assert out["n_reassign"] == 1
    assert out["suspects"][0]["best_alt_name"] == "Al"
    assert "phase_check" in result.metadata
    assert result.metadata["phase_check"]["margin_map"].shape == (NR * NC,)

    orig_pid = np.asarray(result.xmap.phase_id).copy()
    out2 = _run(ix.phase_reassign(ix.PhaseReassignRequest()))
    assert out2["n_grains_applied"] == 1
    assert out2["n_pixels_changed"] == 24
    assert out2["undo_available"] is True
    new_pid = np.asarray(result.xmap.phase_id).reshape(NR, NC)
    assert (new_pid[:, :3] == 1).all() and (new_pid[:, 5:] == 2).all()
    # the answered pixels are blanked in the margin layer
    mm = result.metadata["phase_check"]["margin_map"].reshape(NR, NC)
    assert np.isnan(mm[:, :3]).all()

    out3 = _run(ix.phase_reassign_undo(ix.PhaseReassignRequest()))
    assert out3["n_restored"] == 24
    assert np.array_equal(np.asarray(result.xmap.phase_id), orig_pid)
    assert "phase_reassign_undo" not in result.metadata
    assert "phase_check" not in result.metadata          # forced re-check


def test_endpoint_reassign_requires_check_first(monkeypatch):
    from fastapi import HTTPException
    result = _fake_two_phase_result()
    ix = _patch_endpoint_seams(monkeypatch, result)
    with pytest.raises(HTTPException) as exc:
        _run(ix.phase_reassign(ix.PhaseReassignRequest()))
    assert exc.value.status_code == 400


def test_endpoint_check_rejects_non_spherical(monkeypatch):
    from fastapi import HTTPException
    result = _fake_two_phase_result()
    result.metadata["indexing_method"] = "hough"
    ix = _patch_endpoint_seams(monkeypatch, result)
    with pytest.raises(HTTPException) as exc:
        _run(ix.phase_check(ix.PhaseCheckRequest()))
    assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# Margin layer (phase_map.py)
# ---------------------------------------------------------------------------

def test_phase_margin_layer_renders_and_404s_before_compute():
    from unittest.mock import patch
    from fastapi import HTTPException
    from backend.api.routes.phase_map import _compute_layer_rgba

    fake = MagicMock()
    fake.original_shape = (NR, NC)
    fake.metadata = {}
    with patch("backend.api.routes.phase_map.get_last_indexing_result",
               return_value=fake):
        with pytest.raises(HTTPException) as exc:
            _compute_layer_rgba(kind="phase_margin")
        assert exc.value.status_code == 404

    margin = np.full(NR * NC, np.nan)
    margin[:3] = -0.17           # suspect → red-ish, opaque
    margin[3] = +0.10            # checked, stored wins → green-ish
    fake.metadata = {"phase_check": {"margin_map": margin}}
    with patch("backend.api.routes.phase_map.get_last_indexing_result",
               return_value=fake):
        rgba = _compute_layer_rgba(kind="phase_margin")
    assert rgba.shape == (NR, NC, 4)
    flat = rgba.reshape(-1, 4)
    assert (flat[:4, 3] == 255).all()                    # finite → opaque
    assert (flat[4:, 3] == 0).all()                      # NaN → transparent
    # red channel dominates on the losing pixels, green on the winning one
    assert flat[0, 0] > flat[0, 1]
    assert flat[3, 1] > flat[3, 0]


# ---------------------------------------------------------------------------
# Review hardening (2026-07-12 adversarial review)
# ---------------------------------------------------------------------------

def test_apply_identity_guard_skips_when_map_changed_since_check():
    """HIGH finding: grain ids are first-encounter ordinals of the
    segmentation — after an orientation edit the same id can point at a
    DIFFERENT physical grain. The apply step must verify count+centroid and
    skip on mismatch instead of flipping a good grain."""
    from backend.spherical_gpu.pipeline.phase_reassignment import apply_reassignment
    full_q, pf, report = _checked()
    # Simulate an orientation edit between check and apply: rotate the top
    # row of grain W by 30° about z → W splits, grain numbering shifts.
    q2 = full_q.copy()
    a = np.radians(30.0) / 2.0
    q2[[0, 1, 2]] = np.array([np.cos(a), 0.0, 0.0, np.sin(a)])
    new_pf, new_q, applied, skipped = apply_reassignment(
        q2, pf, NR, NC, report, PHASES, _hough_fn_factory())
    assert not applied
    assert skipped and skipped[0]["skip_reason"] in (
        "grain changed since check", "grain no longer found")
    assert np.array_equal(new_pf, pf)
    assert np.array_equal(new_q, q2)


def test_check_map_stored_render_failure_is_never_a_loss():
    """LOW finding: a non-finite STORED score (e.g. transient GPU OOM) is a
    scoring failure, not evidence against the phase — candidates must not
    win by default."""
    from backend.spherical_gpu.pipeline.phase_reassignment import check_map
    full_q, pf = _grid()

    def alpha_fn(flats, quats):
        flats = np.asarray(flats).reshape(-1)
        return np.where(flats % NC < 3, np.nan, 0.45)     # W render fails

    def al_fn(flats, quats):
        return np.full(np.asarray(flats).size, 0.50)      # candidate strong

    hough = _hough_fn_factory()
    report, margin = check_map(full_q, pf, NR, NC, PHASES,
                               {1: al_fn, 2: alpha_fn}, hough)
    assert report["n_reassign"] == 0
    no_score = [e for e in report["grains"] if e["decision"] == "no-score"]
    assert len(no_score) == 1 and no_score[0]["phase_id"] == 2
    assert not hough.calls                                # candidates never scored
    assert np.isnan(margin).all()


def test_check_map_candidate_needs_min_hough_pixels():
    """LOW finding: a candidate scored on a tiny Hough-lucky subset must not
    beat the stored phase's full-sample median."""
    from backend.spherical_gpu.pipeline.phase_reassignment import check_map
    full_q, pf = _grid()
    fns, _ = _score_fns()
    # Hough succeeds on only ONE pixel of the suspect grain's sample.
    calls = []

    def hough(cand_pid, flats):
        flats = np.asarray(flats).reshape(-1)
        calls.append(flats)
        out = np.full((flats.size, 4), np.nan)
        out[0] = IDENT
        return out

    report, _ = check_map(full_q, pf, NR, NC, PHASES, fns, hough)
    assert report["n_reassign"] == 0                      # candidate rejected
    keeps = [e for e in report["grains"] if e["decision"] == "keep"]
    assert len(keeps) == 1 and "best_alt_phase" not in keeps[0]


# ---------------------------------------------------------------------------
# V4 — manual per-grain phase assignment from the Compare-phases view
# ---------------------------------------------------------------------------

def test_assign_phase_endpoint_reassigns_clicked_grain(monkeypatch):
    """Clicking 'assign Al' on an alpha grain flips exactly that grain's
    pixels to Al with per-pixel Hough orientations; the shared
    /phase-reassign/undo restores it."""
    result = _fake_two_phase_result()
    ix = _patch_endpoint_seams(monkeypatch, result)
    orig_pid = np.asarray(result.xmap.phase_id).copy()

    out = _run(ix.assign_phase_to_grain(ix.AssignPhaseRequest(
        row=0, col=0, target_phase_id=1)))
    assert out["n_pixels_changed"] == 24            # grain W = cols 0-2
    assert out["grain_size"] == 24
    assert out["phase_to"] == "Al" and out["phase_from"] == "alpha"
    assert out["undo_available"] is True
    new_pid = np.asarray(result.xmap.phase_id).reshape(NR, NC)
    assert (new_pid[:, :3] == 1).all()              # reassigned
    assert (new_pid[:, 3:5] == 1).all() and (new_pid[:, 5:] == 2).all()
    # grain-flip undo (orientation-only) was invalidated — one undo semantic
    assert "grain_flip_undo" not in result.metadata

    out2 = _run(ix.phase_reassign_undo(ix.PhaseReassignRequest()))
    assert out2["n_restored"] == 24
    assert np.array_equal(np.asarray(result.xmap.phase_id), orig_pid)


def test_assign_phase_rejects_same_phase_and_unknown(monkeypatch):
    from fastapi import HTTPException
    result = _fake_two_phase_result()
    ix = _patch_endpoint_seams(monkeypatch, result)
    with pytest.raises(HTTPException) as e1:
        _run(ix.assign_phase_to_grain(ix.AssignPhaseRequest(
            row=0, col=0, target_phase_id=2)))     # already alpha
    assert e1.value.status_code == 400
    with pytest.raises(HTTPException) as e2:
        _run(ix.assign_phase_to_grain(ix.AssignPhaseRequest(
            row=0, col=0, target_phase_id=99)))
    assert e2.value.status_code == 400


def test_assign_phase_works_for_zero_pixel_target(monkeypatch):
    """Review M1: a systematically mis-indexed phase may own ZERO pixels —
    the compare dropdown still offers it (it has an SHT), and assigning it is
    exactly the hard case the tool exists for. Must not 400."""
    from types import SimpleNamespace
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation
    full_q, pf = _grid()
    x, y = np.meshgrid(np.arange(NC, dtype=float), np.arange(NR, dtype=float))
    xmap = CrystalMap(
        rotations=Rotation(full_q), phase_id=pf.copy(),
        x=x.ravel(), y=y.ravel(),
        phase_list=PhaseList(names=["Al", "alpha", "Si"],
                             point_groups=["m-3m", "m-3", "m-3m"],
                             ids=[1, 2, 3]),
    )
    result = SimpleNamespace(
        xmap=xmap, original_shape=(NR, NC),
        selection_mask=np.ones((NR, NC), bool),
        metadata={
            "indexing_method": "spherical",
            "sht_paths_by_phase": {1: "al.sht", 2: "alpha.sht", 3: "si.sht"},
            "detector_geometry": {"pat_width": 8, "pat_height": 8,
                                  "pc_x": 0.5, "pc_y": 0.5, "pc_z": 0.5},
        },
    )
    ix = _patch_endpoint_seams(monkeypatch, result)
    # orix prunes the zero-pixel phase from the CrystalMap's PhaseList — the
    # endpoint reconstructs it from its CIF. Fake the CIF loading.
    import ebsd_utils
    import indexing_controller as ic
    from orix.crystal_map import Phase
    monkeypatch.setattr(ebsd_utils, "sanitize_cif", lambda p_: p_)
    # endpoint names the reconstructed phase after the CIF stem
    monkeypatch.setattr(ic, "_resolve_cif_for_sht", lambda s: "Si.cif")
    monkeypatch.setattr(Phase, "from_cif",
                        classmethod(lambda cls, p_: Phase(name="Si", point_group="m-3m")))
    out = _run(ix.assign_phase_to_grain(ix.AssignPhaseRequest(
        row=0, col=0, target_phase_id=3)))          # Si owns 0 pixels
    assert out["phase_to"] == "Si"
    # the reconstructed phase is IN the PhaseList now (downstream lookups work)
    assert result.xmap.phases[3].name == "Si"
    assert out["n_pixels_changed"] == 24
    new_pid = np.asarray(result.xmap.phase_id).reshape(NR, NC)
    assert (new_pid[:, :3] == 3).all()


# ---------------------------------------------------------------------------
# Island check — the wrong pixels INSIDE a grain, which the per-grain check
# skips by construction (MIN_GRAIN_PX = 5).
# ---------------------------------------------------------------------------

def test_island_check_endpoint_reports_without_touching_the_result(monkeypatch):
    """Read-only by contract: it is the evidence, not the edit.

    The stored map must come back byte-identical, because this endpoint is what
    the user looks at BEFORE deciding whether anything should change.
    """
    # Built with the island already in place: orix's CrystalMap does not take
    # kindly to having phase_id reassigned after construction.
    result = _fake_result_with_island(row=6, col=6)
    ix = _patch_endpoint_seams(monkeypatch, result)

    before_pid = np.asarray(result.xmap.phase_id).copy()
    before_rot = np.asarray(result.xmap.rotations.data).copy()

    out = _run(ix.island_check(ix.IslandCheckRequest(max_variants=0)))

    assert out["n_islands"] >= 1
    assert set(out) >= {"n_phase_swap", "n_variant_flip", "n_unresolved", "findings"}
    # Every finding names both sides in words, not just ids.
    for f in out["findings"]:
        assert f["stored_name"] and f["enclosing_name"]
    assert np.array_equal(np.asarray(result.xmap.phase_id), before_pid)
    assert np.array_equal(np.asarray(result.xmap.rotations.data), before_rot)


# ---------------------------------------------------------------------------
# Stage 2 folded into the one button (2026-09-07)
#
# "Check phases" used to answer only about whole grains, so on a deformed 7050
# it reported 3 grains / 23 px while the defect the user was pointing at -- Al
# pixels sitting inside an Al7FeCu2 particle -- was 19 islands it never even
# counted (MIN_GRAIN_PX = 5). These pin that both stages now run from the one
# check and are applied by the one reassign, with the one undo.
# ---------------------------------------------------------------------------

def test_phase_check_also_reports_the_islands_inside_grains(monkeypatch):
    result = _fake_result_with_island(row=6, col=6)
    ix = _patch_endpoint_seams(monkeypatch, result)

    out = _run(ix.phase_check(ix.PhaseCheckRequest()))

    assert out["n_islands"] >= 1
    assert out["n_islands_reassign"] >= 1, "stage 2 found nothing to offer"
    assert out["island_error"] is None
    # The findings must be actionable: named phases, not bare ids.
    assert all(f["stored_name"] and f["enclosing_name"] for f in out["island_findings"])
    # Kept on the result, because the reassign is a separate request.
    assert isinstance(result.metadata["phase_check"]["islands"], dict)


def test_reassign_applies_the_islands_and_undo_puts_them_back(monkeypatch):
    result = _fake_result_with_island(row=6, col=6)
    ix = _patch_endpoint_seams(monkeypatch, result)

    _run(ix.phase_check(ix.PhaseCheckRequest()))
    before_pid = np.asarray(result.xmap.phase_id).copy()
    before_rot = np.asarray(result.xmap.rotations.data).copy()

    out = _run(ix.phase_reassign(ix.PhaseReassignRequest()))

    assert out["n_islands_applied"] >= 1
    assert out["n_island_pixels"] >= 1
    # Counted apart from the grain stage -- they are different repairs.
    assert "n_grains_applied" in out
    after_pid = np.asarray(result.xmap.phase_id)
    assert not np.array_equal(after_pid, before_pid), "reported a change it did not make"

    _run(ix.phase_reassign_undo(ix.PhaseReassignRequest()))
    assert np.array_equal(np.asarray(result.xmap.phase_id), before_pid)
    assert np.allclose(np.asarray(result.xmap.rotations.data), before_rot)


def test_a_second_reassign_does_not_rewrite_what_it_already_fixed(monkeypatch):
    """The applied findings are marked, so pressing again is a no-op.

    Without that the second press would 'succeed' with the same numbers while
    changing nothing -- and it would overwrite the undo record with an empty
    one, quietly costing the user the way back.
    """
    result = _fake_result_with_island(row=6, col=6)
    ix = _patch_endpoint_seams(monkeypatch, result)
    _run(ix.phase_check(ix.PhaseCheckRequest()))
    first = _run(ix.phase_reassign(ix.PhaseReassignRequest()))
    assert first["n_islands_applied"] >= 1

    second = _run(ix.phase_reassign(ix.PhaseReassignRequest()))
    assert second["n_islands_applied"] == 0
    assert second["n_pixels_changed"] == 0
