"""Unit tests for the pseudo-symmetry grain-flip v2 coset-snap flood fill
(backend/spherical_gpu/pseudosym.py::grain_snap_floodfill).

v1 (`_grain_floodfill` + one rigid C) failed on variant-split grains: the
spherical wrong-basin pick scatters ONE physical grain across different
pseudo-variants (40-70 deg apart), which a plain 5 deg orientation flood fill
cannot cross. v2 snaps every pixel to ITS OWN best pseudo-variant (gradient-
tracking BFS), so a single apply corrects the whole grain while preserving
intra-grain distortions. The full render-NCC + apply-to-grain flow is verified
end-to-end on real SampleB data by a separate harness (2026-06-28 changelog);
here we lock in the pure segmentation + snap logic.
"""
from __future__ import annotations
import numpy as np
import pytest


def _q_axis_angle(axis, deg):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    a = np.radians(deg) / 2.0
    return np.array([np.cos(a), *(np.sin(a) * axis)])


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _angle_deg(qa, qb, pg="m-3"):
    from backend.spherical_gpu.pseudosym import same_orientation_angle_deg
    return float(same_orientation_angle_deg(np.asarray(qa)[None, :], np.asarray(qb), pg)[0])


def _two_grain_grid(nr=4, nc=4):
    """Left half = grain A (~5 deg about z), right half = grain B (~45 deg about z).

    45 deg about z is NOT a pseudo-variant of 5 deg about z under the m-3m\\m-3
    coset (variants sit at 90 deg multiples about z / cube axes), so the two
    grains must stay separate even for the snap fill.
    """
    qA = _q_axis_angle([0, 0, 1], 5.0)
    qB = _q_axis_angle([0, 0, 1], 45.0)
    full_q = np.zeros((nr * nc, 4))
    phase = np.zeros(nr * nc, int)
    for r in range(nr):
        for c in range(nc):
            full_q[r * nc + c] = qA if c < nc // 2 else qB
    return full_q, phase, nr, nc, qA, qB


def _fill(full_q, phase, nr, nc, start, q_target, pg="m-3", thr=10.0, cap=60.0,
          phase_id=0):
    from backend.spherical_gpu.pseudosym import grain_snap_floodfill
    return grain_snap_floodfill(full_q, phase, nr, nc, start, phase_id, pg,
                                q_target, threshold_deg=thr,
                                max_total_deg=cap)


# ---------------------------------------------------------------------------
# Segmentation rules that must still hold
# ---------------------------------------------------------------------------

def test_fill_segments_one_grain():
    full_q, phase, nr, nc, qA, _qB = _two_grain_grid()
    grain, new_q, _stats = _fill(full_q, phase, nr, nc, (0, 0), qA)
    assert len(grain) == 8                       # only the left half
    assert all(c < 2 for (_r, c) in grain)


def test_fill_from_other_grain():
    full_q, phase, nr, nc, _qA, qB = _two_grain_grid()
    grain, _new_q, _stats = _fill(full_q, phase, nr, nc, (0, 3), qB)
    assert len(grain) == 8
    assert all(c >= 2 for (_r, c) in grain)


def test_fill_respects_threshold():
    # gradient row, 8 deg per column: thr=5 keeps only the start pixel
    nr, nc = 1, 4
    full_q = np.stack([_q_axis_angle([0, 0, 1], 8.0 * c) for c in range(nc)])
    phase = np.zeros(nc, int)
    grain, _new_q, _stats = _fill(full_q, phase, nr, nc, (0, 0), full_q[0], thr=5.0)
    assert grain == [(0, 0)]


def test_fill_skips_unindexed_and_other_phase():
    full_q, phase, nr, nc, qA, _qB = _two_grain_grid()
    full_q[1] = np.nan                            # (0,1) unindexed
    phase[4] = 9                                  # (1,0) different phase
    grain, _new_q, _stats = _fill(full_q, phase, nr, nc, (0, 0), qA)
    assert (0, 1) not in grain
    assert (1, 0) not in grain
    assert (0, 0) in grain


# ---------------------------------------------------------------------------
# v2: variant-split grains (not
# handled by the earlier segmentation)
# ---------------------------------------------------------------------------

def test_fill_crosses_variant_boundary_and_snaps():
    """One physical grain, right half stuck in a pseudo-variant (90 deg about z,
    in the m-3m\\m-3 coset). v1's plain fill stops at the internal boundary
    (~90 deg >> threshold); the snap fill must cross it and return the WHOLE
    grain with every pixel snapped onto the target variant."""
    nr, nc = 4, 4
    qA = _q_axis_angle([0, 0, 1], 0.0)            # identity (true orientation)
    h = _q_axis_angle([0, 0, 1], 90.0)            # coset op (not in T = m-3 rotations)
    qB = _qmul(h, qA)                             # wrong variant (LEFT: Bunge/orix
    #                                               convention — crystal ops
    #                                               left-multiply sample→crystal g)
    full_q = np.zeros((nr * nc, 4))
    phase = np.zeros(nr * nc, int)
    for r in range(nr):
        for c in range(nc):
            full_q[r * nc + c] = qA if c < nc // 2 else qB
    # sanity: v1 semantics would see the halves ~90 deg apart (under m-3)
    assert _angle_deg(qA, qB) > 40.0
    grain, new_q, stats = _fill(full_q, phase, nr, nc, (0, 0), qA, thr=5.0, cap=15.0)
    assert len(grain) == 16                       # crossed the variant boundary
    for (r, c) in grain:
        assert _angle_deg(new_q[r * nc + c], qA) < 0.5
    assert stats["n_variant"] >= 8                # right half was snapped


def test_fill_preserves_intra_grain_distortion():
    """Deformed grain entirely in a wrong variant: measured q_c = h * t_c (LEFT
    composition — the physical mis-indexing relationship in the Bunge/orix
    convention) with a 1 deg/column bend t_c. The snap must recover ~t_c per
    pixel (NOT overwrite everything with the target) — intra-grain texture is
    preserved."""
    nr, nc = 1, 5
    h = _q_axis_angle([0, 0, 1], 90.0)
    t = [_q_axis_angle([1, 0, 0], 1.0 * c) for c in range(nc)]   # true orientations
    full_q = np.stack([_qmul(h, t[c]) for c in range(nc)])       # measured (flipped)
    phase = np.zeros(nc, int)
    grain, new_q, _stats = _fill(full_q, phase, nr, nc, (0, 0), t[0], thr=5.0, cap=15.0)
    assert len(grain) == nc
    for c in range(nc):
        assert _angle_deg(new_q[c], t[c]) < 0.1   # per-pixel truth recovered
    # texture preserved: last column is ~4 deg from first, not 0
    d = _angle_deg(new_q[nc - 1], new_q[0])
    assert 3.5 < d < 4.5


def test_fill_rigid_fallback_for_holohedral_group():
    """m-3m has no pseudo-variants: the rigid-C candidate must carry the fill
    (v1 behaviour preserved for non-pseudo-symmetric phases, e.g. a Hough
    target picked for a true-cubic phase)."""
    nr, nc = 2, 2
    q = _q_axis_angle([0, 0, 1], 10.0)
    q_target = _qmul(_q_axis_angle([1, 0, 0], 3.0), q)   # 3 deg correction
    full_q = np.tile(q, (nr * nc, 1))
    phase = np.zeros(nr * nc, int)
    grain, new_q, stats = _fill(full_q, phase, nr, nc, (0, 0), q_target,
                                pg="m-3m", thr=5.0, cap=15.0)
    assert len(grain) == 4
    for (r, c) in grain:
        assert _angle_deg(new_q[r * nc + c], q_target, pg="m-3m") < 0.1
    assert stats["n_rigid"] + stats["n_current"] >= 3


def test_fill_anti_drift_cap():
    """Gradient chain: neighbour steps pass the local threshold but the total-
    to-target cap must stop the walk (prevents leaking through low-angle
    boundaries into the next grain)."""
    nr, nc = 1, 5
    full_q = np.stack([_q_axis_angle([0, 0, 1], 4.0 * c) for c in range(nc)])
    phase = np.zeros(nc, int)
    grain, _new_q, _stats = _fill(full_q, phase, nr, nc, (0, 0), full_q[0],
                                  thr=5.0, cap=6.0)
    assert (0, 0) in grain and (0, 1) in grain    # 4 deg total, within cap
    assert (0, 2) not in grain                    # 8 deg total > 6 deg cap
    assert (0, 3) not in grain


def test_fill_start_pixel_gets_exact_target():
    full_q, phase, nr, nc, qA, _qB = _two_grain_grid()
    q_target = _q_axis_angle([0, 1, 0], 7.0)
    grain, new_q, _stats = _fill(full_q, phase, nr, nc, (0, 0), q_target, cap=180.0)
    np.testing.assert_allclose(new_q[0], q_target / np.linalg.norm(q_target), atol=1e-12)


# ---------------------------------------------------------------------------
# Endpoint plumbing: apply-to-grain v2 + undo (fake result, no GPU)
# ---------------------------------------------------------------------------

def _fake_spherical_result(nr=4, nc=4, layout="halves"):
    """Minimal result object satisfying _spherical_pixel_ctx + apply/undo.

    layout="halves": left half qA, right half the flipped variant (coherent
    two-domain / twin geometry). layout="speckle": checkerboard variant mix
    (the mis-indexing salt-and-pepper geometry)."""
    from types import SimpleNamespace
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation
    qA = _q_axis_angle([0, 0, 1], 0.0)
    h = _q_axis_angle([0, 0, 1], 90.0)
    qB = _qmul(h, qA)                             # wrong variant (left)
    quats = np.zeros((nr * nc, 4))
    for r in range(nr):
        for c in range(nc):
            if layout == "speckle":
                quats[r * nc + c] = qB if (r + c) % 2 else qA
            else:
                quats[r * nc + c] = qA if c < nc // 2 else qB
    x, y = np.meshgrid(np.arange(nc, dtype=float), np.arange(nr, dtype=float))
    xmap = CrystalMap(
        rotations=Rotation(quats), phase_id=np.zeros(nr * nc, int),
        x=x.ravel(), y=y.ravel(),
        phase_list=PhaseList(Phase(name="alpha", point_group="m-3")),
    )
    result = SimpleNamespace(
        xmap=xmap,
        original_shape=(nr, nc),
        selection_mask=np.ones((nr, nc), bool),
        metadata={
            "indexing_method": "spherical",
            "sht_paths_by_phase": {0: "dummy.sht"},
            "detector_geometry": {"pat_width": 8, "pat_height": 8,
                                  "pc_x": 0.5, "pc_y": 0.5, "pc_z": 0.5},
        },
    )
    return result, qA, qB


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def test_apply_endpoint_snaps_whole_variant_split_grain(monkeypatch):
    import backend.api.routes.indexing as ix
    result, qA, _qB = _fake_spherical_result()
    monkeypatch.setattr(ix, "_get_result", lambda *_a, **_k: result)
    req = ix.GrainApplyRequest(row=0, col=0, quat=[float(v) for v in qA],
                               threshold_deg=5.0, max_total_deg=15.0)
    out = _run(ix.apply_variant_to_grain(req))
    assert out["grain_size"] == 16                # crossed the variant boundary
    # Only the wrong half actually MOVES — already-correct pixels snap to
    # themselves and are no longer counted/undo-stored (no-op filter).
    assert out["n_changed"] == 8
    assert out["undo_available"] is True
    assert out["snap"]["n_variant"] >= 8
    new = np.asarray(result.xmap.rotations.data).reshape(-1, 4)
    for i in range(16):
        assert _angle_deg(new[i], qA) < 0.5


def test_undo_restores_exact_quaternions(monkeypatch):
    import backend.api.routes.indexing as ix
    result, qA, _qB = _fake_spherical_result()
    before = np.asarray(result.xmap.rotations.data).reshape(-1, 4).copy()
    monkeypatch.setattr(ix, "_get_result", lambda *_a, **_k: result)
    _run(ix.apply_variant_to_grain(ix.GrainApplyRequest(
        row=0, col=0, quat=[float(v) for v in qA])))
    changed = np.asarray(result.xmap.rotations.data).reshape(-1, 4)
    assert not np.allclose(changed, before)       # apply really changed things
    out = _run(ix.undo_grain_flip(ix.GrainUndoRequest()))
    assert out["n_restored"] == 8                 # only truly-changed pixels stored
    after = np.asarray(result.xmap.rotations.data).reshape(-1, 4)
    np.testing.assert_allclose(after, before, atol=1e-12)
    # second undo has nothing left
    import pytest
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        _run(ix.undo_grain_flip(ix.GrainUndoRequest()))


def test_apply_refine_flag_is_failsafe(monkeypatch):
    """refine=True must never break the apply: a rejected/failing refine keeps
    the snapped orientations and reports its status."""
    import backend.api.routes.indexing as ix
    result, qA, _qB = _fake_spherical_result()
    monkeypatch.setattr(ix, "_get_result", lambda *_a, **_k: result)
    monkeypatch.setattr(ix, "_grain_newton_refine",
                        lambda *a, **k: (None, {"status": "rejected",
                                                "reason": "render_ncc_gate"}))
    out = _run(ix.apply_variant_to_grain(ix.GrainApplyRequest(
        row=0, col=0, quat=[float(v) for v in qA], refine=True)))
    assert out["n_changed"] == 8                  # snap still applied (wrong half)
    assert out["refine"]["status"] == "rejected"
    new = np.asarray(result.xmap.rotations.data).reshape(-1, 4)
    assert _angle_deg(new[15], qA) < 0.5


@pytest.mark.parametrize("raised", [
    RuntimeError("CUDA out of memory. Tried to allocate 3.00 GiB"),
    "wrapped",           # what load_or_get_phase raises after the sphere-preview change
])
def test_refine_oom_keeps_the_release_gpu_hint(monkeypatch, raised):
    """A GPU out-of-memory inside the refine -- raw from torch, or wrapped by
    load_or_get_phase into SHTOutOfMemoryError (whose text no longer says
    "out of memory") -- must still end as the actionable 'Release GPU' skip
    and never as a generic error."""
    import types

    import backend.api.routes.indexing as ix
    import tools.pattern_comparison as pc
    from backend.api.services.sht_pattern_renderer import SHTOutOfMemoryError

    exc = (SHTOutOfMemoryError("Not enough memory to load this phase at bandwidth 256.")
           if raised == "wrapped" else raised)

    def _boom(*_a, **_k):
        raise exc

    monkeypatch.setattr(pc, "get_experimental_pattern", _boom)
    result = types.SimpleNamespace(original_shape=(2, 2))
    refined, summary = ix._grain_newton_refine(
        result, {}, "x.sht", "m-3", [(0, 0)], {0: np.array([1.0, 0, 0, 0])})
    assert refined is None
    assert summary["status"] == "skipped", summary
    assert "Release GPU" in summary["reason"]


def test_apply_refine_applies_accepted_pixels(monkeypatch):
    import backend.api.routes.indexing as ix
    result, qA, _qB = _fake_spherical_result()
    monkeypatch.setattr(ix, "_get_result", lambda *_a, **_k: result)
    q_ref = _q_axis_angle([1, 0, 0], 1.0)         # pretend refine moved pixel 5

    def _fake_refine(result_, det, sht, pg, coords, quats, **k):
        return {5: q_ref}, {"status": "applied", "n_refined": 1}
    monkeypatch.setattr(ix, "_grain_newton_refine", _fake_refine)
    out = _run(ix.apply_variant_to_grain(ix.GrainApplyRequest(
        row=0, col=0, quat=[float(v) for v in qA], refine=True)))
    assert out["refine"]["status"] == "applied"
    new = np.asarray(result.xmap.rotations.data).reshape(-1, 4)
    np.testing.assert_allclose(np.abs(np.dot(new[5], q_ref)), 1.0, atol=1e-9)


def test_unify_endpoint_fixes_variant_split_map(monkeypatch):
    """POST /pseudosym/unify on a variant-split fake result: the whole map is
    unified onto the render-NCC-winning class, undo restores exactly."""
    import backend.api.routes.indexing as ix
    import backend.spherical_gpu.pipeline.variant_unification as vu
    result, qA, _qB = _fake_spherical_result()
    result.metadata["sht_paths_by_phase"] = {0: "dummy.sht"}
    before = np.asarray(result.xmap.rotations.data).reshape(-1, 4).copy()
    monkeypatch.setattr(ix, "_get_result", lambda *_a, **_k: result)

    def _fake_build_score(sht, det, get_pattern, max_bandwidth=128):
        def score(flat_idx, quats):
            out = []
            for q in np.atleast_2d(quats):
                out.append(0.5 if _angle_deg(q, qA) < 5.0 else 0.2)
            return np.asarray(out)
        return score
    monkeypatch.setattr(vu, "build_render_score_fn", _fake_build_score)

    out = _run(ix.unify_pseudosym_variants(ix.UnifyVariantsRequest()))
    assert out["n_changed"] > 0
    assert out["n_grains"] >= 1
    assert out["undo_available"] is True
    new = np.asarray(result.xmap.rotations.data).reshape(-1, 4)
    for i in range(new.shape[0]):
        assert _angle_deg(new[i], qA) < 0.5        # whole map on the true class
    # one-level undo restores the speckled original exactly
    out2 = _run(ix.undo_grain_flip(ix.GrainUndoRequest()))
    assert out2["n_restored"] == out["n_changed"]
    np.testing.assert_allclose(
        np.asarray(result.xmap.rotations.data).reshape(-1, 4), before, atol=1e-12)


def test_unify_endpoint_flat_signal_makes_consistent_not_flipped(monkeypatch):
    """SPECKLED map + flat render signal: must still become CONSISTENT
    (speckle is unphysical) onto the DOMINANT class, flagged ambiguous.
    (A coherent two-domain layout would instead be twin-protected —
    covered by the core unit tests.)"""
    import backend.api.routes.indexing as ix
    import backend.spherical_gpu.pipeline.variant_unification as vu
    result, qA, _qB = _fake_spherical_result(layout="speckle")
    result.metadata["sht_paths_by_phase"] = {0: "dummy.sht"}
    monkeypatch.setattr(ix, "_get_result", lambda *_a, **_k: result)
    monkeypatch.setattr(
        vu, "build_render_score_fn",
        lambda *a, **k: (lambda i, q: np.full(len(np.atleast_2d(q)), 0.3)))
    out = _run(ix.unify_pseudosym_variants(ix.UnifyVariantsRequest()))
    assert out["n_ambiguous"] >= 1
    assert out["ambiguous_grains"]
    from backend.spherical_gpu.pipeline.variant_unification import (
        class_label_pixels, proper_coset_class_reps)
    new = np.asarray(result.xmap.rotations.data).reshape(-1, 4)
    reps = proper_coset_class_reps("m-3")
    cls = class_label_pixels(new, new[0], reps, "m-3")
    assert len(set(cls.tolist())) == 1             # consistent


def test_apply_refine_skipped_for_huge_grain(monkeypatch):
    import backend.api.routes.indexing as ix
    result, qA, _qB = _fake_spherical_result()
    monkeypatch.setattr(ix, "_get_result", lambda *_a, **_k: result)
    monkeypatch.setattr(ix, "MAX_REFINE_GRAIN_PX", 4)
    called = []
    monkeypatch.setattr(ix, "_grain_newton_refine",
                        lambda *a, **k: called.append(1) or (None, {}))
    out = _run(ix.apply_variant_to_grain(ix.GrainApplyRequest(
        row=0, col=0, quat=[float(v) for v in qA], refine=True)))
    assert not called
    assert out["refine"]["status"] == "skipped"


# ---------------------------------------------------------------------------
# Reference-orientation adoption: neighbour-grain + reference-
# pixel candidates in the variant gallery. A wrongly indexed grain in a
# FOREIGN basin (neither pseudo-variant nor Hough) gets its correct
# orientation from next door / a user-named pixel — rendered + ranked like
# every other candidate.
# ---------------------------------------------------------------------------

def _fake_two_grain_result(nr=8, nc=8, angle_deg=30.0):
    """Left half grain A (identity), right half a REAL second grain rotated
    `angle_deg` about a low-symmetry axis — NOT a coset variant of A, i.e.
    the foreign-basin geometry the classic candidates cannot reach."""
    from types import SimpleNamespace
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation
    qA = _q_axis_angle([0, 0, 1], 3.0)
    qB = _qmul(_q_axis_angle([1, 2, 3], angle_deg), qA)
    quats = np.zeros((nr * nc, 4))
    for r in range(nr):
        for c in range(nc):
            quats[r * nc + c] = qA if c < nc // 2 else qB
    x, y = np.meshgrid(np.arange(nc, dtype=float), np.arange(nr, dtype=float))
    xmap = CrystalMap(
        rotations=Rotation(quats), phase_id=np.zeros(nr * nc, int),
        x=x.ravel(), y=y.ravel(),
        phase_list=PhaseList(Phase(name="alpha", point_group="m-3")),
    )
    result = SimpleNamespace(
        xmap=xmap, original_shape=(nr, nc),
        selection_mask=np.ones((nr, nc), bool),
        metadata={
            "indexing_method": "spherical",
            "sht_paths_by_phase": {0: "dummy.sht"},
            "detector_geometry": {"pat_width": 8, "pat_height": 8,
                                  "pc_x": 0.5, "pc_y": 0.5, "pc_z": 0.5},
        },
    )
    return result, qA, qB


def test_neighbour_grain_candidates_finds_foreign_basin_neighbour():
    import backend.api.routes.indexing as ix
    from backend.spherical_gpu.pseudosym import same_orientation_angle_deg
    result, _qA, qB = _fake_two_grain_result()
    out = ix._neighbour_grain_candidates(result, result.xmap, 0, "m-3", 0, 0)
    assert len(out) == 1
    lab, q, npx = out[0]
    assert "neighbour" in lab and npx == 32
    # branch-consistent mean of a rigid grain == its orientation (mod holohedry)
    assert float(same_orientation_angle_deg(
        np.asarray(q)[None, :], qB, "m-3")[0]) < 1.0


def test_neighbour_grain_candidates_none_when_isolated(monkeypatch):
    """A single-grain map has no neighbours — fail-soft empty list."""
    import backend.api.routes.indexing as ix
    result, _qA, _qB = _fake_two_grain_result(angle_deg=0.5)   # one grain (0.5° < 5°)
    out = ix._neighbour_grain_candidates(result, result.xmap, 0, "m-3", 0, 0)
    assert out == []


def _patch_variant_render_seams(monkeypatch):
    """Make the variants endpoint runnable without GPU/SHT/CIF files."""
    import io as _io
    import base64 as _b64
    import backend.api.routes.indexing as ix
    import backend.api.services.sht_pattern_renderer as shtr
    import tools.pattern_comparison as pcmp
    import indexing_controller as ic
    from PIL import Image

    rng = np.random.default_rng(3)
    exp = (rng.random((8, 8)) * 255).astype(np.float32)
    sim_png = _io.BytesIO()
    Image.fromarray((rng.random((8, 8)) * 255).astype(np.uint8)).save(
        sim_png, format="PNG")
    sim_b64 = _b64.b64encode(sim_png.getvalue()).decode("ascii")

    monkeypatch.setattr(pcmp, "get_experimental_pattern", lambda *_a, **_k: exp)
    monkeypatch.setattr(pcmp, "detect_circular_aperture", lambda *_a, **_k: False)
    monkeypatch.setattr(shtr, "render_pattern_to_png_b64",
                        lambda **_k: sim_b64)
    monkeypatch.setattr(ic, "_resolve_cif_for_sht", lambda *_a, **_k: "")
    monkeypatch.setattr(ix, "_hough_euler_for_phase", lambda *_a, **_k: None)
    return ix


def test_variants_endpoint_includes_neighbour_and_reference(monkeypatch):
    ix = _patch_variant_render_seams(monkeypatch)
    result, _qA, qB = _fake_two_grain_result()
    monkeypatch.setattr(ix, "_get_result", lambda *_a, **_k: result)

    out = _run(ix.get_pattern_match_variants(row=0, col=0))
    kinds = [c["kind"] for c in out["candidates"]]
    assert "current" in kinds and "neighbour" in kinds
    nb = next(c for c in out["candidates"] if c["kind"] == "neighbour")
    assert nb["disorientation_deg"] > 5.0          # foreign basin, not noise
    assert nb["quat"] and nb["thumbnail"]

    # Reference pixel: names a pixel INSIDE grain B —
    # dedupes against the (identical) neighbour candidate, so asking for a
    # DIFFERENT pixel in grain A dedupes against 'current' instead.
    out2 = _run(ix.get_pattern_match_variants(row=0, col=0, ref_row=7, ref_col=1))
    kinds2 = [c["kind"] for c in out2["candidates"]]
    assert "reference" not in kinds2               # grain-A pixel ≈ current → deduped

    # A reference in a THIRD orientation must appear. Put it in grain B's
    # INTERIOR (not touching grain A) — a pixel adjacent to the clicked
    # grain would surface as a neighbour-grain candidate first and the
    # reference would (correctly) dedupe against it.
    q3 = _qmul(_q_axis_angle([3, 1, 2], 44.0), _qA)
    qd = np.asarray(result.xmap.rotations.data).copy()
    qd[4 * 8 + 6] = q3
    from orix.quaternion import Rotation as _R
    result.xmap._rotations = _R(qd)
    out3 = _run(ix.get_pattern_match_variants(row=0, col=0, ref_row=4, ref_col=6))
    ref = [c for c in out3["candidates"] if c["kind"] == "reference"]
    assert len(ref) == 1 and "4,6" in ref[0]["label"].replace(" ", "")


# ---------------------------------------------------------------------------
# V2 — propagate the correction to all similar grains (render-verified)
# ---------------------------------------------------------------------------

def _fake_nests_result(nr=8, nc=8):
    """One big correct grain (qT everywhere) with THREE 2×2 nests at the same
    wrong orientation qW: the clicked nest (top-left), a mis-indexed sibling
    (bottom-left) and a 'genuinely different' nest (bottom-right) that the
    render gate must protect."""
    from types import SimpleNamespace
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation
    qT = _q_axis_angle([0, 0, 1], 3.0)
    qW = _qmul(_q_axis_angle([1, 2, 3], 30.0), qT)      # foreign basin
    quats = np.tile(qT, (nr * nc, 1)).astype(float)
    nests = {"clicked": [(0, 0), (0, 1), (1, 0), (1, 1)],
             "sibling": [(6, 0), (6, 1), (7, 0), (7, 1)],
             "real":    [(6, 6), (6, 7), (7, 6), (7, 7)]}
    for pix in nests.values():
        for (r, c) in pix:
            quats[r * nc + c] = qW
    # A stripe of ANOTHER phase separates top and bottom — the real Scan1
    # geometry (nests embedded in a multi-phase map). The same-phase flood
    # fill cannot cross it, which is exactly why propagation is needed.
    pid = np.zeros(nr * nc, int)
    pid[4 * nc:(4 + 1) * nc] = 1
    x, y = np.meshgrid(np.arange(nc, dtype=float), np.arange(nr, dtype=float))
    xmap = CrystalMap(
        rotations=Rotation(quats), phase_id=pid,
        x=x.ravel(), y=y.ravel(),
        phase_list=PhaseList(names=["alpha", "Al"],
                             point_groups=["m-3", "m-3m"], ids=[0, 1]),
    )
    result = SimpleNamespace(
        xmap=xmap, original_shape=(nr, nc),
        selection_mask=np.ones((nr, nc), bool),
        metadata={
            "indexing_method": "spherical",
            "sht_paths_by_phase": {0: "dummy.sht"},
            "detector_geometry": {"pat_width": 8, "pat_height": 8,
                                  "pc_x": 0.5, "pc_y": 0.5, "pc_z": 0.5},
        },
    )
    return result, qT, qW, nests


def test_apply_propagate_similar_fixes_siblings_render_verified(monkeypatch):
    import backend.api.routes.indexing as ix
    import backend.spherical_gpu.pipeline.variant_unification as vu
    from backend.spherical_gpu.pseudosym import same_orientation_angle_deg
    result, qT, qW, nests = _fake_nests_result()
    monkeypatch.setattr(ix, "_get_result", lambda *_a, **_k: result)
    monkeypatch.setattr(ix, "_free_interactive_gpu_caches", lambda: None)

    real_flats = {r * 8 + c for (r, c) in nests["real"]}

    def fake_build(sht_path, det, get_pattern, max_bandwidth=128):
        def score(flats, quats):
            flats = np.asarray(flats).reshape(-1)
            quats = np.asarray(quats).reshape(-1, 4)
            out = np.empty(flats.size)
            for i, (f, q) in enumerate(zip(flats, quats)):
                near_T = float(same_orientation_angle_deg(
                    q[None, :], qT, "m-3")[0]) < 5.0
                if int(f) in real_flats:
                    # this nest is GENUINELY at qW: the wrong-basin
                    # orientation renders well, the 'correction' renders badly
                    out[i] = 0.1 if near_T else 0.5
                else:
                    out[i] = 0.5 if near_T else 0.1
            return out
        return score

    monkeypatch.setattr(vu, "build_render_score_fn", fake_build)

    req = ix.GrainApplyRequest(row=0, col=0, quat=[float(v) for v in qT],
                               threshold_deg=5.0, propagate_similar=True)
    out = _run(ix.apply_variant_to_grain(req))
    prop = out["propagate"]
    assert prop["n_candidates"] == 2                 # sibling + real
    assert prop["n_adopted"] == 1 and prop["n_rejected"] == 1
    assert prop["n_pixels"] == 4
    assert out["n_changed"] == 8                     # clicked 4 + sibling 4

    new = np.asarray(result.xmap.rotations.data).reshape(-1, 4)
    for (r, c) in nests["clicked"] + nests["sibling"]:
        assert float(same_orientation_angle_deg(
            new[r * 8 + c][None, :], qT, "m-3")[0]) < 2.0
    for (r, c) in nests["real"]:                     # render gate protected it
        assert float(same_orientation_angle_deg(
            new[r * 8 + c][None, :], qW, "m-3")[0]) < 1e-6

    # ONE undo restores clicked grain + sibling together
    out2 = _run(ix.undo_grain_flip(ix.GrainUndoRequest()))
    assert out2["n_restored"] == 8
    new2 = np.asarray(result.xmap.rotations.data).reshape(-1, 4)
    for (r, c) in nests["clicked"] + nests["sibling"]:
        assert float(same_orientation_angle_deg(
            new2[r * 8 + c][None, :], qW, "m-3")[0]) < 1e-6


def test_apply_propagate_off_is_unchanged_behaviour(monkeypatch):
    """propagate_similar=False (default) must not touch siblings and must
    not build a render scorer at all."""
    import backend.api.routes.indexing as ix
    import backend.spherical_gpu.pipeline.variant_unification as vu
    from backend.spherical_gpu.pseudosym import same_orientation_angle_deg
    result, qT, qW, nests = _fake_nests_result()
    monkeypatch.setattr(ix, "_get_result", lambda *_a, **_k: result)

    def boom(*_a, **_k):
        raise AssertionError("scorer must not be built when propagation is off")
    monkeypatch.setattr(vu, "build_render_score_fn", boom)

    req = ix.GrainApplyRequest(row=0, col=0, quat=[float(v) for v in qT],
                               threshold_deg=5.0)
    out = _run(ix.apply_variant_to_grain(req))
    assert out["n_changed"] == 4 and out["propagate"] is None
    new = np.asarray(result.xmap.rotations.data).reshape(-1, 4)
    for (r, c) in nests["sibling"] + nests["real"]:
        assert float(same_orientation_angle_deg(
            new[r * 8 + c][None, :], qW, "m-3")[0]) < 1e-6


def test_variants_reindex_candidate_and_loud_reference_error(monkeypatch):
    """2026-07-15 user round: (a) opt-in re-index candidate — fresh full
    search for the pixel's own phase, surfaced like every other candidate;
    (b) an unusable reference pixel must FAIL LOUD via reference_error
    (it used to vanish silently — coordinate-order/ROI confusion)."""
    ix = _patch_variant_render_seams(monkeypatch)
    result, qA, _qB = _fake_two_grain_result()
    monkeypatch.setattr(ix, "_get_result", lambda *_a, **_k: result)

    from orix.quaternion import Rotation as _R
    q3 = _qmul(_q_axis_angle([3, 1, 2], 44.0), qA)      # distinct third basin
    eu_deg = [float(v) for v in np.degrees(
        np.asarray(_R(q3[None, :]).to_euler()).reshape(3))]
    seen_kwargs = {}

    def fake_compare(**kw):
        seen_kwargs.update(kw)
        return [{"euler_deg": eu_deg, "phase_id": 0}]

    monkeypatch.setattr(ix, "_compute_phase_compare_results", fake_compare)

    out = _run(ix.get_pattern_match_variants(row=0, col=0, reindex=True,
                                             ref_row=999, ref_col=0))
    kinds = [c["kind"] for c in out["candidates"]]
    assert "reindex" in kinds
    ri = next(c for c in out["candidates"] if c["kind"] == "reindex")
    from backend.spherical_gpu.pseudosym import same_orientation_angle_deg
    assert float(same_orientation_angle_deg(
        np.asarray(ri["quat"])[None, :], q3, "m-3")[0]) < 0.1
    # single-phase compare scoped to the clicked pixel's own phase
    assert list(seen_kwargs["sht_paths_by_phase"].keys()) == [0]
    # out-of-bounds reference fails LOUD, not silent
    assert out["reference_error"] and "outside" in out["reference_error"]
    assert "reference" not in kinds


def test_annotate_compare_rows_with_delta():
    """Same-phase compare rows get Δ° to the stored orientation + an
    adoptable quat; other-phase rows stay untouched (lets the
    user see 'same orientation, only 0.7° off' instead of guessing why
    stored R and compare R disagree)."""
    import backend.api.routes.indexing as ix
    from orix.quaternion import Rotation as _R

    q_stored = _q_axis_angle([0, 0, 1], 20.0)
    # Re-indexed orientation: stored + a small 0.7° tweak about a skew axis.
    q_near = _qmul(_q_axis_angle([1, 2, 3], 0.7), q_stored)
    eu_near = [float(v) for v in np.degrees(
        np.asarray(_R(q_near[None, :]).to_euler()).reshape(3))]
    # And a symmetry-equivalent of that tweak (90° about Z is an m-3m op):
    # must STILL report ~0.7°, not ~90°.
    q_sym = _qmul(_q_axis_angle([0, 0, 1], 90.0), q_near)
    eu_sym = [float(v) for v in np.degrees(
        np.asarray(_R(q_sym[None, :]).to_euler()).reshape(3))]

    rows = [
        {"phase_id": 1, "euler_deg": eu_near},
        {"phase_id": 1, "euler_deg": eu_sym},
        {"phase_id": 2, "euler_deg": [10.0, 20.0, 30.0]},   # other phase
        {"phase_id": 1, "euler_deg": []},                    # degenerate
    ]
    ix._annotate_compare_rows_with_delta(rows, q_stored, 1, "m-3m")

    assert abs(rows[0]["disorientation_deg"] - 0.7) < 0.05
    assert abs(rows[1]["disorientation_deg"] - 0.7) < 0.05
    assert len(rows[0]["quat_wxyz"]) == 4
    assert "disorientation_deg" not in rows[2]
    assert "disorientation_deg" not in rows[3]

    # Fail-soft guards: no stored quat / empty rows are silent no-ops.
    ix._annotate_compare_rows_with_delta(rows, None, 1, "m-3m")
    ix._annotate_compare_rows_with_delta([], q_stored, 1, "m-3m")
