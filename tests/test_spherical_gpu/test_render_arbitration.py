"""Stage 0 of the map-wide variant unification: per grain, the resolver's Hough
orientation is kept unless the raw spherical candidate renders clearly better
(backend/spherical_gpu/pipeline/variant_unification.py::arbitrate_resolver_by_render).

Background (2026-09-09): the resolver trusted Hough on every pixel with a fit and
never rendered it. These tests pin the decision rule with an injected scorer —
no rendering, no GPU. A second change of the same day is covered at the end: the
render-verified adoption of unify_map no longer has a 128-px size cap."""
from __future__ import annotations

import numpy as np
import pytest


def _q_axis_angle(axis, deg):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    a = np.radians(deg) / 2.0
    return np.array([np.cos(a), *(np.sin(a) * axis)])


_ID = np.array([1.0, 0.0, 0.0, 0.0])
_FAR = _q_axis_angle([1, 1, 0], 50.0)       # 50 deg off — no m-3 / m-3m equivalent of identity
_G2 = _q_axis_angle([1, 1, 0], 40.0)        # a second, generically rotated grain


def _scorer(hough_map, scores, calls):
    """Scorer keyed by (flat, 'hough'|'raw'): a quaternion equal to the pixel's Hough
    orientation is the Hough candidate, anything else the raw one."""
    def score_fn(flat_idx, quats):
        out = []
        for f, qq in zip(np.atleast_1d(flat_idx), np.atleast_2d(quats)):
            f = int(f); calls.append(f)
            which = "hough" if abs(float(np.dot(qq, hough_map[f]))) > 0.9999 else "raw"
            out.append(scores[(f, which)] if (f, which) in scores else scores[which])
        return np.asarray(out, float)
    return score_fn


def test_restores_raw_on_a_grain_where_it_renders_clearly_better():
    from backend.spherical_gpu.pipeline.variant_unification import (
        arbitrate_resolver_by_render,
    )
    n_rows, n_cols = 3, 4
    q = np.tile(_ID, (12, 1)); raw = np.tile(_FAR, (12, 1)); ph = np.ones(12, dtype=np.int64)
    calls = []
    new_q, rep = arbitrate_resolver_by_render(
        q, raw, ph, n_rows, n_cols, 1, "m-3", _scorer(q, {"hough": 0.44, "raw": 0.64}, calls))
    assert new_q is not None and np.allclose(new_q, raw)
    assert rep["n_grains"] == 1 and rep["n_grains_scored"] == 1 and rep["n_grains_restored"] == 1
    assert rep["n_px_restored"] == 12 and rep["n_px_same"] == 0
    assert len(calls) <= 16                                  # at most 8 sampled pixels, two candidates each
    assert rep["grains"][0]["decision"] == "restored"
    assert rep["median_score_raw"] == pytest.approx(0.64) and rep["median_score_hough"] == pytest.approx(0.44)


def test_margin_is_a_hysteresis_not_a_coin_flip():
    from backend.spherical_gpu.pipeline.variant_unification import (
        arbitrate_resolver_by_render, ARBITRATION_MARGIN,
    )
    q = np.tile(_ID, (4, 1)); raw = np.tile(_FAR, (4, 1)); ph = np.ones(4, dtype=np.int64)
    new_q, rep = arbitrate_resolver_by_render(
        q, raw, ph, 1, 4, 1, "m-3", _scorer(q, {"hough": 0.50, "raw": 0.50 + ARBITRATION_MARGIN * 0.5}, []))
    assert new_q is None and rep["n_grains_restored"] == 0 and rep["grains"][0]["decision"] == "kept"
    new_q, rep = arbitrate_resolver_by_render(
        q, raw, ph, 1, 4, 1, "m-3", _scorer(q, {"hough": 0.50, "raw": 0.50 + ARBITRATION_MARGIN}, []))
    assert new_q is not None and rep["n_grains_restored"] == 1 and np.allclose(new_q, raw)


def test_symmetry_equivalent_candidates_are_the_same_answer():
    """A 90 deg turn about [001] is a symmetry operation of m-3m: the candidates are
    the same orientation, the grain is not scored at all."""
    from backend.spherical_gpu.pipeline.variant_unification import (
        arbitrate_resolver_by_render,
    )
    q = np.tile(_ID, (6, 1)); raw = np.tile(_q_axis_angle([0, 0, 1], 90.0), (6, 1)); ph = np.ones(6, dtype=np.int64)
    calls = []
    new_q, rep = arbitrate_resolver_by_render(q, raw, ph, 2, 3, 1, "m-3m", _scorer(q, {}, calls))
    assert new_q is None and rep["n_px_same"] == 6 and rep["n_grains_scored"] == 0 and calls == []


def test_two_grains_are_decided_independently():
    """Left half: Hough = identity; right half: Hough = a 40 deg rotation (its own
    grain modulo the supergroup). Raw differs everywhere; it renders better only
    on the right grain — only the right grain is restored, per pixel."""
    from backend.spherical_gpu.pipeline.variant_unification import (
        arbitrate_resolver_by_render,
    )
    n_rows, n_cols = 2, 8
    q = np.tile(_ID, (16, 1)); ph = np.ones(16, dtype=np.int64)
    right = np.array([(f % n_cols) >= 4 for f in range(16)])
    q[right] = _G2
    raw = np.tile(_FAR, (16, 1))
    scores = {}
    for f in range(16):
        scores[(f, "hough")] = 0.50
        scores[(f, "raw")] = 0.70 if right[f] else 0.30
    new_q, rep = arbitrate_resolver_by_render(q, raw, ph, n_rows, n_cols, 1, "m-3", _scorer(q, scores, []))
    assert rep["n_grains"] == 2 and rep["n_grains_scored"] == 2 and rep["n_grains_restored"] == 1
    assert new_q is not None
    assert np.allclose(new_q[right], _FAR) and np.allclose(new_q[~right], _ID)
    assert rep["n_px_restored"] == 8


def test_non_finite_raw_score_never_wins_and_other_phases_untouched():
    from backend.spherical_gpu.pipeline.variant_unification import (
        arbitrate_resolver_by_render,
    )
    q = np.tile(_ID, (6, 1)); raw = np.tile(_FAR, (6, 1))
    ph = np.array([1, 1, 1, 1, 2, 2], dtype=np.int64)          # phase 2 must be ignored
    calls = []
    new_q, rep = arbitrate_resolver_by_render(
        q, raw, ph, 2, 3, 1, "m-3", _scorer(q, {"hough": float("-inf"), "raw": float("nan")}, calls))
    assert new_q is None and rep["n_grains_scored"] == 1 and rep["n_grains_restored"] == 0
    assert all(f < 4 for f in calls) and calls            # only phase-1 pixels were scored


def test_wiring_restores_raw_grain_end_to_end(monkeypatch):
    """unify_after_hough_resolve with raw_eulers: the grain where the raw spherical
    candidate renders better comes back as the raw orientations, the report carries
    the arbitration counts, and a missing raw_eulers keeps the old behaviour."""
    import backend.spherical_gpu.pipeline.variant_unification as VU
    from orix.quaternion import Rotation

    n_rows, n_cols = 3, 4
    N = n_rows * n_cols
    q_h = np.tile(_ID, (N, 1)); raw = np.tile(_FAR, (N, 1))
    eul_h = np.asarray(Rotation(q_h).to_euler(), float)
    eul_r = np.asarray(Rotation(raw).to_euler(), float)
    phase_id = np.ones(N, dtype=np.int64)
    pats = np.zeros((N, 8, 8), np.float32)
    det = {"n_rows": n_rows, "n_cols": n_cols, "pat_height": 8, "pat_width": 8,
           "pc_x": 0.5, "pc_y": 0.5, "pc_z": 0.6, "pixel_size": 70.0}
    calls = []

    def fake_build(sht, det_params, get_pattern, **kw):
        def score_fn(flat_idx, quats):
            out = []
            for f, qq in zip(np.atleast_1d(flat_idx), np.atleast_2d(quats)):
                calls.append(int(f))
                out.append(0.64 if abs(float(np.dot(qq, _ID))) < 0.9999 else 0.44)
            return np.asarray(out, float)
        return score_fn

    monkeypatch.setattr(VU, "build_render_score_fn", fake_build)
    monkeypatch.setattr(VU, "unify_map", lambda *a, **k: (None, {"n_grains": 0, "n_flipped_units": 0,
                                                                  "n_ambiguous": 0, "n_rescued": 0, "grains": []}))
    eul_new, reports = VU.unify_after_hough_resolve(
        eul_h, phase_id, pats, ["x.sht"], [{"point_group": "m-3", "z_rot": 2}], det,
        None, False, {1}, raw_eulers=eul_r)
    assert eul_new is not None
    q_new = np.asarray(Rotation.from_euler(eul_new).data, float)
    assert all(abs(float(np.dot(q_new[i], _FAR))) > 0.9999 for i in range(N))
    arb = reports["arbitration"][1]
    assert arb["n_grains_scored"] == 1 and arb["n_grains_restored"] == 1 and arb["n_px_restored"] == N
    assert calls and len(calls) <= 16

    calls.clear()
    eul_none, reports2 = VU.unify_after_hough_resolve(
        eul_h, phase_id, pats, ["x.sht"], [{"point_group": "m-3", "z_rot": 2}], det,
        None, False, {1})
    assert eul_none is None and "arbitration" not in reports2 and calls == []


def test_adoption_has_no_size_cap_any_more():
    """A 200-px blob of Hough's second basin inside a 900-px grain: before
    2026-09-09 the 128-px cap left it alone; now the render-verified adoption
    reaches it (donor 3x bigger, clear margin) and maps it onto the grain."""
    from backend.spherical_gpu.pipeline.variant_unification import unify_map
    n_rows, n_cols = 33, 33
    n = n_rows * n_cols
    big = _ID
    blob_q = _q_axis_angle([1, 2, 3], 71.9)                        # not an m-3m coset variant of the grain
    q = np.tile(big, (n, 1)); ph = np.ones(n, dtype=np.int64)
    rr, cc = np.divmod(np.arange(n), n_cols)
    blob = (rr >= 10) & (rr < 24) & (cc >= 10) & (cc < 24)          # 14 x 14 = 196 px, fully enclosed
    q[blob] = blob_q
    # the blob is wrong: the grain's orientation renders better on it (clear margin)
    def score_fn(flat_idx, quats):
        out = []
        for f, qq in zip(np.atleast_1d(flat_idx), np.atleast_2d(quats)):
            out.append(0.65 if abs(float(np.dot(qq, big))) > 0.9999 else 0.40)
        return np.asarray(out, float)
    new_q, rep = unify_map(q, ph, n_rows, n_cols, 1, "m-3", score_fn)
    assert new_q is not None
    assert rep["n_adopted"] == 1
    assert all(abs(float(np.dot(new_q[i], big))) > 0.9999 for i in np.flatnonzero(blob))


# ---------------------------------------------------------------------------
# Pseudo-icosahedral variants of cubic approximants (2026-09-09)
# ---------------------------------------------------------------------------

def test_icosahedral_ops_form_the_group_of_60():
    from backend.spherical_gpu.pseudosym import icosahedral_ops, _qmul
    I = icosahedral_ops()
    assert I.shape == (60, 4)
    assert np.allclose(np.linalg.norm(I, axis=1), 1.0)
    # closed under multiplication (up to the sign of a quaternion)
    for a in I[::7]:
        for b in I[::11]:
            c = _qmul(a[None, :], b[None, :])[0]
            assert np.max(np.abs(I @ c)) > 1.0 - 1e-6


def test_pseudo_five_fold_144_is_a_class_rep_for_m3():
    """The operator measured on crop1 — 144 deg about a (0, 1, tau) axis — must be
    one of the variant classes of alpha-Al(Fe,Mn)Si (m-3), so the unification can
    score and choose it."""
    from backend.spherical_gpu.pipeline.variant_unification import class_reps_for_phase
    from backend.spherical_gpu.pseudosym import same_orientation_angle_deg
    tau = (1.0 + 5.0 ** 0.5) / 2.0
    p144 = _q_axis_angle([0.0, 1.0, tau], 144.0)
    reps = class_reps_for_phase("m-3")
    assert reps.shape[0] == 6
    d = same_orientation_angle_deg(reps, p144, "m-3")
    assert float(np.min(d)) < 1.0
    # and a plain cubic phase gets nothing extra
    assert class_reps_for_phase("m-3m").shape[0] == 1


def test_unify_map_recovers_a_grain_stuck_in_the_pseudo_five_fold_basin():
    """A whole grain sits in Hough's second basin (144 deg about a pseudo-five-fold
    axis); the injected scorer prefers the true orientation clearly. unify_map
    (speckle mode: one decision per grain over ALL classes) must bring every
    pixel back within 1 deg of the truth."""
    from backend.spherical_gpu.pipeline.variant_unification import unify_map
    from backend.spherical_gpu.pseudosym import _qmul, same_orientation_angle_deg
    tau = (1.0 + 5.0 ** 0.5) / 2.0
    p144 = _q_axis_angle([0.0, 1.0, tau], 144.0)
    q_true = _q_axis_angle([0.3, -0.5, 0.8], 37.0)
    n_rows, n_cols = 8, 8
    n = n_rows * n_cols
    q = np.tile(_qmul(p144[None, :], q_true[None, :])[0], (n, 1))     # wrong basin everywhere
    ph = np.ones(n, dtype=np.int64)

    def score_fn(flat_idx, quats):
        d = same_orientation_angle_deg(np.atleast_2d(quats), q_true, "m-3")
        return np.where(d < 3.0, 0.79, 0.50)

    new_q, rep = unify_map(q, ph, n_rows, n_cols, 1, "m-3", score_fn)
    assert new_q is not None
    d = same_orientation_angle_deg(new_q, q_true, "m-3")
    assert float(np.max(d)) < 1.0
    assert rep["n_flipped_units"] == 1
