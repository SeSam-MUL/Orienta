"""A rigid grain correction must act on ONE member of each pixel's orbit.

Background (2026-09-09, commit ``e605d0f8``)
--------------------------------------------
An orientation is an orbit ``{S·q}`` and an indexer's export stores an arbitrary
member of it per pixel.  Applying one operator ``C`` on the LEFT is
representative-independent only when ``C`` normalises the crystal group; for a
general ``C`` the images ``C·S_i·q`` of two pixels that carry the SAME physical
orientation are DIFFERENT orientations, and the grain comes back speckled.
``variant_unification.nearest_representative`` was written for exactly that,
after 53 of 61 speckle pixels in the ICAA20 crop turned out to be this.

``phase_reassignment.rigid_grain_quats`` has the same shape -- it carries
``C = q_seed · q_click⁻¹`` across a grain by left-multiplying the STORED
quaternions -- and here ``C`` is not even close to a symmetry: it is the
rotation from the old phase's orientation at the clicked pixel to the new
phase's, i.e. an arbitrary element of SO(3).  Nothing normalises anything.

User-visible: "Assign <phase> to this grain" with the seed fallback (Hough
produced nothing for the target phase) hands back a grain whose IPF colouring
is speckled, at up to the group's largest operator angle, out of pixels that
were one uniform grain before.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.spherical_gpu.pipeline.phase_reassignment import rigid_grain_quats
from backend.spherical_gpu.pseudosym import _qmul, _sym_quats


PG = "m-3"          # the cubic approximants this whole tool chain exists for


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _angle_deg(qa, qb):
    """Plain rotation angle between two quaternions (no symmetry reduction)."""
    d = abs(float(np.dot(np.asarray(qa).ravel(), np.asarray(qb).ravel())))
    return float(np.degrees(2.0 * np.arccos(min(1.0, d))))


def _mixed_representative_grain(seed=3, ops=(0, 4, 9, 17), spread_deg=0.0):
    """One physical orientation per pixel, stored under DIFFERENT orbit members.

    ``spread_deg`` optionally gives each pixel a small genuine deviation first,
    so the same fixture can also check that the real intra-grain rotation field
    survives.
    """
    rng = np.random.default_rng(seed)
    q_base = _unit(rng.normal(size=4))
    sym = _sym_quats(PG)
    rows, truths = [], []
    for k, op in enumerate(ops):
        q_true = q_base
        if spread_deg:
            # A small rotation on the RIGHT: a real move of the crystal, not a
            # change of representative.
            ang = np.radians(spread_deg) * k / max(len(ops) - 1, 1)
            dq = np.array([np.cos(ang / 2), np.sin(ang / 2), 0.0, 0.0])
            q_true = _qmul(q_base.reshape(1, 4), dq.reshape(1, 4))[0]
        truths.append(q_true)
        rows.append(_qmul(sym[op].reshape(1, 4), q_true.reshape(1, 4))[0])
    return np.stack(rows), np.stack(truths), rng


def test_the_fixture_really_holds_one_orientation_in_several_guises():
    """Guard the premise: the stored quaternions must LOOK different and BE the
    same orientation. A fixture that fails this proves nothing."""
    from backend.spherical_gpu.pseudosym import same_orientation_angle_deg
    stored, _truth, _rng = _mixed_representative_grain()
    assert max(_angle_deg(stored[0], s) for s in stored[1:]) > 45.0
    for s in stored[1:]:
        assert float(same_orientation_angle_deg(
            s[None, :], stored[0], PG)[0]) < 1e-6


def test_a_uniform_grain_stays_uniform_through_the_rigid_correction():
    """The defect. Every pixel held one orientation; every pixel must end on
    one orientation."""
    stored, _truth, rng = _mixed_representative_grain()
    q_seed = _unit(rng.normal(size=4))
    out = rigid_grain_quats(stored, q_click=stored[0], q_seed=q_seed,
                            point_group=PG)
    # Compared as |q1.q2| = 1, not as an angle: arccos is so badly conditioned
    # at 1 that two bit-identical quaternions read ~2e-4 deg apart, which would
    # make the tolerance a statement about arccos (same reasoning as
    # tests/test_assign_seed_and_message.py).
    dots = np.abs(out[1:] @ out[0])
    worst_deg = max(_angle_deg(out[0], o) for o in out[1:])
    assert np.allclose(dots, 1.0, atol=1e-12), (
        f"pixels that all held the same orientation came out up to "
        f"{worst_deg:.2f} deg apart -- the operator was applied to a different "
        f"orbit member per pixel (see "
        f"variant_unification.nearest_representative)")


def test_the_clicked_pixel_still_lands_exactly_on_the_seed():
    """Unchanged contract: the orientation the user judged is what gets written."""
    stored, _truth, rng = _mixed_representative_grain()
    q_seed = _unit(rng.normal(size=4))
    out = rigid_grain_quats(stored, q_click=stored[2], q_seed=q_seed,
                            point_group=PG)
    assert abs(float(np.dot(out[2], q_seed))) == pytest.approx(1.0, abs=1e-12)


def test_the_real_intra_grain_rotation_field_survives_the_snap():
    """Snapping must move the REPRESENTATIVE, never the orientation: a grain
    with a genuine 1 deg spread must still have it afterwards -- neither
    flattened to 0 nor inflated by a symmetry operator."""
    stored, truth, rng = _mixed_representative_grain(spread_deg=1.0)
    q_seed = _unit(rng.normal(size=4))
    out = rigid_grain_quats(stored, q_click=stored[0], q_seed=q_seed,
                            point_group=PG)
    for i in range(1, len(out)):
        before = _angle_deg(truth[0], truth[i])
        after = _angle_deg(out[0], out[i])
        assert after == pytest.approx(before, abs=1e-6), (
            f"pixel {i}: intra-grain misorientation {before:.4f} -> {after:.4f}")
    assert _angle_deg(out[0], out[-1]) == pytest.approx(1.0, abs=1e-6)


def test_without_a_point_group_the_old_behaviour_is_untouched():
    """The parameter is opt-in: existing callers that cannot name a symmetry
    keep exactly the arithmetic they had (and its documented limitation)."""
    stored, _truth, rng = _mixed_representative_grain()
    q_seed = _unit(rng.normal(size=4))
    from backend.spherical_gpu.pseudosym import _qconj
    corr = _qmul((q_seed / np.linalg.norm(q_seed)).reshape(1, 4),
                 _qconj(stored[0].reshape(1, 4)))[0]
    naive = _qmul(np.broadcast_to(corr, stored.shape), stored)
    naive /= np.linalg.norm(naive, axis=-1, keepdims=True)
    out = rigid_grain_quats(stored, q_click=stored[0], q_seed=q_seed)
    assert np.allclose(np.abs(np.einsum("ij,ij->i", out, naive)), 1.0, atol=1e-12)


def test_an_unknown_point_group_is_refused_rather_than_silently_ignored():
    stored, _truth, rng = _mixed_representative_grain()
    with pytest.raises(Exception):
        rigid_grain_quats(stored, q_click=stored[0],
                          q_seed=_unit(rng.normal(size=4)),
                          point_group="not-a-point-group")
