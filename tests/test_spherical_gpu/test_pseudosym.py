"""Tests for the pseudo-symmetry resolution layer (backend/spherical_gpu/pseudosym.py).

The spherical SHT correlation lands on a wrong pseudo-symmetric variant for
cubic-approximant intermetallics (m-3, 23). This layer expands the cc peaks by the
pseudo-symmetry coset and selects the variant consistent with a rough Hough
orientation. Pure-function tests here; end-to-end on SampleB is a separate script.
"""
from __future__ import annotations
import numpy as np
import pytest


def q_axis_angle(axis, deg):
    """Unit quaternion (w,x,y,z) for a rotation of `deg` about `axis`."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    a = np.radians(deg) / 2.0
    return np.array([np.cos(a), *(np.sin(a) * axis)])


# ----------------------------------------------------------------------
# disorientation_deg
# ----------------------------------------------------------------------
def test_disorientation_identical_is_zero():
    from backend.spherical_gpu.pseudosym import disorientation_deg
    q = q_axis_angle([1, 2, 3], 47.0)
    assert disorientation_deg(q, q, "m-3") < 1e-3


def test_disorientation_symmetry_equivalent_is_zero():
    # 90 deg about z IS a symmetry operation of m-3m -> disorientation 0
    from backend.spherical_gpu.pseudosym import disorientation_deg
    q = q_axis_angle([0, 0, 1], 90.0)
    assert disorientation_deg(np.array([1.0, 0, 0, 0]), q, "m-3m") < 1e-3


def test_disorientation_small_angle_under_m3():
    # 10 deg about z: under m-3 the nearest symmetry op is identity -> 10 deg
    from backend.spherical_gpu.pseudosym import disorientation_deg
    q = q_axis_angle([0, 0, 1], 10.0)
    assert abs(disorientation_deg(np.array([1.0, 0, 0, 0]), q, "m-3") - 10.0) < 1e-2


def test_disorientation_batched():
    # (N,4) input -> (N,) output
    from backend.spherical_gpu.pseudosym import disorientation_deg
    qs = np.stack([q_axis_angle([0, 0, 1], d) for d in (0.0, 5.0, 10.0)])
    out = disorientation_deg(qs, np.array([1.0, 0, 0, 0]), "m-3")
    assert out.shape == (3,)
    assert np.allclose(out, [0.0, 5.0, 10.0], atol=1e-2)


# ----------------------------------------------------------------------
# same_orientation_angle_deg (fast one-sided dedup metric)
# ----------------------------------------------------------------------
def test_same_orientation_angle_small_rotation():
    # 10 deg about z is NOT a symmetry of m-3 -> the orientations differ by 10 deg
    from backend.spherical_gpu.pseudosym import same_orientation_angle_deg
    qs = np.stack([q_axis_angle([0, 0, 1], d) for d in (0.0, 4.0, 10.0)])
    out = same_orientation_angle_deg(qs, np.array([1.0, 0, 0, 0]), "m-3")
    assert out.shape == (3,)
    assert np.allclose(out, [0.0, 4.0, 10.0], atol=1e-2)


def test_same_orientation_angle_zero_for_symmetry_equivalent():
    # 180 deg about z IS a 2-fold symmetry op of m-3 -> SAME orientation (0 deg)
    from backend.spherical_gpu.pseudosym import same_orientation_angle_deg
    q = q_axis_angle([0, 0, 1], 180.0)
    out = same_orientation_angle_deg(np.atleast_2d(q), np.array([1.0, 0, 0, 0]), "m-3")
    assert out[0] < 1e-2


def test_same_orientation_angle_pseudosymmetry_distinction():
    # 90 deg about z is a symmetry of m-3m but NOT of m-3 (Th has no 4-fold axis).
    # This is exactly the pseudo-symmetry the resolver must handle.
    from backend.spherical_gpu.pseudosym import same_orientation_angle_deg
    q = np.atleast_2d(q_axis_angle([0, 0, 1], 90.0))
    ident = np.array([1.0, 0, 0, 0])
    assert same_orientation_angle_deg(q, ident, "m-3m")[0] < 1e-2     # is a symmetry
    assert abs(same_orientation_angle_deg(q, ident, "m-3")[0] - 90.0) < 1e-2  # is not


# ----------------------------------------------------------------------
# coset_operators + is_pseudosymmetric
# ----------------------------------------------------------------------
def test_coset_m3_has_24_ops():
    from backend.spherical_gpu.pseudosym import coset_operators
    coset = coset_operators("m-3")          # m-3m \ m-3
    assert coset.shape == (24, 4)


def test_coset_ops_are_genuine_pseudo_operators():
    # every coset op is NOT a symmetry of m-3 (it changes the orientation)
    from backend.spherical_gpu.pseudosym import coset_operators, disorientation_deg
    ident = np.array([1.0, 0, 0, 0])
    for s in coset_operators("m-3"):
        assert disorientation_deg(s, ident, "m-3") > 1.0


def test_is_pseudosymmetric():
    from backend.spherical_gpu.pseudosym import is_pseudosymmetric
    assert is_pseudosymmetric("m-3") is True
    assert is_pseudosymmetric("23") is True
    assert is_pseudosymmetric("m-3m") is False
    assert is_pseudosymmetric("4/mmm") is False


# ----------------------------------------------------------------------
# spherical_unreliable — the gate for the AUTO Hough substitution.
#
# Until 2026-09-22 it fired for every z_rot==2 master (m-3/23, -43m, mmm/222/mm2).
# Measured after the decode fix (a4b7710c + 13dd72f4), raw spherical against the
# truth on 48 synthetic patterns per master: m-3 0.17 deg / 100 % <= 2 deg,
# mmm 0.26 / 100 %, but -43m 58 % with the rest exactly 90 deg off; on 81 real
# S-phase patterns (7050) the sphere renders 0.711 vs Hough 0.703 and sits
# 0.33 deg from it. So only -43m (and a z_rot==2 master of unknown class) still
# takes Hough; see tasks/retire-hough/.
# ----------------------------------------------------------------------
def test_zrot2_pointgroup_is_the_fact_not_the_gate():
    from backend.spherical_gpu.pseudosym import zrot2_pointgroup
    for pg in ("mmm", "222", "mm2", "m-3", "23", "-43m", " mmm "):
        assert zrot2_pointgroup(pg) is True, pg
    for pg in ("m-3m", "432", "4/mmm", "6/mmm", "-6m2", "2/m", "-1", "3", "-3m"):
        assert zrot2_pointgroup(pg) is False, pg


def test_spherical_unreliable_pointgroup_is_only_minus43m_now():
    from backend.spherical_gpu.pseudosym import spherical_unreliable_pointgroup
    assert spherical_unreliable_pointgroup("-43m") is True
    assert spherical_unreliable_pointgroup(" -43m ") is True
    for pg in ("mmm", "222", "mm2", "m-3", "23",           # the decode fix covers these
               "m-3m", "432", "4/mmm", "6/mmm", "2/m", "-1"):
        assert spherical_unreliable_pointgroup(pg) is False, pg


def test_spherical_unreliable_gates_on_zrot_then_on_the_class():
    from backend.spherical_gpu.pseudosym import spherical_unreliable
    # not z_rot 2 -> never, whatever the name says
    assert spherical_unreliable(z_rot=4, point_group="-43m") is False
    assert spherical_unreliable(z_rot=1, point_group="m-3") is False
    # z_rot 2: the class decides
    assert spherical_unreliable(z_rot=2, point_group="m-3") is False
    assert spherical_unreliable(z_rot=2, point_group="mmm") is False
    assert spherical_unreliable(z_rot=2, point_group="-43m") is True
    # z_rot 2 with no usable class -> keep the safety net (fail closed)
    assert spherical_unreliable(z_rot=2, point_group=None) is True
    assert spherical_unreliable(z_rot=2, point_group="anything") is True
    # z_rot unavailable: the name decides
    assert spherical_unreliable(z_rot=None, point_group="-43m") is True
    assert spherical_unreliable(z_rot=None, point_group="mmm") is False
    assert spherical_unreliable(z_rot=None, point_group="m-3m") is False
    assert spherical_unreliable(z_rot="2", point_group="-43m") is True
    assert spherical_unreliable(z_rot="bad", point_group="-43m") is True


def test_the_switch_restores_the_old_gate(monkeypatch):
    """ORIENTA_HOUGH_ANCHOR_ALL_ZROT2=1 puts every z_rot==2 class back on Hough —
    the way back, and the A/B the change was measured with."""
    from backend.spherical_gpu.pseudosym import (
        spherical_unreliable, spherical_unreliable_pointgroup)
    monkeypatch.setenv("ORIENTA_HOUGH_ANCHOR_ALL_ZROT2", "1")
    for pg in ("m-3", "23", "mmm", "222", "mm2", "-43m"):
        assert spherical_unreliable(z_rot=2, point_group=pg) is True, pg
        assert spherical_unreliable_pointgroup(pg) is True, pg
    assert spherical_unreliable(z_rot=4, point_group="m-3m") is False
    monkeypatch.setenv("ORIENTA_HOUGH_ANCHOR_ALL_ZROT2", "0")
    assert spherical_unreliable(z_rot=2, point_group="m-3") is False


def test_pseudosymmetric_classes_are_no_longer_hough_anchored():
    # is_pseudosymmetric (m-3, 23) still names the cubic approximants for the
    # variant tools; it no longer implies the Hough substitution.
    from backend.spherical_gpu.pseudosym import is_pseudosymmetric, spherical_unreliable
    for pg in ("m-3", "23"):
        assert is_pseudosymmetric(pg) and not spherical_unreliable(z_rot=2, point_group=pg)


# ----------------------------------------------------------------------
# pseudosym_variant_quats — UNIVERSAL candidate generator (any point group)
# ----------------------------------------------------------------------
def test_variant_quats_cubic_approximant_has_multiple_distinct():
    # m-3 sits in m-3m -> the user can flip among several distinct pseudo-variants
    from backend.spherical_gpu.pseudosym import pseudosym_variant_quats, disorientation_deg
    q = q_axis_angle([1, 2, 3], 30.0)
    V = pseudosym_variant_quats(q, "m-3")
    assert V.shape[1] == 4
    assert V.shape[0] >= 2                      # more than the original
    assert np.allclose(V[0], q)                 # variant 0 is the current orientation
    # every variant is distinct under the TRUE point group
    for i in range(1, V.shape[0]):
        assert disorientation_deg(V[i], q, "m-3") > 3.0


def test_variant_quats_triclinic_has_no_pseudovariants():
    # -1 is its own holohedry -> nothing to flip to
    from backend.spherical_gpu.pseudosym import pseudosym_variant_quats
    q = q_axis_angle([0, 0, 1], 17.0)
    V = pseudosym_variant_quats(q, "-1")
    assert V.shape == (1, 4)
    assert np.allclose(V[0], q)


def test_variant_quats_trigonal_uses_hexagonal_pseudo_holohedry():
    # trigonal -3m -> hexagonal 6/mmm pseudo-symmetry (the 60°-about-c quartz case)
    from backend.spherical_gpu.pseudosym import pseudosym_variant_quats, pseudosym_holohedry
    assert pseudosym_holohedry("-3m") == "6/mmm"
    q = q_axis_angle([1, 1, 4], 25.0)
    V = pseudosym_variant_quats(q, "-3m")
    assert V.shape[0] >= 2                       # has pseudo-variants to flip between


def test_pseudosymmetric_classes_have_a_nonempty_coset():
    # Every class we declare pseudo-symmetric MUST yield a non-empty coset —
    # otherwise resolve_variant would crash (caught -> silent no-op), advertising
    # a resolution that never happens. 432/-43m have a trivial coset under the
    # centrosymmetric holohedry formulation, so they must NOT be declared.
    from backend.spherical_gpu.pseudosym import (
        is_pseudosymmetric, coset_operators, _PSEUDO_HOLOHEDRY)
    for pg in _PSEUDO_HOLOHEDRY:
        assert is_pseudosymmetric(pg) is True
        coset = coset_operators(pg)
        assert coset.ndim == 2 and coset.shape[1] == 4  # always (K, 4)
        assert coset.shape[0] > 0, f"{pg} has an empty coset"
    # 432 and -43m specifically must be absent (empty coset -> would crash)
    assert is_pseudosymmetric("432") is False
    assert is_pseudosymmetric("-43m") is False


def test_resolve_variant_survives_empty_coset():
    # Defensive: if a coset is ever empty, resolve_variant must return the peak
    # itself (candidates = {P}) rather than crash on coset[None,:,:].
    from backend.spherical_gpu import pseudosym as ps
    T = q_axis_angle([1, 1, 1], 30.0)
    H = q_axis_angle([1, 1, 1], 31.0)
    empty = np.zeros((0, 4))
    saved = ps.coset_operators
    ps.coset_operators = lambda pg: empty
    try:
        resolved, candidates, d = ps.resolve_variant(np.atleast_2d(T), H, "m-3")
    finally:
        ps.coset_operators = saved
    assert candidates.shape == (1, 4)
    assert ps.disorientation_deg(resolved, T, "m-3") < 1e-6


# ----------------------------------------------------------------------
# resolve_variant
# ----------------------------------------------------------------------
def _qmul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def test_resolve_variant_recovers_true_from_pseudo_variant():
    # Spherical lands on a pseudo-variant P = s*T (s a coset op) of the true T.
    # Hough is a rough (3 deg off) estimate of T. The resolver must return ~T.
    from backend.spherical_gpu.pseudosym import resolve_variant, coset_operators, disorientation_deg
    T = q_axis_angle([1, 1, 1], 30.0)
    s = coset_operators("m-3")[5]
    P = _qmul(s, T)                                  # spherical's wrong pseudo-variant
    H = _qmul(q_axis_angle([0, 1, 0], 3.0), T)       # Hough ~ true (3 deg)
    resolved, candidates, d = resolve_variant(np.atleast_2d(P), H, "m-3")
    assert disorientation_deg(resolved, T, "m-3") < 5.0
    assert d < 5.0
    assert candidates.shape[1] == 4


def test_resolve_variant_keeps_correct_when_already_right():
    # If a peak already matches Hough, the resolver returns it unchanged-ish.
    from backend.spherical_gpu.pseudosym import resolve_variant, disorientation_deg
    T = q_axis_angle([0, 0, 1], 40.0)
    H = _qmul(q_axis_angle([1, 0, 0], 2.0), T)
    resolved, _c, d = resolve_variant(np.atleast_2d(T), H, "m-3")
    assert disorientation_deg(resolved, T, "m-3") < 3.0
    assert d < 3.0
