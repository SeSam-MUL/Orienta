"""Orientation arithmetic used by grain reconstruction, GOS and KAM.

The reference for every angle is orix (an independent implementation); the
left-side symmetry trap is pinned explicitly because the old grain mean fell
into it.
"""
import numpy as np
import pytest
from orix.quaternion import Misorientation, Orientation, Rotation
from orix.quaternion.symmetry import C2h, D4h, D6h, Oh, Th

from analysis.orientation_math import (
    disorientation_angle, label_means, label_spread, nearest_equivalent,
    proper_ops, qconj, qmul,
)

GROUPS = [Oh, Th, D4h, D6h, C2h]


def _orix_angle(qa, qb, sym):
    oa = Orientation(qa, symmetry=sym)
    ob = Orientation(qb, symmetry=sym)
    m = Misorientation(ob * ~oa, symmetry=(sym, sym)).reduce()
    return np.asarray(m.angle).ravel()


def test_qmul_is_orix_product():
    a, b = Rotation.random(50), Rotation.random(50)
    np.testing.assert_allclose(np.abs(qmul(a.data, b.data)), np.abs((a * b).data), atol=1e-12)


@pytest.mark.parametrize("sym", GROUPS, ids=lambda s: s.name)
def test_disorientation_angle_equals_orix(sym):
    qa, qb = Rotation.random(400).data, Rotation.random(400).data
    ops = proper_ops(sym)
    np.testing.assert_allclose(disorientation_angle(qa, qb, ops), _orix_angle(qa, qb, sym), atol=1e-8)


def test_disorientation_angle_chunking_is_exact():
    ops = proper_ops(Oh)
    qa, qb = Rotation.random(50).data, Rotation.random(50).data
    qb[3] = np.nan
    whole = disorientation_angle(qa, qb, ops)
    np.testing.assert_array_equal(disorientation_angle(qa, qb, ops, chunk=7), whole)
    assert np.isnan(whole[3])


def test_symmetry_acts_from_the_left():
    ops = proper_ops(Oh)
    q = Rotation.random(30).data
    for s in ops:
        left = qmul(np.broadcast_to(s, q.shape), q)
        right = qmul(q, np.broadcast_to(s, q.shape))
        # arccos near 1 has a float floor of ~3e-8 rad (1.7e-6 deg), hence 1e-4 deg
        assert np.degrees(disorientation_angle(q, left, ops)).max() < 1e-4
    # a right product is a different orientation for at least one operator
    worst = max(np.degrees(disorientation_angle(q, qmul(q, np.broadcast_to(s, q.shape)), ops)).max() for s in ops)
    assert worst > 10


def test_nan_in_nan_out():
    ops = proper_ops(Oh)
    qa = Rotation.random(3).data
    qb = qa.copy()
    qb[1] = np.nan
    ang = disorientation_angle(qa, qb, ops)
    assert np.isnan(ang[1]) and np.all(np.isfinite(ang[[0, 2]]))


def test_nearest_equivalent_is_equivalent_and_closest():
    ops = proper_ops(Oh)
    ref = Rotation.random(200).data
    q = Rotation.random(200).data
    eq = nearest_equivalent(q, ref, ops)
    assert np.degrees(disorientation_angle(q, eq, ops)).max() < 1e-4          # same orientation (arccos floor, see above)
    plain = 2 * np.degrees(np.arccos(np.clip(np.abs(np.sum(eq * ref, axis=1)), 0, 1)))
    reduced = np.degrees(disorientation_angle(q, ref, ops))
    np.testing.assert_allclose(plain, reduced, atol=1e-6)                     # and the closest one
    assert np.all(np.sum(eq * ref, axis=1) >= 0)                              # sign aligned


def _noisy_around(center, n, deg, rng):
    axes = rng.normal(size=(n, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    ang = np.deg2rad(rng.normal(0, deg, n))
    small = np.column_stack([np.cos(ang / 2), np.sin(ang / 2)[:, None] * axes])
    return qmul(small, np.broadcast_to(center, (n, 4)))


def test_label_means_on_a_symmetry_branch():
    """A tight grain whose pixels come out on different cubic branches (phi2 near 45 deg)."""
    rng = np.random.default_rng(3)
    ops = proper_ops(Oh)
    center = Rotation.from_axes_angles([0, 0, 1], np.deg2rad(45.0)).data[0]
    q = _noisy_around(center, 300, 0.3, rng)
    # scatter the pixels over symmetry branches: the old mean broke on exactly this
    picks = rng.integers(0, len(ops), len(q))
    q = qmul(ops[picks], q)
    labels = np.ones(len(q), int)
    means = label_means(q, labels, 1, ops)
    assert np.degrees(disorientation_angle(means, center[None], ops))[0] < 0.1
    gos = np.degrees(label_spread(q, labels, means, ops))[0]
    assert gos < 0.5


def test_label_means_empty_label_is_nan():
    ops = proper_ops(Oh)
    q = Rotation.random(4).data
    means = label_means(q, np.array([1, 1, 3, 3]), 3, ops)
    assert np.isnan(means[1]).all() and np.isfinite(means[[0, 2]]).all()


def test_label_spread_never_exceeds_the_largest_pair_angle():
    rng = np.random.default_rng(5)
    ops = proper_ops(Oh)
    center = Rotation.random(1).data[0]
    q = _noisy_around(center, 50, 2.0, rng)
    labels = np.ones(50, int)
    gos = label_spread(q, labels, label_means(q, labels, 1, ops), ops)[0]
    pair_max = max(disorientation_angle(np.broadcast_to(q[i], q.shape), q, ops).max() for i in range(50))
    assert gos <= pair_max
