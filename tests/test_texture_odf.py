"""Texture index and entropy of the 5 degree kernel density (analysis/texture_odf.py).

The exact values come from the mathematics of the de la Vallee Poussin kernel, computed here with scipy and
the gamma function, independent of the module under test:
  int psi^2 = C^2 (2/pi) Gamma(2k + 1/2) Gamma(3/2) / Gamma(2k + 2),  one orientation: J = int psi^2 / |S|,
  int psi ln psi = (2/pi) int psi (ln C + 2k ln cos(w/2)) sin^2(w/2) dw, one orientation: S = ln|S| - int psi ln psi,
  n far-apart orientations: J = J1 / n, S = ln(|S| n) - int psi ln psi.
Overlapping orientations are checked against the pair sum  J = 1/(N^2 |S|) sum_{i,j,s} K(w(g_i, s g_j))  with the
self-convolution K = psi * psi of the kernel from its character series.
"""
from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate
from scipy.special import gammaln

from analysis import texture_odf as T
from analysis.orientation_math import proper_ops, qmul

HALFWIDTH = 5.0
# Tolerances on the exact synthetic values. Measured worst case over every proper point group, 60 positions each
# (identity, random, within 0.02 of a rotation by pi, on and 3 % beside every fundamental-zone face):
# relative J 1.9e-4 and absolute S 1.9e-4, both for 432 (24 operators); all other groups are below 1.6e-4 / 1e-6.
# A wider sweep (1500 positions of 432 near rotations by pi) found J 3.1e-4 and S 6.3e-4 there, a
# quadrature effect of the 4 degree nodes (8e-8 at 2 degrees). The tolerances keep a margin over that and stay well
# below the errors of the angle-based partition weight this module had before (J 3e-3, S 1.5e-2 for 3, 4, 6).
J_TOL = 1e-3          # relative
S_TOL = 1.5e-3        # absolute
# the same quantities against MTEX on the real maps (MTEX's own band limit and entropy quadrature are the limit there)
MTEX_J_TOL = 1e-3
# int f = 1 on the quadrature (measured worst case 2.3e-6 over the sets of the mass test, tolerance 4 x that; a kernel
# truncated at half its cut-off loses about 2e-4) and the pair-sum comparison of overlapping orientations (worst 1.7e-6)
MASS_TOL = 1e-5
PAIR_TOL = 6e-6       # relative


def _group(name):
    from orix.quaternion import symmetry as sym
    return proper_ops(getattr(sym, name))


OH = _group("Oh")


# ------------------------------------------------------------------------------------------------------------------
# independent reference values
# ------------------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _kernel(h_deg=HALFWIDTH):
    h = math.radians(h_deg)
    kappa = math.log(0.5) / (2.0 * math.log(math.cos(h / 2.0)))
    log_inv_c = math.log(2.0 / math.pi) + gammaln(kappa + 0.5) + gammaln(1.5) - gammaln(kappa + 2.0)
    return kappa, -log_inv_c


@lru_cache(maxsize=None)
def _psi_integrals(h_deg=HALFWIDTH):
    """``(int psi^2, int psi ln psi)`` over SO(3) with Haar measure 1."""
    kappa, log_c = _kernel(h_deg)
    int_psi2 = math.exp(2 * log_c + math.log(2.0 / math.pi) + gammaln(2 * kappa + 0.5) + gammaln(1.5)
                        - gammaln(2 * kappa + 2.0))

    def integrand(w):
        lp = log_c + 2.0 * kappa * math.log(math.cos(w / 2.0))
        return (2.0 / math.pi) * math.exp(lp) * lp * math.sin(w / 2.0) ** 2

    val, _ = integrate.quad(integrand, 0.0, math.pi, epsabs=1e-13, epsrel=1e-13, limit=400)
    return int_psi2, val


def exact_one(n_sym, n=1, h_deg=HALFWIDTH):
    """Exact J and S of ``n`` orientations that are far apart, for a group of order ``n_sym``."""
    i2, ilnp = _psi_integrals(h_deg)
    return i2 / (n_sym * n), math.log(n_sym * n) - ilnp


@lru_cache(maxsize=None)
def _k_table(h_deg=HALFWIDTH, n_points=20001):
    """K = psi * psi tabulated on [0, pi] from the character series K(w) = sum a_l^2/(2l+1) chi_l(w)."""
    kappa, log_c = _kernel(h_deg)
    x, wx = np.polynomial.legendre.leggauss(3000)
    w = (x + 1.0) * math.pi / 2.0
    wq = wx * math.pi / 2.0
    psi = np.exp(log_c + 2.0 * kappa * np.log(np.cos(w / 2.0)))
    meas = (2.0 / math.pi) * wq * np.sin(w / 2.0) ** 2
    grid = np.linspace(0.0, math.pi, n_points)
    k = np.zeros(n_points)
    total = None
    for l in range(0, 400):
        chi = np.sin((2 * l + 1) * w / 2.0) / np.sin(w / 2.0)
        a_l = np.sum(psi * chi * meas)
        term = a_l ** 2 / (2 * l + 1)
        if total is None:
            total = a_l ** 2
        k[1:] += term * np.sin((2 * l + 1) * grid[1:] / 2.0) / np.sin(grid[1:] / 2.0)
        k[0] += term * (2 * l + 1)
        if l > 20 and a_l ** 2 < 1e-13 * total:
            break
    return grid, k


def pair_sum_j(quats, ops):
    """J = 1/(N^2 |S|) sum_{i,j,s} K(w(q_i, s q_j)), the independent reference for overlapping orientations."""
    grid, k = _k_table()
    n = len(quats)
    total = 0.0
    for s in ops:
        copies = qmul(s[None, :], quats)
        for i0 in range(0, n, 100):
            d = np.abs(quats[i0:i0 + 100] @ copies.T)
            ang = 2.0 * np.arccos(np.clip(d, 0.0, 1.0))
            total += np.interp(ang, grid, k).sum()
    return total / (n * n * len(ops))


# ------------------------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------------------------
def axis_angle(axis, deg):
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    h = math.radians(deg) / 2.0
    return np.concatenate([[math.cos(h)], math.sin(h) * a])


def rodrigues(r):
    r = np.asarray(r, dtype=float)
    q = np.concatenate([[1.0], r])
    return q / np.linalg.norm(q)


def euler_bunge(phi1, big_phi, phi2):
    """Bunge ZXZ as an (unsymmetrised) unit quaternion; any fixed convention does for these tests."""
    a, b, c = (math.radians(v) / 2.0 for v in (phi1, big_phi, phi2))
    qz1 = np.array([math.cos(a), 0, 0, math.sin(a)])
    qx = np.array([math.cos(b), math.sin(b), 0, 0])
    qz2 = np.array([math.cos(c), 0, 0, math.sin(c)])
    return qmul(qmul(qz1, qx), qz2)


def random_rotations(rng, n):
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def jitter(rng, centre, n, spread_deg):
    """``n`` orientations scattered around ``centre`` by rotations of a normal angle with spread ``spread_deg``."""
    vec = rng.normal(size=(n, 3)) * math.radians(spread_deg) / 2.0
    ang = np.linalg.norm(vec, axis=1)
    small = np.concatenate([np.cos(ang)[:, None], vec * (np.sinc(ang / math.pi))[:, None]], axis=1)
    return qmul(small, centre[None, :])


def assert_exact(quats, ops, want_j, want_s, **kw):
    j, s = T.kde_texture_index_and_entropy(quats, ops, HALFWIDTH, **kw)
    assert j == pytest.approx(want_j, rel=J_TOL)
    assert s == pytest.approx(want_s, abs=S_TOL)


# ------------------------------------------------------------------------------------------------------------------
# kernel constants
# ------------------------------------------------------------------------------------------------------------------
def test_kernel_constants_match_the_derivation():
    kappa, log_c, cut = T.kernel_constants(5.0)
    assert kappa == pytest.approx(363.959328, abs=1e-5)
    assert math.exp(log_c) == pytest.approx(12345.11068, rel=1e-8)
    # the cut-off is where the kernel has fallen to 1e-9 of its peak
    assert math.cos(cut / 2.0) ** (2 * kappa) == pytest.approx(1e-9, rel=1e-6)
    assert 2 * cut > math.radians(50)            # two copies of a pixel (>= 60 degrees apart) never both reach a node


def test_pair_sum_reference_is_tied_to_the_closed_form():
    """K(0) = int psi^2: the character series of the self-convolution agrees with the gamma-function value."""
    grid, k = _k_table()
    assert grid[0] == 0.0
    assert k[0] == pytest.approx(_psi_integrals()[0], rel=1e-6)
    assert np.all(np.diff(k[:2000]) < 0.0)                     # K falls off monotonically near its peak


def test_exact_reference_constants():
    i2, ilnp = _psi_integrals()
    assert i2 == pytest.approx(4371.394191, rel=1e-8)
    assert ilnp == pytest.approx(7.924098843, abs=1e-8)
    j1, s1 = exact_one(24)
    assert j1 == pytest.approx(182.14142464, rel=1e-8)
    assert s1 == pytest.approx(-4.746045013, abs=1e-8)


# ------------------------------------------------------------------------------------------------------------------
# exact synthetic values
# ------------------------------------------------------------------------------------------------------------------
POSITIONS = {
    "identity": np.array([1.0, 0.0, 0.0, 0.0]),
    "generic": euler_bunge(10, 20, 30),
    "near_face_100": axis_angle([1, 0, 0], 44.7),               # the fundamental-zone face of the 90 degree rotation
    "near_face_111": axis_angle([1, 1, 1], 59.7),               # the face of the 120 degree rotation (3-fold axis)
    "near_vertex": rodrigues([math.sqrt(2) - 1, math.sqrt(2) - 1, 3 - 2 * math.sqrt(2)]),
    "near_4fold": axis_angle([1, 0, 0], 89.0),                  # a 4-fold rotation is itself a symmetry element
}


@pytest.mark.parametrize("name", list(POSITIONS))
def test_one_orientation_exact(name):
    want_j, want_s = exact_one(24)
    assert_exact(POSITIONS[name][None, :], OH, want_j, want_s)


@pytest.mark.parametrize("n, h", [(2, 5.0), (3, 5.0), (5, 2.5)])
def test_far_apart_orientations_exact(n, h):
    # n orientations whose symmetric copies are all more than 2 x cut-off apart (a cubic crystal has at most 62.8
    # degrees between two orientations, so five of them need the narrower kernel)
    gap = 2.0 * T.kernel_constants(h)[2] * 1.02
    cands = [POSITIONS["generic"], euler_bunge(100, 50, 5), POSITIONS["near_vertex"], euler_bunge(200, 70, 20),
             euler_bunge(310, 35, 80), POSITIONS["near_face_100"], euler_bunge(250, 100, 140)]
    rng = np.random.default_rng(3)
    chosen = []
    for c in cands + list(random_rotations(rng, 2000)):
        if all(_min_dis(c, k) > gap for k in chosen):
            chosen.append(c)
        if len(chosen) == n:
            break
    assert len(chosen) == n
    want_j, want_s = exact_one(24, n, h)
    j, s = T.kde_texture_index_and_entropy(np.array(chosen), OH, h)
    assert j == pytest.approx(want_j, rel=J_TOL)
    assert s == pytest.approx(want_s, abs=S_TOL)


def _min_dis(a, b):
    from analysis.orientation_math import disorientation_angle
    return float(disorientation_angle(a[None], b[None], OH)[0])


PROPER_GROUPS = ["C1", "C2", "D2", "C3", "D3", "C4", "D4", "C6", "D6", "T", "O"]    # 1 2 222 3 32 4 422 6 622 23 432


def _positions_for(ops, seed=0):
    """Identity, generic, random, near a rotation by pi (|q0| < 0.02) and on / beside the zone faces of ``ops``."""
    rng = np.random.default_rng(seed)
    e = np.array([1.0, 0.0, 0.0, 0.0])
    pos = [("identity", e), ("generic", POSITIONS["generic"])]
    pos += [("random", q) for q in random_rotations(rng, 4)]
    for _ in range(3):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        g0 = rng.uniform(-0.02, 0.02)
        pos.append(("pi", np.concatenate([[g0], v * math.sqrt(1.0 - g0 * g0)])))
    faces = [s for s in ops if abs(s[0]) < 0.9999]       # the face towards s lies half way between e and s
    for s in faces[:4]:
        for f in (0.97, 1.0, 1.03):
            q = e + f * s
            pos.append(("face", q / np.linalg.norm(q)))
    return pos


@pytest.mark.parametrize("group", PROPER_GROUPS)
def test_one_orientation_exact_every_proper_group(group):
    """Exact J = int psi^2 / |S|, S = ln|S| - int psi ln psi for every proper group a phase can have."""
    ops = _group(group)
    want_j, want_s = exact_one(len(ops))
    for kind, q in _positions_for(ops):
        j, s = T.kde_texture_index_and_entropy(q[None, :], ops, HALFWIDTH)
        assert j == pytest.approx(want_j, rel=J_TOL), (group, kind, q)
        assert s == pytest.approx(want_s, abs=S_TOL), (group, kind, q)


def test_overlapping_clusters_match_the_pair_sum():
    rng = np.random.default_rng(7)
    parts = [jitter(rng, POSITIONS["generic"], 100, 3.0),
             jitter(rng, POSITIONS["near_vertex"], 100, 5.0),
             jitter(rng, POSITIONS["near_face_100"], 100, 8.0)]
    q = np.concatenate(parts)
    want = pair_sum_j(q, OH)
    j, s = T.kde_texture_index_and_entropy(q, OH, HALFWIDTH)
    assert j == pytest.approx(want, rel=PAIR_TOL)
    # the density integrates to 1 on the same nodes
    assert T.kde_mass(q, OH, HALFWIDTH) == pytest.approx(1.0, abs=MASS_TOL)
    assert s < 0.0


def test_uniform_orientations_give_the_uniform_values():
    rng = np.random.default_rng(0)
    q = random_rotations(rng, 20000)
    j, s = T.kde_texture_index_and_entropy(q, OH, HALFWIDTH)
    # MTEX gives 1.009620 / -0.004796 for this set; the density integrates to 1 on the same nodes
    assert j == pytest.approx(1.00962, rel=J_TOL)
    assert s == pytest.approx(-0.004796, abs=2e-3)
    assert T.kde_mass(q, OH, HALFWIDTH) == pytest.approx(1.0, abs=1e-4)


# ------------------------------------------------------------------------------------------------------------------
# invariances
# ------------------------------------------------------------------------------------------------------------------
def _cluster_set():
    rng = np.random.default_rng(11)
    return np.concatenate([jitter(rng, POSITIONS["generic"], 60, 4.0),
                           jitter(rng, POSITIONS["near_vertex"], 60, 6.0),
                           jitter(rng, euler_bunge(200, 70, 20), 60, 5.0)])


def test_specimen_rotation_changes_nothing():
    q = _cluster_set()
    j0, s0 = T.kde_texture_index_and_entropy(q, OH, HALFWIDTH)
    rng = np.random.default_rng(5)
    for _ in range(3):
        r = random_rotations(rng, 1)[0]
        # orix: the crystal symmetry acts from the left, a rotation of the specimen from the right
        j, s = T.kde_texture_index_and_entropy(qmul(q, r[None, :]), OH, HALFWIDTH)
        assert j == pytest.approx(j0, rel=J_TOL)
        assert s == pytest.approx(s0, abs=S_TOL)


def test_symmetric_equivalents_change_nothing():
    q = _cluster_set()
    j0, s0 = T.kde_texture_index_and_entropy(q, OH, HALFWIDTH)
    rng = np.random.default_rng(6)
    pick = OH[rng.integers(0, len(OH), size=len(q))]
    j, s = T.kde_texture_index_and_entropy(qmul(pick, q), OH, HALFWIDTH)
    assert j == pytest.approx(j0, rel=J_TOL)
    assert s == pytest.approx(s0, abs=S_TOL)
    # the sign of a quaternion is not a different rotation
    flip = np.where(rng.random(len(q))[:, None] < 0.5, -q, q)
    j, s = T.kde_texture_index_and_entropy(flip, OH, HALFWIDTH)
    assert j == pytest.approx(j0, rel=J_TOL)
    assert s == pytest.approx(s0, abs=S_TOL)


def test_duplicating_every_pixel_changes_nothing():
    q = _cluster_set()
    j0, s0 = T.kde_texture_index_and_entropy(q, OH, HALFWIDTH)
    j, s = T.kde_texture_index_and_entropy(np.concatenate([q, q]), OH, HALFWIDTH)
    assert j == pytest.approx(j0, rel=1e-9)
    assert s == pytest.approx(s0, abs=1e-9)


def test_all_pixels_of_one_orientation_give_the_single_orientation_values():
    want_j, want_s = exact_one(24)
    q = np.repeat(POSITIONS["generic"][None, :], 1000, axis=0)
    assert_exact(q, OH, want_j, want_s)


def test_the_launch_lock_keeps_the_workqueue_layer_alive():
    # numba's workqueue threading layer aborts the process when two threads launch a parallel kernel at once; the
    # module-level lock is what prevents that, so the concurrent calls run in a child process on that layer
    import subprocess
    import sys
    import textwrap
    code = textwrap.dedent("""
        import threading
        import numpy as np
        from orix.quaternion.symmetry import Oh
        from analysis import texture_odf as T
        from analysis.orientation_math import proper_ops
        ops = proper_ops(Oh)
        rng = np.random.default_rng(1)
        q = rng.normal(size=(3000, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        out = []
        ts = [threading.Thread(target=lambda: out.append(T.kde_texture_index_and_entropy(q, ops, 5.0)))
              for _ in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert len(out) == 4 and all(o == out[0] for o in out), out
        print("ok")
    """)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, NUMBA_THREADING_LAYER="workqueue")
    r = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), (r.returncode, r.stdout[-500:], r.stderr[-2000:])


def test_quadrature_is_converged_between_resolutions():
    q = _cluster_set()
    a = T.kde_texture_index_and_entropy(q, OH, HALFWIDTH, resolution_deg=4.0)
    b = T.kde_texture_index_and_entropy(q, OH, HALFWIDTH, resolution_deg=2.5)
    assert a[0] == pytest.approx(b[0], rel=2e-4)
    assert a[1] == pytest.approx(b[1], abs=1e-3)


# ------------------------------------------------------------------------------------------------------------------
# input errors
# ------------------------------------------------------------------------------------------------------------------
def test_input_errors_raise():
    q = POSITIONS["generic"][None, :]
    with pytest.raises(ValueError):
        T.kde_texture_index_and_entropy(np.empty((0, 4)), OH)
    with pytest.raises(ValueError):
        T.kde_texture_index_and_entropy(np.array([[np.nan, 0, 0, 0.0]]), OH)
    with pytest.raises(ValueError):
        T.kde_texture_index_and_entropy(np.array([[np.inf, 0, 0, 0.0]]), OH)
    with pytest.raises(ValueError):
        T.kde_texture_index_and_entropy(q, OH, halfwidth_deg=0.0)
    with pytest.raises(ValueError):
        T.kde_texture_index_and_entropy(q, OH, halfwidth_deg=-5.0)
    with pytest.raises(ValueError):
        T.kde_texture_index_and_entropy(q[:, :3], OH)
    with pytest.raises(ValueError):
        T.kde_texture_index_and_entropy(q, np.empty((0, 4)))


def _two_phase_dataset(n_cubic=30, n_hex=25, seed=2):
    """A map whose cubic phase (m-3m) has id 1 and whose hexagonal phase (6/mmm) has id 0, ids in file order 0, 1."""
    from types import SimpleNamespace
    from orix.crystal_map import CrystalMap, PhaseList
    from orix.quaternion import Rotation, symmetry as sym
    rng = np.random.default_rng(seed)
    q_cub = jitter(rng, POSITIONS["generic"], n_cubic, 4.0)
    q_hex = jitter(rng, POSITIONS["near_vertex"], n_hex, 4.0)
    q = np.concatenate([q_hex, q_cub])
    ids = np.concatenate([np.zeros(n_hex, int), np.ones(n_cubic, int)])
    pl = PhaseList(names=["hex", "cubic"], point_groups=[sym.D6h, sym.Oh], ids=[0, 1])
    xmap = CrystalMap(rotations=Rotation(q), phase_id=ids, phase_list=pl,
                      x=np.arange(len(q), dtype=float), y=np.zeros(len(q)))
    return SimpleNamespace(xmap=xmap), q_hex, q_cub


def test_wrapper_uses_the_symmetry_of_the_masked_pixels_phase():
    ds, q_hex, q_cub = _two_phase_dataset()
    ids = np.asarray(ds.xmap.phase_id)
    got_cub = T.texture_index_and_entropy(ds, ids == 1, HALFWIDTH)
    got_hex = T.texture_index_and_entropy(ds, ids == 0, HALFWIDTH)
    assert len(OH) == 24 and len(_group("D6")) == 12
    assert got_cub == pytest.approx(T.kde_texture_index_and_entropy(q_cub, OH, HALFWIDTH), rel=1e-12)
    assert got_hex == pytest.approx(T.kde_texture_index_and_entropy(q_hex, _group("D6"), HALFWIDTH), rel=1e-12)
    # the point group matters: the cubic pixels with the hexagonal group (or the reverse) give other numbers
    assert got_cub[0] != pytest.approx(T.kde_texture_index_and_entropy(q_cub, _group("D6"), HALFWIDTH)[0], rel=1e-2)


def test_wrapper_refuses_a_mask_that_is_not_one_indexed_phase():
    ds, _q_hex, _q_cub = _two_phase_dataset()
    ids = np.asarray(ds.xmap.phase_id)
    with pytest.raises(ValueError, match=r"mixes phases.*\[0, 1\]"):
        T.texture_index_and_entropy(ds, np.ones(len(ids), dtype=bool))
    unindexed = ids.copy()
    unindexed[3] = -1
    ds.xmap._phase_id = unindexed                     # orix keeps the ids in _phase_id
    with pytest.raises(ValueError, match=r"not-indexed.*-1"):
        T.texture_index_and_entropy(ds, unindexed <= 0)
    with pytest.raises(ValueError, match="no pixel"):
        T.texture_index_and_entropy(ds, np.zeros(len(ids), dtype=bool))
    with pytest.raises(ValueError, match="entries"):
        T.texture_index_and_entropy(ds, np.ones(len(ids) - 1, dtype=bool))


def test_wrapper_drops_non_finite_quaternions_of_the_mask():
    ds, _q_hex, q_cub = _two_phase_dataset()
    ids = np.asarray(ds.xmap.phase_id)
    want = T.kde_texture_index_and_entropy(q_cub[1:], OH, HALFWIDTH)
    from types import SimpleNamespace
    data = np.asarray(ds.xmap.rotations.data).copy()
    data[np.flatnonzero(ids == 1)[0]] = np.nan
    ds2 = SimpleNamespace(xmap=SimpleNamespace(rotations=SimpleNamespace(data=data), phase_id=ids,
                                               phases=ds.xmap.phases))
    assert T.texture_index_and_entropy(ds2, ids == 1, HALFWIDTH) == pytest.approx(want, rel=1e-12)


# ------------------------------------------------------------------------------------------------------------------
# truncation of the kernel, node bound, thread safety, read-only cache
# ------------------------------------------------------------------------------------------------------------------
def test_density_integrates_to_one_so_a_truncated_kernel_is_seen():
    """int f = 1 on the quadrature: the neighbour search must reach the whole cut-off (1e-9 of the peak).

    A kernel truncated at half the cut-off (about 5e-3 of its peak) loses about 2e-4 of the mass; with the
    tolerances above the single-orientation tests see that too (run as a mutation: 24 tests fail), this one sees it
    directly and independently of any reference value.
    """
    rng = np.random.default_rng(9)
    for q in (POSITIONS["generic"][None, :], _cluster_set(), jitter(rng, POSITIONS["near_vertex"], 200, 12.0)):
        assert T.kde_mass(q, OH, HALFWIDTH) == pytest.approx(1.0, abs=MASS_TOL)
    assert T.kde_mass(POSITIONS["near_face_100"][None, :], _group("C4"), HALFWIDTH) == pytest.approx(1.0, abs=MASS_TOL)


def test_broad_cluster_matches_the_pair_sum_tightly():
    rng = np.random.default_rng(13)
    q = jitter(rng, POSITIONS["near_face_111"], 250, 14.0)           # wider than the kernel: many overlapping pairs
    assert T.kde_texture_index_and_entropy(q, OH, HALFWIDTH)[0] == pytest.approx(pair_sum_j(q, OH), rel=PAIR_TOL)


def test_too_fine_a_resolution_is_refused_with_the_halfwidth_limit():
    q = POSITIONS["generic"][None, :]
    with pytest.raises(ValueError, match=r"halfwidth of at least 1\.[0-9] degrees"):
        T.kde_texture_index_and_entropy(q, OH, halfwidth_deg=0.5)
    with pytest.raises(ValueError, match="nodes"):
        T.kde_texture_index_and_entropy(q, OH, halfwidth_deg=5.0, resolution_deg=0.5)
    with pytest.raises(ValueError):
        T.kde_texture_index_and_entropy(q, OH, halfwidth_deg=5.0, resolution_deg=0.0)
    # the smallest halfwidth the default resolution allows still works
    assert T._grid_size(0.8 * 2.0) <= T.MAX_NODES


def test_concurrent_calls_agree_and_the_node_cache_is_read_only():
    import threading
    q = _cluster_set()
    want = T.kde_texture_index_and_entropy(q, OH, HALFWIDTH)
    out = [None] * 4

    def run(i):
        out[i] = T.kde_texture_index_and_entropy(q, OH, HALFWIDTH)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(o == want for o in out)
    _f, w = T.kde_density_on_nodes(q, OH, HALFWIDTH)
    assert not w.flags.writeable
    with pytest.raises(ValueError):
        w[0] = 0.0


# ------------------------------------------------------------------------------------------------------------------
# the pixels are processed in blocks: the memory does not grow with their number
# ------------------------------------------------------------------------------------------------------------------
def _blocks_used(monkeypatch, q, max_copies):
    """The density of ``q`` with blocks of at most ``max_copies`` copies, and the number of blocks that were hashed."""
    calls = []
    real = T._build_hash

    def counting(*args):
        calls.append(len(args[0]))
        return real(*args)

    monkeypatch.setattr(T, "_build_hash", counting)
    monkeypatch.setattr(T, "MAX_CHUNK_COPIES", max_copies)
    f, w = T.kde_density_on_nodes(q, OH, HALFWIDTH)
    return f, w, calls


def test_blocks_of_pixels_give_the_density_of_a_single_pass(monkeypatch):
    q = _cluster_set()
    whole_f, whole_w, whole_calls = _blocks_used(monkeypatch, q, 10 ** 9)
    assert len(whole_calls) == 1
    # 40 pixels per block, 7 blocks or more: the partial densities add up to the single-pass density
    f, w, calls = _blocks_used(monkeypatch, q, 40 * len(OH))
    assert len(calls) >= 5
    np.testing.assert_allclose(f, whole_f, rtol=1e-12, atol=1e-12 * whole_f.max())
    np.testing.assert_array_equal(w, whole_w)
    j, s = T.kde_texture_index_and_entropy(q, OH, HALFWIDTH)           # the default (one block here)
    wsum = w.sum()
    assert float(np.dot(w, f * f) / wsum) == pytest.approx(j, rel=1e-12)


def test_a_block_size_that_does_not_divide_the_pixels_loses_none(monkeypatch):
    q = _cluster_set()
    n = len(q)
    f_odd, _, calls = _blocks_used(monkeypatch, q, 7 * len(OH))        # blocks of 7 pixels, the last one shorter
    assert len(calls) >= n // 7 and n % 7 != 0
    f_one, _, _ = _blocks_used(monkeypatch, q, 10 ** 9)
    np.testing.assert_allclose(f_odd, f_one, rtol=1e-12, atol=1e-12 * f_one.max())
    assert T.kde_mass(q, OH, HALFWIDTH) == pytest.approx(1.0, abs=MASS_TOL)


def test_the_peak_memory_is_bounded_by_the_block_not_by_the_number_of_pixels(monkeypatch):
    import tracemalloc
    rng = np.random.default_rng(3)
    q = jitter(rng, POSITIONS["generic"], 30000, 3.0)
    T.kde_density_on_nodes(q[:50], OH, HALFWIDTH)                      # builds and caches the nodes outside the trace

    def peak(max_copies):
        monkeypatch.setattr(T, "MAX_CHUNK_COPIES", max_copies)
        tracemalloc.start()
        try:
            T.kde_density_on_nodes(q, OH, HALFWIDTH)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    whole, chunked = peak(10 ** 9), peak(100_000)
    # 30 000 pixels x 24 operators = 720 000 copies: the single pass holds ~3 arrays of them (about 70 MB); a block of
    # 100 000 holds a seventh of it, plus the density at the nodes
    assert whole > 40e6
    assert chunked < 0.5 * whole


# ------------------------------------------------------------------------------------------------------------------
# real maps (slow, need the maintainer's scans)
# ------------------------------------------------------------------------------------------------------------------
REAL = {
    "sampleB": ("EBSD_SampleB_extrusion_withPattern Sample_B Arbeitsbereich 1 Elementverteilungsdaten 1.h5oina",
                8593, 25.573309, -2.180473),
    "g114": ("ground_truth/G1_14.h5oina", 169745, 4.276201, -0.802549),
}


@pytest.mark.slow
@pytest.mark.parametrize("name", list(REAL))
def test_real_maps_equal_mtex(name):
    from analysis.ebsd_dataset import EBSDDataset
    from analysis.xmap_loader import load_xmap_file
    from tests.data_deps import TEST_DATA_DIR, first_missing
    base = os.environ.get("ORIENTA_TEST_DATA")
    base = Path(base) if base else TEST_DATA_DIR
    fname, n_px, want_j, want_s = REAL[name]
    path = base / fname
    if first_missing(path) is not None:
        pytest.skip(f"needs {path} (set ORIENTA_TEST_DATA to a Test_data directory that has it)")
    xmap, step, _meta = load_xmap_file(str(path))
    ds = EBSDDataset(xmap, step_size=step)
    mask = np.asarray(ds.xmap.phase_id) >= 0
    assert int(mask.sum()) == n_px
    j, s = T.texture_index_and_entropy(ds, mask, HALFWIDTH)
    print(f"{name}: J={j:.6f} (MTEX {want_j}) S={s:.6f} (MTEX {want_s})")
    assert j == pytest.approx(want_j, rel=MTEX_J_TOL)
    assert s == pytest.approx(want_s, abs=0.05)
