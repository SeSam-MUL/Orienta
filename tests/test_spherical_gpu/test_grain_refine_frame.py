"""The grain-flip Newton polish must hand its seed to Newton in the VOLUME frame.

Background (2026-09-10, ``tasks/fidelity/REPORT.md`` open item 3)
-----------------------------------------------------------------
``backend/spherical_gpu/pipeline/_frame.py`` draws the line: everything inside
the spherical pipeline -- the cc volume, the Wigner-d chain, ``cc_at_rotation``
and ``newton_refine`` -- speaks the *volume* frame, and every orientation that
LEAVES the pipeline is carried into the renderer / EMSphInx crystal frame by a
constant ``C2<1 -1 0>``.  ``refiner.refine_index_result`` does exactly that for
its ANALYTIC_NEWTON branch: it inverts the constant on the way into
``newton_refine`` and re-applies it on the way out.

``routes.indexing._grain_newton_refine`` -- the guarded per-pixel polish behind
the manual pseudo-symmetry grain flip -- runs the same ``newton_refine`` against
the same cc surface, but seeded it with the STORED quaternions, which are in the
renderer frame.  The seed therefore sat a fixed 180 deg about <1 -1 0> away from
the peak it was supposed to polish.

``C2<1 -1 0>`` lies in ``O`` and in ``D4``, so on ``m-3m`` / ``4/mmm`` phases the
cc surface does not notice.  For exactly the ``z_rot == 2`` phases this tool
exists for (m-3, 23, mmm, -43m) it is not a symmetry, so the polish started
outside the ~2 deg render-NCC basin, Newton could not converge there, and the
``move <= max_move_deg`` guard threw every pixel away: the "Refine" checkbox --
which is ON by default in ``PseudoSymmetryPanel.jsx`` -- silently never did
anything on the phases it was written for.
"""
from __future__ import annotations

import math
import types

import numpy as np
import pytest

from backend.spherical_gpu.pipeline._frame import (CRYSTAL_FRAME_FIX_QUAT,
                                                   apply_frame_fix_zxz,
                                                   invert_frame_fix_zxz)


# ----------------------------------------------------------------------
# Pure frame arithmetic
# ----------------------------------------------------------------------
def _quat(eu):
    from orix.quaternion import Rotation
    return np.asarray(
        Rotation.from_euler(np.asarray(eu, dtype=np.float64).reshape(1, 3)).data,
        dtype=np.float64).reshape(4)


def _angle_deg(qa, qb):
    """Plain (NOT symmetry-reduced) rotation angle between two unit quats."""
    d = abs(float(np.dot(np.asarray(qa).ravel(), np.asarray(qb).ravel())))
    return float(np.degrees(2.0 * np.arccos(min(1.0, d))))


def test_pack_then_unpack_returns_the_same_orientation():
    """``invert`` -> ``apply`` must be the identity on the ORIENTATION.

    The seed leaves the caller in the renderer frame, is packed into the volume
    frame for Newton and unpacked again; a polish that moved nothing must give
    the caller back exactly what it handed in.
    """
    rng = np.random.default_rng(7)
    eu = np.stack([rng.uniform(0.0, 2.0 * math.pi, 64),
                   rng.uniform(0.0, math.pi, 64),
                   rng.uniform(0.0, 2.0 * math.pi, 64)], axis=1)
    back = np.stack(apply_frame_fix_zxz(*invert_frame_fix_zxz(
        eu[:, 0], eu[:, 1], eu[:, 2])), axis=1)
    worst = max(_angle_deg(_quat(a), _quat(b)) for a, b in zip(eu, back))
    assert worst < 1e-6, f"round trip moved the orientation by {worst:.6f} deg"


def test_the_naive_seed_is_exactly_the_frame_operator_away():
    """Pin the size of the defect: feeding the stored Euler triple straight to
    Newton starts it a full 180 deg about <1 -1 0> from the peak."""
    rng = np.random.default_rng(11)
    eu = np.array([rng.uniform(0.0, 2.0 * math.pi),
                   rng.uniform(0.2, math.pi - 0.2),
                   rng.uniform(0.0, 2.0 * math.pi)])
    q_ren = _quat(eu)
    q_vol = _quat(np.asarray(invert_frame_fix_zxz(*eu)))
    assert _angle_deg(q_ren, q_vol) == pytest.approx(180.0, abs=1e-4)
    # ... and the operator is the documented constant, not some other 180 deg.
    from backend.spherical_gpu.pseudosym import _qconj, _qmul
    op = _qmul(q_ren.reshape(1, 4), _qconj(q_vol.reshape(1, 4)))[0]
    op = op * (1.0 if op[0] >= 0 else -1.0)
    want = np.asarray(CRYSTAL_FRAME_FIX_QUAT, dtype=np.float64)
    assert abs(abs(float(np.dot(op, want))) - 1.0) < 1e-9, (
        f"operator between the two frames is {np.round(op, 4)}, expected "
        f"+-{np.round(want, 4)}")


# ----------------------------------------------------------------------
# The function under test, on a synthetic cc surface
# ----------------------------------------------------------------------
_BASIN_DEG = 10.0     # stand-in for the ~2 deg render-NCC basin: generous, so
                      # the test cannot pass by luck of a tight tolerance.
_L = 4


class _FakeIndexer:
    """Just enough of Tier1Indexer for _grain_newton_refine's coefficient step.

    The synthetic cc surface below ignores flm/gln entirely -- this test is
    about which FRAME the seed is expressed in, not about the correlation.
    """

    def __init__(self):
        import torch
        self.bandwidth = _L
        self.device = torch.device("cpu")
        self._master_coefs = torch.zeros((_L, _L), dtype=torch.complex128)

    def _run_preprocessing(self, pats):
        return pats

    def _direct_sht_coefs(self, prep):
        import torch
        return torch.zeros((prep.shape[0], _L, _L), dtype=torch.complex128)


class _FakeBackend:
    def __init__(self):
        self._indexers = [_FakeIndexer()]

    def _ensure_built(self, det):
        return None


@pytest.fixture
def refine_harness(monkeypatch):
    """Wire _grain_newton_refine to a synthetic cc surface with a known peak.

    Returns ``(call, seeds_seen, q_stored, q_truth_renderer)``.  The peak sits
    1 deg from the stored orientation *in the volume frame*; a caller that
    seeds Newton in the renderer frame starts 180 deg away from it.
    """
    torch = pytest.importorskip("torch")
    import tools.pattern_comparison as pc
    import backend.api.routes.indexing as ix
    from backend.spherical_gpu._math import sht_newton as sn
    from orix.quaternion import Rotation

    # Stored orientation (renderer frame) and the truth Newton should find.
    eu_stored_ren = np.array([0.7, 0.9, 1.3])
    q_stored = _quat(eu_stored_ren)
    eu_stored_vol = np.asarray(invert_frame_fix_zxz(*eu_stored_ren))
    # 1 deg about x, applied on the right so it is a small move of the same
    # orientation rather than a change of representative.
    half = math.radians(1.0) / 2.0
    dq = np.array([math.cos(half), math.sin(half), 0.0, 0.0])
    from backend.spherical_gpu.pseudosym import _qmul
    q_truth_vol = _qmul(_quat(eu_stored_vol).reshape(1, 4), dq.reshape(1, 4))[0]
    eu_truth_vol = np.asarray(
        Rotation(q_truth_vol.reshape(1, 4)).to_euler(), dtype=np.float64).reshape(3)
    q_truth_ren = _quat(np.asarray(apply_frame_fix_zxz(*eu_truth_vol)))

    seeds_seen: list[np.ndarray] = []

    def _cc(eu_vol_zxz) -> float:
        """A peak of 1.0 at the truth, falling off with angle. Volume frame."""
        d = _angle_deg(_quat(np.asarray(eu_vol_zxz, dtype=np.float64)), q_truth_vol)
        return float(math.cos(math.radians(min(d, 180.0)) / 2.0))

    def fake_cc_at_rotation(flm, gln, eu_zyz, L):
        eu_zxz = sn._zyz_to_zxz(eu_zyz)
        return torch.tensor(_cc(eu_zxz.detach().cpu().numpy()), dtype=torch.float64)

    def fake_newton_refine(flm, gln, eu_seed, L, **kw):
        seed = np.asarray(eu_seed.detach().cpu().numpy(), dtype=np.float64)
        seeds_seen.append(seed)
        if _angle_deg(_quat(seed), q_truth_vol) <= _BASIN_DEG:
            return (torch.as_tensor(eu_truth_vol, dtype=torch.float64),
                    torch.tensor(_cc(eu_truth_vol), dtype=torch.float64), True)
        # Outside the basin Newton cannot reach the peak; report no convergence.
        return (eu_seed.clone(), torch.tensor(_cc(seed), dtype=torch.float64),
                False)

    monkeypatch.setattr(sn, "cc_at_rotation", fake_cc_at_rotation)
    monkeypatch.setattr(sn, "newton_refine", fake_newton_refine)
    monkeypatch.setattr(pc, "get_experimental_pattern",
                        lambda result, r, c: np.ones((4, 4), dtype=np.float32))
    # Render-NCC spot check: a pattern that is simply "how far from the truth",
    # and a score that rewards being closer. The gate is real, the optics are not.
    monkeypatch.setattr(
        ix, "_render_sim_for_ncc",
        lambda sht, quat, det, bw=128: np.full(
            (1, 1), _angle_deg(quat, q_truth_ren), dtype=np.float32))
    monkeypatch.setattr(pc, "compute_ncc_scalar",
                        lambda exp, sim: -float(np.asarray(sim).ravel()[0]))
    monkeypatch.setattr(ix, "_get_phase_compare_backend", lambda key: _FakeBackend())

    def call():
        result = types.SimpleNamespace(original_shape=(1, 1))
        det = {"sample_tilt": 70.0, "tilt": 0.0, "pixel_size": 70.0}
        return ix._grain_newton_refine(
            result, det, "nonexistent.sht", "m-3", [(0, 0)], {0: q_stored},
            bandwidth=_L)

    return call, seeds_seen, q_stored, q_truth_ren


def test_newton_is_seeded_in_the_volume_frame(refine_harness):
    """The seed Newton actually sees must be the stored orientation carried
    BACK into the volume frame -- not the stored triple itself."""
    call, seeds_seen, q_stored, _q_truth = refine_harness
    call()
    assert seeds_seen, "newton_refine was never called"
    seed_q = _quat(seeds_seen[0])
    got = _angle_deg(seed_q, q_stored)
    assert got == pytest.approx(180.0, abs=1e-3), (
        f"Newton was seeded {got:.3f} deg from the stored orientation; expected "
        f"the constant C2<1 -1 0> (180 deg) that separates the cc volume from "
        f"the renderer frame. 0 deg means the renderer-frame triple was handed "
        f"straight to the volume-frame cc surface.")


def test_the_polish_converges_and_returns_a_renderer_frame_quaternion(
        refine_harness):
    """End to end: a peak 1 deg from the stored orientation must be found, pass
    the move guard and the render-NCC gate, and come back in the frame the
    caller writes into the CrystalMap."""
    call, _seeds, q_stored, q_truth_ren = refine_harness
    refined, summary = call()
    assert summary.get("status") == "applied", (
        f"the guarded polish rejected a 1 deg refinement: {summary}")
    assert refined is not None and 0 in refined
    got = _angle_deg(refined[0], q_truth_ren)
    assert got < 1e-3, (
        f"refined orientation is {got:.4f} deg from the truth in the renderer "
        f"frame -- the result was not carried back out of the volume frame")
    # Sanity: it really did move, and by about the 1 deg we planted.
    assert _angle_deg(refined[0], q_stored) == pytest.approx(1.0, abs=1e-3)
