"""Unit tests for map-wide pseudo-symmetry variant unification
(backend/spherical_gpu/pipeline/variant_unification.py).

Covers the design's failure-mode checklist: generic class enumeration across
point groups, supergroup segmentation (merges variant-split grains, does NOT
merge generically-rotated neighbours), the coherence policy (speckle vs
pseudo-merohedral twin), twin protection (coherent domains are only flipped on
a CLEAR render-NCC margin), texture preservation, ambiguity flagging and tiny-
orphan rescue. Scoring is injected as a fake so tests are pure CPU.
"""
from __future__ import annotations
import numpy as np
import pytest

from backend.spherical_gpu.pipeline.variant_unification import (
    class_label_pixels,
    class_reps_for_phase,
    metric_supergroup_ops,
    pair_orientation_angle_deg,
    plan_grain_units,
    proper_coset_class_reps,
    segment_supergroup_grains,
    nearest_representative,
    snap_to_class,
    unify_map,
)
from backend.spherical_gpu.pseudosym import _qmul, _sym_quats, same_orientation_angle_deg


def _q_axis_angle(axis, deg):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    a = np.radians(deg) / 2.0
    return np.array([np.cos(a), *(np.sin(a) * axis)])


def _ang(qa, qb, pg):
    return float(same_orientation_angle_deg(np.atleast_2d(qa), np.asarray(qb), pg)[0])


# ---------------------------------------------------------------------------
# Class enumeration — generic over point groups
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pg,expected", [
    # Counts are at the LAUE level (Friedel): only pattern-DISTINGUISHABLE
    # variants count. -43m/4mm/mm2 have Laue = full holohedry → their
    # "variants" are pattern-identical → 1 class (nothing any method could
    # decide). See proper_coset_class_reps docstring.
    ("m-3", 2), ("23", 2),                       # cubic approximants: holohedry coset only here;
                                                 # the pseudo-icosahedral classes are added by
                                                 # class_reps_for_phase (test below)
    ("-3m", 2),                                  # trigonal (Dauphiné 60°-about-c)
    ("4", 2),                                    # tetragonal Laue 4/m
    ("-43m", 1), ("4mm", 1), ("mm2", 1),         # Laue-holohedral → no-op
    ("m-3m", 1), ("432", 1), ("222", 1), ("6/mmm", 1),  # holohedral → no-op
])
def test_class_counts_across_point_groups(pg, expected):
    reps = proper_coset_class_reps(pg)
    assert reps.shape == (expected, 4), f"{pg}: got {reps.shape[0]} classes"
    # class 0 is the identity
    np.testing.assert_allclose(reps[0], [1, 0, 0, 0], atol=1e-12)
    # all classes are genuinely distinct under the TRUE group
    for i in range(reps.shape[0]):
        for j in range(i + 1, reps.shape[0]):
            assert _ang(reps[i], reps[j], pg) > 10.0


def test_class_reps_for_phase_adds_icosahedral_classes_for_approximants():
    """Cubic approximants (m-3 / 23) get the pseudo-icosahedral coset classes on
    top of the cubic one: identity + 90 deg about <100> + four 72/144-deg
    classes about <0 1 tau>. Everything else is unchanged."""
    from backend.spherical_gpu.pipeline.variant_unification import class_reps_for_phase
    assert class_reps_for_phase("m-3").shape[0] == 6
    assert class_reps_for_phase("23").shape[0] == 6
    for pg, k in (("-3m", 2), ("4", 2), ("m-3m", 1), ("432", 1), ("-43m", 1)):
        assert class_reps_for_phase(pg).shape[0] == k, pg


def test_metric_pseudo_cubic_detection():
    assert metric_supergroup_ops((4.04, 4.04, 4.05, 90, 90, 90)) is not None
    assert metric_supergroup_ops((4.0, 4.0, 5.2, 90, 90, 90)) is None      # real tetragonal
    assert metric_supergroup_ops((4.0, 4.0, 4.0, 90, 90, 120)) is None     # wrong angles
    assert metric_supergroup_ops(None) is None
    # pseudo-cubic tetragonal phase: metric classes extend the point-group set
    base = proper_coset_class_reps("4/mmm").shape[0]
    ext = class_reps_for_phase("4/mmm", lattice=(4.0, 4.0, 4.02, 90, 90, 90)).shape[0]
    assert base == 1 and ext > 1


# ---------------------------------------------------------------------------
# Supergroup segmentation
# ---------------------------------------------------------------------------

def _grid(nr, nc, fill):
    q = np.zeros((nr * nc, 4))
    q[:] = fill
    phase = np.zeros(nr * nc, int)
    return q, phase


def test_segmentation_merges_variant_split_grain():
    """One physical grain, right half in the other m-3 variant → ONE grain
    modulo the supergroup (the case plain orientation segmentation fails)."""
    nr, nc = 4, 6
    qA = _q_axis_angle([0, 0, 1], 3.0)
    h = proper_coset_class_reps("m-3")[1]
    q, phase = _grid(nr, nc, qA)
    for r in range(nr):
        for c in range(nc // 2, nc):
            q[r * nc + c] = _qmul(h[None, :], qA[None, :])[0]
    labels = segment_supergroup_grains(q, phase, nr, nc, 0, "m-3", 5.0)
    assert set(labels.tolist()) == {0}            # exactly one grain


def test_segmentation_keeps_generically_rotated_neighbour_separate():
    """Neighbouring grain of the SAME phase, genuinely rotated (25° about a
    random-ish axis, far from any coset op) → stays a separate grain."""
    nr, nc = 4, 6
    qA = _q_axis_angle([0, 0, 1], 3.0)
    qB = _qmul(_q_axis_angle([1, 2, 3], 25.0)[None, :], qA[None, :])[0]
    q, phase = _grid(nr, nc, qA)
    for r in range(nr):
        for c in range(nc // 2, nc):
            q[r * nc + c] = qB
    labels = segment_supergroup_grains(q, phase, nr, nc, 0, "m-3", 5.0)
    assert len(set(labels.tolist())) == 2


def test_pair_disorientation_vectorised_matches_scalar():
    sym = _sym_quats("m-3m")
    qa = np.stack([_q_axis_angle([0, 0, 1], d) for d in (0, 10, 45, 90)])
    qb = np.stack([_q_axis_angle([0, 0, 1], 0.0)] * 4)
    d = pair_orientation_angle_deg(qa, qb, sym)
    assert d[0] < 1e-6 and abs(d[1] - 10) < 1e-6
    assert abs(d[2] - 45.0) < 1.0                 # 45° z: 4-fold reduces to 45°
    assert d[3] < 1e-6                            # 90° z ∈ m-3m → same orientation


# ---------------------------------------------------------------------------
# Coherence policy: speckle vs twin
# ---------------------------------------------------------------------------

def _checkerboard_classes(nr, nc):
    return np.array([(r + c) % 2 for r in range(nr) for c in range(nc)])


def test_policy_salt_and_pepper_is_speckle_mode():
    nr, nc = 6, 6
    pix = np.arange(nr * nc)
    cls = _checkerboard_classes(nr, nc)
    mode, units = plan_grain_units(pix, cls, nc)
    assert mode == "speckle"
    assert len(units) == 1 and units[0]["pixels"].size == nr * nc


def test_policy_two_blocks_is_domain_mode():
    nr, nc = 6, 8
    pix = np.arange(nr * nc)
    cls = np.array([0 if (f % nc) < nc // 2 else 1 for f in pix])
    mode, units = plan_grain_units(pix, cls, nc, domain_min_px=8)
    assert mode == "domains"
    assert len(units) == 2
    sizes = sorted(u["pixels"].size for u in units)
    assert sizes == [24, 24]


def test_policy_small_island_absorbed_into_domain():
    nr, nc = 6, 8
    pix = np.arange(nr * nc)
    cls = np.array([0 if (f % nc) < nc // 2 else 1 for f in pix])
    cls[1 * nc + 1] = 1                            # 1-px island inside domain 0
    mode, units = plan_grain_units(pix, cls, nc, domain_min_px=8)
    assert mode == "domains"
    owner = [u for u in units if (1 * nc + 1) in u["pixels"]][0]
    assert owner["current_class"] == 0             # absorbed into surrounding domain


# ---------------------------------------------------------------------------
# snap_to_class texture preservation
# ---------------------------------------------------------------------------

def test_snap_preserves_texture():
    reps = proper_coset_class_reps("m-3")
    h = reps[1]
    t = np.stack([_q_axis_angle([1, 0, 0], 1.0 * i) for i in range(5)])   # bent grain
    meas = _qmul(h[None, :], t)                    # all in class 1
    cls = class_label_pixels(meas, t[0], reps, "m-3")
    assert set(cls.tolist()) == {1}
    snapped = snap_to_class(meas, cls, 0, reps)
    for i in range(5):
        assert _ang(snapped[i], t[i], "m-3") < 0.1
    assert 3.5 < _ang(snapped[4], snapped[0], "m-3") < 4.5   # gradient survives


# ---------------------------------------------------------------------------
# unify_map end-to-end (fake scorer)
# ---------------------------------------------------------------------------

def _speckled_map(nr=6, nc=8, pg="m-3", seed=7):
    """One physical grain (slight bend), pixels randomly assigned to the two
    variant classes — the Scan1 salt-and-pepper situation. Returns
    (full_q, phase, truth (n,4))."""
    rng = np.random.default_rng(seed)
    reps = proper_coset_class_reps(pg)
    truth = np.stack([
        _qmul(_q_axis_angle([1, 0, 0], 0.05 * (r + c))[None, :],
              _q_axis_angle([0, 0, 1], 5.0)[None, :])[0]
        for r in range(nr) for c in range(nc)])
    cls = rng.integers(0, 2, nr * nc)
    q = truth.copy()
    m = cls == 1
    q[m] = _qmul(reps[1][None, :], truth[m])
    return q, np.zeros(nr * nc, int), truth


def _truth_scorer(truth, pg="m-3", good=0.5, bad=0.2):
    """Fake render-NCC: high when the candidate orientation is (mod the TRUE
    group) near the ground-truth orientation of that pixel, low otherwise."""
    def score(flat_idx, quats):
        out = []
        for f, qq in zip(flat_idx, np.atleast_2d(quats)):
            d = _ang(qq, truth[int(f)], pg)
            out.append(good if d < 5.0 else bad)
        return np.asarray(out)
    return score


def test_unify_map_fixes_speckle_and_preserves_texture():
    nr, nc = 6, 8
    q, phase, truth = _speckled_map(nr, nc)
    new_q, report = unify_map(q, phase, nr, nc, 0, "m-3",
                              _truth_scorer(truth))
    assert new_q is not None
    assert report["n_grains"] == 1
    assert report["n_flipped_units"] == 1
    for i in range(nr * nc):
        assert _ang(new_q[i], truth[i], "m-3") < 0.2     # right class + texture


def test_unify_map_twin_protection_no_clear_margin_keeps_domains():
    """Two coherent domains (a genuine pseudo-merohedral twin), scorer sees NO
    clear margin → both domains must stay bit-identical (primum non nocere)."""
    nr, nc = 6, 8
    reps = proper_coset_class_reps("m-3")
    qA = _q_axis_angle([0, 0, 1], 5.0)
    qB = _qmul(reps[1][None, :], qA[None, :])[0]
    q = np.zeros((nr * nc, 4))
    for r in range(nr):
        for c in range(nc):
            q[r * nc + c] = qA if c < nc // 2 else qB
    flat = lambda i, qq: np.full(len(i), 0.3)              # flat signal
    new_q, report = unify_map(q, np.zeros(nr * nc, int), nr, nc, 0, "m-3",
                              lambda i, qq: np.full(len(np.atleast_1d(i)), 0.3))
    np.testing.assert_allclose(new_q, q, atol=1e-12)       # NOTHING changed
    assert report["n_flipped_units"] == 0
    assert all(g["mode"] == "domains" for g in report["grains"])


def test_unify_map_twin_with_clear_margin_flips_wrong_domain():
    """Coherent domain whose OTHER class clearly wins the render score → it IS
    flipped (a whole grain anchored on the wrong variant, caught by
    verification)."""
    nr, nc = 6, 8
    reps = proper_coset_class_reps("m-3")
    qA = _q_axis_angle([0, 0, 1], 5.0)
    truth = np.tile(qA, (nr * nc, 1))
    q = truth.copy()
    for r in range(nr):                                     # right half flipped
        for c in range(nc // 2, nc):
            q[r * nc + c] = _qmul(reps[1][None, :], truth[r * nc + c][None, :])[0]
    new_q, report = unify_map(q, np.zeros(nr * nc, int), nr, nc, 0, "m-3",
                              _truth_scorer(truth))
    for i in range(nr * nc):
        assert _ang(new_q[i], truth[i], "m-3") < 0.2
    assert report["n_flipped_units"] >= 1


def test_unify_map_flags_ambiguous_speckle_but_still_unifies():
    nr, nc = 6, 8
    q, phase, truth = _speckled_map(nr, nc)
    new_q, report = unify_map(q, phase, nr, nc, 0, "m-3",
                              lambda i, qq: np.full(len(np.atleast_1d(i)), 0.3))
    assert report["n_ambiguous"] >= 1
    # consistent even though ambiguous: all pixels in ONE class now (the
    # DOMINANT one — flat signal must not arbitrarily flip the grain)
    reps = proper_coset_class_reps("m-3")
    cls = class_label_pixels(new_q, new_q[0], reps, "m-3")
    assert len(set(cls.tolist())) == 1


# ---------------------------------------------------------------------------
# Stage 2.5: render-verified small-grain adoption (non-coset mis-indexing
# basins — e.g. the constant-71.9° second Hough basin observed on real Scan1)
# ---------------------------------------------------------------------------

def _blob_map(nr=8, nc=8, blob=((3, 3), (3, 4), (4, 3), (4, 4), (2, 3))):
    """Big uniform grain + a 5-px blob at a FIXED NON-COSET rotation (35° about
    an arbitrary axis — far from any m-3m op, so the class machinery must NOT
    merge it; only adoption can fix it)."""
    qA = _q_axis_angle([0, 0, 1], 5.0)
    C_bad = _q_axis_angle([2, 1, 3], 35.0)
    q = np.tile(qA, (nr * nc, 1))
    blob_flat = [r * nc + c for (r, c) in blob]
    for f in blob_flat:
        q[f] = _qmul(C_bad[None, :], qA[None, :])[0]
    return q, np.zeros(nr * nc, int), qA, np.asarray(blob_flat)


def test_adoption_fixes_non_coset_basin_blob():
    nr, nc = 8, 8
    q, phase, qA, blob = _blob_map(nr, nc)
    new_q, report = unify_map(q, phase, nr, nc, 0, "m-3",
                              _truth_scorer(np.tile(qA, (nr * nc, 1))))
    assert report["n_adopted"] == 1
    for f in blob:
        assert _ang(new_q[f], qA, "m-3") < 0.5     # blob adopted the matrix branch
    # matrix untouched
    assert _ang(new_q[0], qA, "m-3") < 1e-6


def test_adoption_keeps_real_small_grain():
    """A REAL small grain (its own orientation renders best) must stay
    bit-identical — primum non nocere."""
    nr, nc = 8, 8
    q, phase, qA, blob = _blob_map(nr, nc)
    truth = np.tile(qA, (nr * nc, 1))
    truth[blob] = q[blob]                          # blob orientation is REAL
    new_q, report = unify_map(q, phase, nr, nc, 0, "m-3", _truth_scorer(truth))
    assert report["n_adopted"] == 0
    np.testing.assert_allclose(new_q[blob], q[blob], atol=1e-12)


def test_adoption_flat_signal_keeps_and_flags():
    nr, nc = 8, 8
    q, phase, qA, blob = _blob_map(nr, nc)
    new_q, report = unify_map(q, phase, nr, nc, 0, "m-3",
                              lambda i, qq: np.full(len(np.atleast_2d(qq)), 0.3))
    assert report["n_adopted"] == 0
    np.testing.assert_allclose(new_q[blob], q[blob], atol=1e-12)
    adoption_entries = [g for g in report["grains"] if g["mode"] == "adoption"]
    assert adoption_entries and adoption_entries[0]["decision"] == "kept_ambiguous"


def test_adoption_preserves_blob_internal_texture():
    """Adoption maps the blob by ONE constant rotation — internal gradients
    survive (each pixel keeps its own deviation)."""
    nr, nc = 8, 8
    qA = _q_axis_angle([0, 0, 1], 5.0)
    C_bad = _q_axis_angle([2, 1, 3], 35.0)
    q = np.tile(qA, (nr * nc, 1))
    blob = np.asarray([3 * nc + 3, 3 * nc + 4, 4 * nc + 3, 4 * nc + 4, 2 * nc + 3])
    truth_blob = []
    for j, f in enumerate(blob):
        t = _qmul(_q_axis_angle([1, 0, 0], 0.5 * j)[None, :], qA[None, :])[0]
        truth_blob.append(t)
        q[f] = _qmul(C_bad[None, :], t[None, :])[0]
    truth = np.tile(qA, (nr * nc, 1))
    for j, f in enumerate(blob):
        truth[f] = truth_blob[j]
    new_q, report = unify_map(q, phase_full := np.zeros(nr * nc, int), nr, nc,
                              0, "m-3", _truth_scorer(truth))
    assert report["n_adopted"] == 1
    # Adoption anchors the blob MEAN on the donor mean, so each pixel lands
    # within ~the blob's mean internal deviation of its truth (here ≤ 2°) —
    # the absolute anchor is the donor; per-pixel truth is unknowable without
    # refinement. What must survive exactly is the INTERNAL texture:
    for j, f in enumerate(blob):
        assert _ang(new_q[f], truth_blob[j], "m-3") < 2.0
    d = _ang(new_q[blob[-1]], new_q[blob[0]], "m-3")
    assert 1.5 < d < 2.5                           # 0.5°/px gradient survives


def test_unify_map_noop_for_holohedral_phase():
    nr, nc = 4, 4
    q = np.tile(_q_axis_angle([0, 0, 1], 5.0), (nr * nc, 1))
    new_q, report = unify_map(q, np.zeros(nr * nc, int), nr, nc, 0, "m-3m",
                              lambda i, qq: np.ones(len(np.atleast_1d(i))))
    assert new_q is None
    assert report.get("skipped")


# ---------------------------------------------------------------------------
# REAL-DATA E2E on SampleB alpha-AlFeMnSi (GPU + machine-local data gated)
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_ALPHA_SHT = _ROOT / "Database/EBSD_SHT_Database/alpha-AlFeMnSi_ICSD-52623/Al100.89Fe21.24Mn5.31Si10.62 (alpha-AlFeMnSi_ICSD-52623) [cP138] {20kV}.sht"
_ALPHA_CIF = _ROOT / "Database/CIF_Library/alpha-AlFeMnSi_ICSD-52623.cif"
_SAMPLEB = _ROOT / "Test_data/EBSD_SampleB_extrusion_withPattern Sample_B Arbeitsbereich 1 Elementverteilungsdaten 1.h5oina"
_HAVE_DATA = _ALPHA_SHT.is_file() and _ALPHA_CIF.is_file() and _SAMPLEB.is_file()
_DET = {"pat_width": 156, "pat_height": 128, "pixel_size": 70.0, "binning": 1,
        "tilt": 4.323, "sample_tilt": 70.003, "pc_x": 0.5026108, "pc_y": 0.3271,
        "pc_z": 0.8459, "vendor": "Bruker", "n_cols": 120, "n_rows": 90,
        "step_x": 1.0, "step_y": 1.0, "source_vendor": "Oxford"}


@pytest.mark.skipif(not _HAVE_DATA, reason="SampleB / alpha master not present (machine-local)")
def test_unify_map_sampleb_real_data_end_to_end():
    """Full E2E of the map-wide unification on REAL SampleB patterns:

    Take a 6x6 block inside a validated alpha-AlFeMnSi grain, use the Hough
    orientations (render-verified correct, median NCC ~0.5) as ground truth,
    randomly flip half the pixels to the other m-3 variant class (the exact
    salt-and-pepper the variant-blind Hough anchor produces on noisy data),
    then run unify_map with the REAL renderer scorer. Every pixel must come
    back on the true class, and the decisive render-NCC margin must be real
    (documents that the physical signal actually separates the two classes).
    """
    import h5py
    from backend.spherical_gpu.pipeline.resolution import resolve_map
    from backend.spherical_gpu.pipeline.variant_unification import (
        build_render_score_fn,
    )

    rows = range(48, 54)
    cols = range(58, 64)
    ncols_map = 120
    pixels = [(r, c) for r in rows for c in cols]
    with h5py.File(_SAMPLEB, "r") as f:
        dset = f["1/EBSD/Data/Processed Patterns"]
        frame_avg = np.stack([
            np.stack([dset[(r + dr) * ncols_map + (c + dc)].astype(np.float32)
                      for dr in (-1, 0, 1) for dc in (-1, 0, 1)]).mean(0)
            for r, c in pixels])

    # Ground truth: Hough band-geometry orientations (validated to render
    # ~0.5 on this grain — see test_resolution.py). All 36 px must index.
    res = resolve_map(frame_avg, str(_ALPHA_CIF), _DET, "m-3")
    truth = res["resolved"]                        # (36, 4)
    assert int(res["n_fallback"]) == 0

    # Build a compact grid holding just this block, speckle half the pixels.
    nr, nc = len(rows), len(cols)
    reps = proper_coset_class_reps("m-3")
    rng = np.random.default_rng(3)
    cls_true = rng.integers(0, 2, nr * nc)
    q = truth.copy()
    m = cls_true == 1
    q[m] = _qmul(reps[1][None, :], truth[m])
    # sanity: the speckled map really is scattered across variants
    d_raw = np.array([_ang(q[i], truth[i], "m-3") for i in range(nr * nc)])
    assert d_raw[m].min() > 25.0

    pat_of_flat = {i: frame_avg[i] for i in range(nr * nc)}
    score_fn = build_render_score_fn(
        str(_ALPHA_SHT), _DET, lambda f: pat_of_flat.get(int(f)))
    new_q, report = unify_map(q, np.zeros(nr * nc, int), nr, nc, 0, "m-3",
                              score_fn)
    assert new_q is not None

    # The block contains ONE dominant grain plus a few 1-2 px Hough outliers
    # at a boundary (real data). Identify the dominant grain the same way the
    # algorithm does and assert on IT; the outliers must be rescued.
    labels = segment_supergroup_grains(q, np.zeros(nr * nc, int), nr, nc,
                                       0, "m-3", 5.0)
    main = np.bincount(labels[labels >= 0]).argmax()
    main_pix = np.flatnonzero(labels == main)
    assert main_pix.size >= 25                    # dominant grain really found

    g = max(report["grains"], key=lambda d: d["pixels"])
    assert g["mode"] == "speckle"
    assert g["decision"] == "unified"             # NOT ambiguous — real margin
    assert g["margin"] > 0.03, f"render-NCC margin too weak: {g['margin']}"
    for i in main_pix:
        assert _ang(new_q[i], truth[i], "m-3") < 3.0, (
            f"pixel {i} ended {_ang(new_q[i], truth[i], 'm-3'):.1f} deg off truth")
    # tiny boundary orphans adopt the unified grain's orientation
    n_outliers = int((labels >= 0).sum() - main_pix.size)
    if n_outliers:
        assert report["n_rescued"] >= 1


# ---------------------------------------------------------------------------
# GPU-OOM resilience of the render-NCC scorer (pure CPU — seams monkeypatched)
# ---------------------------------------------------------------------------

_OOM_DET = {"pat_width": 64, "pat_height": 64, "pixel_size": 70.0, "binning": 1,
            "tilt": 0.0, "sample_tilt": 70.0, "pc_x": 0.5, "pc_y": 0.5,
            "pc_z": 0.6, "vendor": "Bruker"}


class _FakeRenderer:
    """Stand-in PatternRenderer whose first `oom_calls` renders raise a CUDA
    OOM RuntimeError, then succeed with a deterministic pattern."""

    def __init__(self, oom_calls):
        self.remaining_oom = int(oom_calls)
        self.n_render = 0
        self.device = "cpu"

    def render(self, pgrid, qt, pc, hw, pixel_size, tilt_deg=None,
               det_tilt_deg=None):
        self.n_render += 1
        if self.remaining_oom > 0:
            self.remaining_oom -= 1
            raise RuntimeError(
                "CUDA out of memory. Tried to allocate 2.00 GiB")
        import torch
        h, w = hw
        return torch.tensor(
            np.linspace(0.0, 1.0, h * w, dtype=np.float64).reshape(h, w))


def _patch_render_seams(monkeypatch, renderer, release_counter):
    """Point the service seams referenced by build_render_score_fn at a fake
    renderer so the scorer runs without a GPU."""
    import backend.api.services.sht_pattern_renderer as svc

    def _get_renderer():
        return renderer

    def _load_or_get_phase(sht_path, max_bandwidth=None):
        return object()          # opaque grid — the fake render ignores it

    def _release():
        release_counter.append(1)
        return 0

    monkeypatch.setattr(svc, "get_renderer", _get_renderer)
    monkeypatch.setattr(svc, "load_or_get_phase", _load_or_get_phase)
    monkeypatch.setattr(svc, "release_gpu_caches", _release)


def test_score_fn_retries_after_transient_oom(monkeypatch):
    """First render OOMs, the one-time cache-release + rebuild retry succeeds →
    the pixel gets a finite render-NCC score (never -inf)."""
    from backend.spherical_gpu.pipeline.variant_unification import (
        build_render_score_fn,
    )
    releases: list[int] = []
    fake = _FakeRenderer(oom_calls=1)
    _patch_render_seams(monkeypatch, fake, releases)

    H, W = _OOM_DET["pat_height"], _OOM_DET["pat_width"]
    exp = np.random.default_rng(0).random((H, W)).astype(np.float32)
    score = build_render_score_fn("dummy.sht", _OOM_DET, lambda f: exp)

    q = _q_axis_angle([0, 0, 1], 5.0)[None, :]
    out = score([0], q)
    assert out.shape == (1,)
    assert np.isfinite(out[0]), f"transient OOM should recover, got {out[0]}"
    assert len(releases) == 1              # rebuilt exactly once
    assert fake.n_render == 2              # one failed + one successful retry


def test_score_fn_persistent_oom_returns_neg_inf(monkeypatch):
    """Render OOMs on every attempt (even after rebuild) → the pixel scores
    -inf so it can never decide a variant class."""
    from backend.spherical_gpu.pipeline.variant_unification import (
        build_render_score_fn,
    )
    releases: list[int] = []
    fake = _FakeRenderer(oom_calls=10_000)      # never recovers
    _patch_render_seams(monkeypatch, fake, releases)

    H, W = _OOM_DET["pat_height"], _OOM_DET["pat_width"]
    exp = np.random.default_rng(1).random((H, W)).astype(np.float32)
    score = build_render_score_fn("dummy.sht", _OOM_DET, lambda f: exp)

    q = _q_axis_angle([0, 0, 1], 5.0)[None, :]
    out = score([0], q)
    assert out.shape == (1,)
    assert out[0] == float("-inf")
    assert len(releases) == 1              # tried to recover once, then gave up
    assert fake.n_render == 2              # original + single retry, no loop


def test_score_fn_non_oom_runtimeerror_propagates(monkeypatch):
    """A RuntimeError that is NOT an OOM must NOT be swallowed by the retry
    path — it propagates so real bugs stay loud."""
    from backend.spherical_gpu.pipeline.variant_unification import (
        build_render_score_fn,
    )

    class _BoomRenderer(_FakeRenderer):
        def render(self, *a, **k):
            raise RuntimeError("shape mismatch: not an OOM")

    releases: list[int] = []
    _patch_render_seams(monkeypatch, _BoomRenderer(0), releases)
    H, W = _OOM_DET["pat_height"], _OOM_DET["pat_width"]
    score = build_render_score_fn(
        "dummy.sht", _OOM_DET,
        lambda f: np.zeros((H, W), np.float32) + 0.5)
    with pytest.raises(RuntimeError, match="not an OOM"):
        score([0], _q_axis_angle([0, 0, 1], 5.0)[None, :])
    assert releases == []                  # never entered OOM recovery


def test_unify_map_all_neg_inf_scores_keeps_dominant_and_flags_ambiguous():
    """If the scorer returns -inf for EVERY candidate (e.g. persistent GPU OOM
    on every render), unify_map must NOT flip to an arbitrary class: it keeps
    the dominant variant (map ends single-class) and flags the grain ambiguous.
    Guards the -inf − -inf == NaN slip-through in the margin comparison."""
    nr, nc = 6, 8
    q, phase, _truth = _speckled_map(nr, nc)
    new_q, report = unify_map(
        q, phase, nr, nc, 0, "m-3",
        lambda i, qq: np.full(len(np.atleast_1d(i)), float("-inf")))
    assert new_q is not None
    assert report["n_ambiguous"] >= 1
    # every ambiguous grain-unit was recorded with a JSON-safe (finite) margin
    for g in report["grains"]:
        assert np.isfinite(g["margin"])
    # despite the useless scorer the speckle is unified onto ONE (dominant) class
    reps = proper_coset_class_reps("m-3")
    cls = class_label_pixels(new_q, new_q[0], reps, "m-3")
    assert len(set(cls.tolist())) == 1


def test_unify_map_rescues_tiny_orphan():
    """A 1-px orphan whose orientation fits NO neighbour even modulo the
    supergroup (Hough total failure) adopts the surrounding grain's
    orientation."""
    nr, nc = 5, 5
    qA = _q_axis_angle([0, 0, 1], 5.0)
    q = np.tile(qA, (nr * nc, 1))
    center = 2 * nc + 2
    q[center] = _q_axis_angle([1, 2, 3], 33.0)             # garbage orientation
    new_q, report = unify_map(q, np.zeros(nr * nc, int), nr, nc, 0, "m-3",
                              lambda i, qq: np.full(len(np.atleast_1d(i)), 0.5))
    assert report["n_rescued"] == 1
    assert _ang(new_q[center], qA, "m-3") < 0.1


# ---------------------------------------------------------------------------
# Mixed symmetry representatives (2026-09-09): T is not normal in I, so an
# operator must act on a consistent member of each pixel's orbit
# ---------------------------------------------------------------------------

def _mixed_reps(q, rng, pg="m-3"):
    """Re-express every row of q by a random crystal-symmetry equivalent S·q:
    the same orientations, other members of their orbits — what an indexer's
    export stores per pixel."""
    S = _sym_quats(pg)
    out = q.copy()
    for i in range(q.shape[0]):
        out[i] = _qmul(S[rng.integers(0, len(S))][None, :], q[i][None, :])[0]
    return out


def test_nearest_representative_returns_orbit_member_closest_to_target():
    rng = np.random.default_rng(1)
    S = _sym_quats("m-3")
    q = np.stack([_q_axis_angle([1, 2, 3], 20.0 + 0.3 * i) for i in range(6)])
    mixed = _mixed_reps(q, rng)
    near = nearest_representative(mixed, q[0], S)
    for i in range(6):
        assert _ang(near[i], q[i], "m-3") < 1e-4                    # same orientation (float rounding ~1e-6)
        assert float(np.dot(near[i], q[0])) > 0.999                  # and the member next to the target


def test_snap_to_class_is_representative_independent_for_icosahedral_classes():
    reps = class_reps_for_phase("m-3")
    assert len(reps) == 6
    rng = np.random.default_rng(3)
    q_ref = _q_axis_angle([1, 2, 3], 20.0)
    truth = np.stack([_qmul(_q_axis_angle([1, 0, 0], 0.4 * i)[None, :], q_ref[None, :])[0]
                      for i in range(12)])                            # bent grain
    k_wrong = 3                                                      # a five-fold class
    meas = _mixed_reps(_qmul(reps[k_wrong][None, :], truth), rng)    # one variant, random representatives
    cls = class_label_pixels(meas, q_ref, reps, "m-3")
    assert set(cls.tolist()) == {k_wrong}                            # labels are representative-independent...
    naive = snap_to_class(meas, cls, 0, reps)                        # ...the bare operator is not
    assert max(_ang(naive[i], truth[i], "m-3") for i in range(12)) > 30.0
    fixed = snap_to_class(meas, cls, 0, reps, "m-3", q_ref)
    for i in range(12):
        assert _ang(fixed[i], truth[i], "m-3") < 0.1
    assert 4.0 < _ang(fixed[11], fixed[0], "m-3") < 4.8              # the 0.4°/px gradient survives


def test_unify_map_five_fold_grain_with_mixed_representatives_comes_out_uniform():
    """The ICAA20 crop1 defect (61 speckle px): a grain sitting in a five-fold
    basin whose stored quaternions are random members of their m-3 orbits.
    After unification every pixel must sit on the truth — no speckle."""
    nr, nc = 6, 8
    rng = np.random.default_rng(11)
    reps = class_reps_for_phase("m-3")
    q_ref = _q_axis_angle([0, 0, 1], 5.0)
    truth = np.stack([_qmul(_q_axis_angle([1, 0, 0], 0.05 * (r + c))[None, :], q_ref[None, :])[0]
                      for r in range(nr) for c in range(nc)])
    q = _mixed_reps(_qmul(reps[2][None, :], truth), rng)             # whole grain in five-fold class 2
    new_q, report = unify_map(q, np.zeros(nr * nc, int), nr, nc, 0, "m-3", _truth_scorer(truth))
    assert report["n_grains"] == 1 and report["n_flipped_units"] == 1
    d = [_ang(new_q[i], truth[i], "m-3") for i in range(nr * nc)]
    assert max(d) < 0.5, d


def test_adoption_with_mixed_representatives_lands_on_the_donor_branch():
    nr, nc = 8, 8
    rng = np.random.default_rng(5)
    q, phase, qA, blob = _blob_map(nr, nc)
    q[blob] = _mixed_reps(q[blob], rng)
    new_q, report = unify_map(q, phase, nr, nc, 0, "m-3",
                              _truth_scorer(np.tile(qA, (nr * nc, 1))))
    assert report["n_adopted"] == 1
    for f in blob:
        assert _ang(new_q[f], qA, "m-3") < 0.5
