"""Tests for the spherical pseudo-symmetry resolution pipeline
(backend/spherical_gpu/pipeline/resolution.py)."""
from __future__ import annotations
import numpy as np


def q_axis_angle(axis, deg):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    a = np.radians(deg) / 2.0
    return np.array([np.cos(a), *(np.sin(a) * axis)])


def test_topk_distinct_dedupes_close_orientations():
    from backend.spherical_gpu.pipeline.resolution import topk_distinct
    A = q_axis_angle([0, 0, 1], 0.0)     # identity
    Ap = q_axis_angle([0, 0, 1], 1.0)    # 1 deg dup of A
    B = q_axis_angle([0, 0, 1], 50.0)
    Bp = q_axis_angle([0, 0, 1], 51.0)   # 1 deg dup of B
    C = q_axis_angle([1, 1, 0], 47.0)
    quats = np.stack([A, Ap, B, Bp, C])
    scores = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
    idx = topk_distinct(quats, scores, "m-3", topk=3, min_sep_deg=5.0)
    assert list(idx) == [0, 2, 4]        # A, B, C — dups dropped


def test_topk_distinct_respects_k():
    from backend.spherical_gpu.pipeline.resolution import topk_distinct
    quats = np.stack([q_axis_angle([0, 0, 1], d) for d in (0, 20, 40, 60, 80)])
    scores = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
    idx = topk_distinct(quats, scores, "m-3", topk=2, min_sep_deg=5.0)
    assert len(idx) == 2
    assert list(idx) == [0, 1]           # highest-score distinct ones


# ----------------------------------------------------------------------
# resolve_eulers — the integration wrapper (cubic = bit-identical passthrough)
# ----------------------------------------------------------------------
def test_resolve_eulers_cubic_passthrough_is_bit_identical():
    """For a non-pseudo-symmetric (true cubic m-3m) phase, resolve_eulers must
    return the raw eulers UNCHANGED and run NO extra compute — the validated
    spherical path stays bit-identical. indexer/patterns are None so the test
    also proves the resolver code path is never entered."""
    from backend.spherical_gpu.pipeline.resolution import resolve_eulers
    raw = np.array([[0.1, 0.2, 0.3], [1.0, 0.5, 2.0]], dtype=np.float64)
    out, info = resolve_eulers(
        patterns=None, raw_eulers=raw,
        cif_path="ignored", det_params={}, point_group="m-3m")
    assert np.array_equal(out, raw)
    assert info is None


def test_resolve_eulers_skips_when_no_cif():
    """If the CIF cannot be resolved (None/empty), the wrapper must fail safe and
    return the raw eulers rather than crash the whole indexing run."""
    from backend.spherical_gpu.pipeline.resolution import resolve_eulers
    raw = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
    out, info = resolve_eulers(
        patterns=None, raw_eulers=raw,
        cif_path="", det_params={}, point_group="m-3")
    assert np.array_equal(out, raw)
    assert info is None


def test_resolve_eulers_invokes_resolver_for_mmm(monkeypatch):
    """mmm (orthorhombic, z_rot==2) is now in the spherical-unreliable set, so the
    wrapper MUST enter the Hough resolution path — not the high-symmetry
    passthrough. The S-phase Al2CuMg is exactly this case."""
    import backend.spherical_gpu.pipeline.resolution as R
    from orix.quaternion import Rotation
    called = {}

    def fake_resolve_map(patterns, cif_path, det_params, point_group,
                         raw_eulers=None, progress=None):
        called["pg"] = point_group
        return {"resolved": np.asarray(Rotation.from_euler(raw_eulers).data),
                "n_fallback": 0, "method": "hough"}

    monkeypatch.setattr(R, "resolve_map", fake_resolve_map)
    raw = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
    out, info = R.resolve_eulers(
        patterns=np.zeros((1, 4, 4), dtype=np.float32), raw_eulers=raw,
        cif_path="some.cif", det_params={}, point_group="mmm")
    assert called.get("pg") == "mmm"
    assert info is not None and info.get("method") == "hough"


def test_resolve_eulers_gates_on_zrot(monkeypatch):
    """resolve_eulers prefers the master's z_rot: z_rot==2 fires even for a name
    not in the fallback set; z_rot!=2 is a passthrough even for an unreliable name
    (this is what makes the map path match the per-pattern z_rot==2 gate)."""
    import backend.spherical_gpu.pipeline.resolution as R
    from orix.quaternion import Rotation
    calls = []

    def fake(patterns, cif_path, det_params, point_group, raw_eulers=None, progress=None):
        calls.append(point_group)
        return {"resolved": np.asarray(Rotation.from_euler(raw_eulers).data),
                "n_fallback": 0, "method": "hough"}

    monkeypatch.setattr(R, "resolve_map", fake)
    raw = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
    p = np.zeros((1, 4, 4), dtype=np.float32)
    # z_rot==2 fires even for an unusual point-group name not in the fallback set
    _, info = R.resolve_eulers(patterns=p, raw_eulers=raw, cif_path="c.cif",
                               det_params={}, point_group="-4", z_rot=2)
    assert info is not None and calls == ["-4"]
    # z_rot!=2 is a passthrough even for an unreliable NAME (z_rot wins)
    calls.clear()
    out, info = R.resolve_eulers(patterns=p, raw_eulers=raw, cif_path="c.cif",
                                 det_params={}, point_group="mmm", z_rot=4)
    assert np.array_equal(out, raw) and info is None and not calls


def test_resolve_eulers_passthrough_for_working_symmetries(monkeypatch):
    """Classes the spherical correlation indexes correctly (high-sym cubic/tetra/
    hex AND the WORKING low-sym z_rot!=2 cases monoclinic 2/m + triclinic -1) must
    stay a bit-identical passthrough — resolve_map must NOT run for them."""
    import backend.spherical_gpu.pipeline.resolution as R

    def boom(*a, **k):
        raise AssertionError("resolve_map must not run for a working symmetry")

    monkeypatch.setattr(R, "resolve_map", boom)
    raw = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
    for pg in ("m-3m", "4/mmm", "6/mmm", "2/m", "-1", "3"):
        out, info = R.resolve_eulers(
            patterns=np.zeros((1, 4, 4), dtype=np.float32), raw_eulers=raw,
            cif_path="some.cif", det_params={}, point_group=pg)
        assert np.array_equal(out, raw) and info is None, pg


# ----------------------------------------------------------------------
# resolve_eulers_multiphase — per-phase Hough substitution for multi-phase maps
# ----------------------------------------------------------------------
def _fake_resolver(calls):
    """A resolve_eulers stand-in that marks every input euler (+1.0) so the test
    can see which pixels were rewritten, and records (point_group, n_pixels)."""
    def fake(patterns, raw_eulers, cif_path, det_params, point_group, z_rot=None, progress=None):
        calls.append((point_group, int(np.asarray(raw_eulers).shape[0])))
        return np.asarray(raw_eulers, dtype=np.float64) + 1.0, {"n_fallback": 0}
    return fake


def test_resolve_multiphase_only_z_rot2_pixels_change(monkeypatch):
    """Only the z_rot==2 phase's assigned pixels get Hough orientations; the
    high-symmetry phase's pixels stay bit-identical."""
    import backend.spherical_gpu.pipeline.resolution as R
    calls = []
    monkeypatch.setattr(R, "resolve_eulers", _fake_resolver(calls))
    raw = np.arange(18, dtype=np.float64).reshape(6, 3)
    pid = np.array([1, 1, 1, 2, 2, 2])           # phase1=mmm(z2), phase2=m-3m(z4)
    masters = [{"point_group": "mmm", "z_rot": 2, "formula": "S"},
               {"point_group": "m-3m", "z_rot": 4, "formula": "Al"}]
    pats = np.zeros((6, 4, 4), dtype=np.float32)
    out, info = R.resolve_eulers_multiphase(pats, raw, pid, masters, ["s.cif", "al.cif"], {})
    assert out is not None
    assert np.array_equal(out[:3], raw[:3] + 1.0)   # mmm pixels resolved
    assert np.array_equal(out[3:], raw[3:])         # Al pixels untouched
    assert info["resolved_phase_ids"] == {1}
    assert calls == [("mmm", 3)]                    # called once, on the 3 mmm pixels


def test_resolve_multiphase_skips_phase_with_no_pixels(monkeypatch):
    import backend.spherical_gpu.pipeline.resolution as R
    calls = []
    monkeypatch.setattr(R, "resolve_eulers", _fake_resolver(calls))
    raw = np.arange(12, dtype=np.float64).reshape(4, 3)
    pid = np.array([1, 1, 1, 1])                  # phase2 (mmm) won zero pixels
    masters = [{"point_group": "m-3m", "z_rot": 4}, {"point_group": "mmm", "z_rot": 2}]
    out, info = R.resolve_eulers_multiphase(np.zeros((4, 4, 4), np.float32), raw, pid,
                                            masters, ["a.cif", "s.cif"], {})
    assert out is None and calls == []           # nothing to resolve


def test_resolve_multiphase_skips_when_no_cif(monkeypatch):
    import backend.spherical_gpu.pipeline.resolution as R
    calls = []
    monkeypatch.setattr(R, "resolve_eulers", _fake_resolver(calls))
    raw = np.zeros((3, 3), dtype=np.float64)
    out, info = R.resolve_eulers_multiphase(
        np.zeros((3, 4, 4), np.float32), raw, np.array([1, 1, 1]),
        [{"point_group": "mmm", "z_rot": 2}], [None], {})
    assert out is None and calls == []


def test_resolve_multiphase_all_high_sym_returns_none(monkeypatch):
    import backend.spherical_gpu.pipeline.resolution as R
    calls = []
    monkeypatch.setattr(R, "resolve_eulers", _fake_resolver(calls))
    raw = np.zeros((4, 3), dtype=np.float64)
    masters = [{"point_group": "m-3m", "z_rot": 4}, {"point_group": "6/mmm", "z_rot": 6}]
    out, info = R.resolve_eulers_multiphase(
        np.zeros((4, 4, 4), np.float32), raw, np.array([1, 1, 2, 2]),
        masters, ["a.cif", "b.cif"], {})
    assert out is None and calls == [] and info["resolved_phase_ids"] == set()


def test_resolve_multiphase_single_phase_resolves_all(monkeypatch):
    """Single phase = one z_rot==2 phase covering all pixels → behaves like the
    single-phase resolve_eulers (all pixels resolved)."""
    import backend.spherical_gpu.pipeline.resolution as R
    calls = []
    monkeypatch.setattr(R, "resolve_eulers", _fake_resolver(calls))
    raw = np.arange(18, dtype=np.float64).reshape(6, 3)
    out, info = R.resolve_eulers_multiphase(
        np.zeros((6, 4, 4), np.float32), raw, np.ones(6, dtype=int),
        [{"point_group": "mmm", "z_rot": 2, "formula": "S"}], ["s.cif"], {})
    assert np.array_equal(out, raw + 1.0)
    assert info["resolved_phase_ids"] == {1} and calls == [("mmm", 6)]


# ----------------------------------------------------------------------
# END-TO-END on real SampleB / alpha-AlFeMnSi (skipped if data absent)
# ----------------------------------------------------------------------
import sys
from pathlib import Path
import pytest

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
def test_resolve_map_sampleb_fixes_wrong_spherical_variant():
    """On real SampleB intermetallic pixels, the resolver must CHANGE the wrong
    raw-spherical orientation to the Hough-consistent (correct) variant."""
    import numpy as np
    import torch
    import h5py
    import kikuchipy as kp
    from backend.spherical_gpu.pipeline.detector import DetectorGeometry
    from backend.spherical_gpu.pipeline.indexer import Tier1Indexer
    from backend.spherical_gpu.pipeline.sht_io import read_sht_master
    from backend.spherical_gpu.pipeline.resolution import resolve_map
    from backend.spherical_gpu.pseudosym import disorientation_deg
    from orix.quaternion import Rotation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geom = DetectorGeometry.from_params(_DET, device=device)
    master = read_sht_master(str(_ALPHA_SHT), device=device)
    idx = Tier1Indexer(geom, master, device, bandwidth=68)

    pixels = [(53, 63), (50, 60), (34, 15)]   # validated/representative grains
    ncols = 120
    with h5py.File(_SAMPLEB, "r") as f:
        dset = f["1/EBSD/Data/Processed Patterns"]
        frame_avg = np.stack([
            np.stack([dset[(r + dr) * ncols + (c + dc)].astype(np.float32)
                      for dr in (-1, 0, 1) for dc in (-1, 0, 1)]).mean(0)
            for r, c in pixels])

    # raw spherical argmax (the WRONG pseudo-variant) for comparison
    eul, _ = idx._index_batch(idx._run_preprocessing(torch.from_numpy(frame_avg).to(device).float()))
    raw = np.asarray(Rotation.from_euler(eul.cpu().numpy()).data)
    raw_eulers = eul.cpu().numpy().astype(np.float64)

    # New production behavior: pseudo-symmetric phases take the Hough orientation
    # (which renders strictly better than the spherical pseudo-variant — see the
    # resolution module docstring). resolve_map no longer re-runs the SHT cc.
    res = resolve_map(frame_avg, str(_ALPHA_CIF), _DET, "m-3", raw_eulers=raw_eulers)
    resolved, hough = res["resolved"], res["hough"]
    assert res["method"] == "hough"
    assert res["nmatch"].shape == (len(pixels),)

    # RE-BASELINED 2026-09-10, crystal-frame fix (tasks/fidelity/03_fix_round1.md).
    # Until the fix, the raw spherical answer was the SYSTEMATIC 90-degree cubic
    # coset partner of the truth on every pixel, so "raw is far from Hough" held
    # everywhere and this loop asserted d_raw > 25 on all three grains. With the
    # constant left C2<1 -1 0> removed from the decode, raw is a legitimate
    # answer again and what is left is the phase's REAL pseudo-symmetry:
    # measured d(raw, Hough) = 71.93 / 71.68 / 0.71 deg — the two big grains sit
    # on the five-fold pseudo-icosahedral variant (~71.7 deg, the approximant
    # ambiguity), while (34, 15) now agrees with Hough to 0.71 deg (it was
    # 1.65 deg before the half-bin decode fix of 2026-09-10). So the
    # invariant that survives is about the RESOLVER, not about raw being broken:
    # whatever raw says, resolve_map hands back the Hough orientation, and where
    # raw disagreed it really did change the answer.
    d_raw_all, d_res_all = [], []
    for i in range(len(pixels)):
        d_raw = disorientation_deg(raw[i], hough[i], "m-3")
        d_res = disorientation_deg(resolved[i], hough[i], "m-3")
        d_raw_all.append(d_raw)
        d_res_all.append(d_res)
        # the chosen orientation IS the Hough one (these grains index cleanly,
        # so no per-pixel fallback to raw is expected).
        assert d_res < 1.0, f"pixel {pixels[i]}: resolved is not the Hough orientation ({d_res:.1f})"
        if d_raw > 25.0:
            # a grain where raw picked another pseudo-variant: the resolver must
            # have MOVED the orientation, not merely relabelled it.
            d_moved = disorientation_deg(raw[i], resolved[i], "m-3")
            assert d_moved > 25.0, (
                f"pixel {pixels[i]}: raw is {d_raw:.1f} deg from Hough but the "
                f"resolver only moved it {d_moved:.1f} deg")
    # Guard on the fix itself: raw must never again be the systematic 90-degree
    # cubic partner of Hough. Under m-3 that partner sits at ~88-90 deg.
    assert not any(80.0 < d < 100.0 for d in d_raw_all), (
        f"raw spherical is back on the 90-degree cubic coset partner: "
        f"d(raw, Hough) = {[round(d, 2) for d in d_raw_all]} deg — the "
        f"crystal-frame correction in pipeline/_frame.py has regressed")

    # GROUND TRUTH: the chosen (Hough) orientation must RENDER the experimental
    # pattern better than the raw spherical pseudo-variant — this is *why* Hough
    # is correct (the user's "Hough gives a perfect result"). Without this, the
    # test would pass even if Hough were also wrong, as long as it differed from
    # raw. Render each orientation through the SHT forward model and compare NCC.
    from backend.spherical_gpu.pipeline.forward import PatternRenderer
    H, W = int(_DET["pat_height"]), int(_DET["pat_width"])
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    disc = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (min(cy, cx) * 0.96) ** 2

    def _dynbg(im):
        s = kp.signals.EBSD(im[None, None].astype(np.float32))
        s.remove_dynamic_background(operation="subtract", filter_domain="frequency")
        return s.data[0, 0].astype(np.float32)

    def _ncc(a, b):
        av = a[disc].astype(np.float64); bv = b[disc].astype(np.float64)
        av -= av.mean(); bv -= bv.mean()
        return float(np.dot(av, bv) / (np.linalg.norm(av) * np.linalg.norm(bv) + 1e-12))

    rnd = PatternRenderer(device=device)
    pgrid = rnd.load_phase(str(_ALPHA_SHT), max_bandwidth=128)
    pc = (float(geom.xpc), float(geom.ypc), float(geom.L))

    def _render_ncc(quat, exp):
        q = torch.tensor(np.asarray(quat)[:4], dtype=torch.float64)
        P = rnd.render(pgrid, q, pc, (H, W), _DET["pixel_size"],
                       tilt_deg=_DET["sample_tilt"], det_tilt_deg=_DET["tilt"]).numpy()
        return _ncc(exp, _dynbg(P))

    nccs_hough, nccs_raw = [], []
    for i in range(len(pixels)):
        exp = _dynbg(frame_avg[i])
        nh, nr = _render_ncc(hough[i], exp), _render_ncc(raw[i], exp)
        nccs_hough.append(nh); nccs_raw.append(nr)
        # RE-BASELINED TWICE, both times because raw got BETTER, never because
        # a bar was in the way.
        #   shipped defect  : raw 0.19-0.21 (the 90-degree partner, noise floor)
        #   crystal-frame fix: raw 0.507 / 0.501 / 0.208, Hough 0.516/0.480/0.296
        #   half-bin fix     : raw 0.613 / 0.609 / 0.280, Hough unchanged
        # Hough does not go through the spherical decode, so its three numbers
        # are the same in all three rows -- which is exactly why "Hough is at
        # least as good as raw" could not survive: it was a statement about raw
        # being broken. Raw now WINS on the two big grains by 0.10 and 0.13.
        # What is asserted here is therefore the claim that still holds and that
        # this loop exists to protect: the Hough orientation the resolver hands
        # back must actually render the measured pattern, on every pixel, rather
        # than collapse. Measured 0.516 / 0.480 / 0.296.
        assert nh > 0.25, (
            f"pixel {pixels[i]}: Hough render-NCC {nh:.3f} -- the resolver's "
            f"chosen orientation does not render this pattern at all")
    # ...and the typical pixel renders well (median ~0.5; tolerates one bad grain).
    assert float(np.median(nccs_hough)) > 0.4, (
        f"median Hough render-NCC {np.median(nccs_hough):.3f} too low — Hough "
        f"orientations are not rendering the patterns")
    # MEASURED, and deliberately NOT asserted: on these three grains the Hough
    # substitution is now behind raw on the median by 0.129 (0.480 vs 0.609) —
    # it was ahead only while raw carried the crystal-frame defect, and the
    # half-bin fix widened the gap. That is the input to the separate decision
    # on retiring the m-3 / z_rot==2 Hough substitution (tasks/fidelity/03_plan.md
    # risk 2, tasks/fidelity/03_fix_round2.md); asserting either direction here
    # would freeze a question three pixels cannot answer.
    # The fix in its own right: raw must render the measured patterns, not sit
    # at the ~0.20 noise floor it did as the 90-degree partner.
    assert float(np.median(nccs_raw)) > 0.45, (
        f"raw spherical render-NCC median {np.median(nccs_raw):.3f} — below the "
        f"0.609 measured after the half-bin decode fix (0.501 before it, ~0.20 "
        f"while raw was the 90-degree partner)")


# ----------------------------------------------------------------------
# MULTI-PHASE E2E on the real 7050 S-phase (MgCuAl2 mmm + Al m-3m)
# ----------------------------------------------------------------------
_7050 = _ROOT / ("Test_data/batch_test/7050EBSD 70502_R Arbeitsbereich 1 "
                 "Elementverteilungsdaten 5.h5oina")
_SHT_MGCUAL2 = _ROOT / "Database/EBSD_SHT_Database/sd_1814127/MgCuAl2 (sd_1814127) [oS16] {20kV}.sht"
_SHT_AL = _ROOT / "Database/EBSD_SHT_Database/Al/Al (Al) [cF4] {20kV}.sht"
_CIF_MGCUAL2 = _ROOT / "Database/CIF_Library/sd_1814127.cif"
_HAVE_MP = all(p.is_file() for p in (_7050, _SHT_MGCUAL2, _SHT_AL, _CIF_MGCUAL2))


@pytest.mark.skipif(not _HAVE_MP, reason="7050 S-phase / MgCuAl2 / Al data not present")
def test_resolve_multiphase_e2e_7050_fixes_only_sphase():
    """Real 2-phase Spherical competition (MgCuAl2 z_rot=2 + Al z_rot=4) on the
    7050 S-phase: resolve_eulers_multiphase must correct ONLY the S-phase pixels'
    orientation to Hough (render-NCC ~0.2 -> ~0.69) and leave Al pixels untouched."""
    import numpy as np
    import torch
    import kikuchipy as kp
    from safe_loader import load_ebsd_safe
    from backend.api.routes.indexing import build_spherical_det_params
    from backend.spherical_gpu.pipeline.detector import DetectorGeometry
    from backend.spherical_gpu.pipeline.indexer import Tier1Indexer
    from backend.spherical_gpu.pipeline.sht_io import read_sht_master
    from backend.spherical_gpu.pipeline.forward import PatternRenderer
    from backend.spherical_gpu.pipeline.resolution import resolve_eulers_multiphase
    from orix.quaternion import Rotation

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sig = load_ebsd_safe(str(_7050), verbose=False)
    det_params = build_spherical_det_params(sig, sig.detector, str(_7050))
    H, W = int(det_params["pat_height"]), int(det_params["pat_width"])
    NCOLS = 136
    patterns = np.ascontiguousarray(np.asarray(sig.data)).reshape(-1, H, W)
    geom = DetectorGeometry.from_params(det_params, device=dev)
    idxS = Tier1Indexer(geom, read_sht_master(str(_SHT_MGCUAL2), device=dev), dev, bandwidth=128)
    idxA = Tier1Indexer(geom, read_sht_master(str(_SHT_AL), device=dev), dev, bandwidth=128)
    rnd = PatternRenderer(device=dev)
    pgS = rnd.load_phase(str(_SHT_MGCUAL2), max_bandwidth=128)
    pgA = rnd.load_phase(str(_SHT_AL), max_bandwidth=128)
    pc = (float(geom.xpc), float(geom.ypc), float(geom.L))
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    disc = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (min(cy, cx) * 0.96) ** 2

    def dynbg(im):
        s = kp.signals.EBSD(im[None, None].astype(np.float32))
        s.remove_dynamic_background(operation="subtract", filter_domain="frequency")
        return s.data[0, 0].astype(np.float32)

    def fa(flat):
        r, c = flat // NCOLS, flat % NCOLS
        return np.stack([patterns[(r + dr) * NCOLS + (c + dc)].astype(np.float32)
                         for dr in (-1, 0, 1) for dc in (-1, 0, 1)]).mean(0)

    def ncc(a, b):
        av = a[disc].astype(np.float64); bv = b[disc].astype(np.float64)
        av -= av.mean(); bv -= bv.mean()
        return float(av @ bv / (np.linalg.norm(av) * np.linalg.norm(bv) + 1e-12))

    def rncc(pg, eu, exp):
        q = torch.tensor(np.asarray(Rotation.from_euler(np.asarray(eu).reshape(1, 3))
                                    .data).reshape(4)[:4], dtype=torch.float64)
        P = rnd.render(pg, q, pc, (H, W), det_params["pixel_size"],
                       tilt_deg=det_params["sample_tilt"], det_tilt_deg=det_params["tilt"]).numpy()
        return ncc(exp, dynbg(P))

    def sph(idx, raw):
        eu, sc = idx._index_batch(idx._run_preprocessing(torch.tensor(raw[None], device=dev).float()))
        return float(sc.reshape(-1)[0]), eu.cpu().numpy().reshape(3)

    pix = [(9, 67), (9, 66), (10, 67), (8, 67),    # S-phase grain
           (15, 30), (28, 128), (5, 100)]          # Al matrix
    fr = [fa(r * NCOLS + c) for (r, c) in pix]
    raw_eul = np.zeros((len(pix), 3)); pid = np.zeros(len(pix), dtype=int)
    for k, raw in enumerate(fr):
        rS, eS = sph(idxS, raw); rA, eA = sph(idxA, raw)
        if rS >= rA:
            pid[k] = 1; raw_eul[k] = eS
        else:
            pid[k] = 2; raw_eul[k] = eA
    masters = [{"point_group": "mmm", "z_rot": 2, "formula": "MgCuAl2"},
               {"point_group": "m-3m", "z_rot": 4, "formula": "Al"}]
    out, info = resolve_eulers_multiphase(
        np.stack(fr).astype(np.float32), raw_eul, pid, masters,
        [str(_CIF_MGCUAL2), None], det_params)

    # The S-phase competition must actually win its grain (else the test data
    # changed); only then is the multi-phase resolution the thing under test.
    assert (pid[:4] == 1).all(), "S-phase grain pixels did not win the phase competition"
    assert out is not None and info["resolved_phase_ids"] == {1}
    for k, (r, c) in enumerate(pix):
        exp = dynbg(fr[k]); pgw = pgS if pid[k] == 1 else pgA
        if pid[k] == 1:
            # S-phase pixel: orientation corrected to Hough -> renders well now.
            assert not np.array_equal(out[k], raw_eul[k])
            assert rncc(pgw, out[k], exp) > 0.4, f"S-phase pixel {(r, c)} not corrected"
        else:
            # Al pixel: z_rot=4, must be left exactly as the spherical output.
            assert np.array_equal(out[k], raw_eul[k]), f"Al pixel {(r, c)} was altered"
