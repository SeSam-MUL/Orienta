"""Unprocessed AZtec patterns are int16 (about 600..6500). They stay int16 in memory, and every cast
to 8 bit has to rescale by the global min/max: a plain ``astype(np.uint8)`` wraps modulo 256 and
turns a pattern into noise. PyEBSDIndex does read int16, but finds other bands than in the 8 bit
rescale of the same patterns (the Oxford reference scan, pyebsdindex 0.3.9.1: 2 of 9 patterns differ by 0.65 and 36
degrees), so the Hough paths hand it the rescale.

The real-data tests read 9 patterns of Oxford reference scan file and skip when it is absent."""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services.load_choices import LoadVariant, to_uint8  # noqa: E402
from tests.test_load_choices import _sampleb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AL_CIF = ROOT / "Database" / "CIF_Library" / "Al.cif"
AL_MASTER = ROOT / "Database" / "EBSD_H5_Cache" / "Al" / "Al_master_E20kV_npx500.h5"
AL_SHT = ROOT / "Database" / "EBSD_SHT_Database" / "Al" / "Al (Al) [cF4] {20kV}.sht"
AL_SHT_OTHER = (ROOT / "Database" / "EBSD_SHT_Database" / "Al"
                / "AlCu (Al2Cu_mp-985806_conventional_standard) [cF4] {20kV}.sht")
N_PATTERNS = 9


def _ramp_signal(dtype=np.int16, lo=600, hi=6500, shape=(2, 3, 4, 5)):
    import kikuchipy as kp

    n = int(np.prod(shape))
    return kp.signals.EBSD(np.linspace(lo, hi, n).round().astype(dtype).reshape(shape))


# --------------------------------------------------------------------------- #
# The casts
# --------------------------------------------------------------------------- #

def test_signal_patterns_3d_does_not_wrap_int16():
    from backend.api.routes.indexing import _signal_patterns_3d

    sig = _ramp_signal()
    pats, (ny, nx, sy, sx) = _signal_patterns_3d(sig)
    assert (ny, nx, sy, sx) == (2, 3, 4, 5)
    assert pats.dtype == np.uint8 and pats.shape == (6, 4, 5)
    flat = pats.ravel().astype(int)
    assert flat[0] == 0 and flat[-1] == 255
    assert (np.diff(flat) >= 0).all(), "the rescale of a ramp must stay monotone (a wrap is a saw tooth)"
    assert len(np.unique(flat)) > 100


def test_signal_patterns_3d_leaves_uint8_and_uint16_as_they_are():
    from backend.api.routes.indexing import _signal_patterns_3d

    u8 = _ramp_signal(np.uint8, 3, 250)
    pats, _ = _signal_patterns_3d(u8)
    assert pats.dtype == np.uint8 and np.array_equal(pats, np.asarray(u8.data).reshape(6, 4, 5))
    u16 = _ramp_signal(np.uint16, 600, 6500)
    pats, _ = _signal_patterns_3d(u16)
    assert pats.dtype == np.uint16 and np.array_equal(pats, np.asarray(u16.data).reshape(6, 4, 5))


@pytest.mark.parametrize("make", [
    lambda: np.linspace(600, 6500, 120).round().astype(np.int16).reshape(6, 4, 5),
    lambda: np.linspace(0, 60000, 120).round().astype(np.uint16).reshape(6, 4, 5),
    lambda: (np.linspace(-3.5, 9.25, 120, dtype=np.float32).reshape(6, 4, 5)),
    lambda: np.where(np.arange(120).reshape(6, 4, 5) % 7 == 0, np.nan,
                     np.linspace(-1, 1, 120).reshape(6, 4, 5)).astype(np.float32),
    lambda: np.full((6, 4, 5), 900, np.int16),
    lambda: np.arange(120).astype(np.uint8).reshape(6, 4, 5),
])
@pytest.mark.parametrize("chunk", [1, 4, 6, 1000])
def test_to_uint8_chunked_equals_to_uint8(make, chunk):
    from backend.api.services.load_choices import to_uint8_chunked

    a = make()
    got = to_uint8_chunked(a, chunk=chunk)
    assert got.dtype == np.uint8 and got.shape == a.shape
    assert np.array_equal(got, to_uint8(a))


def test_to_uint8_chunked_uses_the_global_range_not_the_range_of_a_chunk():
    from backend.api.services.load_choices import to_uint8_chunked

    a = np.zeros((4, 2, 2), np.int16)
    a[0] = 200
    a[3] = 1000
    got = to_uint8_chunked(a, chunk=1)
    assert got[0].max() == 51 and got[3].max() == 255      # a per-chunk range would give 0 and 0


def test_for_pyebsdindex_rescales_signed_integers_and_leaves_the_rest():
    from backend.api.services.load_choices import for_pyebsdindex

    i16 = np.linspace(600, 6500, 40).astype(np.int16).reshape(2, 4, 5)
    out = for_pyebsdindex(i16)
    assert out.dtype == np.uint8 and np.array_equal(out, to_uint8(i16))
    for other in (i16.astype(np.uint8), i16.astype(np.uint16), i16.astype(np.float32)):
        assert for_pyebsdindex(other) is other


# --------------------------------------------------------------------------- #
# The pattern image route
# --------------------------------------------------------------------------- #

@pytest.fixture
def pattern_client(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.api.routes import ebsd_viewer as ev

    holder = {}
    monkeypatch.setattr(ev, "_get_active_signal", lambda: holder["signal"])
    monkeypatch.setattr(ev, "_compute_include_mask", lambda *a, **k: None)
    app = FastAPI()
    app.include_router(ev.router, prefix="/api/ebsd")
    client = TestClient(app)
    client.holder = holder
    return client


def _grey_levels(client, row=0, col=0):
    from PIL import Image

    resp = client.get(f"/api/ebsd/pattern/{row}/{col}")
    assert resp.status_code == 200, resp.text
    png = Image.open(io.BytesIO(base64.b64decode(resp.json()["image"])))
    return np.asarray(png)


def test_pattern_image_route_int16(pattern_client):
    import kikuchipy as kp

    ramp = np.linspace(600, 6500, 12 * 13).round().astype(np.int16).reshape(12, 13)
    pattern_client.holder["signal"] = kp.signals.EBSD(np.stack([ramp] * 4).reshape(2, 2, 12, 13))
    img = _grey_levels(pattern_client)
    assert len(np.unique(img)) > 50, "an int16 pattern must be shown with its grey levels, not saturated"
    assert img.min() == 0 and img.max() == 255


def test_pattern_image_route_int16_spanning_more_than_half_the_int16_range(pattern_client):
    import kikuchipy as kp

    ramp = np.linspace(-20000, 20000, 12 * 13).round().astype(np.int16).reshape(12, 13)
    pattern_client.holder["signal"] = kp.signals.EBSD(np.stack([ramp] * 4).reshape(2, 2, 12, 13))
    img = _grey_levels(pattern_client).astype(int)
    assert len(np.unique(img)) > 50
    assert img.min() == 0 and img.max() == 255
    assert (np.diff(img.ravel()) >= 0).all(), "a subtraction that overflows int16 makes a saw tooth"


# --------------------------------------------------------------------------- #
# The indexers on real unprocessed patterns
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def sampleb_unprocessed():
    """9 indexed unprocessed (int16) Oxford reference scan patterns with their own PCs, read lazily."""
    path = _sampleb()
    if path is None:
        pytest.skip("EBSD_SampleB real file is not available here")
    import h5py

    import safe_loader

    mp = pytest.MonkeyPatch()
    mp.setenv("KIKUCHIPY_LAZY_THRESHOLD_BYTES", "1")
    try:
        sig = safe_loader.load_ebsd_safe(str(path), verbose=False, variant=LoadVariant("1", "unprocessed"))
        with h5py.File(path, "r") as f:
            phase = f["1/EBSD/Data/Phase"][:]
        hits = np.where(phase == 1)[0]
        idx = hits[::len(hits) // N_PATTERNS][:N_PATTERNS]
        n_cols = sig.data.shape[1]
        pats = np.stack([np.asarray(sig.data[i // n_cols, i % n_cols]) for i in idx])
        pcs = np.asarray(sig.detector.pc).reshape(-1, 3)[idx]
        det = sig.detector
    finally:
        mp.undo()
    assert pats.dtype == np.int16 and len(pats) == N_PATTERNS
    return {"pats": pats, "pcs": pcs, "detector": det, "path": path}


def _angle_deg(a, b):
    """Disorientation under the cubic point group (the phase of these tests is Al); a plain rotation
    angle would count every symmetry-equivalent answer as a difference."""
    from orix.quaternion import Orientation
    from orix.quaternion.symmetry import Oh

    return np.asarray(Orientation(a.rotations, symmetry=Oh).angle_with(
        Orientation(b.rotations, symmetry=Oh), degrees=True)).ravel()


def test_hough_same_on_int16_and_rescaled(sampleb_unprocessed):
    import kikuchipy as kp
    from orix.crystal_map import PhaseList

    import ebsd_utils as eu
    from indexing_controller import IndexingConfig, IndexingMethod, hough_index_patterns

    if not AL_CIF.exists():
        pytest.skip("Al.cif not available")
    pats, pcs, base = (sampleb_unprocessed[k] for k in ("pats", "pcs", "detector"))
    phase = eu.hough_phase_from_cif(str(AL_CIF))
    phase.name = "Al"

    def run(p):
        sig = kp.signals.EBSD(p.reshape(3, 3, *p.shape[1:]))
        det = kp.detectors.EBSDDetector(
            shape=p.shape[1:], pc=pcs.reshape(3, 3, 3), sample_tilt=base.sample_tilt,
            tilt=base.tilt, px_size=base.px_size)
        sig.detector = det
        return hough_index_patterns(
            signal=sig, phase_list=PhaseList([phase]), detector=det,
            config=IndexingConfig(method=IndexingMethod.HOUGH, n_bands=9)).xmap

    raw, rescaled = run(pats), run(to_uint8(pats))
    assert (np.asarray(rescaled.phase_id) == 0).all(), "the control (8 bit) must index the phase"
    assert np.array_equal(raw.phase_id, rescaled.phase_id)
    assert _angle_deg(raw, rescaled).max() < 0.5


def test_dictionary_same_on_int16_and_rescaled(sampleb_unprocessed):
    import kikuchipy as kp

    from indexing_controller import IndexingConfig, dictionary_index_patterns

    if not AL_MASTER.exists():
        pytest.skip("Al master pattern not available")
    pats, pcs, base = (sampleb_unprocessed[k] for k in ("pats", "pcs", "detector"))
    det = kp.detectors.EBSDDetector(
        shape=pats.shape[1:], pc=pcs.mean(axis=0), sample_tilt=base.sample_tilt,
        tilt=base.tilt, px_size=base.px_size)
    master = kp.load(str(AL_MASTER))

    def run(p):
        sig = kp.signals.EBSD(p.reshape(3, 3, *p.shape[1:]))
        sig.detector = det
        cfg = IndexingConfig()
        cfg.compute_mode = "cpu"
        cfg.angular_step_deg = 8.0
        return dictionary_index_patterns(signal=sig, dictionary=master, config=cfg, detector=det)

    raw, rescaled = run(pats), run(to_uint8(pats))
    assert np.array_equal(raw.xmap.phase_id, rescaled.xmap.phase_id)
    assert _angle_deg(raw.xmap, rescaled.xmap).max() < 0.5


def _spherical_eulers(patterns, sampleb, device, how, tmp_path=None, phases=1):
    import h5py
    import kikuchipy as kp
    import torch
    from orix.quaternion import Orientation, Rotation
    from orix.quaternion.symmetry import Oh

    from backend.api.routes.indexing import build_spherical_det_params
    from backend.spherical_gpu.backend import BackendConfig, PhaseConfig, SphericalGPUBackend

    pcs, base = sampleb["pcs"], sampleb["detector"]
    sig = kp.signals.EBSD(patterns.reshape(3, 3, *patterns.shape[1:]))
    det = kp.detectors.EBSDDetector(
        shape=patterns.shape[1:], pc=pcs.mean(axis=0), sample_tilt=base.sample_tilt,
        tilt=base.tilt, px_size=base.px_size)
    params = build_spherical_det_params(sig, det, str(sampleb["path"]))
    shts = [AL_SHT, AL_SHT_OTHER][:phases]
    backend = SphericalGPUBackend(BackendConfig(
        phases=[PhaseConfig(sht_file=str(f), bandwidth=68) for f in shts], device=torch.device(device)))
    if how == "array":
        res = backend.index_array(np.ascontiguousarray(patterns), params)
    else:
        path = tmp_path / f"{how}_{patterns.dtype}_{phases}.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("1/EBSD/Data/Patterns", data=patterns)
        res = (backend.index_h5(str(path), params) if how == "h5"
               else backend.index_h5_dataset(str(path), "1/EBSD/Data/Patterns", params))
    euler = np.asarray(res.euler_xyz.numpy(), dtype=np.float64).reshape(-1, 3)
    return Orientation(Rotation.from_euler(euler), symmetry=Oh)


def _spherical_same(sampleb, device, how, tmp_path=None, phases=1):
    raw = _spherical_eulers(sampleb["pats"], sampleb, device, how, tmp_path, phases)
    ref = _spherical_eulers(to_uint8(sampleb["pats"]), sampleb, device, how, tmp_path, phases)
    return np.asarray(raw.angle_with(ref, degrees=True)).ravel()


@pytest.mark.parametrize("how,phases", [
    ("array", 1), ("array", 2), ("h5", 1), ("h5_dataset", 1), ("h5_dataset", 2)])
def test_spherical_same_on_int16_and_rescaled_on_the_cpu(sampleb_unprocessed, tmp_path, how, phases):
    """The spherical indexer's offset-sensitive pre-processing (a zero-padded Gaussian background) makes an
    int16 scan (offset ~600) differ from its 8 bit rescale unless the rescale is applied on the way in."""
    pytest.importorskip("torch")
    if not AL_SHT.exists() or (phases == 2 and not AL_SHT_OTHER.exists()):
        pytest.skip("Al master (.sht) not available")
    diff = _spherical_same(sampleb_unprocessed, "cpu", how, tmp_path, phases)
    assert diff.max() < 0.5, np.round(diff, 1)


@pytest.mark.gpu
def test_spherical_gpu_same_on_int16_and_rescaled(sampleb_unprocessed, tmp_path):
    """GPU run only (opt in with ORIENTA_RUN_GPU_COMPARISON=1): the same comparison on the CUDA device."""
    import os

    torch = pytest.importorskip("torch")
    if os.environ.get("ORIENTA_RUN_GPU_COMPARISON") != "1":
        pytest.skip("GPU comparison runs only in the GPU run (ORIENTA_RUN_GPU_COMPARISON=1)")
    if not torch.cuda.is_available() or not AL_SHT.exists():
        pytest.skip("needs CUDA and the Al master (.sht)")
    for how in ("array", "h5", "h5_dataset"):
        diff = _spherical_same(sampleb_unprocessed, "cuda", how, tmp_path)
        assert diff.max() < 0.5, (how, np.round(diff, 1))


def test_the_emsphinx_export_rescales_signed_integers(tmp_path):
    import h5py

    from indexing_controller import _export_patterns_for_indexebsd

    data = np.linspace(600, 6500, 6 * 4 * 5).round().astype(np.int16).reshape(6, 4, 5)
    src, dst = tmp_path / "src.h5", tmp_path / "dst.h5"
    with h5py.File(src, "w") as f:
        f.create_dataset("1/EBSD/Data/Unprocessed Patterns", data=data)
    _export_patterns_for_indexebsd(str(src), str(dst), "1/EBSD/Data/Unprocessed Patterns")
    with h5py.File(dst, "r") as f:
        out = f["1/EBSD/Data/Unprocessed Patterns"][:]
    assert out.dtype == np.uint8 and np.array_equal(out, to_uint8(data))
    assert out.min() == 0 and out.max() == 255


def test_signed_pattern_range_and_patterns_to_float32():
    from backend.api.services.load_choices import patterns_to_float32, signed_pattern_range

    data = np.linspace(-300, 5000, 90).round().astype(np.int16).reshape(9, 2, 5)
    assert signed_pattern_range(data, chunk=2) == (-300.0, 5000.0)
    assert signed_pattern_range(data.astype(np.uint8)) is None
    assert signed_pattern_range(data.astype(np.float32)) is None
    rng = signed_pattern_range(data)
    got = np.concatenate([patterns_to_float32(data[s:s + 4], rng) for s in range(0, 9, 4)])
    assert got.dtype == np.float32 and np.array_equal(got, to_uint8(data).astype(np.float32))
    plain = patterns_to_float32(data[:2])
    assert plain.dtype == np.float32 and np.array_equal(plain, data[:2].astype(np.float32))
    assert not patterns_to_float32(np.full((2, 2), 7, np.int16), (7.0, 7.0)).any()


# --------------------------------------------------------------------------- #
# What PyEBSDIndex receives, at every call site (no real data needed)
# --------------------------------------------------------------------------- #

class _Seen(Exception):
    pass


def _int16_nickel():
    """kikuchipy's small nickel set (3 x 3 uint8 patterns) as unprocessed-looking int16 (600..5700)."""
    import kikuchipy as kp

    nickel = kp.data.nickel_ebsd_small()
    data = (np.asarray(nickel.data).astype(np.int16) * 20 + 600).astype(np.int16)
    sig = kp.signals.EBSD(data, detector=nickel.detector)
    assert sig.data.dtype == np.int16 and data.max() > 255
    return sig


@pytest.fixture
def pyebsdindex_input(monkeypatch):
    """The dtype of every pattern stack that reaches kikuchipy's Hough entry points or the indexer."""
    import kikuchipy.signals as ks
    from pyebsdindex.ebsd_index import EBSDIndexer

    seen = []

    def record(label, orig, stop):
        def rec(self, *a, **k):
            seen.append((label, np.dtype((self.data if hasattr(self, "data") else a[0]).dtype)))
            if stop:
                raise _Seen()
            return orig(self, *a, **k)
        return rec

    monkeypatch.setattr(ks.EBSD, "hough_indexing", record("hough_indexing", ks.EBSD.hough_indexing, False))
    monkeypatch.setattr(ks.EBSD, "hough_indexing_optimize_pc",
                        record("optimize_pc", ks.EBSD.hough_indexing_optimize_pc, True))

    orig_index = EBSDIndexer.index_pats

    def index_pats(self, patsin=None, *a, **k):
        seen.append(("index_pats", np.dtype(patsin.dtype)))
        return orig_index(self, patsin, *a, **k)

    monkeypatch.setattr(EBSDIndexer, "index_pats", index_pats)
    return seen


def _ni_controller(tmp_path):
    from pc_controller import PCController
    from tests.test_pc_per_pixel_single_pattern import _ni_cif

    sig = _int16_nickel()
    ctrl = PCController()
    ctrl.load_phase(_ni_cif(tmp_path))
    ctrl.detector = sig.detector.deepcopy()
    for r in range(2):
        ctrl.add_pattern((r, 0), np.asarray(sig.data[r, 0]).copy())
    assert ctrl.patterns[0][1].dtype == np.int16
    return ctrl, sig


def _site_pc_index_pattern(tmp_path):
    ctrl, _ = _ni_controller(tmp_path)
    ctrl.index_pattern(0)


def _site_pc_phases_of_patterns(tmp_path):
    ctrl, _ = _ni_controller(tmp_path)
    ctrl.phases_of_patterns([p for _, p in ctrl.patterns], coords=[c for c, _ in ctrl.patterns])


def _site_compute_ci(tmp_path):
    import ebsd_utils as eu

    ctrl, _ = _ni_controller(tmp_path)
    eu.compute_ci(ctrl.detector_at((0, 0)), ctrl.phase_list, ctrl._ensure_indexer((0, 0)),
                  ctrl.patterns[0][1])


def _site_optimize_pc(tmp_path):
    import ebsd_utils as eu

    ctrl, _ = _ni_controller(tmp_path)
    try:
        eu.optimize_pc(ctrl.detector_at((0, 0)), ctrl._ensure_indexer((0, 0)), ctrl.patterns[0][1])
    except _Seen:
        pass


def _site_index_and_simulate(tmp_path):
    from backend.api.routes.pcrefinement import _index_and_simulate

    ctrl, _ = _ni_controller(tmp_path)
    _index_and_simulate(ctrl, 0, sim_cache={})


def _site_index_selected_hough(tmp_path):
    from indexing_controller import _index_selected_hough

    ctrl, sig = _ni_controller(tmp_path)
    _index_selected_hough(sig, [(0, 0)], ctrl.phase_list, ctrl._ensure_indexer((0, 0)),
                          ctrl.detector_at((0, 0)))


def _site_hough_index_patterns(tmp_path):
    from indexing_controller import IndexingConfig, IndexingMethod, hough_index_patterns

    ctrl, sig = _ni_controller(tmp_path)
    hough_index_patterns(
        signal=sig, phase_list=ctrl.phase_list, detector=sig.detector,
        config=IndexingConfig(method=IndexingMethod.HOUGH, n_bands=9))


@pytest.mark.parametrize("site", [
    _site_pc_index_pattern, _site_pc_phases_of_patterns, _site_compute_ci, _site_optimize_pc,
    _site_index_and_simulate, _site_index_selected_hough, _site_hough_index_patterns],
    ids=lambda f: f.__name__[len("_site_"):])
def test_pyebsdindex_gets_the_8_bit_rescale_of_int16_patterns(site, tmp_path, pyebsdindex_input):
    site(tmp_path)
    assert pyebsdindex_input, "the call site never reached PyEBSDIndex: the test checks nothing"
    assert {dt for _, dt in pyebsdindex_input} == {np.dtype(np.uint8)}, pyebsdindex_input


def test_to_uint8_chunked_clips_values_outside_the_handed_range():
    """A stack edited while the range pass ran holds values outside the scan's (min, max); they clip
    at 0 and 255 like the click path does, they do not wrap into noise."""
    from backend.api.services.load_choices import to_uint8_chunked

    a = np.array([[[1000, 6000, 50]]], np.int16)
    got = to_uint8_chunked(a, value_range=(100, 5100))
    # by hand: (1000-100)*255/5000 = 45.9 -> 46; 6000 is above the range -> 255; 50 is below -> 0
    assert got.tolist() == [[[46, 255, 0]]]
