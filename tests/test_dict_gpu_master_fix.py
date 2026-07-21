"""Regression tests for the GPU dictionary Path-A master-loading fix.

The indexing route loads masters with a plain ``kp.load(path)``, which yields
the STEREOGRAPHIC UPPER hemisphere. The GPU forward projection (Path A) needs
the SQUARE-LAMBERT BOTH-hemisphere master and a contiguous array. Before the
fix this failed two/three ways:
  1. hemisphere: shape (n_energy, npx, npx) → "expected (2, npx, npx)".
  2. projection: sampling a stereographic master with the Lambert grid is
     geometrically wrong (silent — patterns look plausible but don't match).
  3. strides:   kikuchipy slices are negative-strided → torch.from_numpy raises.

``_select_energy_slice`` now reloads the master as Lambert+both and returns a
contiguous (2, npx, npx) array. The decisive correctness check is that Path A
then matches the kikuchipy CPU reference (Path B) at NCC ≈ 1.0.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
AL_MASTER = ROOT / "Database" / "EBSD_H5_Cache" / "Al" / "Al_master_E20kV_npx500.h5"

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_CUDA = False


def _load_app_style_master():
    """Master loaded exactly as the indexing route does (stereographic, upper)."""
    import kikuchipy as kp
    return kp.load(str(AL_MASTER))


@pytest.mark.skipif(not AL_MASTER.exists(), reason="Al master pattern not present")
def test_app_style_master_is_single_hemisphere_stereographic():
    """Guards the premise: plain kp.load gives the 'wrong' form for Path A."""
    m = _load_app_style_master()
    assert m.hemisphere == "upper"
    assert m.projection == "stereographic"
    assert m.data.ndim == 3 and m.data.shape[0] != 2  # (n_energy, npx, npx)


@pytest.mark.skipif(not AL_MASTER.exists(), reason="Al master pattern not present")
def test_select_energy_slice_reloads_lambert_both_contiguous():
    from backend.dict_gpu._pcadi.master_to_dict import _select_energy_slice
    m = _load_app_style_master()
    data = _select_energy_slice(m, energy=20.0)
    assert data.ndim == 3 and data.shape[0] == 2       # both hemispheres
    assert data.flags["C_CONTIGUOUS"]                  # torch.from_numpy-safe
    assert data.dtype == np.float32


@pytest.mark.skipif(not AL_MASTER.exists(), reason="Al master pattern not present")
def test_recover_master_path_roundtrips():
    from backend.dict_gpu._pcadi.master_to_dict import _recover_master_path
    m = _load_app_style_master()
    recovered = _recover_master_path(m)
    assert recovered is not None
    assert Path(recovered).resolve() == AL_MASTER.resolve()


@pytest.mark.skipif(not (_HAS_CUDA and AL_MASTER.exists()),
                    reason="needs CUDA + Al master pattern")
def test_path_a_matches_reference_with_app_style_master():
    """Decisive: Path A on the app-style (wrong-form) master must match the
    kikuchipy CPU reference (Path B) — proving the reload fixes projection."""
    import kikuchipy as kp
    from orix.sampling import get_sample_fundamental
    from backend.dict_gpu._pcadi.master_to_dict import gpu_master_to_dict

    detector = kp.detectors.EBSDDetector(shape=(60, 60), pc=(0.5, 0.5, 0.5),
                                         sample_tilt=70.0)
    m_appstyle = _load_app_style_master()             # stereographic, upper
    m_reference = kp.load(str(AL_MASTER), projection="lambert", hemisphere="both")

    rots = get_sample_fundamental(resolution=10, point_group=m_appstyle.phase.point_group)[:24]

    a = gpu_master_to_dict(m_appstyle, rots, detector, energy=20.0,
                           device="cuda", _use_path_b=False).cpu().numpy()
    b = gpu_master_to_dict(m_reference, rots, detector, energy=20.0,
                           device="cuda", _use_path_b=True).cpu().numpy()

    assert a.shape == b.shape

    def ncc(x, y):
        x = x.ravel() - x.mean()
        y = y.ravel() - y.mean()
        return float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))

    scores = np.array([ncc(a[i], b[i]) for i in range(len(a))])
    assert np.median(scores) > 0.999, f"median NCC {np.median(scores):.5f} < 0.999"
    assert scores.min() > 0.99, f"min NCC {scores.min():.5f} < 0.99"


@pytest.mark.skipif(not AL_MASTER.exists(), reason="Al master pattern not present")
def test_cpu_dict_projects_master_into_dictionary():
    """CPU path (compute_mode='cpu') must project a raw master into a
    detector-shaped dictionary. Before the fix it passed the (1001,1001) master
    straight to kikuchipy → "Experimental (H,W) and dictionary (1001,1001)
    signal shapes must be identical"."""
    import kikuchipy as kp
    import numpy as np
    from indexing_controller import dictionary_index_patterns, IndexingConfig

    # tiny synthetic experimental signal (small detector → fast get_patterns)
    exp = kp.signals.EBSD(np.random.default_rng(0).integers(
        0, 255, size=(4, 4, 40, 40), dtype=np.uint8))
    detector = kp.detectors.EBSDDetector(shape=(40, 40), pc=(0.5, 0.5, 0.5),
                                         sample_tilt=70.0)
    master = _load_app_style_master()  # raw master (stereographic/upper)

    cfg = IndexingConfig()
    cfg.compute_mode = "cpu"
    cfg.angular_step_deg = 12.0  # coarse → quick
    mask = np.zeros((4, 4), bool)
    mask[1:3, 1:3] = True  # 4 px

    res = dictionary_index_patterns(
        signal=exp, dictionary=master, config=cfg,
        selection_mask=mask, detector=detector)
    assert res.xmap.size == 4
    assert list(res.xmap.phases.names)  # phase list populated


def test_cpu_dict_leaves_pregenerated_dictionary_untouched(monkeypatch):
    """A pre-generated dictionary signal (no get_patterns) must NOT be
    re-projected — the master-projection branch keys on get_patterns()."""
    import kikuchipy as kp
    import numpy as np
    import indexing_controller as ic

    exp = kp.signals.EBSD(np.zeros((2, 2, 8, 8), dtype=np.uint8))
    pregen_dict = kp.signals.EBSD(np.zeros((5, 8, 8), dtype=np.uint8))
    assert not hasattr(pregen_dict, "get_patterns")

    called = {"projected": False}

    def _boom(*a, **k):
        called["projected"] = True
        raise AssertionError("should not project a pre-generated dictionary")

    monkeypatch.setattr(ic, "_dictionary_signal_from_master", _boom)
    # dictionary_indexing will fail on the dummy data, but the point is only
    # that the projection branch is NOT taken for a get_patterns-less signal.
    try:
        ic.dictionary_index_patterns(
            signal=exp, dictionary=pregen_dict, config=ic.IndexingConfig())
    except Exception:
        pass
    assert called["projected"] is False
