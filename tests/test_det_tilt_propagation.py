"""The detector elevation must reach the renderer, not just the sample tilt.

``PatternRenderer.render``/``render_batch`` build their projection from
``alpha = 90 - sample_tilt + det_tilt`` and default ``det_tilt_deg`` to 0.
Three service modules used to thread only ``tilt_deg`` (the SAMPLE tilt) down
to the renderer, so every pattern they simulated came out rotated by exactly
the detector elevation of the dataset — silently, because 0 is a legal value.

Measured on real data at a FIXED (correct) orientation: rendering with the
correct detector tilt raised NCC by +0.210, while a 3 deg tilt error raised it
by only +0.001. The render-NCC peak is ~2 deg FWHM, so a wrong camera
elevation lands off the peak entirely. Oxford detectors commonly ship a
non-zero elevation, so this is not an exotic configuration.

These tests monkeypatch the renderer and inspect the kwargs it receives.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# A deliberately odd angle: it can only appear in the renderer call if it was
# read from the detector dict and threaded through.
DET_TILT = 12.5
SAMPLE_TILT = 70.0
PAT = 8


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------

def _fake_result(n_rows=2, n_cols=2, sht_path="/fake/Al.sht", with_det_tilt=True):
    """Tiny fake IndexingResult, all pixels indexed to phase 0."""
    result = MagicMock()
    result.original_shape = (n_rows, n_cols)
    result.selection_mask = np.ones((n_rows, n_cols), dtype=bool)
    det = {
        "pat_height": PAT, "pat_width": PAT,
        "pc_x": 0.5, "pc_y": 0.5, "pc_z": 0.6,
        "vendor": "Bruker", "pixel_size": 70.0,
        "sample_tilt": SAMPLE_TILT, "binning": 1,
    }
    if with_det_tilt:
        det["tilt"] = DET_TILT
    result.metadata = {
        "indexing_method": "spherical",
        "sht_paths_by_phase": {0: sht_path},
        "detector_geometry": det,
    }
    xmap = MagicMock()
    xmap.phase_id = np.zeros(n_rows * n_cols, dtype=np.int16)
    xmap.shape = (n_rows, n_cols)
    rot = MagicMock()
    rot.size = n_rows * n_cols
    rot.to_euler.return_value = np.zeros((n_rows * n_cols, 3))
    rot.data = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_rows * n_cols, 1))
    xmap.rotations = rot
    phase = MagicMock()
    phase.name = "Al"
    xmap.phases = {0: phase}
    result.xmap = xmap
    return result


def _pattern(seed=0):
    """A non-degenerate pattern (constant patterns give a 0-norm NCC)."""
    rng = np.random.RandomState(seed)
    return rng.rand(PAT, PAT).astype(np.float32)


def _batch_renderer(n=4):
    """MagicMock renderer whose render/render_batch return usable tensors."""
    r = MagicMock()
    r.render_batch.return_value = torch.tensor(
        np.stack([_pattern(i) for i in range(n)]), dtype=torch.float32
    )
    r.render.side_effect = lambda **kw: torch.tensor(_pattern(1), dtype=torch.float64)
    return r


def _det_tilts_seen(mock_method):
    """All det_tilt_deg values a mocked render/render_batch was called with."""
    assert mock_method.call_args_list, "renderer was never called"
    return [c.kwargs.get("det_tilt_deg", "<MISSING>")
            for c in mock_method.call_args_list]


# ---------------------------------------------------------------------------
# forward_diagnostics
# ---------------------------------------------------------------------------

def test_full_diagnostics_passes_det_tilt_to_render_batch():
    """compute_full_diagnostics -> _render_chunk_four_variants -> render_batch."""
    from backend.api.services import forward_diagnostics as fd
    result = _fake_result()
    renderer = _batch_renderer()

    with patch.object(fd, "load_or_get_phase", return_value=MagicMock()), \
         patch.object(fd, "get_renderer", return_value=renderer), \
         patch.object(fd, "_get_experimental_pattern", return_value=_pattern(0)):
        fd.compute_full_diagnostics(result, max_bandwidth=64)

    seen = _det_tilts_seen(renderer.render_batch)
    assert seen == [DET_TILT] * len(seen), seen
    # The sample tilt must stay separate, not be replaced by the detector tilt.
    assert all(c.kwargs["tilt_deg"] == SAMPLE_TILT
               for c in renderer.render_batch.call_args_list)


def test_full_diagnostics_per_pixel_pc_path_passes_det_tilt():
    """The refined per-pixel-PC branch renders one pixel at a time; same rule."""
    from backend.api.services import forward_diagnostics as fd
    result = _fake_result()
    result.metadata["pc_per_pixel"] = np.tile(
        np.array([4.0, 4.0, 15000.0]), (2, 2, 1)
    )
    renderer = _batch_renderer(n=1)

    with patch.object(fd, "load_or_get_phase", return_value=MagicMock()), \
         patch.object(fd, "get_renderer", return_value=renderer), \
         patch.object(fd, "_get_experimental_pattern", return_value=_pattern(0)):
        fd.compute_full_diagnostics(result, max_bandwidth=64)

    seen = _det_tilts_seen(renderer.render_batch)
    assert seen == [DET_TILT] * len(seen), seen


def test_full_diagnostics_defaults_to_zero_without_tilt_key():
    """A detector dict with no 'tilt' key keeps the old behaviour (0.0)."""
    from backend.api.services import forward_diagnostics as fd
    result = _fake_result(with_det_tilt=False)
    renderer = _batch_renderer()

    with patch.object(fd, "load_or_get_phase", return_value=MagicMock()), \
         patch.object(fd, "get_renderer", return_value=renderer), \
         patch.object(fd, "_get_experimental_pattern", return_value=_pattern(0)):
        fd.compute_full_diagnostics(result, max_bandwidth=64)

    assert _det_tilts_seen(renderer.render_batch) == [0.0] * 4


def test_thumbnail_pair_passes_det_tilt():
    """The anomaly-browser thumbnails re-render and must use the same geometry."""
    from backend.api.services import forward_diagnostics as fd
    result = _fake_result()
    renderer = MagicMock()
    renderer.render.return_value = torch.zeros(64, 64, dtype=torch.float32)

    with patch.object(fd, "load_or_get_phase", return_value=MagicMock()), \
         patch.object(fd, "get_renderer", return_value=renderer), \
         patch.object(fd, "_get_experimental_pattern", return_value=_pattern(0)):
        fd.thumbnail_pair(result, row=0, col=0, size=64)

    assert _det_tilts_seen(renderer.render) == [DET_TILT]


# ---------------------------------------------------------------------------
# sht_pattern_renderer
# ---------------------------------------------------------------------------

def test_forward_ncc_map_passes_det_tilt(tmp_path, monkeypatch):
    """compute_forward_ncc_map reads 'tilt' from detector_geometry."""
    from backend.api.services import sht_pattern_renderer as spr
    import tools.pattern_comparison as pc_mod

    sht = tmp_path / "Al.sht"          # must exist: the service stats the path
    sht.write_bytes(b"placeholder")
    result = _fake_result(sht_path=str(sht))
    renderer = _batch_renderer()

    monkeypatch.setattr(spr, "load_or_get_phase", lambda *a, **k: MagicMock())
    monkeypatch.setattr(spr, "get_renderer", lambda: renderer)
    monkeypatch.setattr(pc_mod, "get_experimental_pattern",
                        lambda r, row, col: _pattern(0))

    spr.compute_forward_ncc_map(result, max_bandwidth=64)

    assert _det_tilts_seen(renderer.render_batch) == [DET_TILT]
    assert renderer.render_batch.call_args.kwargs["tilt_deg"] == SAMPLE_TILT


def test_render_pattern_to_png_b64_forwards_det_tilt(monkeypatch):
    """The single-pattern entry point already had the parameter; keep it wired."""
    from backend.api.services import sht_pattern_renderer as spr
    renderer = MagicMock()
    renderer.render.return_value = torch.tensor(_pattern(0), dtype=torch.float32)
    monkeypatch.setattr(spr, "load_or_get_phase", lambda *a, **k: MagicMock())
    monkeypatch.setattr(spr, "get_renderer", lambda: renderer)

    spr.render_pattern_to_png_b64(
        sht_path="/fake/Al.sht",
        orientation_quat=torch.tensor([1.0, 0, 0, 0], dtype=torch.float64),
        pc_emsoft=(0.0, 0.0, 15000.0),
        detector_shape=(PAT, PAT),
        pixel_size_um=70.0,
        tilt_deg=SAMPLE_TILT,
        det_tilt_deg=DET_TILT,
    )
    assert _det_tilts_seen(renderer.render) == [DET_TILT]


# ---------------------------------------------------------------------------
# refinement
# ---------------------------------------------------------------------------

def test_refine_per_pixel_passes_det_tilt():
    """Stage 1: the LM loop and its finite-difference J_p both render."""
    from backend.api.services.refinement import refine_per_pixel
    renderer = _batch_renderer()
    I_exp = torch.tensor(np.stack([_pattern(3)]), dtype=torch.float64)
    refine_per_pixel(
        renderer=renderer, grid=MagicMock(),
        I_exp=I_exp,
        q0=torch.tensor([[1.0, 0, 0, 0]], dtype=torch.float64),
        pc0=torch.tensor([[4.0, 4.0, 15000.0]], dtype=torch.float64),
        detector_shape=(PAT, PAT), pixel_size_um=70.0,
        tilt_deg=SAMPLE_TILT, det_tilt_deg=DET_TILT,
        max_iter=2,
    )
    seen = _det_tilts_seen(renderer.render)
    assert seen == [DET_TILT] * len(seen), seen
    assert all(c.kwargs["tilt_deg"] == SAMPLE_TILT
               for c in renderer.render.call_args_list)


def test_refine_r_only_passes_det_tilt():
    """Stage 3 renders at a fixed PC but the same detector geometry."""
    from backend.api.services.refinement import refine_r_only
    renderer = _batch_renderer()
    I_exp = torch.tensor(np.stack([_pattern(3)]), dtype=torch.float64)
    refine_r_only(
        renderer=renderer, grid=MagicMock(),
        I_exp=I_exp,
        q0=torch.tensor([[1.0, 0, 0, 0]], dtype=torch.float64),
        pc_fixed=torch.tensor([[4.0, 4.0, 15000.0]], dtype=torch.float64),
        detector_shape=(PAT, PAT), pixel_size_um=70.0,
        tilt_deg=SAMPLE_TILT, det_tilt_deg=DET_TILT,
        max_iter=2,
    )
    seen = _det_tilts_seen(renderer.render)
    assert seen == [DET_TILT] * len(seen), seen


def test_compute_full_refinement_reads_det_tilt_from_detector_dict(monkeypatch):
    """The orchestrator is the entry point that turns det['tilt'] into a kwarg.

    Stage 1 is intercepted: capturing the kwargs it was called with is enough,
    and it keeps the test off the GPU.
    """
    from backend.api.services import refinement as rf
    captured = {}

    class _Stop(RuntimeError):
        pass

    def _spy(**kwargs):
        captured.update(kwargs)
        raise _Stop()

    monkeypatch.setattr(rf, "refine_per_pixel", _spy)
    monkeypatch.setattr(rf, "load_or_get_phase", lambda *a, **k: MagicMock())
    monkeypatch.setattr(rf, "get_renderer", lambda: MagicMock())
    monkeypatch.setattr(rf, "_get_experimental_pattern",
                        lambda r, row, col: _pattern(0))

    result = _fake_result()
    with pytest.raises(_Stop):
        rf.compute_full_refinement(result, smoothness_lambda=0.01)

    assert captured["det_tilt_deg"] == DET_TILT
    assert captured["tilt_deg"] == SAMPLE_TILT


def test_compute_full_refinement_defaults_to_zero_without_tilt_key(monkeypatch):
    from backend.api.services import refinement as rf
    captured = {}

    class _Stop(RuntimeError):
        pass

    def _spy(**kwargs):
        captured.update(kwargs)
        raise _Stop()

    monkeypatch.setattr(rf, "refine_per_pixel", _spy)
    monkeypatch.setattr(rf, "load_or_get_phase", lambda *a, **k: MagicMock())
    monkeypatch.setattr(rf, "get_renderer", lambda: MagicMock())
    monkeypatch.setattr(rf, "_get_experimental_pattern",
                        lambda r, row, col: _pattern(0))

    result = _fake_result(with_det_tilt=False)
    with pytest.raises(_Stop):
        rf.compute_full_refinement(result, smoothness_lambda=0.01)

    assert captured["det_tilt_deg"] == 0.0


# ---------------------------------------------------------------------------
# Static guard: catches NEW render call sites the tests above don't reach
# ---------------------------------------------------------------------------

def test_no_render_call_site_omits_det_tilt():
    """Every ``.render(...)`` / ``.render_batch(...)`` in the three services
    must pass ``det_tilt_deg`` explicitly.

    The monkeypatch tests only cover today's call sites; this one fails when a
    new one is added without the parameter, which is exactly how the original
    bug spread across three modules.
    """
    from backend.api.services import (
        forward_diagnostics, refinement, sht_pattern_renderer,
    )
    offenders = []
    for module in (forward_diagnostics, refinement, sht_pattern_renderer):
        src_path = Path(inspect.getsourcefile(module))
        tree = ast.parse(src_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr not in ("render", "render_batch"):
                continue
            if "det_tilt_deg" not in {kw.arg for kw in node.keywords}:
                offenders.append(f"{src_path.name}:{node.lineno}")
    assert not offenders, (
        "render call sites without det_tilt_deg: " + ", ".join(offenders)
    )
