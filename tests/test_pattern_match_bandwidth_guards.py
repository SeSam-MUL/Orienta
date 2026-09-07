"""The two guards that would have kept the backend alive on 2026-09-07.

What happened: the user opened the Pattern-Match dialog at Quality bw=384 with
Compare phases on. The re-index built its Legendre tables at L=384 -- an
unnormalised recursion that overflows float64 at m=151, so the tables were
NaN -- and asked for ~88 GB of host commit; the render wanted 5.7 GB of VRAM
on a card with 0.8 GB free and thrashed at 100 % GPU; and both ran INSIDE the
async route, so the whole server was deaf for five minutes and then died,
taking the user's unsaved manual phase assignments with it.

Three consequences, pinned here:
  1. the Compare-phases RE-INDEX is capped at the numerically valid bandwidth;
  2. that cap is justified by the recursion itself, not by folklore;
  3. a render grid the GPU cannot hold is refused up front, with numbers.
(The fourth -- the route runs its GPU work via asyncio.to_thread -- is a
structural change verified by reading; it has no cheap unit test.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestCompareReindexCap:
    def test_high_quality_presets_are_capped_for_the_reindex(self):
        from backend.api.routes.indexing import (
            COMPARE_INDEX_BW_MAX, _compare_index_bandwidth,
        )
        assert COMPARE_INDEX_BW_MAX == 128
        assert _compare_index_bandwidth(384) == 128
        assert _compare_index_bandwidth(256) == 128
        assert _compare_index_bandwidth(128) == 128
        assert _compare_index_bandwidth(None) == 128

    def test_a_lower_request_is_honoured(self):
        from backend.api.routes.indexing import _compare_index_bandwidth
        assert _compare_index_bandwidth(64) == 64

    def test_the_cap_sits_below_where_the_legendre_recursion_overflows(self):
        """The reason for the number: at the cap the tables are finite, at the
        next preset up they are not. If someone ever fixes the recursion
        (normalised form), this test is what tells them the cap may move."""
        from backend.api.routes.indexing import COMPARE_INDEX_BW_MAX
        from backend.spherical_gpu.pipeline.indexer import _build_normalized_alf

        cos_theta = np.array([0.0, 0.3, 0.7, 0.99])
        with np.errstate(all="ignore"):
            ok = _build_normalized_alf(COMPARE_INDEX_BW_MAX, cos_theta)
            bad = _build_normalized_alf(256, cos_theta)
        assert np.all(np.isfinite(ok)), "tables are NOT finite at the cap"
        assert not np.all(np.isfinite(bad)), (
            "L=256 now builds finite tables -- the recursion was fixed; "
            "re-evaluate COMPARE_INDEX_BW_MAX"
        )


class TestVramGuard:
    @pytest.fixture
    def cuda_with(self, monkeypatch):
        """Pretend the renderer is on CUDA with a given amount of free VRAM."""
        import types
        from backend.api.services import sht_pattern_renderer as svc

        def _arm(free_gb, total_gb=12.0):
            dev = types.SimpleNamespace(type="cuda", index=0, __str__=lambda s: "cuda:0")
            monkeypatch.setattr(svc, "get_renderer", lambda: types.SimpleNamespace(device=dev))
            import torch
            monkeypatch.setattr(torch.cuda, "mem_get_info",
                                lambda *_a, **_k: (int(free_gb * 2**30), int(total_gb * 2**30)))
        return _arm

    def test_bw384_is_refused_when_the_card_is_full(self, cuda_with):
        from backend.api.services import sht_pattern_renderer as svc
        cuda_with(free_gb=0.8)
        with pytest.raises(svc.SHTRenderError) as e:
            svc._check_vram_for_bandwidth(384)
        msg = str(e.value)
        assert "5.7 GB" in msg and "0.8 GB" in msg, msg
        assert "256 or 128" in msg

    def test_bw384_passes_with_room(self, cuda_with):
        from backend.api.services import sht_pattern_renderer as svc
        cuda_with(free_gb=9.0)
        svc._check_vram_for_bandwidth(384)          # no raise

    def test_bw256_needs_less(self, cuda_with):
        from backend.api.services import sht_pattern_renderer as svc
        cuda_with(free_gb=2.5)
        svc._check_vram_for_bandwidth(256)          # 1.7 * 1.15 < 2.5
        cuda_with(free_gb=1.0)
        with pytest.raises(svc.SHTRenderError):
            svc._check_vram_for_bandwidth(256)

    def test_128_and_none_are_never_refused(self, cuda_with):
        from backend.api.services import sht_pattern_renderer as svc
        cuda_with(free_gb=0.0)
        svc._check_vram_for_bandwidth(128)
        svc._check_vram_for_bandwidth(None)

    def test_cpu_renderer_is_never_refused(self, monkeypatch):
        import types
        from backend.api.services import sht_pattern_renderer as svc
        dev = types.SimpleNamespace(type="cpu", index=None, __str__=lambda s: "cpu")
        monkeypatch.setattr(svc, "get_renderer", lambda: types.SimpleNamespace(device=dev))
        svc._check_vram_for_bandwidth(384)

    def test_a_memory_query_failure_does_not_refuse(self, monkeypatch):
        """Cannot measure -> do not pretend to know."""
        import types
        from backend.api.services import sht_pattern_renderer as svc
        dev = types.SimpleNamespace(type="cuda", index=0, __str__=lambda s: "cuda:0")
        monkeypatch.setattr(svc, "get_renderer", lambda: types.SimpleNamespace(device=dev))
        import torch
        def boom(*_a, **_k):
            raise RuntimeError("no context")
        monkeypatch.setattr(torch.cuda, "mem_get_info", boom)
        svc._check_vram_for_bandwidth(384)
