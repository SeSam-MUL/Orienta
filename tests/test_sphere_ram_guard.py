"""The RAM preflight of the CPU path of the master sphere.

Without a GPU the sphere is built in RAM: about 2.2 GB at bw 256 and 7 GB at bw
384 (measured, see the comment above ``_RAM_NEEDED_GB``). On a 16 GB machine that
is the difference between a clear message and a computer that stops answering.
The expectations below are hand arithmetic on those figures and a 1.15 headroom:
256 needs 2.2 * 1.15 = 2.53 GB available, 384 needs 7.0 * 1.15 = 8.05 GB.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import psutil  # noqa: E402

from backend.api.problem import CODE_HEADER  # noqa: E402
from backend.api.services import sht_pattern_renderer as svc  # noqa: E402
from backend.api.services import sht_sphere  # noqa: E402

GB = 2**30


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(svc, "_RAM_RETAINED_GB", {})
    monkeypatch.setattr(svc, "_RAM_RSS_BASELINE_GB", None)
    svc._PHASE_CACHE.clear()
    sht_sphere.clear_cache()
    yield
    svc._PHASE_CACHE.clear()
    sht_sphere.clear_cache()


@pytest.fixture
def cpu_with(monkeypatch):
    """The renderer is on the CPU and psutil reports ``available`` GB free of 16."""
    dev = types.SimpleNamespace(type="cpu", index=None, __str__=lambda s: "cpu")
    monkeypatch.setattr(svc, "get_renderer", lambda: types.SimpleNamespace(device=dev))

    def _arm(available_gb, total_gb=16.0):
        monkeypatch.setattr(
            psutil, "virtual_memory",
            lambda: types.SimpleNamespace(available=int(available_gb * GB), total=int(total_gb * GB)))
    return _arm


class TestRamPreflight:
    def test_bw384_is_refused_when_less_than_8_05_gb_is_available(self, cpu_with):
        cpu_with(available_gb=8.0)                              # just under 7.0 * 1.15
        with pytest.raises(svc.SHTOutOfMemoryError) as e:
            svc._check_vram_for_bandwidth(384)
        assert "7.0 GB" in str(e.value) and "8.0 GB" in str(e.value), str(e.value)

    def test_bw384_passes_with_8_1_gb(self, cpu_with):
        cpu_with(available_gb=8.1)
        svc._check_vram_for_bandwidth(384)

    def test_bw256_needs_2_53_gb(self, cpu_with):
        cpu_with(available_gb=2.5)
        with pytest.raises(svc.SHTOutOfMemoryError):
            svc._check_vram_for_bandwidth(256)
        cpu_with(available_gb=2.6)
        svc._check_vram_for_bandwidth(256)

    def test_standard_and_none_are_never_refused(self, cpu_with):
        cpu_with(available_gb=0.0)
        svc._check_vram_for_bandwidth(128)
        svc._check_vram_for_bandwidth(None)

    def test_a_psutil_failure_does_not_refuse(self, cpu_with, monkeypatch):
        def boom():
            raise OSError("no counters")
        monkeypatch.setattr(psutil, "virtual_memory", boom)
        svc._check_vram_for_bandwidth(384)

    def test_a_gpu_is_not_checked_against_ram(self, monkeypatch):
        dev = types.SimpleNamespace(type="cuda", index=0, __str__=lambda s: "cuda:0")
        monkeypatch.setattr(svc, "get_renderer", lambda: types.SimpleNamespace(device=dev))
        import torch
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_a, **_k: (12 * GB, 12 * GB))

        asked = []

        def spy():
            # Recorded, not raised: the check swallows exceptions from psutil.
            asked.append(1)
            raise OSError("not needed")
        monkeypatch.setattr(psutil, "virtual_memory", spy)
        svc._check_vram_for_bandwidth(384)
        assert asked == [], "RAM was consulted for a GPU render"


class TestRetainedMemoryCounts:
    """A freed grid's memory stays with the process and the next load of that
    size reuses it, so it must not be counted as missing."""

    def _holding(self, monkeypatch, rss_now, baseline=1.0):
        """A process that started at ``baseline`` GB and holds ``rss_now`` GB."""
        monkeypatch.setattr(svc, "_RAM_RSS_BASELINE_GB", baseline)
        monkeypatch.setattr(svc, "_process_rss_gb", lambda: rss_now)

    def test_memory_a_finished_load_left_behind_is_credited(self, cpu_with, monkeypatch):
        cpu_with(available_gb=3.0)                               # far below 8.05
        with pytest.raises(svc.SHTOutOfMemoryError):
            svc._check_vram_for_bandwidth(384)
        svc._RAM_RETAINED_GB[384] = 6.0                          # 3.0 + 6.0 >= 8.05
        self._holding(monkeypatch, rss_now=8.0)                  # holds 7.0 above the baseline
        svc._check_vram_for_bandwidth(384)

    def test_a_credit_the_process_no_longer_holds_is_not_counted(self, cpu_with, monkeypatch):
        """After a Max run Windows may trim the working set or other programs take
        the memory; the recorded 6 GB must not be credited then."""
        cpu_with(available_gb=3.0)
        svc._RAM_RETAINED_GB[384] = 6.0
        self._holding(monkeypatch, rss_now=1.0)                  # back at the baseline: holds 0
        with pytest.raises(svc.SHTOutOfMemoryError):
            svc._check_vram_for_bandwidth(384)

    def test_the_credit_is_capped_by_what_is_held_now(self, cpu_with, monkeypatch):
        cpu_with(available_gb=3.0)
        svc._RAM_RETAINED_GB[384] = 6.0
        self._holding(monkeypatch, rss_now=5.0)                  # holds 4.0: 3.0 + 4.0 = 7.0 < 8.05
        with pytest.raises(svc.SHTOutOfMemoryError):
            svc._check_vram_for_bandwidth(384)
        self._holding(monkeypatch, rss_now=5.6)                  # holds 4.6: 3.0 + 4.6 = 7.6 < 8.05
        with pytest.raises(svc.SHTOutOfMemoryError):
            svc._check_vram_for_bandwidth(384)
        self._holding(monkeypatch, rss_now=7.0)                  # holds 6.0 = the credit: 9.0 >= 8.05
        svc._check_vram_for_bandwidth(384)

    def test_a_process_below_its_baseline_holds_nothing(self, cpu_with, monkeypatch):
        """Below its baseline the process holds 0, not a negative number: a
        negative credit would refuse a request the free RAM alone satisfies."""
        svc._RAM_RETAINED_GB[384] = 6.0
        self._holding(monkeypatch, rss_now=0.4, baseline=1.0)    # 0.6 below the baseline
        cpu_with(available_gb=3.0)
        with pytest.raises(svc.SHTOutOfMemoryError):             # no credit to help it
            svc._check_vram_for_bandwidth(384)
        cpu_with(available_gb=8.1)                               # 8.1 >= 8.05 on its own
        svc._check_vram_for_bandwidth(384)

    def test_an_unreadable_rss_gives_no_credit(self, cpu_with, monkeypatch):
        cpu_with(available_gb=3.0)
        svc._RAM_RETAINED_GB[384] = 6.0
        monkeypatch.setattr(svc, "_RAM_RSS_BASELINE_GB", 1.0)

        def boom():
            raise OSError("no counters")
        monkeypatch.setattr(svc, "_process_rss_gb", boom)
        with pytest.raises(svc.SHTOutOfMemoryError):
            svc._check_vram_for_bandwidth(384)

    def test_the_baseline_is_the_rss_before_the_first_tracked_load_only(self, cpu_with, monkeypatch, tmp_path):
        cpu_with(available_gb=64.0, total_gb=64.0)
        sht = tmp_path / "x.sht"
        sht.write_bytes(b"")
        sht2 = tmp_path / "y.sht"
        sht2.write_bytes(b"")
        rss = iter([1.5, 4.0, 4.0, 5.0])                         # load 1: 1.5 -> 4.0; load 2: 4.0 -> 5.0

        class Renderer:
            device = types.SimpleNamespace(type="cpu", index=None, __str__=lambda s: "cpu")

            def load_phase(self, *_a, **_k):
                return object()

        monkeypatch.setattr(svc, "_process_rss_gb", lambda: next(rss))
        monkeypatch.setattr(svc, "get_renderer", lambda: Renderer())
        svc.load_or_get_phase(str(sht), max_bandwidth=256)
        svc.load_or_get_phase(str(sht2), max_bandwidth=256)
        assert svc._RAM_RSS_BASELINE_GB == pytest.approx(1.5)

    def test_releasing_the_caches_forgets_the_baseline_too(self):
        svc._RAM_RSS_BASELINE_GB = 1.5
        svc.release_gpu_caches()
        assert svc._RAM_RSS_BASELINE_GB is None

    def test_a_retained_high_grid_does_not_pay_for_a_max_request(self, cpu_with, monkeypatch):
        cpu_with(available_gb=3.0)
        self._holding(monkeypatch, rss_now=10.0)                 # the process holds plenty: not the reason
        # Deliberately large: if a bw-256 leftover were credited to a bw-384
        # request, 3.0 + 5.5 = 8.5 would clear the 8.05 needed. It must not.
        svc._RAM_RETAINED_GB[256] = 5.5
        with pytest.raises(svc.SHTOutOfMemoryError):
            svc._check_vram_for_bandwidth(384)

    def test_a_load_records_how_much_the_process_grew(self, cpu_with, monkeypatch, tmp_path):
        cpu_with(available_gb=64.0, total_gb=64.0)
        sht = tmp_path / "x.sht"
        sht.write_bytes(b"")
        rss = iter([1.0, 6.5])                                   # before, after: grew by 5.5
        monkeypatch.setattr(svc, "_process_rss_gb", lambda: next(rss))

        class Renderer:
            device = types.SimpleNamespace(type="cpu", index=None, __str__=lambda s: "cpu")

            def load_phase(self, *_a, **_k):
                return object()

        monkeypatch.setattr(svc, "get_renderer", lambda: Renderer())
        svc.load_or_get_phase(str(sht), max_bandwidth=384)
        assert svc._RAM_RETAINED_GB[384] == pytest.approx(5.5)

    def _load_with_growth(self, monkeypatch, tmp_path, bw, before, after):
        sht = tmp_path / "x.sht"
        sht.write_bytes(b"")
        rss = iter([before, after])
        monkeypatch.setattr(svc, "_process_rss_gb", lambda: next(rss))

        class Renderer:
            device = types.SimpleNamespace(type="cpu", index=None, __str__=lambda s: "cpu")

            def load_phase(self, *_a, **_k):
                return object()

        monkeypatch.setattr(svc, "get_renderer", lambda: Renderer())
        svc.load_or_get_phase(str(sht), max_bandwidth=bw)

    def test_growth_from_a_parallel_allocation_cannot_exceed_what_a_load_needs(self, cpu_with, monkeypatch, tmp_path):
        """A parallel thread allocating 3 GB during a bw-256 load made the
        process grow 4.81 GB; credited whole, a later bw-256 load passed with
        0.5 GB free. The credit is capped at the 2.2 GB such a load needs."""
        cpu_with(available_gb=64.0, total_gb=64.0)
        self._load_with_growth(monkeypatch, tmp_path, 256, before=1.0, after=5.81)
        assert svc._RAM_RETAINED_GB[256] == pytest.approx(2.2)

    def test_a_smaller_growth_is_kept_as_measured(self, cpu_with, monkeypatch, tmp_path):
        cpu_with(available_gb=64.0, total_gb=64.0)
        self._load_with_growth(monkeypatch, tmp_path, 256, before=1.0, after=2.5)
        assert svc._RAM_RETAINED_GB[256] == pytest.approx(1.5)

    def test_releasing_the_gpu_caches_forgets_the_credit(self):
        svc._RAM_RETAINED_GB[384] = 6.0
        svc.release_gpu_caches()
        assert svc._RAM_RETAINED_GB == {}

    def test_a_standard_load_is_not_tracked(self, cpu_with, monkeypatch, tmp_path):
        sht = tmp_path / "x.sht"
        sht.write_bytes(b"")
        monkeypatch.setattr(svc, "_process_rss_gb", lambda: pytest.fail("RSS read for bw 128"))

        class Renderer:
            device = types.SimpleNamespace(type="cpu", index=None, __str__=lambda s: "cpu")

            def load_phase(self, *_a, **_k):
                return object()

        monkeypatch.setattr(svc, "get_renderer", lambda: Renderer())
        svc.load_or_get_phase(str(sht), max_bandwidth=128)
        assert svc._RAM_RETAINED_GB == {}


def test_the_route_answers_the_coded_short_422_when_the_ram_is_short(cpu_with, monkeypatch, tmp_path):
    monkeypatch.setenv("ORIENTA_ALLOWED_HOSTS", "testserver")
    from fastapi.testclient import TestClient

    from backend.api.main import app

    sht = tmp_path / "x.sht"
    sht.write_bytes(b"")
    monkeypatch.setattr(sht_sphere, "_resolve_sht", lambda _f: sht)
    cpu_with(available_gb=1.0)
    r = TestClient(app, raise_server_exceptions=False).get(
        "/api/database/sphere/x.sht?max_bandwidth=384&target_size=256")
    assert r.status_code == 422, r.text
    assert r.headers.get(CODE_HEADER.lower()) == "sphereOutOfMemory"
    assert "\n" not in r.json()["detail"] and len(r.json()["detail"]) < 160
