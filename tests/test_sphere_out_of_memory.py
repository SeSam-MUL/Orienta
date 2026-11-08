"""The master-sphere route when the GPU (or RAM) cannot hold the grid.

Reported: switching the Database Browser sphere to "High" gave an error, and
the error was either a refusal with numbers the cache had inflated, or a whole
TorchScript trace. What these tests pin:

* the VRAM preflight gives PyTorch's cached-but-free blocks back before it
  asks the driver, so space that is only *reserved* does not cause a refusal;
* a real out-of-memory error (``torch.cuda.OutOfMemoryError``, a TorchScript
  ``RuntimeError``, ``MemoryError``) becomes one short coded 422;
* nothing of the trace reaches the response.

No GPU, no .sht from ``Database/``: the renderer is replaced by a fake.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from backend.api.problem import CODE_HEADER  # noqa: E402
from backend.api.services import sht_pattern_renderer as svc  # noqa: E402
from backend.api.services import sht_sphere  # noqa: E402

TRACE = (
    "The following operation failed in the TorchScript interpreter.\n"
    "Traceback of TorchScript (most recent call last):\n"
    '  File "code/__torch__/forward.py", line 41, in inverse_sht\n'
    "RuntimeError: CUDA out of memory. Tried to allocate 3.50 GiB"
)


def _cuda_device():
    return types.SimpleNamespace(type="cuda", index=0, __str__=lambda s: "cuda:0")


def _cpu_device():
    return types.SimpleNamespace(type="cpu", index=None, __str__=lambda s: "cpu")


# ---------------------------------------------------------------- preflight

class TestPreflightReturnsTheCacheFirst:
    def test_space_that_is_only_reserved_is_not_refused(self, monkeypatch):
        """0.3 GB look free until empty_cache() runs, 10.3 GB afterwards."""
        monkeypatch.setattr(svc, "get_renderer", lambda: types.SimpleNamespace(device=_cuda_device()))
        state = {"emptied": 0}

        def empty_cache():
            state["emptied"] += 1

        def mem_get_info(*_a, **_k):
            free = 10.3 if state["emptied"] else 0.3
            return int(free * 2**30), int(12 * 2**30)

        monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
        monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)

        svc._check_vram_for_bandwidth(256)            # needs 1.8 * 1.15 GB
        assert state["emptied"] == 1

    def test_a_card_that_is_really_full_is_still_refused(self, monkeypatch):
        """empty_cache() must not turn the guard into a no-op."""
        monkeypatch.setattr(svc, "get_renderer", lambda: types.SimpleNamespace(device=_cuda_device()))
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        monkeypatch.setattr(torch.cuda, "mem_get_info",
                            lambda *_a, **_k: (int(0.5 * 2**30), int(12 * 2**30)))
        with pytest.raises(svc.SHTOutOfMemoryError):
            svc._check_vram_for_bandwidth(256)

    def test_cpu_renderer_never_touches_the_cuda_cache(self, monkeypatch):
        monkeypatch.setattr(svc, "get_renderer", lambda: types.SimpleNamespace(device=_cpu_device()))
        called = []
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: called.append(1))
        import psutil
        monkeypatch.setattr(psutil, "virtual_memory",
                            lambda: types.SimpleNamespace(available=64 * 2**30, total=64 * 2**30))
        svc._check_vram_for_bandwidth(384)
        assert called == []


# ------------------------------------------------------- what counts as OOM

class TestIsOutOfMemory:
    def test_torchscript_runtime_error(self):
        assert svc.is_out_of_memory(RuntimeError(TRACE))

    def test_memory_error(self):
        assert svc.is_out_of_memory(MemoryError())

    def test_the_cpu_oom_this_torch_really_raises(self):
        """Not a remembered message: ask the installed torch for a CPU allocation
        no machine can satisfy (8 PiB) and feed its own error to the predicate.
        On the shipped Windows torch 2.11 that is
        ``DefaultCPUAllocator: not enough memory``; the test follows the
        installed version instead of pinning one wording."""
        with pytest.raises(RuntimeError) as real:
            torch.empty(2**50, dtype=torch.float64)
        assert svc.is_out_of_memory(real.value), str(real.value)

    def test_the_linux_allocator_wording(self):
        # torch's alloc_cpu.cpp on Linux ("... can't allocate memory: you tried
        # to allocate N bytes. Error code 12 (Cannot allocate memory)"). It
        # cannot be produced on this machine, so it is the source's text.
        assert svc.is_out_of_memory(RuntimeError(
            "[enforce fail at alloc_cpu.cpp:117] data. DefaultCPUAllocator: "
            "can't allocate memory: you tried to allocate 9 bytes. "
            "Error code 12 (Cannot allocate memory)"))

    def test_a_failed_cublas_workspace(self):
        # The wording spherical_gpu.runtime.is_oom_error already matches.
        assert svc.is_out_of_memory(RuntimeError(
            "CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`"))

    def test_the_cause_is_followed(self):
        try:
            try:
                raise RuntimeError(TRACE)
            except RuntimeError as inner:
                raise ValueError("wrapped") from inner
        except ValueError as outer:
            assert svc.is_out_of_memory(outer)

    def test_the_wording_on_another_exception_type_is_not_oom(self):
        """Only a RuntimeError (what torch raises) is trusted for its wording: a
        ValueError that merely mentions the phrase -- say, in a file name --
        is not an out-of-memory error."""
        assert not svc.is_out_of_memory(ValueError("cannot read 'not enough memory.sht'"))
        assert not svc.is_out_of_memory(KeyError("CUDA out of memory"))

    def test_the_operating_system_reports_oom_on_an_oserror(self):
        class WinNotEnoughMemory(OSError):      # WinError 8, whichever platform runs the test
            winerror = 8

        class WinOutOfMemory(OSError):          # WinError 14
            winerror = 14

        assert svc.is_out_of_memory(WinNotEnoughMemory("Not enough memory resources are available"))
        assert svc.is_out_of_memory(WinOutOfMemory("Not enough storage is available"))
        import errno
        assert svc.is_out_of_memory(OSError(errno.ENOMEM, "Cannot allocate memory"))

    def test_other_oserrors_are_not_oom(self):
        import errno
        assert not svc.is_out_of_memory(OSError(errno.ENOENT, "No such file"))

        class WinAccessDenied(OSError):
            winerror = 5

        assert not svc.is_out_of_memory(WinAccessDenied("Access is denied"))

    def test_an_unrelated_error_is_not_oom(self):
        assert not svc.is_out_of_memory(RuntimeError("shape mismatch (3, 4) vs (5,)"))
        assert not svc.is_out_of_memory(ValueError("bad file"))


# ------------------------------------------------------------- load mapping

class _Boom:
    """A renderer whose load fails with ``exc``."""

    device = _cpu_device()

    def __init__(self, exc):
        self._exc = exc

    def load_phase(self, *_a, **_k):
        raise self._exc


@pytest.fixture
def clean_caches():
    svc._PHASE_CACHE.clear()
    sht_sphere.clear_cache()
    yield
    svc._PHASE_CACHE.clear()
    sht_sphere.clear_cache()


@pytest.mark.usefixtures("clean_caches")
class TestLoadOrGetPhaseMapsOom:
    def test_oom_in_load_phase_becomes_the_short_error_and_is_not_cached(self, monkeypatch, tmp_path):
        sht = tmp_path / "x.sht"
        sht.write_bytes(b"")
        monkeypatch.setattr(svc, "get_renderer", lambda: _Boom(RuntimeError(TRACE)))
        emptied = []
        monkeypatch.setattr(svc, "_release_cuda_cache", lambda: emptied.append(1))

        with pytest.raises(svc.SHTOutOfMemoryError) as e:
            svc.load_or_get_phase(str(sht), max_bandwidth=256)
        assert "TorchScript" not in str(e.value)
        assert len(str(e.value)) < 120
        assert len(svc._PHASE_CACHE) == 0
        assert emptied == [1], "the cache must be handed back after a failed load"

    def test_the_cache_is_released_after_the_failed_frame_is_gone(self, monkeypatch, tmp_path):
        """empty_cache() cannot return what the traceback still references, so
        the release must run after the except block, not inside it."""
        import gc
        import sys
        import weakref

        class Tensorlike:          # stands in for a tensor held by the failed load
            pass

        sht = tmp_path / "x.sht"
        sht.write_bytes(b"")
        holder = {}

        class Boom(_Boom):
            def load_phase(self, *_a, **_k):
                held = Tensorlike()
                holder["ref"] = weakref.ref(held)
                raise RuntimeError(TRACE)

        seen = {}

        def release():
            gc.collect()
            seen["exc_info"] = sys.exc_info()[0]
            seen["held_alive"] = holder["ref"]() is not None

        monkeypatch.setattr(svc, "get_renderer", lambda: Boom(None))
        monkeypatch.setattr(svc, "_release_cuda_cache", release)
        with pytest.raises(svc.SHTOutOfMemoryError):
            svc.load_or_get_phase(str(sht), max_bandwidth=256)
        assert seen["exc_info"] is None, "released while still handling the exception"
        assert seen["held_alive"] is False, "the failed frame still held its objects"

    def test_a_non_oom_error_passes_through_unchanged(self, monkeypatch, tmp_path):
        sht = tmp_path / "x.sht"
        sht.write_bytes(b"")
        monkeypatch.setattr(svc, "get_renderer", lambda: _Boom(RuntimeError("shape mismatch")))
        with pytest.raises(RuntimeError, match="shape mismatch"):
            svc.load_or_get_phase(str(sht), max_bandwidth=256)


# ------------------------------------------------------------------ route

@pytest.fixture
def client(monkeypatch, clean_caches):
    monkeypatch.setenv("ORIENTA_ALLOWED_HOSTS", "testserver")
    from fastapi.testclient import TestClient

    from backend.api.main import app

    return TestClient(app, raise_server_exceptions=False)


def _assert_short_coded_422(r):
    assert r.status_code == 422, r.text
    assert r.headers.get(CODE_HEADER.lower()) == "sphereOutOfMemory"
    detail = r.json()["detail"]
    assert isinstance(detail, str)
    assert "TorchScript" not in detail and "Traceback" not in detail
    assert "\n" not in detail and len(detail) < 160


class TestSphereRoute:
    def test_real_oom_inside_the_renderer(self, client, monkeypatch, tmp_path):
        sht = tmp_path / "x.sht"
        sht.write_bytes(b"")
        monkeypatch.setattr(sht_sphere, "_resolve_sht", lambda _f: sht)
        monkeypatch.setattr(svc, "get_renderer", lambda: _Boom(RuntimeError(TRACE)))
        _assert_short_coded_422(client.get("/api/database/sphere/x.sht?max_bandwidth=256&target_size=192"))

    def test_preflight_refusal(self, client, monkeypatch, tmp_path):
        sht = tmp_path / "x.sht"
        sht.write_bytes(b"")
        monkeypatch.setattr(sht_sphere, "_resolve_sht", lambda _f: sht)
        monkeypatch.setattr(svc, "get_renderer", lambda: types.SimpleNamespace(device=_cuda_device()))
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        monkeypatch.setattr(torch.cuda, "mem_get_info",
                            lambda *_a, **_k: (int(0.4 * 2**30), int(8 * 2**30)))
        _assert_short_coded_422(client.get("/api/database/sphere/x.sht?max_bandwidth=256&target_size=192"))

    def test_oom_that_escapes_as_a_bare_runtime_error(self, client, monkeypatch):
        """Anything that reaches the route's last handler is checked there too."""
        def boom(*_a, **_k):
            raise RuntimeError(TRACE)

        monkeypatch.setattr(sht_sphere, "sphere_from_sht", boom)
        _assert_short_coded_422(client.get("/api/database/sphere/x.sht?max_bandwidth=256"))

    def test_another_failure_is_one_line_without_a_code(self, client, monkeypatch):
        def boom(*_a, **_k):
            raise RuntimeError("first line\nTraceback (most recent call last):\n  File \"a.py\"")

        monkeypatch.setattr(sht_sphere, "sphere_from_sht", boom)
        r = client.get("/api/database/sphere/x.sht")
        assert r.status_code == 422
        assert r.headers.get(CODE_HEADER.lower()) is None
        assert r.json()["detail"] == "sphere reconstruction failed: first line"


def test_the_code_has_a_text_in_all_four_languages():
    for lang in ("en", "de", "ja", "zh"):
        loc = json.loads((ROOT / f"frontend/src/locales/{lang}/databasebrowser.json").read_text(encoding="utf-8"))
        assert loc["errors"]["sphereOutOfMemory"].strip(), lang
        assert loc["sphere"]["useStandard"].strip(), lang


class TestWrappedErrorIsRecognisedByType:
    """The wrapper's own text does not repeat torch's wording, so consumers
    that used to match "out of memory" must be able to ask by type."""

    def test_the_wrapper_is_oom_by_type_alone(self):
        assert svc.is_out_of_memory(svc.SHTOutOfMemoryError("Quality needs more GPU memory."))

    def test_an_unrelated_error_raised_while_handling_an_oom_is_not_oom(self):
        try:
            try:
                raise RuntimeError(TRACE)
            except RuntimeError:
                raise ValueError("a different problem")   # implicit __context__ only
        except ValueError as e:
            assert not svc.is_out_of_memory(e)
