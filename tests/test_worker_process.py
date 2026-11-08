"""The compute worker as a process, driven directly over its pipes."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.worker_helpers import LineReader, gone_within, kill_quietly, send, spawn_raw_worker  # noqa: E402


@pytest.fixture
def worker():
    import threading
    proc = spawn_raw_worker()
    out = LineReader(proc.stdout)
    tail = []
    threading.Thread(target=lambda: [tail.append(l) for l in proc.stderr], daemon=True).start()
    ready = out.until(lambda m: m.get("ev") == "ready", timeout=60)
    yield proc, out, ready, tail
    kill_quietly(proc.pid)


def test_it_says_ready_with_its_pid(worker):
    proc, out, ready, _ = worker
    assert ready["pid"] == proc.pid


def test_a_job_round_trips_a_path_with_umlauts_and_brackets(worker, tmp_path):
    proc, out, _, _ = worker
    f = tmp_path / "Größe [Pearson] {20kV} Ä.sht"
    f.write_bytes(b"x")
    send(proc, {"job": "_stat", "id": "s1", "args": {"path": str(f)}})
    msg = out.until(lambda m: m.get("id") == "s1" and m.get("ev") in ("result", "error"))
    assert msg["ev"] == "result"
    assert msg["result"] == {"path": str(f), "exists": True}


def test_log_records_travel_with_their_logger_name(worker):
    proc, out, _, _ = worker
    send(proc, {"job": "_log", "id": "l1", "args": {"logger": "some.module", "msg": "Grüße"}})
    msg = out.until(lambda m: m.get("ev") == "log" and m.get("name") == "some.module")
    assert msg["level"] == 30 and msg["msg"] == "Grüße"


def test_printing_does_not_break_the_protocol(worker):
    proc, out, _, tail = worker
    send(proc, {"job": "_print", "id": "p1", "args": {}})
    msg = out.until(lambda m: m.get("id") == "p1" and m.get("ev") in ("result", "error"))
    assert msg == {"ev": "result", "id": "p1", "result": {"ok": True}}
    send(proc, {"job": "_echo", "id": "p2", "args": {"still": "fine"}})
    assert out.until(lambda m: m.get("id") == "p2" and m.get("ev") != "imported")["result"] == {"still": "fine"}
    wanted = ["printed by python", "written to sys.stdout", "written to fd 1"]
    if sys.platform == "win32":
        wanted.append("written to the stdout handle")    # Win32 handle, not fd 1
    end = time.monotonic() + 5.0
    while time.monotonic() < end:
        text = b"".join(tail).decode("utf-8", "replace")
        if all(w in text for w in wanted):
            break
        time.sleep(0.05)
    text = b"".join(tail).decode("utf-8", "replace")
    for w in wanted:
        assert w in text, f"{w!r} did not reach stderr; stderr was {text!r}"


def test_a_coded_failure_arrives_as_an_error_with_its_code(worker):
    proc, out, _, _ = worker
    send(proc, {"job": "_raise", "id": "r1", "args": {"code": "sphereOutOfMemory", "msg": "too big"}})
    msg = out.until(lambda m: m.get("id") == "r1" and m.get("ev") not in ("progress", "imported"))
    assert msg == {"ev": "error", "id": "r1", "code": "sphereOutOfMemory", "message": "too big"}


def test_an_unknown_job_is_an_error_not_a_crash(worker):
    proc, out, _, _ = worker
    send(proc, {"job": "nope", "id": "u1", "args": {}})
    msg = out.until(lambda m: m.get("id") == "u1")
    assert msg["ev"] == "error" and msg["code"] == "unknownJob"
    send(proc, {"job": "_echo", "id": "u2", "args": {}})
    assert out.until(lambda m: m.get("id") == "u2" and m.get("ev") != "imported")["ev"] == "result"


def test_a_result_that_cannot_be_serialised_is_an_error_and_the_worker_lives_on(worker):
    proc, out, _, _ = worker
    send(proc, {"job": "_bad_result", "id": "b1", "args": {}})
    msg = out.until(lambda m: m.get("id") == "b1" and m.get("ev") in ("result", "error"))
    assert msg["ev"] == "error" and msg["code"] == "jobFailed"
    assert "\n" not in msg["message"] and "JSON" in msg["message"]
    send(proc, {"job": "_echo", "id": "b2", "args": {"next": 1}})
    assert out.until(lambda m: m.get("id") == "b2" and m.get("ev") != "imported")["result"] == {"next": 1}
    assert proc.poll() is None


def test_a_result_with_a_lone_surrogate_is_an_error_and_the_worker_lives_on(worker):
    proc, out, _, _ = worker
    send(proc, {"job": "_bad_result", "id": "t1", "args": {"kind": "surrogate"}})
    msg = out.until(lambda m: m.get("id") == "t1" and m.get("ev") in ("result", "error"))
    assert msg["ev"] == "error" and msg["code"] == "jobFailed"
    send(proc, {"job": "_echo", "id": "t2", "args": {}})
    assert out.until(lambda m: m.get("id") == "t2" and m.get("ev") != "imported")["ev"] == "result"


def test_unusable_input_lines_do_not_stop_the_reader(worker):
    proc, out, _, _ = worker
    for line in (b"[1]\n", b"null\n", b"42\n", b"not json at all\n",
                 b'{"ev": "cancel", "id": [1]}\n', b'{"job": "_echo", "id": [1], "args": {}}\n'):
        proc.stdin.write(line)
    proc.stdin.flush()
    send(proc, {"job": [1], "id": "z", "args": {}})
    msg = out.until(lambda m: m.get("id") == "z")
    assert msg["ev"] == "error" and msg["code"] == "unknownJob"
    send(proc, {"job": "_echo", "id": "ok", "args": {"fine": True}})
    assert out.until(lambda m: m.get("id") == "ok" and m.get("ev") != "imported")["result"] == {"fine": True}
    # The reader is still alive: a cancel still arrives, and the end of stdin still ends the worker.
    send(proc, {"job": "_sleep", "id": "cc", "args": {"s": 30, "cooperative": True}})
    out.until(lambda m: m.get("id") == "cc" and m.get("ev") == "progress")
    send(proc, {"ev": "cancel", "id": "cc"})
    assert out.until(lambda m: m.get("id") == "cc" and m.get("ev") != "progress", timeout=10) == {
        "ev": "cancelled", "id": "cc"}
    proc.stdin.close()
    assert gone_within(proc.pid, 3.0)


@pytest.mark.parametrize("plain", [False, True])
def test_an_error_text_that_cannot_be_encoded_still_gets_its_reply(worker, plain):
    proc, out, _, _ = worker
    send(proc, {"job": "_raise", "id": "e1",
                "args": {"code": "oddText", "msg": "bad text ", "surrogate": True, "plain": plain}})
    msg = out.until(lambda m: m.get("id") == "e1" and m.get("ev") in ("result", "error"))
    assert msg["ev"] == "error"
    assert msg["code"] == ("jobFailed" if plain else "oddText")
    assert "bad text" in msg["message"] and "udcff" in msg["message"]
    send(proc, {"job": "_echo", "id": "e2", "args": {}})
    assert out.until(lambda m: m.get("id") == "e2" and m.get("ev") != "imported")["ev"] == "result"
    assert proc.poll() is None


def test_a_cooperative_job_stops_on_cancel(worker):
    proc, out, _, _ = worker
    send(proc, {"job": "_sleep", "id": "c1", "args": {"s": 30, "cooperative": True}})
    out.until(lambda m: m.get("id") == "c1" and m.get("ev") == "progress")   # it is running now
    send(proc, {"ev": "cancel", "id": "c1"})
    msg = out.until(lambda m: m.get("id") == "c1" and m.get("ev") != "progress", timeout=10)
    assert msg == {"ev": "cancelled", "id": "c1"}


def test_a_job_cancelled_while_it_waits_is_answered_without_running(worker):
    proc, out, _, _ = worker
    send(proc, {"job": "_sleep", "id": "w1", "args": {"s": 3}})
    send(proc, {"job": "_log", "id": "w2", "args": {"logger": "waiting.job", "msg": "I ran"}})
    send(proc, {"ev": "cancel", "id": "w2"})
    seen = []
    msg = out.until(lambda m: seen.append(m) or m.get("id") == "w2", timeout=10)
    assert msg == {"ev": "cancelled", "id": "w2"}
    assert not [m for m in seen if m.get("name") == "waiting.job"]   # it never started


def test_progress_is_reported(worker):
    proc, out, _, _ = worker
    send(proc, {"job": "_progress", "id": "g1", "args": {"n": 3}})
    first = out.until(lambda m: m.get("id") == "g1" and m.get("ev") == "progress")
    assert first == {"ev": "progress", "id": "g1", "frac": 0.0, "msg": "step 0"}


def test_the_worker_says_imported_after_the_jobs_heavy_imports_and_before_its_first_work(monkeypatch):
    """`ready` comes before any job module is imported; `imported` is the job's
    first message and follows its `preload`, so the backend can time the slow
    part of a cold start."""
    import threading
    monkeypatch.setenv("ORIENTA_WORKER_TEST_PRELOAD_S", "1.5")
    proc = spawn_raw_worker()
    out = LineReader(proc.stdout)
    threading.Thread(target=lambda: [None for _ in proc.stderr], daemon=True).start()
    try:
        out.until(lambda m: m.get("ev") == "ready", timeout=60)
        t0 = time.monotonic()
        send(proc, {"job": "_slow_import", "id": "i1", "args": {"a": 1}})
        first = out.until(lambda m: m.get("id") == "i1", timeout=30)
        waited = time.monotonic() - t0
        assert first == {"ev": "imported", "id": "i1"}
        assert waited >= 1.4, waited                  # not before the preload finished
        assert out.until(lambda m: m.get("id") == "i1")["result"] == {"a": 1}
    finally:
        kill_quietly(proc.pid)


def test_closing_stdin_ends_the_worker(worker):
    proc, out, _, _ = worker
    proc.stdin.close()
    assert gone_within(proc.pid, 3.0)


def test_a_crashing_job_ends_the_process_with_its_code(worker):
    proc, out, _, tail = worker
    send(proc, {"job": "_crash", "id": "x1", "args": {"code": 3}})
    assert proc.wait(timeout=10) == 3
