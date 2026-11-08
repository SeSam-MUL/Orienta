"""The line protocol between the backend and its compute worker."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.worker import protocol  # noqa: E402
from backend.worker import jobs  # noqa: E402


def test_a_message_is_one_utf8_line_and_comes_back_unchanged():
    msg = {"job": "_echo", "id": "a1",
           "args": {"path": "C:/Daten/Größe [Pearson] {20kV}/Ä.sht", "text": "zwei\nZeilen"}}
    line = protocol.encode(msg)
    assert line.endswith(b"\n") and line.count(b"\n") == 1
    assert "Größe".encode("utf-8") in line          # UTF-8, not \u escapes
    assert protocol.decode(line) == msg


def test_floats_survive_exactly():
    values = [0.1, 1 / 3, 2.0 ** -40, 123456.789e-12, float("nan")]
    back = protocol.decode(protocol.encode({"v": values}))["v"]
    assert back[:4] == values[:4]
    assert back[4] != back[4]                     # NaN stays NaN


def test_a_small_result_travels_inline():
    msg = protocol.result_message("j1", {"a": [1, 2, 3]})
    assert msg == {"ev": "result", "id": "j1", "result": {"a": [1, 2, 3]}}
    assert protocol.read_result(msg) == {"a": [1, 2, 3]}


def test_a_result_above_the_limit_travels_as_a_file_that_is_removed_after_reading():
    blob = "x" * (protocol.INLINE_LIMIT + 10)
    msg = protocol.result_message("j2", {"blob": blob})
    assert "result" not in msg
    path = Path(msg["result_file"])
    assert path.is_file() and path.parent.name.startswith("orienta-worker-")
    assert protocol.read_result(msg) == {"blob": blob}
    assert not path.exists()
    assert not path.parent.exists()          # no empty folders left in the temp dir


def test_a_spilled_result_keeps_non_ascii_text():
    text = "Größe α→β " * (protocol.INLINE_LIMIT // 8)      # 'α' and '→' are not in cp1252
    msg = protocol.result_message("j6", {"t": text})
    assert "result_file" in msg
    assert protocol.read_result(msg) == {"t": text}


def test_the_inline_limit_is_four_mebibytes():
    assert protocol.INLINE_LIMIT == 4 * 1024 * 1024


def _body_of_exactly(n_bytes: int):
    """A result whose compact JSON body is exactly ``n_bytes`` long (ASCII)."""
    overhead = len(json.dumps({"b": ""}, separators=(",", ":")))
    result = {"b": "x" * (n_bytes - overhead)}
    assert len(json.dumps(result, separators=(",", ":")).encode("utf-8")) == n_bytes
    return result


def test_a_body_of_exactly_the_limit_is_still_inline_and_one_byte_more_is_a_file():
    at_limit = protocol.result_message("j4", _body_of_exactly(protocol.INLINE_LIMIT))
    assert "result" in at_limit and "result_file" not in at_limit
    over = protocol.result_message("j5", _body_of_exactly(protocol.INLINE_LIMIT + 1))
    assert "result_file" in over and "result" not in over
    protocol.read_result(over)                 # removes the file again


def test_test_jobs_are_hidden_without_the_switch(monkeypatch):
    monkeypatch.delenv("ORIENTA_WORKER_TEST_JOBS", raising=False)
    with pytest.raises(KeyError):
        jobs.lookup("_echo")


def test_test_jobs_are_visible_with_the_switch(monkeypatch):
    monkeypatch.setenv("ORIENTA_WORKER_TEST_JOBS", "1")
    spec = jobs.lookup("_echo")
    assert spec.module == "backend.worker.jobs._testing" and spec.func == "echo"
    assert not spec.is_heavy({})
    assert jobs.lookup("_sleep_heavy").is_heavy({})


def test_an_unknown_job_is_a_key_error():
    with pytest.raises(KeyError):
        jobs.lookup("no-such-job")


@pytest.mark.parametrize("bw, heavy", [(None, False), (128, False), (255, False), (256, True), (384, True)])
def test_a_sphere_is_heavy_from_bandwidth_256(bw, heavy):
    args = {} if bw is None else {"max_bandwidth": bw}
    assert jobs.lookup("sphere").is_heavy(args) is heavy


def test_switches_only_count_with_the_test_jobs(monkeypatch):
    monkeypatch.setenv("ORIENTA_WORKER_TEST_DISABLE", "eof, poll")
    monkeypatch.delenv("ORIENTA_WORKER_TEST_JOBS", raising=False)
    assert jobs.test_switches() == set()
    monkeypatch.setenv("ORIENTA_WORKER_TEST_JOBS", "1")
    assert jobs.test_switches() == {"eof", "poll"}


def test_the_context_reports_progress_and_cancel():
    sent, cancelled = [], set()
    ctx = jobs.JobContext("j3", sent.append, cancelled)
    ctx.progress(0.5, "halfway")
    ctx.progress(None)
    assert sent == [{"ev": "progress", "id": "j3", "frac": 0.5, "msg": "halfway"},
                    {"ev": "progress", "id": "j3", "frac": None, "msg": ""}]
    assert not ctx.cancelled()
    cancelled.add("j3")
    assert ctx.cancelled()
