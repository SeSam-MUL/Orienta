"""Tests for the backend-console capture helpers in start_app.py."""

import io
import subprocess
import sys

import pytest

import start_app


@pytest.fixture()
def log_root(monkeypatch, tmp_path):
    monkeypatch.setattr(start_app, "PROJECT_ROOT", tmp_path)
    return tmp_path


def test_open_backend_console_log_writes_header(log_root):
    fh = start_app._open_backend_console_log()
    assert fh is not None
    fh.close()
    text = (log_root / "logs" / "backend-console.log").read_text(encoding="utf-8")
    assert "session start" in text


def test_open_backend_console_log_rotates_when_large(log_root):
    log_dir = log_root / "logs"
    log_dir.mkdir()
    big = log_dir / "backend-console.log"
    big.write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    fh = start_app._open_backend_console_log()
    assert fh is not None
    fh.close()
    rotated = log_dir / "backend-console.log.1"
    assert rotated.exists()
    assert rotated.stat().st_size > 2 * 1024 * 1024
    # fresh file only holds the new session header
    assert big.stat().st_size < 1024


def test_tee_mirrors_output_to_console_and_file(log_root, capsys):
    fh_dir = log_root / "logs"
    fh_dir.mkdir()
    log_file = fh_dir / "backend-console.log"
    fh = open(log_file, "a", encoding="utf-8")

    proc = subprocess.Popen(
        [sys.executable, "-c", "print('hello from backend'); print('line two')"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    start_app._tee_backend_output(proc, fh)
    proc.wait()
    # give the daemon thread a moment to drain
    import time
    for _ in range(50):
        if "line two" in log_file.read_text(encoding="utf-8"):
            break
        time.sleep(0.1)
    fh.close()

    text = log_file.read_text(encoding="utf-8")
    assert "hello from backend" in text
    assert "line two" in text
    captured = capsys.readouterr()
    assert "hello from backend" in captured.out
