"""Tests for backend/api/file_log.py — the persistent rotating log file."""

import importlib
import logging
import threading
import warnings

import pytest


@pytest.fixture()
def file_log(monkeypatch, tmp_path):
    """Fresh module instance per test, logging into tmp_path."""
    monkeypatch.delenv("ORIENTA_NO_FILE_LOG", raising=False)
    import backend.api.file_log as mod

    mod = importlib.reload(mod)
    yield mod, tmp_path
    # Detach whatever the test installed so later tests / suites are clean.
    root = logging.getLogger()
    for lg in (root, logging.getLogger("uvicorn")):
        for h in list(lg.handlers):
            if isinstance(h, logging.handlers.RotatingFileHandler) and str(tmp_path) in str(
                getattr(h, "baseFilename", "")
            ):
                lg.removeHandler(h)
                h.close()
        for h in list(lg.handlers):
            if isinstance(h, logging.StreamHandler) and any(
                isinstance(f, mod._DropNoisyLoggers) for f in h.filters
            ) and not isinstance(h, logging.handlers.RotatingFileHandler):
                lg.removeHandler(h)
    logging.captureWarnings(False)


def _read(path):
    return path.read_text(encoding="utf-8")


def test_records_land_in_file(file_log):
    mod, tmp = file_log
    path = mod.install_file_logging(log_dir=tmp)
    assert path is not None and path.exists()
    logging.getLogger("backend.api.routes.indexing").info("hello from indexing")
    assert "hello from indexing" in _read(path)


def test_uvicorn_error_records_are_captured(file_log):
    """uvicorn sets propagate=False — the handler must sit on 'uvicorn' too."""
    mod, tmp = file_log
    path = mod.install_file_logging(log_dir=tmp)
    uv = logging.getLogger("uvicorn")
    old_propagate = uv.propagate
    uv.propagate = False  # simulate uvicorn's dictConfig
    try:
        logging.getLogger("uvicorn.error").error("Exception in ASGI application")
    finally:
        uv.propagate = old_propagate
    assert "Exception in ASGI application" in _read(path)


def test_noisy_loggers_are_kept_out(file_log):
    mod, tmp = file_log
    path = mod.install_file_logging(log_dir=tmp)
    logging.getLogger("matplotlib.font_manager").warning("font spam")
    logging.getLogger("PIL.PngImagePlugin").info("chunk spam")
    logging.getLogger("uvicorn.access").info("GET /api/health 200")
    logging.getLogger("app.real").warning("real message")
    text = _read(path)
    assert "font spam" not in text
    assert "chunk spam" not in text
    assert "GET /api/health" not in text
    assert "real message" in text


def test_idempotent_install(file_log):
    mod, tmp = file_log
    p1 = mod.install_file_logging(log_dir=tmp)
    p2 = mod.install_file_logging(log_dir=tmp)
    assert p1 == p2
    file_handlers = [
        h
        for h in logging.getLogger().handlers
        if isinstance(h, logging.handlers.RotatingFileHandler)
        and str(tmp) in str(h.baseFilename)
    ]
    assert len(file_handlers) == 1


def test_opt_out_env(file_log, monkeypatch):
    mod, tmp = file_log
    monkeypatch.setenv("ORIENTA_NO_FILE_LOG", "1")
    assert mod.install_file_logging(log_dir=tmp) is None
    assert not (tmp / mod.LOG_BASENAME).exists()


def test_warnings_are_captured(file_log):
    mod, tmp = file_log
    path = mod.install_file_logging(log_dir=tmp)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn("deprecated thing", UserWarning)
    assert "deprecated thing" in _read(path)


def test_thread_crash_is_logged(file_log):
    mod, tmp = file_log
    path = mod.install_file_logging(log_dir=tmp)

    def boom():
        raise RuntimeError("thread went down")

    t = threading.Thread(target=boom, name="doomed-worker")
    t.start()
    t.join()
    text = _read(path)
    assert "doomed-worker" in text
    assert "thread went down" in text


def test_get_log_paths_lists_existing_files(file_log):
    mod, tmp = file_log
    path = mod.install_file_logging(log_dir=tmp)
    logging.getLogger("x").info("make sure the file exists")
    paths = mod.get_log_paths()
    assert path in paths
    # rotated siblings don't exist yet
    assert all(p.exists() for p in paths)
