"""A missing emsphinx_config.ini is a warning only where the file is expected.

The config drives the EMsoft/EMSphInx automation that runs inside WSL, which
exists on Windows only. A macOS or Linux checkout has no such file and was told
"Config file not found at <path>" as a WARNING on every start, which reads as
a defect to anyone watching the console. Off Windows it is one INFO line that
says why; on Windows it stays a warning, because there the file is expected.
"""
from __future__ import annotations

import logging
import sys

import pytest

from simulation.simulation_controller import SimulationController

LOGGER = "simulation.simulation_controller"


def _build(tmp_path, monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)
    return SimulationController(config_path=tmp_path / "missing" / "emsphinx_config.ini")


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_off_windows_it_is_one_info_line_that_explains_itself(tmp_path, monkeypatch, caplog, platform):
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        _build(tmp_path, monkeypatch, platform)
    records = [r for r in caplog.records if r.name == LOGGER and "config" in r.getMessage().lower()]
    assert not [r for r in records if r.levelno >= logging.WARNING]
    info = [r for r in records if r.levelno == logging.INFO]
    assert len(info) == 1
    assert "Windows" in info[0].getMessage() and "WSL" in info[0].getMessage()


def test_on_windows_it_stays_a_warning(tmp_path, monkeypatch, caplog):
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        _build(tmp_path, monkeypatch, "win32")
    warnings = [r for r in caplog.records
                if r.name == LOGGER and r.levelno == logging.WARNING
                and "Config file not found" in r.getMessage()]
    assert len(warnings) == 1
