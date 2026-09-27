"""The origin-choice warning must not cry wolf on the files we just repaired.

`xtal_io.read_crystal_structure` warns whenever a .xtal in one of the 24
two-origin space groups declares choice 1, because for a file written by the
old converter that is the signature of the silicon bug. After the rebuild of
2026-09-26 the same message fires on Si.xtal and sd_1816951.xtal, which are now
correct — the converter decided their origin from the CIF's own Wyckoff
multiplicity and wrote coordinates that really are choice 1.

So the converter now leaves a record of how it decided, and the reader reads it.
A sidecar rather than a dataset inside the .xtal: EMsoft reads these files too,
and a format change that cannot be tested here is not worth a log line.
"""
from __future__ import annotations

import json
import logging

import h5py
import numpy as np
import pytest

from backend.forward_sim.crystal.xtal_io import (
    origin_choice_is_recorded,
    origin_choice_sidecar,
    read_crystal_structure,
    write_origin_choice_sidecar,
)

_WARN_LOGGER = "backend.forward_sim.crystal.origin_choice"


def _si_xtal(tmp_path, name="Si.xtal"):
    """Silicon as the rebuild writes it: the 8a site in origin choice 1."""
    path = tmp_path / name
    atomdata = np.zeros((5, 1), dtype=np.float32)
    atomdata[:, 0] = (0.75, 0.75, 0.75, 1.0, 0.005)
    with h5py.File(path, "w") as f:
        cd = f.create_group("CrystalData")
        cd.create_dataset("AtomData", data=atomdata)
        cd.create_dataset("Atomtypes", data=np.array([14], dtype=np.int32))
        cd.create_dataset("Natomtypes", data=np.array([1], dtype=np.int32))
        cd.create_dataset("LatticeParameters",
                          data=np.array([0.54309] * 3 + [90.0] * 3, dtype=np.float32))
        cd.create_dataset("SpaceGroupNumber", data=np.array([227], dtype=np.int32))
        cd.create_dataset("SpaceGroupSetting", data=np.array([1], dtype=np.int32))
        cd.create_dataset("CrystalSystem", data=np.array([1], dtype=np.int32))
    return path


def _warnings_from(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.name == _WARN_LOGGER and r.levelno >= logging.WARNING]


def test_a_file_with_no_record_is_still_warned_about(tmp_path, caplog):
    """The legacy case the warning exists for: nothing says how it was made."""
    path = _si_xtal(tmp_path)
    with caplog.at_level(logging.WARNING, logger=_WARN_LOGGER):
        read_crystal_structure(str(path))
    assert any("two origin choices" in m for m in _warnings_from(caplog))


def test_a_file_whose_origin_was_decided_is_not_warned_about(tmp_path, caplog):
    """What the converter now writes for every file it converts."""
    path = _si_xtal(tmp_path)
    write_origin_choice_sidecar(
        path, block="sm_isp_SD0530557-standardized_unitcell", cif_origin_choice=2,
        shifted=True, evidence=["Wyckoff multiplicity says origin choice 2"],
        cif_name="Si.cif")

    with caplog.at_level(logging.WARNING, logger=_WARN_LOGGER):
        structure = read_crystal_structure(str(path))
    assert _warnings_from(caplog) == []
    assert structure.space_group_setting == 1     # and the file still reads the same


@pytest.mark.parametrize("payload", [
    "",                                   # empty
    "{not json",                          # malformed
    '{"schema": 1}',                      # no origin block
    '{"origin": {"evidence": []}}',       # decided nothing
    '{"origin": {"evidence": ["x"], "cif_origin_choice": 7}}',   # nonsense choice
])
def test_a_record_that_says_nothing_useful_does_not_silence_the_warning(tmp_path, caplog, payload):
    """Fail safe: only a record that actually names evidence counts."""
    path = _si_xtal(tmp_path)
    origin_choice_sidecar(path).write_text(payload, encoding="utf-8")
    assert origin_choice_is_recorded(path) is False
    with caplog.at_level(logging.WARNING, logger=_WARN_LOGGER):
        read_crystal_structure(str(path))
    assert any("two origin choices" in m for m in _warnings_from(caplog))


def test_the_record_says_what_was_decided_and_how(tmp_path):
    path = _si_xtal(tmp_path)
    written = write_origin_choice_sidecar(
        path, block="published_cell", cif_origin_choice=1, shifted=False,
        evidence=["the H-M symbol says origin choice 1"], cif_name="Si.cif")
    assert written == origin_choice_sidecar(path)

    doc = json.loads(written.read_text(encoding="utf-8"))
    assert doc["source_cif"] == "Si.cif"
    assert doc["origin"]["cif_origin_choice"] == 1
    assert doc["origin"]["shifted_to_choice_1"] is False
    assert doc["origin"]["evidence"] == ["the H-M symbol says origin choice 1"]


def test_writing_the_record_never_breaks_a_conversion(tmp_path):
    """Best-effort by contract, like the .sht sidecar."""
    unwritable = tmp_path / "no-such-dir" / "Si.xtal"
    assert write_origin_choice_sidecar(
        unwritable, block="b", cif_origin_choice=1, shifted=False, evidence=["e"]) is None
