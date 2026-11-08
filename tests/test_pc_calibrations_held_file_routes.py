"""Saving a calibration, or making a new version, while another program holds the calibration folder's file.

Applying already answered such a file with the status ``notWritable``. The two routes that write the calibration
file itself (``POST /`` and ``/versions``) answered a bare 500; they answer with the same code now, and nothing is
written. A calibration file whose version records ``origin.refinement`` as anything but a record or nothing is
refused when it is read (``unreadable``), so no later code stumbles over it.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from backend.api.services import pc_calibrations as cal
from tests.test_pc_apply_held_calibration import PC_A, PC_B, _hold_the_calibration_file, world  # noqa: F401


def _folder():
    return cal.calibrations_dir()


def test_saving_while_the_file_is_held_answers_with_a_code_and_saves_nothing(world, monkeypatch):
    client, store = world
    _hold_the_calibration_file(monkeypatch)
    r = client.post("/api/pc-calibrations", json={"name": "REM", "dataset": "scan_a"})
    assert r.status_code == 409, r.text
    assert r.headers.get("X-Orienta-Code") == "notWritable"
    assert not list(_folder().glob("*.json")) if _folder().is_dir() else True


def test_a_new_version_while_the_file_is_held_answers_with_a_code_and_writes_no_pc(world, monkeypatch):
    client, store = world
    made = client.post("/api/pc-calibrations", json={"name": "REM", "dataset": "scan_a"}).json()
    store.update_pc("scan_a", (0.51, 0.31, 0.81), source="manual")
    _hold_the_calibration_file(monkeypatch)
    r = client.post(f"/api/pc-calibrations/{made['id']}/versions",
                    json={"dataset": "scan_a", "apply_to": ["scan_b"], "accept_mismatch": ["scan_b"]})
    assert r.status_code == 409, r.text
    assert r.headers.get("X-Orienta-Code") == "notWritable"
    assert len(cal.get(made["id"])["versions"]) == 1
    np.testing.assert_allclose(store.get_pc("scan_b"), PC_B, atol=1e-9)


def test_a_version_that_was_written_before_the_file_was_held_is_kept_and_the_member_list_is_left_alone(world, monkeypatch):
    """The version is written, then the file is held: the answer is a normal one (nothing a person has to undo), the
    files named in ``apply_to`` say ``notWritable`` themselves, and no PC reaches them."""
    client, store = world
    made = client.post("/api/pc-calibrations", json={"name": "REM", "dataset": "scan_a"}).json()
    store.update_pc("scan_a", (0.51, 0.31, 0.81), source="manual")
    real = os.replace
    allowed = {"n": 1}

    def once(src, dst, *a, **k):
        if allowed["n"] > 0:
            allowed["n"] -= 1
            return real(src, dst, *a, **k)
        raise PermissionError(13, "in use by another program")

    monkeypatch.setattr(cal, "_RETRY_PAUSE_S", 0)
    monkeypatch.setattr(os, "replace", once)
    r = client.post(f"/api/pc-calibrations/{made['id']}/versions",
                    json={"dataset": "scan_a", "apply_to": ["scan_b"], "accept_mismatch": ["scan_b"]})
    assert r.status_code == 200, r.text
    assert [a["status"] for a in r.json()["applied"]] == ["notWritable"]
    assert len(cal.get(made["id"])["versions"]) == 2
    np.testing.assert_allclose(store.get_pc("scan_b"), PC_B, atol=1e-9)


def test_a_version_whose_refinement_is_not_a_record_is_unreadable(world):
    client, store = world
    made = client.post("/api/pc-calibrations", json={"name": "REM", "dataset": "scan_a"}).json()
    path = _folder() / f"{made['id']}.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["versions"][0]["origin"]["refinement"] = "PSO"
    path.write_text(json.dumps(doc), encoding="utf-8")
    problems = client.get("/api/pc-calibrations").json()["problems"]
    assert [p.get("code") for p in problems] == ["unreadable"], problems
    r = client.post(f"/api/pc-calibrations/{made['id']}/apply", json={"datasets": ["scan_b"], "accept_mismatch": ["scan_b"]})
    assert r.status_code == 400 and r.headers.get("X-Orienta-Code") == "unreadable", r.text


def test_a_record_or_nothing_is_a_refinement_that_reads(world):
    client, store = world
    made = client.post("/api/pc-calibrations", json={"name": "REM", "dataset": "scan_a"}).json()
    path = _folder() / f"{made['id']}.json"
    for value in ({"n_patterns": 9, "method": "PSO"}, None):
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["versions"][0]["origin"]["refinement"] = value
        path.write_text(json.dumps(doc), encoding="utf-8")
        assert client.get("/api/pc-calibrations").json()["problems"] == []
