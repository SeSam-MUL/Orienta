"""GET /api/pc/status says where the active dataset's PC comes from (``pc_source``).

The PC page shows a "no PC calibrated" hint for ``missing``: the fields hold a
placeholder, and a page that looked like any other invited the user to refine from it.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from kikuchipy.detectors import EBSDDetector

from backend.api.main import app
from backend.api.services.calibration_store import CalibrationStore
from pc_controller import PCController

SHAPE = (60, 60)


class _Sig:
    def __init__(self, tag):
        self.detector = EBSDDetector(shape=SHAPE, pc=(0.5, 0.5, 0.5), sample_tilt=70.0)
        if tag:
            self.metadata = type("M", (), {"get_item": staticmethod(lambda k, d=None: tag)})()


@pytest.fixture
def client(monkeypatch):
    from backend.api.routes import ebsd_viewer
    from backend.api.routes import pcrefinement as R
    store = CalibrationStore()
    ctrl = PCController()
    ctrl.attach_detector(EBSDDetector(shape=SHAPE, pc=(0.5, 0.5, 0.5), sample_tilt=70.0))
    monkeypatch.setattr(R, "_get_controller", lambda: ctrl)
    monkeypatch.setattr(R, "calibration_store", store)
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "ds")
    return TestClient(app), store


def test_a_file_without_a_pc_reports_missing_until_one_is_set(client):
    c, store = client
    store.register("ds", _Sig("default"))
    assert c.get("/api/pc/status").json()["pc_source"] == "missing"
    store.update_pc("ds", [0.48, 0.52, 0.6], source="manual")
    assert c.get("/api/pc/status").json()["pc_source"] == "manual"


def test_a_file_with_a_header_pc_reports_header(client):
    c, store = client
    store.register("ds", _Sig(None))
    assert c.get("/api/pc/status").json()["pc_source"] == "header"


def test_no_dataset_in_the_store_reports_none(client):
    c, _ = client
    assert c.get("/api/pc/status").json()["pc_source"] is None
