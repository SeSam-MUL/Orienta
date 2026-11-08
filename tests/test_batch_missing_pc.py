"""The batch pages are told "no PC" for a file that has none.

Since a file without a pattern centre (an EDAX UP1/UP2 without an ``.osc``) is
registered as ``pc_source == "missing"``, the batch routes passed that word on,
together with kikuchipy's placeholder (0.5, 0.5, 0.5) as the PC. The batch page
knows ``none`` / ``header`` / ``refined`` / ``inherited`` / ``manual``: it showed
the placeholder under a status it has no text for, and a green tick (anything
but ``none`` counts as "has a PC"). They now get ``none`` and no PC, which is what
makes the page offer a PC from another file or a refinement.
"""
import asyncio

import numpy as np
import pytest

from backend.api.services.calibration_store import CalibrationStore


@pytest.fixture(scope="module")
def nickel():
    import kikuchipy as kp
    return kp.data.nickel_ebsd_small()


@pytest.fixture
def batch(monkeypatch, nickel):
    from backend.api.routes import batch_v2 as B
    store = CalibrationStore()
    missing = nickel.deepcopy()
    missing.metadata.set_item("Signal.pc_source", "default")
    store.register("nopc", missing)
    store.register("haspc", nickel.deepcopy())
    monkeypatch.setattr(B, "calibration_store", store)
    return B, store


def test_pc_status_reports_none_and_no_pc_for_a_missing_one(batch):
    B, store = batch
    rows = {r["dataset_name"]: r for r in asyncio.run(B.pc_status(files="nopc,haspc"))}
    assert rows["nopc"]["pc_source"] == "none"
    assert rows["nopc"]["pc"] is None
    assert rows["haspc"]["pc_source"] == "header"
    assert len(rows["haspc"]["pc"]) == 3


def test_quick_load_reports_none_and_no_pc_for_a_missing_one(batch, monkeypatch, tmp_path, nickel):
    B, store = batch
    import safe_loader
    f = tmp_path / "nopc.up1"
    f.write_bytes(b"x")
    sig = nickel.deepcopy()
    sig.metadata.set_item("Signal.pc_source", "default")
    monkeypatch.setattr(safe_loader, "load_ebsd_safe", lambda path, *a, **k: sig)
    res = asyncio.run(B.quick_load(B.QuickLoadRequest(file_path=str(f))))
    assert res["pc_source"] == "none" and res["pc"] is None
    # the store still knows the truth: it is registered as missing
    assert store.get_entry("nopc").pc_source == "missing"
