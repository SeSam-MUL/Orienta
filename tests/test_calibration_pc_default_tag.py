"""A file that carries no PC must be registered as having none.

``safe_loader`` tags a signal whose reader had no pattern centre (an EDAX UP1/UP2
without an ``.osc`` next to it) with ``Signal.pc_source = "default"``, because
kikuchipy then attaches the placeholder (0.5, 0.5, 0.5). ``CalibrationStore.register``
used to look only at whether the detector's PC could be READ - it always could - and
stored the placeholder as ``pc_source == "header"``. Every guard that refuses to work
from a PC nobody supplied tests for ``"missing"``, so none of them ever fired.
"""
import numpy as np
import pytest

from backend.api.services.calibration_store import CalibrationStore


@pytest.fixture(scope="module")
def nickel():
    import kikuchipy as kp
    return kp.data.nickel_ebsd_small()


def _signal(nickel, tag):
    sig = nickel.deepcopy()
    if tag is not None:
        sig.metadata.set_item("Signal.pc_source", tag)
    return sig


def test_a_defaulted_pc_is_registered_as_missing(nickel):
    store = CalibrationStore()
    entry = store.register("up1", _signal(nickel, "default"))
    assert entry.pc_source == "missing"
    assert store.get_pc("up1") is None        # consumers that need a real PC fail loud


def test_a_pc_from_the_osc_sidecar_is_a_real_one(nickel):
    store = CalibrationStore()
    assert store.register("up1", _signal(nickel, "osc")).pc_source == "header"
    assert store.get_pc("up1") is not None


def test_a_file_without_a_tag_keeps_its_header_pc(nickel):
    """h5oina, EDAX h5 and the others carry their PC and set no tag."""
    store = CalibrationStore()
    entry = store.register("h5", _signal(nickel, None))
    assert entry.pc_source == "header"
    np.testing.assert_allclose(store.get_pc("h5"), np.asarray(nickel.detector.pc).reshape(-1, 3).mean(0))


def test_a_typed_pc_ends_the_missing_state(nickel):
    store = CalibrationStore()
    store.register("up1", _signal(nickel, "default"))
    store.update_pc("up1", [0.51, 0.40, 0.66], source="manual")
    entry = store.get_entry("up1")
    assert entry.pc_source == "manual"
    np.testing.assert_allclose(store.get_pc("up1"), [0.51, 0.40, 0.66])


def test_derived_and_cropped_copies_of_a_missing_pc_stay_missing(nickel):
    """A copy of a dataset whose PC nobody supplied still has none."""
    store = CalibrationStore()
    store.register("up1", _signal(nickel, "default"))
    store.register_derived("up1_avg", "up1")
    assert store.get_entry("up1_avg").pc_source == "missing"
    assert store.get_pc("up1_avg") is None


# --- the consumers that must refuse a PC nobody supplied -----------------------

def test_refine_refuses_a_file_without_a_pc(nickel, tmp_path, monkeypatch):
    """Dictionary refinement used to fall back to the signal's own detector, i.e.
    the placeholder, and refine against it."""
    import asyncio
    from types import SimpleNamespace

    from fastapi import HTTPException

    from backend.api.routes import ebsd_viewer
    from backend.api.routes import indexing as I

    store = CalibrationStore()
    store.register("up1", _signal(nickel, "default"))
    master = tmp_path / "master.h5"
    master.write_bytes(b"x")
    monkeypatch.setattr(I, "calibration_store", store)
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", store)
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "up1")
    monkeypatch.setattr(I, "_get_result",
                        lambda *a, **k: SimpleNamespace(method=SimpleNamespace(value="dictionary")))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(I.refine_orientations_endpoint(I.RefineRequest(master_h5_path=str(master))))
    assert exc.value.status_code == 400
    # the coded refusal the indexing starts give (the page translates the code)
    assert exc.value.headers["X-Orienta-Code"] == "pc_missing"
    assert "carries no pattern centre" in exc.value.detail


def test_copy_pc_will_not_spread_a_placeholder(nickel, monkeypatch):
    import asyncio

    from fastapi import HTTPException

    from backend.api.routes import batch_v2 as B

    store = CalibrationStore()
    from backend.api.services.batch_manager import _canonical_file
    store.register("up1", _signal(nickel, "default"), file_key=_canonical_file("/d/up1.up1"))
    store.register("other", _signal(nickel, None), file_key=_canonical_file("/d/other.up1"))
    monkeypatch.setattr(B, "calibration_store", store)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(B.copy_pc(B.CopyPCRequest(
            source_dataset="up1", target_datasets=["other"],
            source_file_path="/d/up1.up1", target_file_paths=["/d/other.up1"])))
    assert exc.value.status_code == 400
    assert "no pattern centre" in exc.value.detail
