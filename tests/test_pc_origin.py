"""Where a file's pattern centre came from (apex / osc / header / missing) and who is told."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import safe_loader  # noqa: E402
from backend.api.services.calibration_store import CalibrationStore, pc_origin_of  # noqa: E402


class _Meta:
    def __init__(self, tag):
        self.tag = tag

    def get_item(self, key, default=None):
        return self.tag if key == "Signal.pc_source" and self.tag is not None else default

    def set_item(self, key, value):
        assert key == "Signal.pc_source"
        self.tag = value


class _Sig:
    """A stand-in signal: a detector (Bruker PC) and the loader's tag."""

    def __init__(self, tag=None, pc=(0.5, 0.5, 0.5)):
        self.detector = SimpleNamespace(shape=(60, 60), pc=np.array(pc), sample_tilt=70.0,
                                        tilt=0.0, azimuthal=0.0)
        self.metadata = _Meta(tag)


# ----------------------------------------------------------------- vocabulary

@pytest.mark.parametrize("tag, origin", [
    ("apex", "apex"), ("osc", "osc"), ("orienta_export", "orienta_export"),
    ("default", "missing"), (None, "header"),
])
def test_the_loaders_tag_names_one_of_the_origins(tag, origin):
    assert pc_origin_of(_Sig(tag)) == origin


def test_a_signal_without_metadata_is_a_header_pc():
    assert pc_origin_of(SimpleNamespace(detector=None)) == "header"


def test_register_records_the_origin_next_to_the_source():
    store = CalibrationStore()
    apex = store.register("a", _Sig("apex", pc=(0.5, 0.6, 0.6)))
    assert (apex.pc_source, apex.pc_origin) == ("header", "apex")
    assert store.get_pc("a") is not None
    miss = store.register("m", _Sig("default"))
    assert (miss.pc_source, miss.pc_origin) == ("missing", "missing")
    assert store.get_pc("m") is None                    # the guards that need a real PC fail loud
    assert store.register("h", _Sig(None)).pc_origin == "header"
    assert store.get_entry("a").to_dict()["pc_origin"] == "apex"


def test_a_typed_pc_changes_the_source_not_the_origin():
    store = CalibrationStore()
    store.register("a", _Sig("apex"))
    store.update_pc("a", [0.51, 0.40, 0.66], source="manual")
    entry = store.get_entry("a")
    assert (entry.pc_source, entry.pc_origin) == ("manual", "apex")


def test_copies_crops_and_the_stash_keep_the_origin():
    store = CalibrationStore()
    store.register("raw", _Sig("apex"))
    assert store.register_derived("copy", "raw").pc_origin == "apex"
    window = SimpleNamespace(row0=0, col0=0, rows=2, cols=2, apply=lambda a: np.asarray(a)[:2, :2])
    assert store.register_cropped("crop", "raw", window).pc_origin == "apex"
    assert store.snapshot(["raw", "copy", "crop"])["crop"].pc_origin == "apex"
    miss = CalibrationStore()
    miss.register("raw", _Sig("default"))
    assert miss.register_derived("copy", "raw").pc_origin == "missing"


def test_a_restored_calibration_does_not_overwrite_what_the_fresh_file_says():
    store = CalibrationStore()
    store.register("raw", _Sig("apex"))
    store.update_pc("raw", [0.51, 0.40, 0.66], source="manual")
    stash = store.snapshot(["raw"])
    store.clear()
    store.register("raw", _Sig(None))                      # the file now loads without a tag
    store.restore(stash, skip="raw")
    assert store.get_entry("raw").pc_origin == "header"    # the fresh entry's own origin


def test_the_load_answer_carries_the_origin(monkeypatch):
    from backend.api.routes import ebsd_viewer
    store = CalibrationStore()
    monkeypatch.setattr(ebsd_viewer, "calibration_store", store)
    store.register("ds", _Sig("apex"))
    assert ebsd_viewer._pc_state_for_response("ds", "apex", _Sig("apex"))["pc_origin"] == "apex"
    # no entry registered (a load path that never registered one): from the signal's own tag
    assert ebsd_viewer._pc_state_for_response("none", "default", _Sig("default"))["pc_origin"] == "missing"


# --------------------------------------------------------------- loader tags

@pytest.mark.parametrize("name", [
    "Pattern.dat", "Setting.txt", "x.ebsp", "a.bmp", "a.PNG", "a.tif", "a.TIFF"])
def test_a_pc_less_format_is_tagged_default(name):
    sig = _Sig(None)
    safe_loader._tag_pc_less_formats(sig, f"/data/{name}")
    assert sig.metadata.get_item("Signal.pc_source") == "default"
    assert pc_origin_of(sig) == "missing"


@pytest.mark.parametrize("name", ["a.h5oina", "a.h5", "a.hdf5", "a.up1", "a.up2", "a.ang"])
def test_formats_that_carry_a_pc_or_tag_themselves_are_left_alone(name):
    sig = _Sig(None)
    safe_loader._tag_pc_less_formats(sig, f"/data/{name}")
    assert sig.metadata.get_item("Signal.pc_source") is None


def test_a_tag_a_loader_set_is_not_overwritten():
    sig = _Sig("osc")
    safe_loader._tag_pc_less_formats(sig, "/data/Pattern.dat")
    assert sig.metadata.get_item("Signal.pc_source") == "osc"


def test_the_wrapper_tags_what_the_implementation_returns(monkeypatch):
    sig = _Sig(None)
    seen = {}

    def impl(file_path, pattern_type, verbose, apex, variant):
        # the wrapper hands the implementation everything it was given, the variant included
        seen.update(path=file_path, apex=apex, variant=variant)
        return sig

    monkeypatch.setattr(safe_loader, "_load_ebsd_safe_impl", impl)
    out = safe_loader.load_ebsd_safe("C:/x/Pattern.dat", verbose=False, apex={"k": 1})
    assert out is sig
    assert sig.metadata.get_item("Signal.pc_source") == "default"
    assert seen == {"path": "C:/x/Pattern.dat", "apex": {"k": 1}, "variant": None}
    out = safe_loader.load_ebsd_safe("C:/x/Pattern.dat", verbose=False, variant="a variant")
    assert seen["variant"] == "a variant"


def test_a_failed_tag_is_a_visible_warning_and_the_signal_is_still_returned(monkeypatch, caplog):
    import logging

    class _Broken:
        def get_item(self, key, default=None):
            raise RuntimeError("metadata unreadable")

    sig = _Sig(None)
    sig.metadata = _Broken()
    monkeypatch.setattr(safe_loader, "_load_ebsd_safe_impl", lambda *a, **k: sig)
    with caplog.at_level(logging.DEBUG, logger=safe_loader.logger.name):
        out = safe_loader.load_ebsd_safe("C:/x/Scan7.ebsp", verbose=False)
    assert out is sig
    warned = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warned) == 1
    text = warned[0].getMessage()
    assert "Scan7.ebsp" in text and "RuntimeError" in text


# ------------------------------------------------- typing a PC on a file without one

def test_typing_a_pc_on_a_file_without_one_unblocks_it_but_not_its_origin():
    from backend.api.services.pc_guard import is_pc_missing

    store = CalibrationStore()
    store.register("m", _Sig("default"))
    assert is_pc_missing("m", store) is True
    assert store.get_pc("m") is None
    typed = [0.48, 0.52, 0.60]                                # the numbers the user typed
    assert store.update_pc("m", typed, source="manual") is True
    entry = store.get_entry("m")
    assert entry.pc_source == "manual"
    assert is_pc_missing("m", store) is False
    assert store.get_pc("m").tolist() == pytest.approx(typed)
    assert entry.pc_origin == "missing"                       # the FILE still offered nothing


# ------------------------------------------------------------------ /pc/status

def test_pc_status_names_the_origin_and_keeps_it_when_the_pc_is_typed(monkeypatch):
    from fastapi.testclient import TestClient
    from kikuchipy.detectors import EBSDDetector

    from backend.api.main import app
    from backend.api.routes import ebsd_viewer
    from backend.api.routes import pcrefinement as R
    from pc_controller import PCController

    store = CalibrationStore()
    ctrl = PCController()
    ctrl.attach_detector(EBSDDetector(shape=(60, 60), pc=(0.5, 0.5, 0.5), sample_tilt=70.0))
    monkeypatch.setattr(R, "_get_controller", lambda: ctrl)
    monkeypatch.setattr(R, "calibration_store", store)
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "ds")
    client = TestClient(app)

    assert client.get("/api/pc/status").json()["pc_origin"] is None       # nothing registered yet
    store.register("ds", _Sig("apex", pc=(0.5, 0.6, 0.6)))
    body = client.get("/api/pc/status").json()
    assert (body["pc_source"], body["pc_origin"]) == ("header", "apex")
    store.update_pc("ds", [0.48, 0.52, 0.6], source="manual")
    body = client.get("/api/pc/status").json()
    assert (body["pc_source"], body["pc_origin"]) == ("manual", "apex")
