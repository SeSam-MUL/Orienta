"""Loading a chosen variant (scan group x pattern source) through ``safe_loader.load_ebsd_safe``,
and the registry that remembers which variant a dataset holds."""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import safe_loader  # noqa: E402
from backend.api.services import variant_registry  # noqa: E402
from backend.api.services.load_choices import LoadChoiceError, LoadVariant  # noqa: E402
from tests.test_load_choices import _sampleb, _text  # noqa: E402

NY, NX, SY, SX = 2, 3, 4, 5
N = NY * NX


def _processed(g: int) -> np.ndarray:
    return (np.arange(N * SY * SX) % 200 + 10 * g).astype(np.uint8).reshape(N, SY, SX)


def _unprocessed(g: int) -> np.ndarray:
    return (np.arange(N * SY * SX) * 7 - 500 + 1000 * g).astype(np.int16).reshape(N, SY, SX)


def _group(f, name: int, *, processed=True, unprocessed=True, processed_bg=True, unprocessed_bg=True):
    g = str(name)
    hdr = f.create_group(f"{g}/EBSD/Header")
    data = f.create_group(f"{g}/EBSD/Data")
    hdr.create_dataset("X Cells", data=NX)
    hdr.create_dataset("Y Cells", data=NY)
    hdr.create_dataset("X Step", data=0.5)
    hdr.create_dataset("Y Step", data=0.5)
    hdr.create_dataset("Pattern Height", data=SY)
    hdr.create_dataset("Pattern Width", data=SX)
    for pc_name, value in (("Pattern Center X", 0.5), ("Pattern Center Y", 0.6), ("Detector Distance", 0.7)):
        data.create_dataset(pc_name, data=np.full(N, value, np.float32))
    if processed:
        data.create_dataset("Processed Patterns", data=_processed(name))
    if unprocessed:
        data.create_dataset("Unprocessed Patterns", data=_unprocessed(name))
    if processed_bg:
        hdr.create_dataset("Processed Static Background", data=np.full((SY, SX), 3 + name, np.uint8))
    if unprocessed_bg:
        hdr.create_dataset("Unprocessed Static Background", data=np.full((SY, SX), 900 + name, np.int16))


def _file(tmp_path, name, groups):
    path = tmp_path / name
    with h5py.File(path, "w") as f:
        _text(f, "Manufacturer", "Oxford Instruments")
        _text(f, "Format Version", "5.0")
        for number, kw in groups:
            _group(f, number, **kw)
    return str(path)


@pytest.fixture
def multi(tmp_path):
    """Groups 1, 2 and 10 (HDF5 key order is 1, 10, 2). Group 2 has no unprocessed background."""
    return _file(tmp_path, "multi.h5oina", [
        (1, {}), (2, {"unprocessed_bg": False}), (10, {}),
    ])


@pytest.fixture
def only_unprocessed(tmp_path):
    return _file(tmp_path, "unproc.h5oina", [(1, {"processed": False, "processed_bg": False})])


def _load(path, variant=None):
    return safe_loader.load_ebsd_safe(path, verbose=False, variant=variant)


def _data(sig):
    return np.asarray(sig.data)


def test_unprocessed_variant_loads_int16_counts(multi):
    sig = _load(multi, LoadVariant("1", "unprocessed"))
    assert _data(sig).dtype == np.int16
    assert np.array_equal(_data(sig), _unprocessed(1).reshape(NY, NX, SY, SX))


def test_processed_variant_loads_uint8(multi):
    sig = _load(multi, LoadVariant("1", "processed"))
    assert _data(sig).dtype == np.uint8
    assert np.array_equal(_data(sig), _processed(1).reshape(NY, NX, SY, SX))


def test_second_group_loads(multi):
    # kikuchipy's own "first group" would be 1 (and its key order 1, 10, 2): group 2 must be named.
    sig = _load(multi, LoadVariant("2", "processed"))
    assert np.array_equal(_data(sig), _processed(2).reshape(NY, NX, SY, SX))
    sig = _load(multi, LoadVariant("10", "unprocessed"))
    assert np.array_equal(_data(sig), _unprocessed(10).reshape(NY, NX, SY, SX))


def test_scan_group_and_source_reach_kikuchipy_explicitly(multi, monkeypatch):
    calls = []
    real = safe_loader._kp().load

    def spy(path, **kw):
        calls.append(kw)
        return real(path, **kw)

    monkeypatch.setattr(safe_loader._kp(), "load", spy)
    _load(multi, LoadVariant("1", "processed"))     # the default one is named too
    _load(multi, LoadVariant("2", "unprocessed"))
    assert calls[0]["scan_group_names"] == ["1"] and calls[0]["processed"] is True
    assert calls[1]["scan_group_names"] == ["2"] and calls[1]["processed"] is False


def test_background_matches_source(multi):
    unproc = _load(multi, LoadVariant("1", "unprocessed"))
    assert unproc.static_background.dtype == np.int16
    assert np.array_equal(unproc.static_background, np.full((SY, SX), 901, np.int16))
    proc = _load(multi, LoadVariant("1", "processed"))
    assert np.array_equal(proc.static_background, np.full((SY, SX), 4, np.uint8))


def test_unprocessed_without_its_own_background_gets_none(multi):
    # group 2 has a Processed Static Background but no Unprocessed one: never the processed one
    sig = _load(multi, LoadVariant("2", "unprocessed"))
    assert sig.static_background is None
    assert np.array_equal(_data(sig), _unprocessed(2).reshape(NY, NX, SY, SX))


def test_only_unprocessed_file_loads(only_unprocessed):
    sig = _load(only_unprocessed, LoadVariant("1", "unprocessed"))
    assert _data(sig).dtype == np.int16
    assert np.array_equal(_data(sig), _unprocessed(1).reshape(NY, NX, SY, SX))
    assert sig.static_background is not None and sig.static_background.dtype == np.int16


def test_a_variant_the_file_does_not_hold_is_refused_with_a_code(multi, tmp_path):
    with pytest.raises(LoadChoiceError) as e:
        _load(multi, LoadVariant("7", "processed"))
    assert e.value.code == "loadVariantUnknown"
    only_proc = _file(tmp_path, "p.h5oina", [(1, {"unprocessed": False, "unprocessed_bg": False})])
    with pytest.raises(LoadChoiceError) as e:
        _load(only_proc, LoadVariant("1", "unprocessed"))
    assert e.value.code == "loadVariantUnknown"
    # a pattern source this format does not have
    with pytest.raises(LoadChoiceError):
        _load(multi, LoadVariant("1", "single"))


def test_a_measurement_without_patterns_is_refused_with_a_code(tmp_path):
    path = _file(tmp_path, "e.h5oina", [(1, {}), (2, {"processed": False, "unprocessed": False})])
    with pytest.raises(LoadChoiceError) as e:
        _load(path, LoadVariant("2", "processed"))
    assert e.value.code == "loadNoPatterns"


def test_default_is_unchanged(multi):
    kp = safe_loader._kp()
    with safe_loader._kikuchipy_oxford_camera_binning_workaround():
        direct = kp.load(multi, lazy=False)
    sig = _load(multi)
    assert _data(sig).dtype == _data(direct).dtype == np.uint8
    assert _data(sig).tobytes() == _data(direct).tobytes()
    assert np.array_equal(sig.static_background, direct.static_background)
    assert sig.metadata.get_item("Orienta.load_variant", None) is None   # no choice, no mirror
    # asking for the default explicitly is the same data
    again = _load(multi, LoadVariant("1", "processed"))
    assert _data(again).tobytes() == _data(direct).tobytes()


def test_variant_mirror_is_the_resolved_variant(multi):
    sig = _load(multi, LoadVariant(None, "unprocessed"))
    assert sig.metadata.Orienta.load_variant.as_dictionary() == {
        "measurement": "1", "pattern_source": "unprocessed"}
    sig = _load(multi, LoadVariant("2", "processed"))
    assert sig.metadata.Orienta.load_variant.as_dictionary() == {
        "measurement": "2", "pattern_source": "processed"}


def test_unified_loader_fallback_loads_the_chosen_source(multi, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("native reader unavailable")

    monkeypatch.setattr(safe_loader._kp(), "load", boom)
    monkeypatch.setattr(safe_loader, "_load_oxford_by_layout", boom)
    sig = _load(multi, LoadVariant("1", "unprocessed"))
    assert np.array_equal(_data(sig), _unprocessed(1).reshape(NY, NX, SY, SX))
    assert np.array_equal(sig.static_background, np.full((SY, SX), 901, np.int16))
    sig = _load(multi, LoadVariant("1", "processed"))
    assert np.array_equal(_data(sig), _processed(1).reshape(NY, NX, SY, SX))
    assert np.array_equal(sig.static_background, np.full((SY, SX), 4, np.uint8))


def test_unified_loader_fallback_refuses_what_it_cannot_choose(multi, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("native reader unavailable")

    monkeypatch.setattr(safe_loader._kp(), "load", boom)
    monkeypatch.setattr(safe_loader, "_load_oxford_by_layout", boom)
    with pytest.raises(LoadChoiceError) as e:        # the fallback reads the first group only
        _load(multi, LoadVariant("2", "processed"))
    assert e.value.code == "loadVariantNeedsNativeReader"


def test_oxford_by_layout_takes_the_variant(multi, tmp_path):
    import shutil
    renamed = str(tmp_path / "copy.h5")                 # an Oxford file not named .h5oina
    shutil.copy(multi, renamed)
    sig = _load(renamed, LoadVariant("2", "unprocessed"))
    assert np.array_equal(_data(sig), _unprocessed(2).reshape(NY, NX, SY, SX))
    assert sig.static_background is None


# ----------------------------------------------------------------------------- registry


def _signal(mirror=None):
    import hyperspy.api as hs
    sig = hs.signals.Signal2D(np.zeros((2, 2, 2)))
    if mirror is not None:
        sig.metadata.set_item("Orienta.load_variant", mirror)
    return sig

@pytest.fixture(autouse=True)
def _clean_registry():
    variant_registry.clear_all()
    yield
    variant_registry.clear_all()


def test_registry_register_and_lookup():
    v = LoadVariant("2", "unprocessed")
    assert variant_registry.variant_for("scan") is None
    variant_registry.register("scan", v)
    assert variant_registry.variant_for("scan") == v
    variant_registry.register("scan", LoadVariant("1", "processed"))      # loading again replaces
    assert variant_registry.variant_for("scan") == LoadVariant("1", "processed")


def test_registry_follows_deepcopy_and_crop():
    v = LoadVariant("2", "unprocessed")
    variant_registry.register("scan", v)
    assert variant_registry.copy_to("scan", "scan_copy") == v           # a deep copy
    assert variant_registry.copy_to("scan_copy", "scan_copy_crop") == v  # a crop of the copy
    assert variant_registry.variant_for("scan_copy") == v
    assert variant_registry.variant_for("scan_copy_crop") == v
    assert variant_registry.copy_to("unknown", "x") is None             # nothing to follow
    assert variant_registry.variant_for("x") is None


def test_registry_drops_with_the_dataset_only():
    variant_registry.register("scan", LoadVariant("1", "processed"))
    variant_registry.copy_to("scan", "scan_copy")
    variant_registry.drop("scan")
    assert variant_registry.variant_for("scan") is None
    assert variant_registry.variant_for("scan_copy") == LoadVariant("1", "processed")
    variant_registry.drop("never_registered")                           # no error


def test_registry_clear_all():
    variant_registry.register("a", LoadVariant("1", "processed"))
    variant_registry.register("b", LoadVariant("2", "processed"))
    variant_registry.clear_all()
    assert variant_registry.variant_for("a") is None and variant_registry.variant_for("b") is None


def test_registry_snapshot_and_restore_keep_what_is_live():
    variant_registry.register("a", LoadVariant("1", "processed"))
    variant_registry.register("b", LoadVariant("2", "unprocessed"))
    snap = variant_registry.snapshot(["a", "b", "missing"])
    assert set(snap) == {"a", "b"}
    variant_registry.clear_all()
    variant_registry.register("a", LoadVariant("7", "processed"))       # freshly loaded raw dataset
    variant_registry.register("c", LoadVariant("1", "processed"))
    variant_registry.restore(snap, skip="c")
    assert variant_registry.variant_for("a") == LoadVariant("7", "processed")   # live wins
    assert variant_registry.variant_for("b") == LoadVariant("2", "unprocessed")
    assert variant_registry.variant_for("c") == LoadVariant("1", "processed")


def test_register_from_signal_reads_the_mirror_once_and_drops_it():
    sig = _signal({"measurement": "2", "pattern_source": "unprocessed"})
    got = variant_registry.register_from_signal("scan", sig)
    assert got == LoadVariant("2", "unprocessed")
    assert variant_registry.variant_for("scan") == got
    assert sig.metadata.get_item("Orienta.load_variant", None) is None        # transport, dropped
    assert variant_registry.register_from_signal("again", sig) is None        # nothing left to read
    assert variant_registry.variant_for("again") is None


def test_register_from_signal_without_a_mirror_clears_a_stale_entry():
    variant_registry.register("scan", LoadVariant("2", "unprocessed"))
    sig = _signal()
    assert variant_registry.register_from_signal("scan", sig) is None
    assert variant_registry.variant_for("scan") is None


def test_register_from_signal_survives_a_stand_in_signal():
    assert variant_registry.register_from_signal("scan", object()) is None


def test_loaded_signal_registers_through_its_mirror(multi):
    sig = _load(multi, LoadVariant("2", "processed"))
    got = variant_registry.register_from_signal("multi", sig)
    assert got == LoadVariant("2", "processed")
    assert sig.metadata.get_item("Orienta.load_variant", None) is None


# ----------------------------------------------------------------------------- the Oxford reference scan


def _sampleb_or_skip():
    path = _sampleb()
    if path is None:
        pytest.skip("EBSD_SampleB real file is not available here")
    return str(path)


def test_sampleb_unprocessed_is_int16_and_equals_the_file(monkeypatch):
    path = _sampleb_or_skip()
    monkeypatch.setenv("KIKUCHIPY_LAZY_THRESHOLD_BYTES", "1")        # lazy: a few patterns are read, not 270 MB
    sig = _load(path, LoadVariant("1", "unprocessed"))
    assert sig.data.dtype == np.int16
    with h5py.File(path, "r") as f:
        ds = f["1/EBSD/Data/Unprocessed Patterns"]
        assert ds.dtype == np.int16
        n_cols = int(f["1/EBSD/Header/X Cells"][()])
        for index in (0, 1, ds.shape[0] // 2, ds.shape[0] - 1):
            row, col = divmod(index, n_cols)
            assert np.array_equal(np.asarray(sig.data[row, col]), ds[index])


def test_sampleb_background_is_its_sources_own(monkeypatch):
    path = _sampleb_or_skip()
    monkeypatch.setenv("KIKUCHIPY_LAZY_THRESHOLD_BYTES", "1")
    with h5py.File(path, "r") as f:
        hdr = f["1/EBSD/Header"]
        has_unproc = "Unprocessed Static Background" in hdr
        has_proc = "Processed Static Background" in hdr
        unproc = hdr["Unprocessed Static Background"][()] if has_unproc else None
        proc = hdr["Processed Static Background"][()] if has_proc else None
    sig = _load(path, LoadVariant("1", "unprocessed"))
    if has_unproc:
        assert np.array_equal(np.asarray(sig.static_background), unproc)
    else:
        assert sig.static_background is None
    sig = _load(path, LoadVariant("1", "processed"))
    if has_proc:
        assert np.array_equal(np.asarray(sig.static_background), proc)
    else:
        assert sig.static_background is None
