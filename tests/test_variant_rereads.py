"""Every re-read of a loaded file takes the loaded variant (measurement x pattern source).

A file can hold two copies of its patterns (Oxford: processed uint8 and unprocessed int16) in
several scan groups. Loading picks one; everything that later opens the file BY PATH (the external
EMSphInx binary, the in-process GPU indexer, the ROI read, the exports, the image-quality read)
must read the same one, or write into the same one. These tests run on one synthetic file whose
processed patterns are all 10 and whose unprocessed patterns are NON-constant int16 counts above
255, so a read of the wrong dataset, a missing write and a wrap to uint8 each give a different
number.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import indexing_controller as ic  # noqa: E402
from backend.api.services import load_choices as lc  # noqa: E402
from backend.api.services.load_choices import LoadChoiceError, LoadVariant  # noqa: E402

NY, NX, SY, SX = 2, 3, 4, 5
N = NY * NX

PROC = LoadVariant("1", "processed")
RAW = LoadVariant("1", "unprocessed")
RAW2 = LoadVariant("2", "unprocessed")
PROC2 = LoadVariant("2", "processed")


def processed(g: int) -> np.ndarray:
    return np.full((N, SY, SX), 10 * g, np.uint8)


def unprocessed(g: int) -> np.ndarray:
    return (np.arange(N * SY * SX) * 3 + 300 + 1000 * (g - 1)).astype(np.int16).reshape(N, SY, SX)


def _text(f, name, value):
    f.create_dataset(name, data=np.bytes_(value))


def _group(f, g: int):
    hdr = f.create_group(f"{g}/EBSD/Header")
    data = f.create_group(f"{g}/EBSD/Data")
    for key, value in (("X Cells", NX), ("Y Cells", NY), ("X Step", 0.5), ("Y Step", 0.5),
                       ("Pattern Height", SY), ("Pattern Width", SX)):
        hdr.create_dataset(key, data=value)
    for pc_name, value in (("Pattern Center X", 0.5), ("Pattern Center Y", 0.6), ("Detector Distance", 0.7)):
        data.create_dataset(pc_name, data=np.full(N, value, np.float32))
    data.create_dataset("Processed Patterns", data=processed(g))
    data.create_dataset("Unprocessed Patterns", data=unprocessed(g))


@pytest.fixture
def two_sources(tmp_path):
    """An Oxford file with measurements 1 and 2, each with both pattern sources."""
    path = tmp_path / "multi.h5oina"
    with h5py.File(path, "w") as f:
        _text(f, "Manufacturer", "Oxford Instruments")
        _text(f, "Format Version", "5.0")
        _group(f, 1)
        _group(f, 2)
    return str(path)


@pytest.fixture
def unlisted_layout(tmp_path):
    """An HDF5 file whose layout the load choices do not describe (no manufacturer, no scan group)."""
    path = tmp_path / "odd.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("raw/patterns", data=np.zeros((N, SY, SX), np.uint8))
    return str(path)


EXPECTED = [
    (None, "1/EBSD/Data/Processed Patterns", lambda: processed(1)),
    (RAW, "1/EBSD/Data/Unprocessed Patterns", lambda: unprocessed(1)),
    (PROC2, "2/EBSD/Data/Processed Patterns", lambda: processed(2)),
    (RAW2, "2/EBSD/Data/Unprocessed Patterns", lambda: unprocessed(2)),
]
EXPECTED_IDS = ["default", "unprocessed", "measurement-2", "measurement-2-unprocessed"]


# ----------------------------------------------------------------------- dataset path (EMSphInx)

@pytest.mark.parametrize("variant,path,_values", EXPECTED, ids=EXPECTED_IDS)
def test_get_pattern_dataset_path_is_the_variants(two_sources, variant, path, _values):
    assert ic._get_pattern_dataset_path(two_sources, variant) == path


def test_get_pattern_dataset_path_without_a_variant_is_todays_default(two_sources):
    assert ic._get_pattern_dataset_path(two_sources) == "1/EBSD/Data/Processed Patterns"


def _nml_config():
    return SimpleNamespace(sht_file="master.sht", bandwidth=88, normed=True, refine=False,
                           circmask=0, gausbckg=False, nregions=10)


def _nml(source, out_dir, variant):
    det = {"n_rows": NY, "n_cols": NX, "pat_width": SX, "pat_height": SY}
    return ic.generate_emsphinx_nml(_nml_config(), source, det, out_dir, variant=variant)


@pytest.mark.parametrize("variant,path,values", EXPECTED, ids=EXPECTED_IDS)
def test_emsphinx_dataset_path_is_the_variants(two_sources, tmp_path, variant, path, values):
    out = tmp_path / "out"
    out.mkdir()
    nml = _nml(two_sources, out, variant).read_text(encoding="utf-8")
    assert f"patdset    = '{path}'," in nml
    compat = out / "multi_compat.h5"
    with h5py.File(compat, "r") as f:                      # the file the binary reads holds that dataset
        stored = f[path][...]
    expected = values()
    if expected.dtype.kind == "i":
        # A signed-integer scan reaches EMSphInx as its 8 bit rescale by the min/max of the whole
        # stack (the rule of the int16 task), never as a wrap modulo 256 and never as another source.
        lo, hi = float(expected.min()), float(expected.max())
        expected = np.rint((expected.astype(np.float64) - lo) * 255.0 / (hi - lo)).astype(np.uint8)
    assert stored.dtype == expected.dtype
    assert np.array_equal(stored, expected)


def test_emsphinx_nml_refuses_a_variant_the_layout_cannot_honour(unlisted_layout, tmp_path):
    with pytest.raises(LoadChoiceError) as exc:
        _nml(unlisted_layout, tmp_path, RAW)
    assert exc.value.code == "variantNotSupported"
    assert exc.value.params == {"where": "_get_pattern_dataset_path"}


# ------------------------------------------------------------------ the temporary H5 (EMSphInx)

class _Signal:
    def __init__(self, data):
        self.data = data


@pytest.fixture
def viewer(two_sources, tmp_path, monkeypatch):
    """The viewer shows the file, edited in memory: float patterns with negative values (a
    background-corrected stack), non-constant."""
    from backend.api.routes import ebsd_viewer
    edited = (np.arange(N * SY * SX, dtype=np.float64).reshape(NY, NX, SY, SX) - 40.0) * 2.5
    monkeypatch.setattr(ebsd_viewer, "is_active_signal_dirty", lambda: True)
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda: _Signal(edited.copy()))
    monkeypatch.setattr(ebsd_viewer, "_ebsd_file_path", two_sources)
    monkeypatch.setattr(ic, "_get_emsoft_data_path", lambda: tmp_path / "emsoft")
    return SimpleNamespace(path=two_sources, edited=edited.reshape(N, SY, SX))


def _temp_contents(temp):
    with h5py.File(temp, "r") as f:
        return {"proc": f["1/EBSD/Data/Processed Patterns"][...],
                "raw": f["1/EBSD/Data/Unprocessed Patterns"][...],
                "proc2": f["2/EBSD/Data/Processed Patterns"][...]}


def test_make_processed_temp_h5_writes_the_variants_dataset(viewer):
    temp = ic._make_processed_temp_h5(viewer.path, variant=RAW)
    try:
        assert temp is not None
        got = _temp_contents(temp)
        assert np.array_equal(got["proc"], processed(1))   # the processed copy is not touched
        assert np.array_equal(got["proc2"], processed(2))
        assert got["raw"].dtype == np.int16                # the dataset keeps its dtype
        assert not np.array_equal(got["raw"], unprocessed(1))   # and now holds the viewer's patterns
    finally:
        Path(temp).unlink(missing_ok=True)


def test_make_processed_temp_h5_int16_is_rescaled_not_clipped(viewer):
    temp = ic._make_processed_temp_h5(viewer.path, variant=RAW)
    try:
        raw = _temp_contents(temp)["raw"].astype(np.int64)
    finally:
        Path(temp).unlink(missing_ok=True)
    assert raw.min() == 0 and raw.max() == np.iinfo(np.int16).max     # the whole int16 range, no wrap
    flat_in, flat_out = viewer.edited.ravel(), raw.ravel()
    assert np.array_equal(np.argsort(flat_in, kind="stable"), np.argsort(flat_out, kind="stable"))


def test_make_processed_temp_h5_without_a_variant_writes_the_processed_dataset(viewer):
    temp = ic._make_processed_temp_h5(viewer.path)
    try:
        got = _temp_contents(temp)
        assert not np.array_equal(got["proc"], processed(1))         # today: the processed dataset is written
        assert got["proc"].dtype == np.uint8
        assert np.array_equal(got["raw"], unprocessed(1))            # and the unprocessed one is not
    finally:
        Path(temp).unlink(missing_ok=True)


def test_make_processed_temp_h5_refuses_a_variant_the_layout_cannot_honour(viewer, unlisted_layout, monkeypatch):
    from backend.api.routes import ebsd_viewer
    monkeypatch.setattr(ebsd_viewer, "_ebsd_file_path", unlisted_layout)
    with pytest.raises(LoadChoiceError) as exc:
        ic._make_processed_temp_h5(unlisted_layout, variant=RAW)
    assert exc.value.code == "variantNotSupported"
    assert exc.value.params == {"where": "_make_processed_temp_h5"}


def test_spherical_index_patterns_hands_the_variant_on(viewer, monkeypatch):
    seen = {}

    def temp(h5, cb=None, use_viewer_patterns=True, variant=None):
        seen["temp"] = variant

    def impl(**kw):
        seen["impl"] = kw.get("variant")
        return "r"

    monkeypatch.setattr(ic, "_make_processed_temp_h5", temp)
    monkeypatch.setattr(ic, "_spherical_index_patterns_impl", impl)
    monkeypatch.setattr(ic, "_active_scan_is_hex", lambda: False)
    ic.spherical_index_patterns(
        h5_path=viewer.path, config=ic.IndexingConfig(method=ic.IndexingMethod.SPHERICAL),
        detector_params={}, variant=RAW)
    assert seen == {"temp": RAW, "impl": RAW}


def test_the_emsphinx_run_names_the_variant_in_its_nml_and_keeps_a_refusals_code(two_sources, tmp_path, monkeypatch):
    """``_spherical_index_patterns_impl`` hands the variant to the NML builder, and a coded refusal
    from there is not re-worded into a RuntimeError (which would lose its code)."""
    sht = tmp_path / "master.sht"
    sht.write_bytes(b"x")
    seen = {}

    def nml(config, h5_path, detector_params, output_dir, progress_callback=None, crop_region=None,
            variant=None):
        seen["variant"] = variant
        raise LoadChoiceError("variantNotSupported", "refused", {"where": "_get_pattern_dataset_path"})

    monkeypatch.setattr(ic, "_find_indexebsd", lambda: "IndexEBSD")
    monkeypatch.setattr(ic, "_get_emsoft_data_path", lambda: tmp_path / "emsoft")
    monkeypatch.setattr(ic, "generate_emsphinx_nml", nml)
    config = ic.IndexingConfig(method=ic.IndexingMethod.SPHERICAL, sht_file=str(sht))
    with pytest.raises(LoadChoiceError) as exc:
        ic._spherical_index_patterns_impl(
            two_sources, config, {"n_rows": NY, "n_cols": NX}, variant=RAW)
    assert seen["variant"] == RAW
    assert exc.value.code == "variantNotSupported"


# -------------------------------------------------------------------------- the stack read

@pytest.mark.parametrize("variant,path,values", EXPECTED, ids=EXPECTED_IDS)
def test_read_h5_pattern_stack_reads_the_variants(two_sources, variant, path, values):
    stack = ic._read_h5_pattern_stack(two_sources, variant=variant)
    assert stack.dtype == values().dtype
    assert np.array_equal(stack, values())
    part = ic._read_h5_pattern_stack(two_sources, indices=np.array([1, 4]), variant=variant)
    assert np.array_equal(part, values()[[1, 4]])


def test_read_h5_pattern_stack_without_a_variant_is_todays_default(two_sources):
    assert np.array_equal(ic._read_h5_pattern_stack(two_sources), processed(1))


def test_read_h5_pattern_stack_refuses_a_variant_the_layout_cannot_honour(unlisted_layout):
    with pytest.raises(LoadChoiceError) as exc:
        ic._read_h5_pattern_stack(unlisted_layout, variant=RAW)
    assert exc.value.code == "variantNotSupported"
    assert exc.value.params == {"where": "_read_h5_pattern_stack"}


def test_materialised_patterns_for_pseudosymmetry_are_the_variants(two_sources):
    meta = [{"formula": "x", "point_group": "-43m", "z_rot": 2}]
    phase_id = np.ones(N, np.int8)
    pats = ic._materialize_patterns_for_resolution(two_sources, meta, phase_id, variant=RAW)
    assert pats is not None and np.array_equal(pats, unprocessed(1))
    pats = ic._materialize_patterns_for_resolution(two_sources, meta, phase_id)
    assert np.array_equal(pats, processed(1))


# --------------------------------------------------------------- the spherical GPU indexer

class _Stop(Exception):
    pass


class _FakeBackend:
    instances = []

    def __init__(self, cfg):
        self.calls = []
        self.arrays = []
        self.runtime = SimpleNamespace(device="cpu", device_name="fake", free_memory_gb=99.0,
                                       cuda_available=False)
        _FakeBackend.instances.append(self)

    def set_cancel_check(self, fn):
        pass

    def invalidate(self):
        pass

    def index_array(self, patterns=None, **kwargs):
        self.calls.append(("index_array", np.asarray(patterns).shape))
        self.arrays.append(np.array(patterns))
        raise _Stop

    def index_h5(self, h5_path=None, dataset_path=None, **kwargs):
        self.calls.append(("index_h5", str(h5_path), dataset_path))
        raise _Stop


@pytest.fixture
def fake(monkeypatch):
    import backend.spherical_gpu.backend as sgb
    from backend.api.routes import ebsd_viewer
    _FakeBackend.instances = []
    monkeypatch.setattr(sgb, "SphericalGPUBackend", _FakeBackend)
    monkeypatch.setattr(sgb, "BackendConfig", lambda phases: SimpleNamespace(phases=phases))
    monkeypatch.setattr(sgb, "PhaseConfig", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda: SimpleNamespace(data=np.zeros((NY, NX, SY, SX))))
    monkeypatch.setattr(ebsd_viewer, "is_active_signal_dirty", lambda: False)
    monkeypatch.setattr(ebsd_viewer, "get_active_include_mask", lambda: None)
    return _FakeBackend


def _gpu_run(source, mask=None, variant=None):
    cfg = SimpleNamespace(sht_file="master.sht", bandwidth=88, normed=True, refine=False,
                          circmask=0, gausbckg=False, nregions=10)
    det = {"n_rows": NY, "n_cols": NX}
    kw = {} if variant is None else {"variant": variant}
    return ic.spherical_gpu_index_patterns(
        str(source), cfg, det, selection_mask=mask, sht_paths=["master.sht"], **kw)


def _roi_mask():
    mask = np.zeros((NY, NX), dtype=bool)
    mask[0, 1] = mask[1, 2] = True                           # flat patterns 1 and 5
    return mask


@pytest.mark.parametrize("variant,path,values", EXPECTED, ids=EXPECTED_IDS)
def test_gpu_roi_read_takes_the_variants_dataset(fake, two_sources, variant, path, values):
    with pytest.raises(_Stop):
        _gpu_run(two_sources, _roi_mask(), variant)
    assert np.array_equal(fake.instances[0].arrays[0], values()[[1, 5]])


@pytest.mark.parametrize("variant,path,values", EXPECTED, ids=EXPECTED_IDS)
def test_gpu_full_map_streams_the_variants_dataset(fake, two_sources, variant, path, values):
    with pytest.raises(_Stop):
        _gpu_run(two_sources, None, variant)
    expected = None if variant is None else path
    assert fake.instances[0].calls == [("index_h5", two_sources, expected)]


def test_gpu_stream_hands_the_variant_to_the_resolver_read(fake, two_sources, monkeypatch):
    """After a streamed run the stack for the pseudo-symmetry resolver is read from the file again:
    it must be the same dataset the stream indexed."""
    import torch

    class _Streaming(_FakeBackend):
        def __init__(self, cfg):
            super().__init__(cfg)
            self._indexers = [SimpleNamespace(master=SimpleNamespace(formula="x", point_group="-43m", z_rot=2))]

        def index_h5(self, h5_path=None, dataset_path=None, **kwargs):
            return SimpleNamespace(euler_xyz=torch.zeros(N, 3), score=torch.zeros(N),
                                   phase_id=torch.ones(N, dtype=torch.int8), runtime_seconds=1.0)

    import backend.spherical_gpu.backend as sgb
    monkeypatch.setattr(sgb, "SphericalGPUBackend", _Streaming)
    seen = []

    def resolver_read(h5_path, masters_meta, phase_id, progress=None, variant=None):
        seen.append(variant)
        raise _Stop

    monkeypatch.setattr(ic, "_materialize_patterns_for_resolution", resolver_read)
    with pytest.raises(_Stop):
        _gpu_run(two_sources, None, RAW)
    assert seen == [RAW]


def test_gpu_roi_refuses_a_variant_the_layout_cannot_honour(fake, unlisted_layout):
    with pytest.raises(LoadChoiceError) as exc:
        _gpu_run(unlisted_layout, _roi_mask(), RAW)
    assert exc.value.code == "variantNotSupported"
    assert exc.value.params == {"where": "spherical_gpu_index_patterns"}


def test_the_indexer_opens_the_named_dataset(two_sources):
    """``Tier1Indexer._open_pattern_dataset`` is the read behind ``index_h5``; it needs no GPU."""
    from backend.spherical_gpu.pipeline.indexer import Tier1Indexer
    opened = Tier1Indexer._open_pattern_dataset(None, two_sources, "2/EBSD/Data/Unprocessed Patterns")
    n_total, dset, ds_path, handle = opened
    try:
        assert (n_total, ds_path) == (N, "2/EBSD/Data/Unprocessed Patterns")
        assert np.array_equal(dset[...], unprocessed(2))
    finally:
        handle.close()
    n_total, dset, ds_path, handle = Tier1Indexer._open_pattern_dataset(None, two_sources)
    try:
        assert ds_path == "1/EBSD/Data/Processed Patterns"         # no name: today's search
    finally:
        handle.close()


def test_the_stream_hands_the_dataset_to_the_open(two_sources):
    """``index_h5oina(dataset_path=...)`` opens exactly that dataset (the open is the seam)."""
    from backend.spherical_gpu.pipeline.indexer import Tier1Indexer
    seen = []

    def open_spy(path, dataset_path=None):
        seen.append((path, dataset_path))
        raise _Stop

    indexer = object.__new__(Tier1Indexer)
    indexer._open_pattern_dataset = open_spy
    with pytest.raises(_Stop):
        indexer.index_h5oina(two_sources, dataset_path="2/EBSD/Data/Unprocessed Patterns")
    assert seen == [(two_sources, "2/EBSD/Data/Unprocessed Patterns")]


def test_a_refusal_of_the_resolver_read_is_not_swallowed(unlisted_layout):
    """Reading the stack for the resolver fails soft on a read error, but a file that cannot name the
    variant is a coded refusal: the resolver must not run on another source's patterns."""
    meta = [{"formula": "x", "point_group": "-43m", "z_rot": 2}]
    with pytest.raises(LoadChoiceError) as exc:
        ic._materialize_patterns_for_resolution(unlisted_layout, meta, np.ones(N, np.int8), variant=RAW)
    assert exc.value.params == {"where": "_read_h5_pattern_stack"}


def test_the_indexer_refuses_a_dataset_that_is_not_there(two_sources):
    from backend.spherical_gpu.pipeline.indexer import Tier1Indexer
    with pytest.raises(KeyError):
        Tier1Indexer._open_pattern_dataset(None, two_sources, "3/EBSD/Data/Unprocessed Patterns")


def test_the_backend_hands_the_dataset_to_the_indexer(two_sources):
    """``index_h5(dataset_path=...)`` streams exactly that dataset, for one phase and for several."""
    from backend.spherical_gpu.backend import SphericalGPUBackend
    seen = []

    class _Indexer:
        circmask, gausbckg_on, nregions_n, normed, bandwidth = 0, False, 10, True, 88

        def index_h5oina(self, path, dataset_path=None, **kw):
            seen.append((path, dataset_path))
            raise _Stop

        def _open_pattern_dataset(self, path, dataset_path=None):
            seen.append((path, dataset_path))
            raise _Stop

    backend = object.__new__(SphericalGPUBackend)
    backend.config = SimpleNamespace(batch_size=8)
    backend._device = SimpleNamespace(type="cpu")
    backend._ensure_built = lambda det: None
    backend._auto_batch_size = lambda ix: 8
    backend._raise_if_cancelled = lambda: None
    backend._cancel_check = None
    for indexers in ([_Indexer()], [_Indexer(), _Indexer()]):
        seen.clear()
        backend._indexers = indexers
        with pytest.raises(_Stop):
            backend.index_h5(two_sources, {}, dataset_path="2/EBSD/Data/Unprocessed Patterns")
        assert seen and all(s == (two_sources, "2/EBSD/Data/Unprocessed Patterns") for s in seen)


# ------------------------------------------------------------------------------- the callers

def _calls(source: str, name: str):
    """The text of every call ``name(`` in ``source``, up to its closing parenthesis."""
    out, start = [], 0
    while True:
        i = source.find(name + "(", start)
        if i < 0:
            return out
        depth, j = 0, i + len(name)
        while True:
            ch = source[j]
            depth += ch == "("
            depth -= ch == ")"
            j += 1
            if depth == 0:
                break
        out.append(source[i:j])
        start = j


def test_the_indexing_routes_pass_the_variant_of_the_dataset_they_run_on():
    root = Path(__file__).resolve().parents[1]
    src = (root / "backend/api/routes/indexing.py").read_text(encoding="utf-8")
    # read once, from the registry, for the dataset the run (or the batch item) is on
    assert "_run_variant = _variant_registry.variant_for(_run_dataset)" in src
    assert "_item_variant = _variant_registry.variant_for(_ev_item._active_dataset)" in src
    n_item = 0
    for name in ("spherical_gpu_index_patterns", "spherical_index_patterns", "run_single_phase_method",
                 "run_per_phase_indexing"):
        calls = [c for c in _calls(src, name) if "h5_path" in c or name == "run_per_phase_indexing"]
        assert calls, name
        for c in calls:
            assert "variant=_run_variant" in c or "variant=_item_variant" in c, \
                f"{name} is called without the dataset's variant:\n{c}"
            n_item += "variant=_item_variant" in c
    assert n_item == 2                                  # the legacy batch's GPU and EMSphInx calls


def test_run_single_phase_method_and_run_per_phase_hand_the_variant_to_the_indexers():
    import inspect
    for fn in (ic.run_single_phase_method, ic.run_per_phase_indexing):
        assert "variant" in inspect.signature(fn).parameters, fn.__name__
    src = inspect.getsource(ic.run_single_phase_method)
    assert all("variant=variant" in c for c in _calls(src, "spherical_gpu_index_patterns")
               + _calls(src, "spherical_index_patterns"))
    per_phase = inspect.getsource(ic.run_per_phase_indexing)
    assert all("variant=variant" in c for c in _calls(per_phase, "run_single_phase_method"))


def test_a_batch_that_loaded_its_signal_stays_on_the_default():
    """Batch v2 loads each file as processed, so its calls carry no variant (None is the default)."""
    root = Path(__file__).resolve().parents[1]
    src = (root / "backend/api/services/batch_manager.py").read_text(encoding="utf-8")
    assert "variant=" not in "".join(_calls(src, "spherical_gpu_index_patterns")
                                     + _calls(src, "spherical_index_patterns"))


def test_load_codes_know_variant_not_supported():
    assert "variantNotSupported" in lc.LOAD_CODES


# ====================================================================== the start of a run

class _Started(Exception):
    """Raised by a stub placed right after the guards: the request got past them."""


@pytest.fixture
def start_run(monkeypatch):
    """POST /api/indexing/start for the active dataset 'ds', a viewer that holds RAW of a file."""
    import kikuchipy as kp
    from fastapi.testclient import TestClient
    from backend.api.main import app
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.routes import indexing as I
    from backend.api.services import variant_registry
    from tests.test_indexing_refuses_missing_pc import _store

    nickel = kp.data.nickel_ebsd_small()

    def run(path, method):
        monkeypatch.setattr(I, "calibration_store", _store(nickel, "ds", "osc"))
        monkeypatch.setattr(ev, "_active_dataset", "ds")
        monkeypatch.setattr(ev, "_get_active_signal", lambda *a, **k: nickel)
        monkeypatch.setattr(ev, "_ebsd_file_path", path)

        def started(*a, **k):
            raise _Started()

        monkeypatch.setattr(I, "_track_task", started)         # the next line after the guards
        variant_registry.register("ds", RAW)
        try:
            return TestClient(app, raise_server_exceptions=True).post("/api/indexing/start", json={"method": method})
        finally:
            variant_registry.drop("ds")
    return run


def test_a_spherical_run_on_a_layout_that_cannot_name_the_variant_is_refused_with_the_code(
        start_run, unlisted_layout):
    resp = start_run(unlisted_layout, "spherical")
    assert resp.status_code == 400
    assert resp.headers["x-orienta-code"] == "variantNotSupported"
    # the prose is for people: it says "this step", it does not name a function of the program
    assert "This step cannot read the unprocessed patterns" in resp.json()["detail"]
    assert "start_indexing" not in resp.json()["detail"]


def test_a_spherical_run_on_a_file_that_names_the_variant_starts(start_run, two_sources):
    with pytest.raises(_Started):
        start_run(two_sources, "spherical")


@pytest.mark.parametrize("method", ["hough", "dictionary"])
def test_only_the_spherical_run_needs_the_file_to_name_the_variant(start_run, unlisted_layout, method):
    """Hough and dictionary index the signal in memory: no re-read of the file, nothing to refuse."""
    with pytest.raises(_Started):
        start_run(unlisted_layout, method)


# ============================================================ the dtype the exports write

def test_fit_to_dtype_keeps_what_fits():
    a = unprocessed(1)
    assert lc.fit_to_dtype(a, np.int16) is a
    out = lc.fit_to_dtype(a.astype(np.float32), np.int16)          # whole numbers inside the range
    assert out.dtype == np.int16 and np.array_equal(out, a)
    out = lc.fit_to_dtype(np.array([0, 255, 17], np.int16), np.uint8)
    assert out.dtype == np.uint8 and out.tolist() == [0, 255, 17]


def test_fit_to_dtype_rescales_what_does_not_fit():
    edited = np.array([[-40.0, 0.0, 12.5, 100.0]])                 # a background-corrected float stack
    out = lc.fit_to_dtype(edited, np.uint8)
    # by hand: (x + 40) / 140 * 255 = 0, 72.86, 95.63, 255
    assert out.dtype == np.uint8 and out.tolist() == [[0, 73, 96, 255]]
    out = lc.fit_to_dtype(edited, np.int16)
    assert out.dtype == np.int16 and out.min() == 0 and out.max() == 32767
    assert np.array_equal(np.argsort(out.ravel()), np.argsort(edited.ravel()))
    assert lc.fit_to_dtype(np.array([-5, 300], np.int16), np.uint8).tolist() == [0, 255]   # out of range
    assert lc.fit_to_dtype(np.full((2, 2), 4.0), np.int16).tolist() == [[4, 4], [4, 4]]    # whole, fits
    with np.errstate(all="raise"):          # a constant stack has no scale: no 0/0 on the way to zeros
        assert lc.fit_to_dtype(np.full((2, 2), 4.5), np.int16).tolist() == [[0, 0], [0, 0]]


def test_fit_to_dtype_to_float_is_a_plain_cast():
    out = lc.fit_to_dtype(np.array([-1, 2], np.int16), np.float32)
    assert out.dtype == np.float32 and out.tolist() == [-1.0, 2.0]


# ============================================================================ the rich export

def _kp(arr4):
    import kikuchipy as kp
    return kp.signals.EBSD(np.asarray(arr4))


def _edited_unprocessed(g: int) -> np.ndarray:
    """The viewer's int16 patterns of measurement g: not the file's, not constant, above 255."""
    return (unprocessed(g) + 7).reshape(NY, NX, SY, SX)


def _datasets(path):
    with h5py.File(path, "r") as f:
        return {g: {s: f[f"{g}/EBSD/Data/{s} Patterns"][...] for s in ("Processed", "Unprocessed")}
                for g in ("1", "2")}


@pytest.mark.parametrize("variant,group", [(RAW, "1"), (RAW2, "2")], ids=["measurement-1", "measurement-2"])
def test_the_rich_export_writes_the_variants_dataset(two_sources, tmp_path, variant, group):
    from backend.api.routes.indexing import _overwrite_patterns_with_processed
    out = tmp_path / "out.h5"
    import shutil
    shutil.copy2(two_sources, out)
    assert _overwrite_patterns_with_processed(str(out), _kp(_edited_unprocessed(int(group))), variant) is True
    got = _datasets(out)
    target = got[group]["Unprocessed"]
    assert target.dtype == np.int16                                    # int16 counts stay int16
    assert np.array_equal(target, _edited_unprocessed(int(group)).reshape(N, SY, SX))
    with h5py.File(out, "r") as f:
        assert bool(f[f"{group}/EBSD/Data/Unprocessed Patterns"].attrs["patterns_processed"])
    for g in ("1", "2"):                                               # nothing else of the copy moved
        assert np.array_equal(got[g]["Processed"], processed(int(g)))
        if g != group:
            assert np.array_equal(got[g]["Unprocessed"], unprocessed(int(g)))


def test_the_rich_export_without_a_variant_writes_the_processed_dataset_as_before(two_sources, tmp_path):
    from backend.api.routes.indexing import _overwrite_patterns_with_processed
    import shutil
    out = tmp_path / "out.h5"
    shutil.copy2(two_sources, out)
    edited = (processed(1) + 1).reshape(NY, NX, SY, SX)
    assert _overwrite_patterns_with_processed(str(out), _kp(edited)) is True
    got = _datasets(out)
    assert np.array_equal(got["1"]["Processed"], processed(1) + 1)
    assert np.array_equal(got["1"]["Unprocessed"], unprocessed(1))


def test_the_rich_export_fits_a_float_edit_into_the_variants_dtype(two_sources, tmp_path):
    from backend.api.routes.indexing import _overwrite_patterns_with_processed
    import shutil
    out = tmp_path / "out.h5"
    shutil.copy2(two_sources, out)
    edited = (np.arange(N * SY * SX, dtype=np.float64).reshape(NY, NX, SY, SX) - 40.0) * 2.5
    assert _overwrite_patterns_with_processed(str(out), _kp(edited), RAW) is True
    got = _datasets(out)["1"]["Unprocessed"]
    assert got.dtype == np.int16 and got.min() == 0 and got.max() == np.iinfo(np.int16).max
    assert np.array_equal(np.argsort(got.ravel(), kind="stable"), np.argsort(edited.ravel(), kind="stable"))


def test_the_rich_export_refuses_a_variant_the_layout_cannot_honour(unlisted_layout):
    from backend.api.routes.indexing import _overwrite_patterns_with_processed
    with pytest.raises(LoadChoiceError) as exc:
        _overwrite_patterns_with_processed(unlisted_layout, _kp(np.zeros((NY, NX, SY, SX), np.uint8)), RAW)
    assert exc.value.code == "variantNotSupported"
    assert exc.value.params == {"where": "_overwrite_patterns_with_processed"}


@pytest.mark.parametrize("variant,processed_flag,group", [
    (RAW, False, "1"), (RAW2, True, "2"), (PROC2, False, "2")],
    ids=["unprocessed-source-copy", "measurement-2-edited", "measurement-2-processed"])
def test_the_rich_export_marks_the_source(two_sources, tmp_path, variant, processed_flag, group):
    from backend.api.routes.indexing import _mark_pattern_source
    import shutil
    out = tmp_path / "out.h5"
    shutil.copy2(two_sources, out)
    _mark_pattern_source(str(out), variant, processed_flag)
    name = "Unprocessed" if variant.pattern_source == "unprocessed" else "Processed"
    with h5py.File(out, "r") as f:
        attrs = f[f"{group}/EBSD/Data/{name} Patterns"].attrs
        assert bool(attrs["patterns_processed"]) is processed_flag
        assert attrs["pattern_source"] == variant.pattern_source
        if group == "1":
            assert "measurement" not in attrs                  # the file's default measurement
        else:
            assert attrs["measurement"] == "2"


@pytest.mark.parametrize("variant", [RAW, RAW2, PROC2], ids=["unprocessed", "measurement-2-unprocessed", "measurement-2"])
def test_a_rich_export_loads_as_the_variant_it_marks(two_sources, tmp_path, variant):
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.routes.indexing import _mark_pattern_source
    import shutil
    out = tmp_path / "out.h5"
    shutil.copy2(two_sources, out)
    assert lc.inspect_file(str(out)).suggested is None                 # a copy that says nothing
    _mark_pattern_source(str(out), variant, False)
    choices = lc.inspect_file(str(out))
    assert choices.suggested == variant
    assert ev._requested_variant(None, choices) == variant             # a load without a choice


def test_exporting_an_export_moves_the_mark(two_sources, tmp_path):
    """A rich export is a copy of its source, and that source can be a rich export of another variant.
    The file then says the new export's variant, also when that is the default one (no variant)."""
    from backend.api.routes.indexing import _mark_pattern_source
    import shutil
    out = tmp_path / "out.h5"
    shutil.copy2(two_sources, out)
    _mark_pattern_source(str(out), RAW, True)
    _mark_pattern_source(str(out), PROC2, False)
    assert lc.inspect_file(str(out)).suggested == PROC2
    _mark_pattern_source(str(out), None, False)
    assert lc.inspect_file(str(out)).suggested is None
    with h5py.File(out, "r") as f:
        # the dataset still holds the viewer's edits: that is a fact about its values, not a choice
        assert bool(f["1/EBSD/Data/Unprocessed Patterns"].attrs["patterns_processed"])


def test_any_pattern_dataset_counts_unprocessed_patterns(tmp_path):
    from backend.api.routes.indexing import _h5_any_pattern_dataset
    path = tmp_path / "only_raw.h5oina"
    with h5py.File(path, "w") as f:
        f.create_dataset("1/EBSD/Data/Unprocessed Patterns", data=unprocessed(1))
    with h5py.File(path, "r") as f:
        assert _h5_any_pattern_dataset(f) is True
    empty = tmp_path / "none.h5oina"
    with h5py.File(empty, "w") as f:
        f.create_dataset("1/EBSD/Data/Band Contrast", data=np.zeros(N, np.uint8))
    with h5py.File(empty, "r") as f:
        assert _h5_any_pattern_dataset(f) is False


# --------------------------------------------------------------- the rich export, through the route

@pytest.fixture
def viewer_app(monkeypatch):
    from fastapi.testclient import TestClient
    from backend.api.main import app
    from backend.api.routes import indexing as I
    from tests.test_load_variant_routes import _reset
    _reset()
    saved, saved_active = dict(I._result_registry), I._active_result_id
    yield TestClient(app)
    _reset()
    I._result_registry.clear()
    I._result_registry.update(saved)
    I._active_result_id = saved_active


def _store_a_result():
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation
    from backend.api.routes import indexing as I
    from indexing_controller import IndexingMethod, IndexingResult
    xmap = CrystalMap(
        rotations=Rotation.from_euler(np.deg2rad(np.array([[i * 9.0, 20.0, 30.0] for i in range(N)]))),
        phase_id=np.zeros(N, dtype=np.int32),
        x=np.tile(np.arange(NX, dtype=np.float32), NY),
        y=np.repeat(np.arange(NY, dtype=np.float32), NX),
        phase_list=PhaseList(phases=[Phase(name="Ni", space_group=225, point_group="m-3m")]))
    result = IndexingResult(
        xmap=xmap, selection_mask=np.ones(N, dtype=bool), original_shape=(NY, NX),
        method=IndexingMethod.HOUGH, confidence_scores=np.full(N, 0.8, dtype=np.float32),
        metadata={"step_size_um": 0.5, "recorded_lattices": {0: [3.52] * 3 + [90, 90, 90]}})
    return I._store_result(result, "hough")


def _export(client, out):
    return client.post("/api/indexing/export", json={"format": "h5", "filename": str(out)})


def _load_variant(client, path, measurement=None, source=None):
    body = {"path": path}
    if source is not None:
        body["variant"] = {"measurement": measurement, "pattern_source": source}
    r = client.post("/api/ebsd/load", json=body)
    assert r.status_code == 200, r.text


def test_export_marks_the_source(viewer_app, two_sources, tmp_path, monkeypatch):
    """The viewer holds the UNPROCESSED patterns, edited: the export writes the edits into that
    dataset (int16, exact), leaves the processed ones, and says which source the file holds."""
    from backend.api.routes import ebsd_viewer as ev
    _load_variant(viewer_app, two_sources, "1", "unprocessed")
    _store_a_result()
    edited = _edited_unprocessed(1)
    monkeypatch.setattr(ev, "is_active_signal_dirty", lambda: True)
    monkeypatch.setattr(ev, "_get_active_signal", lambda: _kp(edited))
    out = tmp_path / "rich.h5"
    resp = _export(viewer_app, out)
    assert resp.status_code == 200, resp.text
    with h5py.File(out, "r") as f:
        raw = f["1/EBSD/Data/Unprocessed Patterns"]
        assert raw.dtype == np.int16
        assert np.array_equal(raw[...], edited.reshape(N, SY, SX))
        assert bool(raw.attrs["patterns_processed"]) is True
        assert raw.attrs["pattern_source"] == "unprocessed"
        assert np.array_equal(f["1/EBSD/Data/Processed Patterns"][...], processed(1))
        assert "pattern_source" not in f["1/EBSD/Data/Processed Patterns"].attrs


def test_the_export_of_an_unedited_unprocessed_variant_says_it_is_not_processed(viewer_app, two_sources,
                                                                              tmp_path):
    _load_variant(viewer_app, two_sources, "1", "unprocessed")
    _store_a_result()
    out = tmp_path / "rich.h5"
    assert _export(viewer_app, out).status_code == 200
    with h5py.File(out, "r") as f:
        raw = f["1/EBSD/Data/Unprocessed Patterns"]
        assert bool(raw.attrs["patterns_processed"]) is False         # the file's own counts
        assert raw.attrs["pattern_source"] == "unprocessed"
        assert np.array_equal(raw[...], unprocessed(1))


def test_the_export_of_the_default_variant_is_what_it_was_before(viewer_app, two_sources, tmp_path, monkeypatch):
    from backend.api.routes import ebsd_viewer as ev
    _load_variant(viewer_app, two_sources)
    _store_a_result()
    edited = (processed(1) + 1).reshape(NY, NX, SY, SX)
    monkeypatch.setattr(ev, "is_active_signal_dirty", lambda: True)
    monkeypatch.setattr(ev, "_get_active_signal", lambda: _kp(edited))
    out = tmp_path / "rich.h5"
    assert _export(viewer_app, out).status_code == 200
    with h5py.File(out, "r") as f:
        proc = f["1/EBSD/Data/Processed Patterns"]
        assert np.array_equal(proc[...], processed(1) + 1)
        assert "pattern_source" not in proc.attrs and "measurement" not in proc.attrs
        assert np.array_equal(f["1/EBSD/Data/Unprocessed Patterns"][...], unprocessed(1))


def test_the_rich_export_on_a_layout_that_cannot_name_the_variant_is_a_coded_400(viewer_app, two_sources,
                                                                                 unlisted_layout, tmp_path,
                                                                                 monkeypatch):
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.services import variant_registry
    _load_variant(viewer_app, two_sources, "1", "unprocessed")
    _store_a_result()
    monkeypatch.setattr(ev, "_ebsd_file_path", unlisted_layout)       # the source the copy is made of
    assert variant_registry.variant_for(ev._active_dataset) == RAW
    out = tmp_path / "rich.h5"
    resp = _export(viewer_app, out)
    assert resp.status_code == 400, resp.text
    assert resp.headers["x-orienta-code"] == "variantNotSupported"
    assert not out.exists()                                           # refused before anything was copied


# ============================================================================== the crop export

def _window():
    from backend.api.services.crop_window import CropWindow
    return CropWindow("multi.h5oina", row0=0, col0=1, rows=2, cols=2, original_shape=(NY, NX))


CROP_IDX = [r * NX + c for r in range(2) for c in range(1, 3)]


def test_the_crop_writes_the_supplied_patterns_into_the_variants_dataset(two_sources, tmp_path):
    from backend.api.services.crop_export import write_cropped_h5oina
    supplied = (np.arange(4 * SY * SX) * 5 + 700).astype(np.int16).reshape(4, SY, SX)    # not the file's
    out = tmp_path / "crop.h5oina"
    write_cropped_h5oina(two_sources, str(out), _window(), patterns=supplied, variant=RAW)
    with h5py.File(out, "r") as f:
        raw = f["1/EBSD/Data/Unprocessed Patterns"]
        assert raw.dtype == np.int16 and np.array_equal(raw[...], supplied)
        # the other source of the measurement is cut from the file, not replaced
        assert np.array_equal(f["1/EBSD/Data/Processed Patterns"][...], processed(1)[CROP_IDX])
        assert np.array_equal(f["2/EBSD/Data/Unprocessed Patterns"][...], unprocessed(2)[CROP_IDX])
        prov = f["CropProvenance"].attrs
        assert prov["pattern_source"] == "unprocessed" and prov["measurement"] == "1"
        assert prov["patterns"] == "processed"


def test_the_crop_without_a_variant_writes_the_processed_dataset_as_before(two_sources, tmp_path):
    from backend.api.services.crop_export import write_cropped_h5oina
    supplied = np.full((4, SY, SX), 77, np.uint8)
    out = tmp_path / "crop.h5oina"
    write_cropped_h5oina(two_sources, str(out), _window(), patterns=supplied)
    with h5py.File(out, "r") as f:
        assert np.array_equal(f["1/EBSD/Data/Processed Patterns"][...], supplied)
        assert np.array_equal(f["1/EBSD/Data/Unprocessed Patterns"][...], unprocessed(1)[CROP_IDX])
        assert "pattern_source" not in f["CropProvenance"].attrs


def test_a_crop_of_the_unprocessed_variant_reloads_as_unprocessed(two_sources, tmp_path):
    """Round trip: write the crop, ask what it holds, load that variant."""
    import safe_loader
    from backend.api.services.crop_export import write_cropped_h5oina
    supplied = (np.arange(4 * SY * SX) * 5 + 700).astype(np.int16).reshape(4, SY, SX)
    out = tmp_path / "crop.h5oina"
    write_cropped_h5oina(two_sources, str(out), _window(), patterns=supplied, variant=RAW)
    choices = lc.inspect_file(str(out))
    assert choices is not None
    assert [s.key for s in choices.measurements[0].sources] == ["processed", "unprocessed"]
    assert lc.pattern_dataset_path(str(out), RAW) == "1/EBSD/Data/Unprocessed Patterns"
    back = safe_loader.load_ebsd_safe(str(out), verbose=False, variant=RAW)
    assert np.asarray(back.data).dtype == np.int16
    assert np.array_equal(np.asarray(back.data), supplied.reshape(2, 2, SY, SX))


def test_the_crop_refuses_a_variant_the_layout_cannot_honour(unlisted_layout, tmp_path):
    from backend.api.services.crop_export import write_cropped_h5oina
    out = tmp_path / "crop.h5oina"
    with pytest.raises(LoadChoiceError) as exc:
        write_cropped_h5oina(unlisted_layout, str(out), _window(),
                             patterns=np.zeros((4, SY, SX), np.uint8), variant=RAW)
    assert exc.value.code == "variantNotSupported"
    assert exc.value.params == {"where": "write_cropped_h5oina"}
    assert not out.exists()


def test_the_crop_route_writes_into_the_loaded_variant(viewer_app, two_sources, tmp_path):
    """The crop holds the variant of its parent: the route hands it to the writer."""
    _load_variant(viewer_app, two_sources, "1", "unprocessed")
    r = viewer_app.post("/api/ebsd/crop", json={"shape": "rect", "row0": 0, "col0": 1, "rows": 2, "cols": 2})
    assert r.status_code == 200, r.text
    out = tmp_path / "crop.h5oina"
    r = viewer_app.post("/api/ebsd/crop/export", json={"path": str(out)})
    assert r.status_code == 200, r.text
    with h5py.File(out, "r") as f:
        assert f["1/EBSD/Data/Unprocessed Patterns"].dtype == np.int16
        assert np.array_equal(f["1/EBSD/Data/Unprocessed Patterns"][...], unprocessed(1)[CROP_IDX])
        assert np.array_equal(f["1/EBSD/Data/Processed Patterns"][...], processed(1)[CROP_IDX])
        assert f["CropProvenance"].attrs["pattern_source"] == "unprocessed"


def test_the_crop_route_on_a_layout_that_cannot_name_the_variant_is_a_coded_400(viewer_app, two_sources,
                                                                                unlisted_layout, tmp_path,
                                                                                monkeypatch):
    import dataclasses
    from backend.api.routes import ebsd_viewer as ev
    _load_variant(viewer_app, two_sources, "1", "unprocessed")
    r = viewer_app.post("/api/ebsd/crop", json={"shape": "rect", "row0": 0, "col0": 1, "rows": 2, "cols": 2})
    assert r.status_code == 200, r.text
    elsewhere = dataclasses.replace(ev.get_active_crop_window(), source_file=unlisted_layout)
    monkeypatch.setattr(ev, "get_active_crop_window", lambda: elsewhere)    # the file the crop is made of
    out = tmp_path / "crop.h5oina"
    r = viewer_app.post("/api/ebsd/crop/export", json={"path": str(out)})
    assert r.status_code == 400, r.text
    assert r.headers["x-orienta-code"] == "variantNotSupported"
    assert not out.exists()


# ======================================================================= the image quality

def test_the_image_quality_is_computed_on_the_variants_patterns(two_sources, monkeypatch):
    from backend.api.services import pattern_quality as pq
    seen = []

    def fake_quality(signal):
        seen.append(np.asarray(signal.data).copy())
        return np.zeros(N)

    monkeypatch.setattr(pq, "compute_image_quality", fake_quality)
    assert pq.image_quality_from_file(two_sources, NY, NX, variant=RAW) is not None
    assert pq.image_quality_from_file(two_sources, NY, NX, variant=RAW2) is not None
    assert pq.image_quality_from_file(two_sources, NY, NX) is not None
    assert [a.dtype for a in seen] == [np.int16, np.int16, np.uint8]
    assert np.array_equal(seen[0], unprocessed(1).reshape(NY, NX, SY, SX))
    assert np.array_equal(seen[1], unprocessed(2).reshape(NY, NX, SY, SX))
    assert np.array_equal(seen[2], processed(1).reshape(NY, NX, SY, SX))     # no variant: today's


def test_the_image_quality_of_a_variant_the_file_lacks_is_not_available(two_sources):
    from backend.api.services import pattern_quality as pq
    assert pq.image_quality_from_file(two_sources, NY, NX, variant=LoadVariant("9", "unprocessed")) is None
