"""Which measurements and pattern sources an EBSD file offers.

A *variant* is a pair (measurement, pattern source): the HDF5 scan group to load and, where the
format keeps two copies, whether the processed or the unprocessed patterns. Oxford files
(``<group>/EBSD/Data/Processed Patterns`` and, optionally, ``Unprocessed Patterns``) are the only
ones with two sources; Bruker, EDAX and kikuchipy h5ebsd keep one pattern dataset per group.

Everything here reads group and dataset metadata (names, ``shape``, ``dtype`` and the few header
scalars that give the grid), never a pattern array, so it is as fast on a 27 GB file as on a small one.
"""
from __future__ import annotations

import os
import re
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROCESSED = "processed"
UNPROCESSED = "unprocessed"
SINGLE = "single"
PATTERN_SOURCES = (PROCESSED, UNPROCESSED, SINGLE)

_H5_FORMATS = ("oxford", "bruker", "edax_h5ebsd", "kikuchipy_h5ebsd")

#: format -> [(source key, pattern dataset in ``EBSD/Data``, static background in ``EBSD/Header`` or None)]
_SOURCES: Dict[str, List[Tuple[str, str, Optional[str]]]] = {
    "oxford": [
        (PROCESSED, "Processed Patterns", "Processed Static Background"),
        (UNPROCESSED, "Unprocessed Patterns", "Unprocessed Static Background"),
    ],
    "bruker": [(SINGLE, "RawPatterns", "StaticBackground")],
    "edax_h5ebsd": [(SINGLE, "Pattern", None)],
    "kikuchipy_h5ebsd": [(SINGLE, "patterns", "static_background")],
}

#: format -> (rows, columns) header datasets in ``EBSD/Header``, as kikuchipy's readers name them
_GRID: Dict[str, Tuple[str, str]] = {
    "oxford": ("Y Cells", "X Cells"),
    "bruker": ("NROWS", "NCOLS"),
    "edax_h5ebsd": ("nRows", "nColumns"),
    "kikuchipy_h5ebsd": ("n_rows", "n_columns"),
}

_NO_PATTERNS = "loadNoPatterns"
_UNKNOWN_VARIANT = "loadVariantUnknown"
_NOT_SUPPORTED = "variantNotSupported"
#: A group of the file could not be read while listing it (or while reading a preview pattern).
INSPECT_FAILED = "loadInspectFailed"

#: Every code a load refusal can carry; the interface words each under ``errors.*``
#: (``tests/test_load_variant_routes.py`` holds this list against the source and the locale fixture).
LOAD_CODES = ("loadCameUpAsOther", "loadInspectFailed", "loadNoPatterns", "loadVariantNeedsNativeReader",
              "loadVariantUnknown", "variantNotSupported")


class LoadChoiceError(ValueError):
    """A refusal the interface can word. ``code`` is what it translates on; ``params`` are the
    values its text may name (the code travels in a header, the prose keeps them in English)."""

    def __init__(self, code: str, message: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.params = dict(params or {})


@dataclass(frozen=True)
class LoadVariant:
    measurement: Optional[str]
    pattern_source: str

    def to_dict(self) -> Dict[str, Any]:
        return {"measurement": self.measurement, "pattern_source": self.pattern_source}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LoadVariant":
        source = d.get("pattern_source", PROCESSED)
        if source not in PATTERN_SOURCES:
            raise LoadChoiceError(_UNKNOWN_VARIANT, f"unknown pattern source {source!r}")
        measurement = d.get("measurement")
        return cls(None if measurement is None else str(measurement), source)

    def is_default(self, choices: "FileChoices") -> bool:
        """True when loading this variant is what loading the file without a choice does."""
        try:
            default = choices.default_variant()
        except ValueError:
            return False
        return (self.pattern_source == default.pattern_source
                and self.measurement in (None, default.measurement))


@dataclass
class PatternSource:
    key: str
    dataset: str
    n_patterns: int
    pattern_shape: Tuple[int, int]
    dtype: str
    has_background: bool
    loadable: bool
    problem: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key, "dataset": self.dataset, "n_patterns": self.n_patterns,
            "pattern_shape": list(self.pattern_shape), "dtype": self.dtype,
            "has_background": self.has_background, "loadable": self.loadable, "problem": self.problem,
        }


@dataclass
class Measurement:
    name: str
    label: str
    n_rows: int
    n_cols: int
    sources: List[PatternSource]

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "label": self.label, "n_rows": self.n_rows, "n_cols": self.n_cols,
                "sources": [s.to_dict() for s in self.sources]}


@dataclass
class FileChoices:
    path: str
    format: str
    measurements: List[Measurement]
    #: The variant a file says it holds, when it offers it: the one a crop was cut from
    #: (``CropProvenance``) or the one a rich export marked on its pattern dataset. A load without a
    #: choice takes this one: the crop of the unprocessed patterns of measurement 2 is a file about
    #: measurement 2, not about the first one that happens to be there.
    suggested: Optional[LoadVariant] = None

    def n_variants(self) -> int:
        """Variants that can be loaded; a measurement without patterns is listed but not counted."""
        return sum(1 for m in self.measurements for s in m.sources if s.loadable)

    def default_variant(self) -> LoadVariant:
        for m in self.measurements:
            for s in m.sources:
                if s.loadable:
                    return LoadVariant(m.name, s.key)
        raise LoadChoiceError(_NO_PATTERNS, f"{Path(self.path).name} holds no measurement with patterns")

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready. Carries no path: the caller that answers a question about a file knows which
        file it was (the dataset paths inside it are HDF5 paths, not folders of this machine)."""
        d: Dict[str, Any] = {"format": self.format, "n_variants": self.n_variants(),
                             "measurements": [m.to_dict() for m in self.measurements]}
        try:
            d["default"] = self.default_variant().to_dict()
        except ValueError:
            d["default"] = None
        if self.suggested is not None:
            d["suggested"] = self.suggested.to_dict()
        return d


def to_uint8(patterns: np.ndarray) -> np.ndarray:
    """uint8 patterns unchanged; any other dtype rescaled by its global min/max to 0..255.

    A plain ``astype(np.uint8)`` wraps int16 values modulo 256 and turns a pattern into noise."""
    a = np.asarray(patterns)
    if a.dtype == np.uint8:
        return a
    if a.size == 0:
        return a.astype(np.uint8)
    floating = a.dtype.kind == "f"
    lo = float(np.nanmin(a) if floating else a.min())
    hi = float(np.nanmax(a) if floating else a.max())
    if not hi > lo:
        return np.zeros(a.shape, dtype=np.uint8)
    out = a.astype(np.float32)
    out -= lo
    out *= 255.0 / (hi - lo)
    np.rint(out, out=out)
    if floating:
        np.nan_to_num(out, copy=False)
    return out.astype(np.uint8)


def fit_to_dtype(patterns: np.ndarray, dtype) -> np.ndarray:
    """``patterns`` as ``dtype`` without wrapping or clipping.

    Unchanged values when they already fit (the same dtype, integers inside the range, floats that
    are whole numbers inside the range). Otherwise, for an integer target, one global linear
    rescale of min..max onto 0..max of the dtype (a background-corrected float stack has negative
    and fractional values; ``astype`` would wrap or truncate them). A float target is a plain cast."""
    a = np.asarray(patterns)
    dtype = np.dtype(dtype)
    if a.dtype == dtype:
        return a
    if dtype.kind not in "iu" or a.size == 0:
        return a.astype(dtype)
    info = np.iinfo(dtype)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    whole = a.dtype.kind in "iu" or bool(np.all(np.isfinite(a)) and np.array_equal(a, np.rint(a)))
    if whole and lo >= info.min and hi <= info.max:
        return a.astype(dtype)
    if not hi > lo:
        return np.zeros(a.shape, dtype=dtype)
    scaled = (a.astype(np.float64) - lo) / (hi - lo) * float(info.max)
    return np.rint(np.nan_to_num(scaled)).astype(dtype)


def to_uint8_chunked(patterns: np.ndarray, chunk: int = 256,
                     value_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """``to_uint8`` by blocks of ``chunk`` patterns: the same values (the global min/max of the whole
    stack), without the float32 copy of the whole stack that ``to_uint8`` makes.

    ``value_range`` is the (min, max) to rescale by when ``patterns`` are a selection out of a larger
    scan: the range of the scan (``scan_value_range``), not of the selection."""
    a = np.asarray(patterns)
    if a.dtype == np.uint8:
        return a
    if a.size == 0 or a.ndim == 0:
        return to_uint8(a)
    floating = a.dtype.kind == "f"
    if value_range is not None:
        lo, hi = float(value_range[0]), float(value_range[1])
    else:
        lo = float(np.nanmin(a) if floating else a.min())
        hi = float(np.nanmax(a) if floating else a.max())
    out = np.zeros(a.shape, dtype=np.uint8)
    if not hi > lo:
        return out
    scale = 255.0 / (hi - lo)
    step = max(1, int(chunk))
    for start in range(0, a.shape[0], step):
        block = a[start:start + step].astype(np.float32)
        block -= lo
        block *= scale
        np.rint(block, out=block)
        if floating:
            np.nan_to_num(block, copy=False)
        # a value outside the handed range (the stack was edited after the range pass) clips like the
        # click path does (``for_signal_indexer``); astype alone would wrap it into noise
        np.clip(block, 0, 255, out=block)
        out[start:start + step] = block.astype(np.uint8)
    return out


def for_pyebsdindex(patterns: np.ndarray, value_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """What PyEBSDIndex gets to see. It does read int16, but its band detection then gives other
    bands than for the 8 bit rescale of the same patterns (the Oxford reference scan unprocessed, pyebsdindex 0.3.9.1:
    2 of 9 patterns differ, by 0.65 and 36 degrees). Signed integers are therefore rescaled to 8 bit
    (``to_uint8_chunked``), so that Hough gives the orientations of the 8 bit rescale; uint8, uint16
    and floats pass through as the very same object.

    ``value_range`` is the range of the whole scan when ``patterns`` are a selection out of it (see
    ``to_uint8_chunked``)."""
    dtype = getattr(patterns, "dtype", None)
    if dtype is not None and np.dtype(dtype).kind == "i":
        return to_uint8_chunked(patterns, value_range=value_range)
    return patterns


#: "the range of this source has not been asked for yet" (``None`` already means "not a signed-integer source")
RANGE_UNKNOWN = object()

#: bytes of one block read by the range pass (a row of a 4-D signal can hold hundreds of patterns)
_RANGE_BLOCK_BYTES = 64 * 1024 * 1024


def signed_pattern_range(source, chunk: int = 256, *, check=None,
                         progress=None) -> Optional[Tuple[float, float]]:
    """Global (min, max) of a signed-integer pattern source (an array or an h5py dataset, read by
    blocks along its first axis), ``None`` for every other dtype. The indexers ask once per run, so
    that every block of an int16 scan is rescaled by the range of the whole scan.

    It reads the whole source, which takes minutes on a 27 GB scan: ``check`` (a zero-argument
    callable that raises when the user cancelled) is called before every block, and
    ``progress(done, total)`` after every block (first-axis items)."""
    if np.dtype(source.dtype).kind != "i" or int(source.shape[0]) == 0:
        return None
    n = int(source.shape[0])
    item_bytes = max(1, int(np.prod(source.shape[1:], dtype=np.int64)) * np.dtype(source.dtype).itemsize)
    step = max(1, min(int(chunk), _RANGE_BLOCK_BYTES // item_bytes))
    lo, hi = np.inf, -np.inf
    for start in range(0, n, step):
        if check is not None:
            check()
        block = np.asarray(source[start:start + step])
        lo, hi = min(lo, float(block.min())), max(hi, float(block.max()))
        if progress is not None:
            progress(min(start + step, n), n)
    return lo, hi


def _dataset_key(dset) -> Optional[tuple]:
    """What a pass over an HDF5 dataset depends on: the file (path, size, modification time), the
    dataset, its shape and dtype. ``None`` when the file cannot be stat-ed (then nothing is cached)."""
    try:
        path = os.path.abspath(dset.file.filename)
        st = os.stat(path)
        return (path, dset.name, st.st_size, st.st_mtime_ns, tuple(dset.shape), str(dset.dtype))
    except Exception:  # noqa: BLE001 - a dataset we cannot key is simply not cached
        return None


_DATASET_RANGES: Dict[tuple, Tuple[float, float]] = {}
_DATASET_RANGES_MAX = 8
#: Two runs on different files can look up, evict and store at the same moment. The lock covers
#: those accesses only, never the pass over the file (minutes on a 27 GB scan).
_DATASET_RANGES_LOCK = threading.Lock()


def dataset_pattern_range(dset, *, check=None, progress=None) -> Optional[Tuple[float, float]]:
    """``signed_pattern_range`` of an h5py dataset, remembered per (file, size, mtime, dataset): a
    second run on the same file does not read a 27 GB scan again, and a file that changed on disk
    is read again. Streamed runs have no signal in memory, so the file is the only stable key."""
    if np.dtype(dset.dtype).kind != "i":
        return None
    key = _dataset_key(dset)
    if key is not None:
        with _DATASET_RANGES_LOCK:
            if key in _DATASET_RANGES:
                return _DATASET_RANGES[key]
    rng = signed_pattern_range(dset, check=check, progress=progress)
    if key is not None and rng is not None:
        with _DATASET_RANGES_LOCK:
            while len(_DATASET_RANGES) >= _DATASET_RANGES_MAX:
                _DATASET_RANGES.pop(next(iter(_DATASET_RANGES)))
            _DATASET_RANGES[key] = rng
    return rng


_SCAN_RANGE_ATTR = "_orienta_scan_range"
_FILE_SOURCE_ATTR = "_orienta_file_source"


def note_file_source(signal, path: str, variant: Optional["LoadVariant"]) -> None:
    """Say which dataset of which file a lazy signed-integer signal reads, as the loader hands it over.

    The lazy array of a file written one chunk per pattern costs one task per pattern: 970 patterns
    per second against 34000 from the file, measured on 485415 patterns (500 s against 14 s, which
    took the first click on such a scan past the client's 5 minutes). So ``scan_value_range`` asks the
    file as long as the signal holds this array and nothing edited it (``forget_scan_range``)."""
    data = getattr(signal, "data", None)
    if not hasattr(data, "dask") or np.dtype(data.dtype).kind != "i":
        return
    try:
        # without a variant the reader takes its own first group, which need not be the file's
        if variant is None and len(getattr(inspect_file(path), "measurements", ())) != 1:
            return
        dataset = pattern_dataset_path(path, variant)
    except Exception:  # noqa: BLE001 - nothing to name, or a group the listing chokes on: the array is read
        return
    try:
        setattr(signal, _FILE_SOURCE_ATTR, (weakref.ref(data), str(path), dataset))
    except (TypeError, AttributeError):
        pass                            # an object that takes no attributes reads the array


def _range_of_file_source(signal, data, *, check=None, progress=None):
    """``dataset_pattern_range`` of the dataset ``signal`` was loaded from (``note_file_source``);
    ``RANGE_UNKNOWN`` when the signal no longer is that dataset or the file cannot say."""
    entry = getattr(signal, _FILE_SOURCE_ATTR, None)
    if not (isinstance(entry, tuple) and len(entry) == 3 and entry[0]() is data):
        return RANGE_UNKNOWN
    import h5py
    try:
        with h5py.File(entry[1], "r") as f:
            dset = f[entry[2]]
            if dset.dtype != data.dtype or dset.size != data.size:
                return RANGE_UNKNOWN
            return dataset_pattern_range(dset, check=check, progress=progress)
    except (OSError, KeyError):
        return RANGE_UNKNOWN


#: Counts the forgets. A pass over the array that overlaps an edit in place (the pass started before
#: the operation ended) has read half-edited values; the edit's forget bumps the counter, and a pass
#: stores its result only if the counter is what it was when the pass started. The store and the
#: bump are made under ``_RANGE_LOCK``, so that no store slips in between a bump and the clearing.
_RANGE_LOCK = threading.Lock()
_forget_generation = 0


def scan_value_range(signal, *, check=None, progress=None) -> Optional[Tuple[float, float]]:
    """(min, max) of the whole pattern array of a loaded signal when it holds signed integers,
    ``None`` otherwise. It is what the map run rescales an int16 scan by, so a single pattern read
    for an indexer must be rescaled by it too (``signal_pattern``).

    Computed once per signal and kept on it. The entry is tied to the array object it was computed
    from (a weak reference), so loading another file, switching, cropping and replacing the
    array (NLPAR) all find it stale at once; an edit in place (background removal,
    averaging, ...) keeps the array, which is why the viewer calls ``forget_scan_range`` when it
    marks the dataset dirty. A pass that overlaps such an edit returns what it read but does not
    keep it (see ``_forget_generation``)."""
    data = getattr(signal, "data", None)
    dtype = getattr(data, "dtype", None)
    if dtype is None or np.dtype(dtype).kind != "i":
        return None
    entry = getattr(signal, _SCAN_RANGE_ATTR, None)
    if isinstance(entry, tuple) and len(entry) == 2:
        ref, rng = entry
        if ref() is data:
            return rng
    generation = _forget_generation
    rng = _range_of_file_source(signal, data, check=check, progress=progress)
    if rng is RANGE_UNKNOWN:
        rng = signed_pattern_range(data, check=check, progress=progress)
    with _RANGE_LOCK:
        if generation == _forget_generation:
            try:
                setattr(signal, _SCAN_RANGE_ATTR, (weakref.ref(data), rng))
            except (TypeError, AttributeError):
                pass                  # an object that takes no attributes: asked again next time
    return rng


def forget_scan_range(signal, *, keep_file_source: bool = False) -> None:
    """Drop the range kept on ``signal`` (its values were changed in place), and keep every pass
    that is running right now from storing the range it has read so far. ``keep_file_source``: the
    signal may share the edited buffer, but its lazy array, if it has one, was not touched."""
    global _forget_generation
    with _RANGE_LOCK:
        _forget_generation += 1
        for attr in (_SCAN_RANGE_ATTR,) if keep_file_source else (_SCAN_RANGE_ATTR, _FILE_SOURCE_ATTR):
            if signal is not None and getattr(signal, attr, None) is not None:
                try:
                    setattr(signal, attr, None)
                except (TypeError, AttributeError):
                    pass


def patterns_to_float32(block, value_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """A block of patterns as float32 for an indexer. With a ``value_range`` (see
    ``signed_pattern_range``) the values are the 0..255 values ``to_uint8`` gives the whole scan;
    without one the plain float32 conversion."""
    if value_range is None:
        return np.asarray(block, dtype=np.float32)
    lo, hi = value_range
    out = np.asarray(block).astype(np.float32)
    if not hi > lo:
        out[...] = 0.0
        return out
    out -= lo
    out *= 255.0 / (hi - lo)
    np.rint(out, out=out)
    return out


def for_signal_indexer(signal, pattern) -> np.ndarray:
    """A pattern (or a block of patterns) of ``signal`` as an indexer must see it: exactly the
    values the map run sees for the same pixel. A signed-integer scan gives the uint8 rescale by the
    range of the whole scan (``scan_value_range``), the same numbers as ``to_uint8(signal.data)``;
    every other dtype is returned unchanged."""
    pattern = np.asarray(pattern)
    value_range = scan_value_range(signal)
    if value_range is None:
        return pattern
    return np.clip(patterns_to_float32(pattern, value_range), 0, 255).astype(np.uint8)


def signal_pattern(signal, row: int, col: int) -> np.ndarray:
    """Pattern (row, col) of ``signal`` for an indexer (``for_signal_indexer``). Raises what
    ``signal.data[row, col]`` raises."""
    return for_signal_indexer(signal, signal.data[row, col])


def _natural_key(name: str):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name)]


def _text_value(value: Any) -> str:
    value = np.asarray(value).reshape(-1)[0] if np.ndim(value) else value
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    return str(value)


def _manufacturer(f) -> Optional[str]:
    import h5py
    for key, obj in f.items():
        if key.lower() == "manufacturer" and isinstance(obj, h5py.Dataset):
            return _text_value(obj[()]).lower()
    for key, val in f.attrs.items():
        if key.lower() == "manufacturer":
            return _text_value(val).lower()
    return None


def _scan_groups(f) -> List[str]:
    """Top-level groups with an ``EBSD/Data`` child, in natural order (``2`` before ``10``)."""
    import h5py
    names = []
    for key in f.keys():
        try:
            obj = f[key]
        except KeyError:  # broken soft or external link
            continue
        if isinstance(obj, h5py.Group) and "EBSD/Data" in obj:
            names.append(key)
    return sorted(names, key=_natural_key)


def _classify_h5(f, scans: List[str]) -> str:
    man = _manufacturer(f) or ""
    for needle, fmt in (("oxford", "oxford"), ("bruker", "bruker"), ("edax", "edax_h5ebsd"),
                        ("kikuchipy", "kikuchipy_h5ebsd")):
        if needle in man:
            return fmt
    for name in scans:
        data = f[name]["EBSD/Data"]
        for fmt in _H5_FORMATS:
            if any(ds in data for _, ds, _ in _SOURCES[fmt]):
                return fmt
    return "other"


#: the two sections every NORDIF ``Setting.txt`` has (kikuchipy's readers look both up by name)
_NORDIF_SECTIONS = ("[Microscope]", "[Area]")
_NORDIF_HEAD_BYTES = 256 * 1024


def _is_nordif_setting(path: str) -> bool:
    """True when ``path`` reads like a NORDIF settings file: both sections of ``_NORDIF_SECTIONS``
    stand alone on a line (a byte order mark before the first is tolerated)."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(_NORDIF_HEAD_BYTES)
    except OSError:
        return False
    lines = {line.strip().lstrip("﻿\xef\xbb\xbf") for line in head.decode("latin-1").splitlines()}
    return all(section in lines for section in _NORDIF_SECTIONS)


def _is_nordif_pattern_file(path: str) -> bool:
    """A ``.dat`` file is NORDIF's pattern file when the ``Setting.txt`` kikuchipy's reader would
    look for beside it is a NORDIF settings file. A ``.dat`` has no content of its own to tell."""
    folder = Path(path).parent
    try:
        names = os.listdir(folder)
    except OSError:
        return False
    return any(name.lower() == "setting.txt" and _is_nordif_setting(str(folder / name)) for name in names)


def detect_format(path: str) -> str:
    """The layout of an EBSD file: oxford, bruker, edax_h5ebsd, kikuchipy_h5ebsd, apex, edax_up,
    nordif, oxford_binary or other. UP and ``.ebsp`` files are told by extension, HDF5 by layout,
    NORDIF by content: a ``.txt`` that is a NORDIF settings file, a ``.dat`` beside one (any other
    text or data file is ``other``)."""
    path = str(path)
    suffix = Path(path).suffix.lower()
    if suffix in (".up1", ".up2"):
        return "edax_up"
    if suffix == ".txt":
        return "nordif" if _is_nordif_setting(path) else "other"
    if suffix == ".dat":
        return "nordif" if _is_nordif_pattern_file(path) else "other"
    if suffix == ".ebsp":
        return "oxford_binary"
    try:
        import h5py
        if not h5py.is_hdf5(path):
            return "other"
        try:
            from edax_apex import is_apex_file
        except ImportError:
            is_apex_file = None
        if is_apex_file is not None and is_apex_file(path):
            return "apex"
        with h5py.File(path, "r") as f:
            return _classify_h5(f, _scan_groups(f))
    except Exception:  # noqa: BLE001 - unreadable is "not one of ours"
        return "other"


def _scalar_int(group, name: str) -> int:
    if group is None or name not in group:
        return 0
    try:
        return int(np.asarray(group[name][()]).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return 0


def _source(scan: str, hdr, key: str, dataset: str, background: Optional[str], data) -> PatternSource:
    path = f"{scan}/EBSD/Data/{dataset}"
    has_bg = bool(background and hdr is not None and background in hdr)
    ds = data[dataset] if dataset in data else None
    shape = tuple(ds.shape) if ds is not None and hasattr(ds, "shape") else ()
    if len(shape) < 3 or not all(shape):
        return PatternSource(key, path, 0, (0, 0), "" if ds is None else str(ds.dtype), has_bg, False, _NO_PATTERNS)
    n = int(np.prod(shape[:-2]))
    return PatternSource(key, path, n, (int(shape[-2]), int(shape[-1])), str(ds.dtype), has_bg, True, None)


def _measurement(f, fmt: str, scan: str) -> Measurement:
    group = f[scan]
    data = group["EBSD/Data"]
    hdr = group["EBSD/Header"] if "EBSD/Header" in group else None
    rows, cols = _GRID[fmt]
    sources = [_source(scan, hdr, key, dataset, bg, data)
               for key, dataset, bg in _SOURCES[fmt] if dataset in data]
    if not sources:
        key, dataset, bg = _SOURCES[fmt][0]
        sources = [_source(scan, hdr, key, dataset, bg, data)]
    return Measurement(scan, scan, _scalar_int(hdr, rows), _scalar_int(hdr, cols), sources)


def crop_variant(f, measurements: List[Measurement]) -> Optional[LoadVariant]:
    """The variant ``CropProvenance`` names, if the file offers it (loadable), else None."""
    prov = f.get("CropProvenance")
    if prov is None or not hasattr(prov, "attrs"):
        return None
    attrs = prov.attrs
    if "pattern_source" not in attrs:
        return None
    source = _text_value(attrs["pattern_source"])
    name = _text_value(attrs["measurement"]) if "measurement" in attrs else None
    if name is None:
        # written for the default measurement: the first one that has patterns
        name = next((m.name for m in measurements if any(s.loadable for s in m.sources)), None)
    for m in measurements:
        if m.name == name:
            for src in m.sources:
                if src.key == source and src.loadable:
                    return LoadVariant(m.name, source)
    return None


def _marked_variant(f, measurements: List[Measurement]) -> Optional[LoadVariant]:
    """The variant a rich export marked (``pattern_source`` on the pattern dataset it holds). The
    dataset itself says the measurement and the source."""
    for m in measurements:
        for src in m.sources:
            if src.loadable and "pattern_source" in f[src.dataset].attrs:
                return LoadVariant(m.name, src.key)
    return None


def inspect_file(path: str) -> Optional[FileChoices]:
    """The measurements and pattern sources of an HDF5 EBSD file, or None for a file with exactly one
    variant by construction (``.up1/.up2``, ``.dat``, ``.ebsp``, ``.txt``, EDAX APEX) and for anything
    that is not a recognised h5ebsd layout or cannot be opened."""
    path = str(path)
    fmt = detect_format(path)
    if fmt not in _H5_FORMATS:
        return None
    try:
        import h5py
        with h5py.File(path, "r") as f:
            scans = _scan_groups(f)
            if not scans:
                return None
            measurements = [_measurement(f, fmt, s) for s in scans]
            # a rich export of a crop still holds the crop's record; the crop writer drops marks
            suggested = _marked_variant(f, measurements) or crop_variant(f, measurements)
            return FileChoices(path, fmt, measurements, suggested)
    except OSError:
        return None


def pattern_dataset_path(h5_path: str, variant: Optional[LoadVariant]) -> str:
    """The HDF5 path of the pattern dataset a variant reads. ``None`` is what loading without a choice
    reads: the first measurement, processed patterns (the first source that exists when it has none)."""
    choices = inspect_file(h5_path)
    if choices is None:
        raise LoadChoiceError(_NO_PATTERNS, f"{Path(str(h5_path)).name} has no pattern dataset to choose")
    return _variant_dataset(choices, h5_path, variant)


def _variant_dataset(choices: FileChoices, h5_path: str, variant: Optional[LoadVariant]) -> str:
    """``pattern_dataset_path`` for choices that were already listed."""
    if variant is None:
        variant = choices.default_variant()
    for m in choices.measurements:
        if variant.measurement not in (None, m.name):
            continue
        for s in m.sources:
            if s.key != variant.pattern_source:
                continue
            if not s.loadable:
                raise LoadChoiceError(s.problem or _NO_PATTERNS, f"{m.name} / {s.key} has no patterns")
            return s.dataset
    raise LoadChoiceError(
        _UNKNOWN_VARIANT,
        f"{Path(str(h5_path)).name} has no {variant.pattern_source} patterns in measurement {variant.measurement!r}")


def reread_dataset_path(h5_path: str, variant: LoadVariant, where: str) -> str:
    """The pattern dataset a file re-reader (``where`` names it) must read for ``variant``.

    A re-reader that opens the file by path cannot fall back to another source: where the file's
    layout is not one the load choices describe it refuses with ``variantNotSupported`` and its own
    name in ``params["where"]`` (the prose says "this step": it is read by people, ``where`` is for
    the log and the tests). A variant the described layout lacks keeps its own code."""
    if inspect_file(h5_path) is None:
        raise LoadChoiceError(
            _NOT_SUPPORTED,
            f"This step cannot read the {variant.pattern_source} patterns of measurement "
            f"{variant.measurement!r} of {Path(str(h5_path)).name}: the layout of this file does not "
            f"name its pattern sources",
            {"where": where})
    return pattern_dataset_path(h5_path, variant)


def measurement_groups(f, variant: Optional[LoadVariant], default: Optional[str] = None) -> List[str]:
    """The top-level HDF5 groups a reader of per-measurement data looks in, in the order it tries them.

    THE rule for "which measurement does this reader read", for every reader that opens a file by
    path or holds an open handle (Band Contrast, the acquisition header, Euler/Phase, EDS, electron
    images, ...):

    * ``variant`` names a measurement: that group alone. A reader never falls through to another
      measurement; a file that lacks the group refuses with ``variantNotSupported`` (the file is not
      the one the variant was loaded from, or it was changed since).
    * no variant, or one that names no measurement: what the reader did before variants existed,
      ``[default]`` for a reader that had a fixed group (the Oxford ``"1"``), else every top-level key
      in file order (the first group that has the channel wins).
    """
    name = None if variant is None else variant.measurement
    if name is None:
        return [default] if default is not None else list(f.keys())
    if name not in f:
        raise LoadChoiceError(
            _NOT_SUPPORTED,
            f"{Path(str(getattr(f, 'filename', '') or 'The file')).name} has no measurement {name!r}, so "
            f"the data of the measurement that was loaded cannot be read from it",
            {"measurement": name})
    return [name]


def variant_from_stamp(stamp: Any) -> Optional[LoadVariant]:
    """The ``LoadVariant`` a result stamp (``{"measurement", "pattern_source"}``, or None for the
    default variant) describes. An unreadable stamp is the default: it never reaches a reader as
    an error of its own."""
    if not isinstance(stamp, dict) or not stamp.get("pattern_source"):
        return None
    try:
        return LoadVariant.from_dict(stamp)
    except LoadChoiceError:
        return None


def middle_pattern(h5_path: str, measurement: str, source: str) -> np.ndarray:
    """The middle pattern of one pattern source of an HDF5 file, as it is stored.

    "Middle" is the pattern with flat scan index ``n // 2`` of the ``n`` patterns of the source (a
    4-D dataset is indexed row-major). Only that one pattern is read from the file; nothing is
    loaded. ``LoadChoiceError`` for a file that offers no choices (``loadNoPatterns``), a measurement
    or source it lacks (``loadVariantUnknown``) or a source without patterns."""
    choices = inspect_file(h5_path)
    if choices is None:
        raise LoadChoiceError(_NO_PATTERNS, f"{Path(str(h5_path)).name} has no patterns to preview")
    variant = LoadVariant.from_dict({"measurement": measurement, "pattern_source": source})
    dataset = _variant_dataset(choices, h5_path, variant)
    import h5py
    with h5py.File(h5_path, "r") as f:
        dset = f[dataset]
        lead = tuple(int(n) for n in dset.shape[:-2])
        index = np.unravel_index(int(np.prod(lead, dtype=np.int64)) // 2, lead)
        key = int(index[0]) if len(lead) == 1 else tuple(int(i) for i in index)
        return np.asarray(dset[key])


def preview_png(h5_path: str, measurement: str, source: str) -> bytes:
    """A PNG of ``middle_pattern``, stretched by ``to_uint8`` (the value range of this one pattern)."""
    import io
    from PIL import Image
    image = to_uint8(middle_pattern(h5_path, measurement, source))
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return buf.getvalue()
