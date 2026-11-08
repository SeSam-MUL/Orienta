"""The loaders that open a file again by path, and the answer of a load, on a file loaded as a variant.

Three things met when the APEX loading and the pattern-variant loading were brought together:

- a loader that re-opens a loaded file by path (the image-quality fallback, the cross-file Pattern
  Match, the batch loader) has to hand the loader BOTH what it remembers of the file: the metadata
  pairing of an EDAX measurement and the measurement x pattern source it was loaded as;
- the cross-file Pattern Match reads a result's file as the variant the result ran on;
- the answer of a load describes the dataset that is active after a stash was restored, also when the
  file was loaded as a variant.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import safe_loader  # noqa: E402
from backend.api.routes import ebsd_viewer as ev  # noqa: E402
from backend.api.services import pattern_quality  # noqa: E402
from backend.api.services.load_choices import LoadVariant  # noqa: E402
from tests.test_load_variant import NX, NY, SX, SY  # noqa: E402
from tests.test_load_variant_routes import _load, _v, client, multi, other  # noqa: E402,F401
from tools import pattern_comparison  # noqa: E402

RAW2 = LoadVariant("2", "unprocessed")
PAIRING = {"h5_path": "meta.h5", "measurement_id": "00000000-0000-0000-0000-00000000000a"}


@pytest.fixture(autouse=True)
def clean_cache():
    pattern_comparison._source_signal_cache.clear()
    yield
    pattern_comparison._source_signal_cache.clear()


class _Loaded(Exception):
    """Raised by the fake loader after it recorded what it was asked for."""


@pytest.fixture
def recorded(monkeypatch):
    calls = []

    def fake(path, *args, **kwargs):
        calls.append(kwargs)
        raise _Loaded

    monkeypatch.setattr(safe_loader, "load_ebsd_safe", fake)
    # the entry of a file that was paired by hand: the rule that checks the pairing against the metadata
    # file is the switch's and has its own tests
    monkeypatch.setattr(ev, "_usable_stored_pairing", lambda path, stored: (stored, None))
    return calls


def _remember(path, **entry):
    ev._loaded_files[:] = [{"path": str(path), "name": Path(path).stem, **entry}]


def test_the_image_quality_fallback_hands_the_loader_the_pairing_and_the_variant(multi, recorded):
    _remember(multi, apex=PAIRING)
    assert pattern_quality.image_quality_from_file(multi, variant=RAW2) is None      # the fake raised
    assert recorded == [{"verbose": False, "apex": PAIRING, "variant": RAW2}]


def test_the_cross_file_pattern_match_hands_the_loader_the_pairing_and_the_variant(multi, recorded):
    _remember(multi, apex=PAIRING)
    assert pattern_comparison._get_source_signal(multi, RAW2) is None
    assert recorded == [{"verbose": False, "apex": PAIRING, "variant": RAW2}]


def test_the_batch_loader_hands_the_loader_the_pairing_and_the_variant(multi, recorded):
    _remember(multi, apex=PAIRING, variant=_v("2", "unprocessed"))
    assert ev.load_ebsd_file(multi) is False
    assert recorded == [{"apex": PAIRING, "variant": RAW2}]


def test_a_file_with_neither_is_loaded_exactly_as_it_was(multi, recorded):
    _remember(multi)
    pattern_quality.image_quality_from_file(multi)
    pattern_comparison._get_source_signal(multi)
    ev.load_ebsd_file(multi)
    assert recorded == [{"verbose": False}, {"verbose": False}, {}]


# ------------------------------------------------------------ the cross-file Pattern Match

def _unprocessed_counts(group: int) -> np.ndarray:
    return np.array([7 * k - 500 + 1000 * group for k in range(NY * NX * SY * SX)],
                    dtype=np.int16).reshape(NY, NX, SY, SX)


def test_a_result_of_an_unprocessed_measurement_is_read_as_that_variant(multi, monkeypatch):
    """The active file is another one; the result names ``multi`` and the variant it ran on."""
    monkeypatch.setattr(ev, "_ebsd_file_path", "some/other/file.h5oina")
    monkeypatch.setattr(ev, "_get_active_signal", lambda: None)
    stamp = {"measurement": "2", "pattern_source": "unprocessed"}
    result = SimpleNamespace(metadata={"source_file": multi, "load_variant": stamp})
    pattern = pattern_comparison.get_experimental_pattern(result, 1, 2)
    assert pattern is not None
    assert np.array_equal(pattern, _unprocessed_counts(2)[1, 2].astype(np.float32))   # counts of group 2
    # a result of the default variant of the same file reads the processed patterns of group 1
    default = SimpleNamespace(metadata={"source_file": multi})
    pattern = pattern_comparison.get_experimental_pattern(default, 1, 2)
    assert pattern.dtype == np.float32 and pattern.max() < 256
    assert not np.array_equal(pattern, _unprocessed_counts(2)[1, 2].astype(np.float32))
    with h5py.File(multi, "r") as f:
        processed = f["1/EBSD/Data/Processed Patterns"][1 * NX + 2]
    assert np.array_equal(pattern, processed.astype(np.float32))
    assert len(pattern_comparison._source_signal_cache) == 2     # one signal per variant of the file


# ------------------------------------------------------------------------------ the load answer

def test_the_answer_of_a_variant_load_describes_the_dataset_that_is_active_after_the_restore(
        client, multi, other):
    unprocessed = _v("1", "unprocessed")
    assert _load(client, multi, unprocessed).status_code == 200
    r = client.post("/api/ebsd/crop", json={"shape": "rect", "row0": 0, "col0": 0, "rows": 1, "cols": 2,
                                            "name": "u1"})
    assert r.status_code == 200, r.text
    assert _load(client, other).status_code == 200
    answer = _load(client, multi, unprocessed).json()
    # the crop is the active dataset again, and the answer is about it: its grid, its shape
    assert answer["dataset_name"] == "u1" and answer["grid_shape"] == [1, 2]
    assert answer["navigation_shape"] == [2, 1] and answer["data_shape"] == [1, 2, SY, SX]
    assert answer["pattern_count"] == 2
    # what the FILE says stays the file's: the variant it was loaded as, its raw patterns, its format
    assert answer["variant"] == {"measurement": "1", "pattern_source": "unprocessed", "is_default": False}
    assert answer["has_raw_patterns"] is True and answer["format_type"] == "Oxford"
    assert answer["file_path"] == multi
