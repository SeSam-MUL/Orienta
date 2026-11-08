"""A file leaves the loaded-files list: its stash goes, for every variant of it.

The viewer keeps the datasets it must not lose (crops, derived datasets, a PC the user made) in a
stash under the file and the variant the file was loaded as: the plain path for the default variant,
``(path, (measurement, source))`` for another. Letting go of a file (remove, clear, clear all) frees
the lazy signals that hold its pattern file open, and keeps only a PC the user made. That has to hold
for every variant of the file, not only for the default one. Real routes, a synthetic Oxford file
with two scan groups of processed and unprocessed patterns.
"""
from __future__ import annotations

import gc
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.routes import ebsd_viewer as ev  # noqa: E402
from backend.api.services.calibration_store import calibration_store  # noqa: E402
from tests.test_load_variant_routes import _load, _v, client, multi, other  # noqa: E402,F401

DOT = "·"
UNPROCESSED = f"multi {DOT} unprocessed"
TYPED = [0.31, 0.42, 0.53]
TYPED_RAW = [0.27, 0.38, 0.49]


def _crop(client, name):
    r = client.post("/api/ebsd/crop", json={"shape": "rect", "row0": 0, "col0": 0, "rows": 1, "cols": 2,
                                            "name": name})
    assert r.status_code == 200, r.text


def _stashes_of(path):
    canon = ev._canonical_path(path)
    return {k: v for k, v in ev._registry_by_file.items()
            if (k[0] if isinstance(k, tuple) else k) == canon}


def _left_behind_as_two_variants(client, multi, other):
    """``multi`` as its default and as its unprocessed patterns, each cropped, then ``other`` loaded."""
    _load(client, multi)
    _crop(client, "d1")
    _load(client, multi, _v("1", "unprocessed"))
    _crop(client, "u1")
    _load(client, other)
    assert len(_stashes_of(multi)) == 2           # the premise: one stash per variant of the file


def _weakrefs_of(path):
    return [weakref.ref(sig) for stash in _stashes_of(path).values() for sig in stash["raw_signals"].values()]


def test_removing_a_file_drops_the_stash_of_every_variant_of_it(client, multi, other):
    _left_behind_as_two_variants(client, multi, other)
    refs = _weakrefs_of(multi)
    assert len(refs) == 4                          # the raw dataset and the crop of each variant
    r = client.post("/api/ebsd/loaded-files/remove", json={"path": multi})
    assert r.status_code == 200, r.text
    assert _stashes_of(multi) == {}
    gc.collect()
    assert all(ref() is None for ref in refs)      # nothing holds a signal of the removed file any more
    back = _load(client, multi).json()             # a fresh start: no crop of the default variant
    assert back["dataset_name"] == "multi" and sorted(ev._raw_signals) == ["multi"]
    _load(client, multi, _v("1", "unprocessed"))
    assert sorted(ev._raw_signals) == [UNPROCESSED]


@pytest.mark.parametrize("how", ["clear", "clear-all"])
def test_clearing_the_list_drops_the_stash_of_every_variant_of_the_files_it_lets_go_of(
        client, multi, other, how):
    _left_behind_as_two_variants(client, multi, other)
    r = client.post(f"/api/ebsd/loaded-files/{how}")
    assert r.status_code == 200, r.text
    assert _stashes_of(multi) == {}


@pytest.mark.parametrize("how", ["remove", "clear", "clear-all"])
def test_a_file_that_was_only_ever_loaded_as_a_variant_leaves_no_stash_either(client, multi, other, how):
    """No stash under the plain path at all: the only key of the file is a ``(path, variant)`` one."""
    _load(client, multi, _v("1", "unprocessed"))
    _crop(client, "u1")
    _load(client, other)
    stashes = _stashes_of(multi)
    assert len(stashes) == 1 and all(isinstance(key, tuple) for key in stashes)
    if how == "remove":
        r = client.post("/api/ebsd/loaded-files/remove", json={"path": multi})
    else:
        r = client.post(f"/api/ebsd/loaded-files/{how}")
    assert r.status_code == 200, r.text
    assert _stashes_of(multi) == {}


def test_a_pc_the_user_made_stays_for_the_variant_it_was_made_on_and_for_no_other(client, multi, other):
    _load(client, multi)
    calibration_store.update_pc("multi", np.array(TYPED), source="manual")
    _crop(client, "d1")
    _load(client, multi, _v("1", "unprocessed"))
    calibration_store.update_pc(UNPROCESSED, np.array(TYPED_RAW), source="manual")
    _crop(client, "u1")
    _load(client, other)
    assert client.post("/api/ebsd/loaded-files/remove", json={"path": multi}).status_code == 200
    stashes = _stashes_of(multi)
    assert len(stashes) == 2
    for stash in stashes.values():                 # a PC and nothing that holds a file open
        assert stash["raw_signals"] == {} and stash["crops"] == {} and stash["dirty"] == set()
    assert {n: [round(float(v), 6) for v in e.pc_single]
            for stash in stashes.values() for n, e in stash["calibration"].items()
            if n in ("multi", UNPROCESSED)} == {"multi": TYPED, UNPROCESSED: TYPED_RAW}
    # loaded again by path, each variant gets its own PC back and neither gets the other's
    r = _load(client, multi)
    assert r.json()["pc"] == pytest.approx(TYPED)
    r = _load(client, multi, _v("1", "unprocessed"))
    assert r.json()["pc"] == pytest.approx(TYPED_RAW)


def test_the_stash_of_the_active_file_is_not_touched_by_a_clear(client, multi, other):
    _load(client, multi)
    _crop(client, "d1")
    _load(client, multi, _v("1", "unprocessed"))       # the default variant's stash, the file is active
    assert len(_stashes_of(multi)) == 1
    _load(client, other)
    _load(client, multi, _v("1", "unprocessed"))
    client.post("/api/ebsd/loaded-files/clear")
    assert len(_stashes_of(multi)) == 1                # still there: it is the active file's own
