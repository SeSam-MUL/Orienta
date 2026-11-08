"""What pymatgen read from a CIF is kept across backend starts (cif_parse_cache).

Measured 2026-11-08: every start parsed the 46 library CIFs twice with pymatgen before the
Indexing page could list a phase (about 12 s for the compositions, 11.2 s for Crystal Hint).
These tests pin that an unchanged file is not parsed again, that a changed one is, and that
a failure is never kept.
"""
import os
import shutil
import time
from pathlib import Path

import pytest

from backend.api.services import cif_parse_cache

LIB = Path(__file__).resolve().parents[1] / "Database" / "CIF_Library"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ORIENTA_CIF_PARSE_CACHE", str(tmp_path / "cache" / "cif_parse.json"))
    cif_parse_cache._reset_for_tests()
    yield tmp_path / "cache" / "cif_parse.json"
    cif_parse_cache._reset_for_tests()


def _counting(value):
    calls = []

    def compute(path):
        calls.append(path)
        return value
    return compute, calls


def test_an_unchanged_file_is_read_from_the_cache_after_a_restart(cache, tmp_path):
    cif = tmp_path / "x.cif"
    cif.write_text("data_x\n")
    compute, calls = _counting({"formula": "Al", "elements": ["Al"]})
    assert cif_parse_cache.cached("k", cif, compute) == {"formula": "Al", "elements": ["Al"]}
    cif_parse_cache._reset_for_tests()          # a new process: only the file remains
    assert cif_parse_cache.cached("k", cif, compute) == {"formula": "Al", "elements": ["Al"]}
    assert len(calls) == 1
    assert cache.exists()


def test_a_changed_file_is_parsed_again(cache, tmp_path):
    cif = tmp_path / "x.cif"
    cif.write_text("data_x\n")
    compute, calls = _counting("Al")
    cif_parse_cache.cached("k", cif, compute)
    time.sleep(0.02)
    cif.write_text("data_x\n_cell_length_a 4.05\n")   # other size, other mtime
    cif_parse_cache.cached("k", cif, compute)
    os.utime(cif, ns=(cif.stat().st_atime_ns, cif.stat().st_mtime_ns + 1_000))   # same size, newer
    cif_parse_cache.cached("k", cif, compute)
    assert len(calls) == 3


def test_another_kind_is_another_entry(cache, tmp_path):
    cif = tmp_path / "x.cif"
    cif.write_text("data_x\n")
    compute, calls = _counting("Al")
    cif_parse_cache.cached("composition-1", cif, compute)
    cif_parse_cache.cached("composition-2", cif, compute)
    assert len(calls) == 2


def test_a_failure_is_not_kept(cache, tmp_path):
    cif = tmp_path / "x.cif"
    cif.write_text("data_x\n")
    compute, calls = _counting({"error": "pymatgen missing"})
    keep = lambda v: "error" not in v      # noqa: E731
    cif_parse_cache.cached("k", cif, compute, keep=keep)
    cif_parse_cache.cached("k", cif, compute, keep=keep)
    assert len(calls) == 2


def test_an_unreadable_cache_file_costs_the_time_not_the_answer(cache, tmp_path):
    cache.parent.mkdir(parents=True)
    cache.write_text("{not json")
    cif = tmp_path / "x.cif"
    cif.write_text("data_x\n")
    compute, calls = _counting("Al")
    assert cif_parse_cache.cached("k", cif, compute) == "Al"
    assert len(calls) == 1


def test_switched_off_it_always_parses(monkeypatch, tmp_path):
    monkeypatch.setenv("ORIENTA_CIF_PARSE_CACHE", "0")
    cif_parse_cache._reset_for_tests()
    cif = tmp_path / "x.cif"
    cif.write_text("data_x\n")
    compute, calls = _counting("Al")
    cif_parse_cache.cached("k", cif, compute)
    cif_parse_cache.cached("k", cif, compute)
    assert len(calls) == 2


@pytest.mark.skipif(not (LIB / "Al.cif").exists(), reason="the library's Al.cif is not here")
def test_the_real_readers_give_the_same_answer_from_the_cache(cache, tmp_path):
    """Through the readers the app calls: the second start answers from the file with the
    same values, and the Crystal Hint elements are a tuple again."""
    import phase_metadata
    from backend.api.services import crystal_hint_local_library as chl
    cif = tmp_path / "Al.cif"
    shutil.copy(LIB / "Al.cif", cif)
    first_comp = phase_metadata._composition_of(cif)
    first_info = chl._parse_cif(cif)
    assert cif_parse_cache.cached("composition-1", cif, phase_metadata._composition_of) == first_comp
    assert chl._parse_cif_kept(cif) == first_info
    cif_parse_cache._reset_for_tests()

    def must_not_parse(path):
        raise AssertionError("parsed again although the file did not change")
    assert cif_parse_cache.cached("composition-1", cif, must_not_parse) == first_comp
    info = cif_parse_cache.cached("crystal-hint-cif-1", cif, must_not_parse)
    assert tuple(info["elements"]) == first_info["elements"]
    assert chl._parse_cif_kept(cif) == first_info          # elements a tuple again


def test_a_crystal_hint_parse_without_a_space_group_is_not_kept(cache, tmp_path, monkeypatch):
    """pymatgen's symmetry finder failing is not a fact about the file; it must not outlive the start."""
    from backend.api.services import crystal_hint_local_library as chl
    cif = tmp_path / "x.cif"
    cif.write_text("data_x\n")
    calls = []

    def degraded(path):
        calls.append(path)
        return {"formula": "Al", "elements": ("Al",), "space_group": "", "space_group_number": None}
    monkeypatch.setattr(chl, "_parse_cif", degraded)
    chl._parse_cif_kept(cif)
    cif_parse_cache._reset_for_tests()
    chl._parse_cif_kept(cif)
    assert len(calls) == 2
