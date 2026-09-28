"""`found` must be measured on THIS machine, and crystallography must not
depend on whether a sidecar happens to exist.

Three measured defects in `build_sht_info` (2026-09-27), all in the sidecar
branch, all invisible to the existing tests because those write sidecars whose
files they never create and then only assert the NAME:

  1. `found` and `path` were copied out of the sidecar unverified. A sidecar
     records an absolute path belonging to the machine that ran the simulation;
     25 of the recorded paths in the shipped library name the `E:` drive and do
     not resolve on a `C:` checkout. The File Info panel prints the file name
     only when `found` is true (`DatabasePage.jsx:674-675`), so an unverified
     flag is a promise the app cannot keep.
  2. `crystallography` was read from the sidecar, which never carries that key,
     so it came back `{}` for exactly the 13 of 29 masters that HAVE a sidecar
     while the 16 without got it from the .sht binary.
  3. an unreadable sidecar fell through to "recovery" and lost its parameters
     too, although provenance and parameters are independent.

Nothing user-visible was wrong on today's library -- every recorded name does
resolve locally -- which is why this is a correctness fix and not a release
matter. `test_a_sidecar_may_not_promise_a_file_that_is_not_here` is the case
that was one rename away.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.api.services.sht_provenance import build_sht_info

SIDE = ".sht.provenance.json"


def _sht(tmp_path: Path, name="Ni (Ni) [cF4] {20kV}.sht") -> Path:
    p = tmp_path / name
    p.write_bytes(b"not a real sht")   # the binary read is best-effort
    return p


def _sidecar(sht: Path, **over) -> Path:
    doc = {
        "schema": 1,
        "source_xtal": {"name": "Ni.xtal", "path": r"E:\elsewhere\Ni.xtal",
                        "found": True},
        "source_cif": {"name": "Ni.cif", "path": r"E:\elsewhere\Ni.cif",
                       "found": True},
        "reference": "ref",
        "parameters": {"dmin": 0.05, "npx": 500, "engine": "ours"},
    }
    doc.update(over)
    out = sht.with_suffix(SIDE)
    out.write_text(json.dumps(doc), encoding="utf-8")
    return out


# --- 1. existence is measured here ----------------------------------------

def test_a_sidecar_may_not_promise_a_file_that_is_not_here(tmp_path):
    """The defect, in one test: a sidecar claiming files that exist nowhere."""
    sht = _sht(tmp_path)
    _sidecar(sht)                       # says found: true for both
    info = build_sht_info(sht, xtal_dir=tmp_path, cif_dir=tmp_path)
    p = info["provenance"]
    assert p["source_xtal"]["found"] is False
    assert p["source_cif"]["found"] is False
    # the name is still reported -- it is provenance, and it is what the panel
    # would print IF the file were here
    assert p["source_xtal"]["name"] == "Ni.xtal"


def test_a_foreign_path_is_mapped_onto_the_local_library_by_name(tmp_path):
    """The recorded path is `E:\\...`; the file is here. Resolve by NAME."""
    sht = _sht(tmp_path)
    _sidecar(sht)
    (tmp_path / "Ni.xtal").write_bytes(b"x")
    (tmp_path / "Ni.cif").write_text("data_Ni\n", encoding="utf-8")

    p = build_sht_info(sht, xtal_dir=tmp_path, cif_dir=tmp_path)["provenance"]
    assert p["source_xtal"]["found"] is True
    # `path` used to be the absolute local path. It is project-relative now, and
    # the bare name when the file is outside the project -- as `tmp_path` is.
    # See tests/test_phase_library_no_local_paths.py: f7 found this field coming
    # back as `C:\Users\<name>\...` on 29 of 36 phases.
    assert p["source_xtal"]["path"] == "Ni.xtal"
    assert "E:" not in p["source_xtal"]["path"]
    assert "recorded_path" not in p["source_xtal"], (
        "resolved locally, so there is nothing unresolved to report")


def test_an_unresolvable_recorded_path_is_kept_visible_without_the_machine_path(tmp_path):
    """NARROWED, and it is a trade-off rather than a cleanup.

    This used to assert `recorded_path == r"E:\\elsewhere\\Ni.xtal"` verbatim,
    so that a reader could see what the sidecar claimed. But an unresolvable
    recorded path is by definition a path on somebody ELSE's machine, and these
    answers get pasted into bug reports -- `E:\\elsewhere` here, `C:\\Users\\<name>`
    in the wild.

    Reducing it to `Ni.xtal` alone would have quietly killed the field: it would
    then repeat `name` and tell the reader nothing. So the one thing they
    actually needed -- "this master was built against a library that is not this
    one" -- is kept as its own flag.
    """
    sht = _sht(tmp_path)
    _sidecar(sht)
    p = build_sht_info(sht, xtal_dir=tmp_path, cif_dir=tmp_path)["provenance"]
    assert p["source_xtal"]["path"] is None
    assert p["source_xtal"]["recorded_path"] == "Ni.xtal"
    assert p["source_xtal"]["recorded_elsewhere"] is True
    # No part of the foreign path survives. Checked on the VALUES: the key
    # `recorded_elsewhere` legitimately contains the word, and asserting against
    # the whole dict caught that instead of the thing it was looking for.
    values = [v for v in p["source_xtal"].values() if isinstance(v, str)]
    assert not [v for v in values if "E:" in v or "elsewhere" in v], values


def test_a_relative_recorded_path_is_not_flagged_as_elsewhere(tmp_path):
    """The other side of the flag. Sidecars written since this change record a
    project-relative path; calling those "elsewhere" would put a warning on
    every ordinary file and make the flag worthless."""
    sht = _sht(tmp_path)
    _sidecar(sht, source_xtal={"name": "Ni.xtal", "found": True,
                               "path": "Database/XTAL_Library/Ni.xtal"})
    p = build_sht_info(sht, xtal_dir=tmp_path, cif_dir=tmp_path)["provenance"]
    assert p["source_xtal"]["recorded_path"] == "Database/XTAL_Library/Ni.xtal"
    assert p["source_xtal"]["recorded_elsewhere"] is False


def test_it_finds_a_file_in_a_material_subfolder(tmp_path):
    """The library nests by material; a non-recursive lookup would miss it."""
    sht = _sht(tmp_path)
    _sidecar(sht)
    sub = tmp_path / "Ni_material"
    sub.mkdir()
    (sub / "Ni.xtal").write_bytes(b"x")
    p = build_sht_info(sht, xtal_dir=tmp_path, cif_dir=tmp_path)["provenance"]
    assert p["source_xtal"]["found"] is True
    # `found` is what this test is about, and the recursive lookup behind it is
    # unchanged. The path is reported project-relative now, and `tmp_path` is
    # outside the project, so it reduces to the name. Were such a folder to sit
    # under the project the relative form would keep it -- measured today the
    # shipped `Database/XTAL_Library` has NO material subfolders at all, so that
    # is a statement about the code, not about the library.
    assert p["source_xtal"]["path"] == "Ni.xtal"


def test_a_name_with_glob_metacharacters_still_resolves(tmp_path):
    """Library names contain `[`, `(` and `{`.

    `rglob("Al (Al) [cF4].xtal")` treats `[cF4]` as a character class and
    matches nothing, so the lookup has to escape the name. Without the escape
    this test fails while every other one passes.
    """
    sht = _sht(tmp_path)
    _sidecar(sht, source_xtal={"name": "Al (Al) [cF4] {20kV}.xtal",
                               "path": r"E:\elsewhere\x.xtal", "found": True})
    sub = tmp_path / "Al"
    sub.mkdir()
    (sub / "Al (Al) [cF4] {20kV}.xtal").write_bytes(b"x")
    p = build_sht_info(sht, xtal_dir=tmp_path, cif_dir=tmp_path)["provenance"]
    assert p["source_xtal"]["found"] is True


# --- 2. crystallography no longer depends on the sidecar ------------------

def test_the_sidecar_no_longer_suppresses_crystallography(tmp_path, monkeypatch):
    """With a sidecar present the binary must still be read.

    Stubbed, because a real .sht is 300-600 KB of binary and this asserts the
    WIRING, not the parser.
    """
    sht = _sht(tmp_path)
    _sidecar(sht)

    class _S:
        formula = "Ni"; space_group = 225; point_group = "m-3m"
        voltage_kv = 20.0; primary_tilt_deg = 70.0; bandwidth = 384
        crystal_lattice = (0.352, 0.352, 0.352, 90.0, 90.0, 90.0)
        sim_dmin = None; sim_npx = None; sim_numsx = None
        sim_totnum_el = None; sim_bethe = None
        software_version = b"fwd_sim0"; raw_header = None

    import backend.spherical_gpu.pipeline.sht_io as sht_io
    monkeypatch.setattr(sht_io, "read_sht_master", lambda *a, **k: _S())

    info = build_sht_info(sht, xtal_dir=tmp_path, cif_dir=tmp_path)
    crys = info["crystallography"]
    assert crys, "crystallography is empty although the binary carries it"
    assert crys["space_group"] == 225
    assert crys["point_group"] == "m-3m"
    assert crys["lattice"][0] == pytest.approx(0.352)
    # ...and the sidecar still wins for the parameters
    assert info["parameters"]["source"] == "sidecar"
    assert info["parameters"]["npx"] == 500
    assert info["provenance"]["origin"] == "sidecar"


# --- 3. a broken sidecar loses only what it broke -------------------------

def test_an_unreadable_sidecar_does_not_discard_the_rest(tmp_path):
    sht = _sht(tmp_path)
    sht.with_suffix(SIDE).write_text("{ this is not json", encoding="utf-8")
    (tmp_path / "Ni.xtal").write_bytes(b"x")

    info = build_sht_info(sht, xtal_dir=tmp_path, cif_dir=tmp_path)
    assert info["provenance"]["origin"] == "recovered"
    assert info["provenance"]["source_xtal"]["found"] is True, (
        "the .xtal is right there; a broken sidecar is no reason to deny it")


def test_no_sidecar_at_all_still_works(tmp_path):
    sht = _sht(tmp_path)
    (tmp_path / "Ni.xtal").write_bytes(b"x")
    info = build_sht_info(sht, xtal_dir=tmp_path, cif_dir=tmp_path)
    assert info["provenance"]["origin"] == "recovered"
    assert info["provenance"]["source_xtal"]["found"] is True


def test_a_sidecar_naming_nothing_falls_back_to_the_convention(tmp_path):
    """An old sidecar with only parameters: the .xtal name comes from the stem."""
    sht = _sht(tmp_path)
    sht.with_suffix(SIDE).write_text(json.dumps(
        {"schema": 1, "parameters": {"dmin": 0.07}}), encoding="utf-8")
    (tmp_path / "Ni.xtal").write_bytes(b"x")
    info = build_sht_info(sht, xtal_dir=tmp_path, cif_dir=tmp_path)
    assert info["provenance"]["source_xtal"]["name"] == "Ni.xtal"
    assert info["provenance"]["source_xtal"]["found"] is True
    assert info["parameters"]["dmin"] == 0.07
