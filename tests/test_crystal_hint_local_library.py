"""Unit tests for crystal_hint_local_library."""
from __future__ import annotations

from pathlib import Path

import pytest

# Needs the maintainer's crystal library / measurement data, which a clone
# does not have — skip with the missing path named, never fail.
from tests.data_deps import CIF_LIBRARY, requires

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from backend.api.services.crystal_hint_local_library import (
    _system_from_sg_name,
    _system_from_sg_number,
    LocalEntry,
    rebuild_index,
    search,
)

_SHT_DIR = PROJECT_ROOT / "Database" / "EBSD_SHT_Database"


def _require_sht_library():
    """Skip when the SHT library is absent, instead of passing vacuously.

    ``Database/EBSD_SHT_Database`` is gitignored and rebuilt by the Simulation
    tab, so on a fresh clone it is empty. A test that compares the index
    against the disk then finds nothing on either side and passes while
    verifying nothing — the "an .sht named [hP6] must still bind to its cubic
    CIF" bug would go uncaught and the suite would say green. Same rule as
    the data guards merged in d368bee9: when the data is missing, say so.
    """
    if not _SHT_DIR.is_dir() or not any(_SHT_DIR.rglob("*.sht")):
        pytest.skip(f"needs simulated masters in {_SHT_DIR.name} — "
                    "run the Simulation tab, or this proves nothing")


def _sht_file_for(key):
    """The .sht file the library would match to ``key``, or None.

    An APPROXIMATION of the production rule, valid for the ``sd_*`` keys this
    file uses. Production (``crystal_hint_local_library.build_index``) walks
    SHT -> entries longest-key-first; this walks key -> first SHT containing
    it. The two disagree for a key that is a substring of another ("Al" is in
    "Al13Fe4"), so do not reuse it for short keys. It cannot hide the failure
    that matters: if matching broke, this finds the file, the index does not,
    and the assertion fires.
    """
    if not _SHT_DIR.is_dir():
        return None
    return next((p for p in sorted(_SHT_DIR.rglob("*.sht")) if key in p.stem), None)


def test_a_repair_backup_is_not_offered_as_a_phase(tmp_path, monkeypatch):
    """`<name>.P1-backup-<date>.cif` is a backup, not a phase.

    Found via test_curated_phases_have_sht: three such files from 2026-09-13
    sat in the index, so the phase list offered the UN-symmetrised P1 copy of
    a phase the user already had — the shape that makes a Hough reflector
    library explode (see the 2026-08-03 note on non-standard settings). The
    file must stay on disk; it must not be a phase.
    """
    from backend.api.services import crystal_hint_local_library as lib

    cif_dir = tmp_path / "CIF_Library"
    cif_dir.mkdir()
    real = cif_dir / "Al7FeCu2.cif"
    backup = cif_dir / "Al7FeCu2.P1-backup-2026-09-13.cif"
    body = (PROJECT_ROOT / "Database" / "CIF_Library" / "Al.cif")
    text = body.read_text(encoding="utf-8", errors="replace") if body.is_file() else \
        "data_x\n_cell_length_a 4.05\n_symmetry_space_group_name_H-M 'Fm-3m'\n"
    real.write_text(text, encoding="utf-8")
    backup.write_text(text, encoding="utf-8")

    monkeypatch.setattr(lib, "CIF_DIR", cif_dir)
    monkeypatch.setattr(lib, "XTAL_DIR", tmp_path / "absent_xtal")
    monkeypatch.setattr(lib, "SHT_DIR", tmp_path / "absent_sht")
    keys = set(lib.build_index())

    assert "Al7FeCu2" in keys
    assert not [k for k in keys if "backup" in k.lower()], keys
    assert backup.is_file(), "the backup file itself must not be touched"


def test_system_from_sg_number():
    assert _system_from_sg_number(225) == "cubic"   # Fm-3m
    assert _system_from_sg_number(229) == "cubic"   # Im-3m
    assert _system_from_sg_number(204) == "cubic"   # Im-3
    assert _system_from_sg_number(200) == "cubic"   # Pm-3
    assert _system_from_sg_number(194) == "hexagonal"  # P6_3/mmc
    assert _system_from_sg_number(166) == "trigonal"   # R-3m
    assert _system_from_sg_number(139) == "tetragonal"
    assert _system_from_sg_number(63) == "orthorhombic"
    assert _system_from_sg_number(14) == "monoclinic"
    assert _system_from_sg_number(2) == "triclinic"


def test_system_from_sg_name_cubic():
    assert _system_from_sg_name("Fm-3m") == "cubic"
    assert _system_from_sg_name("Im-3") == "cubic"
    assert _system_from_sg_name("Pm-3") == "cubic"
    assert _system_from_sg_name("Fd-3m") == "cubic"


@requires(CIF_LIBRARY)
def test_index_builds_without_crashing():
    entries = rebuild_index()
    assert len(entries) > 0, "Index should find some CIF/XTAL files"


@requires(CIF_LIBRARY)
def test_index_has_known_phases():
    entries = rebuild_index()
    # Al cF4 should be present
    al = entries.get("Al")
    assert al is not None
    assert al.crystal_system == "cubic"
    assert "Al" in al.elements
    assert 3.9 < al.lattice_a_A < 4.2
    assert al.sht_path is not None  # has matching SHT


@requires(CIF_LIBRARY)
def test_index_alpha_alfesi_correctly_cubic():
    """The cubic alpha-Al(Fe,Mn)Si phases (sd_0302719, sd_1401510) should
    show crystal_system='cubic' despite being labelled [hP6] in SHT filenames.
    This is the key bug uncovered in the FINDINGS_v2 investigation."""
    _require_sht_library()
    entries = rebuild_index()
    for key in ("sd_0302719", "sd_1401510"):
        entry = entries.get(key)
        assert entry is not None, f"{key} not in index"
        assert entry.crystal_system == "cubic", (
            f"{key} should be cubic (from CIF), not {entry.crystal_system}"
        )
        # And where such a file EXISTS it must be matched, despite the
        # filename mislabel — that is the bug this test was written for.
        #
        # Which phases are simulated is machine state, not a contract: the
        # SHT library is gitignored and rebuilt by the Simulation tab.
        # sd_1401510's master WAS made — its folder still holds
        # "...(sd_1401510)...sht.provenance.json" (2026-06-26, engine "ours",
        # dmin 0.09, bw 384) with no .sht beside it, while sd_0302719 has
        # both — but it is not in this working copy, and when it went is not
        # established. Demanding it here made the test assert the
        # maintainer's simulation queue. So compare the index against the
        # disk instead, in both directions.
        on_disk = _sht_file_for(key)
        if on_disk is None:
            assert entry.sht_path is None, (
                f"{key}: index claims {entry.sht_path}, which is not in "
                f"{_SHT_DIR}")
        else:
            assert entry.sht_path is not None, (
                f"{key} should match its SHT {on_disk.name}")


def test_curated_phases_have_sht():
    """Every CURATED CIF/XTAL phase should have a matching SHT file.

    Phases the user downloads from COD/MP via the Crystal Hint "Download
    CIF" button land in Database/CIF_Library/ with a `<formula>_MP-<id>` /
    `<formula>_COD-<id>` key and legitimately have NO SHT yet — the SHT is
    generated later in the Simulation tab. Those are excluded here; only
    the curated base library is required to be SHT-complete.
    """
    import re
    entries = rebuild_index()
    downloaded = re.compile(r"_(MP|COD)-", re.IGNORECASE)
    missing = [k for k, e in entries.items()
               if e.sht_path is None and not downloaded.search(k)]
    assert len(missing) == 0, f"Curated phases without SHT: {missing}"


def test_search_filters_by_chemistry():
    """Sample with only Al should NOT return phases with Ni."""
    matches = search(elements=["Al"], strict_chemistry=True)
    for m in matches:
        if m.entry.elements:
            assert all(e in {"Al"} for e in m.entry.elements), (
                f"{m.entry.key} contains foreign element"
            )


@requires(CIF_LIBRARY)
def test_search_empty_elements_does_not_drop_everything():
    """Bug fix from code review — empty `elements` + strict_chemistry=True
    used to silently return no matches. Now: skip the chemistry filter and
    score down by 30% so it's clear chemistry wasn't checked."""
    matches = search(elements=[], strict_chemistry=True)
    assert len(matches) >= 5, (
        f"Empty chemistry should NOT drop all matches (got {len(matches)})"
    )
    # Score should be ≤ 0.7 (not 1.0) when chemistry isn't validated
    for m in matches[:5]:
        assert m.plausibility <= 0.71, (
            f"{m.entry.key}: empty-chemistry plausibility {m.plausibility} too high"
        )


def test_search_excludes_ni_for_al_sample():
    """Al-Si-Cu-Fe sample (AA226 chemistry) should NOT match Ni cF4."""
    matches = search(
        elements=["Al", "Si", "Cu", "Fe", "Mg", "Zn", "Mn"],
        strict_chemistry=True,
    )
    ni_matches = [m for m in matches if "Ni" in m.entry.elements]
    assert len(ni_matches) == 0, "Ni should be filtered out for Al-alloy chemistry"


@requires(CIF_LIBRARY)
def test_search_lattice_range_filter():
    matches = search(
        elements=["Al", "Si", "Cu", "Fe", "Mg"],
        a_range_A=(3.5, 5.0),
        strict_chemistry=True,
    )
    # Al (4.05) and Si (5.43 — outside) — Al should be top, Si lattice-distance > 0
    assert len(matches) > 0
    al_entry = next(m for m in matches if m.entry.key == "Al")
    assert al_entry.lattice_distance == 0.0
    assert al_entry.score == pytest.approx(1.0)


def test_search_system_filter():
    matches = search(
        elements=["Al"],
        crystal_system="cubic",
        strict_chemistry=True,
    )
    for m in matches:
        # Phase should EITHER be cubic OR have system_match=False
        if m.system_match:
            assert m.entry.crystal_system == "cubic"


def test_search_ranks_correctly():
    """Best score first."""
    matches = search(
        elements=["Al", "Si", "Cu", "Fe", "Mg"],
        crystal_system="cubic",
        a_range_A=(3.5, 6.0),
    )
    if len(matches) >= 2:
        assert matches[0].score >= matches[1].score


def test_sd_1816951_is_offered_with_the_reason_it_could_not_be_fully_read():
    """A DECISION REVERSED on 2026-09-27, so the reasoning on both sides stays.

    The file is MgCu2, a cubic Laves phase: its own atom-site table says Cu 16c
    + Mg 8b (Cu 66.7 / Mg 33.3), and sd_1816951.xtal — what EMsoft simulated the
    master from — says the same. pymatgen returns TWO structures, Mg4Cu (40
    sites) and Mg2Cu (24), because the Fd-3m origin choice is mis-expanded, and
    since 2026-09-12 `cif_phase_library.one_structure` refuses such a file rather
    than letting block order decide a composition. That much is unchanged.

    WHAT THIS TEST USED TO ASSERT was that Crystal Hint therefore offers the
    phase to nobody — "a real capability loss, and the honest one". The measured
    cost of that turned out to be higher than the reasoning allowed for:

        search(['Mg','Cu'], strict_chemistry=True)   ->  0 results

    MgCu2 is the library's ONLY Mg-Cu phase, so the phase that *is* the answer
    was the only one missing, and the user saw an empty result rather than a
    caveat. Meanwhile the entry already carried everything Crystal Hint actually
    ranks on — elements (from the .xtal, which reads fine), space group, IT
    number, lattice and formula from the `_sm_*` tags, plus a simulated .sht.

    The distinction the old reasoning missed is WHICH consumer needs the
    structure. Candidate suggestion needs chemistry, symmetry and cell: all
    readable. Hough reflectors computed FROM THIS CIF need the site expansion:
    not readable. So the phase is offered and carries `parse_error` through to
    `CandidateOut.structure_unreadable`, which is what lets a consumer that does
    need the structure refuse it. Offering it silently would be the bad outcome;
    so is hiding it.
    """
    matches = search(elements=["Mg", "Cu"], strict_chemistry=True)
    hits = [m for m in matches if "1816951" in m.entry.key]
    assert hits, "the library's only Mg-Cu phase must not be the only one missing"
    assert hits[0].entry.parse_error, (
        "offered WITHOUT the caveat is worse than not offered at all")

    # An Al-only sample must still not be offered an Mg-Cu phase: being visible
    # is not the same as being unfiltered.
    assert not [m for m in search(elements=["Al"], strict_chemistry=True)
                if "1816951" in m.entry.key]

    cif = PROJECT_ROOT / "Database" / "CIF_Library" / "sd_1816951.cif"
    if not cif.is_file():
        pytest.skip("sd_1816951.cif is not on this machine")
    from backend.api.services.crystal_hint_local_library import _parse_cif
    reason = _parse_cif(cif).get("error", "")
    assert "Mg2Cu" in reason and "Mg4Cu" in reason, reason


def test_search_strict_vs_loose_chemistry():
    """Loose chemistry should return MORE matches than strict."""
    strict_matches = search(elements=["Al"], strict_chemistry=True)
    loose_matches = search(elements=["Al"], strict_chemistry=False)
    assert len(loose_matches) >= len(strict_matches)


def test_prettify_formula_keys_no_collision_in_library():
    """For our current 22-CIF library, the (space_group, elements) keys
    in _PHASE_NICKNAMES must NOT match two different entries — that
    would mean two distinct phases collapse to the same nickname.
    """
    from backend.api.services.crystal_hint_local_library import (
        _PHASE_NICKNAMES, rebuild_index,
    )
    entries = rebuild_index()
    seen: dict[tuple, str] = {}
    for e in entries.values():
        key = (e.space_group.replace(" ", ""), tuple(sorted(e.elements)))
        if key in _PHASE_NICKNAMES:
            # Two different library entries hitting the same nickname is a
            # collision — would mislead the user that they're the same phase.
            prev = seen.get(key)
            if prev and prev != e.key:
                pytest.fail(
                    f"Nickname collision: '{_PHASE_NICKNAMES[key]}' matches "
                    f"both library entries '{prev}' and '{e.key}'"
                )
            seen[key] = e.key


def test_get_index_auto_rebuilds_when_library_changes(monkeypatch):
    """get_index() must rebuild when the .sht/.xtal/.cif library changes on
    disk (e.g. after a re-simulation), so the phase test never serves a stale
    master without a backend restart."""
    import backend.api.services.crystal_hint_local_library as L
    calls = {"n": 0}
    def fake_build():
        calls["n"] += 1
        return {"e": "x"}
    monkeypatch.setattr(L, "build_index", fake_build)
    L._INDEX_CACHE = None
    L._INDEX_SIG = None
    try:
        monkeypatch.setattr(L, "_library_signature", lambda: (5, 1000))
        L.get_index(); L.get_index()
        assert calls["n"] == 1, "stable signature must NOT rebuild"
        monkeypatch.setattr(L, "_library_signature", lambda: (5, 2000))  # a file changed
        L.get_index()
        assert calls["n"] == 2, "changed signature must rebuild"
    finally:
        L._INDEX_CACHE = None
        L._INDEX_SIG = None
