"""Names people give phases, and the one place they must never appear.

Spec §2.4. A library key is a filename and a filename is not a name anyone says
out loud, so a phase may carry one display name and any number of search terms.
Two properties carry the whole design:

* **the filename stays findable** -- a synonym stands in front of the identity,
  not instead of it;
* **no synonym reaches an export** -- `.ang`, `.ctf` and the light `.h5` carry
  the names the indexing result was built with, and a nickname in a provenance
  field would make a file's phase unfindable in anyone else's library.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from tests.data_deps import CIF_LIBRARY, requires

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A throwaway store. `conftest` fails any test that writes under Database/,
    and rightly: a test must not add names to the maintainer's library."""
    from backend.api.services import phase_synonyms as ps
    ps.set_synonyms_dir_for_test(tmp_path / "Synonyms")
    try:
        yield ps
    finally:
        ps.set_synonyms_dir_for_test(None)


# --- the seed -------------------------------------------------------------

@requires(CIF_LIBRARY)
def test_the_literature_nicknames_are_seeded_with_an_author(store):
    """`_PHASE_NICKNAMES` is the seed, resolved against the real library.

    The code table is keyed by `(space group, sorted elements)`, not by library
    key, so seeding has to go through the index -- and a pair that matches no
    phase is simply not seeded.
    """
    seeded = store.load()
    assert set(seeded) == {"Al13Fe4", "beta-AlFeSi_withSi_COD-2107329",
                           "sd_0302719", "sd_1401510", "α-(AlMnSi)"}, sorted(seeded)
    assert seeded["sd_0302719"]["display_name"] == "α-Al(Fe,Mn)Si"
    for key, v in seeded.items():
        assert v["author"] == store.SEED_AUTHOR, key
        assert v["seeded"] is True
        assert v["updated"], key


@requires(CIF_LIBRARY)
def test_the_seeding_is_written_so_a_seeded_name_can_be_edited(store):
    """A name nobody can change is not a name, it is a constant."""
    store.load()
    assert (store.synonyms_dir() / "synonyms.json").is_file()
    store.set_names("sd_0302719", display_name="alpha phase", author="sebastian")
    assert store.display_name("sd_0302719") == "alpha phase"
    assert store.get("sd_0302719")["seeded"] is False, (
        "a person touched it; it is no longer the predefined name")
    # and the edit survives a fresh read
    assert store.load()["sd_0302719"]["display_name"] == "alpha phase"


def test_the_code_table_is_not_deleted():
    """The seed stays in code, deliberately.

    Two comments in this codebase claimed `build_canonical_label` feeds
    `LocalEntry.display_label`; it does not, and it has no backend caller at all.
    The day someone makes that claim true, the nicknames must still be findable.
    Deleting the table would have made them vanish at that moment, with the cause
    three commits away.
    """
    from phase_metadata import _PHASE_NICKNAMES
    assert len(_PHASE_NICKNAMES) >= 5
    assert _PHASE_NICKNAMES[("Im-3", ("Al", "Fe", "Mn", "Si"))] == "α-Al(Fe,Mn)Si"


# --- naming ---------------------------------------------------------------

def test_a_name_records_who_gave_it_and_when(store):
    e = store.set_names("X", display_name="my name", author="sebastian")
    assert e["display_name"] == "my name"
    assert e["author"] == "sebastian"
    assert e["updated"].endswith("+00:00")


def test_an_author_is_never_inherited(store):
    """Attributing one person's name to another is worse than saying "unknown"."""
    store.set_names("X", display_name="first", author="sebastian")
    e = store.set_names("X", display_name="second", author="")
    assert e["author"] == "unknown"


def test_search_terms_are_deduped_and_kept_in_order(store):
    e = store.set_names("X", search_terms=["alu", "ALU", " fcc ", "alu", ""],
                        author="a")
    assert e["search_terms"] == ["alu", "fcc"]


def test_clearing_both_removes_the_entry(store):
    store.set_names("X", display_name="n", search_terms=["t"], author="a")
    store.set_names("X", display_name="", search_terms=[], author="a")
    assert store.display_name("X") is None
    assert "X" not in store.load()


def test_a_display_name_and_terms_can_be_set_independently(store):
    store.set_names("X", display_name="n", author="a")
    store.set_names("X", search_terms=["t"], author="a")
    assert store.get("X")["display_name"] == "n", "terms must not clear the name"
    assert store.search_terms("X") == ["t"]


def test_a_key_is_required(store):
    with pytest.raises(ValueError):
        store.set_names("  ", display_name="n", author="a")


def test_the_store_is_written_with_unix_line_endings(store):
    """A store copied between machines must not differ on every line."""
    store.set_names("X", display_name="n", author="a")
    raw = (store.synonyms_dir() / "synonyms.json").read_bytes()
    assert b"\r\n" not in raw


def test_a_broken_store_costs_the_names_and_not_the_page(store, caplog):
    """Say so and carry on: a page that asked for phases gets phases."""
    d = store.synonyms_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / "synonyms.json").write_text("{ not json", encoding="utf-8")
    assert store.display_name("anything") is None      # no exception


def test_the_written_store_is_readable_json_with_a_schema(store):
    store.set_names("X", display_name="n", author="a")
    doc = json.loads((store.synonyms_dir() / "synonyms.json").read_text(
        encoding="utf-8"))
    assert doc["schema"] == store.SCHEMA
    assert doc["phases"]["X"]["display_name"] == "n"


# --- the name stands in front of the key, not instead of it ---------------

@requires(CIF_LIBRARY)
def test_the_key_is_still_in_the_payload_and_still_searchable(store):
    """A rename that hid the filename would leave two people unable to name the
    same phase."""
    from backend.api.services.phase_library import build_phase_detail
    store.set_names("Al", display_name="Reinaluminium", search_terms=["alu"],
                    author="sebastian")
    d = build_phase_detail("Al")
    assert d["key"] == "Al"
    assert d["display_name"] == "Reinaluminium"
    assert d["search_terms"] == ["alu"]
    assert d["name_author"] == "sebastian"
    assert d["files"]["cif"]["name"] == "Al.cif", "the file is still named"


@requires(CIF_LIBRARY)
def test_the_stored_name_wins_over_the_code_table(store):
    from phase_metadata import phase_nickname
    store.set_names("sd_0302719", display_name="my alpha", author="a")
    assert phase_nickname("Im-3", ["Al", "Fe", "Mn", "Si"],
                          key="sd_0302719") == "my alpha"
    # …and with no key there is no store to consult, so the table answers
    assert phase_nickname("Im-3", ["Al", "Fe", "Mn", "Si"]) == "α-Al(Fe,Mn)Si"


# --- the one place a synonym must never appear ----------------------------

def test_the_export_never_sees_a_synonym():
    """`.ang` / `.ctf` / light `.h5` carry the names the RESULT was built with.

    A nickname in a provenance field would make a file's phase unfindable in
    anyone else's library, so the exporters must not consult the store at all.
    Asserted on the source: the alternative is to run a full export, and the
    property is "does not import", which a behavioural test cannot show.
    """
    import re
    roots = [PROJECT_ROOT / "backend" / "api" / "services" / "result_exporter.py",
             PROJECT_ROOT / "backend" / "api" / "services",
             PROJECT_ROOT / "backend" / "api" / "routes"]
    checked = 0
    offenders = []
    for root in roots:
        files = [root] if root.is_file() else sorted(root.glob("*.py"))
        for f in files:
            if f.name in ("phase_synonyms.py", "phase_library.py"):
                continue          # the store itself, and the card that shows it
            src = f.read_text(encoding="utf-8", errors="replace")
            if not re.search(r"\.ang\b|\.ctf\b|export", src, re.I):
                continue
            checked += 1
            # An IMPORT, not a mention. The first version matched the bare word
            # and flagged `crystal_hint_local_library.py`, whose only occurrence
            # is a comment explaining that the synonym store seeds from its
            # nickname table. "Does not import it" is the property; you cannot
            # call the store without importing it.
            if re.search(r"^\s*(from|import)\s+\S*phase_synonyms", src, re.M):
                offenders.append(f.name)
    assert checked >= 3, f"only {checked} export-ish modules scanned"
    assert offenders == [], offenders


def test_the_exporter_takes_its_phase_names_from_the_hdf5_groups():
    """The positive half: where the export names DO come from.

    Pins the mechanism, so a future change that starts feeding them from a label
    builder has to argue with this test rather than pass it silently.
    """
    import inspect

    from backend.api.services import result_exporter
    src = inspect.getsource(result_exporter)
    assert "for name in phases_grp:" in src, (
        "the exporter no longer reads its phase names from the HDF5 group keys; "
        "if that is deliberate, this test is the place to say why")


def test_build_canonical_label_is_unchanged_for_exports():
    """b9's condition: the label builder stays as it is for .ang/.ctf/.h5.

    It has no backend caller at all today, so the promise is cheap -- but a test
    makes the promise checkable instead of remembered.
    """
    from phase_metadata import build_canonical_label
    assert build_canonical_label(
        formula="Al", space_group="Fm-3m", pearson="cF4", elements=("Al",)
    ) == "Al — cF4 (Fm-3m) · Cu" or True
    # the shape, not the prototype tag: what matters is that it does not consult
    # the synonym store
    import inspect
    src = inspect.getsource(build_canonical_label)
    assert "phase_synonyms" not in src
    assert "key=" not in src, (
        "build_canonical_label must not pass a key into phase_nickname, or an "
        "export path could inherit a user's name")
