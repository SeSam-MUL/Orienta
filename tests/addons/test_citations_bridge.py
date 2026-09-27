from pathlib import Path

import pytest

from backend.api.services.addons import citations_bridge as bridge
from backend.api.services.addons.citations_bridge import (
    AnalysisKeyConflict,
    addon_library_entries,
    register_manifest_citations,
)
from backend.api.services.addons.manifest import parse_manifest
from backend.api.services.citations import render as render_mod
from backend.api.services.citations import steps as steps_mod
from backend.api.services.citations.render import load_library
from backend.api.services.citations.steps import get_step

FIXTURE = Path(__file__).parent / "fixtures" / "valid_addon.toml"
TWO_ANALYSES = Path(__file__).parent / "fixtures" / "two_analyses_addon.toml"

DOI_ID = "doi:10.5281/zenodo.1234567"


def test_registers_one_step_per_analysis():
    """Per ANALYSIS, measured on a manifest that has more than one.

    Against the single-analysis fixture this assertion is vacuous: an
    implementation that looped over ``manifest.analyses[:1]`` keeps it green.
    """
    m = parse_manifest(TWO_ANALYSES)
    assert len(m.analyses) == 2
    register_manifest_citations(m)
    first = get_step("addon.first_analysis")
    second = get_step("addon.second_analysis")
    assert first is not None and second is not None
    assert first.label == "First analysis"
    assert second.label == "Second analysis"
    assert "{method}" in first.sentence
    assert "{n_grains}" in second.sentence


def test_registers_the_step_of_the_single_analysis_fixture():
    m = parse_manifest(FIXTURE)
    register_manifest_citations(m)
    step = get_step("addon.grain_shape_stats")
    assert step is not None
    assert step.label == "Grain shape statistics"
    assert "{method}" in step.sentence


def test_entries_are_built_from_the_doi_references():
    entries = addon_library_entries(parse_manifest(FIXTURE))
    assert DOI_ID in entries
    e = entries[DOI_ID]
    assert e["DOI"] == "10.5281/zenodo.1234567"
    assert e["id"] == DOI_ID


def test_a_doi_entry_does_not_claim_a_structure_it_never_fetched():
    """render.py's house rule: an unlisted CSL type falls back to @misc,
    "the honest choice: it renders without claiming a structure we do not
    know." Nothing is resolved at run time, so a DOI could be a paper, a
    Zenodo deposit or a data set — "article-journal" would render @article
    and assert a journal that may not exist."""
    from backend.api.services.citations.render import _BIBTEX_TYPE, render_bibtex

    entry = addon_library_entries(parse_manifest(FIXTURE))[DOI_ID]
    assert entry["type"] not in _BIBTEX_TYPE or _BIBTEX_TYPE[entry["type"]] == "misc"
    assert render_bibtex([entry]).startswith("@misc{")


def test_a_placeholder_title_names_no_addon():
    """One DOI is one library entry, shared by every add-on that cites it.
    Naming the add-on that happened to register it first would put that name
    on another author's work, in text a researcher may paste into a paper."""
    m = parse_manifest(FIXTURE)
    entry = addon_library_entries(m)[DOI_ID]
    assert m.name not in entry["title"]
    assert entry["DOI"] in entry["title"]


def test_an_unresolvable_reference_is_shown_not_dropped(tmp_path):
    src = FIXTURE.read_text(encoding="utf-8").replace(
        'citations = ["doi:10.5281/zenodo.1234567"]',
        'citations = ["not-a-reference"]')
    p = tmp_path / "x.toml"
    p.write_text(src, encoding="utf-8")
    entries = addon_library_entries(parse_manifest(p))
    assert "not-a-reference" in entries
    assert "unresolved" in entries["not-a-reference"]["title"].lower()


def test_an_unresolvable_reference_names_the_addon_that_declared_it():
    """Unlike the DOI branch: there is no id, DOI or URL that reaches the
    work, so who declared the broken reference is the only actionable thing
    the line can carry."""
    m = parse_manifest(FIXTURE)
    assert m.name in bridge._entry_for("not-a-reference", m)["title"]


def test_a_reference_the_core_library_already_resolves_is_left_alone(tmp_path):
    """`citations = ["orienta"]` means Orienta's own entry, not a second copy
    of it under a new id — which is how an Orienta-authored add-on cites
    honestly without minting a DOI it does not have."""
    src = FIXTURE.read_text(encoding="utf-8").replace(
        'citations = ["doi:10.5281/zenodo.1234567"]', 'citations = ["orienta"]')
    p = tmp_path / "core.toml"
    p.write_text(src, encoding="utf-8")
    entries = addon_library_entries(parse_manifest(p))
    assert "orienta" not in entries


def test_the_addons_own_doi_is_registered_even_without_a_citations_list(tmp_path):
    """Resolution, not a substring.

    The step's id has to be the LIBRARY KEY, prefix included: both readers
    filter with ``cid in library``, so a bare ``10.5281/...`` beside an entry
    keyed ``doi:10.5281/...`` deletes the add-on's own work from BibTeX, from
    the plain list and from the exported .h5 — silently, because a
    filtered-out id leaves no trace anywhere in the output.
    """
    src = FIXTURE.read_text(encoding="utf-8").replace(
        'citations = ["doi:10.5281/zenodo.1234567"]', "citations = []")
    p = tmp_path / "y.toml"
    p.write_text(src, encoding="utf-8")
    m = parse_manifest(p)
    assert any(m.doi in k for k in addon_library_entries(m))

    register_manifest_citations(m)
    step = get_step("addon.grain_shape_stats")
    assert step.citation_ids, "the add-on's own DOI must be cited"
    assert set(step.citation_ids) <= set(load_library())


def test_an_addon_with_neither_doi_nor_citations_states_the_reason(tmp_path):
    src = (FIXTURE.read_text(encoding="utf-8")
           .replace('citations = ["doi:10.5281/zenodo.1234567"]', "citations = []")
           .replace('doi = "10.5281/zenodo.1234567"', 'doi = ""'))
    p = tmp_path / "z.toml"
    p.write_text(src, encoding="utf-8")
    register_manifest_citations(parse_manifest(p))
    step = get_step("addon.grain_shape_stats")
    assert step.citation_ids == ()
    assert step.no_citation_reason


def test_the_entries_reach_the_library_the_panel_actually_reads():
    """The whole point. routes/citations.py and result_exporter both filter
    with `cid in library`, and `library` only ever comes from load_library()."""
    m = parse_manifest(FIXTURE)
    assert DOI_ID not in load_library()
    register_manifest_citations(m)
    library = load_library()
    assert DOI_ID in library
    assert library[DOI_ID]["DOI"] == "10.5281/zenodo.1234567"
    # ...and the core entries are still there.
    assert "orienta" in library and "kikuchipy" in library


def test_an_explicit_library_path_still_merges_nothing():
    """tests/citations/test_render.py pins that a fixture library is exactly
    what its file says. Add-on entries join the no-path branch only."""
    fixture_lib = (Path(__file__).resolve().parents[1] / "citations"
                   / "fixtures" / "library_fixture.json")
    assert fixture_lib.is_file(), fixture_lib
    register_manifest_citations(parse_manifest(FIXTURE))
    assert DOI_ID not in load_library(fixture_lib)


def test_a_second_addon_sharing_a_doi_is_still_told_that_id(tmp_path):
    """``known`` must be the CORE bibliography, not everything load_library()
    currently returns.

    load_library() merges ADDON_LIBRARY_ENTRIES, so after add-on A registers,
    an unsubtracted reading tells add-on B that the CORE resolves the shared
    DOI — and B is told it contributed nothing on its FIRST call. The registry
    is unaffected either way (setdefault); the returned ids are not, and that
    is what a caller reads as "this add-on's works".
    """
    register_manifest_citations(parse_manifest(FIXTURE))
    src = (FIXTURE.read_text(encoding="utf-8")
           .replace('name = "grain-shape-stats"', 'name = "other-addon"')
           .replace('key = "addon.grain_shape_stats"', 'key = "addon.other"'))
    p = tmp_path / "b.toml"
    p.write_text(src, encoding="utf-8")
    second = parse_manifest(p)
    assert second.name == "other-addon"
    assert DOI_ID in register_manifest_citations(second)


def test_registering_twice_leaves_the_first_declaration_standing():
    """Measured on the module attribute, not on a name imported at load time.

    `register_step` mutates `steps.STEP_REGISTRY` in place. A test that does
    `from ...steps import STEP_REGISTRY` and then monkeypatches the module
    attribute is measuring a dict the patched function never touches — which
    is why the previous version of this test passed with the idempotence
    guard deleted AND with random keys registered.

    Counting is not enough either: `register_step` assigns BY KEY, so a
    deleted guard overwrites in place and neither length moves. The guard is
    only observable in what SURVIVES a second manifest that declares the same
    key differently.
    """
    m = parse_manifest(FIXTURE)
    register_manifest_citations(m)
    before_steps = len(steps_mod.STEP_REGISTRY)
    before_entries = len(render_mod.ADDON_LIBRARY_ENTRIES)
    register_manifest_citations(m)
    assert len(steps_mod.STEP_REGISTRY) == before_steps
    assert len(render_mod.ADDON_LIBRARY_ENTRIES) == before_entries
    assert get_step("addon.grain_shape_stats").label == "Grain shape statistics"


def test_a_later_manifest_cannot_redeclare_a_key_already_registered(tmp_path):
    src = FIXTURE.read_text(encoding="utf-8").replace(
        'label = "Grain shape statistics"', 'label = "Something else entirely"')
    p = tmp_path / "clash.toml"
    p.write_text(src, encoding="utf-8")
    clashing = parse_manifest(p)
    assert clashing.analyses[0].label == "Something else entirely"

    register_manifest_citations(parse_manifest(FIXTURE))
    register_manifest_citations(clashing)
    assert get_step("addon.grain_shape_stats").label == "Grain shape statistics"


def test_register_library_entries_tolerates_nothing_to_register():
    before = dict(render_mod.ADDON_LIBRARY_ENTRIES)
    render_mod.register_library_entries(None)
    render_mod.register_library_entries({})
    assert render_mod.ADDON_LIBRARY_ENTRIES == before


def test_a_bibliography_that_cannot_be_read_costs_the_check_not_the_run(
        monkeypatch):
    """The documented degradation: without the core's ids, every reference is
    minted a placeholder — including one the core would have resolved. The run
    continues, and the failure is logged rather than raised."""
    def boom(*_args, **_kwargs):
        raise OSError("library.json is not readable")

    monkeypatch.setattr(bridge, "load_library", boom)
    assert bridge._core_library_ids() == set()
    entries = addon_library_entries(parse_manifest(FIXTURE))
    assert DOI_ID in entries


def test_the_registries_are_restored_between_tests():
    """The isolation fixture in conftest.py is load-bearing:
    tests/citations/test_steps.py pins len(STEP_REGISTRY) == 9 exactly, and
    tests/addons/ sorts first."""
    assert "addon.grain_shape_stats" not in steps_mod.STEP_REGISTRY
    assert "addon.first_analysis" not in steps_mod.STEP_REGISTRY
    assert render_mod.ADDON_LIBRARY_ENTRIES == {}


# --- one analysis key, one add-on -------------------------------------------

BOBS_DOI = "doi:10.5281/zenodo.9999999"


def _bobs_manifest(tmp_path, *, key="addon.grain_shape_stats"):
    """A second add-on that differs from the fixture in EVERY field but the
    analysis key -- the shape a fork of the shipped example arrives in."""
    src = (FIXTURE.read_text(encoding="utf-8")
           .replace('name = "grain-shape-stats"', 'name = "bobs-grain-stats"')
           .replace('doi = "10.5281/zenodo.1234567"',
                    'doi = "10.5281/zenodo.9999999"')
           .replace('citations = ["doi:10.5281/zenodo.1234567"]',
                    'citations = ["doi:10.5281/zenodo.9999999"]')
           .replace('label = "Grain shape statistics"', 'label = "Bob label"')
           .replace('sentence = "Grain shape statistics were computed with '
                    '{method} on {n_grains} grains."',
                    'sentence = "Bob did it with {method}."')
           .replace('key = "addon.grain_shape_stats"', f'key = "{key}"'))
    p = tmp_path / "bob.toml"
    p.write_text(src, encoding="utf-8")
    return parse_manifest(p)


def test_a_declaration_records_which_addon_owns_it():
    """Without an owner on the declaration, "is this key already taken?" and
    "is it taken by SOMEONE ELSE?" are the same question, and only the second
    one is the one that matters."""
    m = parse_manifest(FIXTURE)
    register_manifest_citations(m)
    assert get_step("addon.grain_shape_stats").owner == m.name


def test_a_core_step_carries_no_owner():
    """``owner is None`` means Orienta's own; nothing in the core registry
    may claim to belong to an add-on."""
    for step in steps_mod.STEP_REGISTRY.values():
        assert step.owner is None, step.key


def test_a_second_addon_cannot_inherit_the_first_addons_credit(tmp_path):
    """The defect this guard exists for, measured end to end.

    Bob's code runs and Bob's numbers come back, but the DECLARATION carries
    the citations and the methods sentence -- so a skip-if-declared left
    Alice's DOI and Alice's sentence attached to Bob's run, in the BibTeX, in
    the methods paragraph and in the exported .h5. Bob's own DOI sat in the
    library cited by nothing.
    """
    alice = parse_manifest(FIXTURE)
    register_manifest_citations(alice)
    bob = _bobs_manifest(tmp_path)

    with pytest.raises(AnalysisKeyConflict) as excinfo:
        register_manifest_citations(bob)

    message = str(excinfo.value)
    assert "addon.grain_shape_stats" in message
    assert alice.name in message and bob.name in message
    # Alice's declaration stands, untouched...
    step = get_step("addon.grain_shape_stats")
    assert step.owner == alice.name
    assert step.citation_ids == (DOI_ID,)
    # ...and the refused registration wrote NOTHING, so Bob's work does not
    # sit in the bibliography cited by nobody.
    assert BOBS_DOI not in render_mod.ADDON_LIBRARY_ENTRIES


def test_the_same_addon_registering_twice_is_still_idempotent(tmp_path):
    """The behaviour the refusal must not break: re-registering is how EVERY
    run after the first arrives here, and it is settled."""
    m = parse_manifest(FIXTURE)
    register_manifest_citations(m)
    register_manifest_citations(m)      # must not raise
    assert get_step("addon.grain_shape_stats").owner == m.name

    # ...including from a second copy of the manifest at another path: the
    # owner is the add-on NAME, not the file it was read from.
    same_name = tmp_path / "again.toml"
    same_name.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    register_manifest_citations(parse_manifest(same_name))


def test_only_the_shared_key_is_refused_not_the_whole_idea_of_a_second_addon(
        tmp_path):
    """A different key from a different add-on is ordinary and must pass."""
    register_manifest_citations(parse_manifest(FIXTURE))
    bob = _bobs_manifest(tmp_path, key="addon.bobs_own_key")
    assert BOBS_DOI in register_manifest_citations(bob)
    assert get_step("addon.bobs_own_key").owner == bob.name


def _installed(tmp_path, manifest_text, folder):
    d = tmp_path / folder
    d.mkdir()
    (d / "orienta-addon.toml").write_text(manifest_text, encoding="utf-8")
    return d


def test_a_conflict_with_an_addon_that_is_gone_says_so_and_says_restart(
        tmp_path, monkeypatch):
    """Renaming a folder is an ordinary thing a user does, and it locked them
    out for the life of the process with the remedy "remove one of the two
    add-ons" -- one of which no longer exists.

    The registries stay append-only (a declaration is what an already-recorded
    result's methods paragraph is rendered FROM, so clearing one would
    retroactively change what a finished run claims). What changes is the
    sentence: it names the real state and gives an action that works.
    """
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(tmp_path / "nothing-here"))
    register_manifest_citations(parse_manifest(FIXTURE))     # "grain-shape-stats"
    renamed = _bobs_manifest(tmp_path)
    with pytest.raises(AnalysisKeyConflict) as excinfo:
        register_manifest_citations(renamed)
    message = str(excinfo.value)
    assert "grain-shape-stats" in message
    assert "no longer installed" in message or "is not installed" in message
    assert "restart" in message.lower()


def test_a_conflict_with_an_addon_that_is_still_there_keeps_the_old_remedy(
        tmp_path, monkeypatch):
    """The other branch: both installed, so removing or renaming one is an
    action the user can actually take, and the sentence must say that instead."""
    _installed(tmp_path, FIXTURE.read_text(encoding="utf-8"), "alice")
    monkeypatch.setenv("ORIENTA_ADDON_DIRS", str(tmp_path))
    register_manifest_citations(parse_manifest(FIXTURE))
    with pytest.raises(AnalysisKeyConflict) as excinfo:
        register_manifest_citations(_bobs_manifest(tmp_path))
    message = str(excinfo.value)
    assert "grain-shape-stats" in message
    assert "no longer installed" not in message
    assert "restart" not in message.lower()
