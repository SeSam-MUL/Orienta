"""A CIF whose data_ blocks disagree must not be read as if they agreed.

`_parse_cif_field` returned the FIRST non-empty value it found, across every
`data_` block in the file. 15 of the 39 shipped CIFs (2026-09-27) declare more
than one block, and those 15 are exactly the SpringerMaterials files carrying
the `_sm_*` fields the phase library displays. So the reader was resolving a
possible ambiguity by position, and nothing checked whether there was one.

**The first version of this guard was wrong, and the real library said so.** It
compared every value in the file and reported 43 "contradictions" — none in an
`_sm_*` tag, almost all in `_cell_length_*`. They are not contradictions:
SpringerMaterials writes one crystal in three NAMED cell settings, and
`Al.cif`'s three values for `a` are 4.049 (standardized), 4.049(1) (published,
the same number with its standard uncertainty) and 2.8631 (Niggli-reduced =
4.049/sqrt(2), the fcc primitive edge). Blanking those would have cost 14 files
their lattice parameters. `test_a_second_cell_setting_is_not_a_contradiction`
and `test_the_library_keeps_every_lattice_parameter_it_shows_today` are that
lesson, pinned.

So values are compared only within a declared cell setting. Blocks that do not
declare one — `sm_global`, a single-block CIF, anything from another source —
are compared with each other, which is the case
`test_a_contradicting_tag_is_reported_as_unknown` covers and the case a CIF from
outside this library would hit.

The reader reports a contradiction as unknown and logs the file and the tag. It
does not raise: see `_parse_cif_field`'s docstring for why this differs from
`cif_phase_library.one_structure`, which does refuse.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

import phase_metadata
from phase_metadata import (
    _cif_block_role, _parse_cif_field, extract_metadata_from_cif,
)

LIBRARY = Path(__file__).resolve().parents[1] / "Database" / "CIF_Library"

# A SpringerMaterials file, shortened: identity block plus the same cell three
# times. Values copied from the real `Al.cif`.
_THREE_SETTINGS = """\
data_sm_global
_sm_pearson_symbol               'cF4'
_sm_phase_labels                 'Al'

data_sm_isp_SD1250064-standardized_unitcell
_symmetry_space_group_name_H-M   'Fm-3m'
_cell_length_a                   4.049

data_sm_isp_SD1250064-published_cell
_symmetry_space_group_name_H-M   'Fm-3m'
_cell_length_a                   4.049(1)

data_sm_isp_SD1250064-niggli_reduced_cell
_cell_length_a                   2.8631
"""

_TWO_BLOCKS = """\
data_block_one
_symmetry_space_group_name_H-M   'P m -3'
_symmetry_Int_Tables_number      200
_sm_phase_labels                 'Mn0.5Fe0.5Al5Si0.68'
_sm_pearson_symbol               'cI168'
_cell_length_a                   12.56

data_block_two
_symmetry_space_group_name_H-M   'P m -3'
_symmetry_Int_Tables_number      200
_sm_phase_labels                 'Mn0.5Fe0.5Al5Si0.68'
_sm_pearson_symbol               'cF16'
_cell_length_a                   12.56
"""

_AGREEING = _TWO_BLOCKS.replace("'cF16'", "'cI168'")


# --- the contradiction -----------------------------------------------------

def test_a_contradicting_tag_is_reported_as_unknown():
    """Two blocks, two different Pearson symbols: neither is reported."""
    assert _parse_cif_field(_TWO_BLOCKS, "_sm_pearson_symbol") == ""


def test_it_does_not_quietly_pick_the_first_block():
    """The old behaviour, named so a revert is unmistakable."""
    got = _parse_cif_field(_TWO_BLOCKS, "_sm_pearson_symbol")
    assert got != "cI168", (
        "the first block's value came back, so the contradiction was resolved "
        "by position again")
    assert got != "cF16"


def test_the_warning_names_the_file_and_the_tag(caplog):
    with caplog.at_level(logging.WARNING, logger="phase_metadata"):
        _parse_cif_field(_TWO_BLOCKS, "_sm_pearson_symbol",
                         source="synthetic_two_block.cif")
    assert caplog.records, "a contradiction has to be loud somewhere"
    msg = caplog.records[0].getMessage()
    assert "synthetic_two_block.cif" in msg
    assert "_sm_pearson_symbol" in msg
    # both values, so the reader of the log can see WHAT disagreed
    assert "cI168" in msg and "cF16" in msg


def test_an_unnamed_source_still_warns(caplog):
    """`source` is optional; its absence must not swallow the warning."""
    with caplog.at_level(logging.WARNING, logger="phase_metadata"):
        _parse_cif_field(_TWO_BLOCKS, "_sm_pearson_symbol")
    assert caplog.records
    assert "<unnamed>" in caplog.records[0].getMessage()


# --- everything that must NOT change --------------------------------------

def test_agreeing_blocks_are_unaffected():
    """The normal case for these files: several blocks, same value."""
    assert _parse_cif_field(_AGREEING, "_sm_pearson_symbol") == "cI168"
    assert _parse_cif_field(_AGREEING, "_symmetry_Int_Tables_number") == "200"


def test_a_repeated_identical_value_is_not_a_contradiction(caplog):
    with caplog.at_level(logging.WARNING, logger="phase_metadata"):
        assert _parse_cif_field(_AGREEING, "_sm_phase_labels") == \
            "Mn0.5Fe0.5Al5Si0.68"
    assert not caplog.records


def test_the_cif_null_is_still_skipped():
    """`?` is CIF for "not stated" and is not a competing value.

    The real case: sd_0302719 carries `_symmetry_Int_Tables_number` four times
    as 204, 204, 204, ?.
    """
    text = ("data_a\n_symmetry_Int_Tables_number 204\n"
            "data_b\n_symmetry_Int_Tables_number ?\n")
    assert _parse_cif_field(text, "_symmetry_Int_Tables_number") == "204"


def test_an_absent_tag_is_still_empty():
    assert _parse_cif_field(_TWO_BLOCKS, "_sm_phase_prototype") == ""


def test_a_contradicting_formula_falls_through_to_the_next_tag():
    """"Unknown" is a value every caller already handles.

    `extract_metadata_from_cif` has a fallback chain for the formula, so a
    contradicting `_sm_phase_labels` must not leave the formula empty — it must
    reach `_chemical_formula_sum`.
    """
    text = ("data_a\n_sm_phase_labels 'AlFe'\n_chemical_formula_sum 'Al Fe'\n"
            "data_b\n_sm_phase_labels 'AlMn'\n")
    assert _parse_cif_field(text, "_sm_phase_labels") == ""
    assert _parse_cif_field(text, "_chemical_formula_sum") == "Al Fe"


def test_it_reads_a_real_multi_block_library_cif(tmp_path):
    """End to end through the public function, on a real file.

    sd_0302719 has four blocks and is one of the 15 `_sm_*` files; it must still
    produce its full record.
    """
    cif = LIBRARY / "sd_0302719.cif"
    if not cif.is_file():
        pytest.skip("crystal library not in this checkout")
    meta = extract_metadata_from_cif(cif)
    assert meta.formula == "Mn0.5Fe0.5Al5Si0.68"
    assert meta.pearson == "cI168"
    assert meta.space_group_number == 204
    assert meta.lattice_a is not None


# --- the same cell in three settings is not a disagreement ----------------

def test_a_second_cell_setting_is_not_a_contradiction(caplog):
    """The regression the first version of this guard would have shipped.

    Three values for `a`, one crystal. The standardized one is reported — it is
    the cell that agrees with the Pearson symbol cF4; 2.8631 is the primitive
    edge and would not.
    """
    with caplog.at_level(logging.WARNING, logger="phase_metadata"):
        got = _parse_cif_field(_THREE_SETTINGS, "_cell_length_a", source="Al.cif")
    assert got == "4.049", (
        "the lattice parameter must survive: blanking it costs 14 library files "
        "the number the phase library shows")
    assert not caplog.records, (
        f"a named cell setting was read as a contradiction: "
        f"{[r.getMessage() for r in caplog.records]}")


def test_the_standard_uncertainty_form_alone_is_not_a_contradiction(caplog):
    """`4.049` and `4.049(1)` are one measurement, written twice."""
    text = _THREE_SETTINGS.replace("data_sm_isp_SD1250064-niggli_reduced_cell\n"
                                   "_cell_length_a                   2.8631\n", "")
    with caplog.at_level(logging.WARNING, logger="phase_metadata"):
        assert _parse_cif_field(text, "_cell_length_a") == "4.049"
    assert not caplog.records


def test_two_blocks_in_the_same_setting_still_disagree(caplog):
    """Within one setting there is no innocent explanation left."""
    text = _THREE_SETTINGS + (
        "\ndata_sm_isp_SD1250064-standardized_unitcell\n"
        "_cell_length_a                   9.999\n")
    with caplog.at_level(logging.WARNING, logger="phase_metadata"):
        assert _parse_cif_field(text, "_cell_length_a", source="Al.cif") == ""
    assert caplog.records
    msg = caplog.records[0].getMessage()
    assert "standardized_unitcell" in msg and "Al.cif" in msg


def test_the_block_role_comes_from_the_name():
    assert _cif_block_role("sm_isp_SD1250064-niggli_reduced_cell") == \
        "niggli_reduced_cell"
    assert _cif_block_role("sm_isp_SD1250064-published_cell") == "published_cell"
    assert _cif_block_role("sm_global") == ""
    assert _cif_block_role("block_one") == "", (
        "a block that does not declare a setting must be comparable, or a "
        "foreign two-block CIF goes unchecked")


def test_a_trailing_comment_does_not_hide_the_setting():
    """CIF allows `data_foo  # note`; the comment must not reach the role."""
    assert _cif_block_role("sm_isp_X-published_cell   # the published one") == \
        "published_cell"


def test_an_uppercase_data_header_still_splits(caplog):
    """CIF reserved words are case-insensitive, so `DATA_` is a legal header.

    Splitting case-sensitively left the whole file as ONE block, collapsed the
    three cell settings into one bucket and blanked the lattice parameter — the
    damage this guard exists to prevent, through the other door. Found by review;
    it fails without `re.IGNORECASE`.
    """
    for spelling in ("DATA_", "Data_"):
        text = _THREE_SETTINGS.replace("data_", spelling)
        with caplog.at_level(logging.WARNING, logger="phase_metadata"):
            caplog.clear()
            got = _parse_cif_field(text, "_cell_length_a", source="Al.cif")
        assert got == "4.049", f"{spelling} header lost the lattice parameter"
        assert not caplog.records, f"{spelling}: {[r.getMessage() for r in caplog.records]}"


def test_a_headerless_cif_is_still_read_and_still_checked():
    """A file with no `data_` at all is all preamble.

    Two things at once, because the pre-header branch was unverified and a
    mutation that deleted it passed every other test: the value must come back,
    AND a contradiction inside that text must still be caught — that text is the
    foreign single-block CIF this guard is for.
    """
    assert _parse_cif_field("_cell_length_a 4.049\n_cell_angle_alpha 90\n",
                            "_cell_length_a") == "4.049"
    assert _parse_cif_field("_cell_length_a 4.049\n_cell_length_a 9.999\n",
                            "_cell_length_a") == ""


def test_a_preamble_tag_does_not_contradict_a_real_block():
    """Malformed, and the parent reader tolerated it; keep tolerating it.

    Text before the first header is not a block, so it must not compete with one.
    """
    text = "_cell_length_a 4.049\ndata_only\n_cell_length_a 7.777\n"
    assert _parse_cif_field(text, "_cell_length_a") == "4.049"


def test_a_contradiction_outside_the_first_bucket_is_caught(caplog):
    """The loop must check EVERY setting, not the first one it inserted.

    A mutation that looked at `list(by_setting.items())[:1]` passed all 17 earlier
    tests, because in each of them the offending bucket happened to be inserted
    first. Here the clean `sm_global` and standardized buckets come first and the
    disagreement is in `published_cell`.
    """
    text = _THREE_SETTINGS.replace(
        "data_sm_isp_SD1250064-published_cell\n"
        "_symmetry_space_group_name_H-M   'Fm-3m'\n"
        "_cell_length_a                   4.049(1)\n",
        "data_sm_isp_SD1250064-published_cell\n"
        "_symmetry_space_group_name_H-M   'Fm-3m'\n"
        "_cell_length_a                   4.049(1)\n"
        "_cell_length_a                   9.999\n")
    assert "9.999" in text, "fixture edit missed — the mutation guard is blind"
    with caplog.at_level(logging.WARNING, logger="phase_metadata"):
        assert _parse_cif_field(text, "_cell_length_a", source="Al.cif") == ""
    assert caplog.records
    assert "published_cell" in caplog.records[0].getMessage()


def test_the_origin_choice_is_reported_as_it_is_today():
    """Si states `Fd-3m` and `Fd-3m O1`. Deliberately not a contradiction.

    Different settings, so the guard stays quiet, and the standardized value is
    reported exactly as before this change. The origin choice is settled where a
    .xtal is written, not by blanking a display field.
    """
    text = _THREE_SETTINGS.replace("'Fm-3m'", "'Fd-3m'", 1) \
                          .replace("'Fm-3m'", "'Fd-3m O1'", 1)
    assert _parse_cif_field(text, "_symmetry_space_group_name_H-M") == "Fd-3m"


# --- the guard that keeps the claims above true ----------------------------

def _reader_tags() -> list[str]:
    """Every tag `extract_metadata_from_cif` actually asks for, read from source.

    Hand-maintained lists are why this guard needed three goes: the first survey
    omitted `_cell_length_*` (and produced the "0 contradictions" claim that was
    wrong), and a later one omitted `_cell_angle_*` (and produced "43" where the
    full set gives 68). Deriving it ends the class — a tag added to the reader is
    covered the same day.
    """
    src = (Path(phase_metadata.__file__)).read_text(encoding="utf-8")
    tags = sorted(set(re.findall(
        r"""_parse_cif_field\(\s*\w+\s*,\s*["'](_[A-Za-z0-9_.\-]+)["']""", src)))
    assert len(tags) >= 14, f"only {len(tags)} tags found — did the call shape change?"
    return tags


def _library_cifs() -> list[Path]:
    """The real library, or skip. Never an empty or partial one.

    A review emptied `CIF_Library` and found the lattice test passing on zero
    files while it was the commit's stated evidence. Both library tests now come
    through here, and both require multi-block files — a single-block corpus
    cannot exercise the guard at all, so passing on one would mean nothing.
    """
    if not LIBRARY.is_dir():
        pytest.skip("crystal library not in this checkout")
    cifs = sorted(LIBRARY.glob("*.cif"))
    if len(cifs) < 30:
        pytest.skip(f"only {len(cifs)} CIFs — not the real library")
    multi = [p for p in cifs
             if len(re.findall(r"^data_", p.read_text(encoding="utf-8",
                                                      errors="replace"),
                               re.MULTILINE | re.IGNORECASE)) > 1]
    assert len(multi) >= 15, (
        f"only {len(multi)} multi-block CIFs — this corpus cannot exercise the "
        f"guard, so a pass would prove nothing")
    return cifs


def test_the_shipped_library_has_no_contradictions(caplog):
    """Every library CIF must read without a single contradiction warning.

    This is the test that fires the day a CIF is added whose blocks genuinely
    disagree — and the test that caught the first version of this guard calling
    43 benign multi-setting values a contradiction.
    """
    cifs = _library_cifs()
    tags = _reader_tags()
    with caplog.at_level(logging.WARNING, logger="phase_metadata"):
        for p in cifs:
            text = p.read_text(encoding="utf-8", errors="replace")
            for tag in tags:
                _parse_cif_field(text, tag, source=p.name)
    offenders = [r.getMessage() for r in caplog.records]
    assert offenders == [], (
        f"{len(offenders)} contradiction(s) in the shipped library:\n"
        + "\n".join(offenders[:5]))


def test_the_library_keeps_every_lattice_parameter_it_shows_today():
    """No file may lose a displayed field to this guard.

    The blunt version blanked `lattice_a/b/c` for 14 files. This reads every CIF
    through the public function and requires a lattice parameter wherever the
    file states one at all.
    """
    missing = []
    for p in _library_cifs():
        states_a = "_cell_length_a" in p.read_text(encoding="utf-8",
                                                   errors="replace")
        meta = extract_metadata_from_cif(p)
        if states_a and meta.lattice_a is None:
            missing.append(p.name)
    assert missing == [], f"lattice parameter lost for: {missing}"
