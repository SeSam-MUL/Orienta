from pathlib import Path

import pytest

from backend.api.services.addons.manifest import (
    API_VERSION,
    ManifestError,
    declared_params,
    parse_manifest,
    recordable_params,
)

FIXTURE = Path(__file__).parent / "fixtures" / "valid_addon.toml"


def test_parses_the_declared_fields():
    m = parse_manifest(FIXTURE)
    assert m.name == "grain-shape-stats"
    assert m.display_name == "Grain shape statistics"
    assert m.version == "1.2.0"
    assert m.doi == "10.5281/zenodo.1234567"
    assert m.authors == ("Jane Doe <jane@example.org>",)
    assert m.source_path == FIXTURE


def test_parses_the_analysis():
    a = parse_manifest(FIXTURE).analyses[0]
    assert a.key == "addon.grain_shape_stats"
    assert a.python_name == "grain_shape_stats.run:analyse"
    assert a.citations == ("doi:10.5281/zenodo.1234567",)
    assert a.params_schema["properties"]["method"]["enum"] == ["crofton", "feret"]


def test_declared_params_are_the_schema_properties():
    a = parse_manifest(FIXTURE).analyses[0]
    assert set(declared_params(a)) == {"method", "report_dir"}


def test_an_analysis_without_a_schema_declares_no_parameters():
    """The manifest is the contract: no declaration, nothing accepted."""
    a = parse_manifest(FIXTURE).analyses[0]
    bare = type(a)(a.key, a.label, a.python_name, a.sentence, a.citations, {})
    assert declared_params(bare) == {}
    assert recordable_params(bare) == ()


def test_a_path_parameter_is_declared_but_never_recordable():
    """A recorded path publishes the operator's directory layout into the
    methods paragraph and every exported .h5 — provenance._reject_local_paths
    refuses one outright under pytest."""
    a = parse_manifest(FIXTURE).analyses[0]
    assert "report_dir" in declared_params(a)
    assert recordable_params(a) == ("method",)


def test_a_wrong_api_version_is_refused_by_name(tmp_path):
    p = tmp_path / "a.toml"
    p.write_text(FIXTURE.read_text(encoding="utf-8").replace(
        "api_version = 0", "api_version = 99"), encoding="utf-8")
    with pytest.raises(ManifestError, match="api_version"):
        parse_manifest(p)


def test_the_api_version_is_checked_before_any_other_field(tmp_path):
    """An author on the wrong API must read THAT, not a complaint about a
    field their manifest never got as far as declaring."""
    p = tmp_path / "a2.toml"
    p.write_text("api_version = 99\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="api_version"):
        parse_manifest(p)


def test_a_missing_api_version_is_refused_by_name(tmp_path):
    p = tmp_path / "a3.toml"
    p.write_text('name = "x"\n', encoding="utf-8")
    with pytest.raises(ManifestError, match="api_version"):
        parse_manifest(p)


@pytest.mark.parametrize("bool_literal", ["true", "false"])
def test_a_boolean_api_version_is_not_an_integer(tmp_path, bool_literal):
    """bool is a subclass of int. ``true == 1`` is already refused by the
    plain value check (``declared != API_VERSION``) with no help from the
    ``isinstance`` guard. ``false == 0 == API_VERSION`` is the only input
    that distinguishes the guard from no guard at all — without it, a
    manifest declaring ``api_version = false`` would be silently accepted
    as version 0. Both literals must still be refused, by name."""
    p = tmp_path / f"a4_{bool_literal}.toml"
    p.write_text(FIXTURE.read_text(encoding="utf-8").replace(
        "api_version = 0", f"api_version = {bool_literal}"), encoding="utf-8")
    with pytest.raises(ManifestError, match="api_version"):
        parse_manifest(p)


def test_a_missing_required_field_names_the_field(tmp_path):
    p = tmp_path / "b.toml"
    p.write_text("api_version = 0\nname = \"x\"\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="display_name"):
        parse_manifest(p)


def test_an_analysis_key_must_start_with_addon(tmp_path):
    """Add-on keys are namespaced so they can never collide with a core step."""
    p = tmp_path / "c.toml"
    p.write_text(FIXTURE.read_text(encoding="utf-8").replace(
        'key = "addon.grain_shape_stats"', 'key = "indexing.hough"'), encoding="utf-8")
    with pytest.raises(ManifestError, match="addon\\."):
        parse_manifest(p)


def test_python_name_must_be_module_colon_attr(tmp_path):
    p = tmp_path / "d.toml"
    p.write_text(FIXTURE.read_text(encoding="utf-8").replace(
        'python_name = "grain_shape_stats.run:analyse"',
        'python_name = "grain_shape_stats.run.analyse"'), encoding="utf-8")
    with pytest.raises(ManifestError, match="module:attribute"):
        parse_manifest(p)


def test_sentence_slots_must_be_named(tmp_path):
    """Positional slots break when params arrive as a dict."""
    p = tmp_path / "e.toml"
    p.write_text(FIXTURE.read_text(encoding="utf-8").replace(
        "on {n_grains} grains.", "on {} grains."), encoding="utf-8")
    with pytest.raises(ManifestError, match="named"):
        parse_manifest(p)


def test_a_citation_reference_must_render_as_a_bibtex_key(tmp_path):
    """Add-on entries go through render_bibtex unescaped, as `@misc{<id>,`.
    A comma or a brace there produces a .bib file no reader can parse."""
    p = tmp_path / "g.toml"
    p.write_text(FIXTURE.read_text(encoding="utf-8").replace(
        'citations = ["doi:10.5281/zenodo.1234567"]',
        'citations = ["some, thing"]'), encoding="utf-8")
    with pytest.raises(ManifestError, match="citation"):
        parse_manifest(p)


def test_malformed_toml_is_a_manifest_error_not_a_toml_error(tmp_path):
    p = tmp_path / "f.toml"
    p.write_text("this is not toml [[[", encoding="utf-8")
    with pytest.raises(ManifestError):
        parse_manifest(p)


def test_missing_file_raises_manifest_error(tmp_path):
    with pytest.raises(ManifestError):
        parse_manifest(tmp_path / "nope.toml")


def test_api_version_constant_is_an_int():
    assert isinstance(API_VERSION, int)


# --- the declared types are enforced, not annotated -------------------------

def test_an_unquoted_version_is_a_date_and_is_refused(tmp_path):
    """The one that did real damage, and the reason this section exists.

    ``version = 2026-01-01`` is not a TOML error: tomllib returns a
    ``datetime.date``. The dataclass says ``version: str`` and nothing
    enforced it, so the date travelled into the trust store, where
    ``json.dump`` raised ``TypeError`` -- after ``open(..., "w")`` had
    already truncated the file. Every consent decision on the machine, gone,
    from one badly-written manifest.
    """
    p = tmp_path / "dated.toml"
    p.write_text(FIXTURE.read_text(encoding="utf-8").replace(
        'version = "1.2.0"', "version = 2026-01-01"), encoding="utf-8")
    with pytest.raises(ManifestError) as exc:
        parse_manifest(p)
    assert "version" in str(exc.value)
    assert "date" in str(exc.value)         # names what it got
    assert "quotes" in str(exc.value)       # and what to do about it


@pytest.mark.parametrize("field,replacement", [
    ("name", 'name = 1'),
    ("display_name", 'display_name = true'),
    ("version", 'version = 1.2'),
    ("doi", 'doi = 10.5281'),
    ("requires_orienta", 'requires_orienta = 0.4'),
])
def test_a_top_level_field_must_be_a_string(tmp_path, field, replacement):
    original = {
        "name": 'name = "grain-shape-stats"',
        "display_name": 'display_name = "Grain shape statistics"',
        "version": 'version = "1.2.0"',
        "doi": 'doi = "10.5281/zenodo.1234567"',
        "requires_orienta": 'requires_orienta = ">=0.4,<0.6"',
    }[field]
    p = tmp_path / f"{field}.toml"
    p.write_text(FIXTURE.read_text(encoding="utf-8").replace(
        original, replacement), encoding="utf-8")
    with pytest.raises(ManifestError, match=field):
        parse_manifest(p)


def test_authors_must_be_a_list_and_not_a_bare_string(tmp_path):
    """``tuple("Jane")`` is four authors of one letter each -- in the dialog
    whose whole job is to show the user WHO wrote this."""
    p = tmp_path / "authors.toml"
    p.write_text(FIXTURE.read_text(encoding="utf-8").replace(
        'authors = ["Jane Doe <jane@example.org>"]',
        'authors = "Jane Doe <jane@example.org>"'), encoding="utf-8")
    with pytest.raises(ManifestError, match="authors"):
        parse_manifest(p)


def test_an_author_entry_must_itself_be_a_string(tmp_path):
    p = tmp_path / "author_entry.toml"
    p.write_text(FIXTURE.read_text(encoding="utf-8").replace(
        'authors = ["Jane Doe <jane@example.org>"]',
        'authors = ["Jane Doe <jane@example.org>", 7]'), encoding="utf-8")
    with pytest.raises(ManifestError, match=r"authors\[1\]"):
        parse_manifest(p)


@pytest.mark.parametrize("original,replacement,expected", [
    ('key = "addon.grain_shape_stats"', "key = 42", "key"),
    ('label = "Grain shape statistics"', "label = 42", "label"),
    ('python_name = "grain_shape_stats.run:analyse"',
     "python_name = 42", "python_name"),
    ('sentence = "Grain shape statistics were computed with {method} on '
     '{n_grains} grains."', "sentence = 42", "sentence"),
])
def test_an_analysis_field_must_be_a_string(tmp_path, original, replacement,
                                            expected):
    """Without the check these raised ``AttributeError``/``TypeError`` from
    ``startswith`` / ``count`` / ``Formatter().parse`` -- which
    ``discovery._read`` does NOT catch, so one bad manifest took the whole
    listing down instead of appearing as one rejected row."""
    p = tmp_path / "analysis_field.toml"
    text = FIXTURE.read_text(encoding="utf-8")
    assert original in text
    p.write_text(text.replace(original, replacement), encoding="utf-8")
    with pytest.raises(ManifestError, match=expected):
        parse_manifest(p)


def test_a_citation_entry_must_be_a_string(tmp_path):
    p = tmp_path / "citations.toml"
    p.write_text(FIXTURE.read_text(encoding="utf-8").replace(
        'citations = ["doi:10.5281/zenodo.1234567"]',
        "citations = [2026-01-01]"), encoding="utf-8")
    with pytest.raises(ManifestError, match="citations"):
        parse_manifest(p)


def test_a_params_schema_must_be_a_table(tmp_path):
    p = tmp_path / "schema.toml"
    p.write_text(FIXTURE.read_text(encoding="utf-8").replace(
        "[analyses.params_schema]\ntype = \"object\"",
        "params_schema = \"nope\"\n[analyses.unused]"), encoding="utf-8")
    with pytest.raises(ManifestError, match="params_schema"):
        parse_manifest(p)


def test_every_field_a_rejected_manifest_names_reaches_the_author(tmp_path):
    """A type error must arrive as a ManifestError -- the only class
    ``discovery._read`` turns into a visible, rejected row. Anything else is
    an exception nobody catches."""
    p = tmp_path / "bad.toml"
    p.write_text(FIXTURE.read_text(encoding="utf-8").replace(
        'version = "1.2.0"', "version = 2026-01-01"), encoding="utf-8")
    try:
        parse_manifest(p)
    except ManifestError:
        pass
    except Exception as exc:                        # pragma: no cover
        pytest.fail(f"raised {type(exc).__name__}, which nothing catches: {exc}")


def test_the_experimental_label_is_written_down_without_hedging():
    """Spec §6: API 0's experimental label is made true by mechanism — a hard
    load block (the api_version check above), a separate CHANGELOG-ADDON-API.md,
    and one sentence with no hedging. The block was built and the other two were
    neither built nor declared deferred.

    Pinned verbatim so the sentence cannot be softened later into "we may
    occasionally need to make changes". An author decides whether to build on
    API 0 by reading it.
    """
    changelog = Path(__file__).resolve().parents[2] / "CHANGELOG-ADDON-API.md"
    assert changelog.is_file(), changelog
    text = changelog.read_text(encoding="utf-8")
    assert (
        "API 0: breaks will be announced, there is no deprecation window; "
        "from API 1 the previous number stays loadable for one minor release."
    ) in text
    assert f"API {API_VERSION}" in text


# --- a name and a key are URL path segments too ----------------------------

def _manifest_with(tmp_path, old, new):
    src = FIXTURE.read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    p = tmp_path / "x.toml"
    p.write_text(src.replace(old, new), encoding="utf-8")
    return p


def test_a_name_that_cannot_be_a_url_segment_is_refused(tmp_path):
    """``name`` was never validated at all, and it is a segment of every
    add-on route -- ``/api/addons/{name}/run`` and the ``values_url`` a map
    output advertises. A '/' in it makes that link unreachable while the
    listing shows the add-on as installed all the same."""
    p = _manifest_with(tmp_path, 'name = "grain-shape-stats"',
                       'name = "grain/stats"')
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(p)
    assert "name" in str(excinfo.value)
    assert "/" in str(excinfo.value)


def test_an_analysis_key_that_cannot_be_a_url_segment_is_refused(tmp_path):
    p = _manifest_with(tmp_path, 'key = "addon.grain_shape_stats"',
                       'key = "addon.grain/shape"')
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(p)
    assert "addon.grain/shape" in str(excinfo.value)


def test_the_names_real_addons_use_are_still_accepted(tmp_path):
    for name in ("bc-gmm", "grain_shape_stats", "addon.v1.2"):
        p = _manifest_with(tmp_path, 'name = "grain-shape-stats"',
                           f'name = "{name}"')
        assert parse_manifest(p).name == name


# --- a schema default reaches the provenance trail --------------------------

def test_a_schema_default_that_is_a_toml_date_is_refused(tmp_path):
    """``_require_str`` guards the top-level fields; nothing guarded a value
    one level down, inside ``params_schema``. A default the caller leaves
    alone is recorded, and ``provenance._jsonable`` ends in ``repr()`` -- so
    an unquoted date arrived in a methods paragraph as
    ``datetime.date(2026, 1, 1)``. Same defect as the one that once truncated
    the trust store, one level lower."""
    p = _manifest_with(tmp_path, 'default = "crofton"', "default = 2026-01-01")
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(p)
    assert "method" in str(excinfo.value)
    assert "quotes" in str(excinfo.value) or "date" in str(excinfo.value)


def test_the_default_kinds_an_author_really_writes_are_accepted(tmp_path):
    for value in ('"crofton"', "3", "1.5", "true", '["a", "b"]',
                  "{ a = 1 }"):
        p = _manifest_with(tmp_path, 'default = "crofton"',
                           f"default = {value}")
        assert parse_manifest(p).analyses[0].params_schema


def test_one_manifest_cannot_declare_one_analysis_key_twice(tmp_path):
    """``run_analysis`` takes the FIRST analysis with a key, and the map store
    is keyed by the key -- so a second analysis under the same key is a listed
    control that can never run and whose outputs would be served under the
    first one's name. The cross-add-on half of this is refused in
    ``citations_bridge``; this is the half inside one manifest."""
    src = FIXTURE.read_text(encoding="utf-8")
    block = src[src.index("[[analyses]]"):src.index("[analyses.params_schema]")]
    p = tmp_path / "twice.toml"
    p.write_text(src + "\n" + block.replace(
        'label = "Grain shape statistics"', 'label = "Second"'),
        encoding="utf-8")
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(p)
    assert "addon.grain_shape_stats" in str(excinfo.value)


def test_a_name_of_nothing_but_dots_is_refused(tmp_path):
    p = _manifest_with(tmp_path, 'name = "grain-shape-stats"', 'name = ".."')
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(p)
    assert "name" in str(excinfo.value)


def test_an_analysis_key_of_nothing_but_dots_is_refused(tmp_path):
    """Refused twice over -- the all-dots rule and the ``addon.`` prefix --
    and the test does not care which one speaks, only that one does."""
    p = _manifest_with(tmp_path, 'key = "addon.grain_shape_stats"',
                       'key = ".."')
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(p)
    assert ".." in str(excinfo.value)


def test_an_empty_version_is_refused_because_consent_depends_on_it(tmp_path):
    """A guard whose input the guarded party controls is not a guard.

    The page asks for consent AGAIN when the add-on under a known name has
    changed, comparing the recorded version and DOI with the ones in front of
    it. An empty RECORDED version has to mean "nothing to compare" -- trust
    files written before those fields existed carry none -- so an add-on
    declaring ``version = ""`` would switch its own staleness check off for
    good: delete the folder, drop different code in under the same name, and
    it is enabled with no dialog. Most add-ons declare no DOI (the shipped
    example does not), so the version is the only discriminator left.
    """
    p = _manifest_with(tmp_path, 'version = "1.2.0"', 'version = ""')
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(p)
    assert "version" in str(excinfo.value)


def test_a_version_of_nothing_but_spaces_is_refused_too(tmp_path):
    p = _manifest_with(tmp_path, 'version = "1.2.0"', 'version = "   "')
    with pytest.raises(ManifestError):
        parse_manifest(p)
