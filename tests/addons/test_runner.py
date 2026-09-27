import logging
import sys
from pathlib import Path

import numpy as np
import pytest

from backend.api.services.addons.context import build_context
from backend.api.services.addons.manifest import parse_manifest
from backend.api.services.addons.runner import (
    AddonError,
    probe_import,
    run_analysis,
)
from backend.api.services.addons.trust import TrustStore
from backend.api.services.citations.provenance import get_steps

FIXTURES = Path(__file__).parent / "fixtures"
MANIFEST = FIXTURES / "runnable_addon" / "orienta-addon.toml"
EXPLODING = FIXTURES / "exploding_addon" / "orienta-addon.toml"
RUNNER_LOGGER = "backend.api.services.addons.runner"


class _Result:
    def __init__(self):
        self.original_shape = (3, 4)
        self.metadata = {}
        self.confidence_scores = None
        self.xmap = None


@pytest.fixture(autouse=True)
def _forget_the_module():
    """Only forgets the MODULE. It does NOT put anything on sys.path.

    Reaching past the product with a path fixture is how the folder-install
    half of this contract stayed broken and invisible: every test in the
    PREVIOUS draft of this module used monkeypatch.syspath_prepend, so no test
    could notice that nothing in the product ever made an installed add-on
    importable. (``test_discovery.py`` still uses it, legitimately - see
    test_the_product_makes_a_folder_addon_importable for why the two differ.)
    """
    names = ("runnable_addon_module", "exploding_addon_module",
             "isolated_addon_module", "collided_module",
             "exiting_addon_module", "argparse_addon_module",
             "exiting_import_module")
    for name in names:
        sys.modules.pop(name, None)
    yield
    for name in names:
        sys.modules.pop(name, None)


def _ctx():
    return build_context(_Result(), quality=np.full((3, 4), 2.0),
                         quality_source="native")


def test_the_product_makes_a_folder_addon_importable():
    """C1, the whole folder-install half of the contract.

    Nothing else in the process knows where ``~/.orienta/addons/<name>/`` is;
    only the runner puts it there. This fixture is used by NO other test, so
    the "before" assertion is real whatever order the suite runs in.

    There is no ``syspath_prepend`` in THIS module, and that is the honest
    form of the rule. ``test_discovery.py`` uses it three times (lines 162,
    182, 208) and is right to: discovery does not own ``sys.path``, so a test
    of discovery has to put a package where the product must then find it. The
    runner DOES own ``sys.path`` — putting the add-on there is the job under
    test — so the same fixture here would do the product's work for it and
    hide its absence. That is precisely how the folder-install half of the
    contract stayed broken and invisible.
    """
    import importlib

    root = str((FIXTURES / "isolated_addon").resolve())
    assert root not in sys.path
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("isolated_addon_module")

    run = run_analysis(
        parse_manifest(FIXTURES / "isolated_addon" / "orienta-addon.toml"),
        "addon.isolated", _ctx(), {})
    assert root in sys.path
    assert [o.key for o in run.outputs] == ["ok"]


def test_the_addon_folder_is_appended_not_prepended():
    """An add-on must not be able to shadow the stdlib or site-packages by
    naming a module after one."""
    run_analysis(parse_manifest(MANIFEST), "addon.mean_quality", _ctx(), {})
    root = str((FIXTURES / "runnable_addon").resolve())
    assert root in sys.path
    assert sys.path.index(root) > 0


def test_it_runs_and_returns_validated_outputs():
    run = run_analysis(parse_manifest(MANIFEST), "addon.mean_quality", _ctx(),
                       {"scale": 2.0})
    by_key = {o.key: o for o in run.outputs}
    assert by_key["mean"].value == pytest.approx(4.0)
    assert by_key["scaled"].values.shape == (3, 4)


def test_the_module_is_imported_ONLY_when_it_runs():
    assert "runnable_addon_module" not in sys.modules
    parse_manifest(MANIFEST)
    assert "runnable_addon_module" not in sys.modules, "parsing must not import"
    run_analysis(parse_manifest(MANIFEST), "addon.mean_quality", _ctx(), {})
    assert "runnable_addon_module" in sys.modules


def test_a_failure_names_the_addon_first():
    with pytest.raises(AddonError) as exc:
        run_analysis(parse_manifest(MANIFEST), "addon.mean_quality", _ctx(),
                     {"fail": True})
    assert str(exc.value).startswith("runnable")
    assert exc.value.addon_name == "runnable"
    assert exc.value.analysis_key == "addon.mean_quality"


def test_a_failure_is_counted_against_the_addon(tmp_path):
    trust = TrustStore(tmp_path / "t.json")
    trust.set_enabled("runnable", True, version="0.1.0", doi="")
    with pytest.raises(AddonError):
        run_analysis(parse_manifest(MANIFEST), "addon.mean_quality", _ctx(),
                     {"fail": True}, trust=trust)
    assert trust.crashes("runnable") == 1


def test_a_success_clears_the_crash_count(tmp_path):
    trust = TrustStore(tmp_path / "t.json")
    trust.set_enabled("runnable", True, version="0.1.0", doi="")
    trust.record_crash("runnable")
    run_analysis(parse_manifest(MANIFEST), "addon.mean_quality", _ctx(), {},
                 trust=trust)
    assert trust.crashes("runnable") == 0


def test_an_unknown_analysis_key_is_refused_by_name():
    with pytest.raises(AddonError, match="addon.nope"):
        run_analysis(parse_manifest(MANIFEST), "addon.nope", _ctx(), {})


def test_an_import_failure_is_an_addon_error_naming_the_addon():
    with pytest.raises(AddonError) as exc:
        probe_import(parse_manifest(EXPLODING))
    assert exc.value.addon_name == "exploding"
    assert "explodes on import" in str(exc.value)


def test_a_bad_output_is_an_addon_error_not_a_crash(monkeypatch, caplog):
    """And it is handled by the OutputError clause, not the generic one.

    Deleting ``except OutputError`` left all seventeen of the first round's
    tests green, which made the clause indistinguishable from decoration. The
    two things it actually does differently are asserted here:

    * the validator's message is passed through VERBATIM. It is already a
      finished, author-facing sentence with a remedy in it, and the generic
      clause would prefix it with ``OutputError:`` — its own class name, which
      means nothing to the author reading it;
    * no traceback is logged. A malformed return is not a crash; a stack trace
      in the diagnostics zip invites someone to debug Orienta over it.
    """
    probe_import(parse_manifest(MANIFEST))      # puts it on sys.path, honestly
    import runnable_addon_module  # noqa: F401  (the monkeypatch target)

    monkeypatch.setattr("runnable_addon_module.analyse",
                        lambda context, **kw: ["not an output"])
    with caplog.at_level(logging.DEBUG, logger=RUNNER_LOGGER):
        with pytest.raises(AddonError, match="runnable") as exc:
            run_analysis(parse_manifest(MANIFEST), "addon.mean_quality",
                         _ctx(), {})

    assert str(exc.value).endswith("unknown output type str")
    assert "OutputError" not in str(exc.value)
    assert [r.getMessage() for r in caplog.records
            if r.name == RUNNER_LOGGER and r.levelno >= logging.ERROR] == []


def test_the_step_is_recorded_on_the_result():
    result = _Result()
    run_analysis(parse_manifest(MANIFEST), "addon.mean_quality",
                 build_context(result, quality=np.zeros((3, 4))), {},
                 result=result)
    assert [s["key"] for s in get_steps(result)] == ["addon.mean_quality"]


def test_nothing_is_recorded_when_the_addon_fails():
    result = _Result()
    with pytest.raises(AddonError):
        run_analysis(parse_manifest(MANIFEST), "addon.mean_quality",
                     build_context(result, quality=np.zeros((3, 4))),
                     {"fail": True}, result=result)
    assert get_steps(result) == []


def test_a_bookkeeping_failure_still_names_the_addon(monkeypatch):
    """record_step raises under pytest on an undeclared key or a path-looking
    param. Outside the guard that escaped as a raw ValueError naming no
    add-on - straight into a diagnostics zip looking like an Orienta bug."""
    import backend.api.services.addons.runner as runner_mod

    def _boom(*a, **kw):
        raise ValueError("bookkeeping went wrong")

    monkeypatch.setattr(runner_mod, "record_step", _boom)
    result = _Result()
    with pytest.raises(AddonError, match="runnable"):
        run_analysis(parse_manifest(MANIFEST), "addon.mean_quality",
                     build_context(result, quality=np.zeros((3, 4))), {},
                     result=result)


def test_an_undeclared_parameter_is_refused_from_the_call_and_the_trail():
    """``params`` is an unauthenticated JSON body and record_step writes into
    the methods paragraph and every exported .h5. In production _strict() is
    False, so a path-looking value was merely warned about and then RECORDED."""
    result = _Result()
    run = run_analysis(
        parse_manifest(MANIFEST), "addon.mean_quality",
        build_context(result, quality=np.zeros((3, 4))),
        {"scale": 1.0, "leak": "C:/Users/someone/secret/scan.h5oina"},
        result=result)
    assert run.ignored_params == ("leak",)
    recorded = get_steps(result)[0]["params"]
    assert "leak" not in recorded
    assert "someone" not in repr(recorded)


def test_a_declared_path_parameter_is_passed_but_never_recorded(tmp_path):
    """``path`` is one of the six field types the spec allows, and a recorded
    path publishes the operator's directory layout."""
    result = _Result()
    run = run_analysis(
        parse_manifest(MANIFEST), "addon.mean_quality",
        build_context(result, quality=np.zeros((3, 4))),
        {"report_dir": str(tmp_path)}, result=result)
    assert run.ignored_params == ()          # declared, so it was PASSED
    assert "report_dir" not in run.recorded_params
    recorded = get_steps(result)[0]["params"]
    assert "report_dir" not in recorded
    assert str(tmp_path) not in repr(recorded)


def test_scalar_outputs_reach_the_sentence():
    """Only input params used to be recorded, so ``{n_px}`` could only ever
    render "[n_px not recorded]" - and both example manifests use such a slot.
    """
    from backend.api.services.citations.render import render_methods

    result = _Result()
    run_analysis(parse_manifest(MANIFEST), "addon.mean_quality",
                 build_context(result, quality=np.full((3, 4), 2.0)),
                 {"scale": 1.0}, result=result)
    recorded = get_steps(result)[0]["params"]
    assert recorded["n_px"] == 12
    sentence = render_methods(get_steps(result))
    assert "over 12 pixels" in sentence
    assert "not recorded" not in sentence


def test_an_input_param_wins_over_an_output_of_the_same_name():
    """What the user ASKED for outranks what the add-on says about it.

    The fixture emits a scalar output also called ``scale``, reporting 100x
    the input on purpose, so this has something to be wrong about.
    """
    result = _Result()
    run = run_analysis(parse_manifest(MANIFEST), "addon.mean_quality",
                       build_context(result, quality=np.full((3, 4), 2.0)),
                       {"scale": 3.0}, result=result)
    assert run.recorded_params["scale"] == 3.0
    assert get_steps(result)[0]["params"]["scale"] == 3.0


def test_a_parameter_left_at_its_default_is_recorded_as_what_ran():
    """The trail used to be silent about the one run every user makes first.

    Only what the caller SENT was recorded, and a parameter left alone is
    never sent -- so a sentence slot naming a declared parameter rendered
    "[scale not recorded]" for the button with nothing typed into it.
    Measured on the reference add-on in ``examples/``.

    Fixed HERE rather than in each add-on, because the local workaround --
    emit the parameter as a ``ScalarOutput`` too -- has no form for a string
    or boolean parameter, so it was advice no author could follow in general.
    """
    result = _Result()
    run = run_analysis(parse_manifest(MANIFEST), "addon.mean_quality",
                       build_context(result, quality=np.full((3, 4), 2.0)),
                       {}, result=result)
    assert run.recorded_params["scale"] == 1.0
    assert run.recorded_params["fail"] is False
    assert get_steps(result)[0]["params"]["scale"] == 1.0


def test_a_default_outranks_the_addons_own_scalar_of_the_same_name():
    """A default is an INPUT fact, so it obeys the rule above it.

    The fixture reports ``scale`` as a hundred times its input on purpose.
    Recording 100.0 for a run that used 1.0 would be a false statement about
    the run, in text written to be pasted into a manuscript.
    """
    result = _Result()
    run = run_analysis(parse_manifest(MANIFEST), "addon.mean_quality",
                       build_context(result, quality=np.full((3, 4), 2.0)),
                       {}, result=result)
    assert run.recorded_params["scale"] == 1.0, "the 100x output won"


def test_the_default_of_a_path_parameter_is_still_never_recorded():
    """``format = "path"`` excludes a parameter from the trail by
    DECLARATION, and a default is exactly as capable of being a local path as
    a sent value is. The fixture's ``report_dir`` carries one; if it reached
    the trail, ``_reject_local_paths`` would raise under pytest."""
    result = _Result()
    run = run_analysis(parse_manifest(MANIFEST), "addon.mean_quality",
                       build_context(result, quality=np.full((3, 4), 2.0)),
                       {}, result=result)
    assert "report_dir" not in run.recorded_params
    assert "report_dir" not in get_steps(result)[0]["params"]


# ---------------------------------------------------------------------------
# Fix round 1: the module name is not the add-on's, and it is one namespace.
# ---------------------------------------------------------------------------


def test_two_addons_sharing_a_module_name_do_not_run_each_others_code():
    """The worst thing this module can do, and it used to be silent.

    A top-level module name is one process-wide namespace and
    ``import_module`` answers from ``sys.modules`` first, so the second add-on
    to declare ``collided_module`` got the FIRST one's module back. Its run
    returned the first author's outputs, with no error, attributed to the
    second add-on and cited as the second add-on - in a branch whose entire
    purpose is that an author gets credit for their own work, inside text a
    researcher pastes into a paper.
    """
    a = parse_manifest(FIXTURES / "collide_a" / "orienta-addon.toml")
    b = parse_manifest(FIXTURES / "collide_b" / "orienta-addon.toml")

    run_a = run_analysis(a, "addon.collide_a", _ctx(), {})
    assert [o.value for o in run_a.outputs] == [1.0]     # A's own number

    with pytest.raises(AddonError) as exc:
        run_analysis(b, "addon.collide_b", _ctx(), {})

    # Blamed on B, because B is the one that has to rename its module - but
    # both paths are in the message, because the operator has to see WHICH two
    # add-ons collided to do anything about it.
    assert exc.value.addon_name == "collide_b"
    assert exc.value.analysis_key == "addon.collide_b"
    assert "collide_a" in str(exc.value)
    assert "collide_b" in str(exc.value)


def test_a_module_name_that_shadows_an_orienta_package_is_refused():
    """``analysis`` is a real package at this repository's root.

    An add-on naming its module after one of Orienta's own resolves to
    Orienta's, and is then blamed for an import it did not perform. The
    message has to point at the file that actually answered.
    """
    with pytest.raises(AddonError) as exc:
        probe_import(parse_manifest(
            FIXTURES / "shadow_addon" / "orienta-addon.toml"))

    assert exc.value.addon_name == "shadows"
    import analysis                                    # the real one
    assert str(Path(analysis.__file__).resolve()) in str(exc.value)
    assert "shadow_addon" in str(exc.value)


def test_the_addons_own_module_is_still_accepted_from_a_subpackage_layout():
    """The check is containment, not equality of the parent directory.

    The narrower form - ``module.__file__.parent == root`` - would refuse a
    folder add-on that ships ``mypkg/__init__.py`` instead of a flat module,
    which is an ordinary layout and not a collision at all. The isolated
    fixture is a flat module; this pins that the accepting side of the check
    is not accidentally exact.
    """
    from backend.api.services.addons.runner import _check_module_identity

    manifest = parse_manifest(FIXTURES / "runnable_addon" / "orienta-addon.toml")
    root = (FIXTURES / "runnable_addon").resolve()

    class _FakePackage:
        __name__ = "mypkg"
        __file__ = str(root / "mypkg" / "__init__.py")

    _check_module_identity(manifest, manifest.analyses[0], _FakePackage(), root)


def test_a_builtin_module_name_is_refused_rather_than_run():
    """``getattr(module, "__file__", None)`` is None for a builtin and for a
    namespace package. Neither can be the add-on's code, and a missing
    attribute must not read as "no objection"."""
    from backend.api.services.addons.runner import _check_module_identity

    manifest = parse_manifest(FIXTURES / "runnable_addon" / "orienta-addon.toml")

    class _Builtin:
        __name__ = "sys"

    with pytest.raises(AddonError, match="built-in or namespace"):
        _check_module_identity(manifest, manifest.analyses[0], _Builtin(),
                               (FIXTURES / "runnable_addon").resolve())


# ---------------------------------------------------------------------------
# Fix round 1: the two halves of the credit chain, and a message that raises.
# ---------------------------------------------------------------------------


def test_a_citation_registration_failure_still_names_the_addon(monkeypatch):
    """The sibling of the record_step pin above.

    Only ``record_step``'s position inside the guard was pinned, so moving
    ``register_manifest_citations`` OUT of it kept all seventeen tests green.
    Registration and recording are the two halves of the credit chain; both
    have to be inside, and both have to be held there by a test.
    """
    import backend.api.services.addons.runner as runner_mod

    def _boom(*a, **kw):
        raise ValueError("the bibliography went wrong")

    monkeypatch.setattr(runner_mod, "register_manifest_citations", _boom)
    result = _Result()
    with pytest.raises(AddonError, match="runnable"):
        run_analysis(parse_manifest(MANIFEST), "addon.mean_quality",
                     build_context(result, quality=np.zeros((3, 4))), {},
                     result=result)


def test_an_addon_that_ran_is_cited_even_without_a_result():
    """Registration used to sit under ``if result is not None``.

    An add-on handed nothing to stamp has still RUN, and its works still
    belong in the bibliography - otherwise the citation panel cannot name it
    until its first recorded run, which is a credit gap with no honest reason.
    """
    from backend.api.services.citations.render import load_library
    from backend.api.services.citations.steps import get_step

    assert get_step("addon.mean_quality") is None
    run_analysis(parse_manifest(MANIFEST), "addon.mean_quality", _ctx(), {})
    assert get_step("addon.mean_quality") is not None
    assert "doi:10.5281/zenodo.7777777" in load_library()


def test_an_exception_whose_own_message_raises_is_still_contained(monkeypatch):
    """``f"{exc}"`` runs the ADD-ON's ``__str__``, which is add-on code too.

    Interpolating it inside the very ``raise AddonError(...)`` that exists to
    stop add-on code escaping unnamed is the one place a raw exception could
    still get out of this module.
    """
    probe_import(parse_manifest(MANIFEST))
    import runnable_addon_module  # noqa: F401

    class _Unprintable(Exception):
        def __str__(self):
            raise RuntimeError("even my message explodes")

    def _raise(context, **kw):
        raise _Unprintable()

    monkeypatch.setattr("runnable_addon_module.analyse", _raise)
    with pytest.raises(AddonError) as exc:
        run_analysis(parse_manifest(MANIFEST), "addon.mean_quality", _ctx(), {})

    assert exc.value.addon_name == "runnable"
    assert "_Unprintable" in str(exc.value)
    assert "__str__ raised" in str(exc.value)


# ---------------------------------------------------------------------------
# Fix round 2: SystemExit is a BaseException, and argparse raises it for you.
# ---------------------------------------------------------------------------

EXITING = FIXTURES / "exiting_addon" / "orienta-addon.toml"


def test_an_addon_that_calls_sys_exit_is_named_not_obeyed():
    """``sys.exit`` inside an add-on ends the CALLER's request.

    ``SystemExit`` is a ``BaseException``, so ``except Exception`` never saw
    it: the add-on's mistake arrived as Orienta giving up, attributed to
    nobody. ``KeyboardInterrupt`` stays uncaught, because that one really is
    the operator.
    """
    with pytest.raises(AddonError) as exc:
        run_analysis(parse_manifest(EXITING), "addon.exits", _ctx(), {})

    assert str(exc.value).startswith("exiting")     # its name, first, as ever
    assert exc.value.addon_name == "exiting"
    assert exc.value.analysis_key == "addon.exits"
    assert "SystemExit with code 2" in str(exc.value)
    assert "must RAISE" in str(exc.value)


def test_argparse_exiting_inside_an_addon_is_named_too():
    """The version an author hits without realising they wrote it.

    Nobody types ``sys.exit`` here: ``parse_args()`` prints usage and exits
    with code 2 on bad input. An add-on that parses anything internally takes
    the host down the same path, which is why the message names that 2.
    """
    with pytest.raises(AddonError) as exc:
        run_analysis(parse_manifest(EXITING), "addon.argparses", _ctx(), {})

    assert exc.value.addon_name == "exiting"
    assert exc.value.analysis_key == "addon.argparses"
    assert "SystemExit with code 2" in str(exc.value)
    assert "argparse" in str(exc.value)             # the remedy, for the author


def test_an_addon_that_exits_while_being_imported_is_named(tmp_path):
    """A script-shaped module exits at IMPORT time, which is activation.

    probe_import is the first thing that runs an author's module at all, so it
    is where they meet this - and it must not escape there either.
    """
    trust = TrustStore(tmp_path / "t.json")
    trust.set_enabled("exiting_import", True, version="0.1.0", doi="")

    with pytest.raises(AddonError) as exc:
        probe_import(parse_manifest(
            FIXTURES / "exiting_import_addon" / "orienta-addon.toml"))

    assert exc.value.addon_name == "exiting_import"
    assert "SystemExit with code 3" in str(exc.value)

    # And the run path agrees with the activation path, including the crash
    # count: an add-on that exits is failing, however far it got.
    with pytest.raises(AddonError):
        run_analysis(parse_manifest(
            FIXTURES / "exiting_import_addon" / "orienta-addon.toml"),
            "addon.exits_on_import", _ctx(), {}, trust=trust)
    assert trust.crashes("exiting_import") == 1


def test_an_exit_is_counted_against_the_addon(tmp_path):
    trust = TrustStore(tmp_path / "t.json")
    trust.set_enabled("exiting", True, version="0.1.0", doi="")
    with pytest.raises(AddonError):
        run_analysis(parse_manifest(EXITING), "addon.exits", _ctx(), {},
                     trust=trust)
    assert trust.crashes("exiting") == 1


# --- a run may not be credited with another add-on's declaration ------------

def _twin_manifest(tmp_path):
    """A second add-on with a different name, DOI and sentence, on the SAME
    analysis key as ``runnable_addon`` — a fork of a shipped example."""
    src = (MANIFEST.read_text(encoding="utf-8")
           .replace('name = "runnable"', 'name = "forked"')
           .replace('doi = "10.5281/zenodo.7777777"',
                    'doi = "10.5281/zenodo.8888888"')
           .replace('citations = ["doi:10.5281/zenodo.7777777"]',
                    'citations = ["doi:10.5281/zenodo.8888888"]')
           .replace('sentence = "Mean pattern quality was computed over '
                    '{n_px} pixels at scale {scale}."',
                    'sentence = "The fork did it over {n_px} pixels."'))
    # Beside the original, so its module is still importable from there.
    p = (MANIFEST.parent / "forked-orienta-addon.toml")
    p.write_text(src, encoding="utf-8")
    return p


def test_a_run_under_another_addons_key_is_refused_and_records_nothing(
        tmp_path, request):
    """The credit defect, at the one place it would have been written down.

    Whoever registers first owns the key. The second add-on's run must not be
    stamped onto the result under a declaration that cites someone else.
    """
    from backend.api.services.addons.citations_bridge import (
        register_manifest_citations,
    )

    twin_path = _twin_manifest(tmp_path)
    request.addfinalizer(lambda: twin_path.unlink(missing_ok=True))
    register_manifest_citations(parse_manifest(MANIFEST))    # runnable first

    result = _Result()
    with pytest.raises(AddonError) as excinfo:
        run_analysis(parse_manifest(twin_path), "addon.mean_quality",
                     _ctx(), {}, result=result)

    message = str(excinfo.value)
    assert message.startswith("forked")          # the add-on's own name first
    assert "runnable" in message                 # and who holds the key
    assert get_steps(result) == []               # nothing was recorded


def test_a_key_conflict_is_not_counted_as_a_crash(tmp_path, request):
    """Two installations disagreeing about a key is a configuration problem,
    not the add-on misbehaving — counting it would walk a perfectly good
    add-on towards the automatic disable at CRASH_LIMIT."""
    from backend.api.services.addons.citations_bridge import (
        register_manifest_citations,
    )

    twin_path = _twin_manifest(tmp_path)
    request.addfinalizer(lambda: twin_path.unlink(missing_ok=True))
    register_manifest_citations(parse_manifest(MANIFEST))

    trust = TrustStore(tmp_path / "trust.json")
    with pytest.raises(AddonError):
        run_analysis(parse_manifest(twin_path), "addon.mean_quality",
                     _ctx(), {}, trust=trust)
    assert trust.crashes("forked") == 0


# --- a recorded default must be the value the code actually used ------------

def _variant_manifest(request, *replacements):
    """A manifest beside ``runnable_addon``'s, so its module still imports.

    Not ``orienta-addon.toml``: discovery matches that name exactly, so a
    variant written here is invisible to every test that scans this folder.
    """
    src = MANIFEST.read_text(encoding="utf-8")
    for old, new in replacements:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    path = MANIFEST.parent / "variant-orienta-addon.toml"
    path.write_text(src, encoding="utf-8")
    request.addfinalizer(lambda: path.unlink(missing_ok=True))
    return parse_manifest(path)


def test_a_default_the_function_does_not_take_is_not_recorded(request, caplog):
    """The manifest and the code are each valid on their own, and only their
    JOIN is wrong -- this branch's dominant defect shape.

    The runner passes only what the caller sent, so a declared parameter the
    function does not accept never reaches it and never raises. Its declared
    default was written into the provenance trail all the same: a scientific
    record asserting a value that no line of the run ever saw.
    """
    manifest = _variant_manifest(request, (
        '[analyses.params_schema.properties.scale]',
        '[analyses.params_schema.properties.invented]\ntype = "number"\n'
        'default = 7.0\n\n[analyses.params_schema.properties.scale]'))
    with caplog.at_level(logging.WARNING, logger=RUNNER_LOGGER):
        run = run_analysis(manifest, "addon.mean_quality", _ctx(), {})
    assert "invented" not in run.recorded_params
    assert "invented" in caplog.text


def test_a_default_that_disagrees_with_the_code_records_what_ran(request,
                                                                 caplog):
    """``scale`` defaults to 1.0 in the signature. A manifest claiming 9.0
    describes a run that used 1.0 -- and the sentence names {scale}."""
    manifest = _variant_manifest(request, ("default = 1.0", "default = 9.0"))
    with caplog.at_level(logging.WARNING, logger=RUNNER_LOGGER):
        run = run_analysis(manifest, "addon.mean_quality", _ctx(), {})
    assert run.recorded_params["scale"] == 1.0
    assert "9.0" in caplog.text and "scale" in caplog.text


def test_an_agreeing_default_is_still_recorded(request):
    """The ordinary case, and the reason the middle layer exists at all: the
    most common call there is, the button with nothing typed into it."""
    run = run_analysis(parse_manifest(MANIFEST), "addon.mean_quality",
                       _ctx(), {})
    assert run.recorded_params["scale"] == 1.0
    assert run.recorded_params["fail"] is False


def test_a_sent_value_still_outranks_both(request):
    run = run_analysis(parse_manifest(MANIFEST), "addon.mean_quality",
                       _ctx(), {"scale": 3.0})
    assert run.recorded_params["scale"] == 3.0
