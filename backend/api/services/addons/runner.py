"""The one place add-on code is imported and called.

Everything else in this package reads files. This module is the boundary, and
it is deliberately small: make the add-on importable, import the named
callable, check the module that came back is actually the add-on's, call it
with the parameters it declared, validate what came back, record the step, and
on any failure of the add-on's own name the add-on before anything else.

That last part is not politeness. Without it an add-on's traceback arrives in a
diagnostics zip looking like an Orienta bug, and the wrong person spends the
afternoon on it. It is also why the citation and provenance calls are INSIDE
the guard: they can raise too (``record_step`` raises under pytest on an
undeclared key or a path-looking param), and a raw ValueError from in here
names nobody. The crash COUNTER is guarded for the same reason one layer in:
the trust store is a file, and an unwritable one raising from inside the guard
would replace the add-on failure with an unattributed one from our own
bookkeeping -- see ``_note_crash``.

WHAT THE GUARD DOES NOT CATCH
-----------------------------
An earlier draft of this docstring said "on ANY failure". That was not true,
and an overclaiming docstring in the one module whose job is containment is
worse than a modest one — ``context.py`` admits in the same way that its
read-only flag is a guard against accident, not a boundary. In process, these
escape and nothing here can name the add-on:

* ``KeyboardInterrupt`` — deliberately. That is the operator stopping the
  process, not the add-on misbehaving, and turning it into a failure row
  against someone's add-on would be a lie. ``SystemExit`` USED to be let
  through beside it and is not any more; see ``_exit_detail``;
* ``os._exit``, a segfault or a stack overflow in a C extension the add-on
  imported — the process is gone before any ``except`` runs;
* an add-on that never returns. There is no timeout: interrupting arbitrary
  Python mid-call cannot be done safely in the calling thread;
* memory exhaustion of the whole process;
* anything an add-on raises LATER — from a thread it started, an ``atexit``
  hook, or a callback it registered — which surfaces long after this function
  returned and is attributed to whatever was running then.

Every one of these closes at a separate-process boundary, and keeping that
move possible without changing a single add-on is what the numpy-only,
no-open-handles contract in ``context.py`` and ``outputs.py`` is for.

RE-SCAN, and why this module does not clear anything
----------------------------------------------------
``citations_bridge`` left the decision here: both registries it writes to are
append-only for the life of the process, so an add-on re-scanned after its
manifest changed keeps the declaration it had at FIRST registration.

This module keeps it that way, on purpose. A step's declaration is what an
already-recorded result's methods paragraph is rendered FROM — ``render_methods``
calls ``get_step`` at render time, so the sentence lives only in
``STEP_REGISTRY`` and never in the result. Swapping a declaration mid-session
would therefore retroactively change what a run that already happened claims to
have done, which is the one thing ``provenance`` refuses to do; and clearing one
would make ``record_step`` raise (strict) or warn (production) for a key the
user's result already carries.

So: the runner registers the manifest it is about to run and nothing else.
A changed manifest takes effect on restart. Whoever builds enable/disable owns
the user-facing half of that — it is the only place a user action exists that
could legitimately invalidate a declaration, and it is also the only place with
somewhere to SAY so.
"""
from __future__ import annotations

import importlib
import inspect
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..citations.provenance import record_step
from .citations_bridge import AnalysisKeyConflict, register_manifest_citations
from .context import AddonContext
from .manifest import (
    AddonManifest,
    AnalysisSpec,
    declared_params,
    recordable_params,
)
from .outputs import AddonOutput, OutputError, ScalarOutput, validate_outputs

logger = logging.getLogger(__name__)


class AddonError(Exception):
    """An add-on failed. Its name comes first in the message, always."""

    def __init__(self, addon_name: str, analysis_key: str, detail: str):
        self.addon_name = addon_name
        self.analysis_key = analysis_key
        super().__init__(f"{addon_name} / {analysis_key}: {detail}")


@dataclass(frozen=True)
class AddonRunResult:
    outputs: Tuple[AddonOutput, ...]
    #: Names the caller sent that this analysis does not declare. Reported,
    #: never silently dropped: a run that quietly ignored a setting the user
    #: made is as dishonest as one that recorded something they never sent.
    ignored_params: Tuple[str, ...]
    #: Exactly what was written to the provenance trail.
    recorded_params: Dict[str, Any]


def _describe(exc: BaseException) -> str:
    """``type: message``, for an exception whose message cannot be trusted.

    ``f"{exc}"`` runs the ADD-ON's ``__str__``, which is add-on code like any
    other and can raise — from inside the very ``raise AddonError(...)`` that
    exists to stop add-on code escaping unnamed. ``type(exc).__name__`` is a
    class attribute and is safe.
    """
    try:
        return f"{type(exc).__name__}: {exc}"
    except Exception:       # noqa: BLE001 - the whole point is to catch it
        return (f"{type(exc).__name__}: <the exception's own __str__ raised, "
                "so it has no readable message>")


def _exit_detail(exc: SystemExit) -> str:
    """Why an add-on's ``SystemExit`` is a failure of the add-on, not an exit.

    ``SystemExit`` is a ``BaseException``, so a plain ``except Exception``
    misses it — and it is far more reachable than "an add-on called
    ``sys.exit()``" suggests. ``argparse.ArgumentParser.parse_args`` exits with
    code 2 on bad input rather than raising, so an add-on that parses anything
    internally, or that has a script-shaped error branch, ends the CALLER's
    request. Unattributed, and looking like Orienta gave up.

    ``KeyboardInterrupt`` stays uncaught, and the asymmetry is the point: that
    one really is the operator. ``discovery.py`` refuses to catch either, and
    is right to — discovery imports no add-on code, so nothing raising there
    can be an add-on's doing. Here it can, which is the whole difference.

    The exit code is carried into the message: an author whose ``parse_args``
    exited will recognise the 2, and nothing else in the traceback says it.
    """
    code = getattr(exc, "code", None)
    where = (f"SystemExit with code {code!r}" if code is not None
             else "SystemExit with no code")
    return (
        f"the add-on ended the process instead of failing: {where}. An add-on "
        "is a library inside a running server, not a script: it must RAISE on "
        "failure. Something it called may have done this on its behalf — "
        "argparse's parse_args() exits with code 2 on bad input rather than "
        "raising.")


def _ensure_importable(manifest: AddonManifest) -> Path:
    """Put the add-on's own directory on sys.path and return it.

    Without this a FOLDER-installed add-on — the half the README documents —
    can never be imported: nothing else in the process knows where
    ``~/.orienta/addons/<name>/`` is.

    APPENDED, never prepended: an add-on must not be able to shadow the
    stdlib or site-packages by naming a module after one. ``invalidate_caches``
    is required because a path added after the import machinery cached that
    directory listing is otherwise not seen.

    The decision is made against ``sys.path`` itself — the list the import
    machinery actually reads — and against no second copy of that fact, so it
    is self-healing: if anything removed the root after we added it (pytest's
    ``monkeypatch.syspath_prepend`` restores ``sys.path`` wholesale on undo),
    the next run puts it back instead of failing with a ``ModuleNotFoundError``
    that would name the add-on for our own bookkeeping.
    """
    root = Path(manifest.source_path).resolve().parent
    if str(root) not in sys.path:
        sys.path.append(str(root))
        importlib.invalidate_caches()
    return root


def _check_module_identity(manifest: AddonManifest, analysis: AnalysisSpec,
                           module, root: Path) -> None:
    """Refuse a module that is not this add-on's, however it got that name.

    Appending the root makes an add-on importable; it does not make the name
    the add-on's. A top-level module name is a single, process-wide namespace
    shared by every add-on, the stdlib, every installed package and Orienta's
    own top-level packages, and ``import_module`` answers from ``sys.modules``
    first — so:

    * two add-ons that each ship ``myaddon.py`` resolve to the SAME module.
      Whichever ran first wins, and the second one's run returns the first
      one's outputs, with no error, attributed to the second and cited as the
      second. In a system built so an author gets academic credit, that hands
      one author's credit to another inside text a researcher pastes into a
      paper. It is the worst thing this module can do and it is silent;
    * an add-on whose module is called ``analysis`` resolves to ORIENTA's
      ``analysis`` package, and the add-on is then blamed for whatever that
      import does.

    The test is containment, not equality of the parent directory: a folder
    add-on may perfectly well ship ``mypkg/__init__.py`` rather than a flat
    module, and then the module's own directory is a level BELOW the root.

    KNOWN NARROWING, deliberate: an add-on installed by entry point whose
    manifest sits in a SUB-directory of its package (``my_addon:data/
    orienta-addon.toml``) has a root below its module and is refused. The
    message names both paths and the remedy is to put the manifest at the
    package root, which is the form ``discovery`` documents. Refusing a legal
    layout loudly is the right side to err on when the alternative is
    misattributing authorship quietly.
    """
    location = getattr(module, "__file__", None)
    if not location:
        raise AddonError(
            manifest.name, analysis.key,
            f"{analysis.python_name!r} resolved to a built-in or namespace "
            f"module with no file, which cannot be this add-on's code. Rename "
            f"the module to something that cannot collide, e.g. "
            f"{manifest.name}_{module.__name__}.")
    directory = Path(location).resolve().parent
    if directory != root and root not in directory.parents:
        raise AddonError(
            manifest.name, analysis.key,
            f"{analysis.python_name!r} resolved to {location}, which is not "
            f"inside this add-on at {root}. The module name is taken by other "
            f"code already in this process — another add-on's module, an "
            f"installed package, or one of Orienta's own. Rename it to "
            f"something that cannot collide, e.g. "
            f"{manifest.name}_{module.__name__}.")


def _resolve(manifest: AddonManifest, analysis: AnalysisSpec):
    root = _ensure_importable(manifest)
    module_name, _, attribute = analysis.python_name.partition(":")
    module = importlib.import_module(module_name)
    _check_module_identity(manifest, analysis, module, root)
    return getattr(module, attribute)


def probe_import(manifest: AddonManifest) -> None:
    """Import every analysis this add-on declares, or raise ``AddonError``.

    This is what "an add-on that raises on activation is disabled and says so"
    means for a contract whose only entry point is a named callable: there is
    no activate() hook to call, so activation imports what a run would import.
    It runs on an explicit user action only — never during discovery.
    """
    for analysis in manifest.analyses:
        try:
            func = _resolve(manifest, analysis)
        except AddonError:
            # Already named, by the check that knows why. Re-wrapping would
            # print this add-on's name twice and bury the reason behind it.
            raise
        except (Exception, SystemExit) as exc:
            # SystemExit at IMPORT time is the script-shaped module: a bare
            # parse_args() or sys.exit() at module scope. Activation is exactly
            # where an author meets it, so it must be named here too.
            logger.exception("add-on %s could not be imported", manifest.name)
            detail = (_exit_detail(exc) if isinstance(exc, SystemExit)
                      else _describe(exc))
            raise AddonError(
                manifest.name, analysis.key,
                f"could not import {analysis.python_name!r}: "
                f"{detail}") from exc
        if not callable(func):
            raise AddonError(manifest.name, analysis.key,
                             f"{analysis.python_name!r} is not callable")


def _split_params(analysis: AnalysisSpec,
                  params: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any],
                                                             Tuple[str, ...]]:
    """(what the manifest declares, what it does not)."""
    properties = declared_params(analysis)
    accepted: Dict[str, Any] = {}
    ignored = []
    for name, value in (params or {}).items():
        if name in properties:
            accepted[name] = value
        else:
            ignored.append(name)
    return accepted, tuple(sorted(ignored))


def _note_crash(trust, name: str) -> None:
    """Count a failure against an add-on, and never instead of reporting it.

    Every ``except`` clause below calls this, and it used to call
    ``trust.record_crash`` bare. The trust store is a file: a read-only
    directory, a full disk or another process holding it open makes that raise
    -- and it raises from INSIDE the guard whose whole job is to make sure the
    add-on name comes first, so the add-on failure the user could act on was
    replaced by an unattributed one from Orienta own bookkeeping, on the one
    path where something had already gone wrong.

    ``Exception`` and not an enumeration, for ``routes/addons._record`` reason:
    the first guess at which exception was wrong on first contact with reality,
    and attribution must not depend on guessing right. The traceback is logged,
    so nothing is hidden by catching broadly. ``BaseException`` stays uncaught:
    ``KeyboardInterrupt`` is the operator.

    The cost of swallowing is named: a crash that could not be counted does not
    move the add-on towards the automatic disable at ``CRASH_LIMIT``. That is
    the right trade -- the counter is a convenience, the attribution is the
    contract -- and it is in the log either way.
    """
    if trust is None:
        return
    try:
        trust.record_crash(name)
    except Exception:
        logger.exception(
            "could not record a crash against the add-on %s; the failure is "
            "still reported, but it did not count towards disabling it", name)


def _note_success(trust, name: str) -> None:
    """Reset the crash counter, and never turn a good run into a failure.

    The same unguarded call as ``_note_crash``, on the other path and with a
    worse outcome: here the add-on ran, validated and produced its outputs, and
    an unwritable trust store would have thrown all of it away with a 500 that
    named nobody.
    """
    if trust is None:
        return
    try:
        trust.clear_crashes(name)
    except Exception:
        logger.exception(
            "could not clear the crash count of the add-on %s; its outputs "
            "are unaffected", name)


def _scalar_outputs(outputs) -> Dict[str, Any]:
    """Scalar results, for the sentence. Maps and tables are not sentence
    material and ``record_step`` would only record a map's shape anyway."""
    return {o.key: o.value for o in outputs if isinstance(o, ScalarOutput)}


def _declared_defaults(analysis: AnalysisSpec, allowed: set,
                       func) -> Dict[str, Any]:
    """What a parameter the caller left alone actually ran with.

    Only what the caller SENT used to be recorded, and a parameter left at its
    default is never sent -- so the methods paragraph of the most ordinary
    call there is, the button with nothing typed into it, read
    ``"[n_components not recorded] Gaussian components"``. Measured on the
    reference add-on before this existed. The trail was silent about the one
    run every user makes first.

    It belongs HERE and not in each add-on. The local workaround -- emit the
    parameter as a ``ScalarOutput`` too -- only works for a parameter whose
    value is a number: a string ``method = "crofton"`` or a boolean has no
    ``ScalarOutput`` form at all, so "do it for every parameter your sentence
    names" is advice an author cannot follow. One place, every add-on.

    IT OUTRANKS THE ADD-ON'S OWN SCALAR of the same name, and that is
    deliberate. A default is an INPUT fact -- it is what the parameter WAS --
    and this module's rule is already that inputs outrank what the add-on
    reports about them. ``tests/addons/fixtures/runnable_addon`` reports
    ``scale`` as a hundred times its input on purpose; recording 100.0 for a
    run that used 1.0 would be a false statement about the run, in text
    written to be pasted into a manuscript.

    ``allowed`` is applied here and not afterwards: a ``format = "path"``
    parameter is excluded from the trail BY DECLARATION, and a default is
    exactly as capable of being a local path as a sent value is.

    A schema with no ``default`` contributes nothing. The absence stays an
    absence -- ``render_methods`` names it honestly as "[x not recorded]" --
    because a guessed value in a citation trail is worse than a stated gap.

    CHECKED AGAINST THE CALLABLE, which this module is holding anyway. A
    manifest and a signature are each perfectly valid on their own and only
    their JOIN can be wrong, which is this branch's dominant defect shape; and
    the schema is the half nothing executes, so nothing else can catch its
    drift. Two ways it drifts, both silent before this:

    * the schema declares a parameter the function does not take. The runner
      passes only what the CALLER sent, so nothing raises -- and the declared
      default went into the methods paragraph as a fact about a run in which
      no line ever saw it;
    * the schema's default and the signature's disagree. The caller sent
      nothing, so the SIGNATURE's value is what ran. Recording the schema's
      would state a number the analysis was not given.

    So the value recorded is the signature's, which is the value that ran, and
    a disagreement is logged rather than resolved in the manifest's favour.
    The one-line summary of the rule: the trail records the RUN, and the only
    witness to the run is the code that performed it.

    A callable whose signature cannot be read (a C function, some builtins)
    contributes nothing, with a warning. That is the honest direction: without
    a signature there is no witness, and "not recorded" is readable where a
    number taken on the manifest's word is not.
    """
    declared = {
        name: schema["default"]
        for name, schema in declared_params(analysis).items()
        if name in allowed and isinstance(schema, dict) and "default" in schema
    }
    if not declared:
        return {}
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        logger.warning(
            "add-on %s: the signature of %r cannot be read, so the declared "
            "defaults %s are left out of the provenance trail rather than "
            "recorded on the manifest's word",
            analysis.key, getattr(func, "__name__", func),
            ", ".join(sorted(declared)))
        return {}

    ran: Dict[str, Any] = {}
    for name, value in declared.items():
        parameter = parameters.get(name)
        if parameter is None or parameter.default is inspect.Parameter.empty:
            logger.warning(
                "add-on %s: the manifest declares a default for %r, which "
                "%r does not take as a parameter with a default of its own. "
                "The value the caller left alone never reached the analysis, "
                "so it is NOT recorded; the methods paragraph will name it as "
                "not recorded, which is true.",
                analysis.key, name, getattr(func, "__name__", func))
            continue
        try:
            differs = bool(parameter.default != value)
        except Exception:       # noqa: BLE001 - an add-on's own __eq__
            differs = True
        if differs:
            logger.warning(
                "add-on %s: the manifest declares %r = %r but %r runs with "
                "%r when the caller sends nothing. Recording what ran.",
                analysis.key, name, value,
                getattr(func, "__name__", func), parameter.default)
        ran[name] = parameter.default
    return ran


def run_analysis(manifest: AddonManifest, key: str, context: AddonContext,
                 params: Optional[Dict[str, Any]] = None, *,
                 result=None, trust=None) -> AddonRunResult:
    """Run one analysis of one add-on. The only importer of add-on code."""
    analysis = next((a for a in manifest.analyses if a.key == key), None)
    if analysis is None:
        raise AddonError(manifest.name, key,
                         f"{manifest.name} declares no analysis {key!r}")

    accepted, ignored = _split_params(analysis, params)
    if ignored:
        logger.warning("add-on %s / %s: ignoring undeclared parameter(s) %s",
                       manifest.name, key, ", ".join(ignored))

    try:
        func = _resolve(manifest, analysis)
    except AddonError:
        _note_crash(trust, manifest.name)
        raise               # already named; see probe_import
    except (Exception, SystemExit) as exc:
        _note_crash(trust, manifest.name)
        logger.exception("add-on %s could not be imported", manifest.name)
        detail = (_exit_detail(exc) if isinstance(exc, SystemExit)
                  else _describe(exc))
        raise AddonError(
            manifest.name, key,
            f"could not import {analysis.python_name!r}: {detail}") from exc

    recorded: Dict[str, Any] = {}
    try:
        returned = func(context, **accepted)
        outputs = validate_outputs(returned, shape=context.shape)

        # Inside the guard, all of it. A bookkeeping failure here is still an
        # add-on failure and must still name the add-on.
        allowed = set(recordable_params(analysis))
        # Three layers, weakest first. Scalar results, then the DEFAULTS of
        # declared parameters the caller left alone, then what the caller
        # actually sent. See _declared_defaults for why the middle layer has
        # to exist and why it outranks the add-on's own report.
        recorded = _scalar_outputs(outputs)
        recorded.update(_declared_defaults(analysis, allowed, func))
        recorded.update({k: v for k, v in accepted.items() if k in allowed})
        # ABOVE the result check, deliberately: an add-on that ran and was
        # handed nothing to stamp has still run, and its works still belong in
        # the bibliography. Registering only alongside record_step left the
        # citation panel unable to name an add-on until its first recorded run.
        register_manifest_citations(manifest)
        if result is not None:
            # replace=True, and it is this module's call alone that passes it.
            # An add-on can be re-run on one result as often as its author
            # likes -- tuning a parameter is the ordinary case -- and the map
            # store keys on (name, result_id, analysis_key, key), so the later
            # run REPLACED the earlier outputs. The trail has to agree with
            # the store about which run is current; appending left a methods
            # paragraph claiming both parameter values at once. A core step
            # keeps appending, which is right for it: see record_step.
            record_step(result, key, recorded, replace=True)
    except OutputError as exc:
        _note_crash(trust, manifest.name)
        # Deliberately NOT the generic clause. An OutputError is already a
        # finished, author-facing sentence with a remedy in it, so it is passed
        # through verbatim rather than prefixed with its own class name; and it
        # is a malformed RETURN, not a crash, so it gets no traceback in the
        # log. Both differences are pinned by tests.
        raise AddonError(manifest.name, key, str(exc)) from exc
    except AnalysisKeyConflict as exc:
        # NOT counted as a crash, and no traceback. The add-on's code ran and
        # returned something valid; what failed is that two installations
        # claim one analysis key, which is a configuration fact about the
        # machine. Counting it would walk a working add-on towards the
        # automatic disable at CRASH_LIMIT for someone else's choice of key,
        # and a traceback would point at our own bookkeeping rather than at
        # the two manifests the user has to reconcile. The message already
        # names both; run_analysis stamps this add-on's name in front, and
        # nothing was written to the result or to either registry.
        logger.warning("add-on %s cannot register %s: %s",
                       manifest.name, key, exc)
        raise AddonError(manifest.name, key, str(exc)) from exc
    except SystemExit as exc:
        # A BaseException, so the clause below cannot see it, and reachable
        # without anyone writing sys.exit(): argparse exits on bad input. See
        # _exit_detail for why this is the add-on's failure and
        # KeyboardInterrupt is not.
        _note_crash(trust, manifest.name)
        logger.exception("add-on %s exited during %s", manifest.name, key)
        raise AddonError(manifest.name, key, _exit_detail(exc)) from exc
    except Exception as exc:
        # Exception, not BaseException: KeyboardInterrupt stays uncaught,
        # because that is the operator stopping the process and not the add-on
        # misbehaving. SystemExit is handled above, where it belongs.
        _note_crash(trust, manifest.name)
        logger.exception("add-on %s failed during %s", manifest.name, key)
        raise AddonError(manifest.name, key, _describe(exc)) from exc

    _note_success(trust, manifest.name)

    return AddonRunResult(outputs=outputs, ignored_params=ignored,
                          recorded_params=recorded)
