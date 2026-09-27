"""No recorded param may carry a local filesystem path.

C1. ``eds_phase_strengths`` is keyed by the phase FILE PATH on the wire, and
it was recorded verbatim — so the methods paragraph, which is written to be
pasted into a manuscript and is copied into every exported ``.h5``, printed
the operator's absolute path:

    ...(strengths C:/Users/<name>/.../Database/CIF_Library/Al.cif 0, ...)

A username and a directory layout, leaked to reviewers and co-authors. These
tests pin the general rule (nothing a step records may look like a path) and
the specific re-keying that fixes it.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from backend.api.routes.indexing import (
    _eds_prior_is_on,
    _eds_strengths_by_phase_name,
)
from backend.api.services.citations.provenance import get_steps
from backend.api.services.citations.render import render_methods

# A Windows drive letter ("C:\\", "e:/") anywhere in the value.
_DRIVE = re.compile(r"[A-Za-z]:[\\/]")


def path_like(text: str) -> bool:
    """Does this string look like a local filesystem path?"""
    return ("/" in text or "\\" in text
            or bool(_DRIVE.search(text)) or "Users" in text)


def leaks(value) -> list:
    """Every string inside ``value`` (keys included) that looks like a path."""
    found = []
    if isinstance(value, str):
        if path_like(value):
            found.append(value)
    elif isinstance(value, dict):
        for k, v in value.items():
            found.extend(leaks(str(k)))
            found.extend(leaks(v))
    elif isinstance(value, (list, tuple)):
        for v in value:
            found.extend(leaks(v))
    return found


def test_the_leak_detector_actually_bites():
    """The guard below is only worth having if it fails on the old shape."""
    old = {"strength_by_phase": {
        r"C:\Users\operator\Orienta\Database\CIF_Library\Al.cif": 0.0,
        "C:/Users/operator/Orienta/Database/CIF_Library/Si.cif": 1.0,
    }}
    assert len(leaks(old)) == 2


def _req(method="hough", **kw):
    base = dict(
        method=method, cif_paths=[], master_h5_paths=[], sht_paths=[],
        eds_phase_strengths={}, eds_expected_overrides={},
    )
    base.update(kw)
    return SimpleNamespace(**base)


WINDOWS_PATHS = [
    r"C:\Users\operator\Orienta\Database\CIF_Library\Al.cif",
    r"C:\Users\operator\Orienta\Database\CIF_Library\Si.cif",
]


def test_strengths_are_rekeyed_to_phase_names():
    req = _req(
        cif_paths=list(WINDOWS_PATHS),
        eds_phase_strengths={WINDOWS_PATHS[0]: 0.0, WINDOWS_PATHS[1]: 1.0},
    )
    out = _eds_strengths_by_phase_name(req)
    assert not leaks(out), out
    assert len(out) == 2
    # The value must still travel with its own phase, not just be scrubbed.
    by_zero = [k for k, v in out.items() if v == 0.0]
    by_one = [k for k, v in out.items() if v == 1.0]
    assert len(by_zero) == 1 and len(by_one) == 1
    assert "Al" in by_zero[0]
    assert "Si" in by_one[0]


def test_the_rendered_sentence_is_clean():
    """The actual text a manuscript would receive."""
    req = _req(
        cif_paths=list(WINDOWS_PATHS),
        eds_phase_strengths={WINDOWS_PATHS[0]: 0.0, WINDOWS_PATHS[1]: 1.0},
    )
    sentence = render_methods([{
        "key": "eds.chemistry_prior",
        "params": {
            "strength_by_phase": _eds_strengths_by_phase_name(req),
            "n_adjusted": 349,
        },
    }])
    assert not path_like(sentence), sentence
    assert "349 pixels" in sentence


def test_a_phase_of_the_run_without_a_strength_is_recorded_as_zero():
    """The run's phases, not the request dict's keys: the dict may be keyed
    to another method's files, and a silently missing phase would read as if
    it had not been in the run."""
    req = _req(
        cif_paths=list(WINDOWS_PATHS),
        eds_phase_strengths={WINDOWS_PATHS[1]: 1.0},
    )
    out = _eds_strengths_by_phase_name(req)
    assert len(out) == 2
    assert sorted(out.values()) == [0.0, 1.0]


def test_duplicate_derived_names_do_not_overwrite_each_other():
    dup = [WINDOWS_PATHS[0], WINDOWS_PATHS[0]]
    req = _req(cif_paths=dup, eds_phase_strengths={dup[0]: 0.5})
    out = _eds_strengths_by_phase_name(req)
    assert len(out) == 2, out
    assert not leaks(out), out


def test_prior_on_is_decided_by_configuration_not_by_effect():
    """I3: a prior that was on and flipped nothing is still a prior that ran."""
    on = _req(cif_paths=list(WINDOWS_PATHS),
              eds_phase_strengths={WINDOWS_PATHS[1]: 1.0})
    off = _req(cif_paths=list(WINDOWS_PATHS),
               eds_phase_strengths={WINDOWS_PATHS[1]: 0.0})
    assert _eds_prior_is_on(on) is True
    assert _eds_prior_is_on(off) is False
    assert _eds_prior_is_on(_req(cif_paths=list(WINDOWS_PATHS))) is False


# ---------------------------------------------------------------------------
# ...but "configured" alone claims the prior for runs that documentedly
# ignore it. The rule is configured AND applied.
# ---------------------------------------------------------------------------

SHT_PATHS = [
    r"C:\Users\operator\Orienta\Database\SHT_Library\Al.sht",
    r"C:\Users\operator\Orienta\Database\SHT_Library\Si.sht",
]


def test_multi_phase_spherical_on_a_non_gpu_backend_does_not_claim_the_prior():
    """The multi-phase dispatch warns, in these words, "EDS chemistry is only
    applied on the Spherical GPU backend - this run ignores it", and logs
    "running WITHOUT chemistry". Recording the prior for that run states a
    method that was not used."""
    req = _req("spherical", sht_paths=list(SHT_PATHS), backend="emsphinx",
               eds_phase_strengths={SHT_PATHS[1]: 1.0})
    assert _eds_prior_is_on(req) is False


def test_the_same_run_on_spherical_gpu_does_claim_it():
    """Same strengths, the backend that actually builds sph_weights."""
    req = _req("spherical", sht_paths=list(SHT_PATHS), backend="spherical_gpu",
               eds_phase_strengths={SHT_PATHS[1]: 1.0})
    assert _eds_prior_is_on(req) is True


def test_a_single_phase_run_does_not_claim_the_prior():
    """The prior reweights a phase COMPETITION; all three sites that apply it
    require more than one phase, so with one phase no weight matrix is ever
    built. _apply_particle_rescue refuses the same case."""
    one = _req(cif_paths=[WINDOWS_PATHS[0]],
               eds_phase_strengths={WINDOWS_PATHS[0]: 1.0})
    assert _eds_prior_is_on(one) is False


def test_the_gpu_run_records_n_adjusted_zero_rather_than_nothing():
    """The other half of the rule: on, applied, and it flipped no winner."""
    req = _req("spherical", sht_paths=list(SHT_PATHS), backend="spherical_gpu",
               eds_phase_strengths={SHT_PATHS[1]: 1.0})
    assert _eds_prior_is_on(req) is True
    sentence = render_methods([{
        "key": "eds.chemistry_prior",
        "params": {"strength_by_phase": _eds_strengths_by_phase_name(req),
                   "n_adjusted": 0},
    }])
    assert "adjusting 0 pixels" in sentence
    assert not path_like(sentence), sentence


# ---------------------------------------------------------------------------
# Defence in depth: record_step itself refuses a path-looking param
# ---------------------------------------------------------------------------

class _R:
    """Minimal record_step target."""

    metadata = None


def test_record_step_rejects_a_path_valued_param_under_strict():
    """Strict is on under pytest and off in production, so a future leak is
    a failing test here and never an aborted run there."""
    from backend.api.services.citations.provenance import record_step

    with pytest.raises(ValueError, match="local filesystem path"):
        record_step(_R(), "eds.chemistry_prior",
                    {"strength_by_phase": {WINDOWS_PATHS[0]: 0.0}})


def test_record_step_rejects_a_posix_path_in_a_plain_value_too():
    from backend.api.services.citations.provenance import record_step

    with pytest.raises(ValueError, match="local filesystem path"):
        record_step(_R(), "indexing.hough", {"source": "/home/x/scan.h5"})


def test_record_step_only_warns_when_strict_is_off(monkeypatch, caplog):
    """A production run must never die of bookkeeping."""
    import logging

    from backend.api.services.citations.provenance import get_steps, record_step

    monkeypatch.setenv("ORIENTA_CITATIONS_STRICT", "0")

    r = _R()
    with caplog.at_level(
        logging.WARNING, logger="backend.api.services.citations.provenance"
    ):
        record_step(r, "indexing.hough", {"source": WINDOWS_PATHS[0]})

    assert len(get_steps(r)) == 1
    assert any("local filesystem path" in rec.message for rec in caplog.records)


def test_the_params_the_app_actually_records_pass_the_guard():
    """The guard must not fire on ordinary params - versions, counts, phase
    names and formulas."""
    from backend.api.services.citations.provenance import get_steps, record_step

    r = _R()
    record_step(r, "indexing.dictionary",
                {"orienta_version": "0.3.0", "dict_size": 100347,
                 "angular_step_deg": 2.0})
    record_step(r, "eds.chemistry_prior",
                {"strength_by_phase": {"Al": 0.0, "Al7Cu2Fe": 1.0,
                                       "alpha-Al(Fe,Mn)Si": 1.0},
                 "n_adjusted": 0})
    assert len(get_steps(r)) == 2


@pytest.mark.integration
def test_a_real_run_records_no_path(indexed_result_hough):
    """End to end on a genuine Hough result: nothing it recorded is a path."""
    for step in get_steps(indexed_result_hough):
        assert not leaks(step["params"]), step
