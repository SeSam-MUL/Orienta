"""Which pipeline step is cited with what, and how it describes itself.

Declarations are centralised here rather than living beside each step, because
the owning modules (routes/indexing.py, backend/spherical_gpu/) would import
this module for record_step() and make the import circular. The anti-drift
property is kept by validating keys at call time in provenance.record_step():
an undeclared key is caught by RUNNING the pipeline, not by parsing source.

Note on ``indexing.dictionary``: dictionary indexing uses two things that need
two different citations. ``singh-degraef-di`` (Singh & De Graef, 2016,
"Orientation sampling for dictionary-based diffraction pattern indexing
methods") is the algorithm — how the orientation dictionary is sampled and
matched, per NOTICE.md's attribution. ``emsoft`` (Callahan & De Graef, 2013)
is the Monte-Carlo / dynamical master-pattern simulation that the dictionary's
patterns are rendered from. It is not a citation for the matching algorithm
itself, so ``indexing.spherical`` (which uses the same master patterns but a
different, Lenthe-et-al. matching method) keeps ``emsoft`` too — one entry, two
correct and distinct roles.

Note on the two spherical keys. Orienta can reach a spherical-harmonic result
two ways, and they are different facts about who did the work.
``indexing.spherical`` is Orienta's own GPU implementation
(``backend/spherical_gpu/``, recorded from ``indexing_controller``'s
``spherical_gpu_index_patterns``). ``indexing.spherical_emsphinx`` is the
DEFAULT backend (see ``resolve_spherical_backend``): Orienta shells out to the
authors' own ``IndexEBSD`` binary in WSL, so the numbers are EMSphInx's and the
sentence must say so rather than claim an implementation Orienta did not do for
that run. Both cite ``emsoft``, in the same role as ``indexing.dictionary``
does: the ``.sht`` master patterns both consume are simulated by EMsoft's
``EMEBSDmasterSHT`` — a citation for the simulated references, not for the
matching method.

Which keys are WIRED (i.e. some code path calls ``record_step`` with them):
``indexing.hough``, ``indexing.dictionary``, ``indexing.spherical``,
``indexing.spherical_emsphinx``, ``eds.chemistry_prior``,
``eds.particle_rescue``, ``pseudosym.resolver`` and
``refinement.orientation``.

``preprocessing.background`` is DECLARED BUT DELIBERATELY NEVER RECORDED.
Background removal happens in the EBSD viewer, in place on the loaded signal,
before and independently of any indexing run; no result object exists at that
moment to record it on, and reconstructing it afterwards would be a guess. The
declaration is kept because the recorded-but-undeclared path is the honest
failure mode (see ``record_step``) and because the fixtures and the ``.h5``
round-trip exercise it — but an add-on author reading this registry as the
public contract must not read it as "the app records this". It does not. If a
future preprocessing step wants a citation, it has to record it itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class StepCitation:
    key: str
    label: str
    citation_ids: Tuple[str, ...]
    sentence: str                       # str.format template, named slots only
    no_citation_reason: Optional[str] = None
    #: WHICH ADD-ON DECLARED THIS, or ``None`` for a step of Orienta's own.
    #:
    #: A declaration carries the citations and the methods sentence, so the
    #: registry answers "who gets the credit for a run recorded under this
    #: key?". Without an owner, "this key is already declared" and "this key
    #: is already declared BY SOMEONE ELSE" are the same question, and only
    #: the second decides whether a registration may proceed: re-registering
    #: is how every run after an add-on's first arrives, while a SECOND
    #: add-on on one key would lend the first author's DOI and sentence to
    #: another author's numbers. ``citations_bridge`` reads this field and
    #: refuses that case rather than keeping the first silently.
    owner: Optional[str] = None


_STEPS = (
    StepCitation(
        key="indexing.hough",
        label="Hough indexing",
        citation_ids=("orienta", "pyebsdindex", "kikuchipy", "orix"),
        sentence=(
            "Orientations were determined by Hough/Radon indexing "
            "(PyEBSDIndex) as implemented in Orienta {orienta_version}."
        ),
    ),
    StepCitation(
        key="indexing.dictionary",
        label="Dictionary indexing",
        citation_ids=("orienta", "kikuchipy", "singh-degraef-di", "emsoft", "orix"),
        sentence=(
            "Orientations were determined by dictionary indexing against "
            "{dict_size} simulated patterns on a {angular_step_deg}deg "
            "orientation grid, as implemented in Orienta {orienta_version}."
        ),
    ),
    StepCitation(
        key="indexing.spherical",
        label="Spherical-harmonic indexing",
        citation_ids=("orienta", "lenthe2019", "emsoft", "orix"),
        sentence=(
            "Orientations were determined by spherical-harmonic-transform "
            "indexing at bandwidth {bandwidth}, an independent GPU "
            "reimplementation of the method of Lenthe et al. (2019) in "
            "Orienta {orienta_version}."
        ),
    ),
    StepCitation(
        key="indexing.spherical_emsphinx",
        label="Spherical-harmonic indexing (EMSphInx)",
        citation_ids=("lenthe2019", "orienta", "emsoft", "orix"),
        sentence=(
            "Orientations were determined by spherical-harmonic-transform "
            "indexing at bandwidth {bandwidth} using EMSphInx "
            "(Lenthe et al., 2019), invoked by Orienta {orienta_version}."
        ),
    ),
    StepCitation(
        key="eds.chemistry_prior",
        label="EDS chemistry prior",
        citation_ids=("orienta", "orienta-eds-prior"),
        sentence=(
            "Per-pixel EDS composition was applied as a soft per-phase prior "
            "on the phase assignment (strengths {strength_by_phase}), "
            "adjusting {n_adjusted} pixels."
        ),
    ),
    StepCitation(
        key="eds.particle_rescue",
        label="EDS particle rescue",
        citation_ids=("orienta", "orienta-eds-prior"),
        sentence=(
            "Particles below the EDS interaction volume were resolved by "
            "orientation continuity against the surrounding matrix, "
            "reassigning {n_changed} pixels."
        ),
    ),
    StepCitation(
        key="pseudosym.resolver",
        label="Pseudo-symmetry resolution",
        citation_ids=("orienta", "orix"),
        sentence=(
            "Pseudo-symmetric orientation variants were resolved by "
            "render-based arbitration against the candidate classes of the "
            "phase point group."
        ),
    ),
    StepCitation(
        key="preprocessing.background",
        label="Pattern preprocessing",
        citation_ids=("kikuchipy",),
        sentence=(
            "Patterns were preprocessed before indexing "
            "(background removal: {background})."
        ),
    ),
    StepCitation(
        key="refinement.orientation",
        label="Orientation refinement",
        citation_ids=("orienta", "kikuchipy"),
        sentence=(
            "Orientations were refined against simulated patterns "
            "({n_refined} pixels)."
        ),
    ),
)

STEP_REGISTRY: Dict[str, StepCitation] = {s.key: s for s in _STEPS}


def get_step(key: str) -> Optional[StepCitation]:
    return STEP_REGISTRY.get(key)


def register_step(step: StepCitation) -> None:
    """Add a step declaration. The hook add-ons will use."""
    STEP_REGISTRY[step.key] = step
