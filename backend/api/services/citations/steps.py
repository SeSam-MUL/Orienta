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
