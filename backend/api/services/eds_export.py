"""The numbers leave the app, with everything needed to defend them.

WHY THIS EXISTS. Six user personas were asked what the EDS phase map needs.
Five of six ranked export above presets, unprompted, and all six raised the
same point in six different phrasings: *the settings have to live inside the
exported file*. A preset is an input; a record has to be attached to the
output. Until this module existed, a classified map could be looked at and
screenshotted, and the first question anyone asks about it — "what did you
do?" — had no answer that survived leaving the room.

WHY IT IS A PURE FUNCTION. :func:`build_tables` takes ``(state, at_maps,
geometry, options)`` and returns tables. No FastAPI, no ``h5_session``, no
module globals, no loaded file. Two reasons, both load-bearing:

  * a batch driver over a folder can call it without touching it, which is
    what let the batch feature be deferred honestly rather than silently
    omitted (spec section 2);
  * every number below is unit-testable against a hand-built state, which is
    how the tests check the arithmetic rather than checking that a route
    returns 200.

WHY THREE DENOMINATORS AND WHY THEY ARE IN THE COLUMN NAMES. A user said
flatly: "I will reject any column whose denominator is not in its own header."
He is right, and the three defensible denominators differ by tens of percent
on a real map — the whole exported raster, the pixels that carry a usable EDS
measurement, and the pixels that actually received a phase. A bare
``area_pct`` forces the reader to guess which one, and half of them guess
wrong. So: ``frac_of_scan``, ``frac_of_valid_eds``, ``frac_of_classified``,
each with its own count beside it so a binomial interval is recoverable.

Each fraction's NUMERATOR is drawn from the same set as its denominator. For
``frac_of_valid_eds`` that means the row's pixels that ALSO carry a usable EDS
measurement — ``n_px_valid_eds`` — and not the row's total ``n_px``. Dividing
one set by another was the original bug in this column: a row owning pixels
that carry no EDS measurement had its fraction inflated, and the column could
not sum to 1 across the phases. It sums to 1 now, exactly, because every valid
pixel lands in exactly one phase row (``unclassified`` included).

WHY ``not measured`` AND NEVER ``0``. An element that this scan never mapped
has no measurement, and "Cu 0.0 at%" is a false statement — it asserts that
copper was looked for and found absent. The literal string ``not measured``
goes in the cell. It is deliberately not a number: a spreadsheet that refuses
to average the column is telling the truth about it.

WHY THE MICROMETRE COLUMNS VANISH RATHER THAN DEGRADE. ``get_pixel_sizes()``
legitimately returns ``None``. A size distribution computed on a guessed
1 um/px when the scan was 0.2 is worse than a missing column, because it looks
like an answer. :func:`eds_particles.scale_to_um` raises rather than guessing;
this module catches that and drops every ``*_um*`` column, recording the
reason in the provenance.

WHY BOTH COMPOSITIONS FOR A PARTICLE. The EDS interaction volume during an
EBSD session is larger than the features being mapped, so the rim pixels of a
small particle are particle-plus-matrix mixtures. Averaging over all pixels
therefore drags every small particle toward the matrix and manufactures a
spurious "composition depends on particle size" trend. Both the all-pixel and
the one-pixel-eroded core composition are exported so the user can see the
effect and defend the choice.

WHY ``sd_within`` AND NEVER A STANDARD ERROR. The spread inside a region is
spatial heterogeneity plus counting noise, not the precision of its mean. The
pixels are strongly spatially correlated — the interaction volume exceeds the
step, and the classification box-averages on top of that — so ``sd/sqrt(n)``
would overstate the precision by one to two orders of magnitude. ``sd_within``
and ``n_px`` are exported and the user decides. The name says "within" for the
same reason: a column called ``uncertainty`` would be read as one.

WHY ``phase_auto`` SITS NEXT TO ``phase_final``. "Never overwrite the
software's answer... hiding a disagreement is what gets an expert report
discredited." Where a hand edit or a pinned definition moved a region off the
chemistry's own best candidate, both answers are in the file and
``phase_agrees`` says so in one column.

WHY ``phase_auto`` NOW COMES FROM THE SCORER THAT MADE THE MAP. It did not,
until 2026-08-27. The map is named by ``eds_clustering._match_labels`` ->
``chemistry_score.score_phase_ratio`` — with rule gating, a matrix-element
exemption and a ``min_score`` floor, on each region's eroded interior. The
export recomputed something else: an unweighted mean absolute deviation over
the union of element keys, ungated. Two metrics, and the second one's answer
was printed in a column called ``phase_auto`` beside a boolean called
``phase_agrees``. Measured on the real 90x120 SampleB scan (7 regions, 22
candidates, ZERO hand-painted pixels): 4 of 7 regions read
``phase_agrees = False``. A reader can only understand that as "a human moved
this one", and one did not. A tester drafted a Methods paragraph claiming a
manual override that never happened; another said a boolean reading "a human
changed this" and firing on 57 % of untouched regions is worse than no
boolean, and he is right. ``deciding_rank`` replays the real scorer —
verified to reproduce ``region_phase`` for 7 of 7 regions of that scan — and
``phase_agrees`` is computed against it. The MAD ranking survives under
``*_mad`` names, labelled as the second opinion it is, because a distance in
at% is easier to judge than a 0..1 score and because the region inspector
shows it. ``rank_candidates`` is still the inspector's exact definition and
is unchanged.

WHY EVERY OBJECT IS RANKED ON ITS OWN COMPOSITION. A particle used to inherit
its region's margin. On the real scan two particles of 286 px and 175 px both
printed ``0.366300``, the 660-pixel region's margin — and a number set beside
a 175-pixel object is read as describing it. Every particle now carries its
own ranking; the region's is still there, under the name
``region_margin_score``, which says whose it is.

WHY THE FILE NOW CHECKS ITSELF. A group leader read a finished export and
found the headline wrong: 61 % of an aluminium extrusion classified as
Fe-intermetallic by phases whose pixels carry a third of the iron those
phases are made of. Both numbers were already here, in adjacent columns, and
the file said nothing about the difference between them — "you are one
subtraction away from saying so". The plausibility section below is that
subtraction. It reports; it does not decide. See its header for the criterion
and for every threshold's measured justification.

WHY IT REPORTS THREE FRACTIONS AND NOT ONE. It reported the implausible half
alone, and a reader completes that by subtraction into "the rest is fine".
The rest is not fine, it is UNEXAMINED. Measured on the real 90x120 SampleB
scan (mode=cluster, scale_um=1.5, k=7): 62.33 % of the map sits on rows a
discriminating element contradicts, 37.67 % on rows chemistry cannot check at
all, and 0.00 % on rows a composition check corroborates. Nothing in that map
was corroborated, and the old wording invited the opposite conclusion while
printing neither of the two numbers that would have stopped it. The three
close on the whole raster — see :func:`plausibility_coverage` — so a reader
can see that nothing fell out rather than take it on trust.

WHY THE VERDICT SITS NEXT TO THE PHASE NAME AND NOT AT COLUMN BW. It was
appended at the end, which is free under "additions only at the end" and cost
no FORMAT_VERSION. It also put ``review_flag`` — the cell that says "this row
is questionable" — sixty columns to the right of anything a person looks at,
in the table a QA tester filters row by row on a weekly batch. A flag nobody
scrolls to is not a flag. It moved, and the version moved with it; see
:func:`plausibility_columns`.

WHY THE PICTURES HAVE A KEY IN THE TABLES. A tester matched colours to phases
by counting pixels in the PNG against ``frac_of_scan``: "that worked, but it
is a puzzle, not a legend. Two phases of similar area and I could not have
done it at all." ``phase_map_colour_hex`` and ``region_map_colour_hex`` come
from the same palette functions the renderers call, so the key cannot
disagree with the image; see :func:`colour_columns`.

WHY THE PICTURE IS AN ARTEFACT AND NOT A SCREENSHOT. Same reader: "no picture
told me what the map looked like, so I had no way to eyeball 'does 62 %
intermetallic look like my sample?' — a picture would have caught this in
four seconds." ``phase_map.png`` and ``region_map.png`` are written by
``phase_map_store``'s OWN renderers, the ones the app draws with, so the
picture in the folder is the picture on the screen rather than a second
drawing that can disagree with it.

Spec: docs/superpowers/specs/2026-08-27-eds-presets-and-export-design.md
      (sections 3, 6, 8, 9)
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from backend.api.services import eds_particles
from backend.api.services.eds_particles import (
    CONNECTIVITY,
    ParticleGeometry,
    composition_for_mask,
    find_particles,
    particle_masks,
    scale_to_um,
)

logger = logging.getLogger(__name__)

__all__ = [
    "FORMAT_VERSION",
    "NOT_MEASURED",
    "ExportGeometry",
    "ExportOptions",
    "ExportTables",
    "Table",
    "WrittenFile",
    "WriteResult",
    "build_tables",
    "write_export",
    "build_summary_line",
    "library_report",
    "read_acquisition_header",
    "kanaya_okayama_range_um",
    "phase_label",
    "formula_ascii",
    "deciding_rank",
    "rank_candidates",
    "to_scored_basis",
    "resolve_folder",
    "composition_plausibility",
    "discriminating_elements",
    "nominal_on_measured_basis",
    "background_at_pct",
    "plausibility_columns",
    "plausibility_coverage",
    "colour_columns",
    "REVIEW_REASONS",
]

#: Bumped when a column changes meaning or disappears. Additions at the end of
#: a table do NOT bump it — that is the whole point of "additions only at the
#: end": a downstream sheet that pins this version keeps working when a column
#: is appended, and breaks loudly when one is redefined.
#:
#: 2 (2026-08-27): ``phase_auto`` / ``phase_agrees`` now come from the scorer
#: that made the map instead of a second metric; ``gap_at_pct`` /
#: ``runner_up_gap_at_pct`` / ``margin_at_pct`` are replaced by ``score_auto``
#: / ``runner_up_score`` / ``margin_score`` (different unit AND a flipped sign
#: convention) with the old metric retained under ``*_mad`` names;
#: ``perimeter_um`` -> ``perimeter_crofton_um``; a particle's
#: ``margin_at_pct`` was its REGION's and is now ``region_margin_score``
#: beside the particle's own; ``assignment_source`` gained ``mixed`` and
#: ``untracked``. Every one of those is a redefinition, so the version moves.
#:
#: 3 (2026-08-27): ``particles.assignment_source`` no longer inherits its
#: region's hand/mixed grade — it is graded on the particle's OWN
#: ``n_px_hand`` (a QA tester measured 110 of 167 rows claiming ``mixed``
#: with ``n_px_hand = 0``). That is a redefinition, so the version moves.
#: Everything else in this version is an ADDITION and would not have moved
#: it on its own: ``chemistry_dissents`` on particles;
#: ``third_phase``/``third_score`` in the auto block;
#: ``phase_label``/``formula_ascii``; per-phase particle size summaries; and
#: the ``acquisition``, ``interaction_volume``, ``library``,
#: ``phases_absent``, ``coordinates``, ``peak_overlaps``, ``summary`` and
#: ``hand_edits.log`` blocks in the provenance.
#:
#: 4 (2026-08-27): the ten :func:`plausibility_columns` MOVED. They were
#: appended at the end of phases and regions; they now sit immediately behind
#: the row's phase label, and particles carries them too. A QA tester
#: measured ``review_flag`` at column BW of 76 — sixty columns right of
#: anything a person looks at, in the one table filtered row by row on a
#: weekly batch. Nothing was renamed and nothing changed meaning, so a reader
#: that selects by HEADER is unaffected; a reader that pins column POSITIONS
#: is not, and that is precisely what this number is for. Everything else in
#: this version is an addition and would not have moved it on its own:
#: ``review_flag``/``review_reasons`` on particles, the not-checkable and
#: corroborated coverage in ``provenance.plausibility``, ``legend.txt``, and
#: the ``warnings`` sheet in the workbook.
FORMAT_VERSION = 4

#: What goes in a composition cell for an element this scan never mapped.
#: Deliberately not a number. See the module docstring.
NOT_MEASURED = "not measured"

#: Carried into every export as literal text, because it is the single
#: sentence that decides how far these numbers can be pushed.
SEMI_QUANTITATIVE_CAVEAT = (
    "SEMI-QUANTITATIVE. The at% values in this export come from Aztec window "
    "integral counts converted with a simplified Cliff-Lorimer style "
    "k-factor model and renormalised over the measured elements only. There "
    "is no full ZAF correction, no standardless absorption/fluorescence "
    "correction and no k-factor calibration against a standard. Light "
    "elements are unreliable, carbon and oxygen are excluded from the "
    "scoring entirely, and an element that was not mapped cannot be seen at "
    "all — so the composition is renormalised over a set that may be "
    "incomplete. Use these numbers to distinguish phases, not to certify a "
    "composition."
)

#: Why the classification cannot be replayed from the exported map alone.
#: Applies to a map made before 2026-08-27, or restored from a sidecar
#: written before then: the result survived, the settings that made it did not.
_SETTINGS_NOT_PERSISTED_NOTE = (
    "This map carries no record of its own classification settings. It was "
    "made before those were persisted with the map, so the mode, the "
    "smoothing width and the requested cluster count are reported here only "
    "when the caller supplied them. A value this module could not establish "
    "is null, never a default standing in for the real one."
)

#: The good case: the map states how it was made, and this record is that
#: statement rather than an echo of what the export dialog was told.
_SETTINGS_RECORDED_NOTE = (
    "The classification settings below were recorded WITH the map at the "
    "moment it was classified — they are not the export caller's echo, and "
    "they survive a backend restart. Values the run resolved (the smoothing "
    "width in pixels, the cluster count it settled on) sit beside the values "
    "that were requested (a physical width in microns, or no count at all, "
    "meaning 'choose one')."
)

_DELIMITERS = {",", ";", "\t"}
_DECIMALS = {".", ","}


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

@dataclass
class ExportGeometry:
    """Where the raster came from and how big a pixel is.

    ``step_x_um``/``step_y_um`` are ``None`` when the file carries no usable
    header geometry — a real answer, and the trigger for dropping every
    physical column rather than inventing a scale.
    """

    n_rows: int
    n_cols: int
    step_x_um: Optional[float] = None
    step_y_um: Optional[float] = None
    source_path: Optional[str] = None
    #: Crop window in ORIGINAL-scan coordinates, when this raster is a crop.
    #: Free-form dict as the crop service reports it; passed through verbatim.
    crop_window: Optional[dict] = None
    #: The vendor's own name for the step unit, for the provenance record.
    step_units: str = "um"
    #: The EDS acquisition header, as :func:`read_acquisition_header` returns
    #: it. The EXPLICIT input: a caller that already has the file open should
    #: pass it here rather than making this module re-open the scan.
    #: ``None`` lets :func:`build_tables` read it from ``source_path`` once,
    #: which is what every current caller relies on — see
    #: ``ExportOptions.read_acquisition_header``.
    acquisition: Optional[dict] = None

    @property
    def n_px(self) -> int:
        return int(self.n_rows) * int(self.n_cols)

    @property
    def area_available(self) -> bool:
        for v in (self.step_x_um, self.step_y_um):
            if v is None:
                return False
            try:
                if not math.isfinite(float(v)) or float(v) <= 0.0:
                    return False
            except (TypeError, ValueError):
                return False
        return True


@dataclass
class ExportOptions:
    """Everything the caller decides. Recorded verbatim in the provenance."""

    connectivity: int = CONNECTIVITY
    #: Flags ``below_size_limit``; NEVER deletes a row. Silent exclusion was
    #: named a disqualifier by two personas independently, and it is the
    #: failure mode that quietly halves a particle count.
    min_particle_px: int = 0
    decimal: str = "."
    delimiter: str = ","
    #: Which artefacts to write. The four tables plus xlsx are on by default;
    #: the two heavy ones are opt-in.
    #: ``map_png`` is on by default and the two heavy ones are opt-in.
    artefacts: Dict[str, bool] = field(default_factory=lambda: {
        "phases": True, "regions": True, "particles": True,
        "definitions": True, "xlsx": True, "map_png": True,
        "labels": False, "pixels": False,
    })
    #: ``"suffix"`` appends ``_2``, ``_3``... to an existing folder;
    #: ``"fail"`` raises. Never overwrite — an export folder is evidence.
    on_existing: str = "suffix"
    #: Full sha256 of the source file. Costs a full read; on a 27 GB scan that
    #: is tens of seconds, so the caller can turn it off and the provenance
    #: then says so instead of quietly omitting it.
    hash_source: bool = True
    preset_name: str = ""
    #: The user's phase-colour overrides, exactly as the page holds them, so
    #: the exported PNG is the picture on the screen and not a second one in
    #: default colours. ``None`` means "whatever the palette gives", which is
    #: also what the page shows when nothing was overridden.
    colour_overrides: Optional[Dict[str, str]] = None
    #: ``CompatibilityReport.as_dict()`` when the user overrode a refusal.
    #: Its presence in the file IS the record of the override.
    compatibility: Optional[dict] = None
    #: Smoothing width, when the caller knows it. ``None`` omits the smoothed
    #: composition columns entirely rather than smoothing at a default that
    #: may not be the one the regions were formed on.
    smoothing_scale_px: Optional[int] = None
    #: ``_resolve_scale``'s report, verbatim, when the caller has one.
    scale_report: Optional[dict] = None
    #: Free-form echo of what the UI asked for, so "requested" can sit beside
    #: "resolved" in the provenance for values this module cannot re-derive.
    requested: Dict[str, Any] = field(default_factory=dict)
    #: Read the EDS acquisition header from ``geometry.source_path`` when
    #: ``geometry.acquisition`` is None. A batch driver that has already
    #: collected the headers sets this False and passes them in; the
    #: provenance records which route was taken either way.
    read_acquisition_header: bool = True
    #: The FULL CIF phase library, as ``load_cif_phase_library`` returns it
    #: (``{key: CifPhaseEntry}``). Explicit input for the same reason as
    #: above. ``None`` falls back to the default database path so that
    #: ``provenance.phases_absent`` is populated for callers that predate
    #: this argument.
    library: Optional[Mapping[str, Any]] = None
    #: Where ``library`` came from, for the identity block. Ignored when
    #: ``library`` is None and the default path is used.
    library_path: Optional[str] = None
    #: Stamped into the provenance as-is.
    app_version: Optional[dict] = None
    #: Overridable for deterministic tests.
    timestamp: Optional[datetime] = None

    def validate(self) -> None:
        """Raise ``ValueError`` on a combination that would write wrong numbers."""
        if self.decimal not in _DECIMALS:
            raise ValueError(
                f"decimal must be '.' or ',', got {self.decimal!r}")
        if self.delimiter not in _DELIMITERS:
            raise ValueError(
                f"delimiter must be ',', ';' or a tab, got {self.delimiter!r}")
        if self.decimal == "," and self.delimiter == ",":
            # Not a preference. "3,14" written into a comma-separated file is
            # two fields, and every column after it shifts by one — a German
            # Excel opening it reads a different number in every column.
            raise ValueError(
                "decimal ',' cannot be combined with delimiter ',': the "
                "decimal comma would split every number into two fields. "
                "Use ';' or a tab as the delimiter.")
        if self.connectivity not in (4, 8):
            raise ValueError(
                f"connectivity must be 4 or 8, got {self.connectivity!r}")
        if int(self.min_particle_px) < 0:
            raise ValueError(
                f"min_particle_px must be >= 0, got {self.min_particle_px!r}")
        if self.on_existing not in ("suffix", "fail"):
            raise ValueError(
                f"on_existing must be 'suffix' or 'fail', got {self.on_existing!r}")


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

@dataclass
class Table:
    """One flat table: an explicit column order and rows keyed by column.

    The column list is authoritative and the rows are dicts, so a missing key
    writes an empty cell rather than shifting the row — the failure that
    silently moves every number one column left.
    """

    name: str
    columns: List[str]
    rows: List[Dict[str, Any]]

    def __len__(self) -> int:
        return len(self.rows)


@dataclass
class ExportTables:
    """Everything one export knows, before any of it is written anywhere."""

    phases: Table
    regions: Table
    particles: Table
    definitions: Table
    provenance: Dict[str, Any]
    #: ``region_id``/``phase_id`` int rasters plus the lookup tables that give
    #: the ids meaning. The method developer called this the highest-value
    #: artefact in the set and he is right: it is the only one that lets
    #: somebody else's script reproduce a figure pixel for pixel.
    labels: Dict[str, np.ndarray] = field(default_factory=dict)
    #: Built lazily. A 485k-pixel map with twelve elements is a ~100 MB CSV;
    #: materialising it as a list of dicts costs several times that in RAM for
    #: no reason, so this is a generator factory and the writer streams it.
    pixel_rows: Optional[Any] = None
    pixel_columns: List[str] = field(default_factory=list)
    #: ``{filename: png_bytes}``. Rendered by ``phase_map_store``'s own
    #: renderers so the file is the picture the app draws; see
    #: :func:`_map_images`.
    images: Dict[str, bytes] = field(default_factory=dict)
    warnings: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class WrittenFile:
    name: str
    rows: int
    bytes: int
    #: Set for pictures. `rows` on a PNG is its pixel count; the dialog that
    #: lists the written files printed that as "462 rows" for a 21 x 22 px
    #: map, so a picture also says its size in the unit a reader expects.
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass
class WriteResult:
    folder: str
    files: List[WrittenFile]
    warnings: List[Dict[str, str]]
    provenance: Dict[str, Any]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _warn(bag: List[Dict[str, str]], code: str, message: str) -> None:
    """Stable machine code plus English prose.

    The UI translates on ``code`` and never on the prose — a lesson already
    paid for in this repo in August 2026 — and the prose is the
    fallback for a code the UI has not learned yet.
    """
    bag.append({"code": code, "message": message})


def _flat(arr) -> np.ndarray:
    return np.asarray(arr).ravel()


def _finite_mean(values: np.ndarray) -> Optional[float]:
    if values.size == 0:
        return None
    v = float(values.mean())
    return v if math.isfinite(v) else None


def _sha256_of(path: Path) -> Optional[str]:
    """Streamed, so a 27 GB scan does not become a 27 GB allocation."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
    except OSError as exc:
        logger.info("eds_export: cannot hash %s: %s", path, exc)
        return None
    return h.hexdigest()


def rank_candidates(
    means: Mapping[str, float],
    entries: Sequence[Any],
) -> List[Tuple[int, float]]:
    """``[(phase_index, gap_at_pct), ...]`` best first — THE INSPECTOR'S ranking.

    THIS IS THE RANKING FROM ``routes/eds.py::_region_detail`` AND MUST STAY
    THAT WAY. The inspector shows a user "AlFeMnSi, 2.7 at% off"; anything
    claiming to be that number has to agree to the digit, or the two halves
    of the page are using different notions of "close".

    IT IS NOT THE RANKING THAT MADE THE MAP, and until 2026-08-27 the export
    used it as if it were. The map is named by
    ``eds_clustering._match_labels`` -> ``chemistry_score.score_phase_ratio``,
    with rule gating, a matrix-element exemption and a ``min_score`` floor,
    on the eroded interior of each region. This function is an unweighted
    mean absolute deviation with none of that. Two metrics, and the export
    was printing this one's answer in a column called ``phase_auto`` beside a
    boolean called ``phase_agrees``. Measured on the real 90x120 SampleB scan
    (7 regions, 22 candidates, zero hand-painted pixels): 4 of 7 regions read
    ``phase_agrees = False``, which a reader can only understand as "a human
    moved this one". Nobody had. See :func:`deciding_rank` for the scorer
    that actually decides, and the export now uses that; this function
    survives under ``*_mad`` column names as an explicitly-labelled second
    opinion, because a distance in at% is easier to judge than a 0..1 score
    and because the inspector still shows it.

    The metric is the mean absolute deviation over the UNION of the measured
    element keys and the phase's nominal composition keys, with a missing key
    read as 0 on the side that lacks it. Union rather than intersection is
    what makes a phase containing an element the region does not have score
    badly instead of scoring well on the elements they happen to share.

    ``means`` must be the same dict ``_region_detail`` builds — the wand's
    ``selection_stats(...)["mean_at_pct"]`` with ``_CHEM_IGNORE`` removed —
    so that "the same measure" means the same measure and not merely a
    similar one. Ties break on the lower phase index, which makes the order
    reproducible; ``_region_detail`` leaves them to sort stability, which is
    the same thing given it enumerates in index order.
    """
    out: List[Tuple[int, float]] = []
    for i, e in enumerate(entries):
        comp = getattr(e, "composition", None) or {}
        keys = set(means) | set(comp)
        if not keys:
            continue
        gap = sum(abs(float(means.get(k, 0.0)) - float(comp.get(k, 0.0)))
                  for k in keys)
        out.append((i, gap / len(keys)))
    out.sort(key=lambda t: (t[1], t[0]))
    return out


def to_scored_basis(means: Mapping[str, float]) -> Dict[str, float]:
    """Renormalise a measured mean to sum 100 over the elements it names.

    THE TWO SIDES OF THE SUBTRACTION HAD DIFFERENT DENOMINATORS. The at%
    maps this module is handed are renormalised over EVERY measured element,
    carbon and oxygen included — measured on the real SampleB export, the
    per-pixel at% sum including C and O is 100.000000 everywhere. Drop C and
    O (which every scorer does, ``_CHEM_IGNORE``) and the remainder sums to
    91.297860 at worst, 98.44 on average. A CIF's nominal composition, by
    construction, has no C or O in it and sums to 100. Subtracting one from
    the other charged every phase for a few at% of carbon it could not
    contain, and charged it unevenly across the map.

    Measured on the same scan, over all 7 regions: no winner changed, but the
    margin moved by up to +112.6 % (region 4: 0.194720 -> 0.413980 at%) and
    one RUNNER-UP changed identity (region 6: beta-AlFeSi.cif ->
    sd_0302719.cif). A margin is the number that says whether a call was a
    decision or a coin toss, so a factor of two in it is not cosmetic.

    A dict that sums to nothing is returned unchanged: there is no basis to
    put it on, and scaling by 1/0 would manufacture one.
    """
    total = sum(float(v) for v in means.values())
    if not math.isfinite(total) or total <= 1e-9:
        return {k: float(v) for k, v in means.items()}
    return {k: float(v) * 100.0 / total for k, v in means.items()}


def _margin(ranked: Sequence[Tuple[int, float]], *, higher_is_better: bool = False,
            ) -> Optional[float]:
    """How far clear the winner is of the runner-up, in the ranking's own unit.

    A gap of 2.7 at% is meaningless until the runner-up's is known: 2.7
    against 9.1 is a decision, 2.7 against 2.8 is a coin toss wearing a phase
    name. Same for a score of 0.92 — measured on the real SampleB scan,
    region 6 wins at 0.817624 with a runner-up at 0.808264, a margin of
    0.009360, and that region is genuinely undecided by chemistry alone.

    ``higher_is_better`` picks the sign convention: the deciding scorer
    ranks by score (winner minus runner-up), the MAD second opinion ranks by
    distance (runner-up minus winner). Either way the result is >= 0 for a
    correctly-sorted ranking and larger means more confident.

    WHAT IS NOT PROMISED. An earlier docstring here claimed "never export a
    winner without a margin", and four code paths broke it. ``None`` is
    returned when there IS no runner-up — a one-entry library, or every other
    candidate excluded from the ranking — and that emptiness is itself the
    fact worth seeing: it says the winner was unopposed, not that it was
    convincing. The other three paths that used to leave a winner marginless
    (a zero-pixel region, a failed ranking, and every particle on a
    phase-basis map) are gone: the first two now emit no winner either, and
    the third is ranked on its own composition.
    """
    if len(ranked) < 2:
        return None
    if higher_is_better:
        return float(ranked[0][1] - ranked[1][1])
    return float(ranked[1][1] - ranked[0][1])


# ---------------------------------------------------------------------------
# the scorer that actually decides
# ---------------------------------------------------------------------------

def deciding_rank(
    object_means: Mapping[str, np.ndarray],
    entries: Sequence[Any],
    *,
    matrix_element: Optional[str] = None,
    background: Optional[Mapping[str, float]] = None,
    rule_set: Any = None,
) -> List[List[Tuple[int, float]]]:
    """The map's own scorer, replayed. ``[[(phase_index, score), ...], ...]``.

    One ranked list per object, best first, in the units the classification
    decided in: ``chemistry_score.score_phase_ratio``, 0..1, higher is
    better, with the user's phase rules applied as a GATE exactly as
    ``eds_clustering._match_labels`` applies them — a blocked phase scores 0
    and so loses even to a badly-scoring phase that is allowed.

    ``object_means`` is ``{element: array}`` with one entry per object: each
    object is scored as a single pseudo-pixel, which is what ``_match_labels``
    does with a cluster mean. Batched across objects rather than looped
    because ``score_phase_ratio`` is vectorised over pixels: 22 candidates on
    a 10 000-particle map is 22 calls, not 220 000.

    ``background`` MUST come from the whole map. The median of one value is
    that value, so a background derived from a single object's mean would
    compare it against 1.3x itself and veto every phase — the exact trap
    ``score_phase_ratio`` documents at its ``background is None`` branch.

    Verified 2026-08-27 against the real 90x120 SampleB map (7 regions, 22
    candidates): reproduces ``region_phase`` for 7 of 7 regions, including
    the two the MAD ranking got wrong by naming a different phase entirely.
    """
    n_obj = 0
    for arr in object_means.values():
        n_obj = int(np.asarray(arr).size)
        break
    if n_obj == 0 or not entries:
        return [[] for _ in range(n_obj)]

    one = {el: np.asarray(v, dtype=np.float64).ravel()
           for el, v in object_means.items()}
    bg = dict(background) if background else None
    score = _score_phase_ratio()
    if score is None:
        return [[] for _ in range(n_obj)]

    gate = None
    if rule_set is not None and not getattr(rule_set, "is_empty", True):
        try:
            from backend.api.services.phase_rules import gate_scores as gate
        except Exception:
            logger.debug("eds_export: gate_scores unavailable", exc_info=True)
            gate = None

    scores = np.zeros((len(entries), n_obj), dtype=np.float64)
    for i, e in enumerate(entries):
        try:
            # no_data_score=0.0 for the same reason _match_labels uses it: an
            # object whose pixels carry no measurement must score 0 and stay
            # unnamed, not score a perfect 1.0 and become the most confident
            # call on the map.
            s = np.asarray(score(
                one, getattr(e, "composition", None) or {},
                matrix_element=matrix_element, no_data_score=0.0,
                background=bg), dtype=np.float64)
        except Exception:
            logger.debug("eds_export: scoring failed for entry %d", i,
                         exc_info=True)
            continue
        if gate is not None:
            rule = rule_set.rule_for(str(getattr(e, "key", "")))
            try:
                s, _outcome = gate(s, rule, one, background=bg)
            except Exception:
                logger.debug("eds_export: gating failed for entry %d", i,
                             exc_info=True)
        scores[i] = s

    out: List[List[Tuple[int, float]]] = []
    for j in range(n_obj):
        col = scores[:, j]
        # Descending by score, ties on the lower phase index so the order is
        # reproducible. _match_labels sorts on -score alone and leaves ties
        # to sort stability, which enumerates in index order — the same thing.
        order = sorted(range(len(entries)), key=lambda i: (-col[i], i))
        out.append([(i, float(col[i])) for i in order])
    return out


def _score_phase_ratio():
    """Late import. ``None`` on a slimmed deployment without the scorer.

    Deliberately not a module-level import: `deciding_rank` is the only
    caller and it degrades to "no answer" rather than taking the whole export
    down, which is the same posture every other optional import in this file
    takes.
    """
    try:
        from backend.api.services.chemistry_score import score_phase_ratio
        return score_phase_ratio
    except Exception:
        logger.debug("eds_export: score_phase_ratio unavailable", exc_info=True)
        return None


def _phase_name(entry: Any) -> str:
    """What a phase is called in every table.

    The CIF filename, because that is the identity the store carries and the
    one a user can look up in the library. The formula gets its own column
    rather than being substituted here: two library entries can share a
    formula and differ in space group, and collapsing them would merge two
    rows that mean different things.
    """
    if entry is None:
        return ""
    return str(getattr(entry, "cif_filename", "") or getattr(entry, "key", ""))


#: Database bookkeeping that is not part of a phase's name. Stripped from a
#: filename stem before it is offered as a label, longest first so that
#: ``_mp-570001_symmetrized`` loses both pieces.
_LABEL_NOISE = (
    "_symmetrized", "_conventional", "_primitive", "_standard",
)

#: A filename stem that is only a database id carries no chemistry, so it
#: cannot be a legend entry on its own.
_ID_ONLY = re.compile(r"^(sd|icsd|cod|mp)[-_]?\d+$", re.IGNORECASE)

#: Unicode subscripts as the library writes them.
_SUBSCRIPTS = str.maketrans("\u2080\u2081\u2082\u2083\u2084"
                            "\u2085\u2086\u2087\u2088\u2089",
                            "0123456789")


def formula_ascii(formula: Any) -> str:
    """The formula with its Unicode subscripts flattened to ASCII digits.

    ``Mn\u2084.\u2085\u2081\u2082Al\u2081\u2082\u2087...`` is unreadable in
    half the tools a reader will open this file with, and unpasteable into
    the other half: a subscript survives a CSV and does not survive a
    plotting label, a LaTeX run or a shell. The Unicode form stays under
    ``formula``; this sits beside it.
    """
    return str(formula or "").translate(_SUBSCRIPTS).strip()


def phase_label(entry: Any) -> str:
    """A name a figure legend can carry.

    WHY. ``phase_name`` is ``sd_0302719.cif`` and ``formula`` is
    ``Mn\u2084.\u2085\u2081\u2082Al\u2081\u2082\u2087.\u2082\u2089\u2086...``.
    Two testers said independently that neither can go in a figure legend or
    a report, and they are right: one is a database id and the other is a
    refined site occupancy printed as a formula.

    The filename stem is preferred, because it is what the user named the
    CIF and therefore what they already call the phase, with database
    bookkeeping removed (``_mp-570001``, ``_symmetrized``). When the stem is
    only an id it carries no chemistry, so the label is built from the
    composition instead -- elements in descending at%, hyphen-joined -- with
    the id kept in brackets so the row stays traceable to the library entry.

    Never invented: with neither a stem nor a composition to work from, the
    empty string is returned and the caller shows ``phase_name``.
    """
    if entry is None:
        return ""
    stem = str(getattr(entry, "cif_filename", "") or getattr(entry, "key", ""))
    stem = stem.rsplit(".cif", 1)[0].strip() if stem else ""
    for noise in _LABEL_NOISE:
        stem = stem.replace(noise, "")
    # A trailing database id on an otherwise chemical name: Al6Fe_mp-570001.
    stem = re.sub(r"[_-](mp|icsd|cod|sd)[-_]?\d+$", "", stem,
                  flags=re.IGNORECASE).strip(" _-")
    # `Fe3Si_MP-mp-2199` loses the id above and keeps a bare `_MP`, which is
    # the database's name and not the phase's.
    stem = re.sub(r"[_-](mp|icsd|cod)$", "", stem,
                  flags=re.IGNORECASE).strip(" _-")

    comp = getattr(entry, "composition", None) or {}
    if stem and not _ID_ONLY.match(stem):
        return stem
    if comp:
        els = "-".join(el for el, _v in sorted(
            comp.items(), key=lambda kv: (-float(kv[1]), kv[0])))
        return f"{els} ({stem})" if stem else els
    return stem


# ---------------------------------------------------------------------------
# element bookkeeping
# ---------------------------------------------------------------------------

def _element_universe(
    at_maps: Mapping[str, np.ndarray],
    entries: Sequence[Any],
    state: Any,
) -> Tuple[List[str], List[str]]:
    """``(all_columns, measured)`` element symbols.

    The column set is deliberately WIDER than what was measured: every element
    a participating phase, a region definition or an element weight names gets
    a column too, and it reads ``not measured`` all the way down. That is the
    point — a reader comparing this export against a phase that contains
    copper has to be able to see that copper was never mapped, and an absent
    column looks like an oversight while an absent VALUE is a statement.

    Alphabetical, because stability across datasets beats readability within
    one: a sheet that pins column positions must not re-order because this
    sample happens to have more silicon than the last one.
    """
    measured = sorted(str(k) for k in at_maps.keys())
    extra: set = set()
    for e in entries or []:
        for el in (getattr(e, "composition", None) or {}):
            extra.add(str(el))
    for d in (getattr(state, "region_defs", None) or []):
        if not isinstance(d, dict):
            continue
        for clause in (d.get("elements") or []):
            if isinstance(clause, dict) and clause.get("element"):
                extra.add(str(clause["element"]))
        for clause in (d.get("enrichment") or []):
            if isinstance(clause, dict) and clause.get("element"):
                extra.add(str(clause["element"]))
        for clause in (d.get("ratios") or []):
            if not isinstance(clause, dict):
                continue
            for side in ("numerator", "denominator"):
                if clause.get(side):
                    extra.add(str(clause[side]))
    for el in (getattr(state, "element_weights", None) or {}):
        extra.add(str(el))
    return sorted(set(measured) | extra), measured


def _composition_cells(
    prefix: str,
    columns: Sequence[str],
    measured: Sequence[str],
    stats: Mapping[str, Tuple[float, float]],
    row: Dict[str, Any],
    *, with_sd: bool = True,
) -> None:
    """Fill ``<prefix>mean_at_pct_El`` / ``<prefix>sd_within_at_pct_El``.

    An element outside ``measured`` gets :data:`NOT_MEASURED`. An element that
    IS measured but whose stats are missing for this mask (an empty mask, a
    map of the wrong length) gets an empty cell — "no pixels here" and "never
    mapped" are different statements and must not share a spelling.
    """
    measured_set = set(measured)
    for el in columns:
        mkey = f"{prefix}mean_at_pct_{el}"
        skey = f"{prefix}sd_within_at_pct_{el}"
        if el not in measured_set:
            row[mkey] = NOT_MEASURED
            if with_sd:
                row[skey] = NOT_MEASURED
            continue
        pair = stats.get(el)
        row[mkey] = None if pair is None else float(pair[0])
        if with_sd:
            row[skey] = None if pair is None else float(pair[1])


def _composition_columns(
    prefix: str, columns: Sequence[str], *, with_sd: bool = True,
) -> List[str]:
    out: List[str] = []
    for el in columns:
        out.append(f"{prefix}mean_at_pct_{el}")
        if with_sd:
            out.append(f"{prefix}sd_within_at_pct_{el}")
    return out


# ---------------------------------------------------------------------------
# which phases were never on the list, and which library said so
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS. A QA tester measured the library at 30 entries, of which
# 22 participated and **8 were dropped silently** -- seven because magnesium
# was never mapped, one because nickel was not. The drop happens inside
# ``cif_phase_library.candidates_for``, which is a one-line subset filter
# with no logging and no return channel for what it removed. Her named phase
# beta was in fact present under a different filename and DID participate;
# gamma-Al3FeSi was not in the library at all. Those are opposite problems
# with opposite fixes -- map the element, or add the CIF -- and the export
# gave her no way to tell them apart. ``phases_participating`` answered
# "who competed" and nothing answered "who did not, and why".
#
# THE IDENTITY BLOCK is a separate request, made independently by three
# testers. ``phases_participating`` already carries each entry's formula, so
# a stoichiometry edit between two exports is visible by diffing them. A
# LATTICE-PARAMETER-only edit is not: it changes nothing this module ever
# sees, and one tester said that is precisely the edit he cares about
# between heat treatments. Only a hash of the file itself can catch it, so
# the file is hashed.

#: (interfering element, affected element, the sentence). Only pairs where
#: BOTH elements were mapped are reported: "this scan contains Mn" has to be
#: true of the scan the file describes, and a generic list of every overlap
#: in the periodic table is a list nobody reads.
_PEAK_OVERLAPS: Tuple[Tuple[str, str, str], ...] = (
    ("Mn", "Fe",
     "Mn K-beta at 6.49 keV sits under Fe K-alpha at 6.40 keV, so a fixed "
     "Fe window OVER-REPORTS Fe in an Mn-bearing alloy."),
    ("Al", "Si",
     "The Al K-beta tail at 1.55 keV runs into the Si K-alpha window at "
     "1.74 keV, so an aluminium matrix raises the apparent Si."),
    ("Cr", "Mn",
     "Cr K-beta at 5.95 keV overlaps Mn K-alpha at 5.90 keV."),
    ("Fe", "Co",
     "Fe K-beta at 7.06 keV overlaps Co K-alpha at 6.93 keV."),
    ("Ti", "V",
     "Ti K-beta at 4.93 keV overlaps V K-alpha at 4.95 keV."),
)


def _peak_overlap_block(measured: Sequence[str]) -> Dict[str, Any]:
    """The overlaps that apply TO THIS SCAN, named with their energies.

    The manual says peak overlaps exist. A tester wanted the concrete
    sentence in the file, and he is right about which one: this project's
    samples are Al-Fe-Mn-Si, so the Mn-K-beta-under-Fe-K-alpha overlap is
    live on essentially every scan it produces, and it biases the element
    that decides between an Fe-intermetallic and the matrix.

    Reported only for pairs where BOTH elements were mapped, so the block is
    about the file it sits in. It is a DECLARATION, not a detection: this
    module receives per-element maps and never the spectra, so it cannot
    measure an overlap, and saying so is part of the block.
    """
    have = {str(el) for el in measured}
    applies = [{"interfering": a, "affected": b, "note": text}
               for a, b, text in _PEAK_OVERLAPS if a in have and b in have]
    return {
        "measured_elements": sorted(have),
        "applies_to_this_scan": applies,
        "n_applicable": len(applies),
        "why": (
            "The at% values come from Aztec WINDOW INTEGRALS -- counts in a "
            "fixed energy window per element -- and not from a fitted, "
            "deconvolved spectrum. A window collects every line that falls "
            "inside it, whoever emitted it."),
        "consequence": (
            "An over-reported element is over-reported EVERYWHERE, so it "
            "shifts a phase's measured composition in one direction rather "
            "than scattering it, and it can move a candidate ranking whose "
            "margin is under a couple of at%. Read margin_score and "
            "third_score before trusting a call on an element listed here."),
        "detection": (
            "NOT detected, DECLARED. This module receives per-element maps "
            "and never the spectra, so it cannot measure an overlap. An "
            "empty applies_to_this_scan means no pair from the declared "
            "list has both of its elements mapped here -- never that the "
            "spectra were checked and found clean."),
    }


def _default_library_path() -> Path:
    """``Database/crystal_database.xlsx`` beside the project root.

    ``backend/api/services/eds_export.py`` -> parents[3] is the repo root,
    the same arithmetic ``routes/eds.py::_project_root`` does. Derived, never
    hardcoded: a drive letter in this file would break every other machine.
    """
    return Path(__file__).resolve().parents[3] / "Database" / "crystal_database.xlsx"


def _entries_digest(entries: Iterable[Any]) -> str:
    """sha256 over what this module can SEE of each entry.

    Key, formula, space group and composition, sorted, so it is stable
    across dict ordering. It answers a narrower question than the file hash
    beside it -- "did the chemistry or the symmetry of the candidates
    change" -- and the pair is what lets a reader tell a stoichiometry edit
    from an edit to something this export never reads.
    """
    h = hashlib.sha256()
    for e in sorted(entries, key=lambda x: str(getattr(x, "key", ""))):
        comp = {k: round(float(v), 6)
                for k, v in sorted((getattr(e, "composition", None) or {}).items())}
        h.update(json.dumps([
            str(getattr(e, "key", "")),
            str(getattr(e, "cif_filename", "")),
            str(getattr(e, "formula", "")),
            str(getattr(e, "space_group", "")),
            comp,
        ], sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()


def library_report(
    entries: Sequence[Any],
    measured_elements: Sequence[str],
    *,
    library: Optional[Mapping[str, Any]] = None,
    library_path: Optional[str] = None,
    requested_keys: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """``(identity, phases_absent)`` -- who the library is, and who is missing.

    Three reasons, and they are three different actions for the reader:

      ``element_not_measured``   the phase contains an element this scan
                                 never mapped, so ``candidates_for`` removed
                                 it. ``missing_elements`` names them. Fix:
                                 map the element.
      ``not_requested``          the phase is in the library and every
                                 element of it was measured, but the run was
                                 narrowed to a subset (``phase_keys``). Fix:
                                 widen the selection.
      ``not_in_library``         a key the run ASKED for that the library
                                 does not have. Fix: add the CIF and rebuild
                                 the database.

    The library is read at the default path when the caller supplies none,
    and ``identity["source"]`` says which happened. A failure to read it is
    reported as such, never as an empty library -- "no phases were dropped"
    and "we could not check" are different statements.
    """
    identity: Dict[str, Any] = {}
    absent: List[Dict[str, Any]] = []

    if library is not None:
        lib = dict(library)
        path = Path(library_path) if library_path else None
        identity["source"] = "caller"
    else:
        path = _default_library_path()
        identity["source"] = "default path"
        try:
            from backend.api.services.cif_phase_library import load_cif_phase_library
            lib = dict(load_cif_phase_library(path))
        except Exception:
            logger.debug("eds_export: cannot load the CIF library",
                         exc_info=True)
            lib = {}
            identity["source"] = "unavailable"

    identity["path"] = str(path) if path else None
    identity["n_entries"] = len(lib)
    identity["n_participating"] = len(entries)
    identity["sha256"] = None
    identity["bytes"] = None
    identity["mtime"] = None
    if path is not None:
        try:
            st = path.stat()
            identity["bytes"] = int(st.st_size)
            identity["mtime"] = datetime.fromtimestamp(st.st_mtime).astimezone().isoformat()
            # The FILE hash, not a hash of the parsed entries: a lattice
            # parameter edited between two heat treatments changes nothing
            # this module parses, and only the bytes can see it.
            identity["sha256"] = _sha256_of(path)
        except OSError:
            logger.debug("eds_export: cannot stat the CIF library %s", path,
                         exc_info=True)
    identity["entries_digest"] = _entries_digest(lib.values()) if lib else None
    identity["note"] = (
        "sha256 is of the library FILE, so it changes on any edit -- "
        "including one this export never reads, such as a lattice parameter. "
        "entries_digest hashes only the key, filename, formula, space group "
        "and composition of every entry, so the pair tells a chemistry or "
        "symmetry edit (both hashes move) apart from an edit to anything "
        "else (only sha256 moves). Two exports carrying the same pair were "
        "scored against the same candidates.")

    if not lib:
        identity["phases_absent_note"] = (
            "The library could not be read, so this export cannot say which "
            "phases were left out. Absent, not empty: an empty list would "
            "assert that nothing was dropped.")
        return identity, absent

    measured = {str(el) for el in measured_elements}
    participating = {str(getattr(e, "key", "")) for e in entries}
    requested = ([str(k) for k in requested_keys]
                 if requested_keys else None)
    requested_set = set(requested) if requested else None

    for key, entry in lib.items():
        if str(key) in participating:
            continue
        els = [str(el) for el in (getattr(entry, "elements", None) or [])]
        missing = sorted(el for el in els if el not in measured)
        if missing:
            reason, detail = "element_not_measured", missing
        elif requested_set is not None and str(key) not in requested_set:
            reason, detail = "not_requested", []
        else:
            # Every element measured, not narrowed out, and still absent.
            # Report it honestly rather than inventing one of the two known
            # reasons for it.
            reason, detail = "excluded_upstream", []
        absent.append({
            "key": str(key),
            "cif_filename": str(getattr(entry, "cif_filename", "")),
            "formula": str(getattr(entry, "formula", "")),
            "phase_label": phase_label(entry),
            "elements": els,
            "reason": reason,
            "missing_elements": detail,
        })

    for key in (requested or []):
        if key in lib or key in participating:
            continue
        absent.append({
            "key": str(key), "cif_filename": "", "formula": "",
            "phase_label": str(key), "elements": [],
            "reason": "not_in_library", "missing_elements": [],
        })

    absent.sort(key=lambda d: (d["reason"], d["key"]))
    identity["n_absent"] = len(absent)
    identity["phases_absent_note"] = (
        "Every library entry that did NOT compete, with the reason. "
        "element_not_measured: cif_phase_library.candidates_for removes a "
        "phase whose elements are not all mapped, because the missing ones "
        "would be matched against zeros -- missing_elements names them, and "
        "mapping that element is the fix. not_requested: the run was "
        "narrowed to a subset of the library. not_in_library: the run asked "
        "for a key the library does not have, which is the one case where "
        "the fix is to add the CIF and rebuild. Until 2026-08-27 this "
        "removal was silent: a tester measured 30 entries, 22 competing and "
        "8 dropped, with nothing in the file to say so or why.")
    return identity, absent


# ---------------------------------------------------------------------------
# how the spectra were acquired, and how big the volume they came from is
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS. A method-developer tester opened the source file himself
# and read `/1/EDS/Header`: beam voltage, working distance, process time,
# detector serial, magnification, energy range, channel width, channel count,
# tilt, detector elevation and azimuth, window type and the acquisition date.
# Every one of those was already in the scan and NONE of it reached the
# export -- while the caveat below it asserts "no k-factor calibration", a
# claim he cannot weigh without knowing the kV. Worse, the omission was not
# declared in `not_built` either, so the file gave no sign anything was
# missing. Readers for exactly these keys already exist in this repo
# (tools/h5_viewer_backend.get_eds_header; result_exporter's Beam Voltage
# lookup), so this was a gap in the plumbing and not a hard problem.

#: Keys of ``/<root>/EDS/Header`` promoted to named provenance fields, with
#: the vendor's unit appended to the name. Everything in the group --
#: including keys not listed here -- is also carried verbatim under
#: ``acquisition.raw``, so a field this table does not know about is
#: preserved rather than dropped.
_ACQ_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("Beam Voltage", "beam_voltage_kv"),
    ("Working Distance", "working_distance_mm"),
    ("Magnification", "magnification"),
    ("Process Time", "process_time"),
    ("Energy Range", "energy_range_kev"),
    ("Channel Width", "channel_width_ev"),
    ("Start Channel", "start_channel_ev"),
    ("Number Channels", "number_channels"),
    ("Tilt Angle", "tilt_angle_deg"),
    ("Tilt Axis", "tilt_axis_deg"),
    ("Detector Elevation", "detector_elevation"),
    ("Detector Azimuth", "detector_azimuth"),
    ("Detector Serial Number", "detector_serial"),
    ("Detector Type Id", "detector_type_id"),
    ("Window Type", "window_type"),
    ("Processor Type", "processor_type"),
    ("Drift Correction", "drift_correction"),
    ("Strobe FWHM", "strobe_fwhm_ev"),
    ("Number Frames", "number_frames"),
    ("Binning", "binning"),
    ("Acquisition Date", "acquisition_date"),
    ("Specimen Label", "specimen_label"),
    ("Site Label", "site_label"),
    ("Analysis Label", "analysis_label"),
    ("Project Label", "project_label"),
)


def _h5_scalar(value: Any) -> Any:
    """One H5OINA header value as a plain Python object.

    H5OINA wraps scalars as ``(1,)`` and vectors as ``(1, N)``, so the test
    is on ``ndim`` after ``squeeze`` and never on ``size == 1`` -- the latter
    mishandles both shapes. Bytes are decoded with ``errors="replace"``,
    because a label with one stray byte in it must not cost the export its
    whole header.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    arr = np.squeeze(value)
    if arr.ndim == 0:
        item = arr.item()
        return (item.decode("utf-8", errors="replace")
                if isinstance(item, bytes) else item)
    out = []
    for v in arr.ravel().tolist():
        out.append(v.decode("utf-8", errors="replace")
                   if isinstance(v, bytes) else v)
    return out


def read_acquisition_header(source_path: Optional[str]) -> Optional[dict]:
    """``/<root>/EDS/Header`` as a plain dict, or ``None``.

    Read GENERICALLY: every dataset in the group, plus one level of
    subgroups -- ``Stage Position`` is a group and not a dataset, so a flat
    ``for k in group: group[k][()]`` raises on it. A fixed key list would
    silently drop whatever Aztec adds next, and the whole point of this
    block is that the reader can see what the microscope was doing.

    ``None`` -- never ``{}`` -- when there is no path, no h5py, no such
    group, or the file cannot be opened. The caller turns that into a stated
    reason in the provenance rather than an absent field.
    """
    if not source_path:
        return None
    try:
        import h5py
    except Exception:
        logger.debug("eds_export: h5py unavailable", exc_info=True)
        return None
    try:
        with h5py.File(source_path, "r") as fh:
            group = None
            for root in fh.keys():
                candidate = f"{root}/EDS/Header"
                if candidate in fh:
                    group = fh[candidate]
                    break
            if group is None:
                return None
            out: Dict[str, Any] = {}
            for key in group.keys():
                node = group[key]
                try:
                    if hasattr(node, "keys"):              # a subgroup
                        out[str(key)] = {str(k): _h5_scalar(node[k][()])
                                         for k in node.keys()}
                    else:
                        out[str(key)] = _h5_scalar(node[()])
                except Exception:
                    logger.debug("eds_export: header key %s unreadable", key,
                                 exc_info=True)
            return out
    except Exception:
        logger.debug("eds_export: cannot read the EDS header from %s",
                     source_path, exc_info=True)
        return None


def _acquisition_block(
    geometry: ExportGeometry, opts: ExportOptions,
    warnings: List[Dict[str, str]],
) -> Dict[str, Any]:
    """The named fields, the raw group, and where they came from."""
    raw = geometry.acquisition
    source = "caller"
    if raw is None:
        if opts.read_acquisition_header:
            raw = read_acquisition_header(geometry.source_path)
            source = "source file"
        else:
            source = "not requested"
    if not raw:
        _warn(warnings, "no_acquisition_header",
              "The EDS acquisition header could not be read, so the beam "
              "voltage, working distance, detector and spectrometer "
              "settings are absent from this export. Every measured number "
              "is unaffected; the interaction-volume estimate cannot be "
              "made without the beam voltage.")
        return {
            "available": False,
            "source": source if raw is None else "empty",
            "note": ("No /<root>/EDS/Header was readable. Absent, not "
                     "defaulted: a beam voltage nobody measured would be "
                     "worse than none at all."),
        }

    out: Dict[str, Any] = {"available": True, "source": source}
    for key, name in _ACQ_FIELDS:
        if key in raw:
            out[name] = raw[key]
    out["raw"] = dict(raw)
    out["note"] = (
        "Read from the scan's own /<root>/EDS/Header. The named fields "
        "carry the vendor's unit in their name; `raw` is the whole group "
        "verbatim, so a key this exporter does not know about is preserved "
        "rather than dropped. These settings are what the semi-quantitative "
        "caveat has to be weighed against -- beam_voltage_kv in particular, "
        "which sets both which lines are excited at all and how large the "
        "volume each measurement comes from is (see interaction_volume).")
    return out


#: Kanaya-Okayama constants for the elements this project's samples are made
#: of, so the range estimate still works without pymatgen: (atomic mass
#: g/mol, atomic number, density g/cm3). pymatgen is tried first because it
#: covers the whole table; these are the fallback and they agree with it.
_KO_ELEMENTS: Dict[str, Tuple[float, int, float]] = {
    "Al": (26.982, 13, 2.70), "Si": (28.086, 14, 2.33),
    "Fe": (55.845, 26, 7.87), "Mn": (54.938, 25, 7.47),
    "Mg": (24.305, 12, 1.74), "Cu": (63.546, 29, 8.96),
    "Zn": (65.380, 30, 7.14), "Ni": (58.693, 28, 8.91),
    "Ti": (47.867, 22, 4.51), "Cr": (51.996, 24, 7.19),
}


def kanaya_okayama_range_um(element: str, beam_kv: float) -> Optional[float]:
    """R = 0.0276 A E^1.67 / (Z^0.89 rho), in micrometres.

    The standard electron-range estimate (Kanaya & Okayama 1972) with A in
    g/mol, E in keV and rho in g/cm3. It is the number that decides whether
    an EDS map acquired during an EBSD session can resolve the features it
    appears to show: at 20 kV in aluminium it is **4.19 um**, against this
    project's 0.5 um step. A tester computed exactly that by hand and asked
    for it to be in the file, because "the interaction volume exceeds the
    features" stays prose until it is a length standing beside a size
    distribution.

    It is a RANGE in a pure element, not a resolution -- the X-ray
    generation volume is smaller -- so the provenance reports it as a bound.
    ``None`` when the element is unknown or the voltage is not a usable
    number: an invented density would put a made-up length in a report.
    """
    try:
        kv = float(beam_kv)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(kv) or kv <= 0.0:
        return None
    sym = str(element or "").strip()
    if not sym:
        return None
    mass = z = rho = None
    try:
        from pymatgen.core.periodic_table import Element
        el = Element(sym)
        mass = float(el.atomic_mass)
        z = int(el.Z)
        d = getattr(el, "density_of_solid", None)
        rho = float(d) / 1000.0 if d else None            # kg/m3 -> g/cm3
    except Exception:
        logger.debug("eds_export: pymatgen lookup failed for %s", sym,
                     exc_info=True)
    if not (mass and z and rho):
        fallback = _KO_ELEMENTS.get(sym)
        if fallback is None:
            return None
        mass, z, rho = float(fallback[0]), int(fallback[1]), float(fallback[2])
    try:
        return float(0.0276 * mass * (kv ** 1.67) / ((z ** 0.89) * rho))
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _interaction_volume_block(
    acquisition: Mapping[str, Any], matrix_element: Optional[str],
    geometry: ExportGeometry, particle_rows: Sequence[Mapping[str, Any]],
    area_ok: bool,
) -> Dict[str, Any]:
    """The range, the step, and how many particles are smaller than it.

    THE POINT OF THE BLOCK. ``particles.core_note`` and the module docstring
    already say the interaction volume exceeds the features; that is prose,
    and a reader cannot act on it. Two numbers turn it into a judgement: the
    Kanaya-Okayama range for the matrix at the recorded kV, and the fraction
    of the exported particles whose equivalent circle diameter falls below
    it. On this project's SampleB scan that is 4.19 um against a 0.5 um
    step, and the great majority of the particles -- which is the single
    most important thing to know before quoting a particle composition from
    this file.

    Compared on ``ecd_um`` because the range is the diameter of a roughly
    spherical volume, so diameter against diameter is the like-for-like
    comparison. ``feret_max_um`` would flatter an elongated particle, and
    ``sqrt_area_um`` is a different rating parameter (see ``size_note``).
    """
    kv = acquisition.get("beam_voltage_kv") if acquisition else None
    out: Dict[str, Any] = {
        "beam_voltage_kv": kv,
        "matrix_element": matrix_element,
        "model": "Kanaya-Okayama 1972, R = 0.0276 A E^1.67 / (Z^0.89 rho)",
    }
    rng = (kanaya_okayama_range_um(matrix_element, kv)
           if (kv is not None and matrix_element) else None)
    out["range_um"] = rng
    if rng is None:
        out["note"] = (
            "Not computed: it needs a beam voltage from the acquisition "
            "header AND a matrix element, and at least one of them is "
            "unavailable here. See acquisition.available.")
        return out

    step = None
    if area_ok:
        try:
            step = max(float(geometry.step_x_um), float(geometry.step_y_um))
        except (TypeError, ValueError):
            step = None
    out["step_um"] = step
    out["range_over_step"] = (rng / step) if (step and step > 0) else None

    ecds = [float(r["ecd_um"]) for r in particle_rows
            if r.get("ecd_um") is not None]
    out["n_particles_measured"] = len(ecds)
    if ecds:
        n_small = sum(1 for v in ecds if v < rng)
        out["n_particles_smaller_than_range"] = n_small
        out["frac_particles_smaller_than_range"] = n_small / len(ecds)
    else:
        out["n_particles_smaller_than_range"] = None
        out["frac_particles_smaller_than_range"] = None

    out["note"] = (
        "range_um is the Kanaya-Okayama electron range in the matrix "
        "element at the recorded beam voltage -- a BOUND on the volume each "
        "spectrum comes from, not a resolution (X-ray generation happens "
        "inside it). frac_particles_smaller_than_range compares it against "
        "each particle's ecd_um, diameter against diameter. A particle "
        "below the range is measured together with its surroundings: its "
        "composition is pulled toward the matrix, and the pull scales with "
        "size, which manufactures a spurious composition-versus-size trend. "
        "The core_* columns mitigate that and do not remove it. This is the "
        "number the semi-quantitative caveat should be read next to.")
    return out


# ---------------------------------------------------------------------------
# the core
# ---------------------------------------------------------------------------

def build_tables(
    state: Any,
    at_maps: Mapping[str, np.ndarray],
    *,
    geometry: ExportGeometry,
    options: Optional[ExportOptions] = None,
) -> ExportTables:
    """Every table this export can build, from data alone.

    Parameters
    ----------
    state
        A :class:`~backend.api.services.phase_map_store.PhaseMapState`. Read
        only; nothing here mutates it.
    at_maps
        ``{element_symbol: array}``, at%, one value per pixel. 1-D or 2-D;
        everything below ravels.
    geometry
        Step sizes, grid shape, source path. ``step_*_um = None`` drops every
        physical column.
    options
        See :class:`ExportOptions`. ``None`` uses the defaults.

    Returns
    -------
    ExportTables
        Tables, provenance, label rasters and a generator factory for the
        per-pixel table. Nothing has touched the filesystem yet;
        :func:`write_export` does that.
    """
    opts = options or ExportOptions()
    opts.validate()
    warnings: List[Dict[str, str]] = []

    n_rows = int(geometry.n_rows)
    n_cols = int(geometry.n_cols)
    n_px = n_rows * n_cols

    entries = list(getattr(state, "phase_entries", None) or [])
    phase_grid = np.asarray(state.phase_grid, dtype=np.int64).reshape(n_rows, n_cols)
    flat_phase = phase_grid.ravel()
    score_flat = _flat(getattr(state, "score_grid", None)
                       if getattr(state, "score_grid", None) is not None
                       else np.zeros(n_px, dtype=np.float32)).astype(np.float64)

    locked = getattr(state, "locked_mask", None)
    locked_flat = (np.asarray(locked, dtype=bool).ravel()
                   if locked is not None else np.zeros(n_px, dtype=bool))
    if locked_flat.size != n_px:                    # a stale mask from another grid
        locked_flat = np.zeros(n_px, dtype=bool)

    # --- normalise the composition maps to flat float64 ----------------------
    flat_maps: Dict[str, np.ndarray] = {}
    for el, arr in (at_maps or {}).items():
        v = np.asarray(arr, dtype=np.float64).ravel()
        if v.size != n_px:
            _warn(warnings, "element_shape_mismatch",
                  f"Element {el} has {v.size} values for a {n_rows}x{n_cols} "
                  f"({n_px} pixel) raster and was dropped from the export.")
            continue
        flat_maps[el] = v

    columns_el, measured_el = _element_universe(flat_maps, entries, state)

    # --- the three denominators ---------------------------------------------
    valid_mask = _valid_eds_mask(flat_maps, n_px)
    n_valid = int(valid_mask.sum())
    n_classified = int((flat_phase >= 0).sum())
    if n_valid == 0:
        _warn(warnings, "no_valid_eds",
              "No pixel in this raster carries a usable EDS measurement; "
              "frac_of_valid_eds is empty throughout.")
    if n_classified == 0:
        _warn(warnings, "nothing_classified",
              "No pixel received a phase; frac_of_classified is empty "
              "throughout.")

    area_ok = geometry.area_available
    if not area_ok:
        _warn(warnings, "no_step_size",
              "This file carries no step size for the EDS or EBSD grid, so "
              "every column in micrometres has been omitted. A guessed scale "
              "would look like an answer.")

    # --- optional smoothed composition --------------------------------------
    # Resolved ONCE: the same record drives the smoothed columns and the
    # provenance entry, so the file cannot report a width it did not use.
    smoothing = _resolve_smoothing(state, opts, geometry)
    smoothed_maps, smoothing_note = _smoothed_maps(
        flat_maps, n_rows, n_cols, smoothing, warnings)
    smoothing["note"] = smoothing_note

    # --- regions -------------------------------------------------------------
    region_grid = getattr(state, "region_grid", None)
    has_regions = region_grid is not None
    if has_regions:
        region_grid = np.asarray(region_grid, dtype=np.int64).reshape(n_rows, n_cols)
        flat_region = region_grid.ravel()
    else:
        flat_region = np.full(n_px, -1, dtype=np.int64)

    # --- particles -----------------------------------------------------------
    # A particle is a connected component of one REGION. On a map made in
    # pixel mode there are no regions at all, and returning an empty
    # particles.csv there would be useless rather than honest — so the phase
    # grid stands in, and `particle_basis` in the provenance says which was
    # used. The two are not comparable: two regions carrying the same phase
    # name are one component under the phase grid and two under the region
    # grid.
    if has_regions:
        particle_basis = "region"
        particle_grid = region_grid.astype(np.int32)
    else:
        particle_basis = "phase"
        particle_grid = phase_grid.astype(np.int32)
        _warn(warnings, "particles_from_phases",
              "This map has no regions (it was classified per pixel), so "
              "particles are connected components of the PHASE grid. Two "
              "regions carrying the same phase would be one particle here.")

    particles = find_particles(particle_grid, connectivity=opts.connectivity)

    # --- background, for the enrichment the ranking is defined against -------
    background = _background_levels(flat_maps)

    # --- the gating context the classification decided in --------------------
    # Rebuilt from what the map RECORDED, not from what the export dialog was
    # told, so `phase_auto` is the answer that produced this map rather than a
    # second opinion wearing its name. See `deciding_rank`.
    rule_set = _resolve_rule_set(state)
    matrix_element = _resolve_matrix_element(flat_maps, rule_set)
    stored_min_score = _recorded_settings(state).get("min_score")
    min_score = float(stored_min_score if stored_min_score is not None
                      else getattr(state, "min_score", 0.0) or 0.0)
    ctx = {
        "entries": entries,
        "matrix_element": matrix_element,
        "background": background,
        "rule_set": rule_set,
        "min_score": min_score,
    }

    # --- per-region measurement ---------------------------------------------
    region_records = _region_records(
        state, flat_maps, smoothed_maps, columns_el, measured_el,
        flat_region, flat_phase, locked_flat, valid_mask, score_flat,
        entries, background, geometry, particles, ctx,
        n_px=n_px, n_valid=n_valid, n_classified=n_classified,
        n_rows=n_rows, n_cols=n_cols, area_ok=area_ok, warnings=warnings)

    regions_table = Table(
        name="regions",
        columns=_region_columns(columns_el, area_ok, bool(smoothed_maps)),
        rows=region_records,
    )

    # --- per-particle measurement -------------------------------------------
    particles_table = _particle_table(
        particles, particle_grid, flat_maps, columns_el, measured_el,
        flat_phase, flat_region, locked_flat, valid_mask, entries,
        region_records, background, ctx, geometry=geometry, opts=opts,
        area_ok=area_ok,
        n_px=n_px, n_valid=n_valid, n_classified=n_classified,
        particle_basis=particle_basis, warnings=warnings)

    # --- per-phase measurement ----------------------------------------------
    phases_table = _phase_table(
        state, entries, flat_maps, columns_el, measured_el, flat_phase,
        flat_region, locked_flat, valid_mask, score_flat, particles,
        region_records, background, geometry=geometry, area_ok=area_ok,
        n_px=n_px, n_valid=n_valid, n_classified=n_classified)

    # --- definitions ---------------------------------------------------------
    definitions_table, def_claims = _definitions_table(
        state, flat_maps, background, n_px=n_px, area_ok=area_ok,
        geometry=geometry, warnings=warnings)

    # attribute each region to a definition, now that the claims exist
    _attribute_definitions(region_records, def_claims, flat_region, warnings)
    _grade_particle_source(particles_table, region_records, particle_basis)

    # --- the key to the picture ----------------------------------------------
    # Unconditional, and NOT gated on `map_png`. The colours are a fact about
    # the map the app is showing right now; a reader who exported without the
    # PNGs still has a screenshot in a slide deck, and an empty column would
    # say the phase has no colour rather than that they turned the picture
    # off. BEFORE the plausibility block so the column order comes out
    # deterministic: identity, then its colour, then the verdict on it.
    colours = colour_columns(state, opts, phases_table, regions_table,
                             warnings)

    # --- plausibility --------------------------------------------------------
    # LAST, because it reads the finished cells rather than recomputing them,
    # and `assignment_source` is only final after `_attribute_definitions`.
    # Before `_provenance`, because the warnings it raises have to be in the
    # record — `_provenance` snapshots the bag with `list(warnings)`.
    phase_findings = _apply_plausibility(
        phases_table, entries=entries, index_key="phase_index",
        name_key="phase_name", measured_el=measured_el,
        background=background, matrix_element=matrix_element,
        after="phase_map_colour_hex")
    _apply_plausibility(
        regions_table, entries=entries, index_key="phase_final_index",
        name_key="phase_final", measured_el=measured_el,
        background=background, matrix_element=matrix_element,
        after="phase_map_colour_hex")
    # Particles too. It was the one table without a review_flag, and the QA
    # tester's answer to "which table do you filter row by row on a weekly
    # batch" was this one — 167 rows against 8 phase rows. The findings are
    # discarded on purpose: warnings still come from phases.csv alone, or a
    # single wrong phase would shout once per connected component of it.
    _apply_plausibility(
        particles_table, entries=entries, index_key="phase_index",
        name_key="phase_name", measured_el=measured_el,
        background=background, matrix_element=matrix_element,
        after="phase_formula_ascii")
    plausibility = _plausibility_warnings(
        phase_findings, phases_table.rows, warnings)

    # --- the picture ---------------------------------------------------------
    # The bar on the figure is horizontal, so the x step is its scale. Same
    # guard as `ExportGeometry.area_available`: None, NaN, inf and <= 0 all
    # mean "no bar", not an OverflowError blamed on the renderer.
    _sx = getattr(geometry, "step_x_um", None)
    try:
        _sx = float(_sx) if _sx is not None else None
    except (TypeError, ValueError):
        _sx = None
    images, map_figures = _map_images(
        state, opts, warnings,
        step_um=_sx if _sx is not None and math.isfinite(_sx) and _sx > 0 else None)

    # --- labels --------------------------------------------------------------
    labels = _label_arrays(
        phase_grid, region_grid if has_regions else None,
        particles, particle_grid, entries, locked_flat.reshape(n_rows, n_cols),
        opts.connectivity)

    # --- warnings that need finished tables ----------------------------------
    # BEFORE `_provenance`, which snapshots the bag with `list(warnings)`.
    _warn_ids_renumbered(state, warnings)
    _warn_matrix_particles(phases_table, ctx.get("matrix_element"), warnings)

    # --- provenance ----------------------------------------------------------
    provenance = _provenance(
        state, geometry, opts, entries, measured_el, columns_el,
        n_px=n_px, n_valid=n_valid, n_classified=n_classified,
        # Intersected here, where both masks are in scope, for the same reason
        # the table columns intersect: classified and valid-EDS are different
        # sets and their raw ratio can exceed 1.
        n_classified_valid=int(np.count_nonzero((flat_phase >= 0) & valid_mask)),
        n_regions=len(region_records), n_particles=len(particles_table),
        particle_basis=particle_basis, locked_flat=locked_flat,
        smoothing=smoothing, flat_maps=flat_maps,
        particles_table=particles_table, background=background,
        ctx=ctx, warnings=warnings, area_ok=area_ok,
        plausibility=plausibility, phases_table=phases_table,
        colours=colours)
    # How the `_figure.png` pictures relate to the 1:1 rasters: factor, size,
    # the bar's length. A reader measuring on the figure needs these numbers.
    provenance["map_figures"] = map_figures

    pixel_columns = _pixel_columns(columns_el, area_ok)

    def _pixels() -> Iterator[Dict[str, Any]]:
        return _pixel_rows(
            flat_phase, flat_region, score_flat, locked_flat, valid_mask,
            flat_maps, columns_el, measured_el, entries, particles,
            particle_grid, opts.connectivity, n_rows, n_cols, geometry,
            area_ok)

    return ExportTables(
        phases=phases_table,
        regions=regions_table,
        particles=particles_table,
        definitions=definitions_table,
        provenance=provenance,
        labels=labels,
        pixel_rows=_pixels,
        pixel_columns=pixel_columns,
        images=images,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# pieces of build_tables
# ---------------------------------------------------------------------------

def _map_images(
    state: Any, opts: ExportOptions, warnings: List[Dict[str, str]],
    step_um: Optional[float] = None,
) -> Tuple[Dict[str, bytes], Dict[str, Any]]:
    """``{"phase_map.png": bytes, ...}`` — the picture the app draws — plus
    ``phase_map_figure.png`` / ``region_map_figure.png``, the same pixels
    magnified with a scale bar for a report, and a record of how.

    The 1:1 files are kept exactly as the renderer made them (every pixel one
    scan point, byte-identical to the screen). The tester who opened
    ``phase_map.png`` found a 21 x 22 px picture and no scale, which is right
    for a raster and useless in a report; the ``_figure`` files are for the
    report. ``step_um`` is the scan step; without it the figure carries no
    bar.

    THROUGH ``phase_map_store``'s OWN RENDERERS, decoded from the base64 they
    already return. Not a second drawing: a picture in the export folder that
    coloured a phase differently from the screen would be worse than no
    picture, because a reader comparing the two would conclude one of them is
    a different map. The colour overrides are passed through for the same
    reason — a user who recoloured a phase gets that phase in that colour.

    WHY IT IS OPT-IN-BY-DEFAULT RATHER THAN OPTIONAL. The reader who asked for
    it had used the export to find a wrong headline number and said a picture
    would have caught it in four seconds; the alternative on offer was a
    right-click on a different page into a different dialog with five
    decisions she had no opinion about. A default-off artefact would have been
    the same wall with an extra checkbox.

    A FAILURE HERE COSTS A PICTURE, NEVER THE NUMBERS. Pillow missing, a
    state the renderer cannot draw, a region grid that is not there — each
    warns and returns what it managed. The tables are the export.
    """
    if not (opts.artefacts or {}).get("map_png", True):
        return {}, {}
    try:
        from backend.api.services.phase_map_store import (
            render_phase_map_to_base64,
            render_region_map_to_base64,
        )
    except Exception as exc:
        logger.debug("eds_export: map renderers unavailable", exc_info=True)
        _warn(warnings, "map_png_failed",
              f"The map images could not be rendered ({exc}); every number in "
              f"this export is unaffected.")
        return {}, {}

    import base64

    out: Dict[str, bytes] = {}
    try:
        out["phase_map.png"] = base64.b64decode(
            render_phase_map_to_base64(state, opts.colour_overrides))
    except Exception as exc:
        logger.debug("eds_export: phase map render failed", exc_info=True)
        _warn(warnings, "map_png_failed",
              f"phase_map.png could not be rendered ({exc}); every number in "
              f"this export is unaffected.")

    # Only when there ARE regions. `render_region_map_to_base64` raises on a
    # pixel-mode map, and a warning saying "the region picture is missing"
    # about a map that has no regions would be noise, not information.
    if getattr(state, "region_grid", None) is not None:
        try:
            out["region_map.png"] = base64.b64decode(
                render_region_map_to_base64(state))
        except Exception as exc:
            logger.debug("eds_export: region map render failed", exc_info=True)
            _warn(warnings, "map_png_failed",
                  f"region_map.png could not be rendered ({exc}); every "
                  f"number in this export is unaffected.")

    # The report pictures. A failure costs the figure, never the 1:1 file.
    # No figure when it would be the 1:1 picture again: a map already over
    # 1000 px is not magnified, and without a scan step there is no bar to
    # add, so the file would be a byte-different duplicate.
    figures: Dict[str, Any] = {}
    for name in list(out):
        fig_name = name[:-4] + "_figure.png"
        if not _figure_worthwhile(out[name], step_um):
            continue
        try:
            out[fig_name], figures[fig_name] = _figure_png(out[name], step_um)
        except Exception as exc:
            logger.debug("eds_export: figure render failed", exc_info=True)
            _warn(warnings, "map_png_failed",
                  f"{fig_name} could not be rendered ({exc}); {name} and every "
                  f"number in this export are unaffected.")
    return out, figures


#: What both renderers paint where nothing was assigned: ``(60, 60, 60)``.
#: Read off ``render_phase_map_to_base64`` / ``render_region_map_to_base64``
#: rather than guessed, because a legend that names the wrong grey is worse
#: than no legend -- it tells the reader the hole in their map is a phase.
_NO_DATA_HEX = "#3c3c3c"

#: Cell for a row whose colour this module could not establish. Never a
#: plausible-looking hex: printing somebody else's colour beside this row's
#: name is the one failure a legend must not have.
_UNKNOWN_HEX = ""


def colour_columns(
    state: Any, opts: ExportOptions,
    phases_table: Table, regions_table: Table,
    warnings: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Put the picture's own colours in the tables. Returns the record.

    WHY. The PNGs shipped without a key. A tester matched colours to phases
    by counting pixels in the image and comparing the counts against
    ``frac_of_scan``: "that worked, but it is a puzzle, not a legend. Two
    phases of similar area and I could not have done it at all." Two phases
    of similar area is the ordinary case, so the picture was decorative for
    anybody who had not already read the tables.

    WHY A COLUMN AND NOT A legend.txt. A column is beside the row it
    describes, sorts and filters with it, and survives being pasted into
    something else -- and the reader who needs the legend is already in the
    table, because the fraction they are matching colours against is a
    column. A separate file would have to be joined back by hand.

    THE COLOURS COME FROM THE SAME FUNCTIONS THE RENDERERS CALL, with the
    same overrides, so a legend cannot disagree with the image it describes
    -- the identical reason :func:`_map_images` draws through
    ``phase_map_store`` instead of drawing a second picture.

    TWO PICTURES, SO TWO COLUMNS ON regions.csv. A region is one colour in
    ``region_map.png`` (its own, one per region) and a different colour in
    ``phase_map.png`` (its phase's, shared with every other region carrying
    that phase). Each column names the file it keys, for the same reason
    every fraction in this export names its denominator.

    A FAILURE HERE COSTS A LEGEND, NEVER THE NUMBERS. The palette helpers are
    imported late; an import that is not there warns, leaves the cells empty
    and returns a record saying so.
    """
    phase_col = "phase_map_colour_hex"
    region_col = "region_map_colour_hex"

    record: Dict[str, Any] = {
        "no_data_hex": _NO_DATA_HEX,
        "phase_map_png": (
            f"phases.{phase_col} is the colour that phase is drawn in, and "
            f"regions.{phase_col} is the colour that region is drawn in -- "
            f"the same value for every region carrying the same phase."),
        "region_map_png": (
            f"regions.{region_col} is the colour that region is drawn in. "
            f"One colour per region: several regions can carry the SAME "
            f"phase and they are still different colours in this picture."),
        "no_data_note": (
            f"Pixels the map left unassigned are drawn {_NO_DATA_HEX} in "
            f"both pictures. No table row carries that colour -- the "
            f"unclassified phase row is a row about pixels, and this is the "
            f"colour they are painted -- so it is stated here."),
        "source": (
            "phase_map_store.palette_hex_for_state and "
            "phase_map_store.region_color_hex, the same functions the "
            "renderers call, with the same colour overrides. The legend "
            "therefore cannot disagree with the image."),
        "overrides_applied": sorted(str(k) for k in
                                    (opts.colour_overrides or {})),
    }

    for table, cols in ((phases_table, [phase_col]),
                        (regions_table, [region_col, phase_col])):
        for col in cols:
            if col not in table.columns:
                # Immediately behind the identity the colour belongs to,
                # which is where a reader matching a colour to a name is
                # already looking.
                anchor = ("phase_label" if col == phase_col
                          and table is phases_table else
                          "region_id" if col == region_col else
                          "phase_final_formula_ascii")
                at = (table.columns.index(anchor) + 1
                      if anchor in table.columns else len(table.columns))
                table.columns.insert(at, col)
        for row in table.rows:
            for col in cols:
                row.setdefault(col, _UNKNOWN_HEX)

    try:
        from backend.api.services.phase_map_store import (
            palette_hex_for_state, region_color_hex,
        )
        palette = list(palette_hex_for_state(state, opts.colour_overrides))
    except Exception as exc:
        logger.debug("eds_export: palette helpers unavailable", exc_info=True)
        _warn(warnings, "legend_unavailable",
              f"The colour columns could not be filled ({exc}), so the "
              f"pictures in this export have no key. Every number is "
              f"unaffected.")
        record["available"] = False
        record["reason"] = str(exc)
        return record

    record["available"] = True
    record["palette"] = list(palette)

    def _phase_hex(idx: Any) -> str:
        try:
            i = int(idx)
        except (TypeError, ValueError):
            return _UNKNOWN_HEX
        # -1 is the unclassified row, and it has no palette entry by
        # construction: it is drawn in the no-data grey, not in a phase
        # colour, so that is what the cell says.
        if i < 0:
            return _NO_DATA_HEX
        return str(palette[i]) if i < len(palette) else _UNKNOWN_HEX

    for row in phases_table.rows:
        row[phase_col] = _phase_hex(row.get("phase_index"))

    for row in regions_table.rows:
        row[phase_col] = _phase_hex(row.get("phase_final_index"))
        try:
            row[region_col] = str(region_color_hex(int(row.get("region_id"))))
        except Exception:
            row[region_col] = _UNKNOWN_HEX

    return record


def _warn_ids_renumbered(state: Any, warnings: List[Dict[str, str]]) -> None:
    """A boundary edit renumbers region AND particle ids. Say so, loudly.

    ``region_grid_edited`` was a bare boolean in the provenance whose meaning
    lived in a Python comment. A failure-analyst tester measured 58 of 135
    particle ids changing after a single merge at 1.000 pixel overlap -- the
    ids moved although not one pixel did -- and called it the thing most
    likely to break a figure. A field nobody reads cannot do that job; a
    warning can, because the UI surfaces warnings.
    """
    try:
        summary = state.edit_summary()
    except Exception:
        return
    if not summary or not summary.get("region_grid_edited"):
        return
    _warn(warnings, "region_ids_renumbered",
          "The region grid was edited after classification (a merge, split, "
          "grow or edge snap). Region ids and particle ids are ordered by "
          "(region_id, -n_px, centroid), so those edits RENUMBER them: an id "
          "written on an earlier figure or quoted in an earlier report may "
          "now point at a different object. Every measurement in this export "
          "is of the map as it stands; only the ids are affected. See "
          "provenance.hand_edits.log for what was done.")


def _warn_matrix_particles(
    phases_table: Optional[Table], matrix_element: Optional[str],
    warnings: List[Dict[str, str]],
) -> None:
    """Counting "particles" of the matrix is a category error. Name it.

    A tester read "143 Al particles" off ``phases.csv`` -- connected
    components of the continuous matrix -- and said a student would paste it
    into a thesis. The rows are not wrong: they are components of a phase,
    counted exactly as every other phase's are. What is wrong is calling
    them particles, and the file has to be the thing that says so, because
    the column heading cannot.

    Identified by the matrix ELEMENT the scorer used, not by area: a
    genuinely dominant intermetallic in a small field of view would trip an
    area rule, and the matrix element is the value the classification itself
    decided on.
    """
    if phases_table is None or not matrix_element:
        return
    named = []
    for row in phases_table.rows:
        if int(row.get("phase_index", -1)) < 0:
            continue
        comp = row.get("_nominal_comp") or {}
        n_part = int(row.get("n_particles") or 0)
        if n_part <= 1 or not comp:
            continue
        # The matrix phase is the one whose nominal composition is that
        # single element and nothing else -- Al.cif, not Al6Fe.cif.
        if list(comp.keys()) == [str(matrix_element)]:
            named.append((row.get("phase_name"), n_part))
    for name, n_part in named:
        _warn(warnings, "matrix_counted_as_particles",
              f"{name} is the matrix phase ({matrix_element}), and its "
              f"n_particles = {n_part} counts connected components of a "
              f"CONTINUOUS phase, not particles. The number is what it says "
              f"it is -- components of the matrix left over between the "
              f"second-phase objects -- and it is not an inclusion count. "
              f"Every particles.csv row carrying this phase is such a "
              f"component. Filter on phase_name before quoting a particle "
              f"count or a size distribution.")


def _valid_eds_mask(flat_maps: Mapping[str, np.ndarray], n_px: int) -> np.ndarray:
    """Pixels that carry a usable EDS measurement.

    Through ``chemistry_score.has_chemistry`` rather than a second rule, so
    "valid EDS" means here exactly what it means to the classifier that
    refused to classify those pixels. Falls back to "any element non-zero"
    only if that import is unavailable, which is a slimmed-deployment case
    and not a normal one.
    """
    if not flat_maps:
        return np.zeros(n_px, dtype=bool)
    try:
        from backend.api.services.chemistry_score import has_chemistry
        mask = np.asarray(has_chemistry(dict(flat_maps)), dtype=bool).ravel()
        if mask.size == n_px:
            return mask
    except Exception:
        logger.debug("eds_export: has_chemistry unavailable", exc_info=True)
    total = np.zeros(n_px, dtype=np.float64)
    for arr in flat_maps.values():
        total += np.nan_to_num(arr, nan=0.0)
    return total > 1e-9


def _background_levels(flat_maps: Mapping[str, np.ndarray]) -> Dict[str, float]:
    try:
        from backend.api.services.chemistry_score import background_levels
        return dict(background_levels(dict(flat_maps)))
    except Exception:
        logger.debug("eds_export: background_levels unavailable", exc_info=True)
        return {}


def _ranking_means(
    flat_maps: Mapping[str, np.ndarray],
    mask_flat: np.ndarray,
    background: Mapping[str, float],
) -> Dict[str, float]:
    """The mean dict :func:`rank_candidates` expects.

    Built through ``eds_wand.selection_stats`` and filtered with
    ``_CHEM_IGNORE``, which is precisely what ``_region_detail`` does. Not a
    reimplementation: the enrichment half of that function was already wrong
    once when it was reimplemented by hand ("Al 49.68x enriched" on a map
    whose background IS aluminium), and the mean half is what the candidate
    ranking stands on.

    Returned on the map's OWN basis — unrenormalised, and rounded exactly as
    the inspector rounds it — because matching the inspector is the whole
    point. The export puts it through :func:`to_scored_basis` before
    subtracting a nominal composition from it; see that function for the
    measured cost of not doing so.
    """
    try:
        from backend.api.services.crystal_hint_phase_fit import _CHEM_IGNORE
        from backend.api.services.eds_wand import selection_stats
    except Exception:
        logger.debug("eds_export: selection_stats unavailable", exc_info=True)
        return {}
    stats = selection_stats(dict(flat_maps), mask_flat,
                            background=dict(background) or None)
    return {el: float(v) for el, v in (stats.get("mean_at_pct") or {}).items()
            if el not in _CHEM_IGNORE}


def _resolve_rule_set(state: Any) -> Any:
    """The phase rules the classification gated with, rebuilt from the map.

    ``settings["rules"]`` is the wire-form rule payload, written at the moment
    of classification and carried through the sidecar, so this is the same
    object ``_auto_classify_blocking`` handed to the clustering and not a
    reconstruction from something adjacent. ``None`` when the map predates
    the recording or carried no rules — in which case nothing was gated and
    an ungated replay is the correct replay.
    """
    payload = _recorded_settings(state).get("rules")
    if not payload:
        return None
    try:
        from backend.api.services.phase_rules import rule_set_from_dict
        return rule_set_from_dict(payload)
    except Exception:
        logger.debug("eds_export: rule_set_from_dict unavailable", exc_info=True)
        return None


def _resolve_matrix_element(
    flat_maps: Mapping[str, np.ndarray], rule_set: Any,
) -> Optional[str]:
    """Which element the scorer treated as the solvent.

    Same precedence as ``eds_clustering.cluster_and_match``: the user's
    declaration wins over the inference, in both modes. It is not a
    preference — ratio matching is DEFINED on the non-matrix elements and the
    matrix is exempt from the enrichment gate, so getting this wrong inverts
    the whole metric rather than nudging it.
    """
    declared = list(getattr(rule_set, "matrix_elements", None) or [])
    if declared:
        return str(declared[0])
    try:
        from backend.api.services.chemistry_score import infer_matrix_element
        return infer_matrix_element(dict(flat_maps))
    except Exception:
        logger.debug("eds_export: matrix inference unavailable", exc_info=True)
        return None


def _interior_flat(mask2d: np.ndarray) -> np.ndarray:
    """The eroded interior of an object, falling back to the whole of it.

    :func:`eds_clustering._interior_mask` verbatim — same 3x3 structure, same
    ``border_value=0``, same fallback — because the mean this produces is the
    mean the map was NAMED on. Reimplementing it a shade differently would
    put the export's "the software's own answer" a shade off the software's
    own answer, which is the whole defect being fixed here.

    The fallback matters for thin or small objects: a one-pixel-wide feature
    erodes to nothing, and an empty mean is worse than a rim-contaminated one.

    Cropped to the object's bounding box, then padded with one ring of False
    — which IS ``border_value=0``, including at the raster edge, where a
    pixel is never interior under that convention either. Not a different
    answer, a cheaper route to the same one: eroding the whole raster once
    per particle cost 6.35 s of a 34.3 s profile on a 400x400 map with 7695
    particles. ``test_the_cropped_erosion_is_the_classifications_erosion``
    pins the equivalence, edge-touching and single-pixel objects included,
    because "a shade different from the classification" is precisely the
    defect this whole file is being repaired for.
    """
    rows = np.flatnonzero(mask2d.any(axis=1))
    if rows.size == 0:
        return mask2d.ravel()
    cols = np.flatnonzero(mask2d.any(axis=0))
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1

    padded = np.zeros((r1 - r0 + 2, c1 - c0 + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask2d[r0:r1, c0:c1]
    er = ndimage.binary_erosion(
        padded, structure=np.ones((3, 3), dtype=bool), border_value=0)[1:-1, 1:-1]
    if not er.any():
        return mask2d.ravel()
    out = np.zeros(mask2d.shape, dtype=bool)
    out[r0:r1, c0:c1] = er
    return out.ravel()


def _object_means(
    flat_maps: Mapping[str, np.ndarray], masks: Sequence[np.ndarray],
) -> Dict[str, np.ndarray]:
    """``{element: (n_obj,)}`` — one pseudo-pixel per object, for the scorer.

    An empty mask contributes 0 for every element, which the scorer reads as
    "no measurement" through ``no_data_score`` and answers with a score of 0
    — no phase, rather than a phase asserted about no pixels.
    """
    n = len(masks)
    out: Dict[str, np.ndarray] = {}
    for el, arr in flat_maps.items():
        v = np.zeros(n, dtype=np.float64)
        for j, m in enumerate(masks):
            if m is not None and m.any():
                v[j] = float(arr[m].mean())
        out[el] = v
    return out


def _auto_columns() -> List[str]:
    """The two answers and the one boolean that compares them.

    ``score_*`` comes from the scorer that made the map; ``*_mad`` from the
    mean-absolute-deviation ranking the region inspector shows. Both are in
    the file, both say in their own name which they are, and only the first
    one drives ``phase_agrees``.
    """
    return [
        "phase_auto", "phase_auto_index", "phase_agrees",
        "score_auto", "runner_up_phase", "runner_up_score", "margin_score",
        # THIRD place, because a margin cannot be read without it. A tester
        # put it plainly: a 0.15 margin between first and second means one
        # thing when third is far behind and another when third is 0.01
        # below second, and the second case is a three-way tie the file was
        # reporting as a decision. Deciding metric only -- the MAD block is
        # a second opinion and a third-place second opinion is noise.
        "third_phase", "third_score",
        "phase_second_opinion_mad", "gap_mad_at_pct", "runner_up_mad",
        "margin_mad_at_pct",
    ]


def _auto_cells(
    row: Dict[str, Any],
    ranked: Sequence[Tuple[int, float]],
    mad: Sequence[Tuple[int, float]],
    entries: Sequence[Any],
    final_idx: int,
    min_score: float,
) -> None:
    """Fill the auto-answer block. Empty cells where there is no answer.

    ``phase_auto_index`` is -1 when the best score fails ``min_score``,
    because that is precisely what the classifier does with such a region:
    leave it unassigned. ``score_auto`` is still filled in that case — "the
    best chemistry could manage was 0.21, and the floor is 0.30" is the
    sentence a reader needs, and blanking the score would hide it.

    ``phase_agrees`` is a plain equality including the -1 == -1 case: a
    region the map left unclassified and the chemistry would also leave
    unclassified is an AGREEMENT, and the previous code called it False.
    """
    if not ranked:
        row["phase_auto"] = ""
        row["phase_auto_index"] = -1
        row["score_auto"] = None
        row["runner_up_phase"] = ""
        row["runner_up_score"] = None
        row["third_phase"] = ""
        row["third_score"] = None
        row["margin_score"] = None
        # No replay, so no verdict. False here would assert a disagreement
        # this export did not measure.
        row["phase_agrees"] = None
    else:
        best_idx, best_s = ranked[0]
        cleared = float(best_s) >= float(min_score)
        auto_idx = int(best_idx) if cleared else -1
        row["phase_auto"] = _phase_name(entries[best_idx]) if cleared else ""
        row["phase_auto_index"] = auto_idx
        row["score_auto"] = float(best_s)
        if len(ranked) > 1:
            row["runner_up_phase"] = _phase_name(entries[ranked[1][0]])
            row["runner_up_score"] = float(ranked[1][1])
        else:
            row["runner_up_phase"] = ""
            row["runner_up_score"] = None
        if len(ranked) > 2:
            row["third_phase"] = _phase_name(entries[ranked[2][0]])
            row["third_score"] = float(ranked[2][1])
        else:
            # Empty, not repeated: a library with two candidates has no
            # third place, and printing the runner-up again would invent one.
            row["third_phase"] = ""
            row["third_score"] = None
        row["margin_score"] = _margin(ranked, higher_is_better=True)
        row["phase_agrees"] = bool(int(final_idx) == auto_idx)

    if mad:
        row["phase_second_opinion_mad"] = _phase_name(entries[mad[0][0]])
        row["gap_mad_at_pct"] = float(mad[0][1])
        row["runner_up_mad"] = (_phase_name(entries[mad[1][0]])
                                if len(mad) > 1 else "")
        row["margin_mad_at_pct"] = _margin(mad)
    else:
        row["phase_second_opinion_mad"] = ""
        row["gap_mad_at_pct"] = None
        row["runner_up_mad"] = ""
        row["margin_mad_at_pct"] = None


def _chem_ignore() -> frozenset:
    """The elements every scorer drops. One source, so nothing can drift.

    Empty on a slimmed deployment where the import is unavailable, which
    makes the enrichment denominator wider than the scorer's rather than
    narrower — the direction that under-reports enrichment instead of
    inventing it.
    """
    try:
        from backend.api.services.crystal_hint_phase_fit import _CHEM_IGNORE
        return frozenset(_CHEM_IGNORE)
    except Exception:
        logger.debug("eds_export: _CHEM_IGNORE unavailable", exc_info=True)
        return frozenset()


def _enrichment_columns(columns_el: Sequence[str]) -> List[str]:
    out: List[str] = []
    for el in columns_el:
        out.append(f"enrichment_{el}")
        out.append(f"background_at_pct_{el}")
    return out


def _enrichment_cells(
    row: Dict[str, Any],
    columns_el: Sequence[str],
    measured: Sequence[str],
    means: Mapping[str, float],
    background: Mapping[str, float],
) -> None:
    """How enriched this object is over the map's own background, and over WHAT.

    The background sits in the next column on purpose. A tester asked for
    enrichment WITH its background and got neither; this repo has already
    shipped an "Al 49.68x enriched" readout on a map whose background IS
    aluminium, and the only defence a reader has against that is being able
    to see the divisor.

    Definition is ``eds_wand.selection_stats``' — the object's share of the
    SCORED elements divided by the map-level median share — because the
    classifier's own enrichment gate is defined against that same background,
    and two definitions of "enriched" on one page is how the 49.68x happened.
    Unrounded here; ``selection_stats`` rounds to 2 dp for the screen.

    ``background_at_pct_El`` is that median share as an at%, so it is on the
    same basis as a renormalised composition and can be compared by eye.
    """
    measured_set = set(measured)
    ignore = _chem_ignore()
    total = sum(float(v) for el, v in means.items()
                if el in measured_set and el not in ignore)
    for el in columns_el:
        ekey, bkey = f"enrichment_{el}", f"background_at_pct_{el}"
        if el not in measured_set:
            row[ekey] = NOT_MEASURED
            row[bkey] = NOT_MEASURED
            continue
        bg = float(background.get(el, 0.0))
        row[bkey] = bg * 100.0
        if bg <= 1e-9 or total <= 1e-9 or el not in means:
            # No background to be enriched over, or nothing to divide. A
            # ratio would be an infinity dressed as a measurement.
            row[ekey] = None
        else:
            row[ekey] = (float(means[el]) / total) / bg


def _scored_sum(
    means: Mapping[str, float], measured: Sequence[str],
) -> Optional[float]:
    """The object's at% summed over the SCORED elements — a quality measure.

    Not ``raw_at_pct_sum``: see ``provenance.composition.raw_sum_note``. The
    at% maps arrive already renormalised over every measured element, so this
    is 100 minus whatever carbon and oxygen the object carries. Measured on
    the real SampleB scan it runs 91.30 to 100.00 per pixel, and a region
    sitting at the bottom of that range is a region a third of whose signal
    the scorer never saw.
    """
    measured_set = set(measured)
    ignore = _chem_ignore()
    vals = [float(v) for el, v in means.items()
            if el in measured_set and el not in ignore]
    if not vals:
        return None
    s = sum(vals)
    return s if math.isfinite(s) else None


def _recorded_settings(state: Any) -> Dict[str, Any]:
    """What the map itself says about how it was made. ``{}`` when it says nothing.

    ``PhaseMapState.settings`` is the wire-form classification request, written
    at the moment of classification and carried through the sidecar. Read
    through ``getattr`` because this function is also handed hand-built states
    in tests, and states restored from sidecars written before the settings
    were recorded at all.
    """
    s = getattr(state, "settings", None)
    return dict(s) if isinstance(s, Mapping) else {}


def _box_um(scale_px: Optional[int], step_x, step_y) -> Optional[Dict[str, Any]]:
    """``{"x": .., "y": ..}`` edge lengths of the smoothing box, or ``None``.

    Through ``eds_clustering.scale_box_um`` rather than a second multiplication
    here, so "5 px is a 3.3 um box" means in the export exactly what it means
    on the page. Two edges because the box is square in PIXELS: a scan with
    unequal X and Y steps has a rectangular physical box, and quoting one
    number for it would be a wrong number in a report.
    """
    if scale_px is None:
        return None
    try:
        from backend.api.services.eds_clustering import scale_box_um
        x, y = scale_box_um(int(scale_px), step_x, step_y)
    except Exception:
        logger.debug("eds_export: scale_box_um unavailable", exc_info=True)
        return None
    return None if x is None and y is None else {"x": x, "y": y}


def _resolve_smoothing(
    state: Any, opts: ExportOptions, geometry: ExportGeometry,
) -> Dict[str, Any]:
    """Which smoothing width this export may honestly claim, and from where.

    Three cases, and the note has to differ in each:

    ``map``     the width was recorded with the map. Authoritative — it is the
                width the regions were actually formed on, so it wins over
                anything the export dialog was told. A caller who asked for a
                different one is told it was superseded rather than silently
                obeyed, because the column is named ``smoothed_`` and claims to
                explain THESE regions.
    ``caller``  the map predates the recording and the caller stated a width.
                Usable, and labelled as the caller's assertion.
    ``None``    nobody knows. No width, no box, no smoothed columns.

    A map classified per pixel records ``scale = None``. That is a recorded
    fact — nothing was smoothed — and it must not fall through to the caller's
    width, which is why the test is ``"scale" in stored`` and not truthiness.
    """
    stored = _recorded_settings(state)
    caller_px = (None if opts.smoothing_scale_px is None
                 else int(opts.smoothing_scale_px))

    if "scale" in stored:
        raw = stored.get("scale")
        scale_px = None if raw is None else int(raw)
        # The step AS RESOLVED AT CLASSIFICATION TIME, falling back to this
        # export's geometry. They agree on the same file; the recorded one is
        # preferred because it is the number the width was converted with.
        step_x = stored.get("step_x_um", geometry.step_x_um)
        step_y = stored.get("step_y_um", geometry.step_y_um)
        return {
            "source": "map",
            "scale_px": scale_px,
            "box_um": _box_um(scale_px, step_x, step_y),
            "report": stored.get("scale_report") or opts.scale_report,
            "mode": stored.get("mode"),
            "caller_scale_px": (caller_px if caller_px != scale_px else None),
        }

    return {
        "source": ("caller" if caller_px is not None else None),
        "scale_px": caller_px,
        "box_um": _box_um(caller_px, geometry.step_x_um, geometry.step_y_um),
        "report": opts.scale_report,
        "mode": None,
        "caller_scale_px": None,
    }


def _smoothing_note(rec: Mapping[str, Any]) -> str:
    """One sentence saying where the width came from.

    Never a width this module inferred: a box is quoted only when a step size
    was there to convert it with, and the ``None`` case says plainly that
    nobody recorded one.
    """
    scale_px, box = rec.get("scale_px"), rec.get("box_um")
    box_text = ""
    if box:
        x, y = box.get("x"), box.get("y")
        if x is not None and y is not None and abs(float(x) - float(y)) > 1e-9:
            box_text = f" = a {float(x):g} x {float(y):g} um box"
        elif x is not None or y is not None:
            edge = float(x if x is not None else y)
            box_text = f" = a {edge:g} um box"

    if rec.get("source") == "map":
        if scale_px is None:
            return ("Recorded with the map: it was classified per pixel, so "
                    "nothing was smoothed and there is no smoothed "
                    "composition to report.")
        note = (f"Recorded with the map: the regions were formed on "
                f"composition smoothed with a {int(scale_px)} px box"
                f"{box_text}.")
        if rec.get("caller_scale_px") is not None:
            note += (f" The export caller asked for "
                     f"{int(rec['caller_scale_px'])} px; the recorded width "
                     f"wins, because these columns describe how THESE regions "
                     f"were formed.")
        return note

    if rec.get("source") == "caller":
        return (f"Smoothed composition computed with a {int(scale_px)} px box"
                f"{box_text}, as supplied by the export caller. This map "
                f"carries no record of its own width, so that is the caller's "
                f"assertion and not a measurement of how the regions were "
                f"made.")

    return ("The smoothing width used to form the regions is not "
            "persisted with the map and was not supplied by the "
            "caller, so the smoothed-composition columns were "
            "omitted rather than recomputed at a default width.")


def _smoothed_maps(
    flat_maps: Mapping[str, np.ndarray],
    n_rows: int, n_cols: int,
    rec: Dict[str, Any],
    warnings: List[Dict[str, str]],
) -> Tuple[Dict[str, np.ndarray], str]:
    """The composition the regions were FORMED on, when it is knowable.

    Regions are formed on smoothed composition and reported on raw. Both are
    legitimate and they differ by enough to matter — measured on the SampleB
    silicon particle, 40.97 at% raw against 36.88 smoothed — so ``basis``
    names the reported one and the smoothed mean is exported alongside.

    "When it is knowable" is the catch, and :func:`_resolve_smoothing` settles
    it. Smoothing here at a default would produce a column labelled "smoothed"
    that was smoothed differently from the thing it claims to explain, so the
    columns are omitted unless a width was either recorded with the map or
    stated by the caller.
    """
    scale_px = rec.get("scale_px")
    if scale_px is None:
        return {}, _smoothing_note(rec)
    try:
        from backend.api.services.eds_clustering import _smooth_maps
        sm = _smooth_maps(dict(flat_maps), n_rows, n_cols, int(scale_px))
        return ({el: np.asarray(v, dtype=np.float64).ravel()
                 for el, v in sm.items()},
                _smoothing_note(rec))
    except Exception as exc:
        _warn(warnings, "smoothing_failed",
              f"Could not recompute the smoothed composition ({exc}); the "
              f"smoothed columns were omitted.")
        return {}, f"Smoothing failed: {exc}"


def _region_columns(
    columns_el: Sequence[str], area_ok: bool, with_smoothed: bool,
) -> List[str]:
    # `n_px_valid_eds` sits immediately before the fraction it is the
    # numerator of. The export's stated rule is "additions only at the end",
    # and this breaks it deliberately: a fraction whose count is somewhere
    # else in the row is exactly the column the reader cannot check, and
    # these files have never shipped, so there is no reader to be
    # compatible with — only a column order worth getting right.
    cols = [
        "region_id", "n_px", "n_px_classified", "n_px_hand",
        "frac_of_scan", "n_px_valid_eds", "frac_of_valid_eds",
        "frac_of_classified",
    ]
    if area_ok:
        cols += ["area_um2", "sqrt_area_um", "ecd_um"]
    cols += [
        "phase_final", "phase_final_index", "phase_final_formula",
        # A legend cannot carry `sd_0302719.cif`, and it cannot carry
        # `Mn4.512Al127.296Fe19.488Si16.704` either. See `phase_label`.
        "phase_final_label", "phase_final_formula_ascii",
    ]
    cols += _auto_columns()
    cols += [
        "assignment_source", "definition_name", "definition_match_frac",
        "n_particles", "mean_score", "basis", "scored_at_pct_sum",
    ]
    cols += _composition_columns("", columns_el)
    cols += _enrichment_columns(columns_el)
    if with_smoothed:
        cols += _composition_columns("smoothed_", columns_el, with_sd=False)
    return cols


def _region_records(
    state, flat_maps, smoothed_maps, columns_el, measured_el,
    flat_region, flat_phase, locked_flat, valid_mask, score_flat,
    entries, background, geometry, particles, ctx,
    *, n_px, n_valid, n_classified, n_rows, n_cols, area_ok, warnings,
) -> List[Dict[str, Any]]:
    region_phase = list(getattr(state, "region_phase", None) or [])
    if not region_phase and getattr(state, "region_grid", None) is None:
        return []

    n_particles_by_region: Dict[int, int] = {}
    for p in particles:
        n_particles_by_region[p.region_id] = \
            n_particles_by_region.get(p.region_id, 0) + 1

    # The deciding scorer, replayed for every region in ONE batched pass, on
    # the eroded interior — which is the population `_match_labels` named each
    # region from. Batched because `score_phase_ratio` is vectorised over
    # pixels: 22 candidates over 7 regions is 22 calls, not 154.
    region_2d = flat_region.reshape(n_rows, n_cols)
    # An empty region contributes an all-zero pseudo-pixel, which the scorer
    # reads through `no_data_score` as "no measurement" and answers with 0 —
    # no phase, rather than a phase asserted about no pixels.
    interiors = [_interior_flat(region_2d == sid)
                 for sid in range(len(region_phase))]
    ranked_all = deciding_rank(
        _object_means(flat_maps, interiors), entries,
        matrix_element=ctx["matrix_element"], background=background,
        rule_set=ctx["rule_set"])

    rows: List[Dict[str, Any]] = []
    for sid in range(len(region_phase)):
        mask = flat_region == sid
        n = int(mask.sum())
        # Numerator from the same set as the denominator — see the module
        # docstring. `n` would count pixels that carry no EDS at all.
        n_v = int(np.count_nonzero(mask & valid_mask))
        row: Dict[str, Any] = {
            "region_id": sid,
            "n_px": n,
            "n_px_classified": int((flat_phase[mask] >= 0).sum()) if n else 0,
            "n_px_hand": int(locked_flat[mask].sum()) if n else 0,
            "frac_of_scan": (n / n_px) if n_px else None,
            "n_px_valid_eds": n_v,
            "frac_of_valid_eds": (n_v / n_valid) if n_valid else None,
            "frac_of_classified": (n / n_classified) if n_classified else None,
            "n_particles": int(n_particles_by_region.get(sid, 0)),
            "mean_score": _finite_mean(score_flat[mask]) if n else None,
            # Regions are FORMED on smoothed composition and REPORTED on raw.
            # The flag is not decoration: a threshold copied off a raw reading
            # and typed into a window that is evaluated on smoothed values
            # catches nothing, and this column is where that becomes visible.
            "basis": "raw",
            # Filled by _attribute_definitions once the claims are known.
            "assignment_source": "auto",
            "definition_name": "",
            "definition_match_frac": None,
        }
        if area_ok:
            area = float(n) * float(geometry.step_x_um) * float(geometry.step_y_um)
            row["area_um2"] = area
            # Murakami's inclusion-rating parameter: sqrt(area) is the length
            # a failure analyst rates an inclusion by, and it is not
            # interchangeable with ECD (sqrt(pi/4) = 0.886 apart for a disc,
            # further for anything elongated).
            row["sqrt_area_um"] = math.sqrt(area) if area > 0 else 0.0
            row["ecd_um"] = (2.0 * math.sqrt(area / math.pi)) if area > 0 else 0.0

        final_idx = int(region_phase[sid]) if sid < len(region_phase) else -1
        final_entry = entries[final_idx] if 0 <= final_idx < len(entries) else None
        row["phase_final"] = _phase_name(final_entry) or ("" if final_idx < 0
                                                          else str(final_idx))
        row["phase_final_index"] = final_idx
        row["phase_final_formula"] = (str(getattr(final_entry, "formula", ""))
                                      if final_entry is not None else "")
        row["phase_final_label"] = phase_label(final_entry)
        row["phase_final_formula_ascii"] = formula_ascii(
            getattr(final_entry, "formula", "") if final_entry is not None else "")

        stats = composition_for_mask(dict(flat_maps), mask) if n else {}
        _composition_cells("", columns_el, measured_el, stats, row)
        plain = {el: float(p[0]) for el, p in stats.items()}
        _enrichment_cells(row, columns_el, measured_el, plain, background)
        row["scored_at_pct_sum"] = _scored_sum(plain, measured_el) if n else None
        if smoothed_maps:
            sstats = composition_for_mask(dict(smoothed_maps), mask) if n else {}
            _composition_cells("smoothed_", columns_el, measured_el, sstats, row,
                               with_sd=False)

        # TWO answers, each named for the metric that produced it.
        #
        #   ranked  the scorer that MADE this map, replayed on the same
        #           population it named the region from. This is what
        #           `phase_agrees` compares against, so the boolean now means
        #           "the map differs from what the chemistry alone would
        #           choose" rather than "two different metrics disagree",
        #           which it fired on for 4 of 7 regions of the real SampleB
        #           scan with not one hand-painted pixel anywhere.
        #   mad     the region inspector's mean absolute deviation, kept
        #           because a distance in at% is easier to judge than a 0..1
        #           score — but on the SAME BASIS as the nominal composition
        #           it is subtracted from. See `to_scored_basis`.
        ranked = list(ranked_all[sid]) if (n and sid < len(ranked_all)) else []
        if n and entries:
            mad = rank_candidates(
                to_scored_basis(_ranking_means(flat_maps, mask, background)),
                entries)
        else:
            mad = []
        _auto_cells(row, ranked, mad, entries, final_idx, ctx["min_score"])

        # Graded, not asserted. `hand` on a region with 100 painted pixels out
        # of 3327 tells a reader a human decided the region, and a human
        # decided 3 % of it.
        row["assignment_source"] = _grade_source(
            n_px=n, n_hand=int(row["n_px_hand"]), agrees=row["phase_agrees"])
        rows.append(row)
    return rows


def _grade_source(*, n_px: int, n_hand: int, agrees: Optional[bool],
                  definition: bool = False) -> str:
    """How much of this row a human decided, and whether we can even tell.

    ``hand``       every pixel was painted.
    ``mixed``      some were. The old code said ``hand`` for both, so 100
                   painted pixels claimed a 3327-pixel region.
    ``definition`` a composition window claims it (set by the caller, and it
                   outranks ``auto``: a window is a tracked decision).
    ``auto``       nothing was painted AND the map matches the replayed
                   chemistry. Only then can this row positively assert that
                   nothing touched it.
    ``untracked``  nothing was painted and the map does NOT match. Something
                   moved it that leaves no trace in the stored map — naming a
                   region, merging two, splitting one, snapping an edge; see
                   ``provenance.hand_edits.note``. ``auto`` here would be a
                   claim rather than a measurement.

    ``agrees is None`` means the replay produced no answer at all (an empty
    library, a scoring failure). Then it cannot be graded either way and
    ``untracked`` is the honest cell.
    """
    if n_px > 0 and n_hand >= n_px:
        return "hand"
    if n_hand > 0:
        return "mixed"
    if definition:
        return "definition"
    return "auto" if agrees is True else "untracked"


def _definitions_table(
    state, flat_maps, background, *, n_px, area_ok, geometry, warnings,
) -> Tuple[Table, List[Dict[str, Any]]]:
    """One row per region definition, and what it actually claimed.

    Re-evaluated here rather than trusted from a stored count, because the
    stored map may have been merged, split, grown or edge-snapped since, and
    the honest answer to "what would this window claim" is to ask it again.
    It is asked on the RAW composition — the classification asked it on the
    smoothed composition — so ``definition_match_frac`` on the region rows is
    an attribution strength and not a bookkeeping identity. Both facts are in
    the provenance.
    """
    columns = [
        "kind", "index", "name", "phase_key", "clause_count", "clauses_json",
        "human_readable", "n_px_claimed", "frac_of_scan", "n_px_overlap",
        "reason",
    ]
    if area_ok:
        columns.insert(columns.index("frac_of_scan") + 1, "area_um2")

    raw_defs = list(getattr(state, "region_defs", None) or [])
    if not raw_defs or not flat_maps:
        return Table("definitions", columns, []), []

    try:
        from backend.api.services.phase_rules import (
            region_defs_from_list, region_labels,
        )
    except Exception as exc:
        _warn(warnings, "definitions_unavailable",
              f"Could not parse the region definitions ({exc}); "
              f"definitions.csv is empty.")
        return Table("definitions", columns, []), []

    defs = region_defs_from_list(raw_defs)
    try:
        assign = region_labels(defs, dict(flat_maps),
                               background=dict(background) or None)
    except Exception as exc:
        _warn(warnings, "definitions_eval_failed",
              f"Could not re-evaluate the region definitions ({exc}).")
        return Table("definitions", columns, []), []

    labels = np.asarray(assign.labels).ravel()
    rows: List[Dict[str, Any]] = []
    claims: List[Dict[str, Any]] = []
    for i, d in enumerate(defs):
        raw = raw_defs[i] if i < len(raw_defs) else {}
        n_claimed = int(assign.counts[i]) if i < len(assign.counts) else 0
        row: Dict[str, Any] = {
            "kind": "region_definition",
            "index": i,
            "name": str(getattr(d, "name", "") or ""),
            "phase_key": str(getattr(d, "phase_key", "") or ""),
            "clause_count": (len(d.elements) + len(d.ratios)
                             + len(d.enrichment)),
            # Verbatim, so the window can be read back and retyped. A summary
            # would be the thing that loses the one clause nobody remembers.
            "clauses_json": json.dumps(raw, sort_keys=True, ensure_ascii=False),
            "human_readable": _def_text(d),
            "n_px_claimed": n_claimed,
            "frac_of_scan": (n_claimed / n_px) if n_px else None,
            "n_px_overlap": (int(assign.overlaps[i])
                             if i < len(assign.overlaps) else 0),
            "reason": (assign.reasons[i] if i < len(assign.reasons) else None),
        }
        if area_ok:
            row["area_um2"] = (float(n_claimed) * float(geometry.step_x_um)
                               * float(geometry.step_y_um))
        rows.append(row)
        claims.append({"index": i, "name": row["name"],
                       "mask": labels == i})
    return Table("definitions", columns, rows), claims


def _def_text(d: Any) -> str:
    """One readable line for a window, for the human reading the CSV."""
    parts: List[str] = []
    for c in getattr(d, "elements", ()) or ():
        parts.append(f"{c.element} {_range_text(c.min_at_pct, c.max_at_pct, ' at%')}")
    for c in getattr(d, "ratios", ()) or ():
        parts.append(f"{c.numerator}/{c.denominator} "
                     f"{_range_text(c.min_ratio, c.max_ratio, '')}")
    for c in getattr(d, "enrichment", ()) or ():
        parts.append(f"{c.element} enrichment "
                     f"{_range_text(c.min_factor, c.max_factor, 'x')}")
    return " AND ".join(parts)


def _range_text(lo, hi, unit: str) -> str:
    if lo is not None and hi is not None:
        return f"{lo:g}-{hi:g}{unit}"
    if lo is not None:
        return f">= {lo:g}{unit}"
    if hi is not None:
        return f"<= {hi:g}{unit}"
    return "any"


def _attribute_definitions(
    region_rows: List[Dict[str, Any]],
    claims: Sequence[Dict[str, Any]],
    flat_region: np.ndarray,
    warnings: List[Dict[str, str]],
) -> None:
    """Which hand-written window, if any, this region is.

    Measured rather than looked up. The obvious implementation — "definitions
    occupy region ids 0..n-1, because that is how the clustering labels them"
    — is true at the moment of classification and false immediately after a
    merge, which pops an id and shifts every id above it down by one. So the
    attribution is an OVERLAP: the window that claims the largest share of
    this region's pixels wins, and ``definition_match_frac`` says how large
    that share is, so a weak attribution is visible instead of being asserted.

    ``hand``/``mixed`` beat ``definition``: a hand-painted pixel is an
    explicit decision and the strongest statement in the file about where a
    phase boundary is.
    """
    if not claims:
        return
    for row in region_rows:
        n = int(row.get("n_px") or 0)
        if n == 0:
            continue
        mask = flat_region == int(row["region_id"])
        best_name, best_frac = "", 0.0
        for claim in claims:
            hit = int(np.count_nonzero(mask & claim["mask"]))
            frac = hit / n
            if frac > best_frac:
                best_name, best_frac = claim["name"], frac
        if best_frac > 0.5:
            row["definition_name"] = best_name
            row["definition_match_frac"] = float(best_frac)
            if row.get("assignment_source") not in ("hand", "mixed"):
                row["assignment_source"] = _grade_source(
                    n_px=n, n_hand=int(row.get("n_px_hand") or 0),
                    agrees=row.get("phase_agrees"), definition=True)
        elif best_frac > 0.0:
            # Recorded but not claimed: a window overlapping a third of a
            # region is information, and calling that region "definition" on
            # the strength of it would be an overstatement.
            row["definition_name"] = best_name
            row["definition_match_frac"] = float(best_frac)


def _particle_columns(
    columns_el: Sequence[str], area_ok: bool,
) -> List[str]:
    cols = [
        "particle_id", "region_id", "phase_name", "phase_index",
        "phase_formula", "phase_label", "phase_formula_ascii",
    ]
    # The particle's OWN answer, ranked on the particle's OWN composition.
    # Inheriting the region's was the defect: two particles of 286 px and
    # 175 px both printed the 660-px region's margin, and a number set beside
    # a 175-pixel object is read as describing it.
    cols += _auto_columns()
    cols += [
        "n_px", "n_px_core", "n_px_hand",
        "frac_of_scan", "n_px_valid_eds", "frac_of_valid_eds",
        "frac_of_classified",
    ]
    if area_ok:
        cols += [
            "area_um2", "sqrt_area_um", "ecd_um",
            "feret_max_um", "feret_min_um", "feret_angle_deg",
            "aspect_ratio", "perimeter_crofton_um", "circularity", "solidity",
            "centroid_x_um", "centroid_y_um",
        ]
    cols += [
        "centroid_row", "centroid_col",
        "bbox_row_min", "bbox_row_max", "bbox_col_min", "bbox_col_max",
        "touches_edge", "below_size_limit",
        # WHO decided this particle's phase, and — separately — whether this
        # particle's own chemistry backs it. Two questions, two columns,
        # because one column answering both is the defect that put
        # `assignment_source = mixed` on 110 of 167 rows carrying
        # `n_px_hand = 0`. See `_grade_particle_source`.
        "assignment_source", "chemistry_dissents",
        "region_margin_score", "basis",
        "scored_at_pct_sum",
    ]
    cols += _composition_columns("", columns_el)
    cols += _enrichment_columns(columns_el)
    cols += _composition_columns("core_", columns_el)
    return cols


def _feret_angle_deg(
    hull_px: Optional[np.ndarray], step_x_um: float, step_y_um: float,
) -> Optional[float]:
    """Orientation of the maximum Feret diameter.

    CONVENTION, and it has to be stated or the column is unusable: degrees
    from the +x axis (increasing COLUMN), measured COUNTER-CLOCKWISE as seen
    on the map, folded into [0, 180) because a diameter has no head or tail.
    The raster's row index increases DOWNWARD, so the row component is
    negated — without that every angle in the file is mirrored, and "are the
    inclusions aligned with the extrusion direction" gets the wrong answer
    with full confidence.

    Measured on the same hull ``scale_to_um`` measures ``feret_max_um`` from,
    scaled the same way, so the length and its direction describe one line.
    That has a consequence worth stating: ``feret_max`` on a rectangle is its
    DIAGONAL, so a 1x7 bar reports 171.87 deg and not 180 — the direction of
    the longest caliper, which is what ``feret_max_um`` is the length of. For
    an elongated object the two converge; for a stubby one they do not, and
    ``aspect_ratio`` is the column that says which case a row is.

    ``None`` when the hull is unavailable (the defensive Qhull fallback in
    ``eds_particles._hull_measures``) — an angle guessed from a bounding box
    would be a different quantity.
    """
    if hull_px is None or len(hull_px) < 2:
        return None
    v = np.asarray(hull_px, dtype=np.float64)
    y = v[:, 0] * float(step_y_um)          # rows -> the Y step
    x = v[:, 1] * float(step_x_um)          # cols -> the X step
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    i, j = np.unravel_index(int(np.argmax(d2)), d2.shape)
    if d2[i, j] <= 0.0:
        return None
    ang = math.degrees(math.atan2(-(y[i] - y[j]), x[i] - x[j]))
    return float(ang % 180.0)


def _particle_table(
    particles: Sequence[ParticleGeometry], particle_grid, flat_maps,
    columns_el, measured_el, flat_phase, flat_region, locked_flat, valid_mask,
    entries, region_records, background, ctx, *, geometry, opts, area_ok,
    n_px, n_valid, n_classified, particle_basis, warnings,
) -> Table:
    columns = _particle_columns(columns_el, area_ok)
    rows: List[Dict[str, Any]] = []
    if not particles:
        return Table("particles", columns, rows)

    region_by_id = {int(r["region_id"]): r for r in region_records}
    min_px = int(opts.min_particle_px)
    scale_failed = False

    # Materialised once: the deciding scorer runs over every particle in one
    # batched pass below, and re-deriving the masks per particle would mean
    # walking the label grid twice.
    pairs = list(particle_masks(particle_grid, particles,
                                connectivity=opts.connectivity))
    # Same population rule as a region: the eroded interior, falling back to
    # the whole object. Most particles are small enough to fall back, which
    # is the honest answer for them — a 4-pixel particle has no interior.
    interiors = [_interior_flat(m2) for _p, m2 in pairs]
    ranked_all = deciding_rank(
        _object_means(flat_maps, interiors), entries,
        matrix_element=ctx["matrix_element"], background=background,
        rule_set=ctx["rule_set"])

    for k, (p, mask2d) in enumerate(pairs):
        mask = mask2d.ravel()
        # The core is the SAME erosion find_particles counted for n_px_core —
        # imported rather than redefined, so `n_px_core` and the core
        # composition can never describe different pixel sets.
        core2d = ndimage.binary_erosion(
            mask2d, structure=eds_particles._EROSION_STRUCT)
        core = core2d.ravel()

        # Same set as the denominator — see the module docstring.
        n_v = int(np.count_nonzero(mask & valid_mask))

        row: Dict[str, Any] = {
            "particle_id": p.particle_id,
            "n_px": p.n_px,
            "n_px_core": p.n_px_core,
            "n_px_hand": int(locked_flat[mask].sum()),
            "frac_of_scan": (p.n_px / n_px) if n_px else None,
            "n_px_valid_eds": n_v,
            "frac_of_valid_eds": (n_v / n_valid) if n_valid else None,
            "frac_of_classified": ((p.n_px / n_classified)
                                   if n_classified else None),
            "centroid_row": p.centroid_row,
            "centroid_col": p.centroid_col,
            "bbox_row_min": p.bbox_row_min,
            "bbox_row_max": p.bbox_row_max,
            "bbox_col_min": p.bbox_col_min,
            "bbox_col_max": p.bbox_col_max,
            # An edge particle is truncated: its size is a LOWER BOUND and
            # every size distribution is biased without this column.
            "touches_edge": bool(p.touches_edge),
            # Flag, never a filter. Silent exclusion was named a disqualifier.
            "below_size_limit": bool(min_px > 0 and p.n_px < min_px),
            "basis": "raw",
            "assignment_source": "auto",
        }

        if area_ok:
            try:
                scaled = scale_to_um(p, float(geometry.step_x_um),
                                     float(geometry.step_y_um))
            except ValueError:
                scale_failed = True
            else:
                # Renamed in the row, not in `scale_to_um`: the correction
                # has to survive somebody opening the CSV without the JSON,
                # and a column called `perimeter_um` does not say that it is
                # a Crofton estimate rather than a boundary-pixel count.
                scaled["perimeter_crofton_um"] = scaled.pop("perimeter_um")
                area = float(scaled.get("area_um2") or 0.0)
                scaled["sqrt_area_um"] = math.sqrt(area) if area > 0 else 0.0
                scaled["feret_angle_deg"] = _feret_angle_deg(
                    p.hull_px, float(geometry.step_x_um),
                    float(geometry.step_y_um))
                row.update(scaled)

        # Which region and which phase this particle belongs to. Under
        # `particle_basis == "phase"` the component IS the phase, so the
        # region column is empty rather than invented.
        if particle_basis == "region":
            region_id = int(p.region_id)
            row["region_id"] = region_id
            rec = region_by_id.get(region_id)
            row["phase_name"] = rec.get("phase_final", "") if rec else ""
            row["phase_index"] = rec.get("phase_final_index", -1) if rec else -1
            row["phase_formula"] = rec.get("phase_final_formula", "") if rec else ""
            row["phase_label"] = rec.get("phase_final_label", "") if rec else ""
            row["phase_formula_ascii"] = (
                rec.get("phase_final_formula_ascii", "") if rec else "")
            # The region's margin, under a name that says whose it is. It used
            # to be called `margin_at_pct` and sit beside the particle's size.
            row["region_margin_score"] = (rec.get("margin_score")
                                          if rec else None)
        else:
            row["region_id"] = None
            idx = int(p.region_id)          # the label grid WAS the phase grid
            entry = entries[idx] if 0 <= idx < len(entries) else None
            row["phase_name"] = _phase_name(entry)
            row["phase_index"] = idx
            row["phase_formula"] = (str(getattr(entry, "formula", ""))
                                    if entry is not None else "")
            row["phase_label"] = phase_label(entry)
            row["phase_formula_ascii"] = formula_ascii(
                getattr(entry, "formula", "") if entry is not None else "")
            row["region_margin_score"] = None

        stats = composition_for_mask(dict(flat_maps), mask)
        _composition_cells("", columns_el, measured_el, stats, row)
        plain = {el: float(pair[0]) for el, pair in stats.items()}
        _enrichment_cells(row, columns_el, measured_el, plain, background)
        row["scored_at_pct_sum"] = _scored_sum(plain, measured_el)
        # Core-only. A particle that erodes to nothing legitimately has no
        # core composition; the cells are empty, which is the true statement,
        # and n_px_core = 0 next to it says why.
        core_stats = (composition_for_mask(dict(flat_maps), core)
                      if core.any() else {})
        _composition_cells("core_", columns_el, measured_el, core_stats, row)

        # This particle's own ranking, on this particle's own composition.
        ranked = list(ranked_all[k]) if k < len(ranked_all) else []
        mad = rank_candidates(
            to_scored_basis(_ranking_means(flat_maps, mask, background)),
            entries) if entries else []
        _auto_cells(row, ranked, mad, entries, int(row["phase_index"]),
                    ctx["min_score"])

        # Provenance first, on THIS particle's own painted pixels only. The
        # region's contribution is added by `_grade_particle_source` once the
        # region rows are final, and it contributes only the part of the
        # verdict that is genuinely the region's.
        row["assignment_source"] = _grade_source(
            n_px=int(p.n_px), n_hand=int(row["n_px_hand"]),
            agrees=row["phase_agrees"])
        # A chemistry observation, not a provenance claim. `phase_agrees`
        # carries the same fact in the opposite polarity and is shared with
        # the regions table; this column exists because a reader who meets
        # `phase_agrees = False` beside `assignment_source` reads it as "a
        # human changed this", and on a clean scan it fires on tens of
        # percent of the particles. It is the ordinary consequence of naming
        # a whole region from its interior mean and then measuring each
        # component of it separately.
        row["chemistry_dissents"] = (row["phase_agrees"] is False)
        rows.append(row)

    if scale_failed:
        _warn(warnings, "particle_scale_failed",
              "The step size was rejected while scaling particles; the "
              "micrometre columns are incomplete.")
    if min_px > 0:
        n_flagged = sum(1 for r in rows if r["below_size_limit"])
        if n_flagged:
            _warn(warnings, "below_size_limit",
                  f"{n_flagged} of {len(rows)} particles are smaller than "
                  f"min_particle_px = {min_px}. They are FLAGGED, not "
                  f"removed — filter on below_size_limit if you want them "
                  f"out, and say so when you report the count.")
    return Table("particles", columns, rows)


def _grade_particle_source(
    particles: Table, region_records: Sequence[Dict[str, Any]],
    particle_basis: str,
) -> None:
    """Grade each particle on ITS OWN painted pixels, then on its region's decision.

    THE DEFECT THIS REPLACES. The previous version copied the region's grade
    onto every particle of that region wholesale. Measured by a QA tester on
    a real export: **110 of 167 particles read ``assignment_source = mixed``
    while carrying ``n_px_hand = 0``**, with exactly one particle in the
    whole file actually holding a painted pixel — and the file's own legend
    defines ``mixed`` as "SOME pixels were painted". 110 rows of an
    auditor's table asserted a human decision that did not happen. It is the
    same over-claim that was fixed at region level in the previous round,
    displaced one table down. Reproduced here on the real SampleB scan
    (90x120, mode=cluster, scale_um=1.5, one 36-pixel paint): 3 particles
    read ``mixed``, **2 of them with ``n_px_hand = 0``**.

    THE SPLIT. Two facts were riding in one cell and they belong to
    different owners:

      * ``hand`` / ``mixed`` are statements about PAINTED PIXELS, and a
        particle's painted pixels are its own. They are graded here on this
        particle's ``n_px_hand`` against this particle's ``n_px``, and on
        nothing else. A particle with none of its own is never ``mixed``.
      * ``definition`` / ``auto`` / ``untracked`` are statements about how
        the PHASE NAME was arrived at, and under region basis that decision
        genuinely was the region's — the particle was never named
        individually. So a particle with no painted pixels inherits exactly
        that part of its region's verdict, recomputed with ``n_hand = 0`` so
        the region's painting cannot leak back in.

    Under phase basis there is no region, so the particle's own grade stands.

    WHAT STILL DOES NOT PROPAGATE: ``phase_agrees``, ``chemistry_dissents``,
    ``score_auto`` and the margins stay the PARTICLE's own. They answer a
    different question — does this object's chemistry support the name it
    carries — and a particle dissenting from its region is exactly the row
    worth finding.
    """
    if particle_basis != "region":
        return
    by_id = {int(r["region_id"]): r for r in region_records}
    for row in particles.rows:
        # This particle's own paint is the whole of the hand/mixed verdict.
        if row.get("assignment_source") in ("hand", "mixed"):
            continue
        rec = by_id.get(row.get("region_id"))
        if not rec:
            continue
        # The region's decision, with its own painting deliberately zeroed:
        # a region that is `mixed` because a human touched a different
        # particle still decided THIS particle's phase automatically (or by
        # a definition, or untracked), and that is the part being inherited.
        row["assignment_source"] = _grade_source(
            n_px=int(rec.get("n_px") or 0), n_hand=0,
            agrees=rec.get("phase_agrees"),
            definition=bool(rec.get("definition_name")
                            and (rec.get("definition_match_frac") or 0.0) > 0.5))


def _phase_table(
    state, entries, flat_maps, columns_el, measured_el, flat_phase,
    flat_region, locked_flat, valid_mask, score_flat, particles,
    region_records, background, *, geometry, area_ok, n_px, n_valid,
    n_classified,
) -> Table:
    """One row per phase that owns pixels, plus a real ``unclassified`` row.

    ``unclassified`` is not a leftover to be subtracted from 100 %. Four
    personas asked for it as a first-class row and they are right: it has an
    area, it has a composition, and its composition is the most diagnostic
    thing in a bad map — unclassified pixels that all look like one chemistry
    are a missing library entry, unclassified pixels that look like noise are
    a bad scan.
    """
    columns = [
        "phase_index", "phase_name", "phase_label", "formula",
        "formula_ascii", "cif_filename",
        "space_group", "space_group_number", "crystal_system",
        "n_px", "n_px_hand", "n_regions", "n_particles",
        "frac_of_scan", "n_px_valid_eds", "frac_of_valid_eds",
        "frac_of_classified",
    ]
    if area_ok:
        columns += ["area_um2", "sqrt_area_um",
                    # The size distribution behind n_particles, without
                    # having to group particles.csv by hand. Mean AND max
                    # AND sd, because on this kind of map the mean is
                    # dominated by a cloud of near-resolution objects and
                    # the max is the one that matters to a failure analyst.
                    "particle_area_mean_um2", "particle_area_max_um2",
                    "particle_area_sd_um2"]
    columns += ["mean_score", "median_score", "basis", "scored_at_pct_sum"]
    columns += _composition_columns("", columns_el)
    columns += _enrichment_columns(columns_el)
    columns += [f"nominal_at_pct_{el}" for el in columns_el]

    region_ids_by_phase: Dict[int, int] = {}
    for r in region_records:
        idx = int(r.get("phase_final_index", -1))
        region_ids_by_phase[idx] = region_ids_by_phase.get(idx, 0) + 1

    particles_by_phase: Dict[int, int] = {}
    # Areas kept alongside the count so the phase row can summarise them.
    # From n_px x the two steps, exactly as every other area in this file:
    # re-deriving them from the written particle rows would let the two
    # tables disagree.
    px_by_phase: Dict[int, List[int]] = {}
    if region_records:
        phase_of_region = {int(r["region_id"]): int(r.get("phase_final_index", -1))
                           for r in region_records}
        for p in particles:
            idx = phase_of_region.get(int(p.region_id), -1)
            particles_by_phase[idx] = particles_by_phase.get(idx, 0) + 1
            px_by_phase.setdefault(idx, []).append(int(p.n_px))
    else:
        for p in particles:
            idx = int(p.region_id)
            particles_by_phase[idx] = particles_by_phase.get(idx, 0) + 1
            px_by_phase.setdefault(idx, []).append(int(p.n_px))

    rows: List[Dict[str, Any]] = []

    def _add(idx: int, entry: Any, mask: np.ndarray) -> None:
        n = int(mask.sum())
        # The row's pixels that ALSO carry EDS. Because the phase masks
        # partition the raster (`unclassified` included), these numerators
        # sum to n_valid and the column therefore closes to 1.
        n_v = int(np.count_nonzero(mask & valid_mask))
        row: Dict[str, Any] = {
            "phase_index": idx,
            "phase_name": _phase_name(entry) if entry is not None else "unclassified",
            "phase_label": phase_label(entry) if entry is not None else "unclassified",
            "formula": str(getattr(entry, "formula", "")) if entry is not None else "",
            "formula_ascii": (formula_ascii(getattr(entry, "formula", ""))
                              if entry is not None else ""),
            "cif_filename": str(getattr(entry, "cif_filename", "")) if entry is not None else "",
            "space_group": str(getattr(entry, "space_group", "")) if entry is not None else "",
            "space_group_number": (getattr(entry, "space_group_number", None)
                                   if entry is not None else None),
            "crystal_system": str(getattr(entry, "crystal_system", "")) if entry is not None else "",
            "n_px": n,
            "n_px_hand": int(locked_flat[mask].sum()) if n else 0,
            "n_regions": int(region_ids_by_phase.get(idx, 0)),
            "n_particles": int(particles_by_phase.get(idx, 0)),
            "frac_of_scan": (n / n_px) if n_px else None,
            "n_px_valid_eds": n_v,
            "frac_of_valid_eds": (n_v / n_valid) if n_valid else None,
            # The unclassified row is BY DEFINITION not part of the classified
            # set, so dividing it by that set would print a fraction that can
            # exceed 1 in a column called "frac". Empty is the true answer.
            "frac_of_classified": (None if idx < 0
                                   else ((n / n_classified) if n_classified
                                         else None)),
            "mean_score": _finite_mean(score_flat[mask]) if n else None,
            "median_score": (float(np.median(score_flat[mask]))
                             if n else None),
            "basis": "raw",
        }
        if area_ok:
            area = (float(n) * float(geometry.step_x_um)
                    * float(geometry.step_y_um))
            row["area_um2"] = area
            row["sqrt_area_um"] = math.sqrt(area) if area > 0 else 0.0
            px_area = (float(geometry.step_x_um) * float(geometry.step_y_um))
            areas = np.asarray(px_by_phase.get(idx, []), dtype=np.float64) * px_area
            row["particle_area_mean_um2"] = (float(areas.mean())
                                             if areas.size else None)
            row["particle_area_max_um2"] = (float(areas.max())
                                            if areas.size else None)
            # Population sd (ddof=0), the same convention as sd_within, and
            # None rather than 0.0 for a single particle: one object has no
            # spread to report and "0" would read as "they are all the same
            # size".
            row["particle_area_sd_um2"] = (float(areas.std())
                                           if areas.size > 1 else None)
        stats = composition_for_mask(dict(flat_maps), mask) if n else {}
        _composition_cells("", columns_el, measured_el, stats, row)
        plain = {el: float(pair[0]) for el, pair in stats.items()}
        _enrichment_cells(row, columns_el, measured_el, plain, background)
        row["scored_at_pct_sum"] = _scored_sum(plain, measured_el) if n else None
        nominal = (getattr(entry, "composition", None) or {}) if entry is not None else {}
        for el in columns_el:
            # The library's nominal composition. Always a number when the
            # phase names the element, whether or not the scan measured it —
            # this column is what the phase CLAIMS, next to what was measured.
            row[f"nominal_at_pct_{el}"] = (float(nominal[el]) if el in nominal
                                           else None)
        # Not a column -- `Table` writes `columns` and ignores everything
        # else -- but the matrix-phase check needs the nominal composition as
        # a set, and re-deriving it from the per-element cells would mean
        # telling a genuine 0.0 apart from an absent element by inspecting
        # strings.
        row["_nominal_comp"] = {k: float(v) for k, v in nominal.items()}
        rows.append(row)

    for i, entry in enumerate(entries):
        mask = flat_phase == i
        if not mask.any():
            # A candidate that won no pixels is not a result. It is listed in
            # the provenance's participating-phase list, where it belongs.
            continue
        _add(i, entry, mask)
    _add(-1, None, flat_phase < 0)
    return Table("phases", columns, rows)


def _ustr(values: Sequence[str]) -> np.ndarray:
    """A numpy unicode array that survives ``np.load`` without ``allow_pickle``.

    ``np.array([], dtype="U")`` is not portable across numpy versions, so the
    empty case is spelled out.
    """
    if not len(values):
        return np.empty(0, dtype="U1")
    return np.array([str(v) for v in values], dtype="U")


def _n_phases_with_pixels(state: Any) -> int:
    present = np.unique(np.asarray(state.phase_grid))
    return int(sum(1 for v in present if int(v) >= 0))


def _label_arrays(
    phase_grid, region_grid, particles, particle_grid, entries, locked2d,
    connectivity: int,
) -> Dict[str, np.ndarray]:
    """Int rasters plus the lookup tables that give the ids meaning.

    A raster without a lookup is a picture of some integers. The lookups are
    parallel string arrays rather than a pickled dict so that ``np.load``
    without ``allow_pickle`` — the safe way — still reads everything.
    """
    out: Dict[str, np.ndarray] = {
        "phase_id": np.asarray(phase_grid, dtype=np.int32),
        "locked_mask": np.asarray(locked2d, dtype=bool),
        "phase_lut_index": np.array([i for i in range(len(entries))],
                                    dtype=np.int32),
        "phase_lut_name": _ustr([_phase_name(e) for e in entries]),
        "phase_lut_formula": _ustr([str(getattr(e, "formula", "")) for e in entries]),
        "phase_lut_cif": _ustr([str(getattr(e, "cif_filename", "")) for e in entries]),
        "phase_lut_space_group": _ustr([str(getattr(e, "space_group", "")) for e in entries]),
    }
    if region_grid is not None:
        out["region_id"] = np.asarray(region_grid, dtype=np.int32)

    # Particle raster: 0 = no particle, otherwise the particle_id used in
    # particles.csv, so a row in the CSV and a blob in the raster are the
    # same object.
    part = np.zeros(np.asarray(particle_grid).shape, dtype=np.int32)
    for p, mask in particle_masks(particle_grid, particles,
                                  connectivity=connectivity):
        part[mask] = p.particle_id
    out["particle_id"] = part
    out["particle_lut_id"] = np.array([p.particle_id for p in particles],
                                      dtype=np.int32)
    out["particle_lut_region"] = np.array([p.region_id for p in particles],
                                          dtype=np.int32)
    # -1 rather than 0 for "unclassified"/"no phase", matching phase_grid.
    out["note"] = _ustr([
        "phase_id / region_id: -1 means unclassified / no region.",
        "particle_id: 0 means no particle; ids match particles.csv.",
        f"connectivity used for particles: {int(connectivity)}",
        "phase_lut_* are parallel arrays indexed by phase_id.",
    ])
    return out


def _pixel_columns(columns_el: Sequence[str], area_ok: bool) -> List[str]:
    # 0-based, row-major; x_um = col * step_x_um, y increases DOWNWARD.
    # The convention is spelled out in provenance.coordinates, because this
    # is the table somebody else's script reads and a convention that lives
    # only in the manual is one the script gets wrong.
    cols = ["row", "col"]
    if area_ok:
        cols += ["x_um", "y_um"]
    cols += ["region_id", "particle_id", "phase_index", "phase_name",
             "score", "hand_edited", "has_eds", "scored_at_pct_sum"]
    cols += [f"at_pct_{el}" for el in columns_el]
    return cols


def _pixel_rows(
    flat_phase, flat_region, score_flat, locked_flat, valid_mask,
    flat_maps, columns_el, measured_el, entries, particles, particle_grid,
    connectivity, n_rows, n_cols, geometry, area_ok,
) -> Iterator[Dict[str, Any]]:
    """One row per pixel, yielded not accumulated.

    The method developer ranked this first of everything in the consultation:
    it is the artefact that lets somebody re-derive any of the other three
    tables and check them. It is also the one that is 100 MB on a real map,
    which is why it is opt-in and why this is a generator — the writer streams
    it straight to disk and the whole table is never in memory at once.
    """
    part = np.zeros(n_rows * n_cols, dtype=np.int32)
    for p, mask in particle_masks(particle_grid, particles,
                                  connectivity=connectivity):
        part[mask.ravel()] = p.particle_id

    names = [_phase_name(e) for e in entries]
    measured_set = set(measured_el)
    sx = float(geometry.step_x_um) if area_ok else None
    sy = float(geometry.step_y_um) if area_ok else None

    # The per-pixel at% over the SCORED elements, vectorised once rather than
    # summed per row: 485 000 Python-level sums is the difference between a
    # streamed write and a stall. See `provenance.composition.raw_sum_note`
    # for why this is not called `raw_at_pct_sum`.
    ignore = _chem_ignore()
    scored_els = [el for el in measured_set if el not in ignore]
    scored_sum = (np.sum([flat_maps[el] for el in scored_els], axis=0)
                  if scored_els else None)

    for i in range(n_rows * n_cols):
        r, c = divmod(i, n_cols)
        idx = int(flat_phase[i])
        row: Dict[str, Any] = {"row": r, "col": c}
        if area_ok:
            row["x_um"] = c * sx
            row["y_um"] = r * sy
        row["region_id"] = int(flat_region[i])
        row["particle_id"] = int(part[i])
        row["phase_index"] = idx
        row["phase_name"] = names[idx] if 0 <= idx < len(names) else ""
        row["score"] = float(score_flat[i])
        row["hand_edited"] = bool(locked_flat[i])
        row["has_eds"] = bool(valid_mask[i])
        row["scored_at_pct_sum"] = (None if scored_sum is None
                                    else float(scored_sum[i]))
        for el in columns_el:
            if el not in measured_set:
                row[f"at_pct_{el}"] = NOT_MEASURED
            else:
                row[f"at_pct_{el}"] = float(flat_maps[el][i])
        yield row


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

#: ``region_grid_edited`` was a bare boolean whose meaning lived in a Python
#: comment, while every other field in that block carries its note. A tester
#: measured 58 of 135 particle ids changing after a single merge at 1.000
#: pixel overlap -- the ids moved although not one pixel did -- so this is
#: the field that decides whether a figure labelled with particle or region
#: ids can still be trusted.
_REGION_GRID_EDITED_NOTE = (
    "True means at least one merge, split, grow or edge-snap moved pixels "
    "between regions after classification. Particle ids are ordered by "
    "(region_id, -n_px, centroid), so any of those RENUMBERS both region "
    "ids and particle ids: a merge pops an id and everything above it "
    "shifts down. An id burned onto a figure before such an edit no longer "
    "points at the same object. null means the map carries no edit record "
    "and cannot say either way -- never 'no such edit happened'."
)

#: ``paints`` and ``pixels_painted`` count different things over different
#: spans, and a reader meeting ``paints: 0`` beside ``pixels_painted: 25`` is
#: entitled to conclude that one of them is wrong.
_PAINTS_NOTE = (
    "paints counts paint OPERATIONS since the last classification; "
    "pixels_painted counts the pixels currently carrying a hand assignment "
    "(locked_mask), whenever they were painted. A re-classify starts a fresh "
    "counter but deliberately CARRIES the painted pixels across, so "
    "'paints: 0' beside 'pixels_painted: 25' is the ordinary reading of a "
    "map that was painted and then re-classified. Both are true; they answer "
    "different questions."
)


def _edit_record(state) -> Dict[str, Any]:
    """The hand-edit counts, from the store's record when it kept one.

    Two agents built the halves of this independently: the store learned to
    log merges, splits, grows, edge snaps and region namings, and this file
    kept emitting the hardcoded nulls it had used before that existed. The
    counts were tracked and then thrown away on the way out. This is the join.

    A state with no record at all (a hand-built one in a test, or a map
    restored from a sidecar written before tracking) still gets the honest
    nulls plus the note that says a null means "not tracked", never
    "none happened".
    """
    summary = None
    try:
        fn = getattr(state, "edit_summary", None)
        if callable(fn):
            summary = fn()
    except Exception:      # a record must never cost us the export
        logger.debug("eds_export: edit_summary unavailable", exc_info=True)
        summary = None

    # An untracked map takes the same branch as no record at all. The
    # store phrases its own note differently ('carries no edit record');
    # this file's contract, and its test, is that the words 'not tracked'
    # appear next to a null, because that is the sentence a reader has to
    # come away with.
    if not summary or not summary.get("tracked"):
        return {
            "regions_named": None, "merges": None, "splits": None,
            "grows": None, "edge_snaps": None, "boundary_edits": None,
            "tracked": False,
            # Null, never []: an empty list would say "no edits happened",
            # which is precisely the claim an untracked map cannot make.
            "log": None, "log_cap": None,
            "log_note": ("No edit record exists for this map, so the log is "
                         "null rather than empty — an empty list would "
                         "assert that nothing happened."),
            # Both notes belong here too. An untracked map is exactly the one
            # whose reader has the least to go on, and a note that appears
            # only when the field is populated is a note the reader who needs
            # it never sees.
            "region_grid_edited": None,
            "region_grid_edited_note": _REGION_GRID_EDITED_NOTE,
            "paints_note": _PAINTS_NOTE,
            "note": (
                "Only hand-PAINTED pixels are recorded here, via locked_mask. "
                "This map carries no edit record, so a null means 'not "
                "tracked', never 'none happened'. Note also that a "
                "hand-painted pixel survives a re-classify and a hand-given "
                "region name does not."),
        }

    counts = summary.get("counts") or {}
    out: Dict[str, Any] = {
        "tracked": bool(summary.get("tracked")),
        # THE LOG ITSELF. It was tracked, counted, summarised — and then
        # dropped on the way into the file: the record wrote
        # `log_retained: 5` beside "Complete: every edit since
        # classification is listed" and then listed nothing, so that
        # sentence was false about the file it appeared in. Carried
        # verbatim; `edit_summary()` already returns JSON-shaped entries
        # (`{"op": "merge", "detail": {...}}`).
        #
        # It also resolves an ambiguity nothing else can: `untracked` in
        # assignment_source collapses "a rename happened" with "the replay
        # produced no answer", and an entry with op = name_region tells the
        # two apart.
        "log": [dict(e) for e in (summary.get("log") or [])],
        "log_cap": summary.get("log_cap"),
        "log_note": (
            "One entry per edit since the last classification, oldest "
            "first, as the store recorded it: op plus the operation's own "
            "detail. n_edits is the true total and log_retained is how many "
            "of them are listed — they differ only when log_truncated is "
            "true, in which case these are the most recent log_cap. A "
            "re-classify starts a fresh log, so this is the record SINCE "
            "the classification the rest of this file describes."),
        # `boundary_edits` is the older name for the two ops that move a
        # region's outline. Kept as their sum so a reader who learned the old
        # key still finds it, with the two ops itemised beside it.
        "boundary_edits": (
            None if not summary.get("tracked")
            else int(counts.get("grows") or 0) + int(counts.get("edge_snaps") or 0)),
        "n_edits": summary.get("n_edits"),
        "log_truncated": summary.get("log_truncated"),
        "log_retained": summary.get("log_retained"),
        "note": summary.get("note"),
        # The four ops that renumber region ids, and therefore particle ids.
        # A figure labelled with particle ids and re-exported after a merge
        # will disagree with itself; this is how a reader finds out.
        "region_grid_edited": summary.get("region_grid_edited"),
        "region_grid_edited_note": _REGION_GRID_EDITED_NOTE,
        "paints_note": _PAINTS_NOTE,
    }
    for key, value in counts.items():
        out[str(key)] = value
    return out


def _provenance(
    state, geometry: ExportGeometry, opts: ExportOptions, entries,
    measured_el, columns_el, *, n_px, n_valid, n_classified,
    n_classified_valid, n_regions,
    n_particles, particle_basis, locked_flat, smoothing, flat_maps, background,
    ctx, warnings, area_ok, particles_table=None, plausibility=None,
    phases_table=None, colours=None,
) -> Dict[str, Any]:
    """The full run record. Everything a reader needs to attack the numbers."""
    # `particles_table` is a Table (name/columns/rows), not a sequence of
    # rows. Tolerate either, and an absent table, so a caller that builds no
    # particles still gets a record rather than a TypeError.
    _particle_rows = getattr(particles_table, "rows", particles_table) or []
    now = opts.timestamp or datetime.now().astimezone()

    version = opts.app_version
    if version is None:
        try:
            from backend.api.services.app_version import get_version_info
            version = get_version_info()
        except Exception:
            version = {"app": "Orienta", "version": "unknown"}

    source: Dict[str, Any] = {"path": geometry.source_path}
    if geometry.source_path:
        p = Path(geometry.source_path)
        try:
            source["bytes"] = int(p.stat().st_size)
        except OSError:
            source["bytes"] = None
        if opts.hash_source and source.get("bytes") is not None:
            source["sha256"] = _sha256_of(p)
            if source["sha256"] is None:
                source["sha256_note"] = "the file could not be read for hashing"
        else:
            source["sha256"] = None
            source["sha256_note"] = (
                "not computed: hashing was disabled for this export"
                if not opts.hash_source else
                "not computed: the source file could not be stat-ed")
    else:
        source["bytes"] = None
        source["sha256"] = None
        source["sha256_note"] = "no source path was supplied"

    # Excluded from SCORING. Named explicitly rather than described, because
    # "light elements" is not a set anybody can check against — and named
    # accurately, because the previous wording said they were excluded from
    # the composition renormalisation and they are not.
    ignored = sorted(_chem_ignore())

    try:
        from backend.api.services.cif_phase_library import TIE_TOLERANCE
        tie = float(TIE_TOLERANCE)
    except Exception:
        tie = None

    # The value the replay actually scored with, not a second inference that
    # could quietly differ from it — a wrong matrix element inverts the whole
    # ratio metric, so this cell has to be the one that was used.
    matrix_resolved = ctx.get("matrix_element")

    requested = dict(opts.requested or {})
    # What the MAP says beats what the export dialog was told. The dialog's
    # echo describes the last thing typed into a form; the map's record is
    # what actually ran, and it survives a restart. A key PRESENT in the
    # record wins even when its value is None — "no cluster count was pinned"
    # is a statement, and letting a stale echo fill that hole would print a
    # number nobody chose.
    stored = _recorded_settings(state)

    def _prefer(key: str) -> Any:
        return stored[key] if key in stored else requested.get(key)

    acquisition = _acquisition_block(geometry, opts, warnings)
    interaction = _interaction_volume_block(
        acquisition, matrix_resolved, geometry, _particle_rows, area_ok)
    library_identity, phases_absent = library_report(
        entries, measured_el, library=opts.library,
        library_path=opts.library_path,
        requested_keys=stored.get("phase_keys"))
    if phases_absent:
        n_unmapped = sum(1 for d in phases_absent
                         if d["reason"] == "element_not_measured")
        if n_unmapped:
            _warn(warnings, "phases_dropped_unmeasured_element",
                  f"{n_unmapped} of {library_identity.get('n_entries')} "
                  f"library phases could not compete because they contain an "
                  f"element this scan never mapped. They were removed "
                  f"silently before scoring; provenance.phases_absent names "
                  f"each one and the element it needs.")

    prov: Dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "app": version,
        "exported_at": now.isoformat(),
        # Stated IN the file, because the alternative is a reader finding out
        # by crashing. A tester's `json.load(open('provenance.json'))` died
        # with UnicodeDecodeError on Windows, where open() defaults to
        # cp1252 and this file was written as UTF-8 carrying subscripted
        # formulas. It is now pure ASCII and that call works everywhere.
        "encoding": {
            "provenance_json": "ascii",
            "note": (
                "provenance.json is written with ensure_ascii=True, so the "
                "bytes are pure ASCII and json.load(open(path)) works with "
                "no encoding argument on any platform, including a Windows "
                "Python defaulting to cp1252. Non-ASCII characters -- the "
                "subscripted formulas from the CIF library -- are stored as "
                "\\uXXXX escapes and any JSON parser decodes them back to "
                "the identical string, so nothing is lost or transliterated. "
                "Every phase also carries formula_ascii beside formula for "
                "readers that would rather not see subscripts at all. The "
                "CSV tables are a different convention: UTF-8 with a BOM, "
                "which is what makes Excel open them correctly by "
                "double-click."),
        },
        "source": source,
        "crop_window": geometry.crop_window,
        "grid": {
            "n_rows": geometry.n_rows,
            "n_cols": geometry.n_cols,
            "n_px": n_px,
        },
        "step": {
            # requested-and-resolved even here: "the file was asked for a step
            # size and answered None" is the fact that removed every
            # micrometre column, and it has to be readable as such.
            "requested": requested.get("step"),
            "x_um": geometry.step_x_um,
            "y_um": geometry.step_y_um,
            "units": geometry.step_units,
            "area_available": bool(area_ok),
            "note": (None if area_ok else
                     "This file carries no usable step size for the EDS or "
                     "EBSD grid, so every column in micrometres was omitted. "
                     "A guessed scale would look like an answer."),
        },
        "elements": {
            "measured": list(measured_el),
            "columns": list(columns_el),
            "not_measured": [el for el in columns_el if el not in set(measured_el)],
            "excluded_from_scoring": ignored,
            # CORRECTED 2026-08-27. The previous wording said these elements
            # were excluded from the composition renormalisation. They are
            # not, and the numbers in this very file prove it: measured on
            # the real SampleB export, the per-pixel at% sum INCLUDING C and
            # O is 100.000000 everywhere, and EXCLUDING them it runs
            # 91.297860 to 100.000000 (mean 98.44). A reader who trusted the
            # old sentence would have read every at% cell against the wrong
            # denominator.
            "excluded_note": (
                "Carbon and oxygen ARE included in the reported at% columns: "
                "the at% maps are renormalised over every measured element, "
                "so mean_at_pct_* across all measured elements sums to 100 "
                "including C and O. What C and O are excluded from is "
                "SCORING — chemistry_score._renormalise drops them and "
                "renormalises again over the remainder before any phase is "
                "matched or any rule is evaluated. The two statements are "
                "different and were previously conflated. scored_at_pct_sum "
                "in each table is the at% that survives that drop, so the "
                "size of the discarded fraction is visible per object."),
            "excluded_from_renormalisation": [],
            "excluded_from_renormalisation_note": (
                "Empty by measurement, not by omission: nothing is excluded "
                "from the at% renormalisation. See excluded_note."),
            # The divisor of every enrichment_* cell, once, as at%.
            "background_at_pct": {el: float(v) * 100.0
                                  for el, v in sorted((background or {}).items())},
            "background_note": (
                "chemistry_score.background_levels: the MEDIAN share of each "
                "element over the pixels that carry chemistry, as an at% of "
                "the scored elements. It is what a phase has to beat to be "
                "called present, and it is the divisor of every enrichment_* "
                "column. The matrix of a real map is not clean — this is the "
                "number that says so."),
        },
        "counts": {
            "n_px_total": n_px,
            "n_px_valid_eds": n_valid,
            "n_px_classified": n_classified,
            "n_px_hand_edited": int(locked_flat.sum()),
            "n_phases_with_pixels": _n_phases_with_pixels(state),
            "n_regions": n_regions,
            "n_particles": n_particles,
            "frac_classified_of_scan": (n_classified / n_px) if n_px else None,
            # Same intersection rule as the table columns. A classified pixel
            # need NOT be a valid-EDS pixel — a hand-painted one, for instance —
            # so dividing the raw classified count by the valid count mixes two
            # different sets and can exceed 1. Counted here the way it is
            # counted in every row of every table, so the summary scalar and
            # the tables cannot disagree.
            "n_px_classified_and_valid_eds": n_classified_valid,
            "frac_classified_of_valid_eds": ((n_classified_valid / n_valid)
                                             if n_valid else None),
            # The two particle exclusion counts as NUMBERS, not only inside the
            # English warning string. A caption builder that wanted "81 below
            # the limit, 38 touching the edge" had to parse prose, which is
            # both fragile and untranslatable. Neither count removes a row —
            # both are flags — so these are what a reader needs to state the
            # exclusions honestly ("N found; M below limit; K edge; R used").
            "n_particles_below_size_limit": int(sum(
                1 for r in _particle_rows if r.get("below_size_limit"))),
            "n_particles_touching_edge": int(sum(
                1 for r in _particle_rows if r.get("touches_edge"))),
        },
        "denominators": {
            "frac_of_scan": "n_px / every pixel of the exported raster",
            "frac_of_valid_eds": (
                "n_px_valid_eds / pixels carrying a usable EDS measurement "
                "(chemistry_score.has_chemistry). NUMERATOR AND DENOMINATOR "
                "ARE THE SAME SET: the numerator counts this row's pixels "
                "that themselves carry a usable EDS measurement, not the "
                "row's total n_px. A row can own pixels with no EDS at all, "
                "and counting those would inflate the fraction and stop the "
                "column summing to 1 across the phases."),
            "frac_of_classified": "n_px / pixels that received a phase",
            "note": ("All three are exported because all three are "
                     "defensible and they differ by tens of percent. Every "
                     "fraction has its own count beside it — n_px, or "
                     "n_px_valid_eds for frac_of_valid_eds — so a binomial "
                     "interval is recoverable."),
        },
        "classification": {
            "tolerance": float(getattr(state, "tolerance", float("nan"))),
            # Verified 2026-08-27: `tolerance` is accepted by the request
            # model, stored in the sidecar and drawn as a slider, and
            # `auto_classify_pixels` never reads it. It was superseded by the
            # vetoed L1 scorer and left in place for API compatibility. A
            # preset that carried it would let a user tune it, see nothing
            # change, and conclude the preset was broken.
            "tolerance_effective": False,
            "tolerance_note": (
                "The tolerance slider is inert in this build: no code path "
                "reads it. It is recorded for completeness only."),
            "min_score": float(getattr(state, "min_score", float("nan"))),
            "ambiguity_threshold_at_pct": tie,
            "ambiguity_threshold_note": (
                "cif_phase_library.TIE_TOLERANCE — two candidates within this "
                "score are treated as genuinely undecidable by chemistry."),
            "mode": _prefer("mode"),
            "n_clusters_requested": _prefer("n_clusters"),
            "n_clusters_resolved": n_regions,
            # What the run itself settled on. It can differ from
            # n_clusters_resolved above, which counts the regions in the map
            # AS EXPORTED: a merge removes one. Both are true, about
            # different moments.
            "n_clusters_at_classification": stored.get("k_used"),
            "min_score_requested": _prefer("min_score"),
            "cluster_remainder": stored.get("cluster_remainder"),
            "phase_keys": stored.get("phase_keys"),
            "rules": stored.get("rules"),
            # AutoClassifyRequest carries no matrix element, so this one is
            # the caller's echo in every case.
            "matrix_element_requested": requested.get("matrix_element"),
            "matrix_element_resolved": matrix_resolved,
            "element_weights": dict(getattr(state, "element_weights", None) or {}),
            "element_weights_note": (
                "A weight scales one column of the clustering feature matrix. "
                "It does NOT change any reported composition, and it does NOT "
                "enter the candidate ranking — the ranking is the unweighted "
                "mean absolute deviation, which is a documented and "
                "deliberate difference between the two halves of the page."),
            "smoothing": {
                "scale_px": smoothing.get("scale_px"),
                # The physical width. "5" is not a length and a reader
                # cannot judge it; "5 px = 3.3 um box" is the sentence that
                # decides whether the averaging box was larger than the
                # features being counted. None when this file carries no
                # step size — never a guessed one.
                "box_um": smoothing.get("box_um"),
                "source": smoothing.get("source"),
                "report": smoothing.get("report"),
                "note": smoothing.get("note"),
            },
            "settings_source": ("map" if stored else
                                ("caller" if requested else None)),
            "settings_note": (_SETTINGS_RECORDED_NOTE if stored
                              else _SETTINGS_NOT_PERSISTED_NOTE),
        },
        "determinism": {
            "deterministic": True,
            "random_state": 0,
            "n_init": 10,
            "note": ("Every KMeans site and the GMM in eds_clustering.py fix "
                     "random_state=0 with n_init=10, so re-running the same "
                     "settings on the same data reproduces the same grids. "
                     "Verified 2026-08-27 at eds_clustering.py:150, 252, 323, "
                     "598."),
        },
        "particles": {
            "connectivity": int(opts.connectivity),
            "basis": particle_basis,
            "basis_note": (
                "Connected components of the REGION grid."
                if particle_basis == "region" else
                "This map has no regions, so components were taken from the "
                "PHASE grid. Two regions carrying the same phase are one "
                "particle here."),
            "assignment_source_note": (
                "CORRECTED 2026-08-27, and the correction is why "
                "format_version moved to 3. hand/mixed are graded on THIS "
                "particle's own n_px_hand against its own n_px and on "
                "nothing else: a particle with no painted pixels of its own "
                "can never read 'mixed'. Until this version the region's "
                "grade was copied onto every particle in it, and a QA tester "
                "measured 110 of 167 rows reading 'mixed' while carrying "
                "n_px_hand = 0, with exactly one particle in the file "
                "actually painted. definition/auto/untracked ARE inherited "
                "from the region, because under region basis the phase name "
                "genuinely was decided per region and never per particle — "
                "and they are recomputed with n_hand = 0 so the region's "
                "painting cannot leak back in. Read n_px_hand beside this "
                "column; it is the count the grade is made of."),
            "chemistry_dissents_note": (
                "chemistry_dissents = True means THIS particle's own "
                "composition would name a different phase from the one it "
                "carries. It is a chemistry observation and NOT a provenance "
                "claim — nobody touched the row. It is the same fact as "
                "phase_agrees = False, spelled so it cannot be read as 'a "
                "human changed this'; phase_agrees is shared with the "
                "regions table, where a disagreement DOES carry provenance "
                "weight (see hand_edits.assignment_source_values.untracked). "
                "Expect it often on a clean map: a region is named from its "
                "eroded interior mean and each connected component of it is "
                "then measured separately, so components on the region's "
                "chemical edge dissent as a matter of course. Measured on "
                "the real SampleB scan (90x120, cluster mode, 1.5 um "
                "smoothing, nothing painted): 94 of 135 particles dissent "
                "while every one of them is provenance-clean."),
            "min_particle_px": int(opts.min_particle_px),
            # The value AND the default, so "was this run configured, or did
            # it inherit?" is answerable from the file. A tester reported the
            # scripted and the clicked export disagreeing here — the API
            # defaulting to 0 while the dialog sent 4 — which would make two
            # exports of one map carry different `below_size_limit`
            # populations with nothing in either file to say so. MEASURED on
            # this repo at 2026-08-27 and the report does not reproduce:
            # ExportOptions, ExportRequest (routes/eds_export.py) and the
            # dialog (ExportDialog.jsx, `useState(0)`) are all 0; the 4 the
            # tester saw is a fixture value in ExportDialog.test.jsx. That
            # they agree TODAY is not a reason to leave it invisible — it is
            # exactly the kind of agreement that drifts silently — so the
            # default travels with the value and any future divergence shows
            # up as `used_default: false` on the run that did not set it.
            "min_particle_px_default": int(
                ExportOptions.__dataclass_fields__["min_particle_px"].default),
            "min_particle_px_used_default": (
                int(opts.min_particle_px)
                == int(ExportOptions.__dataclass_fields__[
                    "min_particle_px"].default)),
            "min_particle_px_note": (
                "A flag, never a filter: particles below this size carry "
                "below_size_limit = True and are still counted and still "
                "present as rows. min_particle_px_default is this module's "
                "own default; when min_particle_px differs from it, a caller "
                "chose the value, and two exports of the same map that "
                "disagree on it will disagree on which rows are flagged."),
            "size_note": (
                "sqrt_area_um is sqrt(area_um2) — Murakami's inclusion-"
                "rating parameter, the length a failure analyst rates an "
                "inclusion by. It is NOT interchangeable with ecd_um: for a "
                "disc the two differ by sqrt(pi)/2 = 0.886, and further for "
                "anything elongated."),
            "feret_angle_note": (
                "feret_angle_deg is the orientation of feret_max_um, in "
                "DEGREES FROM THE +X AXIS (increasing column), measured "
                "COUNTER-CLOCKWISE as seen on the map, folded into [0, 180) "
                "because a diameter has no head or tail. The raster's row "
                "index increases downward and the row component is negated "
                "accordingly — without that every angle would be mirrored. "
                "Measured from the same convex hull feret_max_um is measured "
                "from, scaled the same way, so the length and the direction "
                "describe one line — and feret_max on a rectangle is its "
                "DIAGONAL, so a 1x7 bar reads 171.87 deg rather than 180. "
                "Check aspect_ratio: the two converge for an elongated "
                "object and do not for a stubby one. Empty when the hull was "
                "unavailable; an angle guessed from a bounding box would be "
                "a different quantity."),
            "perimeter_note": (
                "perimeter_crofton_um (renamed from perimeter_um on "
                "2026-08-27, so the correction survives somebody opening the "
                "CSV without this file) is a 4-direction Crofton estimate, "
                "not a "
                "boundary-pixel count. A staircase count overestimates the "
                "perimeter by 4/pi = 1.273 at every size, which reads as a "
                "38 % circularity UNDERESTIMATE. Crofton is unbiased for a "
                "convex set of random orientation and is meaningless below a "
                "few tens of pixels — check n_px before using circularity."),
            "core_note": (
                "core_* columns are measured after a one-pixel 4-connected "
                "erosion. The EDS interaction volume exceeds the step, so rim "
                "pixels of a small particle are particle-plus-matrix mixtures; "
                "using all pixels drags small particles toward the matrix and "
                "manufactures a spurious size-composition trend."),
            "edge_note": (
                "touches_edge = True means the particle is cut off by the "
                "raster border and its size is a LOWER BOUND. Every size "
                "distribution is biased if these are not handled."),
        },
        "composition": {
            "basis": "raw",
            "basis_note": (
                "Regions are FORMED on smoothed composition and REPORTED on "
                "raw. Measured on the SampleB silicon particle: 40.97 at% raw "
                "against 36.88 smoothed, so a threshold read off a raw value "
                "and typed into a window evaluated on smoothed values catches "
                "nothing."),
            "spread": "sd_within",
            "spread_note": (
                "sd_within is the POPULATION standard deviation (ddof=0) of "
                "the pixels in the object — spatial heterogeneity plus "
                "counting noise, not the precision of the mean. No standard "
                "error is exported: the pixels are spatially correlated "
                "(interaction volume > step, plus box smoothing), so "
                "sd/sqrt(n) would overstate precision by one to two orders of "
                "magnitude. Use sd_within and n_px and decide for yourself."),
            "not_measured": NOT_MEASURED,
            "not_measured_note": (
                f"A cell reading '{NOT_MEASURED}' means this scan never "
                f"mapped that element. It is deliberately not 0: '0.0 at%' "
                f"would assert the element was looked for and found absent."),
            "scored_at_pct_sum_note": (
                "The object's at% summed over the elements the scorer keeps "
                "— every measured element except those in "
                "elements.excluded_from_scoring. A data-quality measure: the "
                "at% columns sum to 100 including C and O, so 100 minus this "
                "is how much of the signal was carbon and oxygen. Measured "
                "per pixel on the real SampleB scan it runs 91.30 to "
                "100.00."),
            "raw_sum_note": (
                "A pre-renormalisation total (raw counts, or wt% before "
                "conversion) is NOT RECOVERABLE at this point and is "
                "therefore absent rather than silently approximated. "
                "build_tables receives at% maps only — the counts-to-wt%-to-"
                "at% conversion happens upstream in eds_utils and "
                "renormalises to 100 on the way — so no function in this "
                "module has ever seen the un-normalised total. "
                "scored_at_pct_sum is the recoverable neighbour and is "
                "documented above."),
            "enrichment_note": (
                "enrichment_El is the object's share of the scored elements "
                "divided by the map's own background share of that element, "
                "which is the quantity the classifier's own enrichment gate "
                "is defined on (eds_wand.selection_stats, unrounded). "
                "background_at_pct_El sits in the next column on purpose: it "
                "is that background as an at%, and it is the divisor. This "
                "project has shipped an 'Al 49.68x enriched' readout on a "
                "map whose background IS aluminium, and seeing the divisor "
                "is the only defence a reader has against a repeat. The "
                "background is a MAP-LEVEL constant, so it repeats down the "
                "column; it is also in elements.background_at_pct."),
        },
        "ranking": {
            "deciding": {
                "columns": ["phase_auto", "phase_auto_index", "phase_agrees",
                            "score_auto", "runner_up_phase",
                            "runner_up_score", "margin_score"],
                "metric": ("chemistry_score.score_phase_ratio, 0..1, higher "
                           "is better"),
                "margin": "winner's score minus runner-up's score",
                "population": (
                    "the ERODED INTERIOR of the object (3x3 erosion, falling "
                    "back to the whole object when it erodes away), which is "
                    "the population eds_clustering._match_labels named each "
                    "region from"),
                "composition_basis": (
                    "RAW at%, as measured. The classification SMOOTHS to "
                    "decide which pixels group together, then names the "
                    "group on raw composition — see classification.smoothing "
                    "for the grouping width."),
                "matrix_element": ctx.get("matrix_element"),
                "min_score": ctx.get("min_score"),
                "min_score_note": (
                    "phase_auto_index is -1 where score_auto falls below "
                    "this floor, because that is exactly what the classifier "
                    "does with such an object: leave it unassigned. "
                    "score_auto is still reported there — 'the best chemistry "
                    "could manage was below the floor' is the fact, and "
                    "blanking it would hide it."),
                "rules_applied": bool(
                    ctx.get("rule_set") is not None
                    and not getattr(ctx.get("rule_set"), "is_empty", True)),
                "rules_note": (
                    "User phase rules are a GATE, not a re-score: a blocked "
                    "phase scores 0 and so loses even to a badly-scoring "
                    "phase that is allowed. Replayed here from the rules "
                    "recorded WITH the map."),
                "note": (
                    "THIS is the scorer that produced the map, replayed. "
                    "Verified 2026-08-27 on the real 90x120 SampleB map (7 "
                    "regions, 22 candidates): it reproduces region_phase for "
                    "7 of 7 regions. phase_agrees therefore means 'the map "
                    "differs from what the chemistry alone would choose'. "
                    "Until 2026-08-27 these columns were filled by the "
                    "mean-absolute-deviation ranking below, which is a "
                    "different metric on a different basis; on that same "
                    "scan it made 4 of 7 regions read phase_agrees = False "
                    "with not one hand-painted pixel on the map."),
            },
            "second_opinion": {
                "columns": ["phase_second_opinion_mad", "gap_mad_at_pct",
                            "runner_up_mad", "margin_mad_at_pct"],
                "metric": ("mean absolute deviation over the union of "
                           "element keys, at%, lower is better"),
                "margin": "runner-up's deviation minus the winner's, at%",
                "population": "every pixel of the object",
                "composition_basis": (
                    "RAW at%, RENORMALISED to sum 100 over the scored "
                    "elements before the subtraction. Both sides of the "
                    "difference are then on the same denominator; the "
                    "nominal composition of a CIF contains no C or O and "
                    "sums to 100 by construction, while a measured at% "
                    "includes C and O and sums to 91.3-100 without them. "
                    "Measured on the real SampleB scan, correcting this "
                    "changed no winner but moved a margin by up to +112.6 % "
                    "(region 4: 0.194720 -> 0.413980 at%) and changed one "
                    "runner-up's identity (region 6)."),
                "differs_from_screen": True,
                "differs_from_screen_note": (
                    "The region inspector on screen shows this metric WITHOUT "
                    "the renormalisation, and its means are rounded to 2 dp "
                    "with elements below 0.05 at% dropped. So the file and "
                    "the panel can differ in the third digit and, rarely, in "
                    "the runner-up. The file is the corrected one. "
                    "eds_export.rank_candidates remains the inspector's exact "
                    "definition for anyone who needs to reproduce the panel."),
                "note": (
                    "Exported because a distance in at% is easier to judge "
                    "than a 0..1 score, and because it is what the inspector "
                    "shows. It does NOT decide anything: phase_agrees is "
                    "computed against the deciding metric above. It is also "
                    "UNWEIGHTED even when element_weights are set — see "
                    "classification.element_weights_note."),
            },
        },
        # How the numbers above were CHECKED, next to the record of how they
        # were made. The group leader's line was that the record had become
        # excellent and the result was still not checked; this block is the
        # second half.
        "plausibility": plausibility if plausibility is not None else {
            "method": "not run",
        },
        "acquisition": acquisition,
        "interaction_volume": interaction,
        "coordinates": {
            "origin": "0-based: the first pixel is row 0, col 0.",
            # The claim used to read "every flat index in this export is
            # row * n_cols + col" — and no table carries a flat index, so it
            # documented an absent thing. It says where it applies now: the
            # rasters, which are the artefacts somebody's script actually
            # ravels.
            "order": ("row-major, and that matters for the RASTERS rather "
                      "than for the tables: labels.npz holds "
                      "(n_rows, n_cols) arrays, so a flat index into one of "
                      "them -- what numpy's ravel(), flatnonzero() and "
                      "argmax() return -- is row * n_cols + col. NO TABLE IN "
                      "THIS EXPORT CARRIES A FLAT INDEX. Rows are located by "
                      "centroid_row / centroid_col and by the bbox_* "
                      "columns, all of them 0-based on the same grid, and "
                      "pixels.csv carries row and col as separate columns."),
            "x_um": "col * step_x_um -- the LEFT edge of the pixel, not its centre.",
            "y_um": "row * step_y_um.",
            "y_direction": (
                "y increases DOWNWARD, as the raster is stored and as the "
                "map is drawn. It is not a right-handed map coordinate; a "
                "reader plotting these with y up gets a vertically mirrored "
                "figure, and feret_angle_deg would then be mirrored with it "
                "(that column already negates the row component to be "
                "counter-clockwise ON THE MAP -- see feret_angle_note)."),
            "centroid": ("centroid_row / centroid_col are in PIXELS on the "
                         "same 0-based grid; centroid_x_um / centroid_y_um "
                         "are the same point scaled by the steps."),
            "bbox": ("bbox_row_min..bbox_row_max and bbox_col_min.."
                     "bbox_col_max are INCLUSIVE on both ends, so the height "
                     "is bbox_row_max - bbox_row_min + 1."),
            "note": ("Stated here because pixels.csv and the labels arrays "
                     "are the artefacts somebody else's script reads, and a "
                     "convention that lives only in a manual is a "
                     "convention the script will get wrong."),
        },
        # The key to the pictures. The per-row colours are COLUMNS -- beside
        # the row they describe, where a reader matching a colour to a name
        # already is -- and this block carries what no row can: the grey the
        # unassigned pixels are painted, and where the palette came from.
        "colours": colours if colours is not None else {
            "available": False,
            "reason": "this export did not resolve the palette",
        },
        "peak_overlaps": _peak_overlap_block(measured_el),
        "library": library_identity,
        "phases_absent": phases_absent,
        "phases_participating": [
            {
                "phase_index": i,
                "key": str(getattr(e, "key", "")),
                "cif_filename": str(getattr(e, "cif_filename", "")),
                "phase_label": phase_label(e),
                "formula": str(getattr(e, "formula", "")),
                "formula_ascii": formula_ascii(getattr(e, "formula", "")),
                "space_group": str(getattr(e, "space_group", "")),
                "space_group_number": getattr(e, "space_group_number", None),
                "crystal_system": str(getattr(e, "crystal_system", "")),
                "composition_at_pct": {k: float(v) for k, v in
                                       (getattr(e, "composition", None) or {}).items()},
            }
            for i, e in enumerate(entries)
        ],
        "region_definitions": list(getattr(state, "region_defs", None) or []),
        "region_definitions_note": (
            "Verbatim, as evaluated. definitions.csv re-evaluates them on the "
            "RAW composition to attribute regions; the classification "
            "evaluated them on the SMOOTHED composition, so a "
            "definition_match_frac below 1 is expected near boundaries."),
        "hand_edits": {
            "pixels_painted": int(locked_flat.sum()),
            "frac_of_scan": (float(locked_flat.sum()) / n_px) if n_px else None,
            # Read off the store's own edit record rather than hardcoded.
            # `edit_summary()` returns all-None counts for a map restored
            # from a sidecar written before edits were tracked, and real
            # integers (including the zeros) otherwise — so "0 merges" is a
            # measurement where it can be one and a null where it cannot.
            # Getattr, because build_tables is a pure function and its
            # callers may hand it a hand-built state with no record at all.
            **_edit_record(state),
            "assignment_source_values": {
                "hand": "every pixel of the row was painted by hand.",
                "mixed": (
                    "SOME pixels were painted. Until 2026-08-27 this case "
                    "read 'hand', so 100 painted pixels claimed a "
                    "3327-pixel region. n_px_hand beside it is the count."),
                "definition": (
                    "a composition window claims more than half of the row's "
                    "pixels; definition_name and definition_match_frac say "
                    "which and how strongly."),
                "auto": (
                    "nothing was painted AND the map matches the replayed "
                    "chemistry (phase_agrees = True). Only then can the row "
                    "positively assert that nothing touched it."),
                "untracked": (
                    "nothing was painted and the map does NOT match the "
                    "replayed chemistry — or the replay produced no answer. "
                    "Something moved it that leaves no trace in the stored "
                    "map (see note above). This value used to read 'auto', "
                    "which asserted a provenance the file cannot know."),
            },
        },
        "preset": {
            "name": opts.preset_name or None,
            "compatibility": opts.compatibility,
            "override": bool(
                opts.compatibility
                and not (opts.compatibility or {}).get("ok", True)),
            "note": ("A compatibility report present here with ok = false "
                     "means the user was refused and overrode the refusal. "
                     "That is their right; this is the record of it."),
        },
        "locale": {"decimal": opts.decimal, "delimiter": opts.delimiter},
        "rounding": ("None. Every numeric cell carries full float precision; "
                     "rounding is presentation and belongs in whatever reads "
                     "this file."),
        "caveat": SEMI_QUANTITATIVE_CAVEAT,
        "not_built": {
            "batch_over_folder": (
                "Deferred. The export core is a pure function of (state, "
                "at_maps, geometry, options), so a batch driver can call it "
                "without changing it."),
            "eds_ebsd_confusion_matrix": (
                "Deferred: it needs an independence flag the indexing export "
                "does not record, and the EDS chemistry prior can drive the "
                "indexing it would be checked against, so the number would "
                "be circular."),
            "sensitivity_reruns": "Not built.",
            "effective_n_standard_errors": "Not built — see composition.spread_note.",
            "k_factor_calibration": (
                "Not built. There is no standard, no measured k-factor and "
                "no ZAF correction; acquisition.beam_voltage_kv is now in "
                "this file so the caveat can at least be weighed."),
            "peak_overlap_warnings": (
                "Not DETECTED. This module receives per-element at% maps and "
                "never the spectra, so it cannot measure an overlap. The "
                "overlaps known for this alloy system are declared in "
                "peak_overlaps instead of being silently omitted."),
            "spectra": (
                "Per-pixel spectra are not exported. The acquisition block "
                "carries the spectrometer settings they were taken with "
                "(channel width, channel count, energy range, process "
                "time), so a reader can tell what a re-extraction from the "
                "source file would give."),
        },
        "warnings": list(warnings),
    }
    prov["summary"] = build_summary_line(prov, phases_table)
    prov["summary_note"] = (
        "A pasteable one-liner assembled from the values above, nothing "
        "new. It is also written to caption.txt beside the tables, because "
        "the dialog that offers one on screen is gone four weeks later and "
        "the folder is not. A clause whose value is unavailable is OMITTED "
        "rather than printed with a placeholder -- the frontend's version "
        "of this line already follows that rule after a Number(null) === 0 "
        "bug printed a real-looking 0.")
    return prov


def build_summary_line(
    provenance: Mapping[str, Any], phases_table: Optional[Table] = None,
) -> str:
    """One sentence a reader can paste under a figure. Values only, no prose.

    TWO TESTERS ASKED FOR IT, and for the same reason: the export dialog
    builds a summary on screen and it never reaches the folder, so the one
    artefact that survives to the writing-up stage is the one without a
    caption. Everything here is already in ``provenance``; this only puts it
    in an order a person reads.

    EVERY CLAUSE IS GUARDED ON ``is None`` AND NOT ON TRUTHINESS. A step of
    0.0 um, a count of 0 regions and a missing value are three different
    things, and the sibling implementation in the frontend shipped a bug of
    exactly this shape (``Number(null) === 0`` printing a confident zero).
    A clause whose value is unavailable is left out; nothing is defaulted.
    """
    parts: List[str] = []

    src = (provenance.get("source") or {}).get("path")
    if src:
        parts.append(Path(str(src)).name.split(".")[0])

    grid = provenance.get("grid") or {}
    step = provenance.get("step") or {}
    rows, cols = grid.get("n_rows"), grid.get("n_cols")
    if rows is not None and cols is not None:
        clause = f"{int(rows)}x{int(cols)} px"
        sx, sy = step.get("x_um"), step.get("y_um")
        if sx is not None and sy is not None:
            clause += (f" at {float(sx):g} um step" if float(sx) == float(sy)
                       else f" at {float(sx):g}x{float(sy):g} um step")
        parts.append(clause)

    cls = provenance.get("classification") or {}
    mode = cls.get("mode")
    if mode:
        clause = f"EDS {mode} classification"
        box = ((cls.get("smoothing") or {}).get("box_um") or {})
        bx = box.get("x") if isinstance(box, Mapping) else None
        if bx is not None:
            clause += f", {float(bx):g} um smoothing box"
        parts.append(clause)

    counts = provenance.get("counts") or {}
    n_reg, n_ph = counts.get("n_regions"), counts.get("n_phases_with_pixels")
    if n_reg is not None and n_ph is not None:
        parts.append(f"{int(n_reg)} regions -> {int(n_ph)} phases")
    n_part = counts.get("n_particles")
    if n_part is not None:
        clause = f"{int(n_part)} particles"
        edge = counts.get("n_particles_touching_edge")
        if edge:
            clause += f" ({int(edge)} touching the raster edge)"
        parts.append(clause)

    frac = counts.get("frac_classified_of_scan")
    if frac is not None:
        parts.append(f"{float(frac) * 100.0:.1f} % of the scan classified")

    if phases_table is not None:
        ranked = sorted(
            (r for r in phases_table.rows
             if int(r.get("phase_index", -1)) >= 0
             and r.get("frac_of_scan") is not None),
            key=lambda r: -float(r["frac_of_scan"]))
        if ranked:
            top = ranked[0]
            name = top.get("phase_label") or top.get("phase_name") or ""
            if name:
                parts.append(f"largest phase {name} at "
                             f"{float(top['frac_of_scan']) * 100.0:.1f} % of the scan")

    kv = (provenance.get("acquisition") or {}).get("beam_voltage_kv")
    if kv is not None:
        parts.append(f"{float(kv):g} kV")
    rng = (provenance.get("interaction_volume") or {}).get("range_um")
    if rng is not None:
        clause = f"interaction range ~{float(rng):.2f} um"
        small = (provenance.get("interaction_volume") or {}).get(
            "frac_particles_smaller_than_range")
        if small is not None:
            clause += f" ({float(small) * 100.0:.0f} % of particles smaller)"
        parts.append(clause)

    painted = (provenance.get("hand_edits") or {}).get("pixels_painted")
    if painted is not None:
        parts.append("no hand-painted pixels" if int(painted) == 0
                     else f"{int(painted)} hand-painted pixels")

    lib = provenance.get("library") or {}
    n_lib, n_part_lib = lib.get("n_entries"), lib.get("n_participating")
    if n_lib is not None and n_part_lib is not None:
        clause = f"library {int(n_part_lib)}/{int(n_lib)} phases competing"
        digest = lib.get("sha256")
        if digest:
            clause += f" (sha256 {str(digest)[:12]})"
        parts.append(clause)

    parts.append("semi-quantitative EDS, no standard and no ZAF correction")

    app = provenance.get("app") or {}
    name = app.get("app") or "Orienta"
    ver = app.get("version")
    when = provenance.get("exported_at")
    tail = f"{name} {ver}" if ver else str(name)
    if when:
        tail += f", exported {str(when)[:19]}"
    parts.append(tail)

    return "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# plausibility: a report ON the result, never an input TO it
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS. A group leader used a finished export to sanity-check the
# science and found the headline number wrong: 61 % of an aluminium extrusion
# classified as Fe-intermetallic, by phases whose assigned pixels carry a
# THIRD of the iron those phases are made of. Both numbers were already in
# phases.csv, in adjacent columns — ``mean_at_pct_Fe`` and
# ``nominal_at_pct_Fe`` — and the file said nothing. Her sentence:
#
#     "You already print both numbers next to each other. You are one
#      subtraction away from saying so."
#
# Sharper than it reads: docs/user-guide/EDS.md describes THIS failure, with
# this phase and this number, as one that was fixed ("a phase requiring
# 11.6 at% Fe was assigned to pixels measuring 3 at% Fe, on 20.95 % of a real
# scan"). It came back on 61 % of a map and the record was silent.
#
# WHAT IT IS NOT. It does not change a single assignment. It does not rank,
# gate, veto or re-score anything; ``phase_final`` and ``phase_grid`` are
# identical with it and without it. It reads the two numbers the export
# already prints and reports the arithmetic between them. The classification
# has one scorer (``deciding_rank``); a second one wearing the word "check"
# would be exactly the two-metrics-in-one-file defect this module was
# repaired for on 2026-08-27.
#
# WHY A RELATIVE CRITERION. The elements that decide a phase identity are the
# small ones. On the scan below, Fe runs 0.16 to 4.27 at% while Al runs 80 to
# 97; an absolute at% threshold that is meaningful for Fe would fire on every
# row for Al, and one that is quiet for Al could never fire for Fe at all.
# A factor is the only criterion that means the same thing at both ends.

#: An element DISCRIMINATES a phase from the matrix only when the phase needs
#: substantially more of it than the map already has everywhere. Below this
#: factor it cannot separate them: a pixel of pure background already reads
#: more than half of what the phase requires, so ordinary mixing covers the
#: difference and a deviation on that element says nothing about the call.
#:
#: Measured on the real SampleB scan (90x120 = 10 800 px, mode=cluster,
#: scale_um=1.5, k=7, 22 candidates) — nominal at% over the map background
#: from :func:`_background_levels`, per element of each assigned phase:
#:
#:     Al in Al.cif          100.00 / 85.66 =  1.17   not discriminating
#:     Al in Al6Fe            85.71 / 85.66 =  1.00   not discriminating
#:     Al in sd_0302719       75.77 / 85.66 =  0.88   not discriminating
#:     Si in sd_0302719        9.94 /  7.34 =  1.35   not discriminating
#:     Mn in sd_0302719        2.69 /  1.26 =  2.14   DISCRIMINATING
#:     Fe in sd_0302719       11.60 /  2.86 =  4.05   DISCRIMINATING
#:     Fe in Al6Fe            14.29 /  2.86 =  4.99   DISCRIMINATING
#:     Fe in Fe3_Al2_Si3      37.50 /  2.86 = 13.10   DISCRIMINATING
#:     Si in Si.cif          100.00 /  7.34 = 13.62   DISCRIMINATING
#:
#: That is the group leader's own line, drawn by the data rather than by a
#: rule: "Al being off by 5 at% means little; Fe being off by 3x means
#: everything". Every Al-bearing phase on an Al-matrix scan falls out at
#: 0.88-1.17 without anybody naming aluminium anywhere.
_DISCRIMINATING_FACTOR = 2.0

#: And it has to be present enough to argue about. An element a phase needs
#: at a few tenths of a percent is below what this semi-quantitative pipeline
#: can resolve (see SEMI_QUANTITATIVE_CAVEAT), so a factor on it is noise
#: wearing a decimal point. The smallest discriminating nominal on the scan
#: above is Mn at 2.69 at%, comfortably clear of this.
_MIN_NOMINAL_AT_PCT = 1.0

#: Twice, or half. The comparison is symmetric — a phase assigned to pixels
#: carrying far MORE of its discriminating element than it contains is the
#: same kind of wrong — so the ratio is always >= 1 and the direction is a
#: separate column.
#:
#: Measured on the same scan, worst discriminating element per assigned phase
#: (measured and nominal both on the scored basis, see `to_scored_basis`):
#:
#:     Al.cif             --      no discriminating element at all
#:     sd_0302719   Mn   1.146    the best-fitting checkable element there is
#:     Si.cif       Si   2.063    a real 134 px Si particle, diluted
#:     sd_0302719   Fe   2.665    <- the group leader's row (4.35 vs 11.60)
#:     Al6Fe        Fe   3.547    <- her other row      (4.03 vs 14.29)
#:     Fe3_Al2_Si3  Fe 226.729    25 px nobody had noticed (0.17 vs 37.50)
#:
#: 2.0 sits in the measured gap between the largest ratio a good call produces
#: here (1.146 — and it is the Mn of the very row that fails at 2.665 on Fe,
#: so the criterion is per-element rather than a blanket verdict on the row)
#: and the smallest a diluted or wrong one does (2.063). It is placed at the
#: TOP of that gap deliberately: under a factor of two on a semi-quantitative
#: at% there is nothing to say.
#:
#: The 2.063 case is honestly reported rather than tuned away. Those 134 px
#: are genuinely silicon, and their measured Si is halved because the EDS
#: interaction volume during an EBSD session is larger than the particle —
#: the effect this module already exports a core composition for. "Deserves a
#: look" is the true statement about it; the coverage floor below is what
#: keeps it out of the headline instead of a threshold fudged to 2.5, which
#: would have left the group leader's own 2.665 row clearing by 6 %.
_IMPLAUSIBLE_FACTOR = 2.0

#: Denominator floor, so a measured zero gives a large finite ratio instead of
#: an infinity. 0.05 at% is the smallest value the region inspector will show;
#: below it there is no measurement to divide by. Fe3_Al2_Si3 above measures
#: 0.165 at% Fe and is not floored — the 226.7 is a real quotient.
_MEASURED_FLOOR_AT_PCT = 0.05

#: A row must cover this much of the scan before it is named in ``warnings``.
#: Measured coverage of the four implausible rows above: 33.45 %, 27.64 %,
#: 1.24 %, 0.23 %. The two the group leader challenged clear 5 % five-fold;
#: the other two are a 134 px particle and a 25 px sliver, which are row-level
#: curiosities and cannot move a headline. This is the "weight by how much of
#: the map the row covers" half of the criterion: EVERY implausible row is
#: flagged in the tables and counted in the summary, but only a row big enough
#: to change what the map says gets its own warning. 5 % is 540 px here.
_WARN_MIN_FRAC_OF_SCAN = 0.05

#: Below this a winner is a coin toss wearing a phase name. Margins measured
#: on the same scan's seven regions: 0.0048, 0.0492, 0.2545, 0.4110, 0.4741,
#: 0.8664, 0.9293 — one genuine tie an order of magnitude below the next, and
#: it is region 1, which is 27.64 % of the map. 0.02 sits in that gap. See
#: :func:`_margin`, which already records 0.009360 as "genuinely undecided by
#: chemistry alone".
_COINFLIP_MARGIN = 0.02

#: What ``review_flag`` ORs together. Codes, not prose — ``review_reasons``
#: carries them joined by ``;`` so a bare boolean with three causes is still
#: sortable and filterable.
REVIEW_REASONS = {
    "implausible_composition": (
        "a discriminating element is off by at least "
        f"{_IMPLAUSIBLE_FACTOR:g}x against the phase's nominal composition"),
    "undecided_between_phases": (
        "the winning score is within "
        f"{_COINFLIP_MARGIN:g} of the runner-up's — chemistry did not decide "
        "this row"),
    "untracked_assignment": (
        "nothing was painted by hand and the map still does not match the "
        "replayed chemistry; something moved it that leaves no trace"),
}


def plausibility_columns() -> List[str]:
    """The per-row verdict, placed right behind the row's phase LABEL.

    IT USED TO BE APPENDED AT THE END, and that was defensible for exactly
    one release: appending keeps the "additions only at the end" rule and
    costs no FORMAT_VERSION, so a sheet pinned to the previous version keeps
    working and simply does not see the new columns.

    It is not defensible now that people are using the file. A QA tester
    measured ``review_flag`` at column BW of 76 — the cell that says "this
    row is questionable" sitting sixty columns to the right of anything a
    human looks at, in the table she filters row by row on a weekly batch.
    A flag nobody scrolls to is a flag that does not flag.

    So the block moved, and FORMAT_VERSION moved with it (3 -> 4). That is
    the rule working rather than the rule being broken: "additions only at
    the end" says an APPEND is free, and it says anything else breaks
    loudly. A position-pinned reader of version 3 now fails on the version
    it pins instead of silently reading ``worst_element`` as
    ``mean_at_pct_Al``. Every column keeps its NAME and its meaning, so a
    reader that selects by header — every reader in this repo, and
    ``csv.DictReader`` — is unaffected.

    Behind the LABEL rather than merely near the front: the verdict is about
    the phase this row was given, so it belongs next to the phase this row
    was given. ``phase_label`` in phases and particles,
    ``phase_final_label`` in regions.
    """
    return [
        "plausibility_checked", "plausibility_note",
        "worst_element", "worst_element_measured_at_pct",
        "worst_element_nominal_at_pct", "worst_element_ratio",
        "worst_element_direction", "composition_implausible",
        "review_flag", "review_reasons",
    ]


def background_at_pct(background: Mapping[str, float]) -> Dict[str, float]:
    """``_background_levels`` speaks in SHARES. This is the same thing in at%.

    A trap worth one function: ``chemistry_score.background_levels`` returns
    the median RENORMALISED SHARE, 0..1, not a percentage. Caught by measuring
    against the real scan rather than by reading — the first cut of this check
    compared a nominal 11.60 at% against a background of 0.0286 and the
    warning printed "map background 0.03 at%" for an iron level the export's
    own ``background_at_pct_Fe`` column gives as 2.8633. Every element passed
    the discriminating test at that scale, which quietly turned the whole
    criterion off: aluminium survived only because it is also the matrix
    element. Same x100 that ``_enrichment_cells`` applies for the
    ``background_at_pct_*`` columns, so the check and the columns divide by
    the same number.

    The share is already renormalised over the SCORED elements, which is the
    basis :func:`to_scored_basis` puts the measurement on — so all three
    quantities in the comparison share a denominator.
    """
    out: Dict[str, float] = {}
    for el, v in (background or {}).items():
        try:
            f = float(v) * 100.0
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out[str(el)] = f
    return out


def discriminating_elements(
    nominal: Mapping[str, float],
    background: Mapping[str, float],
    *,
    matrix_element: Optional[str] = None,
) -> List[str]:
    """Which of a phase's elements can testify about this assignment.

    ``nominal`` must already be on the measured basis (see
    :func:`nominal_on_measured_basis`) — an element the scan never mapped
    cannot discriminate anything, because there is nothing to compare it to.
    ``background`` must be in AT%, i.e. through :func:`background_at_pct`, not
    the raw shares ``_background_levels`` returns.

    The matrix element is dropped explicitly as well as by the background
    test, for the same reason ``score_phase_ratio`` exempts it: ratio matching
    is DEFINED on the non-matrix elements. On the SampleB scan the two rules
    agree — aluminium falls out at 0.88-1.17x background on its own — so
    neither is load-bearing alone, which is why both are here.
    """
    out: List[str] = []
    for el, want in nominal.items():
        if matrix_element and el == matrix_element:
            continue
        try:
            w = float(want)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(w) or w < _MIN_NOMINAL_AT_PCT:
            continue
        bg = background.get(el)
        try:
            b = float(bg) if bg is not None else 0.0
        except (TypeError, ValueError):
            b = 0.0
        # b <= 0 means the map has none of this element anywhere, which makes
        # it maximally discriminating rather than unusable.
        if b > 0.0 and w < _DISCRIMINATING_FACTOR * b:
            continue
        out.append(el)
    return out


def nominal_on_measured_basis(
    nominal: Mapping[str, float],
    measured_elements: Iterable[str],
    ignore: Iterable[str] = (),
) -> Dict[str, float]:
    """A phase's nominal composition, renormalised over what this scan can see.

    BOTH SIDES OF THE SUBTRACTION MUST SHARE A DENOMINATOR. That lesson is
    already paid for in this file — see :func:`to_scored_basis`, where a
    measured at% including carbon was being differenced against a CIF that
    contains none. Here it is the mirror image: a phase containing an element
    the scan never mapped would claim a share of a total the measurement
    cannot account for, and every other element of that phase would then read
    systematically low against it.

    On the SampleB scan every element of every assigned phase was measured, so
    this is the identity there — it exists for the phase whose chromium
    nobody mapped.
    """
    drop = set(ignore)
    keep_set = {str(e) for e in measured_elements} - drop
    keep: Dict[str, float] = {}
    for el, v in (nominal or {}).items():
        if str(el) not in keep_set:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f) and f > 0.0:
            keep[str(el)] = f
    total = sum(keep.values())
    if total <= 1e-9:
        return {}
    return {el: v * 100.0 / total for el, v in keep.items()}


def composition_plausibility(
    measured_at_pct: Mapping[str, float],
    nominal_at_pct: Mapping[str, float],
    background: Mapping[str, float],
    measured_elements: Sequence[str],
    *,
    matrix_element: Optional[str] = None,
) -> Dict[str, Any]:
    """One row's verdict. Pure arithmetic on two compositions.

    ``measured_at_pct`` is the row's own mean, raw, as printed in the
    ``mean_at_pct_*`` cells; it is put on the scored basis here so it shares a
    denominator with the nominal. ``background`` is in AT% — the
    ``background_at_pct_*`` column, not the raw share; see
    :func:`background_at_pct`. Returns the cells named by
    :func:`plausibility_columns`, minus ``review_flag``/``review_reasons``,
    which are an OR over this and the row's other columns.

    ``composition_implausible`` is ``None``, never ``False``, when there was
    nothing to check. ``False`` would assert "checked, and fine" about a row
    that was never checked — the same distinction :data:`NOT_MEASURED` draws
    between "measured zero" and "never looked".
    """
    blank: Dict[str, Any] = {
        "plausibility_checked": False,
        "plausibility_note": "",
        "worst_element": "",
        "worst_element_measured_at_pct": None,
        "worst_element_nominal_at_pct": None,
        "worst_element_ratio": None,
        "worst_element_direction": "",
        "composition_implausible": None,
    }

    ignore = _chem_ignore()
    meas_raw = {el: float(v) for el, v in (measured_at_pct or {}).items()
                if el not in ignore}
    if not meas_raw:
        blank["plausibility_note"] = "this row has no measured composition"
        return blank
    meas = to_scored_basis(meas_raw)

    nominal = nominal_on_measured_basis(nominal_at_pct or {}, measured_elements,
                                        ignore)
    if not nominal:
        blank["plausibility_note"] = (
            "this row carries no phase, or the phase names no element this "
            "scan measured")
        return blank

    disc = discriminating_elements(nominal, background or {},
                                   matrix_element=matrix_element)
    if not disc:
        blank["plausibility_note"] = (
            "no element of this phase stands far enough above the map "
            "background to distinguish it from the matrix; chemistry cannot "
            "check this assignment")
        return blank

    worst_el = ""
    worst_ratio = 0.0
    worst_m = 0.0
    worst_n = 0.0
    for el in disc:
        m = float(meas.get(el, 0.0))
        n = float(nominal[el])
        ratio = max(n, m) / max(min(n, m), _MEASURED_FLOOR_AT_PCT)
        if ratio > worst_ratio:
            worst_el, worst_ratio, worst_m, worst_n = el, ratio, m, n

    return {
        "plausibility_checked": True,
        "plausibility_note": "",
        "worst_element": worst_el,
        "worst_element_measured_at_pct": worst_m,
        "worst_element_nominal_at_pct": worst_n,
        "worst_element_ratio": worst_ratio,
        "worst_element_direction": ("deficient" if worst_m < worst_n
                                    else "excess" if worst_m > worst_n
                                    else "exact"),
        "composition_implausible": bool(worst_ratio >= _IMPLAUSIBLE_FACTOR),
    }


def _review_reasons(row: Mapping[str, Any]) -> List[str]:
    """"This row deserves a look", ORed from whatever the row can answer.

    Only reasons the row HAS columns for. The phases table has no margin and
    no assignment source, so it contributes only the composition reason — a
    missing column must not silently read as "fine".
    """
    reasons: List[str] = []
    if row.get("composition_implausible") is True:
        reasons.append("implausible_composition")
    margin = row.get("margin_score")
    if isinstance(margin, (int, float)) and not isinstance(margin, bool):
        if math.isfinite(float(margin)) and float(margin) < _COINFLIP_MARGIN:
            reasons.append("undecided_between_phases")
    if row.get("assignment_source") == "untracked":
        reasons.append("untracked_assignment")
    return reasons


def _row_measured(row: Mapping[str, Any],
                  measured_el: Sequence[str]) -> Dict[str, float]:
    """The row's composition, read back out of the cells the file will print.

    Deliberately not recomputed from the masks. The whole complaint was that
    the two numbers sit in adjacent columns and nothing subtracts them, so the
    check has to be a function of exactly those columns — a second computation
    that drifted from them would put a warning next to numbers that do not
    support it.
    """
    out: Dict[str, float] = {}
    for el in measured_el:
        v = row.get(f"mean_at_pct_{el}")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        f = float(v)
        if math.isfinite(f):
            out[el] = f
    return out


def _apply_plausibility(
    table: Table,
    *,
    entries: Sequence[Any],
    index_key: str,
    name_key: str,
    measured_el: Sequence[str],
    background: Mapping[str, float],
    matrix_element: Optional[str],
    after: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fill the plausibility cells of every row. Returns the findings.

    A finding is one implausible row, with the numbers a warning needs. The
    table is mutated in place — columns inserted, cells filled — because the
    alternative is a parallel structure keyed on row identity, and this
    module's own worst defect class is state held in two places.

    ``after`` names the column the block is placed BEHIND; absent, or absent
    from the table, it falls back to appending at the end. See
    :func:`plausibility_columns` for why the block moved forward and why that
    cost a FORMAT_VERSION.
    """
    block = [c for c in plausibility_columns() if c not in table.columns]
    at = -1
    if after and after in table.columns:
        at = table.columns.index(after) + 1
    if at < 0:
        table.columns.extend(block)
    else:
        table.columns[at:at] = block

    # ONE conversion, here, so nothing downstream can mix shares with at%.
    bg_at_pct = background_at_pct(background)

    findings: List[Dict[str, Any]] = []
    for row in table.rows:
        idx = row.get(index_key)
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            idx = -1
        entry = entries[idx] if 0 <= idx < len(entries) else None
        nominal = (getattr(entry, "composition", None) or {}) if entry else {}
        cells = composition_plausibility(
            _row_measured(row, measured_el), nominal, bg_at_pct, measured_el,
            matrix_element=matrix_element)
        row.update(cells)
        reasons = _review_reasons(row)
        row["review_flag"] = bool(reasons)
        row["review_reasons"] = ";".join(reasons)

        if cells.get("composition_implausible") is True:
            findings.append({
                "table": table.name,
                "row": row.get(name_key, ""),
                "row_id": row.get(index_key),
                "n_px": row.get("n_px"),
                "frac_of_scan": row.get("frac_of_scan"),
                "element": cells["worst_element"],
                "measured_at_pct": cells["worst_element_measured_at_pct"],
                "nominal_at_pct": cells["worst_element_nominal_at_pct"],
                "ratio": cells["worst_element_ratio"],
                "direction": cells["worst_element_direction"],
                "background_at_pct": float(
                    bg_at_pct.get(cells["worst_element"], 0.0) or 0.0),
            })
    return findings


def plausibility_coverage(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """How much of the SCAN each verdict covers. The three add up to the map.

    WHY THIS EXISTS. The export reported both halves of the answer and never
    added them up. Measured on the real 90x120 SampleB scan (mode=cluster,
    scale_um=1.5), the block said "2 of 2 checkable phase rows are
    composition-implausible ... covering 34.69 % of the scan" — and a reader
    in a hurry takes that as "a third is dodgy, two-thirds are fine". The
    two-thirds are not fine. They are 65.31 % of a map that chemistry cannot
    check AT ALL, and the fraction this export actually corroborated is
    0.00 %. Nothing in that map was corroborated; the old wording invited the
    opposite conclusion and printed neither of the numbers that would have
    stopped it.

    THE THREE PARTITION THE RASTER, which is what makes stating all three
    honest rather than merely fuller. :func:`_phase_table` emits one row per
    phase that owns pixels PLUS a real ``unclassified`` row, and those masks
    cover every pixel exactly once — so implausible + not_checkable +
    corroborated == 1. ``frac_of_scan_accounted`` carries the sum rather than
    asserting it, so a reader can see at a glance that nothing fell out (and
    ``n_rows_without_frac`` says how many rows could not contribute, which is
    the only way the sum can fall short).

    ``corroborated`` is deliberately the narrowest word available. A row
    lands there when a discriminating element of its phase WAS compared and
    came back inside :data:`_IMPLAUSIBLE_FACTOR`: a check that passed, not a
    certificate. A row that could not be checked is neither corroborated nor
    contradicted and is counted as exactly that — the same distinction
    :data:`NOT_MEASURED` draws, and the reason ``composition_implausible`` is
    ``None`` rather than ``False`` on an unchecked row.
    """
    n_impl = n_corr = n_notchk = 0
    f_impl = f_corr = f_notchk = 0.0
    n_missing = 0

    for row in rows or []:
        v = row.get("frac_of_scan")
        f = (float(v) if isinstance(v, (int, float))
             and not isinstance(v, bool) and math.isfinite(float(v))
             else None)
        if f is None:
            # Counted, but contributing no area. A row whose fraction the
            # export could not compute must not silently become 0 % of a
            # partition that is supposed to close.
            n_missing += 1
            f = 0.0
        if not row.get("plausibility_checked"):
            n_notchk += 1
            f_notchk += f
        elif row.get("composition_implausible") is True:
            n_impl += 1
            f_impl += f
        else:
            n_corr += 1
            f_corr += f

    return {
        "n_rows_total": len(rows or []),
        "n_rows_checkable": n_impl + n_corr,
        "n_rows_implausible": n_impl,
        "n_rows_corroborated": n_corr,
        "n_rows_not_checkable": n_notchk,
        "frac_of_scan_implausible": f_impl,
        "frac_of_scan_corroborated": f_corr,
        "frac_of_scan_not_checkable": f_notchk,
        "frac_of_scan_accounted": f_impl + f_corr + f_notchk,
        "n_rows_without_frac": n_missing,
    }


def _plausibility_warnings(
    findings: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    warnings: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Name the big ones, count all of them, and return the provenance block.

    ``rows`` is the finished PHASES table. The counts and the three coverage
    fractions are read off it rather than passed in beside it, because a
    caller-supplied ``n_checked`` next to a locally-derived fraction is two
    places holding one fact — this module's own worst defect class.

    Warnings are raised from the PHASES table only. A region and the phase it
    carries are largely the same pixels — region 4 and sd_0302719 are 3613 of
    the same 3613 — so warning from both would say the same thing twice and
    double the fraction of the map it claims to describe. Regions and
    particles still carry the per-row columns, which is where a reader goes
    for the detail.
    """
    cov = plausibility_coverage(rows)
    n_checked = int(cov["n_rows_checkable"])
    total_frac = float(cov["frac_of_scan_implausible"])

    named = sorted(
        (f for f in findings
         if isinstance(f.get("frac_of_scan"), (int, float))
         and float(f["frac_of_scan"]) >= _WARN_MIN_FRAC_OF_SCAN),
        key=lambda f: -float(f["frac_of_scan"]))

    for f in named:
        frac = float(f["frac_of_scan"])
        _warn(warnings, "composition_implausible",
              f"{f['row']} covers {frac * 100:.2f} % of the scan but measures "
              f"{f['measured_at_pct']:.2f} at% {f['element']} against the "
              f"{f['nominal_at_pct']:.2f} at% its nominal composition "
              f"requires — {f['ratio']:.2f}x {f['direction']}. "
              f"{f['element']} distinguishes this phase from the matrix (map "
              f"background {f['background_at_pct']:.2f} at%), so this is not a "
              f"rounding difference. Nothing was changed: check the "
              f"assignment.")

    if findings:
        n_small = len(findings) - len(named)
        # All three numbers, in one sentence, because the reader's error is
        # made by SUBTRACTION: told only that 34.69 % is implausible, a
        # hurried reader completes the map with a 65.31 % that is fine. It is
        # not fine, it is unexamined, and the number that says so — how much
        # of the map a composition check actually corroborated — has to be in
        # the same breath or it is not in the argument at all.
        _warn(warnings, "composition_implausible_summary",
              f"{len(findings)} of {n_checked} checkable phase rows are "
              f"composition-implausible (a discriminating element off by "
              f"{_IMPLAUSIBLE_FACTOR:g}x or more). BY AREA this scan is "
              f"{cov['frac_of_scan_implausible'] * 100:.2f} % implausible, "
              f"{cov['frac_of_scan_not_checkable'] * 100:.2f} % not checkable "
              f"at all, and "
              f"{cov['frac_of_scan_corroborated'] * 100:.2f} % corroborated — "
              f"the three cover the whole map. NOT CHECKABLE IS AN ABSENCE OF "
              f"EVIDENCE, NOT A PASS: no element of those phases stands far "
              f"enough above the map background to testify about the "
              f"assignment. "
              # The manual already draws this conclusion for this scan
              # (docs/user-guide/EDS.md, "That every checkable row on this
              # scan failed is itself the finding"). It belongs in the
              # folder, where the numbers are, and not only in a manual the
              # reader of the folder may never open.
              + ("Nothing in this map was corroborated: every checkable row "
                 "failed. That is itself the finding, and it reads the other "
                 "way round from how it looks — it does not mean the phase "
                 "names are wrong, it means not one of them can be "
                 "corroborated on the element that identifies it. "
                 if n_checked and len(findings) == n_checked else "")
              + (f"{n_small} of them cover less than "
                 f"{_WARN_MIN_FRAC_OF_SCAN * 100:g} % each and are flagged in "
                 f"the tables only — sort phases.csv on worst_element_ratio."
                 if n_small else
                 "Every one of them is named above."))

    return {
        "method": (
            "For each row, the measured composition (mean_at_pct_*, put on "
            "the scored basis) is compared element by element against the "
            "assigned phase's nominal composition (nominal_at_pct_*, "
            "renormalised over the elements this scan measured). Only "
            "DISCRIMINATING elements are compared: those the phase needs at "
            "at least "
            f"{_DISCRIMINATING_FACTOR:g}x the map background and at least "
            f"{_MIN_NOMINAL_AT_PCT:g} at%. The worst such element is "
            "reported as a factor, symmetric in direction."),
        "changes_nothing": (
            "This is a report on the result. No assignment, score, ranking or "
            "grid is touched by it — phase_final is identical with and "
            "without the check."),
        "thresholds": {
            "discriminating_factor_over_background": _DISCRIMINATING_FACTOR,
            "min_nominal_at_pct": _MIN_NOMINAL_AT_PCT,
            "implausible_factor": _IMPLAUSIBLE_FACTOR,
            "measured_floor_at_pct": _MEASURED_FLOOR_AT_PCT,
            "warn_min_frac_of_scan": _WARN_MIN_FRAC_OF_SCAN,
            "coinflip_margin_score": _COINFLIP_MARGIN,
        },
        "n_rows_checkable": int(n_checked),
        "n_rows_implausible": len(findings),
        "n_rows_named_in_warnings": len(named),
        "frac_of_scan_implausible": total_frac,
        # --- the other two thirds of the answer -------------------------
        # The block used to report the implausible fraction alone, which a
        # reader completes by subtraction into "the rest is fine". These are
        # the numbers that stop that: how much of the map chemistry could not
        # examine, and how much it actually corroborated. The three close on
        # the whole raster (see plausibility_coverage).
        "n_rows_not_checkable": int(cov["n_rows_not_checkable"]),
        "frac_of_scan_not_checkable": float(cov["frac_of_scan_not_checkable"]),
        "n_rows_corroborated": int(cov["n_rows_corroborated"]),
        "frac_of_scan_corroborated": float(cov["frac_of_scan_corroborated"]),
        "frac_of_scan_accounted": float(cov["frac_of_scan_accounted"]),
        "n_rows_without_frac": int(cov["n_rows_without_frac"]),
        "coverage_note": (
            "frac_of_scan_implausible + frac_of_scan_not_checkable + "
            "frac_of_scan_corroborated = frac_of_scan_accounted, and that is "
            "1.0 for a complete map: the phase table has one row per phase "
            "that owns pixels plus a real unclassified row, and those cover "
            "every pixel exactly once. CORROBORATED is the narrow word it "
            "looks like — a discriminating element was compared and came "
            "back inside the factor. It is a check that passed, not a "
            "certificate, and it is the only one of the three that is good "
            "news. NOT CHECKABLE is neither good nor bad news: nothing about "
            "those pixels was established either way. Anything short of 1.0 "
            "in frac_of_scan_accounted is n_rows_without_frac rows whose "
            "fraction could not be computed."),
        "rows": list(findings),
        "review_reasons": dict(REVIEW_REASONS),
        "raised_from": (
            "phases.csv only. A region and the phase it carries are largely "
            "the same pixels, and a particle is one connected component of a "
            "region, so warning from all three tables would state one finding "
            "many times over and multiply the fraction of the map it claims. "
            "regions.csv AND particles.csv carry the same per-row columns "
            "(plausibility_checked ... review_reasons), which is where a "
            "reader goes for the detail and what they filter on."),
        "not_checked": (
            "An element the row MEASURES but the phase does not contain is "
            "not examined — only the phase's own discriminating elements are. "
            "Neither is a phase whose elements all sit near the map "
            "background (an Al phase on an Al matrix): plausibility_checked "
            "is False there, with plausibility_note saying why, and that is "
            "an absence of evidence rather than a pass."),
    }


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------

def resolve_folder(
    dest_dir: os.PathLike | str,
    source_path: Optional[str],
    when: Optional[datetime] = None,
    *,
    on_existing: str = "suffix",
) -> Path:
    """``<dest>/<scan-stem>_EDS_<YYYY-MM-DD_HHMM>``, never an existing one.

    An export folder is evidence. Overwriting one silently replaces a record
    somebody may already have cited, and two exports a minute apart would
    otherwise collide on the minute-resolution timestamp. ``suffix`` appends
    ``_2``, ``_3``...; ``fail`` raises ``FileExistsError``.
    """
    when = when or datetime.now()
    stem = "scan"
    if source_path:
        # Multi-suffix names (`foo.h5oina`) must not lose their first suffix,
        # and a stem is only cosmetic anyway — split once on the first dot.
        stem = Path(source_path).name.split(".")[0] or "scan"
    base = f"{stem}_EDS_{when.strftime('%Y-%m-%d_%H%M')}"
    root = Path(dest_dir)
    candidate = root / base
    if not candidate.exists():
        return candidate
    if on_existing == "fail":
        raise FileExistsError(f"{candidate} already exists")
    n = 2
    while (root / f"{base}_{n}").exists():
        n += 1
    return root / f"{base}_{n}"


def _png_size(blob: bytes) -> Tuple[Optional[int], Optional[int]]:
    """``(width, height)`` read from the IHDR; ``(None, None)`` when unreadable.

    Straight off the header rather than through Pillow: the bytes have just
    been produced by Pillow and re-decoding a whole image to learn its shape
    is work for two numbers that are eight big-endian bytes at a fixed offset.
    """
    try:
        if blob[:8] != b"\x89PNG\r\n\x1a\n" or blob[12:16] != b"IHDR":
            return None, None
        w = int.from_bytes(blob[16:20], "big")
        h = int.from_bytes(blob[20:24], "big")
        return int(w), int(h)
    except Exception:
        return None, None


def _png_pixel_count(blob: bytes) -> int:
    """Width x height from the IHDR. ``0`` on anything unreadable."""
    w, h = _png_size(blob)
    return int(w * h) if w is not None and h is not None else 0


#: Below this long side (px) a map is a "small map" whose picture must be
#: magnified to be a picture at all. The same rule as the app's export dialog
#: (`frontend/src/components/common/imageExport.js`, `defaultFactor`): the M5
#: tester's 21 x 22 px export was unusable for a report.
_SMALL_MAP_PX = 200
_SMALL_MAP_TARGET_PX = 800
_FIGURE_FACTORS = (1, 2, 4, 8, 16)


def _figure_factor(long_side_px: int) -> int:
    """Integer magnification for the report picture of a map.

    Small maps: the smallest factor of at least 8 that brings the long side to
    ``_SMALL_MAP_TARGET_PX``, capped at 16 (21 px -> 16x = 336 px, 120 px ->
    8x = 960 px). Larger maps: the largest factor up to 4 that keeps the long
    side within 4000 px, so a 1000 px map becomes 4000 px and a 3000 px map
    stays as it is.
    """
    long_side = max(1, int(long_side_px))
    if long_side < _SMALL_MAP_PX:
        for f in (8, 16):
            if long_side * f >= _SMALL_MAP_TARGET_PX:
                return f
        return 16
    for f in (4, 2, 1):
        if long_side * f <= 4000:
            return f
    return 1


def _figure_worthwhile(blob: bytes, step_um: Optional[float]) -> bool:
    """Would the figure differ from the 1:1 file? Magnified, or with a bar."""
    w, h = _png_size(blob)
    if w is None or h is None:
        return False
    if step_um and step_um > 0:
        return True
    return _figure_factor(max(w, h)) > 1


def _nice_bar_um(target_um: float) -> float:
    """The largest 1/2/5 x 10^n length not above ``target_um``."""
    import math
    if not (target_um > 0):
        return 0.0
    exp = math.floor(math.log10(target_um))
    best = 10.0 ** exp
    for m in (1, 2, 5):
        cand = m * 10.0 ** exp
        if cand <= target_um * (1 + 1e-9):
            best = cand
    return float(best)


def _format_um(value_um: float) -> str:
    text = f"{value_um:.3f}".rstrip("0").rstrip(".")
    return f"{text} µm"


def _figure_png(blob: bytes, step_um: Optional[float]) -> Tuple[bytes, Dict[str, Any]]:
    """The report version of a map picture: magnified, with a scale bar.

    Nearest-neighbour, so every pixel keeps the colour the screen renderer
    gave it and the legend in the tables still describes the picture. The bar
    (white on a black outline, bottom left, label above it) covers about a
    fifth of the width at a 1/2/5 length, and is drawn only when the pixel
    size is known -- a bar of unknown length is worse than none.

    Returns the PNG bytes and a record for provenance.json.
    """
    import io
    from PIL import Image, ImageDraw, ImageFont

    with Image.open(io.BytesIO(blob)) as src:
        im = src.convert("RGBA")
    w, h = im.size
    long_side = max(w, h)
    f = _figure_factor(long_side)
    big = im.resize((w * f, h * f), Image.NEAREST)
    info: Dict[str, Any] = {
        "scale_factor": f, "width": w * f, "height": h * f,
        "source_width": w, "source_height": h,
        # Two pixel sizes, because two rasters: the scan point of the 1:1
        # file, and the pixel of THIS picture (`scale_factor` times finer).
        "source_pixel_um": float(step_um) if step_um else None,
        "figure_pixel_um": (float(step_um) / f) if step_um else None,
        "scalebar_um": None, "scalebar_px": None,
        "resampling": "nearest",
        "note": (
            f"Every source pixel is drawn as a {f} x {f} block of the same "
            f"colour; the colours are those of the 1:1 file. scalebar_px is "
            f"in figure pixels."),
    }
    if step_um and step_um > 0:
        bar_um = _nice_bar_um(0.2 * w * float(step_um))
        if bar_um > 0:
            bar_px = max(1, int(round(bar_um / float(step_um) * f)))
            L = long_side * f
            margin = max(4, int(round(0.03 * L)))
            thick = max(4, int(round(0.015 * L)))
            x0 = margin
            y1 = h * f - margin
            y0 = y1 - thick
            draw = ImageDraw.Draw(big)
            draw.rectangle([x0 - 1, y0 - 1, x0 + bar_px + 1, y1 + 1], fill=(0, 0, 0, 255))
            draw.rectangle([x0, y0, x0 + bar_px, y1], fill=(255, 255, 255, 255))
            label = _format_um(bar_um)
            size = max(10, int(round(0.045 * L)))
            try:
                font = ImageFont.load_default(size=size)
            except TypeError:  # Pillow < 10.1: fixed-size bitmap font
                font = ImageFont.load_default()
            stroke = max(1, size // 8)
            bbox = draw.textbbox((0, 0), label, font=font, stroke_width=stroke)
            text_h = bbox[3] - bbox[1]
            draw.text((x0, y0 - 3 - text_h - bbox[1]), label, font=font,
                      fill=(255, 255, 255, 255), stroke_width=stroke,
                      stroke_fill=(0, 0, 0, 255))
            info.update({"scalebar_um": bar_um, "scalebar_px": bar_px})
    buf = io.BytesIO()
    big.save(buf, format="PNG")
    return buf.getvalue(), info


def _format_cell(value: Any, decimal: str) -> str:
    """One cell, at full precision, in the caller's decimal convention.

    ``repr`` on a float is the shortest string that round-trips, which is
    exactly "no rounding" — ``str(0.1 + 0.2)`` and ``repr(0.1 + 0.2)`` are the
    same thing in Python 3 and both give back the identical double.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        # "True"/"False": what pandas.read_csv parses back as bool without a
        # converter. "true"/"false" does not round-trip.
        return "True" if value else "False"
    if isinstance(value, (np.bool_,)):
        return "True" if bool(value) else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if math.isnan(v):
            return ""
        text = repr(v)
        return text.replace(".", ",") if decimal == "," else text
    return str(value)


def _write_table(
    path: Path, table: Table, opts: ExportOptions,
) -> WrittenFile:
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=opts.delimiter,
                       lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL)
        w.writerow(table.columns)
        for row in table.rows:
            w.writerow([_format_cell(row.get(c), opts.decimal)
                        for c in table.columns])
    return WrittenFile(path.name, len(table.rows), path.stat().st_size)


def _write_pixels(
    path: Path, columns: Sequence[str], rows: Iterable[Dict[str, Any]],
    opts: ExportOptions,
) -> WrittenFile:
    """Streamed. A 485k-pixel map with twelve elements is ~100 MB of text."""
    n = 0
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=opts.delimiter,
                       lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL)
        w.writerow(list(columns))
        for row in rows:
            w.writerow([_format_cell(row.get(c), opts.decimal) for c in columns])
            n += 1
    return WrittenFile(path.name, n, path.stat().st_size)


def _flatten(prefix: str, value: Any, out: List[Tuple[str, Any]]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(value, (list, tuple)):
        out.append((prefix, json.dumps(list(value), ensure_ascii=False,
                                       default=str)))
    else:
        out.append((prefix, value))


#: What an empty data sheet says about itself. A blank sheet reads as a
#: broken export; "no definitions were used" is a RESULT and belongs on the
#: sheet that would otherwise be blank.
_EMPTY_SHEET_NOTES = {
    "definitions": ("No region definitions were used: this map was grouped "
                    "automatically, with no hand-written composition "
                    "windows. Not an omission -- a result."),
    "particles": ("No particles: nothing on this map formed a connected "
                  "component to measure."),
    "regions": ("No regions: this map was classified per pixel, so there is "
                "no region table. See provenance.particles.basis."),
    "phases": "No phase owns a pixel on this map.",
}


#: Where the workbook sends a reader who reaches ``provenance!warnings``.
#: The list stays whole in provenance.json; in the workbook it is a sheet.
_WARNINGS_POINTER = (
    "see the 'warnings' sheet -- one row each, code and message in separate "
    "columns. The full list is also in provenance.json under 'warnings'.")


def _write_warnings_sheet(wb, warnings: Sequence[Mapping[str, str]],
                          bold) -> None:
    """One warning per ROW, wrapped, on the first sheet of the workbook.

    WHY IT IS NOT WHERE IT WAS. Every warning landed in a single cell — B300
    of a 302-row provenance sheet, 2074 characters of unwrapped JSON on one
    line. A tester put it exactly right: this is the most important content
    in the export and it was the worst-presented thing in it, and the
    difference is between "the export warned me" and "the export warned
    somebody who reads JSON".

    FIRST, before the data sheets, because a workbook opens on its first
    sheet and the whole purpose of a warning is to be met before the number
    it is about. ``code`` and ``message`` in separate columns because the
    code is what a reader filters and greps on and the prose is what they
    read; one column holding both is neither.

    Wrapped with a width, because these messages are two to four sentences
    of measured prose by design — they carry the numbers a reader needs to
    check the finding — and Excel's default renders that as one clipped line
    with an invisible remainder. Row heights are deliberately left unset so
    Excel auto-fits them to the wrapped text.
    """
    from openpyxl.styles import Alignment

    ws = wb.create_sheet(title="warnings", index=0)
    ws.append(["#", "code", "message"])
    for cell in ws[1]:
        cell.font = bold

    wrap = Alignment(wrap_text=True, vertical="top")
    for i, w in enumerate(warnings or [], start=1):
        ws.append([i, str(w.get("code", "")), str(w.get("message", ""))])
        ws.cell(row=ws.max_row, column=3).alignment = wrap
        ws.cell(row=ws.max_row, column=2).alignment = Alignment(vertical="top")

    if not warnings:
        # A blank sheet reads as a broken export. "No warnings" is a RESULT,
        # and on this export a notable one.
        ws.append([None, "(none)",
                   "This export raised no warnings. Every check this module "
                   "runs is reported in provenance.json; an empty list here "
                   "means they ran and found nothing to say, not that they "
                   "were skipped."])
        ws.cell(row=ws.max_row, column=3).alignment = wrap

    ws.freeze_panes = "A2"
    if warnings:
        ws.auto_filter.ref = ws.dimensions
    ws.column_dimensions["A"].width = 5
    ws.column_dimensions["B"].width = 34
    ws.column_dimensions["C"].width = 120


def _write_xlsx(
    path: Path, tables: Sequence[Table], provenance: Dict[str, Any],
    warnings: Optional[Sequence[Mapping[str, str]]] = None,
) -> WrittenFile:
    """A warnings sheet, one flat sheet per table, and a provenance sheet.

    Numbers go in as numbers, not as locale-formatted text: the decimal option
    is a CSV concern, and Excel applies the reader's own locale to a real
    numeric cell. Writing "3,14" into a cell here would make it text in every
    locale, which is worse than either convention.

    ``warnings`` defaults to the provenance's own snapshot. The caller passes
    its LIVE bag instead, which is a superset: `map_png_failed` and anything
    else raised after `_provenance` took its `list(warnings)` copy is in the
    live bag and not in the snapshot.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    wb.remove(wb.active)
    bold = Font(bold=True)
    for table in tables:
        ws = wb.create_sheet(title=table.name[:31])
        ws.append(list(table.columns))
        for cell in ws[1]:
            cell.font = bold
        for row in table.rows:
            ws.append([_xlsx_cell(row.get(c)) for c in table.columns])
        ws.freeze_panes = "A2"
        # Filter handles on the header row. These tables are read by
        # sorting and filtering them -- "sort phases.csv on
        # worst_element_ratio" is literally what one of this file's own
        # warnings instructs -- and a reader who has to add the filter
        # themselves does it on the wrong range about half the time,
        # which silently detaches one column from its neighbours.
        if table.rows:
            ws.auto_filter.ref = ws.dimensions
        else:
            # An empty sheet is indistinguishable from a broken export. Say
            # which of the two it is, in the sheet, where the reader is.
            ws.append([_EMPTY_SHEET_NOTES.get(
                table.name, f"(no {table.name} in this export)")])

    ws = wb.create_sheet(title="provenance")
    ws.append(["key", "value"])
    for cell in ws[1]:
        cell.font = bold
    flat: List[Tuple[str, Any]] = []
    # The warnings list is REPLACED by a pointer here, not duplicated: as a
    # flattened value it is the 2074-character single cell this change
    # exists to abolish, and leaving it in place beside the new sheet would
    # keep the unreadable copy and add a readable one. provenance.json still
    # carries the list in full.
    shown = dict(provenance)
    if "warnings" in shown:
        shown["warnings"] = _WARNINGS_POINTER
    _flatten("", shown, flat)
    for k, v in flat:
        ws.append([k, _xlsx_cell(v)])
    ws.column_dimensions["A"].width = 44
    ws.column_dimensions["B"].width = 110

    # Written LAST so it lands at index 0 in front of everything above.
    _write_warnings_sheet(
        wb, list(warnings) if warnings is not None
        else list(provenance.get("warnings") or []), bold)
    wb.active = 0

    wb.save(path)
    return WrittenFile(path.name, sum(len(t) for t in tables),
                       path.stat().st_size)


def _xlsx_cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return None if math.isnan(v) else v
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def write_export(
    tables: ExportTables,
    dest_dir: os.PathLike | str,
    *,
    options: Optional[ExportOptions] = None,
    source_path: Optional[str] = None,
) -> WriteResult:
    """Write the artefacts. Never into a folder that already exists.

    Returns what was written, with row counts and byte sizes, so the caller
    can say "1 842 particles, 412 kB" rather than "done".
    """
    opts = options or ExportOptions()
    opts.validate()
    when = opts.timestamp or datetime.now()
    src = source_path or ((tables.provenance.get("source") or {}).get("path"))
    folder = resolve_folder(dest_dir, src, when, on_existing=opts.on_existing)
    folder.mkdir(parents=True, exist_ok=False)

    want = dict(opts.artefacts or {})
    written: List[WrittenFile] = []
    warnings = list(tables.warnings)

    sheet_tables: List[Table] = []
    for key, table in (("phases", tables.phases), ("regions", tables.regions),
                       ("particles", tables.particles),
                       ("definitions", tables.definitions)):
        if not want.get(key, True):
            continue
        written.append(_write_table(folder / f"{key}.csv", table, opts))
        sheet_tables.append(table)

    # provenance.json is not optional. It is the artefact the whole
    # consultation converged on; an export without it is a screenshot.
    #
    # ensure_ascii=True, AND IT IS NOT A PREFERENCE. This file is full of
    # subscripted formulas (`Mg₁₇Al₁₂`, straight out of the CIF library), so
    # written with ensure_ascii=False it is a UTF-8 file with non-ASCII bytes
    # in it — and `json.load(open('provenance.json'))` on Windows opens with
    # the locale default (cp1252), which raises UnicodeDecodeError. A tester
    # hit exactly that. Every reader is expected to get the encoding argument
    # right; none of them do, and being right about whose fault it is does
    # not open the file.
    #
    # ASCII-escaping costs nothing and loses nothing: `₁` is valid JSON
    # that every parser decodes back to the identical string, so the file
    # round-trips character for character while being byte-for-byte ASCII —
    # readable under cp1252, UTF-8, latin-1 and whatever else a machine
    # defaults to. Verified by literally calling `json.load(open(path))` with
    # no encoding argument; see tests/test_eds_export.py.
    prov_path = folder / "provenance.json"
    prov_path.write_text(
        json.dumps(tables.provenance, indent=2, ensure_ascii=True, default=str),
        encoding="ascii")
    written.append(WrittenFile(prov_path.name, 1, prov_path.stat().st_size))

    # The one-liner, as a file. `provenance.summary` already holds it, but a
    # JSON key is not something anybody pastes under a figure four weeks
    # later; a text file in the folder is. Written unconditionally for the
    # same reason provenance.json is: an export without a caption is the
    # export whose numbers get quoted without their conditions.
    caption = str(tables.provenance.get("summary") or "").strip()
    if caption:
        cap_path = folder / "caption.txt"
        cap_path.write_text(caption + "\n", encoding="utf-8")
        written.append(WrittenFile(cap_path.name, 1, cap_path.stat().st_size))

    # The picture, beside the numbers it describes. `rows` on a PNG is its
    # pixel count — the same convention labels.npz uses, and the number that
    # says at a glance whether the image covers the whole raster.
    if want.get("map_png", True):
        for name, blob in (tables.images or {}).items():
            path = folder / name
            try:
                path.write_bytes(blob)
            except OSError as exc:
                _warn(warnings, "map_png_failed",
                      f"{name} could not be written ({exc}); every number in "
                      f"this export is unaffected.")
                continue
            w, h = _png_size(blob)
            written.append(WrittenFile(
                name, _png_pixel_count(blob), path.stat().st_size,
                width=w, height=h))

    if want.get("labels"):
        npz = folder / "labels.npz"
        np.savez_compressed(npz, **tables.labels)
        written.append(WrittenFile(npz.name, int(tables.labels["phase_id"].size),
                                   npz.stat().st_size))

    if want.get("pixels") and tables.pixel_rows is not None:
        written.append(_write_pixels(
            folder / "pixels.csv", tables.pixel_columns,
            tables.pixel_rows(), opts))

    if want.get("xlsx", True):
        try:
            written.append(_write_xlsx(folder / "summary.xlsx",
                                       sheet_tables or [tables.phases],
                                       tables.provenance, warnings))
        except Exception as exc:
            # The CSVs are already on disk and carry every number. Losing the
            # convenience workbook must not lose the export.
            _warn(warnings, "xlsx_failed",
                  f"The CSV tables were written but summary.xlsx could not "
                  f"be created ({exc}).")

    return WriteResult(str(folder), written, warnings, tables.provenance)
