"""Notes a Hough result carries about itself.

A caveat is ``{"code": str, "phases": [name, ...], "text": str}``. It lives in
``result.metadata["caveats"]`` (a list), is merged by every site that builds a new
result out of per-phase results (``hough_reflectors.merge_build_records``), is
written next to ``hough_reflectors`` in an exported file, and reaches the page in
``task_result`` and ``/result/last``. The page words it from ``code`` in the user's
language and uses ``text`` only for a code it has no sentence for.

Three codes exist, strongest first. None of them names a cause beyond what was
measured: the band search gives no sign for a band normal, so the bands alone do not
always fix which of several equivalent orientations is the true one.

``houghTrigonalSetting``: for the seven space groups of
``phase_cell.PER_REFLECTION_SPACE_GROUPS`` (149, 151, 153, 157, 159, 162, 163)
PyEBSDIndex uses the two-fold axes of the other trigonal setting (like orix, it does
not tell -31m / -3m1, 312 / 321, 31m / 3m1 apart), and Hough often cannot tell apart
orientations 60 degrees apart about c.

``houghHexagonalLaue``: the trigonal and hexagonal space groups of the Laue classes -3
and 6/m (143 to 148, 168 to 176). PyEBSDIndex uses the right operators there; the
orientation it returns is often related to the true one by a rotation that is a
symmetry of the hexagonal cell but not of the crystal.

``houghHexagonalVariants``: a softer hint for the primitive and a few other space groups
of Laue class -3m1, where the same happens in a smaller share of the patterns and only
as a 60 degree turn about c.

The run is not refused and its numbers are not changed; the result says so.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

HOUGH_TRIGONAL_SETTING = "houghTrigonalSetting"
HOUGH_HEXAGONAL_LAUE = "houghHexagonalLaue"
HOUGH_HEXAGONAL_VARIANTS = "houghHexagonalVariants"

# Shown strongest first, in a result and in the banner.
_CODE_ORDER = (HOUGH_TRIGONAL_SETTING, HOUGH_HEXAGONAL_LAUE, HOUGH_HEXAGONAL_VARIANTS)
# Codes that are a hint, not a warning (the log line and the progress line say so).
SOFT_CODES = frozenset({HOUGH_HEXAGONAL_VARIANTS})

# Where the numbers below come from: simulated patterns (60 x 60 px, one detector
# geometry, noise-free and with Poisson noise, 200 random orientations per structure,
# three runs that differ only in how ties are decided) of model structures of the
# space group in question (one atom in a general position, or two) and of a few real
# structures (quartz, calcite, bismuth, antimony, corundum), indexed with PyEBSDIndex
# with the reflector list Orienta builds. The master patterns are kinematic with hard
# band edges. No measured patterns were used. The cells and atom positions of the real
# structures were written from the literature and checked by bond lengths, not read
# from a structure file; for corundum cell and sites come from the literature and were
# not checked against a file.
#
# houghTrigonalSetting: measured for space group 162 only (149, 151, 153, 157, 159
# and 163 were not): 25 to 53 % of the patterns got the orientation turned by 60 degrees
# about c, depending on the structure. The mechanism (the operators the library is built
# with) is the same for all seven groups.
#
# houghHexagonalLaue (below): 23 to 68 % of the patterns got a wrong variant, depending
# on the structure and the crystal class. In 6/m only a 180 degree turn about an axis in
# the basal plane (60 degrees about c is a symmetry of the crystal there and did not
# occur); in -3 with an R lattice (146, 148, hexagonal axes) only that turn about a
# basal axis of the a type; in -3 with a P lattice (143, 147) all three kinds. Measured:
# 143, 146, 147, 148, 168, 174, 175, 176. Not measured, in the list because they share
# the Laue class: 144, 145 (screw-axis partners of 143), 169 to 173 (partners of 168 and
# 174). 146 and 148 in rhombohedral axes were not measured either (the same crystal in
# other axes).
#
# houghHexagonalVariants (below): primitive lattices of Laue class -3m1 gave the
# 60 degree variant in 0 to about 35 % of the patterns, depending on the structure
# (quartz 7.5 %, corundum 7.5 %). None of the measured structures showed a 180 degree
# turn about an axis in the basal plane, so the sentence names only the 60 degree turn.
# Measured: 150, 152, 154, 156, 164, 165, 167. Not measured, in the list because it
# shares the Laue class: 158 (with 156).
#
# In the list although an R lattice: 167 (R-3c), because of one exception. Corundum
# gave the variant in 7.5 % of the patterns (185 of 200 correct), and haematite has the
# same structure; both are common in EBSD. The oxygen sublattice is nearly hexagonal
# close-packed. The other 11 of 12 R structures confused the variants in at most 0.5 %.
#
# Not in any list:
#   * 155, 160, 161, 166 (R lattices): a 60 degree turn about c is not a symmetry of an
#     R lattice (it swaps obverse and reverse), and the measured structures confused
#     the variants in at most 0.5 % of the patterns (161 was not measured).
#   * 177 to 194 (6/mmm): the crystal has the full symmetry of the hexagonal lattice,
#     so there is no variant to confuse (none found in the control structures).
#   * the seven groups of ``houghTrigonalSetting``.
#   * cubic, tetragonal, orthorhombic, monoclinic, triclinic.
# An alternative setting number (diffpy: 1146, 1148 for the rhombohedral axes of 146
# and 148) is reduced with ``% 1000`` like the other lists of this kind. It is the same
# crystal in other axes, so it gets the same note as the hexagonal axes; R groups in
# rhombohedral axes (1155, 1160, 1161, 1166) reduce to the excluded numbers, 1167 to 167.
#
# The two lists are written out and disjoint from each other and from the seven groups
# (a test holds that).
HEXAGONAL_LAUE_SPACE_GROUPS = frozenset({
    143, 144, 145, 146, 147, 148,
    168, 169, 170, 171, 172, 173, 174, 175, 176,
})
HEXAGONAL_VARIANT_SPACE_GROUPS = frozenset({
    150, 152, 154, 156, 158,
    164, 165, 167,
})


def _head(phases: List[str]) -> Tuple[str, bool]:
    """``("Phase A", True)`` for one phase, ``("Phases A, B", False)`` for several."""
    if len(phases) == 1:
        return f"Phase {phases[0]}", True
    return "Phases " + ", ".join(phases), False


def trigonal_setting_text(phases: List[str]) -> str:
    """The English sentence of ``houghTrigonalSetting`` (the page's fallback). It says
    what can happen and gives no number."""
    head, one = _head(phases)
    which = "this space group" if one else "these space groups"
    return (
        f"{head}: For {which} PyEBSDIndex uses the two-fold axes of the other trigonal "
        f"setting, and Hough often cannot tell apart orientations 60 degrees apart about "
        f"c. Spherical or Dictionary indexing may help."
    )


def hexagonal_laue_text(phases: List[str]) -> str:
    """The English sentence of ``houghHexagonalLaue``; no percentage, because the rate
    depends on the structure."""
    head, one = _head(phases)
    crystal = "this crystal" if one else "these crystals"
    return (
        f"{head}: Hough often returns an orientation related to the true one by a "
        f"rotation that is a symmetry of the hexagonal cell but not of {crystal} "
        f"(depending on the crystal class, 180 degrees about an axis in the basal plane "
        f"or 60 degrees about c). Spherical or Dictionary indexing may help."
    )


def hexagonal_variants_text(phases: List[str]) -> str:
    """The English sentence of ``houghHexagonalVariants``; no percentage, because the
    rate depends on the structure."""
    head, one = _head(phases)
    crystal = "this crystal" if one else "these crystals"
    return (
        f"{head}: Hough cannot always tell apart orientations related by a rotation "
        f"that is a symmetry of the hexagonal cell but not of {crystal} (60 degrees "
        f"about c). For important results, Spherical or Dictionary indexing may help."
    )


_TEXT_OF_CODE = {
    HOUGH_TRIGONAL_SETTING: trigonal_setting_text,
    HOUGH_HEXAGONAL_LAUE: hexagonal_laue_text,
    HOUGH_HEXAGONAL_VARIANTS: hexagonal_variants_text,
}


def _entry(code: str, phases: List[str], text: Optional[str] = None) -> Dict[str, Any]:
    if text is None:
        make = _TEXT_OF_CODE.get(code)
        text = make(phases) if make else ""
    return {"code": code, "phases": list(phases), "text": text}


def trigonal_setting_phases(phase_list) -> List[str]:
    """Names of the phases of ``phase_list`` whose space group is one PyEBSDIndex gets
    wrong, in list order, each name once. The set is
    ``phase_cell.needs_per_reflection_structure_factor`` (alternative settings
    included), not a second list."""
    from phase_cell import needs_per_reflection_structure_factor

    names: List[str] = []
    for _pid, phase in phase_list:
        if needs_per_reflection_structure_factor(phase):
            name = str(getattr(phase, "name", "") or "")
            if name not in names:
                names.append(name)
    return names


def _phases_in(phase_list, numbers) -> List[str]:
    """Names of the phases of ``phase_list`` whose space group number (alternative
    setting numbers reduced with ``% 1000``) is in ``numbers``, in list order, each
    name once."""
    names: List[str] = []
    for _pid, phase in phase_list:
        sg = getattr(getattr(phase, "space_group", None), "number", None)
        if sg is not None and int(sg) % 1000 in numbers:
            name = str(getattr(phase, "name", "") or "")
            if name not in names:
                names.append(name)
    return names


def hexagonal_laue_phases(phase_list) -> List[str]:
    """Names of the phases whose space group is in :data:`HEXAGONAL_LAUE_SPACE_GROUPS`."""
    return _phases_in(phase_list, HEXAGONAL_LAUE_SPACE_GROUPS)


def hexagonal_variant_phases(phase_list) -> List[str]:
    """Names of the phases whose space group is in :data:`HEXAGONAL_VARIANT_SPACE_GROUPS`."""
    return _phases_in(phase_list, HEXAGONAL_VARIANT_SPACE_GROUPS)


def hough_caveats(phase_list) -> List[Dict[str, Any]]:
    """The caveats a Hough run over ``phase_list`` must carry (empty: none), the
    stronger first. A phase has at most one of the three (the space groups differ)."""
    out = []
    for code, find in ((HOUGH_TRIGONAL_SETTING, trigonal_setting_phases),
                       (HOUGH_HEXAGONAL_LAUE, hexagonal_laue_phases),
                       (HOUGH_HEXAGONAL_VARIANTS, hexagonal_variant_phases)):
        names = find(phase_list)
        if names:
            out.append(_entry(code, names))
    return out


def log_and_progress(caveat: Dict[str, Any]) -> Tuple[bool, str]:
    """``(is_warning, progress line)`` for one caveat of :func:`hough_caveats`. A
    warning gets the warning sign; a soft hint (:data:`SOFT_CODES`) reads
    ``Hough note: ...`` and is logged below warning level."""
    if caveat["code"] in SOFT_CODES:
        return False, f"Hough note: {caveat['text']}"
    return True, f"⚠ {caveat['text']}"


def merge_caveats(target, sources: Iterable[Any]) -> None:
    """Carry ``metadata["caveats"]`` of per-phase sub-results onto the result that
    merges them. One entry per code; the phases of equal codes are united (each
    name once, in the order met), and the text of a known code is written again for
    the united phases. A source without caveats, or a result whose metadata is not a
    dict, contributes nothing."""
    md = getattr(target, "metadata", None)
    if not isinstance(md, dict):
        return
    merged: Dict[str, Dict[str, Any]] = {}

    def take(entries):
        for e in entries or ():
            if not isinstance(e, dict) or not isinstance(e.get("code"), str):
                continue
            cur = merged.setdefault(e["code"], _entry(e["code"], [], e.get("text") or ""))
            for name in e.get("phases") or ():
                if name not in cur["phases"]:
                    cur["phases"].append(name)

    take(md.get("caveats"))
    for source in sources:
        smd = getattr(source, "metadata", None)
        take(smd.get("caveats") if isinstance(smd, dict) else None)
    out = []
    for code, e in merged.items():
        make = _TEXT_OF_CODE.get(code)
        if make and e["phases"]:
            e["text"] = make(e["phases"])
        out.append(e)
    # the known codes strongest first, any other code after them, each group in the
    # order met (sorted() is stable)
    out.sort(key=lambda e: _CODE_ORDER.index(e["code"]) if e["code"] in _CODE_ORDER
             else len(_CODE_ORDER))
    if out:
        md["caveats"] = out


def clean_caveats(raw: Any) -> Optional[List[Dict[str, Any]]]:
    """``raw`` as read from a file (a trust boundary), or None if it is not a list
    of ``{"code": str, "phases": [str], "text": str}``. Entries of another shape are
    dropped; extra keys are not kept."""
    if not isinstance(raw, list):
        return None
    out = []
    for e in raw:
        if (isinstance(e, dict) and isinstance(e.get("code"), str)
                and isinstance(e.get("text"), str)
                and isinstance(e.get("phases"), list)
                and all(isinstance(p, str) for p in e["phases"])):
            out.append({"code": e["code"], "phases": list(e["phases"]), "text": e["text"]})
    return out or None
