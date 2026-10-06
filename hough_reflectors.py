"""Which reflector families Hough indexing uses, chosen per phase.

What Hough indexing really consumes
-----------------------------------
PyEBSDIndex builds its band-triplet library from a short list of pole
FAMILIES per phase.  Orienta's default list (``ebsd_utils._reflectors_for_phase``)
is a list of ROWS: every symmetry equivalent and both signs of every reflector
with ``d >= 1 A`` and ``|F| > 0.1 |F|max``, cut to the 70 strongest rows.
kikuchipy then folds the rows back to families
(``ReciprocalLatticeVector.unique(use_symmetry=True)``) and PyEBSDIndex drops
any family whose pole is parallel, in index space, to one listed before it
(``BandIndexer.setpolefamilies``): for Al that leaves {200}, {220}, {111}, {311}
out of six families, {400} and {222} being dropped.

This module lets the user see those families and choose them.  A choice is a
per-phase *spec*:

``{"mode": "auto", "rule": {"min_d", "f_threshold", "max_rows"}}``
    the default construction with other numbers;
``{"mode": "custom", "families": [[h, k, l], ...]}``
    exactly these families (one representative each, 3-index also for
    hexagonal phases).

No spec means today's code path, untouched.  Specs are kept in a registry keyed
like the row limits of ``ebsd_utils`` (lower-case CIF stem), because the same CIF
becomes a Hough indexer in many places (the run, the PC refinement page, the
pseudo-symmetry resolver, the phase check, ...) and only the run carries a
request.  ``ebsd_utils.create_indexer`` is the single place that applies them.
"""
from __future__ import annotations

import io
import json
import logging
import math
import os
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: The default construction of ``ebsd_utils.prepare_reflectors``.
DEFAULT_RULE = {"min_d": 1.0, "f_threshold": 0.1, "max_rows": 70}

#: A family whose structure factor is below this fraction of the strongest
#: candidate is extinct for practical purposes (it is zero up to rounding for a
#: reflection forbidden by a glide or screw axis) and is neither listed nor
#: accepted.
EXTINCT_REL_F = 1e-3

#: Smallest d spacing (A) accepted for a hand-added family.
MIN_D_ADDED = 0.3

#: More families than this make no sense for a band-triplet library.
MAX_FAMILIES = 120

#: PyEBSDIndex builds no library from a single pole family (its angle tables
#: need at least two).
MIN_FAMILIES = 2

#: Families sent to the UI at most (all selected ones are always sent).
LIST_LIMIT = 400

SPEC_STORE_FILE = "hough_reflector_specs.json"


class SpecError(ValueError):
    """A reflector spec or a hand-entered family cannot be used.

    ``code`` is stable and machine readable (the UI translates it), ``params``
    carries the values for the message.
    """

    def __init__(self, code, message=None, **params):
        super().__init__(message or code)
        self.code = code
        self.params = params


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _eu():
    """``ebsd_utils``, imported late: it imports this module from inside
    ``create_indexer`` and a top-level import here would be circular."""
    import ebsd_utils
    return ebsd_utils


def phase_key(name_or_path):
    """Registry key; the same as that of the row limits (lower-case stem)."""
    return _eu()._phase_key(name_or_path)


def phase_from_cif(path):
    """The orix ``Phase`` of a CIF, built the way Hough builds its phases.

    Uses ``ebsd_utils.hough_phase_from_cif`` (origin-choice aware) when this
    tree has it, and the plain ``Phase.from_cif(sanitize_cif(path))`` otherwise,
    so the list shown here is the list the indexer will see in either tree.
    The name is the file stem, as everywhere else.
    """
    eu = _eu()
    builder = getattr(eu, "hough_phase_from_cif", None)
    if builder is not None:
        phase = builder(str(path))
    else:
        from orix.crystal_map import Phase
        phase = Phase.from_cif(eu.sanitize_cif(str(path)))
    stem = Path(str(path)).stem
    if phase.name != stem:
        phase.name = stem
    return phase


def _gcd3(v):
    g = 0
    for x in v:
        g = math.gcd(g, abs(int(x)))
    return g or 1


def _direction(rep):
    """The reduced direction of an index triple, sign-normalised: two index
    vectors are parallel exactly when these are equal."""
    g = _gcd3(rep)
    v = [int(x) // g for x in rep]
    for x in v:
        if x != 0:
            if x < 0:
                v = [-y for y in v]
            break
    return tuple(v)


def _is_hexagonal(phase):
    try:
        return bool(phase.is_hexagonal)
    except Exception:  # noqa: BLE001
        return False


def hkl4(rep):
    """Miller-Bravais indices of a 3-index triple."""
    h, k, l = (int(x) for x in rep)
    return [h, k, -(h + k), l]


def family_label(rep, hex4=False):
    """``{200}``, ``{-111}``, ``{10-10}`` (4-index for hexagonal phases)."""
    idx = hkl4(rep) if hex4 else [int(x) for x in rep]
    if all(abs(x) < 10 for x in idx):
        return "{" + "".join(str(x) for x in idx) + "}"
    return "{" + " ".join(str(x) for x in idx) + "}"


def parse_hkl(value, phase):
    """Three-index integer triple from a user entry.

    Accepts a string (``"2 0 0"``, ``"200"`` is NOT split - ambiguous above 9),
    a list of 3 numbers, and for hexagonal / trigonal phases also 4-index
    ``[h, k, i, l]`` which is checked (``i = -(h+k)``) and converted.
    """
    if isinstance(value, str):
        txt = value.replace(",", " ").replace("{", " ").replace("}", " ")
        txt = txt.replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ")
        parts = txt.split()
        if len(parts) == 1 and len(parts[0].replace("-", "")) in (3, 4):
            # "1-10", "111", "10-10": single-digit indices written together
            s = parts[0]
            out, i = [], 0
            while i < len(s):
                if s[i] == "-":
                    out.append(s[i:i + 2])
                    i += 2
                else:
                    out.append(s[i])
                    i += 1
            parts = out
    else:
        parts = list(value or [])
    try:
        nums = [float(p) for p in parts]
    except (TypeError, ValueError):
        raise SpecError("bad_hkl", "Indices must be numbers.", value=str(value))
    if not all(math.isfinite(x) and abs(x - round(x)) < 1e-6 for x in nums):
        raise SpecError("bad_hkl", "Indices must be whole numbers.", value=str(value))
    nums = [int(round(x)) for x in nums]
    if len(nums) == 4:
        if not _is_hexagonal(phase):
            raise SpecError("bad_hkl", "Four indices only apply to hexagonal and "
                            "trigonal phases.", value=str(value))
        h, k, i, l = nums
        if i != -(h + k):
            raise SpecError("bad_hkl", "In hkil the third index must equal -(h+k).",
                            value=str(value))
        nums = [h, k, l]
    if len(nums) != 3:
        raise SpecError("bad_hkl", "Give three indices (hkl) - or four (hkil) for a "
                        "hexagonal phase.", value=str(value))
    if nums == [0, 0, 0]:
        raise SpecError("zero_vector", "(000) is not a reflection.")
    return nums


def phase_fingerprint(phase):
    """What identifies the crystal a spec was made for.

    The registry key is the CIF stem, which two different files can share and
    which stays the same when the file is edited; a custom family list means
    nothing for another cell or space group.
    """
    sg = getattr(phase.space_group, "number", None)
    try:
        lat = [round(float(x), 3) for x in phase.structure.lattice.abcABG()]
    except Exception:  # noqa: BLE001
        lat = []
    return {"space_group": None if sg is None else int(sg), "lattice": lat}


def _structure_hash(phase):
    """Hashable identity of the crystal (cache key)."""
    atoms = []
    try:
        for a in phase.structure:
            xyz = tuple(round(float(x), 4) for x in a.xyz)
            atoms.append((str(a.element), xyz, round(float(getattr(a, "occupancy", 1.0)), 3)))
    except Exception:  # noqa: BLE001
        pass
    fp = phase_fingerprint(phase)
    return (fp["space_group"], tuple(fp["lattice"]), hash(tuple(atoms)))


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------

def _num(value, name, lo, hi):
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise SpecError("bad_spec", f"{name} must be a number.", field=name)
    if not math.isfinite(x) or not (lo <= x <= hi):
        raise SpecError("bad_spec", f"{name} must be between {lo} and {hi}.",
                        field=name, lo=lo, hi=hi)
    return x


def normalise_spec(spec):
    """The canonical form of a spec, or ``None``; ``SpecError`` when unusable."""
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise SpecError("bad_spec", "A reflector spec must be an object or null.")
    mode = spec.get("mode")
    if mode == "auto":
        rule = dict(DEFAULT_RULE)
        given = spec.get("rule") or {}
        if "min_d" in given:
            rule["min_d"] = _num(given["min_d"], "min_d", 0.3, 20.0)
        if "f_threshold" in given:
            rule["f_threshold"] = _num(given["f_threshold"], "f_threshold", 0.0, 1.0)
        if "max_rows" in given:
            rule["max_rows"] = int(_num(given["max_rows"], "max_rows", 1, 1000))
        return {"mode": "auto", "rule": rule}
    if mode == "custom":
        fams = spec.get("families")
        if not isinstance(fams, (list, tuple)) or not fams:
            raise SpecError("empty_spec", "Select at least one reflector family.")
        out, seen = [], set()
        for f in fams:
            try:
                arr = [float(x) for x in f]
            except (TypeError, ValueError):
                raise SpecError("bad_hkl", "Each family needs three numbers.", value=str(f))
            if len(arr) != 3 or not all(math.isfinite(x) and abs(x - round(x)) < 1e-6
                                        for x in arr):
                raise SpecError("bad_hkl", "Each family needs three whole numbers.",
                                value=str(f))
            t = [int(round(x)) for x in arr]
            if t == [0, 0, 0]:
                raise SpecError("zero_vector", "(000) is not a reflection.")
            if tuple(t) in seen:
                continue
            seen.add(tuple(t))
            out.append(t)
        if len(out) > MAX_FAMILIES:
            raise SpecError("too_many", f"At most {MAX_FAMILIES} families.",
                            limit=MAX_FAMILIES)
        norm = {"mode": "custom", "families": out}
        fp = spec.get("phase")
        if isinstance(fp, dict):
            norm["phase"] = {"space_group": fp.get("space_group"),
                             "lattice": [round(float(x), 3) for x in fp.get("lattice", [])]}
        return norm
    raise SpecError("bad_spec", "mode must be 'auto' or 'custom'.", mode=str(mode))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_SPECS: dict = {}
_LOCK = threading.Lock()
_LOADED = False
_VERSION = 0


def _store_path():
    """Where the specs live between sessions, or None when disabled.

    A file of its own beside ``hough_reflector_limits.json``: the older file is
    never read or written here, so an older Orienta keeps working on it.
    Disabled by the same environment variable as the row limits.
    """
    base = _eu()._limits_store_path()
    if base is None:
        return None
    return base.parent / SPEC_STORE_FILE


def _load_locked():
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    path = _store_path()
    if path is None:
        return
    try:
        with io.open(path, encoding="utf-8") as fh:
            stored = json.load(fh)
        for key, value in (stored or {}).items():
            try:
                spec = normalise_spec(value)
            except SpecError:
                logger.warning("ignoring an unusable stored reflector spec for %r", key)
                continue
            if spec is not None:
                _SPECS[str(key)] = spec
    except FileNotFoundError:
        pass
    except Exception:  # noqa: BLE001 - a corrupt file means "no specs"
        logger.warning("could not read %s; reflector specs start empty", path,
                       exc_info=True)


def _save_locked():
    path = _store_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with io.open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(_SPECS, fh, indent=2, sort_keys=True)
            fh.write("\n")
        tmp.replace(path)
    except Exception:  # noqa: BLE001 - failing to persist must not fail the caller
        logger.debug("could not persist reflector specs to %s", path, exc_info=True)


def registry_version():
    """Bumped on every real change; whoever cached an indexer compares it."""
    return _VERSION


def has_specs():
    with _LOCK:
        _load_locked()
        return bool(_SPECS)


def get_spec(name_or_path):
    key = phase_key(name_or_path)
    if not key:
        return None
    with _LOCK:
        _load_locked()
        spec = _SPECS.get(key)
        return json.loads(json.dumps(spec)) if spec is not None else None


def set_spec(name_or_path, spec):
    """Store (or, with ``None``, forget) one phase's spec; returns the stored form.

    Raises ``SpecError`` for an unusable spec.  The version is bumped only when
    the stored value actually changes, so re-sending the same choice does not
    make everything that depends on it rebuild.
    """
    global _VERSION
    key = phase_key(name_or_path)
    if not key:
        raise SpecError("bad_spec", "A phase name or CIF path is required.")
    norm = normalise_spec(spec)
    with _LOCK:
        _load_locked()
        before = _SPECS.get(key)
        if norm is None:
            _SPECS.pop(key, None)
        else:
            _SPECS[key] = norm
        if before != norm:
            _VERSION += 1
            _save_locked()
    return norm


def all_specs():
    with _LOCK:
        _load_locked()
        return json.loads(json.dumps(_SPECS))


def clear_specs(persist=False):
    """Forget every spec (tests start from a clean registry with this)."""
    global _LOADED, _VERSION
    with _LOCK:
        had = bool(_SPECS)
        _SPECS.clear()
        _LOADED = True       # cleared stays cleared: no reload from disk
        if had:
            _VERSION += 1
        if persist:
            _save_locked()


# ---------------------------------------------------------------------------
# Candidate families of a phase
# ---------------------------------------------------------------------------

_CACHE = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 8


def _cached(key, build):
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    value = build()
    with _CACHE_LOCK:
        _CACHE[key] = value
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    return value


class _Table:
    """Candidate families of one phase at one ``min_d``."""

    __slots__ = ("reps", "d", "f", "mult", "row_to_family", "fmax", "min_d")

    def __init__(self):
        self.reps = []            # (h, k, l) int tuples
        self.d = []
        self.f = []
        self.mult = []
        self.row_to_family = {}   # every symmetry-equivalent row -> family index
        self.fmax = 0.0
        self.min_d = None


def _diffsims():
    from diffsims.crystallography import ReciprocalLatticeVector
    return ReciprocalLatticeVector


def _lexmax_rep(rows):
    return max(tuple(int(x) for x in r) for r in rows)


def _build_table(phase, min_d):
    R = _diffsims()
    _eu()._normalize_element_labels(phase)
    phase = phase.deepcopy()      # sanitise_phase() below expands in place
    ref = R.from_min_dspacing(phase, float(min_d))
    # Primitive-hexagonal groups: diffsims cannot filter systematic absences;
    # the structure-factor filter below removes them.
    try:
        ref = ref[ref.allowed]
    except NotImplementedError:
        pass
    ref = ref.unique(use_symmetry=True)
    ref.sanitise_phase()
    ref.calculate_structure_factor()
    f_all = np.abs(np.asarray(ref.structure_factor)).ravel()
    table = _Table()
    table.min_d = float(min_d)
    if f_all.size == 0 or float(f_all.max()) <= 0.0:
        return table
    table.fmax = float(f_all.max())
    keep = f_all >= EXTINCT_REL_F * table.fmax
    ref = ref[keep]
    f_all = f_all[keep]
    sym, mult, idx = ref.symmetrise(return_multiplicity=True, return_index=True)
    rows = np.rint(np.asarray(sym.hkl)).astype(int)
    idx = np.asarray(idx).ravel()
    d = np.asarray(ref.dspacing).ravel()
    for i in range(len(f_all)):
        fam_rows = rows[idx == i]
        table.reps.append(_lexmax_rep(fam_rows))
        table.d.append(float(d[i]))
        table.f.append(float(f_all[i]))
        table.mult.append(int(len(fam_rows)))
        for r in fam_rows:
            table.row_to_family[tuple(int(x) for x in r)] = i
    return table


def candidate_table(phase, min_d=None):
    """The candidate families of ``phase`` with ``d >= min_d`` (cached)."""
    min_d = DEFAULT_RULE["min_d"] if min_d is None else float(min_d)
    key = ("table", _structure_hash(phase), round(min_d, 4))
    return _cached(key, lambda: _build_table(phase, min_d))


def _rule_rows(phase, rule):
    """Rows of the default construction with ``rule`` (cached): what the
    existing code would give for these numbers.

    Returns ``(hkl, rows)``: ``hkl`` exactly as ``ReciprocalLatticeVector.hkl``
    reports them (floats, with the rounding noise of the cartesian round trip),
    ``rows`` the same rounded to integers for lookups.  The noise is kept on
    purpose: which member of a family diffsims reports as its representative can
    depend on it, and the representative decides the order of PyEBSDIndex's pole
    arrays.
    """
    key = ("rows", _structure_hash(phase), rule["min_d"], rule["f_threshold"],
           rule["max_rows"])

    def build():
        ref = _eu()._reflectors_for_phase(phase, min_d=rule["min_d"],
                                          f_threshold=rule["f_threshold"],
                                          max_reflectors=rule["max_rows"])
        hkl = np.asarray(ref.hkl, dtype=float)
        return hkl, np.rint(hkl).astype(int)

    return _cached(key, build)


def _extra_family(phase, rep):
    """Properties of a family that is not in the candidate table (``d`` below
    ``min_d``, or typed in by the user)."""
    R = _diffsims()
    _eu()._normalize_element_labels(phase)
    phase = phase.deepcopy()      # sanitise_phase() expands in place
    r = R(phase, hkl=np.array([rep], dtype=float))
    r.sanitise_phase()
    r.calculate_structure_factor()
    f = float(np.abs(np.asarray(r.structure_factor)).ravel()[0])
    d = float(np.asarray(r.dspacing).ravel()[0])
    sym, mult = r.symmetrise(return_multiplicity=True)
    rows = np.rint(np.asarray(sym.hkl)).astype(int)
    return {"rep": _lexmax_rep(rows), "f": f, "d": d, "mult": int(len(rows)),
            "rows": rows}


def _check_allowed(phase, rep):
    """``SpecError`` when ``rep`` is not a diffracting reflection of the phase."""
    R = _diffsims()
    r = R(phase, hkl=np.array([rep], dtype=float))
    try:
        if not bool(np.asarray(r.allowed).ravel()[0]):
            raise SpecError("forbidden", "Forbidden by the lattice centring.",
                            hkl=list(rep))
    except NotImplementedError:
        pass
    info = _extra_family(phase, rep)
    if info["f"] < 1e-3:
        raise SpecError("forbidden", "Structure factor is zero: extinct.",
                        hkl=list(rep))
    if info["d"] < MIN_D_ADDED:
        raise SpecError("too_fine", f"d = {info['d']:.2f} A is below "
                        f"{MIN_D_ADDED} A.", hkl=list(rep), d=info["d"])
    return info


# ---------------------------------------------------------------------------
# Turning a spec into reflectors
# ---------------------------------------------------------------------------

def _check_fingerprint(phase, spec):
    want = spec.get("phase")
    if not want:
        return
    have = phase_fingerprint(phase)
    same = (want.get("space_group") == have["space_group"]
            and len(want.get("lattice", [])) == len(have["lattice"])
            and all(abs(a - b) <= 2e-3 for a, b in zip(want["lattice"], have["lattice"])))
    if not same:
        raise SpecError(
            "stale_spec",
            f"The reflector selection stored for '{phase.name}' was made for a "
            "different crystal (space group or cell differ). Reset the selection "
            "for this phase and choose again.",
            phase=str(phase.name))


def spec_rows(phase, spec):
    """The reflector rows a spec stands for: ``(hkl floats, int rows)``.

    ``auto``: the default construction with the rule's numbers.  ``custom``: for
    every chosen family the rows the DEFAULT list holds of it, and the whole
    symmetry orbit for a family the default list does not hold.  Taking the
    default's own rows is what makes "the default, written out" the identical
    Hough library as the default itself - a family is represented to PyEBSDIndex
    by one of its rows, and which one comes out of the rows given.
    """
    spec = normalise_spec(spec)
    if spec["mode"] == "auto":
        return _rule_rows(phase, spec["rule"])
    _check_fingerprint(phase, spec)
    for rep in spec["families"]:
        _check_allowed(phase, rep)
    table = candidate_table(phase, DEFAULT_RULE["min_d"])
    d_hkl, d_rows = _rule_rows(phase, DEFAULT_RULE)
    d_fam = np.array([table.row_to_family.get(_row_key(r), -1) for r in d_rows],
                     dtype=int)
    wanted, extra = set(), []
    for rep in spec["families"]:
        i = table.row_to_family.get(tuple(rep))
        if i is not None:
            wanted.add(i)
        else:
            extra.append(rep)
    keep = np.isin(d_fam, list(wanted)) if wanted else np.zeros(len(d_fam), bool)
    hkl_parts = [d_hkl[keep]] if keep.any() else []
    rows_parts = [d_rows[keep]] if keep.any() else []
    # families the default list does not hold (weak, below the cut, or finer
    # than min_d): whole orbit, once even if two reps name the same one
    covered = {_row_key(r) for r in (d_rows[keep] if keep.any() else [])}
    R = _diffsims()
    # families of the table that the default list leaves out entirely
    for i in sorted(wanted):
        if not (d_fam == i).any():
            extra.append(list(table.reps[i]))
    for rep in extra:
        r = R(phase, hkl=np.array([rep], dtype=float)).symmetrise()
        hk = np.asarray(r.hkl, dtype=float)
        rw = np.rint(hk).astype(int)
        fresh = np.array([_row_key(x) not in covered for x in rw], dtype=bool)
        if fresh.any():
            hkl_parts.append(hk[fresh])
            rows_parts.append(rw[fresh])
            covered.update(_row_key(x) for x in rw[fresh])
    return np.concatenate(hkl_parts, axis=0), np.concatenate(rows_parts, axis=0)


def rlv_for_spec(phase, spec):
    """A ``ReciprocalLatticeVector`` (with structure factors, for the Kikuchi
    line overlay) holding the rows of a spec."""
    spec = normalise_spec(spec)
    if spec["mode"] == "auto":
        r = spec["rule"]
        return _eu()._reflectors_for_phase(phase, min_d=r["min_d"],
                                           f_threshold=r["f_threshold"],
                                           max_reflectors=r["max_rows"])
    hkl, _rows = spec_rows(phase, spec)
    R = _diffsims()
    _eu()._normalize_element_labels(phase)
    rlv = R(phase.deepcopy(), hkl=hkl)       # sanitise_phase() expands in place
    rlv.sanitise_phase()
    rlv.calculate_structure_factor()
    return rlv


def spec_for_phase(phase):
    """The registered spec for ``phase`` (by its name) or None."""
    return get_spec(getattr(phase, "name", ""))


def prepare_reflectors(phase_list, min_d=1.0, f_threshold=0.1, max_reflectors=70):
    """``ebsd_utils.prepare_reflectors`` with the registered specs applied.

    Without a spec for any phase this IS ``ebsd_utils.prepare_reflectors``.
    """
    eu = _eu()
    specs = [spec_for_phase(phase_list[pid]) for pid in phase_list.ids]
    if not any(s is not None for s in specs):
        return eu.prepare_reflectors(phase_list, min_d, f_threshold, max_reflectors)
    out = []
    for pid, spec in zip(phase_list.ids, specs):
        phase = phase_list[pid]
        if spec is None:
            out.append(eu._reflectors_for_phase(phase, min_d, f_threshold, max_reflectors))
        else:
            out.append(rlv_for_spec(phase, spec))
    return out[0] if len(out) == 1 else out


def apply_specs(phase_list, ref_hkl, multi):
    """Replace the reflector rows of every phase that has a spec.

    ``ref_hkl`` is what ``create_indexer`` builds: one list of hkl rows (single
    phase) or one list per phase (``multi``).  Returns ``(ref_hkl, flags)`` with
    ``flags[i]`` True for a phase that carries a spec; the input is returned
    unchanged (same object) when no phase has one.
    """
    names = [getattr(p, "name", "") for _pid, p in phase_list]
    specs = [get_spec(n) for n in names]
    if not any(s is not None for s in specs):
        return ref_hkl, [False] * len(names)
    phases = [p for _pid, p in phase_list]
    per_phase = list(ref_hkl) if multi else [ref_hkl]
    out = []
    for i, spec in enumerate(specs):
        if spec is None:
            out.append(per_phase[i])
        else:
            out.append(spec_rows(phases[i], spec)[0].tolist())
    return (out if multi else out[0]), [s is not None for s in specs]


def keep_without_spec_phases(keep, ref_hkl, flags, multi):
    """Row limits (``keep``) with the phases that carry a spec set to no limit.

    A spec says exactly which families a phase uses; the row-based limit must
    not cut them again.  Unchanged when no phase has a spec.
    """
    if not any(flags):
        return keep
    n = len(flags)
    if isinstance(keep, (list, tuple)):
        per = list(keep) + [None] * (n - len(keep))
    else:
        per = [keep] * n
    per = [None if flags[i] else per[i] for i in range(n)]
    return per


# ---------------------------------------------------------------------------
# What PyEBSDIndex will really use
# ---------------------------------------------------------------------------

def effective_families(phase, rows):
    """The pole families PyEBSDIndex keeps for these rows.

    Runs the very steps kikuchipy and PyEBSDIndex run - fold to families with
    ``unique(use_symmetry=True)``, then ``BandIndexer.setpolefamilies`` - so
    nothing is replicated by hand.  Returns ``(kept, unique_reps)``: the kept
    integer triples and every folded representative, in PyEBSDIndex's order
    (a later pole parallel to an earlier one is dropped).
    """
    from pyebsdindex.tripletvote import BandIndexer
    R = _diffsims()
    u = R(phase, hkl=np.asarray(rows, dtype=float)).unique(use_symmetry=True)
    b = BandIndexer()
    b.setpolefamilies(u.hkl)
    kept = np.asarray(b.polefamilies).reshape(-1, 3)
    folded = np.rint(np.asarray(u.hkl)).astype(int)
    return kept, folded


# ---------------------------------------------------------------------------
# The family table the UI shows
# ---------------------------------------------------------------------------

def _row_key(row):
    return tuple(int(x) for x in row)


def describe(phase, spec=None, limit=LIST_LIMIT):
    """The family table of ``phase`` under ``spec`` (None = the default).

    Every candidate family gets ``selected`` (is in the current choice),
    ``effective`` (PyEBSDIndex keeps it) and, when it is a multiple of another
    family's pole, ``parallel_with``.
    """
    spec = normalise_spec(spec)
    if spec is not None and spec["mode"] == "custom":
        _check_fingerprint(phase, spec)
    hex4 = _is_hexagonal(phase)
    rule = spec["rule"] if spec and spec["mode"] == "auto" else dict(DEFAULT_RULE)
    table = candidate_table(phase, rule["min_d"])

    fam = []            # dicts, index = candidate index (+ extras appended)
    for i, rep in enumerate(table.reps):
        fam.append({"rep": rep, "d": table.d[i], "f": table.f[i], "mult": table.mult[i],
                    "rows": None, "idx": i})
    row_to_family = dict(table.row_to_family)

    def _index_of_rows(rows):
        hit = {row_to_family[_row_key(r)] for r in rows if _row_key(r) in row_to_family}
        return hit

    # --- which families are selected ---------------------------------------
    if spec is None:
        sel_hkl, sel_rows = _rule_rows(phase, DEFAULT_RULE)
    else:
        sel_hkl, sel_rows = spec_rows(phase, spec)
    selected = _index_of_rows(sel_rows)
    if spec is not None and spec["mode"] == "custom":
        # a chosen family finer than min_d (or typed in) is not in the table:
        # list it as well
        for rep in spec["families"]:
            if tuple(rep) in row_to_family:
                continue
            info = _check_allowed(phase, rep)
            new_i = len(fam)
            fam.append({"rep": info["rep"], "d": info["d"], "f": info["f"],
                        "mult": info["mult"], "rows": info["rows"], "idx": new_i})
            for r in info["rows"]:
                row_to_family[_row_key(r)] = new_i
            selected.add(new_i)
        selected |= _index_of_rows(sel_rows)

    kept, folded = effective_families(phase, sel_hkl) if len(sel_hkl) else (
        np.zeros((0, 3), int), np.zeros((0, 3), int))
    kept_set = set()
    for r in kept:
        hit = row_to_family.get(_row_key(r))
        if hit is not None:
            kept_set.add(hit)

    # --- parallel groups (index space) --------------------------------------
    groups = {}
    for i, f in enumerate(fam):
        groups.setdefault(_direction(f["rep"]), []).append(i)

    fmax = table.fmax or max((f["f"] for f in fam), default=1.0) or 1.0
    out = []
    for i, f in enumerate(fam):
        label = family_label(f["rep"], hex4)
        mates = [family_label(fam[j]["rep"], hex4)
                 for j in groups[_direction(f["rep"])] if j != i]
        dropped_by = None
        if i in selected and i not in kept_set:
            for j in kept_set:
                if _direction(fam[j]["rep"]) == _direction(f["rep"]):
                    dropped_by = family_label(fam[j]["rep"], hex4)
                    break
        out.append({
            "hkl": [int(x) for x in f["rep"]],
            "hkl4": hkl4(f["rep"]) if hex4 else None,
            "label": label,
            "d": round(f["d"], 4),
            "f": round(f["f"], 4),
            "rel_f": round(f["f"] / fmax, 4),
            "mult": f["mult"],
            "selected": i in selected,
            "effective": i in kept_set,
            "parallel_with": mates,
            "dropped_by": dropped_by,
        })
    out.sort(key=lambda r: (-r["d"], -r["f"], r["hkl"]))
    total = len(out)
    truncated = False
    if total > limit:
        head = [r for r in out[:limit]]
        extra = [r for r in out[limit:] if r["selected"]]
        out = head + extra
        truncated = True
    sg = getattr(phase.space_group, "short_name", None)
    return {
        "phase_name": str(phase.name),
        "point_group": phase.point_group.name if phase.point_group else None,
        "space_group": sg,
        "hexagonal": hex4,
        "mode": "default" if spec is None else spec["mode"],
        "spec": spec,
        "rule": rule,
        "default_rule": dict(DEFAULT_RULE),
        "families": out,
        "n_total": total,
        "truncated": truncated,
        "n_selected": len(selected),
        "n_effective": len(kept_set),
        "effective": [[int(x) for x in r] for r in kept],
        "fingerprint": phase_fingerprint(phase),
    }


def spec_with_fingerprint(phase, spec):
    """``spec`` normalised, a custom one stamped with ``phase``'s fingerprint."""
    spec = normalise_spec(spec)
    if spec is not None and spec["mode"] == "custom":
        spec["phase"] = phase_fingerprint(phase)
    return spec


def validate_family(phase, value, spec=None):
    """Check one hand-entered family against the phase.

    Returns the family's properties; ``SpecError`` (codes ``bad_hkl``,
    ``zero_vector``, ``forbidden``, ``too_fine``, ``duplicate``) when it cannot
    be used.  ``duplicate`` carries the label of the family it equals.  A family
    parallel to a listed one is NOT an error: ``parallel_with`` says which, the
    caller decides (PyEBSDIndex will drop the later of the two).
    """
    rep = parse_hkl(value, phase)
    info = _check_allowed(phase, rep)
    hex4 = _is_hexagonal(phase)
    table = candidate_table(phase, (normalise_spec(spec) or {}).get("rule", DEFAULT_RULE)["min_d"]
                            if spec and spec.get("mode") == "auto" else DEFAULT_RULE["min_d"])
    row_to_family = table.row_to_family
    known = None
    for r in info["rows"]:
        if _row_key(r) in row_to_family:
            known = row_to_family[_row_key(r)]
            break
    rep_canon = info["rep"]
    label = family_label(rep_canon, hex4)
    cur = normalise_spec(spec)
    current = []
    if cur is not None and cur["mode"] == "custom":
        current = [tuple(x) for x in cur["families"]]
    in_spec = False
    for c in current:
        if c == tuple(rep_canon) or (known is not None and row_to_family.get(c) == known):
            in_spec = True
    if in_spec:
        raise SpecError("duplicate", f"{label} is already in the list.",
                        label=label, hkl=list(rep_canon))
    # Parallel to a family the choice already holds: PyEBSDIndex will keep one of
    # the two. (Families that are only candidates do not count: they are not used.)
    mates = []
    for c in current:
        if _direction(c) == _direction(rep_canon):
            lab = family_label(c, hex4)
            if lab != label and lab not in mates:
                mates.append(lab)
    return {
        "hkl": [int(x) for x in rep_canon],
        "hkl4": hkl4(rep_canon) if hex4 else None,
        "label": label,
        "d": round(info["d"], 4),
        "f": round(info["f"], 4),
        "rel_f": round(info["f"] / (table.fmax or info["f"] or 1.0), 4),
        "mult": info["mult"],
        "parallel_with": mates,
        "in_candidates": known is not None,
    }


def expand_top_n(phase, n, rule=None):
    """A custom spec of the ``n`` strongest families that PyEBSDIndex would keep.

    Strongest = largest structure factor.  A family parallel to one already
    taken is skipped (PyEBSDIndex would drop it, and it would only use up one
    of the ``n``).
    """
    n = int(n)
    if n < MIN_FAMILIES:
        raise SpecError("too_few", f"Hough indexing needs at least {MIN_FAMILIES} "
                        "distinct reflector families.", minimum=MIN_FAMILIES)
    rule = dict(DEFAULT_RULE) if rule is None else rule
    table = candidate_table(phase, rule["min_d"])
    order = sorted(range(len(table.reps)), key=lambda i: (-table.f[i], -table.d[i]))
    picked, dirs = [], set()
    for i in order:
        d = _direction(table.reps[i])
        if d in dirs:
            continue
        dirs.add(d)
        picked.append(list(table.reps[i]))
        if len(picked) == n:
            break
    return spec_with_fingerprint(phase, {"mode": "custom", "families": picked})
