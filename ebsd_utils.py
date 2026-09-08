import io
import json
import logging
import os
import re
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from diffsims.crystallography import ReciprocalLatticeVector
from kikuchipy.signals import EBSD

logger = logging.getLogger(__name__)

# A CIF ``_atom_site_type_symbol`` should be a bare element symbol, but
# ICSD / Springer-Materials exports deviate in two ways that make diffsims
# return ZERO scattering for the atom -> empty reflector list -> PC refinement /
# Hough indexing get no bands (the phase looks un-loadable, or worse, indexes
# with atoms silently missing):
#   1. an oxidation-state suffix:        "Fe0+", "Al3+", "O2-"
#   2. a shared (mixed-occupancy) site:  "0.884Al + 0.116Si", "0.8Fe + 0.2Mn"
# Both are normalised to a single clean element symbol. For a mixed site we take
# the DOMINANT element (highest fraction) at full occupancy — the same ordered
# approximation EMsoft/.xtal uses; neighbours in Z (Al/Si, Fe/Mn) scatter almost
# identically so the Kikuchi band geometry is unaffected.
_KNOWN_ELEMENTS = frozenset(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I "
    "Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt "
    "Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu".split()
)
_OXIDATION_SUFFIX_RE = re.compile(r'\s*[0-9]*[+-]$')
_MIX_TERM_RE = re.compile(r'([0-9]*\.?[0-9]+)?\s*([A-Za-z][a-zA-Z]?)')


def _clean_element_label(raw):
    """Map a messy CIF type_symbol to a single clean element symbol.

    Handles oxidation suffixes (``Fe0+``->``Fe``), case (``al``->``Al``) and
    mixed-occupancy sites (``0.884Al + 0.116Si``->``Al``, dominant element).
    Returns the input unchanged if nothing resolves, so behaviour is never
    worse than before.
    """
    e = str(raw).strip()
    s = _OXIDATION_SUFFIX_RE.sub('', e).strip()
    if s in _KNOWN_ELEMENTS:
        return s
    if s.capitalize() in _KNOWN_ELEMENTS:
        return s.capitalize()
    # Mixed / shared site -> dominant element by fraction.
    best, best_frac = None, -1.0
    for frac, el in _MIX_TERM_RE.findall(e):
        el = _OXIDATION_SUFFIX_RE.sub('', el).strip().capitalize()
        if el not in _KNOWN_ELEMENTS:
            continue
        f = float(frac) if frac else 1.0
        if f > best_frac:
            best_frac, best = f, el
    return best if best is not None else e


def _normalize_element_labels(phase):
    """Resolve oxidation states and mixed-occupancy sites to a single clean
    element symbol for every atom, in place, so diffsims can look up scattering
    factors. See ``_clean_element_label``.

    Idempotent, and a verified no-op for already-clean labels (checked across
    the whole CIF library: 5 phases fixed from 0/un-loadable, 2 corrected where
    atoms were silently dropped, 23 unchanged). EMsoft/.xtal paths are
    unaffected — they key on atomic number, not these labels.
    """
    try:
        for a in phase.structure:
            e = str(getattr(a, 'element', '') or '')
            e2 = _clean_element_label(e)
            if e2 and e2 != e:
                a.element = e2
    except Exception:
        logger.debug("element-label normalisation skipped", exc_info=True)
    return phase


def _reflectors_for_phase(phase, min_d=1.0, f_threshold=0.1, max_reflectors=70):
    """Compute filtered reflectors for a single phase."""
    _normalize_element_labels(phase)
    ref = ReciprocalLatticeVector.from_min_dspacing(phase, min_d)
    # ``allowed`` (systematic-absence filter) raises NotImplementedError for
    # primitive-hexagonal space groups in diffsims. Those absent reflections
    # have F≈0 anyway, so the structure-factor threshold below removes them —
    # skip the allowed-filter rather than failing the whole phase.
    try:
        ref = ref[ref.allowed]
    except NotImplementedError:
        logger.debug("phase %s: .allowed not implemented for this space group; "
                     "relying on the structure-factor filter",
                     getattr(phase, 'name', '?'))
    ref = ref.unique(use_symmetry=True).symmetrise()
    ref.sanitise_phase()
    ref.calculate_structure_factor()
    F = np.abs(ref.structure_factor)
    filt = ref[F > f_threshold * F.max()]

    # Limit to strongest max_reflectors
    n_ref = filt.hkl.shape[0]
    if n_ref > max_reflectors:
        sf = np.abs(filt.structure_factor)
        strongest = np.argsort(-sf)[:max_reflectors]
        filt = filt[strongest.tolist()]

    return filt


def prepare_reflectors(phase_list, min_d=1.0, f_threshold=0.1, max_reflectors=70):
    """
    Generates filtered reflectors for each phase in a PhaseList.

    Returns a single ReciprocalLatticeVector for one phase,
    or a list of them for multi-phase indexing.
    """
    ids = phase_list.ids
    if len(ids) == 1:
        return _reflectors_for_phase(phase_list[ids[0]], min_d, f_threshold, max_reflectors)
    return [_reflectors_for_phase(phase_list[pid], min_d, f_threshold, max_reflectors) for pid in ids]


# PyEBSDIndex derives a phase's Laue class from its space-group NUMBER, and its
# lookup table only covers the standard numbers 1-230. A CIF written in a
# NON-STANDARD SETTING hands diffpy/orix an alternate-setting number instead
# (beta-AlFeSi is "A12/a1" -> 4015), which runs off the end of that table and
# resolves to CUBIC m-3m. Every reflector is then expanded into 24 "equivalent"
# poles that are all geometrically distinct on a monoclinic lattice, so the
# number of unique inter-pole angles explodes — and since the band-triplet
# library is provisioned as npoles * nangs**3 / 6, indexing dies with
# "Unable to allocate 1.40 TiB for an array with shape (64177025400, 3)".
# Mapping the number back to its standard setting keeps the crystal system and
# the Laue class (verified for all 284 alternate settings in diffpy's table)
# and brings that phase down to 3.6 GiB provisioned / 6042 real triplets.
_MAX_STANDARD_SPACE_GROUP = 230


def _standard_setting_phase_list(phase_list):
    """Phase list whose space groups all carry a standard 1-230 number.

    Only the Hough indexer sees this — reflectors are still computed from the
    phase exactly as the CIF defines it, so the real centring and systematic
    absences are untouched. Returns the input unchanged when nothing needs
    fixing (the common case).
    """
    off_standard = {}
    for pid, phase in phase_list:
        number = getattr(phase.space_group, "number", None)
        if number is not None and not (1 <= number <= _MAX_STANDARD_SPACE_GROUP):
            off_standard[pid] = number
    if not off_standard:
        return phase_list

    fixed = phase_list.deepcopy()
    for pid, number in off_standard.items():
        phase = fixed[pid]
        standard = number % 1000  # diffpy numbers alternate settings <n>*1000 + <number>
        if not (1 <= standard <= _MAX_STANDARD_SPACE_GROUP):
            raise ValueError(
                f"Phase '{phase.name}' has space group number {number}, which is neither "
                f"a standard space group (1-230) nor a recognised alternate setting. "
                f"PyEBSDIndex cannot determine its symmetry from that — re-export the CIF "
                f"in a standard setting, or index this phase with Dictionary/Spherical."
            )
        logger.warning(
            "phase %s: space group %s (%s) is a non-standard setting; the Hough indexer "
            "gets the standard number %d instead so PyEBSDIndex reads the correct Laue "
            "class (reflectors are unaffected)",
            phase.name, number, getattr(phase.space_group, "short_name", "?"), standard,
        )
        phase.space_group = standard
    return fixed


#: Bytes per row of PyEBSDIndex's band-triplet library: `libANG` (3 x float64)
#: plus `libID` (3 x int64), both allocated at full size in `build_trip_lib`.
_TRIPLET_ROW_BYTES = 48

#: Reflector counts whose predicted library size is REPORTED when the full set
#: does not fit, so the caller (and the user) can choose one knowingly.
#:
#: These are NOT applied automatically, and that is a measured decision rather
#: than caution. Trimming families silently changes the answer: on Ni (m-3m, 58
#: families, 800 real patterns) going to 40 or 32 is bit-identical — 0.000 deg
#: median deviation, the same 495/800 indexed — but 24 collapses to 192/800 at
#: 119.7 deg deviation, and 16 returns a median fit of 180 deg. The failure
#: hands back plausible-looking orientations, not an error. A low-symmetry phase
#: is worse off still: with no symmetry a "family" is a single pole, so the same
#: count covers far fewer directions than it does in a cubic phase.
#:
#: So the machine refuses what it cannot afford and says what each choice would
#: cost; picking one is the user's call, on their phase and their data.
#: Measured on
#: Al7FeCu2.cif (written as `P 1`, so every family is a single pole) with a
#: 156x128 detector — the library is O(npoles * nangs**3), so it collapses fast:
#:
#:     70 reflectors  1,000,494,880 rows   44.7 GiB
#:     50               112,498,750 rows    5.0 GiB
#:     40                32,412,765 rows    1.4 GiB
#:     30                 4,607,680 rows    0.21 GiB
#:     24                   574,860 rows    0.03 GiB
#:
#: The same phase at `mmm` needs 0.59 GiB for all 70, and cubic `m-3m` never
#: gets near the budget — so this ladder is only ever walked by the low-symmetry
#: phases that would otherwise take the machine down with them.
_REFLECTOR_LADDER = (50, 40, 32, 24, 20, 16, 12)

#: Probe ceiling for `predict_triplet_library`, in library rows (48 MiB).
_PREDICT_PROBE_ROWS = 1_000_000

_INDEXER_BUILD_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Per-phase reflector limits
#
# The limit belongs to the PHASE, not to one run: the same CIF is turned into a
# Hough indexer in nine places — the main run, the pseudo-symmetry resolver that
# follows a spherical run, the phase check, single-pixel pattern match, the
# quick test and PC refinement — and only the first of those is reached by a
# parameter on the indexing request. Threading an argument through the other
# eight call chains would mean eight chances to forget one, and forgetting is
# silent: that phase simply becomes un-indexable again on a machine where it had
# just been made to work.
#
# So it is registered once, against the phase, and every builder reads it.
# ---------------------------------------------------------------------------
_PHASE_REFLECTOR_LIMITS: dict = {}
_PHASE_LIMITS_LOCK = threading.Lock()
_PHASE_LIMITS_LOADED = False

#: Set by the test suite so a test run never reads or writes the real user
#: file — the same escape hatch ORIENTA_NO_FILE_LOG gives the log.
_LIMITS_STORE_ENV = "ORIENTA_NO_PHASE_LIMIT_STORE"


def _limits_store_path():
    """Where the limits live between sessions, or None when disabled.

    The per-machine config directory the rest of the app already uses
    (``%APPDATA%/Kikuchipy`` on Windows), reached through the ONE definition of
    it rather than re-deriving the OS convention here. Imported lazily: this
    module sits at the repo root and must not take a hard dependency on the
    backend package, which is not importable in every context that uses it
    (scripts, the beta tree).
    """
    if os.environ.get(_LIMITS_STORE_ENV):
        return None
    try:
        from backend.api.services.user_config_manager import config_path
        base = config_path().parent
    except Exception:  # noqa: BLE001 — a missing backend must not break indexing
        base = Path.home() / ".orienta"
    return base / "hough_reflector_limits.json"


def _load_phase_limits_locked():
    """Read the stored limits once. Caller holds the lock."""
    global _PHASE_LIMITS_LOADED
    if _PHASE_LIMITS_LOADED:
        return
    # Marked loaded FIRST: a broken or absent file must not make every lookup
    # retry the disk, and `clear_phase_reflector_limits` relies on this flag to
    # keep a cleared registry cleared.
    _PHASE_LIMITS_LOADED = True
    path = _limits_store_path()
    if path is None:
        return
    try:
        with io.open(path, encoding="utf-8") as fh:
            stored = json.load(fh)
        for key, value in (stored or {}).items():
            if isinstance(value, (int, float)) and value >= 1:
                _PHASE_REFLECTOR_LIMITS[str(key)] = int(value)
    except FileNotFoundError:
        pass
    except Exception:  # noqa: BLE001 — a corrupt file means "no limits", not a crash
        logger.warning("could not read %s; reflector limits start empty", path,
                       exc_info=True)


def _save_phase_limits_locked():
    """Write the limits out. Caller holds the lock. Best-effort by design:
    failing to persist a preference must never fail the indexing run that set
    it."""
    path = _limits_store_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with io.open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(_PHASE_REFLECTOR_LIMITS, fh, indent=2, sort_keys=True)
            fh.write("\n")
        tmp.replace(path)          # atomic: no half-written file after a crash
    except Exception:  # noqa: BLE001
        logger.debug("could not persist reflector limits to %s", path, exc_info=True)


def _phase_key(name_or_path):
    """Registry key: the file stem, lowercased.

    Callers hand in a CIF path; `create_indexer` only ever sees `phase.name`,
    which the CIF-based builders set to that same stem. Normalising both ends
    through here is what lets a limit set in the phase list be found again by a
    resolver that never heard of the path.
    """
    if not name_or_path:
        return ""
    text = str(name_or_path).replace("\\", "/").rsplit("/", 1)[-1]
    if text.lower().endswith(".cif"):
        text = text[:-4]
    return text.strip().lower()


def set_phase_reflector_limit(name_or_path, max_reflectors):
    """Remember (or forget, with None) one phase's reflector-family limit."""
    key = _phase_key(name_or_path)
    if not key:
        return
    with _PHASE_LIMITS_LOCK:
        _load_phase_limits_locked()
        if max_reflectors is None:
            _PHASE_REFLECTOR_LIMITS.pop(key, None)
        else:
            _PHASE_REFLECTOR_LIMITS[key] = max(1, int(max_reflectors))
        _save_phase_limits_locked()


def get_phase_reflector_limit(name_or_path):
    """The remembered limit for one phase, or None."""
    with _PHASE_LIMITS_LOCK:
        _load_phase_limits_locked()
        return _PHASE_REFLECTOR_LIMITS.get(_phase_key(name_or_path))


def phase_reflector_limits():
    """A copy of the whole registry (for the API and for tests)."""
    with _PHASE_LIMITS_LOCK:
        _load_phase_limits_locked()
        return dict(_PHASE_REFLECTOR_LIMITS)


def clear_phase_reflector_limits(persist: bool = False):
    """Forget every limit.

    `persist=False` by default so a test can start from a clean registry
    WITHOUT deleting the user's stored preferences.
    """
    global _PHASE_LIMITS_LOADED
    with _PHASE_LIMITS_LOCK:
        _PHASE_REFLECTOR_LIMITS.clear()
        # Cleared stays cleared: without this the next lookup would reload the
        # file and quietly bring back what was just cleared.
        _PHASE_LIMITS_LOADED = True
        if persist:
            _save_phase_limits_locked()


def _registered_limits_for(phase_list):
    """Per-phase limits from the registry, or None when none apply.

    Returns a list aligned with the phase list so it drops straight into
    `_trim_reflectors`, which already understands one entry per phase.
    """
    try:
        found = [get_phase_reflector_limit(getattr(p, "name", "")) for _pid, p in phase_list]
    except Exception:  # noqa: BLE001 — a lookup must never break a build
        return None
    return found if any(v is not None for v in found) else None



def _available_memory_bytes():
    """What this machine can actually hand out right now, conservatively.

    On Windows the binding limit is the COMMIT charge, not free RAM — measured
    2026-09-03 on the dev box: 23.4 GiB of physical memory free while only
    2.2 GiB of commit remained. Sizing against free RAM there would have
    green-lit an allocation that cannot succeed, so both are consulted and the
    smaller wins.
    """
    limits = []
    try:
        import psutil
        limits.append(int(psutil.virtual_memory().available))
    except Exception:  # noqa: BLE001 — a missing probe must not block indexing
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = _MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                limits.append(int(st.ullAvailPageFile))
        except Exception:  # noqa: BLE001
            pass
    return min(limits) if limits else (2 << 30)


def _triplet_library_budget_bytes():
    """How much scratch one band-triplet library may claim.

    Half of what is actually free, with a small floor — deliberately NOT a
    constant. A fixed ceiling is wrong in both directions: 2 GiB looked prudent
    and turned out to refuse beta-AlFeSi (2/m), which needs 3.55 GiB and had
    been building here for months; the same 2 GiB would be the entire machine on
    a small laptop. Tying the budget to free memory keeps the phases that fit
    working and refuses the ones that would take the machine down, on whatever
    machine is asking.

    The library is transient scratch for one indexer, so spending half of what
    is free on it is affordable; what is not affordable is asking for 44.7 GiB
    and letting Windows charge every page of it against the commit limit.
    """
    override = os.environ.get("ORIENTA_HOUGH_LIBRARY_BUDGET_MB")
    if override:
        try:
            # Escape hatch for two callers with a reason: a test that wants a
            # deterministic budget instead of whatever the machine happens to
            # have free, and a user who knows their machine better than this
            # heuristic does.
            return int(float(override) * (1 << 20))
        except ValueError:
            logger.warning("ORIENTA_HOUGH_LIBRARY_BUDGET_MB=%r is not a number; ignoring",
                           override)
    return int(max(256 << 20, _available_memory_bytes() * 0.5))


@contextmanager
def _capped_triplet_library(max_rows):
    """Make PyEBSDIndex REFUSE an oversized library instead of asking for it.

    `build_trip_lib` sizes `libANG`/`libID` by over-provisioning ("this
    completely over previsions the arrays", its own comment) and then calls
    `np.zeros`. On Linux those pages are lazy and the ask is survivable; on
    Windows every one of them is charged against the commit limit immediately,
    so a 44.7 GiB request either takes the machine into swap death or makes the
    NEXT allocation fail — which is how a user got "Context failed:
    OUT_OF_HOST_MEMORY" out of an OpenCL call that had nothing to do with it.

    The cap is installed on `tripletvote`'s own `np` reference rather than on
    numpy itself, so nothing outside that module changes, and a lock keeps two
    builds from swapping the shim out from under each other.
    """
    import pyebsdindex.tripletvote as _tv

    real_np = _tv.np

    class _CappedNumpy:
        """Numpy, except `zeros` refuses a library bigger than the budget."""

        def __getattr__(self, name):
            return getattr(real_np, name)

        def zeros(self, shape, *args, **kwargs):
            rows = shape[0] if isinstance(shape, tuple) and shape else None
            if isinstance(rows, (int, real_np.integer)) and rows > max_rows:
                raise MemoryError(
                    f"band-triplet library would need {int(rows):,} rows "
                    f"({int(rows) * _TRIPLET_ROW_BYTES / 2**30:.1f} GiB); "
                    f"budget is {max_rows * _TRIPLET_ROW_BYTES / 2**30:.1f} GiB"
                )
            return real_np.zeros(shape, *args, **kwargs)

    with _INDEXER_BUILD_LOCK:
        _tv.np = _CappedNumpy()
        try:
            yield
        finally:
            _tv.np = real_np


def _is_multiphase(ref_hkl):
    """True when `ref_hkl` is one hkl list PER PHASE rather than one phase's."""
    return bool(ref_hkl) and isinstance(ref_hkl[0], list)         and bool(ref_hkl[0]) and isinstance(ref_hkl[0][0], list)


def _trim_reflectors(ref_hkl, keep):
    """Cut each phase's reflector families down to `keep`.

    `keep` is either one number for every phase, or one per phase — the choice
    is PER PHASE because that is how the cost behaves: in the same run a cubic
    phase can carry all 64 of its families for a few MiB while a symmetry-less
    one would need 44.7 GiB for 70. Forcing them to share a limit would punish
    the cheap phase for the expensive one's CIF.

    `None` in a per-phase list means "no limit for this phase".
    """
    multi = _is_multiphase(ref_hkl)
    if isinstance(keep, (list, tuple)):
        if not multi:
            k = keep[0] if keep else None
            return ref_hkl[:k] if k else ref_hkl
        out = []
        for i, r in enumerate(ref_hkl):
            k = keep[i] if i < len(keep) else None
            out.append(r[:k] if k else r)
        return out
    if not keep:
        return ref_hkl
    return [r[:keep] for r in ref_hkl] if multi else ref_hkl[:keep]


def _normalise_max_reflectors(max_reflectors, ref_hkl, n_full):
    """User's choice -> what `_trim_reflectors` wants, clamped to what exists.

    Accepts one number for all phases, or one per phase (None = that phase
    keeps everything). Asking for more families than a phase HAS is not an
    error — it just means "all of them".
    """
    if max_reflectors is None:
        return n_full
    if isinstance(max_reflectors, (list, tuple)):
        counts = ([len(r) for r in ref_hkl] if _is_multiphase(ref_hkl)
                  else [len(ref_hkl)])
        return [
            None if (i >= len(max_reflectors) or not max_reflectors[i])
            else max(1, min(int(max_reflectors[i]), counts[i] if i < len(counts) else n_full))
            for i in range(len(counts))
        ]
    return max(1, min(int(max_reflectors), n_full)) if max_reflectors else n_full


def predict_triplet_library(detector, phase_list, reflectors, counts=None, nBands=12):
    """What the band-triplet library WOULD cost, per reflector-family count.

    Returns ``[(count, rows, bytes), ...]``, largest count first, without ever
    allocating one: the cap raises on the request, and the request carries the
    size. Feeds the message below and, through it, the user's choice.
    """
    if isinstance(reflectors, list):
        ref_hkl = [r.hkl.tolist() for r in reflectors]
        n_full = max((len(r) for r in ref_hkl), default=0)
    else:
        ref_hkl = reflectors.hkl.tolist()
        n_full = len(ref_hkl)
    if counts is None:
        counts = (n_full, *(k for k in _REFLECTOR_LADDER if k < n_full))
    fixed = _standard_setting_phase_list(phase_list)
    out = []
    for keep in counts:
        rows = None
        try:
            # Probe cap, not 1 row: `build_trip_lib` also allocates small
            # bookkeeping arrays (pole pairs, family indices) and a cap of 1
            # would report THEIR size instead of the library's. A million rows
            # is 48 MiB — above every one of those and far below any library
            # worth warning about.
            with _capped_triplet_library(_PREDICT_PROBE_ROWS):
                detector.get_indexer(fixed, _trim_reflectors(ref_hkl, keep), nBands=nBands)
            rows = 0                      # built under the probe: negligible
        except MemoryError as e:
            m = re.search(r"need ([0-9,]+) rows", str(e))
            if m:
                rows = int(m.group(1).replace(",", ""))
        except Exception:                 # noqa: BLE001 — sizing must not raise
            continue
        if rows is not None:
            out.append((int(keep), rows, rows * _TRIPLET_ROW_BYTES))
    return out


def create_indexer(detector, phase_list, reflectors, nBands=12, tSigma=2, rSigma=2,
                   max_reflectors=None):
    """
    Builds a Hough indexer with the given parameters.

    reflectors can be a single ReciprocalLatticeVector or a list of them
    (one per phase for multi-phase indexing).

    `max_reflectors` limits how many reflector families each phase contributes.
    It is the USER's control, not a default we apply behind their back, because
    the choice is not free: measured on Ni (m-3m, 58 families, 800 real
    patterns) 40 and 32 families are bit-identical to the full set (0.000 deg
    median deviation, the same 495/800 indexed), while 24 collapses to 192/800
    at 119.7 deg and 16 returns a median fit of 180 deg — and it does that
    QUIETLY, handing back orientations that look like data. So the count is
    offered, with what each one costs, and never chosen for you.

    What IS automatic is the refusal: the band-triplet library is sized from the
    family count and grows as O(npoles * nangs**3), so a symmetry-less CIF can
    ask for 44.7 GiB. That request is not attempted on a machine that cannot
    afford it (see `_triplet_library_budget_bytes`) — on Windows it is charged
    against the commit limit in full, which is how one such build made an
    unrelated OpenCL call fail with OUT_OF_HOST_MEMORY and, on a weaker
    machine, would simply have taken it down.
    """
    if isinstance(reflectors, list):
        ref_hkl = [r.hkl.tolist() for r in reflectors]
        n_full = max((len(r) for r in ref_hkl), default=0)
    else:
        ref_hkl = reflectors.hkl.tolist()
        n_full = len(ref_hkl)

    # An explicit argument wins; otherwise the phase's registered limit applies,
    # which is how the eight builders that know nothing about the indexing
    # request still honour what the user set in the phase list.
    limits = max_reflectors if max_reflectors is not None else _registered_limits_for(phase_list)
    keep = _normalise_max_reflectors(limits, ref_hkl, n_full)
    budget = _triplet_library_budget_bytes()
    fixed_phases = _standard_setting_phase_list(phase_list)

    try:
        with _capped_triplet_library(max(1, budget // _TRIPLET_ROW_BYTES)):
            return detector.get_indexer(
                fixed_phases, _trim_reflectors(ref_hkl, keep),
                nBands=nBands, tSigma=tSigma, rSigma=rSigma
            )
    except MemoryError as e:
        names = ", ".join(
            f"{p.name} ({p.point_group.name if p.point_group else 'unknown symmetry'})"
            for _, p in phase_list
        )
        # What WOULD fit, so the message ends in a decision the user can take
        # rather than in a dead end.
        table = []
        try:
            for cnt, _rows, nbytes in predict_triplet_library(
                    detector, phase_list, reflectors, nBands=nBands):
                if cnt > keep:
                    continue
                table.append(
                    f"{cnt} -> {nbytes / 2**30:.2f} GiB" if nbytes
                    else f"{cnt} -> under {_PREDICT_PROBE_ROWS * _TRIPLET_ROW_BYTES / 2**20:.0f} MiB"
                )
        except Exception:  # noqa: BLE001 — the advice is a bonus, not the error
            pass
        advice = ("  Reflector families vs. memory: " + ", ".join(table) + "."
                  if table else "")
        raise MemoryError(
            f"Hough indexing needs more memory than this machine can spare for "
            f"{names}: {e}. About {_available_memory_bytes() / 2**30:.1f} GiB is free "
            f"right now.{advice} Lower the reflector count for this phase, free memory, "
            f"or index it with Dictionary or Spherical indexing instead. "
            f"(A CIF stored without symmetry — space group P 1 — is the usual cause: "
            f"every reflector family is then a single pole.)"
        ) from e


def log_index_data(xmap, index_data, band_data):
    """
    Consolidated debug logging for xmap, index_data and band_data.
    """
    logger.debug("=== xmap ===")
    logger.debug(" type: %s", type(xmap))
    if hasattr(xmap, "phases"):
        logger.debug(" phases: %s", xmap.phases)
    if hasattr(xmap, "rotations"):
        try:
            data = getattr(xmap.rotations, "data", None)
            logger.debug(" rotations.data.shape: %s",
                         data.shape if data is not None else "(no .data attribute)")
        except Exception as e:
            logger.debug(" error reading rotations: %s", e)
    else:
        logger.debug(" no .rotations attribute on xmap")

    logger.debug("=== index_data ===")
    logger.debug(" type: %s", type(index_data))
    if hasattr(index_data, "dtype"):
        logger.debug(" fields: %s", index_data.dtype.names)
    shp = getattr(index_data, "shape", None)
    if shp is not None:
        logger.debug(" shape: %s", shp)
    else:
        try:
            logger.debug(" length: %s", len(index_data))
        except Exception:
            logger.debug(" cannot get length")

    logger.debug("=== band_data ===")
    logger.debug(" type: %s", type(band_data))
    try:
        n = len(band_data)
        logger.debug(" number of band_data items: %d", n)
        for i, bd in enumerate(band_data[:min(n, 3)]):
            logger.debug("  band_data[%d].shape: %s", i, getattr(bd, "shape", bd))
    except Exception as e:
        logger.debug(" error reading band_data: %s", e)


def optimize_pc(detector, indexer, pattern, method="PSO", search_limit=0.05, batch=True):
    """
    Optimizes the Pattern Center via PSO using kikuchipy's hough_indexing_optimize_pc.
    Returns the mean of the resulting PC and the result object.
    """
    # pattern is 2D (H, W); hough_indexing_optimize_pc requires at least one
    # navigation axis — shape (1, 1, H, W) satisfies this requirement.
    ebsd = EBSD(pattern[np.newaxis, np.newaxis], detector=detector)

    # search_limit is only supported by PSO (not Nelder-Mead)
    extra_kwargs = {}
    if method.upper() == "PSO":
        extra_kwargs["search_limit"] = search_limit

    det = ebsd.hough_indexing_optimize_pc(
        pc0=list(detector.pc),
        indexer=indexer,
        method=method,
        batch=batch,
        **extra_kwargs
    )

    # Mean over all PC proposals with validation
    pc_flat = getattr(det, 'pc_flattened', None)
    if pc_flat is None or pc_flat.size == 0:
        logger.warning("PC optimization returned empty result, using initial PC")
        mean_pc = np.array(list(detector.pc), dtype=float)
    else:
        mean_pc = np.mean(pc_flat, axis=0)
    return tuple(mean_pc), det


def compute_ci(detector, phase_list, indexer, pattern):
    """
    Performs a simple Hough indexing and returns the CI value.
    """
    if indexer is None:
        from ebsd_utils import create_indexer, prepare_reflectors
        reflectors = prepare_reflectors(phase_list)
        indexer = create_indexer(detector, phase_list, reflectors)
    
    # pattern is 2D (H, W); hough_indexing requires at least one navigation axis.
    ebsd = EBSD(pattern[np.newaxis, np.newaxis], detector=detector)
    # Run indexing with the configured indexer
    xmap, index_data, band_data = ebsd.hough_indexing(
        phase_list, indexer,
        return_index_data=True,
        return_band_data=True,
        verbose=0
    )
    # CI is stored in index_data['cm']
    try:
        ci = float(index_data['cm'].mean())
    except Exception:
        ci = float(index_data['cm'][0, 0])
    return ci

def validate_pc_array(pc_array):
    """
    Validate an array of PC values.

    Args:
        pc_array: Array of shape (N, 3) with PC values

    Returns:
        Tuple (is_valid, message) — True if all PCs are in reasonable range
    """
    arr = np.asarray(pc_array)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return False, f"PC array shape {arr.shape}, expected (N, 3)"

    # Reasonable PC ranges (typical for EBSD)
    pcx, pcy, pcz = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(pcx < -0.5) or np.any(pcx > 1.5):
        return False, f"PCx out of range: [{pcx.min():.3f}, {pcx.max():.3f}]"
    if np.any(pcy < -0.5) or np.any(pcy > 1.5):
        return False, f"PCy out of range: [{pcy.min():.3f}, {pcy.max():.3f}]"
    if np.any(pcz < 0.1) or np.any(pcz > 2.0):
        return False, f"PCz out of range: [{pcz.min():.3f}, {pcz.max():.3f}]"

    return True, "OK"


def pc_variation_stats(pc_array):
    """
    Compute statistics of PC variation across calibration points.

    Args:
        pc_array: Array of shape (N, 3) with PC values

    Returns:
        Dict with mean, std, range for each PC component
    """
    arr = np.asarray(pc_array)
    labels = ['pcx', 'pcy', 'pcz']
    stats = {}
    for i, label in enumerate(labels):
        vals = arr[:, i]
        stats[label] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
            'range': float(np.ptp(vals)),
        }
    return stats


# Track sanitize_cif temp files so they can be cleaned at process exit.
# Without this, each call to sanitize_cif on a pymatgen CIF leaked one
# tempfile (~2 KB) into %TEMP%; a long batch indexing run could accumulate
# thousands of stale .cif files over weeks of use.
_SANITIZED_CIF_TMPS: list = []


def _cleanup_sanitized_cifs():
    """Remove any tempfiles created by sanitize_cif. Registered via atexit."""
    import os as _os
    for p in _SANITIZED_CIF_TMPS:
        try:
            if _os.path.exists(p):
                _os.remove(p)
        except OSError:
            pass
    _SANITIZED_CIF_TMPS.clear()


import atexit as _atexit
_atexit.register(_cleanup_sanitized_cifs)


def sanitize_cif(path):
    """
    Fix pymatgen-generated CIF files that have content before the first data_ block.
    Returns path to a sanitized temp file, or the original path if already valid.

    Temp files created here are tracked in _SANITIZED_CIF_TMPS and removed
    at process exit via atexit, so a long-running backend doesn't leak
    files into %TEMP% over weeks.
    """
    import re
    import tempfile

    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    # Already valid: starts with optional comments then data_
    if re.match(r'\s*(#.*\n)*\s*data_', content):
        return path
    # Find first data_ line and keep only from there
    match = re.search(r'^(data_)', content, re.MULTILINE)
    if match:
        sanitized = content[match.start():]
        tmp = tempfile.NamedTemporaryFile(suffix='.cif', delete=False, mode='w', encoding='utf-8')
        tmp.write(sanitized)
        tmp.close()
        _SANITIZED_CIF_TMPS.append(tmp.name)
        logger.debug("Sanitized CIF %s (stripped %d chars before data_ block)", path, match.start())
        return tmp.name
    return path


def validate_pattern(pattern):
    """
    Validate a pattern is a 2D array. Returns (True, array) or (False, reason).
    """
    try:
        arr = np.asarray(pattern)
        if arr.ndim != 2:
            return False, f"Pattern has {arr.ndim} dimensions, must be 2D."
        return True, arr
    except Exception as e:
        return False, f"Pattern is not a valid array: {e}"
