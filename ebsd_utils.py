import logging
import re
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


def create_indexer(detector, phase_list, reflectors, nBands=12, tSigma=2, rSigma=2):
    """
    Builds a Hough indexer with the given parameters.

    reflectors can be a single ReciprocalLatticeVector or a list of them
    (one per phase for multi-phase indexing).
    """
    if isinstance(reflectors, list):
        ref_hkl = [r.hkl.tolist() for r in reflectors]
    else:
        ref_hkl = reflectors.hkl.tolist()
    try:
        return detector.get_indexer(
            _standard_setting_phase_list(phase_list), ref_hkl,
            nBands=nBands, tSigma=tSigma, rSigma=rSigma
        )
    except MemoryError as e:
        # PyEBSDIndex over-provisions the band-triplet library from the reflector
        # count, and that grows steeply as symmetry drops. Name the phases so the
        # user knows which one to trim instead of reading a raw numpy traceback.
        names = ", ".join(
            f"{p.name} ({p.point_group.name if p.point_group else 'unknown symmetry'})"
            for _, p in phase_list
        )
        raise MemoryError(
            f"Hough indexing could not build the band-triplet library for {names}: {e}. "
            f"PyEBSDIndex sizes that library from the number of reflector families, which "
            f"grows steeply for low-symmetry phases — lower max_reflectors for this phase, "
            f"or index it with Dictionary or Spherical indexing instead."
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
