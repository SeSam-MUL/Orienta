"""Simulation-parity harness — pattern-vs-pattern NCC + indexing-parity.

This is the *simulation-parity* counterpart to the master-level NCC gate in
:mod:`backend.forward_sim.validate.ncc_vs_emsoft`.  Where that module compares the
**master pattern** of OUR dynamical forward model against the EMsoft oracle master
(a Lambert-square-vs-Lambert-square NCC), this module compares the **projected
detector patterns** — the quantity an EBSD user actually indexes against.

It answers the user's direct question:

    "Do EBSD patterns simulated from OUR master match patterns simulated from the
     EMsoft master, pattern-to-pattern, in the *old vs new* workflow?"

Why a separate harness
----------------------
The master NCC (0.74–0.76 at npx=50, iter-8) is a *worst-case* number: it scores
every Lambert-square pixel, including the high-spatial-frequency band edges that
our coarse npx=50 grid aliases.  A projected EBSD pattern only samples a
**70°-tilted detector cone** of the master (all pixel directions land on the upper
hemisphere — verified), and the detector itself low-pass-filters via the bilinear
``grid_sample``.  So the *pattern* NCC is the production-relevant metric and is
expected to be **higher** than the master NCC.

Identical projector — the only difference is the master content
---------------------------------------------------------------
Both the OUR-master and the EMsoft-master patterns run through the **same three
functions** in ``backend/dictionary_gpu/`` — ``detector_pixel_directions`` (PC +
tilt → directions), ``rotate_directions_by_quaternions`` (quats × directions), and
``sample_master_at_directions`` (``direction_to_lambert`` + ``F.grid_sample``,
``align_corners=True``, ``padding_mode='border'``).  The *only* difference between
the two projected pattern batches is the content of the ``hemispheres`` tensor
fed to ``project_patterns``.  That is the correct isolation: the NCC measures the
master difference, not a projector difference.

OUR master is wrapped directly (design option (i)): the saved ``(2*npx+1)²``
northern-hemisphere array is stacked into a ``(2, M, M)`` hemispheres tensor (south
= copy of north).  The southern copy is sound because (a) ``direction_to_lambert``
already flips ``z<0`` directions onto the upper square before sampling, (b) the
EMsoft ``mLPNH`` is upper/lower symmetric to ~0.999 for cubic phases, and (c) the
70°-tilted detector geometry puts **all** pixel directions on the upper hemisphere
regardless.

Two metrics (per orientation, then summarised)
----------------------------------------------
METRIC 1 — per-orientation pattern NCC (masked to the detector disc), using the
same zero-mean NCC as :func:`ncc_vs_emsoft.ncc_zero_mean`.  Reported: mean, median,
5th-percentile tail.  Gate: median ≥ 0.95 (SP0+SP1 target).

METRIC 2 — indexing parity.  We build a self-consistent dictionary from the EMsoft
patterns (the reference), then for each ground-truth orientation ``i`` index its
OUR-master pattern against the EMsoft dictionary by NCC-argmax; the matched
dictionary orientation is compared to ``q_i`` via the Oh-symmetry-reduced
disorientation.  Parity = fraction with disorientation < 2°.  Gate: ≥ 0.90.

CLI
---
``python -m backend.forward_sim.validate.pattern_parity Ni --npx 50``
``python -m backend.forward_sim.validate.pattern_parity Al --npx 50 --n-orient 100``
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch

from ..runtime import ForwardSimError, get_device
from .ncc_vs_emsoft import _ORACLE, _project_root, ncc_zero_mean

# Default fixed detector geometry (design §detector).
_DET_SHAPE = (80, 80)
_DET_PC = (0.5, 0.5, 0.5)
_DET_TILT = 70.0
_DET_DETECTOR_TILT = 0.0
_DET_AZIMUTHAL = 0.0

_PARITY_DEG = 2.0  # disorientation gate for METRIC 2

_ITER_DIR = Path("tasks/forward_sim/iterations/11")


# ---------------------------------------------------------------------------
# Saved OUR-master lookup (iter-8 npx50 arrays, for speed)
# ---------------------------------------------------------------------------
def _saved_our_master(phase: str, npx: int) -> Optional[Path]:
    """Path to a saved OUR-master ``.npy`` if present (iter-8 at npx50), else None.

    The iter-8 arrays live under ``tasks/forward_sim/iterations/8/`` and are the
    current best master (absflg=3, dmin=0.10).  Reusing them avoids a ~500 s
    per-phase rebuild when the requested ``npx`` matches.
    """
    p = _project_root() / "tasks" / "forward_sim" / "iterations" / "8" / f"our_{phase}_npx{npx}.npy"
    return p if p.is_file() else None


def _build_our_master_array(
    phase: str,
    *,
    npx: int,
    energy_kV: float,
    dmin: float,
    absflg: int,
    device: torch.device,
) -> np.ndarray:
    """Build OUR dynamical master for ``phase`` at ``npx`` → ``(2*npx+1, 2*npx+1)``.

    Imports the heavy dynamical stack lazily so the harness module is cheap to
    import (and the unit test can monkey-free a tiny build).  Returns a float32
    numpy array (northern-hemisphere Lambert square).
    """
    from ..crystal.structure_matrix import read_bethe_parameters
    from ..crystal.xtal_io import read_crystal_structure
    from ..dynamical.master_builder import build_master
    from ..mc.emsoft_mc_input import load_mc

    oracle = _project_root() / _ORACLE[phase]
    structure = read_crystal_structure(str(oracle))
    mc = load_mc(str(oracle))
    bethe = read_bethe_parameters(str(oracle))
    ekevs = mc.EkeVs.detach().cpu().numpy().ravel()
    e_idx = int(np.argmin(np.abs(ekevs - energy_kV)))

    ours_t = build_master(
        structure,
        mc,
        npx=npx,
        energy_idx=e_idx,
        dmin=dmin,
        device=device,
        bethe_params=bethe,
        absflg=absflg,
    )
    return ours_t.detach().cpu().numpy().astype(np.float32)


def _our_hemispheres(
    phase: str,
    *,
    npx: int,
    energy_kV: float,
    dmin: float,
    absflg: int,
    device: torch.device,
    allow_saved: bool = True,
) -> Tuple[torch.Tensor, str]:
    """Return OUR (2, M, M) hemispheres tensor + a provenance string.

    Prefers the saved iter-8 ``.npy`` (when ``allow_saved`` and the npx matches);
    otherwise rebuilds via the dynamical core.  South hemisphere = copy of north
    (design option (i)).
    """
    source = "rebuilt"
    arr: Optional[np.ndarray] = None
    if allow_saved:
        saved = _saved_our_master(phase, npx)
        if saved is not None:
            arr = np.load(str(saved)).astype(np.float32)
            source = f"saved:{saved.name}"
    if arr is None:
        arr = _build_our_master_array(
            phase, npx=npx, energy_kV=energy_kV, dmin=dmin, absflg=absflg, device=device
        )
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"OUR master must be square 2D, got {arr.shape}")
    hemi_np = np.stack([arr, arr], axis=0)  # (2, M, M), south = north copy
    hemi = torch.from_numpy(hemi_np).to(device=device, dtype=torch.float32)
    return hemi, source


# ---------------------------------------------------------------------------
# Orientation sampling
# ---------------------------------------------------------------------------
def _sample_orientations(n_orient: int, *, resolution: float = 7.5, seed: int = 0) -> np.ndarray:
    """Sample ``n_orient`` orientations from the cubic (Oh) fundamental zone.

    Uses orix ``get_sample_fundamental(resolution, point_group=Oh)`` (≈2000 points
    at 7.5°) and sub-samples a deterministic ``n_orient`` subset.  Returns
    ``(n_orient, 4)`` float32 quaternions ``(w, x, y, z)`` in the orix/kikuchipy
    convention (identity = ``[1, 0, 0, 0]``).
    """
    from orix.quaternion.symmetry import Oh
    from orix.sampling import get_sample_fundamental

    rot = get_sample_fundamental(resolution=resolution, point_group=Oh)
    q = np.asarray(rot.data, dtype=np.float32)  # (N, 4)
    N = q.shape[0]
    if n_orient >= N:
        return q
    # Deterministic even-stride sub-sample for reproducible FZ coverage.
    idx = np.linspace(0, N - 1, n_orient).round().astype(int)
    idx = np.unique(idx)
    if idx.size < n_orient:  # round-collisions: top up with an RNG draw
        rng = np.random.default_rng(seed)
        extra = rng.choice(np.setdiff1d(np.arange(N), idx), n_orient - idx.size, replace=False)
        idx = np.sort(np.concatenate([idx, extra]))
    return q[idx]


def _disorientation_deg_batch(q_gt: np.ndarray, q_hat: np.ndarray) -> np.ndarray:
    """Oh-symmetry-reduced disorientation (deg) between matched orientation pairs.

    ``q_gt``/``q_hat``: ``(N, 4)`` quaternion arrays.  Returns ``(N,)`` degrees.
    """
    from orix.quaternion import Misorientation, Rotation
    from orix.quaternion.symmetry import Oh

    ra = Rotation(np.asarray(q_gt, dtype=float))
    rb = Rotation(np.asarray(q_hat, dtype=float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mo = Misorientation((~ra) * rb, symmetry=(Oh, Oh))
        mo = mo.reduce()
        return np.rad2deg(np.asarray(mo.angle).ravel())


# ---------------------------------------------------------------------------
# NCC-argmax indexing (self-contained, no kikuchipy dictionary step)
# ---------------------------------------------------------------------------
def _masked_zscore(patterns: torch.Tensor, mask_flat: torch.Tensor) -> torch.Tensor:
    """Zero-mean / unit-L2-normalise each pattern over the disc, masked elsewhere=0.

    ``patterns``: ``(N, P)`` flattened patterns.  ``mask_flat``: ``(P,)`` bool.
    Returns ``(N, P)`` where, over the mask, each row is mean-subtracted and
    L2-normalised so that ``A @ B.T`` yields the zero-mean NCC matrix directly;
    out-of-mask entries are exactly 0 so they contribute nothing to the dot.
    """
    m = mask_flat.to(patterns.dtype)
    k = m.sum().clamp(min=1.0)
    mean = (patterns * m).sum(dim=1, keepdim=True) / k
    centred = (patterns - mean) * m
    norm = centred.norm(dim=1, keepdim=True).clamp(min=1e-30)
    return centred / norm


def pattern_parity(
    phase: str,
    *,
    npx: int = 50,
    dmin: float = 0.10,
    absflg: int = 1,
    n_orient: int = 100,
    energy_kV: float = 20.0,
    detector_shape: Tuple[int, int] = _DET_SHAPE,
    pc: Tuple[float, float, float] = _DET_PC,
    sample_tilt_deg: float = _DET_TILT,
    detector_tilt_deg: float = _DET_DETECTOR_TILT,
    azimuthal_deg: float = _DET_AZIMUTHAL,
    resolution_deg: float = 7.5,
    device: torch.device | str | None = None,
    allow_saved: bool = True,
    save_images: bool = True,
    n_example_pngs: int = 3,
) -> dict:
    """Measure pattern-vs-pattern NCC + indexing parity for ``phase`` (Ni / Al).

    Projects ``n_orient`` EBSD detector patterns from BOTH the EMsoft master and
    OUR master at the **same orientations and geometry, through the same
    projector**, then:

      * METRIC 1 — per-orientation masked zero-mean pattern NCC (OUR vs EMsoft).
      * METRIC 2 — indexing parity: index each OUR pattern against the EMsoft
        dictionary (NCC-argmax) and count matches within ``2°`` (Oh-reduced
        disorientation) of the ground-truth orientation.

    Args:
        phase: ``"Ni"`` or ``"Al"`` (selects the oracle ``.h5`` + OUR master).
        npx: Lambert half-grid size of OUR master (grid ``2*npx+1``).  At ``50``
            the saved iter-8 ``.npy`` is reused when ``allow_saved``.
        dmin / absflg: forwarded to :func:`build_master` only when rebuilding
            (ignored when a saved master is used).
        n_orient: number of FZ orientations to project + score.
        energy_kV: accelerating voltage (closest ``EkeVs`` bin used).
        detector_shape / pc / sample_tilt_deg / detector_tilt_deg / azimuthal_deg:
            fixed detector geometry (design defaults).
        resolution_deg: FZ sampling resolution for the orientation grid.
        device: compute device; defaults to :func:`get_device` (honours
            ``FORWARD_SIM_DEVICE``).
        allow_saved: reuse the saved iter-8 master ``.npy`` when present.
        save_images: write ``n_example_pngs`` side-by-side our/EMsoft/absdiff PNGs.
        n_example_pngs: how many example orientations to render.

    Returns:
        dict with ``phase``, ``npx``, ``n_orient``, ``device``, ``our_source``,
        ``pattern_ncc`` (list), ``pattern_ncc_mean/median/p05``,
        ``indexing_parity``, ``disorientation_deg`` (list),
        ``self_consistency`` (EMsoft-vs-EMsoft argmax identity fraction),
        ``wall_s``, and (when saved) ``example_pngs``.

    Raises:
        ValueError: unknown phase or missing oracle file.
        ForwardSimError: no CUDA and ``FORWARD_SIM_DEVICE`` unset.
    """
    from backend.dictionary_gpu.detector import detector_pixel_directions
    from backend.dictionary_gpu.master_loader import load_master_pattern
    from backend.dictionary_gpu.projector import project_patterns
    from tools.pattern_comparison import circular_mask

    if phase not in _ORACLE:
        raise ValueError(f"unknown phase {phase!r}; known: {sorted(_ORACLE)}")
    oracle = _project_root() / _ORACLE[phase]
    if not oracle.is_file():
        raise ValueError(f"oracle file not found for {phase!r}: {oracle}")

    dev = torch.device(device) if device is not None else get_device()
    t0 = time.time()

    # --- Orientations (ground truth) -------------------------------------------
    quats_np = _sample_orientations(n_orient, resolution=resolution_deg)
    n = quats_np.shape[0]
    quats = torch.from_numpy(quats_np).to(device=dev, dtype=torch.float32)

    # --- Detector pixel directions (shared by both masters) --------------------
    pix = detector_pixel_directions(
        detector_shape,
        pc=pc,
        tilt_deg=sample_tilt_deg,
        detector_tilt_deg=detector_tilt_deg,
        azimuthal_deg=azimuthal_deg,
        device=str(dev),
    )

    # --- EMsoft master → patterns ----------------------------------------------
    em = load_master_pattern(str(oracle), energy_kv=energy_kV, device=str(dev))
    dict_em = project_patterns(quats, pix, em.hemispheres, detector_shape)  # (N, H, W)

    # --- OUR master → patterns (same projector, same orientations) -------------
    our_hemi, our_source = _our_hemispheres(
        phase,
        npx=npx,
        energy_kV=energy_kV,
        dmin=dmin,
        absflg=absflg,
        device=dev,
        allow_saved=allow_saved,
    )
    dict_our = project_patterns(quats, pix, our_hemi, detector_shape)  # (N, H, W)

    H, W = detector_shape
    mask = circular_mask(detector_shape, radius_frac=1.0)  # (H, W) bool
    mask_flat = torch.from_numpy(mask.ravel()).to(dev)

    em_flat = dict_em.reshape(n, H * W)
    our_flat = dict_our.reshape(n, H * W)

    # --- METRIC 1: per-orientation masked zero-mean pattern NCC ----------------
    em_z = _masked_zscore(em_flat, mask_flat)    # (N, P), rows unit-L2 over disc
    our_z = _masked_zscore(our_flat, mask_flat)
    pattern_ncc_t = (em_z * our_z).sum(dim=1)    # (N,) diagonal NCC
    pattern_ncc = pattern_ncc_t.detach().cpu().numpy().astype(np.float64)

    # --- METRIC 2: indexing parity (NCC-argmax vs EMsoft dictionary) -----------
    # Self-consistency: EMsoft pattern i argmax against EMsoft dict must be i.
    sim_em = em_z @ em_z.T                        # (N, N) NCC matrix
    self_argmax = sim_em.argmax(dim=1)
    self_consistency = float(
        (self_argmax == torch.arange(n, device=dev)).float().mean().item()
    )
    # OUR pattern i argmax against EMsoft dict → matched ground-truth orientation.
    sim_cross = our_z @ em_z.T                    # (N, N)
    matched_idx = sim_cross.argmax(dim=1).detach().cpu().numpy()
    q_hat = quats_np[matched_idx]                 # matched orientation per pixel
    diso = _disorientation_deg_batch(quats_np, q_hat)
    indexing_parity = float(np.mean(diso < _PARITY_DEG))

    wall_s = time.time() - t0

    result = {
        "phase": phase,
        "npx": npx,
        "n_orient": n,
        "absflg": int(absflg),
        "dmin": dmin,
        "energy_kV": energy_kV,
        "device": str(dev),
        "our_source": our_source,
        "detector_shape": list(detector_shape),
        "pc": list(pc),
        "sample_tilt_deg": sample_tilt_deg,
        "pattern_ncc": pattern_ncc.tolist(),
        "pattern_ncc_mean": float(np.mean(pattern_ncc)),
        "pattern_ncc_median": float(np.median(pattern_ncc)),
        "pattern_ncc_p05": float(np.percentile(pattern_ncc, 5)),
        "pattern_ncc_min": float(np.min(pattern_ncc)),
        "indexing_parity": indexing_parity,
        "self_consistency": self_consistency,
        "disorientation_deg": diso.tolist(),
        "disorientation_median_deg": float(np.median(diso)),
        "wall_s": wall_s,
    }

    if save_images:
        pngs = _save_example_pngs(
            phase, npx, dict_em, dict_our, mask, n_example_pngs
        )
        result["example_pngs"] = pngs

    return result


# ---------------------------------------------------------------------------
# FINE indexing-parity — sub-2° angular precision (not just coarse basin recovery)
# ---------------------------------------------------------------------------
def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of quaternion arrays ``a`` ⊗ ``b`` (``(w, x, y, z)``).

    Broadcasting: ``a`` ``(..., 4)`` × ``b`` ``(..., 4)`` → ``(..., 4)``.  Used to
    compose a fixed local-offset cloud onto each base orientation without a
    per-base orix call (orix ``get_sample_local`` has a large fixed per-call
    overhead, so 50 calls is pathologically slow — we call it ONCE around
    identity and compose).
    """
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def _fine_local_dictionary(
    bases: np.ndarray,
    *,
    resolution_deg: float,
    grid_width_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a dense local-grid dictionary around each base orientation.

    A single local cubochoric grid of width ``grid_width_deg`` at spacing
    ``resolution_deg`` (orix :func:`get_sample_local` around the **identity**) is
    composed onto every base via fast quaternion multiplication —
    ``dict_b = q_base ⊗ offset`` — so each base is surrounded by the same local
    cloud.  (Calling ``get_sample_local`` once per base is correct but
    pathologically slow: ~minutes for 50 bases due to orix's per-call overhead.
    Composing a shared offset cloud is mathematically equivalent — a rigid
    left-rotation of the local grid — and runs in milliseconds.)

    The ``resolution_deg`` is the *angular sampling pitch* of the dictionary —
    the quantity that bounds how close the nearest dictionary entry can be to an
    arbitrary query.  A pitch well below the 2° parity gate (default 1.0°,
    grid_width 3°) guarantees that a 0.5–1.5° perturbation of a base has a
    dictionary neighbour within ~½·pitch, so a correctly-wired indexer can
    recover it to <2°; a *miswired* projector cannot.

    Args:
        bases: ``(B, 4)`` base quaternions ``(w, x, y, z)``.
        resolution_deg: local-grid sampling pitch (the dictionary resolution).
        grid_width_deg: half-extent of the local grid around each base.

    Returns:
        ``(dict_quats, dict_base_idx)`` — ``dict_quats`` is ``(D, 4)`` float32,
        ``dict_base_idx`` is ``(D,)`` int (which base each entry surrounds; for
        diagnostics only).
    """
    from orix.quaternion import Rotation
    from orix.sampling import get_sample_local

    # ONE local grid around the identity (the shared offset cloud).
    local = get_sample_local(
        resolution=float(resolution_deg),
        center=Rotation.identity(),
        grid_width=float(grid_width_deg),
    )
    offset = np.asarray(local.data, dtype=np.float64)  # (g, 4)
    g = offset.shape[0]
    B = bases.shape[0]

    # dict[b, k] = q_base[b] ⊗ offset[k]   → (B, g, 4)
    base_b = bases.astype(np.float64)[:, None, :]          # (B, 1, 4)
    off_b = offset[None, :, :]                              # (1, g, 4)
    dict_q = _quat_mul(base_b, off_b).reshape(B * g, 4)     # (B*g, 4)
    # Normalise (guard float drift) + canonical sign (w >= 0).
    dict_q /= np.linalg.norm(dict_q, axis=1, keepdims=True)
    dict_q[dict_q[:, 0] < 0] *= -1.0
    base_idx = np.repeat(np.arange(B), g)
    return dict_q.astype(np.float32), base_idx.astype(int)


def _perturb_orientations(
    bases: np.ndarray, *, min_deg: float, max_deg: float, seed: int
) -> np.ndarray:
    """Perturb each base by a random small rotation in ``[min_deg, max_deg]``.

    A uniformly-random axis and a uniform angle in the band are composed onto each
    base (``q_query = q_base * q_pert``), so the query sits a known small angle
    away from its base — the precision the fine dictionary must resolve.

    Returns ``(B, 4)`` float32 query quaternions.
    """
    from orix.quaternion import Rotation

    rng = np.random.default_rng(seed)
    B = bases.shape[0]
    axes = rng.standard_normal((B, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angles = np.deg2rad(rng.uniform(min_deg, max_deg, size=B))
    pert = Rotation.from_axes_angles(axes, angles)
    q_query = (Rotation(bases.astype(float)) * pert).data.astype(np.float32)
    return q_query


def fine_pattern_parity(
    phase: str,
    *,
    npx: int = 50,
    dmin: float = 0.10,
    absflg: int = 1,
    n_base: int = 50,
    perturb_min_deg: float = 0.5,
    perturb_max_deg: float = 1.5,
    fine_resolution_deg: float = 1.0,
    grid_width_deg: float = 3.0,
    parity_deg: float = _PARITY_DEG,
    energy_kV: float = 20.0,
    base_resolution_deg: float = 10.0,
    detector_shape: Tuple[int, int] = _DET_SHAPE,
    pc: Tuple[float, float, float] = _DET_PC,
    sample_tilt_deg: float = _DET_TILT,
    detector_tilt_deg: float = _DET_DETECTOR_TILT,
    azimuthal_deg: float = _DET_AZIMUTHAL,
    device: torch.device | str | None = None,
    allow_saved: bool = True,
    seed: int = 0,
) -> dict:
    """FINE indexing parity: sub-2° angular precision, OUR pattern vs EMsoft dict.

    Where :func:`pattern_parity`'s METRIC 2 tests *coarse basin recovery* (the
    query orientations ARE the dictionary, so a correct indexer trivially returns
    0° disorientation), this probes the harder question the SP1 closeout asks:

        Projected from OUR master and indexed against a **fine** dictionary built
        from the EMsoft master, is a small (``0.5–1.5°``) orientation
        perturbation recovered to **better than 2°**?

    Procedure
    ---------
    1. Sample ``n_base`` base orientations from the cubic FZ (deterministic).
    2. Perturb each base by a random ``[perturb_min_deg, perturb_max_deg]``
       rotation → the query orientations (one per base).
    3. Build a dense local dictionary around the bases via orix
       ``get_sample_local(resolution=fine_resolution_deg, grid_width=…)`` —
       sampling pitch ``fine_resolution_deg`` (≪ 2°), so a near neighbour of each
       query exists.
    4. Project the queries from **OUR** master and the dictionary from the
       **EMsoft** master, through the **same projector**.
    5. NCC-argmax each query against the dictionary; compare the matched
       dictionary orientation to the *true query* orientation by Oh-reduced
       disorientation.  Parity = fraction ``< parity_deg``.

    A perfect-but-discretised oracle (dictionary built from the *EMsoft* master,
    queries also from the *EMsoft* master) is also indexed → ``self_parity``,
    the achievable ceiling at this pitch (isolates dictionary discretisation from
    the OUR-vs-EMsoft master difference).

    Returns:
        dict with ``phase``, ``npx``, ``n_base``, ``fine_resolution_deg``,
        ``grid_width_deg``, ``parity_deg``, ``dict_size``, ``our_source``,
        ``fine_parity`` (OUR-vs-EMsoft fraction < parity_deg), ``self_parity``
        (EMsoft-vs-EMsoft ceiling), ``disorientation_deg`` (list, OUR matches),
        ``disorientation_median_deg``, ``perturb_median_deg``, ``device``,
        ``wall_s``.

    Raises:
        ValueError: unknown phase or missing oracle file.
        ForwardSimError: no CUDA and ``FORWARD_SIM_DEVICE`` unset.
    """
    from backend.dictionary_gpu.detector import detector_pixel_directions
    from backend.dictionary_gpu.master_loader import load_master_pattern
    from backend.dictionary_gpu.projector import project_patterns
    from tools.pattern_comparison import circular_mask

    if phase not in _ORACLE:
        raise ValueError(f"unknown phase {phase!r}; known: {sorted(_ORACLE)}")
    oracle = _project_root() / _ORACLE[phase]
    if not oracle.is_file():
        raise ValueError(f"oracle file not found for {phase!r}: {oracle}")

    dev = torch.device(device) if device is not None else get_device()
    t0 = time.time()

    # --- (1) base orientations + (2) perturbed queries + (3) fine dictionary ---
    bases = _sample_orientations(n_base, resolution=base_resolution_deg, seed=seed)
    n = bases.shape[0]
    queries_np = _perturb_orientations(
        bases, min_deg=perturb_min_deg, max_deg=perturb_max_deg, seed=seed + 1
    )
    perturb_deg = _disorientation_deg_batch(bases, queries_np)  # actual offsets

    dict_np, _dict_base_idx = _fine_local_dictionary(
        bases, resolution_deg=fine_resolution_deg, grid_width_deg=grid_width_deg
    )
    dict_size = dict_np.shape[0]

    queries = torch.from_numpy(queries_np).to(device=dev, dtype=torch.float32)
    dict_q = torch.from_numpy(dict_np).to(device=dev, dtype=torch.float32)

    # --- detector directions (shared) + both masters ---------------------------
    pix = detector_pixel_directions(
        detector_shape, pc=pc, tilt_deg=sample_tilt_deg,
        detector_tilt_deg=detector_tilt_deg, azimuthal_deg=azimuthal_deg,
        device=str(dev),
    )
    em = load_master_pattern(str(oracle), energy_kv=energy_kV, device=str(dev))
    our_hemi, our_source = _our_hemispheres(
        phase, npx=npx, energy_kV=energy_kV, dmin=dmin, absflg=absflg,
        device=dev, allow_saved=allow_saved,
    )

    H, W = detector_shape
    mask = circular_mask(detector_shape, radius_frac=1.0)
    mask_flat = torch.from_numpy(mask.ravel()).to(dev)

    # EMsoft dictionary patterns (the reference index) — z-scored once.
    dict_em_pat = project_patterns(dict_q, pix, em.hemispheres, detector_shape)
    dict_em_z = _masked_zscore(dict_em_pat.reshape(dict_size, H * W), mask_flat)

    # OUR query patterns + (for the ceiling) EMsoft query patterns.
    our_q_pat = project_patterns(queries, pix, our_hemi, detector_shape)
    our_q_z = _masked_zscore(our_q_pat.reshape(n, H * W), mask_flat)
    em_q_pat = project_patterns(queries, pix, em.hemispheres, detector_shape)
    em_q_z = _masked_zscore(em_q_pat.reshape(n, H * W), mask_flat)

    # NCC-argmax each query against the EMsoft dictionary.
    our_match = (our_q_z @ dict_em_z.T).argmax(dim=1).detach().cpu().numpy()
    self_match = (em_q_z @ dict_em_z.T).argmax(dim=1).detach().cpu().numpy()

    diso_our = _disorientation_deg_batch(queries_np, dict_np[our_match])
    diso_self = _disorientation_deg_batch(queries_np, dict_np[self_match])
    fine_parity = float(np.mean(diso_our < parity_deg))
    self_parity = float(np.mean(diso_self < parity_deg))

    wall_s = time.time() - t0
    return {
        "phase": phase,
        "npx": npx,
        "n_base": n,
        "absflg": int(absflg),
        "dmin": dmin,
        "energy_kV": energy_kV,
        "device": str(dev),
        "our_source": our_source,
        "fine_resolution_deg": float(fine_resolution_deg),
        "grid_width_deg": float(grid_width_deg),
        "parity_deg": float(parity_deg),
        "perturb_band_deg": [float(perturb_min_deg), float(perturb_max_deg)],
        "perturb_median_deg": float(np.median(perturb_deg)),
        "dict_size": int(dict_size),
        "fine_parity": fine_parity,
        "self_parity": self_parity,
        "disorientation_deg": diso_our.tolist(),
        "disorientation_median_deg": float(np.median(diso_our)),
        "self_disorientation_median_deg": float(np.median(diso_self)),
        "wall_s": wall_s,
    }


def _save_example_pngs(
    phase: str,
    npx: int,
    dict_em: torch.Tensor,
    dict_our: torch.Tensor,
    mask: np.ndarray,
    n_examples: int,
) -> list[str]:
    """Write side-by-side our|EMsoft|absdiff example PNGs (titled), one per row.

    Each PNG is a horizontal strip: OUR pattern, EMsoft pattern, and their
    abs-difference (each panel disc-normalised so the diff reflects pattern
    mismatch, not the arbitrary overall scale).  Titled with phase + index + NCC.
    """
    from PIL import Image, ImageDraw

    n = dict_em.shape[0]
    em = dict_em.detach().cpu().numpy().astype(np.float64)
    ours = dict_our.detach().cpu().numpy().astype(np.float64)
    idxs = np.linspace(0, n - 1, min(n_examples, n)).round().astype(int)
    idxs = np.unique(idxs)

    _ITER_DIR.mkdir(parents=True, exist_ok=True)
    out_paths: list[str] = []
    label_h = 16

    for k in idxs:
        e = _disc_z(em[k], mask)
        o = _disc_z(ours[k], mask)
        d = np.abs(e - o)
        ncc = float(
            np.dot((e - e[mask].mean())[mask].ravel(), (o - o[mask].mean())[mask].ravel())
            / (
                np.linalg.norm((e - e[mask].mean())[mask].ravel())
                * np.linalg.norm((o - o[mask].mean())[mask].ravel())
                + 1e-30
            )
        )
        panels = [_to_uint8(o, mask), _to_uint8(e, mask), _to_uint8(d, mask)]
        H, W = panels[0].shape
        strip = np.zeros((H + label_h, W * 3 + 4, 3), dtype=np.uint8)
        for j, p in enumerate(panels):
            x0 = j * (W + 2)
            strip[label_h : label_h + H, x0 : x0 + W, :] = np.stack([p] * 3, axis=-1)
        img = Image.fromarray(strip)
        draw = ImageDraw.Draw(img)
        draw.text((1, 1), f"our  emsoft  absdiff  {phase} o#{k} NCC={ncc:.3f}", fill=(255, 255, 0))
        out = _ITER_DIR / f"parity_{phase}_npx{npx}_o{k:03d}.png"
        img.save(str(out))
        out_paths.append(str(out))
    return out_paths


def _disc_z(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero-mean / unit-std normalise ``arr`` over its masked disc (out-of-disc=0)."""
    vals = arr[mask]
    mu = float(vals.mean()) if vals.size else 0.0
    sd = float(vals.std()) if vals.size else 1.0
    if sd == 0.0:
        sd = 1.0
    out = (arr - mu) / sd
    return np.where(mask, out, 0.0)


def _to_uint8(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Scale an array to uint8 over its in-disc min/max (out-of-disc black)."""
    vals = arr[mask]
    lo = float(vals.min()) if vals.size else 0.0
    hi = float(vals.max()) if vals.size else 1.0
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    norm = np.where(mask, norm, 0.0)
    return (norm * 255.0).astype(np.uint8)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pattern-vs-pattern NCC + indexing parity (OUR vs EMsoft master)."
    )
    p.add_argument("phase", choices=sorted(_ORACLE), help="phase (Ni or Al)")
    p.add_argument("--npx", type=int, default=50, help="OUR master Lambert half-grid (2*npx+1)")
    p.add_argument("--n-orient", type=int, default=100, help="number of FZ orientations")
    p.add_argument("--dmin", type=float, default=0.10, help="reflection dmin (nm), rebuild only")
    p.add_argument("--absflg", type=int, default=1, choices=(1, 3), help="absorption flag, rebuild only")
    p.add_argument("--energy", type=float, default=20.0, help="accelerating voltage (kV)")
    p.add_argument("--no-saved", action="store_true", help="force rebuild (ignore saved .npy)")
    p.add_argument("--no-images", action="store_true", help="skip example PNGs")
    p.add_argument("--device", default=None, help="torch device override (cpu/cuda)")
    p.add_argument(
        "--fine",
        action="store_true",
        help="run the FINE sub-2 deg precision parity instead of the coarse metrics",
    )
    p.add_argument("--n-base", type=int, default=50, help="[fine] base orientations")
    p.add_argument(
        "--fine-resolution", type=float, default=1.0, help="[fine] dictionary pitch (deg)"
    )
    p.add_argument("--grid-width", type=float, default=3.0, help="[fine] local-grid half-extent (deg)")
    return p


def _run_fine(args) -> int:
    res = fine_pattern_parity(
        args.phase,
        npx=args.npx,
        dmin=args.dmin,
        absflg=args.absflg,
        n_base=args.n_base,
        fine_resolution_deg=args.fine_resolution,
        grid_width_deg=args.grid_width,
        energy_kV=args.energy,
        device=args.device,
        allow_saved=not args.no_saved,
    )
    print(
        f"FINE-PARITY({res['phase']}, npx={res['npx']}, n_base={res['n_base']}, "
        f"src={res['our_source']}):"
    )
    print(
        f"  dict pitch={res['fine_resolution_deg']:.2f} deg  width={res['grid_width_deg']:.1f} deg  "
        f"dict_size={res['dict_size']}  perturb median={res['perturb_median_deg']:.2f} deg"
    )
    print(
        f"  FINE parity (<{res['parity_deg']:.0f} deg) = {res['fine_parity']:.3f}  "
        f"(self/ceiling {res['self_parity']:.3f}, "
        f"median disorientation {res['disorientation_median_deg']:.2f} deg)"
    )
    print(f"  device={res['device']}  wall={res['wall_s']:.1f}s")
    return 0


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        if args.fine:
            return _run_fine(args)
        res = pattern_parity(
            args.phase,
            npx=args.npx,
            n_orient=args.n_orient,
            dmin=args.dmin,
            absflg=args.absflg,
            energy_kV=args.energy,
            device=args.device,
            allow_saved=not args.no_saved,
            save_images=not args.no_images,
        )
    except ForwardSimError as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        f"PATTERN-PARITY({res['phase']}, npx={res['npx']}, n={res['n_orient']}, "
        f"src={res['our_source']}):"
    )
    print(
        f"  pattern NCC  mean={res['pattern_ncc_mean']:.4f}  "
        f"median={res['pattern_ncc_median']:.4f}  p05={res['pattern_ncc_p05']:.4f}  "
        f"min={res['pattern_ncc_min']:.4f}"
    )
    print(
        f"  indexing parity (<{_PARITY_DEG:.0f} deg) = {res['indexing_parity']:.3f}  "
        f"(self-consistency {res['self_consistency']:.3f}, "
        f"median disorientation {res['disorientation_median_deg']:.2f} deg)"
    )
    print(f"  device={res['device']}  wall={res['wall_s']:.1f}s")
    for png in res.get("example_pngs", []):
        print(f"  wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
