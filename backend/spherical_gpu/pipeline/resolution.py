"""Pseudo-symmetry orientation resolution for cubic-approximant intermetallics.

For a cubic-approximant phase (point group m-3 / 23) the SHT-spherical SO(3)
correlation lands on a WRONG pseudo-symmetric variant (the Kikuchi-band geometry
is fully cubic m-3m while the crystal is only m-3). The PRODUCTION fix
(:func:`resolve_map` / :func:`resolve_eulers`) replaces those orientations with
**Hough band-geometry indexing**, which is the reliable orientation source for
these phases.

Why Hough and not the coset resolver: measured on real SampleB alpha-AlFeMnSi
(render-NCC of each orientation vs the experimental pattern, 36 px), Hough renders
at a uniform median 0.51 (min 0.46, ZERO failures) and is >= the coset-resolved
orientation on 36/36 pixels, whereas the coset min-disorientation-to-Hough
selection is bimodal (22% of pixels render at ~0.19, as bad as the raw wrong
variant). The renderer does not honour the full m-3 symmetry, so "closest to Hough
under m-3" is the wrong selection criterion — picking Hough directly is both more
correct AND much faster (no per-pixel SHT cc re-correlation, the old ~50 s silent
pass). See the project changelog (2026-06-28).

The coset machinery (:func:`extract_topk_peaks`, :func:`topk_distinct`,
``pseudosym.resolve_variant``) is retained as a tested alternative strategy
(spherical-precision variant selection) but is NOT in the default path.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from ..pseudosym import same_orientation_angle_deg


def topk_distinct(
    quats: np.ndarray,
    scores: np.ndarray,
    point_group: str,
    topk: int,
    min_sep_deg: float,
) -> np.ndarray:
    """Indices of the top-`topk` DISTINCT orientations (by score), where distinct
    means > `min_sep_deg` from every already-kept orientation (same-orientation-
    modulo-symmetry test).

    Greedy: walk orientations in descending score; keep one if it is far enough
    from all kept ones; stop at `topk`. Uses the cheap ONE-SIDED symmetry metric
    (:func:`same_orientation_angle_deg`) — the correct notion of "distinct
    orientation of the same phase" and ~M times faster than the both-sided
    disorientation, which matters because this runs per pixel over a whole map.
    """
    order = np.argsort(scores)[::-1]
    kept_idx: list[int] = []
    kept_q = np.empty((0, 4), dtype=np.float64)
    for i in order:
        i = int(i)
        if kept_q.shape[0]:
            ang = same_orientation_angle_deg(kept_q, quats[i], point_group)  # (K,)
            if ang.min() <= min_sep_deg:
                continue
        kept_idx.append(i)
        kept_q = np.vstack([kept_q, quats[i]])
        if len(kept_idx) >= topk:
            break
    return np.asarray(kept_idx, dtype=int)


def _decode_bins(flat_idx: np.ndarray, L: int) -> np.ndarray:
    """Decode cc-volume flat bin indices -> Bunge ZXZ Euler (radians).

    Mirrors ``Tier1Indexer._decode_peak`` (integer-bin, no sub-bin refinement);
    bin resolution ~ 2*pi/(2L-1). Sufficient for variant selection — the resolved
    orientation is a coset variant of a peak, so its precision is the peak's.
    """
    size = 2 * L - 1
    off = L - 1
    scale = 2.0 * math.pi / size
    hp = math.pi / 2.0
    tp = 2.0 * math.pi
    a = flat_idx // (size * size)
    rem = flat_idx % (size * size)
    b = rem // size
    c = rem % size
    alpha = ((a - off + size) % size) * scale
    gamma = ((off - c + size) % size) * scale
    phi1 = (alpha + hp) % tp
    phi2 = (gamma - hp) % tp
    phi1 = (phi1 + 3.0 * hp) % tp
    return np.stack([phi1, b * scale, phi2], axis=1)


def extract_topk_peaks(
    indexer,
    patterns,
    topk: int = 8,
    min_sep_deg: float = 10.0,
    point_group: str = "m-3",
    n_bins: int = 2000,
) -> list[np.ndarray]:
    """Top-K DISTINCT spherical cc-volume peaks per pattern, as quaternions.

    Uses the indexer's own SHT correlation + rDen volume (no change to the
    validated ``_index_batch``). Returns a list (one entry per pattern) of
    ``(Ki, 4)`` quaternion arrays, highest-score first.
    """
    import torch
    from orix.quaternion import Rotation
    from .._math.sht_cc import rs2cc_

    device = indexer.device
    pats = torch.as_tensor(np.asarray(patterns, dtype=np.float32), device=device)
    if pats.ndim == 2:
        pats = pats[None]
    prep = indexer._run_preprocessing(pats)
    L = indexer.bandwidth
    cc = rs2cc_(L, indexer._direct_sht_coefs(prep),
                indexer._master_coefs, indexer._wigner_table).real.float()
    nc = (cc * indexer._rDen_vol).reshape(pats.shape[0], -1)   # (B, S^3)
    nb = min(n_bins, nc.shape[1])
    topv, topi = torch.topk(nc, nb, dim=1)            # (B, nb)
    # ONE orix call for ALL pixels' candidate bins (orix object creation per call
    # is the cost; batching it amortises it to ~0 per pixel).
    eul = _decode_bins(topi.reshape(-1).cpu().numpy(), L)        # (B*nb, 3)
    quats = np.asarray(Rotation.from_euler(eul).data).reshape(pats.shape[0], nb, 4)
    scores = topv.cpu().numpy()
    out: list[np.ndarray] = []
    for b in range(quats.shape[0]):
        idx = topk_distinct(quats[b], scores[b], point_group, topk, min_sep_deg)
        out.append(quats[b][idx])
    return out


def batch_hough_orientations(
    patterns,
    cif_path: str,
    det_params: dict,
    progress=None,
    chunk: int = 256,
):
    """Hough band-geometry orientations for a batch — builds the PyEBSDIndex
    indexer ONCE and indexes the patterns in chunks (so progress can be reported
    while a large map runs, instead of one silent blocking call).

    `patterns` must be background-flattened (dynamic-BG removed). Returns
    ``(quats (B,4), fit (B,), nmatch (B,))`` — ``fit`` is the mean band angular
    deviation in degrees (lower = better) and ``nmatch`` the number of matched
    bands, both used to detect Hough failures for the per-pixel fallback.

    `progress(msg)` is called once per chunk with a count + rate metric.
    """
    from orix.crystal_map import Phase, PhaseList
    from ebsd_utils import sanitize_cif, prepare_reflectors, create_indexer
    from kikuchipy.detectors import EBSDDetector
    from kikuchipy.signals import EBSD

    phase = Phase.from_cif(sanitize_cif(str(cif_path)))
    try:
        phase.name = Path(cif_path).stem
    except Exception:
        pass
    pl = PhaseList(phase)
    H, W = int(det_params["pat_height"]), int(det_params["pat_width"])
    det = EBSDDetector(
        shape=(H, W),
        sample_tilt=float(det_params.get("sample_tilt", 70.0)),
        tilt=float(det_params.get("tilt", 0.0)),
        pc=(float(det_params["pc_x"]), float(det_params["pc_y"]), float(det_params["pc_z"])),
        convention="bruker",
        binning=int(det_params.get("binning", 1)),
    )
    indexer = create_indexer(det, pl, prepare_reflectors(pl), nBands=12)   # build ONCE

    pats = np.asarray(patterns, dtype=np.float32)
    B = pats.shape[0]
    quats = np.zeros((B, 4), dtype=np.float64)
    fit = np.full(B, np.inf, dtype=np.float64)
    nmatch = np.zeros(B, dtype=np.float64)
    t0 = time.time()
    for s in range(0, B, max(1, int(chunk))):
        e = min(s + max(1, int(chunk)), B)
        eb = EBSD(pats[s:e], detector=det)
        out = eb.hough_indexing(pl, indexer, verbose=0, return_index_data=True)
        xmap, idata = out[0], out[1]
        quats[s:e] = np.asarray(xmap.rotations.data).reshape(-1, 4)[: e - s]

        def _last_row(name):
            a = np.asarray(idata[name])
            return (a[-1] if a.ndim > 1 else a).ravel()[: e - s]
        try:
            fit[s:e] = _last_row("fit")
            nmatch[s:e] = _last_row("nmatch")
        except Exception:
            pass  # quality optional; leave fit=inf/nmatch=0 -> treated as failure
        if progress:
            rate = e / max(time.time() - t0, 1e-3)
            try:
                progress(f"Hough anchor: {e}/{B} patterns ({rate:.0f} px/s)")
            except Exception:
                pass
    return quats, fit, nmatch


def resolve_eulers(
    patterns,
    raw_eulers: np.ndarray,
    cif_path: str,
    det_params: dict,
    point_group: str,
    z_rot=None,
    progress=None,
):
    """Integration wrapper: replace a spherical map's orientations for a pseudo-
    symmetric phase with the reliable Hough orientations, returning Bunge-ZXZ
    Euler angles ready to drop into the live indexing path.

    - For a NON pseudo-symmetric phase (true cubic m-3m, hexagonal, etc.) it
      returns ``raw_eulers`` UNCHANGED and runs no extra compute — the
      validated spherical path stays bit-identical.
    - For a pseudo-symmetric cubic-approximant phase (m-3, 23) it runs Hough
      band-geometry indexing (which renders strictly better than the spherical
      pseudo-variant — see module docstring) and returns those Eulers, keeping
      the raw spherical orientation only on the rare pixel where Hough failed.
    - On ANY failure (no CIF, Hough error, ...) it FAILS SAFE: returns
      ``raw_eulers`` unchanged so a resolver problem can never break indexing.

    `progress(msg, pct=None)` is forwarded for live log output.

    Returns
    -------
    (eulers, info) : (np.ndarray (B,3), dict | None)
        ``info`` carries the Hough/raw quaternions + per-pixel fit/nmatch +
        ``n_fallback`` (for diagnostics / a future manual-flip UI).
    """
    from ..pseudosym import spherical_unreliable

    # Fire for ALL masters the spherical correlation can't reliably index
    # (z_rot==2): cubic approximants m-3/23, cubic -43m, and orthorhombic
    # mmm/222/mm2. Prefer the master's actual z_rot; fall back to the point-group
    # name. True-cubic/hex/tetragonal and the working low-sym classes (2/m, -1)
    # stay a bit-identical passthrough.
    if not spherical_unreliable(z_rot, point_group):
        return raw_eulers, None
    if not cif_path or patterns is None:
        return raw_eulers, None
    try:
        from orix.quaternion import Rotation
        res = resolve_map(patterns, cif_path, det_params, point_group,
                          raw_eulers=raw_eulers, progress=progress)
        eulers = np.asarray(Rotation(res["resolved"]).to_euler(), dtype=np.float64)
        return eulers, res
    except Exception:
        # Fail safe: never let a resolver problem break the indexing run.
        if progress:
            try:
                progress("Pseudo-symmetry resolution FAILED — keeping raw "
                         "spherical orientations (may be wrong for intermetallics)")
            except Exception:
                pass
        return raw_eulers, None


def resolve_map(
    patterns,
    cif_path: str,
    det_params: dict,
    point_group: str,
    raw_eulers: np.ndarray | None = None,
    progress=None,
    fit_max_deg: float = 3.0,
    nmatch_min: int = 4,
) -> dict:
    """Best orientations for a pseudo-symmetric-phase batch = Hough band geometry,
    with a per-pixel fallback to the raw spherical orientation where Hough failed.

    `patterns`: (B, H, W) experimental patterns (as fed to the spherical indexer).
    Hough consumes their dynamic-BG-removed version. The spherical SHT correlation
    is NOT re-run here (it lands on the wrong pseudo-variant for these phases — see
    module docstring), so this is also much faster than the old coset path.

    Returns a dict: ``resolved`` (B,4) chosen quaternions, ``hough`` (B,4),
    ``fit``/``nmatch`` (B,) Hough quality, ``n_fallback`` int, ``method`` str.
    """
    import kikuchipy as kp

    def _emit(msg, pct=None):
        if progress:
            try:
                progress(msg, pct)
            except TypeError:
                try:
                    progress(msg)
                except Exception:
                    pass
            except Exception:
                pass

    pats = np.asarray(patterns, dtype=np.float32)
    B = pats.shape[0]
    _emit(f"Pseudo-symmetry ({point_group}): dynamic-BG for {B} patterns…", 0.90)
    # copy=True: remove_dynamic_background mutates in place and kp.signals.EBSD
    # aliases an already-float32 input — without the copy we would silently
    # background-subtract the caller's live pattern array.
    sig = kp.signals.EBSD(np.array(pats, dtype=np.float32, copy=True))
    sig.remove_dynamic_background(operation="subtract", filter_domain="frequency")

    t0 = time.time()
    hough, fit, nmatch = batch_hough_orientations(
        sig.data, cif_path, det_params, progress=lambda m: _emit(m, 0.95))
    dt = time.time() - t0

    resolved = hough.copy()
    bad = (nmatch < nmatch_min) | (fit > fit_max_deg) | ~np.isfinite(fit)
    n_fallback = int(bad.sum())
    if raw_eulers is not None and n_fallback:
        from orix.quaternion import Rotation
        raw_q = np.asarray(Rotation.from_euler(np.asarray(raw_eulers)).data).reshape(-1, 4)
        if raw_q.shape[0] == B:
            resolved[bad] = raw_q[bad]

    rate = B / max(dt, 1e-3)
    good_fit = float(np.median(fit[np.isfinite(fit)])) if np.isfinite(fit).any() else float("nan")
    _emit(
        f"Pseudo-symmetry: resolved {B} px via Hough in {dt:.1f}s ({rate:.0f} px/s, "
        f"median fit {good_fit:.2f}°"
        + (f", {n_fallback} kept spherical: Hough failed" if n_fallback else "")
        + ")",
        0.98,
    )
    return {
        "resolved": resolved,
        "hough": hough,
        "fit": fit,
        "nmatch": nmatch,
        "n_fallback": n_fallback,
        "method": "hough",
    }
