"""Lattice parameter estimator for the Crystal Hint feature.

Detects Kikuchi bands via Hough transform, estimates d-spacings from
band positions, infers crystal system + lattice 'a' parameter from
d-spacing ratios.

v1: cubic-focused with FCC/BCC/diamond ratio tables. Hexagonal /
tetragonal can be added later.

See docs/superpowers/specs/2026-05-26-crystal-hint-feature-design.md Section 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter


# Wavelength of electron in Angstroms at accelerating voltage `kv` (kV).
# Relativistic correction included.
def wavelength_from_voltage_kv(kv: float) -> float:
    """Return electron wavelength in Angstroms for voltage in kilovolts.

    λ_A = 12.2643 / sqrt(V_eV * (1 + 0.97845e-6 * V_eV))
    where V_eV is voltage in electron-volts.
    """
    if kv <= 0 or kv > 1000 or not np.isfinite(kv):
        raise ValueError(f"Invalid voltage: {kv} kV (must be in (0, 1000))")
    V_eV = kv * 1000.0
    return 12.2643 / np.sqrt(V_eV * (1.0 + 0.97845e-6 * V_eV))


@dataclass
class LatticeReport:
    d_spacings_A: list[float] = field(default_factory=list)
    a_estimate_A: Optional[float] = None
    a_range_A: Optional[tuple[float, float]] = None
    confidence: str = "none"  # "high" | "medium" | "low" | "none"
    crystal_system_best: Optional[str] = None
    candidate_hkls: list[tuple[int, int, int]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "d_spacings_A": [float(d) for d in self.d_spacings_A],
            "a_estimate_A": float(self.a_estimate_A) if self.a_estimate_A else None,
            "a_range_A": list(self.a_range_A) if self.a_range_A else None,
            "confidence": self.confidence,
            "crystal_system_best": self.crystal_system_best,
            "candidate_hkls": [list(h) for h in self.candidate_hkls],
            "warnings": self.warnings,
        }


# Tabulated d-spacing ratios. For each "system + lattice family", we list
# the typical low-index hkl reflections in decreasing d (largest first)
# along with their normalised d-ratios = d_hkl / d_largest.
#
# For cubic: d_hkl = a / sqrt(h² + k² + l²)
#
# Allowed reflections per Bravais lattice:
#   FCC (Fm-3m):       (h+k+l) all even or all odd  → 111, 200, 220, 311, 222, 400
#   BCC (Im-3m):       h+k+l even                   → 110, 200, 211, 220, 310
#   Diamond (Fd-3m):   FCC + (h+k+l ≡ 4n) extinction → 111, 220, 311, 400
SYSTEM_RATIOS: dict[str, dict[str, list[tuple[tuple[int, int, int], float]]]] = {
    "cubic": {
        "FCC (Fm-3m / Fd-3m)": [
            ((1, 1, 1), 1.0),
            ((2, 0, 0), 1.0 / np.sqrt(4 / 3)),
            ((2, 2, 0), 1.0 / np.sqrt(8 / 3)),
            ((3, 1, 1), 1.0 / np.sqrt(11 / 3)),
        ],
        "BCC (Im-3m)": [
            ((1, 1, 0), 1.0),
            ((2, 0, 0), 1.0 / np.sqrt(2)),
            ((2, 1, 1), 1.0 / np.sqrt(3)),
            ((2, 2, 0), 1.0 / np.sqrt(4)),
            ((3, 1, 0), 1.0 / np.sqrt(5)),
        ],
    },
}


def _hkl_norm(hkl: tuple[int, int, int]) -> float:
    """sqrt(h² + k² + l²) — for cubic d = a / hkl_norm."""
    return float(np.sqrt(hkl[0] ** 2 + hkl[1] ** 2 + hkl[2] ** 2))


def detect_bands_hough(
    pattern: np.ndarray,
    n_peaks: int = 8,
    threshold_rel: float = 0.45,
) -> list[dict]:
    """Detect Kikuchi bands via straight-line Hough transform.

    Returns a list of band dicts:
        [{"rho": float, "theta_rad": float, "score": float}, ...]
    sorted by score (highest first).
    """
    try:
        from skimage.transform import hough_line, hough_line_peaks
    except ImportError as exc:
        raise RuntimeError("skimage required for Hough band detection") from exc

    # Normalise + gentle smoothing to suppress noise
    pat = pattern.astype(np.float32)
    pat = pat - gaussian_filter(pat, sigma=max(pat.shape) / 6.0)
    pat = pat - pat.min()
    if pat.max() > 0:
        pat = pat / pat.max()
    # Edge response: gradient magnitude amplifies band edges
    gy, gx = np.gradient(pat)
    edge = np.sqrt(gy * gy + gx * gx).astype(np.float32)

    # Hough at coarse angle grid
    theta_grid = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False)
    h_acc, theta, rho = hough_line(edge, theta=theta_grid)
    accum_max = h_acc.max() if h_acc.size else 1.0
    threshold = threshold_rel * accum_max

    peaks = hough_line_peaks(
        h_acc, theta, rho,
        num_peaks=n_peaks,
        threshold=threshold,
        min_distance=8,
        min_angle=10,
    )
    bands = []
    for accum, t, r in zip(*peaks):
        bands.append({"rho": float(r), "theta_rad": float(t), "score": float(accum)})
    return bands


def _bragg_angle_from_band(
    band: dict,
    detector_geom: dict,
) -> float:
    """Convert band position (rho, theta in Hough) to Bragg angle 2θ_B.

    Simple geometric model: the perpendicular distance from the pattern
    center (PC_x, PC_y) to the band-line gives the projected distance d_px,
    which relates to 2θ_B by tan(2θ_B) = d_px * pixel_size / detector_distance.

    detector_geom keys (Bruker convention):
        pc_x, pc_y (in [0, 1] of pattern width/height)
        pat_width, pat_height (in pixels)
        pc_z   (in fraction of pattern width — detector distance proxy)

    Returns the half-angle θ_B in degrees.
    """
    pc_x = detector_geom["pc_x"]
    pc_y = detector_geom["pc_y"]
    pc_z = detector_geom["pc_z"]
    pat_w = detector_geom["pat_width"]

    # PC in pixel coordinates
    cx = pc_x * pat_w
    cy = pc_y * detector_geom.get("pat_height", pat_w)

    # Hough representation: a line at angle theta_rad has perpendicular distance rho from origin
    # We need the perpendicular distance from (cx, cy) to the line
    rho = band["rho"]
    t = band["theta_rad"]
    perp_dist_px = abs(cx * np.cos(t) + cy * np.sin(t) - rho)

    # Detector distance in pixels: pc_z * pat_width
    L_px = pc_z * pat_w
    if L_px <= 0:
        return 0.0

    # 2 * θ_B = arctan(perp_dist_px / L_px)
    two_theta_rad = np.arctan(perp_dist_px / L_px)
    return float(np.degrees(two_theta_rad / 2.0))  # half-angle θ_B in degrees


def _match_d_ratios(
    d_spacings: list[float],
    system: str,
) -> tuple[Optional[str], Optional[list[tuple[int, int, int]]], float]:
    """Find the SYSTEM_RATIOS entry that best matches the observed d-ratios.

    Returns (lattice_family_name, hkls_in_order, residual_norm).
    None family if no good match (residual > 0.15).
    """
    if system not in SYSTEM_RATIOS:
        return (None, None, float("inf"))
    if len(d_spacings) < 2:
        return (None, None, float("inf"))

    sorted_d = sorted(d_spacings, reverse=True)
    observed_ratios = [d / sorted_d[0] for d in sorted_d[1:]]

    best = None
    for family, table in SYSTEM_RATIOS[system].items():
        # table is in decreasing d (largest first), already normalised
        ref_ratios = [r for (_h, r) in table[1:]]
        n_compare = min(len(observed_ratios), len(ref_ratios))
        if n_compare == 0:
            continue
        diff = np.array(observed_ratios[:n_compare]) - np.array(ref_ratios[:n_compare])
        residual = float(np.sqrt(np.mean(diff ** 2)))
        hkls = [h for (h, _r) in table[: n_compare + 1]]
        if best is None or residual < best[2]:
            best = (family, hkls, residual)

    if best is None:
        return (None, None, float("inf"))
    family, hkls, residual = best
    if residual > 0.15:
        return (None, None, residual)
    return best


def estimate_lattice(
    pattern: np.ndarray,
    detector_geom: dict,
    crystal_system_hint: Optional[str] = None,
) -> LatticeReport:
    """Estimate lattice parameter 'a' from Kikuchi band positions.

    Args:
        pattern: 2D pattern (H, W), float-convertible.
        detector_geom: dict with pc_x, pc_y, pc_z, pat_width, pat_height,
            voltage_kv (optional, defaults to 20 if missing/corrupt).
        crystal_system_hint: one of "cubic", "hexagonal", "tetragonal".
            If None, defaults to "cubic" (v1).

    Returns LatticeReport. Confidence reflects how well the band ratios
    match a known crystal-system table.
    """
    if pattern.ndim != 2:
        raise ValueError(f"pattern must be 2D (got shape {pattern.shape})")

    report = LatticeReport()
    system = crystal_system_hint or "cubic"

    # Voltage handling (handle corrupt header gracefully)
    voltage_kv = detector_geom.get("voltage_kv", 20.0)
    try:
        if not (1.0 < voltage_kv < 100.0):
            raise ValueError("voltage_kv outside [1, 100]")
        wavelength_A = wavelength_from_voltage_kv(voltage_kv)
    except (ValueError, TypeError):
        report.warnings.append(
            f"voltage_kv={voltage_kv} corrupt — falling back to 20 kV"
        )
        wavelength_A = wavelength_from_voltage_kv(20.0)

    # Band detection
    try:
        bands = detect_bands_hough(pattern, n_peaks=8)
    except Exception as exc:
        report.warnings.append(f"Hough band detection failed: {exc}")
        return report

    if len(bands) < 2:
        report.warnings.append(f"only {len(bands)} bands detected — need ≥2 for ratios")
        return report

    # Convert band-line positions to a pseudo-Bragg angle.
    #
    # ⚠ Honesty note: a Kikuchi band is a gnomonic projection of a
    # diffracting plane; its trace passes *through* the pattern centre, and
    # the Bragg angle corresponds to the band's projected WIDTH, not its
    # perpendicular distance from PC. The Hough peak we detect gives us the
    # band's trace, not its width — so the conversion below produces a
    # rough lattice scale that is correlated with, but not equal to, the
    # true a-parameter. Use this as a coarse filter (helps narrow the
    # external-DB search) and not as a calibrated measurement.
    report.warnings.append(
        "Lattice 'a' is a coarse heuristic from Hough band positions, "
        "not a calibrated measurement — treat estimate as ±20%"
    )
    d_spacings = []
    for band in bands:
        theta_B_deg = _bragg_angle_from_band(band, detector_geom)
        if theta_B_deg <= 0.0001:
            continue
        sin_th = np.sin(np.radians(theta_B_deg))
        if sin_th <= 0:
            continue
        d = wavelength_A / (2.0 * sin_th)
        if 0.5 < d < 50.0:  # plausible range
            d_spacings.append(d)
    d_spacings.sort(reverse=True)
    report.d_spacings_A = d_spacings

    if len(d_spacings) < 2:
        report.warnings.append("Could not derive ≥2 plausible d-spacings")
        return report

    # Ratio matching
    family, hkls, residual = _match_d_ratios(d_spacings, system)
    if family is None:
        report.warnings.append(
            f"No {system} lattice family fit observed ratios (residual={residual:.3f})"
        )
        return report

    # Lattice parameter from largest d-spacing × first hkl norm (cubic)
    if hkls:
        first_hkl = hkls[0]
        a_estimate = d_spacings[0] * _hkl_norm(first_hkl)
        report.a_estimate_A = a_estimate
        report.a_range_A = (a_estimate * 0.80, a_estimate * 1.20)
        report.candidate_hkls = hkls
        report.crystal_system_best = f"{system} — {family}"

    # Confidence based on band count + residual
    if len(d_spacings) >= 4 and residual < 0.05:
        report.confidence = "high"
    elif len(d_spacings) >= 3 and residual < 0.10:
        report.confidence = "medium"
    elif len(d_spacings) >= 2 and residual < 0.15:
        report.confidence = "low"
    else:
        report.confidence = "none"

    return report
