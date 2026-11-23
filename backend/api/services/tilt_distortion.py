"""Tilt distortion check for Oxford ``.h5oina`` files.

For EBSD the sample is tilted (typically 70 degrees), but AZtec stores the
electron images and the maps with square pixels (``X Step == Y Step``). On the
sample surface one image direction is then foreshortened by
``k = 1 / cos(tilt)`` -- 2.92 at 70 degrees. A grain size, a particle shape or
a scale bar read off such an image is wrong by that factor in one direction.

Which direction is foreshortened is NOT taken from the header: on the data
this was developed on, ``Tilt Axis`` was 0 while the foreshortening lay in X.
The header only says *whether* to expect a distortion and *how large* it
should be; the direction is measured on the image itself:

1. A Gaussian low-pass (sigma 15 px) is subtracted to remove large-scale
   brightness gradients.
2. The image is cut into 4 x 4 tiles. Each tile's autocorrelation (via FFT,
   normalised by the overlap so the estimate is unbiased, and to 1 at zero
   shift) gives a correlation length along X (``Lx``) and along Y (``Ly``):
   the shift at which it falls to 0.5, linearly interpolated.
3. The 8 tiles with the most texture (highest standard deviation) are kept
   and the median of ``Lx / Ly`` over them is the measurement.

If the sample texture is isotropic on average, a foreshortened direction shows
as a shorter correlation length in that direction. Blur widens structures in
both directions equally and pulls the ratio towards 1; that is why a value
between 0.6 and 0.75 is still reported as "probably X foreshortened".

The module is read-only and independent of the routes: :func:`check_h5oina`
takes a path and returns a JSON-ready dict in the format agreed for this check
(German keys, see ``docs/user-guide/TiltDistortion.md``). It never raises for
a file it cannot judge -- it says ``unklar`` and why.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Tilt below this counts as "untilted" (k ~ 1): the measurement still runs,
#: as a control.
TILT_SUSPECT_DEG = 10.0
#: Gaussian low-pass subtracted before the autocorrelation (px).
HIGHPASS_SIGMA_PX = 15.0
#: Tiles per side.
N_TILES = 4
#: How many of the most textured tiles decide.
KEEP_TILES = 8
#: Fewer usable tiles than this: the image is too texture-poor to judge.
MIN_TILES = 4
#: Below this (relative to the image's own value range) a tile counts as flat.
FLAT_TILE_REL_STD = 1e-3
#: Two pixel-per-step values that differ by less than this belong to the same
#: coordinate system (the map inherits the image's distortion).
SAME_GRID_TOL = 0.15
#: How far a measured ratio may lie from cos(tilt) (relative) and still count
#: as "~ cos(tilt)". The reference measurements at 70 deg spread 0.29-0.41
#: around cos 70 = 0.342.
COS_MATCH_TOL = 0.40

# Classification limits for the median Lx/Ly.
RATIO_X_SURE = 0.60
RATIO_X_PROBABLE = 0.75
RATIO_Y_UNCLEAR = 1.30
RATIO_Y_SURE = 1.70

BEFUND_X = "X gestaucht"
BEFUND_Y = "Y gestaucht"
BEFUND_OK = "unverzerrt"
BEFUND_UNCLEAR = "unklar"


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

def _gaussian_highpass(img: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    a = np.asarray(img, dtype=np.float64)
    return a - gaussian_filter(a, sigma=sigma, mode="reflect")


def _crossing(profile: np.ndarray, level: float = 0.5) -> float:
    """First shift (>0) at which ``profile`` falls to ``level``, linearly
    interpolated between the two pixels around the crossing. NaN when it never
    falls that far within the profile."""
    p = np.asarray(profile, dtype=np.float64)
    below = np.flatnonzero(p[1:] <= level)
    if below.size == 0:
        return float("nan")
    i = int(below[0]) + 1          # p[i] <= level < p[i-1]
    a, b = p[i - 1], p[i]
    if not (np.isfinite(a) and np.isfinite(b)) or a == b:
        return float(i)
    return float((i - 1) + (a - level) / (a - b))


def tile_correlation_lengths(tile: np.ndarray) -> Tuple[float, float]:
    """(Lx, Ly) of one tile: the shifts along X (columns) and Y (rows) at which
    its autocorrelation falls to 0.5.

    The autocorrelation is computed with zero padding (no wrap-around) and
    divided by the number of overlapping pixel pairs at each shift, so a long
    shift is not penalised just for overlapping less."""
    t = np.asarray(tile, dtype=np.float64)
    t = t - t.mean()
    h, w = t.shape
    fh, fw = 2 * h, 2 * w
    F = np.fft.rfft2(t, s=(fh, fw))
    ac = np.fft.irfft2(F * np.conj(F), s=(fh, fw))
    ones = np.fft.rfft2(np.ones_like(t), s=(fh, fw))
    overlap = np.fft.irfft2(ones * np.conj(ones), s=(fh, fw))
    overlap = np.maximum(np.round(overlap), 1.0)
    ac = ac / overlap
    if not ac[0, 0] > 0:
        return float("nan"), float("nan")
    ac = ac / ac[0, 0]
    # Only up to half the tile: beyond that too few pairs overlap to trust.
    lx = _crossing(ac[0, : w // 2 + 1])
    ly = _crossing(ac[: h // 2 + 1, 0])
    return lx, ly


@dataclass
class AnisotropyMeasurement:
    lx_ly_median: Optional[float]
    tiles_used: int
    tiles_total: int
    tile_ratios: List[float] = field(default_factory=list)
    lx_median: Optional[float] = None
    ly_median: Optional[float] = None
    reason: Optional[str] = None


def measure_anisotropy(image: np.ndarray, *, sigma: float = HIGHPASS_SIGMA_PX,
                       n_tiles: int = N_TILES, keep: int = KEEP_TILES,
                       min_tiles: int = MIN_TILES) -> AnisotropyMeasurement:
    """Median ``Lx / Ly`` over the ``keep`` most textured of ``n_tiles``^2
    tiles of the high-passed ``image`` (2-D, rows = Y, columns = X).

    ``lx_ly_median`` is None (with ``reason``) when fewer than ``min_tiles``
    tiles give a finite ratio -- a texture-poor image is reported as such, not
    forced into a number."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:                       # RGB(A): luminance
        img = img[..., :3] @ np.array([0.299, 0.587, 0.114])
    if img.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {img.shape}")
    h, w = img.shape
    th, tw = h // n_tiles, w // n_tiles
    total = n_tiles * n_tiles
    if th < 8 or tw < 8:
        return AnisotropyMeasurement(None, 0, total, reason="Bild zu klein für die Kachelmessung")

    hp = _gaussian_highpass(img, sigma)
    span = float(np.ptp(img)) or 1.0
    tiles = []
    for r in range(n_tiles):
        for c in range(n_tiles):
            t = hp[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
            tiles.append((float(t.std()), t))
    tiles.sort(key=lambda st: st[0], reverse=True)

    ratios, lxs, lys = [], [], []
    for std, t in tiles[:keep]:
        if std / span < FLAT_TILE_REL_STD:
            continue
        lx, ly = tile_correlation_lengths(t)
        if np.isfinite(lx) and np.isfinite(ly) and ly > 0 and lx > 0:
            ratios.append(lx / ly)
            lxs.append(lx)
            lys.append(ly)
    if len(ratios) < min_tiles:
        return AnisotropyMeasurement(
            None, len(ratios), total, tile_ratios=[round(x, 3) for x in ratios],
            reason="zu wenig Textur für eine Messung (texturarm)")
    return AnisotropyMeasurement(
        float(np.median(ratios)), len(ratios), total,
        tile_ratios=[round(x, 3) for x in ratios],
        lx_median=float(np.median(lxs)), ly_median=float(np.median(lys)))


def classify_ratio(ratio: Optional[float]) -> Tuple[str, Optional[str]]:
    """(befund, sicherheit) for a median Lx/Ly on its own (no header)."""
    if ratio is None or not np.isfinite(ratio):
        return BEFUND_UNCLEAR, None
    if ratio < RATIO_X_SURE:
        return BEFUND_X, "sicher"
    if ratio < RATIO_X_PROBABLE:
        return BEFUND_X, "wahrscheinlich"
    if ratio <= RATIO_Y_UNCLEAR:
        return BEFUND_OK, "sicher"
    if ratio <= RATIO_Y_SURE:
        return BEFUND_UNCLEAR, None
    return BEFUND_Y, "sicher"


def stretch_factor(tilt_deg: float) -> float:
    """k = 1 / cos(tilt). Raises for a tilt that has no finite factor."""
    c = math.cos(math.radians(float(tilt_deg)))
    if c <= 1e-6:
        raise ValueError(f"tilt {tilt_deg} deg has no finite stretch factor")
    return 1.0 / c


@dataclass
class Verdict:
    befund: str
    sicherheit: Optional[str]
    faktor: float                      # factor to stretch the foreshortened direction by
    faktor_gemessen: Optional[float]
    hinweise: List[str]


def reconcile(tilt_deg: float, ratio: Optional[float], *, channel: str = "SE") -> Verdict:
    """Combine the header tilt with the image measurement (step 4).

    ``faktor`` is what the correction uses: the theoretical ``k`` when the
    measurement confirms the header, 1 when nothing is to be corrected or the
    case is unclear (an unclear image is never stretched by a guess)."""
    hints: List[str] = []
    tilted = tilt_deg >= TILT_SUSPECT_DEG
    k = stretch_factor(tilt_deg) if tilted else 1.0
    cos_t = 1.0 / k
    cls, sure = classify_ratio(ratio)
    measured = None
    if ratio is not None and np.isfinite(ratio) and ratio > 0:
        measured = round(1.0 / ratio, 2) if ratio < 1 else round(ratio, 2)

    if channel.upper().startswith("FSD"):
        hints.append("FSD-Kanal: nur ergänzend, Scan-Artefakte können auch ohne Kippung "
                     "Lx/Ly von 1,1 bis 2,5 ergeben; nicht allein danach entscheiden.")

    if ratio is None:
        hints.append("Keine Messung möglich (texturarm): Richtung unbekannt.")
        return Verdict(BEFUND_UNCLEAR, None, 1.0, None, hints)

    def near(value, target):
        return abs(value - target) <= COS_MATCH_TOL * target

    if tilted:
        if cls == BEFUND_X:
            if near(ratio, cos_t) or sure == "wahrscheinlich":
                if sure == "wahrscheinlich":
                    hints.append("Lx/Ly zwischen 0,6 und 0,75: wahrscheinlich X gestaucht, "
                                 "Unschärfe senkt den Kontrast.")
                return Verdict(BEFUND_X, sure, k, measured, hints)
            hints.append(f"Lx/Ly = {ratio:.2f} weicht deutlich von cos(Kippung) = {cos_t:.2f} ab: "
                         "unklar, manuell prüfen.")
            return Verdict(BEFUND_UNCLEAR, None, 1.0, measured, hints)
        if cls == BEFUND_Y:
            if near(1.0 / ratio, cos_t):
                hints.append("Stauchung liegt in Y, nicht in X.")
                return Verdict(BEFUND_Y, sure, k, measured, hints)
            hints.append(f"Lx/Ly = {ratio:.2f} passt nicht zu 1/cos(Kippung) = {k:.2f}: "
                         "unklar, manuell prüfen.")
            return Verdict(BEFUND_UNCLEAR, None, 1.0, measured, hints)
        if cls == BEFUND_OK:
            hints.append("Widerspruch: Header-Kippung ≥ 10°, aber keine messbare Stauchung. "
                         "Möglicherweise ist eine Tilt-Korrektur aktiv oder der Kippwert steht nur "
                         "im Header: unklar, manuell prüfen.")
            return Verdict(BEFUND_UNCLEAR, None, 1.0, measured, hints)
        hints.append(f"Lx/Ly = {ratio:.2f} liegt im unklaren Bereich 1,3–1,7.")
        return Verdict(BEFUND_UNCLEAR, None, 1.0, measured, hints)

    # Header says untilted.
    if cls == BEFUND_OK:
        return Verdict(BEFUND_OK, "sicher", 1.0, measured, hints)
    if cls == BEFUND_X and sure == "sicher":
        hints.append(f"Widerspruch: Header-Kippung < 10°, aber Lx/Ly = {ratio:.2f}: unklar, "
                     f"manuell prüfen (gemessener Faktor {1.0 / ratio:.2f}).")
    else:
        hints.append(f"Header-Kippung < 10°, Lx/Ly = {ratio:.2f} ist nicht eindeutig: "
                     "unklar, manuell prüfen.")
    return Verdict(BEFUND_UNCLEAR, None, 1.0, measured, hints)


# --------------------------------------------------------------------------
# File access
# --------------------------------------------------------------------------

def _scalar(ds) -> Optional[float]:
    try:
        a = np.asarray(ds[()]).reshape(-1)
        return float(a[0]) if a.size else None
    except Exception:
        return None


def _vector(ds) -> Optional[List[float]]:
    try:
        return [float(x) for x in np.asarray(ds[()]).reshape(-1)]
    except Exception:
        return None


def _hval(header, name: str) -> Optional[float]:
    return _scalar(header[name]) if header is not None and name in header else None


def _measurements(f) -> List[str]:
    """Top-level measurement groups ("1", "2", ...), numeric order."""
    keys = [k for k in f.keys() if str(k).isdigit()]
    return sorted(keys, key=int)


def _tilt_of(group) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """(holder tilt deg, stage tilt deg, which header) -- first header that has
    a ``Tilt Angle``: Electron Image, then EBSD, then EDS."""
    for area in ("Electron Image", "EBSD", "EDS"):
        h = group.get(f"{area}/Header")
        if h is None or "Tilt Angle" not in h:
            continue
        holder = _hval(h, "Tilt Angle")
        stage = _hval(h, "Stage Position/Tilt")
        return (math.degrees(holder) if holder is not None else None,
                math.degrees(stage) if stage is not None else None, area)
    return None, None, None


def _image_datasets(group) -> List[Tuple[str, object]]:
    """Every image dataset under ``Electron Image/Data`` as (name, dataset),
    name relative to Data (e.g. ``SE/Elektronenbild 33``)."""
    import h5py
    data = group.get("Electron Image/Data")
    out: List[Tuple[str, object]] = []
    if data is None:
        return out

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out.append((name, obj))
    data.visititems(visit)
    return out


def _as_image(ds, nx: Optional[int], ny: Optional[int]) -> np.ndarray:
    a = np.asarray(ds[()])
    if a.ndim == 1:
        if not nx or not ny or a.size != nx * ny:
            raise ValueError(f"flat image of {a.size} values does not fit {ny} x {nx}")
        a = a.reshape(int(ny), int(nx))      # X runs fastest
    elif a.ndim == 2 and a.shape[1] in (3, 4) and nx and ny and a.shape[0] == nx * ny:
        a = a.reshape(int(ny), int(nx), a.shape[1])
    return a


def _channel_of(name: str) -> str:
    head = name.split("/", 1)[0].upper()
    if head.startswith("FSD") or head.startswith("FSE"):
        return "FSD"
    if head.startswith("SE"):
        return "SE"
    if head.startswith("BSE"):
        return "BSE"
    return head or "?"


def _correction(befund: str, faktor: float, nx: int, ny: int, step: Optional[float]):
    if befund == BEFUND_X and faktor > 1:
        return (f"X × {faktor:.2f}, Y × 1", [int(round(nx * faktor)), int(ny)])
    if befund == BEFUND_Y and faktor > 1:
        return (f"X × 1, Y × {faktor:.2f}", [int(nx), int(round(ny * faktor))])
    return ("keine", [int(nx), int(ny)])


def _r(x, nd=4):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def check_h5oina(path: str, measurement: Optional[str] = None) -> Dict:
    """Run the full check (steps 1-6) on one ``.h5oina`` file.

    ``measurement`` limits it to one top-level group ("1", "2" ...); default is
    every measurement in the file (with more than one, image names carry the
    group as prefix)."""
    import h5py

    out: Dict = {
        "datei": os.path.basename(str(path)),
        "kippung_header_grad": None,
        "tischkippung_grad": None,
        "faktor_k_theoretisch": None,
        "bilder": [],
        "karten": [],
        "hinweise": [],
    }
    with h5py.File(str(path), "r") as f:
        groups = [measurement] if measurement else _measurements(f)
        groups = [g for g in groups if g in f]
        if not groups:
            out["hinweise"].append("Keine Messgruppe (1, 2, …) in der Datei.")
            out["befund"] = BEFUND_UNCLEAR
            out["zusammenfassung"] = "Verzerrung unklar: keine Messgruppe gefunden."
            return out
        multi = len(groups) > 1
        se_verdicts: List[Verdict] = []
        for gname in groups:
            g = f[gname]
            prefix = f"{gname}/" if multi else ""
            holder, stage, src = _tilt_of(g)
            if holder is None:
                out["hinweise"].append(f"{prefix or 'Messung'}: kein 'Tilt Angle' im Header.")
                tilt = 0.0
            else:
                # The effective tilt is the holder pre-tilt plus the stage tilt.
                tilt = holder + (stage or 0.0)
            if out["kippung_header_grad"] is None and holder is not None:
                out["kippung_header_grad"] = round(holder, 2)
                out["tischkippung_grad"] = round(stage, 2) if stage is not None else None
                out["kippung_wirksam_grad"] = round(tilt, 2)
                if tilt >= TILT_SUSPECT_DEG:
                    out["faktor_k_theoretisch"] = round(stretch_factor(tilt), 2)
                else:
                    out["faktor_k_theoretisch"] = 1.0
            # Step 2: noted, never decisive.
            for area in ("Electron Image", "EBSD"):
                h = g.get(f"{area}/Header")
                if h is None:
                    continue
                noted = {k: _hval(h, k) for k in ("Tilt Axis", "Scan Rotation") if k in h}
                if noted:
                    out.setdefault("header_richtungsangaben", {})[f"{prefix}{area}"] = {
                        k: _r(v) for k, v in noted.items()}

            ih = g.get("Electron Image/Header")
            img_nx = int(_hval(ih, "X Cells") or 0) or None
            img_ny = int(_hval(ih, "Y Cells") or 0) or None
            img_step = _hval(ih, "X Step")
            img_step_y = _hval(ih, "Y Step")
            if img_step is not None and img_step_y is not None and img_step > 0 \
                    and abs(img_step - img_step_y) / img_step > 1e-3:
                out["hinweise"].append(
                    f"{prefix}Elektronenbild: X Step ({img_step:.4g}) ≠ Y Step ({img_step_y:.4g}) – "
                    "möglicherweise ist die Tilt-Korrektur schon angewendet.")

            image_verdicts: List[Tuple[str, Verdict]] = []
            for name, ds in _image_datasets(g):
                entry: Dict = {"name": prefix + name, "kanal": _channel_of(name)}
                try:
                    img = _as_image(ds, img_nx, img_ny)
                except Exception as exc:  # noqa: BLE001 -- report, never fail the file
                    entry.update(befund=BEFUND_UNCLEAR, fehler=str(exc))
                    out["bilder"].append(entry)
                    continue
                ny_, nx_ = img.shape[:2]
                meas = measure_anisotropy(img)
                v = reconcile(tilt, meas.lx_ly_median, channel=entry["kanal"])
                corr, target = _correction(v.befund, v.faktor, nx_, ny_, img_step)
                entry.update({
                    "pixel": [int(nx_), int(ny_)],
                    "step_nominal_um": _r(img_step),
                    "lx_ly_median": _r(meas.lx_ly_median, 3),
                    "kacheln_verwendet": meas.tiles_used,
                    "befund": v.befund,
                    "sicherheit": v.sicherheit,
                    "faktor_gemessen": v.faktor_gemessen,
                    "korrektur": corr,
                    "um_pro_px_nach_korrektur": ([_r(img_step), _r(img_step)]
                                                 if img_step is not None else None),
                    "zielgroesse_px": target,
                })
                if meas.reason:
                    entry["messhinweis"] = meas.reason
                if v.hinweise:
                    entry["hinweise"] = v.hinweise
                out["bilder"].append(entry)
                image_verdicts.append((entry["kanal"], v))

            se = [v for ch, v in image_verdicts if ch == "SE"]
            if not se:
                se = [v for ch, v in image_verdicts if ch not in ("FSD",)]
            if se:
                se_verdicts.extend(se)
            elif image_verdicts:
                out["hinweise"].append(f"{prefix}Nur FSD-Bilder vorhanden: keine Entscheidung allein "
                                       "anhand von FSD.")

            # Layered images: not measured (overlays of maps), they follow the file finding.
            for lname in (g.get("Layered Image") or {}).keys() if "Layered Image" in g else []:
                lh = g.get(f"Layered Image/{lname}/Header")
                lx = int(_hval(lh, "X Cells") or 0) if lh is not None else 0
                ly = int(_hval(lh, "Y Cells") or 0) if lh is not None else 0
                out["bilder"].append({
                    "name": f"{prefix}Layered Image/{lname}", "kanal": "Schichtbild",
                    "pixel": [lx, ly] if lx and ly else None,
                    "step_nominal_um": _r(_hval(lh, "X Step")) if lh is not None else None,
                    "lx_ly_median": None, "befund": None,
                    "hinweis": "nicht gemessen; erbt den Befund des Elektronenbilds",
                })

            # Step 5: maps.
            for area in ("EBSD", "EDS"):
                mh = g.get(f"{area}/Header")
                if mh is None:
                    continue
                mx = int(_hval(mh, "X Cells") or 0)
                my = int(_hval(mh, "Y Cells") or 0)
                step = _hval(mh, "X Step")
                rs = _vector(mh["Relative Size"]) if "Relative Size" in mh else None
                karte: Dict = {"typ": area, "raster": [mx, my], "step_nominal_um": _r(step)}
                if rs and len(rs) >= 2 and img_nx and img_ny and mx and my:
                    px = [rs[0] * img_nx / mx, rs[1] * img_ny / my]
                    karte["px_pro_schritt_im_bild"] = [round(px[0], 2), round(px[1], 2)]
                    mean = (px[0] + px[1]) / 2.0
                    karte["erbt_bildverzerrung"] = bool(mean > 0 and abs(px[0] - px[1]) / mean
                                                        < SAME_GRID_TOL)
                else:
                    karte["px_pro_schritt_im_bild"] = None
                    karte["erbt_bildverzerrung"] = None
                if area == "EBSD" and "Hit Rate" in mh:
                    hr = _hval(mh, "Hit Rate")
                    karte["hit_rate_prozent"] = _r(hr, 2)
                    if hr is not None and hr == 0:
                        out["hinweise"].append(f"{prefix}EBSD-Hit-Rate 0 %: Phasenkarte nicht aussagekräftig.")
                out["karten"].append(karte)

    # File verdict: from the SE images (step 3 decides on SE).
    if se_verdicts:
        befunde = {v.befund for v in se_verdicts}
        if len(befunde) == 1:
            file_v = se_verdicts[0]
            befund = file_v.befund
            faktor = max(v.faktor for v in se_verdicts)
        else:
            befund, faktor = BEFUND_UNCLEAR, 1.0
            out["hinweise"].append("Die SE-Bilder der Datei widersprechen sich: "
                                   + ", ".join(sorted(befunde)) + ".")
    else:
        befund, faktor = BEFUND_UNCLEAR, 1.0
        if not out["bilder"]:
            out["hinweise"].append("Kein Elektronenbild in der Datei: Richtung nicht messbar.")

    for karte in out["karten"]:
        step = karte.get("step_nominal_um")
        if step is None:
            karte["wahrer_schritt_um"] = None
        elif karte.get("erbt_bildverzerrung") and befund == BEFUND_X:
            karte["wahrer_schritt_um"] = [round(step * faktor, 4), step]
        elif karte.get("erbt_bildverzerrung") and befund == BEFUND_Y:
            karte["wahrer_schritt_um"] = [step, round(step * faktor, 4)]
        else:
            karte["wahrer_schritt_um"] = [step, step]

    for e in out["bilder"]:
        if e.get("kanal") == "Schichtbild":
            e["befund"] = befund

    if befund in (BEFUND_X, BEFUND_Y) and faktor > 1:
        out["hinweise"].append(
            "Korrektur gilt streng nur für die Grundebene; bei starkem Relief werden aufragende "
            "Strukturen in der Breite überzeichnet. Bilder mit Lanczos, Karten mit Nearest "
            "Neighbour strecken; der Maßstabsbalken gilt danach horizontal.")
    out["befund"] = befund
    out["faktor_korrektur"] = round(faktor, 2)
    out["zusammenfassung"] = summary_sentence(befund, faktor)
    return out


def summary_sentence(befund: str, faktor: float) -> str:
    k = f"{faktor:.2f}".replace(".", ",")
    if befund == BEFUND_X:
        return f"Verzerrt: ja, X gestaucht, Faktor {k}."
    if befund == BEFUND_Y:
        return f"Verzerrt: ja, Y gestaucht, Faktor {k}."
    if befund == BEFUND_OK:
        return "Verzerrt: nein."
    return "Verzerrt: unklar, manuell prüfen."


@lru_cache(maxsize=16)
def _cached(path: str, mtime_ns: int, size: int, measurement: Optional[str]) -> Dict:
    return check_h5oina(path, measurement)


def check_h5oina_cached(path: str, measurement: Optional[str] = None) -> Dict:
    """:func:`check_h5oina`, cached per (path, mtime, size): the load answer and
    ``/ebsd/info`` can both ask without measuring twice."""
    st = os.stat(path)
    return _cached(os.path.abspath(path), st.st_mtime_ns, st.st_size, measurement)


def is_h5oina(path: str) -> bool:
    return str(path).lower().endswith(".h5oina")


__all__ = [
    "check_h5oina", "check_h5oina_cached", "measure_anisotropy", "classify_ratio",
    "reconcile", "stretch_factor", "tile_correlation_lengths", "is_h5oina",
    "BEFUND_X", "BEFUND_Y", "BEFUND_OK", "BEFUND_UNCLEAR",
]
