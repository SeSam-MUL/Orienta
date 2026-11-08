"""Sum-spectrum fit: weighted NNLS inside a Nelder-Mead over four parameters.

``fit_sum_spectrum`` returns a :class:`FileFit` that later stages (per-pixel
fits, quantification) reuse: the non-linear parameters, the composition that
set the absorption term, and the pile-up shapes are fixed by the sum
spectrum, so pixels only solve the linear problem.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, nnls

from . import tables
from .cube import (DEFAULT_BLOCK, H5Like, _open, _spectrum, channel_blocks,
                   live_time, sum_spectrum)
from .model import (MIN_OVERVOLTAGE, MODEL_VERSION, NonlinearParams,
                    SpectrumMeta, SpectrumModel, fwhm, label, measured_pileup)

#: Elements whose K lines are too soft to enter the first-pass composition.
_SOFT = ("C", "N", "O")
N_PASSES = 3
START = NonlinearParams(0.0, 1.0, 0.13, 3e-4)


@dataclass
class FileFit:
    """Result of a sum-spectrum fit.

    ``A``, ``coef``, ``names``, ``mask`` and ``pile_cols`` / ``comp`` belong
    together: ``A`` is the design matrix built with exactly that composition
    and those pile-up shapes.
    """
    meta: SpectrumMeta
    families: List[Tuple[str, str]]
    params: NonlinearParams
    names: List[str]
    A: np.ndarray
    mask: np.ndarray
    coef: np.ndarray
    chi2_red: float
    comp: Dict[str, float]
    pile_cols: Dict[str, np.ndarray] = field(default_factory=dict)
    model_version: str = MODEL_VERSION


def _solve(model: SpectrumModel, y: np.ndarray, params: NonlinearParams,
           comp: Dict[str, float]):
    A, names, mask = model.design(params, comp)
    ym = y[mask]
    wt = 1 / np.sqrt(np.maximum(ym, 1))
    coef, _ = nnls(A * wt[:, None], ym * wt, maxiter=5000)
    fit = A @ coef
    dof = ym.size - len(coef) - 4
    chi2 = float(np.sum((ym - fit) ** 2 / np.maximum(ym, 1)) / dof)
    return chi2, coef, names, A, mask


def _composition(model: SpectrumModel, names: List[str], coef: np.ndarray
                 ) -> Dict[str, float]:
    """First-pass weight composition from the K lines (for self-absorption)."""
    from eds_utils import UnknownLineError, _emission_k
    areas = dict(zip(names, coef))
    comp: Dict[str, float] = {}
    for fam in model.families:
        el, f = fam
        if f != "K" or el in _SOFT:
            continue
        intensity = areas[label(fam)] * model.main_weight[fam]
        try:
            comp[el] = intensity * _emission_k(el, f, model.meta.beam_kv)
        except UnknownLineError:
            pass
    return {k: v for k, v in comp.items() if v > 0}


def fit_sum_spectrum(y: np.ndarray, meta: SpectrumMeta,
                     families: List[Tuple[str, str]]) -> FileFit:
    """Fit the sum spectrum ``y`` (counts per channel on ``meta``'s grid)."""
    y = np.asarray(y, dtype=float)
    if y.shape != (meta.n_channels,):
        raise ValueError(f"spectrum has shape {y.shape}, header says "
                         f"{meta.n_channels} channels")
    model = SpectrumModel(meta, families, pile_cols=measured_pileup(y, meta))
    p0 = START.as_array()
    comp: Dict[str, float] = {"Al": 1.0}

    def objective(p):
        if not (0.09 < p[2] < 0.2 and p[3] > 1e-6):
            return 1e30
        return _solve(model, y, NonlinearParams.from_array(p), comp)[0]

    result = None
    for _ in range(N_PASSES):
        # pass 2/3: composition-aware absorption and model-based pile-up
        r = minimize(objective, p0, method="Nelder-Mead", options=dict(
            xatol=1e-5, fatol=1e-3, maxiter=400,
            initial_simplex=[p0, p0 + [0.005, 0, 0, 0], p0 + [0, 0.002, 0, 0],
                             p0 + [0, 0, 0.01, 0], p0 + [0, 0, 0, 3e-4]]))
        p0 = r.x
        params = NonlinearParams.from_array(p0)
        chi2, coef, names, A, mask = _solve(model, y, params, comp)
        result = (params, chi2, coef, names, A, mask, dict(comp),
                  {k: v.copy() for k, v in model.pile_cols.items()})
        new_comp = _composition(model, names, coef)
        if sum(new_comp.values()) > 0:
            comp = new_comp
        model.rebuild_pileup(params, coef, names, A, mask)
    params, chi2, coef, names, A, mask, comp_used, pile_used = result
    return FileFit(meta=meta, families=[tuple(f) for f in families],
                   params=params, names=names, A=A, mask=mask, coef=coef,
                   chi2_red=chi2, comp=comp_used, pile_cols=pile_used)


def net_areas(ff: FileFit) -> Dict[str, float]:
    """Counts in the main line of every family, e.g. ``{"Fe K": ...}``."""
    cmap = dict(zip(ff.names, ff.coef))
    return {label(f): float(cmap[label(f)] * w)
            for f, w in _main_weights(ff).items()}


def _main_weights(ff: FileFit) -> Dict[Tuple[str, str], float]:
    from . import tables
    return {tuple(f): tables.main_line(*f)[1] for f in ff.families}


def forward(ff: FileFit, areas: Dict[str, float],
            params: NonlinearParams | None = None) -> np.ndarray:
    """Model spectrum on the full channel grid.

    Family blocks take their amplitude from ``areas`` (main-line counts per
    label; a family not listed contributes nothing); tails, pile-up and
    continuum keep the amplitudes in ``ff.coef``.  ``params`` overrides the
    fitted non-linear parameters.
    """
    params = ff.params if params is None else params
    model = SpectrumModel(ff.meta, ff.families, pile_cols=ff.pile_cols)
    A, names, mask = model.design(params, ff.comp)
    if names != ff.names:
        raise ValueError("FileFit columns do not match the model")
    weights = {label(f): w for f, w in _main_weights(ff).items()}
    coef = ff.coef.copy()
    for j, nm in enumerate(names):
        if nm in weights:
            coef[j] = areas.get(nm, 0.0) / weights[nm]
    out = np.zeros(ff.meta.n_channels)
    out[mask] = A @ coef
    return out


# ---------------------------------------------------------------------------
# per-pixel projection
# ---------------------------------------------------------------------------
def _scaled(Aa: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Unit-norm columns and their norms (the columns differ by many decades)."""
    norm = np.linalg.norm(Aa, axis=0)
    norm[norm == 0] = 1.0
    return Aa / norm, norm


def project_cube(h5file: H5Like, ff: FileFit, root: str = "1",
                 block: int = DEFAULT_BLOCK,
                 progress: Optional[Callable[[float], None]] = None,
                 y_sum: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Net main-line intensity of every family and pixel, counts per live second.

    The non-linear parameters, the composition and the active set (columns
    with ``coef > 0``) come from the sum-spectrum fit ``ff``; each pixel then
    needs one weighted linear projection ``P = (A'WA)^-1 A'W`` with
    ``W = 1 / max(sum spectrum, 1)`` (the sum fit's own weights; a constant
    factor cancels, so this reproduces its active-set solution exactly).  The cube is read in blocks of
    ``block`` channels for all pixels (it is chunked per channel).  ``y_sum``
    is the sum spectrum if the caller has it; otherwise it is read first.
    Values are float32, one flat array of ``n_px`` per family label, finite
    and NOT clipped: a weak line can come out negative in a noisy pixel, and
    only unclipped maps sum exactly to the sum-spectrum fit (live-time
    weighted sum over all pixels == ``net_areas``).  Clipping happens where
    the numbers are used: per-pixel quantification and display.  A pixel with
    no live time gets 0.
    """
    with _open(h5file) as f:
        ds = _spectrum(f, root)
        n_px, n_ch = ds.shape
        if n_ch != ff.meta.n_channels:
            raise ValueError(f"cube has {n_ch} channels, the fit was made on "
                             f"{ff.meta.n_channels}")
        lt = live_time(f, root)
        if lt.size != n_px:
            raise ValueError(f"{lt.size} live times for {n_px} pixels")
        first = 0.0
        if y_sum is None:
            y_sum = sum_spectrum(f, root, block, progress=(
                None if progress is None else lambda x: progress(0.5 * x)))
            first = 0.5
        keep = ff.coef > 0
        An, norm = _scaled(ff.A[:, keep])
        W = 1.0 / np.maximum(np.asarray(y_sum, dtype=float)[ff.mask], 1.0)
        AW = (An * W[:, None]).T
        try:
            P = np.linalg.solve(An.T @ AW.T, AW)
        except np.linalg.LinAlgError:
            P = np.linalg.lstsq(An.T @ AW.T, AW, rcond=None)[0]
        P /= norm[:, None]  # k x n_mask, in the original column units
        rank = np.cumsum(ff.mask) - 1  # channel -> row of A
        idx = np.flatnonzero(ff.mask)
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        C = np.zeros((n_px, int(keep.sum())))
        for c0, c1 in channel_blocks(n_ch, block, lo, hi):
            sel = ff.mask[c0:c1]
            if sel.any():
                Y = ds[:, c0:c1][:, sel].astype(np.float64)
                C += Y @ P[:, rank[c0:c1][sel]].T
            if progress is not None:
                progress(first + (1 - first) * (c1 - lo) / (hi - lo))
    col = {nm: k for k, nm in enumerate(np.asarray(ff.names)[keep])}
    live = lt > 0
    out: Dict[str, np.ndarray] = {}
    for fam, wgt in _main_weights(ff).items():
        nm = label(fam)
        net = np.zeros(n_px)
        if nm in col:
            net[live] = C[live, col[nm]] * wgt / lt[live]
        out[nm] = net.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# element detection
# ---------------------------------------------------------------------------
DETECT_Z = 5.0
#: An L/M family only counts as evidence for an element whose K is not excited,
#: and only when its main line is at least this hard (below it the soft-line
#: continuum shape and the window absorb too much to tell L from misfit), keV.
DETECT_L_MIN_KEV = 1.0


@dataclass
class Detection:
    """A family that stands out of the noise: main-line counts, sigma and z."""
    element: str
    family: str
    area: float
    sigma: float
    z: float


def _sigmas(ff: FileFit) -> np.ndarray:
    """Standard error of every column amplitude (0 for columns off the active set).

    Covariance of the weighted linear fit on the active set, ``pinv(A'WA)``
    with Poisson weights from the fitted spectrum, inflated by
    ``sqrt(max(chi2_red, 1))`` so that model misfit widens the error bars.
    """
    keep = ff.coef > 0
    sigma = np.zeros(len(ff.coef))
    if not keep.any():
        return sigma
    An, norm = _scaled(ff.A[:, keep])
    W = 1.0 / np.maximum(ff.A @ ff.coef, 1.0)
    cov = np.linalg.pinv(An.T @ (An * W[:, None]))
    sigma[keep] = (np.sqrt(np.clip(np.diag(cov), 0, None)) / norm
                   * np.sqrt(max(ff.chi2_red, 1.0)))
    return sigma


def significance(ff: FileFit) -> List[Detection]:
    """Area, sigma and z of every modelled family (``z = 0`` where sigma is 0)."""
    sig = dict(zip(ff.names, _sigmas(ff)))
    cmap = dict(zip(ff.names, ff.coef))
    out = []
    for fam, wgt in _main_weights(ff).items():
        nm = label(fam)
        area, s = float(cmap[nm] * wgt), float(sig[nm] * wgt)
        out.append(Detection(fam[0], fam[1], area, s, area / s if s > 0 else 0.0))
    return out


def _k_excited(element: str, beam_kv: float) -> bool:
    try:
        edge = tables.edge_kev(element, "K")
    except KeyError:
        return False
    return edge is not None and beam_kv / edge >= MIN_OVERVOLTAGE


def detect(ff: FileFit) -> List[Detection]:
    """Families the spectrum itself shows, whatever the instrument exported.

    A K family counts at ``z >= 5``.  An L or M family counts at ``z >= 5``
    only for an element whose K line the beam does not excite (overvoltage
    below :data:`~backend.eds_spectrum.model.MIN_OVERVOLTAGE`) and only when
    its main line is at least 1 keV: below that, or when K is available, L
    is too easily a misfit of the soft continuum to be taken as evidence.
    """
    found = []
    for d in significance(ff):
        if d.z < DETECT_Z:
            continue
        if d.family == "K":
            found.append(d)
        elif (not _k_excited(d.element, ff.meta.beam_kv)
              and tables.main_line(d.element, d.family)[0] >= DETECT_L_MIN_KEV):
            found.append(d)
    return found


# ---------------------------------------------------------------------------
# fit quality
# ---------------------------------------------------------------------------
QUALITY_FRACTION = 0.05
QUALITY_MIN_AREA = 0.01


@dataclass
class QualityWarning:
    """Residual of the sum fit around a family's main line, relative to its area."""
    family: str
    residual_fraction: float


def quality(ff: FileFit, y: np.ndarray) -> List[QualityWarning]:
    """Families whose main line the model does not reproduce.

    The residual ``y - model`` is summed over +-FWHM of each main line and
    divided by the family's net area; a family warns when that ratio exceeds
    5 % in absolute value and its area is more than 1 % of the largest.
    """
    y = np.asarray(y, dtype=float)
    resid = y[ff.mask] - ff.A @ ff.coef
    E = (ff.params.offset_kev + ff.params.gain * ff.meta.energies())[ff.mask]
    areas = net_areas(ff)
    biggest = max(areas.values(), default=0.0)
    out = []
    for fam in ff.families:
        area = areas[label(fam)]
        if area <= 0 or area <= QUALITY_MIN_AREA * biggest:
            continue
        c = tables.main_line(*fam)[0]
        win = np.abs(E - c) <= fwhm(c, ff.params.fwhm_mnka_kev)
        frac = float(resid[win].sum() / area)
        if abs(frac) > QUALITY_FRACTION:
            out.append(QualityWarning(label(fam), frac))
    return out
