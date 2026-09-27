"""Band-contrast mixture model, as an Orienta add-on.

This is the core's ``analysis/bc_analysis.py`` rebuilt against the add-on
contract, and it exists to prove the contract rather than to replace the core
version -- both ship. If a change to the contract breaks this file, the
contract changed.

It also does the one thing the core version's own docstring asks for and its
code forbids: the component count is a parameter, not the constant 3. The core
fixes it because ``BCGMMResult`` has hard-coded ``bc_low``/``bc_mid``/``bc_high``
fields; the contract's outputs are a list, so nothing here has to know the
count in advance.

WHAT THIS REBUILD DOES **NOT** CARRY, said here rather than left out
--------------------------------------------------------------------
``analysis/bc_analysis.py`` has three entry points. Two of them are below.
The third, ``grain_average_bc``, reads ``grains.grain_ids`` and
``grains.n_grains`` -- and ``AddonContext`` carries no grains. It could not be
faked from what the context DOES carry either: reconstructing grains means
misorientations between neighbouring pixels *under the crystal's point group*,
and the symmetry operators live in orix, which an add-on must not import (that
is the dependency claim this example exists to make true). Computing raw
axis-angle differences without symmetry would produce grain boundaries where
there are none. So it is absent, and named. See the README.

WHAT IT CHANGES ON PURPOSE
--------------------------
The core histogram is nailed to the range 0-230 with a width of 10, which are
native band-contrast numbers. ``context.quality`` may instead be Orienta's
COMPUTED pattern quality, which lives in 0-1 -- the same fixed range would put
every pixel of such a scan in the first bin and report it as a result. The
range here is taken from the data, and the bin width may be chosen from the
data too.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.mixture import GaussianMixture

from backend.api.services.addons.outputs import (
    MapOutput,
    ScalarOutput,
    TableOutput,
)

#: Below this many usable pixels per component the fit is arithmetic rather
#: than a measurement. The core refuses under 10 values for its fixed three.
MIN_PIXELS_PER_COMPONENT = 10

#: The core's fit settings, kept so the two implementations can be compared on
#: the same scan. ``spherical`` is one variance per component, which is what
#: the MATLAB original fits; the seed makes a run reproducible, which matters
#: for something whose numbers end up in a methods paragraph.
COVARIANCE_TYPE = "spherical"
REG_COVAR = 1e-6
MAX_ITER = 2000
N_INIT = 10
RANDOM_STATE = 42

#: Guard against a pathological ``bin_width`` turning a histogram into an
#: out-of-memory event. A user asking for 0.001 on a 0-255 map means it, but
#: not at a quarter of a million rows in a table the browser renders.
MAX_BINS = 2000


def _positive_int(value, name: str) -> int:
    """A count the caller sent, or a sentence saying why it is not one.

    Nothing between the user and here validates this. ``params_schema`` is
    read by no validator today -- the runner filters parameters by NAME and
    passes the values through -- so ``minimum``/``maximum`` in the manifest
    are documentation for a form that does not exist yet. An add-on that
    leans on them hands its user a scikit-learn traceback instead of a
    sentence.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"{name} must be a whole number, not {value!r} "
            f"({type(value).__name__})")
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be at least 1, not {number}")
    return number


def _non_negative_float(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise ValueError(
            f"{name} must be a number, not {value!r} ({type(value).__name__})")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(
            f"{name} must be a finite number of zero or more, not {number!r}. "
            "Zero means 'choose the width from the data'.")
    return number


def _freedman_diaconis_width(values: np.ndarray) -> float:
    """A bin width taken from the data, by a stated rule and not by taste.

    ``2 * IQR / n**(1/3)``. It is here so that ``bin_width = 0`` has an
    answer for a quality map whose whole range is smaller than one band-
    contrast bin, without this file guessing at which metric it was handed.
    A degenerate spread (every value equal, or an IQR of zero on a heavily
    tied map) has no width to derive, so the caller is told to state one
    rather than given a number invented here.
    """
    q1, q3 = np.percentile(values, [25, 75])
    iqr = float(q3 - q1)
    if iqr <= 0:
        spread = float(values.max() - values.min())
        if spread <= 0:
            raise ValueError(
                "every usable pixel has the same pattern-quality value, so "
                "there is no distribution to bin or to decompose")
        # A flat-topped or heavily tied distribution: fall back to the full
        # spread over a fixed number of bins, which is stated, not guessed.
        return spread / 32.0
    return 2.0 * iqr / float(values.size) ** (1.0 / 3.0)


def _bin_edges(values: np.ndarray, bin_width: float) -> np.ndarray:
    lo = float(values.min())
    hi = float(values.max())
    width = bin_width if bin_width > 0 else _freedman_diaconis_width(values)
    if width <= 0:
        raise ValueError(
            f"a bin width of {width!r} cannot divide a range of "
            f"{hi - lo!r}; state a positive bin_width")
    n_bins = int(math.ceil((hi - lo) / width)) or 1
    if n_bins > MAX_BINS:
        raise ValueError(
            f"a bin width of {width:g} would make {n_bins} bins over the "
            f"range {lo:g} to {hi:g}; at most {MAX_BINS} are produced. Ask "
            "for a wider bin, or 0 to choose one from the data.")
    return lo + width * np.arange(n_bins + 1, dtype=float)


def _mixture_mass(edges: np.ndarray, means: np.ndarray, stds: np.ndarray,
                  weights: np.ndarray) -> np.ndarray:
    """Probability the fitted mixture puts in each bin.

    Exact, through the normal CDF, rather than density-at-the-centre times
    width: the approximation is wrong by a visible amount exactly where the
    histogram is most interesting, at a narrow, tall component. ``math.erf``
    is the standard library, so this costs no dependency.
    """
    def cdf(x: float) -> float:
        total = 0.0
        for mean, std, weight in zip(means, stds, weights):
            if std <= 0:
                total += float(weight) * (1.0 if x >= mean else 0.0)
                continue
            z = (x - float(mean)) / (float(std) * math.sqrt(2.0))
            total += float(weight) * 0.5 * (1.0 + math.erf(z))
        return total

    cumulative = np.array([cdf(float(edge)) for edge in edges], dtype=float)
    return np.diff(cumulative)


def analyse(context, n_components: int = 3, bin_width: float = 10.0):
    """Decompose a scan's pattern-quality distribution into Gaussians.

    ``context.report(...)`` is called below and is a no-op on a real run
    today: Orienta's route installs no progress sink, and the calls are
    discarded. They are here because the call is part of the contract an
    author writes against, not because anything is listening yet.
    """
    quality = context.quality
    if quality is None:
        raise ValueError(
            "this analysis needs a pattern-quality map, and this result has "
            "none -- load a scan with band contrast, or compute pattern "
            "quality first")

    n_components = _positive_int(n_components, "n_components")
    bin_width = _non_negative_float(bin_width, "bin_width")

    grid = np.asarray(quality, dtype=float)
    usable = np.isfinite(grid)
    values = grid[usable]
    if values.size < n_components * MIN_PIXELS_PER_COMPONENT:
        raise ValueError(
            f"{values.size} usable pixels is too few for {n_components} "
            f"components; at least {MIN_PIXELS_PER_COMPONENT} per component "
            "are needed for the fit to be a measurement rather than "
            "arithmetic")

    # A mixture asked for more components than the data has distinct values
    # does not fail and does not refuse: it returns the extra components with
    # mean 0.0 and weight 0.0, and reports ``converged_ = True``. Measured on
    # a constant band-contrast map with the defaults:
    #
    #     means [50. 0. 0.] · weights [1. 0. 0.] · converged_ True
    #
    # Those zeros would land in the components table, in the scalars and in
    # the provenance trail as findings -- a scientific tool inventing two
    # populations at BC 0 that no pixel of the scan is anywhere near.
    # scikit-learn does emit a ConvergenceWarning naming it, and the add-on
    # contract has no channel that carries a warning, so the refusal has to
    # be here.
    n_distinct = int(np.unique(values).size)
    if n_distinct < n_components:
        raise ValueError(
            f"this map has only {n_distinct} distinct pattern-quality "
            f"value(s), which cannot support {n_components} components: the "
            f"fit would return {n_components - n_distinct} component(s) of "
            "zero weight at value 0 and report them as findings. Ask for at "
            f"most {n_distinct} component(s), or accept that this map has no "
            "distribution to decompose.")

    context.report("building the histogram", 0.2)
    edges = _bin_edges(values, bin_width)
    counts, edges = np.histogram(values, bins=edges)
    total = int(counts.sum())
    probability = counts / total if total else np.zeros_like(counts, float)

    context.report("fitting the mixture", 0.6)
    model = GaussianMixture(
        n_components=n_components, covariance_type=COVARIANCE_TYPE,
        reg_covar=REG_COVAR, max_iter=MAX_ITER, n_init=N_INIT,
        random_state=RANDOM_STATE)
    model.fit(values.reshape(-1, 1))

    # ---------------------------------------------------------------- ranks
    # Components come back in NO defined order, so every number that names a
    # component -- the scalars, the table rows AND the per-pixel map -- is put
    # through this one permutation. Sorting the scalars while leaving the map
    # on scikit-learn's own labels is the silent version of this bug: right
    # shape, right dtype, right value range, right label, wrong population.
    means = model.means_.ravel()
    order = np.argsort(means)
    rank_of_component = np.empty(n_components, dtype=int)
    rank_of_component[order] = np.arange(n_components)

    means = means[order]
    stds = np.sqrt(np.asarray(model.covariances_).ravel())[order]
    weights = np.asarray(model.weights_).ravel()[order]

    posterior = model.predict_proba(values.reshape(-1, 1))
    ranked = rank_of_component[np.argmax(posterior, axis=1)]

    # NaN, not -1 and not 0: a pixel with no pattern-quality value belongs to
    # no component, and a sentinel inside the value range would be drawn as
    # the lowest-quality population.
    label_map = np.full(grid.shape, np.nan, dtype=float)
    label_map[usable] = ranked.astype(float)
    confidence_map = np.full(grid.shape, np.nan, dtype=float)
    confidence_map[usable] = posterior.max(axis=1)

    context.report("assembling the outputs", 0.9)
    fitted = _mixture_mass(edges, means, stds, weights)
    outputs = [
        TableOutput(
            key="histogram",
            label="Pattern-quality histogram",
            columns=("bin_start", "bin_end", "count", "probability",
                     "fitted_probability"),
            rows=[(float(edges[i]), float(edges[i + 1]), int(counts[i]),
                   float(probability[i]), float(fitted[i]))
                  for i in range(len(counts))],
        ),
        TableOutput(
            key="components",
            label="Fitted components, lowest quality first",
            columns=("rank", "mean", "std", "area_fraction", "n_px_assigned"),
            rows=[(int(rank), float(means[rank]), float(stds[rank]),
                   float(weights[rank]), int(np.count_nonzero(ranked == rank)))
                  for rank in range(n_components)],
        ),
        # Where the facts that are not numbers go. A scalar output is a
        # float, so the quality source -- native band contrast or Orienta's
        # computed pattern quality, which are different measurements -- has
        # nowhere else in this contract to be said. See the README.
        TableOutput(
            key="fit_summary",
            label="What was fitted, and to what",
            columns=("quality_source", "n_components", "n_px_used",
                     "n_distinct_values", "bin_width_used", "value_min",
                     "value_max", "converged", "covariance_type", "n_init",
                     "max_iter", "reg_covar", "random_seed"),
            rows=[(context.quality_source, int(n_components),
                   int(values.size), n_distinct,
                   float(edges[1] - edges[0]),
                   float(values.min()), float(values.max()),
                   bool(model.converged_), COVARIANCE_TYPE, int(N_INIT),
                   int(MAX_ITER), float(REG_COVAR), int(RANDOM_STATE))],
        ),
        MapOutput(
            key="component_map",
            label="Mixture component (rank; 0 is the lowest-quality one)",
            values=label_map,
            vmin=0.0,
            vmax=float(n_components - 1),
        ),
        MapOutput(
            key="assignment_probability",
            label="Probability of the component this pixel was assigned to",
            values=confidence_map,
            vmin=0.0,
            vmax=1.0,
        ),
        # Named to match the sentence's {n_px} slot: the runner merges scalar
        # outputs into the recorded params so a sentence can state what the
        # analysis FOUND, not only what it was asked to do.
        ScalarOutput(key="n_px", label="Pixels used", value=float(values.size)),
        # NOT "n_components" as a scalar as well. That was this add-on's local
        # patch for a defect in the runner -- a parameter left at its default
        # was never recorded, so the sentence read "[n_components not
        # recorded]" on the most ordinary call there is. The runner now merges
        # declared defaults into the recorded params itself, which is the only
        # place it can work: a string or boolean parameter has no ScalarOutput
        # form, so the local patch was advice no author could follow in
        # general. Left here as the note that the gap was real and where it
        # went.
        #
        # The seed IS recorded, because it changes the numbers. Every other
        # fit setting is a module constant and is reported in fit_summary.
        ScalarOutput(key="random_seed", label="Random seed",
                     value=float(RANDOM_STATE)),
    ]
    for rank in range(n_components):
        # No ``unit``: the quality map is native band contrast (0-255,
        # dimensionless) on one scan and a computed 0-1 index on another, and
        # this add-on is not told which. Naming a unit it cannot know would
        # be the guess the contract forbids; ``fit_summary`` says which
        # measurement it was instead.
        outputs.append(ScalarOutput(
            key=f"component_{rank}_mean",
            label=f"Component {rank + 1} mean",
            value=float(means[rank])))
        outputs.append(ScalarOutput(
            key=f"component_{rank}_std",
            label=f"Component {rank + 1} standard deviation",
            value=float(stds[rank])))
        outputs.append(ScalarOutput(
            key=f"component_{rank}_area_fraction",
            label=f"Component {rank + 1} area fraction",
            value=float(weights[rank])))
    context.report("done", 1.0)
    return outputs
