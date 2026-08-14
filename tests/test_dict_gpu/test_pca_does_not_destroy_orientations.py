"""PCA truncation must not be enabled for speed it does not buy.

`run_dictionary_index` used to auto-enable a 256-component PCA for any
selection of >= 2000 patterns — i.e. every full map. Measured through this
function on LoGainNi (7440 px, 2 deg grid, 100,347 orientations), against
Hough as ground truth (Hough median fit 0.27 deg on this dataset):

    setting            median disorientation   <2 deg   reported NCC   seconds
    PCA off                       0.95 deg      95.9 %      0.469        37.5
    PCA k=256                     1.06 deg      93.9 %      0.603        39.0
    PCA k=1024                    0.94 deg      97.3 %      0.564        75.4

Two things follow, and only these two:

1. It buys no speed here (39.0 s vs 37.5 s), so auto-enabling it on selection
   size alone is a cost with no benefit. It stays available for the case it was
   built for — a dictionary that does not fit in VRAM.
2. The reported score is not comparable across settings. In a truncated
   subspace the correlation is dominated by structure common to every Ni
   pattern, so k=256 reports 0.603 where the full correlation gives 0.469. A
   user reading NCC as match quality is misled by a number that went UP because
   information was thrown away.

An earlier version of this file claimed k=256 randomised the orientation field
(28 deg neighbour misorientation). That claim came from HiGainNi, whose
patterns are too noisy for template matching under any setting, and from a
neighbour-misorientation metric that reduced symmetry on one side only, which
over-states disagreement between cubic orientation fields. Re-measured against
Hough on data that template matching can actually index, the orientation field
survives k=256 essentially intact. The defaults below are still the right ones,
for the two reasons above — not for that one.
"""
import inspect

import pytest

from backend.dict_gpu.pipeline import indexer as indexer_mod


def test_pca_is_not_auto_enabled_for_speed_alone():
    """A lossy approximation may buy memory, never speed.

    The old code enabled PCA whenever the selection reached 2000 patterns,
    which is every full map, so every full-map run was silently approximated
    for a speed win that the measurement above does not show.
    """
    src = inspect.getsource(indexer_mod.run_dictionary_index)
    assert "PCA_PAYOFF_MIN_PATTERNS" in src, "threshold constant vanished"
    ns: dict = {}
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("PCA_PAYOFF_MIN_PATTERNS"):
            exec(s, {"float": float}, ns)
            break
    assert "PCA_PAYOFF_MIN_PATTERNS" in ns, "could not read the threshold"
    # inf => never triggered by selection size alone. Any finite value must at
    # least be far above a normal map, or full maps get approximated again.
    assert ns["PCA_PAYOFF_MIN_PATTERNS"] > 100_000, (
        f"PCA auto-enables for speed at {ns['PCA_PAYOFF_MIN_PATTERNS']} patterns — "
        "that is a normal map size, and the measurement shows no speed win there"
    )


def test_memory_pressure_still_allows_pca():
    """Without it a dictionary that does not fit in VRAM cannot run at all."""
    src = inspect.getsource(indexer_mod.run_dictionary_index)
    assert "need_pca_for_memory" in src
    assert "need_pca_for_memory or worth_pca_for_speed" in src


def test_component_count_keeps_most_of_the_variance():
    """When memory pressure does force PCA, keep enough that the reported score
    stays close to the full correlation: k=1024 reports 0.564 against 0.469
    full, k=256 reports 0.603."""
    sig = inspect.signature(indexer_mod.run_dictionary_index)
    k = sig.parameters["pca_components"].default
    assert k >= 512, f"pca_components={k} distorts the reported score too far"


def test_caller_can_still_force_pca_either_way():
    src = inspect.getsource(indexer_mod.run_dictionary_index)
    assert "PCA forced ON by caller" in src
    assert "PCA forced OFF by caller" in src


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
