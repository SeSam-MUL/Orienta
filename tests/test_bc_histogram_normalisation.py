"""The BC histogram of the workbook is normalised as MATLAB ``histcounts(..., "Normalization", "probability")``
(EBSDanalysis_frame.m:269): by the number of finite values, not by the number that fall inside the bins."""
from types import SimpleNamespace

import numpy as np
import pytest


def test_the_bc_histogram_is_divided_by_all_finite_values_not_by_those_in_range():
    from analysis.bc_analysis import calculate_bc_histogram
    bc = np.concatenate([np.full(60, 15.0), np.full(20, 105.0), np.full(20, 250.0),     # 20 % above the last edge
                         np.full(7, np.nan), np.full(3, np.inf)])
    res = calculate_bc_histogram(SimpleNamespace(bc=bc))
    assert res.counts[1] == pytest.approx(0.60) and res.counts[10] == pytest.approx(0.20)
    assert res.counts.sum() == pytest.approx(0.80)                  # not 1.0: the 20 % above 230 are in no bin
    assert len(res.counts) == 23
    # a map inside the range sums to one
    inside = calculate_bc_histogram(SimpleNamespace(bc=np.linspace(0, 229.9, 1000)))
    assert inside.counts.sum() == pytest.approx(1.0)
    # no finite value: all counts are 0
    assert calculate_bc_histogram(SimpleNamespace(bc=np.full(5, np.nan))).counts.sum() == 0.0
