"""Drift guard: run the real paths and assert what they recorded.

This is the test that makes the registry honest. It runs with
ORIENTA_CITATIONS_STRICT on (the default), so an undeclared key raises here.
"""
import pytest

from backend.api.services.citations.provenance import get_steps


def test_hough_run_records_its_step(indexed_result_hough):
    keys = [s["key"] for s in get_steps(indexed_result_hough)]
    assert "indexing.hough" in keys


def test_the_method_step_carries_the_app_version(indexed_result_hough):
    step = next(s for s in get_steps(indexed_result_hough)
                if s["key"] == "indexing.hough")
    assert step["params"].get("orienta_version")


def test_eds_prior_off_records_nothing(indexed_result_hough):
    """Absence is the record that it was off."""
    keys = [s["key"] for s in get_steps(indexed_result_hough)]
    assert "eds.chemistry_prior" not in keys


def test_merging_two_real_per_phase_hough_runs_carries_the_step(
    h5oina_path, al_cif_path,
):
    """The multi-phase merge sites (a 'compare phases' or per-phase EDS
    Hough split run) build a *fresh* IndexingResult from several real
    per-phase sub-results and call merge_provenance to carry their steps
    forward. Exercise that against two genuine hough_index_patterns runs,
    not stand-ins for them — both runs are the same real job on the same
    ROI, so both record indexing.hough with identical params, and the
    merge must dedupe to exactly one step (not zero, not two)."""
    pytest.importorskip("pyebsdindex")
    import numpy as np
    from orix.crystal_map import Phase, PhaseList

    from backend.api.services.citations.provenance import merge_provenance
    from ebsd_utils import sanitize_cif
    from indexing_controller import (
        IndexingConfig,
        IndexingMethod,
        IndexingResult,
        hough_index_patterns,
    )
    from safe_loader import load_ebsd_safe

    signal = load_ebsd_safe(str(h5oina_path), verbose=False)
    n_rows, n_cols = signal.data.shape[:2]
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    mask[18:30, 2:14] = True

    phase = Phase.from_cif(sanitize_cif(str(al_cif_path)))
    phase.name = "Al"
    phase_list = PhaseList(phase)

    sub_a = hough_index_patterns(
        signal, phase_list, signal.detector,
        IndexingConfig(method=IndexingMethod.HOUGH), mask,
    )
    sub_b = hough_index_patterns(
        signal, phase_list, signal.detector,
        IndexingConfig(method=IndexingMethod.HOUGH), mask,
    )

    merged = IndexingResult(
        xmap=sub_a.xmap, selection_mask=mask,
        original_shape=(n_rows, n_cols), method=IndexingMethod.HOUGH,
    )
    merge_provenance(merged, [sub_a, sub_b])

    keys = [s["key"] for s in get_steps(merged)]
    assert keys == ["indexing.hough"], (
        "two identical real Hough sub-results must dedupe to one step, "
        f"got {keys}"
    )
