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
