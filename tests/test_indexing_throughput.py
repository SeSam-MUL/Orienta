"""Every indexing method reports its throughput, in the same words.

Only Spherical did. Hough reported nothing, the GPU dictionary path reported
per-phase timings but no rate, and the CPU path's rate — kikuchipy prints
``Indexing speed: 2026.19698 patterns/s`` itself — went to the Python log and
never reached the app.

The point of one wording is that the numbers can be compared. Measured on
LoGainNi, 3720 px, in one run:

    Hough:      3,720 patterns in 0.9 s ·  4,007 pat/s (single thread)
    Dictionary: 3,720 patterns in 0.2 s · 17,002 pat/s (GPU, 100,347 entries)
    Dictionary: 3,720 patterns in 1.4 s ·  2,676 pat/s (CPU, 11,935 entries)

All three time the MATCHING step only — not the file read, not the dictionary
projection, not the CrystalMap build. Folding those in would make a fast
method on a big dictionary look slow for the wrong reason.
"""
import inspect
import re

import pytest

import indexing_controller
from indexing_controller import indexing_rate_line


# --------------------------------------------------------------------------
# the shared sentence
# --------------------------------------------------------------------------
def test_rate_is_patterns_over_seconds():
    line = indexing_rate_line("Hough", 3720, 0.9285)
    assert "3,720 patterns" in line
    assert "0.9 s" in line
    m = re.search(r"([\d,]+) pat/s", line)
    assert m, line
    assert int(m.group(1).replace(",", "")) == round(3720 / 0.9285)


def test_device_detail_is_parenthesised():
    assert indexing_rate_line("Dictionary", 10, 1.0, "GPU, 100 entries").endswith(
        "(GPU, 100 entries)")
    # and omitted entirely when there is nothing to say
    assert indexing_rate_line("Hough", 10, 1.0).endswith("pat/s")


def test_thousands_separators_survive_large_runs():
    line = indexing_rate_line("Dictionary", 196608, 12.5, "GPU")
    assert "196,608 patterns" in line
    assert "15,729 pat/s" in line


def test_a_run_too_short_to_time_reports_no_rate():
    """Dividing by a sub-millisecond clock says more about the clock than the
    code, so report the count and the milliseconds instead."""
    line = indexing_rate_line("Dictionary", 4, 2e-5)
    assert "pat/s" not in line
    assert "ms" in line
    assert "4 patterns" in line


def test_zero_seconds_does_not_divide_by_zero():
    line = indexing_rate_line("Hough", 1, 0.0)
    assert "pat/s" not in line


@pytest.mark.parametrize("method", ["Hough", "Dictionary", "Spherical"])
def test_all_three_methods_produce_the_same_shape(method):
    line = indexing_rate_line(method, 1000, 2.0, "dev")
    assert line == f"{method}: 1,000 patterns in 2.0 s · 500 pat/s (dev)"


# --------------------------------------------------------------------------
# each path actually emits one
# --------------------------------------------------------------------------
def test_hough_times_the_matching_step_and_reports():
    src = inspect.getsource(indexing_controller.hough_index_patterns)
    assert "_rate_t0 = _t.perf_counter()" in src
    assert 'indexing_rate_line(\n        "Hough"' in src or 'indexing_rate_line(' in src
    # the timer must start before the indexer call, not around the whole function
    assert src.index("_rate_t0 = _t.perf_counter()") < src.index("if use_ray:")


def test_dictionary_cpu_reports_its_own_rate_and_forwards_kikuchipys():
    src = inspect.getsource(indexing_controller.dictionary_index_patterns)
    assert 'indexing_rate_line("Dictionary"' in src
    assert "Indexing speed:" in src, "kikuchipy's own figure is not forwarded"
    assert "Dictionary (kikuchipy)" in src


def test_dictionary_cpu_counts_the_projected_dictionary_not_the_master():
    """`dict_size` is read before a raw master is projected, so it is 2 — the
    two hemispheres. The entry count has to come from the dictionary that was
    actually searched."""
    src = inspect.getsource(indexing_controller.dictionary_index_patterns)
    assert "_n_entries = len(dictionary.data)" in src
    assert src.index("_n_entries = len(dictionary.data)") > src.index(
        "_dictionary_signal_from_master")


def test_gpu_path_reports_live_and_final():
    from backend.dict_gpu.pipeline import indexer as indexer_mod
    src = inspect.getsource(indexer_mod.run_dictionary_index)
    assert "_match_t0 = time.perf_counter()" in src
    assert "matching {_tile_i}/{_n_tiles} tiles" in src, "no live rate"
    assert "pat/s (GPU," in src, "no final rate"
    # at most one line a second, or a fast run buries the log
    assert "_now - _last_report >= 1.0" in src


def test_gpu_live_rate_uses_the_fraction_of_tiles_done():
    """Every experimental pattern is compared against every dictionary tile,
    so after k of n tiles the work done is n_sel * k/n — not n_sel."""
    from backend.dict_gpu.pipeline import indexer as indexer_mod
    src = inspect.getsource(indexer_mod.run_dictionary_index)
    assert "_done = n_sel * _tile_i / _n_tiles" in src


def test_spherical_uses_the_shared_sentence_too():
    src = inspect.getsource(indexing_controller.spherical_gpu_index_patterns)
    assert 'indexing_rate_line(\n            "Spherical"' in src
    # multi-phase counts every comparison, which is why it says "effective"
    assert "n_indexed * n_phases" in src


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
