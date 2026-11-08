"""The reflector-count ladder the refusal and the one-click answer walk down.

Measured on the crystal library (band-triplet library, GiB): Al13Fe4 64 -> 0.75,
60 -> 0.54, 56 -> 0.31; beta-AlFeSi 64 -> 2.64, 60 -> 1.73, 56 -> 1.09, 50 -> 0.49.
A step of four rows near the top already buys what 50 -> 40 -> 32 gave, so the
ladder is fine there and stops at 40 rows / 4 effective families (PyEBSDIndex's own
defaults are 4 families for cubic and 8 for hexagonal phases). The floor was 32 and was
raised to 40 later, a conservative change: on the S phase (mmm) of a real aluminium-alloy map
32 rows gave 0.3 % of the reference pixels within 5 degrees, 40 rows 38.0 %.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ebsd_utils as eu  # noqa: E402
from tests.test_hough_reflectors import NI_CIF, _phase, _plist  # noqa: E402


def test_the_ladder_and_its_floor():
    assert eu._REFLECTOR_LADDER == (64, 60, 56, 50, 44, 40)
    assert list(eu._REFLECTOR_LADDER) == sorted(eu._REFLECTOR_LADDER, reverse=True)
    assert (eu.MIN_FIT_ROWS, eu.MIN_FIT_FAMILIES) == (40, 4)
    assert min(eu._REFLECTOR_LADDER) == eu.MIN_FIT_ROWS


def test_the_prediction_walks_the_ladder_below_the_full_list(tmp_path):
    ni, _ = _phase(tmp_path, NI_CIF, "Ni")

    class Recorder:
        def __init__(self):
            self.rows = []

        def get_indexer(self, phase_list, reflectors, **kw):
            self.rows.append(len(reflectors))

    class Refl:                       # 70 rows, the full default list's size
        hkl = np.arange(210, dtype=float).reshape(70, 3) + 1.0

    det = Recorder()
    out = eu.predict_triplet_library(det, _plist(ni), Refl())
    assert [c for c, _r, _b in out] == [70, 64, 60, 56, 50, 44, 40]
    assert det.rows == [70, 64, 60, 56, 50, 44, 40]


def test_a_row_limit_keeps_the_strongest_rows_not_the_first(tmp_path):
    """Al (fcc, 64 default rows, in diffsims' order). A limit of 21 rows built {200} and
    {220} and dropped the strongest {111}; it means the 21 strongest rows: {111} 8,
    {200} 6 and 7 of {220} (|F| of fcc falls with d: {111} > {200} > {220})."""
    from tests.test_hough_reflectors import AL_CIF, _RecordingDetector
    al, _ = _phase(tmp_path, AL_CIF, "Al")
    pl = _plist(al)
    refl = eu.prepare_reflectors(pl)
    det = _RecordingDetector()
    eu.create_indexer(det, pl, refl, max_reflectors=21)
    sent = det.calls[0][1]
    assert len(sent) == 21
    fams = [tuple(sorted(abs(int(round(x))) for x in r)) for r in sent]
    assert fams.count((1, 1, 1)) == 8 and fams.count((0, 0, 2)) == 6
    assert fams.count((0, 2, 2)) == 7
    # what nothing cuts keeps its order, row for row
    for limit in (None, 64, 500):
        det2 = _RecordingDetector()
        eu.create_indexer(det2, pl, refl, max_reflectors=limit)
        assert det2.calls[0][1] == refl.hkl.tolist()


def test_the_cost_table_of_the_row_limit_prices_the_strongest_rows_too(tmp_path):
    """predict_triplet_library trims like create_indexer, so the table the refusal shows
    prices the list a limit would really build."""
    from tests.test_hough_reflectors import AL_CIF, _RecordingDetector
    al, _ = _phase(tmp_path, AL_CIF, "Al")
    pl = _plist(al)
    refl = eu.prepare_reflectors(pl)
    det = _RecordingDetector()
    eu.predict_triplet_library(det, pl, refl, counts=(21,))
    rows = det.calls[0][1]
    assert len(rows) == 21
    assert sum(tuple(sorted(abs(int(round(x))) for x in r)) == (1, 1, 1) for r in rows) == 8


def test_the_floor_refuses_a_list_of_32_rows_that_would_have_fitted(tmp_path, monkeypatch):
    """Orthorhombic Cmcm (synthetic cell of test_phase_cell), library MiB by rows:
    70 -> 713, 64/60/56 -> 284, 50 -> 74, 44/40 -> 42, 32 -> 7.2. At a budget of 20 MiB
    only the 32-row list fits; with the floor at 40 rows nothing is offered. (A
    LATER, conservative floor: 32 rows on the S phase of a real map gave 0.3 % of the
    pixels within 5 degrees, 40 rows 38 %.)"""
    import hough_reflectors as hr
    from tests.test_phase_cell import _from_cif
    monkeypatch.setattr(eu, "_triplet_library_budget_bytes", lambda: 20 * (1 << 20))
    monkeypatch.setattr(eu, "_PREDICT_PROBE_ROWS", 10_000)
    phase, _ = _from_cif(tmp_path, "orthorhombic-Cmcm")
    found = hr.largest_list_that_fits(phase)
    assert found["rows"] is None and found["reason"] == "nothing_fits"
    # the floor is the only reason: with it lowered the 32-row list is there
    monkeypatch.setattr(eu, "_REFLECTOR_LADDER", (64, 60, 56, 50, 44, 40, 32))
    monkeypatch.setattr(eu, "MIN_FIT_ROWS", 32)
    assert hr.largest_list_that_fits(phase)["rows"] == 32
