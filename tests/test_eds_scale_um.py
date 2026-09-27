"""Smoothing as a physical length, not a pixel count.

``scale`` is a box width in PIXELS, and a pixel is not a length: 5 px at a
0.2 um step is a 1 um averaging box, 5 px at a 2 um step is a 10 um one. A
saved recipe replayed across two step sizes is therefore two different
physical analyses presented under one name, with nothing on screen saying so.
``scale_um`` states the box in microns and lets each dataset convert it.

The load-bearing test in this file is
:func:`test_absent_scale_um_is_bit_identical_to_the_old_path` — this code sits
under every phase map the app has ever made, so the untouched path has to be
the same arithmetic, not merely similar behaviour.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.main import app
from backend.api.routes import eds as eds_routes
from backend.api.services.cif_phase_library import CifPhaseEntry, candidates_for
from backend.api.services.eds_clustering import (
    DEFAULT_SCALE, cluster_and_match, scale_box_um, scale_px_from_um,
)
from backend.api.services.phase_map_store import get_phase_map_store

N_ROWS = N_COLS = 12

#: Which patched CIF loader the route reached, per request — see ``_classify``.
_LIBRARY_CALLS = []


# ---------------------------------------------------------------------------
# scale_px_from_um — the arithmetic
# ---------------------------------------------------------------------------

def test_it_divides_the_length_by_the_step():
    assert scale_px_from_um(1.0, 0.2) == 5
    assert scale_px_from_um(3.29, 0.6579) == 5


def test_it_rounds_to_the_nearest_whole_pixel():
    # The filter takes one integer, so a fractional box cannot be applied;
    # the nearest whole pixel is the honest answer and the report says what
    # that came to in microns.
    assert scale_px_from_um(0.44, 0.1) == 4
    assert scale_px_from_um(0.46, 0.1) == 5
    assert scale_px_from_um(0.25, 0.1) == 3       # half-up, not banker's


def test_zero_is_an_answer_not_an_error():
    # A width below half a pixel genuinely cannot be applied, and 0 is a
    # legitimate setting meaning "no smoothing" — never a negative box.
    assert scale_px_from_um(0.04, 0.1) == 0
    assert scale_px_from_um(0.0, 0.1) == 0
    assert scale_px_from_um(-2.0, 0.1) == 0


@pytest.mark.parametrize("bad_step", [None, 0.0, -0.5, float("nan"), "abc"])
def test_a_bad_step_size_raises_instead_of_guessing(bad_step):
    # Inventing a step would put a wrong physical width into the provenance,
    # which is worse than admitting the step is unknown. The caller decides
    # to fall back, and has to say so.
    with pytest.raises(ValueError):
        scale_px_from_um(1.0, bad_step)


def test_the_same_length_is_the_same_box_at_two_step_sizes():
    """The entire point of the change.

    0.5 um is 5 px on a 0.1 um scan and 2 px on a 0.25 um one — different
    pixel counts, the same physical averaging box.
    """
    fine, coarse = scale_px_from_um(0.5, 0.1), scale_px_from_um(0.5, 0.25)
    assert (fine, coarse) == (5, 2)
    assert scale_box_um(fine, 0.1, 0.1)[0] == pytest.approx(0.5)
    assert scale_box_um(coarse, 0.25, 0.25)[0] == pytest.approx(0.5)


def test_the_box_is_reported_on_both_axes():
    # Square in pixels does not mean square in microns. Quoting one number
    # for a scan with different X and Y steps is a wrong number in a report.
    assert scale_box_um(4, 0.5, 0.25) == (2.0, 1.0)


def test_an_unknown_step_gives_no_physical_edge():
    assert scale_box_um(5, None, 0.2) == (None, 0.2 * 5)
    assert scale_box_um(5, 0.0, None) == (None, None)


# ---------------------------------------------------------------------------
# _eds_step_um — where the step size comes from
# ---------------------------------------------------------------------------

class _FakeExtractor:
    def __init__(self, sizes):
        self._sizes = sizes

    def get_pixel_sizes(self):
        return self._sizes


def _with_extractor(monkeypatch, sizes):
    monkeypatch.setattr(eds_routes, "is_open", lambda: True)
    monkeypatch.setattr(eds_routes, "get_extractor",
                        lambda: _FakeExtractor(sizes))


def test_step_size_comes_from_the_eds_area(monkeypatch):
    _with_extractor(monkeypatch, {
        "eds": {"x": 0.6579, "y": 0.6579, "units": "um", "source": "step"},
        "ebsd": {"x": 9.9, "y": 9.9, "units": "um", "source": "step"},
        "electron_image": {"x": 0.0621, "y": 0.0621, "units": "um"},
    })
    assert eds_routes._eds_step_um() == (0.6579, 0.6579)


def test_it_falls_back_to_the_ebsd_area(monkeypatch):
    # EDS and EBSD share one acquisition grid on these files, which is what
    # lets an EDS map be indexed with EBSD (row, col) at all.
    _with_extractor(monkeypatch, {
        "eds": None,
        "ebsd": {"x": 0.6579, "y": 0.5, "units": "um"},
        "electron_image": {"x": 0.0621, "y": 0.0621, "units": "um"},
    })
    assert eds_routes._eds_step_um() == (0.6579, 0.5)


def test_it_never_borrows_the_electron_image_step(monkeypatch):
    # Measured 2026-08-14: the electron image sits on a 10.6x finer step over
    # a different field of view. Borrowing it would shrink every reported
    # length by an order of magnitude.
    _with_extractor(monkeypatch, {
        "eds": None, "ebsd": None,
        "electron_image": {"x": 0.0621, "y": 0.0621, "units": "um"},
    })
    assert eds_routes._eds_step_um() == (None, None)


def test_a_step_in_unknown_units_is_treated_as_unknown(monkeypatch):
    _with_extractor(monkeypatch, {"eds": {"x": 500.0, "y": 500.0,
                                          "units": "nm"}})
    assert eds_routes._eds_step_um() == (None, None)


def test_an_extractor_that_cannot_answer_costs_a_unit_not_the_run(monkeypatch):
    class _NoGeometry:
        pass

    monkeypatch.setattr(eds_routes, "is_open", lambda: True)
    monkeypatch.setattr(eds_routes, "get_extractor", lambda: _NoGeometry())
    assert eds_routes._eds_step_um() == (None, None)

    def _boom():
        raise RuntimeError("h5 handle is dead")

    monkeypatch.setattr(eds_routes, "get_extractor", _boom)
    assert eds_routes._eds_step_um() == (None, None)


def test_no_file_open_means_no_step(monkeypatch):
    monkeypatch.setattr(eds_routes, "is_open", lambda: False)
    assert eds_routes._eds_step_um() == (None, None)


# ---------------------------------------------------------------------------
# The endpoint
# ---------------------------------------------------------------------------

def _at_maps():
    """A 12x12 aluminium matrix with a 4x4 silicon particle in one corner."""
    n = N_ROWS * N_COLS
    al = np.full(n, 94.0)
    si = np.full(n, 4.0)
    fe = np.full(n, 2.0)
    g = np.arange(n)
    particle = ((g // N_COLS < 4) & (g % N_COLS < 4))
    al[particle], si[particle], fe[particle] = 40.0, 59.0, 1.0
    return {"Al": al, "Si": si, "Fe": fe}


def _library():
    def entry(key, comp):
        return CifPhaseEntry(
            key=key, cif_filename=key, formula=key, space_group="Fm-3m",
            space_group_number=225, crystal_system="cubic",
            composition=comp, elements=sorted(comp),
        )

    return {"Al.cif": entry("Al.cif", {"Al": 100.0}),
            "Si.cif": entry("Si.cif", {"Si": 100.0})}


@pytest.fixture
def client(monkeypatch, tmp_path):
    """A loaded file with known composition and a two-phase library.

    ``is_open`` is pinned False so the step size is genuinely unknown unless
    a test says otherwise — the store and the session are per process, and a
    step leaking in from another test would make these assertions lie.

    BOTH loader names are patched. /auto-classify moved from
    ``load_cif_phase_library`` to ``load_cif_phase_library_with_skips`` on
    2026-09-13 (6e1d5d73, the composition-contradiction warning), while this
    file was written before that and committed after it (2f511775). Patching
    only the old name left the route reading the real Database/CIF_Library,
    so the grid comparisons below were against a ~30-phase candidate list and
    failed on the phase ids (3/5 against 1/0) while the shapes matched
    exactly. Nothing said the fake library had been bypassed, which is why
    ``_classify`` now proves the call happened.
    """
    at_maps = _at_maps()
    # A PATH, not a bare name. Every successful set_classification writes a
    # .phase_map.npz sidecar next to the file it was given, so "fake.h5oina"
    # dropped fake.h5oina.phase_map.npz into the repository root on every run
    # of this file — measured: of the EDS suites, only this one did.
    scan = tmp_path / "fake.h5oina"
    monkeypatch.setattr(
        eds_routes, "_build_at_pct_maps_for_loaded_file",
        lambda: (at_maps, N_ROWS, N_COLS, str(scan)),
    )

    def _loader(*_a, **_kw):
        _LIBRARY_CALLS.append("plain")
        return _library()

    def _loader_with_skips(*_a, **_kw):
        _LIBRARY_CALLS.append("with_skips")
        return _library(), []

    monkeypatch.setattr(eds_routes, "load_cif_phase_library", _loader)
    monkeypatch.setattr(eds_routes, "load_cif_phase_library_with_skips",
                        _loader_with_skips)
    monkeypatch.setattr(eds_routes, "is_open", lambda: False)
    _LIBRARY_CALLS.clear()
    get_phase_map_store().clear()
    try:
        yield TestClient(app), at_maps
    finally:
        _LIBRARY_CALLS.clear()
        get_phase_map_store().clear()


def _classify(c, **kw):
    """Classify, and prove the route read the fixture's library while doing it.

    A monkeypatch on a name the route no longer calls fails silently. It
    cannot be caught by the phase NAMES either: the real
    Database/CIF_Library also holds Al.cif and Si.cif, so a bypassed patch
    still produces a map that matches "Al" and "Si" — only the candidate
    INDICES move (3 and 5 against 0 and 1), which reads like a broken feature.
    So the guard is the call itself.
    """
    before = len(_LIBRARY_CALLS)
    r = c.post("/api/eds/auto-classify", json=kw)
    assert r.status_code == 200, r.text
    assert len(_LIBRARY_CALLS) > before, (
        "auto-classify did not call either patched loader, so it read the "
        "real Database/CIF_Library: the fixture patches names the route no "
        "longer uses")
    return r.json()


def _stored_grids():
    state = get_phase_map_store().get_state()
    assert state is not None
    return np.asarray(state.phase_grid), np.asarray(state.region_grid)


def test_absent_scale_um_is_bit_identical_to_the_old_path(client):
    """The regression guard. Every existing phase map runs through here.

    The reference is computed by calling the clustering service directly with
    ``scale=None`` — the untouched pre-change entry point — on the same maps
    and the same candidate list, so this compares the route's arithmetic
    against the arithmetic it had before, not against itself.
    """
    c, at_maps = client
    body = _classify(c, min_score=0.3)
    phase_grid, region_grid = _stored_grids()

    ref_phase, ref_region, _matches, ref_k = cluster_and_match(
        at_pct_per_element=at_maps, n_rows=N_ROWS, n_cols=N_COLS,
        candidates=candidates_for(_library(), at_maps.keys()),
        k=None, min_score=0.3, scale=None,
    )
    assert np.array_equal(phase_grid, ref_phase)
    assert np.array_equal(region_grid, ref_region)
    assert body["k_used"] == ref_k
    # And it says, honestly, that no physical width was involved.
    assert body["scale_source"] == "pixels"
    assert body["scale_px_used"] == DEFAULT_SCALE
    assert body["scale_um_requested"] is None


def test_a_pixel_scale_still_wins_when_no_um_is_given(client):
    c, at_maps = client
    body = _classify(c, scale=0)
    assert body["scale_px_used"] == 0
    assert body["scale_source"] == "pixels"

    phase_grid, _ = _stored_grids()
    ref_phase, _r, _m, _k = cluster_and_match(
        at_pct_per_element=at_maps, n_rows=N_ROWS, n_cols=N_COLS,
        candidates=candidates_for(_library(), at_maps.keys()),
        k=None, min_score=0.3, scale=0,
    )
    assert np.array_equal(phase_grid, ref_phase)


def test_scale_um_wins_and_the_response_reports_resolved_values(
        client, monkeypatch):
    c, _ = client
    monkeypatch.setattr(eds_routes, "_eds_step_um", lambda: (0.1, 0.1))
    body = _classify(c, scale=99, scale_um=0.5)

    assert body["scale_source"] == "um"
    # RESOLVED, not requested: 99 px was asked for in pixels and lost.
    assert body["scale_px_used"] == 5
    assert body["scale_um_used"] == pytest.approx(0.5)
    assert body["scale_um_requested"] == pytest.approx(0.5)
    assert body["step_x_um"] == pytest.approx(0.1)
    assert body["step_y_um"] == pytest.approx(0.1)
    assert body["scale_box_um"] == {"x": pytest.approx(0.5),
                                    "y": pytest.approx(0.5)}


def test_the_same_um_is_the_same_physical_box_on_two_scans(client, monkeypatch):
    """Acceptance item 7 of the design, through the endpoint."""
    c, _ = client

    monkeypatch.setattr(eds_routes, "_eds_step_um", lambda: (0.1, 0.1))
    fine = _classify(c, scale_um=0.5)
    monkeypatch.setattr(eds_routes, "_eds_step_um", lambda: (0.25, 0.25))
    coarse = _classify(c, scale_um=0.5)

    assert fine["scale_px_used"] != coarse["scale_px_used"]   # 5 vs 2
    assert fine["scale_um_used"] == pytest.approx(coarse["scale_um_used"])
    assert fine["scale_um_used"] == pytest.approx(0.5)


def test_the_resolved_um_is_the_rounded_box_not_the_request(client, monkeypatch):
    # 0.33 um on a 0.1 um step is 3.3 px -> 3 px -> 0.3 um. Reporting the
    # request back would overstate the box by 10 %.
    c, _ = client
    monkeypatch.setattr(eds_routes, "_eds_step_um", lambda: (0.1, 0.1))
    body = _classify(c, scale_um=0.33)
    assert body["scale_px_used"] == 3
    assert body["scale_um_used"] == pytest.approx(0.3)
    assert body["scale_um_requested"] == pytest.approx(0.33)


def test_a_rectangular_pixel_reports_a_rectangular_box(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr(eds_routes, "_eds_step_um", lambda: (0.1, 0.2))
    body = _classify(c, scale_um=0.5)
    assert body["scale_px_used"] == 5
    assert body["scale_box_um"]["x"] == pytest.approx(0.5)
    assert body["scale_box_um"]["y"] == pytest.approx(1.0)


def test_scale_um_without_a_step_falls_back_and_says_so(client):
    """No step size in the file: the request cannot be honoured.

    It must fall back to the pixel value AND be visible in the response —
    a physical width that silently became a pixel count is the failure this
    whole change exists to prevent.
    """
    c, at_maps = client                      # is_open() is False here
    body = _classify(c, scale=3, scale_um=0.5)

    assert body["scale_source"] == "pixels_no_step"
    assert body["scale_px_used"] == 3
    assert body["scale_um_used"] is None
    assert body["step_x_um"] is None and body["step_y_um"] is None
    assert body["scale_um_requested"] == pytest.approx(0.5)
    assert "step size" in body["scale_note"]

    phase_grid, _ = _stored_grids()
    ref_phase, _r, _m, _k = cluster_and_match(
        at_pct_per_element=at_maps, n_rows=N_ROWS, n_cols=N_COLS,
        candidates=candidates_for(_library(), at_maps.keys()),
        k=None, min_score=0.3, scale=3,
    )
    assert np.array_equal(phase_grid, ref_phase)


def test_pixel_mode_does_not_claim_a_smoothing_width(client, monkeypatch):
    # Per-pixel matching never smooths, so a scale in the provenance there
    # would be a false entry. The step sizes are still true.
    c, _ = client
    monkeypatch.setattr(eds_routes, "_eds_step_um", lambda: (0.1, 0.1))
    body = _classify(c, mode="pixel", scale_um=0.5)
    assert body["scale_source"] == "not_applicable"
    assert body["scale_px_used"] is None
    assert body["scale_um_used"] is None
    assert body["step_x_um"] == pytest.approx(0.1)


def test_the_preview_resolves_the_same_way_as_the_run(client, monkeypatch):
    """Otherwise the preview smooths on a different box than the run.

    That disagreement is exactly what the preview endpoint was built to
    remove, so it has to accept the same physical width.
    """
    c, _ = client
    monkeypatch.setattr(eds_routes, "_eds_step_um", lambda: (0.1, 0.1))
    r = c.post("/api/eds/phase-map/region-defs/preview",
               json={"region_defs": [{"name": "Si", "elements": [
                   {"element": "Si", "min_at_pct": 20}]}], "scale_um": 0.5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scale"] == 5
    assert body["scale_px_used"] == 5
    assert body["scale_source"] == "um"
    assert body["scale_um_used"] == pytest.approx(0.5)


def test_seeding_from_a_pixel_resolves_the_same_way(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr(eds_routes, "_eds_step_um", lambda: (0.1, 0.1))
    r = c.post("/api/eds/phase-map/region-defs/seed-from-pixel",
               json={"row": 1, "col": 1, "scale_um": 0.5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scale"] == 5
    assert body["scale_px_used"] == 5
    assert body["scale_source"] == "um"
