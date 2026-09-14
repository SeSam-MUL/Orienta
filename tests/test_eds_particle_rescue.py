"""Particle rescue for pattern-degenerate pairs (backend/api/services/eds_particle_rescue.py).

Synthetic grids built to mirror the measured situation of 2026-09-13
(Probe B, Arbeitsbereich 1): an Al matrix grain, Si particles diluted by the
EDS interaction volume, and the two ways a 30 at% Si pixel can arise -- the
surface IS the particle (own orientation) or the beam sits on matrix beside
or above it (orientation continues the grain).
"""
import numpy as np
import pytest
from orix.quaternion import Orientation, symmetry

from backend.api.services.eds_particle_rescue import (
    RescuePair, rescue_particles, select_rescue_pairs,
)

AL, SI = 1, 2
OH = symmetry.Oh


def _quat(euler_deg):
    return np.asarray(Orientation.from_euler(np.deg2rad(euler_deg)).data, dtype=float).reshape(4)


MATRIX_Q = _quat([10.0, 20.0, 30.0])
PARTICLE_Q = _quat([70.0, 60.0, 15.0])      # ~50 deg from the matrix under Oh
SECOND_GRAIN_Q = _quat([200.0, 45.0, 80.0])


def _grid(h=20, w=20):
    pid = np.full((h, w), AL, dtype=int)
    q = np.tile(MATRIX_Q, (h, w, 1))
    si = np.full((h, w), 1.6)                # matrix background, as measured
    return pid, q, si


def _pair():
    return RescuePair(particle_pid=SI, matrix_pid=AL, element="Si",
                      particle_name="Si", matrix_name="Al")


def test_small_particle_with_own_orientation_becomes_particle():
    """Blob C of the crop: 13 px, 28 at% Si, 43 deg off the matrix, all Al."""
    pid, q, si = _grid()
    blob = np.zeros_like(pid, dtype=bool); blob[8:11, 8:12] = True
    si[blob] = 28.0
    q[blob] = PARTICLE_Q
    res = rescue_particles(pid, q, si, _pair(), OH)
    assert res.to_particle.sum() == blob.sum()
    assert (res.phase_id_2d[blob] == SI).all()
    assert res.to_matrix.sum() == 0
    assert (res.phase_id_2d[~blob] == AL).all()
    assert res.report["n_to_particle"] == blob.sum() and res.report["n_blobs"] == 1
    # the input grid is untouched (pure function)
    assert (pid[blob] == AL).all()


def test_beam_on_matrix_beside_particle_stays_matrix():
    """Blob D of the crop: 29 at% Si but the orientation continues the grain."""
    pid, q, si = _grid()
    blob = np.zeros_like(pid, dtype=bool); blob[8:11, 8:10] = True
    si[blob] = 29.0
    res = rescue_particles(pid, q, si, _pair(), OH)
    assert res.n_changed == 0
    assert res.report["n_blobs"] == 1
    assert res.report["blobs"][0]["median_min_angle_deg"] < 1.0


def test_particle_rim_pixel_that_continues_the_matrix_goes_back():
    """Backward direction: a Si-assigned rim pixel whose surface is matrix."""
    pid, q, si = _grid()
    core = np.zeros_like(pid, dtype=bool); core[6:12, 6:12] = True
    si[core] = 80.0; pid[core] = SI; q[core] = PARTICLE_Q
    # one rim pixel: prior said Si (55 at%), orientation is the matrix grain
    pid[6, 12] = SI; si[6, 12] = 55.0
    res = rescue_particles(pid, q, si, _pair(), OH)
    assert res.to_matrix.sum() == 1 and res.to_matrix[6, 12]
    assert res.phase_id_2d[6, 12] == AL
    # the core keeps its phase: nothing there is attached to the matrix
    assert (res.phase_id_2d[core] == SI).all()


def test_particle_sharing_the_matrix_orientation_is_left_alone():
    """Measured on the crop: a 92-px Si particle continues the grain below it
    to 0.5 deg. Continuity says nothing there -- neither direction may act."""
    pid, q, si = _grid()
    core = np.zeros_like(pid, dtype=bool); core[6:12, 6:12] = True
    si[core] = 77.0; pid[core] = SI                  # q stays MATRIX_Q everywhere
    pid[6, 12] = SI; si[6, 12] = 55.0                # a rim pixel the prior called Si
    res = rescue_particles(pid, q, si, _pair(), OH)
    assert res.n_changed == 0
    assert res.report["blobs"][0]["n_core_free"] == 0


def test_rim_pixel_that_belongs_to_the_particle_crystal_stays_particle():
    """Backward guard 1 (`~in_core`): a rim pixel with a DIRECT matrix
    neighbour of the same orientation, but continuing the particle's own
    crystal -> stays. The rim rule alone would flip it."""
    pid, q, si = _grid()
    core = np.zeros_like(pid, dtype=bool); core[6:12, 6:12] = True
    si[core] = 80.0; pid[core] = SI; q[core] = PARTICLE_Q
    pid[6, 12] = SI; si[6, 12] = 55.0; q[6, 12] = PARTICLE_Q   # continues the core
    q[5, 13] = PARTICLE_Q                                      # direct matrix neighbour matches
    res = rescue_particles(pid, q, si, _pair(), OH)
    assert res.to_matrix.sum() == 0
    assert res.phase_id_2d[6, 12] == SI


def test_matrix_oriented_pixel_inside_the_shell_is_not_sent_back():
    """Backward guard 2 (direct neighbour only): a Si-assigned pixel with the
    matrix orientation whose nearest true-matrix pixel is 3 px away stays.
    With a 3-px reach the rule would flip it (it is not in the core)."""
    pid, q, si = _grid(24, 24)
    shell = np.zeros_like(pid, dtype=bool); shell[4:16, 4:16] = True
    core = np.zeros_like(pid, dtype=bool); core[7:13, 7:13] = True
    si[shell] = 55.0; si[core] = 80.0; pid[shell] = SI; q[shell] = PARTICLE_Q
    q[6, 10] = MATRIX_Q                     # inside the 3-px shell, matrix-oriented
    res = rescue_particles(pid, q, si, _pair(), OH, ring_px=3)
    assert res.to_matrix.sum() == 0 and res.phase_id_2d[6, 10] == SI
    q[4, 10] = MATRIX_Q                     # the same pixel AT the rim flips
    res = rescue_particles(pid, q, si, _pair(), OH, ring_px=3)
    assert res.to_matrix.sum() == 1 and res.phase_id_2d[4, 10] == AL


def test_deep_core_with_the_rim_pixels_orientation_keeps_it_particle():
    """Backward guard 3 (`core_zone`): a particle with two orientations --
    the half next to the rim pixel is its own crystal, the deeper part shares
    the matrix orientation (blob r47-57 of the crop). A matrix-oriented rim
    pixel matches nothing in reach except that deeper core -> it stays. The
    deeper core has no matrix in reach and is only seen through core_zone."""
    pid, q, si = _grid(24, 24)
    core = np.zeros_like(pid, dtype=bool); core[4:16, 4:16] = True
    si[core] = 80.0; pid[core] = SI
    q[core] = PARTICLE_Q
    q[4:16, 10:16] = MATRIX_Q                # right half shares the matrix orientation
    res = rescue_particles(pid, q, si, _pair(), OH, ring_px=3)
    assert res.to_matrix.sum() == 0
    assert res.phase_id_2d[4, 12] == SI and res.phase_id_2d[4, 5] == SI
    assert res.report["blobs"][0]["n_core_free"] > 0


def test_route_glue_keeps_the_not_indexed_phase(monkeypatch):
    """Consensus maps carry -1 for unassigned pixels and orix files that as
    phase 'not_indexed' under id -1; the rebuild that adds a pruned phase
    must keep it (measured before: repr IndexError, phases[-1] KeyError)."""
    from pathlib import Path
    from types import SimpleNamespace
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation
    import backend.api.routes.indexing as route
    import backend.api.services.eds_indexing_prior as prior
    from indexing_controller import declare_phase_id_base
    al = Path("Database/EBSD_SHT_Database/Al/Al (Al) [cF4] {20kV}.sht")
    sht = Path("Database/EBSD_SHT_Database/Si/Si (Si) [cF8] {20kV}.sht")
    if not (al.is_file() and sht.is_file()):
        pytest.skip("Al/Si masters not on this machine")
    pid, q, si = _grid(); pid[:] = 0                            # consensus: base 0
    blob = np.zeros_like(pid, dtype=bool); blob[8:11, 8:12] = True
    q[blob] = PARTICLE_Q; si[blob] = 30.0
    pid[0, 0:3] = -1                                            # three unassigned pixels
    xmap = CrystalMap(rotations=Rotation(q.reshape(-1, 4)), phase_id=pid.reshape(-1),
                      x=np.tile(np.arange(20), 20).astype(float),
                      y=np.repeat(np.arange(20), 20).astype(float),
                      phase_list=PhaseList(phases=[Phase(name="Al", point_group="m-3m"),
                                                   Phase(name="Si", point_group="m-3m")], ids=[0, 1]))
    assert list(xmap.phases.ids) == [-1, 0]
    declare_phase_id_base(xmap, 0)
    result = SimpleNamespace(xmap=xmap, selection_mask=np.ones((20, 20), bool),
                             original_shape=(20, 20), metadata={})
    req = SimpleNamespace(method="spherical", sht_paths=[str(al), str(sht)],
                          master_h5_paths=[], cif_paths=[],
                          eds_phase_strengths={str(al): 1.0, str(sht): 1.0}, eds_expected_overrides={})
    monkeypatch.setattr(prior, "measured_atpct_per_pixel",
                        lambda sel, expected_shape=None: [{"Al": 100.0 - v, "Si": v} for v in si.reshape(-1)])
    monkeypatch.setattr(prior, "expected_at_pct_for_phases",
                        lambda formulas, overrides=None, paths=None: [{"Al": 100.0}, {"Si": 100.0}])
    report = route._apply_particle_rescue(result, req, result.selection_mask)
    assert report is not None and report["n_to_particle"] == blob.sum()
    assert list(xmap.phases.ids) == [-1, 0, 1]
    assert xmap.phases[-1].name == "not_indexed"
    repr(xmap)
    assert (np.asarray(xmap.phase_id).reshape(20, 20)[0, 0:3] == -1).all()


def test_large_blob_costs_its_rim_not_its_area():
    """Interior pixels are undecidable by construction and must not enter the
    pairwise angle matrices (measured before: 2025 px -> 1.1 GB)."""
    import time
    pid, q, si = _grid(70, 70)
    core = np.zeros_like(pid, dtype=bool); core[5:65, 5:65] = True   # 3600 px
    si[core] = 80.0; pid[core] = SI; q[core] = PARTICLE_Q
    t0 = time.perf_counter()
    res = rescue_particles(pid, q, si, _pair(), OH, ring_px=3)
    assert time.perf_counter() - t0 < 5.0
    assert res.n_changed == 0
    b = res.report["blobs"][0]
    assert b["n_undecidable"] == (60 - 6) ** 2       # everything deeper than 3 px


def test_blobs_below_min_size_are_counted_not_decided():
    pid, q, si = _grid()
    si[8, 8] = 30.0; q[8, 8] = PARTICLE_Q                      # 1 px
    si[14, 3:5] = 30.0; q[14, 3:5] = PARTICLE_Q                # 2 px
    res = rescue_particles(pid, q, si, _pair(), OH)
    assert res.n_changed == 0 and res.report["n_too_small"] == 3
    res2 = rescue_particles(pid, q, si, _pair(), OH, min_blob_px=1)
    assert res2.to_particle.sum() == 3


def test_particle_interior_far_from_matrix_is_untouched():
    """Big particle: interior pixels have no matrix within ring_px -> no decision."""
    pid, q, si = _grid(30, 30)
    core = np.zeros_like(pid, dtype=bool); core[5:25, 5:25] = True
    si[core] = 85.0; pid[core] = SI; q[core] = PARTICLE_Q
    res = rescue_particles(pid, q, si, _pair(), OH, ring_px=3)
    assert res.n_changed == 0
    assert res.report["blobs"][0]["n_undecidable"] > 0


def test_enrichment_needs_both_absolute_and_relative_threshold():
    pid, q, si = _grid()
    q[8:11, 8:11] = PARTICLE_Q
    si[8:11, 8:11] = 6.0          # 3.75x background but under 15 at%
    assert rescue_particles(pid, q, si, _pair(), OH).n_changed == 0
    si2 = np.full_like(si, 12.0)  # background high -> 15 at% is not 5x
    si2[8:11, 8:11] = 20.0
    res = rescue_particles(pid, q, si2, _pair(), OH)
    assert res.n_changed == 0 and res.report["threshold_at_pct"] == 60.0


def test_matrix_element_is_never_rescued():
    """Seen from the Al side, 'Al enriched' means nothing: skipped with reason."""
    pid, q, al = _grid()
    al[:] = 90.0
    pair = RescuePair(particle_pid=AL, matrix_pid=SI, element="Al")
    pid[0:3, 0:3] = SI
    res = rescue_particles(pid, q, al, pair, OH)
    assert res.n_changed == 0 and "not below" in res.report["skipped"]


def test_particle_touching_a_second_matrix_grain_is_judged_against_its_own_neighbours():
    """A grain boundary next to the blob: attachment is to ANY nearby matrix
    pixel, so the blob only counts as its own crystal when it matches neither."""
    pid, q, si = _grid()
    q[:, 10:] = SECOND_GRAIN_Q                   # two matrix grains
    blob = np.zeros_like(pid, dtype=bool); blob[8:11, 8:12] = True
    si[blob] = 30.0
    q[blob] = SECOND_GRAIN_Q                     # continues the right-hand grain
    assert rescue_particles(pid, q, si, _pair(), OH).n_changed == 0
    q[blob] = PARTICLE_Q                         # matches neither grain
    assert rescue_particles(pid, q, si, _pair(), OH).to_particle.sum() == blob.sum()


def test_unindexed_and_unmeasured_pixels_are_ignored():
    pid, q, si = _grid()
    blob = np.zeros_like(pid, dtype=bool); blob[8:11, 8:12] = True
    si[blob] = 30.0; q[blob] = PARTICLE_Q
    pid[9, 9] = -1                                # unindexed inside the blob
    si[8, 8] = np.nan                             # unmeasured inside the blob
    res = rescue_particles(pid, q, si, _pair(), OH)
    assert res.phase_id_2d[9, 9] == -1
    assert res.phase_id_2d[8, 8] == AL
    assert res.to_particle.sum() == blob.sum() - 2


def test_symmetry_is_honoured():
    """A 90 deg rotation about <100> is the identity under Oh: still matrix."""
    pid, q, si = _grid()
    blob = np.zeros_like(pid, dtype=bool); blob[8:11, 8:12] = True
    si[blob] = 30.0
    rot90 = Orientation.from_euler(np.deg2rad([[90.0, 0.0, 0.0]])) * Orientation(MATRIX_Q)
    q[blob] = np.asarray(rot90.data, dtype=float).reshape(4)
    assert rescue_particles(pid, q, si, _pair(), OH).n_changed == 0


def test_shape_errors_are_loud():
    pid, q, si = _grid()
    with pytest.raises(ValueError):
        rescue_particles(pid, q[..., :3], si, _pair(), OH)
    with pytest.raises(ValueError):
        rescue_particles(pid, q, si[:-1], _pair(), OH)


# --- the real crop (Probe B, Arbeitsbereich 1, 80x48, 2026-09-13) ----------
#
# Fixture: phase ids of the run (1 = Al, 2 = Si, EDS prior on), Si at% from
# the same quantification, and the Al-master orientation of every pixel
# (the Al fit and the Si fit agree to 0.1 deg on this pair, so one set serves
# both directions). Numbers in the assertions are the measured outcome; a
# change here is a change of behaviour, not a broken test.

@pytest.fixture(scope="module")
def ab1_crop():
    from pathlib import Path
    from orix.quaternion import Rotation
    f = np.load(Path(__file__).parent / "fixtures" / "eds_particle_rescue_ab1_crop.npz")
    ph = f["phase"].astype(int); si = f["si"].astype(float)
    q = np.asarray(Rotation.from_euler(f["euler_al_fit"].astype(float)).data).reshape(80, 48, 4)
    return ph, q, si


def test_real_crop_missed_particles_become_si_and_the_matrix_case_stays(ab1_crop):
    ph, q, si = ab1_crop
    res = rescue_particles(ph, q, si, _pair(), OH)
    new = res.phase_id_2d
    # B (rows 21-22, cols 45-47): own crystal, 48-54 deg off every grain around
    assert (new[21:23, 45:48] == SI).all()
    # C (row 36, cols 41-46): own crystal, 33 deg off
    assert (new[36, 41:47] == SI).all()
    # A (rows 17-20, cols 0-5): continues the 5-px Si core beside it
    assert (new[18, 0:6] == SI).all() and (new[19, 3:7] == SI).all()
    # D (rows 49-51, cols 6-7): 29 at% Si but the orientation IS the matrix grain
    assert (new[49:52, 6:8] == AL).all()
    # halo pixels of B at 15-21 at% that continue the matrix stay Al
    assert new[20, 41] == AL and new[23, 47] == AL
    # nothing below the enrichment threshold moves, nothing indexed as Si with
    # more than 60 at% Si goes back, and no matrix pixel far from any particle moves
    assert not (res.to_particle | res.to_matrix)[si < 15].any()
    assert si[res.to_matrix].max() < 60.0
    assert (new[(si < 3)] == AL).all()
    # snapshot of the whole crop (see module comment)
    assert int(res.to_particle.sum()) == 94
    assert int(res.to_matrix.sum()) == 11
    assert res.report["n_too_small"] == 2


# --- route glue ------------------------------------------------------------

def _fake_run(monkeypatch, si_map, strengths, pg_si="m-3m"):
    """A spherical-shaped result on a 20x20 grid: Al matrix with one enriched
    blob of its own orientation; the EDS map and compositions are stubbed."""
    from types import SimpleNamespace
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation
    import backend.api.routes.indexing as route
    import backend.api.services.eds_indexing_prior as prior

    from indexing_controller import declare_phase_id_base

    pid, q, _ = _grid()
    blob = np.zeros_like(pid, dtype=bool); blob[8:11, 8:12] = True
    q[blob] = PARTICLE_Q
    # one Si pixel in a corner keeps the Si phase in the orix phase list
    # (orix drops a phase no pixel carries -- the no-pixel case is covered
    # by test_point_group_of_an_absent_phase_comes_from_its_file)
    pid[0, 0] = SI; q[0, 0] = PARTICLE_Q
    phases = PhaseList(
        phases=[Phase(name="Al", point_group="m-3m"), Phase(name="Si", point_group=pg_si)],
        ids=[1, 2])
    xmap = CrystalMap(rotations=Rotation(q.reshape(-1, 4)), phase_id=pid.reshape(-1),
                      x=np.tile(np.arange(20), 20).astype(float),
                      y=np.repeat(np.arange(20), 20).astype(float), phase_list=phases)
    declare_phase_id_base(xmap, 1)
    result = SimpleNamespace(xmap=xmap, selection_mask=np.ones((20, 20), bool),
                             original_shape=(20, 20), metadata={})
    req = SimpleNamespace(method="spherical", sht_paths=["Al.sht", "Si.sht"],
                          master_h5_paths=[], cif_paths=[],
                          eds_phase_strengths=strengths, eds_expected_overrides={})
    monkeypatch.setattr(prior, "measured_atpct_per_pixel",
                        lambda sel, expected_shape=None: [
                            {"Al": 100.0 - v, "Si": v} for v in si_map.reshape(-1)])
    monkeypatch.setattr(prior, "expected_at_pct_for_phases",
                        lambda formulas, overrides=None, paths=None: [{"Al": 100.0}, {"Si": 100.0}])
    monkeypatch.setattr(route, "_phase_formulas_for_paths", lambda paths: ["Al", "Si"])
    return route, result, req, blob


def test_route_glue_rewrites_the_xmap_and_records_provenance(monkeypatch):
    si = np.full((20, 20), 1.6); si[8:11, 8:12] = 30.0
    route, result, req, blob = _fake_run(monkeypatch, si, {"Al.sht": 0.75, "Si.sht": 0.75})
    lines = []
    report = route._apply_particle_rescue(result, req, result.selection_mask, lines.append)
    assert report is not None and report["n_to_particle"] == blob.sum() and report["n_to_matrix"] == 0
    new = np.asarray(result.xmap.phase_id).reshape(20, 20)
    assert (new[blob] == SI).all()
    rest = ~blob; rest[0, 0] = False                      # the corner Si pixel of the fake
    assert (new[rest] == AL).all() and new[0, 0] == SI
    assert result.metadata["eds_particle_rescue"] is report
    assert sorted(report["changed_rows"]) == sorted(np.flatnonzero(blob.reshape(-1)).tolist())
    assert report["old_phase_ids"] == [AL] * blob.sum()
    assert lines and "Si: 12 px -> Si" in lines[0]


@pytest.mark.parametrize("strengths", [
    {"Al.sht": 0.0, "Si.sht": 0.0},          # prior off
    {"other.cif": 0.75},                      # strength keyed to another method's file
    {"Al.sht": 0.75, "Si.sht": 0.0},          # the particle phase was not weighed
    {"Al.sht": 0.0, "Si.sht": 0.75},          # the matrix phase was not weighed
])
def test_route_glue_is_a_noop_unless_the_prior_weighed_both_phases(monkeypatch, strengths):
    si = np.full((20, 20), 1.6); si[8:11, 8:12] = 30.0
    route, result, req, blob = _fake_run(monkeypatch, si, strengths)
    before = np.asarray(result.xmap.phase_id).copy()
    assert route._apply_particle_rescue(result, req, result.selection_mask) is None
    assert (np.asarray(result.xmap.phase_id) == before).all()
    assert "eds_particle_rescue" not in result.metadata


def test_route_glue_reads_a_missing_element_as_zero_not_unmeasured(monkeypatch):
    """The quantification drops zeros from a pixel's dict; a matrix pixel
    with 0 at% Si is still matrix and still a ring anchor."""
    si = np.full((20, 20), 0.0); si[8:11, 8:12] = 30.0
    route, result, req, blob = _fake_run(monkeypatch, si, {"Al.sht": 0.75, "Si.sht": 0.75})
    import backend.api.services.eds_indexing_prior as prior
    monkeypatch.setattr(prior, "measured_atpct_per_pixel",
                        lambda sel, expected_shape=None: [
                            ({"Al": 100.0 - v, "Si": v} if v > 0 else {"Al": 100.0})
                            for v in si.reshape(-1)])
    report = route._apply_particle_rescue(result, req, result.selection_mask)
    assert report is not None and report["n_to_particle"] == blob.sum()


def test_route_glue_adds_the_phase_orix_pruned():
    """The run this pass exists for: NOT ONE pixel indexed as Si, so orix
    dropped Si from the phase list. After the rescue the map must still
    print, colour and export: Si is in the list with name and point group."""
    from pathlib import Path
    from types import SimpleNamespace
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation
    from _pytest.monkeypatch import MonkeyPatch
    import backend.api.routes.indexing as route
    import backend.api.services.eds_indexing_prior as prior
    from indexing_controller import declare_phase_id_base
    al = Path("Database/EBSD_SHT_Database/Al/Al (Al) [cF4] {20kV}.sht")
    sht = Path("Database/EBSD_SHT_Database/Si/Si (Si) [cF8] {20kV}.sht")
    if not (al.is_file() and sht.is_file()):
        pytest.skip("Al/Si masters not on this machine")
    pid, q, si = _grid()
    blob = np.zeros_like(pid, dtype=bool); blob[8:11, 8:12] = True
    q[blob] = PARTICLE_Q; si[blob] = 30.0
    xmap = CrystalMap(rotations=Rotation(q.reshape(-1, 4)), phase_id=pid.reshape(-1),
                      x=np.tile(np.arange(20), 20).astype(float),
                      y=np.repeat(np.arange(20), 20).astype(float),
                      phase_list=PhaseList(phases=[Phase(name="Al", point_group="m-3m"),
                                                   Phase(name="Si", point_group="m-3m")], ids=[1, 2]))
    assert list(xmap.phases.ids) == [1]                     # orix pruned Si
    declare_phase_id_base(xmap, 1)
    result = SimpleNamespace(xmap=xmap, selection_mask=np.ones((20, 20), bool),
                             original_shape=(20, 20), metadata={})
    req = SimpleNamespace(method="spherical", sht_paths=[str(al), str(sht)],
                          master_h5_paths=[], cif_paths=[],
                          eds_phase_strengths={str(al): 1.0, str(sht): 1.0}, eds_expected_overrides={})
    mp = MonkeyPatch()
    try:
        mp.setattr(prior, "measured_atpct_per_pixel",
                   lambda sel, expected_shape=None: [{"Al": 100.0 - v, "Si": v} for v in si.reshape(-1)])
        mp.setattr(prior, "expected_at_pct_for_phases",
                   lambda formulas, overrides=None, paths=None: [{"Al": 100.0}, {"Si": 100.0}])
        report = route._apply_particle_rescue(result, req, result.selection_mask)
    finally:
        mp.undo()
    assert report is not None and report["n_to_particle"] == blob.sum()
    assert list(xmap.phases.ids) == [1, 2]
    assert xmap.phases[2].name.startswith("Si") and xmap.phases[2].point_group.name == "m-3m"
    assert list(xmap.phases_in_data.ids) == [1, 2]
    repr(xmap)                                              # was IndexError before the fix


def test_route_glue_skips_pairs_the_pattern_can_separate(monkeypatch):
    si = np.full((20, 20), 1.6); si[8:11, 8:12] = 30.0
    route, result, req, blob = _fake_run(monkeypatch, si, {"Al.sht": 0.75, "Si.sht": 0.75},
                                         pg_si="m-3")
    before = np.asarray(result.xmap.phase_id).copy()
    assert route._apply_particle_rescue(result, req, result.selection_mask) is None
    assert (np.asarray(result.xmap.phase_id) == before).all()


def test_route_glue_skips_the_corner_si_pixel_but_reports_it(monkeypatch):
    """The corner Si pixel of the fake is a 1-px blob: too small to decide."""
    si = np.full((20, 20), 1.6); si[8:11, 8:12] = 30.0; si[0, 0] = 60.0
    route, result, req, blob = _fake_run(monkeypatch, si, {"Al.sht": 0.75, "Si.sht": 0.75})
    report = route._apply_particle_rescue(result, req, result.selection_mask)
    assert np.asarray(result.xmap.phase_id).reshape(20, 20)[0, 0] == SI
    assert report["pairs"][-1]["n_too_small"] == 1


def test_point_group_of_an_absent_phase_comes_from_its_file():
    """orix drops a phase no pixel carries; the run that needs the rescue most
    (no Si pixel at all) must still find the Si point group -- from the SHT."""
    from pathlib import Path
    from types import SimpleNamespace
    import backend.api.routes.indexing as route
    sht = Path("Database/EBSD_SHT_Database/Si/Si (Si) [cF8] {20kV}.sht")
    if not sht.is_file():
        pytest.skip("Si master not on this machine")
    xmap = SimpleNamespace(phases={})          # nothing in the map
    pg, name = route._phase_point_group_and_name(xmap, 2, str(sht))
    assert pg == "m-3m" and name.startswith("Si")
    cif = Path("Database/CIF_Library/Si.cif")
    if cif.is_file():
        assert route._phase_point_group_and_name(xmap, 2, str(cif))[0] == "m-3m"
    assert route._phase_point_group_and_name(xmap, 2, "nowhere/Si.sht") == (None, "Si")


def test_route_glue_never_raises(monkeypatch):
    si = np.full((20, 20), 1.6)
    route, result, req, blob = _fake_run(monkeypatch, si, {"Al.sht": 0.75, "Si.sht": 0.75})
    import backend.api.services.eds_indexing_prior as prior
    monkeypatch.setattr(prior, "measured_atpct_per_pixel",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert route._apply_particle_rescue(result, req, result.selection_mask) is None


# --- pair selection -------------------------------------------------------

def test_select_pairs_al_si_gives_si_particle_in_al_matrix_only():
    pairs = select_rescue_pairs(
        [1, 2],
        {1: {"Al": 100.0}, 2: {"Si": 100.0}},
        {1: "m-3m", 2: "m-3m"},
        {1: "Al", 2: "Si"},
    )
    assert pairs == [
        RescuePair(particle_pid=1, matrix_pid=2, element="Al", particle_name="Al", matrix_name="Si"),
        RescuePair(particle_pid=2, matrix_pid=1, element="Si", particle_name="Si", matrix_name="Al"),
    ]


def test_select_pairs_requires_same_point_group():
    """alpha-AlFeMnSi (m-3) vs Al (m-3m): the pattern separates them -> no pair."""
    pairs = select_rescue_pairs(
        [1, 2],
        {1: {"Al": 100.0}, 2: {"Al": 73.0, "Fe": 15.0, "Mn": 4.0, "Si": 8.0}},
        {1: "m-3m", 2: "m-3"},
    )
    assert pairs == []


def test_select_pairs_picks_the_most_abundant_discriminating_element():
    pairs = select_rescue_pairs(
        [1, 2],
        {1: {"Al": 100.0}, 2: {"Mg": 66.7, "Si": 33.3}},
        {1: "m-3m", 2: "m-3m"},
    )
    assert [p.element for p in pairs if p.particle_pid == 2] == ["Mg"]


def test_select_pairs_skips_phases_without_composition_or_point_group():
    assert select_rescue_pairs([1, 2], {1: {"Al": 100.0}, 2: {}}, {1: "m-3m", 2: "m-3m"}) == []
    assert select_rescue_pairs([1, 2], {1: {"Al": 100.0}, 2: {"Si": 100.0}}, {1: "m-3m", 2: None}) == []
