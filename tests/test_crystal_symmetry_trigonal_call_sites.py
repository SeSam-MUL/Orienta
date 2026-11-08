"""Where the seven trigonal groups with their two-folds along Y are used.

``test_crystal_symmetry_trigonal.py`` pins the groups. These tests pin that the
code which names, builds, saves and reduces by a phase's point group asks for
the right one: a phase of space group 162 (-31m) and a phase of space group 167
(-3m) are both "-3m" to orix, and must stay two groups wherever the group is
handed on as text, cached by text, or written to a file.

Expectations come from the operations of the diffpy space group or from a table
written out in this file, never from the code under test.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from diffpy.structure import Lattice, Structure
from diffpy.structure.spacegroups import GetSpaceGroup
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Rotation
from orix.quaternion.symmetry import D3, D3d, D3y, C3v

import crystal_symmetry as cs
from crystal_symmetry import FramePhase, frame_symmetry, phase_point_group_key, point_group_key

HEX = Lattice(4.8, 4.8, 4.5, 90, 90, 120)
SEVEN = {149: "312", 151: "312", 153: "312", 157: "31m", 159: "31m",
         162: "-31m", 163: "-31m"}


def _structure():
    return Structure(lattice=HEX)


def _map(phases: dict, n=4):
    """A map whose pixels are split over the given phases {id: phase}."""
    ids = sorted(phases)
    pid = np.array([ids[i % len(ids)] for i in range(n * n)])
    rng = np.random.default_rng(3)
    q = rng.normal(size=(n * n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    ys, xs = np.divmod(np.arange(n * n), n)
    return CrystalMap(rotations=Rotation(q), phase_id=pid, x=xs.astype(float),
                      y=ys.astype(float),
                      phase_list=PhaseList(phases=[phases[i] for i in ids], ids=ids))


def _cartesian_rotation_parts(sg):
    """Rotation part (det +1) of every operation of space group ``sg`` in the
    crystal frame X || a, Z || c*, as a sorted list of rounded 9-tuples."""
    a, c = 4.8, 4.5
    m = np.array([[a, a * np.cos(np.radians(120)), 0.0],
                  [0.0, a * np.sin(np.radians(120)), 0.0],
                  [0.0, 0.0, c]])
    minv = np.linalg.inv(m)
    out = []
    for op in GetSpaceGroup(sg).iter_symops():
        r = m @ np.asarray(op.R, float) @ minv
        r = r * np.sign(np.linalg.det(r))
        out.append(tuple(np.round(r, 6).ravel() + 0.0))
    return sorted(out)


def _quat_matrices(quats):
    mats = np.asarray(Rotation(np.asarray(quats)).to_matrix())
    return sorted(tuple(np.round(m, 6).ravel() + 0.0) for m in mats)


# ----------------------------------------------------------------- sht_io ---
def test_sht_table_gives_the_texts_of_the_seven():
    from backend.spherical_gpu.pipeline.sht_io import _SG_TO_POINT_GROUP

    for sg, text in SEVEN.items():
        assert _SG_TO_POINT_GROUP[sg] == text
        assert frame_symmetry(_SG_TO_POINT_GROUP[sg]) is frame_symmetry(space_group=sg)


#: The table of point-group names by space-group number as it was before the
#: seven had texts of their own: (first, last, name), International Tables.
_OLD_RANGES = [
    (1, 1, "1"), (2, 2, "-1"), (16, 24, "222"), (25, 46, "mm2"), (47, 74, "mmm"),
    (75, 80, "4"), (81, 82, "-4"), (83, 88, "4/m"), (89, 98, "422"),
    (99, 110, "4mm"), (111, 122, "-42m"), (123, 142, "4/mmm"),
    (143, 146, "3"), (147, 148, "-3"), (149, 155, "32"), (156, 161, "3m"),
    (162, 167, "-3m"), (168, 173, "6"), (174, 174, "-6"), (175, 176, "6/m"),
    (177, 182, "622"), (183, 186, "6mm"), (187, 190, "-6m2"), (191, 194, "6/mmm"),
    (195, 199, "23"), (200, 206, "m-3"), (207, 214, "432"), (215, 220, "-43m"),
    (221, 230, "m-3m"),
]


def test_sht_table_is_bit_identical_for_every_other_number():
    from backend.spherical_gpu.pipeline.sht_io import _SG_TO_POINT_GROUP

    want = {n: name for lo, hi, name in _OLD_RANGES for n in range(lo, hi + 1)}
    checked = 0
    for n, name in want.items():
        if n in SEVEN:
            continue
        assert _SG_TO_POINT_GROUP[n] == name, n
        checked += 1
    assert checked == 230 - 13 - 7        # no monoclinic (3..15), not the seven
    assert len(_SG_TO_POINT_GROUP) == 230


# -------------------------------------------------------------- pseudosym ---
def test_sym_quats_of_minus_31m_are_the_diffpy_operations():
    from backend.spherical_gpu.pseudosym import _sym_quats

    q = _sym_quats("-31m")
    assert q.shape == (12, 4)
    assert _quat_matrices(q) == _cartesian_rotation_parts(162)
    assert _quat_matrices(_sym_quats("31m")) == _cartesian_rotation_parts(157)
    assert _quat_matrices(_sym_quats("312")) == _cartesian_rotation_parts(149)


def test_sym_quats_of_minus_3m_are_orix_d3d_and_differ_from_minus_31m():
    from backend.spherical_gpu.pseudosym import _sym_quats

    assert np.array_equal(_sym_quats("-3m"), np.asarray(D3d.data, dtype=np.float64))
    assert _quat_matrices(_sym_quats("-3m")) == _cartesian_rotation_parts(164)
    assert _quat_matrices(_sym_quats("-3m")) != _quat_matrices(_sym_quats("-31m"))
    assert np.array_equal(_sym_quats("3m"), np.asarray(C3v.data, dtype=np.float64))


def test_holohedry_of_the_three_texts_is_6_mmm():
    from backend.spherical_gpu.pseudosym import pseudosym_holohedry

    for text in ("312", "31m", "-31m"):
        assert pseudosym_holohedry(text) == "6/mmm"


def test_pseudosym_helpers_tell_the_two_settings_apart():
    from backend.spherical_gpu.pseudosym import (_snap_ops, _sym_quats,
                                                 same_orientation_angle_deg)

    ident = np.array([[1.0, 0.0, 0.0, 0.0]])
    c2x = np.array([0.0, 1.0, 0.0, 0.0])           # 2-fold about X
    c2y = np.array([0.0, 0.0, 1.0, 0.0])           # 2-fold about Y
    # -3m (X setting): the X two-fold is a symmetry operation, the Y one is not.
    assert same_orientation_angle_deg(ident, c2x, "-3m")[0] < 1e-3
    assert same_orientation_angle_deg(ident, c2y, "-3m")[0] == pytest.approx(60.0, abs=1e-3)
    # -31m (Y setting): the other way round.
    assert same_orientation_angle_deg(ident, c2y, "-31m")[0] < 1e-3
    assert same_orientation_angle_deg(ident, c2x, "-31m")[0] == pytest.approx(60.0, abs=1e-3)
    # The snap operators are cached by the text; each text keeps its own group.
    h31, in31 = _snap_ops("-31m")
    h3, in3 = _snap_ops("-3m")
    assert in31.sum() == 12 and in3.sum() == 12
    # Each cache entry holds the rotations of its own group, not the other one's.
    assert set(_quat_matrices(h31[in31])) == set(_cartesian_rotation_parts(162))
    assert set(_quat_matrices(h3[in3])) == set(_cartesian_rotation_parts(164))
    assert set(_cartesian_rotation_parts(162)) != set(_cartesian_rotation_parts(164))


def test_proper_coset_of_minus_31m_is_the_other_half_of_d6():
    from backend.spherical_gpu.pipeline.variant_unification import proper_coset_class_reps
    from backend.spherical_gpu.pseudosym import same_orientation_angle_deg

    reps = proper_coset_class_reps("-31m")
    assert reps.shape[0] == 2                      # D6 over D3: two classes
    # The nontrivial class is the 60-degree turn about c (or an X two-fold):
    # no element of -31m, so the representative is 60 degrees from the identity.
    ident = np.array([[1.0, 0.0, 0.0, 0.0]])
    assert same_orientation_angle_deg(ident, reps[1], "-31m")[0] == pytest.approx(60.0, abs=1e-3)
    # For -3m the nontrivial class is the one containing the Y two-fold.
    reps3 = proper_coset_class_reps("-3m")
    assert reps3.shape[0] == 2
    assert same_orientation_angle_deg(ident, reps3[1], "-3m")[0] == pytest.approx(60.0, abs=1e-3)


# ------------------------------------------------------------------- grid ---
@pytest.mark.parametrize("text", ["312", "31m", "-31m"])
def test_grid_gives_the_same_orientations_for_the_text_and_the_object(text):
    from backend.dict_gpu.pipeline.grid import sample_orientations

    by_text = sample_orientations(text, 10.0)
    by_object = sample_orientations(frame_symmetry(text), 10.0)
    assert by_text.size == by_object.size > 0
    assert np.array_equal(np.asarray(by_text.data), np.asarray(by_object.data))


def test_grid_of_minus_31m_is_not_the_grid_of_minus_3m():
    from backend.dict_gpu.pipeline.grid import sample_orientations

    a = sample_orientations("-31m", 10.0)
    b = sample_orientations("-3m", 10.0)
    c = sample_orientations(D3d, 10.0)
    assert np.array_equal(np.asarray(b.data), np.asarray(c.data))
    assert a.size != b.size or not np.allclose(np.asarray(a.data), np.asarray(b.data))


# ------------------------------------------------------- Phase construction ---
def test_phase_symmetry_arguments():
    assert cs.phase_symmetry_arguments("-31m", 162) == {"space_group": 162}
    assert cs.phase_symmetry_arguments("312", 149) == {"space_group": 149}
    assert cs.phase_symmetry_arguments("31m", "159") == {"space_group": 159}
    # everything else is the point group alone, as before
    assert cs.phase_symmetry_arguments("-3m", 164) == {"point_group": "-3m"}
    assert cs.phase_symmetry_arguments("-31m", None) == {"point_group": "-31m"}
    assert cs.phase_symmetry_arguments("m-3m", 225) == {"point_group": "m-3m"}
    assert cs.phase_symmetry_arguments("-3m", 162) == {"point_group": "-3m"}
    assert cs.phase_symmetry_arguments("2/m", 12) == {"point_group": "2/m"}


def _index_result(n):
    import torch
    from backend.spherical_gpu.pipeline.indexer import IndexResult

    rng = np.random.default_rng(5)
    return IndexResult(
        euler_xyz=torch.tensor(rng.uniform(0.2, 3.0, size=(n, 3)), dtype=torch.float32),
        score=torch.tensor(rng.uniform(0.1, 0.9, size=n), dtype=torch.float32),
        phase_id=torch.ones(n, dtype=torch.int8), runtime_seconds=1.0)


def test_to_crystal_map_carries_the_space_group_of_the_seven():
    from backend.spherical_gpu.pipeline.output import to_crystal_map

    xmap = to_crystal_map(_index_result(6), 2, 3, 1.0, 1.0, point_group="-31m",
                          phase_name="Cr2N", space_group=162)
    ph = xmap.phases[1]
    assert ph.space_group.number == 162
    assert ph.point_group is frame_symmetry("-31m")


def test_to_crystal_map_is_unchanged_for_every_other_phase():
    from backend.spherical_gpu.pipeline.output import to_crystal_map

    for pg, sg in (("-3m", 164), ("m-3m", 225), ("2/m", 12), ("32", 150)):
        xmap = to_crystal_map(_index_result(6), 2, 3, 1.0, 1.0, point_group=pg,
                              phase_name="x", space_group=sg)
        assert xmap.phases[1].space_group is None
        assert xmap.phases[1].point_group.name == (
            "321" if pg == "32" else pg)        # orix reads "32" as "321"
    legacy = to_crystal_map(_index_result(6), 2, 3, 1.0, 1.0, point_group="m-3m",
                            phase_name="x")                  # no space group given
    assert legacy.phases[1].space_group is None


def _roundtrip_through_the_app(monkeypatch, tmp_path, xmap):
    """Export through the route the app saves with, load with the app's reader and
    let the app's registry frame the phases (what ``_store_result`` does)."""
    from tests.test_export_roi import _patch_route_seams, _run, _release_h5_session
    from backend.api.routes.analysis import _load_kikuchipy_rich_h5

    n = xmap.rotations.size
    mask = np.ones((2, 3), bool)
    result = SimpleNamespace(xmap=xmap, original_shape=(2, 3), selection_mask=mask,
                             confidence_scores=np.full(n, 0.5, np.float32),
                             method=SimpleNamespace(value="spherical"), metadata={})
    ix = _patch_route_seams(monkeypatch, result)
    out = tmp_path / "out.h5"
    _run(ix.export_indexing_result(ix.ExportRequest(
        format="h5_light", filename=str(out), include_eds=False,
        include_detector=False)))
    _release_h5_session()
    back = _load_kikuchipy_rich_h5(str(out))
    cs.frame_xmap(back)
    return back


def test_a_saved_and_reloaded_spherical_result_keeps_minus_31m(monkeypatch, tmp_path):
    from backend.spherical_gpu.pipeline.output import to_crystal_map

    xmap = to_crystal_map(_index_result(6), 2, 3, 1.0, 1.0, point_group="-31m",
                          phase_name="Cr2N", space_group=162)
    back = _roundtrip_through_the_app(monkeypatch, tmp_path, xmap)
    assert back.phases[0].point_group is frame_symmetry("-31m")


def test_without_the_space_group_the_reload_falls_back_to_orix_minus_3m(monkeypatch, tmp_path):
    """What the spherical result did before the space group went into the phase."""
    from backend.spherical_gpu.pipeline.output import to_crystal_map

    xmap = to_crystal_map(_index_result(6), 2, 3, 1.0, 1.0, point_group="-31m",
                          phase_name="Cr2N")
    back = _roundtrip_through_the_app(monkeypatch, tmp_path, xmap)
    assert back.phases[0].point_group is not frame_symmetry("-31m")
    assert back.phases[0].point_group == D3d


def test_project_save_and_load_keep_minus_31m(tmp_path):
    import project_manager
    from indexing_controller import IndexingMethod, IndexingResult
    from backend.spherical_gpu.pipeline.output import to_crystal_map

    xmap = to_crystal_map(_index_result(6), 2, 3, 1.0, 1.0, point_group="-31m",
                          phase_name="Cr2N", space_group=162)
    entry = project_manager.GalleryEntry(
        name="cr2n", method="Spherical", phase_data=None,
        indexing_result=IndexingResult(
            xmap=xmap, selection_mask=np.ones((2, 3), bool), original_shape=(2, 3),
            method=IndexingMethod.SPHERICAL))
    project_manager.save_project(str(tmp_path / "proj"), [entry], {})
    _meta, entries = project_manager.load_project(str(tmp_path / "proj"))
    loaded = entries[0].indexing_result.xmap
    assert point_group_key(loaded.phases[1].point_group) == "-31m"


def test_tier1_indexer_hands_its_masters_space_group_to_the_map(tmp_path, monkeypatch):
    import h5py
    from backend.spherical_gpu.pipeline.indexer import Tier1Indexer

    path = tmp_path / "scan.h5oina"
    with h5py.File(path, "w") as f:
        h = f.create_group("1/EBSD/Header")
        h["Y Cells"] = np.array([2])
        h["X Cells"] = np.array([3])
        h["X Step"] = np.array([0.5])
        h["Y Step"] = np.array([0.5])
    ix = Tier1Indexer.__new__(Tier1Indexer)
    ix.master = SimpleNamespace(point_group="-31m", space_group=162, formula="Cr2N")
    monkeypatch.setattr(Tier1Indexer, "index_h5oina", lambda self, *a, **k: _index_result(6))
    xmap = ix.index_h5oina_to_xmap(str(path))
    assert xmap.phases[1].space_group.number == 162
    assert xmap.phases[1].point_group is frame_symmetry("-31m")


def test_cpu_dictionary_samples_the_phases_marked_group(monkeypatch):
    """The dictionary of a master is sampled over the group the phase has in the
    crystal frame, handed over as the group object (a text would lose it)."""
    import indexing_controller as ic
    import backend.dict_gpu.pipeline.grid as grid

    seen = []

    def stop(point_group, step):
        seen.append(point_group)
        raise RuntimeError("stop after sampling")

    monkeypatch.setattr(grid, "sample_orientations", stop)
    master = SimpleNamespace(hemisphere="both", projection="lambert",
                             phase=Phase("Cr2N", space_group=162, structure=_structure()))
    with pytest.raises(RuntimeError, match="stop after sampling"):
        ic._dictionary_signal_from_master(master, None, 5.0)
    assert seen[0] is frame_symmetry("-31m")


# --------------------------------------------- the spherical controller ---
class _FakeBackend:
    """Stands in for the GPU backend: one Cr2N master of space group 162."""

    def __init__(self, cfg):
        self.config = cfg
        self.runtime = SimpleNamespace(device_name="cpu", free_memory_gb=99.0,
                                       cuda_available=False, device="cpu")
        self._indexers = [SimpleNamespace(master=SimpleNamespace(
            formula="Cr2N", point_group="-31m", space_group=162, z_rot=1))]

    def set_cancel_check(self, fn):
        pass

    def index_h5(self, **_kw):
        return _index_result(6)

    def invalidate(self):
        pass


def test_spherical_controller_builds_the_phase_with_its_space_group(monkeypatch):
    import backend.spherical_gpu.backend as gpu_backend
    import indexing_controller as ic

    monkeypatch.setattr(gpu_backend, "SphericalGPUBackend", _FakeBackend)
    monkeypatch.setattr(ic, "_get_processed_patterns_if_dirty", lambda *a, **k: None)
    monkeypatch.setattr(ic, "_materialize_patterns_for_resolution", lambda *a, **k: None)
    cfg = ic.IndexingConfig(method=ic.IndexingMethod.SPHERICAL)
    result = ic.spherical_gpu_index_patterns(
        "unused.h5", cfg, {"n_rows": 2, "n_cols": 3}, sht_paths=["Cr2N.sht"])
    phase = result.xmap.phases[1]
    assert phase.space_group.number == 162
    assert phase.point_group is frame_symmetry("-31m")


# ----------------------------------------------------------- route helpers ---
def _two_phase_result():
    """Cr2N (162, -31m) and corundum (167, -3m) as plain orix phases, framed."""
    cr2n = Phase("Cr2N", space_group=162, structure=_structure())
    corundum = Phase("Corundum", space_group=167, structure=_structure())
    xmap = _map({1: cr2n, 2: corundum})
    cs.frame_xmap(xmap)
    return xmap


def test_phase_point_group_and_name_gives_the_key():
    from backend.api.routes.indexing import _phase_point_group_and_name

    xmap = _two_phase_result()
    assert _phase_point_group_and_name(xmap, 1, "Cr2N.sht")[0] == "-31m"
    assert _phase_point_group_and_name(xmap, 2, "Corundum.sht")[0] == "-3m"


def test_rescue_pairs_do_not_join_cr2n_and_corundum():
    """The pairing asks for the SAME point group; -31m and -3m are two."""
    from backend.api.routes.indexing import _phase_point_group_and_name
    from backend.api.services.eds_particle_rescue import select_rescue_pairs

    xmap = _two_phase_result()
    pgs = {pid: _phase_point_group_and_name(xmap, pid, "x.sht")[0] for pid in (1, 2)}
    expected = {1: {"Cr": 66.0, "N": 33.0}, 2: {"Al": 40.0, "O": 60.0}}
    assert select_rescue_pairs([1, 2], expected, pgs, {1: "Cr2N", 2: "Corundum"}) == []
    same = {1: "-3m", 2: "-3m"}
    assert select_rescue_pairs([1, 2], expected, same, {1: "Cr2N", 2: "Corundum"}) != []


def _spherical_result(xmap, n=4):
    md = {"indexing_method": "spherical",
          "sht_paths_by_phase": {1: "a.sht", 2: "b.sht"},
          "detector_geometry": {"dummy": 1}}
    return SimpleNamespace(xmap=xmap, metadata=md, original_shape=(n, n),
                           selection_mask=np.ones((n, n), bool))


def test_pixel_context_hands_on_the_key():
    from backend.api.routes.indexing import _spherical_pixel_ctx

    xmap = _two_phase_result()
    result = _spherical_result(xmap)
    pg_cr2n = _spherical_pixel_ctx(result, 0, 0)[5]       # pixel 0 -> id 1
    pg_corundum = _spherical_pixel_ctx(result, 0, 1)[5]         # pixel 1 -> id 2
    assert (pg_cr2n, pg_corundum) == ("-31m", "-3m")


def test_phase_check_context_holds_the_key_per_phase():
    from backend.api.routes.indexing import _phase_check_ctx

    ctx = _phase_check_ctx(_spherical_result(_two_phase_result()), build_scorers=False)
    assert ctx["phases"][1] == "-31m"
    assert ctx["phases"][2] == "-3m"


# --------------------------------------------------------------- phase map ---
@pytest.mark.parametrize("order", [(1, 2), (2, 1)])
def test_ipf_keys_do_not_share_a_triangle_between_cr2n_and_corundum(order):
    from backend.api.routes.phase_map import _group_phases_by_symmetry

    cr2n = Phase("Cr2N", space_group=162, structure=_structure())
    corundum = Phase("Corundum", space_group=167, structure=_structure())
    phases = {order[0]: cr2n if order[0] == 1 else corundum,
              order[1]: corundum if order[1] == 2 else cr2n}
    xmap = _map(phases)
    groups = _group_phases_by_symmetry(xmap)
    assert sorted(groups) == ["-31m", "-3m"]
    assert groups["-31m"]["phases"] == ["Cr2N"]
    assert groups["-3m"]["phases"] == ["Corundum"]
    assert groups["-31m"]["symmetry"] is frame_symmetry("-31m").laue
    assert groups["-3m"]["symmetry"] == D3d.laue


def test_ipf_keys_still_share_a_triangle_between_equal_groups():
    from backend.api.routes.phase_map import _group_phases_by_symmetry

    a = Phase("Corundum", space_group=167, structure=_structure())
    b = Phase("Bi2Te3", space_group=166, structure=_structure())
    groups = _group_phases_by_symmetry(_map({1: a, 2: b}))
    assert list(groups) == ["-3m"] and groups["-3m"]["phases"] == ["Corundum", "Bi2Te3"]


def test_grain_consistent_ipf_colouring_hands_the_key_to_the_helpers(monkeypatch):
    """tools.phase_map_generator hands each phase's group to the pseudo-symmetry
    helpers as text; Cr2N must arrive as "-31m" and corundum as "-3m"."""
    import backend.spherical_gpu.pseudosym as ps
    from tools import phase_map_generator as pmg

    seen = []
    real = ps._sym_quats

    def spy(pg):
        seen.append(pg)
        return real(pg)

    monkeypatch.setattr(ps, "_sym_quats", spy)
    pmg.compute_ipf_colors_grain_consistent(_two_phase_result(), "Z", n_rows=4, n_cols=4)
    assert "-31m" in seen and "-3m" in seen


# ---------------------------------------------------------- the old D3y ---
def test_a_plain_312_from_orix_reaches_the_marked_group_through_a_phase():
    ph = FramePhase("p", structure=_structure())
    ph.point_group = D3y
    assert point_group_key(ph.point_group) == "312"
    assert phase_point_group_key(ph) == "312"


def test_phase_point_group_key_of_nothing_is_none():
    assert phase_point_group_key(None) is None
    assert phase_point_group_key(Phase("x")) is None


# ------------------------------------- unify_pseudosym_variants, file branches ---
def _cubic_phase():
    return Phase("Al", space_group=225,
                 structure=Structure(lattice=Lattice(4.05, 4.05, 4.05, 90, 90, 90)))


def test_unify_endpoint_hands_each_phase_its_key_and_writes_the_orientations(monkeypatch):
    """Cr2N (162) must reach ``unify_map`` as "-31m", the cubic control as "m-3m",
    and what ``unify_map`` returns must land in the map's rotations."""
    import asyncio

    import backend.api.routes.indexing as ix
    import backend.spherical_gpu.pipeline.variant_unification as vu

    xmap = _map({1: Phase("Cr2N", space_group=162, structure=_structure()),
                 2: _cubic_phase()})
    result = _spherical_result(xmap)
    before = np.asarray(xmap.rotations.data).reshape(-1, 4).copy()
    seen = {}

    def fake_unify(full_q, phase_full, n_rows, n_cols, pid, pg, score_fn, **kw):
        seen[pid] = pg
        new = full_q.copy()
        new[phase_full == pid] = [1.0, 0.0, 0.0, 0.0]       # identity: a visible edit
        return new, {"n_grains": 1}

    monkeypatch.setattr(ix, "_get_result", lambda *_a, **_k: result)
    monkeypatch.setattr(vu, "unify_map", fake_unify)
    monkeypatch.setattr(vu, "build_render_score_fn", lambda *a, **k: (lambda i, q: None))
    out = asyncio.run(ix.unify_pseudosym_variants(ix.UnifyVariantsRequest()))

    assert seen == {1: "-31m", 2: "m-3m"}
    assert out["n_changed"] > 0 and out["undo_available"] is True
    after = np.asarray(xmap.rotations.data).reshape(-1, 4)
    assert np.allclose(np.abs(after[:, 0]), 1.0)             # every pixel was edited
    assert not np.allclose(after, before)


_CIF = """data_{name}
_cell_length_a {a}
_cell_length_b {a}
_cell_length_c {c}
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma {gamma}
_symmetry_space_group_name_H-M '{hm}'
_symmetry_Int_Tables_number {number}
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
{atom} {atom} 0.0 0.0 0.0
"""


def _write_cif(tmp_path, name, **kw):
    path = tmp_path / f"{name}.cif"
    path.write_text(_CIF.format(name=name, **kw), encoding="utf-8")
    return str(path)


def _empty_xmap():
    """A map that holds phase 1 only, so a lookup of phase 2 falls to the file."""
    return _map({1: _cubic_phase()})


@pytest.mark.parametrize("name, kw, expected", [
    ("Cr2N", dict(a=4.8, c=4.5, gamma=120, hm="P -3 1 m", number=162, atom="Cr"),
     "-31m"),
    ("Al", dict(a=4.05, c=4.05, gamma=90, hm="F m -3 m", number=225, atom="Al"),
     "m-3m"),
])
def test_phase_point_group_and_name_reads_the_key_from_a_cif(tmp_path, name, kw, expected):
    from backend.api.routes.indexing import _phase_point_group_and_name

    path = _write_cif(tmp_path, name, **kw)
    assert _phase_point_group_and_name(_empty_xmap(), 2, path) == (expected, name)


@pytest.mark.parametrize("name, sg, expected", [
    ("Cr2N", 162, "-31m"), ("Al", 225, "m-3m")])
def test_phase_point_group_and_name_reads_the_key_from_a_kikuchipy_file(
        tmp_path, monkeypatch, name, sg, expected):
    """The ``.h5`` branch asks ``kikuchipy.load(path, lazy=True)`` for its phase.
    A real master file needs EMsoft output, so the loader is a stand-in that
    returns a signal object carrying a plain orix phase of the space group."""
    import kikuchipy as kp
    from backend.api.routes.indexing import _phase_point_group_and_name

    lattice = _structure().lattice if sg == 162 else Lattice(4.05, 4.05, 4.05, 90, 90, 90)
    phase = Phase(name, space_group=sg, structure=Structure(lattice=lattice))
    asked = []

    def fake_load(path, lazy=False, **kw):
        asked.append((path, lazy))
        return SimpleNamespace(phase=phase)

    monkeypatch.setattr(kp, "load", fake_load)
    path = str(tmp_path / f"{name}_master.h5")
    assert _phase_point_group_and_name(_empty_xmap(), 2, path) == (expected, f"{name}_master")
    assert asked == [(path, True)]


# --------------------------- tables keyed by the name of the group ("312") ---
@pytest.mark.parametrize("setting, name", [("312", "312"), ("31m", "3m"), ("-31m", "-3m")])
def test_tsl_code_of_the_marked_groups_is_32(setting, name):
    from backend.api.services import ang_export

    sym = frame_symmetry(setting)
    assert sym.name == name
    assert ang_export.tsl_symmetry_code(sym) == "32"


@pytest.mark.parametrize("setting", ["312", "31m", "-31m"])
def test_pole_figure_defaults_of_the_marked_groups_are_the_hexagonal_ones(setting):
    from backend.api.services.pole_figure import default_hkl_families

    assert default_hkl_families(frame_symmetry(setting)) == [(0, 0, 1), (1, 0, 0), (1, 0, 1)]
    assert default_hkl_families(frame_symmetry(setting)) == default_hkl_families(D3)


@pytest.mark.parametrize("point_group, sg", [("312", None), ("312", 149), ("3m", 157),
                                          ("-3m", 162), ("32", 150)])
def test_batch_ctf_export_gives_laue_class_7_to_every_trigonal_name(
        tmp_path, monkeypatch, point_group, sg):
    """The name the checkpoint records ("312" for 149) must have a Laue class.

    The phase is built from the recorded symmetry alone, whose default hexagonal
    cell (1 1 1 90 90 120) the exporter does not recognise as a placeholder; the
    structure's lattice is therefore ignored here so that the recorded cell is
    the one used, as it is for a cell that is not hexagonal."""
    from tests.test_crystal_symmetry_call_sites import _checkpoint_with_phase
    from backend.api.services import ang_export
    from backend.api.services.result_exporter import export_ang_ctf

    monkeypatch.setattr(ang_export, "lattice_from_structure", lambda *a, **k: None)

    source, cw = _checkpoint_with_phase(
        tmp_path, point_group, sg, [4.8, 4.8, 4.5, 90, 90, 120])
    _ang, ctf = export_ang_ctf(cw.checkpoint_path, str(tmp_path / "o"),
                               source_h5_path=str(source))
    rows = open(ctf, encoding="utf-8").read().splitlines()
    phase_row = rows[next(i for i, r in enumerate(rows) if r.startswith("Phases")) + 1]
    # a;b;c  alpha;beta;gamma  name  Laue class  space group
    assert phase_row.split("\t")[3] == "7"


@pytest.mark.parametrize("sg, label", [(149, "312"), (157, "3m"), (162, "-3m"),
                                       (150, "32"), (164, "-3m")])
@pytest.mark.parametrize("kind", [Phase, FramePhase])
def test_reflector_panel_labels_the_point_group_alike_for_plain_and_frame_phase(
        sg, label, kind):
    import hough_reflectors as hr

    phase = kind("p", space_group=sg, structure=_structure())
    assert hr.describe(phase, None)["point_group"] == label
