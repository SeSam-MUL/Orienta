"""The grain tools (flood fill, enclosed islands, phase check) on the six neighbours of a hexagonal scan."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services.scan_geometry import ScanGeometry  # noqa: E402
from backend.api.services.scan_neighbours import Topology  # noqa: E402
from backend.spherical_gpu.pipeline import phase_reassignment as pr  # noqa: E402
from backend.spherical_gpu.pipeline.island_check import check_islands, find_enclosed_islands  # noqa: E402
from backend.spherical_gpu.pseudosym import grain_snap_floodfill  # noqa: E402

SIGNS = (0.5, -0.5)


def _qz(deg):
    a = np.radians(deg) / 2.0
    return np.array([np.cos(a), 0.0, 0.0, np.sin(a)])


def _distances(shift, n_rows, n_cols):
    g = ScanGeometry.hex(n_rows, n_cols, 1.0, odd_row_shift=shift)
    x, y = g.positions()
    pts = np.stack([x.ravel(), y.ravel()], axis=1)
    return np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])


def _zigzag(shift, n_rows, n_cols, start_col):
    """One cell per row, each one step from the previous one but never in the same column: found from the
    positions. The cells touch only through the diagonal link of the offset lattice."""
    dist = _distances(shift, n_rows, n_cols)
    chain = [(0, start_col)]
    for r in range(1, n_rows):
        pr_, pc = chain[-1]
        for c in (pc - 1, pc + 1):
            if 0 <= c < n_cols and abs(dist[pr_ * n_cols + pc, r * n_cols + c] - 1.0) < 1e-6:
                chain.append((r, c))
                break
    assert len(chain) == n_rows
    return chain


# ----------------------------------------------------------------------- the flood fill

@pytest.mark.parametrize("shift", SIGNS)
def test_the_flood_fill_follows_the_diagonal_links_of_a_hexagonal_scan(shift):
    nr = nc = 8
    chain = _zigzag(shift, nr, nc, start_col=4)
    qa = _qz(5.0)
    full_q = np.tile(qa, (nr * nc, 1))
    phase = np.ones(nr * nc, dtype=int)
    for r, c in chain:
        phase[r * nc + c] = 0                                  # the grain: only the chain is phase 0
    hexagonal, _, _ = grain_snap_floodfill(full_q, phase, nr, nc, chain[0], 0, "m-3", qa,
                                           topology=Topology("hex", nr, nc, shift))
    legacy, _, _ = grain_snap_floodfill(full_q, phase, nr, nc, chain[0], 0, "m-3", qa)
    assert sorted(hexagonal) == sorted(chain)                  # the whole chain: eight points
    assert legacy == [chain[0]]                                # nothing but the point clicked


def test_a_square_topology_fills_like_the_index_grid():
    nr, nc = 6, 7
    rng = np.random.default_rng(5)
    qa = _qz(5.0)
    full_q = np.tile(qa, (nr * nc, 1))
    full_q[rng.random(nr * nc) < 0.2] = _qz(40.0)
    phase = np.zeros(nr * nc, dtype=int)
    old = grain_snap_floodfill(full_q, phase, nr, nc, (2, 3), 0, "m-3", qa)
    new = grain_snap_floodfill(full_q, phase, nr, nc, (2, 3), 0, "m-3", qa, topology=Topology("square", nr, nc))
    assert sorted(old[0]) == sorted(new[0])
    assert old[2] == new[2]


# --------------------------------------------------------------------- enclosed islands

@pytest.mark.parametrize("shift", SIGNS)
def test_a_diagonal_pair_is_one_island_on_a_hexagonal_scan_and_two_on_the_index_grid(shift):
    nr = nc = 6
    pair = tuple(_zigzag(shift, nr, nc, start_col=2)[1:3])        # two cells that touch only diagonally
    phase = np.zeros(nr * nc, dtype=int)
    for r, c in pair:
        phase[r * nc + c] = 1
    hexagonal = find_enclosed_islands(phase, nr, nc, topology=Topology("hex", nr, nc, shift))
    legacy = find_enclosed_islands(phase, nr, nc)
    assert [i["pixels"].size for i in hexagonal] == [2]
    assert sorted(i["pixels"].size for i in legacy) == [1, 1]
    dist = _distances(shift, nr, nc)
    for px, nb in hexagonal[0]["neighbour_of"].items():
        assert abs(dist[px, nb] - 1.0) < 1e-6                      # the orientation is borrowed from a point one step away


def test_a_square_topology_finds_the_islands_of_the_index_grid():
    nr = nc = 7
    phase = np.zeros(nr * nc, dtype=int)
    for r, c in ((1, 1), (4, 4), (4, 5)):
        phase[r * nc + c] = 1
    old = find_enclosed_islands(phase, nr, nc)
    new = find_enclosed_islands(phase, nr, nc, topology=Topology("square", nr, nc))
    assert [list(i["pixels"]) for i in old] == [list(i["pixels"]) for i in new]


def _isolated_middle(shift, start_col, nr=9, nc=9):
    """Every way to build this: an island of three points in a zigzag (a, m, b, row by row) in a field of phase 0,
    where m has no indexed neighbour of its own and a and b each have ONE indexed neighbour outside the island,
    ``ra`` and ``rb``, taken among the neighbours of a and b that are not neighbours of m. m must borrow from one
    of the two. Yields ``(topology, phase, m, ra, rb)`` for every choice of the pair."""
    topology = Topology("hex", nr, nc, shift)
    a, m, b = (r * nc + c for r, c in _zigzag(shift, nr, nc, start_col=start_col)[2:5])
    island = {a, m, b}
    near_m = set(topology.neighbours(m))
    for ra in (n for n in topology.neighbours(a) if n not in island and n not in near_m):
        for rb in (n for n in topology.neighbours(b) if n not in island and n not in near_m):
            if ra == rb:
                continue
            phase = np.zeros(nr * nc, dtype=int)
            phase[[a, m, b]] = 1
            for px in island:
                for nb in topology.neighbours(px):
                    if nb not in island and nb not in (ra, rb):
                        phase[nb] = -1
            yield topology, phase, m, ra, rb


@pytest.mark.parametrize("shift", SIGNS)
def test_a_pixel_with_no_neighbour_of_its_own_borrows_from_the_nearest_ring_pixel_on_the_sample(shift):
    nr = nc = 9
    x, y = (a.ravel() for a in ScanGeometry.hex(nr, nc, 1.0, odd_row_shift=shift).positions())
    cases = 0
    for start_col in range(2, 7):
        for topology, phase, m, ra, rb in _isolated_middle(shift, start_col):
            on_sample = [float(np.hypot(x[r] - x[m], y[r] - y[m])) for r in (ra, rb)]
            in_index = [abs(r // nc - m // nc) + abs(r % nc - m % nc) for r in (ra, rb)]
            if abs(on_sample[0] - on_sample[1]) < 1e-6:
                continue                                              # a tie on the sample: the next test
            if int(np.argmin(on_sample)) == int(np.argmin(in_index)):
                continue                                              # the premise: the two rules disagree
            (island,) = find_enclosed_islands(phase, nr, nc, topology=topology)
            assert island["neighbour_of"][m] == (ra, rb)[int(np.argmin(on_sample))]
            cases += 1
    assert cases > 0


@pytest.mark.parametrize("shift", SIGNS)
def test_two_ring_pixels_at_the_same_distance_go_to_the_lower_index(shift):
    # Two ring pixels at the same distance from m in exact arithmetic (sqrt(3) steps, a tie by construction)
    # differ in the last bit of the float distance, which must not decide: the lower index wins, as on the
    # index grid.
    nr = nc = 9
    x, y = (a.ravel() for a in ScanGeometry.hex(nr, nc, 1.0, odd_row_shift=shift).positions())
    cases = 0
    for start_col in range(2, 7):
        for topology, phase, m, ra, rb in _isolated_middle(shift, start_col):
            if abs(np.hypot(x[ra] - x[m], y[ra] - y[m]) - np.hypot(x[rb] - x[m], y[rb] - y[m])) > 1e-9:
                continue
            (island,) = find_enclosed_islands(phase, nr, nc, topology=topology)
            assert island["neighbour_of"][m] == min(ra, rb)
            cases += 1
    assert cases >= 4

@pytest.mark.parametrize("shift", SIGNS)
def test_check_islands_counts_the_islands_of_the_topology_it_is_given(shift):
    nr = nc = 6
    pair = _zigzag(shift, nr, nc, start_col=2)[1:3]
    phase = np.zeros(nr * nc, dtype=int)
    for r, c in pair:
        phase[r * nc + c] = 1
    full_q = np.tile(_qz(5.0), (nr * nc, 1))
    phases = {0: "m-3", 1: "m-3"}
    hexagonal = check_islands(full_q, phase, nr, nc, phases, {}, topology=Topology("hex", nr, nc, shift))
    legacy = check_islands(full_q, phase, nr, nc, phases, {})
    assert (hexagonal["n_islands"], legacy["n_islands"]) == (1, 2)


# ----------------------------------------------------------------------- the core sample

@pytest.mark.parametrize("shift", SIGNS)
def test_the_core_of_a_hexagonal_grain_needs_all_six_neighbours_inside(shift):
    nr = nc = 12
    topology = Topology("hex", nr, nc, shift)
    p0 = 6 * nc + 6
    # a neighbour of p0 that is no index-grid neighbour of it (another row AND another column)
    diagonal = next(n for n in topology.neighbours(p0) if n // nc != 6 and n % nc != 6)
    grain = np.array([r * nc + c for r in range(2, 10) for c in range(2, 10) if r * nc + c != diagonal])
    inset = set(grain.tolist())
    hexagonal = set(pr._grain_core_sample(grain, nr, nc, k=40, topology=topology).tolist())
    legacy = set(pr._grain_core_sample(grain, nr, nc, k=40).tolist())
    assert p0 in legacy and p0 not in hexagonal              # its four index neighbours are inside, one of its six is not
    for p in hexagonal:
        nbs = topology.neighbours(int(p))
        assert len(nbs) == 6 and all(n in inset for n in nbs)
    assert hexagonal < legacy


@pytest.mark.parametrize("shift", SIGNS)
def test_a_point_on_the_edge_of_the_scan_is_never_core(shift):
    nr = nc = 8
    topology = Topology("hex", nr, nc, shift)
    grain = np.arange(nr * nc)                                       # the whole scan is one grain
    sample = set(pr._grain_core_sample(grain, nr, nc, k=40, topology=topology).tolist())
    six = {p for p in range(nr * nc) if len(topology.neighbours(p)) == 6}
    assert sample == six and len(six) < nr * nc


def _core_from_positions(grain, shift, n_rows, n_cols):
    """The sorted grain points with six neighbours at one step, all of them in the grain: from the positions."""
    dist = _distances(shift, n_rows, n_cols)
    inset = {int(g) for g in grain}
    core = []
    for p in sorted(inset):
        nbs = np.flatnonzero((dist[p] > 1e-6) & (dist[p] <= 1.0 + 1e-6))
        if len(nbs) == 6 and all(int(n) in inset for n in nbs):
            core.append(p)
    return np.asarray(core, dtype=np.int64)


def _block(n_cols, r0, r1, c0, c1):
    return np.asarray([r * n_cols + c for r in range(r0, r1) for c in range(c0, c1)], dtype=np.int64)


@pytest.mark.parametrize("shift", SIGNS)
def test_a_thin_chain_has_no_core_and_the_sample_is_the_whole_chain(shift):
    nr = nc = 12
    topology = Topology("hex", nr, nc, shift)
    chain = np.asarray(sorted(r * nc + c for r, c in _zigzag(shift, nr, nc, start_col=5)), dtype=np.int64)
    assert _core_from_positions(chain, shift, nr, nc).size == 0              # the premise: nothing to prefer
    sample = pr._grain_core_sample(chain, nr, nc, k=16, topology=topology)
    np.testing.assert_array_equal(sample, chain)                              # twelve points, all of them
    np.testing.assert_array_equal(
        pr._grain_core_sample(chain[::-1], nr, nc, k=16, topology=topology), chain)   # unsorted input


@pytest.mark.parametrize("shift", SIGNS)
def test_a_core_smaller_than_half_the_sample_falls_back_to_the_whole_grain(shift):
    nr = nc = 14
    topology = Topology("hex", nr, nc, shift)
    grain = _block(nc, 2, 6, 2, 10)                                           # 4 x 8 points
    core = _core_from_positions(grain, shift, nr, nc)
    assert 3 <= core.size < 20 and grain.size == 32                           # a core, but under k // 2 = 20
    sample = pr._grain_core_sample(grain, nr, nc, k=40, topology=topology)
    np.testing.assert_array_equal(sample, np.sort(grain))                     # the grain, not the core


@pytest.mark.parametrize("shift", SIGNS)
def test_a_grain_without_core_is_thinned_evenly_to_k(shift):
    nr = nc = 14
    topology = Topology("hex", nr, nc, shift)
    grain = _block(nc, 3, 5, 1, 13)                                           # two rows of 12: no core
    assert _core_from_positions(grain, shift, nr, nc).size == 0
    sample = pr._grain_core_sample(grain, nr, nc, k=12, topology=topology)
    np.testing.assert_array_equal(sample, np.sort(grain)[::2])                # every second point, by hand


@pytest.mark.parametrize("k, every", [(12, 3), (9, 4), (18, 2)])
@pytest.mark.parametrize("shift", SIGNS)
def test_a_core_larger_than_k_is_thinned_evenly_over_its_whole_length(shift, k, every):
    nr = nc = 14
    topology = Topology("hex", nr, nc, shift)
    grain = _block(nc, 2, 10, 2, 10)
    core = _core_from_positions(grain, shift, nr, nc)
    assert core.size == 36                                                    # 36 = k * every
    sample = pr._grain_core_sample(grain, nr, nc, k=k, topology=topology)
    np.testing.assert_array_equal(sample, core[::every])                      # first, then every `every`-th
    assert sample.max() > core[k]                                             # not just the first k points


@pytest.mark.parametrize("block", [1, 5, 7])
@pytest.mark.parametrize("shift", SIGNS)
def test_the_blocks_the_core_is_found_in_do_not_change_it(monkeypatch, shift, block):
    nr = nc = 14
    topology = Topology("hex", nr, nc, shift)
    grain = _block(nc, 2, 11, 2, 11)                                          # 81 points: 81 // block + 1 blocks
    core = _core_from_positions(grain, shift, nr, nc)
    assert 0 < core.size < grain.size
    monkeypatch.setattr(pr, "_CORE_BLOCK", block)
    got = pr._hex_core(np.sort(grain), topology)
    np.testing.assert_array_equal(got, core)                                  # every block adds its core points
    sample = pr._grain_core_sample(grain, nr, nc, k=2 * core.size, topology=topology)
    np.testing.assert_array_equal(sample, core)                               # k // 2 = core.size: the core is used


# --------------------------------------------------------- a topology of another grid is refused

@pytest.mark.parametrize("which", ["flood", "islands", "check_islands", "core"])
def test_a_topology_of_another_grid_is_refused_not_applied_quietly(which):
    nr = nc = 6
    wrong = Topology("hex", nr + 1, nc, 0.5)                       # one row more than the map
    qa = _qz(5.0)
    full_q = np.tile(qa, (nr * nc, 1))
    phase = np.zeros(nr * nc, dtype=int)
    call = {
        "flood": lambda: grain_snap_floodfill(full_q, phase, nr, nc, (2, 2), 0, "m-3", qa, topology=wrong),
        "islands": lambda: find_enclosed_islands(phase, nr, nc, topology=wrong),
        "check_islands": lambda: check_islands(full_q, phase, nr, nc, {0: "m-3"}, {}, topology=wrong),
        "core": lambda: pr._grain_core_sample(np.arange(nr * nc), nr, nc, topology=wrong),
    }[which]
    with pytest.raises(ValueError, match="grid"):
        call()


# --------------------------------------------------------------- the phase check and its apply

@pytest.fixture
def seen(monkeypatch):
    got = {"segment": [], "core": []}
    segment, core = pr.segment_supergroup_grains, pr._grain_core_sample

    def spy_segment(*args, **kwargs):
        got["segment"].append(kwargs.get("topology"))
        return segment(*args, **kwargs)

    def spy_core(*args, **kwargs):
        got["core"].append(kwargs.get("topology"))
        return core(*args, **kwargs)

    monkeypatch.setattr(pr, "segment_supergroup_grains", spy_segment)
    monkeypatch.setattr(pr, "_grain_core_sample", spy_core)
    return got


def test_the_check_and_the_apply_use_the_topology_they_are_given(seen):
    from tests.test_phase_reassignment import NC as COLS, NR as ROWS, PHASES, _grid, _hough_fn_factory, _score_fns
    topology = Topology("hex", ROWS, COLS, 0.5)
    full_q, phase = _grid()
    fns, _calls = _score_fns()
    hough = _hough_fn_factory()
    report, _margin = pr.check_map(full_q, phase, ROWS, COLS, PHASES, fns, hough, topology=topology)
    assert report["n_reassign"] == 1
    assert seen["segment"] and all(t is topology for t in seen["segment"])           # the grains
    assert seen["core"] and all(t is topology for t in seen["core"])                 # the sample that decides them
    before = len(seen["segment"])
    pr.apply_reassignment(full_q, phase, ROWS, COLS, report, PHASES, hough, score_fns=fns, topology=topology)
    assert len(seen["segment"]) > before and all(t is topology for t in seen["segment"])   # re-derived with the same one
