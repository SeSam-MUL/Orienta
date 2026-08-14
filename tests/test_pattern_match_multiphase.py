"""Multi-phase Dictionary runs must still show the simulated best match.

The merged result is built fresh in the multi-phase branch of
``backend/api/routes/indexing.py`` and used to carry neither ``dictionary``
nor ``dict_path``; ``build_consensus_xmap`` builds a bare CrystalMap with no
``prop``, so ``simulation_indices`` was gone too. Result: the pattern-match
dialog said "Dictionary not in memory" for every multi-phase run, no matter
how well it had indexed.

Each phase is indexed against its OWN dictionary file, so a single
``dict_path`` cannot describe the merged result — hence
``per_phase_match_sources``.
"""
from pathlib import Path

import numpy as np
import pytest

from tools.pattern_comparison import get_best_match_pattern

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _FakePhase:
    def __init__(self, name):
        self.name = name


class _FakePhases:
    def __init__(self, names):
        self._by_id = dict(enumerate(_FakePhase(n) for n in names))

    def __getitem__(self, pid):
        return self._by_id[pid]


class _FakeXmap:
    def __init__(self, phase_id, names):
        self.phase_id = np.asarray(phase_id)
        self.phases = _FakePhases(names)
        self.prop = {}


class _FakeResult:
    def __init__(self, xmap, shape, metadata, mask=None):
        self.xmap = xmap
        self.original_shape = shape
        self.metadata = metadata
        self.selection_mask = np.ones(shape, dtype=bool) if mask is None else mask


def _dict_file(tmp_path, n=6, h=8, w=8):
    """A kikuchipy-written dictionary, i.e. what generation actually produces."""
    import kikuchipy as kp
    from orix.crystal_map import CrystalMap, PhaseList, Phase
    from orix.quaternion import Rotation

    rng = np.random.default_rng(0)
    data = rng.random((n, h, w), dtype=np.float32)
    sig = kp.signals.EBSD(data)
    sig.xmap = CrystalMap(
        rotations=Rotation.random(n), phase_list=PhaseList(Phase(name="Si"))
    )
    out = tmp_path / "si_dict.h5"
    sig.save(str(out), overwrite=True)
    return out, data


def test_multi_phase_reads_the_winning_phase_dictionary(tmp_path):
    path, data = _dict_file(tmp_path)

    # 2x2 grid: Al wins two pixels, Si the other two.
    phase_id = np.array([0, 1, 1, 0])
    sim_si = np.full((4, 1), -1, dtype=np.int64)
    sim_si[1, 0] = 3          # pixel (0,1) -> dictionary row 3
    sim_si[2, 0] = 5          # pixel (1,0) -> dictionary row 5

    result = _FakeResult(
        _FakeXmap(phase_id, ["Al", "Si"]),
        (2, 2),
        {
            "multi_phase": True,
            "per_phase_match_sources": {
                "Si": {"dict_path": str(path), "simulation_indices": sim_si},
            },
        },
    )

    got = get_best_match_pattern(result, row=0, col=1)
    assert got is not None, "still 'Dictionary not in memory'"
    np.testing.assert_allclose(got, data[3], rtol=1e-6)

    got2 = get_best_match_pattern(result, row=1, col=0)
    np.testing.assert_allclose(got2, data[5], rtol=1e-6)


def test_pixel_of_a_phase_without_a_source_returns_none(tmp_path):
    """Al has no recorded source here — must degrade, not read Si's file."""
    path, _ = _dict_file(tmp_path)
    sim_si = np.full((4, 1), 2, dtype=np.int64)
    result = _FakeResult(
        _FakeXmap(np.array([0, 1, 1, 0]), ["Al", "Si"]),
        (2, 2),
        {"per_phase_match_sources": {
            "Si": {"dict_path": str(path), "simulation_indices": sim_si}}},
    )
    assert get_best_match_pattern(result, row=0, col=0) is None   # Al pixel
    assert get_best_match_pattern(result, row=0, col=1) is not None  # Si pixel


def test_unassigned_pixel_returns_none(tmp_path):
    path, _ = _dict_file(tmp_path)
    sim = np.full((4, 1), 1, dtype=np.int64)
    result = _FakeResult(
        _FakeXmap(np.array([-1, 1, 1, 1]), ["Al", "Si"]),
        (2, 2),
        {"per_phase_match_sources": {
            "Si": {"dict_path": str(path), "simulation_indices": sim}}},
    )
    assert get_best_match_pattern(result, row=0, col=0) is None


def test_out_of_range_index_is_refused_not_clamped(tmp_path):
    path, _ = _dict_file(tmp_path, n=6)
    sim = np.full((4, 1), 99, dtype=np.int64)
    result = _FakeResult(
        _FakeXmap(np.array([1, 1, 1, 1]), ["Al", "Si"]),
        (2, 2),
        {"per_phase_match_sources": {
            "Si": {"dict_path": str(path), "simulation_indices": sim}}},
    )
    assert get_best_match_pattern(result, row=0, col=0) is None


def test_single_phase_path_is_untouched():
    """No per_phase_match_sources -> old behaviour (returns None here)."""
    result = _FakeResult(
        _FakeXmap(np.array([0, 0, 0, 0]), ["Al"]), (2, 2), {}
    )
    assert get_best_match_pattern(result, row=0, col=0) is None


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
