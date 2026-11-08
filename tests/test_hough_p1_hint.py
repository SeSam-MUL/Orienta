"""The Hough memory error names the hidden symmetry of a P 1 phase.

A P 1 CIF makes every reflector family a single pole and the band-triplet
library tens of GiB. When the atoms of such a phase have more symmetry, the
error says which group and which action fixes it (the Crystal Database's "Save
symmetrised copy"); for a phase that genuinely has none it keeps the generic
sentence. Both go through the real refusal in ``create_indexer``, not through a
mocked message.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymatgen.core import Lattice, Structure  # noqa: E402
from pymatgen.io.cif import CifWriter  # noqa: E402

from ebsd_utils import create_indexer, hough_phase_from_cif, prepare_reflectors  # noqa: E402
from tests.synthetic_cif import write_synthetic_p1_cif  # noqa: E402


def _phase_list(path):
    from orix.crystal_map import PhaseList

    ph = hough_phase_from_cif(str(path))
    ph.name = Path(path).stem
    return PhaseList(ph)


def _detector():
    import kikuchipy as kp

    return kp.detectors.EBSDDetector(shape=(60, 60), pc=(0.5, 0.5, 0.5), sample_tilt=70.0)


def _refuse(monkeypatch, path):
    monkeypatch.setenv("ORIENTA_NO_PHASE_LIMIT_STORE", "1")
    monkeypatch.setenv("ORIENTA_HOUGH_LIBRARY_BUDGET_MB", "64")   # far below any P 1 library
    pl = _phase_list(path)
    with pytest.raises(MemoryError) as ei:
        create_indexer(_detector(), pl, prepare_reflectors(pl), nBands=12)
    return str(ei.value)


def _noisy_p1_zirconium(tmp_path):
    zr = Structure.from_spacegroup("P6_3/mmc", Lattice.hexagonal(3.23, 5.15), ["Zr"], [[1 / 3, 2 / 3, 0.25]])
    rng = np.random.default_rng(1)
    cart = zr.cart_coords + rng.uniform(-0.003, 0.003, zr.cart_coords.shape)
    noisy = Structure(zr.lattice, zr.species, zr.lattice.get_fractional_coords(cart))
    path = tmp_path / "ZrNoisyP1.cif"
    path.write_text(str(CifWriter(noisy)), encoding="utf-8")
    return path


def test_p1_zirconium_error_names_the_group_and_the_action(tmp_path, monkeypatch):
    path = _noisy_p1_zirconium(tmp_path)
    msg = _refuse(monkeypatch, path)
    assert "ZrNoisyP1 is stored as space group P 1" in msg
    assert "P6_3/mmc (194)" in msg
    assert "Save symmetrised copy" in msg


def test_the_batch_failure_message_names_the_group_too(tmp_path, monkeypatch):
    # The other place that explains a Hough memory failure (phase check, batch).
    from backend.api.routes.indexing import _describe_hough_failure
    from tests.test_hough_failure_message import _CLError, _pin_the_machine

    path = _noisy_p1_zirconium(tmp_path)
    _pin_the_machine(monkeypatch, path, 8 << 30)          # 8 GiB free, the ask is ~29
    msg = _describe_hough_failure(_CLError("Context failed: OUT_OF_HOST_MEMORY"), path)
    assert "P 1" in msg
    assert "P6_3/mmc (194)" in msg and "Save symmetrised copy" in msg


def test_a_phase_without_hidden_symmetry_keeps_the_generic_sentence(tmp_path, monkeypatch):
    msg = _refuse(monkeypatch, write_synthetic_p1_cif(tmp_path))
    assert "space group P 1" in msg and "usual cause" in msg
    assert "Save symmetrised copy" not in msg


def _noisy_p1_aluminium_ambiguous(tmp_path):
    """fcc Al with +-0.008 A of noise (seed 1): Cm (8) at 0.01 and 0.02 A, Fm-3m (225)
    from 0.05 A."""
    al = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.05), ["Al"], [[0, 0, 0]])
    cart = al.cart_coords + np.random.default_rng(1).uniform(-0.008, 0.008, al.cart_coords.shape)
    path = tmp_path / "AlNoisyP1.cif"
    path.write_text(str(CifWriter(Structure(al.lattice, al.species, al.lattice.get_fractional_coords(cart)))), encoding="utf-8")
    return path


def test_the_memory_error_states_the_ladder_where_the_group_depends_on_the_tolerance(tmp_path, monkeypatch):
    msg = _refuse(monkeypatch, _noisy_p1_aluminium_ambiguous(tmp_path))
    assert "AlNoisyP1 is stored as space group P 1" in msg
    assert "0.01 A: Cm (8); 0.02 A: Cm (8); 0.05 A: Fm-3m (225); 0.1 A: Fm-3m (225)" in msg
    assert "choose a tolerance" in msg and "Save symmetrised copy" in msg
    assert "has the symmetry" not in msg                       # no single guess


def test_the_batch_failure_message_states_the_ladder_too(tmp_path, monkeypatch):
    from backend.api.routes.indexing import _describe_hough_failure
    from tests.test_hough_failure_message import _CLError, _pin_the_machine

    path = _noisy_p1_aluminium_ambiguous(tmp_path)
    _pin_the_machine(monkeypatch, path, 2 << 30)               # 2 GiB free, the ask is ~5
    msg = _describe_hough_failure(_CLError("Context failed: OUT_OF_HOST_MEMORY"), path)
    assert "0.01 A: Cm (8)" in msg and "0.05 A: Fm-3m (225)" in msg and "choose a tolerance" in msg
