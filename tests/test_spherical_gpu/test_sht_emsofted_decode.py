"""Task 4 — decode the EMsoftED SimulationData block in ``read_sht_master``.

The ``.sht`` master file carries an 88-byte EMsoftED ``SimulationData`` block per
crystal (written by :func:`backend.forward_sim.io.sht_writer._build_emsoft_ed`).
The reader historically SKIPPED that block; this test pins that ``read_sht_master``
now decodes the simulation provenance (dmin, npx, numsx, totnum_el, Bethe params)
into the additive optional ``sim_*`` fields on :class:`SHTMasterFile`.

Round-trip: write a ``.sht`` with known ``dmin``/``npx``/``numsx``/``totnum_el``
via :func:`write_sht`, read it back, assert the decoded ``sim_*`` fields match.

Runs on CPU (``FORWARD_SIM_DEVICE=cpu``) so it needs no CUDA.
"""
from __future__ import annotations

import os

os.environ.setdefault("FORWARD_SIM_DEVICE", "cpu")

import numpy as np  # noqa: E402

from backend.forward_sim.crystal.xtal_io import Atom, CrystalStructure  # noqa: E402
from backend.forward_sim.io.sht_writer import write_sht  # noqa: E402
from backend.spherical_gpu.pipeline.sht_io import read_sht_master  # noqa: E402


def _tiny_cubic_master(npx: int) -> np.ndarray:
    """Minimal Lambert NH master the writer accepts: a (2*npx+1, 2*npx+1) square.

    The writer requires an ODD square side (= 2*npx+1) — its only structural
    constraint on the master grid; the harmonic content is irrelevant to the
    ``sim_*`` round-trip this test exercises.
    """
    side = 2 * npx + 1
    return np.random.RandomState(0).rand(side, side).astype(np.float64)


def _cubic_structure(sg: int = 225) -> CrystalStructure:
    """Smallest valid structure the writer needs (single Al atom, Fm-3m).

    ``CrystalStructure`` (from ``backend.forward_sim.crystal.xtal_io``) has no
    ``simple_cubic`` constructor; it is a plain frozen dataclass, so we build it
    directly — the same fixture the writer's own tests use.
    """
    return CrystalStructure(
        lattice=(0.405, 0.405, 0.405, 90.0, 90.0, 90.0),
        atoms=[Atom(Z=13, xyz=(0.0, 0.0, 0.0), occ=1.0, B=0.005)],
        space_group=sg,
        crystal_system=1,
    )


def test_emsofted_fields_roundtrip(tmp_path):
    # Build the smallest valid structure the writer needs (single Al atom, Fm-3m).
    struct = _cubic_structure(sg=225)
    npx = 120
    out = tmp_path / "probe.sht"
    write_sht(
        str(out), _tiny_cubic_master(npx), struct,
        bandwidth=16, energy_kV=20.0, npx=npx, dmin=0.04,
        numsx=333, totnum_el=2_000_000, sg_eff=225,
        device="cpu",
    )
    parsed = read_sht_master(str(out), device="cpu")
    assert abs(parsed.sim_dmin - 0.04) < 1e-6
    assert parsed.sim_npx == 120
    assert parsed.sim_numsx == 333
    assert parsed.sim_totnum_el == 2_000_000
    assert len(parsed.sim_bethe) == 4
