"""TDD tests for gpu_sim_runner.py — MC source strategy, ekev guard,
non-cubic preflight, and GPU job lock serialisation.

AUTOHW: the Orienta Engine is now FULLY EMsoft-FREE.  The MC source strategy is:
  (a) existing MC .h5 cache hit,
  (b) Orienta Engine GPU MC (cupy CUDA kernel) when CUDA + cupy compile,
  (c) Orienta Engine numba CPU MC, else
  (d) the Orienta Engine PyTorch CPU loop.
Strategies (b)/(c)/(d) all route through ``run_gpu_mc(engine="auto")`` (the
fan-out is internal to run_gpu_mc); EMsoft EMMCOpenCL / WSL is NO LONGER a
fallback in this pipeline.  ``_run_emsoft_mc`` was removed.

These tests verify:
  1. MC source strategy (EMsoft-free): (a) existing .h5, (b) GPU MC when
     CUDA+cupy, (c/d) Orienta Engine CPU MC otherwise — never EMsoft.
  2. ekev <= 10.0 raises ValueError (fail-loud)
  3. Non-cubic structures are supported (build both hemispheres, no cubic raise)
  4. GPU build lock serialises concurrent acquisitions (no overlap)
  5. Device resolution is hardware-adaptive (CPU-only host runs with no env var)
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


@pytest.fixture(autouse=True)
def _outputs_to_tmp(tmp_path, monkeypatch):
    """Keep every test in this file out of the user's phase library.

    These tests drive the real ``run_gpu_simulation`` with the heavy steps
    mocked, but not ``write_provenance_sidecar`` — and the runner takes its
    output root from a module constant pointing at
    ``Database/EBSD_SHT_Database``. Nine of them therefore wrote provenance
    records into the real library (``Ni (Ni) {10kV}``, ``{20kV}``,
    ``Mg (Mg) {20kV}``), each claiming a source .xtal of ``/fake/…`` as FOUND
    for a simulation that never ran. In a worktree ``Database/`` is a junction,
    so that is the directory the app reads.

    Autouse and file-wide because the writes come from several call sites,
    including a shared ``_run_noncubic`` helper that takes no ``tmp_path``.
    ``_cif_library_dir()`` is derived from ``_SHT_DB.parent``, so it is pinned
    separately — reading the user's library is fine, writing to it is not.
    """
    from backend.api.services import gpu_sim_runner
    from tests.data_deps import CIF_LIBRARY

    monkeypatch.setattr(gpu_sim_runner, "_SHT_DB", tmp_path / "EBSD_SHT_Database")
    monkeypatch.setattr(gpu_sim_runner, "_H5_CACHE", tmp_path / "EBSD_H5_Cache")
    # Only pin the READ when there is a library to read. Unconditionally
    # pinning it would make these tests depend on Database/ on a machine that
    # has none, where until now they merely behaved as they always did — and
    # this fixture is autouse, so a skip here would also skip the tests in this
    # file that never touch a CIF (the lock and ekev guards).
    if CIF_LIBRARY.is_dir():
        monkeypatch.setattr(gpu_sim_runner, "_cif_library_dir", lambda: CIF_LIBRARY)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _real_xtal(tmp_path, name="Ni.xtal"):
    """A .xtal that actually exists, so nothing has to lie about it.

    The runner guards its input with `xtal.exists()` and passes the same path
    to write_provenance_sidecar, which records `found: xtal_path.exists()`.
    Faking existence globally satisfied the guard AND wrote `found: true` for a
    path that never existed — four such records sat in the user's phase library
    from 2026-08-07. The content is irrelevant here: read_crystal_structure is
    mocked in every one of these tests.
    """
    f = Path(tmp_path) / name
    f.write_bytes(b"")
    return f


def _make_cubic_structure():
    s = MagicMock()
    s.space_group = 225
    s.lattice = (0.352, 0.352, 0.352, 90.0, 90.0, 90.0)
    s.atoms = [MagicMock()]
    return s


def _make_hexagonal_structure():
    s = MagicMock()
    s.space_group = 194
    # Hexagonal: a=b≠c, alpha=beta=90°, gamma=120°
    s.lattice = (0.321, 0.321, 0.521, 90.0, 90.0, 120.0)
    s.atoms = [MagicMock()]
    return s


def _make_mock_mc_data(ekev: float = 20.0):
    """Return a realistic MCData mock with 11 energy bins."""
    import torch
    mock = MagicMock()
    mock.n_energy = 11
    mock.EkeVs = torch.linspace(10.0, ekev, 11)
    mock.accum_e = torch.zeros(501, 501, 11)
    mock.depth_step = 1.0
    mock.depth_max = 100.0
    return mock


def _make_dummy_master():
    import torch
    import numpy as np
    m = MagicMock()
    m.isfinite.return_value.all.return_value.item.return_value = True
    m.shape = (101, 101)
    m.detach.return_value.cpu.return_value.numpy.return_value = np.zeros(
        (101, 101), dtype="float32"
    )
    return m


# ---------------------------------------------------------------------------
# 1. MC source strategy
# ---------------------------------------------------------------------------

class TestMcSourceStrategy:
    """Verify the EMsoft-free MC source strategy in run_gpu_simulation (AUTOHW)."""

    def test_strategy_a_existing_mc_h5_used_without_running_mc(self, tmp_path):
        """If a cached MC .h5 already exists, load_mc is called and run_gpu_mc is NOT."""
        from backend.api.services import gpu_sim_runner

        structure = _make_cubic_structure()
        mc_data = _make_mock_mc_data()
        load_mc_mock = MagicMock(return_value=mc_data)
        run_gpu_mc_mock = MagicMock(return_value=mc_data)
        import torch

        # No Path.exists patch: both files are MADE, under tmp_path.
        #
        # This test used to fake existence globally, and that is how four false
        # provenance records ended up in the user's phase library on 2026-08-07
        # — write_provenance_sidecar asks `xtal_path.exists()`, was told "yes"
        # about a path that never existed, and wrote `found: true` to disk.
        # A patch that makes the filesystem lie to the code under test also
        # makes it lie to whatever that code PERSISTS.
        xtal = tmp_path / "Ni.xtal"
        xtal.write_bytes(b"")            # content is irrelevant: the reader is mocked
        existing_mc = gpu_sim_runner.mc_h5_path("Ni", 20, 70.0, 501)
        existing_mc.parent.mkdir(parents=True, exist_ok=True)
        existing_mc.write_bytes(b"")     # the cache hit strategy (a) is about to take

        log_lines: list = []

        with patch("backend.forward_sim.runtime.resolve_device_adaptive", return_value=torch.device("cpu")), \
             patch("backend.forward_sim.crystal.xtal_io.read_crystal_structure", return_value=structure), \
             patch("backend.forward_sim.mc.emsoft_mc_input.load_mc", load_mc_mock), \
             patch("backend.forward_sim.mc.gpu_monte_carlo.run_gpu_mc", run_gpu_mc_mock), \
             patch("backend.forward_sim.dynamical.master_builder.build_master", return_value=_make_dummy_master()), \
             patch("backend.forward_sim.io.mc_h5.write_mc_h5"), \
             patch("backend.forward_sim.io.master_h5.write_master_h5"), \
             patch("backend.forward_sim.io.sht_writer.write_sht"), \
             patch("backend.forward_sim.io.sht_writer._formula_from_atoms", return_value="Ni"):

            gpu_sim_runner.run_gpu_simulation(
                str(xtal),
                {"output_type": "sht_only", "ekev": 20.0},
                progress_cb=lambda p, m: None,
                log_cb=lambda l: log_lines.append(l),
            )

        # Strategy (a): load_mc called, run_gpu_mc NOT called
        load_mc_mock.assert_called_once()
        run_gpu_mc_mock.assert_not_called()

        # And the provenance record the run left behind tells the truth about
        # its source crystal — the thing the old blanket patch made impossible.
        sidecars = list((tmp_path / "EBSD_SHT_Database").rglob("*.provenance.json"))
        assert len(sidecars) == 1, sidecars
        doc = json.loads(sidecars[0].read_text(encoding="utf-8"))
        assert doc["source_xtal"]["path"] == str(xtal)
        assert doc["source_xtal"]["found"] is True   # because it really is there

        # Log must mention strategy (a) cache hit
        combined = " ".join(log_lines)
        assert "existing_cache" in combined or "existing" in combined.lower()

    def test_strategy_b_gpu_mc_when_cuda_cupy(self, tmp_path):
        """AUTOHW: on a CUDA host with cupy, Orienta Engine GPU MC (run_gpu_mc) runs and NO
        EMsoft helper exists to call.  mc_source must be gpu_native."""
        from backend.api.services import gpu_sim_runner

        # The EMsoft helper was removed entirely — assert it is gone.
        assert not hasattr(gpu_sim_runner, "_run_emsoft_mc"), (
            "_run_emsoft_mc must be removed — the Ours pipeline is EMsoft-free"
        )

        structure = _make_cubic_structure()
        mc_data = _make_mock_mc_data()
        import torch

        run_gpu_mc_mock = MagicMock(return_value=mc_data)
        load_mc_mock = MagicMock(return_value=mc_data)
        # If anything tries to probe WSL, fail the test loudly.
        wsl_probe = MagicMock(side_effect=AssertionError("Ours must not probe WSL/EMsoft"))
        log_lines: list = []

        with patch("backend.forward_sim.runtime.resolve_device_adaptive", return_value=torch.device("cuda")), \
             patch("backend.forward_sim.crystal.xtal_io.read_crystal_structure", return_value=structure), \
             patch(
                 "backend.forward_sim.mc.composition.mc_composition_from_structure",
                 return_value=MagicMock(mean_Z=28.0, mean_A=58.693, rho=8.908),
             ), \
             patch("backend.forward_sim.mc.emsoft_mc_input.load_mc", load_mc_mock), \
             patch("backend.forward_sim.mc.gpu_monte_carlo.run_gpu_mc", run_gpu_mc_mock), \
             patch("backend.forward_sim.mc.gpu_monte_carlo.cupy_mc_available", return_value=True), \
             patch("backend.forward_sim.dynamical.master_builder.build_master", return_value=_make_dummy_master()), \
             patch("backend.forward_sim.io.mc_h5.write_mc_h5"), \
             patch("backend.forward_sim.io.master_h5.write_master_h5"), \
             patch("backend.forward_sim.io.sht_writer.write_sht"), \
             patch("backend.forward_sim.io.sht_writer._formula_from_atoms", return_value="Ni"), \
             patch("backend.api.services.gpu_sim_runner._wsl_available", wsl_probe):

            gpu_sim_runner.run_gpu_simulation(
                str(_real_xtal(tmp_path)),
                {"output_type": "sht_only", "ekev": 20.0},
                progress_cb=lambda p, m: None,
                log_cb=lambda l: log_lines.append(l),
            )

        # Our GPU MC ran; WSL/EMsoft never probed.
        run_gpu_mc_mock.assert_called_once()
        wsl_probe.assert_not_called()

        combined = " ".join(log_lines)
        assert "mc_source=gpu_native" in combined
        assert "Orienta Engine GPU MC" in combined
        # The EMsoft-free path must NOT mention EMsoft running.
        assert "EMMCOpenCL" not in combined
        assert "emsoft_wsl" not in combined

    def test_strategy_cd_cpu_mc_when_no_cupy_kernel(self, tmp_path):
        """AUTOHW: CUDA present but cupy MC kernel does NOT compile -> Orienta Engine CPU MC
        runs (run_gpu_mc engine=auto -> numba/pytorch); NO EMsoft, loud warning."""
        from backend.api.services import gpu_sim_runner

        structure = _make_cubic_structure()
        mc_data = _make_mock_mc_data()
        import torch

        run_gpu_mc_mock = MagicMock(return_value=mc_data)
        wsl_probe = MagicMock(side_effect=AssertionError("Ours must not probe WSL/EMsoft"))
        log_lines: list = []

        with patch("backend.forward_sim.runtime.resolve_device_adaptive", return_value=torch.device("cuda")), \
             patch("backend.forward_sim.crystal.xtal_io.read_crystal_structure", return_value=structure), \
             patch(
                 "backend.forward_sim.mc.composition.mc_composition_from_structure",
                 return_value=MagicMock(mean_Z=28.0, mean_A=58.693, rho=8.908),
             ), \
             patch("backend.forward_sim.mc.gpu_monte_carlo.run_gpu_mc", run_gpu_mc_mock), \
             patch("backend.forward_sim.mc.gpu_monte_carlo.cupy_mc_available", return_value=False), \
             patch("backend.forward_sim.dynamical.master_builder.build_master", return_value=_make_dummy_master()), \
             patch("backend.forward_sim.io.mc_h5.write_mc_h5"), \
             patch("backend.forward_sim.io.master_h5.write_master_h5"), \
             patch("backend.forward_sim.io.sht_writer.write_sht"), \
             patch("backend.forward_sim.io.sht_writer._formula_from_atoms", return_value="Ni"), \
             patch("backend.api.services.gpu_sim_runner._wsl_available", wsl_probe):

            gpu_sim_runner.run_gpu_simulation(
                str(_real_xtal(tmp_path)),
                {"output_type": "sht_only", "ekev": 20.0},
                progress_cb=lambda p, m: None,
                log_cb=lambda l: log_lines.append(l),
            )

        # Orienta Engine CPU MC ran; WSL/EMsoft never probed.
        run_gpu_mc_mock.assert_called_once()
        wsl_probe.assert_not_called()
        combined = " ".join(log_lines)
        assert "mc_source=cpu_fallback" in combined
        # A machine that HAS a GPU it cannot use: this one still shouts, and
        # still names the cost. Asserted on the marker word rather than the
        # sentence, which was reworded once already.
        assert "WARNING" in combined
        assert "slower" in combined.lower()
        assert "cupy" in combined
        # "EMsoft-free" used to be asserted here as prose. The guarantee it
        # stood for is checked far more strongly two lines above
        # (wsl_probe.assert_not_called()) and by the absence below, so the
        # wording is no longer load-bearing.
        assert "EMMCOpenCL" not in combined

    def test_strategy_cd_cpu_mc_on_cpu_host_is_reported_calmly(self, tmp_path):
        """AUTOHW: on a CPU-only host (no CUDA), the Orienta Engine CPU MC runs,
        it is reported as INFORMATION rather than a warning, and no EMsoft is
        involved.

        Renamed from ..._with_loud_warning: the loud warning was removed on
        purpose (M5 tester, 2026-09-25 — on a machine with no NVIDIA card this
        is the normal path, and the orange line read as a failure). A test name
        that still promises the warning is a lie the next reader has to
        disprove."""
        from backend.api.services import gpu_sim_runner

        structure = _make_cubic_structure()
        mc_data = _make_mock_mc_data()
        import torch

        run_gpu_mc_mock = MagicMock(return_value=mc_data)
        wsl_probe = MagicMock(side_effect=AssertionError("Ours must not probe WSL/EMsoft"))
        log_lines: list = []

        with patch("backend.forward_sim.runtime.resolve_device_adaptive", return_value=torch.device("cpu")), \
             patch("backend.forward_sim.crystal.xtal_io.read_crystal_structure", return_value=structure), \
             patch(
                 "backend.forward_sim.mc.composition.mc_composition_from_structure",
                 return_value=MagicMock(mean_Z=28.0, mean_A=58.693, rho=8.908),
             ), \
             patch("backend.forward_sim.mc.gpu_monte_carlo.run_gpu_mc", run_gpu_mc_mock), \
             patch("backend.forward_sim.dynamical.master_builder.build_master", return_value=_make_dummy_master()), \
             patch("backend.forward_sim.io.mc_h5.write_mc_h5"), \
             patch("backend.forward_sim.io.master_h5.write_master_h5"), \
             patch("backend.forward_sim.io.sht_writer.write_sht"), \
             patch("backend.forward_sim.io.sht_writer._formula_from_atoms", return_value="Ni"), \
             patch("backend.api.services.gpu_sim_runner._wsl_available", wsl_probe):

            gpu_sim_runner.run_gpu_simulation(
                str(_real_xtal(tmp_path)),
                {"output_type": "sht_only", "ekev": 20.0},
                progress_cb=lambda p, m: None,
                log_cb=lambda l: log_lines.append(l),
            )

        # Orienta Engine CPU MC must have been called; WSL never probed.
        run_gpu_mc_mock.assert_called_once()
        wsl_probe.assert_not_called()

        combined = " ".join(log_lines)
        assert "mc_source=cpu_fallback" in combined
        # What the user reads, as opposed to the "[gpu_sim] ..." trace lines
        # that share the same callback.
        user_facing = " ".join(l for l in log_lines if not l.startswith("[gpu_sim]"))

        # On a machine with NO CUDA device this is the normal path, so the
        # user-facing line is information and must NOT carry the word WARNING:
        # SimulationPage colours any line containing it orange, and an M5 tester
        # read the old warning as "PyTorch is broken" (2026-09-25). The prose
        # assertions that used to live here ("no CUDA device", "EMsoft-free")
        # were what made this test fail when that was fixed, so what is pinned
        # now is the MEANING, with a pattern rather than a sentence.
        assert not re.search(r"WARNING", user_facing), user_facing
        assert re.search(r"no CUDA (GPU|device)", user_facing, re.I), user_facing
        # it still says what the speed costs, in either direction
        assert re.search(r"faster|slower", user_facing, re.I), user_facing
        # and which engine ran, by the one pinned name
        assert "Orienta Engine" in user_facing, user_facing

        assert "EMMCOpenCL" not in combined

    def test_no_emsoft_references_in_source(self):
        """Static guard: run_gpu_simulation must not invoke EMsoft for MC.

        The Ours pipeline source must not reference the removed _run_emsoft_mc nor
        the EMMCOpenCL-run path; only the _wsl_available capability probe (used by
        the router + capability reporting) may remain.
        """
        src = (
            _ROOT / "backend" / "api" / "services" / "gpu_sim_runner.py"
        ).read_text(encoding="utf-8")
        assert "_run_emsoft_mc" not in src, "_run_emsoft_mc must be removed"
        assert "def _run_emsoft_mc" not in src
        # The MC strategy must not load/run an EMsoft-produced .h5 as a fallback.
        assert "emsoft_wsl" not in src


# ---------------------------------------------------------------------------
# 2. ekev guard
# ---------------------------------------------------------------------------

class TestEkevGuard:
    """ekev <= Ehistmin (10.0) must raise ValueError before any MC work."""

    @pytest.mark.parametrize("ekev", [10.0, 9.9, 5.0, 1.0, 0.01])
    def test_ekev_at_or_below_ehistmin_raises(self, ekev, tmp_path):
        """run_gpu_simulation must raise ValueError for ekev <= 10.0 kV."""
        from backend.api.services import gpu_sim_runner
        import torch

        structure = _make_cubic_structure()
        run_gpu_mc_mock = MagicMock()

        with patch("backend.forward_sim.runtime.resolve_device_adaptive", return_value=torch.device("cpu")), \
             patch("backend.forward_sim.crystal.xtal_io.read_crystal_structure", return_value=structure), \
             patch("backend.forward_sim.mc.gpu_monte_carlo.run_gpu_mc", run_gpu_mc_mock):


            with pytest.raises(ValueError, match="Ehistmin"):
                gpu_sim_runner.run_gpu_simulation(
                    str(_real_xtal(tmp_path)),
                    {"output_type": "sht_only", "ekev": ekev},
                    progress_cb=lambda p, m: None,
                    log_cb=lambda l: None,
                )

        # MC must NOT have been called
        run_gpu_mc_mock.assert_not_called()

    def test_ekev_above_ehistmin_does_not_raise(self, tmp_path):
        """ekev = 10.1 kV is just above the threshold and must not raise."""
        from backend.api.services import gpu_sim_runner
        import torch

        structure = _make_cubic_structure()
        mc_data = _make_mock_mc_data(ekev=10.1)

        # Just need it to get past the guard — we don't need a full successful run
        with patch("backend.forward_sim.runtime.resolve_device_adaptive", return_value=torch.device("cpu")), \
             patch("backend.forward_sim.crystal.xtal_io.read_crystal_structure", return_value=structure), \
             patch("backend.api.services.gpu_sim_runner._wsl_available", return_value=False), \
             patch(
                 "backend.forward_sim.mc.composition.mc_composition_from_structure",
                 return_value=MagicMock(mean_Z=28.0, mean_A=58.693, rho=8.908),
             ), \
             patch("backend.forward_sim.mc.gpu_monte_carlo.run_gpu_mc", return_value=mc_data), \
             patch("backend.forward_sim.dynamical.master_builder.build_master", return_value=_make_dummy_master()), \
             patch("backend.forward_sim.io.mc_h5.write_mc_h5"), \
             patch("backend.forward_sim.io.sht_writer.write_sht"), \
             patch("backend.forward_sim.io.sht_writer._formula_from_atoms", return_value="Ni"):

            # Should NOT raise ValueError about Ehistmin
            try:
                gpu_sim_runner.run_gpu_simulation(
                    str(_real_xtal(tmp_path)),
                    {"output_type": "sht_only", "ekev": 10.1},
                    progress_cb=lambda p, m: None,
                    log_cb=lambda l: None,
                )
            except ValueError as exc:
                assert "Ehistmin" not in str(exc), (
                    f"Should not raise Ehistmin error for ekev=10.1: {exc}"
                )


# ---------------------------------------------------------------------------
# 3. Non-cubic preflight
# ---------------------------------------------------------------------------

class TestNonCubicRouting:
    """SP6: non-cubic structures are now SUPPORTED (general dsm map + SH build).

    They must NOT raise a 'cubic' ValueError, must build BOTH hemispheres
    (NH != SH in general), and must hand the southern master to write_sht so the
    .sht encodes the full sphere (SP6-iter2 Gap 1 + Gap 3).
    """

    def _run_noncubic(self, write_sht_mock, build_master_mock, tmp_path):
        """Drive run_gpu_simulation for a hexagonal (non-cubic) phase with all the
        heavy steps mocked; returns nothing (assertions are on the mocks)."""
        from backend.api.services import gpu_sim_runner
        import torch

        structure = _make_hexagonal_structure()
        mc_data = _make_mock_mc_data()

        with patch("backend.forward_sim.runtime.resolve_device_adaptive", return_value=torch.device("cpu")), \
             patch("backend.forward_sim.crystal.xtal_io.read_crystal_structure", return_value=structure), \
             patch("backend.api.services.gpu_sim_runner._wsl_available", return_value=False), \
             patch(
                 "backend.forward_sim.mc.composition.mc_composition_from_structure",
                 return_value=MagicMock(mean_Z=20.0, mean_A=40.0, rho=3.0),
             ), \
             patch("backend.forward_sim.mc.gpu_monte_carlo.run_gpu_mc", return_value=mc_data), \
             patch("backend.forward_sim.dynamical.master_builder.build_master", build_master_mock), \
             patch("backend.forward_sim.io.mc_h5.write_mc_h5"), \
             patch("backend.forward_sim.io.sht_writer.write_sht", write_sht_mock), \
             patch("backend.forward_sim.io.sht_writer._formula_from_atoms", return_value="Mg"):

            gpu_sim_runner.run_gpu_simulation(
                str(_real_xtal(tmp_path, "Mg.xtal")),
                {"output_type": "sht_only", "ekev": 20.0},
                progress_cb=lambda p, m: None,
                log_cb=lambda l: None,
            )

    def test_non_cubic_does_not_raise_cubic_error(self, tmp_path):
        """SP6: a hexagonal structure runs through the general path (no cubic raise)."""
        build_master_mock = MagicMock(return_value=_make_dummy_master())
        write_sht_mock = MagicMock()
        try:
            self._run_noncubic(write_sht_mock, build_master_mock, tmp_path)
        except ValueError as exc:
            assert "cubic" not in str(exc).lower(), (
                f"non-cubic structure must not raise a cubic error post-SP6: {exc}"
            )
        # The master builder must have been invoked (general path reached).
        assert build_master_mock.call_count >= 1

    def test_non_cubic_builds_both_hemispheres(self, tmp_path):
        """SP6: a non-cubic build evaluates BOTH the northern and southern grid."""
        build_master_mock = MagicMock(return_value=_make_dummy_master())
        self._run_noncubic(MagicMock(), build_master_mock, tmp_path)
        hemis = [c.kwargs.get("hemisphere") for c in build_master_mock.call_args_list]
        assert "north" in hemis, hemis
        assert "south" in hemis, hemis

    def test_non_cubic_passes_sh_master_to_write_sht(self, tmp_path):
        """SP6-iter2 Gap 3: the southern master is threaded into write_sht so the
        .sht harmonics encode the full sphere (not an NH-only mirror)."""
        sentinel_nh = _make_dummy_master()
        sentinel_sh = _make_dummy_master()
        # First call (north) -> NH sentinel; second (south) -> SH sentinel.
        build_master_mock = MagicMock(side_effect=[sentinel_nh, sentinel_sh])
        write_sht_mock = MagicMock()
        self._run_noncubic(write_sht_mock, build_master_mock, tmp_path)
        assert write_sht_mock.call_count == 1
        kwargs = write_sht_mock.call_args.kwargs
        assert "our_master_SH" in kwargs, "write_sht must receive our_master_SH"
        assert kwargs["our_master_SH"] is sentinel_sh, (
            "the southern master built must be the one passed to write_sht"
        )

    def test_cubic_structure_does_not_raise(self, tmp_path):
        """Ni (cubic FCC) must not raise a non-cubic error."""
        from backend.api.services import gpu_sim_runner
        import torch

        structure = _make_cubic_structure()
        mc_data = _make_mock_mc_data()

        with patch("backend.forward_sim.runtime.resolve_device_adaptive", return_value=torch.device("cpu")), \
             patch("backend.forward_sim.crystal.xtal_io.read_crystal_structure", return_value=structure), \
             patch("backend.api.services.gpu_sim_runner._wsl_available", return_value=False), \
             patch(
                 "backend.forward_sim.mc.composition.mc_composition_from_structure",
                 return_value=MagicMock(mean_Z=28.0, mean_A=58.693, rho=8.908),
             ), \
             patch("backend.forward_sim.mc.gpu_monte_carlo.run_gpu_mc", return_value=mc_data), \
             patch("backend.forward_sim.dynamical.master_builder.build_master", return_value=_make_dummy_master()), \
             patch("backend.forward_sim.io.mc_h5.write_mc_h5"), \
             patch("backend.forward_sim.io.sht_writer.write_sht"), \
             patch("backend.forward_sim.io.sht_writer._formula_from_atoms", return_value="Ni"):

            try:
                gpu_sim_runner.run_gpu_simulation(
                    str(_real_xtal(tmp_path)),
                    {"output_type": "sht_only", "ekev": 20.0},
                    progress_cb=lambda p, m: None,
                    log_cb=lambda l: None,
                )
            except ValueError as exc:
                # Any ValueError here is a real problem if it's about cubic
                assert "cubic" not in str(exc).lower(), (
                    f"Cubic structure should not raise cubic error: {exc}"
                )


# ---------------------------------------------------------------------------
# 3b. L3 — build_sh gate must stay `not cubic`, NOT `not centrosymmetric`.
#
# The writer's copy-mirror is an IDENTITY copy (mLPSH = mLPNH), correct only when
# NH is invariant under the both-axis Lambert flip — true for cubic (and for
# high-symmetry centro cells), but FALSE for low-symmetry centro cells.  Measured
# on monoclinic-centro Al13Fe4 (SG12) and beta-AlFeSi (SG15): identity copy gives
# SH NCC ~0.71–0.76 vs 1.00 for the direct south build (which returns NH.flip).
# This test pins that invariant so nobody relaxes the gate to `is_centrosymmetric`.
# ---------------------------------------------------------------------------
class TestBuildShGate:
    _XTAL = (
        _ROOT / "Database" / "XTAL_Library" / "Al13Fe4.xtal"
    )  # monoclinic C2/m, centrosymmetric, NOT flip-invariant

    @pytest.mark.skipif(
        not _XTAL.exists(),
        reason="Database/XTAL_Library/Al13Fe4.xtal not present (gitignored library)",
    )
    def test_noncubic_centro_identity_copy_is_wrong_sh(self):
        """For a low-symmetry centrosymmetric cell the writer's identity copy-mirror
        (mLPSH = mLPNH) is a POOR southern hemisphere, while build_master(south)
        is exact — so build_sh must trigger on `not cubic`, not `not centro`."""
        import numpy as np
        from backend.forward_sim.crystal.xtal_io import read_crystal_structure
        from backend.forward_sim.mc.gpu_monte_carlo import run_gpu_mc, MCConfig
        from backend.forward_sim.mc.composition import mc_composition_from_structure
        from backend.forward_sim.dynamical.master_builder import build_master, _is_cubic

        s = read_crystal_structure(str(self._XTAL))
        # Precondition: this cell is exactly the dangerous case (non-cubic + centro).
        assert not _is_cubic(s)
        assert s.is_centrosymmetric

        comp = mc_composition_from_structure(s)
        cfg = MCConfig(
            starting_E_keV=20.0, Ehistmin=10.0, Ebinsize=1.0, nE=11,
            depth_step=1.0, depth_max=100.0, sig_deg=70.0, omega_deg=0.0,
            n_dir=501, n_dir_z=51, n_simulations=60_000,
            n_electrons_parallel=60_000, n_max_steps=300, device="cpu",
        )
        mc = run_gpu_mc(comp.mean_Z, comp.mean_A, comp.rho, cfg)
        eidx = mc.n_energy - 1
        common = dict(npx=20, energy_idx=eidx, dmin=0.13, device="cpu", dir_chunk="auto")
        nh = build_master(s, mc, hemisphere="north", **common).detach().cpu().numpy().astype(np.float64)
        sh = build_master(s, mc, hemisphere="south", **common).detach().cpu().numpy().astype(np.float64)

        def _ncc(a, b):
            nz = np.isfinite(a) & np.isfinite(b) & ((a != 0) | (b != 0))
            a = a[nz] - a[nz].mean(); b = b[nz] - b[nz].mean()
            d = np.sqrt((a * a).sum() * (b * b).sum())
            return float((a * b).sum() / d)

        ncc_copy = _ncc(nh, sh)          # writer identity copy-mirror would use NH as SH
        ncc_south = _ncc(sh, sh)         # the actual south build (sanity = 1.0)
        # The identity copy is clearly WRONG here (would be ~1.0 if it were valid).
        assert ncc_copy < 0.95, (
            f"identity copy-mirror unexpectedly matched the true SH (NCC={ncc_copy:.4f}); "
            f"if this cell is flip-invariant pick a lower-symmetry monoclinic cell"
        )
        assert ncc_south == pytest.approx(1.0, abs=1e-9)

    def test_gate_uses_cubic_not_centrosymmetric(self):
        """Static guard: the build_sh gate must be `not structure_is_cubic`.

        Relaxing it to `is_centrosymmetric` would write the wrong SH for the cells
        above, so we pin the source text (cheap, no .xtal / GPU needed)."""
        src = (
            _ROOT / "backend" / "api" / "services" / "gpu_sim_runner.py"
        ).read_text(encoding="utf-8")
        assert "build_sh = not structure_is_cubic" in src
        assert "build_sh = not structure.is_centrosymmetric" not in src


# ---------------------------------------------------------------------------
# 4. GPU job lock serialisation
# ---------------------------------------------------------------------------

class TestGpuJobLock:
    """GPU build lock must serialise concurrent acquisitions — no overlap."""

    def test_lock_serialises_two_concurrent_acquisitions(self):
        """Two threads acquiring _gpu_build_lock must NOT run simultaneously.

        We measure the time of 'inside lock' overlap: with serialisation,
        thread B must start its critical section only after thread A has
        released the lock.
        """
        from backend.api.routes import simulation as sim_mod

        overlap_detected = threading.Event()
        inside_count = [0]
        inside_count_lock = threading.Lock()
        timeline: list = []

        def _worker(name: str, hold_seconds: float) -> None:
            with sim_mod._gpu_build_lock:
                with inside_count_lock:
                    inside_count[0] += 1
                    count_now = inside_count[0]
                    if count_now > 1:
                        overlap_detected.set()
                timeline.append((name, "enter", time.monotonic()))
                time.sleep(hold_seconds)
                timeline.append((name, "exit", time.monotonic()))
                with inside_count_lock:
                    inside_count[0] -= 1

        t1 = threading.Thread(target=_worker, args=("A", 0.05))
        t2 = threading.Thread(target=_worker, args=("B", 0.05))

        t1.start()
        time.sleep(0.005)  # give t1 a head start
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        assert not overlap_detected.is_set(), (
            "Two threads were inside _gpu_build_lock simultaneously — "
            "the lock is NOT serialising!"
        )

        # Verify ordering: A must exit before B enters
        events = {(n, e): ts for n, e, ts in timeline}
        assert events[("A", "exit")] <= events[("B", "enter")] + 1e-3, (
            f"B entered before A exited: "
            f"A.exit={events.get(('A','exit')):.4f}  "
            f"B.enter={events.get(('B','enter')):.4f}"
        )

    def test_lock_present_in_simulation_module(self):
        """_gpu_build_lock must be a threading.Lock on the simulation module."""
        from backend.api.routes import simulation as sim_mod
        assert hasattr(sim_mod, "_gpu_build_lock"), (
            "_gpu_build_lock not found on simulation module"
        )
        lock = sim_mod._gpu_build_lock
        # A threading.Lock can be acquired; type check is duck-typed
        assert hasattr(lock, "acquire") and hasattr(lock, "release"), (
            "_gpu_build_lock does not look like a threading.Lock"
        )

    def test_start_gpu_endpoint_acquires_lock(self):
        """The run_gpu_sim closure in /start-gpu must reference _gpu_build_lock."""
        import ast
        src = (
            _ROOT / "backend" / "api" / "routes" / "simulation.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(src)

        found_lock_ref = False
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "start_gpu_simulation":
                body = ast.unparse(node)
                if "_gpu_build_lock" in body:
                    found_lock_ref = True
                    break

        assert found_lock_ref, (
            "start_gpu_simulation does not reference _gpu_build_lock — "
            "concurrent GPU jobs would run unserialized"
        )

    def test_batch_gpu_endpoint_acquires_lock(self):
        """The run_batch_gpu closure in /batch/start-gpu must reference _gpu_build_lock."""
        import ast
        src = (
            _ROOT / "backend" / "api" / "routes" / "simulation.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(src)

        found_lock_ref = False
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "start_batch_gpu":
                body = ast.unparse(node)
                if "_gpu_build_lock" in body:
                    found_lock_ref = True
                    break

        assert found_lock_ref, (
            "start_batch_gpu does not reference _gpu_build_lock — "
            "concurrent GPU batch jobs would run unserialized"
        )

    def test_batch_gpu_empty_cache_present(self):
        """run_batch_gpu must call torch.cuda.empty_cache() between phases."""
        import ast
        src = (
            _ROOT / "backend" / "api" / "routes" / "simulation.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(src)

        found_empty_cache = False
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "start_batch_gpu":
                body = ast.unparse(node)
                if "empty_cache" in body:
                    found_empty_cache = True
                    break

        assert found_empty_cache, (
            "start_batch_gpu does not call torch.cuda.empty_cache() — "
            "VRAM may not be freed between phases in a large-cell batch"
        )
