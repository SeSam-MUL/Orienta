"""Tests for HDF5 checkpoint writer (_multiphase.h5)."""
import os
import numpy as np
import pytest
import h5py
from backend.api.services.checkpoint_writer import CheckpointWriter


class TestCheckpointWriter:

    def test_derive_path(self, tmp_path):
        source = str(tmp_path / "Sample_A.h5oina")
        cw = CheckpointWriter(source)
        assert cw.checkpoint_path == str(tmp_path / "Sample_A_multiphase.h5")

    def test_write_phase_result_creates_file(self, tmp_path, small_ci_map, small_orientation_map):
        source = str(tmp_path / "Sample_A.h5oina")
        cw = CheckpointWriter(source)
        cw.init_metadata(grid_shape=(10, 10), batch_id="test-123", method="spherical")
        cw.write_phase_result(
            phase_name="Ferrit",
            ci_map=small_ci_map,
            orientation_map=small_orientation_map,
            metadata={"phase_file": "/p/Fe.sht", "ci_mean": 0.72, "duration_sec": 15.0},
        )
        assert cw.checkpoint_path_exists()
        with h5py.File(cw.checkpoint_path, "r") as f:
            assert "phases/Ferrit/ci" in f
            assert "phases/Ferrit/orientation" in f
            assert f["phases/Ferrit/ci"].shape == (10, 10)

    def test_phase_already_done(self, tmp_path, small_ci_map, small_orientation_map):
        source = str(tmp_path / "Sample_A.h5oina")
        cw = CheckpointWriter(source)
        cw.init_metadata(grid_shape=(10, 10), batch_id="test-123", method="spherical")
        assert cw.phase_already_done("Ferrit") is False
        cw.write_phase_result("Ferrit", small_ci_map, small_orientation_map, {"phase_file": "/p", "ci_mean": 0.5, "duration_sec": 1.0})
        assert cw.phase_already_done("Ferrit") is True

    def test_get_completed_phases(self, tmp_path, small_ci_map, small_orientation_map):
        source = str(tmp_path / "Sample_A.h5oina")
        cw = CheckpointWriter(source)
        cw.init_metadata(grid_shape=(10, 10), batch_id="test-123", method="spherical")
        cw.write_phase_result("Al", small_ci_map, small_orientation_map, {"phase_file": "/p", "ci_mean": 0.5, "duration_sec": 1.0})
        cw.write_phase_result("Fe", small_ci_map, small_orientation_map, {"phase_file": "/p", "ci_mean": 0.6, "duration_sec": 2.0})
        assert set(cw.get_completed_phases()) == {"Al", "Fe"}

    def test_compute_auto_assignment(self, tmp_path):
        source = str(tmp_path / "Sample_A.h5oina")
        cw = CheckpointWriter(source)
        cw.init_metadata(grid_shape=(3, 3), batch_id="test-123", method="spherical")
        # Phase A: high CI everywhere
        ci_a = np.array([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4], [0.3, 0.2, 0.1]], dtype=np.float32)
        ori_a = np.zeros((3, 3, 3), dtype=np.float32)
        cw.write_phase_result("PhaseA", ci_a, ori_a, {"phase_file": "/p", "ci_mean": 0.5, "duration_sec": 1.0})
        # Phase B: high CI in bottom-right
        ci_b = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
        ori_b = np.ones((3, 3, 3), dtype=np.float32)
        cw.write_phase_result("PhaseB", ci_b, ori_b, {"phase_file": "/p", "ci_mean": 0.5, "duration_sec": 1.0})

        cw.compute_auto_assignment(confidence_threshold=0.3)

        with h5py.File(cw.checkpoint_path, "r") as f:
            best_id = f["auto_assignment/best_phase_id"][:]
            best_ci = f["auto_assignment/best_ci"][:]
            uncertainty = f["auto_assignment/uncertainty"][:]
            # Top-left should be PhaseA (idx 0), bottom-right PhaseB (idx 1)
            assert best_id[0, 0] == 0  # PhaseA wins at (0,0): 0.9 > 0.1
            assert best_id[2, 2] == 1  # PhaseB wins at (2,2): 0.9 > 0.1
            assert np.isclose(best_ci[0, 0], 0.9)
            assert np.isclose(uncertainty[1, 1], 0.0)  # Tie at (1,1): 0.5 vs 0.5

    def test_read_phase_ci(self, tmp_path, small_ci_map, small_orientation_map):
        source = str(tmp_path / "Sample_A.h5oina")
        cw = CheckpointWriter(source)
        cw.init_metadata(grid_shape=(10, 10), batch_id="test-123", method="spherical")
        cw.write_phase_result("Al", small_ci_map, small_orientation_map, {"phase_file": "/p", "ci_mean": 0.5, "duration_sec": 1.0})
        read_ci = cw.read_phase_ci("Al")
        np.testing.assert_array_almost_equal(read_ci, small_ci_map)

    def test_read_all_phase_cis(self, tmp_path, small_ci_map, small_orientation_map):
        source = str(tmp_path / "Sample_A.h5oina")
        cw = CheckpointWriter(source)
        cw.init_metadata(grid_shape=(10, 10), batch_id="test-123", method="spherical")
        cw.write_phase_result("Al", small_ci_map, small_orientation_map, {"phase_file": "/p", "ci_mean": 0.5, "duration_sec": 1.0})
        cw.write_phase_result("Fe", small_ci_map * 0.5, small_orientation_map, {"phase_file": "/p", "ci_mean": 0.25, "duration_sec": 2.0})
        all_cis = cw.read_all_phase_cis()
        assert "Al" in all_cis
        assert "Fe" in all_cis

    def test_validate_ok(self, tmp_path, small_ci_map, small_orientation_map):
        source = str(tmp_path / "Sample_A.h5oina")
        cw = CheckpointWriter(source)
        cw.init_metadata(grid_shape=(10, 10), batch_id="test-123", method="spherical")
        cw.write_phase_result("Al", small_ci_map, small_orientation_map, {"phase_file": "/p", "ci_mean": 0.5, "duration_sec": 1.0})
        result = cw.validate()
        assert result.valid is True

    def test_validate_detects_corrupt(self, tmp_path):
        source = str(tmp_path / "Sample_A.h5oina")
        cw = CheckpointWriter(source)
        # Write garbage to the checkpoint path
        with open(cw.checkpoint_path, "wb") as f:
            f.write(b"NOT_HDF5_CONTENT")
        result = cw.validate()
        assert result.valid is False

    def test_write_manual_override(self, tmp_path, small_ci_map, small_orientation_map):
        source = str(tmp_path / "Sample_A.h5oina")
        cw = CheckpointWriter(source)
        cw.init_metadata(grid_shape=(10, 10), batch_id="test-123", method="spherical")
        cw.write_phase_result("Al", small_ci_map, small_orientation_map, {"phase_file": "/p", "ci_mean": 0.5, "duration_sec": 1.0})
        override_map = np.full((10, 10), -1, dtype=np.int8)
        override_map[5, 5] = 0  # Override pixel (5,5) to phase index 0
        source_map = np.zeros((10, 10), dtype=np.uint8)
        source_map[5, 5] = 1  # click
        cw.write_manual_override(override_map, source_map)
        with h5py.File(cw.checkpoint_path, "r") as f:
            assert f["manual_override/phase_id"][5, 5] == 0
            assert f["manual_override/override_source"][5, 5] == 1

    def test_tmp_file_cleanup_on_resume(self, tmp_path):
        source = str(tmp_path / "Sample_A.h5oina")
        cw = CheckpointWriter(source)
        # Simulate crash: leave a .tmp file
        tmp_file = cw.checkpoint_path + ".tmp"
        with open(tmp_file, "w") as f:
            f.write("crash artifact")
        # New writer should detect and clean up
        cw2 = CheckpointWriter(source)
        assert not os.path.exists(tmp_file)
