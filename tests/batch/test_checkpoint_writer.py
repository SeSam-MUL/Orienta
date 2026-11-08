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


# ---------------------------------------------------------------- geometry stamp

from backend.api.services.checkpoint_writer import read_checkpoint_geometry
from backend.api.services.scan_geometry import (
    ScanGeometry, geometry_from_attrs, geometry_to_attrs,
)

# Hand calculation: a hexagonal scan of step 2.0 um has rows 2.0 * sqrt(3) / 2 = 1.7320508 um apart.


def _hex_3x4():
    return ScanGeometry.hex(3, 4, 2.0)          # odd rows +0.5, shift not confirmed


def _new_cw(tmp_path):
    return CheckpointWriter(str(tmp_path / "Sample_H.h5oina"))


class TestGeometryAttrs:

    def test_hex_to_attrs(self):
        attrs = geometry_to_attrs(_hex_3x4())
        assert attrs["grid_type"] == "HexGrid"
        assert attrs["odd_row_shift"] == 0.5
        assert attrs["shift_confirmed"] is False
        assert attrs["step_x"] == 2.0
        assert attrs["step_y"] == pytest.approx(1.7320508, abs=1e-7)

    def test_square_to_attrs(self):
        attrs = geometry_to_attrs(ScanGeometry.square(5, 6, 1.5, 1.5))
        assert attrs["grid_type"] == "SqrGrid"
        assert attrs["odd_row_shift"] == 0.0
        assert attrs["shift_confirmed"] is True
        assert (attrs["step_x"], attrs["step_y"]) == (1.5, 1.5)

    def test_from_attrs_round_trip_hex_and_square(self):
        for g in (_hex_3x4(), ScanGeometry.hex(3, 4, 2.0, odd_row_shift=-0.5, shift_confirmed=True),
                  ScanGeometry.square(5, 6, 1.5, 0.75)):
            back = geometry_from_attrs(geometry_to_attrs(g), g.grid_shape)
            assert back == g

    def test_from_attrs_accepts_numpy_and_bytes(self):
        attrs = {"grid_type": np.bytes_(b"HexGrid"), "odd_row_shift": np.float64(0.5),
                 "shift_confirmed": np.bool_(False), "step_x": np.float64(2.0),
                 "step_y": np.float64(2.0 * 0.8660254037844386)}
        g = geometry_from_attrs(attrs, (3, 4))
        assert g.is_hex and g.grid_shape == (3, 4) and g.odd_row_shift == 0.5
        assert g.shift_confirmed is False

    @pytest.mark.parametrize("missing", ["odd_row_shift", "shift_confirmed", "step_x", "step_y"])
    def test_hex_with_a_missing_attribute_raises(self, missing):
        attrs = geometry_to_attrs(_hex_3x4())
        del attrs[missing]
        with pytest.raises(ValueError, match=missing):
            geometry_from_attrs(attrs, (3, 4))

    def test_unknown_grid_type_raises(self):
        attrs = geometry_to_attrs(_hex_3x4())
        attrs["grid_type"] = "TriGrid"
        with pytest.raises(ValueError, match="TriGrid"):
            geometry_from_attrs(attrs, (3, 4))

    def test_grid_shape_that_differs_from_the_stamp_raises(self):
        attrs = geometry_to_attrs(_hex_3x4())
        attrs["grid_shape"] = [3, 4]
        assert geometry_from_attrs(attrs, (3, 4)).grid_shape == (3, 4)
        with pytest.raises(ValueError, match="grid_shape"):
            geometry_from_attrs(attrs, (4, 3))


class TestCheckpointGeometry:

    def test_hex_round_trip_through_the_file(self, tmp_path):
        cw = _new_cw(tmp_path)
        cw.init_metadata(grid_shape=(3, 4), batch_id="b", method="spherical",
                         step_size_um=2.0, geometry=_hex_3x4())
        with h5py.File(cw.checkpoint_path, "r") as f:
            meta = f["metadata"].attrs
            assert str(meta["grid_type"]) == "HexGrid"
            assert float(meta["odd_row_shift"]) == 0.5
            assert bool(meta["shift_confirmed"]) is False
            assert float(meta["step_x"]) == 2.0
            assert float(meta["step_y"]) == pytest.approx(1.7320508, abs=1e-7)
            assert float(meta["step_size_um"]) == 2.0         # stays = step_x
        g = read_checkpoint_geometry(cw.checkpoint_path)
        assert g == _hex_3x4()
        assert g.is_hex and g.grid_shape == (3, 4) and g.shift_confirmed is False

    def test_square_round_trip_through_the_file(self, tmp_path):
        cw = _new_cw(tmp_path)
        cw.init_metadata(grid_shape=(5, 6), batch_id="b", method="spherical",
                         step_size_um=1.5, geometry=ScanGeometry.square(5, 6, 1.5))
        assert read_checkpoint_geometry(cw.checkpoint_path) == ScanGeometry.square(5, 6, 1.5)

    def test_geometry_alone_sets_step_size_um(self, tmp_path):
        cw = _new_cw(tmp_path)
        cw.init_metadata(grid_shape=(3, 4), batch_id="b", method="spherical", geometry=_hex_3x4())
        with h5py.File(cw.checkpoint_path, "r") as f:
            assert float(f["metadata"].attrs["step_size_um"]) == 2.0

    def test_legacy_checkpoint_reads_as_none(self, tmp_path):
        cw = _new_cw(tmp_path)
        cw.init_metadata(grid_shape=(10, 10), batch_id="b", method="spherical", step_size_um=1.0)
        with h5py.File(cw.checkpoint_path, "r") as f:
            assert "grid_type" not in f["metadata"].attrs
        assert read_checkpoint_geometry(cw.checkpoint_path) is None

    def test_hex_checkpoint_missing_its_shift_raises(self, tmp_path):
        cw = _new_cw(tmp_path)
        cw.init_metadata(grid_shape=(3, 4), batch_id="b", method="spherical", geometry=_hex_3x4())
        with h5py.File(cw.checkpoint_path, "a") as f:
            del f["metadata"].attrs["odd_row_shift"]
        with pytest.raises(ValueError, match="odd_row_shift"):
            read_checkpoint_geometry(cw.checkpoint_path)

    def test_unknown_grid_type_in_the_file_raises(self, tmp_path):
        cw = _new_cw(tmp_path)
        cw.init_metadata(grid_shape=(3, 4), batch_id="b", method="spherical", geometry=_hex_3x4())
        with h5py.File(cw.checkpoint_path, "a") as f:
            f["metadata"].attrs["grid_type"] = "TriGrid"
        with pytest.raises(ValueError, match="TriGrid"):
            read_checkpoint_geometry(cw.checkpoint_path)

    def test_geometry_that_does_not_fit_grid_shape_raises(self, tmp_path):
        cw = _new_cw(tmp_path)
        with pytest.raises(ValueError, match="grid"):
            cw.init_metadata(grid_shape=(4, 3), batch_id="b", method="spherical", geometry=_hex_3x4())
        assert not cw.checkpoint_path_exists()

    def test_step_size_that_contradicts_the_geometry_raises(self, tmp_path):
        cw = _new_cw(tmp_path)
        with pytest.raises(ValueError, match="step"):
            cw.init_metadata(grid_shape=(3, 4), batch_id="b", method="spherical",
                             step_size_um=5.0, geometry=_hex_3x4())

    def test_resume_with_the_same_geometry_is_idempotent(self, tmp_path):
        cw = _new_cw(tmp_path)
        cw.init_metadata(grid_shape=(3, 4), batch_id="b", method="spherical", geometry=_hex_3x4())
        cw.init_metadata(grid_shape=(3, 4), batch_id="b2", method="spherical", geometry=_hex_3x4())
        assert read_checkpoint_geometry(cw.checkpoint_path) == _hex_3x4()

    @pytest.mark.parametrize("other", [
        ScanGeometry.hex(3, 4, 2.0, odd_row_shift=-0.5),          # other direction
        ScanGeometry.hex(3, 4, 2.0, shift_confirmed=True),        # other confirmation
        ScanGeometry.hex(3, 4, 3.0),                              # other step
        ScanGeometry.square(3, 4, 2.0),                           # other kind
    ])
    def test_resume_with_a_different_geometry_raises(self, tmp_path, other):
        cw = _new_cw(tmp_path)
        cw.init_metadata(grid_shape=(3, 4), batch_id="b", method="spherical", geometry=_hex_3x4())
        with pytest.raises(ValueError, match="geometry"):
            cw.init_metadata(grid_shape=(3, 4), batch_id="b", method="spherical", geometry=other)
        assert read_checkpoint_geometry(cw.checkpoint_path) == _hex_3x4()

    def test_resume_of_a_legacy_checkpoint_with_a_hex_geometry_raises(self, tmp_path):
        cw = _new_cw(tmp_path)
        cw.init_metadata(grid_shape=(3, 4), batch_id="b", method="spherical", step_size_um=2.0)
        with pytest.raises(ValueError, match="geometry"):
            cw.init_metadata(grid_shape=(3, 4), batch_id="b", method="spherical", geometry=_hex_3x4())
        assert read_checkpoint_geometry(cw.checkpoint_path) is None

    def test_resume_without_a_geometry_of_a_hexagonal_checkpoint_is_refused_and_keeps_the_stamp(self, tmp_path):
        """No geometry means the file is not a hexagonal scan (a square one whose step came from elsewhere): the
        kind is still compared."""
        cw = _new_cw(tmp_path)
        cw.init_metadata(grid_shape=(3, 4), batch_id="b", method="spherical", geometry=_hex_3x4())
        with pytest.raises(CheckpointPairingChanged) as raised:
            cw.init_metadata(grid_shape=(3, 4), batch_id="b", method="spherical")
        assert raised.value.code == "checkpointPairingChanged"
        assert "not hexagonal" in str(raised.value)
        assert read_checkpoint_geometry(cw.checkpoint_path) == _hex_3x4()

    @pytest.mark.parametrize("stored", [None, ScanGeometry.square(3, 4, 2.0)], ids=["legacy", "square stamp"])
    def test_resume_without_a_geometry_of_a_square_checkpoint_is_as_before(self, tmp_path, stored):
        cw = _new_cw(tmp_path)
        cw.init_metadata(grid_shape=(3, 4), batch_id="b", method="spherical", step_size_um=2.0, geometry=stored)
        cw.init_metadata(grid_shape=(3, 4), batch_id="b2", method="spherical", step_size_um=0.5)
        assert read_checkpoint_geometry(cw.checkpoint_path) == stored


# ---------------------------------------------------------------- the pairing a checkpoint was written for

from backend.api.services.checkpoint_writer import CheckpointPairingChanged   # noqa: E402

APEX_A = {"h5_path": "/some/folder/session.h5", "measurement_id": "GUID-A", "measurement_name": "Area 1 / Map 1"}
APEX_B = {"h5_path": "/some/folder/session.h5", "measurement_id": "GUID-B", "measurement_name": "Area 1 / Map 2"}


def _attrs(cw):
    with h5py.File(cw.checkpoint_path, "r") as f:
        return {k: f["metadata"].attrs[k] for k in f["metadata"].attrs}


class TestCheckpointPairing:

    def test_the_pairing_is_stored_by_the_file_name_the_id_and_the_name(self, tmp_path):
        cw = _new_cw(tmp_path)
        cw.init_metadata(grid_shape=(3, 4), batch_id="b", method="spherical", geometry=_hex_3x4(), apex=APEX_A)
        attrs = _attrs(cw)
        assert attrs["apex_h5_name"] == "session.h5"                    # the name, not the folder of this machine
        assert attrs["apex_measurement_id"] == "GUID-A"
        assert attrs["apex_measurement_name"] == "Area 1 / Map 1"

    def test_a_file_without_a_pairing_stores_empty_texts(self, tmp_path):
        cw = _new_cw(tmp_path)
        cw.init_metadata(grid_shape=(3, 4), batch_id="b", method="spherical", geometry=_hex_3x4())
        attrs = _attrs(cw)
        assert (attrs["apex_h5_name"], attrs["apex_measurement_id"], attrs["apex_measurement_name"]) == ("", "", "")

    def test_checkpoint_other_pairing_refused(self, tmp_path):
        cw = _new_cw(tmp_path)
        cw.init_metadata(grid_shape=(3, 4), batch_id="b", method="spherical", geometry=_hex_3x4(), apex=APEX_A)
        with pytest.raises(CheckpointPairingChanged) as raised:
            cw.init_metadata(grid_shape=(3, 4), batch_id="b", method="spherical", geometry=_hex_3x4(), apex=APEX_B)
        assert raised.value.code == "checkpointPairingChanged"
        assert isinstance(raised.value, ValueError)
        assert "GUID-A" in str(raised.value) and "GUID-B" in str(raised.value)
        assert _attrs(cw)["apex_measurement_id"] == "GUID-A"            # nothing was overwritten

    def test_a_pairing_where_there_was_none_and_none_where_there_was_one_are_refused(self, tmp_path):
        cw = _new_cw(tmp_path)
        cw.init_metadata(grid_shape=(3, 4), batch_id="b", method="spherical", geometry=_hex_3x4())
        with pytest.raises(CheckpointPairingChanged):
            cw.init_metadata(grid_shape=(3, 4), batch_id="b", method="spherical", geometry=_hex_3x4(), apex=APEX_A)
        other = CheckpointWriter(str(tmp_path / "Other.h5oina"))
        other.init_metadata(grid_shape=(3, 4), batch_id="b", method="spherical", geometry=_hex_3x4(), apex=APEX_A)
        with pytest.raises(CheckpointPairingChanged):
            other.init_metadata(grid_shape=(3, 4), batch_id="b", method="spherical", geometry=_hex_3x4())

    def test_the_same_pairing_resumes_even_when_the_metadata_file_was_renamed(self, tmp_path):
        """The measurement is the GUID, not the name of the file that holds it."""
        cw = _new_cw(tmp_path)
        cw.init_metadata(grid_shape=(3, 4), batch_id="b", method="spherical", geometry=_hex_3x4(), apex=APEX_A)
        renamed = {**APEX_A, "h5_path": "/elsewhere/renamed.h5"}
        cw.init_metadata(grid_shape=(3, 4), batch_id="b2", method="spherical", geometry=_hex_3x4(), apex=renamed)
        assert _attrs(cw)["apex_measurement_id"] == "GUID-A"

    def test_a_checkpoint_written_before_pairings_were_stored_is_not_refused_for_one(self, tmp_path):
        """No ``apex_measurement_id`` attribute at all: the checkpoint cannot say what it was paired with."""
        cw = _new_cw(tmp_path)
        cw.init_metadata(grid_shape=(3, 4), batch_id="b", method="spherical", geometry=_hex_3x4())
        with h5py.File(cw.checkpoint_path, "a") as f:
            for key in ("apex_h5_name", "apex_measurement_id", "apex_measurement_name"):
                del f["metadata"].attrs[key]
        cw.init_metadata(grid_shape=(3, 4), batch_id="b", method="spherical", geometry=_hex_3x4(), apex=APEX_A)

    def test_a_different_geometry_is_the_same_coded_refusal(self, tmp_path):
        cw = _new_cw(tmp_path)
        cw.init_metadata(grid_shape=(3, 4), batch_id="b", method="spherical", geometry=_hex_3x4())
        with pytest.raises(CheckpointPairingChanged) as raised:
            cw.init_metadata(grid_shape=(3, 4), batch_id="b", method="spherical",
                             geometry=ScanGeometry.square(3, 4, 2.0))
        assert raised.value.code == "checkpointPairingChanged"

    def test_the_other_refusals_of_init_metadata_are_not_that_code(self, tmp_path):
        """A geometry that does not fit the grid given is a caller's mistake, not a changed pairing."""
        cw = _new_cw(tmp_path)
        with pytest.raises(ValueError) as raised:
            cw.init_metadata(grid_shape=(4, 3), batch_id="b", method="spherical", geometry=_hex_3x4())
        assert not isinstance(raised.value, CheckpointPairingChanged)

