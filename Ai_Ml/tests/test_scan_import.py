"""Tests for the common import infrastructure (ScanData, euler conversion, format detection)."""

from __future__ import annotations

import struct
from pathlib import Path

import h5py
import numpy as np
import pytest

from ebsd_ai.data.scan_import import (
    ScanData,
    ScanFormat,
    degrees_to_radians,
    detect_format,
    euler_to_quaternion,
)


# ---------------------------------------------------------------------------
# Euler-to-quaternion conversion
# ---------------------------------------------------------------------------


class TestEulerToQuaternion:
    """Tests for euler_to_quaternion()."""

    def test_identity(self) -> None:
        """Euler (0, 0, 0) → quaternion (1, 0, 0, 0)."""
        q = euler_to_quaternion(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-12)

    def test_single_rotation_phi1(self) -> None:
        """Pure phi1 rotation (Z axis)."""
        angle = np.pi / 4  # 45 degrees
        q = euler_to_quaternion(np.array([angle, 0.0, 0.0]))
        # Z-rotation by phi1: q = (cos(phi1/2), 0, 0, sin(phi1/2))
        expected_w = np.cos(angle / 2)
        expected_z = np.sin(angle / 2)
        np.testing.assert_allclose(
            q, [expected_w, 0, 0, expected_z], atol=1e-12
        )

    def test_single_rotation_phi2(self) -> None:
        """Pure phi2 rotation (Z axis)."""
        angle = np.pi / 3  # 60 degrees
        q = euler_to_quaternion(np.array([0.0, 0.0, angle]))
        expected_w = np.cos(angle / 2)
        expected_z = np.sin(angle / 2)
        np.testing.assert_allclose(
            q, [expected_w, 0, 0, expected_z], atol=1e-12
        )

    def test_single_rotation_phi(self) -> None:
        """Pure Phi rotation (X axis) with phi1=phi2=0."""
        angle = np.pi / 2  # 90 degrees
        q = euler_to_quaternion(np.array([0.0, angle, 0.0]))
        # When phi1=phi2=0: sigma=0, delta=0
        # w = cos(Phi/2)*cos(0) = cos(Phi/2)
        # x = sin(Phi/2)*cos(0) = sin(Phi/2)
        # y = sin(Phi/2)*sin(0) = 0
        # z = cos(Phi/2)*sin(0) = 0
        expected_w = np.cos(angle / 2)
        expected_x = np.sin(angle / 2)
        np.testing.assert_allclose(
            q, [expected_w, expected_x, 0, 0], atol=1e-12
        )

    def test_unit_quaternion(self) -> None:
        """Result is always a unit quaternion."""
        rng = np.random.default_rng(42)
        angles = rng.uniform(0, 2 * np.pi, size=(100, 3))
        quats = euler_to_quaternion(angles)
        norms = np.linalg.norm(quats, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_positive_scalar(self) -> None:
        """Scalar part (w) is always non-negative (canonical form)."""
        rng = np.random.default_rng(123)
        angles = rng.uniform(0, 2 * np.pi, size=(200, 3))
        quats = euler_to_quaternion(angles)
        assert np.all(quats[:, 0] >= 0)

    def test_batch_shape(self) -> None:
        """Batch of N Euler angles → (N, 4) quaternions."""
        angles = np.zeros((50, 3))
        quats = euler_to_quaternion(angles)
        assert quats.shape == (50, 4)
        assert quats.dtype == np.float64

    def test_single_shape(self) -> None:
        """Single Euler angle (3,) → (4,) quaternion."""
        q = euler_to_quaternion(np.array([0.0, 0.0, 0.0]))
        assert q.shape == (4,)
        assert q.dtype == np.float64

    def test_known_values(self) -> None:
        """Verify against known Bunge → quaternion conversion values.

        phi1=90°, Phi=90°, phi2=0° is a well-known orientation.
        """
        phi1 = np.pi / 2
        big_phi = np.pi / 2
        phi2 = 0.0
        q = euler_to_quaternion(np.array([phi1, big_phi, phi2]))

        # Manual calculation:
        # sigma = (pi/2 + 0) / 2 = pi/4
        # delta = (pi/2 - 0) / 2 = pi/4
        # w = cos(pi/4) * cos(pi/4) = 0.5
        # x = sin(pi/4) * cos(pi/4) = 0.5
        # y = sin(pi/4) * sin(pi/4) = 0.5
        # z = cos(pi/4) * sin(pi/4) = 0.5
        np.testing.assert_allclose(q, [0.5, 0.5, 0.5, 0.5], atol=1e-12)

    def test_full_rotation_phi1(self) -> None:
        """phi1=2*pi should be the same as phi1=0 (identity)."""
        q = euler_to_quaternion(np.array([2 * np.pi, 0.0, 0.0]))
        # cos(pi)=-1, sin(pi)=0 → q = (-1, 0, 0, 0) → canonical: (1, 0, 0, 0)
        np.testing.assert_allclose(abs(q[0]), 1.0, atol=1e-10)
        np.testing.assert_allclose(q[1:], 0.0, atol=1e-10)

    def test_unsupported_convention(self) -> None:
        """Non-bunge convention raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported Euler convention"):
            euler_to_quaternion(np.array([0.0, 0.0, 0.0]), convention="roe")

    def test_wrong_shape(self) -> None:
        """Wrong array shape raises ValueError."""
        with pytest.raises(ValueError, match="Expected euler_angles"):
            euler_to_quaternion(np.array([[0.0, 0.0]]))

    def test_symmetry_phi1_phi2(self) -> None:
        """Swapping phi1 and phi2 with Phi=0 gives equivalent orientations."""
        # With Phi=0, both are just Z rotations; phi1+phi2 determines the net rotation
        angle = np.pi / 6
        q1 = euler_to_quaternion(np.array([angle, 0.0, 0.0]))
        q2 = euler_to_quaternion(np.array([0.0, 0.0, angle]))
        # Both should yield the same quaternion (just Z rotation by angle)
        np.testing.assert_allclose(q1, q2, atol=1e-12)


class TestDegreesToRadians:
    """Tests for degrees_to_radians()."""

    def test_zero(self) -> None:
        assert degrees_to_radians(np.array(0.0)) == 0.0

    def test_ninety(self) -> None:
        np.testing.assert_allclose(
            degrees_to_radians(np.array(90.0)), np.pi / 2
        )

    def test_batch(self) -> None:
        deg = np.array([0.0, 90.0, 180.0, 360.0])
        rad = degrees_to_radians(deg)
        np.testing.assert_allclose(rad, [0, np.pi / 2, np.pi, 2 * np.pi])


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


class TestDetectFormat:
    """Tests for detect_format()."""

    def test_ang_extension(self, tmp_path: Path) -> None:
        """File with .ang extension → ANG."""
        p = tmp_path / "scan.ang"
        p.write_text("# header\n1 2 3\n")
        assert detect_format(p) == ScanFormat.ANG

    def test_ctf_extension(self, tmp_path: Path) -> None:
        """File with .ctf extension → CTF."""
        p = tmp_path / "scan.ctf"
        p.write_text("Channel Text File\n")
        assert detect_format(p) == ScanFormat.CTF

    def test_h5oina_extension(self, tmp_path: Path) -> None:
        """File with .h5oina extension containing EBSD group → H5OINA."""
        p = tmp_path / "scan.h5oina"
        with h5py.File(p, "w") as f:
            g = f.create_group("1")
            g.create_group("EBSD")
        assert detect_format(p) == ScanFormat.H5OINA

    def test_h5_with_ebsd_group(self, tmp_path: Path) -> None:
        """Generic .h5 with EBSD group → H5OINA."""
        p = tmp_path / "scan.h5"
        with h5py.File(p, "w") as f:
            g = f.create_group("Measurement1")
            g.create_group("EBSD")
        assert detect_format(p) == ScanFormat.H5OINA

    def test_h5_kikuchipy(self, tmp_path: Path) -> None:
        """HDF5 with 'Scan 1' group → KIKUCHIPY."""
        p = tmp_path / "scan.h5"
        with h5py.File(p, "w") as f:
            f.create_group("Scan 1")
        assert detect_format(p) == ScanFormat.KIKUCHIPY

    def test_h5_emsoft(self, tmp_path: Path) -> None:
        """HDF5 with 'EMData' group → EMSOFT."""
        p = tmp_path / "scan.h5"
        with h5py.File(p, "w") as f:
            f.create_group("EMData")
        assert detect_format(p) == ScanFormat.EMSOFT

    def test_h5_unknown(self, tmp_path: Path) -> None:
        """HDF5 with no recognizable groups → UNKNOWN."""
        p = tmp_path / "scan.h5"
        with h5py.File(p, "w") as f:
            f.create_group("random_data")
        assert detect_format(p) == ScanFormat.UNKNOWN

    def test_unknown_extension(self, tmp_path: Path) -> None:
        """File with unknown extension → UNKNOWN."""
        p = tmp_path / "scan.xyz"
        p.write_text("data\n")
        assert detect_format(p) == ScanFormat.UNKNOWN

    def test_missing_file(self, tmp_path: Path) -> None:
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            detect_format(tmp_path / "nonexistent.ang")

    def test_non_hdf5_with_h5_extension(self, tmp_path: Path) -> None:
        """File with .h5 extension but not HDF5 content → UNKNOWN."""
        p = tmp_path / "fake.h5"
        p.write_bytes(b"not hdf5 at all")
        assert detect_format(p) == ScanFormat.UNKNOWN

    def test_emsoft_nml(self, tmp_path: Path) -> None:
        """HDF5 with 'NMLparameters' → EMSOFT."""
        p = tmp_path / "emsoft.h5"
        with h5py.File(p, "w") as f:
            f.create_group("NMLparameters")
        assert detect_format(p) == ScanFormat.EMSOFT

    def test_hdf5_extension(self, tmp_path: Path) -> None:
        """.hdf5 extension with EBSD → H5OINA."""
        p = tmp_path / "scan.hdf5"
        with h5py.File(p, "w") as f:
            g = f.create_group("1")
            g.create_group("EBSD")
        assert detect_format(p) == ScanFormat.H5OINA


# ---------------------------------------------------------------------------
# ScanData
# ---------------------------------------------------------------------------


class TestScanData:
    """Tests for the ScanData dataclass."""

    def _make_scan(
        self,
        n: int = 100,
        n_phases: int = 3,
        with_patterns: bool = True,
        with_eds: bool = True,
        rng: np.random.Generator | None = None,
    ) -> ScanData:
        """Create a valid ScanData for testing."""
        if rng is None:
            rng = np.random.default_rng(42)

        patterns = None
        if with_patterns:
            patterns = rng.integers(0, 256, (n, 60, 80), dtype=np.uint8)

        phase_ids = rng.integers(0, n_phases, size=n, dtype=np.int32)
        phase_names = [f"Phase_{i}" for i in range(n_phases)]
        euler_angles = rng.uniform(0, 2 * np.pi, (n, 3))
        confidence = rng.uniform(0.0, 1.0, n).astype(np.float32)

        eds_data = None
        if with_eds:
            eds_data = {
                "Fe": rng.uniform(0, 80, n).astype(np.float32),
                "Cr": rng.uniform(0, 20, n).astype(np.float32),
            }

        return ScanData(
            patterns=patterns,
            phase_ids=phase_ids,
            phase_names=phase_names,
            euler_angles=euler_angles,
            confidence_scores=confidence,
            eds_data=eds_data,
            scan_shape=(10, 10),
            step_sizes=(0.5, 0.5),
            source_format=ScanFormat.ANG,
            source_file="test.ang",
            metadata={"kv": 20.0},
        )

    def test_n_points_from_patterns(self) -> None:
        """n_points derived from patterns shape."""
        scan = self._make_scan(n=50)
        assert scan.n_points == 50

    def test_n_points_from_phase_ids(self) -> None:
        """n_points from phase_ids when no patterns."""
        scan = self._make_scan(n=30, with_patterns=False)
        assert scan.n_points == 30

    def test_has_patterns(self) -> None:
        scan = self._make_scan(with_patterns=True)
        assert scan.has_patterns is True

    def test_no_patterns(self) -> None:
        scan = self._make_scan(with_patterns=False)
        assert scan.has_patterns is False

    def test_has_eds(self) -> None:
        scan = self._make_scan(with_eds=True)
        assert scan.has_eds is True

    def test_no_eds(self) -> None:
        scan = self._make_scan(with_eds=False)
        assert scan.has_eds is False

    def test_eds_none(self) -> None:
        scan = ScanData()
        assert scan.has_eds is False

    def test_eds_empty_dict(self) -> None:
        scan = ScanData(eds_data={})
        assert scan.has_eds is False

    def test_orientations_property(self) -> None:
        """orientations converts Euler angles to quaternions."""
        scan = self._make_scan(n=10)
        quats = scan.orientations
        assert quats.shape == (10, 4)
        # All should be unit quaternions
        norms = np.linalg.norm(quats, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_orientations_empty(self) -> None:
        """orientations returns zero quaternions when euler_angles empty."""
        scan = ScanData(phase_ids=np.array([0, 1, 2], dtype=np.int32))
        quats = scan.orientations
        assert quats.shape == (3, 4)
        np.testing.assert_allclose(quats, 0.0)

    def test_validate_ok(self) -> None:
        """Valid ScanData has no issues."""
        scan = self._make_scan()
        issues = scan.validate()
        assert issues == []

    def test_validate_empty(self) -> None:
        """Empty ScanData reports zero points."""
        scan = ScanData()
        issues = scan.validate()
        assert any("0 points" in i for i in issues)

    def test_validate_phase_id_out_of_range(self) -> None:
        """phase_ids exceeding phase_names length detected."""
        scan = self._make_scan()
        scan.phase_ids = np.array([0, 1, 99] + [0] * 97, dtype=np.int32)
        issues = scan.validate()
        assert any("max phase_id" in i for i in issues)

    def test_validate_negative_phase_ids(self) -> None:
        """Negative phase_ids detected."""
        scan = self._make_scan()
        scan.phase_ids[0] = -1
        issues = scan.validate()
        assert any("negative" in i for i in issues)

    def test_validate_euler_shape_mismatch(self) -> None:
        """Euler angles shape mismatch detected."""
        scan = self._make_scan(n=100)
        scan.euler_angles = np.zeros((50, 3))
        issues = scan.validate()
        assert any("euler_angles" in i for i in issues)

    def test_validate_confidence_shape_mismatch(self) -> None:
        """Confidence scores shape mismatch detected."""
        scan = self._make_scan(n=100)
        scan.confidence_scores = np.zeros(50, dtype=np.float32)
        issues = scan.validate()
        assert any("confidence_scores" in i for i in issues)

    def test_validate_patterns_ndim(self) -> None:
        """Patterns with wrong ndim detected."""
        scan = self._make_scan()
        scan.patterns = np.zeros((100, 60), dtype=np.uint8)  # 2D instead of 3D
        issues = scan.validate()
        assert any("ndim" in i for i in issues)

    def test_validate_patterns_n_mismatch(self) -> None:
        """Mismatched patterns vs phase_ids detected."""
        scan = self._make_scan(n=100, with_patterns=False)
        # n_points=100 (from phase_ids), but patterns has only 50
        scan.patterns = np.zeros((50, 60, 80), dtype=np.uint8)
        issues = scan.validate()
        # n_points=100 (from phase_ids since no patterns initially),
        # but now patterns.shape[0]=50 triggers the mismatch check
        assert len(issues) > 0

    def test_validate_eds_shape_mismatch(self) -> None:
        """EDS array length mismatch detected."""
        scan = self._make_scan(n=100)
        assert scan.eds_data is not None
        scan.eds_data["Fe"] = np.zeros(50, dtype=np.float32)
        issues = scan.validate()
        assert any("eds_data" in i for i in issues)

    def test_validate_phase_ids_shape(self) -> None:
        """phase_ids shape mismatch detected."""
        scan = self._make_scan(n=100)
        scan.phase_ids = np.array([0, 1, 2], dtype=np.int32)
        issues = scan.validate()
        assert any("phase_ids" in i for i in issues)

    def test_default_construction(self) -> None:
        """Default ScanData has sensible defaults."""
        scan = ScanData()
        assert scan.patterns is None
        assert scan.n_points == 0
        assert scan.phase_names == ["Unknown"]
        assert scan.source_format == ScanFormat.UNKNOWN
        assert scan.has_patterns is False
        assert scan.has_eds is False

    def test_scan_shape_and_step(self) -> None:
        """scan_shape and step_sizes stored correctly."""
        scan = self._make_scan()
        assert scan.scan_shape == (10, 10)
        assert scan.step_sizes == (0.5, 0.5)

    def test_metadata(self) -> None:
        """Arbitrary metadata stored correctly."""
        scan = self._make_scan()
        assert scan.metadata["kv"] == 20.0

    def test_source_info(self) -> None:
        """Source format and file tracked."""
        scan = self._make_scan()
        assert scan.source_format == ScanFormat.ANG
        assert scan.source_file == "test.ang"
