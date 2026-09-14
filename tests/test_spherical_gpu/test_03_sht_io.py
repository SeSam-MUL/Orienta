"""Tests for the EMsoft .sht binary master pattern reader."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from backend.spherical_gpu.exceptions import SHTReadError
from backend.spherical_gpu.pipeline.sht_io import (
    SHTMasterFile,
    read_sht_master,
)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_read_sht_returns_master_file_object(sht_path):
    """Reading the oracle-embedded SHT yields a populated SHTMasterFile."""
    sht = read_sht_master(sht_path, device=_device())
    assert isinstance(sht, SHTMasterFile)
    # Aluminum FCC, 20 kV; Fm-3m space group → m-3m point group.
    assert sht.point_group == "m-3m"
    assert sht.space_group == 225
    assert sht.voltage_kv == pytest.approx(20.0, abs=0.5)
    assert sht.formula.startswith("Al")
    # Lattice parameter for FCC Al: ~0.405 nm
    assert sht.crystal_lattice[0] == pytest.approx(0.405, abs=0.01)


def test_read_sht_bandwidth_is_384(sht_path):
    """The Al SHT was generated with the EMsoft default master-pattern bandwidth.

    Note: this is the master-pattern STORAGE bandwidth (384), distinct from
    the EMSphInx INDEXING bandwidth used at correlation time (typically 68
    or 88), which is set by the user in the NML.
    """
    sht = read_sht_master(sht_path, device=_device())
    assert sht.bandwidth == 384


def test_read_sht_coefs_shape(sht_path):
    """Coefficients dense layout is (bw, bw) complex, indexed [m, l]."""
    sht = read_sht_master(sht_path, device=_device())
    bw = sht.bandwidth
    assert sht.coefs_ml.shape == (bw, bw)
    assert sht.coefs_ml.dtype in (torch.complex64, torch.complex128)


def test_read_sht_lower_triangle_is_zero(sht_path):
    """l < m entries are zero by construction (alm undefined for l < |m|)."""
    sht = read_sht_master(sht_path, device=_device())
    bw = sht.bandwidth
    coefs = sht.coefs_ml.cpu().numpy()
    for m in range(min(bw, 32)):  # check first 32 m's for speed
        for ll in range(m):
            assert coefs[m, ll] == 0.0


def test_read_sht_zrot_zero_rows_are_zero(sht_path):
    """For Fm-3m (zRot=4), m % 4 != 0 rows are entirely zero."""
    sht = read_sht_master(sht_path, device=_device())
    coefs = sht.coefs_ml.cpu().numpy()
    z = sht.z_rot
    assert z == 4
    for m in range(min(sht.bandwidth, 16)):
        if m % z != 0:
            assert np.all(coefs[m, :] == 0.0), f"row m={m} should be all zero (zRot={z})"


def test_read_sht_some_nonzero_entries(sht_path):
    """Not all coefs are zero (sanity check)."""
    sht = read_sht_master(sht_path, device=_device())
    coefs = sht.coefs_ml.cpu().numpy()
    n_nonzero = int(np.count_nonzero(coefs))
    assert n_nonzero > 100, f"Only {n_nonzero} nonzero coefs — decoder likely broken"


def test_read_sht_device_placement(sht_path):
    """Coefficients land on the requested device."""
    cpu = torch.device("cpu")
    sht = read_sht_master(sht_path, device=cpu)
    assert sht.coefs_ml.device.type == "cpu"
    if torch.cuda.is_available():
        sht_gpu = read_sht_master(sht_path, device=torch.device("cuda"))
        assert sht_gpu.coefs_ml.device.type == "cuda"


def test_read_sht_invalid_magic_raises(tmp_path):
    """Garbage bytes raise SHTReadError, not silent garbage."""
    bad = tmp_path / "garbage.sht"
    bad.write_bytes(b"this is not an sht file" * 100)
    with pytest.raises(SHTReadError, match="[Bb]ad magic"):
        read_sht_master(str(bad), device=_device())


def test_read_sht_truncated_file_raises(tmp_path, sht_path):
    """A truncated SHT (header only) raises SHTReadError."""
    truncated = tmp_path / "truncated.sht"
    full = open(sht_path, "rb").read()
    truncated.write_bytes(full[:60])  # past magic but truncated
    with pytest.raises(SHTReadError):
        read_sht_master(str(truncated), device=_device())


def test_read_sht_nonexistent_path_raises(tmp_path):
    with pytest.raises(SHTReadError, match="not found"):
        read_sht_master(str(tmp_path / "no_such_file.sht"), device=_device())


def test_doub_cnt_matches_num_harm(sht_path):
    """The header's doubCnt must match what NumHarm() computes — this is the
    invariant SHT files must satisfy. If our parser passes this, the symmetry
    bookkeeping in unpack is correct.

    For the Al sample: NumHarm(384, 4, 0x07) = 9312, file claims 9312.
    """
    sht = read_sht_master(sht_path, device=_device())
    assert sht.raw_harm.doub_cnt == 9312
    assert sht.bandwidth == 384
    assert sht.z_rot == 4
    assert sht.cmp_flg == 0x07
