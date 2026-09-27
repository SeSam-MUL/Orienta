"""Tests for the SP3 EMSphInx ``.sht`` serialiser (Task: SP3 part 2).

:func:`backend.forward_sim.io.sht_writer.write_sht` reconstructs our GPU-computed
northern-hemisphere Lambert master on a Driscoll-Healy sphere, forward-transforms it
to spherical-harmonic coefficients, and writes the EMSphInx ``*sht`` binary so the
project reader :mod:`backend.spherical_gpu.pipeline.sht_io` (and the EMSphInx C++
``sht::File::read``) can load it.

Coverage levels:

* **CRC algorithm parity (no oracle data, always runs except for the Ni file)** —
  the EMSphInx-specific CRC reproduces the real Ni ``.sht`` trailer bit-exactly.
* **Round-trip (always runs)** — write a synthetic master, ``sht_io``-read it back,
  assert the header / lattice / point-group / formula metadata, that the read-back
  coefficients equal what we wrote (FP precision), that our file's own CRC trailer
  is self-consistent, and that the lossless (``sg_eff=1``) reconstruction matches
  the band-limited master.
* **Cached-Ni round-trip (runs when the Ni oracle ``.h5`` is present)** — write a
  ``.sht`` from the real cached Ni master, read it back with ``sht_io``, and assert
  the container fidelity (read-back reconstruction == written reconstruction) and
  that the reconstruction NCC equals the pure-CSHT band-limit ceiling.

Runs on CPU (``FORWARD_SIM_DEVICE=cpu``) so it needs no CUDA.
"""
from __future__ import annotations

import os
import struct

os.environ.setdefault("FORWARD_SIM_DEVICE", "cpu")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from backend.forward_sim.crystal.xtal_io import Atom, CrystalStructure  # noqa: E402
from backend.forward_sim.io.sht_writer import (  # noqa: E402
    _crc32c_emsoft,
    _forward_sht_coefs,
    _lambert_nh_to_dh_grid,
    _num_harm,
    _pack_harm,
    _SG_CMP,
    _SG_ROT,
    write_sht,
)
from backend.spherical_gpu.pipeline.sht_io import read_sht_master  # noqa: E402
from backend.spherical_gpu._math.sht_inverse import (  # noqa: E402
    inverse_sht_to_dh_grid,
)

NI_SHT = "Database/EBSD_SHT_Database/Ni/Ni (Ni) [cF4] {20kV}.sht"
NI_CACHE = "Database/EBSD_H5_Cache/Ni/Ni_master_E20kV_npx500.h5"


def _toy_structure(sg: int = 225) -> CrystalStructure:
    """A cubic (Ni-like Fm-3m) structure for the synthetic round-trips."""
    return CrystalStructure(
        lattice=(0.3524, 0.3524, 0.3524, 90.0, 90.0, 90.0),
        atoms=[Atom(Z=28, xyz=(0.0, 0.0, 0.0), occ=1.0, B=0.003644)],
        space_group=sg,
        crystal_system=1,
    )


def _smooth_lambert_master(npx: int) -> torch.Tensor:
    """A smooth, near-band-limited master on the modified-Lambert square.

    Built as a low-order polynomial of the direction (recovered per pixel via the
    EMsoft/kikuchipy Lambert inverse), zeroed outside the inscribed disc.
    """
    from backend.dictionary_gpu.lambert import lambert_to_direction

    m = 2 * npx + 1
    idx = torch.arange(m, dtype=torch.float64)
    rr, cc = torch.meshgrid(idx, idx, indexing="ij")
    x = (cc - npx) / npx
    y = (rr - npx) / npx
    q = x * x + y * y
    xy = torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1)
    dirs, _ = lambert_to_direction(xy)
    val = (dirs[:, 2] ** 2 + 0.4 * dirs[:, 0] + 0.3 * dirs[:, 1]).reshape(m, m)
    val = torch.where(q <= 1.0, val, torch.zeros_like(val))
    return val.to(torch.float32)


def _asym_sh_lambert_master(npx: int) -> torch.Tensor:
    """A southern master DISTINCT from :func:`_smooth_lambert_master`'s NH.

    Adds an odd-in-z-ish term so NH != SH — i.e. a synthetic non-centrosymmetric
    master whose ``NH − SH`` (equator-antisymmetric) part is genuinely non-zero.
    """
    from backend.dictionary_gpu.lambert import lambert_to_direction

    m = 2 * npx + 1
    idx = torch.arange(m, dtype=torch.float64)
    rr, cc = torch.meshgrid(idx, idx, indexing="ij")
    x = (cc - npx) / npx
    y = (rr - npx) / npx
    q = x * x + y * y
    xy = torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1)
    dirs, _ = lambert_to_direction(xy)
    # Distinct polynomial: different x/y weights + a constant offset.
    val = (dirs[:, 2] ** 2 - 0.6 * dirs[:, 0] + 0.9 * dirs[:, 1] + 0.5).reshape(m, m)
    val = torch.where(q <= 1.0, val, torch.zeros_like(val))
    return val.to(torch.float32)


def _ncc(a: torch.Tensor, b: torch.Tensor) -> float:
    av = a.flatten().to(torch.float64).cpu().numpy()
    bv = b.flatten().to(torch.float64).cpu().numpy()
    a0 = av - av.mean()
    b0 = bv - bv.mean()
    return float((a0 * b0).sum() / np.sqrt((a0 * a0).sum() * (b0 * b0).sum()))


# --- CRC algorithm parity against the real EMSphInx file ----------------------
@pytest.mark.skipif(
    not os.path.exists(NI_SHT), reason="real Ni .sht oracle not present"
)
def test_crc_matches_real_sht_trailer():
    """Our EMSphInx-specific CRC reproduces the real Ni .sht 4-byte trailer."""
    raw = open(NI_SHT, "rb").read()
    stored = struct.unpack("<I", raw[-4:])[0]
    # The C++ File::computeHash hashes the whole body (everything but the trailer)
    # contiguously; crc32c chains, so crc32c(raw[:-4]) must equal the trailer.
    assert _crc32c_emsoft(raw[:-4], 0) == stored


# --- synthetic round-trip: write -> sht_io read-back --------------------------
def test_write_read_roundtrip_metadata_and_coefs(tmp_path):
    npx = 20
    bw = 24
    nh = _smooth_lambert_master(npx)
    struct_ = _toy_structure()
    out = str(tmp_path / "toy.sht")

    ret = write_sht(
        out, nh, struct_, bandwidth=bw, energy_kV=20.0, dmin=0.05,
        name="Ni", symbol="cF4",
    )
    assert ret == out
    assert os.path.exists(out)

    # The reader (sht_io) loads it — this is the format-acceptance gate.
    m = read_sht_master(out)
    assert m.bandwidth == bw
    assert m.space_group == 225           # sgEff
    assert m.point_group == "m-3m"        # cubic, derived from sgEff
    assert m.z_rot == 4                   # SpaceGroupRot(225)
    assert m.cmp_flg == 0x7               # SpaceGroupCmp(225)
    assert abs(m.voltage_kv - 20.0) < 1e-4
    assert abs(m.primary_tilt_deg - 70.0) < 1e-4
    assert m.formula == "Ni"
    np.testing.assert_allclose(
        np.asarray(m.crystal_lattice), np.asarray(struct_.lattice),
        rtol=0, atol=1e-5,
    )
    # doubCnt is exactly NumHarm(bw, z_rot, cmp_flg).
    assert m.raw_harm.doub_cnt == _num_harm(bw, m.z_rot, m.cmp_flg)

    # The read-back coefficients equal what we wrote (FP precision) on every
    # stored position. (The cmp_flg=0x7 packing legitimately zeros the
    # symmetry-forbidden entries; we compare where the reader returns non-zero.)
    coefs_written = _forward_sht_coefs(
        _lambert_nh_to_dh_grid(nh, npx, bw, torch.device("cpu")),
        bw, torch.device("cpu"),
    )
    coefs_read = m.coefs_ml.cpu()
    stored = coefs_read.abs() > 0
    assert stored.any()
    max_diff = float((coefs_written.cpu() - coefs_read)[stored].abs().max())
    assert max_diff < 1e-6, f"stored coef mismatch {max_diff}"


def test_written_file_crc_is_self_consistent(tmp_path):
    """The 4-byte CRC our writer appends matches the body it wrote."""
    out = str(tmp_path / "crc.sht")
    write_sht(out, _smooth_lambert_master(16), _toy_structure(),
              bandwidth=20, energy_kV=20.0)
    raw = open(out, "rb").read()
    trailer = struct.unpack("<I", raw[-4:])[0]
    assert _crc32c_emsoft(raw[:-4], 0) == trailer


def test_lossless_reconstruction_matches_master(tmp_path):
    """With sg_eff=1 (no symmetry compression) the .sht is lossless: the master
    reconstructed from the read-back coefficients matches the band-limited input."""
    npx = 40
    bw = 24
    nh = _smooth_lambert_master(npx)
    dev = torch.device("cpu")
    out = str(tmp_path / "lossless.sht")
    # sg_eff=1 => z_rot=1, cmp_flg=0 => store the full complex block (lossless).
    write_sht(out, nh, _toy_structure(), bandwidth=bw, energy_kV=20.0, sg_eff=1)

    m = read_sht_master(out)
    assert m.z_rot == 1
    assert m.cmp_flg == 0x0

    dh = _lambert_nh_to_dh_grid(nh, npx, bw, dev)
    recon = inverse_sht_to_dh_grid(
        m.coefs_ml, bandwidth=bw, device=dev, max_bandwidth=bw
    )
    # Lossless container + near-band-limited smooth master => high NCC.
    assert _ncc(dh, recon) > 0.99


def test_container_fidelity_read_equals_written(tmp_path):
    """The .sht is a bit-faithful container: the read-back coefficients equal the
    symmetry-compressed coefficients we wrote, so reconstruction from the read-back
    set equals reconstruction from the same set after applying the compression.

    (We compare against the *compressed* written coefficients — masking out the
    symmetry-forbidden entries the cmp_flg packing legitimately drops — to isolate
    container fidelity from the lossy compression, which is a separate, correct
    behaviour governed by the space group.)
    """
    npx = 30
    bw = 28
    nh = _smooth_lambert_master(npx)
    dev = torch.device("cpu")
    out = str(tmp_path / "fidelity.sht")
    write_sht(out, nh, _toy_structure(), bandwidth=bw, energy_kV=20.0)

    m = read_sht_master(out)
    coefs_written = _forward_sht_coefs(
        _lambert_nh_to_dh_grid(nh, npx, bw, dev), bw, dev
    )
    # Apply the same compression the writer applies: the reader only returns the
    # symmetry-allowed (stored) positions non-zero, so mask the written coefs to
    # those same positions before comparing reconstructions.
    stored_mask = m.coefs_ml.cpu().abs() > 0
    coefs_written_masked = torch.where(
        stored_mask, coefs_written.cpu(), torch.zeros_like(coefs_written.cpu())
    )

    recon_read = inverse_sht_to_dh_grid(
        m.coefs_ml, bandwidth=bw, device=dev, max_bandwidth=bw
    )
    recon_written = inverse_sht_to_dh_grid(
        coefs_written_masked, bandwidth=bw, device=dev, max_bandwidth=bw
    )
    # Bit-faithful container: the two reconstructions are identical to FP precision.
    assert _ncc(recon_read, recon_written) > 0.99999
    assert float((recon_read - recon_written).abs().max()) < 1e-3


# --- full sphere (NH + SH) for non-centrosymmetric crystals (SP6-iter2 Gap 1) -
def test_sh_none_is_bit_identical_to_legacy_nh_only(tmp_path):
    """With ``our_master_SH=None`` the full-sphere refactor is byte-for-byte the
    same .sht as the legacy NH-only serialisation (centrosymmetric/cubic stays
    bit-identical).  We pin it two ways: the DH-grid sampler reproduces the NH-only
    grid exactly, and passing SH == NH gives the same bytes as SH = None."""
    npx = 20
    bw = 24
    dev = torch.device("cpu")
    nh = _smooth_lambert_master(npx)

    # The DH sampler with SH=None must equal sampling NH for BOTH hemispheres
    # (legacy behaviour: every DH direction sampled the NH array).
    dh_none = _lambert_nh_to_dh_grid(nh, npx, bw, dev, lambert_sh=None)
    dh_shnh = _lambert_nh_to_dh_grid(nh, npx, bw, dev, lambert_sh=nh)
    assert torch.equal(dh_none, dh_shnh), "SH==NH must reproduce the NH-only grid"

    out_none = str(tmp_path / "none.sht")
    out_shnh = str(tmp_path / "shnh.sht")
    write_sht(out_none, nh, _toy_structure(), bandwidth=bw, energy_kV=20.0)
    write_sht(out_shnh, nh, _toy_structure(), our_master_SH=nh,
              bandwidth=bw, energy_kV=20.0)
    assert open(out_none, "rb").read() == open(out_shnh, "rb").read()


def test_noncentrosymmetric_full_sphere_recovers_both_hemispheres(tmp_path):
    """For a non-centrosymmetric master (NH != SH) the .sht encodes the FULL sphere.

    EMsoft's ``SH_analyze`` transforms both hemispheres, so the odd-``l+m`` (equator-
    antisymmetric, NH − SH) harmonics carry the asymmetry.  We write a .sht from a
    distinct NH/SH pair with NO symmetry compression (``sg_eff=1``, lossless), read
    it back, reconstruct the full DH sphere, and assert it recovers BOTH the NH (top,
    z>0 rows) AND the SH (bottom, z<0 rows) — which an NH-only (mirror) .sht could
    not (its bottom rows would mirror the NH, not match SH)."""
    npx = 40
    bw = 28
    dev = torch.device("cpu")
    nh = _smooth_lambert_master(npx)
    sh = _asym_sh_lambert_master(npx)
    # Sanity: the synthetic SH genuinely differs from NH.
    assert _ncc(nh, sh) < 0.999, "test fixture SH must differ from NH"

    out = str(tmp_path / "noncentro.sht")
    write_sht(out, nh, _toy_structure(sg=1), our_master_SH=sh,
              bandwidth=bw, energy_kV=20.0, sg_eff=1)

    m = read_sht_master(out)
    assert m.z_rot == 1 and m.cmp_flg == 0x0     # lossless container

    # The DH grid that was transformed (full sphere, NH+SH).
    dh_full = _lambert_nh_to_dh_grid(nh, npx, bw, dev, lambert_sh=sh)
    # An NH-only (mirror) grid for comparison.
    dh_nh_only = _lambert_nh_to_dh_grid(nh, npx, bw, dev, lambert_sh=None)
    # The two DH grids must differ in their southern (lower) rows.
    assert not torch.equal(dh_full, dh_nh_only)

    recon = inverse_sht_to_dh_grid(
        m.coefs_ml, bandwidth=bw, device=dev, max_bandwidth=bw
    )
    # The reconstruction matches the FULL-sphere grid (both hemispheres), and is
    # CLOSER to it than to the NH-only mirror grid (the asymmetry is encoded).
    assert _ncc(recon, dh_full) > 0.99
    assert _ncc(recon, dh_full) > _ncc(recon, dh_nh_only)


def test_noncentrosymmetric_odd_lpm_coefs_nonzero(tmp_path):
    """A non-centrosymmetric NH!=SH master produces non-zero odd-``l+m`` harmonics.

    For an equator-symmetric (NH==SH) master the odd-``l+m`` coefficients vanish (the
    centrosymmetric SH symmetry).  An asymmetric master must populate them — this is
    the direct fingerprint of full-sphere (NH−SH) encoding."""
    npx = 30
    bw = 24
    dev = torch.device("cpu")
    nh = _smooth_lambert_master(npx)
    sh = _asym_sh_lambert_master(npx)

    coefs_sym = _forward_sht_coefs(
        _lambert_nh_to_dh_grid(nh, npx, bw, dev, lambert_sh=None), bw, dev
    )
    coefs_full = _forward_sht_coefs(
        _lambert_nh_to_dh_grid(nh, npx, bw, dev, lambert_sh=sh), bw, dev
    )
    # Build the odd-(l+m) mask over the stored [m, l], l >= m block.
    mm = torch.arange(bw)[:, None]
    ll = torch.arange(bw)[None, :]
    odd_lpm = ((ll + mm) % 2 == 1) & (ll >= mm)
    sym_odd_energy = float(coefs_sym[odd_lpm].abs().pow(2).sum())
    full_odd_energy = float(coefs_full[odd_lpm].abs().pow(2).sum())
    # Equator-symmetric: odd-(l+m) energy is ~0 (machine noise).
    assert sym_odd_energy < 1e-6, sym_odd_energy
    # Full sphere with NH!=SH: odd-(l+m) energy is substantial.
    assert full_odd_energy > 1e-3, full_odd_energy


def test_rejects_sh_shape_mismatch(tmp_path):
    with pytest.raises(ValueError, match="our_master_SH"):
        write_sht(str(tmp_path / "bad.sht"), _smooth_lambert_master(8),
                  _toy_structure(), our_master_SH=torch.zeros(11, 11),
                  bandwidth=16, energy_kV=20.0)


# --- shape / argument guards --------------------------------------------------
def test_rejects_even_side(tmp_path):
    with pytest.raises(ValueError):
        write_sht(str(tmp_path / "bad.sht"), torch.zeros(10, 10),
                  _toy_structure(), bandwidth=16, energy_kV=20.0)


def test_rejects_non_square(tmp_path):
    with pytest.raises(ValueError):
        write_sht(str(tmp_path / "bad.sht"), torch.zeros(11, 13),
                  _toy_structure(), bandwidth=16, energy_kV=20.0)


def test_rejects_bad_bandwidth(tmp_path):
    with pytest.raises(ValueError):
        write_sht(str(tmp_path / "bad.sht"), _smooth_lambert_master(8),
                  _toy_structure(), bandwidth=0, energy_kV=20.0)


def test_rejects_npx_mismatch(tmp_path):
    with pytest.raises(ValueError):
        write_sht(str(tmp_path / "bad.sht"), _smooth_lambert_master(8),
                  _toy_structure(), bandwidth=16, energy_kV=20.0, npx=99)


def test_rejects_bad_sg_eff(tmp_path):
    with pytest.raises(ValueError):
        write_sht(str(tmp_path / "bad.sht"), _smooth_lambert_master(8),
                  _toy_structure(), bandwidth=16, energy_kV=20.0, sg_eff=999)


# --- LEVER 4: vectorised _pack_harm is bit-identical to the reference loop -----
def _ref_pack_harm_loop(coefs_ml, bw, n, f):
    """Verbatim copy of the original (pre-Lever-4) PackHarm (m, l) double loop.

    Kept here as the bit-identity oracle: the vectorised :func:`_pack_harm` must
    reproduce this array byte-for-byte for every space group / bandwidth, so the
    .sht body / CRC never change.
    """
    inv = bool(f & 0x01)
    mir_z = bool(f & 0x02)
    mir_y = bool(f & 0x04)
    mir_x = bool(f & 0x08)
    if mir_x and mir_y:
        raise ValueError("compression flags 0x04 and 0x08 are mutually exclusive")
    out = []
    for m in range(bw):
        if n > 1 and m % n != 0:
            continue
        if mir_y:
            t = 1
        elif mir_x:
            t = 1 if (m % (n * 2) == 0) else 2
        else:
            t = 0
        row = coefs_ml[m]
        for ll in range(m, bw):
            if (inv and ll % 2 != 0) or (mir_z and (ll + m) % 2 != 0):
                continue
            v = row[ll]
            if t == 0:
                out.append(float(v.real))
                out.append(float(v.imag))
            elif t == 1:
                out.append(float(v.real))
            else:
                out.append(float(v.imag))
    return np.asarray(out, dtype="<f8")


def test_pack_harm_vectorised_is_byte_identical_to_loop():
    """The Lever-4 vectorised PackHarm matches the reference (m, l) loop byte-for-byte
    across all 230 space groups (every distinct z_rot/cmp_flg) and several bandwidths,
    including an odd bandwidth and the degenerate bw=1 — the packed-double order and
    values must be bit-identical so the .sht body and CRC do not change."""
    rng = np.random.default_rng(42)
    tested_combos = set()
    for bw in (1, 16, 17, 24, 28, 64):
        coefs = (
            rng.standard_normal((bw, bw)) + 1j * rng.standard_normal((bw, bw))
        ).astype(np.complex128)
        mm = np.arange(bw)[:, None]
        ll = np.arange(bw)[None, :]
        coefs[ll < mm] = 0  # writer zeros the systematic lower triangle
        for sg in range(1, 231):
            n = _SG_ROT[sg - 1]
            f = _SG_CMP[sg - 1]
            ref = _ref_pack_harm_loop(coefs, bw, n, f)
            vec = _pack_harm(coefs, bw, n, f)
            assert vec.shape == ref.shape, (bw, sg, n, hex(f), vec.shape, ref.shape)
            assert vec.dtype == np.dtype("<f8")
            # bit-identical: raw bytes must match exactly (not just allclose).
            assert vec.tobytes() == ref.tobytes(), f"bw={bw} sg={sg} n={n} f={hex(f)}"
            tested_combos.add((n, f))
    # Sanity: we actually exercised the distinct compression regimes, not just one.
    assert len(tested_combos) >= 20


# --- cached-Ni real-master round-trip (oracle-gated) --------------------------
@pytest.mark.skipif(
    not os.path.exists(NI_CACHE), reason="Ni oracle .h5 not present in this checkout"
)
def test_cached_ni_master_roundtrip(tmp_path):
    """Write a .sht from the REAL cached Ni master, read it back, and assert the
    container fidelity + that the reconstruction NCC equals the pure-CSHT band
    ceiling (so any roundtrip loss is band-limit truncation, not a serialiser bug)."""
    import h5py
    from backend.forward_sim.crystal.xtal_io import read_crystal_structure
    from backend.spherical_gpu._math.sht import CSHT

    struct_ = read_crystal_structure(NI_CACHE)
    with h5py.File(NI_CACHE, "r") as f:
        nh = torch.tensor(f["EMData/EBSDmaster/mLPNH"][0, 10].astype(np.float32))
    npx = 500
    bw = 64
    dev = torch.device("cpu")

    out = str(tmp_path / "Ni_ours.sht")
    write_sht(out, nh, struct_, bandwidth=bw, energy_kV=20.0, dmin=0.05,
              name="Ni", symbol="cF4")

    m = read_sht_master(out)
    assert m.bandwidth == bw
    assert m.point_group == "m-3m"
    assert m.formula == "Ni"

    dh = _lambert_nh_to_dh_grid(nh, npx, bw, dev)

    # Container fidelity: read-back coefs reconstruct identically to written coefs.
    coefs_written = _forward_sht_coefs(dh, bw, dev)
    recon_read = inverse_sht_to_dh_grid(
        m.coefs_ml, bandwidth=bw, device=dev, max_bandwidth=bw
    )
    recon_written = inverse_sht_to_dh_grid(
        coefs_written, bandwidth=bw, device=dev, max_bandwidth=bw
    )
    assert _ncc(recon_read, recon_written) > 0.999

    # The reconstruction NCC equals the pure-CSHT fsht->isht ceiling at this bw:
    # the only loss is band-limit truncation (matches the reader's own path).
    csht = CSHT(L=bw, device=dev, precision="double")
    iso = csht.isht(csht.fsht(dh.unsqueeze(0)))[0].real
    ceiling = _ncc(dh, iso)
    got = _ncc(dh, recon_read)
    # recon must not be worse than the band ceiling by more than the cubic
    # compression's tiny symmetry residual.
    assert got >= ceiling - 0.02, f"recon NCC {got} below CSHT ceiling {ceiling}"


# --- production bw=384 .sht from the real Ni master (oracle-gated) ------------
@pytest.mark.skipif(
    not os.path.exists(NI_CACHE), reason="Ni oracle .h5 not present in this checkout"
)
def test_production_bw384_ni_recon_ncc_and_emsoft_parity(tmp_path):
    """Production bw=384 .sht from the real npx=500 Ni master.

    This is the production-quality gate: at the EMSphInx production band-limit 384
    the sharp Ni Kikuchi bands ARE captured (the bw=88 band-limit loss is gone), so

      * sht_io loads it,
      * the master reconstructed from our bw=384 .sht matches the master at NCC
        > 0.97 (the bw=88 ceiling was only ~0.83), and that NCC equals the pure-CSHT
        band ceiling (so any residual is band truncation, not a serialiser bug), and
      * the HarmonicsData structure (bw, zRot, cmpFlg, doubCnt) and crystallography
        (sgEff, point group) are bit-identical to the REAL EMsoft bw=384 Ni .sht.

    write_sht is called WITHOUT an explicit bandwidth to also pin that the default
    is the production bw=384.
    """
    import h5py
    from backend.forward_sim.crystal.xtal_io import read_crystal_structure
    from backend.spherical_gpu._math.sht import CSHT

    struct_ = read_crystal_structure(NI_CACHE)
    with h5py.File(NI_CACHE, "r") as f:
        nh = torch.tensor(f["EMData/EBSDmaster/mLPNH"][0, 10].astype(np.float32))
    npx = (nh.shape[0] - 1) // 2
    assert npx >= 200, "production bw=384 needs a high-res master (npx>=200)"
    bw = 384
    dev = torch.device("cpu")

    out = str(tmp_path / "Ni_bw384.sht")
    # No explicit bandwidth => exercises the production default.
    write_sht(out, nh, struct_, energy_kV=20.0, dmin=0.05, name="Ni", symbol="cF4")

    m = read_sht_master(out)
    assert m.bandwidth == bw                 # default IS the production bw
    assert m.point_group == "m-3m"
    assert m.formula == "Ni"

    # Reconstruction NCC at bw=384 — HIGH (band sharpness recovered).
    dh = _lambert_nh_to_dh_grid(nh, npx, bw, dev)
    recon = inverse_sht_to_dh_grid(
        m.coefs_ml, bandwidth=bw, device=dev, max_bandwidth=bw
    )
    recon_ncc = _ncc(dh, recon)
    assert recon_ncc > 0.97, f"bw=384 recon NCC {recon_ncc} not high (band sharpness lost?)"

    # The recon NCC equals the pure-CSHT fsht->isht band ceiling: the only loss is
    # band-limit truncation, not a serialiser defect.
    csht = CSHT(L=bw, device=dev, precision="double")
    iso = csht.isht(csht.fsht(dh.unsqueeze(0)))[0].real
    ceiling = _ncc(dh, iso)
    assert recon_ncc >= ceiling - 0.02, f"recon {recon_ncc} below ceiling {ceiling}"

    # Structural + crystallographic parity vs the REAL EMsoft bw=384 Ni .sht.
    if os.path.exists(NI_SHT):
        rm = read_sht_master(NI_SHT)
        assert rm.bandwidth == 384            # confirm the oracle is bw=384
        assert m.bandwidth == rm.bandwidth
        assert m.z_rot == rm.z_rot            # 4 (SpaceGroupRot[225])
        assert m.cmp_flg == rm.cmp_flg        # 0x7 (SpaceGroupCmp[225])
        assert m.raw_harm.doub_cnt == rm.raw_harm.doub_cnt   # 9312
        assert m.space_group == rm.space_group  # sgEff 225
        assert m.point_group == rm.point_group  # m-3m


def test_space_group_rot_table_matches_the_crystal_classes():
    """_SG_ROT must hold 230 values, one per space group, and each must be the
    order of the z rotation of that group's point group (-4 and -42m -> 2, -6 and
    -6m2 -> 3, 23 / m-3 / -43m -> 2). Review finding 2026-09-22: the table had 231
    entries -- the orthorhombic block one 2 too long -- so every group from 75 on
    read its predecessor's value; 15 groups got a wrong z_rot, and with it the
    wrong harmonic packing (_pack_harm), among them Pm-3m (221: 2 instead of 4),
    P-43m (215: 4 instead of 2) and P23 (195: 6 instead of 2)."""
    ranges = [((1, 15), 1),                       # triclinic, monoclinic
              ((16, 74), 2),                      # orthorhombic
              ((75, 80), 4), ((81, 82), 2),       # 4, -4
              ((83, 110), 4), ((111, 122), 2),    # 4/m 422 4mm, -42m
              ((123, 142), 4),                    # 4/mmm
              ((143, 167), 3),                    # trigonal
              ((168, 173), 6), ((174, 174), 3),   # 6, -6
              ((175, 186), 6), ((187, 190), 3),   # 6/m 622 6mm, -6m2
              ((191, 194), 6),                    # 6/mmm
              ((195, 206), 2), ((207, 214), 4),   # 23 m-3, 432
              ((215, 220), 2), ((221, 230), 4)]   # -43m, m-3m
    expected = {}
    for (a, b), z in ranges:
        for sg in range(a, b + 1):
            expected[sg] = z
    assert sorted(expected) == list(range(1, 231))
    assert len(_SG_ROT) == 230, len(_SG_ROT)
    wrong = {sg: (_SG_ROT[sg - 1], z) for sg, z in expected.items() if _SG_ROT[sg - 1] != z}
    assert not wrong, wrong
