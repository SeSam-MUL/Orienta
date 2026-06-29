"""Read EMsoft .sht binary master pattern files.

Format reference: SHTfile spec (William C. Lenthe, 2019), pulled by EMSphInx
via CMake FetchContent. The authoritative byte-level spec is in:
``https://github.com/EMsoft-org/SHTfile/blob/master/sht_file.in.hpp``

This is a self-contained binary format — NOT HDF5. See
``tasks/sht-format-research.md`` for the byte-level documentation, plus
``tasks/_research_sht/sht_file.in.hpp`` for the C++ reference cached locally.

Layout summary (verified against ``Al (Al) [cF4] {20kV}.sht``)::

    Offset  Size  Block                       Content
       0    40    FileHeader                  magic '*sht', ver 1.1, modality, keV, doi/note lens
            +N    doi  + notes (each pad8)    UTF-8, padded to 8-byte multiples
            8     MasterPatternData           numXtal, sgEff, pijk, rotSense, ...
       per crystal:
            72    CrystalData                 sgNum, lattice (6 f4), quat (4 f4), strLens
            32×N  AtomData                    x,y,z (24ths), occ, charge, debWal, Z
            5×pad8 strings                    formula/name/symbol/refs/note
       per crystal:
            simMetaSize  SimulationData       88-byte EMsoftED block for EBSD
            8     HarmonicsData header        bw (i2), zRot (i1), cmpFlg (i1), doubCnt (i4)
            doubCnt × 8  packed real alm      LE float64, symmetry-compressed
            4     CRC-32C trailer             poly 0x1edc6f41 (Castagnoli)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from ..exceptions import SHTReadError


# Map crystallographic space group number (1-230) -> orix point-group name.
# EMSphInx reports the EFFECTIVE space group ("sgEff" in MasterPatternData),
# which encodes the underlying point group. The mapping is unambiguous: each
# of the 230 space groups belongs to exactly one of 32 point groups, grouped
# by crystal system per the International Tables for Crystallography.
def _build_sg_to_pg_table() -> dict[int, str]:
    """Build the full SG -> point-group lookup. Static, cheap to call once."""
    table: dict[int, str] = {}
    # Triclinic
    table[1] = "1"
    table[2] = "-1"
    # Monoclinic
    for sg in range(3, 6):    table[sg] = "2"      # P2, P2_1, C2
    for sg in range(6, 10):   table[sg] = "m"      # Pm, Pc, Cm, Cc
    for sg in range(10, 16):  table[sg] = "2/m"    # P2/m..C2/c
    # Orthorhombic
    for sg in range(16, 25):  table[sg] = "222"
    for sg in range(25, 47):  table[sg] = "mm2"
    for sg in range(47, 75):  table[sg] = "mmm"    # includes Cmcm (63), Pnma (62)
    # Tetragonal
    for sg in range(75, 81):  table[sg] = "4"
    for sg in range(81, 83):  table[sg] = "-4"
    for sg in range(83, 89):  table[sg] = "4/m"
    for sg in range(89, 99):  table[sg] = "422"
    for sg in range(99, 111): table[sg] = "4mm"
    for sg in range(111, 123): table[sg] = "-42m"
    for sg in range(123, 143): table[sg] = "4/mmm" # includes I4/mmm (139, 140)
    # Trigonal
    for sg in range(143, 147): table[sg] = "3"
    for sg in range(147, 149): table[sg] = "-3"
    for sg in range(149, 156): table[sg] = "32"
    for sg in range(156, 162): table[sg] = "3m"
    for sg in range(162, 168): table[sg] = "-3m"   # includes R-3m (166), R-3c (167)
    # Hexagonal
    for sg in range(168, 174): table[sg] = "6"
    table[174] = "-6"
    for sg in range(175, 177): table[sg] = "6/m"
    for sg in range(177, 183): table[sg] = "622"
    for sg in range(183, 187): table[sg] = "6mm"
    for sg in range(187, 191): table[sg] = "-6m2"
    for sg in range(191, 195): table[sg] = "6/mmm" # includes P6_3/mmc (194)
    # Cubic
    for sg in range(195, 200): table[sg] = "23"
    for sg in range(200, 207): table[sg] = "m-3"
    for sg in range(207, 215): table[sg] = "432"
    for sg in range(215, 221): table[sg] = "-43m"  # includes F-43m (216)
    for sg in range(221, 231): table[sg] = "m-3m"  # includes Fm-3m (225), Im-3m (229), Fd-3m (227)
    assert len(table) == 230, f"SG table incomplete: {len(table)} entries"
    return table


_SG_TO_POINT_GROUP: dict[int, str] = _build_sg_to_pg_table()


def _pad8(n: int) -> int:
    return n if (n % 8 == 0) else n + (8 - (n % 8))


@dataclass
class _FileHeader:
    magic: bytes
    version: Tuple[int, int]
    software_version: bytes
    modality: int
    beam_energy_kv: float
    primary_angle: float
    secondary_angle: float
    doi: str
    notes: str


@dataclass
class _AtomData:
    x: float
    y: float
    z: float
    occ: float
    charge: float
    deb_wal: float
    z_atom: int


@dataclass
class _CrystalData:
    sg_num: int
    sg_set: int
    sg_axis: int
    sg_cell: int
    origin: Tuple[float, float, float]
    lattice: Tuple[float, ...]   # (a, b, c, alpha, beta, gamma)
    rotation: Tuple[float, ...]  # quaternion (w, x, y, z)
    weight: float
    atoms: List[_AtomData]
    formula: str
    name: str
    symbol: str
    refs: str
    note: str


@dataclass
class _HarmonicsHeader:
    bw: int
    z_rot: int
    cmp_flg: int
    doub_cnt: int


@dataclass
class SHTMasterFile:
    """Parsed contents of an EMsoft .sht binary master pattern file.

    The harmonic coefficients are returned as a dense ``(bw, bw)`` complex
    matrix indexed as ``coefs[m, l]`` (NumPy convention; matches EMSphInx's
    internal ``alm[m * bw + l]`` layout). Lower-triangle ``l < m`` is zero;
    rows for forbidden ``m % z_rot != 0`` are zero; entries that violate
    the symmetry compression flags are zero.
    """
    bandwidth: int
    z_rot: int
    cmp_flg: int
    coefs_ml: torch.Tensor                 # complex, shape (bw, bw), indexed [m, l]
    point_group: str
    space_group: int
    voltage_kv: float
    primary_tilt_deg: float
    crystal_lattice: Tuple[float, ...]     # (a, b, c, alpha, beta, gamma) in nm/deg
    formula: str
    source_path: str
    raw_header: _FileHeader = field(repr=False)
    raw_crystal: _CrystalData = field(repr=False)
    raw_harm: _HarmonicsHeader = field(repr=False)
    # EMsoftED SimulationData provenance (decoded from the 88-byte block; None when
    # the file carries no simulation block, simMetaSize == 0).
    sim_dmin: Optional[float] = None
    sim_npx: Optional[int] = None
    sim_numsx: Optional[int] = None
    sim_totnum_el: Optional[int] = None
    sim_bethe: Optional[Tuple[float, float, float, float]] = None


def _num_harm(bw: int, n: int, f: int) -> int:
    """Direct port of HarmonicsData::NumHarm (sht_file.in.hpp:1672)."""
    inv = bool(f & 0x01)
    mir_z = bool(f & 0x02)
    mir_y = bool(f & 0x04)
    mir_x = bool(f & 0x08)
    if mir_x and mir_y:
        raise SHTReadError(
            "Invalid cmpFlg: 0x04 (mirY) and 0x08 (mirX) are mutually exclusive"
        )
    num = 0
    for m in range(bw):
        if n > 1 and m % n != 0:
            continue
        for ll in range(m, bw):
            if inv and ll % 2 != 0:
                continue
            if mir_z and (ll + m) % 2 != 0:
                continue
            num += 1
    if not (mir_x or mir_y):
        num *= 2
    return num


def _unpack_harm(packed: np.ndarray, bw: int, n: int, f: int) -> np.ndarray:
    """Direct port of HarmonicsData::UnpackHarm (sht_file.in.hpp:1766).

    Returns dense ``(bw, bw)`` complex128 array indexed ``[m, l]``.
    """
    inv = bool(f & 0x01)
    mir_z = bool(f & 0x02)
    mir_y = bool(f & 0x04)
    mir_x = bool(f & 0x08)
    if mir_x and mir_y:
        raise SHTReadError(
            "Invalid cmpFlg: 0x04 (mirY) and 0x08 (mirX) are mutually exclusive"
        )
    out = np.zeros((bw, bw), dtype=np.complex128)
    it = iter(packed.tolist())
    try:
        for m in range(bw):
            if n > 1 and m % n != 0:
                continue
            if mir_y:
                t = 1
            elif mir_x:
                t = 1 if (m % (n * 2) == 0) else 2
            else:
                t = 0
            for ll in range(m, bw):
                if (inv and ll % 2 != 0) or (mir_z and (ll + m) % 2 != 0):
                    continue
                if t == 0:
                    re = next(it)
                    im = next(it)
                    out[m, ll] = complex(re, im)
                elif t == 1:
                    out[m, ll] = complex(next(it), 0.0)
                else:
                    out[m, ll] = complex(0.0, next(it))
    except StopIteration as exc:
        raise SHTReadError(
            f"Coefficient buffer exhausted at m={m}, l={ll} — packed array "
            "shorter than NumHarm() expects."
        ) from exc
    return out


def read_sht_master(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> SHTMasterFile:
    """Read an EMsoft .sht binary master pattern file.

    Parameters
    ----------
    path : str or Path
    device : torch device
        Where to place the harmonic coefficient tensor.

    Returns
    -------
    SHTMasterFile

    Raises
    ------
    SHTReadError
        On magic mismatch, unsupported version, truncated file, unsupported
        simMetaSize, or coefficient-block / NumHarm mismatch.
    """
    p = Path(path)
    if not p.is_file():
        raise SHTReadError(f"SHT file not found: {p}")

    raw = p.read_bytes()
    if len(raw) < 44:
        raise SHTReadError(f"SHT file too short ({len(raw)} bytes): {p}")

    # FileHeader
    magic = raw[0:4]
    if magic == b"*sht":
        endian = "<"
    elif magic == b"*SHT":
        endian = ">"
    else:
        raise SHTReadError(f"Bad magic in {p}: {magic!r}")

    try:
        ver_major, ver_minor = struct.unpack_from(f"{endian}bb", raw, 4)
        if (ver_major, ver_minor) != (1, 1):
            raise SHTReadError(
                f"Unsupported SHT version {ver_major}.{ver_minor} (only 1.1 known)"
            )
        sw_ver = raw[8:16]
        modality = raw[16]
        beam_energy, primary_angle, secondary_angle, _reserved = struct.unpack_from(
            f"{endian}ffff", raw, 20
        )
        doi_len, note_len = struct.unpack_from(f"{endian}hh", raw, 36)

        off = 40
        doi_bytes = raw[off : off + _pad8(doi_len)]
        off += _pad8(doi_len)
        notes_bytes = raw[off : off + _pad8(note_len)]
        off += _pad8(note_len)
        header = _FileHeader(
            magic=magic, version=(ver_major, ver_minor),
            software_version=sw_ver, modality=modality,
            beam_energy_kv=beam_energy, primary_angle=primary_angle,
            secondary_angle=secondary_angle,
            doi=doi_bytes.rstrip(b"\x00").decode("utf-8", errors="replace"),
            notes=notes_bytes.rstrip(b"\x00").decode("utf-8", errors="replace"),
        )

        # MasterPatternData
        num_xtal = struct.unpack_from(f"{endian}b", raw, off)[0]
        sg_eff = raw[off + 1]
        # pijk = struct.unpack_from(f"{endian}b", raw, off + 2)[0]  # unused here
        # rot_sense = struct.unpack_from(f"{endian}b", raw, off + 3)[0]
        # modality_mp = raw[off + 4]
        # vendor = raw[off + 5]
        sim_meta_size = struct.unpack_from(f"{endian}h", raw, off + 6)[0]
        off += 8

        crystals: List[_CrystalData] = []
        for _ in range(num_xtal):
            x_off = off
            sg_num = raw[x_off]
            sg_set, sg_axis, sg_cell = struct.unpack_from(
                f"{endian}bbb", raw, x_off + 1
            )
            ori = struct.unpack_from(f"{endian}fff", raw, x_off + 4)
            lat = struct.unpack_from(f"{endian}6f", raw, x_off + 16)
            rot = struct.unpack_from(f"{endian}4f", raw, x_off + 40)
            weight = struct.unpack_from(f"{endian}f", raw, x_off + 56)[0]
            n_atoms, f_len, m_len, s_len, r_len, nt_len = struct.unpack_from(
                f"{endian}6h", raw, x_off + 60
            )
            off = x_off + 72

            atoms: List[_AtomData] = []
            for _i in range(n_atoms):
                ax, ay, az, occ, charge, deb_wal, _resfp = struct.unpack_from(
                    f"{endian}7f", raw, off
                )
                z_atom = struct.unpack_from(f"{endian}b", raw, off + 28)[0]
                atoms.append(_AtomData(ax, ay, az, occ, charge, deb_wal, z_atom))
                off += 32

            def _read_str(n: int) -> str:
                nonlocal off
                padded = _pad8(n)
                s = raw[off : off + padded]
                off += padded
                return s.rstrip(b"\x00").decode("utf-8", errors="replace")

            crystals.append(
                _CrystalData(
                    sg_num=sg_num, sg_set=sg_set, sg_axis=sg_axis, sg_cell=sg_cell,
                    origin=ori, lattice=lat, rotation=rot, weight=weight,
                    atoms=atoms,
                    formula=_read_str(f_len), name=_read_str(m_len),
                    symbol=_read_str(s_len), refs=_read_str(r_len),
                    note=_read_str(nt_len),
                )
            )

        # SimulationData (EMsoftED, 88 bytes/crystal). Layout: the writer's
        # backend.forward_sim.io.sht_writer._build_emsoft_ed (offsets relative to
        # the block start: totNumEl(i8)@+48, numSx(i2)@+56, Bethe 4×f4@+60,
        # dMin(f4)@+76, numPx(i2)@+80). Decoded from the FIRST crystal only.
        sim_dmin = sim_npx = sim_numsx = sim_totnum_el = None
        sim_bethe = None
        for _ci in range(num_xtal):
            if sim_meta_size == 0:
                continue
            if sim_meta_size != 88:
                raise SHTReadError(
                    f"Unsupported simMetaSize={sim_meta_size}; only 88 (EMsoftED) "
                    "is implemented."
                )
            if _ci == 0:  # first crystal only
                sim_totnum_el = struct.unpack_from(f"{endian}q", raw, off + 48)[0]
                sim_numsx = struct.unpack_from(f"{endian}h", raw, off + 56)[0]
                c1, c2, c3, sgdb = struct.unpack_from(f"{endian}4f", raw, off + 60)
                sim_bethe = (c1, c2, c3, sgdb)
                sim_dmin = struct.unpack_from(f"{endian}f", raw, off + 76)[0]
                sim_npx = struct.unpack_from(f"{endian}h", raw, off + 80)[0]
            off += sim_meta_size

        # HarmonicsData
        if off + 8 > len(raw):
            raise SHTReadError("File truncated before HarmonicsData header")
        bw = struct.unpack_from(f"{endian}h", raw, off)[0]
        z_rot = struct.unpack_from(f"{endian}b", raw, off + 2)[0]
        cmp_flg = struct.unpack_from(f"{endian}b", raw, off + 3)[0]
        doub_cnt = struct.unpack_from(f"{endian}i", raw, off + 4)[0]
        off += 8

        if bw <= 0 or bw > 2048:
            raise SHTReadError(f"Implausible bandwidth bw={bw}")

        expected = _num_harm(bw, z_rot, cmp_flg)
        if expected != doub_cnt:
            raise SHTReadError(
                f"NumHarm({bw}, {z_rot}, {cmp_flg}) = {expected} but file claims "
                f"doubCnt={doub_cnt}"
            )

        coefs_bytes_needed = doub_cnt * 8
        if off + coefs_bytes_needed + 4 > len(raw):
            raise SHTReadError(
                f"File truncated: need {coefs_bytes_needed} bytes for coefs + 4 "
                f"for CRC, have {len(raw) - off}"
            )

        dt = np.dtype("<f8") if endian == "<" else np.dtype(">f8")
        alm_packed = np.frombuffer(raw, dtype=dt, count=doub_cnt, offset=off).copy()
        off += coefs_bytes_needed

        # CRC trailer (parsed for completeness; we don't enforce the CRC since
        # crc32c is an extra dependency and the structural checks above are
        # already very strong).
        _crc_trailer = struct.unpack_from(f"{endian}I", raw, off)[0]
        if off + 4 != len(raw):
            raise SHTReadError(
                f"Trailing bytes after CRC ({len(raw) - off - 4} extra)"
            )
    except struct.error as e:
        raise SHTReadError(f"Failed to parse SHT structure in {p}: {e}") from e

    # Decompress harmonic coefficients into (bw, bw) complex
    coefs_dense = _unpack_harm(alm_packed, bw, z_rot, cmp_flg)
    coefs_t = torch.from_numpy(coefs_dense).to(device)

    # Map space group to crystallographic point group (orix convention)
    point_group = _SG_TO_POINT_GROUP.get(int(sg_eff))
    if point_group is None:
        raise SHTReadError(
            f"Effective space group {sg_eff} not in _SG_TO_POINT_GROUP table. "
            "Add an entry mapping it to the orix point-group name."
        )

    return SHTMasterFile(
        bandwidth=int(bw),
        z_rot=int(z_rot),
        cmp_flg=int(cmp_flg),
        coefs_ml=coefs_t,
        point_group=point_group,
        space_group=int(sg_eff),
        voltage_kv=float(beam_energy),
        primary_tilt_deg=float(primary_angle),
        crystal_lattice=tuple(crystals[0].lattice),
        formula=crystals[0].formula,
        source_path=str(p),
        raw_header=header,
        raw_crystal=crystals[0],
        raw_harm=_HarmonicsHeader(
            bw=bw, z_rot=z_rot, cmp_flg=cmp_flg, doub_cnt=doub_cnt
        ),
        sim_dmin=sim_dmin, sim_npx=sim_npx, sim_numsx=sim_numsx,
        sim_totnum_el=sim_totnum_el, sim_bethe=sim_bethe,
    )
