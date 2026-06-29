"""Assemble the provenance + parameters document for one .sht (File Info panel).

Sidecar-first; falls back to recovering from the .sht binary + .xtal
(useReference) + linked CIF. Nothing is invented — unknown fields are null.
"""
from __future__ import annotations
import json
from pathlib import Path

from phase_metadata import find_linked_cif, extract_metadata_from_xtal, _SHT_REGEX

# The .sht writer's DEFAULT doi (the EMsoft/EMSphInx software paper) — NOT a
# crystal-source citation. Never surface it as the structure's reference.
_EMSOFT_SOFTWARE_DOI = "https://doi.org/10.1016/j.ultramic.2019.112841"


def _stem_from_sht(sht_path: Path) -> str:
    m = _SHT_REGEX.match(sht_path.stem)
    if m and (m.group("phase") or "").strip():
        return m.group("phase").strip()
    return sht_path.stem


def read_xtal_reference(xtal_path) -> str:
    """Read /CrystalData/useReference (DOI/citation) from a .xtal. '' on failure."""
    try:
        import h5py  # noqa: PLC0415
        with h5py.File(str(xtal_path), "r") as f:
            v = f["/CrystalData/useReference"][()]
    except Exception:
        return ""
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace").strip()
    if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
        if len(v) == 0:
            return ""
        v = v[0]
        if isinstance(v, bytes):
            return v.decode("utf-8", "replace").strip()
    return str(v).strip()


def write_provenance_sidecar(sht_path, *, xtal_path, cif_dir, params) -> Path:
    """Write ``<sht>.sht.provenance.json`` next to a .sht (source xtal/cif + params).

    Shared by BOTH engines — the GPU "Ours" runner and the EMsoft controller —
    so provenance coverage is uniform. ``params`` should include ``engine``
    (``"ours"``/``"emsoft"``) so a file can be attributed without re-reading the
    binary. Best-effort by contract: callers wrap this in try/except and never
    fail a simulation on a sidecar error.
    """
    sht_path = Path(sht_path)
    xtal_path = Path(xtal_path)
    cif = Path(cif_dir) / f"{xtal_path.stem}.cif"
    try:
        reference = read_xtal_reference(xtal_path) or ""
    except Exception:
        reference = ""
    doc = {
        "schema": 1,
        "source_xtal": {"name": xtal_path.name, "path": str(xtal_path),
                        "found": xtal_path.exists()},
        "source_cif": {"name": cif.name, "path": str(cif), "found": cif.exists()},
        "reference": reference,
        "parameters": dict(params),
    }
    out = sht_path.with_suffix(".sht.provenance.json")
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return out


def _engine_from_software_version(sw) -> str:
    """Map a .sht FileHeader ``software`` field to the producing engine.

    Our GPU forward-sim writer stamps ``b"fwd_sim0"``; a real EMsoft binary
    stamps its own version string. Returns ``"ours" | "emsoft" | "unknown"``.
    """
    if sw is None:
        return "unknown"
    if isinstance(sw, bytes):
        sw = sw.decode("ascii", "replace")
    sw = str(sw).replace("\x00", "").strip()
    if not sw:
        return "unknown"
    return "ours" if "fwd_sim" in sw.lower() else "emsoft"


def build_sht_info(sht_path: Path, *, xtal_dir: Path, cif_dir: Path) -> dict:
    sht_path = Path(sht_path)
    sidecar = sht_path.with_suffix(".sht.provenance.json")
    if sidecar.exists():
        try:
            doc = json.loads(sidecar.read_text(encoding="utf-8"))
            params = doc.get("parameters", {})
            return {
                "filename": sht_path.name,
                "provenance": {
                    "source_xtal": doc.get("source_xtal"),
                    "source_cif": doc.get("source_cif"),
                    "reference": doc.get("reference", ""),
                    "engine": params.get("engine") or doc.get("engine") or "unknown",
                    "origin": "sidecar",
                },
                "crystallography": doc.get("crystallography", {}),
                "parameters": {**params, "source": "sidecar"},
            }
        except Exception:
            pass  # fall through to recovery

    stem = _stem_from_sht(sht_path)
    xtal = Path(xtal_dir) / f"{stem}.xtal"
    cif = find_linked_cif(sht_path, Path(cif_dir))
    reference = ""
    if xtal.exists():
        # Controller decision #1: the provenance "reference" is the .xtal's
        # useReference DOI/citation, NOT the (empty) phase_name from the xtal.
        reference = read_xtal_reference(xtal)

    crys, params = {}, {"source": "unknown"}
    engine = "unknown"
    try:
        from backend.spherical_gpu.pipeline.sht_io import read_sht_master  # noqa: PLC0415
        s = read_sht_master(str(sht_path), device="cpu")
        engine = _engine_from_software_version(
            getattr(s, "software_version", None)
            or getattr(getattr(s, "raw_header", None), "software_version", None)
        )
        crys = {
            "formula": s.formula, "space_group": s.space_group,
            "point_group": s.point_group, "voltage_kV": s.voltage_kv,
            "tilt_deg": s.primary_tilt_deg, "bandwidth": s.bandwidth,
            "lattice": list(s.crystal_lattice),
        }
        if s.sim_dmin is not None:
            params = {"dmin": s.sim_dmin, "npx": s.sim_npx, "numsx": s.sim_numsx,
                      "totnum_el": s.sim_totnum_el, "bethe": list(s.sim_bethe or []),
                      "source": "sht_binary"}
        # The literature citation is embedded in the .sht binary header (real
        # EMsoft files always carry it; "Ours" files now bake in the .xtal's
        # useReference too). Recover it when the .xtal link is broken/renamed/
        # missing — but never surface the writer's software-paper default DOI.
        if not reference:
            doi = (getattr(getattr(s, "raw_header", None), "doi", "") or "").strip()
            if doi and doi != _EMSOFT_SOFTWARE_DOI:
                reference = doi
    except Exception:
        pass  # honest unknown

    return {
        "filename": sht_path.name,
        "provenance": {
            "source_xtal": {"name": xtal.name, "path": str(xtal), "found": xtal.exists()},
            "source_cif": {"name": cif.name if cif else None,
                           "path": str(cif) if cif else None, "found": cif is not None},
            "reference": reference,
            "engine": engine,
            "origin": "recovered",
        },
        "crystallography": crys,
        "parameters": params,
    }
