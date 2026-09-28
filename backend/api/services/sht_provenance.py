"""Assemble the provenance + parameters document for one .sht (File Info panel).

Three sources, each used for what it can actually know:

* the **sidecar** ``<name>.sht.provenance.json`` -- which files this master was
  made from, its literature reference and the simulation parameters. Cheap.
* the **.sht binary** -- crystallography, and the parameters when no sidecar
  exists. 300-600 KB, read on demand.
* **this machine's filesystem** -- whether those files are here. Never the
  sidecar: see :func:`_file_entry`.

Nothing is invented — unknown fields are null.
"""
from __future__ import annotations
import glob as _glob
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath

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
    # Project-relative, because this file is written INTO `Database/` and the
    # library is copied and mailed between machines (§2.5.3). An absolute path
    # here would put the simulating account's name in a file that travels, and
    # it is worth nothing to the recipient: their copy is somewhere else.
    doc = {
        "schema": 1,
        "source_xtal": {"name": xtal_path.name, "path": _public_path(xtal_path),
                        "found": xtal_path.exists()},
        "source_cif": {"name": cif.name, "path": _public_path(cif),
                       "found": cif.exists()},
        "reference": reference,
        "parameters": dict(params),
    }
    out = sht_path.with_suffix(".sht.provenance.json")
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return out


def _resolve_in_library(name: str, search_dir) -> Path | None:
    """Find `name` in THIS machine's library, or None.

    A sidecar records an ABSOLUTE path, and that path belongs to whichever
    machine ran the simulation. Measured over the shipped library: 25 of the
    recorded paths name the `E:` drive and do not resolve on a `C:` checkout,
    while every recorded FILE NAME does resolve locally. So the name is the
    portable part and the path is not, which is why resolution goes through the
    name.

    `glob.escape` because library names really do contain glob metacharacters
    (`Al (Al) [cF4] {20kV}` for masters, and a CIF may be `Mn2(AlSi)5_…`); an
    unescaped `[cF4]` is a character class and would match nothing.
    """
    if not name:
        return None
    base = Path(search_dir)
    direct = base / name
    if direct.is_file():
        return direct
    try:
        for cand in base.rglob(_glob.escape(name)):
            if cand.is_file():
                return cand
    except OSError:
        pass
    return None


def _is_absolute_anywhere(s) -> bool:
    """Absolute in EITHER flavour: `C:\\…`, `\\\\server\\…` or `/…`.

    One question asked of both flavours, because the answer decides whether a
    string is sanitised at all, and the same string gets opposite answers on
    Windows and on Linux.
    """
    s = str(s)
    return PureWindowsPath(s).is_absolute() or PurePosixPath(s).is_absolute()


def _basename_anyflavour(s) -> str:
    """The last component, splitting on BOTH separators.

    `Path(s).name` is flavour-bound: on Linux a Windows path has no separators
    at all, so `.name` hands back the entire string -- account name included.
    That is how a local path reached a manuscript in 2026-09-24.
    """
    last = re.split(r"[\\/]", str(s))[-1].strip()
    return last or str(s)


def _public_path(p) -> str:
    """A path fit to leave the machine: project-relative, POSIX, or the bare name.

    Every string this module hands to an endpoint passes through here. An
    absolute path carries the account name of whoever ran the simulation --
    `C:\\Users\\<name>\\...` -- and these answers are read by the phase library
    and the Database browser, pasted into bug reports, and quoted in writing.
    This project has already had a local path reach a manuscript once (the
    `Path.stem` finding of 2026-09-24), which is why the rule is "never emit
    one" rather than "strip it where someone noticed".

    The name alone is the fallback, not a guess at a relative path: a file
    outside the project (a library on another drive) has no relative form, and
    inventing one would point at something that is not there.

    An ALREADY RELATIVE path is passed through (normalised to POSIX). Sidecars
    written since this change record one, and reducing those to a bare name
    would have thrown away the very information the field exists for -- the
    first version of this function did exactly that.

    "Absolute" is decided in BOTH path flavours, and the basename is taken on
    both separators, because neither `Path.is_absolute` nor `Path.name` means the
    same thing on two platforms and the library is mailed between them. Measured
    on this Windows host before the fix:

        /Users/other/…/NotHere.xtal  ->  returned VERBATIM, elsewhere=False

    because `PureWindowsPath("/Users/…").is_absolute()` is False (no drive), so a
    POSIX path took the "already relative" branch. The mirror is worse, because
    it is the one that matters today: every recorded path in this tree's sidecars
    is an absolute Windows path, `PurePosixPath(r"C:\\Users\\…").is_absolute()` is
    False, so on the .dmg and the AppImage they would all pass through with the
    account name in them. This is the `Path(p).stem` finding of 2026-09-24 again
    -- same function family, same reason, and `Path.name` carries it too:
    `PosixPath(r"C:\\Users\\x\\Al.xtal").name` is the WHOLE string.
    """
    s = str(p)
    if not _is_absolute_anywhere(s):
        return s.replace("\\", "/")
    try:
        from backend.api.services.crystal_hint_local_library import PROJECT_ROOT
        return str(Path(s).relative_to(PROJECT_ROOT)).replace("\\", "/")
    except Exception:
        return _basename_anyflavour(s)


def _is_outside_project(p) -> bool:
    """Was this recorded on a machine whose layout is not ours?

    Deliberately NOT "the sanitised string differs from the original": an
    absolute path pointing INSIDE this project also differs (it becomes
    relative), and calling that "elsewhere" would flag ordinary local files.

    Both flavours, for the same reason as `_public_path`: asking only the host's
    flavour made a foreign-platform path answer False -- "this is local" about
    the one kind of path that certainly is not.
    """
    s = str(p)
    if not _is_absolute_anywhere(s):
        return False
    try:
        from backend.api.services.crystal_hint_local_library import PROJECT_ROOT
        Path(s).relative_to(PROJECT_ROOT)
        return False
    except Exception:
        return True


def _file_entry(name: str, search_dir, recorded_path: str | None = None) -> dict:
    """One `source_xtal`/`source_cif` block, with `found` measured HERE.

    `found` is never copied from a sidecar. The sidecar is provenance -- which
    file this master was made from -- and a claim about a filesystem is not
    provenance: it was true on the machine that wrote it, at that moment. The
    consumer treats `found` as "you can open this" (the File Info panel prints
    the name only when `found` is true), so an unverified flag is a promise the
    app cannot keep.

    `path` is PROJECT-RELATIVE when the file is here, and None when it is not.
    It used to be the absolute local path, on 29 of 36 phases. No CLIENT reads
    it -- the File Info panel and the phase library both print `name` -- so it
    was a machine path travelling to the browser for no one.

    One caller did read it, and it was this file: the reference recovery below
    opened `xtal_entry["path"]`. Changing the field silently emptied every
    recovered citation until a test caught it. A value for the client and a
    path for this process are two different things; they are no longer the
    same variable.

    `recorded_path` is the harder half, and it is a TRADE-OFF, not a cleanup.
    It exists so a reader can see what the sidecar claimed when we cannot find
    the file -- and an unresolvable recorded path is, by definition, a path on
    somebody else's machine, which is the one string that must not be repeated.
    Reduced to its bare name it says nothing `name` did not already say.

    So the string is sanitised and the SIGNAL is kept separately:
    `recorded_elsewhere` is true when what the sidecar recorded was an absolute
    path from outside this project. That is the part a reader actually needed --
    "this master was built against a library that is not this one" -- and it
    survives without carrying an account name into a bug report.
    """
    local = _resolve_in_library(name, search_dir)
    entry: dict = {"name": name or None,
                   "path": _public_path(local) if local else None,
                   "found": local is not None}
    if local is None and recorded_path:
        entry["recorded_path"] = _public_path(recorded_path)
        entry["recorded_elsewhere"] = _is_outside_project(recorded_path)
    return entry


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
    """The File Info document for one .sht.

    Was sidecar-OR-binary; is now sidecar-AND-binary, with existence measured
    locally. Three defects that produced, all measured on the shipped library
    (2026-09-27):

    * ``found`` and ``path`` were copied out of the sidecar unverified, so the
      panel promised files using a flag from another machine and a path on a
      drive this one does not have (25 of the recorded paths). It happened to
      be right about presence today; a synthetic sidecar naming two files that
      exist nowhere still came back ``found: true``.
    * ``crystallography`` was read from the sidecar, which never carries it, so
      it was ``{}`` for exactly the 13 of 29 masters that HAVE a sidecar while
      the 16 without got it from the binary. The split was an artefact of which
      source answered, not of the files.
    * a malformed sidecar fell through to "recovery" and lost its parameters
      as well, although the two are independent.

    The binary read is the one cost this adds: 300-600 KB per call, on a route
    that serves one selected file. Anything that wants many at once should ask
    for what it needs rather than calling this per card.
    """
    sht_path = Path(sht_path)
    sidecar_doc: dict = {}
    sidecar = sht_path.with_suffix(".sht.provenance.json")
    if sidecar.exists():
        try:
            loaded = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                sidecar_doc = loaded
        except Exception:
            sidecar_doc = {}  # unreadable: recover, but do not pretend

    def _recorded(key: str) -> tuple[str, str | None]:
        """(name, recorded_path) the sidecar states for `key`, if any."""
        v = sidecar_doc.get(key)
        if isinstance(v, dict):
            return (str(v.get("name") or ""), v.get("path"))
        return ("", None)

    # The .xtal/CIF NAMES come from the sidecar when it has them (it knows what
    # the simulation actually read, including names no convention would guess --
    # `α-(AlMnSi).xtal` is not derivable from its material folder). Existence is
    # then measured here, never taken from the sidecar.
    x_name, x_recorded = _recorded("source_xtal")
    c_name, c_recorded = _recorded("source_cif")

    stem = _stem_from_sht(sht_path)
    if not x_name:
        x_name = f"{stem}.xtal"
    if not c_name:
        linked = find_linked_cif(sht_path, Path(cif_dir))
        c_name = linked.name if linked else ""

    xtal_entry = _file_entry(x_name, xtal_dir, x_recorded)
    cif_entry = _file_entry(c_name, cif_dir, c_recorded)

    reference = str(sidecar_doc.get("reference") or "").strip()
    if not reference and xtal_entry["found"]:
        # Controller decision #1: the provenance "reference" is the .xtal's
        # useReference DOI/citation, NOT the (empty) phase_name from the xtal.
        #
        # RESOLVED AGAIN, not read back out of `xtal_entry["path"]`: that field
        # is what we SEND (project-relative, or a bare name for a file outside
        # the project) and it is not openable from an arbitrary working
        # directory. Opening it returned '' for every recovered reference -- a
        # citation silently missing, which is the failure this project cares
        # about most. A field for the client and a path for this process are two
        # different things and must not be the same variable.
        local_xtal = _resolve_in_library(x_name, xtal_dir)
        reference = read_xtal_reference(local_xtal) if local_xtal else ""

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

    # Parameters: the sidecar when it has them (it records what the run was
    # ASKED for), the binary otherwise (what the file ended up carrying).
    side_params = sidecar_doc.get("parameters")
    if isinstance(side_params, dict) and side_params:
        params = {**side_params, "source": "sidecar"}
        engine = side_params.get("engine") or sidecar_doc.get("engine") or engine

    return {
        "filename": sht_path.name,
        "provenance": {
            "source_xtal": xtal_entry,
            "source_cif": cif_entry,
            "reference": reference,
            "engine": engine,
            # What the PROVENANCE came from. Existence is always local and
            # crystallography always the binary, so this no longer implies
            # which source answered for those.
            "origin": "sidecar" if sidecar_doc else "recovered",
        },
        "crystallography": crys,
        "parameters": params,
    }
