"""Crystal Database routes: conversion without a reference, P 1 files.

All against a library in ``tmp_path`` -- never the real ``Database/``. The
expectations are literal strings and known space-group numbers.
"""
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.routes import database as db_route  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# The routes, against a library in tmp_path (never the real Database/)
# ---------------------------------------------------------------------------

def _al_cif(extra: str = "") -> str:
    # Al, Fm-3m (225), atom at the origin; written by hand, not by the code under test.
    return f"""\
data_Al
{extra}_symmetry_space_group_name_H-M   'F m -3 m'
_symmetry_Int_Tables_number   225
_cell_length_a   4.05
_cell_length_b   4.05
_cell_length_c   4.05
_cell_angle_alpha   90
_cell_angle_beta   90
_cell_angle_gamma   90
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Al1  Al  0  0  0  1.0
"""


@pytest.fixture
def lib(monkeypatch, tmp_path):
    cif_dir = tmp_path / "Database" / "CIF_Library"
    cif_dir.mkdir(parents=True)
    dwf_dir = tmp_path / "crystal-structures-for-ebsd-main" / "calculationxtal"
    dwf_dir.mkdir(parents=True)
    (dwf_dir / "DWF.xlsx").write_bytes((REPO / "crystal-structures-for-ebsd-main" / "calculationxtal" / "DWF.xlsx").read_bytes())
    monkeypatch.setattr(db_route, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(db_route, "_get_server_db_root", lambda: None)
    app = FastAPI()
    app.include_router(db_route.router, prefix="/api/database")
    return TestClient(app), cif_dir, tmp_path


def _convert(c, name, **extra):
    return c.post("/api/database/convert-cif-to-xtal", json={"cif_path": name, **extra})


def _use_reference(tmp_path, name="Al.xtal"):
    import h5py

    with h5py.File(tmp_path / "Database" / "XTAL_Library" / name, "r") as f:
        ds = f["CrystalData/useReference"]
        value = ds[()][0]
        value = value.decode() if isinstance(value, bytes) else str(value)
        status = ds.attrs.get("orienta_reference_status")
        if isinstance(status, bytes):
            status = status.decode()
        return value, status


def test_info_reports_a_missing_reference_as_a_soft_warning(lib):
    c, cif_dir, _ = lib
    (cif_dir / "Al.cif").write_text(_al_cif())
    info = c.get("/api/database/cif/Al.cif/info").json()
    assert info["reference_missing"] is True
    assert info["fit_for_xtal"] is True            # no longer a gate
    assert info["fit_warning_codes"] == ["no_reference"]
    assert info["fit_warnings"] == ["No reference or DOI found."]   # English fallback, same order


def test_convert_without_reference_is_refused_with_a_code_and_writes_nothing(lib):
    c, cif_dir, tmp_path = lib
    (cif_dir / "Al.cif").write_text(_al_cif())
    r = _convert(c, "Al.cif")
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "reference_missing"
    assert not (tmp_path / "Database" / "XTAL_Library" / "Al.xtal").exists()


def test_convert_declared_unpublished_writes_the_declared_value_and_the_flag(lib):
    c, cif_dir, tmp_path = lib
    (cif_dir / "Al.cif").write_text(_al_cif())
    r = _convert(c, "Al.cif", reference_declaration="unpublished")
    assert r.status_code == 200, r.text
    assert r.json()["reference"] == "unpublished (declared by user)"
    assert r.json()["reference_status"] == "declared_unpublished"
    value, status = _use_reference(tmp_path)
    assert value == "unpublished (declared by user)"
    assert status == "declared_unpublished"
    # The provenance view reads the same flag back.
    info = c.get("/api/database/xtal/Al/info").json()
    assert info["reference_status"] == "declared_unpublished"
    assert info["useReference"] == "unpublished (declared by user)"


def test_a_real_reference_beats_the_declaration(lib):
    c, cif_dir, tmp_path = lib
    (cif_dir / "Al.cif").write_text(_al_cif("_publ_section_doi '10.1000/real'\n"))
    r = _convert(c, "Al.cif", reference_declaration="unpublished")
    assert r.status_code == 200, r.text
    value, status = _use_reference(tmp_path)
    assert value == "10.1000/real" and status is None
    assert c.get("/api/database/xtal/Al/info").json()["reference_status"] == "found"


def test_unquoted_doi_reaches_the_xtal(lib):
    # Before: info said "10.1000/bare", the converter wrote "none found".
    c, cif_dir, tmp_path = lib
    (cif_dir / "Al.cif").write_text(_al_cif("_publ_section_doi 10.1000/bare\n"))
    assert c.get("/api/database/cif/Al.cif/info").json()["doi"] == "10.1000/bare"
    assert _convert(c, "Al.cif").status_code == 200
    assert _use_reference(tmp_path)[0] == "10.1000/bare"


def test_adding_a_doi_through_the_meta_route_unblocks_the_conversion(lib):
    c, cif_dir, tmp_path = lib
    (cif_dir / "Al.cif").write_text(_al_cif())
    assert _convert(c, "Al.cif").status_code == 422
    r = c.patch("/api/database/cif/Al.cif/meta", json={"doi": "10.5555/entered"})
    assert r.status_code == 200
    info = c.get("/api/database/cif/Al.cif/info").json()
    assert info["doi"] == "10.5555/entered" and info["reference_missing"] is False
    assert _convert(c, "Al.cif").status_code == 200
    assert _use_reference(tmp_path)[0] == "10.5555/entered"


def test_meta_route_accepts_a_citation_with_backslashes(lib):
    c, cif_dir, _ = lib
    (cif_dir / "Al.cif").write_text(_al_cif())
    r = c.patch("/api/database/cif/Al.cif/meta", json={"reference": "J. Phys. \\1 (2020)"})
    assert r.status_code == 200, r.text
    assert c.get("/api/database/cif/Al.cif/info").json()["reference"] == "J. Phys. \\1 (2020)"


def test_unknown_declaration_value_is_a_400(lib):
    c, cif_dir, _ = lib
    (cif_dir / "Al.cif").write_text(_al_cif())
    assert _convert(c, "Al.cif", reference_declaration="whatever").status_code == 400


def test_no_atoms_is_still_a_hard_422_even_when_declared_unpublished(lib):
    c, cif_dir, tmp_path = lib
    (cif_dir / "Empty.cif").write_text(
        "data_Empty\n_symmetry_space_group_name_H-M 'F m -3 m'\n_symmetry_Int_Tables_number 225\n"
        "_cell_length_a 4.05\n_cell_length_b 4.05\n_cell_length_c 4.05\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
    )
    r = _convert(c, "Empty.cif", reference_declaration="unpublished")
    assert r.status_code == 422
    assert "No atomic sites" in r.json()["detail"]
    assert not (tmp_path / "Database" / "XTAL_Library" / "Empty.xtal").exists()


def test_batch_info_flags_missing_references(lib):
    c, cif_dir, _ = lib
    (cif_dir / "Al.cif").write_text(_al_cif())
    (cif_dir / "AlDoi.cif").write_text(_al_cif("_publ_section_doi 10.1000/x\n"))
    res = c.post("/api/database/cif/batch-info", json={"filenames": ["Al.cif", "AlDoi.cif"]}).json()["results"]
    by = {r["filename"]: r for r in res}
    assert by["Al.cif"]["reference_missing"] is True
    assert by["AlDoi.cif"]["reference_missing"] is False
    assert by["AlDoi.cif"]["doi"] == "10.1000/x"


# ---------------------------------------------------------------------------
# P 1 files with hidden symmetry
# ---------------------------------------------------------------------------

def _noisy_p1_al_text() -> str:
    """Conventional fcc Al (Fm-3m, 225) written as P 1, every coordinate moved
    by at most 0.0004 fractional units (~0.0016 A)."""
    return """\
data_Al_p1
_symmetry_space_group_name_H-M   'P 1'
_symmetry_Int_Tables_number   1
_cell_length_a   4.05
_cell_length_b   4.05
_cell_length_c   4.05
_cell_angle_alpha   90
_cell_angle_beta   90
_cell_angle_gamma   90
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Al1  Al  0.0004  -0.0003  0.0002  1.0
  Al2  Al  0.5003   0.4997  0.0002  1.0
  Al3  Al  0.4997   0.0003  0.5002  1.0
  Al4  Al  0.0003   0.5002  0.4998  1.0
"""


def test_info_names_the_detected_group_of_a_p1_file(lib):
    c, cif_dir, _ = lib
    (cif_dir / "AlP1.cif").write_text(_noisy_p1_al_text())
    info = c.get("/api/database/cif/AlP1.cif/info").json()
    assert info["space_group_number"] == 1
    assert info["detected_space_group"]["number"] == 225
    assert info["detected_space_group"]["symbol"] == "Fm-3m"
    assert "p1_hidden_symmetry" in info["fit_warning_codes"]
    # The code list and the English fallbacks stay parallel.
    assert len(info["fit_warning_codes"]) == len(info["fit_warnings"])
    assert any("Fm-3m" in w for w in info["fit_warnings"])


def test_a_file_with_real_symmetry_gets_no_p1_finding(lib):
    c, cif_dir, _ = lib
    (cif_dir / "Al.cif").write_text(_al_cif())
    info = c.get("/api/database/cif/Al.cif/info").json()
    assert "detected_space_group" not in info
    assert "p1_hidden_symmetry" not in info["fit_warning_codes"]


def test_import_reports_hidden_symmetry(lib, tmp_path):
    c, _, _ = lib
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "AlP1.cif").write_text(_noisy_p1_al_text())
    (incoming / "Al.cif").write_text(_al_cif())
    r = c.post("/api/database/add-cif", json={"file_path": str(incoming / "AlP1.cif")}).json()
    assert r["detected_space_group"]["number"] == 225
    assert [w["code"] for w in r["warnings"]] == ["p1_hidden_symmetry"]
    ok = c.post("/api/database/add-cif", json={"file_path": str(incoming / "Al.cif")}).json()
    assert "warnings" not in ok and "detected_space_group" not in ok


def test_symmetrise_route_writes_a_new_file_and_never_the_original(lib):
    import hashlib

    c, cif_dir, _ = lib
    src = cif_dir / "AlP1.cif"
    src.write_text(_noisy_p1_al_text())
    before = hashlib.sha256(src.read_bytes()).hexdigest()

    r = c.post("/api/database/cif/AlP1.cif/symmetrise")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "AlP1_sym.cif" and body["space_group_number"] == 225
    assert hashlib.sha256(src.read_bytes()).hexdigest() == before

    copy_info = c.get("/api/database/cif/AlP1_sym.cif/info").json()
    assert copy_info["space_group_number"] == 225
    assert "detected_space_group" not in copy_info



def test_symmetrise_route_refusals(lib):
    c, cif_dir, _ = lib
    (cif_dir / "Al.cif").write_text(_al_cif())
    (cif_dir / "Mixed.cif").write_text(
        _noisy_p1_al_text().replace("Al1  Al  0.0004  -0.0003  0.0002  1.0",
                                    "Al1  Al  0.0004  -0.0003  0.0002  0.5")
    )
    r = c.post("/api/database/cif/Al.cif/symmetrise")
    assert r.status_code == 422 and r.json()["detail"]["code"] == "not_p1"
    r = c.post("/api/database/cif/Mixed.cif/symmetrise")
    assert r.status_code == 422 and r.json()["detail"]["code"] == "disordered"
    assert c.post("/api/database/cif/Missing.cif/symmetrise").status_code == 404
    assert sorted(p.name for p in cif_dir.iterdir()) == ["Al.cif", "Mixed.cif"]


# ---------------------------------------------------------------------------
# A DOI is only taken from a value that is one
# ---------------------------------------------------------------------------

def test_a_citation_loop_is_read_by_column_by_the_converter(lib):
    # `primary` is the value of the _citation_id column: it was once written to
    # useReference as the DOI. The DOI column's first row is the DOI.
    c, cif_dir, tmp_path = lib
    (cif_dir / "Al.cif").write_text(
        _al_cif("loop_\n_citation_id\n_citation_year\n_citation_doi\n"
                "primary 1999 10.1000/loopvalue\nsecondary 2005 10.1000/second\n")
    )
    info = c.get("/api/database/cif/Al.cif/info").json()
    assert info["doi"] == "10.1000/loopvalue" and info["reference_missing"] is False
    assert _convert(c, "Al.cif").status_code == 200
    assert _use_reference(tmp_path)[0] == "10.1000/loopvalue"


def test_a_loop_without_a_doi_value_still_asks_for_a_reference(lib):
    c, cif_dir, tmp_path = lib
    (cif_dir / "Al.cif").write_text(_al_cif("loop_\n_citation_id\n_citation_doi\nprimary n/a\n"))
    assert c.get("/api/database/cif/Al.cif/info").json()["doi"] == ""
    r = _convert(c, "Al.cif")
    assert r.status_code == 422 and r.json()["detail"]["code"] == "reference_missing"


def test_a_doi_link_in_the_cif_reaches_the_xtal_as_the_bare_doi(lib):
    c, cif_dir, tmp_path = lib
    (cif_dir / "Al.cif").write_text(_al_cif("_publ_section_doi 'https://doi.org/10.1000/linked'\n"))
    assert _convert(c, "Al.cif").status_code == 200
    assert _use_reference(tmp_path)[0] == "10.1000/linked"


def test_meta_route_refuses_something_that_is_not_a_doi(lib):
    c, cif_dir, _ = lib
    text = _al_cif()
    (cif_dir / "Al.cif").write_text(text)
    r = c.patch("/api/database/cif/Al.cif/meta", json={"doi": "n/a"})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "invalid_doi"
    assert (cif_dir / "Al.cif").read_text() == text
    ok = c.patch("/api/database/cif/Al.cif/meta", json={"doi": "https://doi.org/10.1000/ok"})
    assert ok.status_code == 200
    assert c.get("/api/database/cif/Al.cif/info").json()["doi"] == "10.1000/ok"


def test_editing_a_doi_on_a_file_that_only_has_citation_doi_changes_what_is_read(lib):
    c, cif_dir, _ = lib
    (cif_dir / "Al.cif").write_text(_al_cif("_citation_doi 10.1000/old\n"))
    assert c.patch("/api/database/cif/Al.cif/meta", json={"doi": "10.1000/new"}).status_code == 200
    assert c.get("/api/database/cif/Al.cif/info").json()["doi"] == "10.1000/new"
    assert (cif_dir / "Al.cif").read_text().count("10.1000/old") == 0


def test_info_reports_a_one_atom_p1_cell_with_hidden_symmetry(lib):
    # Primitive fcc Al (a = 4.05 A conventional), one atom, written as P 1.
    c, cif_dir, _ = lib
    (cif_dir / "AlPrim.cif").write_text("""data_AlPrim
_symmetry_space_group_name_H-M   'P 1'
_symmetry_Int_Tables_number   1
_cell_length_a   2.86378246
_cell_length_b   2.86378246
_cell_length_c   2.86378246
_cell_angle_alpha   60
_cell_angle_beta   60
_cell_angle_gamma   60
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Al1  Al  0  0  0  1.0
""")
    info = c.get("/api/database/cif/AlPrim.cif/info").json()
    assert info["space_group_number"] == 1
    assert info["detected_space_group"]["number"] == 225


def test_symmetrise_in_a_read_only_folder_is_a_422_without_the_path(lib, monkeypatch):
    import builtins

    c, cif_dir, tmp_path = lib
    (cif_dir / "AlP1.cif").write_text(_noisy_p1_al_text())
    real = builtins.open

    def fake_open(file, mode="r", *a, **kw):
        if "x" in str(mode) and "_sym" in Path(str(file)).name:
            raise PermissionError(13, "Permission denied", str(file))
        return real(file, mode, *a, **kw)

    monkeypatch.setattr(builtins, "open", fake_open)
    r = c.post("/api/database/cif/AlP1.cif/symmetrise")
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "not_writable"
    assert str(tmp_path) not in r.text and "Users" not in r.text


def test_an_unexpected_failure_while_symmetrising_does_not_leak_the_path(lib, monkeypatch):
    from backend.api.services import cif_symmetry

    c, cif_dir, tmp_path = lib
    (cif_dir / "AlP1.cif").write_text(_noisy_p1_al_text())

    def boom(_path):
        raise RuntimeError(f"cannot open {tmp_path / 'secret-user-folder' / 'x.cif'}")

    monkeypatch.setattr(cif_symmetry, "save_symmetrised_copy", boom)
    r = c.post("/api/database/cif/AlP1.cif/symmetrise")
    assert r.status_code == 500
    assert "secret-user-folder" not in r.text and str(tmp_path) not in r.text


def test_symmetrise_twice_the_second_answer_names_the_existing_copy(lib):
    c, cif_dir, _ = lib
    (cif_dir / "AlP1.cif").write_text(_noisy_p1_al_text())
    assert c.post("/api/database/cif/AlP1.cif/symmetrise").json()["name"] == "AlP1_sym.cif"
    r = c.post("/api/database/cif/AlP1.cif/symmetrise")
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "same_content_exists"
    assert r.json()["detail"]["existing"] == "AlP1_sym.cif"
    assert sorted(p.name for p in cif_dir.iterdir()) == ["AlP1.cif", "AlP1_sym.cif"]


def test_importing_an_ordinary_file_never_starts_the_symmetry_analysis(lib, tmp_path, monkeypatch):
    # Si.cif took 4.5 s here before the text pre-check: every import paid for an
    # analysis that only a P 1 file needs. The call is RECORDED, not made to
    # raise: _p1_finding swallows exceptions, so a raising stub proves nothing.
    from backend.api.services import cif_symmetry

    c, _, _ = lib
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "Al.cif").write_text(_al_cif())
    (incoming / "AlP1.cif").write_text(_noisy_p1_al_text())
    calls = []
    real = cif_symmetry.analyse_p1

    def recording(path, *a, **kw):
        calls.append(Path(path).name)
        return real(path, *a, **kw)

    monkeypatch.setattr(cif_symmetry, "analyse_p1", recording)
    r = c.post("/api/database/add-cif", json={"file_path": str(incoming / "Al.cif")}).json()
    assert r["success"] is True and "warnings" not in r
    assert calls == []                                   # a file that declares Fm-3m is never analysed
    c.post("/api/database/add-cif", json={"file_path": str(incoming / "AlP1.cif")})
    assert calls == ["AlP1.cif"]                         # a P 1 file is


def test_the_analysis_runs_off_the_event_loop(lib, tmp_path, monkeypatch):
    # While add-cif and symmetrise wait for a slow analysis, another request must
    # be answered at once; with a synchronous body it waited the full second.
    import asyncio
    import time

    import httpx
    from backend.api.services import cif_symmetry

    c, cif_dir, _ = lib
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "AlP1.cif").write_text(_noisy_p1_al_text())
    (cif_dir / "AlP1b.cif").write_text(_noisy_p1_al_text())
    real = cif_symmetry.analyse_p1

    def slow(path, *a, **kw):
        time.sleep(1.0)
        return real(path, *a, **kw)

    monkeypatch.setattr(cif_symmetry, "analyse_p1", slow)

    async def scenario(url, payload):
        transport = httpx.ASGITransport(app=c.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            slow_call = asyncio.create_task(client.post(url, json=payload) if payload else client.post(url))
            # Time the sleep too: a handler that blocks the loop also holds back
            # the timer, so the measure starts before it, not after.
            t0 = time.perf_counter()
            await asyncio.sleep(0.2)                     # the slow analysis is running now
            quick = await client.get("/api/database/cache/status")
            quick_seconds = time.perf_counter() - t0
            slow_response = await slow_call
            return quick.status_code, quick_seconds, slow_response.status_code

    for url, payload in (
        ("/api/database/add-cif", {"file_path": str(incoming / "AlP1.cif")}),
        ("/api/database/cif/AlP1b.cif/symmetrise", None),
    ):
        status, quick_seconds, slow_status = asyncio.run(scenario(url, payload))
        assert status == 200 and slow_status == 200, url
        assert quick_seconds < 0.7, f"{url}: the event loop was blocked for {quick_seconds:.2f} s"


def test_meta_route_refuses_a_reference_that_is_too_long(lib):
    c, cif_dir, _ = lib
    text = _al_cif()
    (cif_dir / "Al.cif").write_text(text)
    r = c.patch("/api/database/cif/Al.cif/meta", json={"reference": "y" * 2001})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "too_long"
    assert (cif_dir / "Al.cif").read_text() == text
    assert c.patch("/api/database/cif/Al.cif/meta", json={"reference": "y" * 2000}).status_code == 200


def test_a_published_structure_with_journal_fields_is_never_offered_as_unpublished(lib):
    # Raw COD / ICSD style: authors, journal, year, volume and pages as fields, no DOI
    # and no _publ_section_references.
    c, cif_dir, tmp_path = lib
    (cif_dir / "Al.cif").write_text(_al_cif(
        "loop_\n_publ_author_name\n'Panday P.K.'\n'Schubert K.'\n"
        "_journal_name_full 'Journal of the Less-Common Metals'\n"
        "_journal_page_first 395\n_journal_page_last 401\n_journal_volume 18\n_journal_year 1969\n"
    ))
    info = c.get("/api/database/cif/Al.cif/info").json()
    assert info["reference_missing"] is False and info["doi"] == ""
    expected = "Panday P.K., Schubert K.: Journal of the Less-Common Metals 18 (1969) 395-401"
    assert info["reference"] == expected
    assert "no_reference" not in info["fit_warning_codes"]
    r = _convert(c, "Al.cif")                                   # no 422, no dialog
    assert r.status_code == 200, r.text
    assert r.json()["reference_status"] == "found"
    assert _use_reference(tmp_path) == (expected, None)
    batch = c.post("/api/database/cif/batch-info", json={"filenames": ["Al.cif"]}).json()["results"][0]
    assert batch["reference_missing"] is False


# ---------------------------------------------------------------------------
# The ladder through the routes
# ---------------------------------------------------------------------------

def _ambiguous_al_text() -> str:
    """fcc Al (a = 4.05) with +-0.008 A of noise (seed 1), as a P 1 CIF. Its group is
    Cm (8) at 0.01 and 0.02 A and Fm-3m (225) at 0.05 and 0.1 A."""
    import numpy as np
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.cif import CifWriter

    al = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.05), ["Al"], [[0, 0, 0]])
    cart = al.cart_coords + np.random.default_rng(1).uniform(-0.008, 0.008, al.cart_coords.shape)
    return str(CifWriter(Structure(al.lattice, al.species, al.lattice.get_fractional_coords(cart))))


def test_info_shows_the_ladder_and_offers_no_single_group(lib):
    c, cif_dir, _ = lib
    (cif_dir / "AlNoisy.cif").write_text(_ambiguous_al_text())
    info = c.get("/api/database/cif/AlNoisy.cif/info").json()
    assert info["p1_ambiguous"] is True
    assert "detected_space_group" not in info
    assert [(r["symprec"], r["number"]) for r in info["symmetry_ladder"]] == [(0.01, 8), (0.02, 8), (0.05, 225), (0.1, 225)]
    assert "p1_ambiguous_symmetry" in info["fit_warning_codes"]
    assert "p1_hidden_symmetry" not in info["fit_warning_codes"]
    assert any("0.01 A: Cm (8)" in w for w in info["fit_warnings"])


def test_a_stable_file_carries_its_ladder_too(lib):
    c, cif_dir, _ = lib
    (cif_dir / "AlP1.cif").write_text(_noisy_p1_al_text())
    info = c.get("/api/database/cif/AlP1.cif/info").json()
    assert info["detected_space_group"]["number"] == 225
    assert [r["number"] for r in info["symmetry_ladder"]] == [225, 225, 225, 225]
    assert "p1_ambiguous" not in info


def test_import_reports_an_ambiguous_ladder(lib, tmp_path):
    c, _, _ = lib
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "AlNoisy.cif").write_text(_ambiguous_al_text())
    r = c.post("/api/database/add-cif", json={"file_path": str(incoming / "AlNoisy.cif")}).json()
    assert r["p1_ambiguous"] is True and "detected_space_group" not in r
    assert [w["code"] for w in r["warnings"]] == ["p1_ambiguous_symmetry"]


def test_symmetrise_asks_for_a_tolerance_and_then_writes_the_chosen_one(lib):
    c, cif_dir, _ = lib
    (cif_dir / "AlNoisy.cif").write_text(_ambiguous_al_text())
    r = c.post("/api/database/cif/AlNoisy.cif/symmetrise")
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["code"] == "ambiguous_tolerance"
    assert [(x["symprec"], x["number"]) for x in detail["ladder"]] == [(0.01, 8), (0.02, 8), (0.05, 225), (0.1, 225)]
    assert sorted(p.name for p in cif_dir.iterdir()) == ["AlNoisy.cif"]

    chosen = c.post("/api/database/cif/AlNoisy.cif/symmetrise", json={"symprec": 0.05})
    assert chosen.status_code == 200, chosen.text
    body = chosen.json()
    assert (body["space_group_number"], body["symprec"], body["tolerance_chosen"]) == (225, 0.05, True)

    bad = c.post("/api/database/cif/AlNoisy.cif/symmetrise", json={"symprec": 0.03})
    assert bad.status_code == 422 and bad.json()["detail"]["code"] == "invalid_tolerance"


# ---------------------------------------------------------------------------
# A header that contradicts itself
# ---------------------------------------------------------------------------

_CONTRADICTORY_AL = (
    "data_Al\n_cell_length_a 4.05\n_cell_length_b 4.05\n_cell_length_c 4.05\n"
    "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
    "_symmetry_space_group_name_H-M 'F m -3 m'\n_symmetry_Int_Tables_number 1\n"
    "loop_\n_symmetry_equiv_pos_as_xyz\n'x, y, z'\n"
    "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
    "_atom_site_fract_y\n_atom_site_fract_z\nAl1 Al 0 0 0\n"
)


def test_info_names_a_contradictory_header_and_offers_no_copy(lib):
    c, cif_dir, _ = lib
    (cif_dir / "AlHeader.cif").write_text(_CONTRADICTORY_AL, encoding="utf-8")
    info = c.get("/api/database/cif/AlHeader.cif/info").json()
    assert "contradictory_header" in info["fit_warning_codes"]
    assert info["header_contradiction"] == {"number": 1, "symbol": "F m -3 m"}
    assert "detected_space_group" not in info and "p1_ambiguous" not in info
    assert any("space group number 1" in w and "F m -3 m" in w for w in info["fit_warnings"])


def test_import_reports_a_contradictory_header(lib, tmp_path):
    c, _, _ = lib
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "AlHeader.cif").write_text(_CONTRADICTORY_AL, encoding="utf-8")
    r = c.post("/api/database/add-cif", json={"file_path": str(incoming / "AlHeader.cif")}).json()
    assert r["header_contradiction"] == {"number": 1, "symbol": "F m -3 m"}
    assert [w["code"] for w in r["warnings"]] == ["contradictory_header"]


def test_symmetrise_refuses_a_contradictory_header_with_its_code(lib):
    c, cif_dir, _ = lib
    (cif_dir / "AlHeader.cif").write_text(_CONTRADICTORY_AL, encoding="utf-8")
    r = c.post("/api/database/cif/AlHeader.cif/symmetrise")
    assert r.status_code == 422 and r.json()["detail"]["code"] == "contradictory_header"
    assert sorted(p.name for p in cif_dir.iterdir()) == ["AlHeader.cif"]


def test_info_names_the_opposite_contradiction_too(lib):
    c, cif_dir, _ = lib
    text = _CONTRADICTORY_AL.replace("'F m -3 m'", "'P 1'").replace("_symmetry_Int_Tables_number 1", "_symmetry_Int_Tables_number 225")
    (cif_dir / "AlOpposite.cif").write_text(text, encoding="utf-8")
    info = c.get("/api/database/cif/AlOpposite.cif/info").json()
    assert "contradictory_header" in info["fit_warning_codes"]
    assert info["header_contradiction"] == {"number": 225, "symbol": "P 1"}
    assert "detected_space_group" not in info


# ---------------------------------------------------------------------------
# A check that could not be completed is said, not swallowed
# ---------------------------------------------------------------------------

def test_info_reports_a_check_that_failed_at_every_tolerance(lib, monkeypatch):
    from backend.api.services import cif_symmetry

    c, cif_dir, _ = lib
    (cif_dir / "AlP1.cif").write_text(_noisy_p1_al_text())

    def broken(structure, symprec=cif_symmetry.SYMPREC):
        raise RuntimeError("spglib got no cell")

    monkeypatch.setattr(cif_symmetry, "spacegroup_analyzer", broken)
    info = c.get("/api/database/cif/AlP1.cif/info").json()
    assert "p1_check_failed" in info["fit_warning_codes"]
    assert info["p1_check_error"] == "RuntimeError: spglib got no cell"
    assert any("could not be completed" in w and "spglib got no cell" in w for w in info["fit_warnings"])
    assert "detected_space_group" not in info


def test_an_exception_in_the_check_is_logged_as_a_warning_and_reported(lib, monkeypatch, caplog):
    import logging

    from backend.api.services import cif_symmetry

    c, cif_dir, _ = lib
    (cif_dir / "AlP1.cif").write_text(_noisy_p1_al_text())

    def explode(*a, **k):
        raise ValueError("unexpected")

    monkeypatch.setattr(cif_symmetry, "analyse_p1", explode)
    with caplog.at_level(logging.WARNING, logger="backend.api.routes.database"):
        info = c.get("/api/database/cif/AlP1.cif/info").json()
    assert info["p1_check_error"] == "ValueError: unexpected"
    assert "p1_check_failed" in info["fit_warning_codes"]
    assert [r.levelno for r in caplog.records if "P1 symmetry check failed" in r.getMessage()] == [logging.WARNING]


def test_import_reports_a_failed_check(lib, tmp_path, monkeypatch):
    from backend.api.services import cif_symmetry

    c, _, _ = lib
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "AlP1.cif").write_text(_noisy_p1_al_text())
    monkeypatch.setattr(cif_symmetry, "analyse_p1", lambda *a, **k: (_ for _ in ()).throw(ValueError("unexpected")))
    r = c.post("/api/database/add-cif", json={"file_path": str(incoming / "AlP1.cif")}).json()
    assert [w["code"] for w in r["warnings"]] == ["p1_check_failed"]
    assert r["p1_check_error"] == "ValueError: unexpected"
