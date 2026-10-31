"""The CIF info panel shows the CIF's own atom sites.

The panel is headed
"Asymmetric unit" but listed what orix makes of the file. For magnesium, whose
CIF says `Mg 0.33333 0.66667 0.25000`, it showed `0.57735 0.57735 0.25` -- not
the file's fractional coordinates, not even fractional in the hexagonal frame --
and for silicon (origin choice 2, one 8a site written at 1/8,1/8,1/8) sixteen
atoms of the choice-1 expansion.

All expectations are read off the CIF text written in the tests.
"""
from __future__ import annotations

import math

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytest.importorskip("pymatgen")
pytest.importorskip("orix")

from backend.api.routes import database as db_route  # noqa: E402

_MG = """data_Mg
_publ_section_doi 'doi:10.1000/test'
_symmetry_space_group_name_H-M 'P 63/m m c'
_symmetry_Int_Tables_number 194
_cell_length_a 3.20927
_cell_length_b 3.20927
_cell_length_c 5.21033
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
loop_
_symmetry_equiv_pos_as_xyz
'x,y,z'
'-y,x-y,z'
'-x+y,-x,z'
'-x,-y,z+1/2'
'y,-x+y,z+1/2'
'x-y,x,z+1/2'
'y,x,-z'
'x-y,-y,-z'
'-x,-x+y,-z'
'-y,-x,-z+1/2'
'-x+y,y,-z+1/2'
'x,x-y,-z+1/2'
'-x,-y,-z'
'y,-x+y,-z'
'x-y,x,-z'
'x,y,-z+1/2'
'-y,x-y,-z+1/2'
'-x+y,-x,-z+1/2'
'-y,-x,z'
'-x+y,y,z'
'x,x-y,z'
'y,x,z+1/2'
'x-y,-y,z+1/2'
'-x,-x+y,z+1/2'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Mg 0.33333 0.66667 0.25000
"""

# Silicon exactly as the library's Si.cif states it: ONE 8a site, written at
# (1/8,1/8,1/8), which is origin choice 2 of Fd-3m. Operators are not listed.
_SI = """data_Si
_publ_section_doi 'doi:10.1000/test'
_symmetry_space_group_name_H-M 'F d -3 m'
_symmetry_Int_Tables_number 227
_cell_length_a 5.4309
_cell_length_b 5.4309
_cell_length_c 5.4309
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_chemical_formula_sum 'Si8'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_Wyckoff_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Si1 Si 8a 0.125 0.125 0.125 1
"""

# Displacement parameter in the coordinate loop, as U (Angstrom^2), plus one
# mixed site (two rows, one position, occupancies 0.7 + 0.3).
_U_SAME_LOOP = """data_X
_publ_section_doi 'doi:10.1000/test'
_symmetry_space_group_name_H-M 'F m -3 m'
_symmetry_Int_Tables_number 225
_cell_length_a 4.05
_cell_length_b 4.05
_cell_length_c 4.05
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_U_iso_or_equiv
_atom_site_occupancy
Al1 Al 0 0 0 0.0100 0.7
Cu1 Cu 0 0 0 0.0200 0.3
"""

# The displacement column in a SECOND loop (keyed by _atom_site_label_1) is not
# the coordinate rows' and must not be read by position.
_B_OTHER_LOOP = """data_X
_publ_section_doi 'doi:10.1000/test'
_symmetry_space_group_name_H-M 'F m -3 m'
_symmetry_Int_Tables_number 225
_cell_length_a 4.05
_cell_length_b 4.05
_cell_length_c 4.05
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Al1 Al 0 0 0 1
Cu1 Cu 0.5 0.5 0.5 1
loop_
_atom_site_label_1
_atom_site_B_iso_or_equiv
Cu1 0.9
Al1 0.1
"""


@pytest.fixture
def client(monkeypatch, tmp_path):
    (tmp_path / "Database" / "CIF_Library").mkdir(parents=True)
    (tmp_path / "Database" / "XTAL_Library").mkdir(parents=True)
    monkeypatch.setattr(db_route, "_project_root", lambda: tmp_path)
    app = FastAPI()
    app.include_router(db_route.router, prefix="/api/database")
    return TestClient(app), tmp_path / "Database" / "CIF_Library"


def _info(client, name, text):
    c, lib = client
    (lib / name).write_text(text, encoding="utf-8")
    r = c.get(f"/api/database/cif/{name}/info")
    assert r.status_code == 200, r.text
    return r.json()


def test_magnesium_shows_the_cifs_fractional_coordinates(client):
    info = _info(client, "Mg.cif", _MG)
    assert len(info["atoms"]) == 1
    a = info["atoms"][0]
    assert (a["element"], a["x"], a["y"], a["z"], a["occ"]) == ("Mg", 0.33333, 0.66667, 0.25, 1.0)
    assert info["n_sites"] == 1


def test_silicon_in_origin_choice_2_shows_its_one_8a_site_not_sixteen_atoms(client):
    info = _info(client, "Si.cif", _SI)
    assert [(a["element"], a["x"], a["y"], a["z"], a["wyckoff"]) for a in info["atoms"]] == \
        [("Si", 0.125, 0.125, 0.125, "8a")]


def test_a_displacement_in_the_coordinate_loop_is_converted_to_b(client):
    atoms = _info(client, "AlCu.cif", _U_SAME_LOOP)["atoms"]
    assert [a["element"] for a in atoms] == ["Al", "Cu"]
    assert [a["occ"] for a in atoms] == [0.7, 0.3]
    # B = 8 pi^2 U
    assert atoms[0]["dwf"] == pytest.approx(8 * math.pi ** 2 * 0.01, abs=1e-4)
    assert atoms[1]["dwf"] == pytest.approx(8 * math.pi ** 2 * 0.02, abs=1e-4)


def test_a_displacement_in_another_loop_is_not_read_by_row_position(client):
    """Cu is listed second in the coordinate loop but first in the B loop; by
    position Al would get Cu's 0.9."""
    atoms = _info(client, "AlCu_b.cif", _B_OTHER_LOOP)["atoms"]
    assert [a["dwf"] for a in atoms] == [None, None]


def test_a_cif_without_displacement_gives_none_not_zero(client):
    assert _info(client, "Mg.cif", _MG)["atoms"][0]["dwf"] is None


def test_the_batch_endpoint_the_list_cache_uses_gives_the_same_sites(client):
    c, lib = client
    (lib / "Mg.cif").write_text(_MG, encoding="utf-8")
    r = c.post("/api/database/cif/batch-info", json={"filenames": ["Mg.cif"]})
    assert r.status_code == 200, r.text
    atoms = r.json()["results"][0]["atoms"]
    assert [(a["x"], a["y"], a["z"]) for a in atoms] == [(0.33333, 0.66667, 0.25)]


# --- the panel must stay cheap and off the event loop ----------------------------------
# Choosing the block through analyse_cif ran the origin
# analysis (SpaceGroup.from_int_number, ~2.3 s per call for group 227, four calls
# per file) and the two async routes ran it inline: batch-info over the 49 library
# CIFs held the event loop (warm: 35 s before, 3.3 s after).

def test_listing_the_sites_does_not_run_the_origin_analysis(client, monkeypatch):
    from backend.forward_sim.crystal import cif_origin

    def _boom(*a, **k):
        raise AssertionError("analyse_cif is not needed to LIST a CIF's sites")

    monkeypatch.setattr(cif_origin, "analyse_cif", _boom)
    atoms = _info(client, "Si.cif", _SI)["atoms"]
    assert [(a["x"], a["y"], a["z"]) for a in atoms] == [(0.125, 0.125, 0.125)]


_OCC_CASES = """data_X
_publ_section_doi 'doi:10.1000/test'
_symmetry_space_group_name_H-M 'P 1'
_symmetry_Int_Tables_number 1
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
A1 Al 0.0 0.0 0.0 0
A2 Al 0.5 0.5 0.5 ?
A3 Al 0.25 0.25 0.25 0.4
"""


def test_a_stated_zero_occupancy_stays_zero_and_a_missing_one_is_one(client):
    """`occupancy = _number(...) or 1.0` turned a stated 0 into 1."""
    atoms = _info(client, "occ.cif", _OCC_CASES)["atoms"]
    assert [(a["label"], a["occ"]) for a in atoms] == [("A1", 0.0), ("A2", 1.0), ("A3", 0.4)]


def test_the_space_group_is_built_once_per_process():
    from backend.forward_sim.crystal import cif_origin

    assert cif_origin._space_group(194) is cif_origin._space_group(194)


@pytest.mark.parametrize("route", ["info", "batch"])
def test_the_routes_do_not_hold_the_event_loop(monkeypatch, tmp_path, route):
    """A slow parse (stubbed to sleep 0.6 s) must not stall other coroutines."""
    import asyncio
    import time

    from backend.forward_sim.crystal import cif_origin

    (tmp_path / "Database" / "CIF_Library").mkdir(parents=True)
    (tmp_path / "Database" / "XTAL_Library").mkdir(parents=True)
    (tmp_path / "Database" / "CIF_Library" / "Mg.cif").write_text(_MG, encoding="utf-8")
    monkeypatch.setattr(db_route, "_project_root", lambda: tmp_path)

    real = cif_origin.read_cif_sites

    def _slow(path):
        time.sleep(0.6)
        return real(path)

    monkeypatch.setattr(cif_origin, "read_cif_sites", _slow)

    async def scenario():
        gaps, stop = [], False

        async def ticker():
            last = time.monotonic()
            while not stop:
                await asyncio.sleep(0.02)
                now = time.monotonic()
                gaps.append(now - last)
                last = now

        task = asyncio.create_task(ticker())
        if route == "info":
            await db_route.cif_info("Mg.cif")
        else:
            await db_route.cif_batch_info(db_route.BatchInfoRequest(filenames=["Mg.cif"]))
        stop = True
        await task
        return max(gaps)

    assert asyncio.run(scenario()) < 0.4
