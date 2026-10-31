"""Iron has two Debye-Waller values: BCC (alpha-ferrite) and FCC (austenite).

``DWF.xlsx`` is keyed on
the element and carried only ``Fe (BCC)`` (0.3328 A^2), so an austenite CIF was
converted with the ferrite B factor. Peng et al. 1996, Table 1 (300 K) gives
Fe FCC as 0.5710 A^2 = 0.005710 nm^2, Fe BCC as 0.3328 A^2 = 0.003328 nm^2.

All expectations are hand-written from those two published numbers.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import h5py
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.routes import database as db_route  # noqa: E402
from backend.api.services.dwf_table import build_dwf_map, split_label  # noqa: E402

_REAL_DWF = (Path(__file__).resolve().parents[1]
             / "crystal-structures-for-ebsd-main" / "calculationxtal" / "DWF.xlsx")

BCC, FCC = 0.003328, 0.00571
_ROWS = [("Al", 0.007993), ("Fe (BCC)", BCC), ("Fe (FCC)", FCC)]


# --- the lookup rule --------------------------------------------------------------

def test_labels_split_into_symbol_and_structure():
    assert split_label("Fe (FCC)") == ("Fe", "FCC")
    assert split_label("Al") == ("Al", None)
    assert split_label("  Fe  (bcc) ") == ("Fe", "BCC")


def test_elemental_iron_in_an_fcc_space_group_gets_the_fcc_value():
    assert build_dwf_map(_ROWS, spg_number=225, n_elements=1)["Fe"] == FCC


def test_elemental_iron_in_a_bcc_space_group_gets_the_bcc_value():
    assert build_dwf_map(_ROWS, spg_number=229, n_elements=1)["Fe"] == BCC


def test_iron_in_a_compound_keeps_the_first_row_even_in_an_fcc_group():
    """Fe3Si is Fm-3m too, but it is not iron's own crystal."""
    assert build_dwf_map(_ROWS, spg_number=225, n_elements=2)["Fe"] == BCC


def test_without_a_space_group_the_first_row_is_used():
    assert build_dwf_map(_ROWS)["Fe"] == BCC


def test_an_element_with_one_row_is_unaffected_by_the_structure():
    for spg in (225, 229, 194, None):
        assert build_dwf_map(_ROWS, spg_number=spg, n_elements=1)["Al"] == 0.007993


def test_a_missing_variant_falls_back_to_the_first_row_not_to_a_guess():
    rows = [("Fe (BCC)", BCC)]
    assert build_dwf_map(rows, spg_number=225, n_elements=1)["Fe"] == BCC


# --- the shipped table ------------------------------------------------------------

def test_the_shipped_table_has_both_iron_rows_from_peng():
    df = pd.read_excel(_REAL_DWF)
    rows = {str(r["Element"]): r for _, r in df.iterrows()}
    assert abs(float(rows["Fe (BCC)"]["DWB 300 K"]) - 0.003328) < 1e-9
    assert abs(float(rows["Fe (FCC)"]["DWB 300 K"]) - 0.00571) < 1e-9
    assert "Peng" in str(rows["Fe (FCC)"]["Reference"])
    labels = [str(v) for v in df["Element"]]
    assert labels.index("Fe (BCC)") < labels.index("Fe (FCC)"), "BCC must stay the default (first)"


def test_the_shipped_table_still_resolves_iron_as_designed():
    df = pd.read_excel(_REAL_DWF)
    rows = list(zip(df["Element"], df["DWB 300 K"]))
    assert build_dwf_map(rows, spg_number=225, n_elements=1)["Fe"] == pytest.approx(0.00571)
    assert build_dwf_map(rows, spg_number=229, n_elements=1)["Fe"] == pytest.approx(0.003328)
    assert build_dwf_map(rows, spg_number=225, n_elements=2)["Fe"] == pytest.approx(0.003328)


# --- the converter route ----------------------------------------------------------

@pytest.fixture
def root(monkeypatch, tmp_path):
    calc = tmp_path / "crystal-structures-for-ebsd-main" / "calculationxtal"
    calc.mkdir(parents=True)
    shutil.copy(_REAL_DWF, calc / "DWF.xlsx")
    (tmp_path / "Database" / "CIF_Library").mkdir(parents=True)
    (tmp_path / "Database" / "XTAL_Library").mkdir(parents=True)
    monkeypatch.setattr(db_route, "_project_root", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def client(root):
    app = FastAPI()
    app.include_router(db_route.router, prefix="/api/database")
    return TestClient(app)


def _write_cif(root, name, spacegroup, a, species, coords):
    pytest.importorskip("pymatgen")
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.cif import CifWriter
    s = Structure.from_spacegroup(spacegroup, Lattice.cubic(a), species, coords)
    path = root / "Database" / "CIF_Library" / f"{name}.cif"
    CifWriter(s).write_file(str(path))
    # The converter's fit check wants a reference; give it a DOI (right after
    # the data_ line, before any loop).
    lines = path.read_text(encoding="utf-8").splitlines()
    at = next(i for i, ln in enumerate(lines) if ln.startswith("data_")) + 1
    lines.insert(at, "_publ_section_doi '10.1107/S0108767396005382'")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _b_factors(root, name):
    with h5py.File(root / "Database" / "XTAL_Library" / f"{name}.xtal", "r") as f:
        cd = f["CrystalData"]
        return list(cd["Atomtypes"][()]), [float(v) for v in cd["AtomData"][()][4]]


def test_converting_austenite_uses_the_fcc_value(client, root):
    cif = _write_cif(root, "gamma_Fe", "Fm-3m", 3.591, ["Fe"], [[0, 0, 0]])
    r = client.post("/api/database/convert-cif-to-xtal", json={"cif_path": str(cif)})
    assert r.status_code == 200, r.text
    z, b = _b_factors(root, "gamma_Fe")
    assert z == [26]
    assert b == [pytest.approx(0.00571, rel=1e-5)]


def test_converting_ferrite_uses_the_bcc_value(client, root):
    cif = _write_cif(root, "alpha_Fe", "Im-3m", 2.8665, ["Fe"], [[0, 0, 0]])
    r = client.post("/api/database/convert-cif-to-xtal", json={"cif_path": str(cif)})
    assert r.status_code == 200, r.text
    assert _b_factors(root, "alpha_Fe")[1] == [pytest.approx(0.003328, rel=1e-5)]


def test_converting_fe3si_keeps_the_first_row_for_its_iron(client, root):
    """BiF3-type Fe3Si, Fm-3m with two elements: iron is not in its own crystal."""
    cif = _write_cif(root, "Fe3Si", "Fm-3m", 5.65, ["Si", "Fe", "Fe"],
                     [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]])
    r = client.post("/api/database/convert-cif-to-xtal", json={"cif_path": str(cif)})
    assert r.status_code == 200, r.text
    z, b = _b_factors(root, "Fe3Si")
    fe_b = [bb for zz, bb in zip(z, b) if zz == 26]
    assert fe_b and all(v == pytest.approx(0.003328, rel=1e-5) for v in fe_b)


# --- the table editor's endpoint ---------------------------------------------------

def test_updating_the_fcc_row_by_its_label_leaves_the_bcc_row_alone(client, root):
    r = client.patch("/api/database/dwf", json={"updates": [
        {"element": "Fe (FCC)", "dwb_300k": 0.006, "reference": "test"}]})
    assert r.json() == {"updated": ["Fe (FCC)"], "added": []}
    df = pd.read_excel(root / "crystal-structures-for-ebsd-main" / "calculationxtal" / "DWF.xlsx")
    by = {str(r["Element"]): float(r["DWB 300 K"]) for _, r in df.iterrows()}
    assert by["Fe (FCC)"] == pytest.approx(0.006)
    assert by["Fe (BCC)"] == pytest.approx(0.003328)


def test_updating_the_bare_symbol_changes_only_the_first_iron_row(client, root):
    client.patch("/api/database/dwf", json={"updates": [
        {"element": "Fe", "dwb_300k": 0.004, "reference": "test"}]})
    df = pd.read_excel(root / "crystal-structures-for-ebsd-main" / "calculationxtal" / "DWF.xlsx")
    by = {str(r["Element"]): float(r["DWB 300 K"]) for _, r in df.iterrows()}
    assert by["Fe (BCC)"] == pytest.approx(0.004)
    assert by["Fe (FCC)"] == pytest.approx(0.00571)
