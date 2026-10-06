"""Adding a phase to the Indexing page by file path.

The Indexing page used to offer only the phases found in the library. When the
app runs in a plain browser (``start_app.py``) there is no native file dialog,
so a phase outside the library could not be added at all. The page now sends
the typed or pasted path to ``POST /api/indexing/files/from-path``, which has
to say clearly what is wrong with a path (missing, not a file, wrong
extension, not readable as a phase) and, when it is fine, return the same kind
of record the library listing returns.
"""
import struct

import pytest

from backend.api.services.phase_path import PhasePathError, inspect_phase_path

AL_CIF = """data_Al
_cell_length_a 4.0495
_cell_length_b 4.0495
_cell_length_c 4.0495
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'F m -3 m'
_symmetry_Int_Tables_number 225
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Al1 Al 0 0 0
"""

NO_ATOMS_CIF = AL_CIF.split("loop_")[0]


@pytest.fixture
def al_cif(tmp_path):
    p = tmp_path / "MyAl.cif"
    p.write_text(AL_CIF, encoding="utf-8")
    return p


def _code(method, path):
    with pytest.raises(PhasePathError) as ei:
        inspect_phase_path(method, str(path))
    return ei.value


# ---------------------------------------------------------------- Hough / CIF

def test_a_good_cif_returns_a_library_shaped_record(al_cif):
    rec = inspect_phase_path("hough", str(al_cif))
    assert rec["path"] == str(al_cif)
    assert rec["filename"] == "MyAl.cif"
    assert rec["file_type"] == "cif"
    assert rec["user_added"] is True
    assert rec["in_library"] is False
    # the fields the Indexing page reads off a library record
    for key in ("display_label", "formula", "space_group", "crystal_system",
                "element_group", "elements", "lattice_a", "space_group_number"):
        assert key in rec, key
    assert "Al" in rec["elements"]
    assert rec["space_group_number"] == 225


@pytest.mark.parametrize("wrap", ['"{}"', "'{}'", "  {}  ", ' "{}" '])
def test_quotes_and_spaces_pasted_around_a_path_are_ignored(al_cif, wrap):
    """Windows' "Copy as path" puts the path in double quotes."""
    rec = inspect_phase_path("hough", wrap.format(al_cif))
    assert rec["path"] == str(al_cif)


def test_empty_path(tmp_path):
    for raw in ("", "   ", '""'):
        err = _code("hough", raw)
        assert err.code == "empty"


def test_a_relative_path_is_refused_not_guessed(al_cif):
    """The backend's working directory is not something the user can see."""
    err = _code("hough", "MyAl.cif")
    assert err.code == "not_absolute"


def test_missing_file(tmp_path):
    err = _code("hough", tmp_path / "nope.cif")
    assert err.code == "not_found"
    assert str(tmp_path / "nope.cif") in err.params["path"]


def test_a_folder_is_not_a_phase_file(tmp_path):
    err = _code("hough", tmp_path)
    assert err.code == "not_a_file"


@pytest.mark.parametrize("method,name,expected", [
    ("hough", "Al.sht", ".cif"),
    ("hough", "Al.txt", ".cif"),
    ("spherical", "Al.cif", ".sht"),
    ("dictionary", "Al.cif", ".h5"),
])
def test_wrong_extension_names_what_the_method_wants(tmp_path, method, name, expected):
    p = tmp_path / name
    p.write_bytes(b"x")
    err = _code(method, p)
    assert err.code == "wrong_extension"
    assert err.params["expected"] == expected
    assert err.params["method"] == method


def test_text_that_is_not_a_cif(tmp_path):
    p = tmp_path / "junk.cif"
    p.write_text("this is not a cif file", encoding="utf-8")
    assert _code("hough", p).code == "unreadable"


def test_a_cif_without_atoms_is_not_a_phase(tmp_path):
    p = tmp_path / "empty_cell.cif"
    p.write_text(NO_ATOMS_CIF, encoding="utf-8")
    assert _code("hough", p).code == "unreadable"


def test_unknown_method(al_cif):
    err = _code("nope", al_cif)
    assert err.code == "unknown_method"


# ----------------------------------------------------------------- Spherical

def _sht_header(magic=b"*sht", ver=(1, 1), total=64):
    head = magic + struct.pack("bb", *ver) + b"\x00" * (total - 6)
    return head


def test_a_sht_file_is_accepted_on_its_header(tmp_path):
    p = tmp_path / "Al (Al) [cF4] {20kV}.sht"
    p.write_bytes(_sht_header())
    rec = inspect_phase_path("spherical", str(p))
    assert rec["file_type"] == "sht"
    assert rec["user_added"] is True
    assert rec["formula"] == "Al"          # from the file name convention


@pytest.mark.parametrize("payload", [
    _sht_header(magic=b"abcd"),
    _sht_header(ver=(2, 0)),
    b"*sht",                                  # truncated
    b"",
])
def test_a_file_that_is_not_an_sht(tmp_path, payload):
    p = tmp_path / "fake.sht"
    p.write_bytes(payload)
    assert _code("spherical", p).code == "unreadable"


# ---------------------------------------------------------------- Dictionary

def test_a_master_pattern_h5_is_accepted(tmp_path):
    import h5py
    import numpy as np
    p = tmp_path / "Al_master_E20kV_npx500.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("EMData/EBSDmaster/mLPNH", data=np.zeros((1, 1, 3, 3)))
    rec = inspect_phase_path("dictionary", str(p))
    assert rec["file_type"] == "master"
    assert rec["user_added"] is True


def test_an_h5_that_is_not_a_master_pattern(tmp_path):
    import h5py
    p = tmp_path / "other.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("something", data=[1, 2, 3])
    assert _code("dictionary", p).code == "not_a_phase"


def test_a_file_named_h5_that_is_not_hdf5(tmp_path):
    p = tmp_path / "fake.h5"
    p.write_text("hello")
    assert _code("dictionary", p).code == "unreadable"


# ------------------------------------------------------------------ the route

@pytest.fixture
def client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.routes import indexing
    app = FastAPI()
    app.include_router(indexing.router, prefix="/api/indexing")
    return TestClient(app)


def test_route_returns_the_record(client, al_cif):
    r = client.post("/api/indexing/files/from-path",
                    json={"method": "hough", "path": str(al_cif)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["file"]["path"] == str(al_cif)
    assert body["file"]["user_added"] is True


def test_route_error_carries_a_stable_code_and_a_sentence(client, tmp_path):
    r = client.post("/api/indexing/files/from-path",
                    json={"method": "hough", "path": str(tmp_path / "nope.cif")})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["code"] == "not_found"
    assert "nope.cif" in detail["message"]
    assert detail["params"]["path"].endswith("nope.cif")


def test_listing_route_is_not_shadowed(client):
    """GET /files/{method} still answers; the new POST is a different route."""
    r = client.get("/api/indexing/files/nope")
    assert r.status_code == 400


def test_a_path_inside_the_library_is_marked_as_such():
    from path_utils import DATABASE_SUBFOLDERS, get_local_database_path
    lib = get_local_database_path() / DATABASE_SUBFOLDERS["cif_library"] / "Al.cif"
    if not lib.is_file():
        pytest.skip("library Al.cif not present")
    rec = inspect_phase_path("hough", str(lib))
    assert rec["in_library"] is True
