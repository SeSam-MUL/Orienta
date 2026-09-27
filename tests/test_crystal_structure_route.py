from fastapi.testclient import TestClient

from backend.api.main import app

# Needs the maintainer's crystal library / measurement data, which a clone
# does not have — skip with the missing path named, never fail.
from tests.data_deps import CIF_LIBRARY, XTAL_LIBRARY, requires

client = TestClient(app)


@requires(CIF_LIBRARY / "Al.cif")
def test_structure_route_cif_ok():
    r = client.get("/api/database/structure/Al.cif")
    assert r.status_code == 200
    body = r.json()
    assert len(body["atoms"]) == 4
    assert body["space_group"]["number"] == 225
    assert body["source"] == "cif"


@requires(XTAL_LIBRARY / "Ni.xtal")
def test_structure_route_xtal_ok():
    r = client.get("/api/database/structure/Ni.xtal")
    assert r.status_code == 200
    body = r.json()
    assert len(body["atoms"]) == 4
    assert body["source"] == "xtal"


def test_structure_route_missing_404():
    r = client.get("/api/database/structure/nope.cif")
    assert r.status_code == 404


def test_structure_route_bad_type_422():
    r = client.get("/api/database/structure/whatever.txt")
    assert r.status_code == 422
