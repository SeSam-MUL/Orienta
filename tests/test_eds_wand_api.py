"""Wand endpoints: the write guards, not the maths (that is test_eds_wand)."""
import base64
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.services.cif_phase_library import CifPhaseEntry
from backend.api.services.phase_map_store import get_phase_map_store

client = TestClient(app)


def _entry(name, comp):
    return CifPhaseEntry(
        key=name, cif_filename=name, formula=name, space_group="",
        space_group_number=None, crystal_system="", composition=comp,
        elements=sorted(comp),
    )


@pytest.fixture
def loaded_map():
    store = get_phase_map_store()
    store.set_classification(
        phase_grid=np.zeros((4, 4), dtype=np.int32),
        score_grid=np.full((4, 4), 0.9, dtype=np.float32),
        phase_entries=[_entry("Al.cif", {"Al": 100.0}),
                       _entry("Si.cif", {"Si": 100.0})],
        tolerance=15.0, min_score=0.3,
    )
    yield store
    store.clear()


def _packed(mask: np.ndarray) -> str:
    return base64.b64encode(np.packbits(mask.ravel()).tobytes()).decode()


def test_assign_without_a_map_is_refused():
    get_phase_map_store().clear()
    r = client.post("/api/eds/phase-map/wand-assign",
                    json={"mask_b64": _packed(np.ones((4, 4), bool)),
                          "phase_index": 0})
    assert r.status_code == 400
    assert "auto-classify" in r.json()["detail"]


def test_a_mask_smaller_than_the_map_is_refused(loaded_map):
    """A truncated selection would paint a region the user never chose."""
    r = client.post("/api/eds/phase-map/wand-assign",
                    json={"mask_b64": base64.b64encode(b"\x00").decode(),
                          "phase_index": 0})
    assert r.status_code == 400
    assert "pixels" in r.json()["detail"]


def test_garbage_mask_is_refused(loaded_map):
    r = client.post("/api/eds/phase-map/wand-assign",
                    json={"mask_b64": "not base64 !!", "phase_index": 0})
    assert r.status_code == 400


def test_an_out_of_range_phase_is_refused(loaded_map):
    r = client.post("/api/eds/phase-map/wand-assign",
                    json={"mask_b64": _packed(np.ones((4, 4), bool)),
                          "phase_index": 99})
    assert r.status_code == 400


def test_assign_marks_the_pixels_as_hand_set(loaded_map):
    mask = np.zeros((4, 4), bool)
    mask[1, 1] = mask[1, 2] = True
    r = client.post("/api/eds/phase-map/wand-assign",
                    json={"mask_b64": _packed(mask), "phase_index": 1})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_assigned"] == 2
    assert body["n_locked"] == 2
    st = loaded_map.get_state()
    assert st.phase_grid[1, 1] == 1 and st.phase_grid[1, 2] == 1
    assert st.phase_grid[0, 0] == 0
    assert st.locked_mask[1, 1]


def test_assign_can_mark_a_region_unclassified(loaded_map):
    mask = np.zeros((4, 4), bool)
    mask[0, 0] = True
    r = client.post("/api/eds/phase-map/wand-assign",
                    json={"mask_b64": _packed(mask), "phase_index": -1})
    assert r.status_code == 200, r.text
    assert loaded_map.get_state().phase_grid[0, 0] == -1


def test_field_without_an_open_file_is_refused():
    r = client.post("/api/eds/phase-map/wand-field", json={"row": 0, "col": 0})
    assert r.status_code in (400, 500)
    assert r.status_code == 400 or "detail" in r.json()
