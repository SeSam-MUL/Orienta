"""Small robustness cases of the grain analysis: all grains culled, zero quaternions, GOS and ECD maps by lookup."""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from orix.crystal_map import CrystalMap, PhaseList
from orix.quaternion import Rotation

from analysis.ebsd_dataset import EBSDDataset
from analysis.grain_analysis import reconstruct_grains
from tests.synthetic_grain_map import make_map, to_xmap


@pytest.fixture
def client_with_map():
    import backend.api.routes.analysis as ar
    from backend.api.main import app
    m = make_map()
    saved = ar._dataset
    ar._dataset = EBSDDataset(to_xmap(m), step_size=m["step"])
    try:
        yield TestClient(app), ar
    finally:
        ar._dataset = saved


# --- the KAM map of a dataset whose grains are all culled --------------------------------
def test_kam_map_without_any_grain_is_computed_without_grains(client_with_map):
    client, ar = client_with_map
    ar._dataset.grains = reconstruct_grains(ar._dataset, min_pixels=10 ** 6, fill=False)
    assert ar._dataset.grains.n_grains == 0
    r = client.get("/api/analysis/map/kam")
    assert r.status_code == 200
    body = r.json()
    assert np.isfinite(body["min_val"]) and np.isfinite(body["max_val"])
    from analysis.deformation_analysis import calculate_kam
    plain = calculate_kam(ar._dataset)
    assert body["max_val"] == pytest.approx(float(np.nanmax(plain)))


# --- a zero quaternion is not a measurement ----------------------------------------------
class _RawRotations:
    """What a loader that skips orix's normalisation hands over: the raw quaternion rows."""

    def __init__(self, data):
        self.data = data


class _RawRotationXmap(CrystalMap):
    """A CrystalMap whose ``rotations`` returns raw rows (orix 0.14 turns a zero row into NaN on read)."""

    @property
    def rotations(self):
        return _RawRotations(self._raw)


def _identity_map(ny=12, nx=12, zero_at=None):
    q = np.tile(Rotation.identity(1).data, (ny * nx, 1)).astype(float)
    xmap = CrystalMap(rotations=Rotation(q), phase_id=np.zeros(ny * nx, int),
                      x=np.tile(np.arange(nx, dtype=float), ny), y=np.repeat(np.arange(ny, dtype=float), nx),
                      phase_list=PhaseList(names=["Al"], point_groups=["m-3m"]))
    if zero_at is not None:
        raw = q.copy()
        raw[zero_at] = 0.0
        xmap.__class__ = _RawRotationXmap
        xmap._raw = raw
    return EBSDDataset(xmap, step_size=1.0)


def test_grid_arrays_turns_a_zero_quaternion_into_nan():
    from analysis.grain_reconstruction import grid_arrays
    zero_at = 6 * 12 + 6
    q, _ = grid_arrays(_identity_map(zero_at=zero_at))
    assert np.isnan(q.reshape(-1, 4)[zero_at]).all()
    assert np.isfinite(np.delete(q.reshape(-1, 4), zero_at, axis=0)).all()


@pytest.mark.parametrize("min_pixels", [1, 3])
def test_a_zero_quaternion_inside_a_grain_is_closed_not_its_own_grain(min_pixels):
    zero_at = 6 * 12 + 6
    ds = _identity_map(zero_at=zero_at)
    assert np.all(ds.xmap.rotations.data[zero_at] == 0.0)
    g = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=min_pixels, fill=False)
    assert g.n_grains == 1
    assert g.closed_mask.reshape(-1)[zero_at]
    assert g.grain_ids.reshape(-1)[zero_at] == 1
    assert g.grain_size[0] == 143 and g.n_closed[0] == 1


# --- GOS / ECD maps by lookup ------------------------------------------------------------
def _per_grain_image_loop(values, grain_ids, shape, convert=lambda v: v):
    """The first implementation of the GOS and ECD map routes, with the background the routes have since: a pixel
    without a grain (label 0) or with a label that has no value has no grain quantity, so it is NaN, not 0."""
    data = np.full(shape, np.nan, dtype=float)
    for i, v in enumerate(values):
        data[grain_ids == (i + 1)] = convert(v)
    return data


def test_per_grain_image_equals_the_per_grain_loop():
    from backend.api.routes.analysis import _per_grain_image
    rng = np.random.default_rng(0)
    grain_ids = rng.integers(0, 8, size=(15, 17))           # 0 = no grain
    values = rng.random(7) * 3
    values[2] = np.nan
    np.testing.assert_array_equal(_per_grain_image(values, grain_ids),
                                  _per_grain_image_loop(values, grain_ids, grain_ids.shape))
    np.testing.assert_array_equal(_per_grain_image(np.degrees(values), grain_ids),
                                  _per_grain_image_loop(values, grain_ids, grain_ids.shape, np.degrees))


def test_per_grain_image_ignores_a_label_without_a_value():
    from backend.api.routes.analysis import _per_grain_image
    grain_ids = np.array([[0, 1, 2], [3, 4, 2]])
    values = np.array([10.0, 20.0])                        # labels 3 and 4 have no value
    np.testing.assert_array_equal(_per_grain_image(values, grain_ids),
                                  _per_grain_image_loop(values, grain_ids, grain_ids.shape))


@pytest.mark.parametrize("kind", ["gos", "ecd"])
def test_gos_and_ecd_routes_draw_the_values_of_the_per_grain_loop(client_with_map, kind):
    client, ar = client_with_map
    g = reconstruct_grains(ar._dataset, min_pixels=1, fill=False)
    ar._dataset.grains = g
    r = client.get(f"/api/analysis/map/{kind}")
    assert r.status_code == 200
    if kind == "gos":
        expect = _per_grain_image_loop(g.gos, g.grain_ids, g.grain_ids.shape, np.degrees)
    else:
        expect = _per_grain_image_loop(g.ecd, g.grain_ids, g.grain_ids.shape)
    assert r.json()["min_val"] == pytest.approx(float(np.nanmin(expect)))
    assert r.json()["max_val"] == pytest.approx(float(np.nanmax(expect)))
