"""The tables derived from a scan do not outlive the scan.

The neighbour tables and the hexagonal cell index are big arrays kept by ``lru_cache``. Loading
another file (``scan_geometry.clear_all``) has to release them, and the number kept is bounded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import hex_raster, scan_geometry, scan_neighbours  # noqa: E402
from backend.api.services.scan_geometry import ScanGeometry  # noqa: E402
from backend.api.services.scan_neighbours import Topology  # noqa: E402


@pytest.fixture(autouse=True)
def _empty():
    scan_geometry.clear_all()
    yield
    scan_geometry.clear_all()


def _caches():
    return {"table": scan_neighbours._table, "pairs": scan_neighbours._pairs,
            "cell_index": hex_raster._cell_index_cached}


def _build_one_of_each():
    topology = Topology("hex", 6, 7, 0.5)
    topology.table()
    topology.table(rings=2, include_self=True)
    topology.pairs()
    hex_raster.cell_index(ScanGeometry.hex(6, 7, 2.0))


def test_the_tables_are_there_after_they_are_built():
    _build_one_of_each()
    # one ring, and a disc of two rings with the point itself: two tables; one list of pairs; one index
    assert _caches()["table"].cache_info().currsize == 2
    assert _caches()["pairs"].cache_info().currsize == 1
    assert _caches()["cell_index"].cache_info().currsize == 1


def test_clear_all_releases_every_table():
    _build_one_of_each()
    scan_geometry.clear_all()
    for name, cache in _caches().items():
        assert cache.cache_info().currsize == 0, name


def test_a_table_is_built_again_after_the_release_and_is_the_same():
    topology = Topology("hex", 6, 7, 0.5)
    before = topology.table().copy()
    scan_geometry.clear_all()
    assert scan_neighbours._table.cache_info().currsize == 0
    after = topology.table()
    assert (after == before).all() and scan_neighbours._table.cache_info().currsize == 1


def test_clear_all_still_forgets_the_geometries():
    scan_geometry.set_geometry("a.h5", ScanGeometry.square(3, 4, 1.0))
    assert scan_geometry.geometry_for("a.h5") is not None
    scan_geometry.clear_all()
    assert scan_geometry.geometry_for("a.h5") is None


def test_the_number_of_tables_kept_is_bounded():
    # eight different scans, one table each: no more than the bound stay
    for n in range(4, 12):
        Topology("hex", n, n + 1, 0.5).table()
        Topology("hex", n, n + 1, 0.5).pairs()
    assert scan_neighbours._table.cache_info().currsize <= 4
    assert scan_neighbours._pairs.cache_info().currsize <= 4
    assert scan_neighbours._table.cache_info().maxsize == scan_neighbours._pairs.cache_info().maxsize == 4
