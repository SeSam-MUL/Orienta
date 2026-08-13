"""Phase colours must be STABLE per phase name across results and COLLISION-FREE
for many phases.

2026-06-03 fix: ``build_phase_color_map`` used to assign colour by sorted
phase-id POSITION, so a phase's hue changed when a different result had a
different set/order of phases ("the colours are different when I switch
results"). It now derives a continuous HSV hue from the phase NAME.
"""
from __future__ import annotations

import sys

import pytest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.routes.phase_map import (  # noqa: E402
    build_phase_color_map, _hsv_phase_color_by_name,
)

# The user's actual 14-phase multiphase result.
USER_PHASES = [
    "Al", "Al7FeCu2", "MgCuAl2", "Al6Fe", "Fe4Al13",
    "Mn0.52Fe1.08Al4.85Si0.55", "Mn0.5Fe0.5Al5Si0.68", "MnFe2Al0.25Si0.75",
    "Fe23Al81Si15", "Fe3Al2Si3", "Al4FeSi", "Fe3Si4Al2", "Al3Fe2Si", "Mg2Si",
]


class _Ph:
    def __init__(self, name):
        self.name = name


class _Phases:
    def __init__(self, mapping):
        self._m = mapping

    def __getitem__(self, pid):
        return self._m[pid]


class _XMap:
    def __init__(self, phase_id, names):
        self.phase_id = np.asarray(phase_id)
        self.phases = _Phases({pid: _Ph(nm) for pid, nm in names.items()})


def test_phase_colour_stable_across_results():
    """The same phase name gets the same colour in two results that have
    different phase SETS and different phase-id numbering."""
    names_a = {i: USER_PHASES[i] for i in range(len(USER_PHASES))}
    cmap_a = build_phase_color_map(_XMap(list(range(len(USER_PHASES))), names_a))
    by_name_a = {names_a[pid]: cmap_a[pid] for pid in cmap_a}

    # A different result: a subset, DIFFERENT pid numbering.
    names_b = {5: "Al", 2: "Mg2Si", 9: "Fe23Al81Si15"}
    cmap_b = build_phase_color_map(_XMap([5, 2, 9, 5, 2], names_b))
    by_name_b = {names_b[pid]: cmap_b[pid] for pid in cmap_b}

    for nm in ("Al", "Mg2Si", "Fe23Al81Si15"):
        assert by_name_a[nm] == by_name_b[nm], f"{nm} colour must be stable across results"


def test_14_phases_collision_free():
    names = {i: USER_PHASES[i] for i in range(len(USER_PHASES))}
    cmap = build_phase_color_map(_XMap(list(range(len(USER_PHASES))), names))
    colors = [tuple(round(c, 5) for c in cmap[pid]) for pid in cmap]
    assert len(set(colors)) == len(colors), "all 14 phases must get distinct colours"


def test_color_override_by_name_wins():
    names = {0: "Al", 1: "Mg2Si"}
    cmap = build_phase_color_map(_XMap([0, 1, 0], names),
                                 color_overrides={"Al": "#ff0000"})
    assert tuple(round(c, 3) for c in cmap[0]) == (1.0, 0.0, 0.0)


def test_hsv_by_name_deterministic_and_distinct():
    assert _hsv_phase_color_by_name("Al") == _hsv_phase_color_by_name("Al")
    assert _hsv_phase_color_by_name("Al") != _hsv_phase_color_by_name("Fe23Al81Si15")


def test_ipf_key_swatch_uses_the_map_colours():
    """The bar above each triangle names a phase — with the map's colour.

    It used to come from the legacy 8-slot palette with overrides ignored, so
    the swatch showed a colour that appeared nowhere on the map: measured
    #c792ea beside a triangle whose phase was painted #559df2.
    """
    from backend.api.routes.phase_map import _phase_colors_by_name, build_phase_color_map

    class _Phase:
        def __init__(self, pid, name):
            self.id = pid
            self.name = name

    class _Phases:
        def __init__(self, items):
            self._items = items

        def __iter__(self):
            return iter(self._items)

        # build_phase_color_map looks a phase up by id; without this the name
        # comes back None and a colour override can never match.
        def __getitem__(self, pid):
            return self._items[int(pid)]

    class _XMap:
        def __init__(self, items):
            self.phases = _Phases(items)

    xmap = _XMap([_Phase(0, "Al"), _Phase(1, "Al7FeCu2"), _Phase(2, "Si")])

    by_id = build_phase_color_map(xmap, None)
    by_name = _phase_colors_by_name(xmap, None)
    assert by_name["Al"] == by_id[0]
    assert by_name["Al7FeCu2"] == by_id[1]
    assert by_name["Si"] == by_id[2]

    # ...and a user's pick reaches the swatch, as it reaches the map.
    picked = _phase_colors_by_name(xmap, {"Al": "#ff0000"})
    assert all(abs(a - b) < 1e-6 for a, b in zip(picked["Al"], (1.0, 0.0, 0.0)))
    assert picked["Si"] == by_name["Si"]


def test_ipf_key_swatch_survives_a_broken_xmap():
    """A key without swatches beats no key at all."""
    from backend.api.routes.phase_map import _phase_colors_by_name

    class _Broken:
        @property
        def phases(self):
            raise RuntimeError("no phases here")

    assert _phase_colors_by_name(_Broken(), None) == {}


def test_scale_info_reports_the_colours_the_map_was_painted_with():
    """A colour bar has to name the same colormap the pixels came from."""
    from backend.api.routes.phase_map import _scale_info

    info = _scale_info("RdYlGn", 0.18, 0.47, unit="")
    assert info["min"] == 0.18 and info["max"] == 0.47
    assert info["cmap"] == "RdYlGn"
    assert len(info["stops"]) == 16
    # RdYlGn runs dark red -> dark green; sampled from matplotlib, not guessed.
    assert info["stops"][0] == "#a50026"
    assert info["stops"][-1] == "#006837"
    assert all(s.startswith("#") and len(s) == 7 for s in info["stops"])


def test_scale_info_carries_the_unit_when_there_is_one():
    from backend.api.routes.phase_map import _scale_info

    assert _scale_info("magma", 0.0, 2.5, unit="px")["unit"] == "px"
    assert _scale_info("magma", 0.0, 2.5)["unit"] == ""


def test_edax_iq_is_read_when_there_is_no_band_contrast(tmp_path):
    """A file with a quality channel must produce a quality map.

    Oxford writes "Band Contrast", EDAX writes "IQ". Reading only the Oxford
    name left EDAX-flavoured files — including this app's own rich exports —
    with no map at all, and the layer refused to draw.
    """
    import h5py
    import numpy as np
    from backend.api.services import pattern_quality as pq

    path = tmp_path / "edax_like.h5"
    rows, cols = 4, 5
    iq = np.arange(rows * cols, dtype=np.float32) * 10.0
    with h5py.File(path, "w") as f:
        g = f.create_group("Scan1/EBSD")
        g.create_dataset("Data/IQ", data=iq)
        g.create_dataset("Header/nColumns", data=np.array([cols], dtype=np.int32))
        g.create_dataset("Header/nRows", data=np.array([rows], dtype=np.int32))

    got = pq.read_native_image_quality(str(path), rows, cols)
    assert got is not None
    assert got.shape == (rows, cols)
    assert got[0, 0] == 0.0 and got[-1, -1] == pytest.approx(190.0)

    # ...and the shared precedence hands it back, labelled as what it is.
    qm = pq.get_quality_map(rows, cols, source_file=str(path), allow_compute=False)
    assert qm is not None
    assert qm.metric == "image_quality"
    assert qm.source == "native"
    assert "Image Quality" in qm.label
    assert qm.value_range == (0.0, 190.0)


def test_oxford_band_contrast_still_wins(tmp_path):
    """Both channels present: the vendor's band contrast is what BC means."""
    import h5py
    import numpy as np
    from backend.api.services import pattern_quality as pq

    path = tmp_path / "both.h5"
    rows, cols = 3, 3
    with h5py.File(path, "w") as f:
        g = f.create_group("1/EBSD")
        g.create_dataset("Data/Band Contrast", data=np.full(rows * cols, 200, dtype=np.uint8))
        g.create_dataset("Data/IQ", data=np.full(rows * cols, 5000.0, dtype=np.float32))
        g.create_dataset("Header/X Cells", data=np.array([cols], dtype=np.int32))
        g.create_dataset("Header/Y Cells", data=np.array([rows], dtype=np.int32))

    qm = pq.get_quality_map(rows, cols, source_file=str(path), allow_compute=False)
    assert qm is not None and qm.metric == "band_contrast"
    assert qm.array.mean() == pytest.approx(200.0)


def test_a_file_with_neither_says_so(tmp_path):
    import h5py
    import numpy as np
    from backend.api.services import pattern_quality as pq

    path = tmp_path / "empty.h5"
    with h5py.File(path, "w") as f:
        f.create_group("Scan1/EBSD/Data").create_dataset("CI", data=np.zeros(9, dtype=np.float32))

    assert pq.read_native_image_quality(str(path), 3, 3) is None
    assert pq.get_quality_map(3, 3, source_file=str(path), allow_compute=False) is None
