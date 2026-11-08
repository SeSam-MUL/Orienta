"""Every place Orienta computes |F| from an orix phase goes through the fixed path.

hcp is the sharpest case: two atoms on 2c of P6_3/mmc give the closed form
|F| = 2 f(g) |cos(pi((h-k)/3 + l/2))|, which forbids 000l (l odd) and the whole
{2-1-11} family and allows {10-10} at about half the strongest reflection. With
diffsims' own ``sanitise_phase`` the extinct (2 -1 1) came out at |F| = 8.7 and
{10-10} was missing from alpha-Ti's default list.

The expectations are that closed form (f(g) is a common factor taken once from a
lone atom) and, for the reflector tables, a list built from it row by row. Each
test below reaches ``|F|`` through one of the consumers named in its docstring;
none of them calls ``phase_cell`` itself.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("orix")
pytest.importorskip("diffsims")

from orix.crystal_map import PhaseList  # noqa: E402

import ebsd_utils as eu  # noqa: E402
import hough_reflectors as hr  # noqa: E402
from tests.test_phase_cell import (  # noqa: E402
    CRYSTALS, _asymmetric_phase, _f_single, _from_cif)


@pytest.fixture(autouse=True)
def clean_registry():
    hr.clear_specs()
    eu.clear_phase_reflector_limits()
    hr._CACHE.clear()
    eu._FILTERED_CACHE.clear()
    del eu._FILTERED_CACHE_ORDER[:]
    yield
    hr.clear_specs()
    eu.clear_phase_reflector_limits()


HCP = ["hcp-Mg", "alpha-Ti"]

#: Where the phase comes from: a CIF (the whole cell, as diffpy expands it) or only
#: its asymmetric unit. The second is what shows whether a consumer completes the
#: cell itself: without that, one atom has no extinctions at all.
SOURCES = ["cif", "asymmetric"]


def _make(tmp_path, name, source):
    if source == "cif":
        return _from_cif(tmp_path, name)[0]
    return _asymmetric_phase(name)


def _lattice(name):
    return CRYSTALS[name][1]


def _element(name):
    return CRYSTALS[name][2][0][0]


def _closed_form(name, hkl):
    """|F| of hcp (2 atoms on 2c), for an (n, 3) array of 3-index hkl."""
    hkl = np.asarray(hkl, float)
    h, k, l = hkl.T
    return (2 * _f_single(_element(name), _lattice(name), hkl)
            * np.abs(np.cos(np.pi * ((h - k) / 3 + l / 2))))


def _all_rows(name, d_min):
    lat = _lattice(name)
    from diffpy.structure import Lattice
    L = Lattice(*lat)
    rows = np.array([h for h in itertools.product(range(-9, 10), repeat=3) if h != (0, 0, 0)],
                    float)
    return rows[1.0 / L.rnorm(rows) >= d_min - 1e-9]


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("name", HCP)
def test_default_reflector_rows_are_the_closed_form_rows(tmp_path, name, source):
    """``ebsd_utils._filtered_reflectors`` (Hough default: d >= 1 A and |F| > 0.1 max):
    the same rows the closed form selects."""
    phase = _make(tmp_path, name, source)
    rows = _all_rows(name, 1.0)
    F = _closed_form(name, rows)
    want = {tuple(int(x) for x in r) for r in rows[F > 0.1 * F.max()]}
    filt = eu._filtered_reflectors(phase, 1.0, 0.1)
    got = {tuple(int(round(x)) for x in r) for r in np.asarray(filt.hkl)}
    assert got == want
    # the strong first-order prism is in, and the extinct families are not
    assert (1, 0, 0) in got and (0, 0, 2) in got
    assert not got & {(2, -1, 1), (1, 1, 1), (0, 0, 1), (0, 0, 3), (2, 2, 1)}


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("name", HCP)
def test_reflector_stage_table_holds_the_closed_form_values(tmp_path, name, source):
    """``hough_reflectors.stage_table``: the table the family list is drawn from."""
    phase = _make(tmp_path, name, source)
    table = hr.stage_table(phase)
    assert len(table.reps) > 5
    want = _closed_form(name, np.array(table.reps, float))
    assert np.allclose(table.f, want, rtol=1e-9)
    assert table.fmax == pytest.approx(_closed_form(name, _all_rows(name, 1.0)).max(), rel=1e-9)
    # {10-10}: |F| = 2 f |cos(pi/3)| = f, about half of {0002}, and it is listed
    assert (1, 0, 0) in table.row_to_family
    i = table.row_to_family[(1, 0, 0)]
    assert table.f[i] / table.fmax > 0.3
    assert (2, -1, 1) not in table.row_to_family


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("name", HCP)
def test_hand_added_family_is_refused_when_extinct(tmp_path, name, source):
    """``_extra_family`` / ``_check_allowed``: the extinction lock for a typed family."""
    phase = _make(tmp_path, name, source)
    for rep in ((2, -1, 1), (0, 0, 1), (1, 1, 1), (0, 0, 3)):
        assert hr._extra_family(phase, rep)["f"] < 1e-6, rep
        with pytest.raises(hr.SpecError) as e:
            hr._check_allowed(phase, rep)
        assert e.value.code == "forbidden", rep
    for rep in ((1, 0, 0), (0, 0, 2), (1, 0, 1), (2, -1, 0)):
        info = hr._check_allowed(phase, rep)
        want = float(_closed_form(name, np.array([rep], float))[0])
        assert info["f"] == pytest.approx(want, rel=1e-9), rep


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("name", HCP)
def test_kikuchi_overlay_reflectors_carry_the_closed_form_values(tmp_path, name, source):
    """``rlv_for_spec`` (overlay of a custom selection) and the default path of
    ``prepare_reflectors``: structure factors of the vectors handed to kikuchipy."""
    phase = _make(tmp_path, name, source)
    hr.set_spec(phase.name, {"mode": "custom", "families": [[1, 0, 0], [0, 0, 2], [1, 0, 1]]})
    rlv = hr.rlv_for_spec(phase, hr.get_spec(phase.name))
    got = np.abs(np.asarray(rlv.structure_factor)).ravel()
    want = _closed_form(name, np.asarray(rlv.hkl))
    assert len(got) >= 12 and np.allclose(got, want, rtol=1e-9)

    hr.clear_specs()
    refl = hr.prepare_reflectors(PhaseList(phases=[phase]))
    got = np.abs(np.asarray(refl.structure_factor)).ravel()
    want = _closed_form(name, np.asarray(refl.hkl))
    assert len(got) > 10 and np.allclose(got, want, rtol=1e-9)
    assert got.min() > 0.1 * _closed_form(name, _all_rows(name, 1.0)).max()


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("name", HCP)
def test_indexing_overlay_simulation_uses_the_fixed_structure_factors(tmp_path, monkeypatch, name, source):
    """``indexing_controller._simulate_pattern_for_orientation`` (geometrical overlay of an
    indexed pixel): the reflectors it gives kikuchipy's simulator."""
    import kikuchipy as kp
    import kikuchipy.simulations as ksim
    from orix.quaternion import Rotation

    import indexing_controller as ic

    phase = _make(tmp_path, name, source)
    seen = {}

    class Recorder:
        def __init__(self, ref, *a, **k):
            seen["ref"] = ref
            raise RuntimeError("stop here: the reflectors are all this test wants")

    monkeypatch.setattr(ksim, "KikuchiPatternSimulator", Recorder)
    detector = kp.detectors.EBSDDetector(shape=(60, 60), pc=(0.5, 0.5, 0.5), sample_tilt=70.0)
    out = ic._simulate_pattern_for_orientation(
        detector, PhaseList(phases=[phase]), Rotation.identity(), 0)
    assert out is None and "ref" in seen
    ref = seen["ref"]
    got = np.abs(np.asarray(ref.structure_factor)).ravel()
    want = _closed_form(name, np.asarray(ref.hkl))
    assert len(got) > 20 and np.allclose(got, want, rtol=1e-9)
    # kikuchipy draws a line for every reflector it gets, whatever its |F|: the list
    # must already be the default rule's (d >= 1 A, |F| above a tenth of the strongest),
    # so no forbidden family is drawn
    rows = {tuple(int(round(x)) for x in r) for r in np.asarray(ref.hkl)}
    all_rows = _all_rows(name, 1.0)
    F = _closed_form(name, all_rows)
    want_rows = {tuple(int(x) for x in r) for r in all_rows[F > 0.1 * F.max()]}
    assert rows == want_rows
    assert not rows & {(2, -1, 1), (1, 1, 1), (0, 0, 1), (0, 0, 3), (2, 2, 1)}
    assert got.min() > 0.1 * F.max()


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("name", HCP)
def test_candidate_table_of_the_extended_list_holds_the_closed_form_values(tmp_path, name, source):
    """``hough_reflectors.candidate_table`` (the "show weaker families" list, d down to
    0.7 A) and ``describe(extended=True)`` built on it."""
    phase = _make(tmp_path, name, source)
    table = hr.candidate_table(phase, 0.8)
    assert len(table.reps) > 20
    assert np.allclose(table.f, _closed_form(name, np.array(table.reps, float)), rtol=1e-9)
    assert (1, 0, 0) in table.row_to_family
    for row in ((2, -1, 1), (0, 0, 1), (1, 1, 1), (0, 0, 3)):
        assert row not in table.row_to_family, row
    labels = {f["label"] for f in hr.describe(phase, extended=True)["families"]}
    assert "{10-10}" in labels and "{2-1-11}" not in labels and "{0001}" not in labels


@pytest.mark.parametrize("source", SOURCES)
def test_the_indexing_overlay_really_renders_an_image(tmp_path, source):
    """The geometrical overlay returned None under matplotlib 3.10 (tostring_rgb is gone
    and a bare Figure cannot draw), so the quick-test image never appeared. Real
    simulator, real renderer: an image of the detector's shape with lines on a black ground."""
    import kikuchipy as kp
    from orix.quaternion import Rotation

    import indexing_controller as ic

    phase = _make(tmp_path, "alpha-Ti", source)
    detector = kp.detectors.EBSDDetector(shape=(60, 80), pc=(0.5, 0.5, 0.5), sample_tilt=70.0)
    img = ic._simulate_pattern_for_orientation(
        detector, PhaseList(phases=[phase]), Rotation.from_euler([20, 40, 30], degrees=True), 0)
    assert img is not None and img.shape == (60, 80) and img.dtype == np.float32
    assert img.max() > 0 and (img == 0).any()
