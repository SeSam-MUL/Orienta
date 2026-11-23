"""Tilt distortion check for .h5oina files (backend/api/services/tilt_distortion.py).

Two kinds of evidence:

* The decision rules (header tilt + measured Lx/Ly -> finding) against the
  reference measurements of the data set "Masterarbeit_Schlegl_2026_07"
  (SE images, Lx/Ly as measured there). Those files are not distributed, so
  the rules are checked on their measured values.
* The measurement itself on synthetic images whose distortion is known by
  construction: an isotropic texture, foreshortened in X (or Y) by cos 70 deg.
"""
import math

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("scipy")
from scipy.ndimage import gaussian_filter, zoom  # noqa: E402

from backend.api.services import tilt_distortion as td  # noqa: E402

K70 = 1.0 / math.cos(math.radians(70.0))


def _iso(h, w, seed=0, s=3.0):
    rng = np.random.default_rng(seed)
    return gaussian_filter(rng.standard_normal((h, w)), s)


def _foreshortened_x(h=768, w=1024, seed=0):
    wide = _iso(h, int(w * K70) + 2, seed)
    return zoom(wide, (1, 1 / K70), order=3)[:, :w]


def _foreshortened_y(h=768, w=1024, seed=0):
    tall = _iso(int(h * K70) + 2, w, seed)
    return zoom(tall, (1 / K70, 1), order=3)[:h]


def _u8(a):
    a = (a - a.min()) / max(np.ptp(a), 1e-12)
    return (a * 255).astype(np.uint8)


# --- Reference cases (decision rules on the measured values) ---------------

REFERENCE = [
    # (file, header tilt, Lx/Ly (SE), expected finding)
    ("6m_EBSD_TEST AB6, Elektronenbild 33", 70, 0.34, td.BEFUND_X),
    ("6m_EBSD_TEST AB10, Elektronenbild 38", 70, 0.39, td.BEFUND_X),
    ("6m_EBSD_TEST AB11, Elektronenbild 39", 70, 0.29, td.BEFUND_X),
    ("6m_EBSD_TEST AB3, Elektronenbild 30", 70, 0.41, td.BEFUND_X),
    ("6m_EBSD_TEST AB5, Elektronenbild 32", 70, 0.38, td.BEFUND_X),
    ("6m_EBSD_TEST AB4, Elektronenbild 31", 0, 1.01, td.BEFUND_OK),
    ("6m_EBSD_TEST AB7, Elektronenbild 35", 0, 1.01, td.BEFUND_OK),
    ("6_m AB4, Elektronenbild 18", 0, 1.04, td.BEFUND_OK),
    ("Projekt 1, Probe 1, Elektronenbild 1", 70, 1.63, td.BEFUND_UNCLEAR),
    ("Projekt 1, Probe 1, Elektronenbild 2", 70, 1.02, td.BEFUND_UNCLEAR),
]


@pytest.mark.parametrize("name,tilt,ratio,expected", REFERENCE, ids=[r[0] for r in REFERENCE])
def test_reference_cases(name, tilt, ratio, expected):
    v = td.reconcile(tilt, ratio)
    assert v.befund == expected
    if expected == td.BEFUND_X:
        assert v.faktor == pytest.approx(2.92, abs=0.01)
    else:
        assert v.faktor == 1.0          # never stretched by a guess


def test_contradiction_untilted_but_foreshortened_reports_measured_factor():
    v = td.reconcile(0.0, 0.34)
    assert v.befund == td.BEFUND_UNCLEAR
    assert v.faktor_gemessen == pytest.approx(2.94, abs=0.01)
    assert any("Widerspruch" in h for h in v.hinweise)


def test_probable_band_is_reported_as_probable():
    v = td.reconcile(70.0, 0.68)
    assert v.befund == td.BEFUND_X and v.sicherheit == "wahrscheinlich"


def test_fsd_is_never_decisive_alone():
    v = td.reconcile(0.0, 1.9, channel="FSD")
    assert any("FSD" in h for h in v.hinweise)


@pytest.mark.parametrize("ratio,expected", [
    (0.34, (td.BEFUND_X, "sicher")), (0.7, (td.BEFUND_X, "wahrscheinlich")),
    (1.0, (td.BEFUND_OK, "sicher")), (1.5, (td.BEFUND_UNCLEAR, None)),
    (2.9, (td.BEFUND_Y, "sicher")), (None, (td.BEFUND_UNCLEAR, None)),
])
def test_classification_table(ratio, expected):
    assert td.classify_ratio(ratio) == expected


def test_stretch_factor_70_deg():
    assert td.stretch_factor(70.0) == pytest.approx(2.9238, abs=1e-3)


# --- Measurement on images with a known distortion -------------------------

def test_measures_x_foreshortening():
    m = td.measure_anisotropy(_foreshortened_x())
    assert m.lx_ly_median == pytest.approx(math.cos(math.radians(70)), abs=0.06)
    assert m.tiles_used == td.KEEP_TILES


def test_measures_isotropic_image_as_one():
    m = td.measure_anisotropy(_iso(768, 1024, seed=3))
    assert m.lx_ly_median == pytest.approx(1.0, abs=0.1)


def test_measures_y_foreshortening():
    m = td.measure_anisotropy(_foreshortened_y())
    assert m.lx_ly_median == pytest.approx(K70, rel=0.15)


def test_texture_poor_image_is_not_forced_into_a_number():
    flat = np.tile(np.linspace(0, 1, 1024), (768, 1))
    m = td.measure_anisotropy(flat)
    assert m.lx_ly_median is None and "texturarm" in m.reason


# --- Whole file -------------------------------------------------------------

def _write_h5oina(path, image, *, tilt_deg=70.0, stage_deg=0.0, channel="SE",
                  name="Elektronenbild 33 (Input1)", img_step=0.0621,
                  map_cells=(12, 9), map_step=0.658, px_per_step=(9.83, 8.66),
                  hit_rate=15.7, flat=True):
    ny, nx = image.shape
    with h5py.File(path, "w") as f:
        ih = f.create_group("1/Electron Image/Header")
        ih["X Cells"] = np.array([nx], np.int32)
        ih["Y Cells"] = np.array([ny], np.int32)
        ih["X Step"] = np.array([img_step], np.float32)
        ih["Y Step"] = np.array([img_step], np.float32)
        ih["Tilt Angle"] = np.array([math.radians(tilt_deg)], np.float32)
        ih["Tilt Axis"] = np.array([0.0], np.float32)
        ih["Stage Position/Tilt"] = np.array([math.radians(stage_deg)], np.float32)
        data = image.ravel() if flat else image
        f.create_dataset(f"1/Electron Image/Data/{channel}/{name}", data=data)
        mx, my = map_cells
        rs = [px_per_step[0] * mx / nx, px_per_step[1] * my / ny]
        for area in ("EBSD", "EDS"):
            h = f.create_group(f"1/{area}/Header")
            h["X Cells"] = np.array([mx], np.int32)
            h["Y Cells"] = np.array([my], np.int32)
            h["X Step"] = np.array([map_step], np.float32)
            h["Y Step"] = np.array([map_step], np.float32)
            h["Relative Size"] = np.array(rs, np.float32)
            h["Relative Offset"] = np.array([0.4, 0.4], np.float32)
            h["Tilt Angle"] = np.array([math.radians(tilt_deg)], np.float32)
            if area == "EBSD":
                h["Hit Rate"] = np.array([hit_rate], np.float32)


def test_file_like_ab6_x_foreshortened(tmp_path):
    p = tmp_path / "AB6.h5oina"
    _write_h5oina(p, _u8(_foreshortened_x()))
    r = td.check_h5oina(str(p))
    assert r["kippung_header_grad"] == pytest.approx(70.0, abs=0.01)
    assert r["faktor_k_theoretisch"] == pytest.approx(2.92, abs=0.01)
    assert r["befund"] == td.BEFUND_X
    (img,) = r["bilder"]
    assert img["name"] == "SE/Elektronenbild 33 (Input1)"
    assert img["pixel"] == [1024, 768]
    assert img["korrektur"] == "X × 2.92, Y × 1"
    assert img["zielgroesse_px"] == [2994, 768]
    ebsd = next(k for k in r["karten"] if k["typ"] == "EBSD")
    assert ebsd["raster"] == [12, 9]
    assert ebsd["px_pro_schritt_im_bild"] == pytest.approx([9.83, 8.66], abs=0.01)
    assert ebsd["erbt_bildverzerrung"] is True
    assert ebsd["wahrer_schritt_um"][0] == pytest.approx(1.92, abs=0.01)
    assert ebsd["wahrer_schritt_um"][1] == pytest.approx(0.658, abs=1e-3)
    assert ebsd["hit_rate_prozent"] == pytest.approx(15.7, abs=0.01)
    assert r["zusammenfassung"] == "Verzerrt: ja, X gestaucht, Faktor 2,92."


def test_file_untilted_isotropic(tmp_path):
    p = tmp_path / "AB4.h5oina"
    _write_h5oina(p, _u8(_iso(768, 1024, seed=5)), tilt_deg=0.0, flat=False)
    r = td.check_h5oina(str(p))
    assert r["befund"] == td.BEFUND_OK
    ebsd = next(k for k in r["karten"] if k["typ"] == "EBSD")
    assert ebsd["wahrer_schritt_um"] == pytest.approx([0.658, 0.658], abs=1e-3)


def test_file_tilted_header_but_isotropic_image_is_unclear(tmp_path):
    p = tmp_path / "projekt1.h5oina"
    _write_h5oina(p, _u8(_iso(768, 1024, seed=7)))
    r = td.check_h5oina(str(p))
    assert r["befund"] == td.BEFUND_UNCLEAR
    assert r["faktor_korrektur"] == 1.0
    assert any("Widerspruch" in h for h in r["bilder"][0]["hinweise"])


def test_file_effective_tilt_adds_stage_tilt(tmp_path):
    p = tmp_path / "stage.h5oina"
    _write_h5oina(p, _u8(_foreshortened_x()), tilt_deg=0.0, stage_deg=70.0)
    r = td.check_h5oina(str(p))
    assert r["kippung_wirksam_grad"] == pytest.approx(70.0, abs=0.01)
    assert r["befund"] == td.BEFUND_X


def test_file_with_only_fsd_does_not_decide(tmp_path):
    p = tmp_path / "fsd.h5oina"
    _write_h5oina(p, _u8(_foreshortened_x()), channel="FSD", name="FSD Upper")
    r = td.check_h5oina(str(p))
    assert r["befund"] == td.BEFUND_UNCLEAR
    assert any("FSD" in h for h in r["hinweise"])


def test_map_on_its_own_grid_does_not_inherit(tmp_path):
    p = tmp_path / "grid.h5oina"
    _write_h5oina(p, _u8(_foreshortened_x()), px_per_step=(9.83, 4.0))
    r = td.check_h5oina(str(p))
    ebsd = next(k for k in r["karten"] if k["typ"] == "EBSD")
    assert ebsd["erbt_bildverzerrung"] is False
    assert ebsd["wahrer_schritt_um"] == pytest.approx([0.658, 0.658], abs=1e-3)


def test_file_without_electron_image_is_unclear(tmp_path):
    p = tmp_path / "noimg.h5oina"
    with h5py.File(p, "w") as f:
        h = f.create_group("1/EBSD/Header")
        h["X Cells"] = np.array([4], np.int32)
        h["Y Cells"] = np.array([3], np.int32)
        h["X Step"] = np.array([1.0], np.float32)
        h["Tilt Angle"] = np.array([math.radians(70)], np.float32)
    r = td.check_h5oina(str(p))
    assert r["befund"] == td.BEFUND_UNCLEAR
    assert r["karten"][0]["erbt_bildverzerrung"] is None


def test_cached_check_measures_once(tmp_path):
    p = tmp_path / "c.h5oina"
    _write_h5oina(p, _u8(_foreshortened_x()))
    a = td.check_h5oina_cached(str(p))
    b = td.check_h5oina_cached(str(p))
    assert a is b


# --- Wiring into the load answer --------------------------------------------

def test_load_helper_reports_h5oina_and_ignores_other_formats(tmp_path):
    ev = pytest.importorskip("backend.api.routes.ebsd_viewer")
    p = tmp_path / "AB6.h5oina"
    _write_h5oina(p, _u8(_foreshortened_x()))
    r = ev._tilt_check_for(str(p))
    assert r["befund"] == td.BEFUND_X
    assert ev._tilt_check_for(str(tmp_path / "scan.up1")) is None
    assert ev._tilt_check_for(None) is None


def test_load_helper_never_fails_a_load(tmp_path):
    ev = pytest.importorskip("backend.api.routes.ebsd_viewer")
    p = tmp_path / "broken.h5oina"
    p.write_bytes(b"not hdf5")
    r = ev._tilt_check_for(str(p))
    assert r["befund"] == td.BEFUND_UNCLEAR and r["fehler"]


def test_tilt_check_route(tmp_path, monkeypatch):
    ev = pytest.importorskip("backend.api.routes.ebsd_viewer")
    import asyncio
    p = tmp_path / "AB6.h5oina"
    _write_h5oina(p, _u8(_foreshortened_x()))
    monkeypatch.setattr(ev, "_ebsd_file_path", str(p))
    monkeypatch.setattr(ev, "_active_dataset", None)
    out = asyncio.run(ev.tilt_check())
    assert out["tilt_check"]["befund"] == td.BEFUND_X
    monkeypatch.setattr(ev, "_ebsd_file_path", None)
    assert asyncio.run(ev.tilt_check()) == {"tilt_check": None}
