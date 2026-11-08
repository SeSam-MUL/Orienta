"""Spectrum model and sum-spectrum fit (backend/eds_spectrum)."""
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from backend.eds_spectrum.fit import FileFit, fit_sum_spectrum, forward, net_areas
from backend.eds_spectrum import tables
from backend.eds_spectrum.model import (MODEL_VERSION, fwhm, NonlinearParams,
                                        SpectrumMeta, SpectrumModel,
                                        default_families, label)

META = SpectrumMeta(start_kev=-0.1967, width_kev=0.01, n_channels=2048,
                    beam_kv=20.0)
FAMS = [("O", "K"), ("Al", "K"), ("Si", "K"), ("Mn", "K"), ("Mn", "L"),
        ("Fe", "K"), ("Fe", "L"), ("Cu", "K"), ("Cu", "L"), ("Zn", "K"),
        ("Zn", "L")]
AREAS = {"Al K": 5e6, "Si K": 6e5, "Fe K": 4e5, "Mn K": 9e4, "Cu K": 1.5e5,
         "Cu L": 4e5, "Zn L": 2e4}
COMP = {"Al": 0.78, "Si": 0.10, "Fe": 0.05, "Cu": 0.06, "Mn": 0.01}
BENCH = Path(__file__).resolve().parents[1] / "tasks" / "eds-spectrum-fit" / "bench"


def truth(meta, fams, areas, params, bg_per_channel=40.0):
    """A FileFit holding a physically shaped truth, built without fitting."""
    model = SpectrumModel(meta, fams)
    A, names, mask = model.design(params, COMP)
    coef = np.zeros(len(names))
    for j, nm in enumerate(names):
        if nm in areas:
            coef[j] = areas[nm]
        elif nm.startswith("Al tail"):
            coef[j] = 0.002 * areas.get("Al K", 0.0)
        elif nm.startswith("Si tail"):
            coef[j] = 0.002 * areas.get("Si K", 0.0)
    bg = [j for j, nm in enumerate(names) if nm.startswith("bg")]
    E = (params.offset_kev + params.gain * model.E0)[mask]
    probe = (E > 2.5) & (E < 3.5)
    shape = A[:, bg].sum(axis=1)
    coef[bg] = bg_per_channel * (meta.width_kev / 0.01) / shape[probe].mean()
    total = float((A @ coef).sum())
    # pile-up shapes built from the truth itself, with small physical amplitudes
    model.rebuild_pileup(params, coef, names, A, mask)
    A2, names2, mask2 = model.design(params, COMP)
    coef2 = np.zeros(len(names2))
    for nm, c in zip(names, coef):
        coef2[names2.index(nm)] = c
    for nm, frac in (("pile lo", 3e-4), ("pile hi", 2e-4), ("pile 3", 1e-5)):
        coef2[names2.index(nm)] = frac * total
    A, names, mask = A2, names2, mask2
    return FileFit(meta=meta, families=list(fams), params=params, names=names,
                   A=A, mask=mask, coef=coef2, chi2_red=0.0, comp=dict(COMP),
                   pile_cols=dict(model.pile_cols), model_version=MODEL_VERSION)


@pytest.fixture(scope="module")
def synth():
    cache = {}

    def make(seed, offset=0.0, f0=0.13, meta=META, fams=FAMS, areas=AREAS):
        key = (offset, f0, meta, tuple(fams))
        if key not in cache:
            cache[key] = truth(meta, fams, areas,
                               NonlinearParams(offset, 1.0, f0, 3e-4))
        ff = cache[key]
        mean = forward(ff, areas)
        return np.random.default_rng(seed).poisson(mean).astype(float)
    return make


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_recovers_areas_unbiased(seed, synth):
    ff = fit_sum_spectrum(synth(seed), META, FAMS)
    got = net_areas(ff)
    for k, v in AREAS.items():
        assert abs(got[k] - v) < 3 * np.sqrt(v) + 0.01 * v, (k, got[k], v)


def test_recovers_nonlinear_params(synth):
    ff = fit_sum_spectrum(synth(0, offset=0.004, f0=0.140), META, FAMS)
    assert ff.params.offset_kev == pytest.approx(0.004, abs=0.002)
    assert ff.params.fwhm_mnka_kev == pytest.approx(0.140, abs=0.003)


def test_absent_element_area_is_zero_not_negative(synth):
    got = net_areas(fit_sum_spectrum(synth(0), META, FAMS + [("Ni", "K")]))
    assert got["Ni K"] >= 0
    assert got["Ni K"] < 3 * np.sqrt(sum(AREAS.values()) * 1e-3)


def test_header_calibration_is_used(synth):
    meta2 = replace(META, start_kev=-0.050, width_kev=0.020, n_channels=1024,
                    beam_kv=15.0)
    ff = fit_sum_spectrum(synth(0, meta=meta2), meta2, FAMS)
    assert ff.meta.width_kev == 0.020 and ff.chi2_red < 2.0
    got = net_areas(ff)
    assert abs(got["Al K"] - AREAS["Al K"]) < 3 * np.sqrt(AREAS["Al K"]) + 0.01 * AREAS["Al K"]


def test_chi2_near_one_on_pure_poisson(synth):
    assert fit_sum_spectrum(synth(0), META, FAMS).chi2_red < 1.5


def test_channel_count_must_match_header():
    with pytest.raises(ValueError, match="channels"):
        fit_sum_spectrum(np.ones(100), META, FAMS)


def test_filefit_is_self_consistent(synth):
    ff = fit_sum_spectrum(synth(0), META, FAMS)
    assert ff.model_version == MODEL_VERSION
    assert ff.A.shape == (int(ff.mask.sum()), len(ff.names)) and len(ff.coef) == len(ff.names)
    assert {"pile lo", "pile hi", "pile 3"} <= set(ff.names)
    # forward with the fitted areas reproduces A @ coef
    refit = forward(ff, net_areas(ff))
    assert np.allclose(refit[ff.mask], ff.A @ ff.coef, rtol=1e-9)


def test_forward_is_zero_for_unlisted_families(synth):
    ff = fit_sum_spectrum(synth(0), META, FAMS)
    on = forward(ff, {"Al K": 1e6})
    off = forward(ff, {})
    assert (on - off).sum() == pytest.approx(1e6, rel=0.02)


def test_default_families():
    fams = default_families(["Fe Kα1", "Au", "Cu"])
    assert ("Fe", "K") in fams and ("Zr", "L") in fams and ("Au", "L") in fams
    assert ("Fe", "L") in fams and ("Cu", "L") in fams
    assert all(len(f) == 2 for f in fams)
    low = default_families([], SpectrumMeta(0.0, 0.01, 2048, 5.0))
    assert ("Al", "K") in low and ("Fe", "K") not in low and ("Zn", "K") not in low


def test_default_families_overvoltage_gate():
    meta15 = SpectrumMeta(0.0, 0.01, 2048, 15.0)
    fams = default_families([], meta15)
    assert ("Zn", "K") in fams  # 15 / 9.659 = 1.55
    meta12 = SpectrumMeta(0.0, 0.01, 2048, 12.0)
    assert ("Zn", "K") not in default_families([], meta12)
    assert ("Zn", "L") in default_families([], meta12)


def test_every_non_family_column_has_weight_and_tails_sit_below_their_line():
    """A dead (all-zero) column cannot be seen by recovery or parity tests."""
    params = NonlinearParams(0.0, 1.0, 0.13, 3e-4)
    model = SpectrumModel(META, FAMS, pile_cols={
        "pile lo": np.ones(META.n_channels), "pile hi": np.ones(META.n_channels),
        "pile 3": np.ones(META.n_channels)})
    A, names, mask = model.design(params, COMP)
    fam_labels = {label(f) for f in FAMS}
    E = META.energies()[mask]
    checked = set()
    for j, nm in enumerate(names):
        if nm in fam_labels:
            continue
        assert A[:, j].sum() > 0, f"column {nm} is identically zero"
        checked.add(nm)
        if "tail" in nm:
            line = tables.main_line(nm.split()[0], "K")[0]
            col = A[:, j]
            width = fwhm(line, params.fwhm_mnka_kev)
            # the tail is the low-side shelf smeared by the detector line shape:
            # the smearing puts 13-30 % of its area just above the line energy,
            # practically all within one FWHM of it (>= 99 % measured)
            assert col[E < line + width].sum() / col.sum() >= 0.95, nm
            assert col[E < line].sum() / col.sum() >= 0.65, nm
            assert (col * E).sum() / col.sum() < line, nm
    assert {"Al tail20", "Al tail80", "Si tail20", "Si tail80", "pile lo",
            "pile hi", "pile 3"} <= checked
    assert sum(n.startswith("bg") for n in checked) == 10


def test_label():
    assert label(("Fe", "K")) == "Fe K"


BENCH_FAMS = ([(e, "K") for e in ("C", "N", "O", "Mg", "Al", "Si")]
              + [(e, f) for e in ("Ti", "Cr", "Mn", "Fe", "Ni", "Cu", "Zn") for f in "KL"]
              + [("Zr", "L")])


def test_matches_bench_on_alpha_sum():
    z = np.load(BENCH / "bench.npz")
    info = json.loads((BENCH / "bench.json").read_text(encoding="utf-8"))
    m = info["regions"]["AB4_alpha_eroded"]["meta"]
    meta = SpectrumMeta(m["start_kev"], m["width_kev"], m["n_ch"], m["beam_kv"],
                        detector=m["detector"], tilt_deg=m["tilt_deg"])
    t0 = time.time()
    ff = fit_sum_spectrum(z["AB4_alpha_eroded"], meta, BENCH_FAMS)
    seconds = time.time() - t0
    ref = json.loads((BENCH / "nets_nnls.json").read_text())["AB4_alpha_eroded"]
    got = net_areas(ff)
    assert ff.chi2_red < 90  # regenerated bench: 77.1 (113 with dead tail columns)
    for k in ("Al K", "Si K", "Fe K", "Mn K", "Cu K", "Cu L"):
        assert got[k] == pytest.approx(ref[k], rel=0.01), (k, got[k], ref[k])
    assert seconds < 60
