"""Phase comparison per particle (backend/api/services/particle_phase_ncc.py).

The statistics are checked on scores whose top-10 mean, spread and maximum are
known by construction; the LaTeX on the shape of the two tables of the VL12
phase assignment (German decimal comma, bold maximum, footnote for an
isostructural pair).
"""
import numpy as np
import pytest

from backend.api.services import particle_phase_ncc as pp


def test_stats_use_the_ten_best_patterns():
    scores = [0.10, 0.20] + [0.50 + 0.01 * i for i in range(10)] + [None, float("nan")]
    st = pp.phase_stats(scores)
    top = np.array([0.50 + 0.01 * i for i in range(10)])
    assert st["n"] == 10 and st["n_total"] == 12
    assert st["mean"] == pytest.approx(top.mean())
    assert st["std"] == pytest.approx(top.std(ddof=1))
    assert st["max"] == pytest.approx(0.59)


def test_stats_with_fewer_patterns_than_top_n():
    st = pp.phase_stats([0.4, 0.6], top_n=10)
    assert st["n"] == 2 and st["mean"] == pytest.approx(0.5)
    st1 = pp.phase_stats([0.4])
    assert st1["std"] is None and st1["max"] == pytest.approx(0.4)
    assert pp.phase_stats([None, float("nan")])["mean"] is None


def test_stats_refuse_top_n_zero():
    with pytest.raises(ValueError):
        pp.phase_stats([0.5], top_n=0)


def _const(v, n=12):
    return [v] * n


def test_particle_ranking_next_best_and_delta():
    sp = pp.summarize_particle("VL12b", {
        "Mg2Si": _const(0.293), "Al13Fe4": _const(0.599), "Al6Fe": _const(0.193)})
    assert sp["assigned"] == "Al13Fe4" and sp["next_best"] == "Mg2Si"
    assert sp["delta"] == pytest.approx(0.306)
    assert sp["indistinguishable"] is False
    assert [r["phase_key"] for r in sp["phases"]] == ["Al13Fe4", "Mg2Si", "Al6Fe"]


def test_isostructural_pair_is_flagged_not_hidden():
    sp = pp.summarize_particle("VL12c", {"Al6Fe": _const(0.565), "MnAl6": _const(0.562)})
    assert sp["assigned"] == "Al6Fe"
    assert sp["delta"] == pytest.approx(0.003)
    assert sp["indistinguishable"] is True


def test_unscored_phase_goes_last_and_does_not_win():
    sp = pp.summarize_particle("P", {"A": [None, None], "B": _const(0.3)})
    assert sp["assigned"] == "B" and sp["next_best"] is None and sp["delta"] is None
    assert sp["phases"][-1]["phase_key"] == "A"


VL12 = [
    {"name": "VL12a", "scores": {"Mg2Si": _const(0.541), "Al13Fe4": _const(0.282),
                                 "Al6Fe": _const(0.188), "MnAl6": _const(0.191),
                                 "AlCu": _const(0.501)}},
    {"name": "VL12c", "scores": {"Mg2Si": _const(0.279), "Al13Fe4": _const(0.242),
                                 "Al6Fe": _const(0.565), "MnAl6": _const(0.562),
                                 "AlCu": _const(0.239)}},
]
INFO = {
    "Mg2Si": {"label": "Mg2Si", "space_group": "Fm-3m"},
    "Al13Fe4": {"label": "Al13Fe4", "space_group": "C2/m"},
    "Al6Fe": {"label": "Al6Fe", "space_group": "Cmcm"},
    "MnAl6": {"label": "MnAl6", "space_group": "Cmcm"},
    "AlCu": {"label": "AlCu, cF4", "space_group": "Fm-3m"},
}


def test_matrix_and_best_per_particle():
    s = pp.summarize(VL12, INFO)
    assert s["phase_order"] == ["Mg2Si", "Al13Fe4", "Al6Fe", "MnAl6", "AlCu"]
    assert s["matrix"]["AlCu"]["VL12a"] == pytest.approx(0.501)
    assert s["best_per_particle"] == {"VL12a": "Mg2Si", "VL12c": "Al6Fe"}


def test_assignment_table_matches_the_thesis_layout():
    tex = pp.latex_assignment_table(pp.summarize(VL12, INFO))
    assert r"\begin{tabularx}{\linewidth}" in tex
    assert r"VL12a & Mg$_2$Si ($Fm\bar{3}m$)" in tex
    assert r"$0{,}541$ & AlCu, cF4 ($0{,}501$) & $0{,}040$" in tex
    # isostructural pair: footnote mark on the phase and the note below
    assert r"Al$_6$Fe\textsuperscript{a} ($Cmcm$)" in tex
    assert "nicht unterscheidbar" in tex
    assert "zehn Muster" in tex


def test_candidate_table_bolds_the_maximum_per_particle():
    tex = pp.latex_candidate_table(pp.summarize(VL12, INFO))
    assert r"Mg$_2$Si & $Fm\bar{3}m$ & $\mathbf{0{,}541}$ & $0{,}279$ \\" in tex
    assert r"Al$_6$Fe & $Cmcm$ & $0{,}188$ & $\mathbf{0{,}565}$ \\" in tex
    assert r"MnAl$_6$ & $Cmcm$ & $0{,}191$ & $0{,}562$ \\" in tex


def test_missing_value_is_a_text_dash_not_two_minus_signs():
    s = pp.summarize([{"name": "P", "scores": {"A": _const(0.5)}},
                      {"name": "Q", "scores": {"B": _const(0.4)}}])
    tex = pp.latex_candidate_table(s)
    assert "$--$" not in tex and " & -- " in tex


@pytest.mark.parametrize("formula,tex", [
    ("Al13Fe4", "Al$_{13}$Fe$_4$"),
    ("Mg2Si", "Mg$_2$Si"),
    ("Mn0.5Fe0.5Al5Si0.68", "Mn$_{0,5}$Fe$_{0,5}$Al$_5$Si$_{0,68}$"),
    ("AlCu, cF4", "AlCu, cF4"),
    ("tau11-Al4Fe1.7Si", r"$\tau_{11}$-Al$_4$Fe$_{1,7}$Si"),
    ("alpha-Al(Fe,Mn)Si (sd_0302719)", r"$\alpha$-Al(Fe,Mn)Si (sd\_0302719)"),
])
def test_formula_tex(formula, tex):
    assert pp.formula_tex(formula) == tex


@pytest.mark.parametrize("sg,tex", [
    ("Fm-3m", r"$Fm\bar{3}m$"), ("Im-3", r"$Im\bar{3}$"), ("P63/mmc", "$P6_3/mmc$"),
    ("C2/m", "$C2/m$"), ("Fd-3m", r"$Fd\bar{3}m$"), ("", ""),
])
def test_space_group_tex(sg, tex):
    assert pp.space_group_tex(sg) == tex


def test_csv_has_one_row_per_particle_and_phase_with_decimal_comma():
    csv = pp.csv_table(pp.summarize(VL12, INFO))
    lines = csv.strip().splitlines()
    assert len(lines) == 1 + 2 * 5
    assert lines[1].startswith("VL12a;Mg2Si;Fm-3m;1;0,5410;0,0000;0,5410;10;12;ja;0,0400;nein")


def test_disc_pixels_clip_to_the_scan():
    assert pp.disc_pixels(0, 0, 3, 4) == [0]
    assert sorted(pp.disc_pixels(5, 1, 3, 4)) == [1, 4, 5, 6, 9]
    assert sorted(pp.disc_pixels(0, 1, 3, 4)) == [0, 1, 4]
    with pytest.raises(ValueError):
        pp.disc_pixels(12, 1, 3, 4)


# --- Route: the job collects the compare engine's per-pattern NCCs ----------

class _Entry:
    def __init__(self, key, formula, sg):
        self.key, self.formula, self.display_formula = key, formula, formula
        self.space_group, self.sht_path, self.cif_path = sg, f"/lib/{key}.sht", None


@pytest.fixture
def idx(monkeypatch):
    ix = pytest.importorskip("backend.api.routes.indexing")
    entries = [_Entry("mg2si", "Mg2Si", "Fm-3m"), _Entry("al13fe4", "Al13Fe4", "C2/m")]
    det = {"n_rows": 4, "n_cols": 5, "pat_height": 8, "pat_width": 8}
    monkeypatch.setattr(ix, "_get_active_signal_for_phase_test", lambda: object())
    monkeypatch.setattr(ix, "_get_detector_for_phase_test", lambda **kw: (None, det))
    monkeypatch.setattr(ix, "_phase_test_library_entries",
                        lambda keys: ([e for e in entries if e.key in keys], []))
    monkeypatch.setattr(ix, "_particle_ncc_pixel",
                        lambda signal, pix, n_cols, **kw: (np.full((8, 8), float(pix)), None))

    def fake_compare(*, exp_pattern, sht_paths_by_phase, **kw):
        pix = float(exp_pattern[0, 0])
        # pixel 0..9 (particle A): Mg2Si wins; pixel >= 10 (particle B): Al13Fe4 wins
        a = 0.5 + 0.001 * pix if pix < 10 else 0.25
        b = 0.3 if pix < 10 else 0.6 - 0.001 * pix
        return [{"phase_id": 0, "r_score": a}, {"phase_id": 1, "r_score": b}]
    monkeypatch.setattr(ix, "_compute_phase_compare_results", fake_compare)
    monkeypatch.setattr(ix, "_particle_ncc_busy", False)
    return ix


def _run(ix, body):
    import asyncio

    async def go():
        req = ix.ParticlePhaseNccRequest(**body)
        start = await ix.particle_phase_ncc_start(req)
        for _ in range(500):
            await asyncio.sleep(0.01)
            prog = await ix.particle_phase_ncc_progress(start["job_id"])
            if prog["status"] != "running":
                return start, prog
        raise AssertionError("job did not finish")
    return asyncio.run(go())


def test_route_collects_scores_per_particle_and_builds_tables(idx):
    start, prog = _run(idx, {
        "particles": [{"name": "A", "pixel_indices": list(range(10))},
                      {"name": "B", "center_index": 17, "radius_px": 1}],
        "phase_keys": ["mg2si", "al13fe4"], "top_n": 3})
    assert start["total"] == 10 + 4      # disc of radius 1 around (3,2) on a 4 x 5 scan
    assert prog["status"] == "done", prog["error"]
    res = prog["result"]
    a, b = res["particles"]
    assert a["assigned"] == "mg2si" and b["assigned"] == "al13fe4"
    top = next(r for r in a["phases"] if r["phase_key"] == "mg2si")
    assert top["n"] == 3 and top["mean"] == pytest.approx(0.508) and top["max"] == pytest.approx(0.509)
    assert len(res["patterns"]) == 14
    assert "Mg$_2$Si" in res["latex"]["zuordnung"] and "\\mathbf" in res["latex"]["kandidaten"]
    assert idx._particle_ncc_busy is False


def test_route_refuses_duplicate_names_and_pixels_outside(idx):
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        _run(idx, {"particles": [{"name": "A", "pixel_indices": [1]},
                                 {"name": "A", "pixel_indices": [2]}],
                   "phase_keys": ["mg2si"]})
    with pytest.raises(HTTPException):
        _run(idx, {"particles": [{"name": "A", "pixel_indices": [99]}], "phase_keys": ["mg2si"]})


def test_request_needs_pixels_and_phases():
    ix = pytest.importorskip("backend.api.routes.indexing")
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ix.ParticlePhaseNccRequest(particles=[{"name": "A"}], phase_keys=["x"])
    with pytest.raises(ValidationError):
        ix.ParticlePhaseNccRequest(particles=[{"name": "A", "center_index": 0}], phase_keys=[])
