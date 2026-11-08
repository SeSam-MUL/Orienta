"""The RX / substructure texture split reaches the workbook (route export and batch); workbook names are reserved."""
import threading

import numpy as np
from types import SimpleNamespace

import pytest
from openpyxl import load_workbook

from tests.test_analysis_page_sync import _reconstruct, _write_ctf, client_two_phase  # noqa: F401


def _fake_texture_factory():
    from analysis.texture_components import TEXTURE_PRESETS
    names = list(TEXTURE_PRESETS["FCC_Rolling"])

    def fake(dataset=None, *a, **k):
        # the texture step itself (about 90 s of orix symmetry set-up) is not under test; the exporter needs these.
        # The phase is the one the real step would take, so the RX / texture split sees the phases of the map.
        from analysis.texture_analysis import _phase_id_of, _phase_label
        pid = _phase_id_of(dataset, k.get("texture_phase", "most_measured"))
        return SimpleNamespace(texture_index=1.0, entropy=1.0, component_fractions={n: 5.0 for n in names},
                               plane_111_fraction=0.1, plane_100_fraction=0.1, plane_110_fraction=0.1,
                               phase_id=pid, phase_name=_phase_label(dataset, pid))
    return fake, names


@pytest.fixture(autouse=True)
def split_calls(request, monkeypatch):
    """The real split costs minutes per call (orix symmetry set-up, like the texture step itself): every test but
    the ones marked ``slow`` gets a stand-in that records its arguments and returns fixed percentages."""
    calls = []
    if request.node.get_closest_marker("slow") is None:
        import analysis.texture_analysis as ta
        _, names = _fake_texture_factory()

        def fake_split(dataset, grains, rx_grain_ids, components, tolerance_deg=10.0, texture_phase="most_measured"):
            calls.append({"dataset": dataset, "grains": grains, "rx_grain_ids": np.asarray(rx_grain_ids),
                          "components": components, "tolerance_deg": tolerance_deg, "texture_phase": texture_phase})
            return {"rx": {n: 12.5 for n in names}, "substructure": {n: 87.5 for n in names},
                    "n_rx_pixels": 1, "n_substructure_pixels": 1}

        monkeypatch.setattr(ta, "calculate_texture_split_by_rx", fake_split)
    return calls


def _spy_on_the_real_split(monkeypatch):
    """Let the real split run and keep the arguments it was called with."""
    import analysis.texture_analysis as ta
    real, seen = ta.calculate_texture_split_by_rx, []

    def spy(dataset, grains, rx_grain_ids, components, tolerance_deg=10.0, texture_phase="most_measured"):
        seen.append((dataset, grains, np.asarray(rx_grain_ids), components, tolerance_deg))
        return real(dataset, grains, rx_grain_ids, components, tolerance_deg, texture_phase=texture_phase)

    monkeypatch.setattr(ta, "calculate_texture_split_by_rx", spy)
    return seen


def _check_split_against_the_hand_computation(path, names, seen):
    """The two group blocks of the written sheet equal the whole-map per-pixel answer, masked by hand."""
    from tests.texture_reference_impl import split_by_hand, use_variants_of_the_code_under_test
    use_variants_of_the_code_under_test()           # the reference counts the variants of the code under test
    (dataset, grains, rx_ids, components, tolerance) = seen[0]
    by_hand = split_by_hand(dataset, grains, rx_ids, components, tolerance)
    ws = load_workbook(path)["SmallVSLargeGrain"]
    k = len(names)
    for block, key in ((1, "substructure"), (2, "rx")):
        written = [ws.cell(row=3, column=2 + block * k + j).value for j in range(k)]
        want = [by_hand[key][n] for n in names]
        assert len(written) == len(want)
        for w, h in zip(written, want):
            assert (w is None) == (h is None)
            if h is not None:
                assert w == pytest.approx(h, abs=1e-9)
    # not a vacuous pass: some component really has pixels in a group, and the two groups differ
    assert any(v for v in by_hand["rx"].values() if v) or any(v for v in by_hand["substructure"].values() if v)


def _row3_numbers(ws, names):
    """{block title: values in row 3} of the three blocks of SmallVSLargeGrain."""
    k = len(names)
    out = {}
    for b, title in enumerate(("all Grains", "Substructure Grains", "RX Grains")):
        out[title] = [ws.cell(row=3, column=2 + b * k + j).value for j in range(k)]
    return out


def _check_sheet(path, names):
    ws = load_workbook(path)["SmallVSLargeGrain"]
    assert [ws.cell(row=2, column=2 + j).value for j in range(3 * len(names))] == names * 3
    assert ws.cell(row=3, column=1).value
    row = _row3_numbers(ws, names)
    assert all(v == 5.0 for v in row["all Grains"])
    split = [v for t in ("Substructure Grains", "RX Grains") for v in row[t]]
    assert any(isinstance(v, (int, float)) for v in split), row          # values below row 2, not only titles
    assert all(0.0 <= v <= 100.0 for v in split if v is not None)


# ---------------------------------------------------------------------------------------------
# the split in the route export and in the batch
# ---------------------------------------------------------------------------------------------
def _export_through_the_route(client, tmp_path, name="doc.xlsx"):
    out = tmp_path / name
    r = client.post("/api/analysis/export/excel", json={"output_path": str(out), "dataset_name": "scan"})
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "completed" and task["result"]["warnings"] == [], task
    return out


def test_the_route_export_writes_the_texture_split(client_two_phase, monkeypatch, tmp_path, split_calls):
    import analysis.texture_analysis as ta
    from analysis.texture_components import TEXTURE_PRESETS
    client, ar = client_two_phase
    fake, names = _fake_texture_factory()
    monkeypatch.setattr(ta, "analyze_texture", fake)
    _reconstruct(client)
    out = _export_through_the_route(client, tmp_path)
    ws = load_workbook(out)["SmallVSLargeGrain"]
    row = _row3_numbers(ws, names)
    assert row["Substructure Grains"] == [87.5] * len(names) and row["RX Grains"] == [12.5] * len(names)
    _check_sheet(out, names)
    # the split gets the RX analysis' own grains, the preset of the texture step and its tolerance
    (call,) = split_calls
    assert call["components"] == TEXTURE_PRESETS["FCC_Rolling"] and call["tolerance_deg"] == 10.0
    assert call["grains"] is ar._dataset.grains
    want = ar._collect_export_results([])["rx"].rx_grain_ids
    np.testing.assert_array_equal(call["rx_grain_ids"], want)


@pytest.mark.slow
def test_the_route_export_writes_the_real_texture_split(client_two_phase, monkeypatch, tmp_path):
    import analysis.texture_analysis as ta
    client, ar = client_two_phase
    fake, names = _fake_texture_factory()
    monkeypatch.setattr(ta, "analyze_texture", fake)
    _reconstruct(client)
    seen = _spy_on_the_real_split(monkeypatch)
    out = _export_through_the_route(client, tmp_path)
    _check_sheet(out, names)
    assert len(seen) == 1
    _check_split_against_the_hand_computation(out, names, seen)


def test_the_route_export_names_a_failing_split_in_the_warnings(client_two_phase, monkeypatch):
    import analysis.texture_analysis as ta
    client, ar = client_two_phase
    fake, _ = _fake_texture_factory()
    monkeypatch.setattr(ta, "analyze_texture", fake)
    monkeypatch.setattr(ta, "calculate_texture_split_by_rx",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no split")))
    _reconstruct(client)
    warnings = []
    results = ar._collect_export_results(warnings)
    assert "texture_by_rx" not in results and "texture" in results
    assert any(w.startswith("texture_by_rx: RuntimeError: no split") for w in warnings)


def test_the_route_export_has_no_split_without_an_rx_result(client_two_phase, monkeypatch):
    import analysis.texture_analysis as ta
    client, ar = client_two_phase
    fake, _ = _fake_texture_factory()
    monkeypatch.setattr(ta, "analyze_texture", fake)
    _reconstruct(client)
    del ar._dataset.xmap.prop["bc"]                      # no native band contrast: no RX analysis
    results = ar._collect_export_results([])
    assert "rx" not in results and "texture_by_rx" not in results


def test_the_batch_writes_the_texture_split(tmp_path, monkeypatch, split_calls):
    from analysis import batch_processor as bp
    from analysis.texture_components import get_texture_preset
    fake, names = _fake_texture_factory()
    monkeypatch.setattr(bp, "analyze_texture", fake)
    folder = tmp_path / "scans"
    folder.mkdir()
    _write_ctf(folder / "scan.ctf")
    out = tmp_path / "doc.xlsx"
    proc = bp.BatchProcessor(str(folder), str(out), min_grain_pixels=1, grain_alpha=0.0, tolerance_deg=7.5)
    proc.process_all()
    res = proc.result()
    assert res.processed and not res.failed and res.workbook_path == str(out), res
    assert [w for w in res.warnings if "texture_by_rx" in w["reason"]] == []
    _check_sheet(out, names)
    (call,) = split_calls                                  # the components and the tolerance of the texture step
    assert call["components"] == get_texture_preset("FCC_Rolling") and call["tolerance_deg"] == 7.5


@pytest.mark.slow
def test_the_batch_writes_the_real_texture_split(tmp_path, monkeypatch):
    from analysis import batch_processor as bp
    fake, names = _fake_texture_factory()
    monkeypatch.setattr(bp, "analyze_texture", fake)
    # orientations on two FCC rolling components (Cube, Brass), so that the fractions are not all zero
    _write_ctf(tmp_path / "scan.ctf", eulers=((2.0, 1.0, 0.0), (35.0, 45.0, 0.0)))
    out = tmp_path / "out" / "doc.xlsx"
    seen = _spy_on_the_real_split(monkeypatch)
    proc = bp.BatchProcessor(str(tmp_path), str(out), min_grain_pixels=1, grain_alpha=0.0)
    proc.process_all()
    assert proc.result().processed and not proc.result().failed
    _check_sheet(out, names)
    assert len(seen) == 1
    _check_split_against_the_hand_computation(out, names, seen)


def test_the_batch_warns_when_the_split_fails(tmp_path, monkeypatch):
    import analysis.texture_analysis as ta
    from analysis import batch_processor as bp
    fake, _ = _fake_texture_factory()
    monkeypatch.setattr(bp, "analyze_texture", fake)
    monkeypatch.setattr(ta, "calculate_texture_split_by_rx",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no split")))
    _write_ctf(tmp_path / "scan.ctf")
    proc = bp.BatchProcessor(str(tmp_path), str(tmp_path / "out" / "doc.xlsx"), min_grain_pixels=1, grain_alpha=0.0)
    results = proc.process_file(tmp_path / "scan.ctf")
    assert results is not None and "texture_by_rx" not in results and "rx" in results
    assert any(w["reason"].startswith("texture_by_rx: RuntimeError: no split") for w in proc.warnings)


# ---------------------------------------------------------------------------------------------
# workbook names
# ---------------------------------------------------------------------------------------------
def test_free_workbook_path_reserves_the_name_with_an_empty_file(tmp_path):
    from analysis.excel_exporter import free_workbook_path
    first = free_workbook_path(tmp_path / "sub" / "doc.xlsx")
    assert first == tmp_path / "sub" / "doc.xlsx" and first.is_file() and first.stat().st_size == 0
    second = free_workbook_path(tmp_path / "sub" / "doc.xlsx")
    assert second == tmp_path / "sub" / "doc_2.xlsx" and second.is_file()
    assert free_workbook_path(tmp_path / "sub" / "doc.xlsx").name == "doc_3.xlsx"


def test_concurrent_picks_never_get_the_same_name(tmp_path):
    from analysis.excel_exporter import free_workbook_path
    n = 12
    barrier = threading.Barrier(n)
    got, lock = [], threading.Lock()

    def pick():
        barrier.wait(10)
        p = free_workbook_path(tmp_path / "doc.xlsx")
        with lock:
            got.append(p)

    threads = [threading.Thread(target=pick) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert len(got) == n and len(set(got)) == n
    assert all(p.is_file() for p in got)


def test_release_removes_only_an_empty_reservation(tmp_path):
    from analysis.excel_exporter import free_workbook_path, release_workbook_reservation
    empty = free_workbook_path(tmp_path / "a.xlsx")
    release_workbook_reservation(empty)
    assert not empty.exists()
    written = free_workbook_path(tmp_path / "b.xlsx")
    written.write_bytes(b"PK content")
    release_workbook_reservation(written)
    assert written.read_bytes() == b"PK content"
    release_workbook_reservation(tmp_path / "never-existed.xlsx")          # no error


def test_the_exporter_treats_the_empty_reservation_as_new(tmp_path):
    from analysis.excel_exporter import ExcelExporter, free_workbook_path
    path = free_workbook_path(tmp_path / "doc.xlsx")
    ex = ExcelExporter(str(path))
    ex.save()
    ex.close()
    assert "Sheet1" in load_workbook(path).sheetnames


def test_a_batch_that_writes_no_workbook_leaves_no_reservation(tmp_path):
    from analysis import batch_processor as bp
    folder = tmp_path / "scans"
    folder.mkdir()
    (folder / "bad.h5").write_bytes(b"")                  # cannot be read: no file gives a result
    out = tmp_path / "out" / "doc.xlsx"
    proc = bp.BatchProcessor(str(folder), str(out))
    proc.process_all()
    assert proc.result().workbook_path is None and not out.exists()


def test_a_batch_whose_save_fails_leaves_no_reservation(tmp_path, monkeypatch):
    from analysis import batch_processor as bp
    fake, _ = _fake_texture_factory()
    monkeypatch.setattr(bp, "analyze_texture", fake)
    monkeypatch.setattr(bp.ExcelExporter, "save", lambda self: (_ for _ in ()).throw(PermissionError("locked")))
    _write_ctf(tmp_path / "scan.ctf")
    out = tmp_path / "out" / "doc.xlsx"
    proc = bp.BatchProcessor(str(tmp_path), str(out), min_grain_pixels=1, grain_alpha=0.0)
    proc.process_all()
    assert proc.result().save_error and not out.exists()


def test_two_batches_on_one_path_write_two_workbooks_and_the_first_is_untouched(tmp_path, monkeypatch):
    from analysis import batch_processor as bp
    fake, _ = _fake_texture_factory()
    monkeypatch.setattr(bp, "analyze_texture", fake)
    folder = tmp_path / "scans"
    folder.mkdir()
    _write_ctf(folder / "scan.ctf")
    out = tmp_path / "doc.xlsx"
    first = bp.BatchProcessor(str(folder), str(out), min_grain_pixels=1, grain_alpha=0.0)
    first.process_all()
    size = out.stat().st_size
    second = bp.BatchProcessor(str(folder), str(out), min_grain_pixels=1, grain_alpha=0.0)
    second.process_all()
    assert second.result().workbook_path == str(tmp_path / "doc_2.xlsx")
    assert out.stat().st_size == size and (tmp_path / "doc_2.xlsx").stat().st_size > 0


def test_a_failing_route_export_leaves_no_reservation(client_two_phase, monkeypatch, tmp_path):
    import analysis.excel_exporter as ee
    client, ar = client_two_phase
    monkeypatch.setattr(ar, "_collect_export_results", lambda warnings: {})
    monkeypatch.setattr(ee, "export_to_excel", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))
    out = tmp_path / "doc.xlsx"
    r = client.post("/api/analysis/export/excel", json={"output_path": str(out), "dataset_name": "d"})
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "failed" and not out.exists()
