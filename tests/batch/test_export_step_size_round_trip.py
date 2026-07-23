"""Step-size round-trip for the result exporter.

Regression net for the bug where step_size was lost between the indexer
and every downstream output (.ang/.ctf/_light.h5), causing the PC
refinement axes manager to come back with scale=1.0 and quantitative
analysis to silently use 1 µm coordinates regardless of the real EBSD
step. The fix wires step_size_um through:

   signal.axes_manager → CheckpointWriter.init_metadata
                       → /metadata.attrs[step_size_um] in checkpoint
                       → export_all reads it
                       → /Indexing.attrs[step_size_um] in .light/.h5
                       → CTF header XStep/YStep
                       → ang/ctf coordinates in µm
                       → _load_kikuchipy_rich_h5 scales x/y → xmap.step_sizes
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from backend.api.services.checkpoint_writer import CheckpointWriter
from backend.api.services.result_exporter import (
    _read_step_size_from_checkpoint,
    export_all,
    export_ang_ctf,
)


@pytest.fixture
def cubic_checkpoint(tmp_path: Path):
    """A 4×3 multiphase checkpoint with two cubic phases + auto-assignment.

    step_size_um is intentionally 0.5 (a common real-world Oxford step)
    so a regression where the exporter substitutes 1.0 is visible as a
    factor-of-two coordinate error.

    The fake h5oina now also carries a Band Contrast dataset and an
    acquisition header with a tilt of 70°. Tests can assert that:
    - the real BC reaches the .ctf and .ang
    - the .ctf TiltAngle uses the caller-supplied tilt, NOT the 70° in
      the h5oina header (caller is the calibration store).
    """
    source = tmp_path / "FakeSample.h5oina"
    n_pixels = 12  # 3 rows × 4 cols
    # Distinct BC values per pixel so the test can spot any reordering.
    bc_values = np.arange(10, 10 + n_pixels, dtype=np.uint8)
    with h5py.File(source, "w") as f:
        ebsd_data = f.create_group("1/EBSD/Data")
        ebsd_data.create_dataset("Band Contrast", data=bc_values)
        hdr = f.create_group("1/EBSD/Header")
        hdr.create_dataset("Beam Voltage", data=np.float32(20.0))
        hdr.create_dataset("Tilt Angle", data=np.float32(70.0))
        hdr.create_dataset("Magnification", data=np.float32(500.0))
    cw = CheckpointWriter(str(source))
    cw.init_metadata(
        grid_shape=(3, 4), batch_id="step-test",
        method="spherical", step_size_um=0.5,
    )
    rng = np.random.default_rng(0)
    # Top half is Al-favourable, bottom half Cu-favourable so the
    # auto-assignment picks both phases — needed for the .ctf phase
    # lookup loop to exercise more than just the first phase.
    ci_a = np.full((3, 4), 0.4, dtype=np.float32)
    ci_b = np.full((3, 4), 0.4, dtype=np.float32)
    ci_a[:2] = 0.8  # Al wins top two rows
    ci_b[2:] = 0.8  # Cu wins bottom row
    ori_a = rng.uniform(0, 0.1, size=(3, 4, 3)).astype(np.float32)
    ori_b = rng.uniform(0.4, 0.5, size=(3, 4, 3)).astype(np.float32)
    cw.write_phase_result(
        "Al", ci_a, ori_a,
        {"phase_file": "Al.cif", "ci_mean": 0.75, "duration_sec": 1.0,
         "space_group": 225, "point_group": "m-3m"},
    )
    cw.write_phase_result(
        "Cu", ci_b, ori_b,
        {"phase_file": "Cu.cif", "ci_mean": 0.45, "duration_sec": 1.0,
         "space_group": 225, "point_group": "m-3m"},
    )
    cw.compute_auto_assignment(confidence_threshold=0.0)
    return source, cw


def test_step_size_persists_in_checkpoint(cubic_checkpoint):
    _src, cw = cubic_checkpoint
    assert _read_step_size_from_checkpoint(cw.checkpoint_path) == 0.5


def test_export_all_writes_step_into_light_h5(cubic_checkpoint, tmp_path):
    source, cw = cubic_checkpoint
    out = tmp_path / "out"
    out.mkdir()
    results = export_all(
        source_h5_path=str(source),
        checkpoint_path=cw.checkpoint_path,
        output_dir=str(out),
        formats=["h5_light"],
        # No source EDS / quality groups in the fixture — tolerated.
        include_eds=False,
    )
    assert results.get("h5_light"), f"light export failed: {results.get('_errors')}"
    with h5py.File(results["h5_light"], "r") as f:
        idx_attr = float(f["Indexing"].attrs["step_size_um"])
        det_attr = float(f["Detector"].attrs["step_size_um"])
        params_attr = float(f["Indexing/Parameters"].attrs["step_size_um"])
    # Three places, one truth — divergence here means a writer drifted.
    assert idx_attr == 0.5
    assert det_attr == 0.5
    assert params_attr == 0.5


def test_light_h5_has_xy_coordinate_datasets(cubic_checkpoint, tmp_path):
    """The .light file must carry per-pixel X/Y µm coordinates so MTEX can
    place the orientations on a grid without reading the source h5oina.
    Convention: X = column · step, Y = row · step (matches the .ang writer)."""
    source, cw = cubic_checkpoint
    out = tmp_path / "out"
    out.mkdir()
    results = export_all(
        source_h5_path=str(source),
        checkpoint_path=cw.checkpoint_path,
        output_dir=str(out),
        formats=["h5_light"],
        include_eds=False,
    )
    assert results.get("h5_light"), f"light export failed: {results.get('_errors')}"
    with h5py.File(results["h5_light"], "r") as f:
        X = np.array(f["Indexing/X"])
        Y = np.array(f["Indexing/Y"])
    assert X.shape == (3, 4) and Y.shape == (3, 4), f"coord shapes {X.shape}/{Y.shape}"
    cc, rr = np.meshgrid(np.arange(4), np.arange(3))
    np.testing.assert_allclose(X, cc * 0.5, err_msg="X must be col·step")
    np.testing.assert_allclose(Y, rr * 0.5, err_msg="Y must be row·step")


def test_ctf_header_has_real_step(cubic_checkpoint, tmp_path):
    source, cw = cubic_checkpoint
    out = tmp_path / "out"
    out.mkdir()
    ang, ctf = export_ang_ctf(
        cw.checkpoint_path, str(out),
        write_ang=False, write_ctf=True,
        source_h5_path=str(source),
    )
    assert ctf is not None
    text = Path(ctf).read_text(encoding="utf-8")
    assert "XStep\t0.5" in text, f"CTF missing real XStep:\n{text[:400]}"
    assert "YStep\t0.5" in text


def test_ang_coordinates_are_in_microns(cubic_checkpoint, tmp_path):
    """orix.io.save uses xmap.x/y as-is. With our fix x/y are in µm so
    the ANG line for the bottom-right pixel of a 4-col × 3-row scan with
    step=0.5 should be at (1.5, 1.0) µm — not (3, 2) pixels."""
    source, cw = cubic_checkpoint
    out = tmp_path / "out"
    out.mkdir()
    ang, _ = export_ang_ctf(
        cw.checkpoint_path, str(out),
        write_ang=True, write_ctf=False,
        source_h5_path=str(source),
    )
    assert ang is not None
    # ANG header has '# XSTEP' / '# YSTEP' lines (orix writes them).
    head = "\n".join(Path(ang).read_text(encoding="utf-8").splitlines()[:40])
    assert "0.5" in head, f"ANG missing 0.5 µm step:\n{head}"


def test_reader_recovers_step_size(cubic_checkpoint, tmp_path):
    """The reader path that feeds Phase Map / PC Refinement must
    rebuild a CrystalMap whose .step_sizes equals what we wrote — that
    is what populates the axes manager downstream."""
    source, cw = cubic_checkpoint
    out = tmp_path / "out"
    out.mkdir()
    results = export_all(
        source_h5_path=str(source),
        checkpoint_path=cw.checkpoint_path,
        output_dir=str(out),
        formats=["h5_light"],
        include_eds=False,
    )
    assert results.get("h5_light")
    from backend.api.routes.analysis import _load_kikuchipy_rich_h5
    xmap = _load_kikuchipy_rich_h5(results["h5_light"])
    sx = float(getattr(xmap, "dx", 0.0) or 0.0)
    assert sx == pytest.approx(0.5, abs=1e-6), (
        f"reader returned step={sx} — axes manager would still be wrong"
    )


def test_ctf_carries_real_bc_and_zero_mad(cubic_checkpoint, tmp_path):
    """BC column must hold real h5oina Band Contrast values, MAD column
    must be zero (we re-index from scratch — no MAD of our own)."""
    source, cw = cubic_checkpoint
    out = tmp_path / "out"
    out.mkdir()
    ang, ctf = export_ang_ctf(
        cw.checkpoint_path, str(out),
        write_ang=False, write_ctf=True,
        source_h5_path=str(source),
    )
    assert ctf is not None
    lines = Path(ctf).read_text(encoding="utf-8").splitlines()
    # Find the data block — first line after the column header.
    data_start = next(
        i for i, ln in enumerate(lines)
        if ln.startswith("Phase\tX\tY\tBands")
    ) + 1
    rows = [ln.split("\t") for ln in lines[data_start:] if ln.strip()]
    assert len(rows) == 12, f"expected 12 data rows, got {len(rows)}"
    # CTF column order: Phase X Y Bands Error E1 E2 E3 MAD BC BS
    mads = [float(r[8]) for r in rows]
    bcs  = [int(r[9])   for r in rows]
    assert all(m == 0.0 for m in mads), f"MAD column not all zero: {mads}"
    expected_bc = list(range(10, 22))
    assert bcs == expected_bc, f"BC column doesn't match h5oina source: {bcs}"


def test_ctf_no_native_bc_writes_zeros_and_provenance(cubic_checkpoint, tmp_path):
    """With no source h5oina there is no native Band Contrast. The CTF
    BC column must be honest zeros (never a CI×255 surrogate) and the
    Prj header line must carry a provenance note."""
    source, cw = cubic_checkpoint
    out = tmp_path / "out"
    out.mkdir()
    ang, ctf = export_ang_ctf(
        cw.checkpoint_path, str(out),
        write_ang=False, write_ctf=True,
        source_h5_path=None,  # no source → no native BC
    )
    assert ctf is not None
    lines = Path(ctf).read_text(encoding="utf-8").splitlines()
    prj_line = next(ln for ln in lines if ln.startswith("Prj\t"))
    assert "no native Band Contrast" in prj_line, (
        f"Prj line missing provenance note: {prj_line!r}"
    )
    data_start = next(
        i for i, ln in enumerate(lines)
        if ln.startswith("Phase\tX\tY\tBands")
    ) + 1
    rows = [ln.split("\t") for ln in lines[data_start:] if ln.strip()]
    assert len(rows) == 12, f"expected 12 data rows, got {len(rows)}"
    bcs = [int(r[9]) for r in rows]
    assert all(b == 0 for b in bcs), f"BC column must be honest zeros: {bcs}"


def test_ctf_native_bc_prj_line_has_no_note(cubic_checkpoint, tmp_path):
    """When native BC IS present the Prj header stays byte-identical
    (no provenance note) so native round-trip behaviour is unchanged."""
    source, cw = cubic_checkpoint
    out = tmp_path / "out"
    out.mkdir()
    ang, ctf = export_ang_ctf(
        cw.checkpoint_path, str(out),
        write_ang=False, write_ctf=True,
        source_h5_path=str(source),
    )
    assert ctf is not None
    text = Path(ctf).read_text(encoding="utf-8")
    assert "Prj\tOrienta multi-phase batch\n" in text, (
        "native-BC Prj line must remain exactly the original text"
    )
    assert "no native Band Contrast" not in text


def test_ang_no_native_bc_header_note(cubic_checkpoint, tmp_path):
    """With no native Band Contrast orix writes zeros into the ANG IQ
    column (no fake). We add a header comment noting the IQ column is
    empty and NOT a confidence surrogate."""
    source, cw = cubic_checkpoint
    out = tmp_path / "out"
    out.mkdir()
    ang, _ = export_ang_ctf(
        cw.checkpoint_path, str(out),
        write_ang=True, write_ctf=False,
        source_h5_path=None,  # no source → no native BC
    )
    assert ang is not None
    text = Path(ang).read_text(encoding="utf-8")
    assert "no native Band Contrast" in text, (
        f"ANG header missing no-native-BC provenance note:\n{text[:600]}"
    )
    # And the IQ column must actually be honest zeros — no confidence
    # surrogate. IQ is column 6 (index 5), same as the native round-trip
    # test parses it.
    lines = text.splitlines()
    data_rows = [ln.split() for ln in lines if ln and not ln.startswith("#")]
    assert data_rows, "ANG has no data rows"
    iqs = {float(r[5]) for r in data_rows}
    assert iqs == {0.0}, f"ANG IQ column must be all zeros with no native BC; got {iqs}"


def test_ctf_uses_caller_supplied_tilt(cubic_checkpoint, tmp_path):
    """Caller passes a calibrated tilt of 71.5° — the .ctf header must
    show that value, not the 70° from the h5oina acquisition header."""
    source, cw = cubic_checkpoint
    out = tmp_path / "out"
    out.mkdir()
    ang, ctf = export_ang_ctf(
        cw.checkpoint_path, str(out),
        write_ang=False, write_ctf=True,
        source_h5_path=str(source),
        sample_tilt=71.5,
    )
    assert ctf is not None
    text = Path(ctf).read_text(encoding="utf-8")
    assert "TiltAngle\t71.5" in text, f"CTF didn't use calibrated tilt:\n{text[:400]}"
    assert "TiltAngle\t70.0" not in text


def test_ang_carries_calibrated_tilt_and_pc(cubic_checkpoint, tmp_path):
    """Post-edited .ang header must contain calibrated TILT + x/y/z-star
    lines — orix.io.save itself doesn't take detector geometry."""
    source, cw = cubic_checkpoint
    out = tmp_path / "out"
    out.mkdir()
    ang, _ = export_ang_ctf(
        cw.checkpoint_path, str(out),
        write_ang=True, write_ctf=False,
        source_h5_path=str(source),
        sample_tilt=71.5,
        pc=[0.512, 0.488, 0.673],
    )
    assert ang is not None
    text = Path(ang).read_text(encoding="utf-8")
    assert "# TILT" in text and "71.5" in text, f"ANG missing TILT:\n{text[:600]}"
    assert "# x-star" in text and "0.512" in text
    assert "# y-star" in text and "0.488" in text
    assert "# z-star" in text and "0.673" in text


def test_ang_iq_column_holds_real_bc(cubic_checkpoint, tmp_path):
    """orix.io.save writes xmap.prop['iq'] into the ANG IQ column. We
    inject real BC there before saving so the .ang has MTEX-grade
    image quality, not zeros."""
    source, cw = cubic_checkpoint
    out = tmp_path / "out"
    out.mkdir()
    ang, _ = export_ang_ctf(
        cw.checkpoint_path, str(out),
        write_ang=True, write_ctf=False,
        source_h5_path=str(source),
    )
    assert ang is not None
    # Read back and check at least one IQ value matches the BC we
    # planted in the fixture (10..21). orix's column order in ANG is
    # phi1, Phi, phi2, x, y, IQ, CI, phase, sem, ... — IQ is column 6.
    lines = Path(ang).read_text(encoding="utf-8").splitlines()
    data_rows = [ln.split() for ln in lines if ln and not ln.startswith("#")]
    assert data_rows, "ANG has no data rows"
    iqs = sorted({float(r[5]) for r in data_rows})
    # All BC values in [10..21] should appear in the IQ column.
    for expected in (10.0, 21.0):
        assert expected in iqs, f"ANG IQ column missing {expected}; got {iqs}"


def test_light_contains_full_eds_payload(tmp_path: Path):
    """The .light file must carry the COMPLETE EDS payload from the
    source h5oina — Window Integral counts, X/Y, Live Time, the full
    Header AND the per-pixel raw Spectrum dataset. Without the
    Spectrum, the EDS-driven phase-preselection workflow can't operate
    on the .light alone and has to keep the multi-GB source open."""
    src = tmp_path / "FakeWithEDS.h5oina"
    n_pixels = 12
    rng = np.random.default_rng(123)
    spectrum = rng.integers(0, 100, size=(n_pixels, 2048), dtype=np.uint16)
    al_counts = rng.integers(0, 1000, size=n_pixels, dtype=np.int32)
    with h5py.File(src, "w") as f:
        ebsd = f.create_group("1/EBSD/Data")
        ebsd.create_dataset("Band Contrast", data=np.arange(10, 22, dtype=np.uint8))
        f.create_group("1/EBSD/Header").create_dataset("Tilt Angle", data=np.float32(70))
        eds = f.create_group("1/EDS/Data")
        wi = eds.create_group("Window Integral")
        wi.create_dataset("Al", data=al_counts)
        wi.create_dataset("Fe", data=rng.integers(0, 1000, size=n_pixels, dtype=np.int32))
        eds.create_dataset("X", data=np.arange(n_pixels) * 0.5)
        eds.create_dataset("Y", data=np.zeros(n_pixels))
        eds.create_dataset("Live Time", data=np.full(n_pixels, 0.1))
        eds.create_dataset("Spectrum", data=spectrum)
        hdr = f.create_group("1/EDS/Header")
        hdr.create_dataset("Beam Voltage", data=np.float32(20))

    cw = CheckpointWriter(str(src))
    cw.init_metadata(grid_shape=(3, 4), batch_id="eds-test", method="spherical", step_size_um=0.5)
    cw.write_phase_result(
        "Al", np.full((3, 4), 0.7, dtype=np.float32),
        np.zeros((3, 4, 3), dtype=np.float32),
        {"phase_file": "Al.cif", "ci_mean": 0.7, "duration_sec": 1.0,
         "space_group": 225, "point_group": "m-3m"},
    )
    cw.compute_auto_assignment(confidence_threshold=0.0)

    out = tmp_path / "out"
    out.mkdir()
    # Default include_eds=True — we verify that the default actually
    # delivers the full payload, since "include_eds defaults true" was
    # half the user-visible fix.
    results = export_all(
        source_h5_path=str(src), checkpoint_path=cw.checkpoint_path,
        output_dir=str(out), formats=["h5_light"],
    )
    light_path = results["h5_light"]
    assert light_path

    with h5py.File(light_path, "r") as f:
        for key in (
            "EDS/Data/Window Integral/Al",
            "EDS/Data/Window Integral/Fe",
            "EDS/Data/X",
            "EDS/Data/Y",
            "EDS/Data/Live Time",
            "EDS/Data/Spectrum",
            "EDS/Header/Beam Voltage",
        ):
            assert key in f, f"missing /{key} in .light"
        # Spectrum must round-trip byte-for-byte; this is the whole
        # point of "full EDS in light" — losing a byte means losing
        # quantification accuracy.
        np.testing.assert_array_equal(np.array(f["EDS/Data/Spectrum"]), spectrum)
        np.testing.assert_array_equal(np.array(f["EDS/Data/Window Integral/Al"]), al_counts)
        assert f["Documentation"].attrs["eds_included"]
        assert f["Documentation"].attrs["eds_full"]


def test_export_refuses_zero_step(tmp_path):
    """Exporter raises rather than silently producing garbage when the
    checkpoint never recorded a step."""
    source = tmp_path / "NoStep.h5oina"
    with h5py.File(source, "w") as f:
        f.create_group("1/EBSD/Data")
    cw = CheckpointWriter(str(source))
    cw.init_metadata(grid_shape=(2, 2), batch_id="zero", method="spherical")
    rng = np.random.default_rng(0)
    cw.write_phase_result(
        "Al",
        rng.uniform(size=(2, 2)).astype(np.float32),
        rng.uniform(size=(2, 2, 3)).astype(np.float32),
        {"phase_file": "Al.cif", "ci_mean": 0.5, "duration_sec": 0.1,
         "space_group": 225, "point_group": "m-3m"},
    )
    cw.compute_auto_assignment(confidence_threshold=0.0)
    with pytest.raises(ValueError, match="step_size_um"):
        _read_step_size_from_checkpoint(cw.checkpoint_path)
