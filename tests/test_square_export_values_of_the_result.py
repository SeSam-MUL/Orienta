"""The step and the detector shape an export states come from the result, not from the open file.

A result is exported while another file can be open. The scan step (XSTEP of the ``.ang``, ``X``/``Y``
of the light ``.h5``) and the TSL pattern centre (x-star, y-star, z-star of the ``.ang``) used to be
read from the signal of that open file, so a square result of a 3.0 um scan came out with the 7.0 um
step of whatever file was open, and a z-star of 0.6 where the result's detector gives 0.8.

Rule under test: the step is the result's (``step_size_um`` of its metadata, else its square
``scan_geometry`` stamp when the steps are known); only a result that states none and whose own source
file is the open one takes the open file's step. The detector shape is the result's
(``detector_geometry``) or, when the open file is the result's source, that file's; otherwise the TSL
values are left unwritten, with a note.

Hand values. TSL from Bruker: x_T = x_B, y_T = 1 - y_B, z_T = z_B * nrows / min(nrows, ncols).
With (x, y, z)_B = (0.5, 0.7, 0.6): y_T = 0.3; shape (60, 80) gives z_T = 0.6 * 60 / 60 = 0.6, shape
(80, 60) gives z_T = 0.6 * 80 / 60 = 0.8.
"""
from __future__ import annotations

import re
import types
from pathlib import Path

import h5py
import numpy as np
import pytest

from backend.api.routes import indexing as indexing_mod
from backend.api.services.scan_geometry import ScanGeometry
# fixtures and helpers of the sibling file (the registry restore is autouse there)
from tests.test_euler_frame_follows_the_result import (  # noqa: F401
    ROWS, COLS, _export, _open_file, _result, files, restore_indexing_registry)

PC_BRUKER = [0.5, 0.7, 0.6]
NOTE = "the TSL values need the detector shape, which was not known here"


def _signal(scale, shape):
    """The stand-in of the viewer's signal: the navigation step of the open file and its detector."""
    return types.SimpleNamespace(
        axes_manager=types.SimpleNamespace(navigation_axes=[types.SimpleNamespace(scale=scale)]),
        detector=types.SimpleNamespace(shape=shape))


def _viewer(monkeypatch, path, result, *, scale=7.0, shape=(60, 80)):
    """``path`` is open in the viewer, with a signal of that file (step ``scale``, ``shape``)."""
    from backend.api.routes import ebsd_viewer
    _open_file(monkeypatch, path, result)
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda: _signal(scale, shape))


def _square(source_file, *, stamp_step=None, steps_known=True, md_step=None, detector=None):
    result = _result(source_file, hex_scan=False)
    result.metadata.pop("step_size_um", None)
    if md_step is not None:
        result.metadata["step_size_um"] = md_step
    if stamp_step is not None:
        result.metadata["scan_geometry"] = ScanGeometry.square(
            ROWS, COLS, stamp_step, steps_known=steps_known).to_dict()
    if detector is not None:
        result.metadata["detector_geometry"] = {"pat_height": detector[0], "pat_width": detector[1]}
    result.metadata["pc_at_run"] = {"pc": PC_BRUKER, "sample_tilt": 70.0, "tilt": 0.0, "azimuthal": 0.0}
    return result


def _ang(tmp_path, name="r.ang"):
    out = _export("ang", tmp_path / name)
    text = Path(out).read_text(encoding="utf-8")
    values = {}
    for key in ("XSTEP", "x-star", "y-star", "z-star"):
        m = re.search(rf"^# {key}:?\s+([-0-9.eE+]+)", text, re.M)
        values[key] = float(m.group(1)) if m else None
    return values, text


def _refused(tmp_path, fmt="ang"):
    from fastapi.testclient import TestClient
    from backend.api.main import app
    out = tmp_path / f"x.{'ang' if fmt == 'ang' else 'h5'}"
    resp = TestClient(app).post("/api/indexing/export", json={"format": fmt, "filename": str(out)})
    assert resp.status_code == 400, resp.text
    assert not out.exists()                                  # refused before a file was made
    return resp.text


# ---------------------------------------------------------------------------------- the step

def test_the_step_of_a_square_result_is_its_stamp_not_the_open_files(monkeypatch, tmp_path, files):
    """Stamp 3.0 um; the open file (another one) has 7.0 um. XSTEP is 3.0."""
    _viewer(monkeypatch, files["edax"], _square(files["oxford"], stamp_step=3.0), scale=7.0)
    values, _ = _ang(tmp_path)
    assert values["XSTEP"] == pytest.approx(3.0)


def test_the_step_a_result_states_in_its_metadata_comes_before_its_stamp(monkeypatch, tmp_path, files):
    _viewer(monkeypatch, files["edax"], _square(files["oxford"], stamp_step=3.0, md_step=2.0))
    assert _ang(tmp_path)[0]["XSTEP"] == pytest.approx(2.0)


@pytest.mark.parametrize("kind", ["no stamp", "steps not known", "hex stamp of a resampled map"])
def test_a_square_result_that_states_no_step_is_refused_with_another_file_open(
        monkeypatch, tmp_path, files, kind):
    result = _square(files["oxford"])
    if kind == "steps not known":
        result.metadata["scan_geometry"] = ScanGeometry.square(ROWS, COLS, 1.0, steps_known=False).to_dict()
    if kind == "hex stamp of a resampled map":
        # the hexagonal step (2.0) is not the step of the square map made from it: it is not taken
        result.metadata["scan_geometry"] = ScanGeometry.hex(ROWS, COLS, 2.0, odd_row_shift=0.5).to_dict()
        result.metadata["hex_resampled"] = True
    _viewer(monkeypatch, files["edax"], result, scale=7.0)
    assert "Cannot determine the scan step size" in _refused(tmp_path)


def test_the_light_h5_of_a_result_that_states_no_step_is_refused_with_another_file_open(
        monkeypatch, tmp_path, files):
    _viewer(monkeypatch, files["edax"], _square(files["oxford"]), scale=7.0)
    assert "Cannot determine the scan step size" in _refused(tmp_path, "h5_light")


def test_the_light_h5_of_a_stamped_result_states_its_step(monkeypatch, tmp_path, files):
    _viewer(monkeypatch, files["edax"], _square(files["oxford"], stamp_step=3.0), scale=7.0)
    out = _export("h5_light", tmp_path / "light.h5")
    with h5py.File(out, "r") as f:
        assert float(f["Indexing"].attrs["step_size_um"]) == pytest.approx(3.0)
        x = f["Indexing/X"][()]
    assert float(x[0, 1] - x[0, 0]) == pytest.approx(3.0)          # one column apart: one step


def test_with_its_own_file_open_a_result_without_a_step_takes_the_files(monkeypatch, tmp_path, files):
    """Nothing changes for the result's own file: the old lookup runs (7.0 um)."""
    _viewer(monkeypatch, files["oxford"], _square(files["oxford"]), scale=7.0)
    assert _ang(tmp_path)[0]["XSTEP"] == pytest.approx(7.0)


def test_with_its_own_file_open_a_stamped_result_has_the_files_step_too(monkeypatch, tmp_path, files):
    """The stamp was taken from that file's navigation axis: the same number as before."""
    _viewer(monkeypatch, files["oxford"], _square(files["oxford"], stamp_step=3.0), scale=3.0)
    assert _ang(tmp_path)[0]["XSTEP"] == pytest.approx(3.0)


def test_a_result_without_a_source_file_takes_the_open_files_step_as_before(monkeypatch, tmp_path, files):
    result = _square(files["oxford"])
    del result.metadata["source_file"]
    _viewer(monkeypatch, files["edax"], result, scale=7.0)
    assert _ang(tmp_path)[0]["XSTEP"] == pytest.approx(7.0)


# ------------------------------------------------------------------------- the detector shape

@pytest.mark.parametrize("own, detector, file_shape, z_star", [
    (False, (80, 60), (60, 80), 0.8),     # the result's shape, not the open file's (0.6)
    (False, (60, 80), (80, 60), 0.6),     # and the other way round (the file would give 0.8)
    (True, None, (80, 60), 0.8),          # own file open: its shape, as before
    (True, None, (60, 80), 0.6),
])
def test_z_star_follows_the_detector_of_the_result(
        monkeypatch, tmp_path, files, own, detector, file_shape, z_star):
    result = _square(files["oxford"], md_step=2.0, detector=detector)
    _viewer(monkeypatch, files["oxford" if own else "edax"], result, shape=file_shape)
    values, text = _ang(tmp_path)
    assert values["x-star"] == pytest.approx(0.5)
    assert values["y-star"] == pytest.approx(0.3)
    assert values["z-star"] == pytest.approx(z_star)
    assert NOTE not in text


def test_a_result_with_no_detector_shape_leaves_the_tsl_values_and_says_so(monkeypatch, tmp_path, files):
    """Another file is open and the result states no shape: no z-star is made up from that file."""
    _viewer(monkeypatch, files["edax"], _square(files["oxford"], md_step=2.0), shape=(80, 60))
    values, text = _ang(tmp_path)
    assert values["x-star"] == 0.0 and values["y-star"] == 0.0 and values["z-star"] == 0.0
    assert NOTE in text
