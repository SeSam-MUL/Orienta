"""One meaning of the detector azimuthal angle on every path, on every kikuchipy.

kikuchipy 0.12.0 reversed what ``EBSDDetector.azimuthal`` means. Orienta keeps
the EMsoft / kikuchipy 0.11 meaning everywhere, and the single place that
translates is ``backend.api.services.detector_convention``. These tests hold
that down at three levels:

* the helper itself (version rules, no mutation, the exact inverse),
* the pattern a kikuchipy projection gives against Orienta's own GPU projection,
* every place that hands a detector to kikuchipy for projection, found by
  parsing the source, so a new call that forgets the helper fails here.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.services import detector_convention as dc  # noqa: E402


def _kikuchipy_release() -> tuple:
    return dc._installed_release()


def _detector(azimuthal=0.0, **kw):
    from kikuchipy.detectors import EBSDDetector
    args = dict(shape=(48, 64), pc=(0.43, 0.61, 0.52), sample_tilt=70.0, tilt=10.0,
                azimuthal=azimuthal, convention="bruker")
    args.update(kw)
    return EBSDDetector(**args)


# --------------------------------------------------------------------------- #
# The helper
# --------------------------------------------------------------------------- #

def test_below_0_12_the_detector_is_handed_over_as_it_is(monkeypatch):
    monkeypatch.setattr(dc, "_installed_release", lambda: (0, 11, 3))
    det = _detector(5.0)
    assert dc.for_kikuchipy_projection(det) is det
    assert det.azimuthal == 5.0


@pytest.mark.parametrize("release", [(0, 12, 1), (0, 12, 2), (0, 13, 1)])
@pytest.mark.parametrize("azimuthal", [5.0, -5.0, 0.0])
def test_from_0_12_1_a_copy_with_the_opposite_sign_is_handed_over(
        monkeypatch, release, azimuthal):
    monkeypatch.setattr(dc, "_installed_release", lambda: release)
    det = _detector(azimuthal)
    pc_before = det.pc.copy()
    out = dc.for_kikuchipy_projection(det)
    assert out is not det
    assert out.azimuthal == -azimuthal
    # The caller's detector is untouched, in every field.
    assert det.azimuthal == azimuthal
    assert np.array_equal(det.pc, pc_before)
    # Nothing else differs: the sign is the whole difference between the two geometries.
    assert (out.shape, out.tilt, out.sample_tilt, out.px_size, out.binning) == \
        (det.shape, det.tilt, det.sample_tilt, det.px_size, det.binning)
    assert np.array_equal(out.pc, det.pc)
    assert out.pc is not det.pc            # no shared memory with the caller's detector


def test_the_copy_does_not_alias_the_callers_azimuthal(monkeypatch):
    monkeypatch.setattr(dc, "_installed_release", lambda: (0, 13, 1))
    det = _detector(5.0)
    out = dc.for_kikuchipy_projection(det)
    out.azimuthal = 9.0
    assert det.azimuthal == 5.0


def test_kikuchipy_0_12_0_is_refused(monkeypatch):
    monkeypatch.setattr(dc, "_installed_release", lambda: (0, 12, 0))
    with pytest.raises(dc.UnsupportedKikuchipyVersion, match="0.12.0"):
        dc.for_kikuchipy_projection(_detector(5.0))


@pytest.mark.parametrize("text,expected", [
    ("0.11.3", (0, 11, 3)), ("0.13.1", (0, 13, 1)), ("0.12.1.dev3", (0, 12, 1)),
    ("0.13.1+local", (0, 13, 1)), ("0.12", (0, 12, 0)),
])
def test_the_installed_version_is_read_from_the_release_part(monkeypatch, text, expected):
    import kikuchipy
    monkeypatch.setattr(kikuchipy, "__version__", text)
    assert dc._installed_release() == expected


def test_the_helper_is_the_inverse_of_the_version_change_in_the_installed_kikuchipy():
    """On the real kikuchipy: below 0.12 the sign is kept, from 0.12.1 it is flipped."""
    det = _detector(5.0)
    out = dc.for_kikuchipy_projection(det)
    assert out.azimuthal == (5.0 if _kikuchipy_release() < (0, 12, 0) else -5.0)
    assert det.azimuthal == 5.0


@pytest.mark.parametrize("zero", [0.0, -0.0])
def test_a_zero_angle_stays_a_plain_zero_not_negative_zero(monkeypatch, zero):
    """-0.0 would be written into saved h5 headers."""
    monkeypatch.setattr(dc, "_installed_release", lambda: (0, 13, 1))
    out = dc.for_kikuchipy_projection(_detector(zero))
    assert out.azimuthal == 0.0
    assert not np.signbit(out.azimuthal)
    assert not np.signbit(np.float64(out.azimuthal))


def test_a_converted_detector_is_refused_a_second_time(monkeypatch):
    """Converting twice flips back to Orienta's sign and projects with the wrong angle."""
    monkeypatch.setattr(dc, "_installed_release", lambda: (0, 13, 1))
    det = _detector(5.0)
    once = dc.for_kikuchipy_projection(det)
    assert getattr(once, dc.CONVERTED_MARK) is True
    assert not hasattr(det, dc.CONVERTED_MARK)         # the caller's detector is never marked
    with pytest.raises(dc.DetectorAlreadyConverted, match="already converted"):
        dc.for_kikuchipy_projection(once)
    # The original still converts, as often as asked.
    assert dc.for_kikuchipy_projection(det).azimuthal == -5.0


def test_the_mark_survives_kikuchipys_own_deepcopy(monkeypatch):
    """A copy of a converted detector is still a converted detector."""
    monkeypatch.setattr(dc, "_installed_release", lambda: (0, 13, 1))
    once = dc.for_kikuchipy_projection(_detector(5.0))
    copied = once.deepcopy()
    assert getattr(copied, dc.CONVERTED_MARK, False) is True
    with pytest.raises(dc.DetectorAlreadyConverted):
        dc.for_kikuchipy_projection(copied)


def test_below_0_12_nothing_is_marked_and_a_second_pass_is_harmless(monkeypatch):
    monkeypatch.setattr(dc, "_installed_release", lambda: (0, 11, 3))
    det = _detector(5.0)
    again = dc.for_kikuchipy_projection(dc.for_kikuchipy_projection(det))
    assert again is det and not hasattr(det, dc.CONVERTED_MARK)
    assert det.azimuthal == 5.0


def test_double_wrap_on_the_installed_kikuchipy():
    """Whatever kikuchipy is installed: a second pass either raises or changes nothing."""
    det = _detector(5.0)
    once = dc.for_kikuchipy_projection(det)
    if _kikuchipy_release() < (0, 12, 0):
        assert dc.for_kikuchipy_projection(once).azimuthal == 5.0
    else:
        with pytest.raises(dc.DetectorAlreadyConverted):
            dc.for_kikuchipy_projection(once)


@pytest.fixture
def fresh_warning(monkeypatch):
    monkeypatch.setattr(dc, "_warned", False)


def _azimuthal_warnings(caplog):
    return [r for r in caplog.records
            if r.name == dc.logger.name and r.levelname == "WARNING"
            and "azimuthal angle is" in r.getMessage()]


@pytest.mark.parametrize("release", [(0, 11, 3), (0, 13, 1)])
def test_a_non_zero_angle_is_warned_about_once_per_process(
        monkeypatch, caplog, fresh_warning, release):
    monkeypatch.setattr(dc, "_installed_release", lambda: release)
    with caplog.at_level("WARNING", logger=dc.logger.name):
        dc.for_kikuchipy_projection(_detector(4.0))
        dc.for_kikuchipy_projection(_detector(4.0))
        dc.for_kikuchipy_projection(_detector(-9.0))
    found = _azimuthal_warnings(caplog)
    assert len(found) == 1, [r.getMessage() for r in found]
    assert found[0].getMessage() == dc.azimuthal_angle_notice(4.0)


@pytest.mark.parametrize("release", [(0, 11, 3), (0, 13, 1)])
def test_a_zero_angle_is_never_warned_about(monkeypatch, caplog, fresh_warning, release):
    monkeypatch.setattr(dc, "_installed_release", lambda: release)
    with caplog.at_level("WARNING", logger=dc.logger.name):
        dc.for_kikuchipy_projection(_detector(0.0))
        dc.for_kikuchipy_projection(_detector(0.0))
    assert _azimuthal_warnings(caplog) == []
    # A zero angle does not use up the once.
    with caplog.at_level("WARNING", logger=dc.logger.name):
        dc.for_kikuchipy_projection(_detector(3.0))
    assert len(_azimuthal_warnings(caplog)) == 1


# --------------------------------------------------------------------------- #
# Direction cosines: Orienta's own math (EMsoft) against the installed kikuchipy
# --------------------------------------------------------------------------- #

_GEOMETRIES = [
    ((60, 60), (0.5, 0.5, 0.5), 70.0, 0.0),
    ((48, 64), (0.43, 0.61, 0.52), 70.0, 10.0),
    ((60, 80), (0.55, 0.40, 0.65), 68.0, 5.0),
    ((96, 80), (0.35, 0.66, 0.80), 75.7, 3.44),
]


@pytest.mark.parametrize("azimuthal", [-7.0, -0.5, 0.0, 0.5, 7.0, 23.0])
@pytest.mark.parametrize("shape,pc,sample_tilt,tilt", _GEOMETRIES)
def test_kikuchipy_through_the_helper_agrees_with_orientas_own_direction_cosines(
        shape, pc, sample_tilt, tilt, azimuthal):
    """Needs no GPU: Orienta's torch copy runs on the CPU too."""
    torch = pytest.importorskip("torch")
    from backend.dict_gpu._pcadi._projection.direction_cosines import (
        compute_direction_cosines,
    )
    from kikuchipy.signals.util._master_pattern import _get_direction_cosines_from_detector

    det = _detector(azimuthal, shape=shape, pc=pc, sample_tilt=sample_tilt, tilt=tilt)
    ref = np.asarray(_get_direction_cosines_from_detector(dc.for_kikuchipy_projection(det)))
    ref = ref.reshape(shape[0], shape[1], 3)
    ours = compute_direction_cosines(det, device="cpu", dtype=torch.float64).numpy()
    np.testing.assert_allclose(ours, ref, atol=1e-9, rtol=0)


def test_without_the_helper_a_flipped_kikuchipy_would_disagree():
    """The control: on kikuchipy >= 0.12.1 the unconverted detector is 2*w off."""
    if _kikuchipy_release() < (0, 12, 1):
        pytest.skip("kikuchipy < 0.12 reads the angle the way Orienta does")
    torch = pytest.importorskip("torch")
    from backend.dict_gpu._pcadi._projection.direction_cosines import (
        compute_direction_cosines,
    )
    from kikuchipy.signals.util._master_pattern import _get_direction_cosines_from_detector

    det = _detector(7.0)
    raw = np.asarray(_get_direction_cosines_from_detector(det)).reshape(48, 64, 3)
    ours = compute_direction_cosines(det, device="cpu", dtype=torch.float64).numpy()
    cos = np.clip((raw * ours).sum(-1), -1, 1)
    assert np.degrees(np.arccos(cos)).max() > 10.0


# --------------------------------------------------------------------------- #
# Patterns: kikuchipy get_patterns through the helper against Orienta's GPU path
# --------------------------------------------------------------------------- #

def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


@pytest.mark.gpu
@pytest.mark.parametrize("azimuthal", [5.0, -5.0, 0.0])
@pytest.mark.parametrize("tilt", [0.0, 10.0])
def test_kikuchipy_patterns_through_the_helper_equal_orientas_gpu_patterns(azimuthal, tilt):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    import kikuchipy as kp
    from orix.quaternion import Rotation
    from backend.dict_gpu._pcadi.master_to_dict import gpu_master_to_dict

    mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both")
    rot = Rotation.from_euler(np.deg2rad([[10, 20, 30], [120, 40, 250], [300, 80, 15]]))
    det = _detector(azimuthal, shape=(60, 60), pc=(0.47, 0.62, 0.55), tilt=tilt)

    kp_pat = np.asarray(
        mp.get_patterns(rotations=rot, detector=dc.for_kikuchipy_projection(det),
                        energy=20, compute=True).data, dtype=np.float64).reshape(-1, 60, 60)
    gpu = gpu_master_to_dict(mp, rot, det, energy=20.0, device="cuda",
                             dtype=torch.float32).cpu().numpy().astype(np.float64)
    # Path B is the kikuchipy projection behind its own call site.
    path_b = gpu_master_to_dict(mp, rot, det, energy=20.0, device="cuda",
                                dtype=torch.float32, _use_path_b=True
                                ).cpu().numpy().astype(np.float64)
    for i in range(kp_pat.shape[0]):
        assert _ncc(kp_pat[i], gpu[i]) > 0.9999
        assert _ncc(kp_pat[i], path_b[i]) > 0.9999
        assert _ncc(path_b[i], gpu[i]) > 0.9999


@pytest.mark.gpu
def test_the_two_signs_give_different_patterns_so_the_comparison_can_fail():
    """Control for the test above: +5 and -5 are not the same pattern."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    import kikuchipy as kp
    from orix.quaternion import Rotation
    from backend.dict_gpu._pcadi.master_to_dict import gpu_master_to_dict

    mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both")
    rot = Rotation.from_euler(np.deg2rad([[10, 20, 30]]))
    plus = gpu_master_to_dict(mp, rot, _detector(5.0, shape=(60, 60)), energy=20.0,
                              device="cuda").cpu().numpy().astype(np.float64)[0]
    minus = gpu_master_to_dict(mp, rot, _detector(-5.0, shape=(60, 60)), energy=20.0,
                               device="cuda").cpu().numpy().astype(np.float64)[0]
    assert _ncc(plus, minus) < 0.9


# --------------------------------------------------------------------------- #
# The notice
# --------------------------------------------------------------------------- #

def test_no_notice_for_a_zero_angle():
    assert dc.azimuthal_notice(_detector(0.0)) is None
    calls = []
    dc.report_azimuthal(_detector(0.0), calls.append)
    assert calls == []


def test_the_notice_says_value_convention_and_what_ignores_the_angle():
    msg = dc.azimuthal_notice(_detector(-3.5))
    assert "-3.5" in msg
    assert "EMsoft" in msg
    assert "not been verified" in msg
    assert "Hough" in msg and "spherical" in msg
    assert "\n" not in msg                      # one line


def test_report_goes_through_the_progress_channel_exactly_once():
    calls = []
    dc.report_azimuthal(_detector(2.0), calls.append)
    assert len(calls) == 1 and calls[0].startswith("WARNING: Detector azimuthal angle is 2")


def test_dictionary_generation_announces_a_non_zero_angle_once():
    from backend.dictionary_gpu.pipeline import _announce_azimuthal
    seen = []
    _announce_azimuthal(4.0, lambda f, m: seen.append((f, m)))
    assert len(seen) == 1 and "azimuthal angle is 4" in seen[0][1]
    seen.clear()
    _announce_azimuthal(0.0, lambda f, m: seen.append((f, m)))
    assert seen == []


def _innermost_function(tree, target):
    best = None
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(n is target for n in ast.walk(fn)):
                best = fn
    return best


#: The calls that run an indexing method over the patterns.
_DISPATCH = {"run_per_phase_indexing", "spherical_gpu_index_patterns",
             "hough_index_patterns", "dictionary_index_patterns",
             "spherical_index_patterns"}


def _call_name(node):
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def test_the_indexing_routes_report_the_angle_before_the_work_starts():
    src = (ROOT / "backend/api/routes/indexing.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "report_azimuthal"]
    channels = sorted(ast.unparse(c.args[1]) for c in calls)
    assert channels == ["_log", "_progress"], channels
    for call in calls:
        fn = _innermost_function(tree, call)
        dispatch = [n.lineno for n in ast.walk(fn)
                    if isinstance(n, ast.Call) and _call_name(n) in _DISPATCH]
        assert dispatch, f"{fn.name}: no indexing dispatch found; the scan is blind"
        assert call.lineno < min(dispatch), (
            f"{fn.name}: report_azimuthal at line {call.lineno} comes after the first "
            f"indexing call at line {min(dispatch)}")


def test_the_dictionary_pipeline_announces_the_angle_on_both_backends():
    """Delete either call and the user is not told on that backend."""
    src = (ROOT / "backend/dictionary_gpu/pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    work = {"has_cuda", "load_master_pattern", "EBSDDetector", "generate_dictionary"}
    announced = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) == "_announce_azimuthal":
            fn = _innermost_function(tree, node)
            if fn.name == "_announce_azimuthal":
                continue
            announced[fn.name] = node.lineno
            first_work = min(n.lineno for n in ast.walk(fn)
                             if isinstance(n, ast.Call) and _call_name(n) in work)
            assert node.lineno < first_work, (
                f"{fn.name}: the angle is announced at line {node.lineno}, after the "
                f"work starts at line {first_work}")
    assert set(announced) == {"generate_dictionary_gpu", "generate_dictionary_cpu"}, announced


# --------------------------------------------------------------------------- #
# Every place that hands a detector to kikuchipy for projection
# --------------------------------------------------------------------------- #

#: Calls into kikuchipy that project or simulate with a detector.
_PROJECTION_CALLS = {"get_patterns", "on_detector", "refine_orientation",
                     "refine_projection_center", "refine_orientation_projection_center"}

#: Application code lives in the repository root and in these folders.
_APPLICATION_DIRS = ("backend", "simulation", "analysis", "tools", "Ai_Ml")

#: The sites that exist today, ``file -> number of projection calls``. A positive
#: anchor: a scan that finds nothing would also pass, so this one fails when a
#: call disappears from the scan (renamed, moved, wrapped in something the scan
#: cannot see).
_KNOWN_SITES = {
    "indexing_controller.py": 3,
    "simulation/dictionary_generator.py": 2,
    "backend/dict_gpu/_pcadi/master_to_dict.py": 1,
    "backend/api/routes/pcrefinement.py": 1,
}

_HELPER = "for_kikuchipy_projection"


def _application_trees():
    listing = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, check=True,
                             capture_output=True, text=True).stdout.splitlines()
    assert len(listing) > 100, "git ls-files found no project: is this a checkout?"
    for rel in listing:
        if "/" in rel and rel.split("/", 1)[0] not in _APPLICATION_DIRS:
            continue
        path = ROOT / rel
        if not path.is_file():
            continue
        try:
            yield rel, ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue


def _is_helper_call(node) -> bool:
    return (isinstance(node, ast.Call)
            and ((isinstance(node.func, ast.Name) and node.func.id == _HELPER)
                 or (isinstance(node.func, ast.Attribute) and node.func.attr == _HELPER)))


def _enclosing_function(tree, target):
    best = None
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(n is target for n in ast.walk(fn)):
                # innermost wins: ast.walk is breadth-first, so later = deeper
                best = fn
    return best


def _detector_expression(call, scope):
    """The expression passed as the detector of a projection call, or None."""
    for kw in call.keywords:
        if kw.arg == "detector":
            return kw.value
    if isinstance(call.func, ast.Attribute) and call.func.attr == "on_detector" and call.args:
        return call.args[0]
    # refine_orientation(**refine_kwargs) with ``refine_kwargs = dict(detector=...)``
    for kw in call.keywords:
        if kw.arg is None and isinstance(kw.value, ast.Name) and scope is not None:
            for n in ast.walk(scope):
                if (isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                        and isinstance(n.value.func, ast.Name) and n.value.func.id == "dict"
                        and any(isinstance(t, ast.Name) and t.id == kw.value.id
                                for t in n.targets)):
                    for k in n.value.keywords:
                        if k.arg == "detector":
                            return k.value
    return None


def _goes_through_the_helper(expr, scope) -> bool:
    if expr is None:
        return False
    if _is_helper_call(expr):
        return True
    if isinstance(expr, ast.Name) and scope is not None:
        assigns = [n for n in ast.walk(scope)
                   if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == expr.id for t in n.targets)]
        return bool(assigns) and all(_is_helper_call(a.value) for a in assigns)
    return False


def _projection_calls():
    for rel, tree in _application_trees():
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in _PROJECTION_CALLS):
                yield rel, tree, node


def test_every_kikuchipy_projection_gets_its_detector_from_the_helper():
    unwrapped = []
    for rel, tree, call in _projection_calls():
        scope = _enclosing_function(tree, call)
        expr = _detector_expression(call, scope)
        if not _goes_through_the_helper(expr, scope):
            unwrapped.append(f"{rel}:{call.lineno} {ast.unparse(call.func)}"
                             f"(detector={ast.unparse(expr) if expr is not None else '?'})")
    assert not unwrapped, (
        "kikuchipy reads the azimuthal angle with its own, version-dependent sign; "
        "pass the detector through backend.api.services.detector_convention."
        "for_kikuchipy_projection: " + "; ".join(unwrapped))


def test_the_scan_still_sees_the_known_sites():
    found: dict[str, int] = {}
    for rel, _tree, _call in _projection_calls():
        found[rel] = found.get(rel, 0) + 1
    for rel, minimum in _KNOWN_SITES.items():
        assert found.get(rel, 0) >= minimum, (rel, found.get(rel, 0), minimum)
    assert set(found) <= set(_KNOWN_SITES), (
        "a new file projects through kikuchipy; add it to _KNOWN_SITES after "
        f"checking it uses the helper: {sorted(set(found) - set(_KNOWN_SITES))}")


def test_orientas_own_projection_does_not_convert_the_angle():
    """Converting before Orienta's own math, or twice, would flip it back."""
    for rel in ("backend/dict_gpu/_pcadi/_projection/direction_cosines.py",
                "backend/dictionary_gpu/detector.py",
                "backend/dictionary_gpu/pipeline.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert _HELPER not in src, rel


def test_the_helper_would_be_found_by_the_scan_if_it_were_missing():
    """The scan is not blind: the same call without the helper is reported."""
    tree = ast.parse("def f(m, d, r):\n    return m.get_patterns(rotations=r, detector=d)\n")
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call))
    scope = _enclosing_function(tree, call)
    assert not _goes_through_the_helper(_detector_expression(call, scope), scope)
    ok = ast.parse(f"def f(m, d, r):\n    x = {_HELPER}(d)\n"
                   "    return m.get_patterns(rotations=r, detector=x)\n")
    call = next(n for n in ast.walk(ok) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute))
    scope = _enclosing_function(ok, call)
    assert _goes_through_the_helper(_detector_expression(call, scope), scope)


# --------------------------------------------------------------------------- #
# Saved dictionaries carry Orienta's value of the angle
# --------------------------------------------------------------------------- #

def _header_azimuth(path) -> float:
    import h5py
    found = []

    def visit(name, obj):
        if name.endswith("Header/azimuth_angle"):
            found.append(float(np.ravel(obj[()])[0]))

    with h5py.File(path, "r") as f:
        f.visititems(visit)
    assert len(found) == 1, found
    return found[0]


@pytest.mark.parametrize("chunk", [10_000, 5], ids=["unchunked", "chunked"])
@pytest.mark.parametrize("azimuthal", [4.0, -4.0, 0.0])
def test_the_saved_dictionary_header_carries_orientas_angle_not_kikuchipys(
        tmp_path, monkeypatch, chunk, azimuthal):
    """Header azimuth_angle == sidecar azimuthal == the angle the user gave."""
    import json
    import kikuchipy as kp
    import backend.dictionary_gpu.pipeline as pipe

    mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both")
    monkeypatch.setattr(kp, "load", lambda *a, **k: mp)
    monkeypatch.setattr(pipe, "CPU_CHUNK_SIZE", chunk)

    out = tmp_path / "dict.h5"
    result = pipe.generate_dictionary_cpu(
        master_path=str(tmp_path / "ni_master.h5"), detector_shape=(20, 20),
        pc=(0.5, 0.5, 0.5), sample_tilt=70.0, energy_kv=20.0, resolution_deg=30.0,
        detector_tilt_deg=0.0, azimuthal_deg=azimuthal, output_path=str(out))
    # The two cases really are the two branches of generate_dictionary.
    assert (result.metadata.n_orientations > chunk) == (chunk == 5)
    sidecar = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    header = _header_azimuth(out)
    assert sidecar["azimuthal"] == azimuthal
    assert header == azimuthal
    assert not (azimuthal == 0.0 and np.signbit(header))
