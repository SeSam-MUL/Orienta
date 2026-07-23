"""Tests for the /api/phasemap/layer endpoint and its raw-RGBA helper.

The helper produces transparent-background pixel arrays for frontend
compositing. No scalebar, no legend, native resolution.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _make_fake_result(n_rows=8, n_cols=8, n_phases=2, with_bc=True, include_scores=True):
    """Build a minimal stub indexing result with predictable phase ids.

    Pattern: top half phase 0 (Fe), bottom half phase 1 (Al), one row
    in the middle is unindexed (-1). BC values increase linearly with column.

    When ``include_scores=False`` the underlying xmap is built with a tight
    ``spec`` that omits ``scores`` so ``hasattr(xmap, 'scores')`` returns
    False — used to test the bare-'ci' fail-loud path.
    """
    result = MagicMock()
    result.original_shape = (n_rows, n_cols)
    result.selection_mask = None
    phase_ids = np.zeros((n_rows, n_cols), dtype=np.int16)
    phase_ids[n_rows // 2:] = 1
    phase_ids[n_rows // 2 - 1] = -1   # unindexed strip

    if include_scores:
        xmap = MagicMock()
    else:
        xmap = MagicMock(spec=['phase_id', 'phases', 'rotations', 'shape', 'prop'])
    xmap.phase_id = phase_ids.ravel()
    xmap.shape = (n_rows, n_cols)
    fe_phase = MagicMock(name="Fe"); fe_phase.name = "Fe"
    al_phase = MagicMock(name="Al"); al_phase.name = "Al"
    xmap.phases = {0: fe_phase, 1: al_phase}
    rot = MagicMock()
    rot.data = np.zeros((n_rows * n_cols, 4))  # quaternions
    rot.data[:, 0] = 1.0
    xmap.rotations = rot

    if with_bc:
        bc = np.tile(np.linspace(0, 255, n_cols, dtype=np.float32), (n_rows, 1))
        xmap.prop = {"bc": bc.ravel()}
    else:
        xmap.prop = {}

    result.xmap = xmap
    result.metadata = {}
    return result


def test_compute_layer_rgba_phase_returns_HxWx4_with_alpha():
    """Phase layer must return (H,W,4) RGBA with alpha=0 on unindexed pixels."""
    from backend.api.routes.phase_map import _compute_layer_rgba
    fake = _make_fake_result(n_rows=8, n_cols=8)
    with patch("backend.api.routes.phase_map.get_last_indexing_result",
               return_value=fake), \
         patch("backend.api.routes.phase_map.get_analysis_dataset",
               return_value=None):
        arr = _compute_layer_rgba(kind="phase")
    assert arr.shape == (8, 8, 4)
    assert arr.dtype == np.uint8
    # Unindexed row (index 3) must be fully transparent
    assert (arr[3, :, 3] == 0).all(), "unindexed pixels must have alpha=0"
    # Indexed rows must be opaque
    assert (arr[0, :, 3] == 255).all()
    assert (arr[7, :, 3] == 255).all()


def test_compute_layer_rgba_bc_returns_greyscale_with_alpha():
    """BC layer is greyscale; alpha=0 on the unindexed strip."""
    from backend.api.routes.phase_map import _compute_layer_rgba
    fake = _make_fake_result(n_rows=8, n_cols=8, with_bc=True)
    with patch("backend.api.routes.phase_map.get_last_indexing_result",
               return_value=fake), \
         patch("backend.api.routes.phase_map.get_analysis_dataset",
               return_value=None):
        arr = _compute_layer_rgba(kind="bc")
    assert arr.shape == (8, 8, 4)
    # Greyscale: R == G == B per pixel (on the indexed rows)
    assert (arr[0, :, 0] == arr[0, :, 1]).all()
    assert (arr[0, :, 1] == arr[0, :, 2]).all()


def test_compute_layer_rgba_bc_sourced_via_pattern_quality_service():
    """When the xmap carries no prop['bc'], the BC layer must source the
    native Band Contrast for the result's own source_file through the central
    pattern_quality.read_native_band_contrast service (not the old path-based
    helper). We monkeypatch the service to return a known 2D array and assert
    the rendered layer is a non-empty greyscale with the expected alpha.
    """
    from backend.api.routes.phase_map import _compute_layer_rgba
    n_rows, n_cols = 8, 8
    # No prop['bc'] → forces the native-source-file fallback path.
    fake = _make_fake_result(n_rows=n_rows, n_cols=n_cols, with_bc=False)
    fake.metadata = {"source_file": "some_oxford_file.h5oina"}
    # A known BC array (increasing with column) so we can verify it was used.
    known_bc = np.tile(
        np.linspace(0, 255, n_cols, dtype=np.float64), (n_rows, 1)
    )
    called = {}

    def fake_read(source_file, n_r=None, n_c=None):
        called["source_file"] = source_file
        return known_bc

    with patch("backend.api.routes.phase_map.get_last_indexing_result",
               return_value=fake), \
         patch("backend.api.routes.phase_map.get_analysis_dataset",
               return_value=None), \
         patch("backend.api.services.pattern_quality.read_native_band_contrast",
               side_effect=fake_read):
        arr = _compute_layer_rgba(kind="bc")

    # The service was consulted with the result's own source file.
    assert called["source_file"] == "some_oxford_file.h5oina"
    assert arr.shape == (n_rows, n_cols, 4)
    assert arr.dtype == np.uint8
    # Greyscale: R == G == B per pixel.
    assert (arr[..., 0] == arr[..., 1]).all()
    assert (arr[..., 1] == arr[..., 2]).all()
    # Non-empty: the increasing BC ramp yields varying grey values.
    assert arr[..., 0].max() > arr[..., 0].min()
    # Native-source BC shows all pixels (bc_factor was None → alpha all True).
    assert (arr[..., 3] == 255).all()


def test_compute_layer_rgba_rejects_unknown_kind():
    """Unknown kind must raise an HTTPException with 400."""
    from fastapi import HTTPException
    from backend.api.routes.phase_map import _compute_layer_rgba
    fake = _make_fake_result()
    with patch("backend.api.routes.phase_map.get_last_indexing_result",
               return_value=fake), \
         patch("backend.api.routes.phase_map.get_analysis_dataset",
               return_value=None):
        with pytest.raises(HTTPException) as exc:
            _compute_layer_rgba(kind="bogus")
        assert exc.value.status_code == 400


def test_compute_layer_rgba_no_data_raises_400():
    """No indexing result and no analysis dataset → 400, not 500."""
    from fastapi import HTTPException
    from backend.api.routes.phase_map import _compute_layer_rgba
    with patch("backend.api.routes.phase_map.get_last_indexing_result",
               return_value=None), \
         patch("backend.api.routes.phase_map.get_analysis_dataset",
               return_value=None):
        with pytest.raises(HTTPException) as exc:
            _compute_layer_rgba(kind="phase")
        assert exc.value.status_code == 400


def test_compute_layer_rgba_ci_without_scores_raises_400():
    """Bare 'ci' kind on a result lacking scores must fail loudly, not return NaN."""
    from fastapi import HTTPException
    from backend.api.routes.phase_map import _compute_layer_rgba
    fake = _make_fake_result(include_scores=False)
    with patch("backend.api.routes.phase_map.get_last_indexing_result",
               return_value=fake), \
         patch("backend.api.routes.phase_map.get_analysis_dataset",
               return_value=None):
        with pytest.raises(HTTPException) as exc:
            _compute_layer_rgba(kind="ci")
        assert exc.value.status_code == 400
        assert "no ci scores" in exc.value.detail.lower()


def test_layer_endpoint_returns_base64_png():
    """GET /layer?kind=phase returns valid base64 PNG with shape metadata."""
    from fastapi.testclient import TestClient
    from backend.api.main import app
    fake = _make_fake_result(n_rows=8, n_cols=8)
    with patch("backend.api.routes.phase_map.get_last_indexing_result",
               return_value=fake), \
         patch("backend.api.routes.phase_map.get_analysis_dataset",
               return_value=None):
        client = TestClient(app)
        resp = client.get("/api/phasemap/layer", params={"kind": "phase"})
    assert resp.status_code == 200
    body = resp.json()
    assert "image" in body
    assert body["shape"] == [8, 8]
    import base64
    png_bytes = base64.b64decode(body["image"])
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n", "Invalid PNG signature"


def test_layer_endpoint_unknown_kind_returns_400():
    from fastapi.testclient import TestClient
    from backend.api.main import app
    fake = _make_fake_result()
    with patch("backend.api.routes.phase_map.get_last_indexing_result",
               return_value=fake), \
         patch("backend.api.routes.phase_map.get_analysis_dataset",
               return_value=None):
        client = TestClient(app)
        resp = client.get("/api/phasemap/layer", params={"kind": "bogus"})
    assert resp.status_code == 400


def test_layer_endpoint_no_data_returns_400():
    from fastapi.testclient import TestClient
    from backend.api.main import app
    with patch("backend.api.routes.phase_map.get_last_indexing_result",
               return_value=None), \
         patch("backend.api.routes.phase_map.get_analysis_dataset",
               return_value=None):
        client = TestClient(app)
        resp = client.get("/api/phasemap/layer", params={"kind": "phase"})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Forward NCC Diagnostic Layer (Task 11)
# ---------------------------------------------------------------------------
def _fake_result_with_diagnostics():
    """Fake spherical result with a complete forward_diagnostics block."""
    result = MagicMock()
    result.original_shape = (4, 4)
    result.selection_mask = None
    result.xmap = MagicMock()
    result.xmap.phase_id = np.zeros(16, dtype=np.int16)
    result.metadata = {
        "forward_diagnostics": {
            "ncc_map":              np.linspace(0.1, 0.9, 16).reshape(4, 4).astype(np.float32),
            "local_anomaly_map":    np.zeros((4, 4), dtype=np.float32),
            "pc_sensitivity_map":   np.linspace(0, 1, 16).reshape(4, 4).astype(np.float32),
            "pattern_residual_map": np.full((4, 4), 0.2, dtype=np.float32),
        },
    }
    return result


@pytest.mark.parametrize("kind", ["forward_ncc", "local_anomaly", "pc_sensitivity", "pattern_residual"])
def test_compute_layer_rgba_diagnostic_kinds(kind):
    from backend.api.routes.phase_map import _compute_layer_rgba
    fake = _fake_result_with_diagnostics()
    with patch("backend.api.routes.phase_map.get_last_indexing_result", return_value=fake), \
         patch("backend.api.routes.phase_map.get_analysis_dataset", return_value=None):
        rgba = _compute_layer_rgba(kind=kind)
    assert rgba.shape == (4, 4, 4)
    assert rgba.dtype == np.uint8
    # alpha non-zero somewhere (the map has values)
    assert rgba[..., 3].any()


def test_compute_layer_rgba_diagnostic_404_when_not_computed():
    from fastapi import HTTPException
    from backend.api.routes.phase_map import _compute_layer_rgba
    fake = MagicMock()
    fake.original_shape = (4, 4)
    fake.selection_mask = None
    fake.metadata = {}  # no forward_diagnostics
    with patch("backend.api.routes.phase_map.get_last_indexing_result", return_value=fake), \
         patch("backend.api.routes.phase_map.get_analysis_dataset", return_value=None):
        with pytest.raises(HTTPException) as ei:
            _compute_layer_rgba(kind="forward_ncc")
    assert ei.value.status_code == 404
    assert "not yet computed" in str(ei.value.detail)


def test_compute_layer_rgba_local_anomaly_all_zero_no_divide_warning():
    """Degenerate all-zero anomaly map must not produce divide-by-zero NaN."""
    import warnings
    from backend.api.routes.phase_map import _compute_layer_rgba
    fake = _fake_result_with_diagnostics()
    fake.metadata["forward_diagnostics"]["local_anomaly_map"] = np.zeros((4, 4), dtype=np.float32)
    with patch("backend.api.routes.phase_map.get_last_indexing_result", return_value=fake), \
         patch("backend.api.routes.phase_map.get_analysis_dataset", return_value=None):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)  # turn the warning into an error
            rgba = _compute_layer_rgba(kind="local_anomaly")
    assert rgba.shape == (4, 4, 4)
    # All values mapped to the colormap midpoint (alpha=255 because no NaN)
    assert (rgba[..., 3] == 255).all()


# ---------------------------------------------------------------------------
# Joint R+PC Refinement Layers (Phase B, Task 10)
# ---------------------------------------------------------------------------
def _fake_result_with_refinement():
    result = MagicMock()
    result.original_shape = (4, 4)
    result.selection_mask = None
    result.xmap = MagicMock()
    result.xmap.phase_id = np.zeros(16, dtype=np.int16)
    result.metadata = {
        "refinement": {
            "refined_ncc_map":         np.linspace(0.5, 0.9, 16).reshape(4, 4).astype(np.float32),
            "convergence_status_map":  np.ones((4, 4), dtype=np.float32),
            "orientation_delta_map":   np.linspace(0, 2, 16).reshape(4, 4).astype(np.float32),
            "pc_delta_x_map":          np.linspace(-1, 1, 16).reshape(4, 4).astype(np.float32),
            "pc_delta_y_map":          np.linspace(-0.5, 0.5, 16).reshape(4, 4).astype(np.float32),
            "pc_delta_l_map":          np.linspace(-50, 50, 16).reshape(4, 4).astype(np.float32),
        },
    }
    return result


@pytest.mark.parametrize("kind", [
    "refined_ncc", "convergence_status", "orientation_delta",
    "pc_delta_x", "pc_delta_y", "pc_delta_l",
])
def test_compute_layer_rgba_refinement_kinds(kind):
    from backend.api.routes.phase_map import _compute_layer_rgba
    fake = _fake_result_with_refinement()
    with patch("backend.api.routes.phase_map.get_last_indexing_result", return_value=fake), \
         patch("backend.api.routes.phase_map.get_analysis_dataset", return_value=None):
        rgba = _compute_layer_rgba(kind=kind)
    assert rgba.shape == (4, 4, 4)
    assert rgba.dtype == np.uint8
    assert rgba[..., 3].any()


def test_compute_layer_rgba_refinement_404_when_not_computed():
    from fastapi import HTTPException
    from backend.api.routes.phase_map import _compute_layer_rgba
    fake = MagicMock()
    fake.original_shape = (4, 4)
    fake.metadata = {}
    with patch("backend.api.routes.phase_map.get_last_indexing_result", return_value=fake), \
         patch("backend.api.routes.phase_map.get_analysis_dataset", return_value=None):
        with pytest.raises(HTTPException) as ei:
            _compute_layer_rgba(kind="refined_ncc")
    assert ei.value.status_code == 404


# ---------------------------------------------------------------------------
# Bug 3 — IPF rendering must work when the xmap has duplicate phase NAMES.
# ---------------------------------------------------------------------------
def _make_real_xmap(names, point_groups):
    """Build a real orix CrystalMap with the given per-phase names / point
    groups. Pixels are split evenly between the phases; ``names`` may contain
    duplicates (the spherical-indexing case Bug 3 reproduces)."""
    from orix.crystal_map import CrystalMap, PhaseList
    from orix.quaternion import Rotation

    n_phases = len(names)
    side = 4
    n = side * side
    pl = PhaseList(
        names=list(names),
        point_groups=list(point_groups),
        ids=list(range(n_phases)),
    )
    pid = np.zeros(n, dtype=int)
    per = n // n_phases
    for i in range(n_phases):
        pid[i * per:(i + 1) * per] = i
    pid[n_phases * per:] = n_phases - 1  # leftover pixels → last phase
    x = np.tile(np.arange(side), side)
    y = np.repeat(np.arange(side), side)
    # Non-identity rotations so IPF colours are not all the same corner.
    rot = Rotation.from_euler(
        np.column_stack([
            np.linspace(0, 1.0, n),
            np.linspace(0, 0.5, n),
            np.linspace(0, 0.3, n),
        ])
    )
    return CrystalMap(rotations=rot, phase_id=pid, x=x, y=y, phase_list=pl)


def test_compute_ipf_colors_handles_duplicate_phase_names():
    """Bug 3: an xmap with two phases sharing a NAME must still IPF-render.

    The old code did ``xmap[phase.name]`` which matched BOTH duplicate-named
    phases and made orix raise "command that only permits one phase".
    """
    from tools.phase_map_generator import compute_ipf_colors
    xmap = _make_real_xmap(
        names=["Mn0.5Fe0.5Al5Si0.68", "Mn0.5Fe0.5Al5Si0.68"],
        point_groups=["m-3m", "m-3m"],
    )
    # Must not raise.
    rgb = compute_ipf_colors(xmap, "Z")
    assert rgb.shape == (4, 4, 3)
    assert rgb.dtype.kind == "f"
    assert np.all((rgb >= 0.0) & (rgb <= 1.0))


def test_compute_ipf_colors_old_name_selection_would_have_failed():
    """Sanity check: confirm name-based selection on the duplicate xmap is
    exactly the failure mode Bug 3 describes — so the fix above is meaningful."""
    xmap = _make_real_xmap(
        names=["Dup", "Dup"],
        point_groups=["m-3m", "m-3m"],
    )
    with pytest.raises(ValueError, match="only permits one phase"):
        _ = xmap["Dup"].orientations


def test_layer_endpoint_ipf_with_duplicate_phase_names():
    """GET /layer?kind=ipf-z must succeed on a duplicate-name xmap."""
    from fastapi.testclient import TestClient
    from backend.api.main import app
    xmap = _make_real_xmap(
        names=["Mn0.5Fe0.5Al5Si0.68", "Mn0.5Fe0.5Al5Si0.68"],
        point_groups=["m-3m", "m-3m"],
    )
    fake = MagicMock()
    fake.original_shape = (4, 4)
    fake.selection_mask = None
    fake.xmap = xmap
    fake.metadata = {}
    with patch("backend.api.routes.phase_map.get_last_indexing_result", return_value=fake), \
         patch("backend.api.routes.phase_map.get_analysis_dataset", return_value=None):
        client = TestClient(app)
        resp = client.get("/api/phasemap/layer", params={"kind": "ipf-z"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["shape"] == [4, 4]


# ---------------------------------------------------------------------------
# Bug 4a — collision-free phase palette.
# ---------------------------------------------------------------------------
def test_build_phase_color_map_distinct_for_many_phases():
    """N phases must yield N distinct colours — no two phases share a colour,
    even well past the old 8-slot palette."""
    from backend.api.routes.phase_map import build_phase_color_map
    from orix.crystal_map import CrystalMap, PhaseList
    from orix.quaternion import Rotation
    n_phases = 19
    pl = PhaseList(
        names=[f"Phase{i}" for i in range(n_phases)],
        point_groups=["m-3m"] * n_phases,
        ids=list(range(n_phases)),
    )
    xmap = CrystalMap(
        rotations=Rotation.identity((n_phases,)),
        phase_id=np.arange(n_phases),
        x=np.arange(n_phases), y=np.zeros(n_phases),
        phase_list=pl,
    )
    cmap = build_phase_color_map(xmap)
    assert len(cmap) == n_phases
    rounded = {tuple(round(c, 4) for c in rgb) for rgb in cmap.values()}
    assert len(rounded) == n_phases, "phase palette produced duplicate colours"


def test_build_phase_color_map_distinct_for_duplicate_names():
    """Two phases with the SAME name still get DIFFERENT colours (keyed on
    unique phase id, not name)."""
    from backend.api.routes.phase_map import build_phase_color_map
    xmap = _make_real_xmap(names=["Dup", "Dup"], point_groups=["m-3m", "m-3m"])
    cmap = build_phase_color_map(xmap)
    assert len(cmap) == 2
    assert cmap[0] != cmap[1], "duplicate-named phases must not collide in colour"


def test_build_phase_color_map_stable_across_calls():
    """Colour assignment is deterministic — same result, same colours."""
    from backend.api.routes.phase_map import build_phase_color_map
    xmap = _make_real_xmap(names=["A", "B", "C"], point_groups=["m-3m"] * 3)
    assert build_phase_color_map(xmap) == build_phase_color_map(xmap)


def test_build_phase_color_map_override_wins():
    """A color_overrides entry (keyed by phase name) overrides the HSV hue."""
    from backend.api.routes.phase_map import build_phase_color_map
    xmap = _make_real_xmap(names=["Al", "Fe"], point_groups=["m-3m", "m-3m"])
    cmap = build_phase_color_map(xmap, {"Al": "#ff0000"})
    assert cmap[0] == (1.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Bug 4b — /phase-legend endpoint.
# ---------------------------------------------------------------------------
def test_phase_legend_endpoint_structure():
    """GET /phase-legend returns phase_id / phase_name / color_hex per phase."""
    from fastapi.testclient import TestClient
    from backend.api.main import app
    xmap = _make_real_xmap(names=["Al", "Fe", "Si"], point_groups=["m-3m"] * 3)
    fake = MagicMock()
    fake.original_shape = (4, 4)
    fake.selection_mask = None
    fake.xmap = xmap
    fake.metadata = {}
    with patch("backend.api.routes.phase_map.get_last_indexing_result", return_value=fake), \
         patch("backend.api.routes.phase_map.get_analysis_dataset", return_value=None):
        client = TestClient(app)
        resp = client.get("/api/phasemap/phase-legend")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "phases" in body and "unindexed" in body
    phases = body["phases"]
    assert len(phases) == 3
    assert [p["phase_name"] for p in phases] == ["Al", "Fe", "Si"]
    for p in phases:
        assert set(p.keys()) == {"phase_id", "phase_name", "color_hex"}
        assert p["color_hex"].startswith("#") and len(p["color_hex"]) == 7
    assert len({p["color_hex"] for p in phases}) == 3
    assert body["unindexed"]["color_hex"] == "#44475a"


def test_phase_legend_matches_layer_colors():
    """The legend's colours must match the colours build_phase_color_map (and
    therefore the phase LAYER) bakes in."""
    from fastapi.testclient import TestClient
    from backend.api.main import app
    from backend.api.routes.phase_map import build_phase_color_map
    xmap = _make_real_xmap(names=["Al", "Fe", "Si"], point_groups=["m-3m"] * 3)
    fake = MagicMock()
    fake.original_shape = (4, 4)
    fake.selection_mask = None
    fake.xmap = xmap
    fake.metadata = {}
    with patch("backend.api.routes.phase_map.get_last_indexing_result", return_value=fake), \
         patch("backend.api.routes.phase_map.get_analysis_dataset", return_value=None):
        client = TestClient(app)
        resp = client.get("/api/phasemap/phase-legend")
        cmap = build_phase_color_map(xmap)
    legend = {p["phase_id"]: p["color_hex"] for p in resp.json()["phases"]}
    for pid, rgb in cmap.items():
        r, g, b = (int(round(c * 255)) for c in rgb)
        assert legend[pid] == f"#{r:02x}{g:02x}{b:02x}"


def test_phase_legend_endpoint_no_data_returns_400():
    from fastapi.testclient import TestClient
    from backend.api.main import app
    with patch("backend.api.routes.phase_map.get_last_indexing_result", return_value=None), \
         patch("backend.api.routes.phase_map.get_analysis_dataset", return_value=None):
        client = TestClient(app)
        resp = client.get("/api/phasemap/phase-legend")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Per-phase IPF view (community standard: one IPF map / colour key per phase).
# ---------------------------------------------------------------------------

def _fake_result_from_xmap(xmap, shape=(4, 4)):
    fake = MagicMock()
    fake.original_shape = shape
    fake.selection_mask = None
    fake.xmap = xmap
    fake.metadata = {}
    return fake


def test_ipf_phase_filter_is_alpha_only_mask():
    """phase_filter must hide other phases via alpha WITHOUT touching RGB."""
    from backend.api.routes.phase_map import _compute_layer_rgba
    xmap = _make_real_xmap(names=["Al", "alpha"], point_groups=["m-3m", "m-3"])
    fake = _fake_result_from_xmap(xmap)
    pid_2d = np.asarray(xmap.phase_id).reshape(4, 4)
    with patch("backend.api.routes.phase_map.get_last_indexing_result",
               return_value=fake), \
         patch("backend.api.routes.phase_map.get_analysis_dataset",
               return_value=None):
        arr_all = _compute_layer_rgba(kind="ipf-z")
        arr_p0 = _compute_layer_rgba(kind="ipf-z", phase_filter=0)
        arr_p1 = _compute_layer_rgba(kind="ipf-z", phase_filter=1)
    # Unfiltered: every indexed pixel opaque.
    assert (arr_all[..., 3] == 255).all()
    # Filter phase 0: exactly the phase-0 pixels stay opaque.
    assert ((arr_p0[..., 3] == 255) == (pid_2d == 0)).all()
    # Filter phase 1: the complement.
    assert ((arr_p1[..., 3] == 255) == (pid_2d == 1)).all()
    # Alpha-only contract: the colour math is untouched by the filter.
    assert np.array_equal(arr_p0[..., :3], arr_all[..., :3])
    assert np.array_equal(arr_p1[..., :3], arr_all[..., :3])


def test_ipf_phase_filter_composes_with_grain_stabilized():
    """The filter is applied after colour computation, so it must work
    identically for the grain-consistent (v2) colouring path."""
    from backend.api.routes.phase_map import _compute_layer_rgba
    xmap = _make_real_xmap(names=["Al", "alpha"], point_groups=["m-3m", "m-3"])
    fake = _fake_result_from_xmap(xmap)
    pid_2d = np.asarray(xmap.phase_id).reshape(4, 4)
    with patch("backend.api.routes.phase_map.get_last_indexing_result",
               return_value=fake), \
         patch("backend.api.routes.phase_map.get_analysis_dataset",
               return_value=None):
        arr = _compute_layer_rgba(kind="ipf-z", phase_filter=1,
                                  grain_stabilized=True)
    assert ((arr[..., 3] == 255) == (pid_2d == 1)).all()


def test_layer_endpoint_accepts_phase_filter_param():
    """GET /layer?kind=ipf-z&phase_filter=1 → 200 and a PNG whose alpha
    matches the phase-1 mask. phase_filter=-1 keeps the legacy behaviour."""
    import base64 as _b64
    import io as _io
    from PIL import Image
    from fastapi.testclient import TestClient
    from backend.api.main import app
    xmap = _make_real_xmap(names=["Al", "alpha"], point_groups=["m-3m", "m-3"])
    fake = _fake_result_from_xmap(xmap)
    pid_2d = np.asarray(xmap.phase_id).reshape(4, 4)
    with patch("backend.api.routes.phase_map.get_last_indexing_result",
               return_value=fake), \
         patch("backend.api.routes.phase_map.get_analysis_dataset",
               return_value=None):
        client = TestClient(app)
        resp = client.get("/api/phasemap/layer",
                          params={"kind": "ipf-z", "phase_filter": 1})
        resp_all = client.get("/api/phasemap/layer",
                              params={"kind": "ipf-z", "phase_filter": -1})
    assert resp.status_code == 200, resp.text
    assert resp_all.status_code == 200, resp_all.text
    png = _b64.b64decode(resp.json()["image"])
    arr = np.array(Image.open(_io.BytesIO(png)).convert("RGBA"))
    assert ((arr[..., 3] == 255) == (pid_2d == 1)).all()
    png_all = _b64.b64decode(resp_all.json()["image"])
    arr_all = np.array(Image.open(_io.BytesIO(png_all)).convert("RGBA"))
    assert (arr_all[..., 3] == 255).all()


def test_group_phases_by_symmetry_only_phase_id():
    """only_phase_id restricts the IPF-key grouping to one phase, so the
    on-screen colour key matches a phase-filtered IPF layer."""
    from backend.api.routes.phase_map import _group_phases_by_symmetry
    xmap = _make_real_xmap(names=["Al", "alpha"], point_groups=["m-3m", "m-3"])
    groups_all = _group_phases_by_symmetry(xmap)
    assert len(groups_all) == 2  # two distinct Laue classes
    groups_p1 = _group_phases_by_symmetry(xmap, only_phase_id=1)
    assert len(groups_p1) == 1
    (laue_name, info), = groups_p1.items()
    assert info["phases"] == ["alpha"]
    groups_p0 = _group_phases_by_symmetry(xmap, only_phase_id=0)
    assert len(groups_p0) == 1
    (_, info0), = groups_p0.items()
    assert info0["phases"] == ["Al"]


def test_ipf_key_endpoint_accepts_phase_filter():
    """GET /ipf-key?phase_filter=<pid> → 200 with a key restricted to that
    phase (rendered PNG differs from the all-phases key)."""
    from fastapi.testclient import TestClient
    from backend.api.main import app
    xmap = _make_real_xmap(names=["Al", "alpha"], point_groups=["m-3m", "m-3"])
    fake = _fake_result_from_xmap(xmap)
    with patch("backend.api.routes.phase_map.get_last_indexing_result",
               return_value=fake), \
         patch("backend.api.routes.phase_map.get_analysis_dataset",
               return_value=None):
        client = TestClient(app)
        resp_all = client.get("/api/phasemap/ipf-key", params={"direction": "Z"})
        resp_p0 = client.get("/api/phasemap/ipf-key",
                             params={"direction": "Z", "phase_filter": 0})
    assert resp_all.status_code == 200, resp_all.text
    assert resp_p0.status_code == 200, resp_p0.text
    assert resp_p0.json()["image"] != resp_all.json()["image"]


def test_ipf_key_endpoint_accepts_orientation():
    """orientation=vertical stacks the key triangles in a column (side panel
    next to the map) — must render and differ from the horizontal row."""
    from fastapi.testclient import TestClient
    from backend.api.main import app
    xmap = _make_real_xmap(names=["Al", "alpha"], point_groups=["m-3m", "m-3"])
    fake = _fake_result_from_xmap(xmap)
    with patch("backend.api.routes.phase_map.get_last_indexing_result",
               return_value=fake), \
         patch("backend.api.routes.phase_map.get_analysis_dataset",
               return_value=None):
        client = TestClient(app)
        resp_h = client.get("/api/phasemap/ipf-key", params={"direction": "Z"})
        resp_v = client.get("/api/phasemap/ipf-key",
                            params={"direction": "Z", "orientation": "vertical"})
    assert resp_h.status_code == 200, resp_h.text
    assert resp_v.status_code == 200, resp_v.text
    assert resp_v.json()["image"] != resp_h.json()["image"]
