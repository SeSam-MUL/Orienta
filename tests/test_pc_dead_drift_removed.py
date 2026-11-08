"""The "PC Drift Analysis" control, which could not work, is gone.

The button called ``/api/pc/drift``, which asked the controller for
``analyze_pc_drift`` - a method that never existed - and so always answered with
a placeholder sentence ("requires multiple calibration patterns"), whatever it
was given. A button that answers like that is read as "the data has no drift"
or "my patterns are bad". The working drift displays (Aztec's per-pixel PC map,
the PC map of a grid calibration) are separate and stay.
"""
from pc_controller import PCController


def test_the_controller_has_no_drift_analysis():
    assert not hasattr(PCController, "analyze_pc_drift")


def test_there_is_no_drift_route():
    from backend.api.routes import pcrefinement
    paths = {getattr(r, "path", "") for r in pcrefinement.router.routes}
    assert "/drift" not in paths


def test_the_api_client_has_no_drift_call():
    from pathlib import Path
    api = (Path(__file__).resolve().parents[1] / "frontend" / "src" / "services" / "api.js").read_text(encoding="utf-8")
    assert "/api/pc/drift" not in api and "analyzeDrift" not in api
