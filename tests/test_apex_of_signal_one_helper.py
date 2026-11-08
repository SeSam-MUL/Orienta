"""One reader of ``Signal.apex``, in a service, for the viewer route, the batch route and the batch manager.

The viewer route used to own it, and the batch route imported the private name from there while the batch
manager kept a second copy. A fix to one would not have reached the others."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from backend.api.services import scan_geometry

BACKEND = Path(__file__).resolve().parents[1] / "backend"
APEX = {"h5_path": "session.h5", "measurement_id": "GUID-A", "measurement_name": "Area 1 / Map 1", "pairing": "guid"}


def _signal(apex=None):
    import kikuchipy as kp
    signal = kp.signals.EBSD(np.zeros((2, 2, 3, 3), dtype=np.uint8))
    if apex is not None:
        signal.metadata.set_item("Signal.apex", apex)
    return signal


def test_the_helper_gives_the_pairing_as_plain_data():
    got = scan_geometry.apex_of_signal(_signal(APEX))
    assert got == APEX and type(got) is dict


def test_the_helper_gives_none_for_a_file_without_a_pairing_and_for_a_stand_in():
    assert scan_geometry.apex_of_signal(_signal()) is None
    assert scan_geometry.apex_of_signal(object()) is None


def test_no_other_module_defines_its_own_reader_of_signal_apex():
    """Every module under backend/ that reads ``Signal.apex`` goes through the service."""
    own = re.compile(r"^\s*def\s+_?apex_of_signal\b", re.M)
    defs = [p.relative_to(BACKEND).as_posix() for p in BACKEND.rglob("*.py")
            if own.search(p.read_text(encoding="utf-8"))]
    assert defs == ["api/services/scan_geometry.py"], defs


def test_the_batch_manager_and_both_routes_use_the_service_helper():
    from backend.api.routes import batch_v2, ebsd_viewer
    from backend.api.services import batch_manager
    for module in (batch_manager, batch_v2, ebsd_viewer):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "scan_geometry" in source and "apex_of_signal(" in source, module.__name__
        assert not hasattr(module, "_apex_of_signal"), module.__name__
