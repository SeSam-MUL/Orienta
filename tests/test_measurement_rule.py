"""The one rule that names the measurement a file reader reads, and what a refusal answers.

Same synthetic two-measurement file as every other ``test_measurement_*`` module: two
measurements on the same 2x3 grid whose channels differ on every pixel.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from backend.api.services.load_choices import LoadChoiceError, LoadVariant
from tests.two_measurements import *  # noqa: F401,F403  (constants and fixtures)
from tests.two_measurements import _grid, _group, _xmap  # noqa: F401


def test_the_active_variant_is_the_datasets_and_only_for_its_file(viewer, two, tmp_path):
    from backend.api.routes import ebsd_viewer as ev
    viewer(V2)
    assert ev.active_variant() == V2
    assert ev.active_variant(two) == V2
    assert ev.active_variant(str(tmp_path / "other.h5oina")) is None


def test_one_rule_names_the_group_to_read(two):
    from backend.api.services.load_choices import measurement_groups
    with h5py.File(two, "r") as f:
        # no variant: every top-level key in file order, as the readers walked before
        assert measurement_groups(f, None) == list(f.keys()) == ["1", "2", "Format Version", "Manufacturer"]
        assert measurement_groups(f, None, default="1") == ["1"]
        assert measurement_groups(f, V2) == ["2"] == measurement_groups(f, V2, default="1")
        assert measurement_groups(f, LoadVariant(None, "processed")) == list(f.keys())
        with pytest.raises(LoadChoiceError) as e:
            measurement_groups(f, V7)
        assert e.value.code == "variantNotSupported"


def test_a_stamp_is_read_the_same_way_everywhere():
    from backend.api.services import variant_registry as vr
    from backend.api.services.load_choices import variant_from_stamp
    assert vr.variant_of_result(SimpleNamespace(metadata={})) is None
    assert vr.variant_of_result(None) is None
    both = SimpleNamespace(metadata={"load_variant": V1.to_dict(), "indexed_variant": V2_RAW.to_dict()})
    assert vr.variant_of_result(both) == V2_RAW                 # an imported file's own stamp wins
    assert vr.stamp_of_result(both) == V2_RAW.to_dict()
    assert variant_from_stamp({"measurement": "2"}) is None     # no pattern source: unreadable = default
    assert variant_from_stamp({"measurement": "2", "pattern_source": "nonsense"}) is None


def test_an_unhandled_refusal_in_a_route_answers_400_with_the_code(two):
    """The app-wide handler: a reader that cannot follow the variant answers like the routes that
    catch it themselves (400, English prose in detail, the code in the header)."""
    from fastapi import APIRouter
    from fastapi.testclient import TestClient
    from backend.api.main import app

    router = APIRouter()

    @router.get("/api/_test_unfollowable")
    def unfollowable():
        from backend.api.services.load_choices import measurement_groups
        with h5py.File(two, "r") as f:
            measurement_groups(f, V7)

    app.router.routes[:0] = router.routes          # ahead of the static mount of a built frontend
    try:
        resp = TestClient(app).get("/api/_test_unfollowable")
    finally:
        app.router.routes[:] = [r for r in app.router.routes
                                if getattr(r, "path", "") != "/api/_test_unfollowable"]
    assert resp.status_code == 400
    assert resp.headers["x-orienta-code"] == "variantNotSupported"
    assert "has no measurement '7'" in resp.json()["detail"]
