import logging
import pickle
from pathlib import Path

import numpy as np
import pytest

from backend.api.services.addons.context import (
    AddonContext,
    ContextError,
    build_context,
)


def _module_level_report(message, fraction=None):
    """Picklable by reference — the callback that CAN cross a process."""


class _FakeResult:
    """Only what build_context is allowed to read."""

    def __init__(self):
        self.original_shape = (4, 5)
        self.metadata = {"step_size_um": 0.5, "source_file": "C:/data/scan.h5oina"}
        self.confidence_scores = np.full((4, 5), 0.8)
        self.xmap = None


def _real_xmap(rows=4, cols=5):
    """A real orix CrystalMap, because the two bugs this guards against are
    both properties of the real object and invisible against a fake."""
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation

    n = rows * cols
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    rot = Rotation.from_euler(
        np.random.default_rng(0).uniform(0, 1, (n, 3)))
    phase_id = np.zeros(n, dtype=int)
    phase_id[:3] = -1                       # orix keeps not_indexed at id -1
    phase_id[5:9] = 1
    phases = PhaseList([Phase(name="Al", point_group="m-3m"),
                        Phase(name="Si", point_group="m-3m")])
    return CrystalMap(rotations=rot, x=x.ravel(), y=y.ravel(),
                      phase_id=phase_id, phase_list=phases)


def test_context_carries_the_declared_fields():
    c = build_context(_FakeResult(), quality=np.zeros((4, 5)), quality_source="native")
    assert c.shape == (4, 5)
    assert c.step_size_um == 0.5
    assert c.quality_source == "native"
    assert c.source_file == Path("C:/data/scan.h5oina")


def test_euler_is_a_grid_of_three_angles_per_pixel():
    """The field the facade is named after. orix returns (N, 3), whose ndim
    equals the grid's — which is how an ndim-based tail test produced None
    for every real result while every fake set xmap = None and saw nothing."""
    r = _FakeResult()
    r.xmap = _real_xmap()
    c = build_context(r)
    assert c.euler_rad is not None
    assert c.euler_rad.shape == (4, 5, 3)
    expected = np.asarray(r.xmap.rotations.to_euler())
    assert np.allclose(c.euler_rad[0, 1], expected[1])
    assert np.allclose(c.euler_rad[3, 4], expected[19])


def test_phase_ids_are_a_grid():
    r = _FakeResult()
    r.xmap = _real_xmap()
    assert build_context(r).phase_ids.shape == (4, 5)


def test_phase_names_map_the_ids_that_are_actually_in_the_grid():
    """orix's PhaseList yields (id, Phase) TUPLES, so `p.name` raises on every
    element; and it carries not_indexed at id -1, so a positional tuple would
    mislabel every pixel. Both are properties of the real object."""
    r = _FakeResult()
    r.xmap = _real_xmap()
    c = build_context(r)
    assert isinstance(c.phase_names, dict)
    assert c.phase_names[0] == "Al"
    assert c.phase_names[1] == "Si"
    assert c.phase_names[-1] == "not_indexed"
    assert all(isinstance(k, int) for k in c.phase_names)
    # the mapping is usable the obvious way round
    assert c.phase_names[int(c.phase_ids[0, 0])] == "not_indexed"


def test_units_are_in_the_field_names():
    """The core has GOS in radians beside KAM in degrees; the facade does not.

    Written as a positive rule over the ACTUAL names. The previous form,
    `assert "step_size" not in names`, passes whatever the fields are called:
    a set of exact strings never contains the bare prefix.
    """
    names = set(AddonContext.__dataclass_fields__)
    assert names == {
        "shape", "step_size_um", "phase_ids", "phase_names", "euler_rad",
        "confidence", "quality", "quality_source", "eds_at_pct",
        "source_file", "report",
    }
    for bare, required in (("step_size", "step_size_um"), ("euler", "euler_rad")):
        matching = {n for n in names if n == bare or n.startswith(bare + "_")}
        assert matching == {required}, matching


def test_a_context_of_plain_data_is_picklable():
    """The mechanical proof that a separate process stays possible."""
    c = build_context(_FakeResult(), quality=np.zeros((4, 5)))
    again = pickle.loads(pickle.dumps(c))
    assert again.shape == c.shape
    assert again.quality.shape == (4, 5)


def test_a_callback_that_cannot_travel_says_so_by_raising():
    """Stated by failing, not by silently restoring the no-op.

    BOTH exception types, because a callback made where the work started is a
    LOCAL — a closure in a request handler, a lambda in a test — and CPython's
    pickler says so with `AttributeError: Can't pickle local object`, which is
    not a subclass of PicklingError. Asserting only PicklingError made this
    test red on the very case it describes.
    """
    def live(message, fraction=None):       # a local: __qualname__ has <locals>
        pass

    c = build_context(_FakeResult(), report=live)
    with pytest.raises((pickle.PicklingError, AttributeError)):
        pickle.dumps(c)


def test_a_callback_that_can_travel_is_not_replaced_on_the_way():
    """The other half, and the half the rejected __setstate__ would break.

    A module-level function pickles by reference, so this context round-trips
    — and what comes back must be the SAME callback, not the no-op. Without
    this, "nothing is silently replaced" is a claim with no guard under it.
    """
    c = build_context(_FakeResult(), report=_module_level_report)
    again = pickle.loads(pickle.dumps(c))
    assert again.report is _module_level_report


def test_the_context_holds_no_rich_objects():
    c = build_context(_FakeResult())
    allowed = (np.ndarray, str, int, float, bool, tuple, dict, type(None), Path)
    for name in AddonContext.__dataclass_fields__:
        value = getattr(c, name)
        if callable(value):
            continue
        assert isinstance(value, allowed), f"{name} is {type(value)}"


def test_arrays_are_handed_out_read_only():
    """Guards against the accident: a slip in an add-on cannot write here."""
    c = build_context(_FakeResult(), quality=np.zeros((4, 5)))
    with pytest.raises(ValueError):
        c.quality[0, 0] = 1.0


def test_the_read_only_flag_is_a_guard_not_a_boundary():
    """Pinned so nobody later upgrades the claim in a docstring.

    `_frozen` hands out a VIEW; `view.base` is the caller's own array and is
    still writable. The spec says the add-on model is not a sandbox, and this
    flag does not make it one — a real boundary is a separate process.
    """
    caller = np.zeros((4, 5))
    c = build_context(_FakeResult(), quality=caller)
    assert c.quality.base is caller
    c.quality.base[0, 0] = 99.0
    assert caller[0, 0] == 99.0


def test_report_defaults_to_a_no_op():
    build_context(_FakeResult()).report("halfway", 0.5)


def test_report_is_forwarded_when_given():
    seen = []
    c = build_context(_FakeResult(), report=lambda m, f=None: seen.append((m, f)))
    c.report("step", 0.25)
    assert seen == [("step", 0.25)]


def test_missing_optional_data_is_none_not_a_guess():
    c = build_context(_FakeResult())
    assert c.quality is None and c.quality_source is None
    assert c.eds_at_pct == {}
    assert c.euler_rad is None and c.phase_ids is None
    assert c.phase_names == {}


def test_a_result_without_a_step_size_says_none():
    r = _FakeResult()
    r.metadata = {}
    assert build_context(r).step_size_um is None


def test_a_resolved_step_size_wins_over_the_metadata():
    """Nothing in product code ever writes metadata['step_size_um'], so the
    caller that resolved it the way the repo does must be able to say so —
    and must outrank a stale copy if one ever appears. Task 8b is that
    caller."""
    r = _FakeResult()                       # metadata carries 0.5
    assert build_context(r, step_size_um=0.25).step_size_um == 0.25


def test_a_result_without_a_shape_is_refused_with_the_reason():
    """A context with shape () accepts no map output at all — every one is
    rejected as 'does not match the scan grid ()'. Refuse up front instead."""
    r = _FakeResult()
    r.original_shape = None
    with pytest.raises(ContextError, match="shape"):
        build_context(r)


def test_eds_maps_are_carried_as_grids():
    c = build_context(_FakeResult(),
                      eds_at_pct={"Al": np.full(20, 95.0), "Si": np.zeros((4, 5))})
    assert set(c.eds_at_pct) == {"Al", "Si"}
    assert c.eds_at_pct["Al"].shape == (4, 5)
    assert c.eds_at_pct["Al"][0, 0] == 95.0
    with pytest.raises(ValueError):
        c.eds_at_pct["Si"][0, 0] = 1.0


LOGGER = "backend.api.services.addons.context"


def test_a_map_that_does_not_fit_warns_and_still_reports_the_field_absent(caplog):
    """The asymmetry with eds_at_pct stays; only its visibility changes.

    ``confidence`` and ``quality`` are typed Optional, so None is a legal
    "absent" and the downgrade is correct — pinned below. But these arrays are
    computed inside Orienta and cannot honestly fail to fit the grid they were
    computed on, so a mismatch here is a bug somewhere else in the product,
    and DEBUG is off in every real deployment. WARNING, naming the field and
    both shapes: "value array (17,) does not fit the (4, 5) grid" tells a
    maintainer nothing about WHICH field just went missing.
    """
    r = _FakeResult()
    r.confidence_scores = np.zeros(17)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        c = build_context(r, quality=np.zeros(9))

    assert c.confidence is None and c.quality is None      # unchanged

    warnings = [rec.getMessage() for rec in caplog.records
                if rec.levelno >= logging.WARNING and rec.name == LOGGER]
    assert any("confidence" in m and "(17,)" in m and "(4, 5)" in m
               for m in warnings), warnings
    assert any("quality" in m and "(9,)" in m and "(4, 5)" in m
               for m in warnings), warnings


def test_phase_ids_that_do_not_fit_are_warned_about_by_name(caplog):
    """The other field built through _grid, so no call site is left anonymous.

    A stand-in xmap rather than a real CrystalMap: the point is the naming of
    the field, and a real one cannot be given a phase_id of the wrong length.
    """
    class _MisshapenXmap:
        phase_id = np.zeros(7, dtype=int)
        rotations = None
        phases = []

    r = _FakeResult()
    r.xmap = _MisshapenXmap()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        c = build_context(r)

    assert c.phase_ids is None
    assert any("phase_ids" in rec.getMessage() and "(7,)" in rec.getMessage()
               for rec in caplog.records if rec.name == LOGGER), caplog.records


def test_a_value_array_that_is_not_pixel_first_warns_and_names_the_field(caplog):
    """The third _grid branch: the size divides, but the axes are the wrong
    way round. (2, 10, 2) has 40 elements on a 20-pixel grid, so the size
    test passes and the reshape would SUCCEED — silently transposing the
    data. It is refused for that reason, and refusing has to be audible."""
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        c = build_context(_FakeResult(), quality=np.zeros((2, 10, 2)))

    assert c.quality is None
    assert any("quality" in r.getMessage() and "(2, 10, 2)" in r.getMessage()
               and "(4, 5)" in r.getMessage()
               for r in caplog.records if r.name == LOGGER), caplog.records


def test_an_unreadable_phase_list_warns_and_names_the_field(caplog):
    """_phase_names' blanket handler. It returns {} for an xmap whose phase
    list cannot be walked, and {} is also what a context legitimately has
    when there is no xmap at all — so without the warning the two are
    indistinguishable to anyone reading a real deployment's log."""
    class _UnreadablePhases:
        def __iter__(self):
            raise RuntimeError("phase list unreadable")

    class _Xmap:
        phase_id = np.zeros(20, dtype=int)
        rotations = None
        phases = _UnreadablePhases()

    r = _FakeResult()
    r.xmap = _Xmap()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        c = build_context(r)

    assert c.phase_names == {}
    assert c.phase_ids is not None          # the rest of the context survives
    assert any("phase_names" in rec.getMessage()
               for rec in caplog.records if rec.name == LOGGER), caplog.records


def test_euler_angles_that_cannot_be_read_warn_and_name_the_field(caplog):
    """The handler around to_euler(), which is the field the facade is named
    after and the one that was already None for every real result once."""
    class _Rotations:
        @staticmethod
        def to_euler():
            raise RuntimeError("rotations unreadable")

    class _Xmap:
        phase_id = np.zeros(20, dtype=int)
        rotations = _Rotations()
        phases = []

    r = _FakeResult()
    r.xmap = _Xmap()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        c = build_context(r)

    assert c.euler_rad is None
    assert any("euler_rad" in rec.getMessage()
               for rec in caplog.records if rec.name == LOGGER), caplog.records


def test_euler_angles_that_do_not_fit_are_named_euler_rad_too(caplog):
    """Pins the field STRING at the _grid call site, not just the handler
    around it: to_euler() returns cleanly here, and the mismatch is found
    inside _grid. Renaming that literal otherwise changes nothing any test
    can see."""
    class _Rotations:
        @staticmethod
        def to_euler():
            return np.zeros(7)              # 7 does not divide 20

    class _Xmap:
        phase_id = np.zeros(20, dtype=int)
        rotations = _Rotations()
        phases = []

    r = _FakeResult()
    r.xmap = _Xmap()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        c = build_context(r)

    assert c.euler_rad is None
    assert any("euler_rad" in rec.getMessage() and "(7,)" in rec.getMessage()
               and "(4, 5)" in rec.getMessage()
               for rec in caplog.records if rec.name == LOGGER), caplog.records


def test_an_eds_map_that_does_not_fit_is_refused_not_nulled():
    """'Shown as missing, never silently dropped' cannot mean a None inside a
    dict the contract types as arrays — every add-on that indexes it crashes
    on a scan nobody warned them about."""
    with pytest.raises(ContextError, match="Al"):
        build_context(_FakeResult(), eds_at_pct={"Al": np.zeros(7)})


# --- the mappings are guarded like the arrays are ---------------------------

def test_the_context_mappings_refuse_the_ordinary_accident():
    """``_frozen`` guards every ARRAY and the two dicts were left plain, so
    an add-on could write into the phase names of a stored result. The same
    standard as the arrays: a guard against the accident, not a boundary."""
    c = build_context(_FakeResult(), eds_at_pct={"Al": np.zeros((4, 5))})
    for mapping in (c.phase_names, c.eds_at_pct):
        with pytest.raises(TypeError):
            mapping["x"] = 1
        with pytest.raises(TypeError):
            mapping.clear()
        with pytest.raises(TypeError):
            mapping.update({"y": 2})


def test_the_guard_does_not_cost_the_picklability_it_exists_beside():
    """The reason this is not ``MappingProxyType``: a mappingproxy cannot be
    pickled at all, and picklability is the mechanical proof that a separate
    process stays possible. Measured here rather than asserted."""
    c = build_context(_FakeResult(), eds_at_pct={"Al": np.zeros((4, 5))})
    again = pickle.loads(pickle.dumps(c))
    assert dict(again.eds_at_pct) .keys() == {"Al"}
    with pytest.raises(TypeError):
        again.phase_names["x"] = 1
