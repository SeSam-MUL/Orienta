"""The phase of a Dictionary result keeps its space group and its setting.

A dictionary loaded from a file carries a plain orix phase. For space groups
149, 151 and 153 such a phase names its point group "32" (two-fold axes along X)
while the crystal has them along Y ("312"). Handing the name and the space group
together to the result phase made orix drop the space group and leave D3 with
the wrong axes. These tests build the result phase with the expression the
indexer uses.
"""
import warnings

import pytest
from orix.crystal_map import Phase as PlainPhase

from backend.dict_gpu.pipeline.indexer import _result_phase
from crystal_symmetry import FramePhase, point_group_key


def _sg(phase):
    return None if phase.space_group is None else phase.space_group.number


@pytest.mark.parametrize("source_cls", [PlainPhase, FramePhase])
@pytest.mark.parametrize("sg,key", [(149, "312"), (151, "312"), (153, "312")])
def test_trigonal_y_setting_survives(sg, key, source_cls):
    src = source_cls(name="p", space_group=sg)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # orix warns when it drops the group
        out = _result_phase(src)
    assert isinstance(out, FramePhase)
    assert _sg(out) == sg
    assert point_group_key(out.point_group) == key
    assert out.name == "p"


@pytest.mark.parametrize("sg,key", [(157, "31m"), (159, "31m"), (162, "-31m"),
                                    (163, "-31m")])
def test_other_trigonal_y_groups(sg, key):
    out = _result_phase(PlainPhase(name="p", space_group=sg))
    assert _sg(out) == sg
    assert point_group_key(out.point_group) == key


@pytest.mark.parametrize("sg,name", [(164, "-3m"), (225, "m-3m"), (194, "6/mmm"),
                                     (62, "mmm")])
def test_other_space_groups_unchanged(sg, name):
    src = PlainPhase(name="p", space_group=sg)
    out = _result_phase(src)
    assert isinstance(out, FramePhase)
    assert _sg(out) == sg
    assert out.point_group.name == name
    assert point_group_key(out.point_group) == point_group_key(src.point_group)
    # The value is what the expression built before the change.
    before = FramePhase(name="p", point_group=src.point_group,
                        space_group=src.space_group, structure=src.structure)
    assert _sg(before) == _sg(out)
    assert before.point_group.name == out.point_group.name


def test_phase_without_space_group_keeps_its_point_group():
    src = PlainPhase(name="p", point_group="4/mmm")
    out = _result_phase(src)
    assert out.point_group.name == "4/mmm"
    assert out.space_group is None


def test_structure_is_carried():
    src = PlainPhase(name="p", space_group=149)
    assert _result_phase(src).structure is not None
