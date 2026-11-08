"""The two readers of a scan stamp refuse the same stamps.

``hex_raster.geometry_from_stamp`` serves the routes that draw a map, ``scan_neighbours.topology_of_stamp``
the grain tools. Before they shared the rule, a stamp that says it has a valid mask was a 400 on the
first and ran on the second.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import hex_raster  # noqa: E402
from backend.api.services.scan_geometry import ScanGeometry  # noqa: E402
from backend.api.services.scan_neighbours import topology_of_stamp  # noqa: E402

HEX = ScanGeometry.hex(4, 5, 0.5).to_dict()
SQUARE = ScanGeometry.square(4, 5, 0.5).to_dict()

#: (stamp, what each reader does with it), written by hand
CASES = {
    "hex": (HEX, "reads"),
    "no stamp": (None, "nothing"),
    "unknown kind": ({"kind": "triangle"}, "nothing"),
    "square": (SQUARE, "nothing"),
    # a square stamp costs a square run nothing, even one with a mask flag
    "square with a mask": ({**SQUARE, "has_valid_mask": True}, "nothing"),
    "hex with a mask": ({**HEX, "has_valid_mask": True}, "refuses"),
    "hex, infinite rows": ({**HEX, "n_rows": float("inf")}, "refuses"),
    "hex, no row shift": ({**HEX, "odd_row_shift": 0.0}, "refuses"),
}


def _geometry_reader(stamp):
    return hex_raster.geometry_from_stamp(stamp)


def _topology_reader(stamp):
    return topology_of_stamp(stamp)


def _outcome(reader, stamp) -> str:
    try:
        got = reader(stamp)
    except (ValueError, OverflowError):
        return "refuses"
    return "nothing" if got is None else "reads"


@pytest.mark.parametrize("reader", [_geometry_reader, _topology_reader], ids=["geometry_from_stamp", "topology_of_stamp"])
@pytest.mark.parametrize("name", list(CASES))
def test_each_reader_does_what_the_rule_says(reader, name):
    stamp, expected = CASES[name]
    assert _outcome(reader, stamp) == expected


@pytest.mark.parametrize("name", list(CASES))
def test_the_two_readers_agree(name):
    stamp, _ = CASES[name]
    assert _outcome(_geometry_reader, stamp) == _outcome(_topology_reader, stamp)


def test_the_refusal_of_a_masked_stamp_says_why():
    with pytest.raises(ValueError, match="valid mask"):
        _geometry_reader({**HEX, "has_valid_mask": True})
    with pytest.raises(ValueError, match="cannot be read.*valid mask"):
        _topology_reader({**HEX, "has_valid_mask": True})
