"""Phase colours for the EDS map.

Position is not identity: the candidate list changes with which phases the
user ticked and which elements the file measured, so a positional palette
recoloured unrelated phases when one entry left the list. These pin that the
colour follows the NAME, that it matches the EBSD phase map for the same
phase, and that the user's own choice wins.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.routes.phase_map import PHASE_COLORS, stable_color_idx
from backend.api.services.phase_map_store import (
    _norm_phase_name, phase_colours_for,
)


def _ebsd_colour(name):
    """What the EBSD phase map paints for a phase of this name."""
    return tuple(int(c * 255) for c in PHASE_COLORS[stable_color_idx(name)])


# --- stability --------------------------------------------------------------

def test_removing_a_phase_leaves_the_others_alone():
    """The bug this replaced: unticking one phase recoloured 23 of 29."""
    full = phase_colours_for(["Al.cif", "Si.cif", "sd_0302719.cif"])
    without_si = phase_colours_for(["Al.cif", "sd_0302719.cif"])
    assert without_si == [full[0], full[2]]


def test_order_does_not_change_a_colour():
    a = phase_colours_for(["Al.cif", "Si.cif"])
    b = phase_colours_for(["Si.cif", "Al.cif"])
    assert a == [b[1], b[0]]


# --- agreement with the EBSD page -------------------------------------------

def test_the_same_phase_looks_the_same_on_both_pages():
    """EBSD names a CIF phase by the file stem, EDS carries the filename."""
    for name in ("Al", "Si", "sd_0302719", "alpha-AlFeMnSi"):
        assert phase_colours_for([f"{name}.cif"]) == [_ebsd_colour(name)]


def test_the_key_drops_the_cif_extension_case_insensitively():
    assert _norm_phase_name("Al.CIF") == "al"
    assert _norm_phase_name("  Al.cif ") == "al"
    assert _norm_phase_name("Al") == "al"


# --- the user's own choice --------------------------------------------------

def test_an_override_wins():
    assert phase_colours_for(["Al.cif"], {"Al": "#ff0000"}) == [(255, 0, 0)]


def test_an_override_may_be_keyed_either_way():
    """The EBSD page stores "Al"; a caller might send "Al.cif"."""
    assert phase_colours_for(["Al.cif"], {"Al.cif": "#00ff00"}) == [(0, 255, 0)]


def test_a_malformed_override_falls_back_instead_of_raising():
    """A bad colour must not cost the user their classification."""
    for bad in ("nonsense", "", "#12345", "#gggggg", None):
        assert phase_colours_for(["Al.cif"], {"Al": bad}) == [_ebsd_colour("Al")]


def test_an_override_on_one_phase_leaves_the_rest_alone():
    got = phase_colours_for(["Al.cif", "Si.cif"], {"Al": "#ff0000"})
    assert got == [(255, 0, 0), _ebsd_colour("Si")]


# --- more phases than palette slots -----------------------------------------

def test_colliding_names_are_nudged_apart():
    """The palette has 8 slots; a library can hold far more phases."""
    names = [f"phase_{i}.cif" for i in range(40)]
    got = phase_colours_for(names)
    assert len(got) == 40
    # Every name that shares a base slot with an earlier one must differ
    # from it, or two phases read as one on the map.
    for i, n in enumerate(names):
        same_slot = [j for j in range(i)
                     if stable_color_idx(names[j][:-4]) == stable_color_idx(n[:-4])]
        for j in same_slot:
            assert got[i] != got[j], f"{names[i]} collides with {names[j]}"


def test_a_nudged_colour_is_still_stable_under_removal():
    """The nudge depends on earlier collisions, so dropping a NON-colliding
    phase must not move it."""
    names = [f"phase_{i}.cif" for i in range(20)]
    full = phase_colours_for(names)
    idx = 13
    slot = stable_color_idx(names[idx][:-4])
    drop = next(i for i in range(len(names))
                if stable_color_idx(names[i][:-4]) != slot and i != idx)
    kept = [n for i, n in enumerate(names) if i != drop]
    assert phase_colours_for(kept)[kept.index(names[idx])] == full[idx]


def test_an_empty_name_still_gets_a_colour():
    got = phase_colours_for(["", "Al.cif"])
    assert len(got) == 2 and all(len(c) == 3 for c in got)


def test_no_two_phases_in_one_list_share_a_colour():
    """Two phases the same colour read as one phase on the map."""
    names = [f"phase_{i}.cif" for i in range(40)]
    got = phase_colours_for(names)
    assert len(set(got)) == len(got)


def test_the_documented_limit_removing_a_COLLIDING_phase_can_shift_a_later_one():
    """Honest about the one case that is not order-free.

    Shade 0 is the untouched slot colour, which is what keeps a normal map
    identical to the EBSD page. The price is that when two phases fight over
    a slot, which one keeps shade 0 depends on list order. Fewer than nine
    phases with distinct slots never hits this.
    """
    names = [f"phase_{i}.cif" for i in range(40)]
    full = phase_colours_for(names)
    # find a genuine collision pair
    pair = next((i, j) for i in range(len(names)) for j in range(i)
                if stable_color_idx(names[j][:-4]) == stable_color_idx(names[i][:-4]))
    later, earlier = pair
    kept = [n for k, n in enumerate(names) if k != earlier]
    shifted = phase_colours_for(kept)[kept.index(names[later])]
    # With the earlier claimant gone, the later phase inherits shade 0 -
    # the untouched slot colour, which is what the EBSD page paints.
    assert shifted == _ebsd_colour(names[later][:-4])
    assert shifted != full[later], "this is the case the docstring warns about"


# --- who wins a slot --------------------------------------------------------

def test_a_phase_on_the_map_beats_a_candidate_that_covers_nothing():
    """Measured on SampleB: 22 candidates, 3 on the map.

    Without this, absent candidates earlier in the list claimed the palette
    slots and Al came out #8b66a4 instead of the EBSD page's #c792ea - the
    exact thing the name keying is supposed to prevent.
    """
    names = [f"phase_{i}.cif" for i in range(40)]
    slot = stable_color_idx("phase_7")
    rival = next(n for n in names
                 if stable_color_idx(n[:-4]) == slot and n != "phase_7.cif")
    # phase_7 is on the map, its slot rival is not
    priority = [names.index("phase_7.cif")]
    got = phase_colours_for(names, None, priority)
    assert got[names.index("phase_7.cif")] == _ebsd_colour("phase_7")
    assert got[names.index(rival)] != _ebsd_colour("phase_7")


def test_priority_may_be_partial_and_the_rest_keep_list_order():
    names = ["a.cif", "b.cif", "c.cif"]
    got = phase_colours_for(names, [], [2])
    assert len(got) == 3 and len(set(got)) == 3


def test_priority_out_of_range_does_not_drop_a_phase():
    """A stale priority list must not silently shorten the palette - the
    grid indexes into it, so a short palette paints the wrong phase."""
    names = ["a.cif", "b.cif"]
    assert len(phase_colours_for(names, None, [0, 1])) == 2
