from pathlib import Path

import pytest

from backend.api.services.crystal_structure import (
    cpk_color,
    covalent_radius,
    load_structure,
    structure_payload,
)

# Every path below is a real file from the maintainer's library, which a
# clone does not have: need() skips and names it instead of failing.
from tests.data_deps import need

_ROOT = Path(__file__).resolve().parents[1]
_CIF = _ROOT / "Database" / "CIF_Library"
_XTAL = _ROOT / "Database" / "XTAL_Library"


def test_cpk_color_known_and_fallback():
    assert cpk_color("Fe").lower() == "#e06633"
    assert cpk_color("Al").lower() == "#bfa6a6"
    assert cpk_color("Zzz") == "#b0b0b0"  # unknown -> grey fallback


def test_covalent_radius_positive():
    assert 0.5 < covalent_radius("Al") < 2.0
    assert covalent_radius("Zzz") == pytest.approx(1.25)


def test_load_cif_al_is_fcc_ordered():
    s = load_structure(need(_CIF / "Al.cif"))
    assert len(s) == 4  # conventional FCC cell
    assert s.is_ordered
    assert {sp.symbol for site in s for sp in site.species} == {"Al"}


def test_load_xtal_ni_is_four_atoms():
    s = load_structure(need(_XTAL / "Ni.xtal"))
    assert len(s) == 4
    assert {sp.symbol for site in s for sp in site.species} == {"Ni"}


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_structure(_CIF / "does_not_exist.cif")


def test_xtal_partial_occupancy_merged_near_cif_count():
    # EMsoft stores each partial-occ species as its own AtomData row; naive
    # from_spacegroup gives 378 overlapping atoms. merge_sites must collapse
    # them to ~ the CIF's 138 partial-occupancy sites.
    s_cif = load_structure(need(_CIF / "alpha-AlFeMnSi_ICSD-52623.cif"))
    s_xtal = load_structure(need(_XTAL / "alpha-AlFeMnSi_ICSD-52623.xtal"))
    assert not s_cif.is_ordered
    assert not s_xtal.is_ordered
    assert len(s_xtal) < 200  # not the un-merged 378
    assert abs(len(s_xtal) - len(s_cif)) <= 0.25 * len(s_cif)


def test_payload_al_shape():
    pl = structure_payload(need(_CIF / "Al.cif"))
    assert pl["source"] == "cif"
    assert len(pl["atoms"]) == 4
    a0 = pl["atoms"][0]
    assert a0["element"] == "Al"
    assert len(a0["cart"]) == 3 and len(a0["frac"]) == 3
    assert a0["occ"] == 1.0
    assert a0["color"].lower() == "#bfa6a6"
    assert len(pl["cell_vectors"]) == 3 and len(pl["cell_vectors"][0]) == 3
    assert pl["space_group"]["number"] == 225
    assert pl["meta"]["n_atoms"] == 4
    assert pl["meta"]["disordered"] is False


def test_payload_bonds_are_segments_in_range():
    pl = structure_payload(need(_CIF / "Al.cif"))
    assert len(pl["bonds"]) > 0
    for seg in pl["bonds"]:
        (x1, y1, z1), (x2, y2, z2) = seg
        d = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5
        assert 0.1 < d < 4.0  # sane nearest-neighbour range


def test_payload_xtal_source_tag_and_disorder():
    pl = structure_payload(need(_XTAL / "alpha-AlFeMnSi_ICSD-52623.xtal"))
    assert pl["source"] == "xtal"
    assert pl["meta"]["disordered"] is True
    mixed = [a for a in pl["atoms"] if len(a["species"]) > 1]
    assert mixed  # at least one shared site


def test_payload_polyhedra_around_minority_sites():
    # Al7FeCu2: framework = Al (most abundant); Fe/Cu sites get coordination
    # polyhedra of >= 4 Al vertices each.
    pl = structure_payload(need(_CIF / "Al7FeCu2.cif"))
    polys = pl["polyhedra"]
    assert len(polys) > 0
    centers = {p["element"] for p in polys}
    assert centers  # non-framework centres only
    assert "Al" not in centers  # Al is the framework, never a polyhedron centre
    for p in polys:
        assert len(p["vertices"]) >= 4
        assert all(len(v) == 3 for v in p["vertices"])
        assert len(p["center"]) == 3
        assert p["color"].startswith("#")


def test_payload_single_element_has_no_polyhedra():
    # A single-element cell has no coordination polyhedra (nothing to coordinate).
    assert structure_payload(need(_CIF / "Al.cif"))["polyhedra"] == []
    assert structure_payload(need(_XTAL / "Ni.xtal"))["polyhedra"] == []


@pytest.mark.parametrize(
    "stem", ["sd_1401510", "sd_1802610"], ids=["sd_1401510", "sd_1802610"]
)
def test_xtal_float32_occupancy_regression(stem):
    # Regression for the float32 occupancy round-off: these two XTAL files summed
    # a shared site to 1.0000000149 and raised "occupancies sum to more than 1".
    # After rounding at source they load, and match their CIF twin's atom count.
    pl_xtal = structure_payload(need(_XTAL / f"{stem}.xtal"))
    pl_cif = structure_payload(need(_CIF / f"{stem}.cif"))
    assert len(pl_xtal["atoms"]) > 0
    assert len(pl_xtal["atoms"]) == len(pl_cif["atoms"])


def _library_files(subdir, ext):
    base = _ROOT / "Database" / subdir
    return sorted(base.rglob(f"*.{ext}")) if base.is_dir() else []


@pytest.mark.parametrize(
    "path", _library_files("CIF_Library", "cif"), ids=lambda p: p.name
)
def test_every_library_cif_loads(path):
    """Standing guarantee: a library CIF yields a valid payload — or is REFUSED
    by name, with the reason.

    The second half is not a softening. `8d521673` (2026-09-12) put one guard
    in front of every CIF reader: a file that parses to two different
    compositions is refused rather than letting data-block order decide which
    one the app shows. `sd_1816951.cif` is such a file — a SpringerMaterials
    export carrying a standardized, a published AND a niggli-reduced cell,
    which come out as Mg2Cu and Mg4Cu. Picking one silently is the failure
    mode that put 16 atoms in Si.cif (2026-09-14); refusing is the fix.

    This test is older (2026-07-14) than that guard and demanded a payload
    from every file, so it went red on the one file the guard protects us
    from. What must never happen is a *silent* wrong structure, and that is
    what the two branches below pin.
    """
    from backend.api.services.cif_phase_library import AmbiguousCifError

    try:
        pl = structure_payload(path)
    except AmbiguousCifError as exc:
        message = str(exc)
        assert path.name in message, message
        # Names both readings, so the reader can check the file rather than
        # guess what "ambiguous" meant.
        assert "different compositions" in message, message
        # And the refused set is PINNED. Without this the escape hatch would
        # quietly widen: a newly added ambiguous CIF would turn this test
        # green instead of red, and nobody would be asked whether the library
        # should carry a file the app refuses to read.
        assert path.name == "sd_1816951.cif", (
            f"{path.name} is refused as ambiguous and nothing said so. Either "
            f"fix the file, or add it here deliberately: {message}")
        return
    assert len(pl["atoms"]) > 0
    assert len(pl["cell_vectors"]) == 3 and all(len(r) == 3 for r in pl["cell_vectors"])
    assert pl["space_group"]["number"] >= 1


@pytest.mark.parametrize(
    "path", _library_files("XTAL_Library", "xtal"), ids=lambda p: p.name
)
def test_every_library_xtal_loads(path):
    # Standing guarantee: every XTAL in the shipped library yields a valid payload.
    pl = structure_payload(path)
    assert len(pl["atoms"]) > 0
    assert len(pl["cell_vectors"]) == 3 and all(len(r) == 3 for r in pl["cell_vectors"])
    assert pl["space_group"]["number"] >= 1
