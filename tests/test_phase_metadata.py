"""Tests for phase_metadata module."""
import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Every path below is a real file from the maintainer's library, which a
# clone does not have: need() skips and names it instead of failing.
from tests.data_deps import need


# Real CIF files from the Database
CIF_DIR = Path(__file__).parent.parent / "Database" / "CIF_Library"


class TestParseCifField:
    def test_springer_materials_tab_delimited(self):
        """SpringerMaterials CIFs use tab-delimited single-quoted values."""
        from phase_metadata import extract_metadata_from_cif
        meta = extract_metadata_from_cif(need(CIF_DIR / "sd_0302719.cif"))
        assert meta.formula == "Mn0.5Fe0.5Al5Si0.68"
        assert meta.space_group == "Im-3"
        assert meta.pearson == "cI168"
        assert meta.source == "cif"

    def test_springer_al13fe4_formula_is_fe4al13(self):
        """Al13Fe4.cif stores _sm_phase_labels as 'Fe4Al13' (Fe first)."""
        from phase_metadata import extract_metadata_from_cif
        meta = extract_metadata_from_cif(need(CIF_DIR / "Al13Fe4.cif"))
        assert meta.formula == "Fe4Al13"
        assert meta.space_group == "C12/m1"
        assert meta.pearson == "mS102"

    def test_materials_project_space_delimited(self):
        """Materials Project CIFs use _chemical_formula_structural, no Pearson."""
        from phase_metadata import extract_metadata_from_cif
        meta = extract_metadata_from_cif(need(CIF_DIR / "Al6Fe_mp-570001_symmetrized.cif"))
        assert meta.formula == "Al6Fe"
        assert meta.space_group == "Cmcm"
        assert meta.pearson == ""

    def test_iucr_journal_structural_with_underscores(self):
        """IUCr CIFs may have underscores in _chemical_formula_structural."""
        from phase_metadata import extract_metadata_from_cif
        meta = extract_metadata_from_cif(need(CIF_DIR / "Fe3Al2Si4.cif"))
        assert meta.formula == "Fe3Si4Al2"  # underscores stripped
        assert "mcm" in meta.space_group.lower().replace(" ", "")

    def test_iucr_beta_alfesi_sum_formula(self):
        """beta-AlFeSi.cif only has _chemical_formula_sum."""
        from phase_metadata import extract_metadata_from_cif
        meta = extract_metadata_from_cif(need(CIF_DIR / "beta-AlFeSi.cif"))
        assert meta.formula == "Al4FeSi"  # 'Al4 Fe Si' normalized

    def test_multi_block_first_nonempty_wins(self):
        """Multi-data-block CIF: first non-empty space group is used."""
        from phase_metadata import extract_metadata_from_cif
        meta = extract_metadata_from_cif(need(CIF_DIR / "sd_0302719.cif"))
        assert meta.space_group == "Im-3"

    def test_sd_1802610_hexagonal(self):
        """sd_1802610 is hexagonal P63/mmc, not cubic."""
        from phase_metadata import extract_metadata_from_cif
        meta = extract_metadata_from_cif(need(CIF_DIR / "sd_1802610.cif"))
        assert meta.space_group == "P63/mmc"
        assert meta.pearson == "hP28"

    def test_sd_1962794_heusler(self):
        """sd_1962794 is Heusler Fm-3m / cF16."""
        from phase_metadata import extract_metadata_from_cif
        meta = extract_metadata_from_cif(need(CIF_DIR / "sd_1962794.cif"))
        assert meta.formula == "MnFe2Al0.25Si0.75"
        assert meta.space_group == "Fm-3m"
        assert meta.pearson == "cF16"


class TestShtFilenameParsing:
    def test_full_sht_filename(self):
        """Parse full SHT filename with all components."""
        from phase_metadata import extract_metadata_from_sht_filename
        meta = extract_metadata_from_sht_filename(
            Path("Al6Fe (beta-AlFeSi) [oC16] {20kV}.sht")
        )
        assert meta.formula == "Al6Fe"
        assert meta.phase_name == "beta-AlFeSi"
        assert meta.pearson == "oC16"
        assert meta.source == "sht_filename"

    def test_minimal_sht_filename(self):
        """Parse SHT with only formula."""
        from phase_metadata import extract_metadata_from_sht_filename
        meta = extract_metadata_from_sht_filename(Path("Al.sht"))
        assert meta.formula == "Al"
        assert meta.phase_name == ""
        assert meta.pearson == ""

    def test_sht_display_label(self):
        """Display label is the composition (formula) only; structure is shown
        separately by the UI."""
        from phase_metadata import extract_metadata_from_sht_filename
        meta = extract_metadata_from_sht_filename(
            Path("Al6Fe (beta-AlFeSi) [oC16] {20kV}.sht")
        )
        assert meta.display_label == "Al6Fe"


class TestH5XtalParsing:
    def _make_mock_h5(self, has_crystal_data=True, atoms=None, counts=None, sgn=None):
        """Build a mock h5py.File with optional CrystalData."""
        mock_file = MagicMock()
        if not has_crystal_data:
            mock_file.__contains__ = lambda self, key: False
            return mock_file

        mock_file.__contains__ = lambda self, key: key == "CrystalData"

        cd = {}
        if atoms is not None:
            atom_ds = MagicMock()
            atom_ds.__getitem__ = lambda self, key=(): np.array(atoms)
            cd["Atomtypes"] = atom_ds
        if counts is not None:
            count_ds = MagicMock()
            count_ds.__getitem__ = lambda self, key=(): np.array(counts)
            cd["Natomtypes"] = count_ds
        if sgn is not None:
            sgn_ds = MagicMock()
            sgn_ds.__getitem__ = lambda self, key=(): np.array([sgn])
            cd["SpaceGroupNumber"] = sgn_ds

        mock_crystal = MagicMock()
        mock_crystal.__contains__ = lambda self, key: key in cd
        mock_crystal.__getitem__ = lambda self, key: cd[key]
        mock_file.__getitem__ = lambda self, key: mock_crystal if key == "CrystalData" else MagicMock()
        return mock_file

    def test_h5_with_crystaldata(self):
        """Extract formula and space group from HDF5 CrystalData group."""
        from phase_metadata import extract_metadata_from_h5

        mock_file = self._make_mock_h5(atoms=[13, 26], counts=[6, 1], sgn=63)

        with patch("h5py.File") as mock_open:
            mock_open.return_value.__enter__ = lambda s: mock_file
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            meta = extract_metadata_from_h5(Path("test.h5"))

        assert "Al" in meta.formula
        assert "Fe" in meta.formula
        assert meta.source == "h5_crystaldata"

    def test_h5_without_crystaldata(self):
        """H5 without CrystalData returns filename stem."""
        from phase_metadata import extract_metadata_from_h5

        mock_file = self._make_mock_h5(has_crystal_data=False)

        with patch("h5py.File") as mock_open:
            mock_open.return_value.__enter__ = lambda s: mock_file
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            meta = extract_metadata_from_h5(Path("Al6Fe_master.h5"))

        assert meta.formula == "Al6Fe_master"
        assert meta.source == "stem"


class TestFormulasMatch:
    def test_exact_match(self):
        from phase_metadata import _formulas_match
        assert _formulas_match("Al6Fe", "Al6Fe")

    def test_case_insensitive(self):
        from phase_metadata import _formulas_match
        assert _formulas_match("al6fe", "Al6Fe")

    def test_substring_does_NOT_match(self):
        """_formulas_match was deliberately changed to exact match only because
        substring matching let short formulas like 'Al' match everything
        containing 'Al'. Regression guard: confirm substring does NOT match."""
        from phase_metadata import _formulas_match
        assert not _formulas_match("Al6Fe", "Al6Fe_mp-570001_symmetrized")
        assert not _formulas_match("Al", "Al6Fe")  # the root of the original bug

    def test_no_match(self):
        from phase_metadata import _formulas_match
        assert not _formulas_match("Al6Fe", "Ni")

    def test_empty_returns_false(self):
        from phase_metadata import _formulas_match
        assert not _formulas_match("", "Al6Fe")
        assert not _formulas_match("Al6Fe", "")


class TestFindLinkedCif:
    def test_stem_match(self, tmp_path):
        """Direct stem match finds CIF."""
        from phase_metadata import find_linked_cif
        cif = tmp_path / "Al13Fe4.cif"
        cif.write_text("_sm_phase_labels 'Fe4Al13'\n")
        result = find_linked_cif(Path("Al13Fe4.h5"), tmp_path)
        assert result == cif

    def test_no_match_returns_none(self, tmp_path):
        """No matching CIF returns None."""
        from phase_metadata import find_linked_cif
        result = find_linked_cif(Path("Unknown.sht"), tmp_path)
        assert result is None

    def test_formula_match_for_sht(self, tmp_path):
        """SHT formula search finds matching CIF."""
        from phase_metadata import find_linked_cif
        cif = tmp_path / "Al6Fe_mp-570001_symmetrized.cif"
        cif.write_text("_chemical_formula_structural   Al6Fe\n")
        result = find_linked_cif(
            Path("Al6Fe (beta-AlFeSi) [oC16] {20kV}.sht"), tmp_path
        )
        assert result == cif


class TestGetPhaseMetadata:
    def test_cif_direct(self):
        """CIF file returns CIF metadata directly."""
        from phase_metadata import get_phase_metadata
        meta = get_phase_metadata(need(CIF_DIR / "sd_0302719.cif"))
        assert meta.formula == "Mn0.5Fe0.5Al5Si0.68"
        assert meta.source == "cif"

    def test_unknown_ext_returns_stem(self):
        """Unknown extension returns filename stem."""
        from phase_metadata import get_phase_metadata
        meta = get_phase_metadata(Path("random_file.xyz"))
        assert meta.formula == "random_file"
        assert meta.source == "stem"

    def test_sht_without_cif_uses_filename(self):
        """SHT with no CIF library falls back to filename parsing."""
        from phase_metadata import get_phase_metadata
        meta = get_phase_metadata(
            Path("Al6Fe (beta-AlFeSi) [oC16] {20kV}.sht"),
            cif_library_dir=None,
        )
        assert meta.formula == "Al6Fe"
        assert meta.source == "sht_filename"


class TestDisplayLabel:
    """display_label is JUST the chemical composition (formula). The crystal
    system / space group / Pearson are shown SEPARATELY by the UI, never mashed
    into this string (per the user's Crystal-Database-style requirement)."""

    def test_is_composition_only(self):
        from phase_metadata import PhaseMetadata, _build_display_label
        meta = PhaseMetadata(formula="Mn0.5Fe0.5Al5Si0.68", space_group="Im-3", pearson="cI168")
        assert _build_display_label(meta) == "Mn0.5Fe0.5Al5Si0.68"

    def test_no_structure_or_phasename_in_label(self):
        from phase_metadata import PhaseMetadata, _build_display_label
        meta = PhaseMetadata(formula="Al6Fe", space_group="Cmcm", pearson="oC16",
                             phase_name="beta-AlFeSi")
        assert _build_display_label(meta) == "Al6Fe"

    def test_stem_only(self):
        from phase_metadata import PhaseMetadata, _build_display_label
        meta = PhaseMetadata(formula="sd_0302719")
        assert _build_display_label(meta) == "sd_0302719"


class TestSubscripts:
    def test_format_formula_subscripts(self):
        from phase_metadata import format_formula_subscripts
        assert format_formula_subscripts("Al13Fe4") == "Al₁₃Fe₄"
        assert format_formula_subscripts("Mn4.512Al127") == "Mn₄.₅₁₂Al₁₂₇"
        assert format_formula_subscripts("Al") == "Al"
        assert format_formula_subscripts("") == ""


class TestCanonicalLabel:
    """The ONE phase-identity format used everywhere (chemistry -> structure -> tag)."""

    def test_clean_formula(self):
        from phase_metadata import build_canonical_label
        label = build_canonical_label(formula="Al2Cu", space_group="I4/mcm", pearson="tI12")
        assert "Al2Cu" in label
        assert "tI12 (I4/mcm)" in label

    def test_decimal_to_elements_with_nickname(self):
        from phase_metadata import build_canonical_label
        label = build_canonical_label(
            formula="Al100.89Fe21.24Mn5.31Si10.62", space_group="Im-3", pearson="cI138")
        assert "(Al,Fe,Mn,Si)" in label   # ugly decimal -> element list
        assert "cI138 (Im-3)" in label
        assert "Al(Fe,Mn)Si" in label     # nickname tag appended
        assert "100.89" not in label

    def test_no_pearson_keeps_space_group(self):
        from phase_metadata import build_canonical_label
        label = build_canonical_label(formula="Al6Fe", space_group="Cmcm")
        assert "Al6Fe" in label
        assert "Cmcm" in label
        assert "()" not in label

    def test_empty_returns_blank(self):
        from phase_metadata import build_canonical_label
        assert build_canonical_label(formula="") == ""

    def test_phase_nickname_lookup(self):
        from phase_metadata import phase_nickname
        assert phase_nickname("Im-3", ["Al", "Fe", "Mn", "Si"]) == "α-Al(Fe,Mn)Si"
        assert phase_nickname("I m -3", ("Si", "Al", "Mn", "Fe")) == "α-Al(Fe,Mn)Si"
        assert phase_nickname("Fm-3m", ["Al"]) == ""


class TestDegeneracyHelpers:
    def test_phase_metadata_has_new_fields_with_defaults(self):
        from phase_metadata import PhaseMetadata
        m = PhaseMetadata()
        assert m.lattice_a is None and m.lattice_b is None and m.lattice_c is None
        assert m.lattice_alpha is None and m.lattice_beta is None and m.lattice_gamma is None
        assert m.space_group_number is None
        assert m.laue_class == ""
        assert m.centering == ""

    def test_strip_cif_su_removes_uncertainty_parens(self):
        from phase_metadata import _strip_cif_su
        assert _strip_cif_su("12.643(2)") == 12.643
        assert _strip_cif_su("7.53941600") == 7.539416
        assert _strip_cif_su("90.0") == 90.0
        assert _strip_cif_su("") is None
        assert _strip_cif_su("?") is None
        assert _strip_cif_su("notanumber") is None

    def test_laue_class_from_sg_number_covers_all_systems(self):
        from phase_metadata import _laue_class_from_sg_number
        assert _laue_class_from_sg_number(1) == "-1"      # triclinic
        assert _laue_class_from_sg_number(12) == "2/m"    # monoclinic C2/m
        assert _laue_class_from_sg_number(63) == "mmm"    # orthorhombic Cmcm
        assert _laue_class_from_sg_number(139) == "4/mmm" # tetragonal
        assert _laue_class_from_sg_number(166) == "-3m"   # trigonal
        assert _laue_class_from_sg_number(194) == "6/mmm" # hexagonal
        assert _laue_class_from_sg_number(204) == "m-3"   # cubic Im-3
        assert _laue_class_from_sg_number(225) == "m-3m"  # cubic Fm-3m
        assert _laue_class_from_sg_number(None) == ""
        assert _laue_class_from_sg_number(0) == ""

    def test_centering_from_hm_symbol(self):
        from phase_metadata import _centering_from_hm
        assert _centering_from_hm("Cmcm") == "C"
        assert _centering_from_hm("Im-3") == "I"
        assert _centering_from_hm("Fm-3m") == "F"
        assert _centering_from_hm("P63/mmc") == "P"
        assert _centering_from_hm("C12/m1") == "C"
        assert _centering_from_hm("") == ""
        assert _centering_from_hm("xyz") == ""


class TestCifLatticeExtraction:
    def test_mnal6_lattice_and_laue(self):
        from phase_metadata import extract_metadata_from_cif
        meta = extract_metadata_from_cif(need(CIF_DIR / "MnAl6_mp-173_symmetrized.cif"))
        assert abs(meta.lattice_a - 7.539416) < 1e-3
        assert abs(meta.lattice_b - 6.460304) < 1e-3
        assert abs(meta.lattice_c - 8.819153) < 1e-3
        assert abs(meta.lattice_alpha - 90.0) < 1e-6
        assert meta.space_group_number == 63
        assert meta.laue_class == "mmm"
        assert meta.centering == "C"

    def test_al6fe_is_isostructural_with_mnal6(self):
        """Al6Fe and MnAl6: same Cmcm, lattice within ~1.5% -> EBSD-degenerate."""
        from phase_metadata import extract_metadata_from_cif
        al6fe = extract_metadata_from_cif(need(CIF_DIR / "Al6Fe_mp-570001_symmetrized.cif"))
        assert al6fe.laue_class == "mmm"
        assert al6fe.centering == "C"
        assert al6fe.lattice_a is not None

    def test_get_phase_metadata_on_sht_propagates_lattice(self):
        """An SHT file linked to a CIF must carry the CIF's lattice params."""
        from phase_metadata import get_phase_metadata
        cif_dir = CIF_DIR
        sht = Path(__file__).parent.parent / "Database" / "EBSD_SHT_Database" / "Mn" / "MnAl3 (MnAl6_mp-173_symmetrized) [mP12] {20kV}.sht"
        # Only the CIF, NOT the .sht: get_phase_metadata reads the .sht path for
        # its NAME and then loads the linked CIF, so the .sht need not exist —
        # and it does not, on any machine here. Requiring it skipped a test that
        # passes, which is how this guard was caught being too strict.
        # Without the CIF the call returns source="sht_filename" and the
        # assertion below would fail rather than skip; that is what it guards.
        need(cif_dir / "MnAl6_mp-173_symmetrized.cif")
        meta = get_phase_metadata(sht, cif_library_dir=cif_dir)
        assert meta.source == "cif"
        assert meta.lattice_a is not None
        assert meta.laue_class == "mmm"
        assert meta.centering == "C"


class TestXtalLatticeExtraction:
    def test_mnal6_xtal_carries_lattice(self):
        from phase_metadata import extract_metadata_from_xtal
        xtal = Path(__file__).parent.parent / "Database" / "XTAL_Library" / "MnAl6_mp-173_symmetrized.xtal"
        meta = extract_metadata_from_xtal(need(xtal))
        assert meta.lattice_a is not None and meta.lattice_a > 0
        assert meta.space_group_number == 63
        assert meta.laue_class == "mmm"
