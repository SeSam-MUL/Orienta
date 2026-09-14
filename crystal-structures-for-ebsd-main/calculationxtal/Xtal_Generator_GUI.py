#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMsoft .xtal Generator (V3.8 - Final Reference Parsing & Display)
-------------------------------------------------
Ensures full and correct display of publication references by robustly
decoding HTML entities and normalizing whitespace before adding to ComboBox.
"""

import sys
import os
import re
import h5py
import numpy as np
import pandas as pd
import spglib
from pathlib import Path
from datetime import date, datetime
from functools import partial
import html # For HTML entity decoding

from pymatgen.core.structure import Structure 
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from PyQt5.QtCore import Qt, QSettings, pyqtSignal
from PyQt5.QtGui import QIcon, QColor
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QListWidget, QListWidgetItem,
    QComboBox, QTableWidget, QTableWidgetItem, QAbstractItemView,
    QMessageBox, QGroupBox, QStatusBar, QProgressDialog
)

# --- HELPER FUNCTION TO GET ICON PATH ---
def icon_path(icon_name):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, 'icons', icon_name)

# --- CUSTOM WIDGETS ---
class CIFListWidget(QListWidget):
    filesDropped = pyqtSignal(list)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setToolTip("Drag & Drop CIF files or folders here")
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.acceptProposedAction()
    def dropEvent(self, event):
        paths = []
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_dir():
                paths.extend(path.glob('**/*.cif')); paths.extend(path.glob('**/*.CIF'))
            elif path.suffix.lower() == '.cif':
                paths.append(path)
        if paths: self.filesDropped.emit(paths)

# --- MAIN APPLICATION CLASS ---
class EMsoftXtalGenerator(QWidget):
    ORG_NAME, APP_NAME = "EMsoftTools", "XtalGenerator"
    SETTING_LAST_DWF_PATH = "lastDWFPath"
    CREATOR_TAG, PROGRAM_NAME_TAG = b"EMsoftXtalGenerator_py_v3", b"AutoXtalGen_v3"
    CRYSTAL_SYSTEMS = {
        1: "Cubic", 2: "Tetragonal", 3: "Orthorhombic", 4: "Hexagonal",
        5: "Rhombohedral/Trigonal", 6: "Monoclinic", 7: "Triclinic"
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EMsoft .xtal Generator")
        self.setWindowIcon(QIcon(icon_path('atom.svg'))) 
        self.resize(1200, 800)
        self.loaded_cif_paths, self.parsed_data = [], {}
        self.settings = QSettings(self.ORG_NAME, self.APP_NAME)
        self._build_ui()
        self.combo_useReference.currentTextChanged.connect(self.on_reference_edited)
        self._load_settings()
        self._auto_detect_dwf()
        self.lstCIFs.currentRowChanged.connect(self.on_cif_selected)
        self.lstCIFs.filesDropped.connect(self.add_cif_paths)
        self.tblAsymm.itemChanged.connect(self.on_table_item_changed)
    
    def _build_ui(self):
        main_layout = QHBoxLayout()
        main_layout.addWidget(self._create_file_list_group(), stretch=2)
        right_panel = QVBoxLayout()
        right_panel.addWidget(self._create_parameters_group(), stretch=1)
        right_panel.addWidget(self._create_save_group())
        main_layout.addLayout(right_panel, stretch=5)
        outer_layout = QVBoxLayout(self)
        outer_layout.addLayout(main_layout)
        self.statusBar = QStatusBar()
        outer_layout.addWidget(self.statusBar)
    
    def _create_file_list_group(self):
        group = QGroupBox("CIF Files")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 20, 10, 10)
        self.lstCIFs = CIFListWidget()
        layout.addWidget(self.lstCIFs)
        row1_layout = QHBoxLayout()
        self.btnLoadOne = self._create_button("Load One", "file-plus.svg", self.load_one_cif)
        self.btnLoadMultiple = self._create_button("Load Multiple", "files.svg", self.load_multiple_cifs)
        row1_layout.addWidget(self.btnLoadOne)
        row1_layout.addWidget(self.btnLoadMultiple)
        self.btnLoadFolder = self._create_button("Load Folder", "folder.svg", self.load_cifs_from_folder)
        layout.addLayout(row1_layout)
        layout.addWidget(self.btnLoadFolder)
        return group

    def _create_parameters_group(self):
        group = QGroupBox("Crystal Parameters")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 20, 10, 10)
        top_row_layout = QHBoxLayout()
        top_row_layout.addWidget(QLabel("Debye-Waller Factors:"))
        self.le_dwf_path = QLineEdit(); self.le_dwf_path.setReadOnly(True)
        top_row_layout.addWidget(self.le_dwf_path, stretch=1)
        self.btnBrowseDWF = self._create_button("Browse", "file.svg", self.browse_dwf)
        self.btnLoadParse = self._create_button("Re-Parse", "refresh-cw.svg", self.on_load_parse_single)
        top_row_layout.addWidget(self.btnBrowseDWF)
        top_row_layout.addWidget(self.btnLoadParse)
        layout.addLayout(top_row_layout)
        grid = QGridLayout(); grid.setColumnStretch(1, 1); grid.setSpacing(10)
        self.le_cif_path = self._create_parameter_row(grid, 0, "Current CIF:", editable=False)
        self.le_crystal_system = self._create_parameter_row(grid, 1, "CrystalSystem:")
        self.le_spacegroup_number = self._create_parameter_row(grid, 2, "SpaceGroupNumber:")
        self.le_spacegroup_setting = self._create_parameter_row(grid, 3, "SpaceGroupSetting:")
        self.le_lattice_params = self._create_parameter_row(grid, 4, "Lattice [nm, °]:")
        self.le_natomtypes = self._create_parameter_row(grid, 5, "N(asymm):")
        grid.addWidget(QLabel("useReference:"), 6, 0)
        self.combo_useReference = QComboBox(); self.combo_useReference.setEditable(True)
        grid.addWidget(self.combo_useReference, 6, 1, 1, 2)
        layout.addLayout(grid)
        layout.addWidget(QLabel("Asymmetric-Unit Sites (Double-click to edit occ/DWF):"))
        self.tblAsymm = QTableWidget(0, 6)
        self.tblAsymm.setHorizontalHeaderLabels(["Z", "x", "y", "z", "occ", "DWF [nm²]"])
        self.tblAsymm.verticalHeader().setVisible(False)
        self.tblAsymm.setEditTriggers(QAbstractItemView.DoubleClicked)
        layout.addWidget(self.tblAsymm, stretch=1)
        return group

    def _create_save_group(self):
        group = QGroupBox("Export")
        layout = QHBoxLayout(group); layout.setContentsMargins(10, 10, 10, 10); layout.addStretch()
        self.btnSaveSingle = self._create_button("Save Selected", "save.svg", self.save_single_xtal)
        self.btnSaveAll = self._create_button("Save All (Batch)", "archive.svg", self.save_all_xtal)
        layout.addWidget(self.btnSaveSingle)
        layout.addWidget(self.btnSaveAll)
        return group

    def _create_button(self, text, icon_name, on_click):
        btn = QPushButton(text)
        if os.path.exists(icon_path(icon_name)): btn.setIcon(QIcon(icon_path(icon_name)))
        btn.clicked.connect(on_click)
        btn.setCursor(Qt.PointingHandCursor)
        return btn

    def _create_parameter_row(self, grid, row, label, editable=True):
        line_edit = QLineEdit(); line_edit.setReadOnly(True)
        grid.addWidget(QLabel(label), row, 0)
        grid.addWidget(line_edit, row, 1)
        if editable:
            btn = self._create_button("", "edit-3.svg", partial(self.on_edit_value, line_edit))
            btn.setObjectName("EditButton")
            grid.addWidget(btn, row, 2)
            if "CrystalSystem" in label:
                legend = "\n".join(f"  {k} = {v}" for k,v in self.CRYSTAL_SYSTEMS.items())
                grid.itemAtPosition(row, 0).widget().setToolTip(f"CrystalSystem Legend:\n{legend}")
        return line_edit
    
    def _load_settings(self):
        path = self.settings.value(self.SETTING_LAST_DWF_PATH, "")
        if Path(path).exists():
            self.le_dwf_path.setText(path)
            self.statusBar.showMessage("Loaded last used DWF file path.", 3000)

    def _auto_detect_dwf(self):
        """Auto-detect DWF.xlsx in the same directory if no path is set."""
        if not self.le_dwf_path.text():
            default_dwf = Path(__file__).parent / "DWF.xlsx"
            if default_dwf.exists():
                self.le_dwf_path.setText(str(default_dwf))
                self.settings.setValue(self.SETTING_LAST_DWF_PATH, str(default_dwf))
                self.statusBar.showMessage("Auto-detected DWF.xlsx.", 3000)

    def load_cif_by_path(self, file_path):
        """Load a CIF file by its path (called from Excel viewer double-click)."""
        p = Path(file_path)
        if not p.exists():
            self.statusBar.showMessage(f"File not found: {p.name}", 4000)
            return
        if p not in self.loaded_cif_paths:
            self.add_cif_paths([p])
        idx = self.loaded_cif_paths.index(p)
        self.lstCIFs.setCurrentRow(idx)

    def _batch_parse_all(self):
        """Parse all loaded CIF files that haven't been parsed yet."""
        dwf_path = self.le_dwf_path.text().strip()
        if not dwf_path or not Path(dwf_path).exists():
            QMessageBox.warning(self, "Missing DWF", "Select a DWF file first.")
            return

        unparsed = [
            (i, p) for i, p in enumerate(self.loaded_cif_paths)
            if p not in self.parsed_data
        ]
        if not unparsed:
            self.statusBar.showMessage("All CIF files already parsed.", 3000)
            return

        progress = QProgressDialog("Parsing CIF files...", "Cancel", 0, len(unparsed), self)
        progress.setWindowModality(Qt.WindowModal)

        for count, (idx, cif_path) in enumerate(unparsed):
            progress.setValue(count)
            progress.setLabelText(f"Parsing: {cif_path.name}")
            QApplication.processEvents()
            if progress.wasCanceled():
                break
            self._parse_cif_and_fill_ui(cif_path, dwf_path)

        progress.setValue(len(unparsed))
        parsed_count = sum(1 for p in self.loaded_cif_paths if p in self.parsed_data)
        self.statusBar.showMessage(
            f"Batch parse complete: {parsed_count}/{len(self.loaded_cif_paths)} parsed.", 5000
        )

    def add_cif_paths(self, paths):
        added_count = 0
        for p in sorted(list(set(paths))):
            if p not in self.loaded_cif_paths:
                self.loaded_cif_paths.append(p)
                item = QListWidgetItem(p.name)
                item.setIcon(QIcon(icon_path('file.svg')))
                self.lstCIFs.addItem(item)
                added_count += 1
        if added_count > 0:
            self.statusBar.showMessage(f"Added {added_count} new CIF file(s).", 4000)
            if self.lstCIFs.currentRow() == -1: self.lstCIFs.setCurrentRow(0)
    
    def load_one_cif(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CIF", "", "CIF Files (*.cif *.CIF)")
        if path: self.add_cif_paths([Path(path)])

    def load_multiple_cifs(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select CIFs", "", "CIF Files (*.cif *.CIF)")
        if paths: self.add_cif_paths([Path(p) for p in paths])

    def load_cifs_from_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            paths = list(Path(folder).glob('**/*.cif'))
            self.add_cif_paths(paths)
            if paths and self.le_dwf_path.text():
                reply = QMessageBox.question(
                    self, "Batch Parse",
                    f"Parse all {len(paths)} CIF files now?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    self._batch_parse_all()

    def browse_dwf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select DWF Excel File", "", "Excel Files (*.xlsx *.xls)")
        if path and path != self.le_dwf_path.text():
            if self.parsed_data and QMessageBox.question(self, "DWF File Changed", "This will clear all parsed data. Continue?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.No:
                return
            self.parsed_data.clear()
            for i in range(self.lstCIFs.count()):
                item = self.lstCIFs.item(i)
                item.setText(self.loaded_cif_paths[i].name)
                item.setForeground(self.palette().color(self.foregroundRole()))
                item.setIcon(QIcon(icon_path('file.svg')))
            self.statusBar.showMessage("Cache cleared due to new DWF file.", 5000)
            self.le_dwf_path.setText(path)
            self.settings.setValue(self.SETTING_LAST_DWF_PATH, path)

    def on_edit_value(self, line_edit):
        line_edit.setReadOnly(False); line_edit.setFocus()
        line_edit.editingFinished.connect(lambda: line_edit.setReadOnly(True))

    def on_cif_selected(self, index):
        if index < 0: return
        cif_path = self.loaded_cif_paths[index]
        self.le_cif_path.setText(str(cif_path))
        if cif_path in self.parsed_data:
            self._fill_ui_from_parsed_data(cif_path)
        else:
            dwf_path = self.le_dwf_path.text().strip()
            if not dwf_path or not Path(dwf_path).exists():
                self.statusBar.showMessage("Select a valid DWF file to parse.", 5000); return
            self._parse_cif_and_fill_ui(cif_path, dwf_path)

    def on_load_parse_single(self):
        current_row = self.lstCIFs.currentRow()
        if current_row < 0: self.statusBar.showMessage("No CIF selected.", 3000); return
        dwf_path = self.le_dwf_path.text().strip()
        if not dwf_path or not Path(dwf_path).exists(): QMessageBox.warning(self, "Missing DWF", "Select DWF file."); return
        cif_path = self.loaded_cif_paths[current_row]
        if cif_path in self.parsed_data: del self.parsed_data[cif_path]
        self._parse_cif_and_fill_ui(cif_path, dwf_path)

    def on_table_item_changed(self, item):
        current_row_idx = self.lstCIFs.currentRow()
        if current_row_idx < 0: return
        cif_path = self.loaded_cif_paths[current_row_idx]
        if cif_path not in self.parsed_data: return
        row, col = item.row(), item.column()
        try:
            new_value = float(item.text())
            if col == 4: self.parsed_data[cif_path]["occ_list"][row] = new_value
            elif col == 5: self.parsed_data[cif_path]["dwf_list"][row] = new_value
            self.statusBar.showMessage(f"Updated atom {row+1}.", 2000)
        except (ValueError, IndexError):
            self.statusBar.showMessage("Invalid input. Reverting.", 3000)
            self._fill_ui_from_parsed_data(cif_path)
    
    def on_reference_edited(self, new_text: str):
        """
        This slot is triggered whenever the text in the useReference ComboBox changes.
        It saves the new text back to the parsed_data cache for the currently selected file.
        """
        current_row = self.lstCIFs.currentRow()
        # Ensure a file is actually selected to avoid errors on startup or when the list is cleared
        if current_row < 0:
            return
        
        # Get the path of the currently selected CIF file
        cif_path = self.loaded_cif_paths[current_row]
        
        # Check if we have data for this path
        if cif_path in self.parsed_data:
            # Get the list of references for this CIF
            ref_list = self.parsed_data[cif_path].get("references", [])
            
            # If the new text is not already in the list, add it to the beginning.
            # This makes the user's edit the new default choice for this file.
            # We also remove any previous occurrences of the same text to avoid duplicates.
            cleaned_new_text = new_text.strip()
            if cleaned_new_text in ref_list:
                ref_list.remove(cleaned_new_text)
                
                ref_list.insert(0, cleaned_new_text)
                
                # Update the cache
                self.parsed_data[cif_path]["references"] = ref_list
                
                # Optional: Give the user some feedback
                self.statusBar.showMessage(f"Reference for '{cif_path.name}' updated.", 2000)
    
    
     
    def _parse_cif_and_fill_ui(self, cif_path, dwf_path):
        self.statusBar.showMessage(f"Parsing '{cif_path.name}'...", 0); QApplication.processEvents()
        item_index = self.loaded_cif_paths.index(cif_path)
        item = self.lstCIFs.item(item_index)
        item.setIcon(QIcon(icon_path('refresh-cw.svg')))
        try:
            parser = CifParser(str(cif_path))
            structures = parser.parse_structures()
            if not structures:
                raise ValueError("Pymatgen could not parse any structure from the CIF file.")
            structure: Structure = structures[0] 

            ordered_structure_for_spglib: Structure
            if not structure.is_ordered:
                self.statusBar.showMessage(f"Disordered structure for {cif_path.name}. Ordering for spglib...", 0); QApplication.processEvents()
                try:
                    temp_struct = structure.get_primitive_structure(tolerance=0.25)
                    ordered_structure_for_spglib = temp_struct.get_sorted_structure()
                    if not ordered_structure_for_spglib.is_ordered:
                        possible_orderings = temp_struct.get_orderings()
                        if possible_orderings: ordered_structure_for_spglib = possible_orderings[0]
                        else: raise ValueError("Could not derive an ordered structure using get_orderings().")
                except Exception as e_order_complex:
                    print(f"Warning: Complex ordering for {cif_path.name} failed ({e_order_complex}). Simpler ordering."); QApplication.processEvents()
                    species, coords = [], []
                    for site in structure:
                        if site.is_ordered: species.append(site.specie)
                        else: species.append(max(site.species, key=site.species.get))
                        coords.append(site.frac_coords)
                    ordered_structure_for_spglib = Structure(structure.lattice, species, coords)
            else:
                ordered_structure_for_spglib = structure
            
            if not hasattr(ordered_structure_for_spglib, 'atomic_numbers'):
                if ordered_structure_for_spglib.is_ordered:
                     raise AttributeError(f"Ordered structure for spglib is missing 'atomic_numbers'. Structure: {ordered_structure_for_spglib.formula}")
                else: 
                    atomic_numbers_list = []
                    for site_idx, site in enumerate(ordered_structure_for_spglib):
                        if site.is_ordered: atomic_numbers_list.append(site.specie.number)
                        elif site.species: atomic_numbers_list.append(site.species.elements[0].number)
                        else: raise ValueError(f"Site {site_idx} in {cif_path.name} has no species information.")
                    sym_data_input_tuple = (ordered_structure_for_spglib.lattice.matrix, ordered_structure_for_spglib.frac_coords, atomic_numbers_list)
            else:
                sym_data_input_tuple = (ordered_structure_for_spglib.lattice.matrix, ordered_structure_for_spglib.frac_coords, ordered_structure_for_spglib.atomic_numbers)
            
            sym_data = spglib.get_symmetry_dataset(sym_data_input_tuple, symprec=1e-5)

            raw_text = cif_path.read_text(encoding="utf8", errors="ignore")
            found_refs = []
            doi_patterns = [r"_publ_section_doi\s+['\"]([^'\"]+)['\"]", r"_citation_doi\s+['\"]([^'\"]+)['\"]", r"_citation_journal_doi\s+['\"]([^'\"]+)['\"]"]
            for pat in doi_patterns:
                for m_doi in re.findall(pat, raw_text):
                    if m_doi.strip() not in found_refs: found_refs.append(m_doi.strip())
            
            if not found_refs:
                matches = re.findall(r"_publ_section_references\s*;\s*(.*?)\s*;", raw_text, flags=re.DOTALL)
                for m_ref in matches:
                    cite_text = m_ref.strip()
                    plain = re.sub(r"<[^>]+>", "", cite_text) # Strip HTML tags
                    plain = html.unescape(plain) # Decode HTML entities
                    plain = re.sub(r'\s+', ' ', plain).strip() # Normalize whitespace
                    if plain and (plain not in found_refs): found_refs.append(plain)
            if not found_refs: found_refs.append("none found") # Fallback
            
            dwf_map = {str(r["Element"]).split()[0]: float(r["DWB 300 K"]) for _, r in pd.read_excel(dwf_path).iterrows()}
            a, b, c = np.array(structure.lattice.abc) / 10.0; alpha, beta, gamma = structure.lattice.angles
            sga = SpacegroupAnalyzer(structure, symprec=1e-5)
            csys_num = {v.lower(): k for k, v in self.CRYSTAL_SYSTEMS.items()}.get(sga.get_crystal_system().lower(), 7)
            spg_number = sga.get_space_group_number()
            origin_shift = sym_data.get('origin_shift', np.zeros(3))
            spacegroup_setting = 2 if np.linalg.norm(origin_shift) > 1e-6 else 1
            reps = sga.get_symmetrized_structure().equivalent_sites
            
            self.parsed_data[cif_path] = { 
                "csys": csys_num, "spg_number": spg_number, "spg_setting": spacegroup_setting, 
                "lattice": (a, b, c, alpha, beta, gamma), "N": len(reps), 
                "Z_list": [r[0].specie.number if r[0].is_ordered else list(r[0].species.keys())[0].number for r in reps],
                "coords": np.array([r[0].frac_coords for r in reps]).T, 
                "occ_list": [sum(r[0].species[el] for el in r[0].species) if not r[0].is_ordered else 1.0 for r in reps],
                "dwf_list": [dwf_map.get(r[0].specie.symbol if r[0].is_ordered else list(r[0].species.keys())[0].symbol, 0.005) for r in reps],
                "references": found_refs 
            }
            
            self._fill_ui_from_parsed_data(cif_path)
            item.setForeground(QColor("#50fa7b")); item.setIcon(QIcon(icon_path('check-circle.svg')))
            self.statusBar.showMessage(f"Successfully parsed '{cif_path.name}'.", 4000)
            
        except Exception as e:
            QMessageBox.critical(self, "Parsing Error", f"Could not parse '{cif_path.name}':\n\n{type(e).__name__}: {e}")
            self.statusBar.showMessage(f"Failed to parse '{cif_path.name}'.", 5000)
            if cif_path in self.parsed_data: del self.parsed_data[cif_path]
            item.setForeground(QColor("#ff5555")); item.setIcon(QIcon(icon_path('alert-circle.svg')))

    def _fill_ui_from_parsed_data(self, cif_path):
        if cif_path not in self.parsed_data: return
        data = self.parsed_data[cif_path]
        self.tblAsymm.itemChanged.disconnect(self.on_table_item_changed)
        self.le_crystal_system.setText(f"{data['csys']} ({self.CRYSTAL_SYSTEMS.get(data['csys'], 'N/A')})"); self.le_spacegroup_number.setText(str(data['spg_number'])); self.le_spacegroup_setting.setText(str(data['spg_setting']))
        self.le_lattice_params.setText(f"{data['lattice'][0]:.6f}, {data['lattice'][1]:.6f}, {data['lattice'][2]:.6f}, {data['lattice'][3]:.4f}, {data['lattice'][4]:.4f}, {data['lattice'][5]:.4f}")
        self.le_natomtypes.setText(str(data['N'])); self.combo_useReference.clear(); self.combo_useReference.addItems(data['references'])
        self.tblAsymm.setRowCount(data['N'])
        for i in range(data['N']):
            self.tblAsymm.setItem(i, 0, self._create_table_item(str(data['Z_list'][i]), False, Qt.AlignCenter)); self.tblAsymm.setItem(i, 1, self._create_table_item(f"{data['coords'][0, i]:.6f}", False))
            self.tblAsymm.setItem(i, 2, self._create_table_item(f"{data['coords'][1, i]:.6f}", False)); self.tblAsymm.setItem(i, 3, self._create_table_item(f"{data['coords'][2, i]:.6f}", False))
            self.tblAsymm.setItem(i, 4, self._create_table_item(f"{data['occ_list'][i]:.5f}")); self.tblAsymm.setItem(i, 5, self._create_table_item(f"{data['dwf_list'][i]:.5f}"))
        self.tblAsymm.resizeColumnsToContents()
        self.tblAsymm.itemChanged.connect(self.on_table_item_changed)
    
    def _create_table_item(self, text, editable=True, align=Qt.AlignVCenter | Qt.AlignRight):
        item = QTableWidgetItem(text); item.setTextAlignment(align)
        if not editable: item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        return item

    def save_single_xtal(self):
        current_row = self.lstCIFs.currentRow()
        if current_row < 0: QMessageBox.warning(self, "No CIF", "Select a CIF first."); return
        cif_path = self.loaded_cif_paths[current_row]
        if cif_path not in self.parsed_data: QMessageBox.warning(self, "Not Parsed", "Parse the file first."); return
        save_path, _ = QFileDialog.getSaveFileName(self, "Save .xtal", cif_path.with_suffix('.xtal').name, "HDF5 .xtal (*.xtal)")
        if save_path:
             try: self._write_xtal_file(cif_path, Path(save_path)); QMessageBox.information(self, "Saved", f"Wrote .xtal to:\n{save_path}")
             except Exception as e: QMessageBox.critical(self, "Write Error", f"Could not write file:\n{e}")

    def save_all_xtal(self):
        parsed_paths = [p for p in self.loaded_cif_paths if p in self.parsed_data]
        if not parsed_paths: QMessageBox.warning(self, "No Parsed Files", "Parse files before batch saving."); return
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not folder: return
        out_folder = Path(folder)
        progress = QProgressDialog("Saving .xtal files...", "Cancel", 0, len(parsed_paths), self)
        progress.setWindowModality(Qt.WindowModal)
        for i, cif_path in enumerate(parsed_paths):
            progress.setValue(i); progress.setLabelText(f"Saving: {cif_path.name}"); QApplication.processEvents()
            if progress.wasCanceled(): self.statusBar.showMessage("Batch save canceled.", 4000); break
            self._write_xtal_file(cif_path, out_folder / cif_path.with_suffix('.xtal').name)
        progress.setValue(len(parsed_paths))
        self.statusBar.showMessage(f"Batch save complete.", 5000)
        
    # In Ihrer EMsoftXtalGenerator-Klasse in Xtal_Generator_GUI.py:

    def _write_xtal_file(self, cif_path, out_xtal):
    # Diese Funktion verwendet jetzt für Batch-Prozesse ausschließlich gecachte Daten,
    # um die Korrektheit sicherzustellen.

        if cif_path not in self.parsed_data:
            print(f"WARNUNG: Keine geparsten Daten für {cif_path.name} gefunden. Überspringe Speichern.")
            return
        
        # 1) Hole das vollständige Datendictionary für DIESEN spezifischen CIF-Pfad aus dem Cache.
        data = self.parsed_data[cif_path]

        try:
        # Lese ALLE Werte direkt aus dem 'data'-Dictionary, NICHT aus den GUI-Elementen.
            csys, spg_num, spg_set = (
                data['csys'],
                data['spg_number'],
                data['spg_setting']
                )
            # Gitterparameter (a, b, c, alpha, beta, gamma) sind ein Tupel im Cache
            lp_vals = data['lattice']
            N = data['N']
            Z_arr = np.array(data["Z_list"], dtype=np.int32)
            atomdata = np.vstack([
                data["coords"],
                data["occ_list"],
                data["dwf_list"]
            ]).astype(np.float32)

            # Referenz direkt aus dem gecachten Datensatz nehmen
            found_refs_for_this_cif = data.get("references", [])
            ref_str = found_refs_for_this_cif[0] if found_refs_for_this_cif else "none found"
        
            # Einzige Ausnahme: Wenn die aktuell angezeigte Datei gespeichert wird ("Save Selected"),
            # respektieren wir die eventuell vom Nutzer in der ComboBox geänderte Referenz.
            if self.le_cif_path.text() == str(cif_path):
                ref_str = self.combo_useReference.currentText().strip() or ref_str

            ref_bytes = ref_str.encode("utf-8", errors="ignore")

        except (KeyError, IndexError) as e:
            print(f"FEHLER: Unvollständige oder fehlerhafte Cache-Daten für {cif_path.name}. Schlüssel {e} fehlt. Speichern abgebrochen.")
            return

        # 2) HDF5-Datei schreiben (dieser Teil bleibt gleich)
        with h5py.File(str(out_xtal), "w") as f:
            cd = f.create_group("CrystalData")

            cd.create_dataset("CrystalSystem",       data=[csys])
            cd.create_dataset("SpaceGroupNumber",    data=[spg_num])
            cd.create_dataset("SpaceGroupSetting",   data=[spg_set])
            cd.create_dataset("LatticeParameters",   data=lp_vals)
            cd.create_dataset("Natomtypes",          data=[N])
            cd.create_dataset("Atomtypes",           data=Z_arr)
            cd.create_dataset("AtomData",            data=atomdata)

            cd.create_dataset(
                "useReference",
                data=[ref_bytes],
                dtype=h5py.special_dtype(vlen=bytes)
            )

            # Metadaten für die Erstellung
            for name, val in [
                ("CreationDate",  date.today().isoformat().encode("utf-8")),
                ("CreationTime",  datetime.now().strftime("%H:%M:%S").encode("utf-8")),
                ("Creator",       self.CREATOR_TAG),
                ("ProgramName",   self.PROGRAM_NAME_TAG),
            ]:
                cd.create_dataset(
                    name,
                    data=[val],
                    dtype=h5py.special_dtype(vlen=bytes)
                )
    
        print(f"Successfully wrote {cif_path.name} to {out_xtal.name}")
        

def set_style(app):
    COLOR_BACKGROUND, COLOR_FOREGROUND = "#282a36", "#f8f8f2"
    COLOR_WIDGET_BACKGROUND, COLOR_WIDGET_BORDER = "#44475a", "#6272a4"
    COLOR_ACCENT, COLOR_ACCENT_HOVER = "#8be9fd", "#a4ffff"
    app.setStyleSheet(f"""
        QWidget {{ background-color: {COLOR_BACKGROUND}; color: {COLOR_FOREGROUND}; font-size: 10pt; font-family: Segoe UI, Roboto, sans-serif; }}
        QGroupBox {{ font-size: 12pt; font-weight: bold; border: 1px solid {COLOR_WIDGET_BORDER}; border-radius: 8px; margin-top: 1em; }}
        QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top center; padding: 0 10px; background-color: {COLOR_BACKGROUND}; }}
        QLineEdit, QComboBox, QTableWidget, QListWidget {{ background-color: {COLOR_WIDGET_BACKGROUND}; border: 1px solid {COLOR_WIDGET_BORDER}; border-radius: 4px; padding: 5px; }}
        QLineEdit:focus, QComboBox:focus, QTableWidget:focus {{ border: 1px solid {COLOR_ACCENT}; }}
        QPushButton {{ background-color: {COLOR_WIDGET_BORDER}; color: {COLOR_FOREGROUND}; font-weight: bold; border: none; border-radius: 4px; padding: 8px 12px; }}
        QPushButton:hover {{ background-color: {COLOR_ACCENT}; color: {COLOR_BACKGROUND}; }}
        QPushButton#EditButton {{ padding: 6px; max-width: 35px; min-width: 35px; }}
        QStatusBar {{ font-size: 9pt; }}
        QScrollBar:vertical {{ border: none; background: {COLOR_WIDGET_BACKGROUND}; width: 10px; margin: 0; }}
        QScrollBar::handle:vertical {{ background: {COLOR_WIDGET_BORDER}; min-height: 20px; border-radius: 5px; }}
        QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{ height: 0px; background: none; }}
        QTableWidget::item:selected, QListWidget::item:selected {{ background-color: {COLOR_ACCENT}; color: {COLOR_BACKGROUND}; }}
    """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    set_style(app)
    window = EMsoftXtalGenerator()
    window.show()
    sys.exit(app.exec_())