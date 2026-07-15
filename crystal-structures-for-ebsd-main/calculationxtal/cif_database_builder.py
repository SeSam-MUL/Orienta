#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIF to Excel Database Builder v8.0 (The Expert)
----------------------------------
The online search is now a two-stage process. It first looks for a
Materials Project ID in the filename for a precise query. If not found,
it falls back to a more robust formula-based search on COD and MP.
"""
import os
import re
import sys
import html
import argparse
import pandas as pd
from pathlib import Path
import numpy as np
import spglib
from tqdm import tqdm
import logging
import time
import h5py

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _PLOT_AVAILABLE = True
except ImportError:
    _PLOT_AVAILABLE = False
try:
    import xlsxwriter
    _XLSX_WRITER_AVAILABLE = True
except ImportError:
    _XLSX_WRITER_AVAILABLE = False
try:
    import requests
    from bs4 import BeautifulSoup
    _WEB_SEARCH_AVAILABLE = True
except ImportError:
    _WEB_SEARCH_AVAILABLE = False

from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
try:
    from pymatgen.ext.matproj import MPRester
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

# --- CONFIGURATION ---
MP_API_KEY = "Fd6So1NzJA4usfiT" # Ihr eingefügter API Schlüssel

SCRIPT_DIR = Path(__file__).parent
SEARCH_FOLDER = SCRIPT_DIR / 'Cif_Files'
XTAL_OUTPUT_FOLDER = SCRIPT_DIR / 'Xtal_Output'
EXCEL_FILE_PATH = SCRIPT_DIR / 'crystal_database.xlsx'
LOG_FILE_PATH = SCRIPT_DIR / 'database_builder.log'
DWF_FILE_NAME = 'dwf_factors.xlsx'

AUTO_MANAGED_COLUMNS = ['Composition', 'Space Group', 'Crystal System', 'Lattice Parameters (Å)', 'DOI', 'Reference Text', 'Fit for .xtal', 'Structure Fingerprint', 'Warnings', 'File Path']
USER_MANAGED_COLUMNS = ['Phase Name']
COLUMN_ORDER = USER_MANAGED_COLUMNS + [col for col in AUTO_MANAGED_COLUMNS if col != 'Possible Duplicates'] + ['Possible Duplicates', 'CIF File Name']
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(LOG_FILE_PATH, mode='a'), logging.StreamHandler(sys.stdout)])

# --- HELPER FUNCTIONS ---
def _format_formula_with_subscripts(formula: str) -> str:
    subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return re.sub(r'(\d)', lambda m: m.group(1).translate(subscript_map), formula)

def _generate_fingerprint(data: dict) -> str:
    try:
        sub_to_norm_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
        formula = data.get('Composition', '').translate(sub_to_norm_map)
        sg_num = data.get('Space Group', 'SG0').split('(')[-1].replace(')', '')
        lp_str = ''.join([f"{float(p):.3f}" for p in re.findall(r'[\d\.]+', data.get('Lattice Parameters (Å)', ''))])
        return f"SG{sg_num}-F{formula}-L{lp_str}"
    except: return "invalid-fingerprint"
    
def _extract_mp_id(filename: str) -> str | None:
    """ Extracts a Materials Project ID (e.g., mp-12345) from a filename. """
    match = re.search(r'(mp-\d+)', filename)
    return match.group(1) if match else None

def autoformat_excel_sheet(workbook, worksheet, df: pd.DataFrame):
    if not _XLSX_WRITER_AVAILABLE: return
    wrap_format = workbook.add_format({'text_wrap': True, 'valign': 'top'})
    for idx, col in enumerate(df.columns):
        if col == 'DOI Status': continue
        series = df[col]
        max_len = max((series.astype(str).map(len).max(), len(str(series.name)))) if not series.empty else len(str(col))
        max_len = min(max_len + 2, 70)
        if col in ['Lattice Parameters (Å)', 'Warnings', 'Possible Duplicates', 'File Path', 'Reference Text']:
            worksheet.set_column(idx, idx, max_len, wrap_format)
        else: worksheet.set_column(idx, idx, max_len)
    logging.info("Excel sheet auto-formatted.")

def find_reference_online(row: pd.Series) -> str:
    """ Meta-search for DOI: First by MP-ID, then COD, then MP by formula. """
    sub_to_norm_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    formula = row['Composition'].translate(sub_to_norm_map)
    filename = row['CIF File Name']
    
    # --- Step 1: Precise search using Materials Project ID from filename ---
    mp_id = _extract_mp_id(filename)
    if mp_id and _MP_AVAILABLE and MP_API_KEY and MP_API_KEY != "IHREN_API_SCHLÜSSEL_HIER_EINFÜGEN":
        try:
            with MPRester(MP_API_KEY) as mpr:
                # Get all default data for this specific material ID
                data = mpr.get_data(mp_id)
                if data and data[0].get('doi'):
                    doi = data[0]['doi']
                    logging.info(f"  -> Found DOI for {filename} via MP-ID '{mp_id}': {doi}")
                    return doi
        except Exception as e:
            logging.warning(f"  -> MP-ID search for {mp_id} failed: {e}")

    # --- Step 2: Search Crystallography Open Database (COD) by structure ---
    if _WEB_SEARCH_AVAILABLE:
        try:
            sg = row['Space Group'].split(' ')[0]
            params = [float(p) for p in re.findall(r'[\d\.]+', row['Lattice Parameters (Å)'])]
            a, b, c = params[0], params[1], params[2]
            payload = {'formula': formula, 'sg': sg, 'a': f'{a-0.1}:{a+0.1}', 'b': f'{b-0.1}:{b+0.1}', 'c': f'{c-0.1}:{c+0.1}', 'search': 'Search'}
            r = requests.get("http://www.crystallography.net/cod/search.html", params=payload, timeout=20)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'html.parser')
            result_link = soup.select_one('a[href^="/cod/"]')
            if result_link:
                entry_url = f"http://www.crystallography.net{result_link['href']}"
                r_entry = requests.get(entry_url, timeout=20); r_entry.raise_for_status()
                soup_entry = BeautifulSoup(r_entry.text, 'html.parser')
                doi_tag = soup_entry.find(lambda tag: tag.name == 'a' and 'doi.org' in tag.get('href', ''))
                if doi_tag and doi_tag.text:
                    logging.info(f"  -> Found DOI for {filename} via COD: {doi_tag.text.strip()}")
                    return doi_tag.text.strip()
        except Exception: pass
    
    # --- Step 3: Broad search on Materials Project by formula (Fallback) ---
    if _MP_AVAILABLE and MP_API_KEY and MP_API_KEY != "IHREN_API_SCHLÜSSEL_HIER_EINFÜGEN":
        try:
            with MPRester(MP_API_KEY) as mpr:
                results = mpr.get_data(formula) # Get all default data
                if results:
                    for res in results:
                        if res.get('doi'):
                            logging.info(f"  -> Found DOI for {filename} via MP formula search: {res['doi']}")
                            return res['doi']
        except Exception as e:
            logging.error(f"  -> MP formula search for {filename} failed: {e}")

    return "none found"

# --- CORE FUNCTIONS and MAIN ---
# (The rest of the script, including parse_cif_file, main, etc., remains unchanged from the last version where we added the self-healing write_doi_to_cif function)

def write_doi_to_cif(cif_path: Path, doi: str):
    """ Appends a _publ_section_doi tag to the given .cif file. """
    try:
        content = cif_path.read_text(encoding='utf-8')
        if '_publ_section_doi' in content: return
        new_line = f"\n_publ_section_doi      '{doi}'\n"
        with cif_path.open('a', encoding='utf-8') as f: f.write(new_line)
        logging.info(f"    -> Injected DOI into {cif_path.name}")
    except Exception as e:
        logging.error(f"    -> Could not write DOI to {cif_path.name}. Error: {e}")
        
def parse_cif_file(cif_path: Path) -> dict:
    try:
        structure = CifParser(str(cif_path)).get_structures(primitive=False)[0]
        sga = SpacegroupAnalyzer(structure, symprec=1e-5)
        composition = structure.composition.reduced_formula
        spg_symbol, spg_number = sga.get_space_group_symbol(), sga.get_space_group_number()
        crystal_system_str = sga.get_crystal_system()
        a, b, c, alpha, beta, gamma = structure.lattice.parameters
        lattice_params_str = f"a={a:.4f} b={b:.4f} c={c:.4f} α={alpha:.3f} β={beta:.3f} γ={gamma:.3f}"
        raw_text = cif_path.read_text(encoding="utf8", errors="ignore")
        dois = re.findall(r"(?:_publ_section_doi|_citation_doi)\s+['\"]?([^'\"\s]+)['\"]?", raw_text)
        found_doi = dois[0].strip() if dois else "none found"
        found_ref_text = "none found"
        if found_doi == "none found":
            ref_matches = re.findall(r"_publ_section_references\s*;\s*(.*?)\s*;", raw_text, re.DOTALL)
            if ref_matches: found_ref_text = re.sub(r'\s+', ' ', html.unescape(ref_matches[0])).strip()
        warnings, is_fit = [], True
        if found_doi == "none found" and found_ref_text == "none found": is_fit = False; warnings.append("No reference or DOI found.")
        if len(structure) == 0: is_fit = False; warnings.append("No atomic sites.")
        if spg_number is None or spg_number < 1: is_fit = False; warnings.append("No space group.")
        if not structure.is_ordered: warnings.append("Disordered structure.")
        formatted_composition = _format_formula_with_subscripts(composition)
        parsed_data = {"Composition": formatted_composition, "Space Group": f"{spg_symbol} ({spg_number})","Crystal System": crystal_system_str, "Lattice Parameters (Å)": lattice_params_str,"DOI": found_doi, "Reference Text": found_ref_text,"Fit for .xtal": "Yes" if is_fit else "No","CIF File Name": cif_path.name, "File Path": str(cif_path.resolve()),"Warnings": "; ".join(warnings) if warnings else ""}
        parsed_data_for_fp = parsed_data.copy(); parsed_data_for_fp['Composition'] = composition
        parsed_data["Structure Fingerprint"] = _generate_fingerprint(parsed_data_for_fp)
        return parsed_data
    except Exception: return None

def build_database(cif_folder=None, output_path=None, skip_online=False,
                   extra_cif_folders=None, progress_callback=None):
    """Scan CIF folder(s), create/update crystal_database.xlsx, return DataFrame.

    Args:
        cif_folder: Path to CIF folder. Defaults to SEARCH_FOLDER.
        output_path: Path for output Excel. Defaults to EXCEL_FILE_PATH.
        skip_online: If True, skip online DOI search (faster).
        extra_cif_folders: Optional list of additional CIF folder paths to scan.
        progress_callback: Optional callable(current, total, message) for GUI progress.

    Returns:
        pd.DataFrame with the crystal database, or None on error.
    """
    cif_folder = Path(cif_folder) if cif_folder else SEARCH_FOLDER
    output_path = Path(output_path) if output_path else EXCEL_FILE_PATH

    logging.info("Starting CIF Database Builder...")
    if not cif_folder.exists():
        logging.error(f"Directory '{cif_folder}' not found.")
        return None

    try:
        df = pd.read_excel(output_path).fillna('') if output_path.exists() else pd.DataFrame(columns=COLUMN_ORDER)
    except Exception as e:
        logging.error(f"Could not read Excel file: {e}")
        df = pd.DataFrame(columns=COLUMN_ORDER)

    if not df.empty and 'CIF File Name' in df.columns:
        df.drop_duplicates(subset=['CIF File Name'], keep='last', inplace=True, ignore_index=True)

    # Collect CIF files from all folders
    all_cif_paths_raw = list(cif_folder.rglob('*.cif')) + list(cif_folder.rglob('*.CIF'))
    if extra_cif_folders:
        for extra in extra_cif_folders:
            extra = Path(extra)
            if extra.exists():
                all_cif_paths_raw += list(extra.rglob('*.cif')) + list(extra.rglob('*.CIF'))

    unique_paths_dict = {p.resolve(): p for p in all_cif_paths_raw}
    all_cif_paths = list(unique_paths_dict.values())

    # Remove rows for CIF files that no longer exist on disk
    if not df.empty and 'CIF File Name' in df.columns:
        on_disk_names = {p.name for p in all_cif_paths}
        before_prune = len(df)
        df = df[df['CIF File Name'].isin(on_disk_names)].reset_index(drop=True)
        pruned = before_prune - len(df)
        if pruned > 0:
            logging.info(f"Removed {pruned} entries for CIF files no longer on disk.")

    if all_cif_paths:
        total = len(all_cif_paths)
        logging.info(f"Processing {total} unique .cif files on disk...")
        existing_files = dict(zip(df['CIF File Name'], df.index))
        new_rows_list = []

        for i, cif_path in enumerate(all_cif_paths):
            if progress_callback:
                progress_callback(i, total, f"Parsing: {cif_path.name}")

            parsed_data = parse_cif_file(cif_path)
            if not parsed_data:
                continue
            if cif_path.name in existing_files:
                idx = existing_files[cif_path.name]
                for col in AUTO_MANAGED_COLUMNS:
                    if col in parsed_data:
                        df.loc[idx, col] = parsed_data[col]
            else:
                new_rows_list.append({col: parsed_data.get(col, '') for col in COLUMN_ORDER})

        if new_rows_list:
            df = pd.concat([df, pd.DataFrame(new_rows_list)], ignore_index=True)

    # Online DOI search (optional)
    if not skip_online:
        df['DOI Status'] = 'pre_existing'
        refs_to_find = df[(df['DOI'] == 'none found') & (df['Reference Text'] == 'none found')]
        if not refs_to_find.empty:
            logging.info(f"Found {len(refs_to_find)} entries without references. Starting online search...")
            for idx, row in refs_to_find.iterrows():
                found_doi = find_reference_online(row)
                if found_doi != "none found":
                    df.loc[idx, 'DOI'] = found_doi
                    df.loc[idx, 'Fit for .xtal'] = "Yes"
                    df.loc[idx, 'DOI Status'] = 'found_online'
                    cif_path_to_update = Path(row['File Path'])
                    if cif_path_to_update.exists():
                        write_doi_to_cif(cif_path_to_update, found_doi)
                time.sleep(0.5)
    else:
        df['DOI Status'] = 'pre_existing'

    # Duplicate detection
    df['Possible Duplicates'] = ''
    is_duplicate = df.duplicated(subset=['Structure Fingerprint'], keep=False)
    if is_duplicate.any():
        peer_list_str = df[is_duplicate].groupby('Structure Fingerprint')['CIF File Name'].transform(lambda x: ', '.join(x))
        df.loc[is_duplicate, 'Possible Duplicates'] = "Peers: " + peer_list_str

    # Save to Excel
    try:
        logging.info(f"Saving database to {output_path}...")
        final_df = df.reindex(columns=COLUMN_ORDER)
        if _XLSX_WRITER_AVAILABLE:
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                workbook = writer.book
                worksheet = workbook.add_worksheet('Sheet1')
                header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#D7E4BC', 'border': 1})
                highlight_format = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
                for col_num, value in enumerate(final_df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                for row_num, row_data in enumerate(final_df.to_dict('records')):
                    for col_num, col_name in enumerate(final_df.columns):
                        cell_value = row_data.get(col_name, '')
                        if col_name == 'DOI' and df.iloc[row_num]['DOI Status'] == 'found_online':
                            worksheet.write(row_num + 1, col_num, cell_value, highlight_format)
                        else:
                            worksheet.write(row_num + 1, col_num, cell_value)
                autoformat_excel_sheet(workbook, worksheet, final_df)
        else:
            final_df.to_excel(output_path, index=False)
        logging.info("Database updated successfully.")
    except Exception as e:
        logging.error(f"Could not save Excel file (is it open?): {e}")

    if progress_callback:
        progress_callback(len(all_cif_paths) if all_cif_paths else 0,
                          len(all_cif_paths) if all_cif_paths else 0, "Done")

    return final_df


def main():
    parser = argparse.ArgumentParser(description="CIF Database Builder v8.0")
    parser.add_argument('--plot', action='store_true', help='Generate summary plots.')
    parser.add_argument('--generate-xtal', action='store_true', help='Generate .xtal files.')
    parser.add_argument('--skip-online', action='store_true', help='Skip online DOI search.')
    args = parser.parse_args()

    df = build_database(skip_online=args.skip_online)
    if df is not None and args.plot and _PLOT_AVAILABLE:
        generate_summary_plots(df)


if __name__ == "__main__":
    main()