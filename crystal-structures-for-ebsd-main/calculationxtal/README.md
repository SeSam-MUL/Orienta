================================================================================
          User Manual: CIF/Xtal Management & GUI Generator
================================================================================

Welcome to the detailed user manual for CIF and Xtal management.
This document explains all features and workflows of the two tools
in this folder: `cif_database_builder.py` and `Xtal_Generator_GUI.py`.


    +--------------------------------------------------------------------------+
    | [IMPORTANT] PREREQUISITE NOTICE                                          |
    |                                                                          |
    | This guide assumes you have met the prerequisites listed in the          |
    | README.md of the parent folder (../README.md), mainly the Python         |
    | environment of Orienta.                                                  |
    |                                                                          |
    | The installation of Python, the virtual environment setup, etc.,         |
    | will not be explained again here.                                        |
    +--------------------------------------------------------------------------+


---------------------------------
    TABLE OF CONTENTS
---------------------------------

    1.  Purpose of These Tools
    2.  The Command-Line Tool: `cif_database_builder.py`
        2.1 The Online Search Logic in Detail
    3.  The Generated Excel File (`crystal_database.xlsx`)
        3.1 Meaning of the Columns
        3.2 Why This File Is Not Under Version Control
    4.  The GUI Tool: `Xtal_Generator_GUI.py`
        4.1 Visual Overview and UI Elements
        4.2 The `DWF.xlsx` Input File: Format and Usage
    5.  Expert Workflow: The "Magic" Update Feature
    6.  Specific Troubleshooting


********************************************************************************
1. PURPOSE OF THESE TOOLS
********************************************************************************

The scripts in this folder are the starting point for the entire simulation
workflow. Their purpose is to turn an unsorted collection of `.cif` files
into organized, simulation-ready `.xtal` files.

  > `cif_database_builder.py`:
  >    Your tool for bulk analysis. It scans hundreds of CIF files, creates
  >    a clear Excel database, and enriches the data with online information.

  > `Xtal_Generator_GUI.py`:
  >    Your "surgical" tool. With this graphical application, you process a
  >    single crystal structure to create a perfect `.xtal` file for the
  >    EMsoft simulation.


********************************************************************************
2. THE COMMAND-LINE TOOL: `cif_database_builder.py`
********************************************************************************

This script is executed from the command line (PowerShell or CMD).

    --> COMMAND (Default):
        python cif_database_builder.py

    --> COMMAND (With custom paths):
        python cif_database_builder.py --input /path/to/cif_folder --output my_database.xlsx

---------------------------------
   2.1 The Online Search Logic in Detail
---------------------------------

If a CIF file does not contain a reference or DOI, the script attempts to
find this information online automatically. This process follows a clear
hierarchy to achieve the best possible results:

    STEP 1: CHECK THE CIF FILE CONTENT
        --> First, the content of the `.cif` file is searched for an existing
            `_citation_doi` tag. If one is found, the search for this file ends.

    STEP 2: CHECK THE FILENAME (PRECISION SEARCH)
        --> If Step 1 fails, the script searches the filename for a Materials
            Project ID in the format `(mp-XXXXXX)`. If an ID is found, a
            targeted query is sent to the Materials Project to get the exact DOI.

    STEP 3: FORMULA-BASED SEARCH (FALLBACK)
        --> If Step 2 is also unsuccessful, the chemical formula of the
            structure is determined. This formula is then used to search for
            matching entries on the websites of the **Crystallography Open
            Database (COD)** and the **Materials Project (MP)**.


********************************************************************************
3. THE GENERATED EXCEL FILE (`crystal_database.xlsx`)
********************************************************************************

This file is your central cockpit for managing the CIF collection.

---------------------------------
   3.1 Meaning of the Columns
---------------------------------
    * `DOI Status`:
        --> Indicates the origin of the DOI: `found_in_cif`, `found_online` (which
            is highlighted in green), `manual_entry`, or `not_found`.

    * `Formula` / `Space Group Sym.` / `a, b, c, ...`:
        --> The essential crystallographic data extracted by `pymatgen` for a
            quick overview.

---------------------------------
   3.2 Why This File Is Not Under Version Control
---------------------------------
`crystal_database.xlsx` is listed in the `.gitignore` file of the parent folder
(`crystal-structures-for-ebsd-main/`). This is intentional.

    * The CIF files are the source data and the Python scripts are the code.
      `crystal_database.xlsx` is generated from them, so it is a local work
      artifact and not part of the source.

    * If every user committed their own version of it, the files would always
      differ and every update would produce merge conflicts, with no way to
      tell which version is the correct one.

    * Share the source data (the `.cif` files) and the code instead. Each user
      runs the script locally and gets an up-to-date Excel file.


********************************************************************************
4. THE GUI TOOL: `Xtal_Generator_GUI.py`
********************************************************************************

Launch the graphical application from the command line.

    --> COMMAND:
        python Xtal_Generator_GUI.py

---------------------------------
   4.1 Visual Overview and UI Elements
---------------------------------
  +-------------------------------------------------------------------+
  | [1] Load CIF Button                                               |
  +-------------------------------------------------------------------+
  | ... [2] Crystal Data ... | ... [3] Atom Site Table ...             |
  +-------------------------------------------------------------------+
  | [4] useReference Dropdown (editable)   [5] DWF file + Browse      |
  +-------------------------------------------------------------------+
  | [6] Generate .xtal File Button                                    |
  +-------------------------------------------------------------------+

    --> `[4] useReference`: Populated with the references found in the CIF file;
        you can edit the text before generating the file.
    --> `[5] DWF file`: The Debye-Waller factor table, `DWF.xlsx` (see 4.2). It is
        picked up automatically when it sits next to the script; "Browse"
        selects another file.

---------------------------------
   4.2 The `DWF.xlsx` Input File: Format and Usage
---------------------------------
`DWF.xlsx` is the table of Debye-Waller factors that the GUI uses for the
`DWF [nm^2]` column of every atom site. For the GUI to read it correctly, the
first sheet must have these column names:

  | Element | DWB 300 K | Reference                                     |
  |---------|-----------|-----------------------------------------------|
  | C       | 0.001442  | Peng 1996, Acta Cryst A52 456, Table 1, 300 K |

    * `Element`: the element symbol. An element that has more than one crystal
      structure has one row per structure, with the structure in brackets
      (`Fe (BCC)`, `Fe (FCC)`); the GUI uses the first row of such an element.
    * `DWB 300 K`: the isotropic Debye-Waller factor B at 300 K in nm^2
      (B in nm^2 = B in A^2 x 0.01).
    * `Reference`: where the value comes from. The GUI does not read this
      column; keep it filled so that every value stays traceable.

An element that is not in the table gets the generic value 0.005 nm^2. The
sources of the shipped values and the rules for several structures per element
are in `DWF_SOURCES.md`.


********************************************************************************
5. EXPERT WORKFLOW: THE "MAGIC" UPDATE FEATURE
********************************************************************************

This is the most powerful feature of the `cif_database_builder.py` script for
systematically improving your CIF collection.

    STEP 1: Run `cif_database_builder.py`.

    STEP 2: Open the generated `crystal_database.xlsx`.

    STEP 3: Find a row where `DOI Status` is `not_found`.

    STEP 4: Research the correct DOI online and **manually enter it into the
            "DOI" column in the Excel sheet.**

    STEP 5: Save and close the Excel file.

    STEP 6: **Run `cif_database_builder.py` again.**

    **WHAT HAPPENS NOW (THE "MAGIC"):**
    The script will read the Excel file, detect your new manual entry, and
    automatically and permanently write this new DOI back into the
    original `.cif` file. Your data collection gets better with every run.


********************************************************************************
6. SPECIFIC TROUBLESHOOTING
********************************************************************************

    * PROBLEM: An `Error parsing .cif file...` occurs.
        SOLUTION: The CIF file is likely corrupted. Validate it online with the
                  **IUCr checkCIF service** (https://checkcif.iucr.org/).

    * PROBLEM: The online search fails.
        SOLUTION: The `MP_API_KEY` environment variable is likely not set.

    * PROBLEM: Every atom site gets the Debye-Waller factor 0.005 nm^2.
        SOLUTION: The element is missing from `DWF.xlsx`, or the file does not
                  have the `Element` and `DWB 300 K` columns (see Section 4.2).