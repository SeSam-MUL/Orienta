================================================================================
          User Manual: CIF/Xtal Management & GUI Generator
================================================================================

Welcome to the detailed user manual for CIF and Xtal management.
This document explains all features and workflows of the two tools
in this folder: `cif_database_builder.py` and `Xtal_Generator_GUI.py`.


    +--------------------------------------------------------------------------+
    | [IMPORTANT] PREREQUISITE NOTICE                                          |
    |                                                                          |
    | This guide assumes you have successfully completed the entire initial    |
    | setup (Parts A, B, and C) as described in the main, top-level README     |
    | file of this project.                                                    |
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
        3.2 Why This File Doesn't Belong on GitLab
    4.  The GUI Tool: `Xtal_Generator_GUI.py`
        4.1 Visual Overview and UI Elements
    5.  The `DWF.xlsx` Input File: Format and Usage
    6.  Expert Workflow: The "Magic" Update Feature
    7.  Specific Troubleshooting


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
   3.2 Why This File Doesn't Belong on GitLab
---------------------------------
You will notice that the `crystal_database.xlsx` file is listed in the
project's `.gitignore` file. This is intentional and very important.

    * ANALOGY: Think of the `.cif` files as the "ingredients" and the Python
      scripts as the "recipe". The `crystal_database.xlsx` is the "cooked meal".

    * THE PROBLEM: If every user uploaded their own version of the "cooked meal"
      (the Excel file) to GitLab, it would lead to constant conflicts ("Merge
      Conflicts"), as the files would always differ. You would never know which
      version is the "correct" one.

    * THE SOLUTION: You only share the ingredients (source data like `.cif`) and
      the recipe (the code) via GitLab. Every user then runs the script locally
      to generate their own, up-to-date "meal" (the Excel file).

    **The Excel file is a local, generated work artifact, not part of the source code.**


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
  | [4] Publication Reference Dropdown  [5] DWF Reference Dropdown    |
  +-------------------------------------------------------------------+
  | [6] Generate .xtal File Button                                    |
  +-------------------------------------------------------------------+

    --> `[4] Publication Reference`: Populated with references from the CIF file.
    --> `[5] DWF Reference`: Populated from the `DWF.xlsx` file.

---------------------------------
   4.2 The `DWF.xlsx` Input File: Format and Usage
---------------------------------
This is a **critical input file** for the GUI, serving as your personal database
for Debye-Waller factors. For the GUI to read it correctly, it must have the
following structure and column names:

  | ReferenceName                 | ElementName | B_iso |
  |-------------------------------|-------------|-------|
  | International Tables Vol. C   | Carbon      | 0.54  |


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

    * PROBLEM: The `DWF Reference` dropdown in the GUI is empty.
        SOLUTION: Check if `DWF.xlsx` is in the correct folder and has the
                  correct column format (see Section 4.2).