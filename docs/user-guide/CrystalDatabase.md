# Crystal Database

## What it does

The Crystal Database tool manages the crystal structures that Orienta uses for
indexing and pattern simulation. It is the bridge between human-readable
**CIF** files (Crystallographic Information Files, the standard format from
structure databases and the literature) and the **`.xtal`** format that EMsoft
and the simulation pipeline require.

Concretely, it lets you:

- Import CIF files into the local library (`Database/CIF_Library/`).
- Read the crystallographic parameters of each CIF (space group, crystal
  system, lattice parameters, atom sites, occupancy, reference/DOI).
- Convert a CIF into an EMsoft `.xtal` file
  (`Database/XTAL_Library/`), correctly handling disordered/mixed-occupancy
  sites and looking up Debye–Waller factors per element.
- Build and browse a searchable index of all structures
  (`crystal_database.xlsx`).
- Edit the Debye–Waller factor (DWF) reference table.
- Optionally sync the CIF/XTAL library with a shared server location.

The conversion is what turns a structure you found in a paper into something you
can actually simulate a master pattern / SHT for and then index against.

## When to use it

This sits at the very **start** of the indexing workflow, before any
simulation. Use it whenever you need a phase that is not yet in your library:

1. Obtain a CIF for the phase (from the literature, COD, Materials Project, or
   the [Crystal Hint](CrystalHint.md) external search).
2. Import it here and check its parameters.
3. Convert it to `.xtal`.
4. Move to the [Simulation](Simulation.md) tool to generate the master
   pattern / SHT.
5. Index your EBSD data against the new phase.

It is also the place to curate references and DOIs and to verify that a
structure is "fit for conversion" (ordered, has a space group, has atom sites,
and carries a citation).

## How to use it — step by step

1. **Open the tool.** When the page becomes active it scans
   `Database/CIF_Library/` and lists every CIF on the left, with a badge
   indicating whether a matching `.xtal` already exists. It also parses each CIF
   in the background so the parameters are ready when you click.

2. **Import a CIF.** In the CIF list, use **Load Files…** to pick one or more
   `.cif` files (a native file dialog in the desktop app). Each file is copied
   into the library with duplicate detection — identical files are skipped, and
   a same-name-but-different-content file is flagged. **Load Folder…** lets you
   point at a folder of CIFs.

3. **Inspect a structure.** Click a CIF in the list. The **Crystal Parameters**
   panel on the right shows the phase name, space group + number, crystal
   system, lattice parameters (a, b, c, α, β, γ), the atom site table, and the
   reference/DOI. A "fit for conversion" indicator tells you whether the CIF can
   be converted, with warnings (e.g. *no reference*, *no atomic sites*,
   *disordered structure*).

4. **Convert one CIF.** With a CIF selected and marked fit, click **Convert** in
   the action bar. This writes a `.xtal` into the XTAL library and reports the
   space group and atom count. The `.xtal` badge then appears on that CIF.

5. **Batch convert.** Tick the checkboxes of several CIFs and click
   **Batch Convert**. Files flagged "not fit for conversion" are skipped
   automatically; the status bar reports converted / failed / skipped counts.

6. **Build the database index.** Click **Build Database** in the toolbar. This
   parses all CIFs and writes `crystal_database.xlsx` (with a live progress
   bar). **View Database** then opens a table of all structures that you can
   browse and edit cell-by-cell.

7. **Edit Debye–Waller factors.** Click **View DWF Table** to open the DWF
   dialog. You can edit existing B-factors, add a literature reference per
   element, and add entirely new elements. New/edited values are used by the
   next CIF → `.xtal` conversion.

8. **Delete.** Tick CIFs and use **Delete Selected** to remove the CIF (and its
   matching `.xtal`). You can also click a `.xtal` badge to delete just that
   converted file.

9. **Sync (optional).** If a shared server is configured, the toolbar shows a
   **Sync** button. It uploads local-only files, downloads server-only files,
   and detects conflicts by content hash. When conflicts exist, a dialog lets
   you choose **Keep Local**, **Keep Server**, or **Keep Both** per file.

## Inputs & outputs

**Inputs**

- `.cif` files imported from disk (copied into `Database/CIF_Library/`).
- The Debye–Waller factor table
  (`crystal-structures-for-ebsd-main/calculationxtal/DWF.xlsx`), read during
  conversion. A per-element B-factor is required; unknown elements fall back to
  a generic default.

**Outputs**

- `.xtal` files in `Database/XTAL_Library/` (EMsoft HDF5 crystal files), ready
  for simulation.
- `crystal_database.xlsx` — the searchable structure index used by the
  database viewer.
- Updated CIF files when you edit DOI/reference, and an updated `DWF.xlsx` when
  you edit Debye–Waller factors.

## Tips & notes

- **Conversion is gated for unfit CIFs.** A CIF with no space group, no atom
  sites, or no reference/DOI is blocked from conversion. Add a reference/DOI in
  the parameters panel if that is the only thing missing.
- **Disordered / mixed-occupancy structures convert, but simulate poorly.**
  The converter preserves every co-occupying species on a shared site (so
  minority elements like Mn/Si sharing an Al site are not dropped), but EMsoft's
  dynamical theory assumes an ordered crystal. Strongly disordered structures
  (low minimum site occupancy) can make the downstream master-pattern
  simulation extremely slow or unstable — the parameters panel warns about this.
- **Get the Debye–Waller factor right.** A missing element silently falls back
  to a generic B-factor; for accurate simulation, add the published value via
  the DWF table before converting.
- **Large unit cells are slow to simulate later.** Conversion itself is fast,
  but big cells force many reflections at fine `dmin`, which the Simulation tool
  will warn about. This is a downstream concern, not a conversion error.
- **Sync is optional** and only appears when a server location is configured and
  reachable. With no server, the library is purely local.
- Editing crystal data here changes files on disk; deletions remove the CIF
  and/or `.xtal` permanently.
