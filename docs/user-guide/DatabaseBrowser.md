# Database Browser

## What it does

The Database Browser is the central file manager for everything in your local
crystal/EBSD database. It scans the on-disk library and presents every file in
five categories, with search, material filtering, location/size info, previews,
and safe deletion:

| Tab | Files | Role |
|-----|-------|------|
| **SHT** | `.sht` | Spherical-harmonic master files for spherical indexing |
| **MC h5** | Monte-Carlo `.h5` | Electron-scattering intermediates |
| **Master H5** | master `.h5` | Master patterns for dictionary indexing |
| **CIF** | `.cif` | Crystal structure definitions |
| **XTAL** | `.xtal` | EMsoft crystal files (converted from CIF) |

For each file it shows where it lives — **local**, **server**, or **both** —
and its size, and it offers previews: a thumbnail of master/SHT patterns, an
interactive rotatable 3-D master-pattern sphere for `.sht` files, and a
provenance / simulation-parameters panel for `.sht` files.

Unlike the [Crystal Database](CrystalDatabase.md) tool (which *creates*
structures and converts CIF → XTAL), the Database Browser is for *inspecting,
locating, and removing* the simulation artifacts your library accumulates.

## When to use it

Use the Database Browser to:

- **Audit your library** — see what master patterns / SHTs you actually have,
  for which phases and voltages.
- **Inspect a simulation result** — preview a master pattern, rotate an SHT
  sphere, or read which `.xtal`/CIF and parameters produced a `.sht`.
- **Free disk space** — delete stale or duplicate simulation files. The
  monitored library has a nominal size budget shown in the cache stats.
- **Sync with a shared server** (when configured) — push/pull CIF/XTAL files and
  see which files exist locally vs on the server.

It is a maintenance/inspection tool, used alongside (not instead of) the
Crystal Database and Simulation tools.

## How to use it — step by step

1. **Open the tool.** It scans `Database/` and (if configured) the server
   location, then lists files. It auto-refreshes every few seconds while the
   page is visible.
2. **Pick a category tab** (SHT / MC h5 / Master H5 / CIF / XTAL).
3. **Filter the table.** Use the search box for filename text and the material
   dropdown to narrow to one material subfolder.
4. **Read the columns.** Each row shows the filename, material, **location**
   (local / server / both, colour-coded), and size.
5. **Preview a file.** Click a row to open the preview panel on the right:
   - For master/SHT files: a rendered thumbnail (with metadata such as space
     group, energy, and a warning badge if a master pattern is a known-corrupt
     artifact that should be regenerated).
   - For a local `.sht`: an interactive **rotatable master-pattern sphere**
     (drag to rotate, scroll to zoom) plus a **File Info** panel with provenance
     and the simulation parameters that produced it.
   - For local files, an **Open in HDF5 Viewer** action.
6. **Download a server-only file.** Rows that exist only on the server can be
   pulled to local.
7. **Delete files (cascade-aware).** Select one or more files and start a delete.
   A dialog resolves *associated* files — for example, deleting a Monte-Carlo
   `.h5` surfaces the master pattern and SHT derived from it, and deleting a
   master surfaces the dictionaries built from it. You then choose the scope:
   **Local**, **Server**, or **Everywhere** (only the options that apply are
   offered). Required dependents cannot be unchecked.
8. **Sync all (optional).** When a server is configured, **Sync All** runs the
   bidirectional CIF/XTAL sync (upload local-only, download server-only, detect
   conflicts by content hash) and reports the counts.
9. **Check cache stats.** The bottom bar shows the total library size and file
   count against the nominal budget.

## Inputs & outputs

**Inputs**

- The on-disk `Database/` library (`CIF_Library`, `XTAL_Library`,
  `EBSD_SHT_Database`, `EBSD_H5_Cache`, `Dictionary_Library`).
- Optionally, a configured server database root for location info and sync.

**Outputs / side effects**

- Read-only previews (thumbnails, SHT sphere, provenance/parameter panel).
- File **deletions** from local and/or server, including cascade-resolved
  dependents.
- **Downloads** of server-only files to local.
- **Sync** of CIF/XTAL files between local and server, with conflict resolution.

The browser does not create simulation files — it manages files produced by the
[Crystal Database](CrystalDatabase.md) and [Simulation](Simulation.md) tools.

## Tips & notes

- **Cascade delete prevents orphans.** Deleting an MC `.h5` or a master surfaces
  the files derived from it so you can remove a whole chain together. Review the
  resolved list before confirming — deletion is permanent.
- **There is no "clear all cache" button by design.** The counted "cache" is the
  real library on disk, and CIF/XTAL files are user-authored and not
  re-downloadable, so a blanket wipe is intentionally not offered — delete files
  explicitly instead.
- **Location colours:** local, server, and both are colour-coded so you can see
  at a glance what is only on the server (and downloadable) versus already local.
- **The SHT sphere viewer loads a large plotting library on demand.** The 3-D
  Plotly view is lazy-loaded the first time you open a sphere, so there can be a
  brief delay; the sphere reconstruction itself runs on a worker thread to keep
  the UI responsive, and may take longer at higher bandwidth/resolution.
- **Corrupt-master warning.** A master-pattern thumbnail may be flagged as a
  stale, disc-masked artifact — if you see that badge, delete and regenerate the
  master in the Simulation tool.
- **Sync is only available when a server is configured and online.** With no
  server, all files are local and the sync/download actions do nothing.
