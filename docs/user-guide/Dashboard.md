# Dashboard

## What it does

The Dashboard is Orienta's home screen. It presents every analysis module as a
clickable card, grouped by where it belongs in the EBSD workflow, and keeps a
list of recently opened files for quick re-loading. It does no scientific
computation itself — it is the launch pad and navigation hub for the rest of the
application.

The modules are arranged in two workflow-ordered rows:

- **Data & Calibration** — EBSD Viewer, EDS, PC Refinement, Crystal Database.
- **Indexing & Analysis** — Indexing, Phase Maps, EBSD Analysis, ML Hub, Batch,
  Refinement.

Each card shows a short description, and a green **Ready** badge appears when the
prerequisite state for that module is satisfied (for example, the EBSD Viewer,
Indexing, PC Refinement, and EDS cards become "Ready" once a file is open; Phase
Maps becomes "Ready" once an indexing result exists; EBSD Analysis becomes
"Ready" once an analysis dataset is loaded).

## When to use it

Use the Dashboard at the start of a session and whenever you want to jump
between modules. It is the natural place to:

- Re-open a file you worked on recently.
- See, at a glance, which steps of the workflow are unlocked (via the Ready
  badges).
- Discover the available modules and the order they are meant to be used in.

## How to use it — step by step

1. **Open a module** — click any card (or focus it with the keyboard and press
   Enter / Space) to navigate to that module. Hover a card to see a tooltip with
   more detail.
2. **Re-open a recent file** — in the **Recent Files** panel, click an entry to
   load it. The list shows the file name plus its folder, and is capped at the
   8 most recent files.
3. **Quick-load the last file** — click the green **Quick Load** button to open
   the most recently used file in one click.
4. **Clear the recent list** — click **Clear** and confirm. This only empties
   the list; it does not delete any files on disk.
5. **See keyboard shortcuts** — click the small hint at the bottom of the page to
   expand a reference of the global navigation shortcuts (for example
   `Ctrl+2` → EBSD Viewer, `Ctrl+H` → HDF5 Viewer, `Ctrl+B` → toggle sidebar).
6. **Change the background** (optional) — the Dashboard background style (clean
   dot-grid, crystal, or fog) is set on the Settings page; the Dashboard picks up
   the change the next time you return to it.

## Inputs & outputs

- **Inputs:** none required. The Recent Files list and the chosen background are
  read from the browser's local storage, so they persist between sessions on the
  same machine.
- **Outputs:** navigation events to other modules, and a file-open request when
  you pick a recent file (the target module then loads the data).

There is no file or data export from the Dashboard itself.

## Tips & notes

- **Recent Files are local-only.** They are stored in the browser/Electron local
  storage of this installation. They are not validated for existence — if a file
  has been moved or deleted, clicking it will fail in the module that tries to
  load it.
- **Ready badges reflect live application state**, not disk contents. They turn
  on only after the relevant data has actually been loaded or computed in the
  current session.
- Restarting the backend clears in-memory data, so the Ready badges reset even
  though the Recent Files list (local storage) survives.
- No GPU, WSL, or long-running computation is involved here — the Dashboard is
  purely UI.
