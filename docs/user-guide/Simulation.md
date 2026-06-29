# Simulation

## What it does

The Simulation tool generates the **simulated reference data** that EBSD indexing
needs: from a crystal structure (`.xtal`) it produces a **master pattern**
and/or a **spherical-harmonic-transform (SHT)** file by simulating electron
backscatter diffraction.

It supports two compute engines:

- **Ours** (default) — a self-contained GPU/CPU forward-simulation pipeline
  (PyTorch Monte Carlo + dynamical master-pattern builder). It needs no external
  installation; it runs on a CUDA GPU when available and falls back to CPU.
- **EMsoft** (optional) — the established EMsoft Monte-Carlo + master-pattern
  + SHT pipeline, run through WSL. It is only selectable when WSL + EMsoft are
  detected on the machine.

The three output types map directly onto the three indexing methods:

| Output type | File(s) | Used by |
|-------------|---------|---------|
| **SHT only** | `.sht` | Spherical indexing (EMSphInx-style) |
| **Master Pattern only** | `.h5` master | Dictionary indexing (kikuchipy) |
| **Both** | `.sht` + `.h5` | Both indexing methods |

## When to use it

Run a simulation whenever a phase you want to index against does **not yet have**
a master pattern / SHT in your library. The typical chain is:

1. Get the structure into `.xtal` form with the
   [Crystal Database](CrystalDatabase.md) tool.
2. **Here:** simulate the master pattern and/or SHT for that `.xtal` at the
   relevant accelerating voltage.
3. Index your EBSD data against the new phase.

Use **Simulate All Missing** to find every `.xtal` in your library that still
lacks output for the chosen voltage and queue them in one batch.

## How to use it — step by step

The page has three tabs: **Configure**, **Queue**, and **History**. The engine
switch (**Ours / EMsoft**) lives in the header; EMsoft is greyed out unless it
is detected.

### Configure

1. **Choose the engine.** Leave it on **Ours** for the built-in pipeline (the
   header shows the hardware path it will use, e.g. your GPU name or CPU cores),
   or switch to **EMsoft** if it is available.
2. **Pick a crystal file.** Select the `.xtal` to simulate (via the crystal
   picker / file input). The crystal picker shows, per file, whether MC / Master
   / SHT already exist at the chosen voltage.
3. **Set the key parameters:**
   - **Voltage (kV)** — accelerating voltage (default 20 kV). Output files are
     tagged with the voltage, so simulate at the voltage your data was acquired
     at.
   - **Sample tilt** — typically 70°.
   - **dmin** — smallest d-spacing included; smaller = more reflections =
     slower. The default is 0.05; large unit cells may need a coarser value.
   - **Output type** — SHT only, Master only, or Both (see table above).
   - **Total electrons**, **resolution / npx**, and **threads** as needed. The
     "Ours" engine needs far fewer electrons than EMsoft (the default adjusts
     automatically when you switch engines).
   - **Advanced** Monte-Carlo and master-pattern options are available for fine
     control.
4. **Compute mode (EMsoft only).** When EMsoft is selected, an
   Auto / GPU-only / CPU-only switch chooses the OpenCL vs OpenMP path for the
   EMsoft binaries. This card is hidden under the "Ours" engine because that
   pipeline picks GPU-vs-CPU itself.
5. **Start.** Click **Start Simulation** to queue a single job. The view
   switches to the Queue tab.

### Simulate All Missing (batch)

1. Click **Simulate All Missing**. The tool scans the XTAL library for phases
   lacking output at the current voltage and opens a pre-launch dialog.
2. The dialog lists each missing phase with its largest lattice parameter, a
   **recommended dmin** (coarser for large cells so the run finishes in
   reasonable time), and a **disorder warning** for mixed-occupancy structures
   (severe cases are unchecked by default so they are not even attempted).
3. Adjust per-phase `dmin` if you like (or "use recommended for all"), confirm,
   and the selected phases are launched as a sequential batch.

You can also queue a batch from a manual selection of `.xtal` files.

### Queue

Shows running, pending, queued, and finished jobs with live progress and log
lines. You can stop a single job or cancel a whole batch. Polling continues even
if you navigate away while jobs are running.

### History

A persisted record of completed and failed jobs (crystal, method, status,
timestamps, parameters, and a link to the saved log file). You can clear the
history.

## Inputs & outputs

**Inputs**

- A `.xtal` crystal file (from `Database/XTAL_Library/`).
- Simulation parameters (voltage, tilt, dmin, electron count, resolution,
  threads, output type, advanced MC/master options).

**Outputs** (written under `Database/`)

- Monte-Carlo `.h5` files in `Database/EBSD_H5_Cache/`.
- Master-pattern `.h5` files in `Database/EBSD_H5_Cache/` (for dictionary
  indexing).
- `.sht` files in `Database/EBSD_SHT_Database/` (for spherical indexing).
- Per-job log files and a `simulation_history.json` record.

File names encode the parameters (material, voltage, etc.) so different runs do
not overwrite each other.

## Tips & notes

- **"Ours" needs no WSL/EMsoft and runs on GPU when present.** It is the
  recommended default; the header label tells you whether it resolved to GPU
  (CUDA) or CPU. EMsoft is optional and only enabled when WSL + EMsoft are
  detected — if EMsoft becomes unavailable the page falls back to "Ours"
  automatically.
- **GPU work is serialised.** Concurrent GPU simulation jobs (single or batch)
  queue on a single GPU lock to avoid running two master-pattern builds at once
  and exhausting VRAM. After each job the GPU memory is freed.
- **Large unit cells are the main time sink.** At a fine `dmin`, big cells
  generate huge reflection lists and can take hours to days. Take the
  recommended per-phase `dmin` in the "Simulate All Missing" dialog seriously.
- **Disordered structures can stall.** Strongly mixed-occupancy phases can hang
  the master-pattern computation; the batch dialog flags these and leaves the
  worst ones unchecked by default. See the disorder note in the
  [Crystal Database](CrystalDatabase.md) docs.
- **Match the voltage to your acquisition.** Output is voltage-tagged and only
  the file matching your data's accelerating voltage is useful for indexing.
- **Long EMsoft steps can look frozen.** The EMsoft SHT step computes internally
  for a long time without log output; the queue shows an elapsed-time / CPU-load
  heartbeat so a slow run is distinguishable from a stalled one.
- **The "Ours" forward-simulation pipeline is comparatively new** relative to
  the long-established EMsoft path. For phases where exact agreement with EMsoft
  matters, you can switch to the EMsoft engine (when available) and compare.
