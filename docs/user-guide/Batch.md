# Batch

## What it does

The Batch module indexes **many EBSD files against many phases in one run**. You
point it at a folder, it discovers the EBSD files and the available phase files,
and it builds a job queue of *file × phase* combinations. A guided seven-step
wizard collects everything a robust multi-phase run needs — file list, pattern
centres, preprocessing, phase list, indexing method and parameters — then runs
the queue with a live dashboard (progress, memory, per-job status, ETA) and
optional automatic export of the results.

It is built on top of the same indexing engines as the single-scan Indexing page
(Hough, Spherical/SHT, Dictionary), driven through the backend `/api/batch-v2/*`
routes and a per-process batch manager.

## When to use it

Use Batch when you have **more than one scan** to process, or one scan you want to
test against **several candidate phases**, and you want it to run unattended:

- a whole experiment's worth of H5OINA files,
- multi-phase indexing where each candidate phase is a separate job,
- consistent preprocessing and pattern-centre handling across files (inherit one
  refined PC across a series), and
- automatic export to light/rich `.h5`, `.ang`, or `.ctf` so results are ready for
  Phase Map, Analysis, or Refinement.

For a single scan and quick parameter exploration, use the Indexing page instead.

## How to use it — step by step

The left sidebar is a file tree; the main area is the wizard with a step bar
across the top. You can click any step to jump to it.

### Step 1 — Files

1. Click **Add Folder** and choose a directory. Batch scans it (recursively) for
   EBSD files (`.h5oina`, `.h5`, `.hdf5`, `.hdf`, `.ebsp`), skipping previous
   checkpoint/result files, then quick-loads each file's metadata (grid shape,
   pattern shape, pattern centre, EDS availability). Loaded files show a ✓ and an
   **EDS** tag where applicable.

### Step 2 — Pattern Centre (PC)

2. Review the per-file PC table (value + source: header / inherited / refined).
   Use **Refresh PC** to re-read, or **Open PC Refinement** to calibrate a PC and
   then inherit it across the series. A good PC is critical — indexing quality
   depends on it.

### Step 3 — Preprocessing, Post-processing & Export

3. Choose **preprocessing**: frame averaging (with a 3/5/7 window) and background
   removal (dynamic/static). These are applied per pattern before indexing.
4. Choose **post-processing** clean-up thresholds: CI threshold, uncertainty
   threshold, and minimum cluster size (small isolated mis-indexed clusters are
   removed).
5. Configure **export**: tick **Auto-export** and pick formats — **Light H5**
   (compact, recommended), **.ang** / **.ctf** (for MTEX), or **Rich H5** (full,
   large — a size warning is shown). Optionally **include EDS** in the light H5,
   and set an output directory. A live **estimated total output size** is shown.

### Step 4 — Method & Phases

6. Pick the indexing **method**: **Hough**, **Spherical**, or **Dictionary**.
7. The **Available Phases** list auto-discovers the phase files compatible with the
   chosen method (it re-scans when you return to the page, so newly simulated
   phases appear). Select the phases to index, then **Apply to All** to set them on
   every file.

### Step 5 — Method parameters

8. Set method-specific options: **Spherical** — bandwidth, n-regions, refine
   toggle; **Dictionary** — keep-N, metric (NCC/NDP); **Hough** — number of bands.

### Step 6 — Queue (review & start)

9. Review the readiness table (PC present? phases assigned? file ready?) and the
   summary "*N of M files ready, J jobs*". Click **Start Batch**. A **pre-flight
   check** runs first; if any check fails the batch is not started and the failing
   reasons are shown so you can fix and retry.

### Step 7 — Progress (dashboard)

10. Watch the live **dashboard**: overall progress bar, jobs done/failed, ETA,
    RAM gauge, the preprocessing that actually ran, the current job, and the full
    job table. Use **Pause / Resume / Stop** as needed. When complete you can
    **View Report** (an HTML summary) and, per finished file, **open the result in
    Refinement / Analysis / Phase Map**.

## Inputs & outputs

- **Inputs:**
  - A folder of EBSD files (Oxford H5OINA primarily; other HDF5/EBSP detected).
  - Discovered phase files (`.sht` / master `.h5` / `.cif`) matching the chosen
    method, from your local and (if configured) server databases.
  - Per-file pattern centres (from file header, inherited, or refined).
- **In-session outputs:**
  - A running batch with per-job status held by the batch manager (rehydrated from
    a stored batch ID after a page reload), plus a memory gauge and ETA.
- **File outputs (when auto-export is on):**
  - Per file: `result_<stem>_light.h5`, `result_<stem>.ang`, `result_<stem>.ctf`,
    and/or `result_<stem>.h5` (rich), written next to the source or in the chosen
    export directory.
  - A per-batch HTML **report**.

## Tips & notes

- **Spherical indexing needs WSL + EMSphInx.** The Spherical method runs through a
  WSL-hosted EMSphInx install; make sure it is set up (see Settings) before
  selecting it for a batch.
- **GPU helps Dictionary and Spherical.** Dictionary (GPU path) and the SHT
  spherical engine use the GPU where available and fall back to CPU otherwise —
  large batches are much faster with a CUDA GPU.
- **Mind the export size.** **Rich H5** roughly mirrors the source size (can be
  many GB across a folder); the wizard warns and turns the estimate red past
  ~10 GB. Light H5 is the recommended default for downstream tools.
- **Pattern centre first.** Files without a usable PC are flagged not-ready in the
  Step 6 readiness table; refine/inherit a PC before starting.
- **The pre-flight gate is your friend.** If Start does nothing, read the
  pre-flight failure messages — the batch is intentionally not created until the
  checks pass.
- **Long runs may be paced.** When the host is configured with conservative
  call/CPU limits, very large batches can take a long time; the ETA and RAM gauge
  help you judge progress. The dashboard rehydrates after an app restart if the
  batch ID is still known.
