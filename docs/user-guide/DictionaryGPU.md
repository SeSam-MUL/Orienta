# Dictionary (GPU)

> **Status: experimental / internal tool — not part of the standard workflow.**
> This page exists in the codebase and has a working backend, but it is **not
> mounted in the application shell**: there is no sidebar entry and no route wired
> into the main window, so it is not reachable through normal navigation in the
> shipped build. It is documented here for developers and advanced users who
> invoke the backend endpoints directly. Routine dictionary generation should be
> done through the [Indexing](../README.md) and [Simulation](Simulation.md) pages.

## What it does

The Dictionary (GPU) tool generates a **simulated EBSD pattern dictionary** from
an existing master pattern, accelerated on the GPU. Given a master pattern and a
detector geometry, it samples a grid of crystal orientations over SO(3) and
forward-projects each orientation into a simulated detector pattern, then saves
the whole set to disk. That dictionary is the reference set against which
experimental patterns are matched during dictionary indexing.

The standalone tool deliberately has **no shared application state**: you point it
at a master-pattern file and configure everything on the page, so the GPU
dictionary generator can be iterated on in isolation.

## When to use it

Conceptually, dictionary generation sits between **simulation** (which produces
the master pattern) and **dictionary indexing** (which consumes the dictionary).

In the shipped application you do **not** use this page for that step — dictionary
generation is handled inside the normal indexing/simulation flow. Use this
standalone tool only when you are:

- Developing or benchmarking the GPU dictionary-generation pipeline itself, or
- Driving the `/api/dictionary-gpu/*` endpoints directly for scripting/testing.

## How to use it — step by step

*(Applies when the page is run directly, e.g. during development.)*

1. **Master pattern** — paste the full path to an EMsoft master-pattern `.h5`
   file. The path is validated by the backend when you generate.
2. **Detector settings**:
   - **Pattern size** — detector height × width in pixels (default 60 × 60).
   - **PC (x, y, z)** — the pattern centre as three fractional values
     (default 0.5, 0.5, 0.5).
   - **Sample tilt** — sample tilt in degrees (default 70°).
3. **Orientation settings**:
   - **Resolution** — the SO(3) sampling step in degrees (default 5°). Smaller
     values produce far more orientations (the count grows steeply as resolution
     decreases), so they take longer and produce larger files.
   - **Normalize** — optional per-pattern normalisation of the generated
     patterns.
4. Click **Generate**. The tool submits the job and polls progress roughly twice
   a second, showing a progress bar and status message. The button is disabled
   until a master path is set and while a job is running.
5. When the job finishes, the **result** shows the number of generated patterns
   and the output file path. The new dictionary appears in the list below.
6. The **dictionaries list** shows previously generated dictionaries (name,
   pattern count, file size, creation time) with **Refresh** and per-row
   **Delete** (with confirmation).

## Inputs & outputs

**Inputs**

- An EMsoft **master-pattern `.h5`** file (path).
- Detector geometry: pattern shape, pattern centre, sample tilt.
- Orientation sampling: angular resolution (degrees) and the normalise flag.

**Outputs**

- A generated **dictionary `.h5`** file plus a JSON sidecar with metadata,
  written into the project's `tasks/` directory.
- Metadata reported back to the UI: number of orientations/patterns and the
  output path.
- Delete removes both the `.h5` and its `.json` sidecar.

## Tips & notes

- **Not in the standard UI.** As noted above, there is no sidebar button or route
  for this page in the shipped build — it is an internal/experimental tool.
- **GPU is optional; it falls back to CPU.** The backend uses CUDA when a
  compatible GPU is available and automatically falls back to CPU otherwise.
  Generation is dramatically faster on a CUDA GPU; on CPU it still works but is
  much slower.
- **VRAM-aware and OOM-resilient.** On GPU the batch size is chosen from available
  VRAM, and if a batch runs out of memory the pipeline halves the batch size and
  retries — so out-of-memory conditions are handled gracefully rather than
  crashing the job.
- **Resolution drives cost.** Halving the angular resolution increases the number
  of orientations sharply, which increases both the generation time and the
  output file size. Start coarse (e.g. 5°) and refine only if needed.
- **Output lands in `tasks/`.** Generated dictionaries and their sidecars are
  written under the project's `tasks/` directory regardless of where the process
  was started from.
