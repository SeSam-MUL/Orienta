# Installing Orienta — Beginner's Guide

This is a complete, step-by-step walkthrough for installing and running
**Orienta**, an EBSD (Electron Backscatter Diffraction) pattern-analysis
application. No prior experience with Python environments or Node.js is assumed —
just follow the steps in order.

Orienta runs in two ways:

- **Browser version** — a Python web server you open in your normal web browser.
- **Desktop version** — a native desktop window (Electron) that wraps the same app.

Both versions run the *same* Python backend, so the installation steps below are
shared. The only difference is the command you use to launch the app (Step 4).

> **Quick reference (experienced users):**
> ```bash
> conda create -n ebsd python=3.11
> conda activate ebsd
> conda install -c conda-forge "blas=*=openblas"
> pip install -r requirements.txt
> cd frontend && npm install && npm run build && cd ..
> python start_app.py        # browser version
> # or: python start_desktop.py   # desktop version
> ```

---

## Step 0 — Prerequisites

Install these before you begin. The first two are **required**; the last two are
**optional** and only needed for advanced features.

### Required

1. **Anaconda or Miniconda** (a Python distribution that manages isolated
   environments). Either works; Miniconda is smaller.
   - Download: <https://www.anaconda.com/download>
   - During installation, the default options are fine. On Windows, this gives you
     an "Anaconda Prompt" in the Start menu — use that prompt for the `conda`
     commands below.

2. **Node.js 18 or newer** (used to build the user interface, and to run the
   desktop window).
   - Download: <https://nodejs.org/>
   - Pick the **LTS** version. The installer adds the `node` and `npm` commands to
     your system.

### Optional

3. **NVIDIA CUDA 12** — only if you have an NVIDIA GPU and want GPU-accelerated
   indexing and simulation. Without it, the app automatically falls back to the
   CPU (slower, but fully functional). The CUDA runtime libraries are installed
   automatically via `pip` in Step 2; you only need a recent NVIDIA driver on your
   machine.

4. **WSL + EMsoft / EMSphInx** — only if you want **master-pattern simulation** or
   **spherical indexing**. These are external scientific tools that run inside the
   Windows Subsystem for Linux (WSL). They are installed and licensed separately by
   you, and are **not** required for loading data, Hough indexing, Dictionary
   indexing, EDS analysis, phase maps, or the rest of the app.

---

## Step 1 — Get the code

Choose **one** of the following.

**Option A — Clone with Git** (recommended; makes updates easy):

```bash
git clone <your-repo-url>
cd <repo-folder>
```

**Option B — Download a ZIP:**

1. On the repository web page, click the green **Code** button → **Download ZIP**.
2. Unzip it somewhere convenient (avoid paths with unusual characters).
3. Open a terminal (Anaconda Prompt on Windows) and `cd` into the unzipped folder.

All remaining commands are run **from the repository root** (the folder that
contains `requirements.txt`, `start_app.py`, and the `frontend/` directory),
unless a step says otherwise.

---

## Step 2 — Create the Python environment

Orienta needs a dedicated Python 3.11 environment named **`ebsd`**. Run these
commands in your Anaconda Prompt / terminal:

```bash
conda create -n ebsd python=3.11
conda activate ebsd
```

> After `conda activate ebsd`, your prompt should show `(ebsd)`. Run the rest of
> the Python commands in this guide **while this environment is active**.

### 2a — Install an optimized BLAS (important!)

```bash
conda install -c conda-forge "blas=*=openblas"
```

**Why this matters:** BLAS is the math library NumPy uses for linear algebra,
which is at the heart of nearly every EBSD calculation. If NumPy ends up using
the slow *reference* BLAS, computations can be on the order of **~1000× slower**.
Installing OpenBLAS first ensures the whole stack is fast.

### 2b — Install the Python packages

```bash
pip install -r requirements.txt
```

This installs kikuchipy, orix, diffsims, FastAPI, PyTorch, and everything else
Orienta needs.

#### A note on PyTorch (GPU vs CPU)

`requirements.txt` contains this line near the PyTorch entries:

```
--extra-index-url https://download.pytorch.org/whl/cu121
torch>=2.0
torchvision>=0.15
```

The `--extra-index-url` line tells `pip` to download the **CUDA 12.1** build of
PyTorch, which enables GPU acceleration on NVIDIA hardware. This is the default
and works fine even on machines without a GPU (PyTorch simply won't use CUDA).

If you are on a **CPU-only machine** and prefer the smaller CPU build of PyTorch,
you can install plain `torch` yourself instead:

```bash
# Optional: install CPU-only PyTorch instead of the CUDA build
pip install torch torchvision
pip install -r requirements.txt
```

(Installing `torch` first means the requirements step won't pull the CUDA wheel.)
The application behaves identically; only GPU-accelerated steps differ in speed.

---

## Step 3 — Build the user interface

The web interface is a React app that must be built once before first use:

```bash
cd frontend
npm install
npm run build
cd ..
```

- `npm install` downloads the frontend dependencies (this can take a few minutes
  the first time).
- `npm run build` compiles the interface into `frontend/dist/`, which the backend
  then serves.

You only need to repeat this when the frontend source changes.

---

## Step 4 — Run Orienta

There are two ways to launch the app. Both use the **`ebsd`** conda environment
you created in Step 2, so make sure it is active (`conda activate ebsd`).

### Browser version

From the repository root:

```bash
python start_app.py
```

This starts the FastAPI backend and automatically opens the app in your default
web browser at **<http://127.0.0.1:8000>**. Leave the terminal window open while
you use the app; press **Ctrl+C** in it to stop the backend.

Two useful flags:

- `python start_app.py --headless` — the backend **keeps running even if you close
  the browser tab**. Use this for long batch jobs or simulations so a browser
  crash or accidental tab-close doesn't kill a multi-hour run. Only Ctrl+C in the
  terminal stops it.
- `python start_app.py --dev` — starts the React **dev server** (with hot reload)
  instead of serving the pre-built UI. This is for development; normal users don't
  need it.

### Desktop version

From the repository root:

```bash
python start_desktop.py
```

This opens Orienta in a **native desktop window** (Electron) instead of a browser
tab. It requires **Node.js** (Step 0) and the `npm install` from **Step 3**.

> **Important — the desktop app still needs the `ebsd` Python environment.**
> The Electron desktop window is only a shell: it launches the *same* Python
> backend behind the scenes. If the conda `ebsd` environment (Step 2) isn't set
> up, the desktop app will fail to start its backend. The Electron shell tries to
> auto-detect a conda environment that has the scientific packages installed, so
> creating the `ebsd` environment as described is all you need.

---

## First run / verify

Once the app is open (browser or desktop):

1. Go to the **EBSD Viewer** page.
2. **Load a data file** — an Oxford Instruments **`.h5oina`** file or an EDAX
   **`.h5`** file. (Large EBSD datasets are not included in this repository; use
   your own.)
3. Confirm that **patterns display** — you should see the diffraction patterns
   render in the viewer.

If patterns show up, your installation is working correctly.

---

## Troubleshooting

**"Port 8000 is already in use" / the app won't start**
Another program (often a previous Orienta backend that didn't shut down) is using
port 8000. Close the other instance, or find and stop the process using that port,
then run `python start_app.py` again.

**"npm not found" / "npm is not recognized"**
Node.js isn't installed or isn't on your PATH. Install it from
<https://nodejs.org/> (LTS version), then **open a new terminal** so the updated
PATH takes effect, and retry Step 3.

**GPU is not detected**
This is harmless — Orienta automatically **falls back to the CPU**. Everything
still works, just more slowly for GPU-accelerated steps. To use the GPU, make sure
you have a recent **NVIDIA driver** and that the CUDA build of PyTorch was
installed (the default in Step 2b).

**Simulation or spherical indexing options seem unavailable**
These rely on **EMsoft / EMSphInx via WSL**, which are **optional** external tools
you install separately (Step 0, item 4). The rest of the app — data loading, Hough
and Dictionary indexing, EDS analysis, phase maps, grain/texture analysis — works
without them.

**The desktop window opens but reports a backend error**
The Electron shell couldn't start the Python backend. Confirm the conda `ebsd`
environment exists and that `pip install -r requirements.txt` (Step 2) completed
successfully. You can verify the backend independently by running the browser
version: `python start_app.py`.

---

## Where to go next

- **User guides** for each page of the app: [`docs/user-guide/`](docs/user-guide/)
- **Project README:** [README.md](README.md)
- **Architecture overview:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
