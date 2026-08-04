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
> cd /path/to/Orienta        # <- FIRST. Must contain requirements.txt
> conda create -n ebsd python=3.11
> conda activate ebsd
> conda install -c conda-forge "blas=*=openblas"
> pip install -r requirements.txt
> cd frontend && npm install && npm run build && cd ..
> python start_app.py        # browser version
> # or: python start_desktop.py   # desktop version
> ```

> **The single most common installation failure:** running `pip install -r
> requirements.txt` from the wrong folder. A freshly opened Anaconda Prompt starts
> in your **home folder** (e.g. `C:\Users\YourName`), not in Orienta, so `pip`
> reports that `requirements.txt` does not exist. **Step 1 below shows how to get
> into the right folder — do not skip it.**

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
   Windows Subsystem for Linux (WSL).

   **You do not install these by hand.** Orienta has a built-in installer under
   **Settings → EMsoft + EMSphInx Installation** that sets up WSL, creates the
   Linux user and builds both tools for you. Do this *after* the app runs — see
   [Step 5](#step-5--optional-emsoft--emsphinx-via-the-built-in-installer).

   They are **not** required for loading data, Hough indexing, Dictionary
   indexing, EDS analysis, phase maps, or the rest of the app.

---

## Step 1 — Get the code

Choose **one** of the following.

**Option A — Clone with Git** (recommended; makes updates easy):

```bash
git clone https://github.com/SeSam-MUL/Orienta.git
```

This creates a folder named `Orienta` inside whatever folder your terminal is
currently in.

**Option B — Download a ZIP:**

1. On the [repository page](https://github.com/SeSam-MUL/Orienta), click the green
   **Code** button → **Download ZIP**.
2. Unzip it somewhere convenient (avoid paths with unusual characters).

---

## Step 1b — Go into the Orienta folder (do not skip)

Every remaining command must be run **from the Orienta folder** — the one that
contains `requirements.txt`, `start_app.py` and `frontend/`. When you open an
Anaconda Prompt it starts in your **home folder**, not in Orienta, so this is
almost always the first thing that goes wrong.

**1. Find the full path of the folder.** In Windows Explorer, open the Orienta
folder and click the address bar — it shows something like
`C:\Users\YourName\Downloads\Orienta`. Copy it.

**2. Change into it** (right-click pastes into the Anaconda Prompt):

```bash
cd C:\Users\YourName\Downloads\Orienta
```

> **Windows: if Orienta is on a different drive** (D:, E: …), plain `cd` will
> *not* switch drives in the Anaconda Prompt. Use the `/d` flag:
> ```bash
> cd /d D:\Data\Orienta
> ```

On macOS / Linux:

```bash
cd ~/Downloads/Orienta
```

**3. Verify you are in the right place.** This must print the file name, not an
error:

```bash
dir requirements.txt      # Windows
ls requirements.txt       # macOS / Linux
```

If instead you get *"File Not Found"* / *"No such file or directory"*, you are in
the wrong folder — go back to point 1. Running `pip install -r requirements.txt`
from the wrong folder is the most common installation failure, and the error it
produces (`Could not open requirements file`) does not make the cause obvious.

> Keep this terminal open for the rest of the guide. If you close it and come
> back later, you must `cd` into the folder again (and re-run
> `conda activate ebsd`).

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

## Step 5 — Optional: EMsoft + EMSphInx via the built-in installer

Skip this step unless you need **master-pattern simulation** or **spherical
indexing**. Everything else in Orienta works without it.

You do **not** need to follow EMsoft's own build instructions. Orienta ships an
installer that runs the whole chain for you — including a **required CMake
downgrade** that is easy to get wrong by hand (see the version table below).

### Where to find it

Start Orienta (Step 4), then in the left sidebar open **Settings** and scroll to
the **“EMsoft + EMSphInx Installation”** panel. It has a system check followed by
three numbered steps:

| Panel | What it does |
|---|---|
| **System Check** | Reports your platform, CPU architecture, NVIDIA GPU + driver, and whether WSL is version 1 or 2. Read the warnings here first — they explain, for example, that WSL 1 has no GPU passthrough. |
| **Step 1 — WSL Installation** | Pick the Linux distribution and click **Install WSL**. Windows asks for administrator rights (UAC) and the install runs in a separate elevated console window. Takes several minutes, and **you may have to restart your PC afterwards**. If a broken distribution is detected the button becomes **Repair WSL**. Use **Refresh Status** to re-check once it finishes (or after the restart). |
| **Step 2 — WSL User Setup** | WSL starts out with only a `root` account. Enter a username and password and click **Create User**. Remember this password — Step 3 needs it for `sudo`. (**Reset password** is there if you forget it later.) |
| **Step 3 — EMsoft + EMSphInx Installation** | Enter the password from Step 2 and click **Run in WSL**. The build log streams live into the panel. This is the long one — see the time estimate below. |

**On native Linux and macOS** there is no WSL: the panel shows **“Direct
Installation — No WSL needed”** with a single **Run Install** button, and Steps 1
and 2 are not used.

### Versions — these matter

The installer pins specific versions on purpose. **Do not substitute newer ones.**

| Component | Required version | Why |
|---|---|---|
| **CMake** | **exactly 3.27.9** | **EMsoft does not build with newer CMake.** The installer *removes* the distribution's CMake and installs 3.27.9 from Kitware into `/opt`. If you build EMsoft by hand with whatever `apt` gives you (4.x), the build fails. |
| **Ubuntu (WSL)** | **22.04 LTS (recommended)**; 20.04 and 24.04 also supported | These are the three distributions the wizard offers and the installer knows. Anything else falls back to the 22.04 code path and is untested. |
| **Clang / LLVM** | **18** | Installed from `apt.llvm.org`; required to build POCL (the OpenCL backend). |
| **WSL** | **version 2** | WSL 1 works for CPU-only builds but has **no GPU passthrough**. Upgrade with `wsl --set-version <distro> 2` in an admin PowerShell. |
| **NVIDIA driver** | **470 or newer** (only if you want GPU) | Older drivers cannot pass the GPU through to WSL 2. Without a GPU everything still runs, on CPU. |

### What to expect

- **Disk space:** about **20 GB** free inside WSL.
- **RAM:** **4 GB** minimum. The installer also creates a **16 GB swap file** if
  none exists, because the EMsoft build is memory-hungry.
- **Time:** roughly **30–90 minutes**, depending on your CPU. The log will sit on
  single compile steps for minutes at a time — that is normal, not a freeze.
- **Internet:** required throughout (sources are downloaded during the build).

### If it fails partway through

The installer records each completed phase in `~/.emsoft_install_progress` inside
WSL. **Just click “Run in WSL” again** — it skips everything that already
finished and resumes at the failed phase. You do not have to start over.

The nine phases are: pre-flight checks → swap file → build dependencies → POCL
(OpenCL) → CMake 3.27.9 → EMsoft SDK → EMsoft → EMSphInx → config and PATH →
verification.

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

**`Could not open requirements file: ... requirements.txt` / "No such file or directory"**
You are running `pip` from the wrong folder — almost certainly your home folder,
where a fresh Anaconda Prompt starts. Go back to [Step 1b](#step-1b--go-into-the-orienta-folder-do-not-skip)
and `cd` into the Orienta folder first. On Windows, if Orienta sits on another
drive, use `cd /d D:\path\to\Orienta` — plain `cd` does not switch drives.

**`'conda' is not recognized` / `python` opens the Microsoft Store**
You are in a plain Command Prompt or PowerShell instead of the **Anaconda
Prompt**. Open "Anaconda Prompt" from the Start menu and start again from
Step 1b.

**EMsoft build fails with a CMake error**
EMsoft requires **CMake 3.27.9** and does not build with newer releases. The
built-in installer (Step 5) handles this by removing the distribution's CMake
first — so use it rather than building EMsoft by hand. If you did install EMsoft
manually with a 4.x CMake, remove that CMake and re-run the installer from
**Settings → EMsoft + EMSphInx Installation**.

**The EMsoft installation stopped halfway / the log ended with an error**
Nothing is lost. Click **Run in WSL** again — the installer resumes from the last
completed phase (markers live in `~/.emsoft_install_progress` inside WSL).

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
