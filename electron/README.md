# `electron/` — Desktop Shell

This directory contains the Electron desktop shell for **Orienta**, the EBSD pattern-analysis
application. The shell turns the web app into a self-contained desktop program: on launch it
locates a suitable Python interpreter, spawns the FastAPI backend as a child process, waits for
the backend's health check to pass, and then opens a native window that loads the React frontend.
It also bridges a small set of native desktop capabilities (file/folder/save dialogs, a detached
pole-figure window) to the renderer over a context-isolated IPC layer, and adds resilience so that
a long-running computation survives a frozen or crashed UI process.

## Files

| File | Description |
|------|-------------|
| [`main.js`](main.js) | Electron **main process**: finds Python, starts the backend, waits for readiness, creates the `BrowserWindow`, registers IPC handlers, and manages shutdown/crash recovery. |
| [`preload.js`](preload.js) | **Preload script**: runs with context isolation and exposes a minimal, safe `window.electronAPI` to the renderer via `contextBridge`. |

There are no subdirectories.

## How it works

### Locating Python (`findPython`)
The shell needs a Python environment that has the scientific stack installed. `findPython()`
resolves it in priority order, checking files on disk only (no Python subprocess is launched):

1. **Explicit override** — the `PYTHON_PATH` environment variable, or a project-root
   `.python_path` file containing the interpreter path.
2. **Conda auto-scan** — it walks common conda env roots (`anaconda3`, `miniconda3`,
   `miniforge3`, `mambaforge` under the user home, plus `C:\ProgramData\...` on Windows),
   inspecting each env's `site-packages`. It prefers an env that has both `kikuchipy` **and**
   `pyebsdindex`, falling back to one that has only `kikuchipy`.
3. **System fallback** — `python` (Windows) or `python3` (otherwise).

### Starting and waiting for the backend
`startBackend()` spawns the FastAPI app with
`python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000`, using the project root as
the working directory and piping backend stdout/stderr into the Electron console (prefixed
`[Backend]`). `waitForBackend()` then polls `http://127.0.0.1:8000/api/health` (up to ~15 s) before
the window is shown; if the backend never becomes ready, an error dialog is displayed.

### Loading the frontend
`createWindow()` opens a `1400×900` `BrowserWindow` (Dracula-dark background, menu auto-hidden,
`F12` toggles DevTools). The URL it loads depends on the build mode:

- **Dev** (`!app.isPackaged`): `http://localhost:5173` (the Vite dev server).
- **Production**: `http://127.0.0.1:8000` — the **built React app is served by the backend
  itself**, so relative API URLs resolve correctly (loading via `file://` would break them).

### Crash resilience
A frozen or out-of-memory renderer used to take the backend down with it (killing in-flight
batches). The shell now distinguishes a user-initiated quit from a renderer crash:

- `render-process-gone` reloads the window (up to `MAX_RENDERER_RELOADS = 3`) instead of killing
  the backend; after repeated crashes it leaves the backend running and tells the user they can
  reach it at `http://127.0.0.1:8000` from any browser.
- An `unresponsive` renderer is force-reloaded.
- `KIKUCHIPY_KEEP_BACKEND=1` forces the keep-backend-alive path on every window close.

`killBackend()` shuts the backend down gracefully via `POST /api/shutdown`, then force-kills the
process tree after a short delay (`taskkill /T /F` on Windows). Electron caches/temp are redirected
to `<projectRoot>/.cache/electron` to avoid filling the system drive.

## IPC surface

`preload.js` exposes exactly these methods on `window.electronAPI`; each forwards to a handler
registered in `main.js`:

| `window.electronAPI` method | Main-process channel | Purpose |
|-----------------------------|----------------------|---------|
| `openFile(options)` | `dialog:openFile` | Native open dialog (defaults to HDF5/`.h5oina`/`.h5`/`.hdf5`, CIF, all files); returns the selected path or `null`. |
| `openFolder()` | `dialog:openFolder` | Native folder picker; returns the directory path or `null`. |
| `saveFile(options)` | `dialog:saveFile` | Native save dialog (defaults to PNG/Excel/all files); returns the chosen path or `null`. |
| `openPoleFigure()` | `window:openPoleFigure` | Opens a detached child window at `…/?view=polefigure` for pole-figure display. |

`nodeIntegration` is disabled and `contextIsolation` is enabled, so the renderer only ever sees
the curated `electronAPI` object — never raw Node or `ipcRenderer`.

## Where this fits in the Orienta architecture

```
electron/  (this directory)  ── desktop shell, process supervisor
   ├─ spawns ─▶ FastAPI backend   ../backend/  (uvicorn: backend.api.main:app, port 8000)
   └─ loads  ─▶ React frontend    ../frontend/ (Vite dev server in dev; backend-served build in prod)
```

The shell owns process lifecycle and native desktop integration only; all scientific logic lives
in the backend ([`../backend/`](../backend/)) and the UI lives in the frontend
([`../frontend/`](../frontend/)). See [`../backend/api/main.py`](../backend/api/main.py) for the
backend entry point and [`../frontend/src/main.jsx`](../frontend/src/main.jsx) for the React entry
point.

## Run notes

- **Desktop dev (Electron):** start the Vite dev server first (`cd frontend && npm run dev`), then
  launch the Electron shell (`cd frontend && npm run electron:dev`). In dev mode the window loads
  `http://localhost:5173`.
- **Backend-only / browser mode:** you do not need Electron to run Orienta — the backend can be
  started directly and used in a browser. See [`../start_app.py`](../start_app.py) and the root
  project docs.
- **Python selection:** if auto-detection picks the wrong interpreter, set the `PYTHON_PATH`
  environment variable or create a `.python_path` file in the project root with the absolute path
  to the desired `python` executable.
- **Long-running jobs:** if the UI window crashes during a long computation, the backend is kept
  alive — reopen `http://127.0.0.1:8000` in any browser to continue, then restart the app.
