/**
 * Electron Main Process
 *
 * Starts the FastAPI backend as a child process,
 * waits for it to be ready, then opens the React frontend.
 */

const { app, BrowserWindow, dialog, ipcMain } = require('electron');
const { spawn, execSync } = require('child_process');
const path = require('path');
const http = require('http');
const fs = require('fs');

const BACKEND_PORT = 8000;
const FRONTEND_DEV_PORT = 5173;
const isDev = !app.isPackaged;

// --- Redirect Electron caches to project drive (avoid filling C:) ---
const projectRoot = path.resolve(__dirname, '..');
const cacheDir = path.join(projectRoot, '.cache', 'electron');
try { fs.mkdirSync(cacheDir, { recursive: true }); } catch {}
app.setPath('userData', path.join(cacheDir, 'userData'));
app.setPath('sessionData', path.join(cacheDir, 'sessionData'));
app.setPath('temp', path.join(cacheDir, 'temp'));

let mainWindow = null;
let backendProcess = null;

// Distinguish "user pressed Quit" from "renderer crashed and Electron is
// tearing things down". A 12 h batch should survive the latter — we used
// to kill the backend on every window-all-closed, which silently wiped
// in-flight jobs whenever the renderer froze under accumulated state.
let userInitiatedQuit = false;
// Set when we're quitting Electron because the renderer crashed beyond
// repair. before-quit checks this flag to leave the backend alone so the
// user can still reach a running batch at http://127.0.0.1:8000.
let keepBackendOnQuit = false;
// How many times we've auto-recovered the renderer in this session — bail
// out and quit cleanly if the renderer keeps crashing immediately on reload
// (something is fundamentally wrong, no point thrashing).
let rendererReloadCount = 0;
const MAX_RENDERER_RELOADS = 3;

function findPython() {
  // 1. Explicit override via env var or project-local config file
  if (process.env.PYTHON_PATH && fs.existsSync(process.env.PYTHON_PATH)) {
    return process.env.PYTHON_PATH;
  }
  const configFile = path.join(__dirname, '..', '.python_path');
  if (fs.existsSync(configFile)) {
    const p = fs.readFileSync(configFile, 'utf8').trim();
    if (p && fs.existsSync(p)) return p;
  }

  // 2. Dynamic scan: find conda envs that have kikuchipy + pyebsdindex
  // Checks site-packages on disk — no Python subprocess needed.
  const isWin = process.platform === 'win32';
  const home = isWin
    ? (process.env.USERPROFILE || process.env.HOMEPATH)
    : (process.env.HOME || process.env.HOMEPATH);

  // Directories that may contain conda environments
  const envRoots = [
    path.join(home, 'anaconda3', 'envs'),
    path.join(home, 'miniconda3', 'envs'),
    path.join(home, 'miniforge3', 'envs'),
    path.join(home, 'mambaforge', 'envs'),
    'C:\\ProgramData\\anaconda3\\envs',
    'C:\\ProgramData\\miniconda3\\envs',
  ];

  function pythonExe(envDir) {
    return isWin
      ? path.join(envDir, 'python.exe')
      : path.join(envDir, 'bin', 'python');
  }

  function sitePackagesDir(envDir) {
    if (isWin) return path.join(envDir, 'Lib', 'site-packages');
    // On Linux/macOS the version number is in the path — find it dynamically
    const libDir = path.join(envDir, 'lib');
    if (!fs.existsSync(libDir)) return null;
    const pyDir = fs.readdirSync(libDir).find(d => d.startsWith('python'));
    return pyDir ? path.join(libDir, pyDir, 'site-packages') : null;
  }

  function hasPkg(envDir, pkg) {
    const sp = sitePackagesDir(envDir);
    return sp && fs.existsSync(path.join(sp, pkg));
  }

  let bestWithBoth = null;
  let bestWithKikuchipy = null;

  for (const root of envRoots) {
    if (!fs.existsSync(root)) continue;
    let envNames;
    try { envNames = fs.readdirSync(root); } catch { continue; }

    for (const name of envNames) {
      const envDir = path.join(root, name);
      const pyExe = pythonExe(envDir);
      if (!fs.existsSync(pyExe)) continue;

      const hasKiki = hasPkg(envDir, 'kikuchipy');
      if (!hasKiki) continue;

      const hasPyebsd = hasPkg(envDir, 'pyebsdindex');
      if (hasPyebsd && !bestWithBoth) bestWithBoth = pyExe;
      else if (!bestWithKikuchipy) bestWithKikuchipy = pyExe;

      if (bestWithBoth) break; // Can't do better
    }
    if (bestWithBoth) break;
  }

  const found = bestWithBoth || bestWithKikuchipy;
  if (found) {
    console.log(`[findPython] Auto-detected: ${found}`);
    return found;
  }

  // 3. Last resort: system python
  console.warn('[findPython] No suitable conda env found — falling back to system python');
  return isWin ? 'python' : 'python3';
}

function startBackend() {
  const projectRoot = path.resolve(__dirname, '..');
  const pythonCmd = findPython();
  console.log(`Using Python: ${pythonCmd}`);

  backendProcess = spawn(pythonCmd, [
    '-m', 'uvicorn',
    'backend.api.main:app',
    '--host', '127.0.0.1',
    '--port', String(BACKEND_PORT),
    '--log-level', 'info',
  ], {
    cwd: projectRoot,
    stdio: ['ignore', 'pipe', 'pipe'],
    // On macOS/Linux, start the backend as its own process-group leader so
    // killBackend() can signal the whole group via process.kill(-pid) and not
    // orphan uvicorn workers / leave port 8000 bound. On Windows `detached` has
    // different semantics — taskkill /T already kills the whole tree there.
    detached: process.platform !== 'win32',
    env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1', KIKUCHIPY_WATCHDOG: '1' },
  });

  backendProcess.stdout.on('data', (data) => {
    console.log(`[Backend] ${data.toString().trim()}`);
  });

  backendProcess.stderr.on('data', (data) => {
    console.log(`[Backend] ${data.toString().trim()}`);
  });

  backendProcess.on('error', (err) => {
    console.error('Failed to start backend:', err);
    dialog.showErrorBox('Backend Error', `Failed to start Python backend: ${err.message}`);
  });

  backendProcess.on('exit', (code) => {
    console.log(`Backend exited with code ${code}`);
    backendProcess = null;
  });
}

function waitForBackend(retries = 30) {
  return new Promise((resolve, reject) => {
    let attempts = 0;

    const check = () => {
      attempts++;
      const req = http.get(`http://127.0.0.1:${BACKEND_PORT}/api/health`, (res) => {
        if (res.statusCode === 200) {
          resolve();
        } else if (attempts < retries) {
          setTimeout(check, 500);
        } else {
          reject(new Error('Backend health check failed'));
        }
      });

      req.on('error', () => {
        if (attempts < retries) {
          setTimeout(check, 500);
        } else {
          reject(new Error('Backend not reachable after 15 seconds'));
        }
      });

      req.setTimeout(2000, () => {
        req.destroy();
        if (attempts < retries) {
          setTimeout(check, 500);
        } else {
          reject(new Error('Backend timeout'));
        }
      });
    };

    check();
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    title: 'Orienta - EBSD Pattern Analysis',
    backgroundColor: '#282a36',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
    autoHideMenuBar: true,
    icon: path.join(__dirname, '..', 'resources', 'icon.png'),
  });

  if (isDev) {
    mainWindow.loadURL(`http://localhost:${FRONTEND_DEV_PORT}`);
  } else {
    // In production, load from the backend server (which serves the built frontend)
    // Using file:// would break relative API URLs
    mainWindow.loadURL(`http://127.0.0.1:${BACKEND_PORT}`);
  }

  // DevTools: press F12 to toggle manually
  mainWindow.webContents.on('before-input-event', (event, input) => {
    if (input.key === 'F12') mainWindow.webContents.toggleDevTools();
  });

  // Renderer-crash recovery: a frozen / OOM'd renderer used to take the
  // backend with it (window-all-closed fires after the renderer dies →
  // killBackend() → SIGTERM to Python → 12 h batch dead).  Reload the
  // window instead and leave the backend alone.
  mainWindow.webContents.on('render-process-gone', (event, details) => {
    console.error(`[Renderer] gone: reason=${details.reason} exitCode=${details.exitCode}`);
    if (userInitiatedQuit) return;
    if (rendererReloadCount >= MAX_RENDERER_RELOADS) {
      console.error(`[Renderer] crashed ${rendererReloadCount} times — giving up.`);
      // Crucial: tag this so the upcoming window-all-closed / before-quit
      // chain leaves the backend alive — a long-running batch can still
      // be reached at http://127.0.0.1:BACKEND_PORT from any browser.
      keepBackendOnQuit = true;
      dialog.showErrorBox(
        'Renderer keeps crashing',
        `The UI process crashed ${rendererReloadCount} times (last: ${details.reason}). ` +
        `The backend is still running — open http://127.0.0.1:${BACKEND_PORT} in your browser ` +
        `to keep using it. Restart the app when you're ready.`
      );
      return;
    }
    rendererReloadCount += 1;
    console.log(`[Renderer] reloading window (attempt ${rendererReloadCount}/${MAX_RENDERER_RELOADS})…`);
    try {
      mainWindow.reload();
    } catch (err) {
      console.error('[Renderer] reload failed:', err.message);
    }
  });

  // An unresponsive renderer (GUI thread stuck) is the precursor to a
  // crash. Force a reload so we don't end up with the grey screen the
  // user reported.
  mainWindow.on('unresponsive', () => {
    console.warn('[Renderer] unresponsive — forcing reload to keep backend alive');
    if (userInitiatedQuit) return;
    try { mainWindow.reload(); } catch {}
  });

  mainWindow.on('responsive', () => {
    console.log('[Renderer] back to responsive');
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// Handle file open dialog from renderer
ipcMain.handle('dialog:openFile', async (event, options) => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: options?.filters || [
      { name: 'HDF5 Files', extensions: ['h5oina', 'h5', 'hdf5'] },
      { name: 'CIF Files', extensions: ['cif'] },
      { name: 'All Files', extensions: ['*'] },
    ],
  });
  return result.canceled ? null : result.filePaths[0];
});

ipcMain.handle('dialog:openFolder', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory'],
  });
  return result.canceled ? null : result.filePaths[0];
});

ipcMain.handle('dialog:saveFile', async (event, options) => {
  const result = await dialog.showSaveDialog(mainWindow, {
    filters: options?.filters || [
      { name: 'PNG Image', extensions: ['png'] },
      { name: 'Excel', extensions: ['xlsx'] },
      { name: 'All Files', extensions: ['*'] },
    ],
  });
  return result.canceled ? null : result.filePath;
});

const poleFigureWindows = new Set();

ipcMain.handle('window:openPoleFigure', () => {
  const base = isDev
    ? `http://localhost:${FRONTEND_DEV_PORT}`
    : `http://127.0.0.1:${BACKEND_PORT}`;
  const win = new BrowserWindow({
    width: 1000,
    height: 820,
    title: 'Pole Figures',
    backgroundColor: '#282a36',
    parent: mainWindow,            // closes with the main window
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
    autoHideMenuBar: true,
  });
  win.loadURL(`${base}/?view=polefigure`);
  win.on('closed', () => poleFigureWindows.delete(win));
  poleFigureWindows.add(win);
  return true;
});

app.whenReady().then(async () => {
  console.log('Starting Orienta...');

  startBackend();

  try {
    console.log('Waiting for backend...');
    await waitForBackend();
    console.log('Backend ready!');
  } catch (err) {
    console.error('Backend failed to start:', err.message);
    dialog.showErrorBox(
      'Backend Startup Failed',
      'The Python backend could not be started.\n\n' +
      'Make sure Python and the required packages are installed.\n\n' +
      `Error: ${err.message}`
    );
  }

  createWindow();
});

function killBackend() {
  if (!backendProcess) return;
  const pid = backendProcess.pid;
  console.log(`Stopping backend (PID ${pid})...`);

  // 1. Graceful: ask backend to shut itself down via API
  try {
    const req = http.request(
      { hostname: '127.0.0.1', port: BACKEND_PORT, path: '/api/shutdown', method: 'POST', timeout: 2000 },
      () => {}
    );
    req.on('error', () => {});
    req.end();
  } catch {}

  // 2. Force-kill after short delay to ensure it's gone
  // PID is a number from child_process.spawn — safe to interpolate
  setTimeout(() => {
    if (!backendProcess) return;
    try {
      if (process.platform === 'win32') {
        execSync(`taskkill /PID ${pid} /T /F`, { stdio: 'ignore' });
      } else {
        process.kill(-pid, 'SIGTERM');
      }
    } catch {
      try { backendProcess.kill('SIGKILL'); } catch {}
    }
    backendProcess = null;
  }, 1000);
}

app.on('window-all-closed', () => {
  // Default behaviour (user closed the window normally): kill backend and
  // quit Electron. Same as before this fix.
  //
  // Override: when the renderer crashed beyond MAX_RENDERER_RELOADS, the
  // render-process-gone handler set keepBackendOnQuit=true. We then quit
  // Electron but leave the Python backend running so a 12 h batch can be
  // reattached at http://127.0.0.1:BACKEND_PORT from any browser.
  // KIKUCHIPY_KEEP_BACKEND=1 forces the keep-alive path on every close.
  if (process.env.KIKUCHIPY_KEEP_BACKEND === '1') keepBackendOnQuit = true;
  if (keepBackendOnQuit) {
    console.log(`[App] Window gone, backend kept alive. ` +
                `Open http://127.0.0.1:${BACKEND_PORT} in a browser to keep using it.`);
  }
  app.quit();
});

app.on('before-quit', () => {
  // before-quit fires for every quit path. Honour keepBackendOnQuit so
  // a renderer-crash quit doesn't accidentally tear down the backend
  // (the very thing we're trying to protect).  Vite cleanup runs either
  // way — it's frontend-only.
  if (!keepBackendOnQuit) {
    userInitiatedQuit = true;
    killBackend();
  }
  const viteCacheDir = path.join(projectRoot, 'frontend', 'node_modules', '.vite');
  try { fs.rmSync(viteCacheDir, { recursive: true, force: true }); } catch {}
  const tempDir = path.join(cacheDir, 'temp');
  try { fs.rmSync(tempDir, { recursive: true, force: true }); } catch {}
});
