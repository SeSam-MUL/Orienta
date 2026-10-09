/**
 * Electron Main Process
 *
 * Starts the FastAPI backend as a child process,
 * waits for it to be ready, then opens the React frontend.
 */

const { app, BrowserWindow, dialog, ipcMain, shell } = require('electron');
const platformInfo = require('./platform.js');
const removeDataModule = require('./remove_data.js');
const saveDialogs = require('./save_dialogs.js');
const { openLogFolder } = require('./log_folder.js');
const { spawn, execSync } = require('child_process');
const path = require('path');
const http = require('http');
const fs = require('fs');

// ONE answer to "what should this launch do", read by createWindow AND by
// whenReady. Asking it two different ways in two places is how a user with a
// deleted interpreter ended up staring at ERR_CONNECTION_REFUSED forever.
const {
  orientaHome,
  startupDecision,
  startupPlan,
  checkoutRoot,
  resolveProjectRoot,
  openStartupLog,
  logShellLine,
} = require('./project_root');
const { releasesPageUrl, setOrientaHome, homePointerFile } = require('./update_endpoints');
const startLanguage = require('./start_language');
const { t, shellLanguage, preferredLocale } = require('./strings');
const { createWaitingPage } = require('./waiting_page');

// What the waiting page says, by PHASE ('starting' | 'updating' | 'syncing').
// The one-second clock and the update steps both paint from it, so a message
// set by a step is not overwritten a second later by the clock's own text.
// `shellLocale` is a function declaration below and is only called at paint.
const waitingPage = createWaitingPage({ t, shellLanguage, getLocale: () => shellLocale() });

/**
 * The locale to show the user, asked of the SYSTEM rather than the bundle.
 *
 * `uiLocale()` negotiates against the localizations the .app declares,
 * and Orienta declares none -- so on macOS it says "en" on a German Mac.
 * The setup wizard opened in English there on 2026-09-25 for exactly this
 * reason; its German strings existed the whole time.
 *
 * Not a no-op off macOS, despite what an earlier version of this comment
 * said: a Windows user whose display language is English but whose
 * preferred-language list starts with German now gets German. That is the
 * better answer, but it is a change, not an identity.
 */
function uiLocale() {
  const preferred = typeof app.getPreferredSystemLanguages === 'function'
    ? app.getPreferredSystemLanguages()
    : null;
  return preferredLocale(preferred, app.getLocale());
}

const BACKEND_PORT = 8000;
const FRONTEND_DEV_PORT = 5173;
const isDev = !app.isPackaged;

// --- Redirect Electron caches to project drive (avoid filling C:) ---
// In a PACKAGED build these must not live inside the application directory: an
// NSIS reinstall or update deletes it, taking the user's language choice,
// recent files and window geometry with it. The Orienta home already holds the
// crystal library and survives both. In development, today's behaviour is kept
// exactly as it was.
const projectRoot = path.resolve(__dirname, '..');

/**
 * Why the Orienta home could not be used for Chromium's own storage, or ''.
 *
 * Handing Chromium a directory it cannot write KILLS THE APP at startup, with
 * no window and no log -- before a single line of our code can explain
 * anything. And the way to get there is not exotic: ORIENTA_HOME on a drive
 * that is not mounted, or a managed profile whose LOCALAPPDATA is redirected
 * to a share that is offline. That is the same machine the setup's disk probe
 * is written for, so failing silently here means the one screen built to
 * explain it never appears.
 *
 * So: prove the directory is writable first, and fall back to Electron's own
 * default paths if it is not. The app then starts, and the wizard can say what
 * is wrong.
 */
let homeStorageError = '';
function usableCacheDir() {
  if (!app.isPackaged) return path.join(projectRoot, '.cache', 'electron');
  try {
    const dir = path.join(orientaHome(), 'electron');
    fs.mkdirSync(dir, { recursive: true });
    // mkdir alone is not proof: a read-only share accepts the call for an
    // existing directory and refuses the first write.
    const probe = path.join(dir, '.writable');
    fs.writeFileSync(probe, '');
    fs.rmSync(probe, { force: true });
    return dir;
  } catch (err) {
    homeStorageError = String((err && err.message) || err);
    return null;   // Electron keeps its defaults
  }
}

const cacheDir = usableCacheDir();
if (cacheDir) {
  app.setPath('userData', path.join(cacheDir, 'userData'));
  app.setPath('sessionData', path.join(cacheDir, 'sessionData'));
  app.setPath('temp', path.join(cacheDir, 'temp'));
}

// macOS builds the application menu from `app.name`: the first menu itself, and
// the "About X" / "Quit X" items. Electron reads that from package.json `name`,
// which is "kikuchipy-gui" — `productName` sits under `build`, where only
// electron-builder looks. So the M5 tester read "About kikuchipy-gui" and
// "Quit kikuchipy-gui" in the menu bar of the signed dmg (report
// tasks/mac-test-m5-2026-09-25/bericht.md, point 4).
//
// Naming it here rather than adding a top-level `productName`, because
// `setName` also moves the DEFAULT userData directory — and that directory
// holds the renderer's localStorage: saved phase colours, the dashboard
// background, UI preferences. Adding `productName` to package.json would
// change `app.name` before this file runs, and the pin below could then only
// capture the already-moved path.
//
// HOW THE PIN WORKS, measured against the bundled Electron 44.4.3: Chromium's
// PathService caches the first resolution, so the `getPath` READ on the next
// line is what fixes the directory — in both branches, whether or not the
// block above already overrode it. Reading it looks pointless and is the
// load-bearing part; the `setPath` in the `if` is a belt that never tightens
// under this Electron. It is kept because "caches the first read" is an
// implementation detail, not a promise, and wrapped because setPath throws
// when the directory does not exist — at module scope that would kill the app
// with no window and no log (see the note at the top of this file).
//
// Moving that directory on purpose is a migration, like the backend config
// directory — see tasks/mac-tester/naming-audit.md.
const userDataBeforeRename = app.getPath('userData');   // this read is the pin
app.setName('Orienta');
if (app.getPath('userData') !== userDataBeforeRename) {
  try {
    app.setPath('userData', userDataBeforeRename);
  } catch (err) {
    console.error('[orienta] could not pin userData after rename:', err);
  }
}

let mainWindow = null;
let backendProcess = null;
// What the setup page should show when it is loaded. A query string would
// not survive a reload and additionalArguments would not survive a
// navigation, so Task 12 exposes this over its own IPC channel.
let setupContext = { reason: 'first-run', message: '' };
// Returned by startWaitingPageClock; called once the backend answers or fails.
let stopWaitingClock = null;

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

function findPython({ allowSystemFallback = false } = {}) {
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

  // 3. Last resort: system python — ONLY when the caller asks for it.
  //
  // On a clean Windows machine `python` is an App Execution Alias: spawning it
  // opens the Microsoft Store and exits 9009, maybeRestartBackend tries three
  // more times, and the user gets two modal error boxes stacked over the setup
  // wizard. An installed copy must never reach this line — it has
  // `startupDecision()`, which looks only at its own .python_path.
  if (!allowSystemFallback) {
    logShellLine('[findPython] nothing installed, and no conda env found');
    return null;
  }
  console.warn('[findPython] No suitable conda env found — falling back to system python');
  return isWin ? 'python' : 'python3';
}

// --- Persistent capture of the backend's raw stdout/stderr ---------------
// The pipes below are the ONLY place that sees import-time crashes (a missing
// package kills uvicorn before any Python log handler exists), print() output
// from scientific libs, and CUDA C++-level stderr. console.log alone is lost
// the moment the hosting console closes — and a packaged app has none at all.
let backendLogStream = null;

function openBackendLog(projectRoot) {
  try {
    const logDir = path.join(projectRoot, 'logs');
    fs.mkdirSync(logDir, { recursive: true });
    const logPath = path.join(logDir, 'backend-console.log');
    // One-generation rotation: keep the previous session reachable, cap growth.
    try {
      const st = fs.statSync(logPath);
      if (st.size > 2 * 1024 * 1024) {
        fs.rmSync(logPath + '.1', { force: true });
        fs.renameSync(logPath, logPath + '.1');
      }
    } catch {}
    backendLogStream = fs.createWriteStream(logPath, { flags: 'a' });
    backendLogStream.write(`\n===== session start ${new Date().toISOString()} =====\n`);
  } catch (err) {
    console.error('Could not open backend-console.log:', err);
    backendLogStream = null;
  }
}

function logBackendLine(line) {
  console.log(`[Backend] ${line}`);
  if (backendLogStream) {
    try { backendLogStream.write(line + '\n'); } catch {}
  }
}

// Remembered so maybeRestartBackend's setTimeout can restart with the same
// interpreter rather than asking a question whose answer may have changed.
let lastPythonCmd = null;
let lastProjectRoot = null;

function startBackend(pythonCmd, projectRootArg) {
  // The root comes from the PLAN, never re-derived here. Re-deriving it meant
  // that the moment an installation existed on a developer's machine,
  // `resolveProjectRoot()` returned the installed runtime and dev mode ran the
  // backend out of it — edits to backend/ would have had no effect at all, with
  // nothing said. Two answers to one question, which is the defect this whole
  // task exists to remove.
  const projectRoot = projectRootArg || lastProjectRoot || resolveProjectRoot();
  lastProjectRoot = projectRoot;
  // `pythonCmd` comes from startupDecision() on an installed machine and from
  // findPython() only in a development checkout. The two were previously the
  // same call, and findPython() reads `<app>/.python_path` while the setup
  // writes `<orientaHome>/.python_path` — two different files, so an installed
  // Orienta could never start its own backend.
  const resolved = pythonCmd || lastPythonCmd || findPython({ allowSystemFallback: isDev });
  openBackendLog(projectRoot);
  if (!resolved) {
    logShellLine('No Python interpreter available — not starting the backend.');
    dialog.showErrorBox(
      t(shellLocale(), 'noInterpreterTitle'),
      t(shellLocale(), 'noInterpreterBody'),
    );
    return;
  }
  lastPythonCmd = resolved;
  const pythonCmdResolved = resolved;
  logBackendLine(`Using Python: ${pythonCmdResolved}`);

  backendProcess = spawn(pythonCmdResolved, [
    '-m', 'uvicorn',
    'backend.api.main:app',
    '--host', '127.0.0.1',
    '--port', String(BACKEND_PORT),
    '--log-level', 'info',
  ], {
    cwd: projectRoot,
    stdio: ['ignore', 'pipe', 'pipe'],
    // No console window for the backend on Windows. Without this every spawn
    // flashed a console — harmless once, but a restart loop against a busy
    // port opened and closed them "wie wild" and the desktop was unusable.
    windowsHide: true,
    // On macOS/Linux, start the backend as its own process-group leader so
    // killBackend() can signal the whole group via process.kill(-pid) and not
    // orphan uvicorn workers / leave port 8000 bound. On Windows `detached` has
    // different semantics — taskkill /T already kills the whole tree there.
    detached: process.platform !== 'win32',
    env: {
      ...process.env,
      PYTHONDONTWRITEBYTECODE: '1',
      KIKUCHIPY_WATCHDOG: '1',
      // Last line of defence: the backend polls this PID and exits when we are
      // gone. Covers the paths our own cleanup cannot — Electron being killed
      // outright, or the user closing the console that hosts npm/concurrently.
      KIKUCHIPY_PARENT_PID: String(process.pid),
      // "the parent of this backend is a UI that owns a window". The backend's
      // frontend watchdog trusts a live parent as proof that a user is there,
      // and that is only true HERE: start_app.py also passes its PID, but it
      // blocks in backend_proc.wait(), so its liveness says nothing about
      // whether a browser tab is still open. Without this flag the watchdog
      // could never exit on that path at all.
      KIKUCHIPY_UI_PARENT: '1',
    },
  });

  backendProcess.stdout.on('data', (data) => {
    logBackendLine(data.toString().trim());
  });

  backendProcess.stderr.on('data', (data) => {
    logBackendLine(data.toString().trim());
  });

  backendProcess.on('error', (err) => {
    logBackendLine(`Failed to start backend: ${err.message}`);
    console.error('Failed to start backend:', err);
    dialog.showErrorBox('Backend Error', `Failed to start Python backend: ${err.message}`);
  });

  backendProcess.on('exit', (code) => {
    logBackendLine(`Backend exited with code ${code}`);
    backendProcess = null;
    maybeRestartBackend(code);
  });
}

// --- Is somebody else already on our port? -------------------------------
// A backend left over from an earlier session (or started by hand) keeps
// port 8000. Our own backend then fails to bind, exits with code 1 after a
// few seconds, and a naive restart loop does that three times in a row —
// measured 2026-09-09: three console windows opening and closing while the
// page happily talked to the stale backend. So: ask the port first, and if
// it answers, ask the USER what to do instead of spawning anything.

function probeBackendHealth(timeoutMs = 1500) {
  return new Promise((resolve) => {
    const req = http.get(`http://127.0.0.1:${BACKEND_PORT}/api/health`, (res) => {
      let body = '';
      res.on('data', (d) => { body += d; });
      res.on('end', () => {
        try { resolve(res.statusCode === 200 ? JSON.parse(body) : null); }
        catch { resolve(res.statusCode === 200 ? {} : null); }
      });
    });
    req.on('error', () => resolve(null));
    req.setTimeout(timeoutMs, () => { req.destroy(); resolve(null); });
  });
}

function portOwnerPid() {
  try {
    if (process.platform === 'win32') {
      const out = execSync('netstat -ano -p tcp', { encoding: 'utf8' });
      const line = out.split(/\r?\n/).find((l) => l.includes(`:${BACKEND_PORT} `) && l.includes('LISTENING'));
      const pid = line && line.trim().split(/\s+/).pop();
      return pid ? Number(pid) : null;
    }
    const out = execSync(`lsof -ti tcp:${BACKEND_PORT} -sTCP:LISTEN`, { encoding: 'utf8' });
    const pid = out.trim().split(/\s+/)[0];
    return pid ? Number(pid) : null;
  } catch {
    return null;
  }
}

function killPid(pid) {
  try {
    if (process.platform === 'win32') execSync(`taskkill /PID ${Number(pid)} /T /F`, { stdio: 'ignore' });
    else process.kill(pid, 'SIGTERM');
    return true;
  } catch {
    return false;
  }
}

async function waitForPortFree(maxMs = 10000) {
  const deadline = Date.now() + maxMs;
  while (Date.now() < deadline) {
    if (!(await probeBackendHealth(500))) return true;
    await new Promise((r) => setTimeout(r, 300));
  }
  return false;
}

/**
 * If a backend already answers on the port, let the user decide: stop it and
 * start ours (the default — a stale backend can hold dead file handles after
 * a drive was unplugged, and then fails every load), or keep using it (a deliberate
 * `--headless` batch, for instance). Returns true when it is safe to spawn.
 */
async function resolveBusyPort() {
  const health = await probeBackendHealth();
  if (!health) return true; // nobody there — go ahead
  const pid = portOwnerPid();
  const who = pid ? `process ${pid}` : 'another process';
  const py = health.python_executable ? `\n(${health.python_executable})` : '';
  logBackendLine(`Port ${BACKEND_PORT} already answers /api/health — owned by ${who}`);
  const choice = dialog.showMessageBoxSync({
    type: 'question',
    title: 'A backend is already running',
    message: `Port ${BACKEND_PORT} is already used by an Orienta backend (${who}).${py}`,
    detail: 'Usually this is a backend left over from an earlier session. Stopping it and ' +
            'starting a fresh one is the safe choice; keep it only if you know it is running a job.',
    buttons: ['Stop it and start fresh', 'Keep using it'],
    defaultId: 0,
    cancelId: 1,
    noLink: true,
  });
  if (choice !== 0) {
    logShellLine('User chose to keep the existing backend; not starting our own.');
    keepBackendOnQuit = true; // it is not ours to kill on quit either
    return false;
  }
  if (!pid || !killPid(pid) || !(await waitForPortFree())) {
    dialog.showErrorBox('Could not free the port',
      `The process on port ${BACKEND_PORT} could not be stopped. Stop it yourself (Task Manager, ` +
      `python.exe started ${health.python_executable || ''}) and start Orienta again.`);
    return false;
  }
  logBackendLine(`Stopped stale backend ${pid}; port ${BACKEND_PORT} is free.`);
  return true;
}

// A backend that dies while the window is open (out of memory on a large
// scan — bug report #4/8 —, a crash in a scientific library) used to leave
// the page on "Backend not connected" until the user restarted the whole
// app. Start it again, a few times at most: a backend that dies on every
// start is a configuration problem the dialog should report, not a loop.
const MAX_BACKEND_RESTARTS = 3;
let backendRestarts = 0;

async function maybeRestartBackend(code) {
  if (userInitiatedQuit || keepBackendOnQuit) return;
  // Exit code 1 within seconds of spawning is "could not bind": do not loop,
  // resolve the port instead (asks the user, stops the stale one on request).
  if (await probeBackendHealth()) {
    logBackendLine('Backend exited but the port still answers — another backend owns it.');
    if (await resolveBusyPort()) {
      startBackend();
    }
    return;
  }
  if (backendRestarts >= MAX_BACKEND_RESTARTS) {
    logBackendLine(`Backend died ${backendRestarts} times — not restarting again.`);
    dialog.showErrorBox(
      'Backend keeps stopping',
      `The Python backend stopped ${backendRestarts} times in this session (last exit code ${code}). ` +
      'See logs/backend-console.log for the reason, then restart the app.'
    );
    return;
  }
  backendRestarts += 1;
  logBackendLine(`Backend stopped unexpectedly (exit code ${code}) — restarting (${backendRestarts}/${MAX_BACKEND_RESTARTS}) in 1 s`);
  setTimeout(() => {
    if (userInitiatedQuit) return;
    startBackend();
    waitForBackend().then(
      () => logBackendLine('Backend back after restart'),
      (err) => logBackendLine(`Backend did not come back: ${err.message}`),
    );
  }, 1000);
}

// 360 x 500 ms = 3 minutes. The first start of a fresh installation imports
// the scientific stack cold and blew past the old 15 s (30 retries): the
// "Backend Startup Failed" dialog appeared over a backend that was still
// loading. Found by tasks/install_smoke/fresh_install.ps1; start_app.py has
// the same limit (BACKEND_START_TIMEOUT_S).
const BACKEND_START_RETRIES = 360;

function waitForBackend(retries = BACKEND_START_RETRIES) {
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
          reject(new Error(`Backend not reachable after ${Math.round(retries * 0.5)} seconds`));
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

/**
 * The page shown when there is nothing installed yet, or when what is installed
 * cannot be used.
 *
 * It loads the real wizard, `electron/setup/index.html`, which ships in the
 * package. The static page below is only for the case where that file is
 * missing or unreadable — a damaged installation — where saying so honestly
 * beats a blank window or pointing at a backend that is not running.
 * The React interface cannot serve either purpose — it talks exclusively to a
 * backend that does not exist in these states.
 */
/** Escape text that is interpolated into generated HTML.
 *
 * `decision.message` carries a filesystem path and a raw `err.message`. A path
 * containing "<" silently swallowed the rest of the message, leaving a blank
 * screen where the one explanation was supposed to be. */
function esc(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * Fill in the waiting page and tick its counter.
 *
 * The page is inert (CSP default-src 'none') so the main process writes its
 * text — it must speak the same four languages as the app — and updates the
 * elapsed seconds. An unchanging spinner for three minutes is indistinguishable
 * from a hang; the app's own backend banner counts seconds for exactly this
 * reason, and in a packaged install that banner is unreachable because the
 * React app is not loaded until the backend answers.
 *
 * The words come from the current PHASE (waiting_page.js), not from fixed
 * strings: the update steps set the phase, and a clock that painted the same
 * three strings every second erased their text almost as soon as it appeared.
 */
function startWaitingPageClock(window) {
  return waitingPage.startClock(window);
}

/**
 * Bring `runtime/` up to the version THIS shell carries, before uvicorn imports
 * it.
 *
 * Installing a newer .exe over an existing installation replaced the shell and
 * left the program files alone, because the only thing that ever unpacked them
 * was the first-install wizard. The result was measured on 2026-09-27: a 0.4.6
 * shell running a 0.4.4 runtime, About reporting 0.4.4, and not one of the fixes
 * in between present. See `bundled_update.js` for why the parking route is taken
 * rather than a second unpacker.
 *
 * Returns `{ok}` — and `ok: false` means DO NOT start the backend: the tree is
 * part one release and part the other, and uvicorn importing that is the one
 * outcome worse than not starting.
 */
async function applyBundledUpdate(python) {
  const bundledUpdate = require('./bundled_update');
  const lang = shellLocale();
  let home;
  try {
    home = orientaHome();
  } catch (err) {
    logShellLine(`Runtime update: skipped, no home directory (${err.message})`);
    return { ok: true };
  }

  const bundled = bundledUpdate.bundledPackage(process.resourcesPath);
  const installed = bundledUpdate.installedTag(home);
  const parked = bundledUpdate.alreadyParked(home);
  // Both digests are read HERE so the decision stays a pure function of data:
  // the bundle states its own in the `.sha256` beside it (no hashing), and the
  // runtime states which build it came from since the applier began recording
  // it. Needed because 0.4.6 was rebuilt under one tag — see updateDecision.
  let bundledDigest = null;
  if (bundled) {
    try {
      bundledDigest = bundledUpdate.digestFrom(bundled.sum);
    } catch (err) {
      logShellLine(`Runtime update: the bundled package states no usable digest (${err.message})`);
    }
  }
  const decision = bundledUpdate.updateDecision({
    bundled, installed, parked, bundledDigest,
    installedDigest: bundledUpdate.installedDigest(home),
  });
  logShellLine(`Runtime update: ${decision.action} — ${decision.reason}`);
  if (decision.action === 'none') return { ok: true };

  if (decision.action === 'park') {
    try {
      const record = bundledUpdate.parkBundled({ home, bundled });
      logShellLine(`Runtime update: parked ${record.file} (${record.tag}) for the updater`);
    } catch (err) {
      // Nothing has been touched: the old runtime is whole and will start. Say
      // so loudly rather than let the user believe they are on the new version.
      logShellLine(`Runtime update: could not park the package (${err.message})`);
      await showUpdateProblem(t(lang, 'updateSkippedBody'), lang);
      return { ok: true };
    }
  }

  waitingPage.setPhase(mainWindow, 'updating');
  const applier = path.join(__dirname, 'apply_update.py');
  const verdictFile = path.join(home, 'setup-tmp', 'apply-result.json');
  try {
    fs.mkdirSync(path.dirname(verdictFile), { recursive: true });
  } catch { /* the applier's own failure will say so */ }

  const exitCode = await new Promise((resolve) => {
    let child;
    try {
      child = spawn(python, [applier, '--home', home, '--result-json', verdictFile],
        { windowsHide: true });
    } catch (err) {
      logShellLine(`Runtime update: could not run the updater (${err.message})`);
      resolve(-1);
      return;
    }
    const note = (buf) => String(buf).split(/\r?\n/).forEach((line) => {
      if (line.trim()) logShellLine(`  updater: ${line.trim()}`);
    });
    if (child.stdout) child.stdout.on('data', note);
    if (child.stderr) child.stderr.on('data', note);
    child.on('error', (err) => {
      logShellLine(`Runtime update: the updater did not start (${err.message})`);
      resolve(-1);
    });
    child.on('close', (code) => resolve(code === null ? -1 : code));
  });

  const verdict = bundledUpdate.readVerdict({ file: verdictFile, exitCode });
  if (verdict.applied) {
    logShellLine(`Runtime update: applied ${verdict.tag || '(no tag)'}`);
    await clearRendererCache();
    return { ok: true };
  }
  if (verdict.dirty) {
    logShellLine(`Runtime update: INCOMPLETE — ${verdict.error || 'no reason given'}; `
      + 'not starting the backend');
    await showUpdateProblem(t(lang, 'updateFailedBody'), lang, t(lang, 'updateFailedTitle'));
    return { ok: false };
  }
  if (verdict.error) {
    logShellLine(`Runtime update: not applied — ${verdict.error}`);
    await showUpdateProblem(t(lang, 'updateSkippedBody'), lang);
    return { ok: true };
  }
  logShellLine('Runtime update: nothing to apply');
  return { ok: true };
}

/** The package sync's runner while one exists, so `before-quit` can end its
 *  children. Not the wizard's `runSetup.children`: that set is consulted only
 *  while a SETUP is running. */
let packageSyncRunner = null;

/**
 * Bring the Python packages to the lock this release shipped, before uvicorn
 * imports them. See `setup/package_sync.js` for what it does and why.
 *
 * Returns the sync's own answer; the shell acts on two flags only:
 *   cancelled   the app is quitting -- start nothing
 *   repair      the environment fails verification -- show the repair wizard,
 *               start no backend
 * Everything else -- skipped, failed, synced, nothing to do -- is "go on".
 */
async function runPackageSync(decision, plan) {
  try {
    return await runPackageSyncInner(decision, plan);
  } catch (err) {
    // Even a sync that cannot be LOADED (a damaged installation) must not stop
    // the start: the previous packages are supported too.
    logShellLine(`Package sync: skipped, it could not run (${err.message})`);
    return { ok: true };
  }
}

async function runPackageSyncInner(decision, plan) {
  const packageSync = require('./setup/package_sync');
  const installer = require('./setup/installer');
  const lang = shellLocale();
  let home;
  try {
    home = orientaHome();
  } catch (err) {
    logShellLine(`Package sync: skipped, no home directory (${err.message})`);
    return { ok: true };
  }
  const shellLog = path.join(home, 'logs', 'orienta-shell.log');
  const runner = packageSync.createRunner({ platformName: process.platform });
  packageSyncRunner = runner;
  try {
    return await packageSync.syncPackages(
      {
        home,
        python: plan.python,
        decisionMode: decision.mode,
        projectRootEnv: process.env.ORIENTA_PROJECT_ROOT,
        platformName: process.platform,
        env: process.env,
      },
      {
        run: runner.run,
        isCancelled: () => runner.cancelled,
        log: logShellLine,
        freeBytes: (dir) => installer.measureFree(dir),
        onPhase: (phase) => waitingPage.setPhase(mainWindow, phase),
        runtimeTag: () => require('./bundled_update').installedTag(home),
        // One dialog per (lock digest, reason); the module decides when.
        notify: () => showUpdateProblem(
          t(lang, 'syncSkippedBody', { path: shellLog }), lang, t(lang, 'updateFailedTitle')),
      },
    );
  } finally {
    packageSyncRunner = null;
  }
}

/**
 * The repair wizard, from a window that was created to RUN.
 *
 * `createWindow` hands the setup channels to the window it makes for a setup or
 * repair (a preload argument and `setupContentsId`), and to no other. Loading
 * the wizard into the window of a normal start would render it and answer none
 * of its buttons, so the window is replaced instead. The old one's `closed`
 * handler is removed first: it clears `mainWindow`, which by then names the
 * new window.
 */
function showRepairWizard(message) {
  const old = mainWindow;
  createWindow({ mode: 'repair', message });
  if (old && !old.isDestroyed()) {
    old.removeAllListeners('closed');
    old.destroy();
  }
}

/**
 * Throw away the renderer's HTTP cache, because the program files just changed.
 *
 * The backend now sends `no-store` for the entrypoint, and that is the real
 * fix -- but it only governs responses fetched AFTER it shipped. A copy that a
 * previous version wrote into this cache is already there, with an ETag, a
 * Last-Modified and no `Cache-Control` at all, and Chromium's freshness
 * heuristic will serve it without asking. So the very upgrade this function
 * exists for is the one case the header cannot reach.
 *
 * Measured on 2026-09-27: after the runtime went from v0.4.4 to v0.4.6, About
 * read v0.4.6 from the new backend over `/api` while the window rendered the
 * 0.4.4 `index.html` and its hashed chunks from this cache -- a new version
 * number beside a missing feature. `Ctrl+Shift+R` fixed it, which is not
 * something a user should have to know.
 *
 * Only after an update, never on an ordinary start: a cold start already costs
 * 30-40 s, and re-fetching several megabytes of chunks every time would buy
 * nothing.
 */
async function clearRendererCache() {
  try {
    const { session } = require('electron');
    await session.defaultSession.clearCache();
    logShellLine('Runtime update: cleared the renderer cache, so the new frontend is fetched');
  } catch (err) {
    // Not fatal, and not silent: the header fix covers everything from here
    // on, so the worst case is one stale first paint that a reload clears.
    logShellLine(`Runtime update: could not clear the renderer cache (${err.message})`);
  }
}

/** One dialog, so a failure to update is never only a line in a log file. */
async function showUpdateProblem(message, lang, title) {
  try {
    await dialog.showMessageBox(mainWindow && !mainWindow.isDestroyed() ? mainWindow : null, {
      type: 'warning',
      title: title || t(lang, 'updateFailedTitle'),
      message: title || t(lang, 'updateFailedTitle'),
      detail: message,
      buttons: ['OK'],
      noLink: true,
    });
  } catch (err) {
    logShellLine(`Runtime update: could not show the message (${err.message})`);
  }
}

function loadSetupPlaceholder(window, decision) {
  const wizard = path.join(__dirname, 'setup', 'index.html');
  try {
    if (fs.existsSync(wizard)) {
      setupContext = { reason: decision.mode, message: decision.message || '' };
      window.loadFile(wizard);
      return;
    }
  } catch (err) {
    logShellLine(`Could not load the setup page: ${err.message}`);
  }

  const lang = shellLocale();
  const heading = decision.mode === 'repair'
    ? t(lang, 'needsRepairTitle')
    : t(lang, 'notSetUpTitle');
  const detail = decision.message || t(lang, 'setupMissing');
  // The dead end needs a way out, and when the wizard's own files are missing
  // the only honest one is the releases page. Rendered as the URL ITSELF rather
  // than as "click here": a screenshot of this page then carries the address,
  // which is how a tester will actually report it.
  const releases = releasesPageUrl();
  let logLine = '';
  try {
    logLine = t(lang, 'logLocation', { path: path.join(orientaHome(), 'logs') });
  } catch { /* an unusable ORIENTA_HOME must not blank the page */ }

  // A dead end with a next step is survivable; a dead end without one is not.
  // With the wizard's own files missing, the only honest next step is the
  // releases page, and the person needs to reach it from this screen.
  const body = `<!DOCTYPE html><html lang="${esc(shellLanguage(lang))}"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline';">
<title>Orienta</title><style>
:root{color-scheme:dark}html,body{height:100%;margin:0;background:#282a36;color:#f8f8f2;
font-family:"Segoe UI",system-ui,sans-serif}
main{height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;
gap:1rem;text-align:center;padding:2rem}
h1{font-size:1.4rem;margin:0}p{margin:0;max-width:38rem;line-height:1.55;color:#6272a4}
.path{font-family:ui-monospace,Consolas,monospace;font-size:.8rem;color:#44475a;
word-break:break-all}
a.url{color:#8be9fd;font-family:ui-monospace,Consolas,monospace;word-break:break-all}
</style></head><body><main role="status">
<h1>${esc(heading)}</h1><p>${esc(detail)}</p>
<p>${esc(t(lang, 'downloadHere'))}<br>
<a class="url" href="${esc(releases)}">${esc(releases)}</a></p>
<p class="path">${esc(logLine)}</p>
</main></body></html>`;
  window.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(body)}`);
}

/**
 * @param {{mode: 'dev'|'run'|'setup'|'repair', message?: string}} decision
 *
 * Switches on `decision.mode` AND ON NOTHING ELSE. A second test of whether the
 * machine is installed, made here, is what this task exists to remove.
 */
function createWindow(decision = { mode: 'dev' }) {
  // Open on the display the previous window was on, when the relaunch told us
  // which. Absent, or on a monitor since unplugged, this stays null and the
  // OS places the window exactly as it did before.
  const size = { width: 1400, height: 900 };
  let placement = null;
  try {
    const arg = process.argv.find((a) => String(a).startsWith('--orienta-window-at='));
    if (arg) {
      const [x, y] = arg.split('=')[1].split(',').map(Number);
      const { screen } = require('electron');
      placement = platformInfo.windowPositionFor(screen.getAllDisplays(), { x, y }, size);
    }
  } catch { /* never block the window over where it sits */ }

  mainWindow = new BrowserWindow({
    ...size,
    ...(placement || {}),
    minWidth: 1000,
    minHeight: 700,
    title: 'Orienta - EBSD Pattern Analysis',
    backgroundColor: '#282a36',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
      // The wizard's channels are exposed to THIS window only. One preload
      // serves the application window and every pole-figure window as well, so
      // without this the React app would carry a button able to start an 8 GB
      // install on top of a running one.
      additionalArguments: (decision.mode === 'setup' || decision.mode === 'repair')
        ? ['--orienta-setup-window'] : [],
    },
    autoHideMenuBar: true,
    icon: path.join(__dirname, '..', 'resources', 'icon.png'),
  });

  if (decision.mode === 'setup' || decision.mode === 'repair') {
    setupContentsId = mainWindow.webContents.id;
    // Ctrl+R mid-install restarts the RENDERER while the install carries on in
    // the main process: the wizard comes back to its first screen, Install
    // answers "a setup is already running", and a twenty-minute pip continues
    // with no progress shown anywhere. The natural next move -- closing the
    // window -- then kills it half-written.
    mainWindow.webContents.on('before-input-event', (event, input) => {
      const reload = input.key === 'F5'
        || ((input.control || input.meta) && input.key.toLowerCase() === 'r');
      if (reload && setupRunning) event.preventDefault();
    });
    // The React app cannot run here: it talks exclusively to a backend that
    // does not exist yet. The wizard is plain HTML for the same reason, and
    // `loadSetupPlaceholder` falls back to a static page if it is missing.
    loadSetupPlaceholder(mainWindow, decision);
  } else if (isDev) {
    mainWindow.loadURL(`http://localhost:${FRONTEND_DEV_PORT}`);
  } else {
    // In production the backend SERVES the built frontend, so there is nothing
    // to show until it answers — and a cold start takes 30-40 s. Loading the
    // URL now would leave Chromium's ERR_CONNECTION_REFUSED page on screen with
    // nothing to reload it. whenReady() swaps in the real URL once /api/health
    // responds.
    mainWindow.loadFile(path.join(__dirname, 'setup', 'waiting.html'));
    stopWaitingClock = startWaitingPageClock(mainWindow);
  }

  // DevTools: press F12 to toggle manually
  // An http(s) link anywhere in the shell opens in the SYSTEM browser and the
  // window stays where it is. Without this, clicking the releases link on the
  // dead-end page navigates the app itself to GitHub -- and that page has no
  // back button, so the one way out becomes a second dead end.
  const openOutside = (url) => {
    if (/^https?:\/\//i.test(url)) shell.openExternal(url).catch(() => {});
  };
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    openOutside(url);
    return { action: 'deny' };
  });
  mainWindow.webContents.on('will-navigate', (event, url) => {
    // Not the app's own pages: the dev server, the backend and the local files
    // all navigate legitimately.
    if (/^https?:\/\/(localhost|127\.0\.0\.1)(:|\/|$)/i.test(url)) return;
    if (!/^https?:\/\//i.test(url)) return;
    event.preventDefault();
    openOutside(url);
  });

  mainWindow.webContents.on('before-input-event', (event, input) => {
    if (input.key === 'F12') mainWindow.webContents.toggleDevTools();
  });

  // Renderer-crash recovery: a frozen / OOM'd renderer used to take the
  // backend with it (window-all-closed fires after the renderer dies →
  // killBackend() → SIGTERM to Python → 12 h batch dead).  Reload the
  // window instead and leave the backend alone.
  mainWindow.webContents.on('render-process-gone', (event, details) => {
    console.error(`[Renderer] gone: reason=${details.reason} exitCode=${details.exitCode}`);
    logBackendLine(`[Renderer] gone: reason=${details.reason} exitCode=${details.exitCode}`);
    if (userInitiatedQuit) return;
    if (rendererReloadCount >= MAX_RENDERER_RELOADS) {
      console.error(`[Renderer] crashed ${rendererReloadCount} times — giving up.`);
      // Keeping the backend alive here is OPT-IN only. It was the default, to
      // protect a 12 h batch from a renderer crash, but the cost was worse than
      // the benefit in practice: the surviving process kept port 8000, the next
      // session silently attached to it, and a backend that outlived a USB
      // replug then served dead file handles ("errno 22") on every load. Quit
      // means quit; set KIKUCHIPY_KEEP_BACKEND=1 before launching when you
      // deliberately want a long job to survive the UI.
      keepBackendOnQuit = process.env.KIKUCHIPY_KEEP_BACKEND === '1';
      dialog.showErrorBox(
        'Renderer keeps crashing',
        `The UI process crashed ${rendererReloadCount} times (last: ${details.reason}). ` +
        (keepBackendOnQuit
          ? `The backend is still running — open http://127.0.0.1:${BACKEND_PORT} in your browser ` +
            `to keep using it. Restart the app when you're ready.`
          : `The backend has been shut down with it. Restart the app to continue. ` +
            `(Launch with KIKUCHIPY_KEEP_BACKEND=1 if you need a running job to survive this.)`)
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
    logBackendLine('[Renderer] unresponsive — forcing reload');
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
  // `multiple: true` (the CIF import) returns an ARRAY of paths; every other
  // caller gets one path, as before. The option used to be ignored, so the CIF
  // import could only ever take one file per dialog despite asking for more.
  const multiple = options?.multiple === true;
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: multiple ? ['openFile', 'multiSelections'] : ['openFile'],
    filters: options?.filters || [
      { name: 'HDF5 Files', extensions: ['h5oina', 'h5', 'hdf5'] },
      { name: 'CIF Files', extensions: ['cif'] },
      { name: 'All Files', extensions: ['*'] },
    ],
  });
  if (result.canceled || !result.filePaths?.length) return null;
  return multiple ? result.filePaths : result.filePaths[0];
});

/**
 * Directories the user picked in this session.
 *
 * A batch export writes many files without a dialog per file, so the renderer
 * asks for a folder ONCE and then writes into it. This set is what makes that
 * safe: `fs:writeImageInFolder` below writes only where the user has already
 * pointed, so the renderer can never name a path of its own.
 */
const grantedFolders = new Set();

ipcMain.handle('dialog:openFolder', async (event, options) => {
  // Only an EXPORT picker (`remember: true`) starts in, and updates, the
  // remembered save folder. This handler also serves the pickers for a folder
  // of scans or CIFs to read; remembering those would open the next "Export
  // image…" in the raw-data folder.
  const remember = !!options?.remember;
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory', 'createDirectory'],
    defaultPath: remember ? (saveDialogs.readLastDir(orientaHome()) || undefined) : undefined,
  });
  if (result.canceled || !result.filePaths?.[0]) return null;
  grantedFolders.add(path.resolve(result.filePaths[0]));
  if (remember) saveDialogs.rememberDir(orientaHome(), result.filePaths[0], { isDirectory: true });
  return result.filePaths[0];
});

/**
 * Write one image of a batch into a folder the user picked.
 *
 * `dialog:saveImage` shows a save dialog per file, which is exactly what a
 * series of a dozen element maps must avoid. One call per file rather than one
 * call for the whole series: a 16x series of a large map must not have to sit
 * in memory all at once.
 *
 * Refuses a folder that was not granted, and a name that is anything but a
 * name — a separator or a '..' in it would put the file somewhere else.
 */
ipcMain.handle('fs:writeImageInFolder', async (event, options) => {
  const dir = path.resolve(String(options?.dir || ''));
  if (!grantedFolders.has(dir)) {
    throw new Error('Refusing to write into a folder that was not chosen in this session');
  }
  const name = String(options?.name || '');
  if (!name || name !== path.basename(name) || name === '.' || name === '..') {
    throw new Error(`Refusing to write under the name ${JSON.stringify(name)}`);
  }
  const target = path.join(dir, name);
  await fs.promises.writeFile(target, Buffer.from(options.base64, 'base64'));
  return target;
});

/**
 * A starting directory for a save dialog whose caller named no path.
 *
 * `app.getPath('downloads')` throws when the shell folder is unset or points at
 * an unavailable network location — a redirected profile is enough. Returning
 * undefined then leaves the dialog to its own default, which is the behaviour
 * we had before, instead of failing the whole IPC call.
 */
function defaultSaveDirectory() {
  try {
    return app.getPath('downloads');
  } catch (err) {
    console.warn('[dialog] no downloads folder available:', err.message);
    return undefined;
  }
}

ipcMain.handle('dialog:saveFile', async (event, options) => {
  const result = await dialog.showSaveDialog(mainWindow, {
    // Forwarded when the caller supplies one (phase-map export, crop export);
    // without this the dialog silently opened with an empty name even then.
    // `dialog:saveImage` below has always forwarded it.
    //
    // Two callers pass only `filters` — AnalysisPage's "Browse" buttons — and
    // for those we name a starting DIRECTORY rather than inheriting whatever
    // Electron's default happens to be in this version. It is a directory, so
    // the file name is still empty; the gain is only that the dialog opens
    // somewhere predictable.
    //
    // getPath('downloads') THROWS when the shell folder is unset or redirected
    // to an unavailable network location. Unwrapped, that rejects the IPC call
    // and both Browse buttons do nothing at all — no dialog, no message — since
    // neither has a catch.
    //
    // The folder of the last file saved comes first (save_dialogs.js): a
    // report's figures go into one folder, and every dialog opening in
    // Downloads made the user walk there each time.
    defaultPath: saveDialogs.resolveDefaultPath({
      requested: options?.defaultPath,
      lastDir: saveDialogs.readLastDir(orientaHome()),
      fallbackDir: defaultSaveDirectory(),
    }),
    filters: options?.filters || [
      { name: 'PNG Image', extensions: ['png'] },
      { name: 'Excel', extensions: ['xlsx'] },
      { name: 'All Files', extensions: ['*'] },
    ],
  });
  if (result.canceled || !result.filePath) return null;
  saveDialogs.rememberDir(orientaHome(), result.filePath);
  return result.filePath;
});

/**
 * Save an image produced in the renderer.
 *
 * `dialog:saveFile` above only hands back a path, because its callers pass that
 * path to the Python backend and let it do the writing. An exported PNG/JPEG
 * exists only as bytes in the renderer, so this handler shows the dialog AND
 * writes the file.
 *
 * Returns the written path, or null when the user cancels. Write errors are
 * thrown so the renderer can show them rather than reporting a phantom success.
 */
ipcMain.handle('dialog:saveImage', async (event, options) => {
  const result = await dialog.showSaveDialog(mainWindow, {
    // A bare name is placed in the folder of the last save (see
    // save_dialogs.js), then in Downloads; a caller's absolute path wins.
    defaultPath: saveDialogs.resolveDefaultPath({
      requested: options?.defaultPath || 'image.png',
      lastDir: saveDialogs.readLastDir(orientaHome()),
      fallbackDir: defaultSaveDirectory(),
    }),
    filters: options?.filters || [
      { name: 'PNG Image', extensions: ['png'] },
      { name: 'JPEG Image', extensions: ['jpg', 'jpeg'] },
      { name: 'WebP Image', extensions: ['webp'] },
      { name: 'All Files', extensions: ['*'] },
    ],
  });
  if (result.canceled || !result.filePath) return null;
  await fs.promises.writeFile(result.filePath, Buffer.from(options.base64, 'base64'));
  saveDialogs.rememberDir(orientaHome(), result.filePath);
  return result.filePath;
});

/**
 * Save text the renderer produced — a CSV of an add-on's table.
 *
 * `dialog:saveFile` only returns a PATH, for the backend to write to, and an
 * add-on's numbers never reach the backend: they arrive as one HTTP response
 * and live in the page. So this mirrors `dialog:saveImage`, which exists for
 * the same reason on the image side, rather than stretching that channel to
 * carry text as a fake image.
 *
 * Written VERBATIM, as UTF-8. This handler used to prepend a BOM itself, on
 * the sound ground that Excel on Windows reads a BOM-less UTF-8 CSV as the
 * system code page and turns µm into mojibake. The BOM is still there — it
 * just belongs to the caller now, because this channel does not know what
 * format it is carrying: the same unconditional BOM would corrupt a JSON file
 * saved through it, and it split one decision across two files. In a browser
 * there is no Electron at all, so the download path had to prepend its own —
 * and did not, so the same table came out differently depending on where it
 * was saved from. One owner, in the renderer, where the format is known.
 */
ipcMain.handle('dialog:saveText', async (event, options) => {
  const result = await dialog.showSaveDialog(mainWindow, {
    defaultPath: options?.defaultPath || 'data.csv',
    filters: options?.filters || [
      { name: 'CSV', extensions: ['csv'] },
      { name: 'Text', extensions: ['txt'] },
      { name: 'All Files', extensions: ['*'] },
    ],
  });
  if (result.canceled || !result.filePath) return null;
  await fs.promises.writeFile(result.filePath,
                              String(options?.text ?? ''), 'utf8');
  return result.filePath;
});

// Settings > About > "Show log files". The folder is worked out HERE, from the
// project root the backend log is written under; the renderer sends nothing, so
// this channel cannot be used to open any other folder.
ipcMain.handle('app:openLogFolder', () =>
  openLogFolder({ shell, projectRoot: lastProjectRoot || resolveProjectRoot() }));

// Picture of the window for a problem report. Taken before the report dialog
// opens, so the report shows the screen the user is complaining about rather
// than the dialog covering it.
ipcMain.handle('app:captureScreen', async () => {
  try {
    if (!mainWindow) return null;
    const image = await mainWindow.webContents.capturePage();
    return image.toPNG().toString('base64');
  } catch (err) {
    logBackendLine(`[report] screenshot failed: ${err.message}`);
    return null;
  }
});

/**
 * Remove Orienta's data folder, on the platforms that have no uninstaller.
 *
 * Windows asks these two questions while uninstalling; macOS and Linux have
 * nowhere to ask them, so the application does. The rules live in
 * electron/remove_data.js and are the NSIS uninstaller's, deliberately.
 *
 * `plan` and `remove` are separate calls so the dialog can state what will
 * happen -- and so a refusal ("this folder is not recognisably Orienta's")
 * arrives before any question about deleting gigabytes.
 */
/**
 * Wait until nothing answers on the backend port, or give up.
 *
 * A fixed sleep was the first version and it was a guess: /api/shutdown is
 * fire-and-forget, the signal is asynchronous, uvicorn drains its requests,
 * and a backend this shell only attached to is not killed at all.
 */
function waitForBackendToStop(timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve) => {
    const probe = () => {
      const req = http.request(
        { hostname: '127.0.0.1', port: BACKEND_PORT, path: '/health', method: 'GET', timeout: 800 },
        () => { if (Date.now() > deadline) resolve(false); else setTimeout(probe, 500); },
      );
      req.on('error', () => resolve(true));      // nothing listening: it is gone
      req.on('timeout', () => { req.destroy(); if (Date.now() > deadline) resolve(false); else setTimeout(probe, 500); });
      req.end();
    };
    probe();
  });
}

/**
 * Windows has an uninstaller for this, and it asks the same questions with
 * the application closed. A second way to delete the same gigabytes is a
 * second way to get it wrong -- so the channel is not merely hidden in the
 * interface, it is refused here. The preload decides what is EXPOSED; this
 * decides what is ANSWERED, and the second one is the boundary.
 */
function dataRemovalAvailable() {
  return process.platform === 'darwin' || process.platform === 'linux';
}

ipcMain.handle('data:plan', () => {
  if (!dataRemovalAvailable()) {
    logShellLine('[data] plan refused: this platform has an uninstaller');
    return { ok: false, reason: 'notHere' };
  }
  try {
    return removeDataModule.planRemoval(orientaHome());
  } catch (err) {
    logShellLine(`[data] plan failed: ${err.message}`);
    return { ok: false, reason: 'error', error: String(err.message || err) };
  }
});

ipcMain.handle('data:remove', async (event, options = {}) => {
  if (!dataRemovalAvailable()) {
    logShellLine('[data] removal refused: this platform has an uninstaller');
    return { ok: false, reason: 'notHere', removed: [], kept: [] };
  }
  if (!mainWindow || event.sender.id !== mainWindow.webContents.id) {
    logShellLine('[data] removal refused: not the main window');
    return { ok: false, reason: 'notHere', removed: [], kept: [] };
  }
  const home = orientaHome();
  const removeLibrary = Boolean(options && options.removeLibrary);
  logShellLine(`[data] removing ${home} (crystal library: ${removeLibrary ? 'yes' : 'no'})`);
  // The backend runs FROM this folder. Stopping it first is not tidiness: on
  // Windows its open files cannot be deleted at all, and everywhere else it
  // would keep writing logs into a folder that is being removed.
  // The backend runs FROM this folder, and it may be one this shell attached
  // to rather than started (killBackend returns at once for those). Deleting
  // runtime/ under a live interpreter leaves it half-working: already
  // imported modules keep going, the next import fails in the user's face.
  // So: ask it to stop, then WAIT until the port stops answering.
  try { killBackend(); } catch { /* may already be gone */ }
  const stopped = await waitForBackendToStop(15000);
  if (!stopped) {
    logShellLine('[data] the backend is still answering; nothing was removed');
    return { ok: false, reason: 'backendRunning', removed: [], kept: [] };
  }
  const result = removeDataModule.removeData(home, { removeLibrary });
  logShellLine(`[data] removed ${result.removed ? result.removed.length : 0} entries, `
    + `kept ${result.kept ? result.kept.length : 0}`);
  return result;
});

// Restart after a self-update. The backend must go down with us — the new
// source is only loaded by a fresh Python process, and killBackend() is what
// guarantees port 8000 is free when the relaunched app starts its own.
ipcMain.handle('app:relaunch', () => {
  logBackendLine('[update] relaunching after update');
  userInitiatedQuit = true;
  const relaunch = platformInfo.relaunchOptions();
  // Carry the screen we are on across the restart. This is the handler the
  // setup wizard's "Start Orienta" button uses, and the new process creates
  // its window with no coordinates -- so on the M5 tester's three-monitor
  // desk the wizard ran on one screen and Orienta appeared on another. An
  // argument rather than a file: nothing to write, nothing to go stale, and
  // it cannot outlive the restart it belongs to.
  let where = null;
  try {
    if (mainWindow && !mainWindow.isDestroyed()) {
      const b = mainWindow.getBounds();
      where = `--orienta-window-at=${Math.round(b.x + b.width / 2)},${Math.round(b.y + b.height / 2)}`;
    }
  } catch { /* a window that has already gone simply does not vote */ }

  const args = (relaunch && relaunch.args ? relaunch.args.slice() : process.argv.slice(1))
    .filter((a) => !String(a).startsWith('--orienta-window-at='));
  if (where) args.push(where);
  app.relaunch({ ...(relaunch || {}), args });
  app.quit();
  return true;
});

// ==========================================================================
// the setup wizard
// ==========================================================================

/**
 * Which release to install: the NEWEST one, unless the user picks another.
 *
 * It used to be `v${app.getVersion()}` — the number baked into the .exe at
 * build time — on the argument that the shell and the runtime ship together.
 * That argument was wrong in the way that matters: it pins a 100 MB installer
 * to exactly one release, so the day that release is superseded or removed,
 * every copy anyone still has is dead, with a 404 naming a tag they never
 * typed. A bootstrap installer's whole purpose is to stay valid while the
 * thing it installs moves on.
 *
 * The pairing concern is real but belongs elsewhere: the runtime is served to
 * the shell over HTTP on localhost, not linked into it, and a genuine
 * incompatibility needs an explicit minimum-version declaration — not an
 * accidental one enforced by breaking every download ever made.
 */
let requestedTag = null;   // null = newest
/** A folder the user pointed at, searched before anywhere else. */
let packageDir = null;

/**
 * Downloads and Desktop as WINDOWS resolves them.
 *
 * `path.join(process.env.USERPROFILE, 'Downloads')` is a guess, and it is
 * wrong on every machine where OneDrive has taken the known folders over --
 * the default on a managed university laptop. Electron reads the real shell
 * folder. Each call is guarded because `getPath` throws on some redirected
 * profiles, and a folder we cannot name is simply one we do not search.
 */
function knownDownloadFolders() {
  const dirs = [];
  for (const name of ['downloads', 'desktop', 'documents']) {
    try {
      const dir = app.getPath(name);
      if (dir) dirs.push(dir);
    } catch { /* not resolvable on this profile */ }
  }
  return dirs;
}

/** At most one setup at a time, per process AND per machine.
 *
 * Two concurrent runs share one setup-tmp and one python/ directory: the
 * second one's first act is to delete the first one's half-finished download.
 */
let setupRunning = false;

/** The WebContents id of the setup window, or null.
 *
 * The preload flag decides what is EXPOSED; this decides what is ANSWERED,
 * and the second one is the boundary -- a renderer that reaches the channel
 * some other way still gets nothing.
 */
let setupContentsId = null;

/** Register a setup channel that only the wizard window may call. */
function handleSetup(channel, fn) {
  ipcMain.handle(channel, (event, ...rest) => {
    if (setupContentsId === null || event.sender.id !== setupContentsId) {
      logShellLine(`Refused ${channel} from a window that is not the wizard`);
      throw new Error('not available here');
    }
    return fn(event, ...rest);
  });
}

handleSetup('setup:context', () => ({
  reason: setupContext.reason,
  message: setupContext.message,
  locale: shellLanguage(shellLocale()),
  home: orientaHome(),
  releasesUrl: releasesPageUrl(),
  version: app.getVersion(),
}));

handleSetup('setup:locales', () => {
  // Read per call rather than cached at import: this file is the only thing
  // standing between a failed install and an untranslated error, and a stale
  // cache after a repair would be silent.
  const file = path.join(__dirname, 'setup', 'locales.json');
  return JSON.parse(fs.readFileSync(file, 'utf8'));
});

handleSetup('setup:probe', async () => {
  // cpuBytesNeeded(), not the bare CPU_BYTES_NEEDED: the constant is the
  // WINDOWS floor. Shipping it to the renderer made the macOS wizard label
  // its only option "about 2.5 GB" while the sentence right below it, which
  // does ask the function, said 7.0 GB -- two numbers for the same install,
  // on the same screen. Same shape as chooseRecommendation answering for the
  // host instead of the platform it was handed.
  const { probe, GPU_BYTES_NEEDED, cpuBytesNeeded } = require('./setup/installer');
  const CPU_BYTES_NEEDED = cpuBytesNeeded(process.platform);
  const home = orientaHome();
  if (homeStorageError) {
    // We already know the home is unusable -- it is why Chromium is running on
    // its default paths instead. Saying so here beats letting the probe report
    // it as a disk measurement, and it is the only place the user ever hears
    // about it.
    return {
      gpu: { present: false }, recommendation: 'cpu', blocked: true,
      freeBytes: 0, freeKnown: false,
      reason: `Orienta cannot write to ${home} (${homeStorageError}). `
        + 'This is usually a drive that is not connected, or a user folder kept '
        + 'on a network share. Set ORIENTA_HOME to a folder on this computer '
        + 'and start Orienta again.',
      gpuBytesNeeded: GPU_BYTES_NEEDED, cpuBytesNeeded: CPU_BYTES_NEEDED,
    };
  }
  try {
    const measured = await probe(home);
    return { ...measured, gpuBytesNeeded: GPU_BYTES_NEEDED, cpuBytesNeeded: CPU_BYTES_NEEDED };
  } catch (err) {
    // The wizard must still render. A probe that throws means we know nothing
    // about the machine, which is exactly when the smaller install is right.
    logShellLine(`Probe failed: ${err.message}`);
    return {
      gpu: { present: false }, recommendation: 'cpu',
      freeBytes: 0, freeKnown: false,
      gpuBytesNeeded: GPU_BYTES_NEEDED, cpuBytesNeeded: CPU_BYTES_NEEDED,
    };
  }
});

handleSetup('setup:run', async (event, options) => {
  if (setupRunning) {
    return { ok: false, code: 'unexpected', error: 'a setup is already running' };
  }
  const { runSetup } = require('./setup/installer');
  const mode = options && options.mode === 'gpu' ? 'gpu' : 'cpu';
  const resume = Boolean(options && options.resume);
  const home = orientaHome();
  setupRunning = true;

  // `currentStep` at the moment of failure, kept here because the renderer
  // needs to mark WHICH step failed and the result object carries only the
  // reason. Without it the step list shows a failure with no position.
  let lastStep = 'probe';
  const sender = event.sender;

  try {
    const result = await runSetup({
      mode, home, releaseTag: requestedTag, resume, packageDir,
      knownFolders: knownDownloadFolders(),
    },
      (progress) => {
        if (progress && progress.step) lastStep = progress.step;
        // The window can be gone: a user who closes the wizard mid-install
        // leaves this callback firing into a destroyed WebContents, and an
        // unguarded send throws inside the installer's own progress path.
        if (!sender.isDestroyed()) sender.send('setup:progress', progress);
      });
    logShellLine(`Setup ${result.ok ? 'succeeded' : `failed (${result.code})`} in ${mode} mode`);
    return result.ok ? result : { ...result, step: lastStep };
  } catch (err) {
    logShellLine(`Setup threw: ${err.message}`);
    return { ok: false, code: 'unexpected', error: String(err.message || err), step: lastStep };
  } finally {
    setupRunning = false;
  }
});

/** The releases a user could choose from, for the version picker. */
handleSetup('setup:releases', async () => {
  const { listReleases } = require('./setup/installer');
  try {
    return { ok: true, releases: await listReleases(), selected: requestedTag };
  } catch (err) {
    // The picker is an extra, never a blocker: a failure here must not stop
    // someone installing the newest version.
    logShellLine(`Could not list releases: ${err.message}`);
    return { ok: false, error: String(err.message || err), releases: [] };
  }
});

/** Pin the install to one version, or back to "newest" with null. */
handleSetup('setup:selectRelease', (event, tag) => {
  requestedTag = (typeof tag === 'string' && tag.trim()) ? tag.trim() : null;
  logShellLine(`Release to install: ${requestedTag || 'newest'}`);
  return requestedTag;
});

/**
 * Let the user point at the package themselves.
 *
 * The automatic search covers the folders a downloaded file plausibly lands
 * in, but "plausibly" was doing a lot of work there: the first person to
 * follow the written instructions put it somewhere the search did not look,
 * and got a 404 about a release instead of a word about the file sitting right
 * next to them. A picker ends the category.
 */
handleSetup('setup:pickPackage', async () => {
  const answer = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [{ name: 'Orienta package', extensions: ['zip'] }],
  });
  if (answer.canceled || !answer.filePaths.length) return null;
  const chosen = answer.filePaths[0];
  // The DIRECTORY: the checksum file must be beside it, and the search already
  // knows how to pair them. A zip whose .sha256 is missing is reported
  // honestly rather than worked around.
  packageDir = path.dirname(chosen);
  logShellLine(`Package folder chosen by the user: ${packageDir}`);
  return {
    dir: packageDir,
    file: path.basename(chosen),
    hasChecksum: fs.existsSync(`${chosen}.sha256`),
  };
});

/**
 * Choose where the gigabytes go.
 *
 * Not the same question as where the .exe is installed, and the more important
 * one: this folder holds Python (2.5 GB, or 8 for the graphics-card build) and
 * a crystal library that grows for years. Until this existed it could only be
 * moved with an environment variable — which for the people this is built for
 * means it could not be moved.
 */
handleSetup('setup:pickHome', async () => {
  const answer = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory', 'createDirectory'],
    defaultPath: orientaHome(),
  });
  if (answer.canceled || !answer.filePaths.length) return null;
  // Orienta's own subfolder, never the bare folder the user picked: someone
  // who chooses D:\ must not have python/, runtime/ and logs/ scattered
  // across the root of their drive — and an uninstall has to know what is
  // ours to remove.
  const chosen = path.join(answer.filePaths[0], 'Orienta');
  try {
    // The application's install directory is removed wholesale by its own
    // uninstaller, so data inside it would go with it.
    setOrientaHome(chosen, {
      forbidden: app.isPackaged ? platformInfo.programRoots(process.execPath) : [],
    });
    logShellLine(`Data folder set to ${chosen}`);
    // Chromium's own storage was pointed at the OLD home at startup and
    // cannot be moved under a running process, so the choice takes effect on
    // the next launch. Saying so is better than pretending otherwise.
    return { dir: chosen, restartNeeded: Boolean(cacheDir) };
  } catch (err) {
    logShellLine(`Could not use ${chosen}: ${err.message}`);
    return { dir: chosen, error: String(err.message || err) };
  }
});

handleSetup('setup:cancel', () => {
  const { runSetup } = require('./setup/installer');
  runSetup.killChildren();
  logShellLine('Setup cancelled by the user');
  return true;
});

handleSetup('setup:openLog', async () => {
  const file = path.join(orientaHome(), 'logs', 'orienta-setup.log');
  // The FOLDER when the log does not exist yet: opening a missing file does
  // nothing at all, and a button that does nothing reads as a broken app on the
  // screen where trust is thinnest.
  const target = fs.existsSync(file) ? file : path.dirname(file);
  try {
    await shell.openPath(target);
    return true;
  } catch (err) {
    logShellLine(`Could not open the setup log: ${err.message}`);
    return false;
  }
});

/**
 * Where the wizard's language waits for the app: beside the data-folder
 * pointer, NOT in userData. userData is inside the data folder, and a user who
 * picks a different data folder in the wizard relaunches into a process whose
 * userData is somewhere else — the language was lost exactly then.
 * See electron/start_language.js.
 */
function startLanguageDir() {
  return path.dirname(homePointerFile());
}

/**
 * The language the SHELL should speak.
 *
 * The app's own choice first, the operating system second. Before this the
 * splash screen and the shell dialogs asked `app.getLocale()`, so a user
 * running Orienta in German on an English Mac met an English splash at every
 * start — reported by the M5 tester. The app records its language on every
 * change; a first run has nothing recorded and falls back to the OS, which is
 * the old behaviour.
 */
function shellLocale() {
  // uiLocale(), not app.getLocale(): on a Mac whose system language is German
  // app.getLocale() answers "en", because Orienta declares no bundle
  // localizations. That is the sibling fix on golive, and this chains onto
  // it — the app's own choice first, the system's real preference second.
  let fallback = 'en';
  try {
    fallback = uiLocale();
  } catch { /* before app is ready; 'en' is the shell's own default */ }
  try {
    return startLanguage.appLanguage(startLanguageDir()) || fallback;
  } catch {
    return fallback;
  }
}

/**
 * The app telling the shell what language it is in, so the NEXT start's
 * splash screen matches. Deliberately not the same channel as the wizard's:
 * that one is a hand-off that gets consumed, this one is a memory.
 */
ipcMain.handle('app:setLanguage', (event, lang) => {
  try {
    return startLanguage.recordApp(startLanguageDir(), lang);
  } catch (err) {
    logShellLine(`Could not record the app language: ${err.message}`);
    return false;
  }
});

handleSetup('setup:setLanguage', (event, lang, explicit) => {
  try {
    return startLanguage.record(startLanguageDir(), lang, explicit);
  } catch (err) {
    logShellLine(`Could not record the setup language: ${err.message}`);
    return false;
  }
});

// The blocked screen's only action. Without it that screen has no button at
// all and the window's sole exit is the title bar -- on the one failure the
// user can genuinely do nothing about from inside the wizard.
handleSetup('setup:quit', () => {
  userInitiatedQuit = true;
  app.quit();
  return true;
});

// A setup that is still running when the window closes must not carry on
// writing gigabytes into site-packages after the app is gone.
app.on('before-quit', () => {
  // The package sync's own children, not the wizard's: closing the window during
  // the download or the install must not leave a pip writing into site-packages
  // after the app is gone. A killed install leaves its marker, which makes the
  // next start verify and, if need be, repair.
  if (packageSyncRunner) {
    try {
      packageSyncRunner.killAll();
      logShellLine('Package sync children killed on quit');
    } catch { /* nothing to kill */ }
  }
  if (!setupRunning) return;
  try {
    require('./setup/installer').runSetup.killChildren();
    logShellLine('Setup children killed on quit');
  } catch { /* nothing to kill */ }
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

/**
 * Tell the user the backend never came up, and end the session.
 *
 * Absolute paths, and the SHELL log first: "logs/backend-console.log" is
 * relative to a directory the user has never heard of, and it lives inside
 * runtime/ — the very directory a reinstall replaces, so the evidence is
 * destroyed by the action we would otherwise be recommending.
 */
async function reportBackendFailure(err) {
  logShellLine(`Backend failed to start: ${err.message}`);
  if (stopWaitingClock) stopWaitingClock();
  // Quitting during startup can leave no window to parent a dialog to.
  if (!mainWindow || mainWindow.isDestroyed()) return;

  const lang = shellLocale();
  let shellLog = '';
  try { shellLog = path.join(orientaHome(), 'logs', 'orienta-shell.log'); } catch { /* ignore */ }
  const backendLog = path.join(lastProjectRoot || resolveProjectRoot(), 'logs', 'backend-console.log');

  const choice = dialog.showMessageBoxSync(mainWindow, {
    type: 'error',
    buttons: [t(lang, 'showLogFolder'), t(lang, 'tryAgain'), t(lang, 'close')],
    defaultId: 0,
    cancelId: 2,
    title: t(lang, 'backendFailedTitle'),
    message: t(lang, 'backendFailedMessage'),
    detail: t(lang, 'backendFailedDetail', { shellLog, backendLog }),
  });
  if (choice === 0) {
    // The file itself, not a URL: a list of GitHub release assets is a
    // developer's page and answers none of the likely causes.
    try { shell.showItemInFolder(shellLog || backendLog); } catch { /* ignore */ }
  } else if (choice === 1) {
    // Same AppImage caveat as the update relaunch: restarting the temporary
    // mount starts nothing, and this is the worst moment to vanish.
    const options = platformInfo.relaunchOptions();
    if (options) app.relaunch(options); else app.relaunch();
  }
  // Every branch ends the session. Leaving the window on "Orienta is starting"
  // after telling the user it failed is worse than closing.
  app.quit();
}

app.whenReady().then(async () => {
  // FIRST, before any decision can fail. openBackendLog() is opened only inside
  // startBackend(), so every diagnostic in the paths below would otherwise go
  // to a console that a packaged app does not have. This directory exists
  // independently of the runtime and survives every failure here.
  try {
    openStartupLog(path.join(orientaHome(), 'logs'));
  } catch (err) {
    console.error('Could not open the shell log:', err.message);
  }
  logShellLine('Starting Orienta...');

  // A second launch must not race the first. Without this, double-clicking the
  // icon while Orienta is already running starts a second instance whose
  // busy-port dialog offers — as its DEFAULT button — to kill the first one's
  // backend. The first window then silently reconnects to the replacement and
  // its loaded file, indexing result and running job are gone, with nothing
  // said.
  if (!app.requestSingleInstanceLock()) {
    logShellLine('Another instance owns the lock — bringing it forward instead.');
    app.quit();
    return;
  }
  app.on('second-instance', () => {
    if (!mainWindow || mainWindow.isDestroyed()) return;
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.focus();
  });

  try {
    const decision = isDev
      ? { mode: 'dev', root: checkoutRoot() }
      : startupDecision(shellLocale());
    logShellLine(`Startup decision: ${decision.mode}${decision.message ? ' — ' + decision.message : ''}`);

    // A window exists in every branch, before anything else can fail — and
    // before the busy-port dialog, which was previously an ownerless modal
    // shown with no application behind it.
    createWindow(decision);

    // Only ASK about the port when we would actually spawn. Asking first meant
    // a launch that was never going to start anything (setup / repair) still
    // offered to kill a backend the user had running, with "Stop it and start
    // fresh" as the default button.
    let plan = startupPlan(decision, { packaged: !isDev });
    if (plan.askAboutPort) {
      const spawnOurs = await resolveBusyPort();
      plan = startupPlan(decision, {
        packaged: !isDev, portBusy: !spawnOurs, keepExisting: !spawnOurs,
      });
    }
    logShellLine(`Startup plan: ${JSON.stringify(plan)}`);

    if (!plan.spawn && !plan.thenLoad) return;   // the wizard is on screen

    // BEFORE the backend, never after: uvicorn imports `runtime/`, so replacing
    // those files under a running process is not an option, and starting first
    // would serve the old version for the whole session. Only when we are about
    // to spawn against an installed runtime — a checkout updates itself with
    // git, and the wizard unpacks its own package.
    if (plan.spawn && !isDev) {
      const update = await applyBundledUpdate(plan.python);
      if (!update.ok) {
        if (stopWaitingClock) stopWaitingClock();
        app.quit();
        return;
      }
      waitingPage.setPhase(mainWindow, 'starting');

      // AFTER the program files, because the lock the packages are brought to
      // is one of them; BEFORE the backend, because Windows holds the files of
      // a running interpreter open and uvicorn would import half of the old
      // packages and half of the new. It never blocks for a network problem --
      // the previous packages are supported too -- and it returns `repair` only
      // for an environment that failed verification twice.
      const sync = await runPackageSync(decision, plan);
      if (sync.cancelled) return;      // the app is quitting; start nothing
      if (sync.repair) {
        if (stopWaitingClock) stopWaitingClock();
        showRepairWizard(t(shellLocale(), 'syncRepairBody'));
        return;
      }
      waitingPage.setPhase(mainWindow, 'starting');
    }

    if (plan.spawn) startBackend(plan.python, plan.root);

    if (plan.spawn) {
      try {
        await waitForBackend();
        logShellLine('Backend ready.');
      } catch (err) {
        await reportBackendFailure(err);
        return;
      }
    }
    if (stopWaitingClock) stopWaitingClock();
    if (plan.thenLoad === 'backend') {
      // The wizard's language rides along on the first load, and is removed
      // only once that load has succeeded — a failed load keeps it for the
      // next start instead of losing it.
      const langDir = startLanguageDir();
      mainWindow?.loadURL(`http://127.0.0.1:${BACKEND_PORT}/${startLanguage.queryFor(langDir)}`)
        .then(() => startLanguage.consumed(langDir))
        .catch((err) => logShellLine(`Could not load the interface: ${err.message}`));
    }
    return;
  } catch (err) {
    // startupDecision() is written not to throw, but an unhandled rejection
    // here leaves a process with no window and no message at all.
    logShellLine(`Startup failed: ${err.message}`);
    dialog.showErrorBox('Orienta could not start', err.message);
  }
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

  // 2. Force-kill — SYNCHRONOUSLY.
  //
  // This used to sit in setTimeout(..., 1000). killBackend() is called from
  // 'before-quit', and Electron tears the process down long before a 1 s timer
  // can fire, so the taskkill never actually ran: the backend's survival came
  // down to whether the fire-and-forget /api/shutdown request above happened to
  // complete first. That race is why an orphaned backend kept coming back, held
  // port 8000, and let the next session attach to a stale process.
  // PID is a number from child_process.spawn — safe to interpolate.
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
