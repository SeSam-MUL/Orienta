/**
 * One answer to "what should this launch do", read by createWindow AND by
 * app.whenReady().
 *
 * Earlier drafts asked it two ways in two places — "does .python_path exist"
 * and "does findPython() return something" — and the two disagree for a user
 * whose interpreter has been deleted or quarantined: no wizard is shown AND
 * nothing is started, leaving a window on Chromium's ERR_CONNECTION_REFUSED
 * page with no dialog and no log.
 *
 * NOTHING HERE THROWS except orientaHome() on a malformed override. These
 * functions run inside app.whenReady() and, through startBackend(), inside a
 * setTimeout, where a throw is an unhandled rejection and the window dies
 * silently.
 */
const fs = require('node:fs');
const path = require('node:path');

// One JavaScript definition of the install location, shared with the setup and
// pinned against Python's bundle_update.orienta_home() by a contract test.
const { orientaHome } = require('./update_endpoints');
const { t } = require('./strings');

// Translated like every other message on that screen. It used to be a bare
// English constant, and `setup.js` renders `ctx.message` verbatim as the first
// sentence a German user reads on the first screen.

// --------------------------------------------------------------------------
// the shell's own log
// --------------------------------------------------------------------------
//
// openBackendLog() is called ONLY from startBackend(), so in every state where
// startup fails before the backend there is no stream — and a packaged Electron
// app has no console either, so console.log reaches nobody. This log lives
// under the Orienta home, which exists independently of the runtime and
// survives every failure mode in this file.

let shellLogFile = null;

/**
 * Deliberately `appendFileSync` and not a write stream.
 *
 * This log exists to capture the line immediately BEFORE something goes wrong
 * — a failed decision, a refused update, a startup that never reaches the
 * backend. A stream is asynchronous, so the last line, which is the one that
 * matters, is exactly the line a crash or an `app.quit()` can lose. Synchronous
 * appends cost nothing at this volume (single figures per launch) and cannot.
 */
function openStartupLog(dir) {
  if (shellLogFile) return; // called once; a second call is a no-op
  try {
    fs.mkdirSync(dir, { recursive: true });
    const file = path.join(dir, 'orienta-shell.log');
    // One-generation rotation, like openBackendLog: a machine that fails to
    // start every day must not grow an unbounded file.
    try {
      if (fs.statSync(file).size > 2 * 1024 * 1024) {
        fs.rmSync(`${file}.1`, { force: true });
        fs.renameSync(file, `${file}.1`);
      }
    } catch {
      /* absent, or cannot tell: either way, just append */
    }
    fs.appendFileSync(file, `\n===== shell start ${new Date().toISOString()} =====\n`);
    shellLogFile = file;
  } catch {
    // Must never throw: this is the function that makes the other failures
    // visible, so its own failure cannot be allowed to hide them.
    shellLogFile = null;
  }
}

/** Only for tests, which need each case to start from a clean singleton. */
function closeStartupLog() {
  shellLogFile = null;
}

function logShellLine(line) {
  const stamped = `[${new Date().toISOString()}] ${line}`;
  try {
    if (shellLogFile) fs.appendFileSync(shellLogFile, `${stamped}\n`);
  } catch {
    /* a logger that throws is worse than a silent one */
  }
  console.log(stamped);
}

// --------------------------------------------------------------------------
// filesystem questions, answered in three states rather than two
// --------------------------------------------------------------------------

/** null means ABSENT. Anything else is raised: "cannot tell" is not "absent",
 *  and callers of this decide whether a machine is installed. */
function statOrThrow(target) {
  try {
    return fs.lstatSync(target);
  } catch (err) {
    if (err.code === 'ENOENT') return null;
    throw err;
  }
}

/** The non-throwing form, for the decisions below.
 *
 * `statSync`, not `lstatSync`: a dangling symlink and a OneDrive Files-On-Demand
 * placeholder both satisfy an lstat while being unusable, and `findPython` uses
 * `existsSync` (which DOES follow links) — two interpreter finders in one
 * codebase must not disagree about the same path.
 */
function safeStat(target) {
  try {
    return { stat: fs.statSync(target), error: null };
  } catch (err) {
    if (err.code === 'ENOENT') return { stat: null, error: null };
    return { stat: null, error: err.message };
  }
}

/**
 * The interpreter THIS INSTALLATION recorded, if any.
 *
 *   null                 nothing installed — a first run
 *   {path}               installed and present
 *   {path: null, error}  recorded but missing or unreadable: a REPAIR, not a
 *                        first run, and never a reason to fall back to a conda
 *                        scan or to bare `python` (which on Windows is an App
 *                        Execution Alias that opens the Microsoft Store).
 */
function installedInterpreter() {
  let config;
  try {
    config = path.join(orientaHome(), '.python_path');
  } catch (err) {
    return { path: null, error: err.message };
  }

  const found = safeStat(config);
  if (found.error) return { path: null, error: `${config} cannot be read: ${found.error}` };
  if (!found.stat) return null;

  let recorded = '';
  try {
    recorded = fs.readFileSync(config, 'utf8').trim();
  } catch (err) {
    return { path: null, error: `${config} cannot be read: ${err.message}` };
  }
  if (!recorded) return { path: null, error: `${config} is empty` };

  const exe = safeStat(recorded);
  if (exe.error) {
    return { path: null, error: `the installed interpreter cannot be read (${recorded}): ${exe.error}` };
  }
  if (!exe.stat) {
    return { path: null, error: `the installed interpreter is not there any more (${recorded})` };
  }
  if (!exe.stat.isFile()) {
    // A .python_path naming the ENVIRONMENT DIRECTORY instead of python.exe is
    // an easy mistake for a setup to make, and `spawn` on a directory gives
    // EACCES three minutes before anyone finds out.
    return { path: null, error: `the installed interpreter is not a program (${recorded})` };
  }
  return { path: recorded, error: null };
}

/**
 * The unpacked runtime.
 *
 *   null                 no runtime installed
 *   {path}               installed and usable
 *   {path: null, error}  something is there and we cannot read it
 *
 * The third state is the point, and discarding it was a real defect: a VERSION
 * file locked by antivirus during the first launch after an install answered
 * "no runtime installed", and combined with an absent .python_path that gives
 * `setup` — the first-run wizard, on a machine with a perfectly good runtime
 * sitting next to it. `installedInterpreter` already reported the distinction;
 * the two halves of one decision must not disagree.
 */
function installedRuntime() {
  let runtime;
  try {
    runtime = path.join(orientaHome(), 'runtime');
  } catch (err) {
    return { path: null, error: err.message };
  }
  const version = safeStat(path.join(runtime, 'VERSION'));
  if (version.error) return { path: null, error: `${runtime}: ${version.error}` };
  if (!version.stat) return null;
  if (!version.stat.isFile()) {
    // A half-extraction CAN fake this: an unzip that created a directory named
    // VERSION would otherwise be reported as a complete runtime, and uvicorn
    // would then fail with "No module named backend" three restarts later.
    return { path: null, error: `${path.join(runtime, 'VERSION')} is not a file` };
  }
  return { path: runtime, error: null };
}

/** `{tag, file}` when a package is parked, `{error}` when the record is
 *  unreadable, null when there is nothing to do. */
function pendingUpdate() {
  let file;
  try {
    file = path.join(orientaHome(), 'pending.json');
  } catch (err) {
    return { error: err.message };
  }
  let text;
  try {
    text = fs.readFileSync(file, 'utf8');
  } catch (err) {
    if (err.code === 'ENOENT') return null;
    return { error: `the pending-update record could not be read: ${err.message}` };
  }
  try {
    const record = JSON.parse(text);
    return { tag: String(record.tag || ''), file: String(record.file || '') };
  } catch (err) {
    return { error: `the pending-update record is not readable JSON: ${err.message}` };
  }
}

/** This checkout — the development layout, and the last-resort answer. */
function checkoutRoot() {
  return path.resolve(__dirname, '..');
}

/**
 * Where the backend's code lives.
 *
 * Prefer `decision.root`. This exists for callers that have no decision to
 * hand, and it deliberately does NOT consult the installed runtime when
 * ORIENTA_PROJECT_ROOT names a checkout.
 */
function resolveProjectRoot() {
  if (process.env.ORIENTA_PROJECT_ROOT) return process.env.ORIENTA_PROJECT_ROOT;
  const runtime = installedRuntime();
  return (runtime && runtime.path) || checkoutRoot();
}

/**
 * THE startup decision.
 *
 *   {mode: 'dev',    root}            a checkout: behave exactly as before
 *   {mode: 'setup'}                   nothing installed -> the wizard
 *   {mode: 'run',    python, root}    ready
 *   {mode: 'repair', message}         installed but unusable -> the wizard in
 *                                     repair mode, carrying the reason
 *
 * `run` is returned only when BOTH halves are present. Half an installation is
 * a repair, and saying so is the entire point: the alternative is a window
 * pointed at a backend that was never started.
 */
function startupDecision(locale = 'en') {
  if (process.env.ORIENTA_PROJECT_ROOT) {
    return { mode: 'dev', root: process.env.ORIENTA_PROJECT_ROOT };
  }

  let interpreter = null;
  let runtime = null;
  try {
    interpreter = installedInterpreter();
    runtime = installedRuntime();
  } catch (err) {
    // Belt and braces. The helpers above are written not to throw; if one ever
    // does, a repair screen with a message beats an unhandled rejection.
    return {
      mode: 'repair',
      message: t(locale, 'repairCannotInspect', { detail: err.message }),
    };
  }
  /* eslint-disable-next-line no-unused-vars */

  // "Nothing is installed" must mean exactly that — never an error we failed to
  // distinguish. An unreadable runtime plus an absent .python_path used to land
  // here and show the first-run wizard to someone who already has Orienta.
  if (runtime && runtime.error) {
    return {
      mode: 'repair',
      message: `${runtime.error}. ${t(locale, 'reinstallHint')}`,
    };
  }
  if (!interpreter && !runtime) return { mode: 'setup' };

  if (!interpreter) return { mode: 'repair', message: t(locale, 'repairPythonMissing') };
  if (interpreter.error) {
    return {
      mode: 'repair',
      message: t(locale, 'repairPythonUnreadable', { detail: interpreter.error }),
    };
  }
  if (!runtime) return { mode: 'repair', message: t(locale, 'repairRuntimeMissing') };
  return { mode: 'run', python: interpreter.path, root: runtime.path };
}

/**
 * What the shell should DO, as data — no Electron, no filesystem, no side
 * effects.
 *
 * `startupDecision()` answers "what is installed". This answers "therefore,
 * what happens": which page to show, whether to spawn a backend, with which
 * interpreter and root, and what to load once it answers. Keeping it pure is
 * the only way the branches below can be tested at all — nothing in the test
 * suite can load `main.js`, so every one of these decisions used to be covered
 * by nothing.
 *
 * @param decision  from startupDecision()
 * @param context   {packaged, portBusy, keepExisting}
 *   portBusy      another backend already answers on the port
 *   keepExisting  ...and the user chose to keep it
 */
function startupPlan(decision, context = {}) {
  const { packaged = false, portBusy = false, keepExisting = false } = context;

  // In a packaged build the backend SERVES the interface; in a checkout Vite
  // does. That — not the decision mode — is what decides where the window
  // points, which is why a packaged shell pointed at a checkout
  // (ORIENTA_PROJECT_ROOT, a documented override) used to sit on the spinner
  // for ever: it is `dev` by decision and packaged by deployment.
  const backendServesUi = packaged;
  const backendUrl = 'backend';
  const devServerUrl = 'devServer';

  if (decision.mode === 'setup' || decision.mode === 'repair') {
    // Nothing to start, so nothing to ask about the port. Asking first is how a
    // launch that was never going to spawn anything still offered to kill the
    // user's running backend, with "Stop it and start fresh" as the default.
    return {
      page: 'wizard',
      askAboutPort: false,
      spawn: false,
      python: null,
      root: null,
      thenLoad: null,
      reason: decision.message || '',
    };
  }

  const page = backendServesUi ? 'waiting' : 'devServer';
  const thenLoad = backendServesUi ? backendUrl : devServerUrl;

  if (portBusy && keepExisting) {
    // The backend on that port ANSWERS — that is how we know it is there.
    // Pointing the window at it is the entire purpose of the "keep it" button.
    return {
      page, askAboutPort: true, spawn: false, python: null, root: null,
      thenLoad: backendServesUi ? backendUrl : devServerUrl, reason: 'kept-existing',
    };
  }

  return {
    page,
    askAboutPort: true,
    spawn: true,
    // `root` comes from the DECISION, never re-derived. Re-deriving it meant
    // that once an installation existed on a developer's machine, dev mode
    // silently ran the backend out of the installed runtime instead of the
    // checkout — edits to backend/ would have had no effect, with no message.
    python: decision.python || null,
    root: decision.root || null,
    thenLoad,
    reason: '',
  };
}

module.exports = {
  orientaHome,
  checkoutRoot,
  startupPlan,
  statOrThrow,
  installedInterpreter,
  installedRuntime,
  pendingUpdate,
  resolveProjectRoot,
  startupDecision,
  openStartupLog,
  closeStartupLog,
  logShellLine,
};
