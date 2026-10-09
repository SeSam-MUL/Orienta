/**
 * One start of the shell's package sync, without Electron, for
 * `tests/test_package_sync_seam.py`.
 *
 * It is the REAL `package_sync.js` with the REAL runner and the real pip; only
 * the things that need a window are replaced by spies (the dialog, the waiting
 * page). Everything it saw is printed as one JSON document on stdout, so the
 * test can assert on what actually happened rather than on what was logged.
 *
 * usage: node package_sync_seam_driver.js <repo> <home> <python>
 *
 * environment
 *   SEAM_KILL=close   when pip announces it is about to install, end the sync's
 *                     children the way closing the window does (the app is
 *                     quitting: nothing may be decided from here on)
 *   SEAM_KILL=pip     when pip has removed the first old package, kill pip's
 *                     process tree and nothing else: the app lives on, with an
 *                     environment that is missing a package
 *   everything else   (PIP_INDEX_URL, ORIENTA_PYTORCH_INDEX_CPU, ...) goes to pip
 */
'use strict';

const path = require('node:path');
const childProcess = require('node:child_process');

const [repo, home, python] = process.argv.slice(2);
const sync = require(path.join(repo, 'electron', 'setup', 'package_sync.js'));
const installer = require(path.join(repo, 'electron', 'setup', 'installer.js'));
const platform = require(path.join(repo, 'electron', 'platform.js'));

const kill = process.env.SEAM_KILL || '';
const logs = [];
const calls = [];
const phases = [];
const notified = [];
let lastChild = null;
let killed = false;

// The real spawn, with the child remembered so that "pip died, the app did not"
// can be played without a way for the runner to expose its processes.
const runner = sync.createRunner({
  platformName: process.platform,
  spawn: (exe, args, options) => {
    lastChild = childProcess.spawn(exe, args, options);
    return lastChild;
  },
});

function killPipOnly() {
  const child = lastChild;
  if (!child || !child.pid) return;
  const plan = platform.killTree(child.pid);
  if (plan.kind === 'command') {
    childProcess.execFile(plan.command, plan.args, { windowsHide: true }, () => {});
  } else {
    try { process.kill(plan.target, plan.signal); } catch { /* gone */ }
  }
}

async function run(exe, args, opts = {}) {
  const isPip = args[0] === '-m' && args[1] === 'pip';
  const isInstall = isPip && args.includes('install')
    && !args.includes('--dry-run') && !args.includes('--force-reinstall');
  calls.push({
    pip: isPip,
    kind: !isPip ? (args.includes('-c') ? 'python-c' : 'python')
      : args.includes('--dry-run') ? 'dry'
        : args.includes('--force-reinstall') ? 'reinstall'
          : args.includes('check') ? 'check' : 'install',
  });
  const onLine = (line) => {
    if (opts.onLine) opts.onLine(line);
    if (!isInstall || killed) return;
    if (kill === 'close' && /Installing collected packages/.test(line)) {
      killed = true;
      runner.killAll();
    }
    if (kill === 'pip' && /Successfully uninstalled (kikuchipy|orix|pyebsdindex)-/.test(line)) {
      killed = true;
      killPipOnly();
    }
  };
  return runner.run(exe, args, { ...opts, onLine });
}

(async () => {
  const result = await sync.syncPackages(
    { home, python, decisionMode: 'run', platformName: process.platform, env: process.env },
    {
      run,
      isCancelled: () => runner.cancelled,
      log: (line) => logs.push(line),
      freeBytes: (dir) => installer.measureFree(dir),
      onPhase: (phase) => phases.push(phase),
      notify: async (info) => { notified.push(info); },
      runtimeTag: () => 'v0.4.7',
    },
  );
  // Exit only when the pipe has taken the whole document.
  process.stdout.write(JSON.stringify({
    result, calls, phases, notified, logs, killed, cancelled: runner.cancelled,
  }), () => process.exit(0));
})().catch((err) => {
  process.stderr.write(`driver failed: ${err && err.stack ? err.stack : err}\n`);
  process.exit(2);
});
