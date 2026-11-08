/**
 * What the shell says when the Python backend process has ended.
 *
 * Kept out of main.js so it can be tested: main.js spawns the backend when it
 * is loaded. Until 0.4.8 the only trace of an end was "Backend exited with code
 * N" (and nothing at all when the app itself killed it), so a backend that
 * stopped in the middle of the work could not be told from one the app closed.
 *
 * Three sources make one line:
 *   - the exit code or signal Node reports (on Windows a native crash is an
 *     NTSTATUS such as 0xC0000005, which Node hands over as 3221225477);
 *   - why the app stopped it, when it did (`stopReason`);
 *   - the state file the backend writes every few seconds (backend/lifecycle.py):
 *     the last request, the requests still running, the memory. After a hard
 *     kill or a native crash it is the only record of what the backend was doing.
 */
const fs = require('node:fs');

// NTSTATUS values a crashed or killed Windows process reports, the ones seen
// or plausible for a Python process with native extensions.
const WINDOWS_STATUS = {
  0xC0000005: 'access violation: a native crash in Python or a library',
  0xC00000FD: 'stack overflow: a native crash',
  0xC0000409: 'fail-fast / stack buffer overrun: a native crash',
  0xC0000374: 'heap corruption: a native crash',
  0xC0000017: 'not enough memory',
  0xC000012D: 'the commit limit of the computer (RAM plus page file) was reached',
  0xC000013A: 'stopped by Ctrl+C or by closing its console',
  0xE06D7363: 'an unhandled C++ exception: a native crash',
};

/**
 * @param {?number} code   exit code as Node reports it (null when a signal ended it)
 * @param {?string} signal POSIX signal name or null
 * @param {string} [platform=process.platform]
 * @returns {string}
 */
function describeExit(code, signal, platform = process.platform) {
  if (signal) {
    const hint = signal === 'SIGKILL'
      ? ' (killed from outside, for example by the system when memory ran out)'
      : '';
    return `signal ${signal}${hint}`;
  }
  if (code === null || code === undefined) return 'no exit code';
  if (code === 0) return 'exit code 0 (the backend ended itself)';
  if (code === 1) {
    return 'exit code 1 (a Python error, a port that was taken, or the process was ended from '
      + 'outside, for example in the Task Manager; the lines above say which)';
  }
  if (platform === 'win32') {
    const status = code >>> 0;
    const text = WINDOWS_STATUS[status];
    const hex = `0x${status.toString(16).toUpperCase().padStart(8, '0')}`;
    if (text) return `exit code ${hex} (${text})`;
    if (status >= 0xC0000000) return `exit code ${hex} (a Windows error status)`;
  }
  return `exit code ${code}`;
}

/**
 * The state file the backend left, or null when there is none or it cannot be read.
 * Only a file written by THIS backend process counts (`pid`), so a stale file
 * from an earlier session is never quoted as this one's last state.
 */
function readBackendState(file, pid) {
  if (!file) return null;
  try {
    const s = JSON.parse(fs.readFileSync(file, 'utf8'));
    if (pid && s && s.pid !== pid) return null;
    return s && typeof s === 'object' ? s : null;
  } catch {
    return null;
  }
}

function requestText(r) {
  return r ? `${r.method} ${r.path} at ${r.at}` : 'none';
}

/**
 * The one line for backend-console.log.
 * @param {object} o
 * @param {?number} o.code
 * @param {?string} o.signal
 * @param {?string} o.stopReason  why the app stopped it; null = it ended on its own
 * @param {?object} o.state       readBackendState(...)
 * @param {string} [o.platform]
 */
function backendEndLine({ code, signal, stopReason, state, platform = process.platform }) {
  const parts = [`Backend ended: ${describeExit(code, signal, platform)}.`];
  parts.push(stopReason ? `Stopped by the app: ${stopReason}.` : 'Not stopped by the app: unexpected end.');
  if (state) {
    const running = (state.in_flight || [])
      .map((r) => `${r.method} ${r.path} (${r.running_s} s)`).join(', ') || 'none';
    const mem = state.memory_mib && state.memory_mib.commit !== undefined
      ? `commit ${state.memory_mib.commit} MiB, RSS ${state.memory_mib.rss} MiB` : 'unknown';
    parts.push(`Its last state (written ${state.written}, after ${state.uptime_s} s): `
      + `${state.ended ? `it said it was ending: ${state.ended}; ` : ''}`
      + `last request ${requestText(state.last_request)}; still running: ${running}; memory ${mem}.`);
  } else {
    parts.push('No state file from this backend process.');
  }
  return parts.join(' ');
}

/**
 * What the page shows about an unexpected end (sent over IPC, so plain data).
 */
function backendEndNotice({ code, signal, state, restarting, platform = process.platform }) {
  return {
    at: new Date().toISOString(),
    exit: describeExit(code, signal, platform),
    lastRequest: state && state.last_request
      ? `${state.last_request.method} ${state.last_request.path}` : null,
    running: state ? (state.in_flight || []).map((r) => `${r.method} ${r.path}`) : [],
    commitMiB: state && state.memory_mib ? (state.memory_mib.commit ?? null) : null,
    restarting: Boolean(restarting),
  };
}

module.exports = { describeExit, readBackendState, backendEndLine, backendEndNotice, WINDOWS_STATUS };
