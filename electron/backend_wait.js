/**
 * Wait until the Python backend answers its health check.
 *
 * Kept out of main.js so it can be tested: main.js spawns the backend when it is
 * loaded.
 *
 * Exactly ONE request is in flight at any time, and the limit is wall-clock time,
 * not a number of attempts. The version this replaces scheduled the next attempt
 * from the request's timeout handler AND from its error handler — and
 * `req.destroy()` in the timeout handler fires the error handler too. Every
 * health request that took longer than its 2 s timeout therefore doubled the
 * number of parallel request chains, all counting into one shared attempt
 * counter. On a cold start, while the backend's start-up work keeps it from
 * answering quickly, the "3 minute" wait ran out of attempts after about 20 s
 * and reported "Backend not reachable after 180 seconds" for a backend that was
 * up and answering (measured with a server that answers every request after
 * 3 s: 1, 3, 7, 15 … 511 requests, rejected after 22 s). The app then quit and
 * killed the backend it had just started; the second start, with warm disk
 * caches, worked.
 */
const http = require('node:http');

/**
 * @param {object} options
 * @param {number} options.port
 * @param {number} [options.limitMs=180000]       give up after this long (wall clock)
 * @param {number} [options.intervalMs=500]       pause between two attempts
 * @param {number} [options.requestTimeoutMs=10000] how long one request may take; a
 *   starved backend that answers slowly is still a backend that answers
 * @param {string} [options.host='127.0.0.1']
 * @param {string} [options.path='/api/health']
 * @param {() => number} [options.now=Date.now]
 * @returns {Promise<void>} resolves on the first 200, rejects once the limit is reached
 */
function waitForBackend({
  port,
  limitMs = 180_000,
  intervalMs = 500,
  requestTimeoutMs = 10_000,
  host = '127.0.0.1',
  path = '/api/health',
  now = Date.now,
} = {}) {
  return new Promise((resolve, reject) => {
    const started = now();
    let settled = false;
    let lastProblem = 'no answer';

    const finish = (err) => {
      if (settled) return;
      settled = true;
      if (err) reject(err); else resolve();
    };

    const attempt = () => {
      if (settled) return;
      let done = false;
      // Every way an attempt can end comes through here once; only here is the
      // next attempt scheduled, so a request that times out AND errors cannot
      // start two chains.
      const next = (problem) => {
        if (done) return;
        done = true;
        if (problem) lastProblem = problem;
        if (now() - started >= limitMs) {
          finish(new Error(
            `Backend not reachable after ${Math.round(limitMs / 1000)} seconds (${lastProblem})`));
          return;
        }
        setTimeout(attempt, intervalMs);
      };

      const req = http.get({ host, port, path }, (res) => {
        res.resume();
        if (res.statusCode === 200) {
          done = true;
          finish();
        } else {
          next(`health check answered ${res.statusCode}`);
        }
      });
      req.on('error', (err) => next(err && err.code ? err.code : 'connection error'));
      req.setTimeout(requestTimeoutMs, () => {
        // destroy() makes the request emit 'error'; next() runs once either way.
        next(`no answer within ${Math.round(requestTimeoutMs / 1000)} s`);
        req.destroy();
      });
    };

    attempt();
  });
}

module.exports = { waitForBackend };
