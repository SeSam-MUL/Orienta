/**
 * Waiting for the backend: one health request at a time, and a limit in seconds.
 *
 * The old wait in main.js scheduled the next attempt from a request's timeout
 * handler and again from its error handler (which `req.destroy()` fires), so
 * every slow answer doubled the number of request chains. On a cold start it
 * gave up after about 20 s of a 3-minute wait and quit the app over a backend
 * that was answering. These tests use a real local server that answers slowly.
 */
import { describe, it, expect, afterEach } from 'vitest';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { waitForBackend } = require('../../electron/backend_wait.js');

const here = path.dirname(fileURLToPath(import.meta.url));
const MAIN = fs.readFileSync(path.join(here, '../../electron/main.js'), 'utf8');

let server = null;

/** A server whose every answer takes `delayMs`; counts requests and the most in flight at once. */
function slowServer({ delayMs, status = 200 }) {
  const stats = { requests: 0, inFlight: 0, maxInFlight: 0 };
  server = http.createServer((req, res) => {
    stats.requests += 1;
    stats.inFlight += 1;
    stats.maxInFlight = Math.max(stats.maxInFlight, stats.inFlight);
    const timer = setTimeout(() => {
      stats.inFlight -= 1;
      res.writeHead(status);
      res.end('{}');
    }, delayMs);
    req.on('close', () => clearTimeout(timer));
    res.on('close', () => { if (!res.writableEnded) stats.inFlight -= 1; });
  });
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => resolve({ port: server.address().port, stats }));
  });
}

afterEach(async () => {
  if (server) {
    server.closeAllConnections?.();
    await new Promise((r) => server.close(r));
    server = null;
  }
});

describe('waitForBackend', () => {
  it('keeps one request in flight when every answer is slower than the request timeout', async () => {
    const { port, stats } = await slowServer({ delayMs: 400 });
    const t0 = Date.now();
    await expect(waitForBackend({ port, limitMs: 3000, intervalMs: 50, requestTimeoutMs: 150 }))
      .rejects.toThrow(/not reachable after 3 seconds/);
    const elapsed = Date.now() - t0;
    // The old code doubled the chains on every timeout: with 150 ms timeouts and
    // 50 ms pauses it would have sent hundreds of requests and stopped long before
    // the limit. One chain sends at most limit / (timeout + pause) + 1 = 16.
    expect(stats.maxInFlight).toBe(1);
    expect(stats.requests).toBeLessThanOrEqual(16);
    expect(elapsed).toBeGreaterThanOrEqual(3000);
  });

  it('waits for a slow answer instead of counting it as no answer', async () => {
    const { port, stats } = await slowServer({ delayMs: 1500 });
    await expect(waitForBackend({ port, limitMs: 10_000, intervalMs: 50, requestTimeoutMs: 5000 }))
      .resolves.toBeUndefined();
    expect(stats.requests).toBe(1);
  });

  it('retries while nothing listens and resolves once the backend is up', async () => {
    // Reserve a free port, close it, start the server on it later.
    const probe = http.createServer();
    const port = await new Promise((r) => probe.listen(0, '127.0.0.1', () => r(probe.address().port)));
    await new Promise((r) => probe.close(r));
    const waiting = waitForBackend({ port, limitMs: 10_000, intervalMs: 50, requestTimeoutMs: 1000 });
    await new Promise((r) => setTimeout(r, 600));
    server = http.createServer((req, res) => { res.writeHead(200); res.end('{}'); });
    await new Promise((r) => server.listen(port, '127.0.0.1', r));
    await expect(waiting).resolves.toBeUndefined();
  });

  it('does not count a non-200 answer as ready, and says what it got', async () => {
    const { port } = await slowServer({ delayMs: 0, status: 503 });
    await expect(waitForBackend({ port, limitMs: 500, intervalMs: 50, requestTimeoutMs: 500 }))
      .rejects.toThrow(/answered 503/);
  });
});

describe('main.js', () => {
  it('waits through backend_wait.js, not a copy of its own', () => {
    expect(MAIN).toMatch(/require\('\.\/backend_wait\.js'\)/);
    expect(MAIN).not.toMatch(/req\.setTimeout\(2000/);
    expect(MAIN).toMatch(/waitForBackendHealth\(\{ port: BACKEND_PORT, limitMs: BACKEND_START_LIMIT_MS \}\)/);
    expect(MAIN).toMatch(/const BACKEND_START_LIMIT_MS = 180_000;/);
  });
});
