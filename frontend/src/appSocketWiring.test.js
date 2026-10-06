// @vitest-environment node
/**
 * The reconnecting socket only helps if the app uses it. App.jsx once opened
 * its own socket with a fixed 5 s retry; a refactor back to a bare
 * `new WebSocket` would pass every other test. This reads the source (the same
 * approach as the other wiring tests) and pins three facts: the page's socket
 * comes from openBackendSocket, App.jsx does not build one by hand, and a
 * reconnect re-reads the store from the backend.
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8');

describe('App socket wiring', () => {
  it('uses the reconnecting backend socket', () => {
    expect(app).toMatch(/openBackendSocket\(/);
    expect(app).not.toMatch(/new WebSocket\(/);
  });

  it('closes it on unmount', () => {
    expect(app).toMatch(/socket\.close\(\)/);
  });

  it('re-syncs the stores after a gap', () => {
    const block = app.slice(app.indexOf('openBackendSocket('));
    const status = block.slice(block.indexOf('onStatus'), block.indexOf('onStatus') + 400);
    expect(status).toMatch(/syncFromBackend\(\)/);
  });

  it('hands the heartbeat the HTTP probe, so a busy backend keeps its socket', () => {
    const api = readFileSync(new URL('./services/api.js', import.meta.url), 'utf8');
    const block = api.slice(api.indexOf('export const openBackendSocket'));
    expect(block).toMatch(/probe:\s*probeBackend/);
  });
});
