// @vitest-environment node
/**
 * Opt-in integration check against a REAL backend whose side of the socket is
 * dropped on purpose (a harness closes the server end twice while it keeps
 * broadcasting progress). Skipped unless ORIENTA_LIVE_WS=1.
 *
 * Runs in the node environment on purpose: Node's own WebSocket (undici) talks
 * to the real server, and sends no Origin header, which the backend accepts
 * (only browsers send one). window/document are stubbed because the module
 * subscribes to 'online' and 'visibilitychange'.
 */
import { describe, expect, it, vi } from 'vitest';

import { openReconnectingSocket } from './reconnectingSocket';

const live = process.env.ORIENTA_LIVE_WS === '1';

describe.skipIf(!live)('live reconnect against the real backend', () => {
  it('reconnects after each server-side close and progress flows again', async () => {
    vi.stubGlobal('window', { addEventListener() {}, removeEventListener() {} });
    vi.stubGlobal('document', { addEventListener() {}, removeEventListener() {}, visibilityState: 'visible' });
    const t0 = Date.now();
    const statuses = [];
    const progress = [];
    const handle = openReconnectingSocket({
      url: () => 'ws://127.0.0.1:8044/ws',
      onStatus: (s) => statuses.push([s, Date.now() - t0]),
      onMessage: (m) => { if (m.type === 'progress') progress.push([m.message, Date.now() - t0]); },
    });
    await new Promise((r) => setTimeout(r, 14000));
    handle.close();
    process.stderr.write(`STATUSES ${JSON.stringify(statuses)}
`);
    process.stderr.write(`PROGRESS_COUNT ${progress.length} last ${JSON.stringify(progress.slice(-1))}
`);

    const names = statuses.map(([s]) => s);
    const opens = names.filter((s) => s === 'open').length;
    expect(opens).toBeGreaterThanOrEqual(3);                 // first + two reconnects
    expect(names).toContain('reconnecting');
    // Messages kept arriving after the LAST reconnect.
    const lastOpenAt = statuses.filter(([s]) => s === 'open').at(-1)[1];
    expect(progress.filter(([, at]) => at > lastOpenAt).length).toBeGreaterThan(3);
  }, 30000);
});
