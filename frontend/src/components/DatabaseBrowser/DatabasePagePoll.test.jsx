// @vitest-environment jsdom
/**
 * The 5 s auto-refresh must not disturb what is on screen.
 *
 * Mac tester (M3/M5), 2026-09-25: "in the database browser the image flickered
 * a lot, because it was refreshing". The page polls every 5 s.
 *
 * Two notes on the instrument, both learned the hard way while writing this:
 *
 *  * `vi.useFakeTimers({shouldAdvanceTime: true})` is NOT used. Real time then
 *    keeps advancing the fake clock during every `await`, so the reply lands
 *    before the assertion and the in-flight state cannot be observed at all —
 *    a broken instrument that cheerfully reports "no flicker".
 *  * `cleanup()` is explicit. Without it the previous test's tree stays
 *    mounted, every query finds two of everything, and the assertions measure a
 *    stale component. That happened, which is why nothing is queried before a
 *    `settle()`.
 *
 * jsdom has no WebGL and never paints, so it cannot show a *visual* flicker.
 * What it can do is count how often React asks a component to render, and a
 * redraw follows from that.
 */
import '@testing-library/jest-dom/vitest';
import { render, screen, fireEvent, act, cleanup } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

const REQUEST_MS = 40;          // a plausible local HTTP round trip
const MASTER = 'Ni_master_20kV_npx500.h5';
const SHT = 'Ni (Ni) [cF4] {20kV}.sht';
const ENTRIES = [
  // SHT is the tab that is open by default, and its preview is the WebGL
  // master-pattern sphere — the "image" in the tester's report.
  { name: SHT, filename: SHT, file_type: 'sht', material: 'Ni',
    location: 'local', size_bytes: 555 },
  // file_type 'master' lands in the "Master H5" tab, whose preview is the
  // async 2D thumbnail — an <img>. SHT/CIF/XTAL render WebGL scenes instead.
  { name: MASTER, filename: MASTER, file_type: 'master', material: 'Ni',
    location: 'local', size_bytes: 1234 },
  { name: 'Al.cif', filename: 'Al.cif', file_type: 'CIF', material: 'Al',
    location: 'local', size_bytes: 99 },
];

const browse = vi.fn();
const cacheStatus = vi.fn();
const preview = vi.fn();

vi.mock('../../services/api', () => ({
  dbApi: {
    browse: (...a) => browse(...a),
    cacheStatus: (...a) => cacheStatus(...a),
    preview: (...a) => preview(...a),
    shtInfo: vi.fn().mockResolvedValue({ data: { provenance: {}, parameters: {} } }),
  },
  settingsApi: { get: vi.fn().mockResolvedValue({ data: {} }) },
}));

// Counts how often the WebGL preview is asked to render.
const viewerRenders = { count: 0 };
vi.mock('./MasterSphereViewer', () => ({
  default: function MasterSphereViewerStub() {
    viewerRenders.count += 1;
    return null;
  },
}));

let DatabasePage;

const entriesReply = (entries) => () => new Promise((resolve) => {
  setTimeout(() => resolve({ data: { entries } }), REQUEST_MS);
});

beforeEach(async () => {
  vi.useFakeTimers();
  viewerRenders.count = 0;
  browse.mockReset();
  cacheStatus.mockReset();
  preview.mockReset();
  // A fresh array of fresh objects each call — that is what an HTTP reply is.
  browse.mockImplementation(() => entriesReply(ENTRIES.map((e) => ({ ...e })))());
  cacheStatus.mockImplementation(() => new Promise((resolve) => {
    setTimeout(() => resolve({ data: { used_bytes: 1, max_bytes: 20, file_count: 2 } }),
               REQUEST_MS);
  }));
  preview.mockResolvedValue({ data: { image: 'QUJD' } });   // "ABC" base64
  ({ default: DatabasePage } = await import('./DatabasePage'));
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.resetModules();
});

/** Advance the clock and let everything it started finish. */
async function settle(ms) {
  await act(async () => { await vi.advanceTimersByTimeAsync(ms); });
}

const READY = REQUEST_MS + 5;

function selectTheMaster() {
  fireEvent.click(screen.getByRole('tab', { name: /Master H5/i }));
  fireEvent.click(screen.getByText(MASTER));
}

describe('the 5 s poll', () => {
  it('runs at 5 s and asks the backend again', async () => {
    render(<DatabasePage isActive />);
    await settle(READY);
    expect(browse).toHaveBeenCalledTimes(1);

    await settle(5001 + REQUEST_MS);
    expect(browse).toHaveBeenCalledTimes(2);

    await settle(10000);
    expect(browse).toHaveBeenCalledTimes(4);
    expect(cacheStatus).toHaveBeenCalledTimes(4);   // measured cadence
  });

  it('does not re-fetch the preview image when nothing changed', async () => {
    render(<DatabasePage isActive />);
    await settle(READY);
    selectTheMaster();
    await settle(READY);
    expect(preview).toHaveBeenCalledTimes(1);

    await settle(15000);                            // three polls
    expect(browse.mock.calls.length).toBeGreaterThan(3);
    expect(preview).toHaveBeenCalledTimes(1);
  });

  it('keeps the very same <img> element across polls', async () => {
    // A remount, or a rewritten src, makes the browser decode again and the
    // picture blinks. The assertion is DOM identity, because that is what
    // "it flickered" means.
    render(<DatabasePage isActive />);
    await settle(READY);
    selectTheMaster();
    await settle(READY);

    const before = screen.getByRole('img');
    const srcBefore = before.getAttribute('src');

    await settle(15000);
    expect(browse.mock.calls.length).toBeGreaterThan(3);

    const after = screen.getByRole('img');
    expect(after).toBe(before);                     // not remounted
    expect(after.getAttribute('src')).toBe(srcBefore);
  });

  it('does not put the page back into its loading state', async () => {
    // `loading` dims the Refresh button and swaps its label. Measured at the
    // instant the poll is in flight; after the reply there is nothing to see.
    render(<DatabasePage isActive />);
    await settle(READY);
    const refresh = screen.getByRole('button', { name: 'Refresh' });
    const atRest = `${refresh.textContent}@${refresh.style.opacity}`;

    // Land INSIDE the request. The clock already stands at READY, the poll
    // fires at t=5000 and its reply arrives at t=5000+REQUEST_MS, so advancing
    // by 5001 would overshoot to t=5046 and observe nothing — which is exactly
    // how this test first reported "no flicker" for both versions of the code.
    await settle(5000 - READY + 10);                // t ~ 5010: reply pending
    expect(browse).toHaveBeenCalledTimes(2);
    expect(`${refresh.textContent}@${refresh.style.opacity}`).toBe(atRest);

    await settle(READY);
    expect(`${refresh.textContent}@${refresh.style.opacity}`).toBe(atRest);
  });

  it('a user-initiated refresh still announces itself', async () => {
    // The silent flag must not swallow the visible case.
    render(<DatabasePage isActive />);
    await settle(READY);
    const refresh = screen.getByRole('button', { name: 'Refresh' });
    expect(`${refresh.textContent}@${refresh.style.opacity}`).toBe('Refresh@1');

    fireEvent.click(refresh);
    await act(async () => { await Promise.resolve(); });

    expect(`${refresh.textContent}@${refresh.style.opacity}`).not.toBe('Refresh@1');
  });

  it('a poll that finds the same files re-renders the preview zero times', async () => {
    // The tester's symptom, as a number.
    render(<DatabasePage isActive />);
    await settle(READY);

    // No tab click: SHT is the tab that opens by default.
    fireEvent.click(screen.getByText(SHT));
    await settle(READY);

    const before = viewerRenders.count;
    expect(before).toBeGreaterThan(0);              // it is on screen

    await settle(15000);                            // three polls, nothing new
    expect(browse.mock.calls.length).toBeGreaterThan(3);

    expect(viewerRenders.count).toBe(before);
  });

  it('but a poll that finds a change still gets through', async () => {
    // The guard must not make the page blind: a file that finished downloading
    // has to appear. That is why the comparison includes `location`.
    render(<DatabasePage isActive />);
    await settle(READY);
    expect(screen.getAllByText(/local/i).length).toBeGreaterThan(0);

    browse.mockImplementation(entriesReply([
      { ...ENTRIES[0], location: 'both' },
      { ...ENTRIES[1] },
    ]));

    await settle(5001 + READY);
    expect(screen.getAllByText(/both/i).length).toBeGreaterThan(0);
  });
});

describe('the comparison helpers', () => {
  it('sameEntries ignores a re-serialised reply but sees a real change', async () => {
    const { sameEntries } = await import('./DatabasePage');
    const a = [{ name: 'x', location: 'local', size_bytes: 1 }];
    // Same data, keys in a different order. JSON.stringify would call this a
    // change and re-render the page every 5 s.
    const b = [{ size_bytes: 1, location: 'local', name: 'x' }];
    expect(sameEntries(a, b)).toBe(true);
    expect(sameEntries(a, [{ name: 'x', location: 'both', size_bytes: 1 }])).toBe(false);
    expect(sameEntries(a, [])).toBe(false);
    expect(sameEntries(a, a)).toBe(true);
    expect(sameEntries(null, undefined)).toBe(false);
  });

  it('shallowEqual compares one level and tolerates rubbish', async () => {
    const { shallowEqual } = await import('./DatabasePage');
    expect(shallowEqual({ a: 1 }, { a: 1 })).toBe(true);
    expect(shallowEqual({ a: 1 }, { a: 2 })).toBe(false);
    expect(shallowEqual({ a: 1 }, { a: 1, b: 2 })).toBe(false);
    expect(shallowEqual(null, null)).toBe(true);
    expect(shallowEqual(null, { a: 1 })).toBe(false);
  });
});
