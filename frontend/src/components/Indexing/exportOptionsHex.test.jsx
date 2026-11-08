// @vitest-environment jsdom
/**
 * The page, mounted: the export list offers the square .ang and the .ctf for a result of a hexagonal scan
 * only, and a resampled export says in the log what it did, with the numbers of the header
 * `X-Orienta-Resampled`.
 *
 * Whether the options show follows the geometry stamped on the RESULT the backend holds (`/result/last`),
 * not the dataset the viewer shows: the two differ after a result is imported or another one activated.
 * Everything but the two requests under test answers with an empty body.
 */
import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import i18n from '../../i18n';

const mocks = vi.hoisted(() => ({ getLastResult: vi.fn(), exportResult: vi.fn() }));

vi.mock('../../services/api', async (importOriginal) => {
  const actual = await importOriginal();
  const empty = () => Promise.resolve({ data: {} });
  const group = (overrides = {}) => new Proxy({}, {
    get: (_t, name) => (name in overrides ? overrides[name] : empty),
  });
  return {
    ...actual,
    indexApi: group({
      getLastResult: (...a) => mocks.getLastResult(...a),
      exportResult: (...a) => mocks.exportResult(...a),
    }),
    ebsdApi: group({
      datasets: () => Promise.resolve({ data: { datasets: [{ name: 'scan' }], active: 'scan' } }),
    }),
    pcApi: group(),
    edsApi: group(),
    dictionaryGpuApi: group(),
    phaseMapApi: group(),
    collectionsApi: group(),
    getGpuStatus: empty,
  };
});

vi.mock('./NavigationCanvas', () => ({ default: () => null }));

import IndexingPage from './IndexingPage';

// 3 rows x 4 columns, 2 um between points, rows 1.732 um apart: what a hexagonal result is stamped with
const HEX = { kind: 'hex', n_rows: 3, n_cols: 4, step_x: 2, step_y: 1.7320508, odd_row_shift: 0.5, shift_confirmed: true };
const SQUARE = { kind: 'square', n_rows: 3, n_cols: 4, step_x: 2, step_y: 2 };
const answer = (geometry) => ({ data: { available: true, result_id: 'r1', n_indexed: 12, scan_geometry: geometry } });

// What the backend sends (hand-written, as ASCII JSON): a 7 x 9 grid
const HEADER = '{"step_um":0.5,"rows":7,"cols":9,"source_points":60,"points":63,"unused":2,"shift_confirmed":true}';

const exportSelect = () => document.querySelector('option[value="ang"]').parentElement;
const values = () => [...exportSelect().querySelectorAll('option')].map((o) => o.value);

beforeEach(async () => {
  globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
  window.URL.createObjectURL = vi.fn(() => 'blob:x');
  window.URL.revokeObjectURL = vi.fn();
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
  mocks.getLastResult.mockReset();
  mocks.exportResult.mockReset();
  await i18n.changeLanguage('en');
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe('the export list of the Indexing page', () => {
  it('offers the square .ang and the .ctf after .ang for a result of a hexagonal scan', async () => {
    mocks.getLastResult.mockResolvedValue(answer(HEX));
    render(<IndexingPage isActive />);
    await waitFor(() => expect(values()).toEqual(['', 'h5', 'ang', 'ang_square', 'ctf']), { timeout: 4000 });
    const labels = [...exportSelect().querySelectorAll('option')].map((o) => o.textContent);
    expect(labels.slice(3)).toEqual(['.ang (square grid, resampled)', '.ctf (square grid, resampled)']);
  });

  it.each([
    ['a square result', answer(SQUARE)],
    ['a result that states no geometry', answer(null)],
    ['no result', { data: { available: false } }],
  ])('offers neither for %s', async (_name, reply) => {
    mocks.getLastResult.mockResolvedValue(reply);
    render(<IndexingPage isActive />);
    await waitFor(() => expect(mocks.getLastResult).toHaveBeenCalled(), { timeout: 4000 });
    await act(async () => {});
    expect(values()).toEqual(['', 'h5', 'ang']);
  });

  it('asks again when the page becomes active, so a result imported on another page is seen', async () => {
    mocks.getLastResult.mockResolvedValue(answer(SQUARE));
    const { rerender } = render(<IndexingPage isActive={false} />);
    rerender(<IndexingPage isActive />);
    await waitFor(() => expect(mocks.getLastResult).toHaveBeenCalledTimes(1), { timeout: 4000 });
    expect(values()).toEqual(['', 'h5', 'ang']);
    mocks.getLastResult.mockResolvedValue(answer(HEX));
    rerender(<IndexingPage isActive={false} />);
    rerender(<IndexingPage isActive />);
    await waitFor(() => expect(values()).toContain('ctf'), { timeout: 4000 });
    mocks.getLastResult.mockResolvedValue(answer(SQUARE));
    rerender(<IndexingPage isActive={false} />);
    rerender(<IndexingPage isActive />);
    await waitFor(() => expect(values()).not.toContain('ctf'), { timeout: 4000 });
  });
});

describe('whether the export list can be used', () => {
  it('is enabled for a result the backend holds that this page did not produce, and a choice reaches the export', async () => {
    mocks.getLastResult.mockResolvedValue(answer(HEX));
    mocks.exportResult.mockResolvedValue({ data: new Blob(['x']), headers: {} });
    render(<IndexingPage isActive />);
    await waitFor(() => expect(exportSelect()).toBeEnabled(), { timeout: 4000 });
    fireEvent.change(exportSelect(), { target: { value: 'ctf' } });
    await waitFor(() => expect(mocks.exportResult).toHaveBeenCalledWith('ctf', true, true), { timeout: 4000 });
  });

  it('is enabled for a square result too (the plain exports are available to it)', async () => {
    mocks.getLastResult.mockResolvedValue(answer(SQUARE));
    render(<IndexingPage isActive />);
    await waitFor(() => expect(exportSelect()).toBeEnabled(), { timeout: 4000 });
  });

  it.each([
    ['no result', { data: { available: false } }],
    ['an answer without a result', { data: {} }],
  ])('stays disabled for %s', async (_name, reply) => {
    mocks.getLastResult.mockResolvedValue(reply);
    render(<IndexingPage isActive />);
    await waitFor(() => expect(mocks.getLastResult).toHaveBeenCalled(), { timeout: 4000 });
    await act(async () => {});
    expect(exportSelect()).toBeDisabled();
  });

  it('stays disabled when the request for the result fails', async () => {
    mocks.getLastResult.mockRejectedValue(new Error('network'));
    render(<IndexingPage isActive />);
    await waitFor(() => expect(mocks.getLastResult).toHaveBeenCalled(), { timeout: 4000 });
    await act(async () => {});
    expect(exportSelect()).toBeDisabled();
  });

  it('stays disabled during a run of this page, even when the page becomes active again and the old result is still held', async () => {
    mocks.getLastResult.mockResolvedValue(answer(HEX));
    const { rerender } = render(<IndexingPage isActive />);
    await waitFor(() => expect(exportSelect()).toBeEnabled(), { timeout: 4000 });
    const start = await screen.findByRole('button', { name: 'Start Indexing' });
    await waitFor(() => expect(start).toBeEnabled(), { timeout: 4000 });
    fireEvent.click(start);
    // the run began (the answer carries no task id, so it stays "running"); the list went off with it
    await waitFor(() => expect(exportSelect()).toBeDisabled(), { timeout: 4000 });
    const asked = mocks.getLastResult.mock.calls.length;
    rerender(<IndexingPage isActive={false} />);
    rerender(<IndexingPage isActive />);
    await waitFor(() => expect(mocks.getLastResult.mock.calls.length).toBeGreaterThan(asked), { timeout: 4000 });
    await act(async () => {});
    expect(exportSelect()).toBeDisabled();
  });

  it('is disabled again when the result is gone the next time the page becomes active', async () => {
    mocks.getLastResult.mockResolvedValue(answer(HEX));
    const { rerender } = render(<IndexingPage isActive />);
    await waitFor(() => expect(exportSelect()).toBeEnabled(), { timeout: 4000 });
    mocks.getLastResult.mockResolvedValue({ data: { available: false } });
    rerender(<IndexingPage isActive={false} />);
    rerender(<IndexingPage isActive />);
    await waitFor(() => expect(exportSelect()).toBeDisabled(), { timeout: 4000 });
  });
});

describe('a resampled export', () => {
  const run = async (fmt, response) => {
    mocks.getLastResult.mockResolvedValue(answer(HEX));
    mocks.exportResult.mockResolvedValue({ data: new Blob(['x']), headers: response });
    render(<IndexingPage isActive />);
    await waitFor(() => expect(values()).toContain(fmt), { timeout: 4000 });
    await waitFor(() => expect(exportSelect()).toBeEnabled(), { timeout: 4000 });
    fireEvent.change(exportSelect(), { target: { value: fmt } });
    await waitFor(() => expect(mocks.exportResult).toHaveBeenCalledWith(fmt, true, true), { timeout: 4000 });
  };

  it('puts the grid of the header into the log, with the name of the .ctf', async () => {
    await run('ctf', { 'x-orienta-resampled': HEADER });
    await waitFor(() => expect(document.body.textContent).toContain(
      'Export complete: indexing_result.ctf. Resampled onto a square grid of 7 × 9 points; each took the nearest measured point.'),
    { timeout: 4000 });
  });

  it('names the square .ang as indexing_result_square.ang and downloads it under that name', async () => {
    const downloads = [];
    const create = document.createElement.bind(document);
    vi.spyOn(document, 'createElement').mockImplementation((tag, ...rest) => {
      const el = create(tag, ...rest);
      if (tag === 'a') Object.defineProperty(el, 'download', { set: (v) => downloads.push(v), get: () => '' });
      return el;
    });
    await run('ang_square', { 'x-orienta-resampled': HEADER });
    await waitFor(() => expect(document.body.textContent).toContain(
      'Export complete: indexing_result_square.ang. Resampled onto a square grid of 7 × 9 points'), { timeout: 4000 });
    expect(downloads).toEqual(['indexing_result_square.ang']);
    expect(document.body.textContent).not.toContain('ang_square');
  });

  it('does not invent a grid when the header did not arrive, and still says it was resampled', async () => {
    await run('ctf', {});
    await waitFor(() => expect(document.body.textContent).toContain(
      'Export complete: indexing_result.ctf. It was resampled onto a square grid; each point took the nearest measured point.'),
    { timeout: 4000 });
    expect(document.body.textContent).not.toMatch(/\d+ × \d+ points/);
  });

  it('says nothing about resampling for the plain .ang', async () => {
    await run('ang', {});
    await waitFor(() => expect(document.body.textContent).toContain('Export complete: indexing_result.ang'), { timeout: 4000 });
    expect(document.body.textContent).not.toContain('Resampled onto');
  });

  it('words the log in German', async () => {
    await i18n.changeLanguage('de');
    await run('ctf', { 'x-orienta-resampled': HEADER });
    await waitFor(() => expect(document.body.textContent).toContain(
      'Export abgeschlossen: indexing_result.ctf. Auf 7 × 9 Punkte übertragen; jeder Punkt übernahm den nächsten gemessenen.'),
    { timeout: 4000 });
  });
});
