// @vitest-environment jsdom
/**
 * "Save as…" on the Phase Map page, for a result of a hexagonal scan: the format picker lists the square
 * .ang and the .ctf after the .ang (and only then), and a resampled save says what it did, with the grid of
 * the header `X-Orienta-Resampled`. Real page, real stores, real i18n; only the network and the canvas are
 * mocked (as in PhaseMapPage.hexResult.test.jsx).
 */
import '@testing-library/jest-dom/vitest';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, waitFor, fireEvent, within, act } from '@testing-library/react';
import i18n from '../../i18n';
import useToastStore from '../../stores/useToastStore';

const toasts = () => useToastStore.getState().toasts.map((x) => x.message);

const listResults = vi.fn();
const post = vi.fn();
const activateResult = vi.fn();

vi.mock('../../services/api', () => {
  const ok = (data) => Promise.resolve({ data });
  const anyApi = (overrides = {}) => new Proxy({ ...overrides }, {
    get: (target, prop) => {
      if (prop in target) return target[prop];
      if (typeof prop !== 'string') return undefined;
      target[prop] = vi.fn(() => ok({}));
      return target[prop];
    },
  });
  return {
    default: anyApi({ post: (...a) => post(...a) }),
    indexApi: anyApi({
      listResults: (...a) => listResults(...a),
      activateResult: (...a) => activateResult(...a),
    }),
    phaseMapApi: anyApi({ layer: vi.fn(() => ok({ image: '', scale: null })) }),
    analysisApi: anyApi(),
    ebsdApi: anyApi(),
    h5Api: anyApi(),
    pcApi: anyApi(),
    forwardDiagApi: anyApi({ summary: vi.fn(() => Promise.reject(new Error('404'))) }),
    refinementApi: anyApi({ summary: vi.fn(() => Promise.reject(new Error('404'))) }),
    citationsApi: anyApi({ forResult: vi.fn(() => ok({ steps: [], bibtex: '' })) }),
  };
});
vi.mock('../../services/addonsApi', () => ({
  addonsApi: { mapImage: vi.fn(() => Promise.resolve({ data: { image: '', scale: null } })) },
}));
vi.mock('./LayeredCanvas', () => ({ default: () => <div data-testid="canvas" /> }));
vi.mock('../EDS/LinescanProfilePlot', () => ({ default: () => null }));
vi.mock('../EDS/MagnifierLens', () => ({ default: () => null }));
vi.mock('../PatternMatch/LinkedPatternImage', () => ({ default: () => null }));
vi.mock('../PoleFigure/openPoleFigureWindow', () => ({ openPoleFigureWindow: vi.fn() }));
vi.mock('./AnomalyBrowser/AnomalyBrowserDrawer', () => ({ default: () => null }));

import PhaseMapPage from './PhaseMapPage';

const RESULT_ID = 'spherical_123';
const SQUARE = { kind: 'square', n_rows: 10, n_cols: 10, step_x: 0.5, step_y: 0.5 };
const HEX = { kind: 'hex', n_rows: 10, n_cols: 10, step_x: 0.5, step_y: 0.5, odd_row_shift: 0.5, shift_confirmed: true };
const entry = (scan_geometry) => ({
  id: RESULT_ID, result_id: RESULT_ID, method: 'spherical', is_active: true,
  source_file: 'scan.h5oina', phases: ['Al'], shape: [10, 10], n_indexed: 100, mean_ci: 0.3,
  ...(scan_geometry ? { scan_geometry } : {}),
});
const showing = (scan_geometry) => listResults.mockResolvedValue({ data: { results: [entry(scan_geometry)] } });

// what the backend sends for a 6 x 11 square grid (hand-written ASCII JSON)
const HEADER = '{"step_um":0.5,"rows":6,"cols":11,"source_points":60,"points":66,"unused":0,"shift_confirmed":true}';

const saveFile = vi.fn();

async function openPicker() {
  render(<PhaseMapPage isActive />);
  await waitFor(() => expect(listResults).toHaveBeenCalled());
  const save = await screen.findByRole('button', { name: 'Save as…' });
  await waitFor(() => expect(save).toBeEnabled(), { timeout: 4000 });
  fireEvent.click(save);
  return screen.findByRole('dialog', { name: 'Pick save format' });
}
const cards = (dialog) => within(dialog).getAllByRole('button')
  .map((b) => b.querySelector('span')?.textContent).filter(Boolean);

beforeEach(async () => {
  vi.clearAllMocks();
  saveFile.mockResolvedValue('D:/out/result.ctf');
  window.electronAPI = { saveFile };
  useToastStore.setState({ toasts: [] });
  activateResult.mockResolvedValue({ data: {} });
  post.mockResolvedValue({ data: new Blob(['x']), headers: { 'x-orienta-resampled': HEADER } });
  await i18n.changeLanguage('en');
});
afterEach(() => { cleanup(); delete window.electronAPI; });

describe('the format picker of Save as…', () => {
  it('lists the square .ang and the .ctf after the .ang for a hexagonal result', async () => {
    showing(HEX);
    const dialog = await openPicker();
    expect(cards(dialog)).toEqual(['MTEX .ang', '.ang (square grid, resampled)', '.ctf (square grid, resampled)',
      'Light .h5', 'Rich .h5']);
  });

  it.each([
    ['a square result', SQUARE],
    ['a result that states no geometry', null],
  ])('lists the three formats of before for %s', async (_name, geometry) => {
    showing(geometry);
    const dialog = await openPicker();
    expect(cards(dialog)).toEqual(['MTEX .ang', 'Light .h5', 'Rich .h5']);
  });
});

describe('a result added from an Orienta .h5 file', () => {
  // import-h5 is called only here; its answer carries the geometry the file states
  const importFile = async (scanGeometry) => {
    listResults.mockResolvedValue({ data: { results: [] } });
    window.electronAPI = { saveFile, openFile: vi.fn(async () => 'D:/in/result_light.h5') };
    post.mockImplementation(async (url) => (url === '/api/indexing/import-h5'
      ? { data: { result_id: 'imported_1', label: 'imported result', phases: ['Al'], shape: [10, 10],
        capabilities: { indexing: true }, ebsd_signal_loaded: false, scan_geometry: scanGeometry } }
      : { data: new Blob(['x']), headers: {} }));
    render(<PhaseMapPage isActive />);
    await waitFor(() => expect(listResults).toHaveBeenCalled());
    fireEvent.click(await screen.findByRole('button', { name: 'Add file…' }));
    const save = await screen.findByRole('button', { name: 'Save as…' });
    await waitFor(() => expect(save).toBeEnabled(), { timeout: 4000 });
    fireEvent.click(save);
    return screen.findByRole('dialog', { name: 'Pick save format' });
  };

  it('offers the resampled formats when the file states a hexagonal grid', async () => {
    expect(cards(await importFile(HEX))).toEqual(['MTEX .ang', '.ang (square grid, resampled)',
      '.ctf (square grid, resampled)', 'Light .h5', 'Rich .h5']);
  });

  it.each([['a square grid', SQUARE], ['no geometry', null]])('does not when the file states %s', async (_n, geometry) => {
    expect(cards(await importFile(geometry))).toEqual(['MTEX .ang', 'Light .h5', 'Rich .h5']);
  });
});

describe('saving a hexagonal result resampled', () => {
  const pick = async (title) => {
    showing(HEX);
    const dialog = await openPicker();
    fireEvent.click(within(dialog).getByText(title).closest('button'));
  };

  it('asks for a .ctf, suggests the .ctf name, and says the grid it was written on', async () => {
    await pick('.ctf (square grid, resampled)');
    await waitFor(() => expect(post).toHaveBeenCalled(), { timeout: 4000 });
    expect(saveFile.mock.calls[0][0].defaultPath).toBe('spherical_Al_CI_0.300_100_px.ctf');
    expect(saveFile.mock.calls[0][0].filters[0].extensions).toEqual(['ctf']);
    expect(post.mock.calls[0][0]).toBe('/api/indexing/export');
    expect(post.mock.calls[0][1]).toMatchObject({ format: 'ctf', filename: 'D:/out/result.ctf' });
    await waitFor(() => expect(toasts().join('|')).toContain(
      'Saved "spherical — Al — CI: 0.300 — 100 px" → D:/out/result.ctf. Resampled onto a square grid of 6 × 11 points; each took the nearest measured point.'),
    { timeout: 4000 });
  });

  it('asks for the square .ang under a name that cannot be taken for the hexagonal one', async () => {
    saveFile.mockResolvedValue('D:/out/result_square.ang');
    await pick('.ang (square grid, resampled)');
    await waitFor(() => expect(post).toHaveBeenCalled(), { timeout: 4000 });
    expect(saveFile.mock.calls[0][0].defaultPath).toBe('spherical_Al_CI_0.300_100_px_square.ang');
    expect(post.mock.calls[0][1]).toMatchObject({ format: 'ang_square', filename: 'D:/out/result_square.ang' });
    await waitFor(() => expect(toasts().join('|')).toContain('Resampled onto a square grid of 6 × 11 points'),
      { timeout: 4000 });
  });

  it('does not invent a grid when the header did not arrive', async () => {
    post.mockResolvedValue({ data: new Blob(['x']), headers: {} });
    await pick('.ctf (square grid, resampled)');
    await waitFor(() => expect(toasts().join('|')).toContain('Saved "spherical — Al — CI: 0.300 — 100 px" → D:/out/result.ctf'),
      { timeout: 4000 });
    expect(toasts().join('|')).toContain('It was resampled onto a square grid');
    expect(toasts().join('|')).not.toMatch(/\d+ × \d+ points/);
  });

  it('says nothing about resampling when the .ang is saved as measured', async () => {
    saveFile.mockResolvedValue('D:/out/result.ang');
    post.mockResolvedValue({ data: new Blob(['x']), headers: {} });
    await pick('MTEX .ang');
    await waitFor(() => expect(post).toHaveBeenCalled(), { timeout: 4000 });
    expect(post.mock.calls[0][1]).toMatchObject({ format: 'ang', filename: 'D:/out/result.ang' });
    await act(async () => {});
    expect(toasts()).toEqual([]);
  });

  it('raises a toast with the translated text when the backend refuses the .ctf (the info line is hidden under a map)', async () => {
    // a refusal as axios hands it over for a blob request: status 400, the code in the header, a Blob body
    post.mockRejectedValue(Object.assign(new Error('Request failed with status code 400'), {
      response: {
        status: 400,
        headers: { 'x-orienta-code': 'export_needs_hex_scan' },
        data: new Blob(['{"detail":"export needs a hexagonal scan"}']),
      },
    }));
    await pick('.ctf (square grid, resampled)');
    await waitFor(() => expect(toasts().join('|')).toContain(
      'Save failed: A .ctf file and a square .ang file hold a hexagonal scan resampled to a square grid. '
      + 'This result was already resampled to a square grid, or is not from a hexagonal scan, so there is nothing '
      + 'to resample. Nothing was written.'),
    { timeout: 4000 });
    expect(toasts().join('|')).not.toContain('export needs a hexagonal scan');
  });

  it('words the saved line in German', async () => {
    await i18n.changeLanguage('de');
    showing(HEX);
    render(<PhaseMapPage isActive />);
    await waitFor(() => expect(listResults).toHaveBeenCalled());
    const save = await screen.findByRole('button', { name: /Speichern unter/ });
    await waitFor(() => expect(save).toBeEnabled(), { timeout: 4000 });
    fireEvent.click(save);
    const dialog = await screen.findByRole('dialog');
    fireEvent.click(within(dialog).getByText('.ctf (quadratisches Raster, umgerechnet)').closest('button'));
    await waitFor(() => expect(toasts().join('|')).toContain(
      '→ D:/out/result.ctf. Auf 6 × 11 Punkte übertragen; jeder Punkt übernahm den nächsten gemessenen.'),
    { timeout: 4000 });
  });
});
