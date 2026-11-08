// @vitest-environment jsdom
/**
 * The Phase Map page, mounted, when the result on screen changes between a hexagonal and a square scan (or
 * between two hexagonal ones): a picture belongs to the geometry it was fetched for.
 *
 *  - while the new answer is still on its way the page must not say that a layer is "not on the hexagonal
 *    grid" (the expectation of the OLD result would be applied to the NEW picture);
 *  - a picture that really is not on the raster of a hexagonal result is refused by name (the same
 *    machinery, so the test can fail);
 *  - the pictures are made by the test: their size travels in the "image", and the sizes of the rasters are
 *    those of the independent oracle fixture (tests/fixtures/hex_raster_cases.json).
 */
/* global process */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, waitFor, fireEvent, act } from '@testing-library/react';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const listResults = vi.fn();
// what the Phase Map page's layer requests are answered with, set per test
let answerLayer = () => Promise.resolve({ data: { image: '3x3', scale: null } });
// every set of per-layer messages the canvas was ever rendered with
const seenErrors = [];
// every /render answers with another image, so every refresh of the preview asks available-maps again
let previewNo = 0;
// false: every preview is the same image, so nothing but the page being shown again makes it ask again
let freshPreviews = true;

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
    default: anyApi(),
    indexApi: anyApi({ listResults: (...a) => listResults(...a) }),
    // /render must answer with an image, or the page shows its error instead of the canvas
    phaseMapApi: anyApi({
      layer: vi.fn((...a) => answerLayer(...a)),
      render: vi.fn(() => ok({ image: `preview${freshPreviews ? previewNo += 1 : 0}` })),
    }),
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
  addonsApi: { mapImage: vi.fn(() => Promise.resolve({ data: { image: '3x3', scale: null } })) },
}));
vi.mock('./LayeredCanvas', () => ({
  default: ({ bitmaps, perLayerErrors, bitmapVersion }) => {
    const errors = [...perLayerErrors.values()];
    seenErrors.push(...errors);
    const sizes = [...bitmaps.values()].map((b) => `${b.height}x${b.width}`).join(',');
    return (
      <div data-testid="canvas" data-version={bitmapVersion} data-sizes={sizes} data-errors={errors.join('|')} />
    );
  },
}));
vi.mock('../EDS/LinescanProfilePlot', () => ({ default: () => null }));
vi.mock('../EDS/MagnifierLens', () => ({ default: () => null }));
vi.mock('../PatternMatch/LinkedPatternImage', () => ({ default: () => null }));
vi.mock('../PoleFigure/openPoleFigureWindow', () => ({ openPoleFigureWindow: vi.fn() }));
vi.mock('./AnomalyBrowser/AnomalyBrowserDrawer', () => ({ default: () => null }));

// A "PNG" here is the text "<height>x<width>": fetch(data:) hands it back as the blob, createImageBitmap reads it.
vi.stubGlobal('fetch', vi.fn((url) => {
  const m = /^data:image\/png;base64,(.*)$/.exec(String(url));
  return m ? Promise.resolve({ blob: () => Promise.resolve(m[1]) }) : Promise.reject(new Error(`fetch ${url}`));
}));
vi.stubGlobal('createImageBitmap', vi.fn((blob) => {
  const [h, w] = String(blob).split('x').map(Number);
  return Promise.resolve({ height: h, width: w, close: () => {} });
}));

import PhaseMapPage from './PhaseMapPage';
import { phaseMapApi, indexApi } from '../../services/api';
import useAddonLayerRequests from '../../stores/useAddonLayerRequests';

const FIXTURE = JSON.parse(
  readFileSync(resolve(process.cwd(), '..', 'tests', 'fixtures', 'hex_raster_cases.json'), 'utf8'));
const hexCase = (name) => FIXTURE.geometries.find((c) => c.name === name);
const A = hexCase('a_3x3_right');
const B = hexCase('b_4x5_left');
const rasterOf = (c) => `${c.spec.height}x${c.spec.width}`;
const SQUARE = { kind: 'square', n_rows: 3, n_cols: 3, step_x: 2, step_y: 2 };

const SENTENCE = 'This layer is not drawn on the hexagonal map grid';
const entry = (id, scan_geometry) => ({
  id, result_id: id, label: `result ${id}`, is_active: true, source_file: 'a.h5oina',
  phases: ['Al'], shape: [3, 3], n_pixels: 9, ci_mean: 0.3, scan_geometry,
});
const show = (id, geometry) => listResults.mockResolvedValue({ data: { results: [entry(id, geometry)] } });
const canvas = () => screen.getByTestId('canvas');
const sizes = () => canvas().getAttribute('data-sizes');
const messages = () => canvas().getAttribute('data-errors');
const chip = () => screen.getByRole('tab', { name: /Indexing/ });
const wait = (ms = 30) => new Promise((r) => setTimeout(r, ms));
/** An answer the test releases itself. */
function held(image) {
  let release;
  const promise = new Promise((resolve) => { release = () => resolve({ data: { image, scale: null } }); });
  return { promise, release };
}

async function mountWith(id, geometry, image) {
  show(id, geometry);
  answerLayer = () => Promise.resolve({ data: { image, scale: null } });
  render(<PhaseMapPage isActive />);
  await waitFor(() => expect(screen.getByTestId('canvas')).toBeTruthy());
  // the answer has landed: it is either on the canvas or refused on its row
  await waitFor(() => expect(sizes() !== '' || messages() !== '').toBe(true));
}
async function switchTo(id, geometry) {
  const before = listResults.mock.calls.length;
  show(id, geometry);
  fireEvent.click(chip());
  await waitFor(() => expect(listResults.mock.calls.length).toBeGreaterThan(before));
}

beforeEach(() => {
  vi.clearAllMocks();
  phaseMapApi.availableMaps.mockReset();               // back to "no dataset geometry"
  previewNo = 0;
  freshPreviews = true;
  seenErrors.length = 0;
  useAddonLayerRequests.setState({ requests: [], superseded: [], citationTick: 0 });
});
afterEach(cleanup);

describe('the oracle rasters the cases below rest on', () => {
  it('are not the index grid and differ from one another, so a mix-up can be seen', () => {
    expect(rasterOf(A)).toBe('12x14');
    expect(rasterOf(B)).not.toBe(rasterOf(A));
    expect(rasterOf(A)).not.toBe('3x3');
  });
});

describe('a hexagonal result whose picture is not on the raster', () => {
  it('is refused by name on the layer (so the checks below can fail)', async () => {
    await mountWith('r1', A.geometry, '3x3');            // the index grid, on a hexagonal result
    await waitFor(() => expect(messages()).toContain(SENTENCE));
    expect(sizes()).toBe('');
  });

  it('is shown when the picture is on the raster', async () => {
    await mountWith('r1', A.geometry, rasterOf(A));
    expect(sizes()).toBe(rasterOf(A));
    expect(messages()).toBe('');
  });

  // what the backend reports as `display`, from the oracle's raster of the case
  const displayOf = (c, dx = 0) => ({
    kind: 'hex_raster', factor: c.spec.factor, shape: [c.spec.height, c.spec.width],
    step_um: c.spec.step_um, x0: c.spec.x0 + dx, y0: c.spec.y0,
  });

  it('is shown when the answer says it drew exactly the raster the page holds', async () => {
    show('r1', A.geometry);
    answerLayer = () => Promise.resolve({ data: { image: rasterOf(A), display: displayOf(A) } });
    render(<PhaseMapPage isActive />);
    await waitFor(() => expect(sizes()).toBe(rasterOf(A)));
    expect(messages()).toBe('');
  });

  it('is refused when the picture has the size of the raster but the answer says it starts elsewhere on the sample', async () => {
    show('r1', A.geometry);
    answerLayer = () => Promise.resolve({
      data: { image: rasterOf(A), display: displayOf(A, A.spec.step_um * 2) } });
    render(<PhaseMapPage isActive />);
    await waitFor(() => expect(messages()).toContain(SENTENCE));
    expect(sizes()).toBe('');
  });
});

describe('a change of result never shows the old expectation against the new picture', () => {
  it('hexagonal -> square: no message while the new picture is on its way, and the picture is kept', async () => {
    await mountWith('r1', A.geometry, rasterOf(A));
    const pending = held('3x3');
    answerLayer = () => pending.promise;
    await switchTo('r2', SQUARE);
    await wait();
    expect(sizes()).toBe('');                            // the old picture is gone, the new one not here yet
    expect(messages()).toBe('');
    pending.release();
    await waitFor(() => expect(sizes()).toBe('3x3'));
    expect(messages()).toBe('');
    expect(seenErrors.filter((m) => m.includes(SENTENCE))).toEqual([]);
  });

  it('hexagonal -> another hexagonal raster: the new raster is the one expected', async () => {
    await mountWith('r1', A.geometry, rasterOf(A));
    const pending = held(rasterOf(B));
    answerLayer = () => pending.promise;
    await switchTo('r2', B.geometry);
    await wait();
    expect(messages()).toBe('');
    pending.release();
    await waitFor(() => expect(sizes()).toBe(rasterOf(B)));
    expect(messages()).toBe('');
    expect(seenErrors.filter((m) => m.includes(SENTENCE))).toEqual([]);
  });

  it('square -> hexagonal: the raster is expected from the new result on', async () => {
    await mountWith('r1', SQUARE, '3x3');
    const pending = held(rasterOf(A));
    answerLayer = () => pending.promise;
    await switchTo('r2', A.geometry);
    await wait();
    expect(messages()).toBe('');
    pending.release();
    await waitFor(() => expect(sizes()).toBe(rasterOf(A)));
    expect(messages()).toBe('');
    expect(seenErrors.filter((m) => m.includes(SENTENCE))).toEqual([]);
  });

  it('an answer for the result that was shown before does not replace the new picture', async () => {
    await mountWith('r1', A.geometry, rasterOf(A));
    const late = held(rasterOf(A));                      // asked for r1 ...
    answerLayer = () => late.promise;
    fireEvent.click(screen.getAllByRole('button').find((b) => /CI/.test(b.textContent)) ?? chip());
    await wait();
    const fresh = held('3x3');
    answerLayer = () => fresh.promise;
    await switchTo('r2', SQUARE);                        // ... and then r2 is shown
    await wait();
    late.release();                                      // r1's answer lands first
    await wait();
    expect(sizes()).not.toContain(rasterOf(A));
    fresh.release();
    await waitFor(() => expect(sizes()).toBe('3x3'));
    expect(seenErrors.filter((m) => m.includes(SENTENCE))).toEqual([]);
    expect(phaseMapApi.layer).toHaveBeenCalled();
  });
});

describe('a result without a stamp is drawn with the geometry of the dataset (available-maps)', () => {
  const mapsAnswer = (geometry) => ({ data: { maps: [], dataset_scan_geometry: geometry } });

  it('expects the raster of a hexagonal dataset, and refuses an index-grid picture by name', async () => {
    phaseMapApi.availableMaps.mockResolvedValue(mapsAnswer(A.geometry));
    await mountWith('r1', undefined, '3x3');
    await waitFor(() => expect(messages()).toContain(SENTENCE));
  });

  it('expects nothing of a square dataset', async () => {
    phaseMapApi.availableMaps.mockResolvedValue(mapsAnswer(SQUARE));
    await mountWith('r1', undefined, '3x3');
    await wait(60);
    expect(sizes()).toBe('3x3');
    expect(messages()).toBe('');
  });

  it('is not changed by an answer to an earlier question that arrives late', async () => {
    const early = held('unused');
    phaseMapApi.availableMaps.mockReset();
    // Every question asked before the first preview is in (whoever asks it) is held and, when it lands,
    // says "hexagonal"; the later questions say "square". The page asks again when the preview changes.
    phaseMapApi.availableMaps.mockImplementation(() => (previewNo === 0
      ? early.promise.then(() => mapsAnswer(A.geometry))
      : Promise.resolve(mapsAnswer(SQUARE))));
    await mountWith('r1', undefined, '3x3');
    await wait(60);                                      // the later answers are in, the first one is not
    expect(previewNo).toBeGreaterThan(0);
    expect(sizes()).toBe('3x3');
    early.release();                                     // now the answer of the first question lands
    await wait(60);
    expect(messages()).toBe('');
    expect(sizes()).toBe('3x3');
    expect(seenErrors.filter((m) => m.includes(SENTENCE))).toEqual([]);
  });

  it('follows a dataset another page changed while this one stayed mounted', async () => {
    freshPreviews = false;
    phaseMapApi.availableMaps.mockResolvedValue(mapsAnswer(A.geometry));   // a hexagonal dataset ...
    show('r1', undefined);
    answerLayer = () => Promise.resolve({ data: { image: rasterOf(A), scale: null } });
    const utils = render(<PhaseMapPage isActive />);
    await waitFor(() => expect(sizes()).toBe(rasterOf(A)));
    utils.rerender(<PhaseMapPage isActive={false} />);   // the user goes to another page ...
    phaseMapApi.availableMaps.mockResolvedValue(mapsAnswer(SQUARE));       // ... which loads a square dataset
    answerLayer = () => Promise.resolve({ data: { image: '3x3', scale: null } });
    utils.rerender(<PhaseMapPage isActive />);           // and comes back
    await waitFor(() => expect(sizes()).toBe('3x3'));
    expect(messages()).toBe('');
    expect(seenErrors.filter((m) => m.includes(SENTENCE))).toEqual([]);
  });

  it('says nothing about a picture while the geometry is being read again on showing the page', async () => {
    freshPreviews = false;
    phaseMapApi.availableMaps.mockResolvedValue(mapsAnswer(A.geometry));   // a hexagonal dataset ...
    show('r1', undefined);
    answerLayer = () => Promise.resolve({ data: { image: rasterOf(A), scale: null } });
    const utils = render(<PhaseMapPage isActive />);
    await waitFor(() => expect(sizes()).toBe(rasterOf(A)));
    utils.rerender(<PhaseMapPage isActive={false} />);   // the user goes to another page ...
    // ... which loads a square dataset; the page's own answer for it is slow
    const slow = held('unused');
    phaseMapApi.availableMaps.mockImplementation(() => slow.promise.then(() => mapsAnswer(SQUARE)));
    answerLayer = () => Promise.resolve({ data: { image: '3x3', scale: null } });
    seenErrors.length = 0;
    utils.rerender(<PhaseMapPage isActive />);           // and comes back
    await wait(30);
    // a layer is asked for in the window (an add-on map handed over): its picture is on the index grid of the
    // new, square dataset, which the geometry still held does not accept
    act(() => { useAddonLayerRequests.getState().request('addon:bc-gmm/r1/addon.bc_gmm/component_map#run-1', 'Add-on map'); });
    await wait(100);                                     // the picture has landed, the geometry has not
    expect(messages()).toBe('');                         // no verdict against the old geometry
    expect(seenErrors).toEqual([]);
    slow.release();
    await waitFor(() => expect(sizes()).toContain('3x3'));  // the pictures are fetched again and drawn
    expect(messages()).toBe('');
    expect(seenErrors).toEqual([]);
  });
});

const answerWith = (geometry) => ({ data: { maps: [], dataset_scan_geometry: geometry } });
// the raster of the oracle case as the backend reports it for the picture it drew
const displayOfCase = (c) => ({
  kind: 'hex_raster', factor: c.spec.factor, shape: [c.spec.height, c.spec.width],
  step_um: c.spec.step_um, x0: c.spec.x0, y0: c.spec.y0,
});

describe('the page asks for the dataset geometry when it is shown, and not when it is left', () => {
  it('on a square scan: leaving asks nothing, coming back asks once', async () => {
    freshPreviews = false;
    phaseMapApi.availableMaps.mockResolvedValue(answerWith(SQUARE));
    show('r1', SQUARE);
    answerLayer = () => Promise.resolve({ data: { image: '3x3', scale: null } });
    const utils = render(<PhaseMapPage isActive />);
    await waitFor(() => expect(sizes()).toBe('3x3'));
    await wait(80);
    const atMount = phaseMapApi.availableMaps.mock.calls.length;
    expect(atMount).toBeGreaterThan(0);
    utils.rerender(<PhaseMapPage isActive={false} />);   // the user leaves for another page: no question
    await wait(80);
    expect(phaseMapApi.availableMaps.mock.calls.length - atMount).toBe(0);
    utils.rerender(<PhaseMapPage isActive />);           // and comes back: exactly one
    await wait(80);
    expect(phaseMapApi.availableMaps.mock.calls.length - atMount).toBe(1);
  });

  it('an answer still on its way when the page is left is not thrown away', async () => {
    freshPreviews = false;
    const late = held('unused');
    phaseMapApi.availableMaps.mockImplementation(() => late.promise.then(() => answerWith(A.geometry)));
    show('r1', undefined);
    answerLayer = () => Promise.resolve({ data: { image: '3x3', scale: null } });
    const utils = render(<PhaseMapPage isActive />);
    await waitFor(() => expect(sizes()).toBe('3x3'));    // the index grid, drawn while nothing is known
    const asked = phaseMapApi.availableMaps.mock.calls.length;
    utils.rerender(<PhaseMapPage isActive={false} />);   // left while the question is open
    late.release();
    await wait(80);
    expect(phaseMapApi.availableMaps.mock.calls.length).toBe(asked);   // leaving asked nothing more
    // the answer said "hexagonal", so the index-grid picture is now refused on its row
    await waitFor(() => expect(messages()).toContain(SENTENCE));
  });
});

describe('a picture that names a raster where the page expects none', () => {
  const NOT_EXPECTED = 'drawn on a hexagonal map grid that does not match this scan';
  const hostEl = () => document.querySelector('[data-phasemap-canvas-host]');
  const HOST = { left: 0, top: 0, width: 140, height: 120, right: 140, bottom: 120, x: 0, y: 0, toJSON: () => {} };

  it('is not drawn while the geometry is unknown, and neither a hover nor a click reads it', async () => {
    const early = held('unused');
    phaseMapApi.availableMaps.mockImplementation(() => early.promise.then(() => answerWith(A.geometry)));
    show('r1', undefined);                               // no stamp: the geometry is still being asked for
    answerLayer = () => Promise.resolve({ data: { image: rasterOf(A), display: displayOfCase(A) } });
    render(<PhaseMapPage isActive />);
    await waitFor(() => expect(screen.getByTestId('canvas')).toBeTruthy());
    await wait(80);
    expect(phaseMapApi.layer).toHaveBeenCalled();         // the picture has been asked for and has landed ...
    expect(sizes()).toBe('');                             // ... and is held back, with no message
    expect(messages()).toBe('');
    const host = hostEl();
    host.getBoundingClientRect = () => HOST;
    fireEvent.mouseMove(host, { clientX: 70, clientY: 60 });
    fireEvent.click(host, { clientX: 70, clientY: 60 });
    await wait(200);                                      // the probe waits 60 ms before it asks
    expect(phaseMapApi.probe).not.toHaveBeenCalled();
    expect(indexApi.nccHeatmap).not.toHaveBeenCalled();   // a click would open the pattern match dialog
  });

  it('is drawn once the answer says the scan is hexagonal and the raster is the one drawn', async () => {
    const early = held('unused');
    phaseMapApi.availableMaps.mockImplementation(() => early.promise.then(() => answerWith(A.geometry)));
    show('r1', undefined);
    answerLayer = () => Promise.resolve({ data: { image: rasterOf(A), display: displayOfCase(A) } });
    render(<PhaseMapPage isActive />);
    await wait(80);
    expect(sizes()).toBe('');
    early.release();
    await waitFor(() => expect(sizes()).toBe(rasterOf(A)));
    expect(messages()).toBe('');
  });

  it('is refused by name on its layer once the answer says the scan is square', async () => {
    const early = held('unused');
    phaseMapApi.availableMaps.mockImplementation(() => early.promise.then(() => answerWith(SQUARE)));
    show('r1', undefined);
    answerLayer = () => Promise.resolve({ data: { image: '3x3', display: displayOfCase(A) } });
    render(<PhaseMapPage isActive />);
    await wait(80);
    expect(messages()).toBe('');                          // while the geometry is open: no verdict
    early.release();
    await waitFor(() => expect(messages()).toContain(NOT_EXPECTED));
    expect(sizes()).toBe('');
  });

  it('is refused by name at once on a result stamped as square', async () => {
    await mountWith('r1', SQUARE, '3x3');                 // a picture without a raster is drawn (control)
    expect(sizes()).toBe('3x3');
    cleanup();
    show('r2', SQUARE);
    answerLayer = () => Promise.resolve({ data: { image: '3x3', display: displayOfCase(A) } });
    render(<PhaseMapPage isActive />);
    await waitFor(() => expect(messages()).toContain(NOT_EXPECTED));
    expect(sizes()).toBe('');
  });
});
