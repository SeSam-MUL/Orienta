// @vitest-environment jsdom
/**
 * A gallery chip of another variant of the open file: the backend loads that variant, the path stays
 * the same. The stores and the header used to stay on the old one, because everything that re-read
 * them compared the path. Real page, real stores, real i18n; only the network and the canvas are mocked.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, cleanup, waitFor, act } from '@testing-library/react';

const listResults = vi.fn();
const activateResult = vi.fn();
const loadedFiles = vi.fn();
const info = vi.fn();
const loading = vi.hoisted(() => ({ inFlight: false }));

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
    isFileLoadInFlight: () => loading.inFlight,
    indexApi: anyApi({
      listResults: (...a) => listResults(...a),
      activateResult: (...a) => activateResult(...a),
    }),
    phaseMapApi: anyApi({ layer: vi.fn(() => ok({ image: '', scale: null })) }),
    analysisApi: anyApi(),
    ebsdApi: anyApi({
      info: (...a) => info(...a),
      loadedFiles: (...a) => loadedFiles(...a),
      getMetadata: vi.fn(() => ok({})),
    }),
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

import i18n from '../../i18n';
import useDataStore from '../../stores/useDataStore';
import useLoadedFilesStore from '../../stores/useLoadedFilesStore';
import useAddonLayerRequests from '../../stores/useAddonLayerRequests';
import PhaseMapPage from './PhaseMapPage';

const PATH = 'D:/data/SampleB.h5oina';
// a stored result that is not active: the visit adopts it when that loads nothing, and an activation
// through the gallery or "Save as" runs the same code with a load answer
const STORED = {
  id: 'hough_2', result_id: 'hough_2', is_active: false, method: 'hough', source_file: PATH,
  activation_loads: false,
  phases: [{ id: 0, name: 'Al' }], mean_ci: 0.346, n_indexed: 100, shape: [10, 10],
};
const LOADED = {
  file_path: PATH, grid_shape: [5, 20], scan_geometry: null,
  has_electron_images: true, electron_images: ['SE/Image 1'],
};
const SHOWN = [{ path: PATH, name: 'SampleB · unprocessed', active: true }];

beforeEach(() => {
  vi.clearAllMocks();
  loading.inFlight = false;
  useAddonLayerRequests.setState({ requests: [], superseded: [], citationTick: 0 });
  listResults.mockResolvedValue({ data: { results: [STORED] } });
  info.mockResolvedValue({ data: { loaded: true, file_path: PATH, data_shape: [5, 20, 128, 156] } });
  loadedFiles.mockResolvedValue({ data: { files: SHOWN } });
  useLoadedFilesStore.setState({ files: [] });
  useDataStore.setState({
    isFileOpen: true, filePath: PATH, gridShape: [10, 10], ebsdInfo: { grid_shape: [10, 10] }, fileChangedElsewhere: 0,
  });
});
afterEach(async () => {
  cleanup();
  await act(async () => { await i18n.changeLanguage('en'); });
});

describe('activating a result of another variant of the open file', () => {
  it('brings the store, the header list and the viewer to what the backend now holds', async () => {
    // the header asks when it mounts; only a read after the activation can show the new variant
    let activated = false;
    loadedFiles.mockImplementation(async () => ({ data: { files: activated ? SHOWN : [] } }));
    activateResult.mockImplementation(async () => {
      activated = true;
      return { data: { status: 'ok', reloaded: LOADED } };
    });
    render(<PhaseMapPage isActive />);
    await waitFor(() => expect(useDataStore.getState().fileChangedElsewhere).toBe(1));
    const store = useDataStore.getState();
    expect(store.gridShape).toEqual([5, 20]);                     // the path did not change, the grid did
    expect(store.ebsdInfo).toEqual(LOADED);                       // the viewer's copy of the load answer
    expect(store.electronImages).toEqual(['SE/Image 1']);         // /metadata does not list them
    expect(store.hasElectronImages).toBe(true);
    expect(useLoadedFilesStore.getState().files).toEqual(SHOWN);  // what the header reads
  });

  it('a page visit does not load it: the variant the user just chose stays', async () => {
    listResults.mockResolvedValue({ data: { results: [{ ...STORED, activation_loads: true }] } });
    render(<PhaseMapPage isActive />);
    await waitFor(() => expect(listResults).toHaveBeenCalled());
    await act(async () => { await new Promise((r) => setTimeout(r, 30)); });
    expect(activateResult).not.toHaveBeenCalled();
  });

  it('a page visit does not replace an open file that has no signal', async () => {
    info.mockResolvedValue({ data: { loaded: false, open_file: 'D:/data/eds-only.h5oina' } });
    listResults.mockResolvedValue({ data: { results: [{ ...STORED, activation_loads: true }] } });
    render(<PhaseMapPage isActive />);
    await waitFor(() => expect(info).toHaveBeenCalled());
    await act(async () => { await new Promise((r) => setTimeout(r, 30)); });
    expect(activateResult).not.toHaveBeenCalled();
  });

  it('a page visit with nothing open adopts the result: nothing to lose', async () => {
    info.mockResolvedValue({ data: { loaded: false } });
    activateResult.mockResolvedValue({ data: { status: 'ok', reloaded: null } });
    listResults.mockResolvedValue({ data: { results: [{ ...STORED, activation_loads: true }] } });
    render(<PhaseMapPage isActive />);
    await waitFor(() => expect(activateResult).toHaveBeenCalledWith('hough_2'));
  });

  it.each([
    ['the backend reports no signal', () => info.mockResolvedValue({ data: { loaded: false } })],
    ['the re-read fails', () => info.mockRejectedValue(new Error('network'))],
  ])('the load answer still counts when %s', async (_name, arrange) => {
    arrange();
    // the visit's own question about what is loaded has to pass, only the sync sees the answer above
    const visit = { data: { loaded: true, file_path: PATH } };
    info.mockResolvedValueOnce(visit).mockResolvedValueOnce(visit);
    activateResult.mockResolvedValue({ data: { status: 'ok', reloaded: LOADED } });
    render(<PhaseMapPage isActive />);
    await waitFor(() => expect(useDataStore.getState().fileChangedElsewhere).toBe(1));
    expect(useDataStore.getState().gridShape).toEqual([5, 20]);
    expect(useDataStore.getState().filePath).toBe(PATH);
  });

  it('leaves the stores to a load in flight: it sets them itself, and the backend ends on it', async () => {
    loading.inFlight = true;
    activateResult.mockResolvedValue({ data: { status: 'ok', reloaded: LOADED } });
    render(<PhaseMapPage isActive />);
    await waitFor(() => expect(activateResult).toHaveBeenCalledWith('hough_2'));
    await act(async () => { await new Promise((r) => setTimeout(r, 30)); });
    const store = useDataStore.getState();
    expect(store.fileChangedElsewhere).toBe(0);
    expect(store.gridShape).toEqual([10, 10]);
    expect(store.ebsdInfo).toEqual({ grid_shape: [10, 10] });
  });

  it('a re-read without a load answer keeps the electron images of the open file', async () => {
    useDataStore.setState({ electronImages: ['SE/Image 1'], hasElectronImages: true });
    await useDataStore.getState().syncFromBackend({ force: true });
    expect(useDataStore.getState().gridShape).toEqual([5, 20]);
    expect(useDataStore.getState().electronImages).toEqual(['SE/Image 1']);
    expect(useDataStore.getState().hasElectronImages).toBe(true);
  });

  it('leaves everything alone when nothing was loaded', async () => {
    activateResult.mockResolvedValue({ data: { status: 'ok', reloaded: null } });
    render(<PhaseMapPage isActive />);
    await waitFor(() => expect(activateResult).toHaveBeenCalledWith('hough_2'));
    await act(async () => { await new Promise((r) => setTimeout(r, 30)); });
    expect(useDataStore.getState().fileChangedElsewhere).toBe(0);
    expect(useDataStore.getState().gridShape).toEqual([10, 10]);
  });
});

describe('the chip of a result', () => {
  const chip = async (result, lang) => {
    await act(async () => { await i18n.changeLanguage(lang); });
    listResults.mockResolvedValue({ data: { results: [result] } });
    activateResult.mockResolvedValue({ data: { status: 'ok', reloaded: null } });
    const view = render(<PhaseMapPage isActive />);
    await waitFor(() => expect(view.container.textContent).toContain('CI: 0.346'));
    return view.container.textContent;
  };

  it.each([['en', 'unprocessed'], ['de', 'roh'], ['ja', '未処理'], ['zh', '未处理']])(
    'says which patterns it ran on, in %s', async (lang, word) => {
      const text = await chip({ ...STORED, variant_parts: ['2', 'unprocessed'] }, lang);
      expect(text).toContain(`hough — 2 · ${word} — Al — CI: 0.346`);
    });

  it('says nothing for the default variant', async () => {
    const text = await chip({ ...STORED, variant_parts: [] }, 'en');
    expect(text).toContain('hough — Al — CI: 0.346');
  });
});
