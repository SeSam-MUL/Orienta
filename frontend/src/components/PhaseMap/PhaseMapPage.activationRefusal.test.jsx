// @vitest-environment jsdom
/**
 * The gallery activates a result by loading its source file again, and that load can be refused with
 * an APEX code: a pairing that no longer fits the pattern file, a metadata file that is gone, a
 * pairing the result lost. The toast words it in the interface's language (the texts live with the
 * viewer, errors.* of `ebsdviewer`), not in the backend's English. Measured through the PAGE with the
 * real i18n instance; only the network and the canvas are mocked, as in PhaseMapPage.hexResult.test.jsx.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, cleanup, waitFor, act } from '@testing-library/react';

const listResults = vi.fn();
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
    default: anyApi(),
    isFileLoadInFlight: () => false,
    indexApi: anyApi({
      listResults: (...a) => listResults(...a),
      activateResult: (...a) => activateResult(...a),
    }),
    phaseMapApi: anyApi({ layer: vi.fn(() => ok({ image: '', scale: null })) }),
    analysisApi: anyApi(),
    ebsdApi: anyApi({ info: vi.fn(() => ok({ file_path: 'D:/data/SampleB.h5oina' })) }),
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
import useToastStore from '../../stores/useToastStore';
import useDataStore from '../../stores/useDataStore';
import useAddonLayerRequests from '../../stores/useAddonLayerRequests';
import PhaseMapPage from './PhaseMapPage';
import viewerEn from '../../locales/en/ebsdviewer.json';
import viewerDe from '../../locales/de/ebsdviewer.json';

const PROSE = 'BACKEND ENGLISH SENTENCE';
const refusal = (code) => Object.assign(new Error('Request failed with status code 400'), {
  response: { status: 400, headers: { 'x-orienta-code': code }, data: { detail: PROSE } },
});
// a stored result that is not the active one, of the file that is loaded: the visit adopts it
const STORED = {
  id: 'spherical_123', result_id: 'spherical_123', label: 'spherical - Al', is_active: false,
  source_file: 'D:/data/SampleB.h5oina', phases: ['Al'], shape: [10, 10], n_pixels: 100, ci_mean: 0.3,
};

beforeEach(() => {
  vi.clearAllMocks();
  listResults.mockResolvedValue({ data: { results: [STORED] } });
  useAddonLayerRequests.setState({ requests: [], superseded: [], citationTick: 0 });
  useToastStore.setState({ toasts: [] });
});
afterEach(async () => {
  cleanup();
  await act(async () => { await i18n.changeLanguage('en'); });
});

const syncFromBackend = vi.fn(() => Promise.resolve());
beforeEach(() => { useDataStore.setState({ syncFromBackend }); });
const errorToasts = () => useToastStore.getState().toasts.filter((t) => t.type === 'error').map((t) => t.message);

describe.each([['en', viewerEn], ['de', viewerDe]])('in %s', (lang, viewer) => {
  it.each(['apexGuidMismatch', 'apexUnreadable', 'apexPairingLost'])(
    'a refused activation (%s) is worded by its code, not by the backend sentence', async (code) => {
      await act(async () => { await i18n.changeLanguage(lang); });
      activateResult.mockRejectedValue(refusal(code));
      render(<PhaseMapPage isActive />);
      await waitFor(() => expect(activateResult).toHaveBeenCalledWith('spherical_123'));
      await waitFor(() => expect(errorToasts().length).toBeGreaterThan(0));
      const text = errorToasts().join('\n');
      expect(text).toContain(viewer.errors[code]);
      expect(text).not.toContain(PROSE);
    });
});

describe('a result whose file came up as another variant', () => {
  const OTHER_TEXT = 'The file opened, but as another measurement or pattern source than the one this result was '
    + 'created on. The result was not activated, and the viewer now shows the other one.';

  it('says so and shows what the backend now holds: the page is synchronised with the viewer', async () => {
    activateResult.mockRejectedValue(refusal('loadCameUpAsOther'));
    render(<PhaseMapPage isActive />);
    await waitFor(() => expect(activateResult).toHaveBeenCalledWith('spherical_123'));
    await waitFor(() => expect(errorToasts().join(' ')).toContain(OTHER_TEXT));
    await waitFor(() => expect(syncFromBackend).toHaveBeenCalledTimes(1));
  });

  it('does not stop at the toast when the synchronisation itself fails', async () => {
    const stray = [];
    const onStray = (reason) => stray.push(reason);
    process.on('unhandledRejection', onStray);
    try {
      syncFromBackend.mockRejectedValueOnce(new Error('network'));
      activateResult.mockRejectedValue(refusal('loadCameUpAsOther'));
      render(<PhaseMapPage isActive />);
      await waitFor(() => expect(syncFromBackend).toHaveBeenCalledTimes(1));
      expect(errorToasts().join(' ')).toContain(OTHER_TEXT);
      await act(async () => { await new Promise((r) => setTimeout(r, 50)); });
      expect(stray).toEqual([]);   // the failed synchronisation was caught, not left to escape
    } finally {
      process.off('unhandledRejection', onStray);
    }
  });

  it('leaves the stores alone for a refusal that did not change the viewer', async () => {
    activateResult.mockRejectedValue(refusal('apexGuidMismatch'));
    render(<PhaseMapPage isActive />);
    await waitFor(() => expect(errorToasts().length).toBeGreaterThan(0));
    expect(syncFromBackend).not.toHaveBeenCalled();
  });
});
