// @vitest-environment jsdom
//
// The Analysis page offers the file open in the EBSD Viewer (a scan just
// opened in the viewer met "No indexed data loaded" here and had to be browsed for a second time).
// Offered, not loaded on its own: after an indexing run the map to analyse may be the result.
// Real page, only the HTTP layer mocked.
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';

vi.mock('../../../services/api', () => ({
  analysisApi: {
    status: vi.fn(),
    load: vi.fn(),
    getMap: vi.fn(),
    qualityStats: vi.fn(),
  },
  indexApi: { getLastResult: vi.fn() },
  edsApi: {},
}));

import { analysisApi, indexApi } from '../../../services/api';
import useResultStore from '../../../stores/useResultStore';
import useDataStore from '../../../stores/useDataStore';
import AnalysisPage, { isCrystalMapPath } from '../AnalysisPage';

const VIEWER_FILE = String.raw`C:\data\SampleB area 1.h5oina`;
// label and file name sit on two lines; textContent joins them
const OFFER = /^Use the file open in the EBSD ViewerSampleB area 1\.h5oina$/;

/** GET /api/analysis/status as routes/analysis.py writes it (`loaded`, and the file name as `source_name`). */
const statusLoaded = (sourceName) => ({
  data: {
    loaded: true, shape: [90, 120], has_grains: false, phases: ['Aluminium (Fm-3m)'],
    phase_options: [{ id: 1, name: 'Aluminium' }], step_size: 0.5, source_name: sourceName,
    dataset_token: 'a1', reconstruct: null,
  },
});

function setUpBackend({ loadedName = null } = {}) {
  indexApi.getLastResult.mockRejectedValue(new Error('none'));
  analysisApi.qualityStats.mockRejectedValue(new Error('none'));
  analysisApi.getMap.mockImplementation(() => new Promise(() => {}));
  analysisApi.status.mockResolvedValue(loadedName ? statusLoaded(loadedName) : { data: { loaded: false } });
  // POST /api/analysis/load answers `success`, not `loaded` (a mock that answers
  // `loaded: true` passes while the real page keeps offering the file it had just loaded).
  analysisApi.load.mockImplementation((p) => Promise.resolve({
    data: { success: true, shape: [90, 120], phases: ['Aluminium (Fm-3m)'], phase_options: [{ id: 1, name: 'Aluminium' }],
            source_name: p.split(/[\\/]/).pop(), step_size: 0.5, has_native_bc: true },
  }));
}

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
  useResultStore.getState().clearResults();
  useDataStore.setState({ filePath: null });
  try { sessionStorage.clear(); } catch { /* ignore */ }
});

const offerButton = () => Array.from(document.querySelectorAll('button')).find((b) => OFFER.test(b.textContent));
const pathInput = () => screen.getByPlaceholderText('/path/to/result.h5');
const page = (isActive = true) => <AnalysisPage isActive={isActive} onNavigate={() => {}} />;

describe('the file open in the EBSD Viewer', () => {
  it('is offered and prefilled while no map is loaded here; one click loads exactly it', async () => {
    setUpBackend();
    useDataStore.setState({ filePath: VIEWER_FILE });
    render(page());
    await waitFor(() => expect(offerButton()).toBeTruthy());
    expect(pathInput().value).toBe(VIEWER_FILE);
    expect(analysisApi.load).not.toHaveBeenCalled();          // offered, not loaded on its own

    fireEvent.click(offerButton());
    await waitFor(() => expect(analysisApi.load).toHaveBeenCalledWith(VIEWER_FILE));
    await waitFor(() => expect(offerButton()).toBeUndefined());
  }, 15000);

  it('is not offered when the map here is that file already (page entered again)', async () => {
    setUpBackend({ loadedName: 'SampleB area 1.h5oina' });
    useDataStore.setState({ filePath: VIEWER_FILE });
    render(page());
    await waitFor(() => expect(useResultStore.getState().analysisLoaded).toBe(true));
    expect(offerButton()).toBeUndefined();
  }, 15000);

  it('is offered again when the map was replaced elsewhere while this tab stayed open', async () => {
    setUpBackend();
    useDataStore.setState({ filePath: VIEWER_FILE });
    const view = render(page());
    await waitFor(() => expect(offerButton()).toBeTruthy());
    fireEvent.click(offerButton());
    await waitFor(() => expect(offerButton()).toBeUndefined());

    // a Batch "Open in Analysis" or a load on another page swapped the map; the page is entered again
    analysisApi.status.mockResolvedValue(statusLoaded('7050 batch map.ctf'));
    view.rerender(page(false));
    view.rerender(page(true));
    await waitFor(() => expect(offerButton()).toBeTruthy());
  }, 15000);

  it('is offered while the map here has no file name (an indexing result in memory)', async () => {
    setUpBackend({ loadedName: null });
    analysisApi.status.mockResolvedValue(statusLoaded(null));
    useDataStore.setState({ filePath: VIEWER_FILE });
    render(page());
    await waitFor(() => expect(useResultStore.getState().analysisLoaded).toBe(true));
    await waitFor(() => expect(offerButton()).toBeTruthy());
  }, 15000);

  it('compares file names without case on Windows paths only', async () => {
    setUpBackend({ loadedName: 'sampleb AREA 1.H5OINA' });
    useDataStore.setState({ filePath: VIEWER_FILE });
    const view = render(page());
    await waitFor(() => expect(useResultStore.getState().analysisLoaded).toBe(true));
    expect(offerButton()).toBeUndefined();
    view.unmount();

    analysisApi.status.mockClear();
    analysisApi.status.mockResolvedValue(statusLoaded('map.ang'));
    useDataStore.setState({ filePath: '/data/Map.ang' });
    render(page());
    // before /status answers no file name is known and the offer shows anyway: wait for the answer
    await waitFor(() => expect(analysisApi.status).toHaveBeenCalled());
    await new Promise((r) => setTimeout(r, 100));
    expect(Array.from(document.querySelectorAll('button'))
      .some((b) => /EBSD ViewerMap\.ang$/.test(b.textContent))).toBe(true);
  }, 15000);

  it('is offered when the map here is another file', async () => {
    setUpBackend({ loadedName: 'other.ang' });
    useDataStore.setState({ filePath: VIEWER_FILE });
    render(page());
    await waitFor(() => expect(useResultStore.getState().analysisLoaded).toBe(true));
    await waitFor(() => expect(offerButton()).toBeTruthy());
  }, 15000);

  it('is offered again when the viewer opens another file than the one loaded here', async () => {
    setUpBackend();
    useDataStore.setState({ filePath: VIEWER_FILE });
    render(page());
    await waitFor(() => expect(offerButton()).toBeTruthy());
    fireEvent.click(offerButton());
    await waitFor(() => expect(offerButton()).toBeUndefined());

    useDataStore.setState({ filePath: String.raw`C:\data\other.ctf` });
    await waitFor(() => expect(Array.from(document.querySelectorAll('button'))
      .some((b) => /EBSD Viewerother\.ctf$/.test(b.textContent))).toBe(true));
  }, 15000);

  it('tells a file of the same name in another folder apart from the one loaded here', async () => {
    setUpBackend();
    useDataStore.setState({ filePath: VIEWER_FILE });
    render(page());
    await waitFor(() => expect(offerButton()).toBeTruthy());
    fireEvent.click(offerButton());
    await waitFor(() => expect(offerButton()).toBeUndefined());

    useDataStore.setState({ filePath: String.raw`D:\copy\SampleB area 1.h5oina` });
    await waitFor(() => expect(offerButton()).toBeTruthy());
  }, 15000);

  it('does not overwrite a path the user typed', async () => {
    setUpBackend();
    render(page());
    fireEvent.change(pathInput(), { target: { value: String.raw`D:\mine.ang` } });
    useDataStore.setState({ filePath: VIEWER_FILE });
    await waitFor(() => expect(offerButton()).toBeTruthy());
    expect(pathInput().value).toBe(String.raw`D:\mine.ang`);
  }, 15000);

  it('is neither offered nor prefilled when it is no CrystalMap (pattern file of another kind, none)', async () => {
    setUpBackend();
    useDataStore.setState({ filePath: String.raw`C:\data\patterns.up1` });
    render(page());
    await waitFor(() => expect(analysisApi.status).toHaveBeenCalled());
    await new Promise((r) => setTimeout(r, 50));
    expect(Array.from(document.querySelectorAll('button')).some((b) => /EBSD Viewer/.test(b.textContent))).toBe(false);
    expect(pathInput().value).toBe('');
  }, 15000);
});

describe('isCrystalMapPath', () => {
  it('takes what the Browse filter offers, in any case, and nothing else', () => {
    for (const p of ['a.h5oina', 'a.H5', 'a.hdf5', 'x/y.ang', String.raw`C:\z.CTF`, ' a.ang ']) expect(isCrystalMapPath(p)).toBe(true);
    for (const p of ['a.up1', 'a.h5oina.phase_map.npz', '', null, undefined, 3]) expect(isCrystalMapPath(p)).toBe(false);
  });
});
