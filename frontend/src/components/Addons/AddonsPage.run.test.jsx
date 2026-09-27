// @vitest-environment jsdom
/**
 * A run in flight: progress, the end of it, and the numbers afterwards.
 *
 * The module is mocked here rather than axios — what is under test is the
 * page's behaviour against the six endpoints, and the wire shape is pinned
 * next door in AddonsPage.params.test.jsx.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor, within }
  from '@testing-library/react';

vi.mock('../../services/addonsApi', () => ({
  addonsApi: {
    list: vi.fn(), setEnabled: vi.fn(), runJob: vi.fn(),
    job: vi.fn(), jobResult: vi.fn(), mapImage: vi.fn(),
  },
}));
vi.mock('../../services/api', () => ({
  indexApi: { listResults: vi.fn() },
  default: {},
}));

import { addonsApi } from '../../services/addonsApi';
import { indexApi } from '../../services/api';
import AddonsPage from './AddonsPage';
import useAddonLayerRequests from '../../stores/useAddonLayerRequests';

afterEach(() => { cleanup(); vi.useRealTimers(); });

const ANALYSIS = { key: 'addon.bc_gmm', label: 'Band contrast GMM',
                   params_schema: { type: 'object', properties: {
                     n_components: { type: 'integer', title: 'Components',
                                     default: 3 } } } };

const ADDON = {
  name: 'bc-gmm', display_name: 'BC mixture model', version: '0.1.0',
  authors: ['A <a@example.org>'], doi: '10.5281/zenodo.1', origin: 'directory',
  source_path: 'C:/addons/bc-gmm', enabled: true, known: true,
  known_version: '0.1.0', known_doi: '10.5281/zenodo.1',
  compatible: true, compatibility: '', conflict: '', error: '',
  crashes: 0, crash_limit: 3, disabled_by: null, analyses: [ANALYSIS],
};

const listing = (addons = [ADDON]) => ({
  data: { api_version: 0, orienta_version: '0.4.4',
          searched_paths: ['C:/addons'], addons },
});

const OUTPUTS = [
  { kind: 'scalar', key: 'n_px', label: 'Pixels used', value: 10800, unit: 'px' },
  { kind: 'table', key: 'components', label: 'Components',
    columns: ['component', 'weight'], rows: [[1, 0.5], [2, 0.5]] },
];

beforeEach(() => {
  vi.clearAllMocks();
  addonsApi.list.mockResolvedValue(listing());
  indexApi.listResults.mockResolvedValue({ data: { results: [
    { id: 'res-7', is_active: true, source_file: 'D:/data/SampleB.h5oina' }] } });
  addonsApi.runJob.mockResolvedValue({ data: { job_id: 'job-1' } });
});

async function startRun() {
  render(<AddonsPage />);
  fireEvent.click(await screen.findByRole('button', { name: /band contrast gmm/i }));
  const panel = await screen.findByTestId('addon-run-panel');
  fireEvent.click(within(panel).getByRole('button', { name: /^run$/i }));
  await waitFor(() => expect(addonsApi.runJob).toHaveBeenCalled());
  return panel;
}

describe('while it runs', () => {
  it('shows the add-on’s own progress message and fraction', async () => {
    // context.report is the only channel an add-on has; a page that polled
    // and showed nothing would make the whole seam pointless.
    addonsApi.job.mockResolvedValue({ data: { state: 'running',
                                              message: 'fitting', fraction: 0.5 } });
    const panel = await startRun();
    await waitFor(() => expect(panel.textContent).toContain('fitting'));
    expect(panel.textContent).toMatch(/50\s*%/);
  });

  it('says that reloading the page loses track of the run', async () => {
    // Known limit, stated rather than built: the job id lives in page state
    // and there is no listing endpoint to find it again.
    addonsApi.job.mockResolvedValue({ data: { state: 'running', message: '' } });
    const panel = await startRun();
    await waitFor(() => expect(panel.textContent).toMatch(/reload/i));
  });

  it('refuses a second run while one is in flight, and says why', async () => {
    // Not queued, not silently started: two runs writing into one result is
    // the case the runtime replaces a step for, and the page must not be the
    // component that pretends otherwise.
    addonsApi.job.mockResolvedValue({ data: { state: 'running', message: '' } });
    const panel = await startRun();
    const run = within(panel).getByRole('button', { name: /^run$/i });
    await waitFor(() => expect(run.disabled).toBe(true));
    expect(panel.textContent).toMatch(/already running/i);
    fireEvent.click(run);
    expect(addonsApi.runJob).toHaveBeenCalledTimes(1);
  });
});

describe('when it ends', () => {
  it('stops polling once the job is done', async () => {
    // A poll that never stops is a request every second for the rest of the
    // session, and the health check waits behind each one.
    addonsApi.job
      .mockResolvedValueOnce({ data: { state: 'running', message: 'fitting' } })
      .mockResolvedValue({ data: { state: 'done', message: 'done' } });
    addonsApi.jobResult.mockResolvedValue({ data: { outputs: OUTPUTS,
                                                    ignored_params: [] } });
    await startRun();
    await waitFor(() => expect(addonsApi.jobResult).toHaveBeenCalled());
    const callsAtEnd = addonsApi.job.mock.calls.length;
    await new Promise((r) => setTimeout(r, 1200));
    expect(addonsApi.job.mock.calls.length).toBe(callsAtEnd);
  });

  it('stops polling when the panel goes away', async () => {
    addonsApi.job.mockResolvedValue({ data: { state: 'running', message: '' } });
    await startRun();
    await waitFor(() => expect(addonsApi.job).toHaveBeenCalled());
    cleanup();
    const after = addonsApi.job.mock.calls.length;
    await new Promise((r) => setTimeout(r, 1200));
    expect(addonsApi.job.mock.calls.length).toBe(after);
  });

  it('shows the outputs, with units', async () => {
    addonsApi.job.mockResolvedValue({ data: { state: 'done' } });
    addonsApi.jobResult.mockResolvedValue({ data: { outputs: OUTPUTS,
                                                    ignored_params: [] } });
    await startRun();
    const out = await screen.findByTestId('addon-outputs');
    expect(out.textContent).toContain('Pixels used');
    expect(out.textContent).toContain('px');
    expect(within(out).getAllByRole('columnheader').map((h) => h.textContent))
      .toEqual(['component', 'weight']);
  });

  it('the first run’s numbers are gone the moment a second run starts', async () => {
    // Not "replaced when the second finishes" — that happens by itself, and a
    // test asserting only it passes even when the stale numbers are left on
    // screen for the whole of the second run. The window that matters is
    // between the click and the result: showing run 1's outputs while run 2
    // computes is the page disagreeing with the runtime about which run is
    // current, in the one place a user reads numbers off.
    addonsApi.job.mockResolvedValue({ data: { state: 'done' } });
    addonsApi.jobResult.mockResolvedValue({ data: { outputs: [
      { kind: 'scalar', key: 'n_px', label: 'Pixels used', value: 1 }],
      ignored_params: [] } });
    const panel = await startRun();
    await screen.findByTestId('addon-outputs');

    // The second run does not answer: its job stays running.
    addonsApi.job.mockResolvedValue({ data: { state: 'running', message: '' } });
    fireEvent.click(within(panel).getByRole('button', { name: /^run$/i }));
    await waitFor(() => expect(addonsApi.runJob).toHaveBeenCalledTimes(2));
    await waitFor(() =>
      expect(screen.queryByTestId('addon-outputs')).toBeNull());
  });

  it('copying a table writes TSV: header first, one line per row', async () => {
    // The numbers have to be able to LEAVE the page. They live in one HTTP
    // response and are gone on reload, while the citation text beside them
    // has had a Copy button all along — the page would hand a user the
    // sentence to cite and keep the numbers it is about.
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
    addonsApi.job.mockResolvedValue({ data: { state: 'done' } });
    addonsApi.jobResult.mockResolvedValue({ data: { outputs: OUTPUTS,
                                                    ignored_params: [] } });
    await startRun();
    await screen.findByTestId('addon-outputs');
    fireEvent.click(screen.getByRole('button', { name: /copy/i }));
    await waitFor(() => expect(writeText).toHaveBeenCalled());
    const lines = writeText.mock.calls[0][0].split(String.fromCharCode(10));
    expect(lines[0]).toBe(['component', 'weight'].join(String.fromCharCode(9)));
    expect(lines).toHaveLength(3);       // header + two rows
  });

  it('saving a table goes through Electron’s text channel, with the CSV', async () => {
    const saveText = vi.fn().mockResolvedValue('D:/out.csv');
    window.electronAPI = { saveText };
    addonsApi.job.mockResolvedValue({ data: { state: 'done' } });
    addonsApi.jobResult.mockResolvedValue({ data: { outputs: OUTPUTS,
                                                    ignored_params: [] } });
    await startRun();
    await screen.findByTestId('addon-outputs');
    fireEvent.click(screen.getByRole('button', { name: /save/i }));
    await waitFor(() => expect(saveText).toHaveBeenCalled());
    const { defaultPath, text } = saveText.mock.calls[0][0];
    expect(defaultPath).toMatch(/\.csv$/);
    // The BOM travels in the TEXT now, not prepended by the main process, so
    // a browser download and an Electron save produce the same bytes.
    expect(text.charCodeAt(0)).toBe(0xFEFF);
    expect(text.slice(1).split(String.fromCharCode(10))[0])
      .toBe('component,weight');
    delete window.electronAPI;
  });

  it('the browser download carries the same BOM as the Electron save',
     async () => {
    // The defect this pins: the BOM was prepended by electron/main.js, so the
    // same table saved from a browser build came out BOM-less — and Excel on
    // Windows reads a BOM-less UTF-8 CSV as the system code page, turning a
    // unit like µm into mojibake. Two writers, one of them updated.
    delete window.electronAPI;
    const parts = [];
    const RealBlob = global.Blob;
    global.Blob = function (bits, opts) { parts.push(bits); return new RealBlob(bits, opts); };
    global.URL.createObjectURL = vi.fn(() => 'blob:x');
    global.URL.revokeObjectURL = vi.fn();
    addonsApi.job.mockResolvedValue({ data: { state: 'done' } });
    addonsApi.jobResult.mockResolvedValue({ data: { outputs: OUTPUTS,
                                                    ignored_params: [] } });
    await startRun();
    await screen.findByTestId('addon-outputs');
    fireEvent.click(screen.getByRole('button', { name: /save/i }));
    await waitFor(() => expect(parts.length).toBe(1));
    global.Blob = RealBlob;
    expect(parts[0][0].charCodeAt(0)).toBe(0xFEFF);
  });

  it('names parameters the run discarded', async () => {
    // Reachable exactly when a manifest was edited on disk while the app is
    // open: the cached schema still offers a parameter the manifest no
    // longer declares, the run drops it, and a silent drop means the user
    // believes they set something they did not.
    addonsApi.job.mockResolvedValue({ data: { state: 'done' } });
    addonsApi.jobResult.mockResolvedValue({ data: {
      outputs: OUTPUTS, ignored_params: ['tolerance', 'seed'] } });
    await startRun();
    const out = await screen.findByTestId('addon-run-panel');
    await waitFor(() => expect(out.textContent).toContain('tolerance'));
    expect(out.textContent).toContain('seed');
  });

  it('hands a map to the Phase Maps page with result, run and label', async () => {
    // The id has to carry the result — /api/phasemap/layer takes none, so a
    // layer without it could not be fetched at all — and the run, so a second
    // run is not deduplicated into the first run's cached bitmap.
    const MAP = { kind: 'map', key: 'component_map', label: 'Component map (GMM)' };
    addonsApi.job.mockResolvedValue({ data: { state: 'done' } });
    addonsApi.jobResult.mockResolvedValue({ data: { outputs: [MAP],
                                                    ignored_params: [] } });
    useAddonLayerRequests.getState().clear();
    await startRun();
    await screen.findByTestId('addon-outputs');
    fireEvent.click(screen.getByRole('button', { name: /show as layer/i }));

    const [req] = useAddonLayerRequests.getState().requests;
    expect(req.id).toMatch(
      /^addon:bc-gmm\/res-7\/addon\.bc_gmm\/component_map#.+$/);
    // The author's label travels: built from the id it would read
    // "component_map", and that string would end up in a figure's legend.
    expect(req.label).toBe('Component map (GMM)');
  });

  it('asks the app to show the Phase Maps page, in the shape it reads', async () => {
    // App.jsx reads `e.detail.page`. A bare string passes every type check,
    // dispatches happily, and is ignored — the layer lands on the stack while
    // the user sits on this page wondering whether the click worked.
    const MAP = { kind: 'map', key: 'component_map', label: 'Component map' };
    addonsApi.job.mockResolvedValue({ data: { state: 'done' } });
    addonsApi.jobResult.mockResolvedValue({ data: { outputs: [MAP],
                                                    ignored_params: [] } });
    const seen = [];
    const listener = (e) => seen.push(e.detail);
    window.addEventListener('navigate-to', listener);
    try {
      await startRun();
      await screen.findByTestId('addon-outputs');
      fireEvent.click(screen.getByRole('button', { name: /show as layer/i }));
      expect(seen).toEqual([{ page: 'phasemap' }]);
    } finally {
      window.removeEventListener('navigate-to', listener);
    }
  });

  it('a second run hands over a DIFFERENT layer id', async () => {
    // The mechanism, not just the shape. With one token for both runs the two
    // ids are identical, the layer reducer dedupes and the bitmap cache hits,
    // so the second "show as layer" is a no-op: the user goes on reading the
    // FIRST run's numbers under a label that says otherwise. Asserted on the
    // queue holding two entries, because the queue deduplicates by id — one
    // entry means the ids were the same.
    const MAP = { kind: 'map', key: 'component_map', label: 'Component map' };
    addonsApi.runJob
      .mockResolvedValueOnce({ data: { job_id: 'job-1' } })
      .mockResolvedValueOnce({ data: { job_id: 'job-2' } });
    addonsApi.job.mockResolvedValue({ data: { state: 'done' } });
    addonsApi.jobResult.mockResolvedValue({ data: { outputs: [MAP],
                                                    ignored_params: [] } });
    useAddonLayerRequests.getState().clear();

    const panel = await startRun();
    await screen.findByTestId('addon-outputs');
    fireEvent.click(screen.getByRole('button', { name: /show as layer/i }));

    fireEvent.click(within(panel).getByRole('button', { name: /^run$/i }));
    await waitFor(() => expect(addonsApi.runJob).toHaveBeenCalledTimes(2));
    await screen.findByTestId('addon-outputs');
    fireEvent.click(screen.getByRole('button', { name: /show as layer/i }));

    const ids = useAddonLayerRequests.getState().requests.map((r) => r.id);
    expect(ids).toHaveLength(2);
    expect(ids[0]).not.toBe(ids[1]);
  });

  it('switching analyses clears the previous run’s NUMBERS, not just its form', async () => {
    // The panel used to re-label run A's outputs with analysis B's name: a
    // CSV saved as "<B>-<A's output>.csv", a layer id built from B with A's
    // run token, and "this analysis is already running" about an analysis
    // that is not. Only the parameters were being cleared.
    const TWO = [{ ...ADDON, analyses: [
      ANALYSIS,
      { key: 'addon.other', label: 'Other analysis', params_schema: { type: 'object' } },
    ] }];
    addonsApi.list.mockResolvedValue(listing(TWO));
    addonsApi.job.mockResolvedValue({ data: { state: 'done' } });
    addonsApi.jobResult.mockResolvedValue({ data: { outputs: OUTPUTS,
                                                    ignored_params: [] } });
    await startRun();
    await screen.findByTestId('addon-outputs');

    fireEvent.click(screen.getByRole('button', { name: /other analysis/i }));
    await waitFor(() =>
      expect(screen.queryByTestId('addon-outputs')).toBeNull());
    const panel = screen.getByTestId('addon-run-panel');
    expect(panel.textContent).toContain('Other analysis');
    expect(panel.textContent).not.toMatch(/already running/i);
  });

  it('a finished run marks its own earlier layers superseded', async () => {
    // The server's map store is keyed WITHOUT the run token, so a re-run
    // overwrites in place: a layer from the previous run keeps its id, label
    // and legend and, the next time its bitmap is dropped, redraws the new
    // run's numbers under the old run's name.
    const MAP = { kind: 'map', key: 'component_map', label: 'Component map' };
    addonsApi.job.mockResolvedValue({ data: { state: 'done' } });
    addonsApi.jobResult.mockResolvedValue({ data: { outputs: [MAP],
                                                    ignored_params: [] } });
    useAddonLayerRequests.setState({ superseded: [] });
    await startRun();
    await screen.findByTestId('addon-outputs');
    expect(useAddonLayerRequests.getState().superseded).toEqual([
      'addon:bc-gmm/res-7/addon.bc_gmm/component_map']);
  });

  it('names the step size and quality the run was given', async () => {
    // The backend has returned these two since the first run route and the
    // page printed neither. They belong beside the outputs because they are
    // what the outputs were computed FROM: "Band Contrast (native)" and
    // "Pattern Quality (computed)" are different measurements of the same
    // scan, and an add-on's numbers differ accordingly.
    addonsApi.job.mockResolvedValue({ data: { state: 'done' } });
    addonsApi.jobResult.mockResolvedValue({ data: {
      outputs: OUTPUTS, ignored_params: [],
      context: { step_size_um: 0.5,
                 quality_source: 'Band Contrast (native)' } } });
    await startRun();
    const ctx = await screen.findByTestId('run-context');
    expect(ctx.textContent).toContain('0.5');
    expect(ctx.textContent).toContain('Band Contrast (native)');
  });

  it('prints the step size as a person writes it', async () => {
    // Measured in the running app: a 0.2 µm scan printed
    // "0.20000000298023224 µm". The file holds the step as float32, JSON
    // widens it to double, and the raw number went into the sentence.
    addonsApi.job.mockResolvedValue({ data: { state: 'done' } });
    addonsApi.jobResult.mockResolvedValue({ data: {
      outputs: OUTPUTS, ignored_params: [],
      context: { step_size_um: 0.20000000298023224,
                 quality_source: 'Image Quality (native)' } } });
    await startRun();
    const ctx = await screen.findByTestId('run-context');
    expect(ctx.textContent).toContain('0.2');
    expect(ctx.textContent).not.toContain('0.20000000298023224');
  });

  it('says so when the result carries neither', async () => {
    // The case that matters: a null step size means every length the add-on
    // reports is in pixels, and nothing in the add-on's own output says it.
    addonsApi.job.mockResolvedValue({ data: { state: 'done' } });
    addonsApi.jobResult.mockResolvedValue({ data: {
      outputs: OUTPUTS, ignored_params: [],
      context: { step_size_um: null, quality_source: null } } });
    await startRun();
    const ctx = await screen.findByTestId('run-context');
    expect(ctx.textContent).toMatch(/pixels/i);
    expect(ctx.textContent).toMatch(/none in this result/i);
  });

  it('a failed job shows our sentence AND the add-on’s own words', async () => {
    addonsApi.job.mockResolvedValue({ data: {
      state: 'failed', reason: 'addon_failed',
      error: 'bc-gmm / addon.bc_gmm: ValueError: fewer pixels than components' } });
    await startRun();
    const alert = await screen.findByRole('alert');
    expect(alert.textContent).toContain('The analysis failed');
    expect(alert.textContent).toContain('fewer pixels than components');
    expect(addonsApi.jobResult).not.toHaveBeenCalled();
  });
});
