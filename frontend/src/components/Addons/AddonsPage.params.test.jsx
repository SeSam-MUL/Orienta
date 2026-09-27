// @vitest-environment jsdom
/**
 * Parameters, the result a run is written into, and what goes on the wire.
 *
 * axios itself is mocked here, NOT `services/addonsApi` — so the real client
 * builds the request and these tests can say what was SENT, not merely what
 * the page asked the client for. `edsRequestWire.test.jsx` exists because a
 * defect of exactly that shape survived a green suite: the backend accepted
 * three fields, the UI never sent them, and every component test passed.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor, within }
  from '@testing-library/react';

const wire = vi.hoisted(() => ({ get: {}, post: {}, pendRunJob: false }));

vi.mock('axios', () => {
  const pick = (table, url) => {
    const key = Object.keys(table).find((k) => url.startsWith(k));
    return key ? table[key] : {};
  };
  const inst = {
    get: vi.fn((url) => Promise.resolve({ data: pick(wire.get, url) })),
    post: vi.fn((url) => (wire.pendRunJob && url.includes('/run-job')
      // A run that never answers, so a second click has something to be
      // refused by.
      ? new Promise(() => {})
      : Promise.resolve({ data: pick(wire.post, url) }))),
    put: vi.fn(() => Promise.resolve({ data: {} })),
    delete: vi.fn(() => Promise.resolve({ data: {} })),
    interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } },
  };
  return { default: { create: () => inst, __inst: inst }, __inst: inst };
});

import axios from 'axios';
import AddonsPage from './AddonsPage';
import useResultStore from '../../stores/useResultStore';

afterEach(cleanup);

// THREE defaulted properties, of three kinds, and one with no default. The
// earlier fixture had a single defaulted property — the one every test edited
// — so a page that promoted every declared default into the request passed
// every assertion. The fixture was doing the implementation's work.
const SCHEMA = {
  type: 'object',
  properties: {
    n_components: { type: 'integer', title: 'Components', default: 3,
                    minimum: 1, maximum: 9 },
    tolerance: { type: 'number', title: 'Tolerance', default: 0.5 },
    mode: { type: 'string', title: 'Mode', enum: ['full', 'tied'],
            default: 'full' },
    report_dir: { type: 'string', format: 'path', title: 'Report folder' },
  },
};

const ADDON = {
  name: 'bc-gmm', display_name: 'BC mixture model', version: '0.1.0',
  authors: ['A <a@example.org>'], doi: '10.5281/zenodo.1', origin: 'directory',
  source_path: 'C:/addons/bc-gmm', enabled: true, known: true,
  compatible: true, compatibility: '', conflict: '', error: '',
  crashes: 0, crash_limit: 3, disabled_by: null,
  analyses: [{ key: 'addon.bc_gmm', label: 'Band contrast GMM',
               params_schema: SCHEMA }],
};

const RESULT = {
  id: 'res-7', method: 'spherical', n_indexed: 1200, is_active: true,
  source_file: 'D:/data/SampleB.h5oina', phases: [], mean_ci: 0.4,
};

const runJobCalls = () =>
  axios.__inst.post.mock.calls.filter((c) => c[0].includes('/run-job')).length;

const lastPost = (fragment) => {
  const call = [...axios.__inst.post.mock.calls].reverse()
    .find((c) => c[0].includes(fragment));
  return call ? call[1] : null;
};

beforeEach(() => {
  vi.clearAllMocks();
  useResultStore.setState({ indexingResult: null });
  wire.get = {
    '/api/addons': { api_version: 0, orienta_version: '0.4.4',
                     searched_paths: ['C:/addons'], addons: [ADDON] },
    '/api/indexing/results': { results: [RESULT] },
  };
  wire.post = { '/api/addons': { job_id: 'job-1' } };
  wire.pendRunJob = false;
});

async function openParams() {
  render(<AddonsPage />);
  fireEvent.click(await screen.findByRole('button', { name: /band contrast gmm/i }));
  return screen.findByTestId('addon-run-panel');
}

describe('which result a run is written into', () => {
  it('names the result and its source file', async () => {
    // The citation and the provenance are written INTO that result. A page
    // that runs against one without saying which repeats the mistake this
    // repo already has a rule against: a per-pixel panel must name its pixel.
    const panel = await openParams();
    expect(panel.textContent).toContain('res-7');
    expect(panel.textContent).toContain('D:/data/SampleB.h5oina');
  });

  it('prefers this session’s own run over the backend’s active one', async () => {
    useResultStore.setState({ indexingResult: { result_id: 'res-mine' } });
    const panel = await openParams();
    expect(panel.textContent).toContain('res-mine');
  });

  it('with no result at all, says so and refuses to run', async () => {
    wire.get['/api/indexing/results'] = { results: [] };
    const panel = await openParams();
    expect(panel.textContent).toMatch(/index/i);        // names what is missing
    expect(within(panel).getByRole('button', { name: /^run$/i }).disabled).toBe(true);
  });

  it('never activates a result in order to get one', async () => {
    // /run-job takes the id explicitly, so this page has no reason to touch
    // activation — and activation SWITCHES THE LOADED FILE, the documented
    // autoAdoptGuard hazard.
    await openParams();
    const activations = axios.__inst.post.mock.calls
      .filter((c) => c[0].includes('activate'));
    expect(activations).toEqual([]);
  });
});

describe('what goes on the wire', () => {
  it('sends exactly result_id, analysis_key and params', async () => {
    const panel = await openParams();
    fireEvent.click(within(panel).getByRole('button', { name: /^run$/i }));
    await waitFor(() => expect(lastPost('/run-job')).toBeTruthy());
    const body = lastPost('/run-job');
    expect(Object.keys(body).sort()).toEqual(
      ['analysis_key', 'params', 'result_id']);
    expect(body.result_id).toBe('res-7');
    expect(body.analysis_key).toBe('addon.bc_gmm');
  });

  it('posts to the JOB route, not the blocking one', async () => {
    // axios gives up after 300 s; a dictionary-sized analysis outlives that.
    const panel = await openParams();
    fireEvent.click(within(panel).getByRole('button', { name: /^run$/i }));
    await waitFor(() => expect(axios.__inst.post).toHaveBeenCalled());
    const urls = axios.__inst.post.mock.calls.map((c) => c[0]);
    expect(urls.some((u) => u.endsWith('/run-job'))).toBe(true);
    expect(urls.some((u) => u.endsWith('/run'))).toBe(false);
  });

  it('leaves an untouched parameter OUT of params', async () => {
    // The runner merges declared defaults into the provenance trail. Sending
    // them too would record the same fact twice, from two places that can
    // disagree — and the user's copy would win while looking like theirs.
    const panel = await openParams();
    fireEvent.click(within(panel).getByRole('button', { name: /^run$/i }));
    await waitFor(() => expect(lastPost('/run-job')).toBeTruthy());
    expect(lastPost('/run-job').params).toEqual({});
  });

  it('sends a parameter the user actually changed, as a number', async () => {
    const panel = await openParams();
    fireEvent.change(within(panel).getByLabelText('Components'),
                     { target: { value: '5' } });
    fireEvent.click(within(panel).getByRole('button', { name: /^run$/i }));
    await waitFor(() => expect(lastPost('/run-job')).toBeTruthy());
    // The WHOLE object: the defect this pins put every declared default in
    // beside the edit, and an assertion on one key cannot see that.
    expect(lastPost('/run-job').params).toEqual({ n_components: 5 });
  });

  it('editing one field does not drag the other defaults along', async () => {
    // Measured before the fix: typing a path sent
    // {"n_components":3,"report_dir":"D:/typed"} — two of those three keys
    // chosen by nobody, and recorded by the runner as chosen values.
    const panel = await openParams();
    fireEvent.change(within(panel).getByLabelText('Tolerance'),
                     { target: { value: '0.9' } });
    fireEvent.click(within(panel).getByRole('button', { name: /^run$/i }));
    await waitFor(() => expect(lastPost('/run-job')).toBeTruthy());
    expect(lastPost('/run-job').params).toEqual({ tolerance: 0.9 });
  });

  it('a value the user clears is sent as absent, and the box stays empty', async () => {
    const panel = await openParams();
    const box = within(panel).getByLabelText('Components');
    fireEvent.change(box, { target: { value: '' } });
    expect(box.value).toBe('');           // and does not snap back to 3
    fireEvent.click(within(panel).getByRole('button', { name: /^run$/i }));
    await waitFor(() => expect(lastPost('/run-job')).toBeTruthy());
    expect(lastPost('/run-job').params).toEqual({});
  });

  it('a field typed back to its own default is still the user’s choice', async () => {
    // It says the same thing as the default, but it was CHOSEN — and the
    // filter must be "did the user touch it", not "does it differ".
    const panel = await openParams();
    fireEvent.change(within(panel).getByLabelText('Components'),
                     { target: { value: '9' } });
    fireEvent.change(within(panel).getByLabelText('Components'),
                     { target: { value: '3' } });
    fireEvent.click(within(panel).getByRole('button', { name: /^run$/i }));
    await waitFor(() => expect(lastPost('/run-job')).toBeTruthy());
    expect(lastPost('/run-job').params).toEqual({ n_components: 3 });
  });
});

describe('a schema this build cannot render', () => {
  it('blocks the run and names the parameter', async () => {
    // Running anyway would use the add-on's default for a setting the user
    // was never shown — a run they did not configure, recorded as if they had.
    wire.get['/api/addons'] = {
      api_version: 0, orienta_version: '0.4.4', searched_paths: ['C:/addons'],
      addons: [{ ...ADDON, analyses: [{
        key: 'addon.bc_gmm', label: 'Band contrast GMM',
        params_schema: { type: 'object', properties: {
          weights: { type: 'array', title: 'Weights' } } },
      }] }],
    };
    const panel = await openParams();
    expect(panel.textContent).toContain('weights');
    expect(within(panel).getByRole('button', { name: /^run$/i }).disabled).toBe(true);
  });
});

describe('the panel against a listing that moved on', () => {
  it('an analysis with NO params_schema renders instead of hanging', async () => {
    // A fresh `{type:'object'}` literal per render fired SchemaForm's
    // schema-keyed effect on every render, and the page re-rendered itself
    // forever. Measured before the fix: the test worker was killed at 120 s.
    wire.get['/api/addons'] = {
      api_version: 0, orienta_version: '0.4.4', searched_paths: ['C:/addons'],
      addons: [{ ...ADDON, analyses: [{ key: 'addon.bare', label: 'Bare' }] }],
    };
    render(<AddonsPage />);
    fireEvent.click(await screen.findByRole('button', { name: /bare/i }));
    const panel = await screen.findByTestId('addon-run-panel');
    expect(within(panel).getByRole('button', { name: /^run$/i })).toBeTruthy();
  });

  it('runs the analysis that was CLICKED, not the add-on’s first', async () => {
    wire.get['/api/addons'] = {
      api_version: 0, orienta_version: '0.4.4', searched_paths: ['C:/addons'],
      addons: [{ ...ADDON, analyses: [
        { key: 'addon.first', label: 'First one', params_schema: SCHEMA },
        { key: 'addon.second', label: 'Second one', params_schema: SCHEMA }] }],
    };
    render(<AddonsPage />);
    fireEvent.click(await screen.findByRole('button', { name: /second one/i }));
    const panel = await screen.findByTestId('addon-run-panel');
    fireEvent.click(within(panel).getByRole('button', { name: /^run$/i }));
    await waitFor(() => expect(lastPost('/run-job')).toBeTruthy());
    expect(lastPost('/run-job').analysis_key).toBe('addon.second');
  });

  it('switching analyses forgets the parameters of the previous one', async () => {
    wire.get['/api/addons'] = {
      api_version: 0, orienta_version: '0.4.4', searched_paths: ['C:/addons'],
      addons: [{ ...ADDON, analyses: [
        { key: 'addon.first', label: 'First one', params_schema: SCHEMA },
        { key: 'addon.second', label: 'Second one', params_schema: SCHEMA }] }],
    };
    render(<AddonsPage />);
    fireEvent.click(await screen.findByRole('button', { name: /first one/i }));
    let panel = await screen.findByTestId('addon-run-panel');
    fireEvent.change(within(panel).getByLabelText('Components'),
                     { target: { value: '7' } });
    fireEvent.click(screen.getByRole('button', { name: /second one/i }));
    panel = await screen.findByTestId('addon-run-panel');
    fireEvent.click(within(panel).getByRole('button', { name: /^run$/i }));
    await waitFor(() => expect(lastPost('/run-job')).toBeTruthy());
    // The edit belonged to the other analysis; carrying it over would send a
    // value the user set for a different question.
    expect(lastPost('/run-job').params).toEqual({});
  });

  it('a rescan that drops a parameter stops it being sent', async () => {
    // Measured before the fix: the panel went on offering the removed field
    // and sent the value typed into it — for a parameter the manifest no
    // longer declares.
    render(<AddonsPage />);
    fireEvent.click(await screen.findByRole('button', { name: /band contrast gmm/i }));
    let panel = await screen.findByTestId('addon-run-panel');
    fireEvent.change(within(panel).getByLabelText('Components'),
                     { target: { value: '7' } });

    wire.get['/api/addons'] = {
      api_version: 0, orienta_version: '0.4.4', searched_paths: ['C:/addons'],
      addons: [{ ...ADDON, analyses: [{
        key: 'addon.bc_gmm', label: 'Band contrast GMM',
        params_schema: { type: 'object', properties: {
          tolerance: { type: 'number', title: 'Tolerance', default: 0.5 } } },
      }] }],
    };
    fireEvent.click(screen.getByRole('button', { name: /rescan/i }));
    await waitFor(() =>
      expect(within(screen.getByTestId('addon-run-panel'))
        .queryByLabelText('Components')).toBeNull());
    panel = screen.getByTestId('addon-run-panel');
    fireEvent.click(within(panel).getByRole('button', { name: /^run$/i }));
    await waitFor(() => expect(lastPost('/run-job')).toBeTruthy());
    expect(lastPost('/run-job').params).toEqual({});
  });

  it('an add-on switched off while its panel is open cannot be run', async () => {
    render(<AddonsPage />);
    fireEvent.click(await screen.findByRole('button', { name: /band contrast gmm/i }));
    await screen.findByTestId('addon-run-panel');
    wire.get['/api/addons'] = {
      api_version: 0, orienta_version: '0.4.4', searched_paths: ['C:/addons'],
      addons: [{ ...ADDON, enabled: false }],
    };
    fireEvent.click(screen.getByRole('button', { name: /rescan/i }));
    await waitFor(() => {
      const panel = screen.getByTestId('addon-run-panel');
      expect(within(panel).getByRole('button', { name: /^run$/i }).disabled).toBe(true);
    });
    expect(screen.getByTestId('addon-run-panel').textContent)
      .toMatch(/switched off/i);
  });
});

describe('two runs at once', () => {
  it('a run in flight is not restarted by toggling ANOTHER add-on', async () => {
    // One busy flag meant both "this add-on's switch is in flight" and "this
    // add-on's run is in flight". Toggling a second add-on moved the flag,
    // the first add-on's Run button re-enabled, and a second POST went out
    // with the first still unresolved: two runs writing into one result.
    wire.pendRunJob = true;
    wire.get['/api/addons'] = {
      api_version: 0, orienta_version: '0.4.4', searched_paths: ['C:/addons'],
      addons: [ADDON, { ...ADDON, name: 'other', display_name: 'Other add-on',
                        analyses: [] }],
    };
    render(<AddonsPage />);
    fireEvent.click(await screen.findByRole('button', { name: /band contrast gmm/i }));
    const panel = await screen.findByTestId('addon-run-panel');
    fireEvent.click(within(panel).getByRole('button', { name: /^run$/i }));
    await waitFor(() => expect(runJobCalls()).toBe(1));

    // Disable the other add-on: its request resolves, the page re-reads.
    fireEvent.click(screen.getAllByRole('button', { name: /disable/i })[1]);
    await waitFor(() => expect(axios.__inst.post.mock.calls
      .some((c) => c[0].includes('/other/enabled'))).toBe(true));

    fireEvent.click(within(screen.getByTestId('addon-run-panel'))
      .getByRole('button', { name: /^run$/i }));
    await new Promise((r) => setTimeout(r, 50));
    expect(runJobCalls()).toBe(1);
  });
});

describe('a path parameter', () => {
  it('uses the Electron folder picker when there is one', async () => {
    // openFolder, not openFile: `format = "path"` says only that the value is
    // a path and is kept out of the provenance trail, so the app has to
    // choose — and the picker that can CREATE a directory is the one that
    // fits a report folder. Typing a file path still works.
    const openFolder = vi.fn().mockResolvedValue('D:/reports');
    window.electronAPI = { openFolder };
    const panel = await openParams();
    fireEvent.click(within(panel).getByRole('button', { name: /browse/i }));
    await waitFor(() => expect(openFolder).toHaveBeenCalled());
    await waitFor(() =>
      expect(within(panel).getByLabelText('Report folder').value)
        .toBe('D:/reports'));
    delete window.electronAPI;
  });

  it('a cancelled picker changes nothing', async () => {
    // openFolder resolves to null on cancel, and writing that into the field
    // would send `null` as the parameter value.
    window.electronAPI = { openFolder: vi.fn().mockResolvedValue(null) };
    const panel = await openParams();
    fireEvent.change(within(panel).getByLabelText('Report folder'),
                     { target: { value: 'D:/kept' } });
    fireEvent.click(within(panel).getByRole('button', { name: /browse/i }));
    await waitFor(() =>
      expect(within(panel).getByLabelText('Report folder').value).toBe('D:/kept'));
    delete window.electronAPI;
  });

  it('is still typeable in a browser, where there is no picker', async () => {
    // The app runs in a plain browser in development. A field that only works
    // in Electron is a field that does not work.
    delete window.electronAPI;
    const panel = await openParams();
    const field = within(panel).getByLabelText('Report folder');
    fireEvent.change(field, { target: { value: 'D:/typed' } });
    expect(field.value).toBe('D:/typed');
    fireEvent.click(within(panel).getByRole('button', { name: /^run$/i }));
    await waitFor(() => expect(lastPost('/run-job')).toBeTruthy());
    expect(lastPost('/run-job').params.report_dir).toBe('D:/typed');
  });
});
