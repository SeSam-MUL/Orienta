// @vitest-environment jsdom
//
// The two grain-size settings on the Analysis page (edge-grain rule, ECD basis) and the rule
// that grain maps wait for the grains. Real components, only the HTTP layer mocked: the
// selects reach the request, a changed select recomputes the statistics, what the rule did is
// written in the statistics block, and the map viewer does not ask for a grain map before
// grains exist.
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';

vi.mock('../../../services/api', () => ({
  // the Grain tab's citation panel asks for the grains' citations; left pending, it shows its loading line
  citationsApi: { forAnalysis: vi.fn(() => new Promise(() => {})) },
  analysisApi: {
    reconstructGrains: vi.fn(),
    getTaskStatus: vi.fn(),
    getMap: vi.fn(),
    grainSize: vi.fn(),
    batch: vi.fn(),
  },
  indexApi: {},
  edsApi: {},
}));

import { analysisApi } from '../../../services/api';
import useResultStore from '../../../stores/useResultStore';
import { colors } from '../../../theme/components';
import { BatchTab } from '../AnalysisPage';
import { GrainTabInPage } from './grainTabHarness';

const POLL_TIMEOUT = { timeout: 5000 };

const EDGE_FIELDS = {
  edge_rule: 'matlab_script', ecd_basis: 'cell_area',
  n_grains_total: 98, n_edge_grains: 34, n_grains_used: 64, edge_safeguard_applied: false,
};

const completedWith = (extra) => ({
  data: {
    status: 'completed', progress: 1.0, grain_count: 98,
    result: {
      n_grains: 98, grain_count: 98, alpha: 3.1, fill: true, n_filled: 4,
      mean_ecd: 12.34, median_ecd: 11.0, std_ecd: 3.21, mean_aspect_ratio: 1.678,
      ...EDGE_FIELDS, ...extra,
    },
  },
});

function resetMocks() {
  cleanup();
  vi.resetAllMocks();
  useResultStore.getState().clearResults();
  analysisApi.getMap.mockImplementation(() => new Promise(() => {}));   // maps never answer
}

/** The safeguard notice, among the other status elements of the page. */
const notice = () => screen.queryAllByRole('status').find((el) => /grains kept:/.test(el.textContent));

async function reconstructWith(extra, waitForText = /Total grains:\s+98/) {
  analysisApi.reconstructGrains.mockResolvedValue({ data: { task_id: 't1' } });
  analysisApi.getTaskStatus.mockResolvedValue(completedWith(extra));
  const view = render(<GrainTabInPage />);
  fireEvent.click(screen.getByRole('button', { name: /reconstruct grains/i }));
  await waitFor(() => expect(screen.getByText(waitForText)).toBeTruthy(), POLL_TIMEOUT);
  return view;
}

const mapCalls = (type) => analysisApi.getMap.mock.calls.filter((c) => c[0] === type);

describe('GrainTab: edge rule and ECD basis', () => {
  afterEach(resetMocks);

  it('offers the MATLAB-script rule and the cell area by default, and says what the chosen rule does', () => {
    analysisApi.getMap.mockImplementation(() => new Promise(() => {}));
    render(<GrainTabInPage />);
    expect(screen.getByLabelText('Edge grains').value).toBe('matlab_script');
    expect(screen.getByLabelText('ECD from').value).toBe('cell_area');
    // the sentence is visible without hovering, and is also the select's title
    expect(screen.getByText(/in the outermost two pixel rows or columns of the measured area/)).toBeTruthy();
    expect(screen.getByLabelText('Edge grains').title).toMatch(/in the outermost two pixel rows or columns of the measured area/);
    expect(screen.getByText(/ECD from the area of the grain's measurement cells/)).toBeTruthy();
    fireEvent.change(screen.getByLabelText('Edge grains'), { target: { value: 'outer_row' } });
    expect(screen.getByText(/first or last row or column/)).toBeTruthy();
    expect(screen.queryByText(/in the outermost two pixel rows or columns of the measured area/)).toBeNull();
  });

  it('the chosen values reach reconstructGrains', async () => {
    analysisApi.reconstructGrains.mockResolvedValue({ data: { task_id: 't1' } });
    analysisApi.getTaskStatus.mockResolvedValue(completedWith({}));
    render(<GrainTabInPage />);
    fireEvent.change(screen.getByLabelText('Edge grains'), { target: { value: 'mtex' } });
    fireEvent.change(screen.getByLabelText('ECD from'), { target: { value: 'measured_pixels' } });
    fireEvent.click(screen.getByRole('button', { name: /reconstruct grains/i }));
    await waitFor(() => expect(analysisApi.reconstructGrains).toHaveBeenCalledTimes(1));
    expect(analysisApi.reconstructGrains).toHaveBeenCalledWith(4, 3, 1.5, 3.1, true, 'mtex', 'measured_pixels', null);
  });

  it('the statistics name the rule and what it did', async () => {
    await reconstructWith({});
    expect(screen.getByText(/Edge grains \(MATLAB-script rule\): 34 of 98 left out(?! —)/)).toBeTruthy();
    expect(screen.getByText(/Grains in the statistics:\s+64/)).toBeTruthy();
    expect(screen.getByText(/ECD from:\s+cell area/)).toBeTruthy();
    expect(notice()).toBeUndefined();
  });

  it('keep_all says all grains are kept', async () => {
    await reconstructWith({ edge_rule: 'keep_all', n_edge_grains: 0, n_grains_used: 98 });
    expect(screen.getByText(/Edge grains: all kept/)).toBeTruthy();
  });

  it('the safeguard is a visible notice, not a count of left-out grains', async () => {
    await reconstructWith({ n_grains_total: 76, n_edge_grains: 38, n_grains_used: 76, edge_safeguard_applied: true });
    expect(notice().textContent).toBe(
      'All 76 grains kept: the MATLAB-script rule would leave out 38 edge grains, and the 38 left would be fewer than 50, so the rule keeps all grains.');
    expect(screen.queryByText(/left out$/m)).toBeNull();
    // the number that is averaged is a line of its own, also under the safeguard
    expect(screen.getByText(/Grains in the statistics:\s+76/)).toBeTruthy();
    // the box names the rule also when it left nothing out
    expect(screen.getByText(/Edge grains \(MATLAB-script rule\): none left out \(the rule's limit, see below\)/)).toBeTruthy();
    // an information, not a warning: the neutral colour of the page and not the orange one
    const probe = (color) => { const el = document.createElement('div'); el.style.color = color; return el.style.color; };
    expect(notice().style.color).toBe(probe(colors.cyan));
    expect(notice().style.color).not.toBe(probe(colors.orange));
    expect(notice().getAttribute('role')).toBe('status');
  });

  it('the safeguard of the outermost-row rule is worded as a share', async () => {
    await reconstructWith({
      edge_rule: 'outer_row', n_grains_total: 20, n_edge_grains: 19, n_grains_used: 20, edge_safeguard_applied: true,
    });
    expect(notice().textContent).toBe(
      'All 20 grains kept: the outermost-row rule would leave out 19 of 20 grains, more than 85 %, so it keeps all grains.');
  });

  it('the safeguard names the 15 % condition when 50 or more would be left but under 15 % of the total', async () => {
    await reconstructWith({ n_grains_total: 1000, n_edge_grains: 900, n_grains_used: 1000, edge_safeguard_applied: true }, /Total grains:\s+98/);
    expect(notice().textContent).toBe(
      'All 1000 grains kept: the MATLAB-script rule would leave out 900 edge grains, and the 100 left would be under 15 % of 1000, so the rule keeps all grains.');
  });

  it('keep_all has a grains-in-the-statistics line too', async () => {
    await reconstructWith({ edge_rule: 'keep_all', n_edge_grains: 0, n_grains_used: 98 });
    expect(screen.getByText(/Grains in the statistics:\s+98/)).toBeTruthy();
  });

  it('a failed statistics step: the error line, and no line that suggests numbers exist', async () => {
    // the real shape: the route sets edge_rule and ecd_basis before the statistics run, the counts are absent
    await reconstructWith({
      grain_stats_error: 'no measured pixels',
      mean_ecd: undefined, median_ecd: undefined, std_ecd: undefined, mean_aspect_ratio: undefined,
      n_grains_total: undefined, n_edge_grains: undefined, n_grains_used: undefined,
      edge_safeguard_applied: undefined,
    });
    expect(screen.getByRole('alert').textContent).toBe('Grain-size statistics failed: no measured pixels');
    expect(screen.queryByText(/Edge grains \(/)).toBeNull();
    expect(screen.queryByText(/ECD from:\s/)).toBeNull();
    expect(screen.queryByText(/Mean ECD/)).toBeNull();
  });

  it('a map with no grains: the count 0 and no edge or ECD-basis line', async () => {
    await reconstructWith({
      grain_count: 0, n_grains: 0, mean_ecd: undefined, median_ecd: undefined, std_ecd: undefined,
      mean_aspect_ratio: undefined, n_grains_total: 0, n_edge_grains: 0, n_grains_used: 0,
    }, /Total grains:\s+0\b/);
    expect(screen.queryByText(/Edge grains \(/)).toBeNull();
    expect(screen.queryByText(/ECD from:\s/)).toBeNull();
    expect(screen.queryByText(/Mean ECD/)).toBeNull();
  });

  it('the median line comes back when the answer carries one', async () => {
    await reconstructWith({});
    analysisApi.grainSize.mockResolvedValue(sizeAnswer({ ecd_basis: 'measured_pixels', median_ecd: 9.5 }));
    fireEvent.change(screen.getByLabelText('ECD from'), { target: { value: 'measured_pixels' } });
    await waitFor(() => expect(screen.getByText(/Median ECD:\s+9\.50/)).toBeTruthy());
  });

  it('the help line is linked to its select, the notice is a status, the block is a live region', async () => {
    await reconstructWith({ n_grains_total: 76, n_edge_grains: 38, n_grains_used: 76, edge_safeguard_applied: true });
    const select = screen.getByLabelText('Edge grains');
    const help = document.getElementById(select.getAttribute('aria-describedby'));
    expect(help.textContent).toBe(select.title);
    expect(notice()).toBeTruthy();
    const live = document.querySelector('[aria-live="polite"]');
    expect(live.textContent).toMatch(/Total grains/);
    expect(live.contains(notice())).toBe(true);
  });

  it('picking another option with grains present recomputes the statistics at once', async () => {
    await reconstructWith({});
    analysisApi.grainSize.mockResolvedValue({
      data: {
        success: true, mean_ecd: 20.5, median_ecd: 7.25, std_ecd: 5.5, mean_aspect_ratio: 1.25,
        ...EDGE_FIELDS, ecd_basis: 'measured_pixels',
      },
    });
    fireEvent.change(screen.getByLabelText('ECD from'), { target: { value: 'measured_pixels' } });
    await waitFor(() => expect(analysisApi.grainSize).toHaveBeenCalledTimes(1));
    expect(analysisApi.grainSize).toHaveBeenCalledWith('matlab_script', 'measured_pixels', null);
    await waitFor(() => expect(screen.getByText(/Mean ECD:\s+20\.50/)).toBeTruthy());
    expect(screen.getByText(/ECD from:\s+measured pixels × step²/)).toBeTruthy();
    expect(screen.getByText(/Total grains:\s+98/)).toBeTruthy();          // the count stays
    expect(screen.getByText(/Median ECD:\s+7\.25/)).toBeTruthy();           // the median of the new answer

    fireEvent.change(screen.getByLabelText('Edge grains'), { target: { value: 'mtex' } });
    await waitFor(() => expect(analysisApi.grainSize).toHaveBeenCalledTimes(2));
    expect(analysisApi.grainSize).toHaveBeenLastCalledWith('mtex', 'measured_pixels', null);
  });

  it('the citation panel asks again once the statistics follow another setting', async () => {
    const { citationsApi } = await import('../../../services/api');
    await reconstructWith({});
    await waitFor(() => expect(citationsApi.forAnalysis).toHaveBeenCalledTimes(1));
    analysisApi.grainSize.mockResolvedValue({
      data: { success: true, mean_ecd: 20.5, std_ecd: 5.5, mean_aspect_ratio: 1.25, ...EDGE_FIELDS, ecd_basis: 'measured_pixels' },
    });
    fireEvent.change(screen.getByLabelText('ECD from'), { target: { value: 'measured_pixels' } });
    await waitFor(() => expect(citationsApi.forAnalysis).toHaveBeenCalledTimes(2));
    analysisApi.grainSize.mockResolvedValue({
      data: { success: true, mean_ecd: 20.5, std_ecd: 5.5, mean_aspect_ratio: 1.25, ...EDGE_FIELDS,
        ecd_basis: 'measured_pixels', edge_rule: 'keep_all', n_edge_grains: 0, n_grains_used: 98 },
    });
    fireEvent.change(screen.getByLabelText('Edge grains'), { target: { value: 'keep_all' } });
    await waitFor(() => expect(citationsApi.forAnalysis).toHaveBeenCalledTimes(3));
  });

  it('without grains, picking an option only changes the setting', () => {
    render(<GrainTabInPage />);
    fireEvent.change(screen.getByLabelText('Edge grains'), { target: { value: 'keep_all' } });
    expect(analysisApi.grainSize).not.toHaveBeenCalled();
    expect(screen.getByLabelText('Edge grains').value).toBe('keep_all');
  });

  it('the "Analyze Grain Size" button sends the current values', async () => {
    await reconstructWith({});
    analysisApi.grainSize.mockResolvedValue({ data: { success: true, mean_ecd: 1, std_ecd: 1, mean_aspect_ratio: 1 } });
    fireEvent.change(screen.getByLabelText('Edge grains'), { target: { value: 'outer_row' } });
    await waitFor(() => expect(analysisApi.grainSize).toHaveBeenCalledTimes(1));
    analysisApi.grainSize.mockClear();
    fireEvent.click(screen.getByRole('button', { name: /analyze grain size/i }));
    await waitFor(() => expect(analysisApi.grainSize).toHaveBeenCalledWith('outer_row', 'cell_area', null));
  });

  it('a failed recomputation is shown, not swallowed', async () => {
    await reconstructWith({});
    analysisApi.grainSize.mockRejectedValue({ response: { status: 400, data: { detail: 'Reconstruct grains first' } } });
    fireEvent.change(screen.getByLabelText('ECD from'), { target: { value: 'measured_pixels' } });
    await waitFor(() => expect(screen.getByText('Reconstruct grains first')).toBeTruthy());
  });
});

describe('BatchTab: edge rule and ECD basis', () => {
  afterEach(resetMocks);

  it('sends both settings with the batch request', async () => {
    analysisApi.batch.mockResolvedValue({ data: { task_id: 'b1' } });
    analysisApi.getTaskStatus.mockResolvedValue({ data: { status: 'running', progress: 0.1 } });
    render(<BatchTab analysisLoaded onComplete={() => {}} />);
    expect(screen.getByLabelText('Edge grains').value).toBe('matlab_script');
    expect(screen.getByLabelText('ECD from').value).toBe('cell_area');
    fireEvent.change(screen.getByLabelText('Edge grains'), { target: { value: 'outer_row' } });
    fireEvent.change(screen.getByLabelText('ECD from'), { target: { value: 'measured_pixels' } });
    fireEvent.change(screen.getByPlaceholderText('/path/to/dataset/folder'), { target: { value: 'F' } });
    fireEvent.change(screen.getByPlaceholderText('/path/to/Documentation.xlsx'), { target: { value: 'O.xlsx' } });
    fireEvent.click(screen.getByRole('button', { name: /process folder/i }));
    await waitFor(() => expect(analysisApi.batch).toHaveBeenCalledTimes(1));
    expect(analysisApi.batch).toHaveBeenCalledWith('F', 'O.xlsx', 'outer_row', 'measured_pixels', null, { threshold: 4, minPixels: 3, minIntercept: 1.5, alpha: 3.1, fill: true },
      { gosThreshold: 1.25, gbcFraction: 0.7, gkamThreshold: 0.55, minRadius: 0.4 }, { preset: 'FCC_Rolling', toleranceDeg: 10 }, { kamOrder: 2, kamThreshold: 7, hagbMin: 15 });
  });
});

describe('Map viewer: grain maps wait for the grains', () => {
  afterEach(resetMocks);

  it('without grains the ECD map is not requested and a neutral hint is shown', async () => {
    render(<GrainTabInPage />);
    // the Grain tab opens on the ECD layer
    expect(screen.getAllByText('This layer needs grains. Reconstruct grains first.').length).toBeGreaterThan(0);
    await new Promise((r) => setTimeout(r, 100));
    expect(mapCalls('ecd')).toHaveLength(0);
    expect(analysisApi.getMap).not.toHaveBeenCalled();
    // not shown as a failure
    expect(screen.queryByText(/⚠/)).toBeNull();
  });

  it('a layer that needs no grains is still requested without grains', async () => {
    render(<GrainTabInPage />);
    const layer = Array.from(document.querySelectorAll('select')).find((s) => s.value === 'ecd');
    fireEvent.change(layer, { target: { value: 'bc' } });
    await waitFor(() => expect(mapCalls('bc')).toHaveLength(1));
    expect(mapCalls('ecd')).toHaveLength(0);
  });

  it('after a reconstruction the ECD map is requested once', async () => {
    await reconstructWith({});
    await waitFor(() => expect(mapCalls('ecd')).toHaveLength(1));
    await new Promise((r) => setTimeout(r, 100));
    expect(mapCalls('ecd')).toHaveLength(1);
    expect(screen.queryByText('This layer needs grains. Reconstruct grains first.')).toBeNull();
  });

  it('every layer is fetched again after a new reconstruction', async () => {
    render(<GrainTabInPage />);
    const layer = Array.from(document.querySelectorAll('select')).find((s) => s.value === 'ecd');
    fireEvent.change(layer, { target: { value: 'bc' } });
    await waitFor(() => expect(mapCalls('bc')).toHaveLength(1));
    analysisApi.reconstructGrains.mockResolvedValue({ data: { task_id: 't1' } });
    analysisApi.getTaskStatus.mockResolvedValue(completedWith({}));
    fireEvent.click(screen.getByRole('button', { name: /reconstruct grains/i }));
    await waitFor(() => expect(mapCalls('bc')).toHaveLength(2), POLL_TIMEOUT);
  });
});

// ---------------------------------------------------------------------------
// The map follows the settings
// ---------------------------------------------------------------------------

const sizeAnswer = (extra) => ({
  data: { success: true, mean_ecd: 20.5, median_ecd: 7.25, std_ecd: 5.5, mean_aspect_ratio: 1.25, ...EDGE_FIELDS, ...extra },
});

describe('GrainTab: the map follows the settings', () => {
  afterEach(resetMocks);

  it('a changed ECD basis draws the ECD map again, a changed edge rule does not', async () => {
    await reconstructWith({});
    await waitFor(() => expect(mapCalls('ecd')).toHaveLength(1));
    analysisApi.grainSize.mockResolvedValue(sizeAnswer({ ecd_basis: 'measured_pixels' }));
    fireEvent.change(screen.getByLabelText('ECD from'), { target: { value: 'measured_pixels' } });
    await waitFor(() => expect(mapCalls('ecd')).toHaveLength(2));

    analysisApi.grainSize.mockResolvedValue(sizeAnswer({ ecd_basis: 'measured_pixels', edge_rule: 'mtex' }));
    fireEvent.change(screen.getByLabelText('Edge grains'), { target: { value: 'mtex' } });
    await waitFor(() => expect(screen.getByText(/Edge grains \(MTEX rule\)/)).toBeTruthy());
    await new Promise((r) => setTimeout(r, 100));
    expect(mapCalls('ecd')).toHaveLength(2);
  });
});

describe('GrainTab: the ECD map follows the basis, after the statistics answer', () => {
  afterEach(resetMocks);

  it('the ECD map is fetched again when the answer with the new basis arrives, not when the select changes', async () => {
    await reconstructWith({});
    await waitFor(() => expect(mapCalls('ecd')).toHaveLength(1));
    let answer;
    analysisApi.grainSize.mockImplementation(() => new Promise((resolve) => { answer = resolve; }));
    fireEvent.change(screen.getByLabelText('ECD from'), { target: { value: 'measured_pixels' } });
    await waitFor(() => expect(analysisApi.grainSize).toHaveBeenCalledTimes(1));
    await new Promise((r) => setTimeout(r, 100));
    expect(mapCalls('ecd')).toHaveLength(1);                      // the backend still draws the old basis
    answer(sizeAnswer({ ecd_basis: 'measured_pixels' }));
    await waitFor(() => expect(mapCalls('ecd')).toHaveLength(2));
    // and back again: one more
    analysisApi.grainSize.mockResolvedValue(sizeAnswer({ ecd_basis: 'cell_area' }));
    fireEvent.change(screen.getByLabelText('ECD from'), { target: { value: 'cell_area' } });
    await waitFor(() => expect(mapCalls('ecd')).toHaveLength(3));
  });

  it('an answer that keeps the basis (another edge rule or phase) leaves the ECD map alone', async () => {
    await reconstructWith({});
    await waitFor(() => expect(mapCalls('ecd')).toHaveLength(1));
    analysisApi.grainSize.mockResolvedValue(sizeAnswer({ edge_rule: 'outer_row' }));
    fireEvent.change(screen.getByLabelText('Edge grains'), { target: { value: 'outer_row' } });
    await waitFor(() => expect(analysisApi.grainSize).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(screen.getByText(/Edge grains \(outermost-row rule\)/)).toBeTruthy());
    await new Promise((r) => setTimeout(r, 100));
    expect(mapCalls('ecd')).toHaveLength(1);
  });
});

describe('GrainTab: a recompute with no statistics on screen', () => {
  afterEach(resetMocks);

  it('shows the total of the answer, never "Total grains: 0"', async () => {
    useResultStore.setState({ hasGrains: true });      // e.g. the tab was left and opened again
    render(<GrainTabInPage />);
    analysisApi.grainSize.mockResolvedValue(sizeAnswer({ ecd_basis: 'measured_pixels' }));
    fireEvent.change(screen.getByLabelText('ECD from'), { target: { value: 'measured_pixels' } });
    await waitFor(() => expect(screen.getByText(/Total grains:\s+98/)).toBeTruthy());
    expect(screen.queryByText(/Total grains:\s+0\b/)).toBeNull();
  });

  it('a successful recompute leaves the error of a failed reconstruction on screen', async () => {
    useResultStore.setState({ hasGrains: true });
    analysisApi.reconstructGrains.mockResolvedValue({ data: { task_id: 't1' } });
    analysisApi.getTaskStatus.mockResolvedValue({ data: { status: 'failed', progress: 1, error: 'no grains today' } });
    render(<GrainTabInPage />);
    fireEvent.click(screen.getByRole('button', { name: /reconstruct grains/i }));
    await waitFor(() => expect(screen.getByText('no grains today')).toBeTruthy(), POLL_TIMEOUT);

    analysisApi.grainSize.mockResolvedValue(sizeAnswer({ ecd_basis: 'measured_pixels' }));
    fireEvent.change(screen.getByLabelText('ECD from'), { target: { value: 'measured_pixels' } });
    await waitFor(() => expect(screen.getByText(/Mean ECD:\s+20\.50/)).toBeTruthy());
    expect(screen.getByText('no grains today')).toBeTruthy();
  });

  it('a successful recompute does remove the message of an earlier failed recompute', async () => {
    await reconstructWith({});
    analysisApi.grainSize.mockRejectedValueOnce({ response: { status: 400, data: { detail: 'recompute broke' } } });
    fireEvent.change(screen.getByLabelText('ECD from'), { target: { value: 'measured_pixels' } });
    await waitFor(() => expect(screen.getByText('recompute broke')).toBeTruthy());
    analysisApi.grainSize.mockResolvedValue(sizeAnswer({ ecd_basis: 'cell_area' }));
    fireEvent.change(screen.getByLabelText('ECD from'), { target: { value: 'cell_area' } });
    await waitFor(() => expect(screen.queryByText('recompute broke')).toBeNull());
  });
});

describe('Races: an older answer never overwrites a newer one', () => {
  afterEach(resetMocks);

  it('at most one /grain-size in flight; the last request sent carries the final selection', async () => {
    await reconstructWith({});
    let resolveFirst;
    let resolveSecond;
    analysisApi.grainSize
      .mockImplementationOnce(() => new Promise((r) => { resolveFirst = r; }))
      .mockImplementationOnce(() => new Promise((r) => { resolveSecond = r; }));
    fireEvent.change(screen.getByLabelText('ECD from'), { target: { value: 'measured_pixels' } });
    await waitFor(() => expect(analysisApi.grainSize).toHaveBeenCalledTimes(1));
    expect(analysisApi.grainSize).toHaveBeenLastCalledWith('matlab_script', 'measured_pixels', null);

    // the selection moves on while the first answer is out
    fireEvent.change(screen.getByLabelText('Edge grains'), { target: { value: 'mtex' } });
    fireEvent.change(screen.getByLabelText('Edge grains'), { target: { value: 'outer_row' } });
    await new Promise((r) => setTimeout(r, 400));
    expect(analysisApi.grainSize).toHaveBeenCalledTimes(1);               // never two at once

    resolveFirst(sizeAnswer({ mean_ecd: 11, ecd_basis: 'measured_pixels' }));
    await waitFor(() => expect(analysisApi.grainSize).toHaveBeenCalledTimes(2));
    expect(analysisApi.grainSize).toHaveBeenLastCalledWith('outer_row', 'measured_pixels', null);
    resolveSecond(sizeAnswer({ mean_ecd: 30, edge_rule: 'outer_row', ecd_basis: 'measured_pixels' }));
    await waitFor(() => expect(screen.getByText(/Mean ECD:\s+30\.00/)).toBeTruthy());
    await new Promise((r) => setTimeout(r, 400));
    expect(analysisApi.grainSize).toHaveBeenCalledTimes(2);               // nothing left to send
  });

  it('stepping through the options inside the pause sends only the last one', async () => {
    await reconstructWith({});
    analysisApi.grainSize.mockResolvedValue(sizeAnswer({ edge_rule: 'keep_all' }));
    for (const value of ['mtex', 'outer_row', 'keep_all']) {
      fireEvent.change(screen.getByLabelText('Edge grains'), { target: { value } });
      await new Promise((r) => setTimeout(r, 60));          // a key repeat, well inside the pause
    }
    await waitFor(() => expect(analysisApi.grainSize).toHaveBeenCalledTimes(1));
    await new Promise((r) => setTimeout(r, 400));
    expect(analysisApi.grainSize).toHaveBeenCalledTimes(1);
    expect(analysisApi.grainSize).toHaveBeenCalledWith('keep_all', 'cell_area', null);
  });

  it('a new reconstruction drops a /grain-size answer still on its way', async () => {
    await reconstructWith({});
    let resolveSize;
    analysisApi.grainSize.mockImplementationOnce(() => new Promise((r) => { resolveSize = r; }));
    fireEvent.change(screen.getByLabelText('ECD from'), { target: { value: 'measured_pixels' } });
    await waitFor(() => expect(analysisApi.grainSize).toHaveBeenCalledTimes(1));

    analysisApi.reconstructGrains.mockResolvedValue({ data: { task_id: 't2' } });
    analysisApi.getTaskStatus.mockImplementation(() => new Promise(() => {}));   // the new run never ends
    fireEvent.click(screen.getByRole('button', { name: /reconstruct grains/i }));
    await waitFor(() => expect(analysisApi.reconstructGrains).toHaveBeenCalledTimes(2));

    resolveSize(sizeAnswer({ mean_ecd: 99, ecd_basis: 'measured_pixels' }));
    await new Promise((r) => setTimeout(r, 80));
    expect(screen.queryByText(/Mean ECD/)).toBeNull();           // the old grains' numbers did not land
    expect(screen.queryByText(/Total grains/)).toBeNull();
  });

  it('map answers that arrive out of order: the layer last picked wins', async () => {
    const pending = [];
    analysisApi.getMap.mockImplementation((type) => new Promise((resolve) => { pending.push({ type, resolve }); }));
    render(<GrainTabInPage />);
    const layerSelect = () => Array.from(document.querySelectorAll('select'))
      .find((el) => Array.from(el.options).some((o) => o.value === 'kam'));
    fireEvent.change(layerSelect(), { target: { value: 'bc' } });
    await waitFor(() => expect(pending).toHaveLength(1));
    fireEvent.change(layerSelect(), { target: { value: 'kam' } });
    await waitFor(() => expect(pending).toHaveLength(2));

    pending[1].resolve({ data: { image: 'KAMIMAGE' } });
    await waitFor(() => expect(document.querySelector('img.map-image')?.src).toContain('KAMIMAGE'));
    pending[0].resolve({ data: { image: 'BCIMAGE' } });
    await new Promise((r) => setTimeout(r, 80));
    expect(document.querySelector('img.map-image').src).toContain('KAMIMAGE');
  });
});

describe('Map viewer: every grain layer waits for the grains', () => {
  afterEach(resetMocks);

  it.each(['gos', 'ecd', 'rx', 'grain_boundaries'])('%s is not requested without grains', async (type) => {
    render(<GrainTabInPage />);
    const layerSelect = Array.from(document.querySelectorAll('select'))
      .find((el) => Array.from(el.options).some((o) => o.value === 'kam'));
    if (layerSelect.value !== type) fireEvent.change(layerSelect, { target: { value: type } });
    await waitFor(() => expect(screen.getAllByText('This layer needs grains. Reconstruct grains first.').length).toBeGreaterThan(0));
    await new Promise((r) => setTimeout(r, 60));
    expect(mapCalls(type)).toHaveLength(0);
  });
});

describe('GrainTab: the reconstruction parameters live in the page-level object', () => {
  afterEach(resetMocks);

  it('typed parameters reach reconstructGrains and are still there when the tab is opened again', async () => {
    analysisApi.reconstructGrains.mockResolvedValue({ data: { task_id: 't1' } });
    analysisApi.getTaskStatus.mockResolvedValue(completedWith({}));
    const first = render(<GrainTabInPage />);
    fireEvent.change(screen.getByTitle(/Minimum misorientation/), { target: { value: '6.5' } });
    first.unmount();
    render(<GrainTabInPage />);
    expect(screen.getByTitle(/Minimum misorientation/).value).toBe('6.5');
    fireEvent.click(screen.getByRole('button', { name: /reconstruct grains/i }));
    await waitFor(() => expect(analysisApi.reconstructGrains).toHaveBeenCalledTimes(1));
    expect(analysisApi.reconstructGrains).toHaveBeenCalledWith(6.5, 3, 1.5, 3.1, true, 'matlab_script', 'cell_area', null);
  });
});

describe('BatchTab: starts at the Grain tab choice, then goes its own way', () => {
  afterEach(resetMocks);

  it('initial values from the store; changing them there does not touch the store', () => {
    useResultStore.getState().setGrainStatsSettings({ edgeRule: 'outer_row', ecdBasis: 'measured_pixels' });
    render(<BatchTab analysisLoaded onComplete={() => {}} />);
    expect(screen.getByLabelText('Edge grains').value).toBe('outer_row');
    expect(screen.getByLabelText('ECD from').value).toBe('measured_pixels');
    fireEvent.change(screen.getByLabelText('Edge grains'), { target: { value: 'keep_all' } });
    expect(useResultStore.getState().grainStatsSettings.edgeRule).toBe('outer_row');
  });
});

describe('GrainTab: running reconstruction, pauses and the store', () => {
  afterEach(resetMocks);

  it('the selects and the button are disabled while a reconstruction runs', async () => {
    analysisApi.reconstructGrains.mockResolvedValue({ data: { task_id: 't1' } });
    let finish;
    analysisApi.getTaskStatus.mockImplementation(() => new Promise((r) => { finish = r; }));
    render(<GrainTabInPage />);
    fireEvent.click(screen.getByRole('button', { name: /reconstruct grains/i }));
    await waitFor(() => expect(screen.getByLabelText('Edge grains').disabled).toBe(true));
    expect(screen.getByLabelText('ECD from').disabled).toBe(true);
    expect(screen.getByRole('button', { name: /reconstructing/i }).disabled).toBe(true);
    await waitFor(() => expect(finish).toBeDefined(), POLL_TIMEOUT);
    finish(completedWith({}));
    await waitFor(() => expect(screen.getByLabelText('Edge grains').disabled).toBe(false), POLL_TIMEOUT);
  });

  it('a pick made after a reconstruction is sent although an older request was still in flight', async () => {
    await reconstructWith({});
    let resolveOld;
    analysisApi.grainSize.mockImplementationOnce(() => new Promise((r) => { resolveOld = r; }));
    fireEvent.change(screen.getByLabelText('ECD from'), { target: { value: 'measured_pixels' } });
    await waitFor(() => expect(analysisApi.grainSize).toHaveBeenCalledTimes(1));

    analysisApi.reconstructGrains.mockResolvedValue({ data: { task_id: 't2' } });
    fireEvent.click(screen.getByRole('button', { name: /reconstruct grains/i }));
    await waitFor(() => expect(analysisApi.reconstructGrains).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.getByLabelText('Edge grains').disabled).toBe(false), POLL_TIMEOUT);
    await waitFor(() => expect(screen.getByText(/Total grains:\s+98/)).toBeTruthy());

    analysisApi.grainSize.mockResolvedValue(sizeAnswer({ edge_rule: 'mtex', mean_ecd: 41 }));
    fireEvent.change(screen.getByLabelText('Edge grains'), { target: { value: 'mtex' } });
    await new Promise((r) => setTimeout(r, 400));
    expect(analysisApi.grainSize).toHaveBeenCalledTimes(1);                   // the old request is still out

    resolveOld(sizeAnswer({ mean_ecd: 99 }));                                  // the old grains' numbers
    await waitFor(() => expect(analysisApi.grainSize).toHaveBeenCalledTimes(2));
    expect(analysisApi.grainSize).toHaveBeenLastCalledWith('mtex', 'measured_pixels', null);
    await waitFor(() => expect(screen.getByText(/Mean ECD:\s+41\.00/)).toBeTruthy());
    expect(screen.queryByText(/99\.00/)).toBeNull();
  });

  it('leaving the tab inside the pause still sends the pick', async () => {
    const view = await reconstructWith({});
    analysisApi.grainSize.mockResolvedValue(sizeAnswer({ edge_rule: 'mtex' }));
    fireEvent.change(screen.getByLabelText('Edge grains'), { target: { value: 'mtex' } });
    expect(analysisApi.grainSize).not.toHaveBeenCalled();                     // still inside the pause
    view.unmount();
    expect(analysisApi.grainSize).toHaveBeenCalledTimes(1);
    expect(analysisApi.grainSize).toHaveBeenCalledWith('mtex', 'cell_area', null);
    await waitFor(() => expect(useResultStore.getState().grainStats.meanEcd).toBe(20.5));
  });

  it('grains present but none in the statistics: no means, one line saying so, and the edge line', async () => {
    await reconstructWith({
      mean_ecd: 0, median_ecd: undefined, std_ecd: 0, mean_aspect_ratio: 0,
      n_grains_total: 98, n_edge_grains: 0, n_grains_used: 0,
    });
    expect(screen.getByText(/No grain entered the statistics/)).toBeTruthy();
    expect(screen.queryByText(/Grains in the statistics/)).toBeNull();
    expect(screen.queryByText(/Mean ECD/)).toBeNull();
    expect(screen.queryByText(/Std ECD/)).toBeNull();
    expect(screen.queryByText(/Mean AR/)).toBeNull();
    expect(screen.getByText(/Edge grains \(MATLAB-script rule\): 0 of 98 left out(?! —)/)).toBeTruthy();
  });

  it('the statistics block is kept in the store when the tab is left', async () => {
    const view = await reconstructWith({});
    view.unmount();
    render(<GrainTabInPage />);
    expect(screen.getByText(/Mean ECD:\s+12\.34/)).toBeTruthy();
  });

  it('a failed batch task ends with the message of the backend', async () => {
    analysisApi.batch.mockResolvedValue({ data: { task_id: 'b1' } });
    analysisApi.getTaskStatus.mockResolvedValue({ data: { status: 'failed', progress: 1, error: 'folder does not exist' } });
    render(<BatchTab analysisLoaded onComplete={() => {}} />);
    fireEvent.change(screen.getByPlaceholderText('/path/to/dataset/folder'), { target: { value: 'F' } });
    fireEvent.change(screen.getByPlaceholderText('/path/to/Documentation.xlsx'), { target: { value: 'O.xlsx' } });
    fireEvent.click(screen.getByRole('button', { name: /process folder/i }));
    await waitFor(() => expect(screen.getByText('folder does not exist')).toBeTruthy(), POLL_TIMEOUT);
    expect(screen.getByRole('button', { name: /process folder/i }).disabled).toBe(false);
  });

  it('a batch task the backend forgot ends the wait', async () => {
    analysisApi.batch.mockResolvedValue({ data: { task_id: 'b1' } });
    analysisApi.getTaskStatus.mockRejectedValue({ response: { status: 404 } });
    render(<BatchTab analysisLoaded onComplete={() => {}} />);
    fireEvent.change(screen.getByPlaceholderText('/path/to/dataset/folder'), { target: { value: 'F' } });
    fireEvent.change(screen.getByPlaceholderText('/path/to/Documentation.xlsx'), { target: { value: 'O.xlsx' } });
    fireEvent.click(screen.getByRole('button', { name: /process folder/i }));
    await waitFor(() => expect(screen.getByRole('button', { name: /process folder/i }).disabled).toBe(false), POLL_TIMEOUT);
    expect(analysisApi.getTaskStatus).toHaveBeenCalledTimes(1);
  });
});
