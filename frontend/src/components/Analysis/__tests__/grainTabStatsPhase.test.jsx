// @vitest-environment jsdom
//
// The statistics phase in the Grain tab: the select, what it sends, what the statistics block says, the
// block only while grains exist, and the two tooltips. Real components, only the HTTP layer mocked, with the
// shapes of the real answers (routes/analysis.py `_edge_fields`, `/grain-size`, the reconstruct task).
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
  },
  indexApi: {},
  edsApi: {},
}));

import { analysisApi } from '../../../services/api';
import useResultStore from '../../../stores/useResultStore';
import { GrainTabInPage } from './grainTabHarness';

const POLL_TIMEOUT = { timeout: 5000 };
const PHASES = [{ id: 1, name: 'Aluminium' }, { id: 2, name: 'Si' }, { id: 3, name: 'Fe-rich' }];

/** The fields every grain-size answer carries (`_edge_fields`), for a phase or for all phases. */
const forPhase = (id, name, counts) => ({
  edge_rule: 'matlab_script', ecd_basis: 'cell_area', stats_phase: id, stats_phase_name: name,
  n_grains_total: 98, n_edge_grains: 34, n_grains_used: 64, edge_safeguard_applied: false,
  ...counts,
});
const sizeAnswer = (extra) => ({
  data: {
    success: true, ...forPhase(null, null), mean_ecd: 20.5, std_ecd: 4.2, mean_aspect_ratio: 1.8,
    median_ecd: 19.0, median_aspect_ratio: 1.7, std_aspect_ratio: 0.3, area_weighted_ecd: 25.1,
    mean_maxDimX: 5, mean_maxDimY: 6, std_maxDimX: 1, std_maxDimY: 1, ...extra,
  },
});
const completedWith = (extra) => ({
  data: {
    status: 'completed', progress: 1.0, grain_count: 98,
    result: {
      n_grains: 98, grain_count: 98, alpha: 3.1, fill: true, n_filled: 4,
      mean_ecd: 12.34, median_ecd: 11.0, std_ecd: 3.21, mean_aspect_ratio: 1.678, median_aspect_ratio: 1.5,
      ...forPhase(null, null), ...extra,
    },
  },
});

function setUp({ hasGrains = false } = {}) {
  cleanup();
  vi.resetAllMocks();
  useResultStore.getState().clearResults();
  useResultStore.setState({ phaseOptions: PHASES, hasGrains });
  analysisApi.getMap.mockImplementation(() => new Promise(() => {}));
}

const phaseSelect = () => screen.getByLabelText('Phase');
const optionLabels = (select) => Array.from(select.options).map((o) => o.textContent);

describe('GrainTab: the Phase select', () => {
  afterEach(() => { cleanup(); });

  it('offers All phases and the phases of the map, and says what it does', () => {
    setUp();
    render(<GrainTabInPage />);
    expect(optionLabels(phaseSelect())).toEqual(['All phases', 'Aluminium', 'Si', 'Fe-rich']);
    expect(phaseSelect().value).toBe('all');
    const help = /^Statistics over the grains of all phases\.$/;
    expect(screen.getByText(help)).toBeTruthy();
    expect(phaseSelect().title).toMatch(help);
    // the help line is the select's description
    const described = document.getElementById(phaseSelect().getAttribute('aria-describedby'));
    expect(described.textContent).toMatch(help);
  });

  it('shows a phase id the list does not have under its number, never a wrong phase', () => {
    setUp();
    useResultStore.setState({ phaseOptions: [], grainStatsSettings: { ...useResultStore.getState().grainStatsSettings, statsPhase: 7 } });
    render(<GrainTabInPage />);
    expect(phaseSelect().value).toBe('7');
    expect(optionLabels(phaseSelect())).toEqual(['All phases', 'Phase 7']);
  });

  it('a phase chosen before the reconstruction reaches reconstructGrains', async () => {
    setUp();
    analysisApi.reconstructGrains.mockResolvedValue({ data: { task_id: 't1' } });
    analysisApi.getTaskStatus.mockResolvedValue(completedWith(forPhase(2, 'Si', { n_grains_total: 12, n_edge_grains: 5, n_grains_used: 7 })));
    render(<GrainTabInPage />);
    fireEvent.change(phaseSelect(), { target: { value: '2' } });
    fireEvent.click(screen.getByRole('button', { name: /reconstruct grains/i }));
    await waitFor(() => expect(analysisApi.reconstructGrains).toHaveBeenCalledTimes(1));
    expect(analysisApi.reconstructGrains).toHaveBeenCalledWith(4, 3, 1.5, 3.1, true, 'matlab_script', 'cell_area', 2);
    // the statistics name the phase and keep the total of the map
    await waitFor(() => expect(screen.getByText(/Grains of Si:\s+12 \(all phases: 98\)/)).toBeTruthy(), POLL_TIMEOUT);
  });

  it('with grains present, picking a phase recomputes the statistics for it, and All phases goes back', async () => {
    setUp({ hasGrains: true });
    useResultStore.setState({ grainStats: { count: 98, meanEcd: 12.34, nGrainsTotal: 98, nEdgeGrains: 34, nGrainsUsed: 64, edgeRule: 'matlab_script', ecdBasis: 'cell_area' } });
    analysisApi.grainSize.mockResolvedValue(sizeAnswer(forPhase(2, 'Si', { n_grains_total: 12, n_edge_grains: 5, n_grains_used: 7, mean_ecd: 3.5 })));
    render(<GrainTabInPage />);
    fireEvent.change(phaseSelect(), { target: { value: '2' } });
    await waitFor(() => expect(analysisApi.grainSize).toHaveBeenCalledWith('matlab_script', 'cell_area', 2));
    await waitFor(() => expect(screen.getByText(/Grains of Si:\s+12 \(all phases: 98\)/)).toBeTruthy());
    expect(screen.getByText(/Mean ECD:\s+3\.50/)).toBeTruthy();

    analysisApi.grainSize.mockResolvedValue(sizeAnswer({}));
    fireEvent.change(phaseSelect(), { target: { value: 'all' } });
    // always explicit: null means all phases, a missing field would keep the phase
    await waitFor(() => expect(analysisApi.grainSize).toHaveBeenLastCalledWith('matlab_script', 'cell_area', null));
    await waitFor(() => expect(screen.getByText(/Total grains:\s+98/)).toBeTruthy());
    expect(screen.queryByText(/Grains of Si/)).toBeNull();
  });

  it('the "Analyze Grain Size" button sends the chosen phase too', async () => {
    setUp({ hasGrains: true });
    useResultStore.setState({ grainStatsSettings: { ...useResultStore.getState().grainStatsSettings, statsPhase: 3 } });
    analysisApi.grainSize.mockResolvedValue(sizeAnswer(forPhase(3, 'Fe-rich', { n_grains_total: 4, n_edge_grains: 1, n_grains_used: 3 })));
    render(<GrainTabInPage />);
    fireEvent.click(screen.getByRole('button', { name: /analyze grain size/i }));
    await waitFor(() => expect(analysisApi.grainSize).toHaveBeenCalledWith('matlab_script', 'cell_area', 3));
  });

  it('a phase picked while an older request is in flight is sent when that one is back', async () => {
    setUp({ hasGrains: true });
    let resolveFirst;
    analysisApi.grainSize.mockImplementationOnce(() => new Promise((r) => { resolveFirst = r; }));
    analysisApi.grainSize.mockResolvedValue(sizeAnswer(forPhase(3, 'Fe-rich', { n_grains_total: 4, n_edge_grains: 1, n_grains_used: 3, mean_ecd: 7 })));
    render(<GrainTabInPage />);
    fireEvent.change(phaseSelect(), { target: { value: '2' } });
    await waitFor(() => expect(analysisApi.grainSize).toHaveBeenCalledTimes(1));
    fireEvent.change(phaseSelect(), { target: { value: '3' } });
    await new Promise((r) => setTimeout(r, 400));
    expect(analysisApi.grainSize).toHaveBeenCalledTimes(1);          // one in flight at a time
    resolveFirst(sizeAnswer(forPhase(2, 'Si', { mean_ecd: 99 })));
    await waitFor(() => expect(analysisApi.grainSize).toHaveBeenLastCalledWith('matlab_script', 'cell_area', 3));
    await waitFor(() => expect(screen.getByText(/Mean ECD:\s+7\.00/)).toBeTruthy());
  });
});

describe('GrainTab: the statistics block with a phase', () => {
  afterEach(() => { cleanup(); });

  it('a chosen phase without grains gets one line, no means, no zeros', async () => {
    setUp({ hasGrains: true });
    // the real answer for a phase without grains: counts 0 and means 0.0
    analysisApi.grainSize.mockResolvedValue(sizeAnswer({
      ...forPhase(3, 'Fe-rich', { n_grains_total: 0, n_edge_grains: 0, n_grains_used: 0 }),
      mean_ecd: 0, std_ecd: 0, mean_aspect_ratio: 0, median_ecd: 0, median_aspect_ratio: 0,
    }));
    useResultStore.setState({ grainStats: { count: 98, meanEcd: 12.34, nGrainsTotal: 98, nEdgeGrains: 34, nGrainsUsed: 64, edgeRule: 'matlab_script', ecdBasis: 'cell_area' } });
    render(<GrainTabInPage />);
    fireEvent.change(phaseSelect(), { target: { value: '3' } });
    await waitFor(() => expect(screen.getByText('No grain of Fe-rich in this map')).toBeTruthy());
    expect(screen.queryByText(/Mean ECD/)).toBeNull();
    expect(screen.queryByText(/Mean AR/)).toBeNull();
    expect(screen.queryByText(/0\.00/)).toBeNull();
    expect(screen.queryByText(/Total grains/)).toBeNull();
    expect(screen.queryByText(/ECD from:/)).toBeNull();
    expect(screen.queryByText(/Edge grains \(/)).toBeNull();
  });

  it('with all phases a map without any grain still reads "Total grains: 0", without means', async () => {
    setUp({ hasGrains: true });
    analysisApi.grainSize.mockResolvedValue(sizeAnswer({
      ...forPhase(null, null, { n_grains_total: 0, n_edge_grains: 0, n_grains_used: 0 }), mean_ecd: 0, std_ecd: 0, mean_aspect_ratio: 0, median_ecd: 0,
    }));
    useResultStore.setState({ grainStats: { count: 0, nGrainsTotal: 0 } });
    render(<GrainTabInPage />);
    fireEvent.change(screen.getByLabelText('ECD from'), { target: { value: 'measured_pixels' } });
    await waitFor(() => expect(analysisApi.grainSize).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByText(/Total grains:\s+0/)).toBeTruthy());
    expect(screen.queryByText(/Mean ECD/)).toBeNull();
    expect(screen.queryByText(/No grain of/)).toBeNull();
  });

  it('a phase with its total unknown (the block was empty) still names the phase', async () => {
    setUp({ hasGrains: true });
    analysisApi.grainSize.mockResolvedValue(sizeAnswer(forPhase(2, 'Si', { n_grains_total: 12, n_edge_grains: 5, n_grains_used: 7 })));
    useResultStore.setState({ grainStatsSettings: { ...useResultStore.getState().grainStatsSettings, statsPhase: 2 } });
    render(<GrainTabInPage />);
    fireEvent.click(screen.getByRole('button', { name: /analyze grain size/i }));
    await waitFor(() => expect(screen.getByText(/Grains of Si:\s+12\b/)).toBeTruthy());
    // the 12 grains of the phase are not the total of the map
    expect(screen.queryByText(/Total grains/)).toBeNull();
    expect(screen.queryByText(/all phases:/)).toBeNull();
  });
});

describe('GrainTab: the statistics block only while grains exist', () => {
  afterEach(() => { cleanup(); });
  const block = { count: 98, meanEcd: 12.34, nGrainsTotal: 98, nEdgeGrains: 34, nGrainsUsed: 64, edgeRule: 'matlab_script', ecdBasis: 'cell_area' };

  it('statistics of a map the backend no longer has are not drawn', () => {
    setUp({ hasGrains: false });
    useResultStore.setState({ grainStats: block });
    render(<GrainTabInPage />);
    expect(screen.queryByText('Grain Statistics')).toBeNull();
    expect(screen.queryByText(/Mean ECD/)).toBeNull();
  });

  it('and are drawn as soon as grains exist', () => {
    setUp({ hasGrains: true });
    useResultStore.setState({ grainStats: block });
    render(<GrainTabInPage />);
    expect(screen.getByText('Grain Statistics')).toBeTruthy();
    expect(screen.getByText(/Mean ECD:\s+12\.34/)).toBeTruthy();
  });
});

describe('GrainTab: tooltips', () => {
  afterEach(() => { cleanup(); });

  const WHOLE_MAP = 'Statistics over the grains of this phase only. The edge rule looks at the whole map: a grain next to another phase is not an edge grain.';
  const STRIP = "Statistics over the grains of this phase only. With the MATLAB-script rule the border strip is drawn around this phase's own extent, so grains of a minority phase next to other phases also count as edge grains.";
  const tipOf = (name) => Array.from(phaseSelect().options).find((o) => o.textContent === name).title;

  it('each phase choice has its own sentence, and a phase follows the current edge rule', () => {
    setUp();
    render(<GrainTabInPage />);
    expect(tipOf('All phases')).toBe('Statistics over the grains of all phases.');
    expect(tipOf('Si')).toBe(STRIP);
    expect(tipOf('Aluminium')).toBe(STRIP);
    fireEvent.change(screen.getByLabelText('Edge grains'), { target: { value: 'mtex' } });
    expect(tipOf('Si')).toBe(WHOLE_MAP);
    expect(tipOf('All phases')).toBe('Statistics over the grains of all phases.');
    fireEvent.change(screen.getByLabelText('Edge grains'), { target: { value: 'outer_row' } });
    expect(tipOf('Si')).toBe(WHOLE_MAP);
    fireEvent.change(screen.getByLabelText('Edge grains'), { target: { value: 'keep_all' } });
    expect(tipOf('Si')).toBe('Statistics over the grains of this phase only.');
  });

  it('the help line under the select is the sentence of the chosen option, and follows the rule', () => {
    setUp();
    render(<GrainTabInPage />);
    const help = () => document.getElementById(phaseSelect().getAttribute('aria-describedby')).textContent;
    expect(help()).toBe('Statistics over the grains of all phases.');
    fireEvent.change(phaseSelect(), { target: { value: '2' } });
    expect(help()).toBe(STRIP);
    expect(phaseSelect().title).toBe(STRIP);
    fireEvent.change(screen.getByLabelText('Edge grains'), { target: { value: 'mtex' } });
    expect(help()).toBe(WHOLE_MAP);
  });

  it('says which settings give the numbers of the MATLAB script and which those of MTEX', () => {
    setUp();
    render(<GrainTabInPage />);
    expect(screen.getByText(
      "The MATLAB script's statistics: 'Like the MATLAB script', 'Measured pixels × step²' and the phase the script was run on (its ph). MTEX's: 'Like MTEX' and 'Cell area'.")).toBeTruthy();
    // the names quoted are the beginnings of the option labels
    const labels = (name) => Array.from(screen.getByLabelText(name).options).map((o) => o.textContent);
    expect(labels('Edge grains')[0]).toMatch(/^Like the MATLAB script/);
    expect(labels('Edge grains')[1]).toMatch(/^Like MTEX/);
  });

  it('the Analyze Grain Size button says the statistics are computed anyway', () => {
    setUp({ hasGrains: true });
    useResultStore.setState({ grainStats: { count: 98, meanEcd: 1, nGrainsTotal: 98, nEdgeGrains: 0, nGrainsUsed: 98, edgeRule: 'matlab_script', ecdBasis: 'cell_area' } });
    render(<GrainTabInPage />);
    expect(screen.getByRole('button', { name: /analyze grain size/i }).title).toBe(
      'The statistics above are computed with every reconstruction and every change of the settings; this computes them again now.');
  });

  it('the minimum intercept says it is recorded in the Excel sheet and not applied', () => {
    setUp();
    render(<GrainTabInPage />);
    const field = screen.getByDisplayValue('1.5');
    expect(field.title).toBe('Recorded in the Excel sheet; not applied by the grain reconstruction (as in the MATLAB script).');
  });

  it('the MATLAB-script rule says what a chosen minority phase does to the border strip', () => {
    setUp();
    render(<GrainTabInPage />);
    expect(screen.getByLabelText('Edge grains').value).toBe('matlab_script');
    expect(screen.getByLabelText('Edge grains').title)
      .toMatch(/With a chosen phase, the border strip is drawn around that phase's own extent, so grains of a minority phase next to other phases are left out as well\./);
    // the numbers of the safeguard are still in the sentence
    expect(screen.getByLabelText('Edge grains').title).toMatch(/fewer than 50 grains or less than 15 %/);
  });
});

describe('GrainTab: no line about the grains in the statistics for a map without grains', () => {
  afterEach(() => { cleanup(); });

  it('has no "Grains in the statistics" line when there are no grains at all', () => {
    setUp({ hasGrains: true });
    useResultStore.setState({ grainStats: {
      count: 0, nGrainsTotal: 0, nEdgeGrains: 0, nGrainsUsed: 0, edgeRule: 'matlab_script', ecdBasis: 'cell_area',
    } });
    render(<GrainTabInPage />);
    expect(screen.queryByText(/Grains in the statistics/)).toBeNull();
  });
});

describe('GrainTab: the statistics box says what it was made with', () => {
  afterEach(() => { cleanup(); });
  const block = (extra) => ({
    count: 76, meanEcd: 5.75, medianEcd: 2.94, stdEcd: 7.5, meanAr: 2.263, nGrainsTotal: 76, nEdgeGrains: 0, nGrainsUsed: 76,
    edgeRule: 'matlab_script', ecdBasis: 'cell_area', ...extra,
  });

  it('names the step of the map', () => {
    setUp({ hasGrains: true });
    useResultStore.setState({ grainStats: block({}), stepSize: 0.75 });
    render(<GrainTabInPage />);
    expect(screen.getByText(/Step:\s+0\.75 µm/)).toBeTruthy();
  });

  it('has no step line when the step is not known, and none for a map without grains', () => {
    setUp({ hasGrains: true });
    useResultStore.setState({ grainStats: block({}) });
    render(<GrainTabInPage />);
    expect(screen.queryByText(/Step:/)).toBeNull();
    cleanup();
    useResultStore.setState({ grainStats: block({ count: 0, nGrainsTotal: 0, nGrainsUsed: 0 }), stepSize: 0.75 });
    render(<GrainTabInPage />);
    expect(screen.queryByText(/Step:/)).toBeNull();
  });

  it('names the rule when the safeguard left nothing out, and says where the reason is', () => {
    setUp({ hasGrains: true });
    useResultStore.setState({ grainStats: block({ nEdgeGrains: 38, edgeSafeguardApplied: true }) });
    render(<GrainTabInPage />);
    expect(screen.getByText(/Edge grains \(MATLAB-script rule\): none left out \(the rule's limit, see below\)/)).toBeTruthy();
  });

  it('says next to the minimum intercept that it is recorded only', () => {
    setUp();
    render(<GrainTabInPage />);
    expect(screen.getByText('recorded only')).toBeTruthy();
  });

  it('the edge-rule tooltips say what the 15 % is of', () => {
    setUp();
    render(<GrainTabInPage />);
    const tip = () => screen.getByLabelText('Edge grains').title;
    expect(tip()).toMatch(/fewer than 50 grains or less than 15 % of the grains would remain/);
    fireEvent.change(screen.getByLabelText('Edge grains'), { target: { value: 'mtex' } });
    expect(tip()).toMatch(/fewer than 50 grains or less than 15 % of the grains would remain/);
  });
});

describe('GrainTab: the safeguard notice names the grains it counts', () => {
  afterEach(() => { cleanup(); });
  const stats = (extra) => ({
    count: 98, meanEcd: 3.5, nGrainsTotal: 20, nEdgeGrains: 19, nGrainsUsed: 20, edgeRule: 'outer_row', ecdBasis: 'cell_area',
    edgeSafeguardApplied: true, ...extra,
  });
  const notice = () => screen.getAllByRole('status').find((el) => /kept:/.test(el.textContent));

  it('with a phase, the reasons few and share name the phase too', () => {
    setUp({ hasGrains: true });
    useResultStore.setState({ grainStats: stats({
      edgeRule: 'matlab_script', nGrainsTotal: 40, nEdgeGrains: 30, nGrainsUsed: 40, statsPhase: 2, statsPhaseName: 'Si',
    }) });
    render(<GrainTabInPage />);
    expect(notice().textContent).toBe(
      'All 40 grains of Si kept: the MATLAB-script rule would leave out 30 edge grains, and the 10 left would be fewer than 50, so the rule keeps all grains.');
    cleanup();
    useResultStore.setState({ grainStats: stats({
      edgeRule: 'mtex', nGrainsTotal: 1000, nEdgeGrains: 900, nGrainsUsed: 1000, statsPhase: 2, statsPhaseName: 'Si',
    }) });
    render(<GrainTabInPage />);
    expect(notice().textContent).toBe(
      'All 1000 grains of Si kept: the MTEX rule would leave out 900 edge grains, and the 100 left would be under 15 % of 1000, so the rule keeps all grains.');
  });

  it('with a phase: more than 85 % of the grains of that phase', () => {
    setUp({ hasGrains: true });
    useResultStore.setState({ grainStats: stats({ statsPhase: 2, statsPhaseName: 'Si' }) });
    render(<GrainTabInPage />);
    expect(notice().textContent).toBe(
      'All 20 grains kept: the outermost-row rule would leave out 19 of the 20 grains of Si, more than 85 %, so it keeps all grains of Si.');
  });

  it('with all phases: of all grains, as before', () => {
    setUp({ hasGrains: true });
    useResultStore.setState({ grainStats: stats({ statsPhase: null }) });
    render(<GrainTabInPage />);
    expect(notice().textContent).toBe(
      'All 20 grains kept: the outermost-row rule would leave out 19 of 20 grains, more than 85 %, so it keeps all grains.');
  });
});
