// @vitest-environment jsdom
//
// The RX tab: the colours of the RX map are explained next to the map (red = recrystallized by all four criteria,
// as the numbers and the workbook count; the other colours are GOS bands), and a map without native band contrast
// shows "needs band contrast in the map" instead of an RX number. The four inputs are sent with "Classify" (wire
// test: the numbers that reach the API are the ones in the fields) and the criteria used are printed next to the
// numbers. Real components, only the HTTP layer mocked.
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import fs from 'node:fs';
import path from 'node:path';

vi.mock('../../../services/api', () => ({
  analysisApi: { getMap: vi.fn(), rxClassify: vi.fn(), status: vi.fn() },
  indexApi: {},
  edsApi: {},
}));

import { analysisApi } from '../../../services/api';
import useResultStore from '../../../stores/useResultStore';
import { RXTab } from '../AnalysisPage';

const WITH_BC = {
  image: 'RXIMAGE', rx_drawn: true, rx_needs_bc: false, total_grains: 969,
  rx_fraction: 0.11316, rx_grain_count: 5, high_bc_fraction: 13.384, low_gkam_fraction: 0.17, low_kam_fraction: 8.29,
};
const NO_BC = { image: 'RXIMAGE', rx_drawn: false, rx_needs_bc: true, total_grains: 969 };

afterEach(() => { cleanup(); vi.resetAllMocks(); useResultStore.getState().clearResults(); });

const renderTab = () => render(<RXTab analysisLoaded hasGrains grainsVersion={1} onComplete={() => {}} />);
const classify = () => fireEvent.click(screen.getByRole('button', { name: /classify rx grains/i }));
const field = (label) => screen.getByText(label).closest('div').querySelector('input');
const type = (label, value) => fireEvent.change(field(label), { target: { value } });

describe('RX tab with native band contrast', () => {
  it('names the red of the map and prints the analysis numbers', async () => {
    analysisApi.getMap.mockResolvedValue({ data: WITH_BC });
    analysisApi.rxClassify.mockResolvedValue({ data: WITH_BC });
    renderTab();
    const legend = await screen.findByTestId('rx-legend');
    expect(legend.textContent).toMatch(/Red: recrystallized by all four criteria/);
    expect(legend.textContent).toMatch(/as in the panel and the workbook/);
    expect(legend.textContent).toMatch(/GOS bands/);
    expect(screen.queryByTestId('rx-legend-no-red')).toBeNull();
    classify();
    await waitFor(() => expect(screen.getByText(/RX Fraction:\s+0\.1%/)).toBeTruthy());
    expect(screen.queryByText(/needs band contrast/)).toBeNull();
  });
});

describe('RX tab without native band contrast', () => {
  it('shows "needs band contrast in the map" instead of a number, and the legend says there is no red', async () => {
    analysisApi.getMap.mockResolvedValue({ data: NO_BC });
    analysisApi.rxClassify.mockResolvedValue({ data: NO_BC });
    renderTab();
    const noRed = await screen.findByTestId('rx-legend-no-red');
    expect(noRed.textContent).toMatch(/There is no red: the RX criteria need band contrast in the map/);
    expect(screen.getByTestId('rx-legend').textContent).not.toMatch(/recrystallized by all four criteria/);
    classify();
    await waitFor(() => expect(screen.getByText(/RX fraction: not available \(needs band contrast in the map\)/)).toBeTruthy());
    expect(screen.queryByText(/RX Fraction:\s/)).toBeNull();
    expect(screen.queryByText(/RX Grains:/)).toBeNull();
    expect(screen.queryByTestId('rx-criteria-used')).toBeNull();          // nothing was classified with them
  });
});

describe('RX tab: the four inputs reach the backend', () => {
  it('starts with the values of the script and sends the numbers of the fields with "Classify"', async () => {
    analysisApi.getMap.mockResolvedValue({ data: WITH_BC });
    analysisApi.rxClassify.mockResolvedValue({ data: { ...WITH_BC, rx_gos_threshold: 2.5, rx_gbc_fraction: 0.55, rx_gkam_threshold: 0.8, rx_min_radius_um: 1.5 } });
    renderTab();
    expect(['GOS Threshold:', 'gBC Min Fraction:', 'gKAM Threshold:', 'Min Grain Radius:'].map((l) => field(l).value))
      .toEqual(['1.25', '0.7', '0.55', '0.4']);
    type('GOS Threshold:', '2.5');
    type('gBC Min Fraction:', '0.55');
    type('gKAM Threshold:', '0.8');
    type('Min Grain Radius:', '1.5');
    classify();
    await waitFor(() => expect(analysisApi.rxClassify).toHaveBeenCalledTimes(1));
    expect(analysisApi.rxClassify).toHaveBeenCalledWith({ gosThreshold: 2.5, gbcFraction: 0.55, gkamThreshold: 0.8, minRadius: 1.5 }, 'most_measured');
  });

  it('prints the criteria next to the numbers, and a later edit does not change that line', async () => {
    analysisApi.getMap.mockResolvedValue({ data: WITH_BC });
    analysisApi.rxClassify.mockResolvedValue({ data: WITH_BC });
    renderTab();
    type('GOS Threshold:', '2.5');
    classify();
    const line = await screen.findByTestId('rx-criteria-used');
    expect(line.textContent).toMatch(/^Criteria used: GOS < 2\.5°, gBC > 0\.7 × largest gBC, gKAM < 0\.55°, radius > 0\.4 µm/);
    expect(line.textContent).toMatch(/fixed values of the script/);
    type('GOS Threshold:', '9');
    expect(screen.getByTestId('rx-criteria-used').textContent).toMatch(/GOS < 2\.5°/);
  });

  it('keeps the fields in the store: a typed value is pending until the backend has it, then it is not', async () => {
    analysisApi.getMap.mockResolvedValue({ data: WITH_BC });
    analysisApi.rxClassify.mockResolvedValue({ data: { ...WITH_BC, rx_gos_threshold: 2.5, rx_gbc_fraction: 0.7, rx_gkam_threshold: 0.55, rx_min_radius_um: 0.4 } });
    renderTab();
    type('GOS Threshold:', '2.5');
    expect(useResultStore.getState().rxCriteria.gosThreshold).toBe('2.5');
    expect(useResultStore.getState().rxCriteriaEdited).toBe(true);
    const version = useResultStore.getState().grainsVersion;
    classify();
    await waitFor(() => expect(useResultStore.getState().rxCriteriaEdited).toBe(false));
    expect(useResultStore.getState().rxCriteria.gosThreshold).toBe('2.5');
    expect(useResultStore.getState().grainsVersion).toBe(version + 1);      // the map is fetched again with them
  });

  it('does not send a number outside the range of the backend, and says so', async () => {
    analysisApi.getMap.mockResolvedValue({ data: WITH_BC });
    renderTab();
    type('gBC Min Fraction:', '1.5');
    classify();
    expect(await screen.findByText(/Enter numbers inside the ranges/)).toBeTruthy();
    expect(analysisApi.rxClassify).not.toHaveBeenCalled();
    type('gBC Min Fraction:', '');
    classify();
    expect(analysisApi.rxClassify).not.toHaveBeenCalled();
  });

  it('shows what the backend says when it refuses (a rejected body is a list of entries, not text)', async () => {
    analysisApi.getMap.mockResolvedValue({ data: WITH_BC });
    analysisApi.rxClassify.mockRejectedValue({ response: { status: 422, data: { detail: [{ loc: ['body', 'rx_gos_threshold'], msg: 'Input should be less than or equal to 20' }] } } });
    renderTab();
    classify();
    expect(await screen.findByText(/rx_gos_threshold: Input should be less than or equal to 20/)).toBeTruthy();
  });
});

describe('The legend and the no-band-contrast text exist in all four languages', () => {
  const dir = path.resolve(__dirname, '../../../locales');
  const KEYS = ['reasonNeedsBc', 'reasonNoPixels', 'metricNotAvailable', 'nameRxFraction', 'legendTitle', 'legendRx', 'legendLowGos', 'legendLowGosOnly', 'legendSub',
    'legendDeformed', 'legendNoGrain', 'legendBands', 'legendNoRedBc', 'legendNoRedFailed'];
  it.each(['en', 'de', 'ja', 'zh'])('%s', (lang) => {
    const rx = JSON.parse(fs.readFileSync(path.join(dir, lang, 'analysis.json'), 'utf8')).rx;
    for (const k of KEYS) expect(typeof rx[k] === 'string' && rx[k].length > 0, `${lang} rx.${k}`).toBe(true);
    if (lang === 'de') {
      expect(rx.legendRx).toMatch(/rekristallisiert nach allen vier Kriterien/);
      expect(rx.legendNoRedBc).toMatch(/Bandkontrast/);
      expect(rx.legendBands).toMatch(/Kontext/);
    }
  });
});

describe('RX tab: typed criteria the map does not have yet', () => {
  it('says so while they are typed and not sent, and stops saying so once Classify has sent them', async () => {
    analysisApi.getMap.mockResolvedValue({ data: WITH_BC });
    analysisApi.rxClassify.mockResolvedValue({ data: { ...WITH_BC, rx_gos_threshold: 2.5, rx_gbc_fraction: 0.7, rx_gkam_threshold: 0.55, rx_min_radius_um: 0.4 } });
    renderTab();
    // the live region is there from the start, empty (a region inserted together with its text is often not read)
    expect(screen.getByTestId('rx-not-applied').textContent).toBe('');
    expect(screen.getByTestId('rx-not-applied').getAttribute('aria-live')).toBe('polite');
    type('GOS Threshold:', '2.5');
    expect(screen.getByTestId('rx-not-applied').textContent).toBe('Not applied to this map yet \u2014 press Classify');
    classify();
    await waitFor(() => expect(screen.getByTestId('rx-not-applied').textContent).toBe(''));
  });

  it('says nothing on a map without grains, where Classify cannot be pressed', () => {
    analysisApi.getMap.mockResolvedValue({ data: WITH_BC });
    render(<RXTab analysisLoaded hasGrains={false} grainsVersion={1} onComplete={() => {}} />);
    type('GOS Threshold:', '2.5');
    expect(useResultStore.getState().rxCriteriaEdited).toBe(true);
    expect(screen.getByTestId('rx-not-applied').textContent).toBe('');
    expect(screen.getByRole('button', { name: /classify rx grains/i }).disabled).toBe(true);
  });

  it('a Classify that could not be stored (409) says so, asks the backend what it holds, and keeps what is typed pending', async () => {
    analysisApi.getMap.mockResolvedValue({ data: WITH_BC });
    analysisApi.rxClassify.mockRejectedValue({ response: { status: 409, data: { detail: 'The map changed while classifying; the criteria were not stored. Press Classify again.' } } });
    analysisApi.status.mockResolvedValue({ data: { loaded: true, shape: [4, 5], has_grains: true, phase_options: [], rx_gos_threshold: 1.25, rx_gbc_fraction: 0.7, rx_gkam_threshold: 0.55, rx_min_radius_um: 0.4 } });
    renderTab();
    type('GOS Threshold:', '2.5');
    classify();
    expect(await screen.findByText(/Press Classify again/)).toBeTruthy();
    await waitFor(() => expect(analysisApi.status).toHaveBeenCalledTimes(1));
    expect(useResultStore.getState().rxCriteria.gosThreshold).toBe('2.5');      // never taken back by the answer
    expect(screen.getByTestId('rx-not-applied').textContent).toMatch(/Not applied/);
  });
});
