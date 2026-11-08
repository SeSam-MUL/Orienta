// @vitest-environment jsdom
//
// The scalar analysis maps (band contrast, KAM, GOS, ECD) are drawn with their colormap stretched between their own
// smallest and largest value, so the viewer draws a thin colour bar under the map: the colours of the colormap from the
// answer's `colormap_stops`, the smallest and largest value of `value_range` at its ends, with the unit, in three
// significant digits. The ECD map also names its basis. Maps with fixed colours have no bar.
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';

vi.mock('../../../services/api', () => ({
  analysisApi: { getMap: vi.fn(), status: vi.fn() },
  indexApi: {},
  edsApi: {},
}));

import { analysisApi } from '../../../services/api';
import useResultStore from '../../../stores/useResultStore';
import { TextureTab } from '../AnalysisPage';

const STOPS = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'];

afterEach(() => { cleanup(); vi.resetAllMocks(); useResultStore.getState().clearResults(); });

const answer = (extra) => ({ data: { image: 'IMG', map_type: 'x', shape: [2, 2], min_val: 0, max_val: 1, ...extra } });
const scalar = (min, max, unit, extra = {}) => answer({ value_range: { min, max, unit }, colormap_stops: STOPS, ...extra });

function renderViewer() {
  return render(<TextureTab analysisLoaded hasGrains grainsVersion={1} onComplete={() => {}} />);
}
const pick = (layer) => fireEvent.change(screen.getByLabelText('Layer:'), { target: { value: layer } });
const line = () => screen.getByTestId('map-scale-line').textContent;

describe('the colour bar under a scalar map', () => {
  it('draws the stops as a gradient and prints the ends with the unit', async () => {
    analysisApi.getMap.mockImplementation((type) => Promise.resolve(type === 'kam' ? scalar(0, 3.1234, '°') : answer({})));
    renderViewer();
    pick('kam');
    await waitFor(() => expect(screen.getByTestId('map-scale')).toBeTruthy());
    const bar = screen.getByTestId('map-scale-bar');
    expect(bar.getAttribute('role')).toBe('img');
    expect(bar.style.background).toContain('linear-gradient');
    STOPS.forEach((c) => {
      const [r, g, b] = [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
      expect(bar.style.background.replace(/\s/g, '')).toContain(`rgb(${r},${g},${b})`);
    });
    expect(screen.getByTestId('map-scale-min').textContent).toBe('0');
    expect(screen.getByTestId('map-scale-max').textContent).toBe('3.12 °');
    expect(line()).toBe('0 – 3.12 °');
  });

  it('writes numbers with three significant digits', async () => {
    analysisApi.getMap.mockImplementation((type) => Promise.resolve(type === 'gos' ? scalar(0.012345, 1234.5, '°') : answer({})));
    renderViewer();
    pick('gos');
    await waitFor(() => expect(screen.getByTestId('map-scale')).toBeTruthy());
    expect(screen.getByTestId('map-scale-min').textContent).toBe('0.0123');
    expect(screen.getByTestId('map-scale-max').textContent).toBe('1230 °');
  });

  it('band contrast has no unit', async () => {
    analysisApi.getMap.mockImplementation((type) => Promise.resolve(type === 'bc' ? scalar(12, 187.4, '') : answer({})));
    renderViewer();
    pick('bc');
    await waitFor(() => expect(screen.getByTestId('map-scale')).toBeTruthy());
    expect(line()).toBe('12 – 187');
    expect(screen.getByTestId('map-scale-max').textContent).toBe('187');
  });

  it('the ECD map names its basis, and the line follows the basis when the map is drawn again', async () => {
    let basis = 'cell_area';
    analysisApi.getMap.mockImplementation((type) => Promise.resolve(
      type === 'ecd' ? scalar(0.6, basis === 'cell_area' ? 41.3 : 38.9, 'µm', { ecd_basis: basis }) : answer({})));
    renderViewer();
    pick('ecd');
    await waitFor(() => expect(screen.getByTestId('map-scale')).toBeTruthy());
    expect(line()).toBe('ECD, cell area: 0.6 – 41.3 µm');
    basis = 'measured_pixels';
    fireEvent.click(screen.getByRole('button', { name: /refresh/i }));
    await waitFor(() => expect(line()).toBe('ECD, measured pixels × step²: 0.6 – 38.9 µm'));
  });

  it('the Texture Components map has a legend with one colour per component, and no colormap', async () => {
    const legend = [{ name: 'Cube', color: '#1f77b4' }, { name: 'Goss', color: '#ff7f0e' }];
    analysisApi.getMap.mockImplementation((type) => Promise.resolve(type === 'texture_component'
      ? answer({
        legend, no_component_color: '#808080', outside_color: '#000000', texture_phase: { id: 1, name: 'Al' },
        texture_preset: 'BCC_Rolling', texture_tolerance_deg: 4.5,
      })
      : answer({})));
    renderViewer();
    pick('texture_component');
    await waitFor(() => expect(screen.getByTestId('texture-legend')).toBeTruthy());
    expect(screen.getByText('Texture components of Al (BCC_Rolling, 4.5°)')).toBeTruthy();    // what it was drawn with
    const swatch = (key) => screen.getByTestId(`texture-legend-${key}`).firstChild.style.background;
    expect(swatch('Cube')).toBe('rgb(31, 119, 180)');
    expect(swatch('Goss')).toBe('rgb(255, 127, 14)');
    expect(swatch('none')).toBe('rgb(128, 128, 128)');
    expect(swatch('outside')).toBe('rgb(0, 0, 0)');
    expect(screen.getByTestId('texture-legend-none').textContent).toBe('No component within the tolerance');
    expect(screen.getByTestId('texture-legend-outside').textContent).toBe('Other phase or not indexed');
    expect(screen.getByDisplayValue('gray').disabled).toBe(true);
    expect(screen.queryByTestId('map-scale')).toBeNull();
  });

  it('the legend is gone with the layer', async () => {
    analysisApi.getMap.mockImplementation((type) => Promise.resolve(type === 'texture_component'
      ? answer({ legend: [{ name: 'Cube', color: '#1f77b4' }], no_component_color: '#808080', outside_color: '#000000' })
      : answer({})));
    renderViewer();
    pick('texture_component');
    await waitFor(() => expect(screen.getByTestId('texture-legend')).toBeTruthy());
    pick('ipf');
    await waitFor(() => expect(screen.queryByTestId('texture-legend')).toBeNull());
  });

  it.each(['ipf', 'rx', 'grain_boundaries'])('%s has fixed colours and no bar', async (layer) => {
    analysisApi.getMap.mockResolvedValue(answer({}));
    renderViewer();
    pick(layer);
    await waitFor(() => expect(analysisApi.getMap.mock.calls.some((c) => c[0] === layer)).toBe(true));
    await waitFor(() => expect(screen.getByAltText('EBSD analysis map')).toBeTruthy());
    expect(screen.queryByTestId('map-scale')).toBeNull();
  });

  it('goes away with the map when the next one cannot be drawn', async () => {
    analysisApi.getMap.mockImplementation((type) => (type === 'kam'
      ? Promise.resolve(scalar(0, 2, '°')) : Promise.reject({ response: { data: { detail: 'Reconstruct grains first' } } })));
    renderViewer();
    pick('kam');
    await waitFor(() => expect(screen.getByTestId('map-scale')).toBeTruthy());
    pick('gos');
    await waitFor(() => expect(screen.getByText(/Reconstruct grains first/)).toBeTruthy());
    expect(screen.queryByTestId('map-scale')).toBeNull();
  });
});
