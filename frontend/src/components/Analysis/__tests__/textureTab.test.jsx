// @vitest-environment jsdom
//
// The Texture tab: the phase the texture is computed on (a select, sent with the run, named in the result) and the
// surface planes (the family value as the main number, the MATLAB script's single-direction number beside it).
// Real component, only the HTTP layer mocked, with the shape of the real answer of POST /api/analysis/texture.
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup, within } from '@testing-library/react';

vi.mock('../../../services/api', () => ({
  analysisApi: { getMap: vi.fn(), texture: vi.fn(), textureComponents: vi.fn(), status: vi.fn() },
  indexApi: {},
  edsApi: {},
}));

import { analysisApi } from '../../../services/api';
import useResultStore from '../../../stores/useResultStore';
import { TextureTab } from '../AnalysisPage';

const ANSWER = {
  success: true,
  texture_phase: { id: 1, name: 'Al' },
  components: [{ name: 'Cube', fraction: 12.5 }, { name: 'Goss', fraction: 0.4 }, { name: 'Brass', fraction: 1 }],
  texture_index: 4.2762,
  texture_entropy: -0.8025,
  surface_planes: { '{111}': 8.6, '{100}': 0, '{110}': 0 },
  surface_planes_family: { '{111}': 31.2, '{100}': 20.4, '{110}': 55.5 },
};

afterEach(() => { cleanup(); vi.resetAllMocks(); useResultStore.getState().clearResults(); });

function renderTab() {
  analysisApi.getMap.mockImplementation(() => new Promise(() => {}));
  useResultStore.setState({ phaseOptions: [{ id: 0, name: 'Si' }, { id: 1, name: 'Al' }, { id: 2, name: 'Mg2' }] });
  return render(<TextureTab analysisLoaded hasGrains grainsVersion={1} onComplete={() => {}} />);
}
const analyze = () => fireEvent.click(screen.getByRole('button', { name: /analyze texture/i }));

describe('Texture tab: the phase', () => {
  it('offers the phase with the most indexed pixels first, then the phases of the map, and says what it is for', () => {
    renderTab();
    const select = screen.getByLabelText('Phase:');
    expect(Array.from(select.options).map((o) => [o.value, o.textContent])).toEqual([
      ['most_measured', 'Phase with the most indexed pixels'], ['0', 'Si'], ['1', 'Al'], ['2', 'Mg2'],
    ]);
    expect(select.value).toBe('most_measured');
    expect(screen.getByText(/The texture is computed on this one phase\. The components are cubic/)).toBeTruthy();
  });

  it('sends "most_measured" by default and the id of the chosen phase otherwise', async () => {
    analysisApi.texture.mockResolvedValue({ data: ANSWER });
    renderTab();
    analyze();
    await waitFor(() => expect(analysisApi.texture).toHaveBeenCalledTimes(1));
    expect(analysisApi.texture.mock.calls[0]).toEqual(['most_measured', 'FCC_Rolling', 10]);
    fireEvent.change(screen.getByLabelText('Phase:'), { target: { value: '0' } });
    expect(useResultStore.getState().texturePhase).toBe(0);
    analyze();
    await waitFor(() => expect(analysisApi.texture).toHaveBeenCalledTimes(2));
    expect(analysisApi.texture.mock.calls[1]).toEqual([0, 'FCC_Rolling', 10]);
    fireEvent.change(screen.getByLabelText('Phase:'), { target: { value: 'most_measured' } });
    expect(useResultStore.getState().texturePhase).toBe('most_measured');
  });

  it('starts at the choice the store holds (the backend\'s last choice comes back with the status)', () => {
    useResultStore.setState({ texturePhase: 2 });
    renderTab();
    expect(screen.getByLabelText('Phase:').value).toBe('2');
  });

  it('marks the phases that are not cubic and does not let them be chosen', () => {
    analysisApi.getMap.mockImplementation(() => new Promise(() => {}));
    useResultStore.setState({ phaseOptions: [
      { id: 0, name: 'Si', cubic: true }, { id: 1, name: 'Al', cubic: true }, { id: 2, name: 'Mg2', cubic: false }] });
    render(<TextureTab analysisLoaded hasGrains grainsVersion={1} onComplete={() => {}} />);
    const options = Array.from(screen.getByLabelText('Phase:').options);
    expect(options.map((o) => [o.value, o.textContent, o.disabled])).toEqual([
      ['most_measured', 'Phase with the most indexed pixels', false],
      ['0', 'Si', false], ['1', 'Al', false], ['2', 'Mg2 (not cubic)', true],
    ]);
  });

  it('after a run the map on screen is fetched again (it was drawn with the earlier preset and tolerance)', async () => {
    analysisApi.texture.mockResolvedValue({ data: ANSWER });
    function InPage() {                                  // the page hands the store's grainsVersion to the tab
      const version = useResultStore((state) => state.grainsVersion);
      return <TextureTab analysisLoaded hasGrains grainsVersion={version} onComplete={() => {}} />;
    }
    analysisApi.getMap.mockImplementation(() => new Promise(() => {}));
    useResultStore.setState({ phaseOptions: [{ id: 0, name: 'Si' }, { id: 1, name: 'Al' }] });
    render(<InPage />);
    await waitFor(() => expect(analysisApi.getMap).toHaveBeenCalled());
    const before = analysisApi.getMap.mock.calls.length;
    const version = useResultStore.getState().grainsVersion;
    analyze();
    await waitFor(() => expect(useResultStore.getState().grainsVersion).toBe(version + 1));
    expect(analysisApi.getMap.mock.calls.length).toBeGreaterThan(before);
  });

  it('a run applies the pick: the store no longer holds it as pending', async () => {
    analysisApi.texture.mockResolvedValue({ data: ANSWER });
    renderTab();
    fireEvent.change(screen.getByLabelText('Phase:'), { target: { value: '0' } });
    expect(useResultStore.getState().textureEdited).toBe(true);
    analyze();
    await waitFor(() => expect(useResultStore.getState().textureEdited).toBe(false));
    expect(useResultStore.getState().texturePhase).toBe(0);
  });

  it('shows the component fractions as the percent they are, also below 1 %', async () => {
    analysisApi.texture.mockResolvedValue({ data: ANSWER });
    renderTab();
    analyze();
    await waitFor(() => expect(screen.getByText('Cube')).toBeTruthy());
    const row = (name) => screen.getByText(name).parentElement.textContent;
    expect(row('Cube')).toBe('Cube12.5%');
    expect(row('Goss')).toBe('Goss0.4%');           // not 40.0 %
    expect(row('Brass')).toBe('Brass1.0%');         // not 100.0 %
  });

  it('names the phase the numbers are of', async () => {
    analysisApi.texture.mockResolvedValue({ data: ANSWER });
    renderTab();
    expect(screen.queryByTestId('texture-phase')).toBeNull();
    analyze();
    await waitFor(() => expect(screen.getByTestId('texture-phase').textContent).toBe('Texture of: Al'));
  });

  it('shows the backend\'s reason when the phase is refused (not cubic, unknown)', async () => {
    const detail = 'Texture phase 2 (Mg2) is not cubic (Laue class 6/mmm); the texture components are cubic. Choose a cubic phase: the cubic phases of this map are 0 (Si), 1 (Al).';
    analysisApi.texture.mockRejectedValue({ response: { status: 400, data: { detail } } });
    renderTab();
    fireEvent.change(screen.getByLabelText('Phase:'), { target: { value: '2' } });
    analyze();
    await waitFor(() => expect(screen.getByText(detail)).toBeTruthy());
    expect(screen.queryByTestId('texture-phase')).toBeNull();
  });

  it('a new run replaces the numbers of the earlier one, also when the new answer has none', async () => {
    analysisApi.texture.mockResolvedValueOnce({ data: ANSWER });
    analysisApi.texture.mockResolvedValueOnce({
      data: { ...ANSWER, texture_phase: { id: 0, name: 'Si' }, texture_index: undefined, texture_entropy: undefined },
    });
    renderTab();
    analyze();
    await waitFor(() => expect(screen.getByText('4.276')).toBeTruthy());
    analyze();
    await waitFor(() => expect(screen.getByTestId('texture-phase').textContent).toBe('Texture of: Si'));
    expect(screen.queryByText('4.276')).toBeNull();
  });
});

describe('Texture tab: preset and tolerance angle', () => {
  const tolerance = () => screen.getByTitle(/Maximum misorientation/);

  it('sends the preset and the tolerance angle with the run, and the store keeps them as pending until then', async () => {
    analysisApi.texture.mockResolvedValue({ data: ANSWER });
    analysisApi.textureComponents.mockResolvedValue({ data: { components: [] } });
    renderTab();
    fireEvent.change(screen.getByTitle(/Choose the texture-component preset/), { target: { value: 'BCC_Rolling' } });
    fireEvent.change(tolerance(), { target: { value: '4' } });
    expect(useResultStore.getState().textureEdited).toBe(true);
    analyze();
    await waitFor(() => expect(analysisApi.texture).toHaveBeenCalledTimes(1));
    expect(analysisApi.texture.mock.calls[0]).toEqual(['most_measured', 'BCC_Rolling', 4]);
    await waitFor(() => expect(useResultStore.getState().textureEdited).toBe(false));
    expect(useResultStore.getState().texturePreset).toBe('BCC_Rolling');
    expect(useResultStore.getState().textureTolerance).toBe('4');
  });

  it.each(['', '0', '-3', '31', 'abc'])('a tolerance of "%s" cannot be run: the button is off and the field says why', (typed) => {
    renderTab();
    fireEvent.change(tolerance(), { target: { value: typed } });
    expect(screen.getByRole('button', { name: /analyze texture/i }).disabled).toBe(true);
    expect(screen.getByTestId('texture-tolerance-invalid').textContent).toBe('Tolerance angle: more than 0° and at most 30°.');
    analyze();
    expect(analysisApi.texture).not.toHaveBeenCalled();
  });

  it('30 degrees is allowed', () => {
    renderTab();
    fireEvent.change(tolerance(), { target: { value: '30' } });
    expect(screen.getByRole('button', { name: /analyze texture/i }).disabled).toBe(false);
    expect(screen.queryByTestId('texture-tolerance-invalid')).toBeNull();
  });

  it('shows the preset and the tolerance the store holds', () => {
    useResultStore.setState({ texturePreset: 'BCC_Rolling', textureTolerance: '6.5' });
    renderTab();
    expect(screen.getByTitle(/Choose the texture-component preset/).value).toBe('BCC_Rolling');
    expect(tolerance().value).toBe('6.5');
  });
});

describe('Texture tab: the surface planes', () => {
  it('shows the family value as the main number and the script\'s single-direction number beside it, for each plane', async () => {
    analysisApi.texture.mockResolvedValue({ data: ANSWER });
    renderTab();
    analyze();
    await waitFor(() => expect(screen.getByTestId('plane-{111}')).toBeTruthy());
    const rows = {
      '{111}': ['31.2 %', 'as the MATLAB script, one direction (1 1 1): 8.6 %'],
      '{100}': ['20.4 %', 'as the MATLAB script, one direction (1 0 0): 0.0 %'],
      '{110}': ['55.5 %', 'as the MATLAB script, one direction (1 1 0): 0.0 %'],
    };
    Object.entries(rows).forEach(([key, [family, script]]) => {
      const row = screen.getByTestId(`plane-${key}`);
      expect(within(row).getByText(`${key} planes in the surface (any plane of the family, ±15°)`)).toBeTruthy();
      expect(within(row).getByText(family)).toBeTruthy();
      expect(within(row).getByText(script)).toBeTruthy();
    });
  });

  it('explains in a tooltip on the script\'s number why {100} and {110} read near 0 %', async () => {
    analysisApi.texture.mockResolvedValue({ data: ANSWER });
    renderTab();
    analyze();
    await waitFor(() => expect(screen.getByTestId('plane-{100}')).toBeTruthy());
    const script = within(screen.getByTestId('plane-{100}')).getByText(/as the MATLAB script/);
    expect(script.getAttribute('title')).toBe(
      'The MATLAB script measures one direction per pixel, so the number depends on which of the symmetric equivalents the file stores; on Aztec data {100} and {110} come out near 0 %.');
  });

  it('without family values (an older answer) the main number is a dash and the script\'s number stays', async () => {
    const { surface_planes_family: _gone, ...old } = ANSWER;
    analysisApi.texture.mockResolvedValue({ data: old });
    renderTab();
    analyze();
    await waitFor(() => expect(screen.getByTestId('plane-{111}')).toBeTruthy());
    const row = screen.getByTestId('plane-{111}');
    expect(within(row).getByText('—')).toBeTruthy();
    expect(within(row).getByText('as the MATLAB script, one direction (1 1 1): 8.6 %')).toBeTruthy();
  });
});
