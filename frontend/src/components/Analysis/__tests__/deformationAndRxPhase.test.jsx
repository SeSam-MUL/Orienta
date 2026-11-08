// @vitest-environment jsdom
//
// The Deformation tab sends its three parameters (KAM order, KAM threshold, HAGB limit) and says which ones the numbers
// were made with; the RX tab chooses the phase it works on and sends it with the criteria. Both keep a typed value
// pending until a run has sent it. A rejected request body (422 with a LIST as `detail`) shows a message and does not
// take the page down. Real components, only the HTTP layer mocked, with the shapes of the real answers.
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import fs from 'node:fs';
import path from 'node:path';

vi.mock('../../../services/api', () => ({
  analysisApi: {
    getMap: vi.fn(), deformation: vi.fn(), rxClassify: vi.fn(), texture: vi.fn(), textureComponents: vi.fn(), status: vi.fn(),
  },
  indexApi: {},
  edsApi: {},
}));

import { analysisApi } from '../../../services/api';
import useResultStore from '../../../stores/useResultStore';
import { DeformationTab, RXTab, TextureTab } from '../AnalysisPage';

afterEach(() => { cleanup(); vi.resetAllMocks(); useResultStore.getState().clearResults(); });

const LIST_422 = (field, msg) => ({ response: { status: 422, data: { detail: [{ type: 'greater_than', loc: ['body', field], msg, input: 0 }] } } });
const state = () => useResultStore.getState();

describe('Deformation tab: the three parameters reach the backend', () => {
  const renderTab = (props = {}) => {
    analysisApi.getMap.mockImplementation(() => new Promise(() => {}));
    return render(<DeformationTab analysisLoaded hasGrains grainsVersion={1} onComplete={() => {}} {...props} />);
  };
  const analyze = () => fireEvent.click(screen.getByRole('button', { name: /analyze deformation/i }));
  const field = (label) => screen.getByLabelText(label);

  it('starts with the values of the script, names its fields, and says what the SAGB range is', () => {
    renderTab();
    expect([field('KAM order:'), field('KAM threshold:'), field('HAGB from:')].map((i) => i.value)).toEqual(['2', '7.0', '15.0']);
    expect(screen.getByTestId('deformation-sagb-help').textContent).toBe('SAGB: 3° to the HAGB limit.');
  });

  it('sends the numbers of the fields with "Analyze Deformation" (no body would run with the backend\'s old ones)', async () => {
    analysisApi.deformation.mockResolvedValue({ data: {} });
    renderTab();
    fireEvent.change(field('KAM order:'), { target: { value: '3' } });
    fireEvent.change(field('KAM threshold:'), { target: { value: '5' } });
    fireEvent.change(field('HAGB from:'), { target: { value: '20' } });
    analyze();
    await waitFor(() => expect(analysisApi.deformation).toHaveBeenCalledTimes(1));
    expect(analysisApi.deformation).toHaveBeenCalledWith({ kamOrder: 3, kamThreshold: 5, hagbMin: 20 });
  });

  it('a value outside the backend\'s range is refused before it is sent, and the field says so', () => {
    renderTab();
    fireEvent.change(field('KAM order:'), { target: { value: '2.5' } });
    expect(screen.getByTestId('deformation-invalid').textContent).toMatch(/KAM order a whole number 1–5/);
    expect(screen.getByRole('button', { name: /analyze deformation/i }).disabled).toBe(true);
    fireEvent.change(field('KAM order:'), { target: { value: '2' } });
    fireEvent.change(field('HAGB from:'), { target: { value: '3' } });       // the limit is more than 3
    expect(screen.getByRole('button', { name: /analyze deformation/i }).disabled).toBe(true);
    fireEvent.change(field('HAGB from:'), { target: { value: '15' } });
    fireEvent.change(field('KAM threshold:'), { target: { value: '31' } });
    expect(screen.getByRole('button', { name: /analyze deformation/i }).disabled).toBe(true);
    expect(analysisApi.deformation).not.toHaveBeenCalled();
  });

  it('a typed parameter is pending until a run has sent it; the numbers say what they were made with', async () => {
    analysisApi.deformation.mockResolvedValue({
      data: { mean_kam: 0.5, deformation_params: { kam_order: 3, kam_threshold_deg: 7, hagb_min_deg: 15 } },
    });
    renderTab();
    expect(screen.getByTestId('deformation-not-applied').textContent).toBe('');
    fireEvent.change(field('KAM order:'), { target: { value: '3' } });
    expect(screen.getByTestId('deformation-not-applied').textContent).toMatch(/Not applied to this map yet/);
    analyze();
    await waitFor(() => expect(screen.getByTestId('deformation-params-used')).toBeTruthy());
    expect(screen.getByTestId('deformation-params-used').textContent).toBe('Made with: KAM order 3, KAM threshold 7°, HAGB from 15°');
    expect(screen.getByTestId('deformation-not-applied').textContent).toBe('');
    expect(state().deformationEdited).toBe(false);
  });

  it('a disabled button says why', () => {
    renderTab({ hasGrains: false });
    expect(screen.getByRole('button', { name: /analyze deformation/i }).title).toBe('Reconstruct grains first (Grain Analysis tab).');
  });

  it('a 422 with a list as detail shows a message', async () => {
    analysisApi.deformation.mockRejectedValue(LIST_422('kam_order', 'Input should be less than or equal to 5'));
    renderTab();
    analyze();
    await waitFor(() => expect(screen.getByText('kam_order: Input should be less than or equal to 5')).toBeTruthy());
  });
});

describe('RX tab: the phase', () => {
  const PHASES = [{ id: 1, name: 'Aluminium' }, { id: 2, name: 'Si' }];
  const renderTab = () => {
    analysisApi.getMap.mockImplementation(() => new Promise(() => {}));
    useResultStore.setState({ phaseOptions: PHASES });
    return render(<RXTab analysisLoaded hasGrains grainsVersion={1} onComplete={() => {}} />);
  };
  const classify = () => fireEvent.click(screen.getByRole('button', { name: /classify rx grains/i }));
  const select = () => screen.getByLabelText('Phase:');

  it('offers the phases of the map and the phase with the most indexed pixels, which is the default', () => {
    renderTab();
    expect(Array.from(select().options).map((o) => [o.value, o.textContent])).toEqual([
      ['most_measured', 'Phase with the most indexed pixels'], ['1', 'Aluminium'], ['2', 'Si'],
    ]);
    expect(select().value).toBe('most_measured');
  });

  it('sends the chosen phase with the criteria, says which phase the numbers are of, and is no longer pending', async () => {
    analysisApi.rxClassify.mockResolvedValue({ data: { rx_fraction: 3, rx_phase: { id: 2, name: 'Si' } } });
    renderTab();
    fireEvent.change(select(), { target: { value: '2' } });
    expect(screen.getByTestId('rx-not-applied').textContent).toMatch(/Not applied to this map yet/);
    classify();
    await waitFor(() => expect(analysisApi.rxClassify).toHaveBeenCalledTimes(1));
    expect(analysisApi.rxClassify).toHaveBeenCalledWith({ gosThreshold: 1.25, gbcFraction: 0.7, gkamThreshold: 0.55, minRadius: 0.4 }, 2);
    await waitFor(() => expect(screen.getByTestId('rx-phase-result').textContent).toBe('RX of: Si'));
    expect(screen.getByTestId('rx-not-applied').textContent).toBe('');
    expect(state().rxPhase).toBe(2);
  });

  it('a 422 with a list as detail shows a message', async () => {
    analysisApi.rxClassify.mockRejectedValue(LIST_422('rx_gos_threshold', 'Input should be less than or equal to 20'));
    renderTab();
    classify();
    await waitFor(() => expect(screen.getByText('rx_gos_threshold: Input should be less than or equal to 20')).toBeTruthy());
  });
});

describe('Texture tab', () => {
  const renderTab = (props = {}) => {
    analysisApi.getMap.mockImplementation(() => new Promise(() => {}));
    return render(<TextureTab analysisLoaded hasGrains grainsVersion={1} onComplete={() => {}} {...props} />);
  };

  it('a pick that no run has applied is shown as pending, like the RX criteria', async () => {
    analysisApi.texture.mockResolvedValue({ data: { texture_phase: { id: 1, name: 'Aluminium' }, texture_preset: 'BCC_Rolling', texture_tolerance_deg: 10 } });
    renderTab();
    expect(screen.getByTestId('texture-not-applied').textContent).toBe('');
    fireEvent.change(screen.getByLabelText('Tolerance Angle:'), { target: { value: '12' } });
    expect(screen.getByTestId('texture-not-applied').textContent).toMatch(/Not applied to this map yet/);
    fireEvent.click(screen.getByRole('button', { name: /analyze texture/i }));
    await waitFor(() => expect(screen.getByTestId('texture-made-with')).toBeTruthy());
    expect(screen.getByTestId('texture-not-applied').textContent).toBe('');
  });

  it('needs no grains: the texture runs on the pixels of the phase, and the button says why it is disabled when it is', async () => {
    analysisApi.texture.mockResolvedValue({ data: { texture_phase: { id: 1, name: 'Aluminium' } } });
    renderTab({ hasGrains: false });
    const button = screen.getByRole('button', { name: /analyze texture/i });
    expect(button.disabled).toBe(false);
    fireEvent.click(button);
    await waitFor(() => expect(analysisApi.texture).toHaveBeenCalledTimes(1));
    fireEvent.change(screen.getByLabelText('Tolerance Angle:'), { target: { value: '45' } });
    expect(button.disabled).toBe(true);
    expect(button.title).toBe('Tolerance angle: more than 0° and at most 30°.');
  });

  it('a 422 with a list as detail shows a message', async () => {
    analysisApi.texture.mockRejectedValue(LIST_422('tolerance_deg', 'Input should be greater than 0'));
    renderTab();
    fireEvent.click(screen.getByRole('button', { name: /analyze texture/i }));
    await waitFor(() => expect(screen.getByText('tolerance_deg: Input should be greater than 0')).toBeTruthy());
  });
});

describe('the page never puts the raw `detail` of an error into a message', () => {
  // A FastAPI 422 has a LIST of objects as `detail`; as a React child it takes the whole page down. Every place that makes
  // a message from an error goes through errorMessageFrom (the blob request has its own reader).
  it('AnalysisPage.jsx has no `response.data.detail` outside the reader of the blob', () => {
    const source = fs.readFileSync(path.resolve(__dirname, '../AnalysisPage.jsx'), 'utf8');
    const raw = source.split('\n').filter((line) => /response\??\.data\??\.detail/.test(line));
    expect(raw).toEqual([]);
  });
});
