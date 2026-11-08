// @vitest-environment jsdom
//
// The crystal hint averages the patterns around a point (a radius above 0) and the backend refuses
// that on a hexagonal scan, where the (2r+1) x (2r+1) window on the index grid misses true
// neighbours. All three modes used to start at radius 1, so their first run on a hexagonal file
// failed. Real modes, real data store, real i18n; the only seam is the API.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup, act } from '@testing-library/react';

vi.mock('../../services/api', () => ({
  ebsdApi: { overview: vi.fn(() => Promise.resolve({ data: { image: '' } })) },
  crystalHintApi: {
    analyzePixel: vi.fn(),
    analyzeRegion: vi.fn(),
    qualityCheck: vi.fn(),
    externalSearch: vi.fn(() => Promise.resolve({ data: { matches: [] } })),
  },
}));
vi.mock('./SymmetryReportPanel', () => ({ default: () => null }));
vi.mock('./LatticeReportPanel', () => ({ default: () => null }));
vi.mock('./CandidateList', () => ({ default: () => null }));
vi.mock('./PatternPreview', () => ({ default: () => null }));
vi.mock('./MissingPhaseClusters', () => ({ default: () => null }));
vi.mock('./HistogramBars', () => ({ default: () => null }));

import i18n from '../../i18n';
import { crystalHintApi } from '../../services/api';
import useDataStore from '../../stores/useDataStore';
import useResultStore from '../../stores/useResultStore';
import SinglePixelMode from './SinglePixelMode';
import RegionMode from './RegionMode';
import QualityCheckMode from './QualityCheckMode';

const SQUARE = { kind: 'square', nRows: 4, nCols: 5, gridShape: [4, 5], stepX: 1, stepY: 1, stepsKnown: true, oddRowShift: 0, shiftConfirmed: true };
const HEX = { ...SQUARE, kind: 'hex', oddRowShift: 0.5 };

const MODES = {
  single: {
    element: () => <SinglePixelMode elements={['Al']} presetKey="" isActive />,
    api: 'analyzePixel',
    run: /analyse pixel/i,
  },
  region: {
    element: () => <RegionMode elements={['Al']} presetKey="" isActive />,
    api: 'analyzeRegion',
    run: /analyse region/i,
  },
  quality: {
    element: () => <QualityCheckMode isActive />,
    api: 'qualityCheck',
    run: /run quality check/i,
  },
};

const REASON = {
  en: 'hexagonal scan',
  de: 'hexagonalen Scan nicht verfügbar',
  ja: '六方格子のスキャンでは利用できません',
  zh: '六方网格扫描中不可用',
};

const avgSelect = () => screen.getAllByRole('combobox').find((s) => [...s.options].some((o) => o.value === '2'));

function refusal(code, prose) {
  return Object.assign(new Error('Request failed with status code 400'), {
    response: { status: 400, headers: { 'x-orienta-code': code }, data: { detail: prose } },
  });
}

beforeEach(() => {
  vi.clearAllMocks();
  useDataStore.setState({ isFileOpen: true, ebsdLoaded: true, gridShape: [4, 5], scanGeometry: SQUARE, currentIndex: 0 });
  useResultStore.setState({ indexingResult: { id: 'r1' } });
  for (const name of ['analyzePixel', 'analyzeRegion', 'qualityCheck']) {
    // the request is what is looked at, not a result: a plain failure keeps the page simple
    crystalHintApi[name].mockRejectedValue(new Error('no result in this test'));
  }
});
afterEach(async () => {
  cleanup();
  await act(async () => { await i18n.changeLanguage('en'); });
});

async function runOnce(mode) {
  fireEvent.click(screen.getByRole('button', { name: MODES[mode].run }));
  await waitFor(() => expect(crystalHintApi[MODES[mode].api]).toHaveBeenCalled());
  return crystalHintApi[MODES[mode].api].mock.calls.at(-1)[0];
}

describe.each(Object.keys(MODES))('%s mode', (mode) => {
  it('on a square scan starts at radius 1 and the control can be changed', async () => {
    render(MODES[mode].element());
    expect(avgSelect().value).toBe('1');
    expect(avgSelect().disabled).toBe(false);
    expect((await runOnce(mode)).avgRadius).toBe(1);
  });

  it('on a hexagonal scan runs at radius 0 by default and does not offer to raise it', async () => {
    useDataStore.setState({ scanGeometry: HEX });
    render(MODES[mode].element());
    expect(avgSelect().value).toBe('0');
    expect(avgSelect().disabled).toBe(true);
    expect((await runOnce(mode)).avgRadius).toBe(0);
  });

  it.each(Object.keys(REASON))('says why, in %s, on a hexagonal scan and not on a square one', async (lang) => {
    await act(async () => { await i18n.changeLanguage(lang); });
    useDataStore.setState({ scanGeometry: HEX });
    render(MODES[mode].element());
    const why = screen.getByTestId('hex-avg-locked');
    expect(why.textContent).toContain(REASON[lang]);
    expect(avgSelect().title).toContain(REASON[lang]);
    act(() => { useDataStore.setState({ scanGeometry: SQUARE }); });
    expect(screen.queryByTestId('hex-avg-locked')).toBeNull();
  });

  it('a switch to a square scan gives back the radius that was chosen before', async () => {
    render(MODES[mode].element());
    fireEvent.change(avgSelect(), { target: { value: '2' } });
    act(() => { useDataStore.setState({ scanGeometry: HEX }); });
    expect(avgSelect().value).toBe('0');
    expect((await runOnce(mode)).avgRadius).toBe(0);
    act(() => { useDataStore.setState({ scanGeometry: SQUARE }); });
    expect(avgSelect().value).toBe('2');
    expect((await runOnce(mode)).avgRadius).toBe(2);
  });

  it('shows the refusal of the backend in the language of the page, not its English sentence', async () => {
    crystalHintApi[MODES[mode].api].mockRejectedValue(refusal('hex_neighbourhood_unsupported', 'BACKEND ENGLISH SENTENCE'));
    render(MODES[mode].element());
    const button = screen.getByRole('button', { name: MODES[mode].run });
    await act(async () => { await i18n.changeLanguage('de'); });
    fireEvent.click(button);
    await waitFor(() => expect(document.body.textContent).toContain('sechs Nachbarn'));
    expect(document.body.textContent).not.toContain('BACKEND ENGLISH SENTENCE');
  });
});
