// @vitest-environment jsdom
/**
 * The EDS page waits for the spectra, and says so.
 *
 * A file with a spectrum cube serves no EDS numbers until its spectra are
 * evaluated; every EDS-number endpoint answers 503 meanwhile. Showing the tabs
 * then would mean a page full of refusals -- and, worse, the hooks that load
 * the element list and the default layers would load them EMPTY on the 503
 * and never ask again. So while the fit runs the page shows only its progress,
 * and the tabs mount when the numbers exist.
 *
 * Everything heavy on the page is stubbed: this is about the gate and the
 * notices around it, not about the maps.
 */
import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  render, screen, cleanup, act, fireEvent, waitFor,
} from '@testing-library/react';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k, o) => (o ? `${k}:${JSON.stringify(o)}` : k) }),
}));

const spectrumFitStatus = vi.fn();
vi.mock('../../services/api', () => ({
  edsApi: {
    spectrumFitStatus: (...a) => spectrumFitStatus(...a),
    quantifyPixel: vi.fn(),
  },
}));

const STORE = {
  isFileOpen: true, filePath: '/data/SampleB.h5oina', stepSize: null,
  pixelSizes: null, setPendingPhaseMapIndexing: () => {},
};
vi.mock('../../stores/useDataStore', () => ({ default: (sel) => sel(STORE) }));
vi.mock('../../stores/useResultStore', () => ({
  default: (sel) => sel({ indexingResult: null }),
}));

const addLayer = vi.fn();
const stackStub = {
  layers: [], bitmaps: new Map(), bitmapVersion: 0, errors: new Map(),
  shape: null, layerSources: new Map(), layerCropStatus: new Map(),
  addLayer: (...a) => addLayer(...a), removeLayer: vi.fn(), setOpacity: vi.fn(),
  setBlend: vi.fn(), setVisibility: vi.fn(), reorder: vi.fn(),
  setThreshold: vi.fn(), addMaskFromThreshold: vi.fn(), clear: vi.fn(),
};
vi.mock('./hooks/useEdsLayerStack', () => ({ useEdsLayerStack: () => stackStub }));
// The body's first hook. Recording how many status answers had arrived when
// it first ran is how "the body never mounted before ready" is measured.
const bodyMounts = [];
vi.mock('./hooks/useDefaultLayers', () => ({
  useDefaultLayers: () => {
    bodyMounts.push(spectrumFitStatus.mock.calls.length);
    return {
      ready: true, layers: [], elements: ['Al Kα1', 'Si Kα1'], electronImages: [],
    };
  },
}));
vi.mock('./hooks/useActiveDatasetKey', () => ({
  useActiveDatasetKey: () => null, DATASET_UNKNOWN: Symbol('unknown'),
}));
vi.mock('./hooks/useHoverProbe', () => ({
  useHoverProbe: () => ({ probe: null, error: null, requestProbe: () => {}, clear: () => {} }),
}));
vi.mock('./hooks/useLinescan', () => ({
  useLinescan: () => ({ data: null, fetchProfile: () => {} }),
}));
vi.mock('./hooks/useSuggestPhases', () => ({
  useSuggestPhases: () => ({
    suggestions: null, pixel: null, atomicPct: null, mapPhase: null,
    librarySource: null, librarySize: null, cifNoMatch: null,
    librarySkipped: [], loading: false, error: null,
    run: () => {}, followPixel: () => {},
  }),
}));
// The Region Average answer the panel shows (null: nothing computed yet).
let REGION_DATA = null;
vi.mock('./hooks/useRegionAverage', () => ({
  useRegionAverage: () => ({
    rowStart: 0, setRowStart: () => {}, rowEnd: 0, setRowEnd: () => {},
    colStart: 0, setColStart: () => {}, colEnd: 0, setColEnd: () => {},
    data: REGION_DATA, loading: false, error: null, quantify: () => {},
    onRegionSelected: () => {},
  }),
}));
vi.mock('./hooks/useRegionInspector', () => ({
  default: () => ({ detail: null, loading: false, error: null, inspectPixel: () => {} }),
}));
vi.mock('./useMapBackground', () => ({
  default: () => ({ shape: null }), fovRatio: () => null, BG_NONE: 'none',
}));
vi.mock('./PhaseMapPanel', () => ({
  usePhaseMap: () => ({
    phaseMap: null, mapView: 'phases', selectedRegionId: null,
    setSelectedRegionId: () => {}, mapVersion: 0, regionDefs: [], rules: null,
  }),
  PhaseMapCanvas: () => null,
  PhaseMapControls: () => <div data-testid="phase-map-controls" />,
}));
vi.mock('./OverlayCard', () => ({ default: () => <div data-testid="overlay-card" /> }));
vi.mock('./SwipeCompareController', () => ({ default: () => null }));
vi.mock('./TileGrid', () => ({ default: () => null }));
vi.mock('../PhaseMap/LayerStackPanel', () => ({ default: () => null }));
vi.mock('../common/FileSwitcher', () => ({ default: () => <div data-testid="file-switcher" /> }));
vi.mock('./HoverProbeOverlay', () => ({ default: () => null }));
vi.mock('./ThresholdHistogram', () => ({ default: () => null }));
vi.mock('./LinescanProfilePlot', () => ({ default: () => null }));
vi.mock('../common/ContextMenu', () => ({ default: () => null }));
vi.mock('../common/ImageExportDialog', () => ({ default: () => null }));
vi.mock('./RegionInspector', () => ({ default: () => null }));
vi.mock('./PhaseRules', () => ({ default: () => null }));
vi.mock('./RegionDefs', () => ({ default: () => null, seedDefFromRegion: () => null }));

import EDSPage from './EDSPage';

const answer = (state, extra = {}) => ({
  data: { state, progress: 0, stage: null, error: null, elapsed_s: 0.1,
          fit: null, ...extra },
});
const FIT = {
  model_version: 'eds-spectrum-fit/1', chi2_red: 5.18,
  lines: { Al: { label: 'Al Kα1', family: 'K' }, Si: { label: 'Si Kα1', family: 'K' } },
  detected_not_exported: [{ element: 'Si', family: 'K', z: 8.2,
    label: 'Si Kα1', at_pct_map_sum: 0.4 }],
  quality_warnings: [],
};

const settle = () => act(async () => {
  for (let i = 0; i < 4; i += 1) await Promise.resolve();
});
const tabButton = (key) => screen.queryByRole('button', { name: key });

beforeEach(() => {
  cleanup();
  spectrumFitStatus.mockReset();
  addLayer.mockReset();
  bodyMounts.length = 0;
  REGION_DATA = null;
});

/** No answer at all: a timeout or a dropped connection. */
const noAnswer = () => Object.assign(new Error('timeout of 4000ms exceeded'),
  { code: 'ECONNABORTED' });
afterEach(cleanup);

describe('EDSPage while the spectrum fit runs', () => {
  it('shows the progress line, not the tabs', async () => {
    spectrumFitStatus.mockResolvedValue(answer('running', { progress: 0.42 }));
    render(<EDSPage />);
    await settle();
    expect(screen.getByText('spectrumFit.running:{"percent":42}')).toBeTruthy();
    expect(screen.getByRole('progressbar')).toBeTruthy();
    expect(tabButton('tabs.elements')).toBeNull();
    expect(tabButton('tabs.phaseMap')).toBeNull();
    // The way out stays: switching to another file must not wait for this one.
    expect(screen.getByTestId('file-switcher')).toBeTruthy();
  });

  it('says so when the fit runs inside the backend (its process could not start)', async () => {
    spectrumFitStatus.mockResolvedValue(answer('running', {
      progress: 0.3, runner: 'in_process_fallback',
      runner_reason: 'its own process could not be started (OSError: no python)',
    }));
    render(<EDSPage />);
    await settle();
    const note = document.querySelector('[data-spectrum-fit-in-process]');
    expect(note, 'the stall it brings back has to be explained').toBeTruthy();
    expect(note.textContent).toContain('spectrumFit.inProcess');
    expect(note.textContent).toContain('OSError: no python');
  });

  it('says nothing about where the fit runs when it runs in its own process', async () => {
    spectrumFitStatus.mockResolvedValue(answer('running', {
      progress: 0.3, runner: 'child', runner_reason: null }));
    render(<EDSPage />);
    await settle();
    expect(document.querySelector('[data-spectrum-fit-in-process]')).toBeNull();
  });

  it('shows the tabs when the fit is ready, without the progress line', async () => {
    spectrumFitStatus.mockResolvedValue(answer('ready', { progress: 1, fit: FIT }));
    render(<EDSPage />);
    await settle();
    expect(tabButton('tabs.elements')).toBeTruthy();
    expect(tabButton('tabs.phaseMap')).toBeTruthy();
    expect(screen.queryByRole('progressbar')).toBeNull();
  });

  it('turns from progress into the tabs by itself when the fit finishes', async () => {
    spectrumFitStatus
      .mockResolvedValueOnce(answer('running', { progress: 0.5 }))
      .mockResolvedValue(answer('ready', { progress: 1, fit: FIT }));
    render(<EDSPage />);
    await settle();
    expect(tabButton('tabs.elements')).toBeNull();
    await waitFor(() => expect(tabButton('tabs.elements')).toBeTruthy(),
      { timeout: 3000 });
  });

  it('says it is checking while the first answer is slow -- and shows no tabs', async () => {
    spectrumFitStatus.mockReturnValue(new Promise(() => {}));
    render(<EDSPage />);
    await settle();
    expect(tabButton('tabs.elements')).toBeNull();
    const line = document.querySelector('[data-spectrum-fit-checking]');
    expect(line, 'a bare header while the backend is slow says nothing').toBeTruthy();
    expect(line.textContent).toContain('spectrumFit.checking');
    expect(bodyMounts).toEqual([]);
  });

  it('a status request that fails first is retried; the body mounts only on ready', async () => {
    spectrumFitStatus
      .mockRejectedValueOnce(noAnswer())
      .mockResolvedValueOnce(answer('running', { progress: 0.4 }))
      .mockResolvedValue(answer('ready', { progress: 1, fit: FIT }));
    render(<EDSPage />);
    await settle();
    // Unanswered: still checking, and saying the program is not answering.
    expect(tabButton('tabs.elements')).toBeNull();
    expect(document.querySelector('[data-spectrum-fit-unreachable]')).toBeTruthy();
    await waitFor(() => expect(screen.getByRole('progressbar')).toBeTruthy(),
      { timeout: 3000 });
    expect(tabButton('tabs.elements')).toBeNull();
    expect(document.querySelector('[data-spectrum-fit-unreachable]')).toBeNull();
    await waitFor(() => expect(tabButton('tabs.elements')).toBeTruthy(),
      { timeout: 3000 });
    // The body's hooks first ran after the third answer -- the "ready" one --
    // never on the error and never while the fit ran.
    expect(bodyMounts.length).toBeGreaterThan(0);
    expect(bodyMounts[0]).toBe(3);
  });

  it('a running fit whose backend stops answering keeps the progress and says why it stalls', async () => {
    spectrumFitStatus
      .mockResolvedValueOnce(answer('running', { progress: 0.6 }))
      .mockRejectedValue(noAnswer());
    render(<EDSPage />);
    await settle();
    expect(screen.getByRole('progressbar')).toBeTruthy();
    await waitFor(() => expect(
      document.querySelector('[data-spectrum-fit-unreachable]')).toBeTruthy(),
    { timeout: 3000 });
    expect(document.querySelector('[data-spectrum-fit-unreachable]').textContent)
      .toContain('spectrumFit.unreachable');
    expect(screen.getByRole('progressbar')).toBeTruthy();
    expect(tabButton('tabs.elements')).toBeNull();
    expect(bodyMounts).toEqual([]);
  });

  it('a file without spectra is the page as it always was', async () => {
    spectrumFitStatus.mockResolvedValue(answer('none'));
    render(<EDSPage />);
    await settle();
    expect(tabButton('tabs.elements')).toBeTruthy();
    expect(document.querySelector('[data-spectrum-fit-notice]')).toBeNull();
    expect(document.querySelector('[data-spectrum-fit-failed]')).toBeNull();
  });
});

describe('EDSPage after the fit', () => {
  it('a failed fit is said, and the rest of the page stays usable', async () => {
    spectrumFitStatus.mockResolvedValue(answer('failed', { error: 'cube unreadable' }));
    render(<EDSPage />);
    await settle();
    const alert = document.querySelector('[data-spectrum-fit-failed]');
    expect(alert).toBeTruthy();
    expect(alert.getAttribute('role')).toBe('alert');
    expect(alert.textContent).toContain('spectrumFit.failed');
    expect(alert.textContent).toContain('cube unreadable');
    expect(tabButton('tabs.elements')).toBeTruthy();
  });

  it('names the detected element and "Add as map" puts its map on the overlay', async () => {
    spectrumFitStatus.mockResolvedValue(answer('ready', { progress: 1, fit: FIT }));
    render(<EDSPage />);
    await settle();
    const notice = document.querySelector('[data-spectrum-fit-notice]');
    expect(notice).toBeTruthy();
    expect(notice.textContent).toContain('Si K (8σ, 0.40 at%)');

    // From the other tab: the map lands on the element tab, and that is
    // where the page goes, or the click would appear to do nothing.
    fireEvent.click(tabButton('tabs.phaseMap'));
    expect(screen.queryByTestId('overlay-card')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: /spectrumFit\.addMap/ }));
    expect(addLayer).toHaveBeenCalledTimes(1);
    expect(addLayer.mock.calls[0][0]).toMatchObject({
      id: 'eds-Si Kα1', kind: 'eds-element', element: 'Si Kα1',
    });
    expect(screen.getByTestId('overlay-card')).toBeTruthy();
  });
});

/** `/region-quantify` answers (routes/eds.py), one per source. */
const STAT = (mean) => ({ mean, std: 1, min: mean - 2, max: mean + 2 });
const SUMMED_REGION = {
  row_start: 40, row_end: 49, col_start: 50, col_end: 69, n_pixels: 200,
  data: { Al: { counts: STAT(800), wt_pct: STAT(60), at_pct: STAT(67.5) } },
  composition_method: 'summed_net_counts',
  quantification: { source: 'spectrum' },
};
const MEAN_REGION = {
  ...SUMMED_REGION,
  composition_method: 'mean_of_pixel_at_pct',
  quantification: { source: 'window' },
};

describe('Region Average names a summed composition as one', () => {
  let writeText;
  beforeEach(() => {
    writeText = vi.fn().mockResolvedValue();
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText }, configurable: true,
    });
  });
  const copied = () => writeText.mock.calls[0][0].split('\n')[0].split('\t');

  it('the copied header and the description on the spectrum source', async () => {
    spectrumFitStatus.mockResolvedValue(answer('ready', { progress: 1, fit: FIT }));
    REGION_DATA = SUMMED_REGION;
    render(<EDSPage />);
    await settle();
    expect(screen.getByText('regionAvg.descriptionSummed')).toBeTruthy();
    expect(screen.queryByText('regionAvg.description')).toBeNull();
    expect(screen.getByRole('button', { name: 'regionAvg.compute' })
      .getAttribute('title')).toBe('regionAvg.computeTooltipSummed');
    fireEvent.click(screen.getByRole('button', { name: 'regionAvg.copyAria' }));
    expect(copied()).toEqual(['table.element', 'region.netCountsHeader',
      'region.wtPctSummedHeader', 'region.atPctSummedHeader']);
    expect(writeText.mock.calls[0][0]).not.toMatch(/mean/);
  });

  it('the copied header and the description on the window source', async () => {
    spectrumFitStatus.mockResolvedValue(answer('none'));
    REGION_DATA = MEAN_REGION;
    render(<EDSPage />);
    await settle();
    expect(screen.getByText('regionAvg.description')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'regionAvg.compute' })
      .getAttribute('title')).toBe('regionAvg.computeTooltip');
    fireEvent.click(screen.getByRole('button', { name: 'regionAvg.copyAria' }));
    expect(copied()).toEqual(['table.element', 'region.countsHeader',
      'region.wtPctHeader', 'region.atPctHeader']);
  });

  it('before anything is computed the description follows the file', async () => {
    spectrumFitStatus.mockResolvedValue(answer('ready', { progress: 1, fit: FIT }));
    render(<EDSPage />);
    await settle();
    expect(screen.getByText('regionAvg.descriptionSummed')).toBeTruthy();
    cleanup();
    spectrumFitStatus.mockResolvedValue(answer('none'));
    render(<EDSPage />);
    await settle();
    expect(screen.getByText('regionAvg.description')).toBeTruthy();
  });
});
