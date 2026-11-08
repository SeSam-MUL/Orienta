// @vitest-environment jsdom
/**
 * What the phase-map payload says about itself, on screen.
 *
 * `GET /phase-map` (and every phase-map write) carries two things this panel
 * never read:
 *
 *  - `notices: [{code: "sidecar_made_on_other_source", detail: {sidecar,
 *    current}, message}]` -- the saved map was made on the other EDS numbers.
 *    Its regions and rules were set on window counts; on net intensities the
 *    same thresholds select other pixels. Nothing in the map is converted, so
 *    the only protection is that the user is told;
 *  - `eds_status: {state: "pending" | "failed", ...}` -- the map is shown
 *    while the spectrum fit runs or after it failed, WITHOUT region
 *    compositions. An empty composition column with no reason reads as "this
 *    region has no chemistry".
 *
 * Translated on the code; the backend's English is the fallback.
 */
import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, cleanup, act, fireEvent, screen } from '@testing-library/react';

import en from '../../locales/en/eds.json';
import de from '../../locales/de/eds.json';
import ja from '../../locales/ja/eds.json';
import zh from '../../locales/zh/eds.json';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k, o) => (o ? `${k}:${JSON.stringify(o)}` : k) }),
}));
vi.mock('../../stores/useDataStore', () => ({
  default: (sel) => sel({ filePath: '/tmp/x.h5oina' }),
}));

const getPhaseMap = vi.fn();
const autoClassify = vi.fn();
vi.mock('../../services/api', () => ({
  edsApi: {
    autoClassify: (...a) => autoClassify(...a),
    getPhaseMap: (...a) => getPhaseMap(...a),
    setPhaseColors: vi.fn(() => Promise.resolve({ data: { loaded: false } })),
    clearPhaseMap: vi.fn(() => Promise.resolve({ data: {} })),
    elements: vi.fn(() => Promise.resolve({ data: { elements: [] } })),
    cifPhases: vi.fn(() => Promise.resolve({ data: { phases: [] } })),
    assignRegionPhase: vi.fn(),
  },
}));

import { usePhaseMap, PhaseMapControls, PhaseMapNotices } from './PhaseMapPanel';

/** `_source_notices` in routes/eds.py, verbatim shape. */
const NOTICE = {
  code: 'sidecar_made_on_other_source',
  message: 'This phase map was made on window EDS numbers, this file\'s are '
    + 'spectrum. Nothing in the map was changed or converted; ...',
  detail: { sidecar: 'window', current: 'spectrum' },
};
const MAP = {
  loaded: true, n_rows: 4, n_cols: 4, summary: [], all_phases: [],
  regions: [], n_classified: 16, n_unclassified: 0,
};

const realT = (key, opts = {}) => {
  const val = key.split('.').reduce(
    (o, k) => (o && typeof o === 'object' ? o[k] : undefined), en);
  let s = typeof val === 'string' ? val : (opts.defaultValue ?? key);
  for (const [k, v] of Object.entries(opts)) {
    if (k === 'defaultValue') continue;
    s = s.split(`{{${k}}}`).join(String(v));
  }
  return s;
};
const mockT = (k, o) => (o ? `${k}:${JSON.stringify(o)}` : k);

const flushAll = () => act(async () => {
  for (let i = 0; i < 6; i += 1) await Promise.resolve();
});
function Harness() {
  const handle = usePhaseMap({});
  return <PhaseMapControls handle={handle} />;
}

beforeEach(() => {
  cleanup();
  vi.clearAllMocks();
  getPhaseMap.mockResolvedValue({ data: { loaded: false } });
});
afterEach(cleanup);

describe('PhaseMapNotices', () => {
  it('a map made on window numbers says so, in the user\'s language', () => {
    render(<PhaseMapNotices notices={[NOTICE]} t={realT} />);
    const el = document.querySelector('[data-phase-map-notice="sidecar_made_on_other_source"]');
    expect(el).toBeTruthy();
    expect(el.textContent).toBe(en.phaseMap.notice.otherSource);
  });

  it('the other direction gets its own sentence', () => {
    render(<PhaseMapNotices t={realT} notices={[{
      ...NOTICE, detail: { sidecar: 'spectrum', current: 'window' } }]} />);
    expect(document.querySelector('[data-phase-map-notice]').textContent)
      .toBe(en.phaseMap.notice.otherSourceSpectrum);
  });

  it('a code this build does not know still reaches the user in the backend\'s words', () => {
    render(<PhaseMapNotices t={realT}
      notices={[{ code: 'from_the_future', message: 'Something new.' }]} />);
    expect(document.querySelector('[data-phase-map-notice="from_the_future"]').textContent)
      .toBe('Something new.');
  });

  it('pending compositions say why they are missing', () => {
    render(<PhaseMapNotices t={mockT} edsStatus={{ state: 'pending',
      code: 'eds_spectrum_fit_running', progress: 0.25, stage: 'fit' }} />);
    const el = document.querySelector('[data-eds-status="pending"]');
    expect(el.textContent).toContain('spectrumFit.edsPending');
    expect(el.textContent).toContain('"percent":25');
  });

  it('a failed fit says it failed', () => {
    render(<PhaseMapNotices t={mockT} edsStatus={{ state: 'failed',
      code: 'eds_spectrum_fit_failed', error: 'cube unreadable' }} />);
    const el = document.querySelector('[data-eds-status="failed"]');
    expect(el.getAttribute('role')).toBe('alert');
    expect(el.textContent).toContain('cube unreadable');
  });

  it('renders nothing when there is nothing to say', () => {
    const { container } = render(<PhaseMapNotices t={realT} notices={[]} />);
    expect(container.innerHTML).toBe('');
  });
});

describe('the panel reads them off the payload', () => {
  it('shows the notice of a saved map as soon as it is loaded', async () => {
    getPhaseMap.mockResolvedValue({ data: { ...MAP, notices: [NOTICE],
      quant_source: 'window' } });
    render(<Harness />);
    await flushAll();
    const el = document.querySelector('[data-phase-map-notice="sidecar_made_on_other_source"]');
    expect(el, 'the notice is on the payload and nowhere on screen').toBeTruthy();
    expect(el.textContent).toContain('phaseMap.notice.otherSource');
  });

  it('shows the eds_status of a map loaded while the fit runs', async () => {
    getPhaseMap.mockResolvedValue({ data: { ...MAP, notices: [],
      eds_status: { state: 'pending', code: 'eds_spectrum_fit_running',
        progress: 0.5, stage: 'fit' } } });
    render(<Harness />);
    await flushAll();
    expect(document.querySelector('[data-eds-status="pending"]')).toBeTruthy();
  });

  it('labels the cluster report as the classifier\'s view, not the region composition', async () => {
    autoClassify.mockResolvedValue({ data: { ...MAP, warnings: [],
      clusters: [{ cluster_id: 0, cif_filename: 'Al.cif', percentage: 100,
        score: 0.98, mean_at_pct: { Al: 97.2 }, ambiguous: false }] } });
    render(<Harness />);
    await flushAll();
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'phaseMap.autoClassify' }));
    });
    await flushAll();
    const note = document.querySelector('[data-cluster-composition-note]');
    expect(note, 'the cluster means are not labelled').toBeTruthy();
    expect(note.textContent).toContain('composition.classifier');
  });
});

describe('every language', () => {
  it('has both notice sentences and the two status lines, translated', () => {
    for (const [lang, b] of Object.entries({ en, de, ja, zh })) {
      for (const k of ['otherSource', 'otherSourceSpectrum']) {
        expect(b.phaseMap.notice?.[k], `${lang}.${k}`).toBeTruthy();
      }
      for (const k of ['edsPending', 'edsFailed']) {
        expect(b.spectrumFit?.[k], `${lang}.${k}`).toBeTruthy();
      }
    }
    expect(en.phaseMap.notice.otherSource).toBe(
      'This saved map was made on window numbers; the thresholds are unchanged and may now select different pixels.');
  });
});
