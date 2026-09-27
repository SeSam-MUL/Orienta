// @vitest-environment jsdom
/**
 * Two small honesty items in the classification controls.
 *
 *  1. The Tolerance slider is GONE. It was inert - `auto_classify_pixels`
 *     never read it, `eds_clustering.py` references it zero times - and it
 *     sat on screen behind a grey label saying so. A tester's verdict: "that
 *     your provenance says `tolerance_effective: false` is admirable; that it
 *     is still on my screen is not." A control that cannot change the result
 *     has no claim on the panel. The value is still sent and still recorded,
 *     so the i18n keys stay; only the widget goes.
 *  2. Min score now explains itself where the user is standing. The sentence
 *     existed - in the manual. A slider labelled "Min score (0.30)" tells a
 *     non-expert nothing about what moving it does to the map.
 */
import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react';

vi.mock('../../stores/useDataStore', () => ({
  default: (sel) => sel({ filePath: '/data/scan.h5oina' }),
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k, o) => (o ? `${k}:${JSON.stringify(o)}` : k) }),
}));
vi.mock('../../services/api', () => ({
  edsApi: {
    getPhaseMap: vi.fn(() => Promise.resolve({ data: { loaded: false } })),
    setPhaseColors: vi.fn(() => Promise.resolve({ data: { loaded: false } })),
    cifPhases: vi.fn(() => Promise.resolve({ data: { phases: [] } })),
    elements: vi.fn(() => Promise.resolve({ data: { elements: [] } })),
    autoClassify: vi.fn(),
    clearPhaseMap: vi.fn(),
  },
  edsExportApi: {
    listPresets: vi.fn(() => Promise.resolve({ data: { presets: [] } })),
    getPreset: vi.fn(), checkPreset: vi.fn(), savePreset: vi.fn(),
    deletePreset: vi.fn(), importPreset: vi.fn(), exportPresetFile: vi.fn(),
    exportPreview: vi.fn(), runExport: vi.fn(),
  },
}));

import { usePhaseMap, PhaseMapControls } from './PhaseMapPanel';

function Harness() {
  const handle = usePhaseMap({});
  return <PhaseMapControls handle={handle} />;
}

const flush = () => act(async () => { await Promise.resolve(); await Promise.resolve(); });

beforeEach(() => { cleanup(); vi.clearAllMocks(); });
afterEach(() => cleanup());

describe('the dead tolerance slider', () => {
  it('is not on screen in either grouping mode', async () => {
    render(<Harness />);
    await flush();
    expect(document.querySelector('[data-tolerance-inert]')).toBeNull();

    // Per-pixel mode is where it used to be drawn.
    fireEvent.click(screen.getByRole('button', { name: 'phaseMap.modePixel' }));
    await flush();
    expect(document.querySelector('[data-tolerance-inert]')).toBeNull();
    // And no slider is left labelled with it.
    const sliders = [...document.querySelectorAll('input[type="range"]')];
    expect(sliders.every((el) => el.title !== 'phaseMap.toleranceInert')).toBe(true);
  });

  it('leaves one line for the user who read an older manual', async () => {
    render(<Harness />);
    await flush();
    fireEvent.click(screen.getByRole('button', { name: 'phaseMap.modePixel' }));
    await flush();
    expect(document.querySelector('[data-tolerance-removed]').textContent)
      .toContain('phaseMap.toleranceRemoved');
  });
});

describe('min score explains itself in the panel', () => {
  it('puts the sentence under the slider, not on page 236 of the manual', async () => {
    render(<Harness />);
    await flush();
    const note = document.querySelector('[data-min-score-note]');
    expect(note).toBeTruthy();
    expect(note.textContent).toContain('phaseMap.minScoreNote');
  });
});

/**
 * FIX 6's premise, pinned in source: the confirmation lives in
 * `usePhaseMap.handleAutoClassify`, and the two editors' Apply buttons must
 * route through it. If either is ever rewired to a bare `runClassify`, its
 * tooltip - which now promises "you are asked first" - becomes a lie, and
 * the guard covers two thirds of the ways to lose hand-given names.
 */
describe('all three Apply routes go through the guarded function', () => {
  it('the region-definition and rules editors call handleAutoClassify', async () => {
    const fs = await import('node:fs');
    const path = await import('node:path');
    // `import.meta.url` is an http URL under the jsdom environment, so the
    // path is resolved from the vitest root instead.
    const src = fs.readFileSync(
      path.resolve(process.cwd(), 'src/components/EDS/EDSPage.jsx'), 'utf-8');
    for (const tag of ['<RegionDefs', '<PhaseRules']) {
      const at = src.indexOf(tag);
      expect(at, `${tag} is not rendered by EDSPage`).toBeGreaterThan(-1);
      const block = src.slice(at, src.indexOf('/>', at));
      expect(block, `${tag} must reach the guard`)
        .toContain('onReclassify={phaseMapHandle.handleAutoClassify}');
    }
  });
});
