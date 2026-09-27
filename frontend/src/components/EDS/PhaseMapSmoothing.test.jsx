// @vitest-environment jsdom
/**
 * Smoothing is a LENGTH, and until now only the backend knew it.
 *
 * `scale` alone is not portable: the same "5" is a 2.5 um box at a 0.5 um
 * step and a 3.75 um box at a 0.75 um step — the same setting on screen, a
 * 50 % different physical analysis. The backend grew `scale_um` and a
 * resolved report to fix exactly that; none of it reached a user while the
 * panel sent only pixels.
 *
 * So what is pinned here is the wiring, not the arithmetic:
 *
 *  - exactly ONE statement about the box leaves the panel (um XOR px),
 *  - the RESOLVED width comes back on screen, because "4 px" and "2.0 um"
 *    are different facts and the user asked for the second,
 *  - a um request the file cannot honour SAYS SO. A physical width that
 *    quietly became a pixel count is the failure the field exists to
 *    prevent, so silence there would be worse than not having the field.
 */
import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, render, screen, cleanup } from '@testing-library/react';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k, o) => (o ? `${k}:${JSON.stringify(o)}` : k) }),
}));
vi.mock('../../stores/useDataStore', () => ({
  default: (sel) => sel({ filePath: '/tmp/x.h5oina' }),
}));

const MAP = { loaded: true, n_rows: 4, n_cols: 4, summary: [], all_phases: [], regions: [] };

/** A classify response as the backend flattens it (`response.update(scale_report)`). */
const withScale = (over = {}) => ({
  ...MAP,
  scale_px_used: 4,
  scale_um_used: 2.0,
  scale_box_um: { x: 2.0, y: 2.0 },
  step_x_um: 0.5,
  step_y_um: 0.5,
  scale_source: 'um',
  scale_um_requested: 2.0,
  ...over,
});

const autoClassify = vi.fn();
const elements = vi.fn();
vi.mock('../../services/api', () => ({
  edsApi: {
    autoClassify: (...a) => autoClassify(...a),
    elements: (...a) => elements(...a),
    getPhaseMap: vi.fn(() => Promise.resolve({ data: MAP })),
    clearPhaseMap: vi.fn(() => Promise.resolve({ data: {} })),
    setPhaseColors: vi.fn(() => Promise.resolve({ data: { loaded: false } })),
    cifPhases: vi.fn(() => Promise.resolve({ data: { phases: [] } })),
    assignRegionPhase: vi.fn(),
  },
}));

import { usePhaseMap, SmoothingScale, readScaleReport } from './PhaseMapPanel';

const flush = () => act(async () => {
  await Promise.resolve(); await Promise.resolve(); await Promise.resolve();
});

const t = (k, o) => (o ? `${k}:${JSON.stringify(o)}` : k);

beforeEach(() => {
  cleanup();
  vi.clearAllMocks();
  autoClassify.mockResolvedValue({ data: withScale() });
  elements.mockResolvedValue({ data: { elements: ['Al Kα1', 'Si Kα1', 'Fe Kα1'] } });
});
afterEach(cleanup);

async function mountHook() {
  const r = renderHook(() => usePhaseMap({}));
  await flush();
  return r;
}

describe('which statement about the box leaves the panel', () => {
  it('sends the pixel count while the unit is px — the default path is unchanged', async () => {
    const { result } = await mountHook();
    act(() => { result.current.setScale(7); });
    await act(async () => { await result.current.handleAutoClassify(); });

    const sent = autoClassify.mock.calls[0][0];
    expect(sent.scale).toBe(7);
    expect('scaleUm' in sent).toBe(false);
  });

  it('sends the LENGTH once the unit is um, and never the pixel count beside it', async () => {
    const { result } = await mountHook();
    act(() => {
      result.current.setScale(7);          // a stale pixel value, deliberately
      result.current.setScaleUnit('um');
      result.current.setScaleUm(2.5);
    });
    await act(async () => { await result.current.handleAutoClassify(); });

    const sent = autoClassify.mock.calls[0][0];
    expect(sent.scaleUm).toBe(2.5);
    // Two different statements about one box, with the loser decided by a
    // precedence rule nobody reading the request can see.
    expect('scale' in sent).toBe(false);
  });

  it('falls back to pixels when the um box is empty — a half-typed width is not a width', async () => {
    const { result } = await mountHook();
    act(() => {
      result.current.setScale(5);
      result.current.setScaleUnit('um');
      result.current.setScaleUm(null);
    });
    await act(async () => { await result.current.handleAutoClassify(); });

    const sent = autoClassify.mock.calls[0][0];
    expect(sent.scale).toBe(5);
    expect('scaleUm' in sent).toBe(false);
  });
});

describe('what comes back', () => {
  it('keeps the resolved report from the run', async () => {
    const { result } = await mountHook();
    await act(async () => { await result.current.handleAutoClassify(); });

    expect(result.current.scaleReport).toMatchObject({
      scale_px_used: 4, scale_source: 'um', step_x_um: 0.5,
      scale_box_um: { x: 2.0, y: 2.0 },
    });
  });

  it('writes the resolved pixel count back, so the two readouts cannot disagree', async () => {
    const { result } = await mountHook();
    act(() => {
      result.current.setScale(9);
      result.current.setScaleUnit('um');
      result.current.setScaleUm(2.0);
    });
    await act(async () => { await result.current.handleAutoClassify(); });

    // 9 px was never smoothed with; 4 px was.
    expect(result.current.scale).toBe(4);
  });

  it('leaves the pixel control alone when the run was a pixel run', async () => {
    autoClassify.mockResolvedValue({
      data: withScale({ scale_source: 'pixels', scale_px_used: 4, scale_um_requested: null }),
    });
    const { result } = await mountHook();
    act(() => { result.current.setScale(9); });
    await act(async () => { await result.current.handleAutoClassify(); });
    expect(result.current.scale).toBe(9);
  });

  it('records the measured elements as plain symbols, never the Aztec line names', async () => {
    const { result } = await mountHook();
    expect(result.current.authoredElements).toEqual(['Al', 'Si', 'Fe']);
  });

  it('reports the step size only once a run has answered', async () => {
    const { result } = await mountHook();
    expect(result.current.authoredStepUm).toBe(null);
    await act(async () => { await result.current.handleAutoClassify(); });
    expect(result.current.authoredStepUm).toBe(0.5);
  });
});

describe('readScaleReport', () => {
  it('is null for a payload that makes no statement about the box', () => {
    expect(readScaleReport(MAP)).toBe(null);
    expect(readScaleReport(null)).toBe(null);
  });

  it('copies the contract fields and nothing else', () => {
    const r = readScaleReport({ ...withScale(), scale: 999, unrelated: 1 });
    expect(Object.keys(r).sort()).toEqual([
      'scale_box_um', 'scale_note', 'scale_px_used', 'scale_source',
      'scale_um_requested', 'scale_um_used', 'step_x_um', 'step_y_um',
    ]);
  });
});

describe('what the panel says about the box', () => {
  const base = {
    unit: 'um', setUnit: vi.fn(), scaleUm: 2, setScaleUm: vi.fn(), scale: 4, t,
  };

  it('states the resolved width in both units — "4 px" alone does not answer the question', () => {
    render(<SmoothingScale {...base} report={readScaleReport(withScale())} />);
    const line = document.querySelector('[data-smoothing-resolved]');
    expect(line).toBeTruthy();
    expect(line.textContent).toContain('phaseMap.scaleResolved');
    expect(line.textContent).toContain('"px":4');
    expect(line.textContent).toContain('"um":"2.0"');
  });

  it('quotes BOTH edges when the step differs along x and y', () => {
    render(<SmoothingScale {...base} report={readScaleReport(withScale({
      scale_box_um: { x: 2.0, y: 3.0 }, step_y_um: 0.75,
    }))} />);
    const line = document.querySelector('[data-smoothing-resolved]');
    expect(line.textContent).toContain('phaseMap.scaleResolvedXY');
    expect(line.textContent).toContain('"y":"3.0"');
  });

  it('says the um request could NOT be honoured, rather than falling back in silence', () => {
    render(<SmoothingScale {...base} report={readScaleReport(withScale({
      scale_source: 'pixels_no_step',
      scale_um_used: null, scale_box_um: null,
      step_x_um: null, step_y_um: null,
      scale_note: 'Requested a 2 um smoothing box, but this file carries no step size',
    }))} />);
    const warn = document.querySelector('[data-smoothing-no-step]');
    expect(warn).toBeTruthy();
    expect(warn.getAttribute('role')).toBe('alert');
    expect(warn.textContent).toContain('phaseMap.scaleNoStep');
    // The requested length and the width actually used, both named.
    expect(warn.textContent).toContain('"um":"2.0"');
    expect(warn.textContent).toContain('"px":4');
    // And it must not ALSO claim a resolved physical box.
    expect(document.querySelector('[data-smoothing-resolved]')).toBe(null);
  });

  it('gives the pixel count only, when the file has no step and none was asked for', () => {
    render(<SmoothingScale {...base} unit="px" report={readScaleReport(withScale({
      scale_source: 'pixels', scale_um_used: null, scale_box_um: null,
      step_x_um: null, step_y_um: null, scale_um_requested: null,
    }))} />);
    expect(document.querySelector('[data-smoothing-resolved]').textContent)
      .toContain('phaseMap.scaleResolvedPxOnly');
  });

  it('says per-pixel classification smoothed nothing, instead of quoting a width', () => {
    render(<SmoothingScale {...base} report={readScaleReport(withScale({
      scale_source: 'not_applicable',
      scale_px_used: null, scale_um_used: null, scale_box_um: null,
    }))} />);
    expect(document.querySelector('[data-smoothing-na]')).toBeTruthy();
    expect(document.querySelector('[data-smoothing-resolved]')).toBe(null);
  });

  it('offers the um field only in um, and keeps px reachable — some users think in pixels', () => {
    const { rerender } = render(<SmoothingScale {...base} unit="px" report={null} />);
    expect(screen.queryByLabelText('phaseMap.scaleUmLabel')).toBe(null);
    expect(screen.getByRole('button', { name: 'phaseMap.scaleUnitPx' })
      .getAttribute('aria-pressed')).toBe('true');

    rerender(<SmoothingScale {...base} unit="um" report={null} />);
    expect(screen.getByLabelText('phaseMap.scaleUmLabel').value).toBe('2');
  });

  it('says nothing at all before a run has resolved anything', () => {
    render(<SmoothingScale {...base} report={null} />);
    expect(document.querySelector('[data-smoothing-resolved]')).toBe(null);
    expect(document.querySelector('[data-smoothing-no-step]')).toBe(null);
  });
});
