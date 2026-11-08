// @vitest-environment jsdom
/**
 * The Indexing page's EDS Overlay panel during the spectrum fit: it fetched `/api/eds/elements` once per file, got a 503
 * while the fit ran, emptied the list and never asked again -- "No EDS
 * elements found in dataset" a minute after the fit, on a file with eight.
 */
import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k, o) => (o ? `${k}:${JSON.stringify(o)}` : k) }),
}));
const edsApi = vi.hoisted(() => ({ elements: vi.fn(), getMap: vi.fn() }));
vi.mock('../../services/api', () => ({ edsApi }));
// One object, as the real store keeps it: the panel refetches its maps when
// the colour map CHANGES identity.
const STORE = vi.hoisted(() => ({ colors: { Al: '#ff0000', Fe: '#00ff00' } }));
vi.mock('../../stores/useEdsColorStore', () => ({ default: (sel) => sel(STORE) }));

import EdsOverlayPanel from './EdsOverlayPanel';

/** jsdom never loads an image; this one loads as soon as it has a source. */
class LoadingImage {
  set src(v) { this._src = v; setTimeout(() => this.onload?.(), 0); }
  get src() { return this._src; }
}

const busy = () => Object.assign(new Error('503'), {
  response: { status: 503, data: { detail: { code: 'eds_spectrum_fit_running' } } },
});

function panel(props = {}) {
  const onLayersChange = props.onLayersChange || vi.fn();
  const el = (p) => (
    <EdsOverlayPanel enabled available onToggle={() => {}} fileKey="SampleB::d1"
      onLayersChange={onLayersChange} {...p} />
  );
  const r = render(el(props));
  return { ...r, rerenderWith: (p) => r.rerender(el({ ...props, ...p })), onLayersChange };
}

let realImage;
beforeEach(() => {
  edsApi.elements.mockReset();
  edsApi.getMap.mockReset();
  realImage = globalThis.Image;
  globalThis.Image = LoadingImage;
});
afterEach(() => {
  cleanup();
  globalThis.Image = realImage;
});

describe('EdsOverlayPanel and the spectrum fit', () => {
  it('503 first, then ready: the element list appears without user action', async () => {
    edsApi.elements.mockRejectedValueOnce(busy())
      .mockResolvedValue({ data: { elements: ['Al Kα1', 'Fe Kα1'] } });
    const { rerenderWith } = panel({ fitState: 'running', fitProgress: 0.4 });
    await waitFor(() => expect(edsApi.elements).toHaveBeenCalledTimes(1));
    expect(await screen.findByText('eds:spectrumFit.running:{"percent":40}')).toBeTruthy();
    expect(screen.queryByText('eds.noElements')).toBeNull();

    rerenderWith({ fitState: 'ready', fitProgress: 1 });
    expect(await screen.findByText('Al Kα1')).toBeTruthy();
    expect(screen.getByText('Fe Kα1')).toBeTruthy();
    expect(edsApi.elements).toHaveBeenCalledTimes(2);
    expect(screen.queryByText(/spectrumFit\.running/)).toBeNull();
  });

  it('an element switched on during the fit gets its map when the fit is ready', async () => {
    edsApi.elements.mockResolvedValue({ data: { elements: ['Al Kα1', 'Fe Kα1'] } });
    edsApi.getMap.mockRejectedValueOnce(busy())
      .mockResolvedValue({ data: { image: 'PNGB64' } });
    const onLayersChange = vi.fn();
    const { rerenderWith } = panel({ fitState: 'running', fitProgress: 0.5, onLayersChange });
    fireEvent.click(await screen.findByText('Al Kα1'));
    await waitFor(() => expect(edsApi.getMap).toHaveBeenCalledTimes(1));
    expect(screen.getByText('eds.loading')).toBeTruthy();

    rerenderWith({ fitState: 'ready', fitProgress: 1 });
    await waitFor(() => expect(edsApi.getMap).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(onLayersChange.mock.calls.at(-1)[0]).toHaveLength(1));
    expect(onLayersChange.mock.calls.at(-1)[0][0].element).toBe('Al Kα1');
    expect(screen.queryByText('eds.loading')).toBeNull();
  });

  it('says it is checking before the fit state is known, not "no elements"', async () => {
    edsApi.elements.mockRejectedValue(busy());
    panel({ fitState: null });
    await waitFor(() => expect(edsApi.elements).toHaveBeenCalled());
    expect(await screen.findByText('eds:spectrumFit.checking')).toBeTruthy();
    expect(screen.queryByText('eds.noElements')).toBeNull();
  });

  it('a file without spectra and without elements says so, as before', async () => {
    edsApi.elements.mockResolvedValue({ data: { elements: [] } });
    panel({ fitState: 'none' });
    expect(await screen.findByText('eds.noElements')).toBeTruthy();
  });
});

describe('the Indexing page hands the panel its fit state', () => {
  it('passes the spectrum fit through to EdsOverlayPanel', async () => {
    const { readFileSync } = await import('node:fs');
    const { resolve, dirname } = await import('node:path');
    const { fileURLToPath } = await import('node:url');
    const src = readFileSync(resolve(dirname(fileURLToPath(import.meta.url)),
      'IndexingPage.jsx'), 'utf8');
    expect(src).toMatch(/edsFitState=\{spectrumFit\.state\}/);
    expect(src).toMatch(/edsFitProgress=\{spectrumFit\.progress\}/);
    const panel = src.slice(src.indexOf('<EdsOverlayPanel'),
      src.indexOf('/>', src.indexOf('<EdsOverlayPanel')));
    expect(panel).toMatch(/fitState=\{edsFitState\}/);
    expect(panel).toMatch(/fitProgress=\{edsFitProgress\}/);
  });
});

describe('EdsOverlayPanel: one label per element in wt% and at%', () => {
  const LABELS = ['Al Kα1', 'Cu Kα1', 'Cu Lα1,2', 'Fe Kα1', 'Fe Lα1,2'];
  const LINES = { Al: { label: 'Al Kα1', family: 'K' }, Cu: { label: 'Cu Lα1,2', family: 'L' },
                  Fe: { label: 'Fe Kα1', family: 'K' } };
  const unitSelect = (container) => [...container.querySelectorAll('select')]
    .find((s) => [...s.options].some((o) => o.value === 'at_pct'));

  it('Counts lists every fitted family; At.% the quantified line of each element', async () => {
    edsApi.elements.mockResolvedValue({ data: { elements: LABELS } });
    const { container } = panel({ fitState: 'ready', fitLines: LINES });
    expect(await screen.findByText('Fe Lα1,2')).toBeTruthy();
    expect(screen.getByText('Cu Kα1')).toBeTruthy();
    fireEvent.change(unitSelect(container), { target: { value: 'at_pct' } });
    await waitFor(() => expect(screen.queryByText('Fe Lα1,2')).toBeNull());
    expect(screen.queryByText('Cu Kα1')).toBeNull();
    for (const l of ['Al Kα1', 'Cu Lα1,2', 'Fe Kα1']) expect(screen.getByText(l)).toBeTruthy();
  });

  it('a label switched on in Counts is not drawn twice in At.%', async () => {
    edsApi.elements.mockResolvedValue({ data: { elements: LABELS } });
    edsApi.getMap.mockResolvedValue({ data: { image: 'PNGB64' } });
    const onLayersChange = vi.fn();
    const { container } = panel({ fitState: 'ready', fitLines: LINES, onLayersChange });
    fireEvent.click(await screen.findByText('Fe Kα1'));
    fireEvent.click(screen.getByText('Fe Lα1,2'));
    await waitFor(() => expect(onLayersChange.mock.calls.at(-1)[0]).toHaveLength(2));
    fireEvent.change(unitSelect(container), { target: { value: 'at_pct' } });
    await waitFor(() => expect(onLayersChange.mock.calls.at(-1)[0].map((l) => l.element))
      .toEqual(['Fe Kα1']));
  });

  it('a window file (no fit lines) is unchanged in At.%', async () => {
    edsApi.elements.mockResolvedValue({ data: { elements: ['Cu Kα1', 'Cu Lα1,2'] } });
    const { container } = panel({ fitState: 'none', fitLines: null });
    expect(await screen.findByText('Cu Lα1,2')).toBeTruthy();
    fireEvent.change(unitSelect(container), { target: { value: 'at_pct' } });
    await new Promise((r) => setTimeout(r, 20));
    expect(screen.getByText('Cu Kα1')).toBeTruthy();
    expect(screen.getByText('Cu Lα1,2')).toBeTruthy();
  });

  it('the Indexing page hands the fit lines to the panel', async () => {
    const { readFileSync } = await import('node:fs');
    const { resolve, dirname } = await import('node:path');
    const { fileURLToPath } = await import('node:url');
    const src = readFileSync(resolve(dirname(fileURLToPath(import.meta.url)), 'IndexingPage.jsx'), 'utf8');
    expect(src).toMatch(/edsFitLines=\{spectrumFit\.state === 'ready' \? \(spectrumFit\.fit\?\.lines \?\? null\) : null\}/);
    expect(src).toMatch(/fitLines=\{edsFitLines\}/);
  });
});
