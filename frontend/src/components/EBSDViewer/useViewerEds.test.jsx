// @vitest-environment jsdom
/**
 * The EBSD viewer's "EDS Composition" box.
 *
 * On every load of a file with a spectrum cube the viewer asks for pixel
 * (0, 0) while the spectrum fit runs. Both requests answered 503, the box
 * said "No EDS data loaded", "Send Filter to Indexing" stayed disabled -- and
 * nothing asked again when the fit was done. Now the box says the spectra are
 * being evaluated, asks nothing while they are (each of the two requests would be
 * a 503 and a console error), and asks for the pixel by itself
 * when they are done.
 */
import React, { useEffect } from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, act, waitFor } from '@testing-library/react';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const edsApi = vi.hoisted(() => ({
  spectrumFitStatus: vi.fn(), quantifyPixel: vi.fn(), suggestPhases: vi.fn(),
}));
vi.mock('../../services/api', () => ({ edsApi }));

import { useViewerEds, edsCompositionText } from './useViewerEds';

const t = (k, o) => (o ? `${k}:${JSON.stringify(o)}` : k);
const status = (state, extra = {}) => ({
  data: { state, progress: 0, stage: null, error: null, fit: null, ...extra },
});
const busy = () => Object.assign(new Error('503'), {
  response: { status: 503, data: { detail: { code: 'eds_spectrum_fit_running' } } },
});
/** A /quantify/pixel answer as axios hands it over: the body is `data`. */
const PIXEL = { data: { data: { Al: { counts: 812, wt_pct: 90.1, at_pct: 91.2 },
                                Fe: { counts: 40, wt_pct: 9.9, at_pct: 8.8 } } } };

/** The box as the viewer has it: ask for (0, 0) once, on load. */
function Box({ filePath = '/data/SampleB.h5oina', first = [0, 0] }) {
  const eds = useViewerEds({ filePath, active: true, edsMode: 'At.%' });
  useEffect(() => { eds.fetchEds(...first); }, []);   // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div>
      <div data-testid="box">{edsCompositionText(eds.composition, 'At.%', eds.fit, !!filePath, t)}</div>
      <button type="button" disabled={!eds.composition?.data}>send</button>
      <button type="button" onClick={() => eds.fetchEds(3, 4)}>move</button>
      <button type="button" onClick={() => eds.fetchEdsAfterLoad(0, 0)}>reload</button>
    </div>
  );
}

const box = () => screen.getByTestId('box').textContent;

beforeEach(() => {
  Object.values(edsApi).forEach((f) => f.mockReset());
  edsApi.suggestPhases.mockResolvedValue({ data: { suggestions: [] } });
});
afterEach(cleanup);

/** Let pending promises and effects run. */
const flush = () => act(async () => {
  for (let i = 0; i < 4; i += 1) await new Promise((r) => setTimeout(r, 0));
});

describe('the EBSD viewer EDS box during the spectrum fit', () => {
  it('asks nothing while the fit runs; the composition appears when it is ready', async () => {
    edsApi.spectrumFitStatus
      .mockResolvedValueOnce(status('running', { progress: 0.3 }))
      .mockResolvedValue(status('ready', { progress: 1 }));
    edsApi.quantifyPixel.mockRejectedValue(busy());      // what it would get
    edsApi.suggestPhases.mockRejectedValue(busy());
    render(<Box />);
    await waitFor(() => expect(box()).toContain('eds:spectrumFit.running'));
    expect(box()).toContain('"percent":30');
    expect(box()).not.toContain('eds.noData');
    expect(screen.getByRole('button', { name: 'send' }).disabled).toBe(true);
    expect(edsApi.quantifyPixel).not.toHaveBeenCalled();
    expect(edsApi.suggestPhases).not.toHaveBeenCalled();

    edsApi.quantifyPixel.mockReset().mockResolvedValue(PIXEL);
    edsApi.suggestPhases.mockReset().mockResolvedValue({ data: { suggestions: [] } });
    await waitFor(() => expect(box()).toContain('Al: 91.2%'), { timeout: 3000 });
    expect(box()).toContain('Fe: 8.8%');
    expect(screen.getByRole('button', { name: 'send' }).disabled).toBe(false);
    expect(edsApi.quantifyPixel).toHaveBeenCalledTimes(1);
    expect(edsApi.quantifyPixel.mock.calls[0].slice(0, 2)).toEqual([0, 0]);
    expect(edsApi.suggestPhases).toHaveBeenCalledTimes(1);
  });

  it('asks for the pixel the cursor is on, not the one it asked first', async () => {
    let release;
    edsApi.spectrumFitStatus
      .mockResolvedValueOnce(status('running', { progress: 0.5 }))
      .mockReturnValueOnce(new Promise((r) => { release = r; }));
    edsApi.quantifyPixel.mockResolvedValue(PIXEL);
    render(<Box />);
    await waitFor(() => expect(box()).toContain('eds:spectrumFit.running'));
    await act(async () => { screen.getByRole('button', { name: 'move' }).click(); });
    expect(edsApi.quantifyPixel).not.toHaveBeenCalled();
    await act(async () => { release(status('ready', { progress: 1 })); });
    await waitFor(() => expect(box()).toContain('Al: 91.2%'), { timeout: 3000 });
    expect(edsApi.quantifyPixel).toHaveBeenCalledTimes(1);
    expect(edsApi.quantifyPixel.mock.calls[0].slice(0, 2)).toEqual([3, 4]);
  });

  it('says the fit failed instead of "no EDS", and asks nothing', async () => {
    edsApi.spectrumFitStatus.mockResolvedValue(status('failed', { error: 'cube unreadable' }));
    edsApi.quantifyPixel.mockRejectedValue(Object.assign(new Error('500'),
      { response: { status: 500 } }));
    render(<Box />);
    await waitFor(() => expect(box()).toContain('eds:spectrumFit.failed'));
    expect(box()).toContain('cube unreadable');
    await flush();
    expect(edsApi.quantifyPixel).not.toHaveBeenCalled();
  });

  it('says it is checking before the first status answer, and asks nothing', async () => {
    edsApi.spectrumFitStatus.mockReturnValue(new Promise(() => {}));
    edsApi.quantifyPixel.mockRejectedValue(busy());
    render(<Box />);
    await flush();
    expect(box()).toContain('eds:spectrumFit.checking');
    expect(edsApi.quantifyPixel).not.toHaveBeenCalled();
  });

  it('the previous file\'s numbers leave the box while the new one waits', async () => {
    edsApi.spectrumFitStatus.mockImplementation(() => Promise.resolve(status('none')));
    edsApi.quantifyPixel.mockResolvedValue(PIXEL);
    const { rerender } = render(<Box filePath="/data/A.h5oina" />);
    await waitFor(() => expect(box()).toBe('Al: 91.2% | Fe: 8.8%'));
    edsApi.spectrumFitStatus.mockImplementation(
      () => Promise.resolve(status('running', { progress: 0.1 })));
    rerender(<Box filePath="/data/B.h5oina" />);
    await flush();
    // the viewer asks for a pixel of the new file (on load, or the cursor)
    await act(async () => { screen.getByRole('button', { name: 'move' }).click(); });
    expect(box()).toContain('eds:spectrumFit.running');
    expect(edsApi.quantifyPixel).toHaveBeenCalledTimes(1);   // A's only
  });

  it('a file without EDS still says "no EDS data"', async () => {
    edsApi.spectrumFitStatus.mockResolvedValue(status('none'));
    edsApi.quantifyPixel.mockRejectedValue(Object.assign(new Error('400'),
      { response: { status: 400 } }));
    render(<Box />);
    await waitFor(() => expect(box()).toBe('eds.noData'));
  });

  it('with no file loaded the box says "no EDS data", not "checking"', () => {
    expect(edsCompositionText(null, 'At.%', { state: null }, false, t)).toBe('eds.noData');
  });

  it('a window-path file shows its numbers at once, as before', async () => {
    edsApi.spectrumFitStatus.mockResolvedValue(status('none'));
    edsApi.quantifyPixel.mockResolvedValue(PIXEL);
    render(<Box />);
    await waitFor(() => expect(box()).toBe('Al: 91.2% | Fe: 8.8%'));
    expect(edsApi.quantifyPixel).toHaveBeenCalledTimes(1);
  });
});

describe('the same file loaded again', () => {
  it('a load that starts a new fit asks nothing until the fit is ready', async () => {
    edsApi.spectrumFitStatus.mockResolvedValue(status('ready', { progress: 1 }));
    edsApi.quantifyPixel.mockResolvedValue(PIXEL);
    render(<Box />);
    await waitFor(() => expect(box()).toBe('Al: 91.2% | Fe: 8.8%'));
    expect(edsApi.quantifyPixel).toHaveBeenCalledTimes(1);

    // the path is the same; the backend fits it again (e.g. the file changed)
    edsApi.spectrumFitStatus.mockReset()
      .mockResolvedValueOnce(status('running', { progress: 0.2 }))
      .mockResolvedValue(status('ready', { progress: 1 }));
    edsApi.quantifyPixel.mockReset().mockRejectedValue(busy());
    await act(async () => { screen.getByRole('button', { name: 'reload' }).click(); });
    await waitFor(() => expect(box()).toContain('eds:spectrumFit.running'));
    expect(edsApi.quantifyPixel).not.toHaveBeenCalled();

    edsApi.quantifyPixel.mockReset().mockResolvedValue(PIXEL);
    await waitFor(() => expect(box()).toBe('Al: 91.2% | Fe: 8.8%'), { timeout: 3000 });
    expect(edsApi.quantifyPixel).toHaveBeenCalledTimes(1);
    expect(edsApi.quantifyPixel.mock.calls[0].slice(0, 2)).toEqual([0, 0]);
  });

  it('a load of a file whose fit is cached asks the state once more, then the pixel', async () => {
    edsApi.spectrumFitStatus.mockResolvedValue(status('ready', { progress: 1 }));
    edsApi.quantifyPixel.mockResolvedValue(PIXEL);
    render(<Box />);
    await waitFor(() => expect(box()).toBe('Al: 91.2% | Fe: 8.8%'));
    const asked = edsApi.spectrumFitStatus.mock.calls.length;
    await act(async () => { screen.getByRole('button', { name: 'reload' }).click(); });
    await waitFor(() => expect(edsApi.quantifyPixel).toHaveBeenCalledTimes(2));
    expect(edsApi.spectrumFitStatus.mock.calls.length).toBe(asked + 1);
    expect(box()).toBe('Al: 91.2% | Fe: 8.8%');
  });
});

describe('the viewer uses it', () => {
  const here = dirname(fileURLToPath(import.meta.url));
  const src = readFileSync(resolve(here, 'EBSDViewer.jsx'), 'utf8');
  it('takes its EDS box from useViewerEds and edsCompositionText', () => {
    expect(src).toMatch(/useViewerEds\(\{/);
    expect(src).toMatch(/edsCompositionText\(/);
    // every load asks for its pixel through fetchEdsAfterLoad
    expect(src.match(/fetchEdsAfterLoad\(0, 0\)/g)).toHaveLength(3);
    expect(src).not.toMatch(/\bfetchEds\(0, 0\)/);
    // no second, private copy of the state that stopped asking
    expect(src).not.toMatch(/useState\(null\);\s*\n\s*const \[edsPhases/);
    expect(src).not.toMatch(/setEdsComposition\(/);
  });
});
