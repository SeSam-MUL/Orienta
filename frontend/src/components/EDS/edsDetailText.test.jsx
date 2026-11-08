// @vitest-environment jsdom
/**
 * A failed EDS request must reach the screen as TEXT.
 *
 * The spectrum fit answers every EDS-number request with a structured refusal
 * while it runs (HTTP 503) or after it failed (HTTP 500):
 * `detail: {code, progress | error, message}`. Every error panel on this page
 * put `err.response.data.detail` straight into state and rendered it as a
 * React child, which for an object is a crash into the error boundary, not a
 * message. These tests hold the helper to the shapes the backend sends and the
 * call sites to the helper.
 */
import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { renderHook, act, render } from '@testing-library/react';
import fs from 'node:fs';
import path from 'node:path';

const regionQuantify = vi.fn();
vi.mock('../../services/api', () => ({
  edsApi: { regionQuantify: (...a) => regionQuantify(...a) },
}));

import { edsDetailText } from './edsDetailText';
import { useRegionAverage } from './hooks/useRegionAverage';

/** Exactly what eds_spectrum_service.SpectrumFitPending serialises. */
const RUNNING = {
  response: {
    status: 503,
    data: { detail: {
      code: 'eds_spectrum_fit_running', progress: 0.42, stage: 'fit',
      message: 'The spectra of this map are still being evaluated (42 %). '
        + 'EDS numbers are available when that is done.',
    } },
  },
  message: 'Request failed with status code 503',
};
/** ... and SpectrumFitFailed. */
const FAILED = {
  response: {
    status: 500,
    data: { detail: {
      code: 'eds_spectrum_fit_failed', error: 'cube unreadable',
      message: 'The spectrum fit failed: cube unreadable. EDS numbers are not '
        + 'available for this file.',
    } },
  },
};

const t = (k, o) => `${k}:${JSON.stringify(o)}`;

describe('edsDetailText', () => {
  it('passes a plain string detail through unchanged', () => {
    expect(edsDetailText({ response: { data: { detail: 'No file open' } } }))
      .toBe('No file open');
  });

  it('is null when there is no detail, so the call site keeps its fallback', () => {
    expect(edsDetailText({ message: 'Network Error' })).toBeNull();
    expect(edsDetailText(null)).toBeNull();
    expect(edsDetailText({ response: { data: {} } })).toBeNull();
  });

  it('uses the backend sentence of a structured refusal without t', () => {
    expect(edsDetailText(RUNNING)).toBe(RUNNING.response.data.detail.message);
    expect(edsDetailText(FAILED)).toBe(FAILED.response.data.detail.message);
  });

  it('translates the two spectrum-fit codes when given t, with the prose as fallback', () => {
    const run = edsDetailText(RUNNING, t);
    expect(run).toContain('spectrumFit.running');
    expect(run).toContain('"percent":42');
    expect(run).toContain('defaultValue');
    const fail = edsDetailText(FAILED, t);
    expect(fail).toContain('spectrumFit.failed');
    expect(fail).toContain('"error":"cube unreadable"');
  });

  it('turns a FastAPI validation list into one line', () => {
    const err = { response: { data: { detail: [
      { loc: ['body', 'row'], msg: 'field required', type: 'missing' },
      { loc: ['body', 'col'], msg: 'field required', type: 'missing' },
    ] } } };
    expect(edsDetailText(err)).toBe('field required; field required');
  });

  it('never returns an object, whatever arrives', () => {
    for (const detail of [{ weird: 1 }, [{ a: 1 }], 42, true]) {
      const out = edsDetailText({ response: { data: { detail } } });
      expect(typeof out).toBe('string');
    }
  });
});

describe('the call sites go through it', () => {
  it('the region average shows the refusal as a sentence, and it renders', async () => {
    regionQuantify.mockRejectedValue(RUNNING);
    // i18next's behaviour for a key it has no string for: the defaultValue.
    const tMissing = (k, o) => o?.defaultValue ?? k;
    const { result } = renderHook(() => useRegionAverage('at_pct', tMissing));
    await act(async () => { await result.current.quantify({
      rowStart: 0, rowEnd: 1, colStart: 0, colEnd: 1,
    }); });
    expect(typeof result.current.error).toBe('string');
    expect(result.current.error).toContain('still being evaluated');
    // What the page does with it: a React child. An object would throw here.
    expect(() => render(<div>{result.current.error}</div>)).not.toThrow();
  });

  it('no EDS source reads response.data.detail except through the helper', () => {
    const dir = path.resolve(process.cwd(), 'src/components/EDS');
    const files = [];
    const walk = (d) => {
      for (const f of fs.readdirSync(d)) {
        const p = path.join(d, f);
        if (fs.statSync(p).isDirectory()) walk(p);
        else if (/\.(jsx?)$/.test(f) && !/\.test\./.test(f)
                 && f !== 'edsDetailText.js') files.push(p);
      }
    };
    walk(dir);
    expect(files.length).toBeGreaterThan(20);
    const offenders = files.filter((p) => /data\??\.detail\b/
      .test(fs.readFileSync(p, 'utf8')))
      .map((p) => path.relative(dir, p));
    expect(offenders).toEqual([]);
  });
});
