// @vitest-environment jsdom
/**
 * Counts on the spectrum source are NET counts, and say so.
 *
 * `/quantify/pixel`, `/region-quantify` and `/probe` answer `counts` that, on
 * a file with spectra, are net line counts from the spectrum fit: background
 * subtracted, so a pixel without the element scatters around 0 and can be
 * negative. Under the old header "Counts" a -3 reads as a broken number; the
 * same holds for a counts linescan, which the backend deliberately does NOT
 * clip (`/map` clips for display and says `clipped_for_display`).
 *
 * The composite payloads (the hover probe) never fail for the fit; they leave
 * the EDS part out and say why in `eds_status` -- which must reach the
 * tooltip, or an empty element list reads as "no EDS here".
 */
import React from 'react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, cleanup } from '@testing-library/react';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k, o) => (o ? `${k}:${JSON.stringify(o)}` : k) }),
}));
// Plotly cannot run in jsdom; the plot itself is not what is tested here.
vi.mock('react-plotly.js', () => ({
  default: (props) => (
    <div data-testid="plot" data-ytitle={props.layout?.yaxis?.title} />
  ),
}));

import QuantTable, { RegionQuantTable } from './quantTable';
import HoverProbeOverlay from './HoverProbeOverlay';
import LinescanProfilePlot from './LinescanProfilePlot';

afterEach(cleanup);

const ROWS = [
  { element: 'Al', counts: 812.4, wt_pct: 97.1, at_pct: 97.6 },
  { element: 'Ti', counts: -3.2, wt_pct: 0, at_pct: 0 },
];
const headers = () => [...document.querySelectorAll('th')].map((th) => th.textContent);

describe('the pixel table', () => {
  it('heads the column "net counts" on the spectrum source', () => {
    render(<QuantTable data={ROWS} source="spectrum" />);
    expect(headers()).toContain('table.netCounts');
    expect(headers()).not.toContain('table.counts');
    const th = [...document.querySelectorAll('th')]
      .find((h) => h.textContent === 'table.netCounts');
    expect(th.getAttribute('title')).toBe('table.netCountsTooltip');
  });

  it('keeps "counts" on the window source and when the source is unknown', () => {
    render(<QuantTable data={ROWS} source="window" />);
    expect(headers()).toContain('table.counts');
    cleanup();
    render(<QuantTable data={ROWS} />);
    expect(headers()).toContain('table.counts');
  });

  it('shows a negative net count as the number it is', () => {
    render(<QuantTable data={ROWS} source="spectrum" />);
    expect(document.body.textContent).toContain('-3');
  });
});

/** A `/region-quantify` answer on the spectrum source (routes/eds.py). */
const REGION = {
  row_start: 0, row_end: 9, col_start: 0, col_end: 9, n_pixels: 100,
  data: {
    Al: { counts: { mean: 800, std: 40, min: 700, max: 900 },
          wt_pct: { mean: 97, std: 1, min: 95, max: 99 },
          at_pct: { mean: 97.5, std: 1, min: 95, max: 99 } },
  },
  composition_method: 'summed_net_counts',
  quantification: { source: 'spectrum',
    source_data: 'spectrum fit (eds-spectrum-fit/1): ...' },
};

describe('the region table', () => {
  it('net counts header and the summed-composition note on the spectrum source', () => {
    render(<RegionQuantTable data={REGION} />);
    expect(headers()).toContain('region.netCountsHeader');
    // Wt.% and At.% are ONE composition of the summed nets here, not a mean
    // of pixels; only their ± is per pixel. "(mean±σ)" would be false.
    expect(headers()).toContain('region.wtPctSummedHeader');
    expect(headers()).toContain('region.atPctSummedHeader');
    expect(headers()).not.toContain('region.wtPctHeader');
    expect(headers()).not.toContain('region.atPctHeader');
    const note = document.querySelector('[data-composition-method]');
    expect(note.getAttribute('data-composition-method')).toBe('summed_net_counts');
    expect(note.textContent).toContain('composition.summed');
  });

  it('the mean-of-pixels note and plain counts on the window source', () => {
    render(<RegionQuantTable data={{
      ...REGION, composition_method: 'mean_of_pixel_at_pct',
      quantification: { source: 'window', source_data: 'vendor Window Integral (background included)' },
    }} />);
    expect(headers()).toContain('region.countsHeader');
    expect(headers()).toContain('region.wtPctHeader');
    expect(headers()).toContain('region.atPctHeader');
    expect(headers()).not.toContain('region.atPctSummedHeader');
    const note = document.querySelector('[data-composition-method]');
    expect(note.getAttribute('data-composition-method')).toBe('mean_of_pixel_at_pct');
    expect(note.textContent).toContain('composition.mean');
  });

  it('no note for an answer that does not say how it was made', () => {
    const { composition_method: _m, ...old } = REGION;
    render(<RegionQuantTable data={old} />);
    expect(document.querySelector('[data-composition-method]')).toBeNull();
  });
});

describe('the region table headers say what the columns hold, in every language', () => {
  it('has the summed headers, translated, and none of them claims a mean', async () => {
    const locales = {
      en: (await import('../../locales/en/eds.json')).default,
      de: (await import('../../locales/de/eds.json')).default,
      ja: (await import('../../locales/ja/eds.json')).default,
      zh: (await import('../../locales/zh/eds.json')).default,
    };
    const meanWord = { en: /mean/i, de: /Mittel/, ja: /平均/, zh: /均值/ };
    for (const [lang, b] of Object.entries(locales)) {
      for (const k of ['wtPctSummedHeader', 'atPctSummedHeader']) {
        expect(b.region[k], `${lang}.region.${k}`).toBeTruthy();
        expect(b.region[k], `${lang}.region.${k}`).not.toMatch(meanWord[lang]);
        if (lang !== 'en') expect(b.region[k]).not.toBe(locales.en.region[k]);
      }
    }
    expect(locales.en.region.atPctSummedHeader).toBe('At.% (summed; ± per pixel)');
  });

  it('the Region Average description and tooltip for a summed composition', async () => {
    const locales = {
      en: (await import('../../locales/en/eds.json')).default,
      de: (await import('../../locales/de/eds.json')).default,
      ja: (await import('../../locales/ja/eds.json')).default,
      zh: (await import('../../locales/zh/eds.json')).default,
    };
    const meanWord = { en: /mean/i, de: /Mittel/, ja: /平均/, zh: /均值/ };
    for (const [lang, b] of Object.entries(locales)) {
      for (const k of ['descriptionSummed', 'computeTooltipSummed']) {
        expect(b.regionAvg[k], `${lang}.regionAvg.${k}`).toBeTruthy();
        expect(b.regionAvg[k], `${lang}.regionAvg.${k}`).not.toMatch(meanWord[lang]);
        if (lang !== 'en') expect(b.regionAvg[k]).not.toBe(locales.en.regionAvg[k]);
      }
      // the window source's texts stay as they were: there it IS a mean
      expect(b.regionAvg.description).toMatch(meanWord[lang]);
    }
  });
});

describe('the hover probe', () => {
  const base = { row: 3, col: 4, bc: 120, elements: {}, phase: null };

  it('says the EDS values are pending while the fit runs', () => {
    render(<HoverProbeOverlay visible pos={{ x: 0, y: 0 }} displayMode="at_pct"
      probe={{ ...base, eds_status: { state: 'pending',
        code: 'eds_spectrum_fit_running', progress: 0.37, stage: 'fit' } }} />);
    const el = document.querySelector('[data-probe-eds-status]');
    expect(el).toBeTruthy();
    expect(el.textContent).toContain('spectrumFit.edsPending');
    expect(el.textContent).toContain('"percent":37');
  });

  it('says the fit failed when it did', () => {
    render(<HoverProbeOverlay visible pos={{ x: 0, y: 0 }} displayMode="at_pct"
      probe={{ ...base, eds_status: { state: 'failed',
        code: 'eds_spectrum_fit_failed', error: 'cube unreadable' } }} />);
    const el = document.querySelector('[data-probe-eds-status]');
    expect(el.textContent).toContain('spectrumFit.edsFailed');
    expect(el.textContent).toContain('cube unreadable');
  });

  it('labels counts as net counts on the spectrum source', () => {
    render(<HoverProbeOverlay visible pos={{ x: 0, y: 0 }} displayMode="counts"
      probe={{ ...base, elements: { Al: { counts: 812, wt_pct: 97, at_pct: 97.6 } },
        quantification: { source: 'spectrum' } }} />);
    expect(document.querySelector('[data-probe-net-counts]')).toBeTruthy();
  });

  it('no label in at% mode or on the window source', () => {
    render(<HoverProbeOverlay visible pos={{ x: 0, y: 0 }} displayMode="at_pct"
      probe={{ ...base, quantification: { source: 'spectrum' } }} />);
    expect(document.querySelector('[data-probe-net-counts]')).toBeNull();
    cleanup();
    render(<HoverProbeOverlay visible pos={{ x: 0, y: 0 }} displayMode="counts"
      probe={{ ...base, quantification: { source: 'window' } }} />);
    expect(document.querySelector('[data-probe-net-counts]')).toBeNull();
  });
});

describe('the counts linescan', () => {
  const LAYERS = [{ id: 'eds-Ti Kα1', visible: true }];
  const series = { 'eds-Ti Kα1': [1.2, -0.8, 0.3] };

  it('says the values are net counts and may dip below zero', () => {
    render(<LinescanProfilePlot layers={LAYERS} data={{
      samples: 3, series, display_mode: 'counts',
      quantification: { source: 'spectrum', source_data: 'spectrum fit ...' },
    }} />);
    expect(document.querySelector('[data-linescan-net-counts]').textContent)
      .toContain('linescan.netCountsNote');
    expect(document.querySelector('[data-testid="plot"]').getAttribute('data-ytitle'))
      .toBe('linescan.axisNetCounts');
  });

  it('a window or at% linescan keeps its plain axis', () => {
    render(<LinescanProfilePlot layers={LAYERS} data={{
      samples: 3, series, display_mode: 'counts',
      quantification: { source: 'window' },
    }} />);
    expect(document.querySelector('[data-linescan-net-counts]')).toBeNull();
    expect(document.querySelector('[data-testid="plot"]').getAttribute('data-ytitle'))
      .toBe('linescan.axisValue');
    cleanup();
    render(<LinescanProfilePlot layers={LAYERS} data={{
      samples: 3, series, display_mode: 'at_pct',
      quantification: { source: 'spectrum' },
    }} />);
    expect(document.querySelector('[data-linescan-net-counts]')).toBeNull();
  });
});
