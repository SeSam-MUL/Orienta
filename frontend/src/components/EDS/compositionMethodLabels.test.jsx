// @vitest-environment jsdom
/**
 * One region, one composition -- and the screen says which kind.
 *
 * On the spectrum source a region's composition is its summed net counts
 * quantified once (`_SummedNetComposition`); on the window source it is the
 * mean of its pixels' at%. Every region composition the backend sends says
 * which in `composition_method` ("summed_net_counts" |
 * "mean_of_pixel_at_pct"): the inspector, the region list (legend), the
 * region-definition preview and the wand. Before, the inspector's hint said
 * "Average of every pixel in this region" on both -- false on the one that is
 * now the default.
 *
 * The auto-classify cluster report is a THIRD thing again: the classifier's
 * mean of each group's smoothed interior pixels. It sits next to the region
 * list and must not read as the same number.
 */
import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, cleanup, fireEvent, screen, waitFor } from '@testing-library/react';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k, o) => (o ? `${k}:${JSON.stringify(o)}` : k) }),
}));
const previewRegionDefs = vi.fn();
vi.mock('../../services/api', () => ({
  edsApi: { previewRegionDefs: (...a) => previewRegionDefs(...a) },
}));

import en from '../../locales/en/eds.json';
import de from '../../locales/de/eds.json';
import ja from '../../locales/ja/eds.json';
import zh from '../../locales/zh/eds.json';
import RegionInspector from './RegionInspector';
import RegionPanel from './RegionPanel';
import RegionDefs from './RegionDefs';
import { compositionMethodKey } from './compositionMethod';

beforeEach(() => { cleanup(); vi.clearAllMocks(); });
afterEach(cleanup);

/** `_region_detail` on the spectrum source, trimmed. */
const DETAIL = {
  region_id: 2, n_pixels: 202, percentage: 1.9, color: '#ff0000',
  phase_index: -1, cif_filename: null, formula: null,
  elements: [{ element: 'Si', at_pct: 66.7, spread: 9.1, enrichment: 7.7 },
             { element: 'Al', at_pct: 32.0, spread: 9.0, enrichment: 0.5 }],
  pieces: [202], n_pieces: 1, neighbours: [], candidates: [],
  enrichment_factor: 1.15, composition_method: 'summed_net_counts',
};

describe('compositionMethodKey', () => {
  it('maps the two backend values and nothing else', () => {
    expect(compositionMethodKey('summed_net_counts')).toBe('composition.summed');
    expect(compositionMethodKey('mean_of_pixel_at_pct')).toBe('composition.mean');
    expect(compositionMethodKey(null)).toBeNull();
    expect(compositionMethodKey(undefined)).toBeNull();
    expect(compositionMethodKey('something_new')).toBeNull();
  });
});

describe('the inspector', () => {
  it('says the composition is the summed net counts on the spectrum source', () => {
    render(<RegionInspector detail={DETAIL} />);
    expect(screen.getByText('inspector.compositionHintSummed')).toBeTruthy();
    expect(screen.queryByText('inspector.compositionHint')).toBeNull();
  });

  it('keeps the mean-of-pixels hint on the window source', () => {
    render(<RegionInspector detail={{ ...DETAIL, composition_method: 'mean_of_pixel_at_pct' }} />);
    expect(screen.getByText('inspector.compositionHint')).toBeTruthy();
    expect(screen.queryByText('inspector.compositionHintSummed')).toBeNull();
  });
});

function panelHandle(regions) {
  return {
    phaseMap: { loaded: true, n_rows: 10, n_cols: 10, regions, all_phases: [] },
    regionBusy: false, selectedRegionId: null, setSelectedRegionId: vi.fn(),
    scale: 5, setScale: vi.fn(), nClusters: null, setNClusters: vi.fn(),
    handleMergeRegions: vi.fn(), handleSplitRegion: vi.fn(),
    handleGrowRegion: vi.fn(), handleSnapEdges: vi.fn(),
    setMapView: vi.fn(), background: { id: 'none' }, backgroundChoices: [],
    mode: 'cluster', regionDefs: [],
  };
}
const REGION = {
  region_id: 0, n_pixels: 60, percentage: 60, color: '#00ff00',
  phase_index: -1, cif_filename: null, formula: null,
  mean_at_pct: { Al: 97.1, Fe: 1.2 }, composition_method: 'summed_net_counts',
};

describe('the region list', () => {
  it('says how its compositions were made', () => {
    render(<RegionPanel handle={panelHandle([REGION])} />);
    const el = document.querySelector('[data-composition-method]');
    expect(el).toBeTruthy();
    expect(el.getAttribute('data-composition-method')).toBe('summed_net_counts');
    expect(el.textContent).toContain('composition.summed');
  });

  it('says nothing about it while compositions are left out', () => {
    // `_regions_payload` sends composition_method null while the fit runs.
    render(<RegionPanel handle={panelHandle([
      { ...REGION, mean_at_pct: {}, composition_method: null }])} />);
    expect(document.querySelector('[data-composition-method]')).toBeNull();
  });
});

describe('the region-definition preview', () => {
  it('labels the claimed pixels\' composition with its method', async () => {
    previewRegionDefs.mockResolvedValue({ data: {
      definitions: [{ name: 'a', n_pixels: 202, percentage: 1.9,
                      overlap_pixels: 0, reason: null,
                      mean_at_pct: { Al: 58.1, Si: 40.3 } }],
      unclaimed_pixels: 0, unclaimed_percentage: 0,
      composition_method: 'summed_net_counts' } });
    render(<RegionDefs
      defs={[{ name: 'a', elements: [{ element: 'Si', min_at_pct: 20 }] }]}
      setDefs={vi.fn()} elements={['Al', 'Si']} allPhases={[]}
      clusterRemainder setClusterRemainder={vi.fn()}
      elementWeights={{}} setElementWeights={vi.fn()}
      onReclassify={vi.fn()} busy={false} />);
    fireEvent.click(screen.getByText('defs.preview'));
    const comp = await waitFor(() => screen.getByText(/Al 58\.1 · Si 40\.3 at%/));
    expect(comp.getAttribute('data-composition-method')).toBe('summed_net_counts');
    // The plain tooltip says "the average composition"; on this source it
    // is not an average.
    expect(comp.getAttribute('title')).toBe('defs.previewMeanTooltipSummed');
  });
});

describe('every language says it', () => {
  it('has the method sentences and the summed inspector hint, translated', () => {
    for (const [lang, b] of Object.entries({ en, de, ja, zh })) {
      for (const k of ['summed', 'mean', 'classifier']) {
        expect(b.composition?.[k], `${lang}.composition.${k}`).toBeTruthy();
        if (lang !== 'en') {
          expect(b.composition[k], `${lang}.composition.${k}`).not.toBe(en.composition[k]);
        }
      }
      expect(b.inspector.compositionHintSummed, lang).toBeTruthy();
      expect(b.defs.previewMeanTooltipSummed, lang).toBeTruthy();
    }
  });
});
