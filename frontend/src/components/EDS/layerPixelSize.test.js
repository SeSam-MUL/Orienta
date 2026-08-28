import { describe, it, expect } from 'vitest';
import {
  areaForLayer, pixelSizeForLayer, umPerPxForLayer, scanUmPerPx, maskSourceId,
} from './layerPixelSize';

// Real values, read from the user's "Arbeitsbereich 6" h5oina.
const SIZES = {
  ebsd: { x: 0.6578947, y: 0.6578947, units: 'um', source: 'step' },
  eds: { x: 0.6578947, y: 0.6578947, units: 'um', source: 'step' },
  electron_image: { x: 0.06207658, y: 0.06207658, units: 'um', source: 'step' },
};

describe('areaForLayer', () => {
  it('routes each EDS-page layer id to its acquisition area', () => {
    expect(areaForLayer('electron-SE/Elektronenbild 41 (Input1)')).toBe('electron_image');
    expect(areaForLayer('eds-Al Kα1')).toBe('eds');
    expect(areaForLayer('phase')).toBe('ebsd');
    expect(areaForLayer('ipf-z')).toBe('ebsd');
    expect(areaForLayer('bc')).toBe('ebsd');
    expect(areaForLayer('ci')).toBe('ebsd');
  });

  it('routes the PHASE-MAP page ids too — one resolver, both conventions', () => {
    // useLayerStack names the same two areas `se:` and `eds:`. That page had
    // no pixel-size rule at all and handed its scan step to every export,
    // including exports of an SE layer at the SEM raster.
    expect(areaForLayer('se:SE/Elektronenbild 1')).toBe('electron_image');
    expect(areaForLayer('eds:Al Kα1')).toBe('eds');
    expect(areaForLayer('vbse')).toBe('ebsd');
    expect(areaForLayer('forward-ncc')).toBe('ebsd');
    expect(areaForLayer('grain-boundaries')).toBe('ebsd');
  });

  it('resolves a mask through the layer it masks, by id or by object', () => {
    // `mask-<sourceId>-<stamp>`; the source id itself carries hyphens and
    // slashes, so only the wrapper is stripped.
    const src = 'electron-SE/Elektronenbild 41 (Input1)';
    expect(maskSourceId(`mask-${src}-1756300000000`)).toBe(src);
    expect(areaForLayer(`mask-${src}-1756300000000`)).toBe('electron_image');
    expect(areaForLayer({ id: 'mask-x-1', kind: 'mask', isMaskFor: 'se:SE 1' }))
      .toBe('electron_image');
    expect(areaForLayer({ id: 'phase' })).toBe('ebsd');
  });

  it('does not mistake a non-mask id that merely starts with "mask"', () => {
    expect(maskSourceId('masked-something')).toBe('masked-something');
    expect(maskSourceId('mask-eds-Al')).toBe('mask-eds-Al');  // no -<stamp>
  });
});

describe('umPerPxForLayer / scanUmPerPx', () => {
  it('hand back a bare number or null, never undefined or NaN', () => {
    expect(umPerPxForLayer('eds-Al Kα1', SIZES)).toBeCloseTo(0.6578947, 6);
    expect(umPerPxForLayer('electron-SE', { electron_image: null })).toBeNull();
    expect(umPerPxForLayer('phase', null)).toBeNull();
    expect(umPerPxForLayer('phase', { ebsd: { x: NaN } })).toBeNull();
    expect(umPerPxForLayer('phase', { ebsd: { x: -1 } })).toBeNull();
  });

  it('scanUmPerPx is the scan raster, whatever else is loaded', () => {
    expect(scanUmPerPx(SIZES)).toBeCloseTo(0.6578947, 6);
    expect(scanUmPerPx({ ebsd: null, eds: null, electron_image: SIZES.electron_image }))
      .toBeNull();
  });
});

describe('pixelSizeForLayer', () => {
  it('gives an electron image its OWN pixel size, not the scan step', () => {
    // The bug: one global step size for every tile -> SE scale bar off by 10.6x.
    const got = pixelSizeForLayer('electron-SE/Elektronenbild 41 (Input1)', SIZES);
    expect(got.x).toBeCloseTo(0.06207658, 6);
    expect(got.x).not.toBeCloseTo(SIZES.ebsd.x, 3);
  });

  it('gives an EDS map the scan step', () => {
    expect(pixelSizeForLayer('eds-Fe Kα1', SIZES).x).toBeCloseTo(0.6578947, 6);
  });

  it('gives phase / IPF / BC the scan step', () => {
    for (const id of ['phase', 'ipf-z', 'bc', 'ci']) {
      expect(pixelSizeForLayer(id, SIZES).x).toBeCloseTo(0.6578947, 6);
    }
  });

  it('substitutes EDS for EBSD when there is no EBSD area', () => {
    // EDS-only acquisitions have no EBSD header; Aztec writes both on the
    // same scan raster, verified equal on both real files.
    const edsOnly = { ebsd: null, eds: SIZES.eds, electron_image: SIZES.electron_image };
    expect(pixelSizeForLayer('phase', edsOnly).x).toBeCloseTo(0.6578947, 6);
  });

  it('returns null rather than borrowing another area for an electron image', () => {
    const noElectron = { ebsd: SIZES.ebsd, eds: SIZES.eds, electron_image: null };
    // A bar from the 10.6x-off scan step would look authoritative and be wrong.
    expect(pixelSizeForLayer('electron-SE', noElectron)).toBeNull();
  });

  it('returns null when nothing is known', () => {
    expect(pixelSizeForLayer('eds-Al Kα1', null)).toBeNull();
    expect(pixelSizeForLayer('eds-Al Kα1', {})).toBeNull();
    expect(pixelSizeForLayer('eds-Al Kα1', { eds: { x: 0 } })).toBeNull();
  });

  it('handles the EDS-only file, where both areas share one scale', () => {
    const a12 = {
      ebsd: null,
      eds: { x: 0.1814546, y: 0.1814546, units: 'um', source: 'step' },
      electron_image: { x: 0.1814546, y: 0.1814546, units: 'um', source: 'step' },
    };
    expect(pixelSizeForLayer('eds-Al Kα1', a12).x).toBeCloseTo(0.1814546, 6);
    expect(pixelSizeForLayer('electron-SE/Elektronenbild 41 (Input1)', a12).x)
      .toBeCloseTo(0.1814546, 6);
  });
});
