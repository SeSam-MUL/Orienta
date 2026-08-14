import { describe, it, expect } from 'vitest';
import { areaForLayer, pixelSizeForLayer } from './layerPixelSize';

// Real values, read from the user's "Arbeitsbereich 6" h5oina.
const SIZES = {
  ebsd: { x: 0.6578947, y: 0.6578947, units: 'um', source: 'step' },
  eds: { x: 0.6578947, y: 0.6578947, units: 'um', source: 'step' },
  electron_image: { x: 0.06207658, y: 0.06207658, units: 'um', source: 'step' },
};

describe('areaForLayer', () => {
  it('routes each layer id to its acquisition area', () => {
    expect(areaForLayer('electron-SE/Elektronenbild 41 (Input1)')).toBe('electron_image');
    expect(areaForLayer('eds-Al Kα1')).toBe('eds');
    expect(areaForLayer('phase')).toBe('ebsd');
    expect(areaForLayer('ipf-z')).toBe('ebsd');
    expect(areaForLayer('bc')).toBe('ebsd');
    expect(areaForLayer('ci')).toBe('ebsd');
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
