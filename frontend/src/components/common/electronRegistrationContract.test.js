/**
 * The frontend half of the electron-image registration.
 *
 * The backend returns an electron image already cut to the part of the
 * specimen the EBSD/EDS map covers (`?scope=dataset`). That is only correct if
 * every consumer then (a) asks for the dataset scope, (b) draws the returned
 * image over the WHOLE map box without a second placement of its own, and
 * (c) keeps the micrometres-per-pixel of the image's own area for its scale
 * bar. These are the three places a regression would silently re-open the
 * 12-97 px misregistration measured on aluminium-alloy scans, so each is pinned here
 * against the real sources (the pages themselves need a backend to render).
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { pixelSizeForLayer, areaForLayer } from '../EDS/layerPixelSize';
import { fovRatio } from '../EDS/useMapBackground';

const HERE = dirname(fileURLToPath(import.meta.url));
const src = (rel) => readFileSync(resolve(HERE, rel), 'utf8');

describe('who asks for a registered image', () => {
  it('the phase-map se: layer fetches the dataset scope and forwards the verdict', () => {
    const s = src('../PhaseMap/hooks/useLayerStack.js');
    expect(s).toMatch(/getElectronImage\(name, 'dataset'\)/);
    expect(s).toMatch(/crop: res\.data\.crop/);
  });

  it('the EDS layer stack and the EDS background fetch the dataset scope', () => {
    expect(src('../EDS/hooks/useEdsLayerStack.js'))
      .toMatch(/getElectronImage\(layer\.electronName, 'dataset'\)/);
    const bg = src('../EDS/useMapBackground.js');
    expect(bg).toMatch(/getElectronImage\(parsed\.name, 'dataset'\)/);
    expect(bg).toMatch(/setCrop\(res\.data\?\.crop \|\| null\)/);
  });

  it('standalone electron views stay on the stored image (no scope)', () => {
    expect(src('../HDF5Viewer/hooks/useLayers.js')).toMatch(/getElectronImage\(layer\.name\)/);
    expect(src('../HDF5Viewer/LayerRail/LayerThumb.jsx')).toMatch(/getElectronImage\(layer\.name\)/);
  });
});

describe('the returned image is placed over the full map box, once', () => {
  it('LayeredCanvas stretches every layer bitmap over the whole composite', () => {
    const s = src('../PhaseMap/LayeredCanvas.jsx');
    expect(s).toMatch(/offCtx\.drawImage\(bmp, 0, 0, off\.width, off\.height\)/);
  });

  it('the page does not shift or crop an se: bitmap itself', () => {
    // A second placement (offset/size from the header) in the frontend would
    // be applied on top of the backend's cut: double correction.
    const s = src('../PhaseMap/hooks/useLayerStack.js');
    expect(s).not.toMatch(/relative_?offset|relativeOffset|Relative Offset/i);
    const bg = src('../EDS/useMapBackground.js');
    expect(bg).not.toMatch(/relative_?offset|relativeOffset/i);
  });

  it('the EDS background keeps objectFit contain over the same box', () => {
    // The map and its (now registered) background share one box.
    expect(src('../EDS/PhaseMapPanel.jsx'))
      .toMatch(/objectFit:\s*'contain'/);
  });
});

describe('the scale bar of an se: layer is the image area\'s own', () => {
  // An aluminium-alloy scan: scan 0.1 um/px, electron image 0.01685 um/px (read with h5py).
  const sizes = {
    ebsd: { x: 0.1, y: 0.1 }, eds: { x: 0.1, y: 0.1 },
    electron_image: { x: 0.01684935949742794, y: 0.01684935949742794 },
  };

  it('resolves se: and electron- layers to the image step, not the scan step', () => {
    expect(areaForLayer('se:SE/Elektronenbild 1')).toBe('electron_image');
    expect(pixelSizeForLayer('se:SE/Elektronenbild 1', sizes).x).toBeCloseTo(0.01685, 5);
    expect(pixelSizeForLayer('electron-SE/Elektronenbild 1', sizes).x).toBeCloseTo(0.01685, 5);
    expect(pixelSizeForLayer('phase', sizes).x).toBe(0.1);
  });

  it('the registered image is the map\'s field: a bar of L um is L / step image px', () => {
    // Registered image: 807 px wide (see tests/test_electron_registration.py)
    // x 0.016849 um = 13.60 um = the 136-column map. Stretched over the map box,
    // the bar that is `um` micrometres long is `um / 13.6` of its width in both.
    const imgWidthPx = 807;
    const imageField = imgWidthPx * sizes.electron_image.x;
    const mapField = 136 * sizes.ebsd.x;
    expect(imageField / mapField).toBeCloseTo(1, 2);
    expect(fovRatio(sizes, 136, imgWidthPx)).toBeCloseTo(1, 2);   // no warning
    // …whereas the unregistered full image reproduced the 27 % warning
    expect(fovRatio(sizes, 136, 1024)).toBeCloseTo(1.27, 2);
  });
});
