/**
 * A diagnostic layer must not decide how large the map is drawn.
 *
 * Pressing "Check phases" adds the Phase Check layer, which is transparent
 * everywhere except the grains it flagged. It was voting on the frame, and
 * because `computeAlphaBbox` deliberately abstains for a layer that covers the
 * whole canvas, the full-coverage phase map underneath said nothing and the
 * sparse layer won on its own: a 39x136 scan collapsed onto ~20 pixels, drawn
 * with a 100 nm scalebar instead of 2 um (reported 2026-09-07).
 *
 * The frame is a property of the RESULT — which region was indexed. How many
 * pixels a check found wrong is a finding, not a region.
 */
import { describe, it, expect } from 'vitest';
import { definesRoi, roiDefiningLayers } from './roiFrame';
import { findLayerDef } from './layerSources';

const L = (id, source) => ({ id, source, visible: true });

describe('sparse diagnostics do not define the frame', () => {
  it('excludes Phase Check', () => {
    expect(definesRoi(L('phase-margin', 'diagnostics'))).toBe(false);
  });

  it('excludes Assignment Source', () => {
    expect(definesRoi(L('assignment-source', 'diagnostics'))).toBe(false);
  });

  it('keeps the DENSE diagnostics, which do cover the whole result', () => {
    ['forward-ncc', 'local-anomaly', 'pc-sensitivity', 'pattern-residual']
      .forEach((id) => expect(definesRoi(L(id, 'diagnostics')), id).toBe(true));
  });

  it('leaves the ordinary result and analysis layers alone', () => {
    expect(definesRoi(L('phase', 'result'))).toBe(true);
    expect(definesRoi(L('ipf-z', 'result'))).toBe(true);
    expect(definesRoi(L('kam', 'analysis'))).toBe(true);
    expect(definesRoi(L('refined-ncc', 'refinement'))).toBe(true);
  });

  it('still excludes full-scan sources', () => {
    expect(definesRoi({ id: 'eds-Al', source: 'h5', visible: true })).toBe(false);
  });

  it('still excludes hidden layers', () => {
    expect(definesRoi({ id: 'phase', source: 'result', visible: false })).toBe(false);
  });
});

describe('the user’s stack after pressing Check phases', () => {
  it('is framed by the phase map, not by the check', () => {
    const stack = [L('phase', 'result'), L('phase-margin', 'diagnostics')];
    expect(roiDefiningLayers(stack).map((l) => l.id)).toEqual(['phase']);
  });

  it('a stack of ONLY the sparse layer yields no voter, so the frame is kept', () => {
    // roiDefiningLayers returning [] is the documented "keep what we had"
    // signal — better than snapping to 20 pixels or to the full scan.
    expect(roiDefiningLayers([L('phase-margin', 'diagnostics')])).toEqual([]);
  });
});

describe('the flag lives in the catalogue, so both readers agree', () => {
  it('marks exactly the two sparse layers', () => {
    expect(findLayerDef('phase-margin').sparse).toBe(true);
    expect(findLayerDef('assignment-source').sparse).toBe(true);
    expect(findLayerDef('forward-ncc').sparse).toBeUndefined();
  });

  it('an explicit per-layer override still wins over the catalogue', () => {
    expect(definesRoi({ ...L('phase-margin', 'diagnostics'), sparse: false })).toBe(true);
    expect(definesRoi({ ...L('phase', 'result'), sparse: true })).toBe(false);
  });
});
