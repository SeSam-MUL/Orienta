import { describe, it, expect } from 'vitest';
import {
  tiltFinding, mapsInherit, stretchFor, stretchedAspect, NO_STRETCH, manualStretch, suggestedManualFactor,
} from './tiltStretch';

const AB5 = {
  befund: 'X gestaucht', faktor_korrektur: 2.92,
  karten: [{ typ: 'EDS', erbt_bildverzerrung: true }],
};

describe('tilt display stretch', () => {
  it('stretches X by k for a confirmed X foreshortening', () => {
    expect(tiltFinding(AB5)).toEqual({ sx: 2.92, sy: 1, axis: 'X', factor: 2.92 });
    expect(stretchFor(AB5, 'electron')).toEqual({ sx: 2.92, sy: 1 });
    expect(stretchFor(AB5, 'eds-element')).toEqual({ sx: 2.92, sy: 1 });
    expect(stretchFor(AB5, 'composite')).toEqual({ sx: 2.92, sy: 1 });
  });

  it('stretches Y for a Y finding', () => {
    expect(stretchFor({ ...AB5, befund: 'Y gestaucht' }, 'electron')).toEqual({ sx: 1, sy: 2.92 });
  });

  it('never corrects an unclear, undistorted or failed check, or with the switch off', () => {
    expect(stretchFor({ ...AB5, befund: 'unklar' }, 'electron')).toBe(NO_STRETCH);
    expect(stretchFor({ ...AB5, befund: 'unverzerrt', faktor_korrektur: 1 }, 'electron')).toBe(NO_STRETCH);
    expect(stretchFor({ ...AB5, fehler: 'x' }, 'electron')).toBe(NO_STRETCH);
    expect(stretchFor(null, 'electron')).toBe(NO_STRETCH);
    expect(stretchFor(AB5, 'electron', false)).toBe(NO_STRETCH);
  });

  it('leaves maps on a raster of their own, corrects them when it cannot tell', () => {
    const own = { ...AB5, karten: [{ typ: 'EDS', erbt_bildverzerrung: false }] };
    expect(mapsInherit(own)).toBe(false);
    expect(stretchFor(own, 'eds-element')).toBe(NO_STRETCH);
    expect(stretchFor(own, 'electron')).toEqual({ sx: 2.92, sy: 1 });
    const unknown = { ...AB5, karten: [{ typ: 'EDS', erbt_bildverzerrung: null }] };
    expect(stretchFor(unknown, 'composite')).toEqual({ sx: 2.92, sy: 1 });
  });

  it('gives the CSS aspect ratio of the stretched box', () => {
    expect(stretchedAspect(1024, 768)).toBe('1024/768');
    expect(stretchedAspect(1024, 768, { sx: 2.92, sy: 1 })).toBe('2990.08/768');
    expect(stretchedAspect(12, 9, { sx: 1, sy: 2 })).toBe('12/18');
  });
});

describe('manual correction', () => {
  const unclear = { befund: 'unklar', faktor_k_theoretisch: 2.92, karten: [] };

  it('stretches what the user chose when the check could not confirm', () => {
    expect(stretchFor(unclear, 'electron', true, { axis: 'X', factor: 2.92 })).toEqual({ sx: 2.92, sy: 1 });
    expect(stretchFor(unclear, 'eds-element', true, { axis: 'Y', factor: 2 })).toEqual({ sx: 1, sy: 2 });
    expect(stretchFor(null, 'composite', true, { axis: 'X', factor: 3 })).toEqual({ sx: 3, sy: 1 });
  });

  it('never overrides a confirmed finding, and honours the switch', () => {
    const confirmed = { befund: 'X gestaucht', faktor_korrektur: 2.92, karten: [] };
    expect(stretchFor(confirmed, 'electron', true, { axis: 'Y', factor: 5 })).toEqual({ sx: 2.92, sy: 1 });
    expect(stretchFor(unclear, 'electron', false, { axis: 'X', factor: 2.92 })).toBe(NO_STRETCH);
  });

  it('ignores an unusable factor', () => {
    expect(manualStretch({ axis: 'X', factor: 1 })).toBeNull();
    expect(manualStretch({ axis: 'X', factor: 50 })).toBeNull();
    expect(manualStretch({ axis: 'Z', factor: 2 })).toBeNull();
  });

  it('suggests 1/cos of the header tilt, else 2.92', () => {
    expect(suggestedManualFactor({ faktor_k_theoretisch: 1.555 })).toBe(1.56);
    expect(suggestedManualFactor({ faktor_k_theoretisch: 1 })).toBe(2.92);
    expect(suggestedManualFactor(null)).toBe(2.92);
  });
});
