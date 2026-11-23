import { describe, it, expect } from 'vitest';
import {
  nextParticleName, addParticle, discSize, particleRequest, fmtNcc, phaseRow,
} from './particleNcc';

describe('particle comparison helpers', () => {
  it('picks the next free name', () => {
    expect(nextParticleName([])).toBe('P1');
    expect(nextParticleName([{ name: 'P1' }, { name: 'P3' }])).toBe('P2');
  });

  it('refuses an empty or repeated name and clamps the radius', () => {
    expect(addParticle([], { name: ' ', pixelIndex: 3, radius: 2 }).error).toBe('particleNameMissing');
    expect(addParticle([{ name: 'VL12a' }], { name: 'VL12a', pixelIndex: 3, radius: 2 }).error)
      .toBe('particleNameTaken');
    const out = addParticle([], { name: 'VL12a', pixelIndex: 7, radius: 99 });
    expect(out.particles).toEqual([{ name: 'VL12a', center_index: 7, radius_px: 10 }]);
  });

  it('counts a disc the way the backend cuts it', () => {
    expect(discSize(5, 1, 3, 4)).toBe(5);
    expect(discSize(0, 1, 3, 4)).toBe(3);
    expect(discSize(0, 0, 3, 4)).toBe(1);
    expect(discSize(17, 1, 4, 5)).toBe(4);
  });

  it('sends the selected phases in the picker order with the dialog settings', () => {
    const body = particleRequest({
      particles: [{ name: 'A', center_index: 4, radius_px: 2 }],
      phases: [{ key: 'mg2si' }, { key: 'al' }, { key: 'al13fe4' }],
      selectedKeys: new Set(['al13fe4', 'mg2si']),
      topN: 10,
      settings: { bandwidth: 128, bgRemove: true, usePixelPc: false, aperture: 'circular', apertureRadius: 1 },
    });
    expect(body.phase_keys).toEqual(['mg2si', 'al13fe4']);
    expect(body).toMatchObject({ top_n: 10, max_bandwidth: 128, bg_remove: true, aperture: 'circular' });
    expect(body.particles[0]).toEqual({ name: 'A', center_index: 4, radius_px: 2 });
  });

  it('formats NCC in the reader\'s locale', () => {
    expect(fmtNcc(0.5412, 'de')).toBe('0,541');
    expect(fmtNcc(0.5412, 'en')).toBe('0.541');
    expect(fmtNcc(null, 'de')).toBe('—');
  });

  it('finds a phase row', () => {
    const p = { phases: [{ phase_key: 'a', mean: 0.5 }] };
    expect(phaseRow(p, 'a').mean).toBe(0.5);
    expect(phaseRow(p, 'b')).toBeNull();
  });
});
