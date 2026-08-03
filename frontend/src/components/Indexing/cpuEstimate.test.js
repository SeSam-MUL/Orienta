import { describe, it, expect } from 'vitest';
import {
  cpuPatternsPerSecond,
  estimateCpuSphericalSeconds,
  formatRoughDuration,
} from './cpuEstimate';

describe('cpuPatternsPerSecond', () => {
  it('buckets by bandwidth, and L=128 is much slower than L=68', () => {
    expect(cpuPatternsPerSecond(68)).toBeGreaterThan(cpuPatternsPerSecond(88));
    expect(cpuPatternsPerSecond(88)).toBeGreaterThan(cpuPatternsPerSecond(128));
    // measured: L=128 is ~6x the cost of L=68, not ~1.2x
    expect(cpuPatternsPerSecond(68) / cpuPatternsPerSecond(128)).toBeGreaterThan(3);
  });

  it('accepts the string the <select> actually gives us', () => {
    expect(cpuPatternsPerSecond('88')).toBe(cpuPatternsPerSecond(88));
  });

  it('falls back to the L=88 default for junk input', () => {
    expect(cpuPatternsPerSecond(undefined)).toBe(cpuPatternsPerSecond(88));
  });
});

describe('estimateCpuSphericalSeconds', () => {
  it('scales linearly with the pattern count', () => {
    const one = estimateCpuSphericalSeconds(1000, 88);
    expect(estimateCpuSphericalSeconds(2000, 88)).toBeCloseTo(2 * one, 6);
  });

  it('puts the reported 196k-pattern scan in the hours range, not minutes', () => {
    const hours = estimateCpuSphericalSeconds(196608, 88) / 3600;
    expect(hours).toBeGreaterThan(1);
    expect(hours).toBeLessThan(6);
  });

  it('returns 0 when the pattern count is unknown', () => {
    expect(estimateCpuSphericalSeconds(0, 88)).toBe(0);
    expect(estimateCpuSphericalSeconds(NaN, 88)).toBe(0);
    expect(estimateCpuSphericalSeconds(undefined, 88)).toBe(0);
  });
});

describe('formatRoughDuration', () => {
  it('stays coarse so it never reads as a precise promise', () => {
    expect(formatRoughDuration(45)).toBe('45 s');
    expect(formatRoughDuration(600)).toBe('10 min');
    expect(formatRoughDuration(3600 * 2.5)).toBe('2.5 h');
    expect(formatRoughDuration(3600 * 12)).toBe('12 h');
  });

  it('renders nothing for unknown durations', () => {
    expect(formatRoughDuration(0)).toBe('');
    expect(formatRoughDuration(NaN)).toBe('');
  });
});
