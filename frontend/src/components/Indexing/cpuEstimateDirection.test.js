/**
 * The CPU warning must not put the fast machine on the bigger number.
 *
 * M5 tester, 2026-09-25 (report point B5b): the dialog before a spherical run
 * read
 *
 *   "for 441 patterns expect roughly 18 s instead of a few minutes on a GPU"
 *
 * which has it exactly backwards — 441 patterns on a CUDA GPU take under a
 * second. The sentence was written when the CPU fallback was assumed to cost
 * hours, and "a few minutes on a GPU" was hard-coded prose that no measurement
 * ever produced.
 *
 * Both numbers now come from the measured rate tables, so one cannot end up
 * larger than the other by accident, and these tests pin the ordering rather
 * than the wording.
 */
import { describe, it, expect } from 'vitest';
import {
  estimateCpuSphericalSeconds,
  estimateGpuSphericalSeconds,
  cpuPatternsPerSecond,
  GPU_REFERENCE_RATE,
  formatRoughDuration,
} from './cpuEstimate';
import en from '../../locales/en/indexing.json';
import de from '../../locales/de/indexing.json';
import ja from '../../locales/ja/indexing.json';
import zh from '../../locales/zh/indexing.json';

const LANGS = [['en', en], ['de', de], ['ja', ja], ['zh', zh]];

describe('the GPU is always the faster one', () => {
  it.each([441, 10800, 1, 196608])('for %i patterns', (n) => {
    const cpu = estimateCpuSphericalSeconds(n, 88);
    const gpu = estimateGpuSphericalSeconds(n);
    expect(gpu).toBeLessThan(cpu);
  });

  it('across every bandwidth bucket', () => {
    for (const bw of [63, 68, 88, 96, 128, 256]) {
      expect(estimateGpuSphericalSeconds(441))
        .toBeLessThan(estimateCpuSphericalSeconds(441, bw));
      // and the reference rate really is the larger throughput
      expect(cpuPatternsPerSecond(bw)).toBeLessThan(GPU_REFERENCE_RATE);
    }
  });

  it('the tester\'s own case reads the right way round', () => {
    // 441 patterns: ~18 s on this CPU table, well under a second on the GPU.
    const cpu = formatRoughDuration(estimateCpuSphericalSeconds(441, 88));
    const gpu = formatRoughDuration(estimateGpuSphericalSeconds(441));
    expect(cpu).toBe('18 s');
    expect(gpu).toBe('1 s');            // floored at 1 s by formatRoughDuration
  });

  it('returns 0 for nonsense rather than a misleading number', () => {
    for (const bad of [0, -5, NaN, undefined, null, 'x']) {
      expect(estimateGpuSphericalSeconds(bad)).toBe(0);
    }
  });
});

describe('the sentences name both numbers, in all four languages', () => {
  it.each(LANGS)('%s', (lang, dict) => {
    for (const key of ['cpuFallbackWithEstimate', 'cpuConfirmBodyWithEstimate']) {
      const text = dict.spherical[key];
      expect(text, `${lang}.${key}`).toBeTruthy();
      expect(text, `${lang}.${key} must name the CPU estimate`).toContain('{{estimate}}');
      expect(text, `${lang}.${key} must name the GPU estimate`).toContain('{{gpuEstimate}}');
      // the old prose promised a fixed "few minutes on a GPU"
      expect(text).not.toMatch(/few minutes on a GPU|Minuten auf einer GPU/i);
    }
  });

  it('and there is a label for a machine without CUDA', () => {
    for (const [lang, dict] of LANGS) {
      const label = dict.spherical.backendGpuOnCpu;
      expect(label, lang).toBeTruthy();
      // it must not keep promising a GPU
      expect(label, lang).not.toMatch(/^GPU\b/);
      expect(label, lang).toMatch(/CPU/);
    }
  });

  it('the plain GPU label is untouched for machines that have one', () => {
    expect(en.spherical.backendGpu).toContain('GPU');
  });
});
