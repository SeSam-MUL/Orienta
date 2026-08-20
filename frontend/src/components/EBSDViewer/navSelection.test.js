import { describe, it, expect } from 'vitest';
import {
  boundsOf, rectMask, countSelected, estimateBytes, formatBytes, bytesPerSample,
} from './navSelection';

describe('boundsOf', () => {
  it('spans the drawn points inclusively', () => {
    expect(boundsOf([{ r: 4, c: 9 }, { r: 7, c: 2 }]))
      .toEqual({ row0: 4, col0: 2, rows: 4, cols: 8 });
  });

  it('is a single pixel for a single point', () => {
    expect(boundsOf([{ r: 3, c: 3 }]))
      .toEqual({ row0: 3, col0: 3, rows: 1, cols: 1 });
  });

  it('returns null for nothing drawn', () => {
    expect(boundsOf([])).toBeNull();
    expect(boundsOf(null)).toBeNull();
  });
});

describe('rectMask', () => {
  it('is null — a rectangle selects its whole box', () => {
    expect(rectMask({ row0: 0, col0: 0, rows: 3, cols: 4 })).toBeNull();
  });
});

describe('countSelected', () => {
  it('counts the whole box when there is no mask', () => {
    expect(countSelected(null, { row0: 0, col0: 0, rows: 3, cols: 4 })).toBe(12);
  });

  it('counts the true entries of a mask', () => {
    const mask = new Uint8Array(12);
    mask[0] = 1; mask[5] = 1;
    expect(countSelected(mask, { row0: 0, col0: 0, rows: 3, cols: 4 })).toBe(2);
  });
});

describe('estimateBytes', () => {
  it('is pixels times pattern size times depth', () => {
    // 8x10 pixels of 128x156 uint8
    expect(estimateBytes(8, 10, [128, 156], 1)).toBe(8 * 10 * 128 * 156);
  });

  it('scales with the sample depth', () => {
    expect(estimateBytes(2, 2, [4, 4], 2)).toBe(128);
  });

  it('is 0 when the pattern shape is unknown', () => {
    expect(estimateBytes(8, 10, null, 1)).toBe(0);
  });
});

describe('formatBytes', () => {
  it('reads as a person would say it', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(1536)).toBe('1.5 KB');
    expect(formatBytes(256 * 1024 * 1024)).toBe('256.0 MB');
    expect(formatBytes(3 * 1024 ** 3)).toBe('3.0 GB');
  });
});

describe('bytesPerSample', () => {
  it('reads the width out of the numpy dtype name', () => {
    expect(bytesPerSample('uint8')).toBe(1);
    expect(bytesPerSample('uint16')).toBe(2);
    expect(bytesPerSample('int16')).toBe(2);
    expect(bytesPerSample('float32')).toBe(4);
    expect(bytesPerSample('float64')).toBe(8);
  });

  it('falls back to 1 when the dtype is unknown — the pre-dtype assumption', () => {
    expect(bytesPerSample(null)).toBe(1);
    expect(bytesPerSample(undefined)).toBe(1);
    expect(bytesPerSample('')).toBe(1);
    expect(bytesPerSample('bool')).toBe(1);
    expect(bytesPerSample('uint4')).toBe(1);   // sub-byte: never less than 1
  });
});
