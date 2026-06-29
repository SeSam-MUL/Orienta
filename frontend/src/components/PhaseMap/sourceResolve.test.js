import { describe, it, expect } from 'vitest';
import { resolveSource, alignShape } from './sourceResolve';

describe('resolveSource', () => {
  it('links when result.source_path matches ebsd info path', () => {
    const r = resolveSource({
      resultEntry: { data: { source_path: '/tmp/sample.h5oina', shape: [128, 128] } },
      ebsdInfo: { path: '/tmp/sample.h5oina', shape: [128, 128] },
      analysisStatus: null,
      manualOverride: null,
    });
    expect(r.linked).toBe(true);
    expect(r.sourcePath).toBe('/tmp/sample.h5oina');
    expect(r.reason).toBeNull();
  });

  it('falls back to ebsd info when result has no source_path', () => {
    const r = resolveSource({
      resultEntry: { data: { shape: [256, 256] } },
      ebsdInfo: { path: '/tmp/x.h5oina', shape: [256, 256] },
      analysisStatus: null,
      manualOverride: null,
    });
    expect(r.linked).toBe(true);
    expect(r.sourcePath).toBe('/tmp/x.h5oina');
  });

  it('manual override always wins', () => {
    const r = resolveSource({
      resultEntry: { data: { source_path: '/auto.h5oina' } },
      ebsdInfo: { path: '/wrong.h5oina' },
      analysisStatus: null,
      manualOverride: { path: '/manual.h5oina', shape: [128, 128] },
      resultShape: [128, 128],
    });
    expect(r.sourcePath).toBe('/manual.h5oina');
    expect(r.linked).toBe(true);
  });

  it('unlinked when no source info available', () => {
    const r = resolveSource({
      resultEntry: null, ebsdInfo: null, analysisStatus: null, manualOverride: null,
    });
    expect(r.linked).toBe(false);
    expect(r.reason).toMatch(/no source/i);
  });

  it('unlinked with reason when shape mismatches', () => {
    const r = resolveSource({
      resultEntry: { data: { source_path: '/x.h5oina' } },
      ebsdInfo: { path: '/x.h5oina', shape: [256, 256] },
      analysisStatus: null,
      manualOverride: null,
      resultShape: [128, 128],
    });
    expect(r.linked).toBe(false);
    expect(r.reason).toMatch(/shape mismatch/i);
  });
});

describe('alignShape', () => {
  it('returns same shape when result == source', () => {
    const a = alignShape({ resultShape: [128, 128], sourceShape: [128, 128] });
    expect(a.aligned).toBe(true);
    expect(a.mode).toBe('exact');
  });
  it('auto-crops when offset metadata present and shapes fit', () => {
    const a = alignShape({
      resultShape: [128, 128],
      sourceShape: [256, 256],
      cropOffset: [64, 64],
    });
    expect(a.aligned).toBe(true);
    expect(a.mode).toBe('crop');
    expect(a.crop).toEqual({ row0: 64, row1: 192, col0: 64, col1: 192 });
  });
  it('not aligned when no crop offset and shapes differ', () => {
    const a = alignShape({ resultShape: [128, 128], sourceShape: [256, 256] });
    expect(a.aligned).toBe(false);
    expect(a.reason).toMatch(/shape mismatch/i);
  });
  it('not aligned when crop would exceed source bounds', () => {
    const a = alignShape({
      resultShape: [128, 128],
      sourceShape: [150, 150],
      cropOffset: [64, 64],
    });
    expect(a.aligned).toBe(false);
  });
});
