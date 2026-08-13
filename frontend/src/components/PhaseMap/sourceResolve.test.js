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

describe('resolveSource — the payloads the backend really sends', () => {
  // These are verbatim response shapes: /api/ebsd/info answers with file_path
  // and a 4-D data_shape, /api/indexing/results with source_file. Reading the
  // wrong key made every result look source-less, which removed every EDS and
  // electron-image layer from the Phase Map layer list without saying why.
  const EBSD_INFO = {
    loaded: true,
    file_path: 'C:/data/SampleB.h5oina',
    data_shape: [90, 120, 128, 156],
  };

  it('links through /api/ebsd/info file_path + data_shape', () => {
    const r = resolveSource({ resultEntry: null, ebsdInfo: EBSD_INFO, resultShape: [90, 120] });
    expect(r.linked).toBe(true);
    expect(r.sourcePath).toBe('C:/data/SampleB.h5oina');
    expect(r.sourceShape).toEqual([90, 120]);
    expect(r.reason).toBeNull();
  });

  it('links through a gallery entry carrying source_file', () => {
    const r = resolveSource({
      resultEntry: { data: { source_file: 'C:/data/Other.h5oina' } },
      ebsdInfo: null,
      resultShape: [90, 120],
    });
    expect(r.linked).toBe(true);
    expect(r.sourcePath).toBe('C:/data/Other.h5oina');
  });

  it('still refuses a source of the wrong size', () => {
    const r = resolveSource({ resultEntry: null, ebsdInfo: EBSD_INFO, resultShape: [174, 145] });
    expect(r.linked).toBe(false);
    expect(r.reason).toMatch(/mismatch/i);
  });
});
