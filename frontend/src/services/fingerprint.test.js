import { describe, it, expect } from 'vitest';
import { errorFingerprint, normalizeMessage, topFrame } from './fingerprint';

const STACK_A = `TypeError: Cannot read properties of undefined
    at PhaseMapPage (https://127.0.0.1:8000/assets/index-yKiAoGkI.js:4821:19)
    at renderWithHooks (https://127.0.0.1:8000/assets/vendor-a1b2c3d4.js:120:5)`;

describe('errorFingerprint', () => {
  it('is stable for the same fault', () => {
    const a = errorFingerprint('react-boundary', 'Cannot read x of undefined', STACK_A);
    const b = errorFingerprint('react-boundary', 'Cannot read x of undefined', STACK_A);
    expect(a).toBe(b);
    expect(a).toMatch(/^[0-9a-f]{8}$/);
  });

  it('matches across users when only numbers differ', () => {
    const a = errorFingerprint('ui-error', 'Pixel 123 has no pattern', STACK_A);
    const b = errorFingerprint('ui-error', 'Pixel 4096 has no pattern', STACK_A);
    expect(a).toBe(b);
  });

  it('matches across users when only file paths differ', () => {
    const a = errorFingerprint('ui-error', 'Could not open D:\\data\\SampleB.h5oina', '');
    const b = errorFingerprint('ui-error', 'Could not open C:\\scans\\Other.h5oina', '');
    expect(a).toBe(b);
  });

  it('survives a rebuild — the bundle content hash is ignored', () => {
    const older = STACK_A;
    const newer = STACK_A.replace('index-yKiAoGkI.js', 'index-ZZ99xxQQ.js')
                         .replace('vendor-a1b2c3d4.js', 'vendor-99887766.js');
    expect(errorFingerprint('react-boundary', 'boom', older))
      .toBe(errorFingerprint('react-boundary', 'boom', newer));
  });

  it('ignores line and column drift from unrelated edits', () => {
    const moved = STACK_A.replace(':4821:19', ':5307:23');
    expect(errorFingerprint('react-boundary', 'boom', STACK_A))
      .toBe(errorFingerprint('react-boundary', 'boom', moved));
  });

  it('separates genuinely different faults', () => {
    const crash = errorFingerprint('react-boundary', 'Cannot read x', STACK_A);
    const http = errorFingerprint('http-error', 'POST /api/indexing/start → 500', '');
    const other = errorFingerprint('react-boundary', 'Something else entirely', STACK_A);
    expect(new Set([crash, http, other]).size).toBe(3);
  });

  it('separates the same message from different places', () => {
    const elsewhere = STACK_A.replace('PhaseMapPage', 'IndexingPage');
    expect(errorFingerprint('react-boundary', 'boom', STACK_A))
      .not.toBe(errorFingerprint('react-boundary', 'boom', elsewhere));
  });

  it('works without a stack at all', () => {
    expect(errorFingerprint('ui-error', 'no stack here', undefined))
      .toMatch(/^[0-9a-f]{8}$/);
  });

  it('never throws on odd input', () => {
    expect(() => errorFingerprint(null, undefined, {})).not.toThrow();
  });
});

describe('normalizeMessage', () => {
  it('collapses numbers, paths and quoted values', () => {
    expect(normalizeMessage('Pixel 42 in "SampleB" at C:\\data\\x.h5 failed'))
      .toBe('pixel # in <value> at <path> failed');
  });
});

describe('topFrame', () => {
  it('takes the first frame and drops host, line and hash', () => {
    expect(topFrame(STACK_A)).toBe('at phasemappage (/assets/index-*.js)');
  });

  it('returns empty for a stackless error', () => {
    expect(topFrame('')).toBe('');
  });
});
