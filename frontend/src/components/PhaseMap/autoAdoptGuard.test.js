import { describe, it, expect } from 'vitest';
import { wouldSwitchFile } from './autoAdoptGuard';

// The 2026-08-17 data-loss scenario: the user loads a fresh .h5oina, which
// resets the backend's active result; the Phase-Map visit-sync then saw
// "nothing active", auto-adopted the newest stored result (an import of an
// OLD file), and the activation auto-switched the loaded file away.
// wouldSwitchFile(...) === true is what now blocks that.

describe('wouldSwitchFile', () => {
  it('true when the result belongs to a different file (the data-loss case)', () => {
    expect(wouldSwitchFile(
      'C:\\Users\\x\\OneDrive\\Scan1_Rich_indexed.h5',
      'C:\\Users\\x\\Test_data\\SampleB.h5oina',
    )).toBe(true);
  });

  it('false when the result belongs to the loaded file', () => {
    expect(wouldSwitchFile(
      'C:\\data\\SampleB.h5oina',
      'C:\\data\\SampleB.h5oina',
    )).toBe(false);
  });

  it('tolerates slash direction and case differences (same file ≠ switch)', () => {
    expect(wouldSwitchFile(
      'C:/Data/SampleB.h5oina',
      'c:\\data\\sampleb.h5oina',
    )).toBe(false);
  });

  it('false when no file is loaded — nothing to lose, adoption is safe', () => {
    expect(wouldSwitchFile('C:\\data\\a.h5', null)).toBe(false);
    expect(wouldSwitchFile('C:\\data\\a.h5', undefined)).toBe(false);
    expect(wouldSwitchFile('C:\\data\\a.h5', '')).toBe(false);
  });

  it('false when the result has no source_file (old results) — keep old behaviour', () => {
    expect(wouldSwitchFile(null, 'C:\\data\\a.h5')).toBe(false);
    expect(wouldSwitchFile('', 'C:\\data\\a.h5')).toBe(false);
  });
});
