import { describe, it, expect } from 'vitest';
import { adoptDecision, R_ADOPT_MARGIN } from './adoptDecision';

const QUAT = [1, 0, 0, 0];

describe('adoptDecision', () => {
  it('refuses the real ProbeB case that caused the bug', () => {
    // pixel (323,144): stored renders at 0.262, the re-indexed candidate at
    // 0.140, 44.13 deg apart. The old code offered this swap.
    const d = adoptDecision({
      samePhase: true, quat: QUAT, disorientationDeg: 44.13,
      rCandidate: 0.1397, rStored: 0.2625,
    });
    expect(d.canAdopt).toBe(false);
    expect(d.rejected).toBe(true);
    expect(d.reason).toBe('not-better');
  });

  it('offers the swap when the candidate clearly renders better', () => {
    const d = adoptDecision({
      samePhase: true, quat: QUAT, disorientationDeg: 44.13,
      rCandidate: 0.40, rStored: 0.17,
    });
    expect(d.canAdopt).toBe(true);
    expect(d.rejected).toBe(false);
  });

  it('refuses an improvement smaller than the clear margin', () => {
    const d = adoptDecision({
      samePhase: true, quat: QUAT, disorientationDeg: 10,
      rCandidate: 0.30 + R_ADOPT_MARGIN / 2, rStored: 0.30,
    });
    expect(d.canAdopt).toBe(false);
    expect(d.reason).toBe('not-better');
  });

  it('accepts exactly at the margin', () => {
    const d = adoptDecision({
      samePhase: true, quat: QUAT, disorientationDeg: 10,
      rCandidate: 0.33, rStored: 0.30,
    });
    expect(d.canAdopt).toBe(true);
  });

  it('is not a candidate at all for a different phase', () => {
    const d = adoptDecision({
      samePhase: false, quat: QUAT, disorientationDeg: 44,
      rCandidate: 0.9, rStored: 0.1,
    });
    expect(d.canAdopt).toBe(false);
    expect(d.rejected).toBe(false);
    expect(d.reason).toBe('no-candidate');
  });

  it('is not a candidate when the orientations are effectively identical', () => {
    const d = adoptDecision({
      samePhase: true, quat: QUAT, disorientationDeg: 0.01,
      rCandidate: 0.9, rStored: 0.1,
    });
    expect(d.reason).toBe('no-candidate');
  });

  it('refuses when a render score is missing rather than guessing', () => {
    for (const [rc, rs] of [[null, 0.2], [0.4, null], [NaN, 0.2], [0.4, NaN]]) {
      const d = adoptDecision({
        samePhase: true, quat: QUAT, disorientationDeg: 30,
        rCandidate: rc, rStored: rs,
      });
      expect(d.canAdopt).toBe(false);
      expect(d.reason).toBe('no-r-score');
    }
  });

  it('needs a quaternion to adopt', () => {
    const d = adoptDecision({
      samePhase: true, quat: null, disorientationDeg: 30,
      rCandidate: 0.9, rStored: 0.1,
    });
    expect(d.reason).toBe('no-candidate');
  });

  it('handles being called with nothing', () => {
    expect(adoptDecision().canAdopt).toBe(false);
  });
});
