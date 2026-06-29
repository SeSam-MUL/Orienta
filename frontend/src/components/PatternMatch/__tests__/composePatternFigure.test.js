// @vitest-environment node
import { describe, it, expect } from 'vitest';
import { targetDims, composePatternFigure } from '../composePatternFigure';

describe('targetDims', () => {
  const model = { canvas: { wPx: 1200, hPx: 600 }, elements: [] };
  it('uses explicit target width preserving aspect', () => {
    expect(targetDims(model, { targetWidthPx: 2400 })).toEqual({ W: 2400, H: 1200 });
  });
  it('falls back to scale × canvas size', () => {
    expect(targetDims(model, { scale: 4 })).toEqual({ W: 4800, H: 2400 });
  });
  it('defaults to scale 2 when nothing given', () => {
    expect(targetDims(model, {})).toEqual({ W: 2400, H: 1200 });
  });
});

describe('composePatternFigure guard', () => {
  it('throws when the only panel has no decoded image (would be blank)', async () => {
    const model = { canvas: { wPx: 100, hPx: 50 }, elements: [{ type: 'panel', source: 'experimental' }] };
    await expect(composePatternFigure(model, {/* no experimental */})).rejects.toThrow(/Nothing to export/);
  });
});
