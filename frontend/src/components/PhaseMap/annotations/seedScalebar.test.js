import { describe, expect, it } from 'vitest';
import { seedScalebar } from './seedScalebar';

describe('seedScalebar', () => {
  it('adds a bar of a nice length when the picture has a scale', () => {
    // 21 scan columns at 2x with a 0.5 um step: 42 output px at 0.25 um each;
    // a quarter of the map is 2.625 um -> the nearest 1/2/5 rung is 2 um.
    const out = seedScalebar([], { umPerPx: 0.25, mapWidthPx: 42 });
    expect(out).toHaveLength(1);
    expect(out[0].type).toBe('scalebar');
    expect(out[0].props.lengthUm).toBe(2);
    expect(out[0].id).toMatch(/scalebar|sb/);
  });

  it('leaves a bar the user already placed alone', () => {
    const mine = [{ id: 'sb-1', type: 'scalebar', x: 0.1, y: 0.1, w: 0.3, h: 0.05, props: { lengthUm: 50 } }];
    expect(seedScalebar(mine, { umPerPx: 0.25, mapWidthPx: 42 })).toBe(mine);
  });

  it('adds nothing without a scale', () => {
    const list = [{ id: 't-1', type: 'title', props: { text: 'x' } }];
    expect(seedScalebar(list, { umPerPx: null, mapWidthPx: 42 })).toBe(list);
    expect(seedScalebar(list, { umPerPx: 0, mapWidthPx: 42 })).toBe(list);
    expect(seedScalebar(list, {})).toBe(list);
    expect(seedScalebar(null, { umPerPx: 0.25, mapWidthPx: 0 })).toEqual([]);
  });

  it('keeps the other annotations and appends the bar', () => {
    const list = [{ id: 't-1', type: 'title', props: { text: 'x' } }];
    const out = seedScalebar(list, { umPerPx: 0.5, mapWidthPx: 400 });
    expect(out.map((a) => a.type)).toEqual(['title', 'scalebar']);
    expect(out[1].props.lengthUm).toBe(50);   // 0.5 * 400 * 0.25 = 50
  });
});
