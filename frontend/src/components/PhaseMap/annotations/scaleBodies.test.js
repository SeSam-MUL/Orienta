// @vitest-environment jsdom
import { describe, it, expect } from 'vitest';
import {
  findScaleBody, withScaleBody, withRemoved, respaceLeftColumn, scaleMargins,
} from './exportAnnotEdits';

const KEY = { type: 'colorkey' };
const BC = { type: 'valuescale', layerId: 'bc' };
const CI = { type: 'valuescale', layerId: 'ci' };

describe('scale bodies', () => {
  it('adds a body for an entry and finds it again', () => {
    const { annotations, id } = withScaleBody([], KEY, true);
    expect(annotations).toHaveLength(1);
    expect(annotations[0].type).toBe('colorkey');
    expect(findScaleBody(annotations, KEY)?.id).toBe(id);
  });

  it('keeps value scales apart by the layer they belong to', () => {
    let list = withScaleBody([], BC, true).annotations;
    list = withScaleBody(list, CI, true).annotations;
    expect(list).toHaveLength(2);
    expect(findScaleBody(list, BC).props.layerId).toBe('bc');
    expect(findScaleBody(list, CI).props.layerId).toBe('ci');
    // Off for one, on for the other.
    list = withScaleBody(list, BC, false).annotations;
    expect(findScaleBody(list, BC)).toBeNull();
    expect(findScaleBody(list, CI)).toBeTruthy();
  });

  it('is idempotent in both directions', () => {
    // The checkbox passes its state, not a command, so asking twice must not
    // produce two bars stacked on the same spot.
    const once = withScaleBody([], BC, true).annotations;
    const twice = withScaleBody(once, BC, true).annotations;
    expect(twice).toBe(once);
    const off = withScaleBody(once, BC, false).annotations;
    expect(withScaleBody(off, BC, false).annotations).toBe(off);
  });

  it('stands beside the map, not on it', () => {
    // The reported wish: value scale left of the data, colour key right of it,
    // instead of both lying on top of the map.
    const list = withScaleBody(withScaleBody([], KEY, true).annotations, BC, true).annotations;
    const key = findScaleBody(list, KEY);
    const bar = findScaleBody(list, BC);
    expect(bar.x + bar.w).toBeLessThanOrEqual(0);   // entirely left of the map
    expect(key.x).toBeGreaterThanOrEqual(1);        // entirely right of it
  });

  it('lets two value scales share the left column instead of overlapping', () => {
    let list = withScaleBody([], BC, true).annotations;
    const alone = findScaleBody(list, BC).h;
    list = withScaleBody(list, CI, true).annotations;
    const [a, b] = [findScaleBody(list, BC), findScaleBody(list, CI)];
    expect(a.h).toBeLessThan(alone);                // made room for the second
    expect(a.h).toBeCloseTo(b.h, 6);                // equal shares
    expect(a.y + a.h).toBeLessThanOrEqual(b.y);     // stacked, not overlapping
    // Removing one gives the other the whole column back.
    list = withScaleBody(list, CI, false).annotations;
    expect(findScaleBody(list, BC).h).toBeCloseTo(alone, 6);
  });

  it('leaves a bar the user dragged onto the map where they put it', () => {
    let list = withScaleBody([], BC, true).annotations;
    const moved = { ...findScaleBody(list, BC), x: 0.5, y: 0.5, h: 0.2 };
    list = withScaleBody([moved], CI, true).annotations;
    const still = list.find((a) => a.props.layerId === 'bc');
    expect([still.x, still.y, still.h]).toEqual([0.5, 0.5, 0.2]);
  });

  it('asks for exactly the border the bodies need', () => {
    const list = withScaleBody(withScaleBody([], KEY, true).annotations, BC, true).annotations;
    const m = scaleMargins(list);
    const key = findScaleBody(list, KEY);
    const bar = findScaleBody(list, BC);
    expect(m.left).toBeCloseTo(-bar.x + 0.02, 6);
    expect(m.right).toBeCloseTo(key.x + key.w - 1 + 0.02, 6);
    expect(m.top).toBe(0);
    expect(m.bottom).toBe(0);
    // No scales, no border.
    expect(scaleMargins([])).toEqual({ top: 0, right: 0, bottom: 0, left: 0 });
  });

  it('asks for the whole border the body needs, however far out it sits', () => {
    // It used to stop at half the map per side. A three-phase IPF key beside a
    // wide, short map needs more than that and was sliced (2026-09-07); what
    // the file can hold is the exporter's business, not this function's.
    const far = [{ type: 'colorkey', x: 4, y: 0, w: 1, h: 1, props: {} }];
    expect(scaleMargins(far).right).toBeCloseTo(4 + 1 - 1 + 0.02, 6);
  });

  it('respacing an empty or column-less list changes nothing', () => {
    const onMap = [{ id: 'v', type: 'valuescale', x: 0.5, y: 0.1, w: 0.2, h: 0.3, props: {} }];
    expect(respaceLeftColumn(onMap)).toBe(onMap);
    expect(respaceLeftColumn([])).toEqual([]);
  });

  it('leaves the user-made annotations alone', () => {
    const title = { id: 't', type: 'title', x: 0, y: 0, w: 0.4, h: 0.06, props: { text: 'hi' } };
    const list = withScaleBody([title], KEY, true).annotations;
    expect(list[0]).toBe(title);
    expect(withScaleBody(list, KEY, false).annotations).toEqual([title]);
  });

  it('a body deleted by its own × is simply gone — the checkbox reads that', () => {
    // The box is not a second source of truth; this is what keeps them in step.
    const { annotations, id } = withScaleBody([], BC, true);
    const after = withRemoved(annotations, id);
    expect(findScaleBody(after, BC)).toBeNull();
  });
});
