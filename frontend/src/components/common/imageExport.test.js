import { describe, it, expect } from 'vitest';
import {
  FORMATS, formatById,
  clampCrop, fullCrop, isFullCrop, resizeCrop,
  outputSize, defaultFactor, SCALE_FACTORS, niceScalebar,
  sanitiseFilename, buildFilename,
  drawAnnotations, ellipsise, wrapText, scalebarOptions,
  moveAnnotation, resizeCaptionStyle, resizeScalebarStyle,
  DEFAULT_CAPTION_STYLE, DEFAULT_SCALEBAR_STYLE,
} from './imageExport';

const NAT = { width: 120, height: 90 };   // overview
const PAT = { width: 156, height: 128 };  // pattern

describe('formats', () => {
  it('offers only what canvas.toBlob can encode', () => {
    expect(FORMATS.map((f) => f.id)).toEqual(['png', 'jpeg', 'webp']);
    expect(FORMATS.every((f) => f.mime.startsWith('image/'))).toBe(true);
  });

  it('falls back to PNG for an unknown id', () => {
    expect(formatById('tiff').id).toBe('png');
    expect(formatById(undefined).id).toBe('png');
  });
});

describe('clampCrop', () => {
  it('keeps a valid crop untouched', () => {
    expect(clampCrop({ x: 10, y: 20, width: 30, height: 40 }, NAT))
      .toEqual({ x: 10, y: 20, width: 30, height: 40 });
  });

  it('pulls a crop that hangs over the right/bottom edge back inside', () => {
    expect(clampCrop({ x: 110, y: 85, width: 30, height: 40 }, NAT))
      .toEqual({ x: 90, y: 50, width: 30, height: 40 });
  });

  it('caps the size at the image and never returns a negative origin', () => {
    expect(clampCrop({ x: -50, y: -50, width: 999, height: 999 }, NAT))
      .toEqual({ x: 0, y: 0, width: 120, height: 90 });
  });

  it('never collapses below one pixel', () => {
    const c = clampCrop({ x: 5, y: 5, width: 0, height: -3 }, NAT);
    expect(c.width).toBe(1);
    expect(c.height).toBe(1);
  });

  it('substitutes the full image for non-numeric input', () => {
    expect(clampCrop({ x: NaN, y: undefined, width: 'x', height: null }, NAT))
      .toEqual({ x: 0, y: 0, width: 120, height: 90 });
  });

  it('rounds to whole image pixels', () => {
    expect(clampCrop({ x: 10.4, y: 20.6, width: 30.5, height: 39.4 }, NAT))
      .toEqual({ x: 10, y: 21, width: 31, height: 39 });
  });
});

// The phase map's stacked view auto-zooms to the indexed region, and the export
// dialog opens on that region instead of the whole grid. That is exactly this
// call: a content bbox {x, y, w, h} reshaped to a crop and clamped to the image.
describe('content bbox as the initial crop', () => {
  const asCrop = (b) => ({ x: b.x, y: b.y, width: b.w, height: b.h });

  it('keeps a bbox that already sits inside the image', () => {
    const bbox = { x: 12, y: 30, w: 96, h: 120 };
    expect(clampCrop(asCrop(bbox), { width: 145, height: 174 }))
      .toEqual({ x: 12, y: 30, width: 96, height: 120 });
  });

  it('is a smaller region than the full grid, so the export is not padded', () => {
    const nat = { width: 145, height: 174 };
    const c = clampCrop(asCrop({ x: 12, y: 30, w: 96, h: 120 }), nat);
    expect(c.width * c.height).toBeLessThan(nat.width * nat.height);
    expect(isFullCrop(c, nat)).toBe(false);
  });

  it('pulls a bbox that overhangs back inside instead of failing', () => {
    const c = clampCrop(asCrop({ x: 140, y: 170, w: 40, h: 40 }), { width: 145, height: 174 });
    expect(c.x + c.width).toBeLessThanOrEqual(145);
    expect(c.y + c.height).toBeLessThanOrEqual(174);
  });

  it('falls back to the whole image when there is no bbox', () => {
    const nat = { width: 145, height: 174 };
    expect(fullCrop(nat)).toEqual({ x: 0, y: 0, width: 145, height: 174 });
  });
});

describe('fullCrop / isFullCrop', () => {
  it('round-trips', () => {
    const f = fullCrop(NAT);
    expect(f).toEqual({ x: 0, y: 0, width: 120, height: 90 });
    expect(isFullCrop(f, NAT)).toBe(true);
    expect(isFullCrop({ x: 1, y: 0, width: 119, height: 90 }, NAT)).toBe(false);
  });
});

describe('resizeCrop', () => {
  const start = { x: 20, y: 20, width: 40, height: 30 };

  it('moves without changing the size', () => {
    const r = resizeCrop('move', start, 10, -5, NAT);
    expect(r).toEqual({ x: 30, y: 15, width: 40, height: 30 });
  });

  it('stops a move at the image edge instead of leaving the image', () => {
    const r = resizeCrop('move', start, 999, 999, NAT);
    expect(r).toEqual({ x: 80, y: 60, width: 40, height: 30 });
  });

  it.each([
    ['e', 10, 0, { x: 20, y: 20, width: 50, height: 30 }],
    ['w', 10, 0, { x: 30, y: 20, width: 30, height: 30 }],
    ['s', 0, 10, { x: 20, y: 20, width: 40, height: 40 }],
    ['n', 0, 10, { x: 20, y: 30, width: 40, height: 20 }],
    ['se', 10, 10, { x: 20, y: 20, width: 50, height: 40 }],
    ['nw', 10, 10, { x: 30, y: 30, width: 30, height: 20 }],
    ['ne', 10, 10, { x: 20, y: 30, width: 50, height: 20 }],
    ['sw', 10, 10, { x: 30, y: 20, width: 30, height: 40 }],
  ])('handle %s pins the opposite edge', (handle, dx, dy, expected) => {
    expect(resizeCrop(handle, start, dx, dy, NAT)).toEqual(expected);
  });

  it('collapses to one pixel instead of inverting when dragged past the far edge', () => {
    const r = resizeCrop('e', start, -999, 0, NAT);
    expect(r.width).toBe(1);
    expect(r.x).toBe(20);           // the pinned edge did not move
    const l = resizeCrop('w', start, 999, 0, NAT);
    expect(l.width).toBe(1);
    expect(l.x + l.width).toBe(60); // right edge still pinned
  });

  it('cannot grow a crop beyond the image', () => {
    const r = resizeCrop('se', start, 999, 999, NAT);
    expect(r.x + r.width).toBe(120);
    expect(r.y + r.height).toBe(90);
  });

  it('treats a non-numeric delta as no movement', () => {
    expect(resizeCrop('move', start, NaN, undefined, NAT)).toEqual(start);
  });
});

describe('outputSize', () => {
  const crop = { x: 0, y: 0, width: 120, height: 90 };

  it('multiplies by the factor', () => {
    expect(outputSize(crop, { type: 'factor', factor: 8 }))
      .toMatchObject({ width: 960, height: 720, factor: 8, limited: false });
  });

  it('shrinks below 1:1 for a fractional factor', () => {
    expect(outputSize(crop, { type: 'factor', factor: 0.25 }))
      .toMatchObject({ width: 30, height: 23 });
  });

  it('hits an exact target width and derives the height from the aspect', () => {
    const o = outputSize(crop, { type: 'width', targetWidth: 2000 });
    expect(o.width).toBe(2000);
    expect(o.height).toBe(1500);
    expect(o.factor).toBeCloseTo(2000 / 120, 10);
  });

  it('keeps the crop aspect, not the image aspect', () => {
    const o = outputSize({ x: 0, y: 0, width: 40, height: 10 }, { type: 'width', targetWidth: 400 });
    expect(o.height).toBe(100);
  });

  // Measured in Chrome on 2026-08-10: 201 MP encodes, 300 MP and 373 MP make
  // toBlob() return null. Anything we hand back must stay encodable, or the
  // export dies at the last step with "Encoding to image/png failed".
  const ENCODABLE_MP = 268;

  it.each([
    ['absurd target width', { x: 0, y: 0, width: 120, height: 90 }, { type: 'width', targetWidth: 200000 }],
    ['a tall crop', { x: 0, y: 0, width: 10, height: 400 }, { type: 'width', targetWidth: 19000 }],
    ['the EDS montage at 4x', { x: 0, y: 0, width: 5280, height: 4416 }, { type: 'factor', factor: 4 }],
    ['a big electron image at 16x', { x: 0, y: 0, width: 1024, height: 768 }, { type: 'factor', factor: 16 }],
  ])('keeps %s inside what the browser can encode', (_name, c, mode) => {
    const o = outputSize(c, mode);
    expect(o.width).toBeLessThanOrEqual(16384);
    expect(o.height).toBeLessThanOrEqual(16384);
    expect(o.width * o.height).toBeLessThan(ENCODABLE_MP * 1e6);
  });

  it('says when it had to reduce the request', () => {
    expect(outputSize({ x: 0, y: 0, width: 5280, height: 4416 }, { type: 'factor', factor: 4 }).limited).toBe(true);
    expect(outputSize({ x: 0, y: 0, width: 120, height: 90 }, { type: 'factor', factor: 4 }).limited).toBe(false);
  });

  it('keeps the aspect ratio when it clamps', () => {
    const c = { x: 0, y: 0, width: 5280, height: 4416 };
    const o = outputSize(c, { type: 'factor', factor: 4 });
    expect(o.height / o.width).toBeCloseTo(4416 / 5280, 3);
  });

  it('falls back to 1:1 on garbage input', () => {
    expect(outputSize(crop, { type: 'factor', factor: 0 }).width).toBe(120);
    expect(outputSize(crop, { type: 'width', targetWidth: 'abc' }).width).toBe(120);
    expect(outputSize(crop, undefined).width).toBe(120);
  });
});

describe('defaultFactor', () => {
  it('offers magnification for small maps and shrinking for big sheets', () => {
    expect(defaultFactor({ width: 120 })).toBe(4);      // EBSD overview
    expect(defaultFactor({ width: 156 })).toBe(4);      // detector pattern
    expect(defaultFactor({ width: 1024 })).toBe(2);     // electron image
    expect(defaultFactor({ width: 5280 })).toBeLessThan(1);  // all-maps sheet
  });

  it('only ever returns a listed factor', () => {
    for (const w of [1, 50, 120, 999, 4000, 5280, 20000]) {
      expect(SCALE_FACTORS).toContain(defaultFactor({ width: w }));
    }
  });

  it('survives nonsense input', () => {
    expect(SCALE_FACTORS).toContain(defaultFactor(null));
    expect(SCALE_FACTORS).toContain(defaultFactor({ width: 0 }));
  });
});

describe('niceScalebar', () => {
  it('picks a 1/2/5 length near the target fraction', () => {
    // SampleB: 0.5 µm per scan point, 120 points wide → 60 µm across.
    // A quarter of that is 15 µm → nearest ladder rung is 20.
    const b = niceScalebar(0.5, 120, 0.25);
    expect(b.lengthUnits).toBe(20);
    expect(b.lengthPx).toBe(40);
    expect(b.text).toBe('20');
  });

  it.each([
    [0.5, 120, 0.1, 5],
    [0.05, 120, 0.25, 2],
    [5, 120, 0.25, 200],
    [0.001, 200, 0.25, 0.05],
  ])('stays on the ladder for %f units/px', (upp, w, frac, expected) => {
    expect(niceScalebar(upp, w, frac).lengthUnits).toBeCloseTo(expected, 10);
  });

  it('never returns a bar wider than the crop', () => {
    for (const upp of [0.001, 0.05, 0.5, 5, 500]) {
      for (const w of [3, 17, 120, 1024]) {
        const b = niceScalebar(upp, w, 0.9);
        expect(b).not.toBeNull();
        expect(b.lengthPx).toBeLessThanOrEqual(w + 1e-9);
      }
    }
  });

  it('shows decimals only for sub-unit bars', () => {
    expect(niceScalebar(0.5, 120, 0.25).text).toBe('20');
    expect(niceScalebar(0.001, 200, 0.25).text).toBe('0.05');
  });

  it('returns null when there is no physical scale', () => {
    expect(niceScalebar(0, 120)).toBeNull();
    expect(niceScalebar(undefined, 120)).toBeNull();
    expect(niceScalebar(-1, 120)).toBeNull();
    expect(niceScalebar(0.5, 0)).toBeNull();
  });
});

describe('filenames', () => {
  it('keeps spaces and hyphens', () => {
    expect(sanitiseFilename('Sample_B Arbeitsbereich 1 - EBSD'))
      .toBe('Sample_B Arbeitsbereich 1 - EBSD');
  });

  it('replaces every character a filesystem rejects', () => {
    for (const ch of ['<', '>', ':', '"', '/', '|', '?', '*', String.fromCharCode(92)]) {
      expect(sanitiseFilename(`a${ch}b`)).toBe('a_b');
    }
  });

  it('replaces control characters', () => {
    expect(sanitiseFilename('a bc')).toBe('a_b_c');
  });

  it('trims leading/trailing dots and whitespace', () => {
    expect(sanitiseFilename('  ..name..  ')).toBe('name');
  });

  it('never returns an empty name', () => {
    expect(sanitiseFilename('')).toBe('export');
    expect(sanitiseFilename('...')).toBe('export');
    expect(sanitiseFilename(null)).toBe('export');
  });

  it('swaps the extension to match the format', () => {
    expect(buildFilename('overview.png', 'jpeg')).toBe('overview.jpg');
    expect(buildFilename('overview.jpeg', 'png')).toBe('overview.png');
    expect(buildFilename('overview', 'webp')).toBe('overview.webp');
  });

  it('leaves a non-image extension alone as part of the stem', () => {
    expect(buildFilename('scan.v2', 'png')).toBe('scan.v2.png');
  });
});

// A recording 2D context: enough surface for drawAnnotations, and it lets the
// test assert WHERE things were drawn rather than merely that it did not throw.
function fakeCtx() {
  const calls = [];
  const rec = (name) => (...args) => calls.push([name, ...args]);
  return {
    calls,
    save: rec('save'), restore: rec('restore'),
    beginPath: rec('beginPath'), stroke: rec('stroke'),
    moveTo: rec('moveTo'), lineTo: rec('lineTo'),
    strokeRect: rec('strokeRect'), fillRect: rec('fillRect'), fillText: rec('fillText'),
    measureText: (s) => ({ width: s.length * 6 }),
    set strokeStyle(v) { calls.push(['strokeStyle', v]); },
    set fillStyle(v) { calls.push(['fillStyle', v]); },
    set lineWidth(v) { calls.push(['lineWidth', v]); },
    set font(v) { calls.push(['font', v]); },
    set textAlign(v) { calls.push(['textAlign', v]); },
    set textBaseline(v) { calls.push(['textBaseline', v]); },
  };
}

describe('drawAnnotations', () => {
  const crop = { x: 20, y: 10, width: 60, height: 45 };
  const output = { width: 600, height: 450 };
  const geom = { crop, output, sx: 10, sy: 10 };

  it('draws nothing when nothing is requested', () => {
    const ctx = fakeCtx();
    drawAnnotations(ctx, {}, geom);
    expect(ctx.calls).toEqual([]);
  });

  it('is a no-op on missing arguments instead of throwing', () => {
    expect(() => drawAnnotations(null, {}, geom)).not.toThrow();
    expect(() => drawAnnotations(fakeCtx(), null, geom)).not.toThrow();
    expect(() => drawAnnotations(fakeCtx(), {}, null)).not.toThrow();
  });

  it('places the crosshair at the pixel centre in OUTPUT coordinates', () => {
    const ctx = fakeCtx();
    drawAnnotations(ctx, { crosshair: { row: 30, col: 50 } }, geom);
    // crop starts at (20, 10): col 50 is 30 image px in → 305 output px with the
    // half-pixel centre; row 30 is 20 image px in → 205.
    const moves = ctx.calls.filter((c) => c[0] === 'moveTo');
    expect(moves[0][2]).toBeCloseTo((30 + 0.5 - 10) * 10, 6);   // y of the horizontal arm
    expect(moves[1][1]).toBeCloseTo((50 + 0.5 - 20) * 10, 6);   // x of the vertical arm
  });

  it('offsets the ROI by the crop origin', () => {
    const ctx = fakeCtx();
    drawAnnotations(ctx, { roi: { startRow: 20, startCol: 40, endRow: 30, endCol: 60 } }, geom);
    const rect = ctx.calls.find((c) => c[0] === 'strokeRect');
    expect(rect.slice(1)).toEqual([200, 100, 200, 100]);
  });

  it('scales line width with the output so 16x is not a hairline', () => {
    const small = fakeCtx();
    drawAnnotations(small, { crosshair: { row: 0, col: 0 } }, { crop, output: { width: 60, height: 45 }, sx: 1, sy: 1 });
    const big = fakeCtx();
    drawAnnotations(big, { crosshair: { row: 0, col: 0 } }, { crop, output: { width: 960, height: 720 }, sx: 16, sy: 16 });
    const lw = (c) => c.calls.find((x) => x[0] === 'lineWidth')[1];
    expect(lw(big)).toBeGreaterThan(lw(small));
  });

  it('anchors the scalebar bottom-right and writes the unit', () => {
    const ctx = fakeCtx();
    drawAnnotations(ctx, { scalebar: { lengthPx: 40, text: '20', unitLabel: 'µm' } }, geom);
    const text = ctx.calls.find((c) => c[0] === 'fillText');
    expect(text[1]).toBe('20 µm');
    const bar = ctx.calls.filter((c) => c[0] === 'fillRect').at(-1);
    expect(bar[1] + bar[3]).toBeLessThanOrEqual(output.width);  // inside the right edge
    expect(bar[2]).toBeLessThan(output.height);
    expect(bar[3]).toBeCloseTo(40 * geom.sx, 6);                 // 40 image px at 10x
  });

  it('draws the caption top-left', () => {
    const ctx = fakeCtx();
    drawAnnotations(ctx, { label: 'SampleB · Band Contrast' }, geom);
    const text = ctx.calls.find((c) => c[0] === 'fillText');
    expect(text[1]).toBe('SampleB · Band Contrast');
    expect(text[2]).toBeLessThan(output.width / 2);
    expect(text[3]).toBeLessThan(output.height / 2);
  });

  // Real dataset names here run past 80 characters. They used to be cut off
  // with an ellipsis; now they wrap onto as many lines as they need.
  it('wraps a long caption over several lines instead of truncating it', () => {
    const ctx = fakeCtx();
    const long = 'EBSD_SampleB_extrusion_withPattern Sample_B Arbeitsbereich 1 Elementverteilungsdaten 1 · Band Contrast · 90×120';
    drawAnnotations(ctx, { label: long }, { crop, output: { width: 400, height: 240 }, sx: 6.6, sy: 8 });
    const drawn = ctx.calls.filter((c) => c[0] === 'fillText').map((c) => c[1]);
    expect(drawn.length).toBeGreaterThan(1);
    expect(drawn.some((l) => l.includes('…'))).toBe(false);
    // Every word survives somewhere, and each line is drawn lower than the last.
    expect(drawn.join(' ').replace(/\s+/g, ' ')).toBe(long.replace(/\s+/g, ' '));
    const ys = ctx.calls.filter((c) => c[0] === 'fillText').map((c) => c[3]);
    expect(ys.every((y, i) => i === 0 || y > ys[i - 1])).toBe(true);
  });

  it('places the caption where its style says, not in a fixed corner', () => {
    const geomB = { crop, output: { width: 400, height: 300 }, sx: 6.6, sy: 6.6 };
    const at = (style) => {
      const ctx = fakeCtx();
      drawAnnotations(ctx, { label: 'abc', labelStyle: style }, geomB);
      return ctx.calls.find((c) => c[0] === 'fillText').slice(2, 4);
    };
    const [x1, y1] = at({ x: 0.02, y: 0.02 });
    const [x2, y2] = at({ x: 0.5, y: 0.6 });
    expect(x2).toBeGreaterThan(x1);
    expect(y2).toBeGreaterThan(y1);
  });

  it('omits the backdrop entirely at zero opacity', () => {
    const ctx = fakeCtx();
    drawAnnotations(ctx, { label: 'abc', labelStyle: { backgroundOpacity: 0 } }, geom);
    expect(ctx.calls.some((c) => c[0] === 'fillRect')).toBe(false);
    expect(ctx.calls.some((c) => c[0] === 'fillText')).toBe(true);
  });
});

describe('wrapText', () => {
  const ctx = { measureText: (s) => ({ width: s.length * 6 }) };

  it('breaks on word boundaries, filling each line as far as it goes', () => {
    // 6 px per char: 'one two' = 42, 'three four' = 60 — both fit exactly.
    expect(wrapText(ctx, 'one two three four', 60)).toEqual(['one two', 'three four']);
    expect(wrapText(ctx, 'one two three four', 40)).toEqual(['one', 'two', 'three', 'four']);
  });

  it('splits a single word that cannot fit', () => {
    const lines = wrapText(ctx, 'aaaaaaaaaaaaaaa', 30);
    expect(lines.length).toBeGreaterThan(1);
    expect(lines.join('')).toBe('aaaaaaaaaaaaaaa');
    expect(lines.every((l) => l.length * 6 <= 30)).toBe(true);
  });

  it('keeps everything on one line when it fits', () => {
    expect(wrapText(ctx, 'one two', 600)).toEqual(['one two']);
  });

  it('returns nothing for empty input or a useless width', () => {
    expect(wrapText(ctx, '', 100)).toEqual([]);
    expect(wrapText(ctx, null, 100)).toEqual([]);
    expect(wrapText(ctx, 'abc', 0)).toEqual([]);
  });
});

describe('scalebarOptions', () => {
  it('offers only 1/2/5 lengths that fit the crop', () => {
    const opts = scalebarOptions(0.5, 120);   // 60 µm across
    expect(opts.length).toBeGreaterThan(3);
    expect(opts.every((v) => v <= 60)).toBe(true);
    expect(opts).toContain(10);
    expect(opts).toContain(20);
    expect(opts).not.toContain(100);
    expect([...opts].sort((a, b) => a - b)).toEqual(opts);
  });

  it('returns nothing without a physical scale', () => {
    expect(scalebarOptions(0, 120)).toEqual([]);
    expect(scalebarOptions(0.5, 0)).toEqual([]);
  });
});

describe('annotation dragging', () => {
  it('moves by the given fraction and stays inside the image', () => {
    const s = { x: 0.2, y: 0.3 };
    const moved = moveAnnotation(s, 0.1, -0.1);
    expect(moved.x).toBeCloseTo(0.3, 6);
    expect(moved.y).toBeCloseTo(0.2, 6);
    expect(moveAnnotation(s, 9, 9)).toMatchObject({ x: 1, y: 1 });
    expect(moveAnnotation(s, -9, -9)).toMatchObject({ x: 0, y: 0 });
  });

  it('caption: side grips change the wrap width, top/bottom the font size', () => {
    const s = { ...DEFAULT_CAPTION_STYLE };
    expect(resizeCaptionStyle(s, 'e', 0.2, 0).widthFrac).toBeCloseTo(0.75, 6);
    expect(resizeCaptionStyle(s, 'e', 0.2, 0).fontScale).toBe(s.fontScale);
    expect(resizeCaptionStyle(s, 's', 0, 0.25).fontScale).toBeCloseTo(2, 6);
    expect(resizeCaptionStyle(s, 's', 0, 0.25).widthFrac).toBe(s.widthFrac);
  });

  it('caption: a west grip moves the origin so the right edge stays put', () => {
    const s = { ...DEFAULT_CAPTION_STYLE, x: 0.3, widthFrac: 0.5 };
    const out = resizeCaptionStyle(s, 'w', 0.1, 0);
    expect(out.x).toBeCloseTo(0.4, 6);
    expect(out.widthFrac).toBeCloseTo(0.4, 6);
    expect(out.x + out.widthFrac).toBeCloseTo(s.x + s.widthFrac, 6);
  });

  it('scale bar: grips scale lettering and thickness together', () => {
    const s = { ...DEFAULT_SCALEBAR_STYLE };
    const out = resizeScalebarStyle(s, 'se', 0.125, 0.125);
    expect(out.fontScale).toBeCloseTo(2, 6);
    expect(out.barScale).toBeCloseTo(2, 6);
    const smaller = resizeScalebarStyle(s, 'se', -0.1, -0.1);
    expect(smaller.fontScale).toBeLessThan(1);
  });

  it('never scales to zero or beyond a sane maximum', () => {
    const s = { ...DEFAULT_SCALEBAR_STYLE };
    expect(resizeScalebarStyle(s, 'se', -99, -99).fontScale).toBeGreaterThan(0);
    expect(resizeScalebarStyle(s, 'se', 99, 99).fontScale).toBeLessThanOrEqual(8);
    expect(resizeCaptionStyle({ ...DEFAULT_CAPTION_STYLE }, 'e', -99, 0).widthFrac).toBeGreaterThan(0);
  });
});

describe('ellipsise', () => {
  // 6 px per character, matching fakeCtx.
  const ctx = { measureText: (s) => ({ width: s.length * 6 }) };

  it('leaves text that already fits alone', () => {
    expect(ellipsise(ctx, 'short', 600)).toBe('short');
    expect(ellipsise(ctx, 'abcde', 30)).toBe('abcde');   // exact fit
  });

  it('returns the longest prefix that fits with an ellipsis', () => {
    const out = ellipsise(ctx, 'a'.repeat(100), 120);
    expect(out.endsWith('…')).toBe(true);
    expect(out.length * 6).toBeLessThanOrEqual(120);
    expect((out.length + 1) * 6).toBeGreaterThan(120);   // one more would overflow
  });

  it('gives up rather than overflowing when there is no room', () => {
    expect(ellipsise(ctx, 'anything', 3)).toBe('');
    expect(ellipsise(ctx, 'anything', 0)).toBe('');
    expect(ellipsise(ctx, 'anything', -5)).toBe('');
  });

  it('handles empty and nullish input', () => {
    expect(ellipsise(ctx, '', 100)).toBe('');
    expect(ellipsise(ctx, null, 100)).toBe('');
  });
});
