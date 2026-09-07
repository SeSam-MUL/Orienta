import { describe, it, expect, vi } from 'vitest';
import { keyImageToCanvas, ipfKeyExportItems } from './ipfKeyExport';

/** A canvas that records the calls instead of painting: jsdom has no context. */
function recordingCanvas() {
  const calls = [];
  const ctx = {
    set fillStyle(v) { calls.push(['fillStyle', v]); },
    fillRect: (...a) => calls.push(['fillRect', ...a]),
    drawImage: (...a) => calls.push(['drawImage', a[0]?.tag ?? a[0], ...a.slice(1)]),
  };
  return { canvas: { width: 0, height: 0, getContext: () => ctx }, calls };
}

const KEY = { width: 288, height: 792, tag: 'key' };   // 3 phases, 2.4in x 6.6in

describe('keyImageToCanvas', () => {
  it('keeps the key at its native size', () => {
    const rec = recordingCanvas();
    const out = keyImageToCanvas(KEY, { makeCanvas: () => rec.canvas });
    expect([out.width, out.height]).toEqual([288, 792]);
  });

  it('lays the white plate down BEFORE the key, not over it', () => {
    const rec = recordingCanvas();
    keyImageToCanvas(KEY, { opaque: true, makeCanvas: () => rec.canvas });
    const order = rec.calls.map((c) => c[0]);
    expect(order).toEqual(['fillStyle', 'fillRect', 'drawImage']);
    expect(rec.calls[0][1]).toBe('#ffffff');
    expect(rec.calls[1].slice(1)).toEqual([0, 0, 288, 792]);
  });

  it('paints no plate when transparency was asked for', () => {
    const rec = recordingCanvas();
    keyImageToCanvas(KEY, { opaque: false, makeCanvas: () => rec.canvas });
    expect(rec.calls.map((c) => c[0])).toEqual(['drawImage']);
  });

  it('refuses an image that has not loaded', () => {
    expect(() => keyImageToCanvas(null)).toThrow(/no key image/);
    expect(() => keyImageToCanvas({ width: 0, height: 0 })).toThrow(/no key image/);
  });
});

describe('ipfKeyExportItems', () => {
  const t = (k) => k;

  it('offers both grounds, and they do not write the same file', () => {
    const open = vi.fn();
    const items = ipfKeyExportItems({ t, stem: 'Scan1', open });
    expect(items.map((i) => i.label)).toEqual([
      'imageexport:menuExportIpfKey',
      'imageexport:menuExportIpfKeyAlpha',
    ]);

    items[0].onSelect();
    items[1].onSelect();
    expect(open.mock.calls).toEqual([
      [true, 'Scan1_ipf-key'],
      [false, 'Scan1_ipf-key_transparent'],
    ]);
  });

  it('keeps a title with slashes out of the file name', () => {
    const open = vi.fn();
    ipfKeyExportItems({ t, stem: 'a/b\\c', open })[0].onSelect();
    expect(open).toHaveBeenCalledWith(true, 'a_b_c_ipf-key');
  });
});
