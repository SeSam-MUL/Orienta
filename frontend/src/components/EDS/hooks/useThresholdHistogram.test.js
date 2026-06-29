// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { useThresholdHistogram } from './useThresholdHistogram';

// jsdom does not implement HTMLCanvasElement.getContext('2d') — the hook
// allocates an offscreen canvas, draws the bitmap, then reads ImageData out
// of it. We stub getContext to return a minimal 2d context that hands back
// a deterministic ImageData buffer, so the inline-binning branch can run.
let originalGetContext;
function installCanvasStub(makeImageData) {
  originalGetContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function (type) {
    if (type !== '2d') return null;
    return {
      drawImage() {},
      getImageData: (sx, sy, sw, sh) => ({
        data: makeImageData(sw, sh),
        width: sw,
        height: sh,
      }),
    };
  };
}
function uninstallCanvasStub() {
  if (originalGetContext) HTMLCanvasElement.prototype.getContext = originalGetContext;
}

beforeEach(() => { vi.restoreAllMocks(); });
afterEach(() => { uninstallCanvasStub(); });

describe('useThresholdHistogram', () => {
  it('returns null when bitmap is undefined', () => {
    const { result } = renderHook(() => useThresholdHistogram(undefined));
    expect(result.current).toBeNull();
  });

  it('bins a synthetic 4×4 half-black/half-white image into the expected bins', async () => {
    // Build the ImageData buffer ourselves: left two columns white, right two
    // columns black, alpha=255 everywhere.
    installCanvasStub((w, h) => {
      const data = new Uint8ClampedArray(w * h * 4);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const idx = (y * w + x) * 4;
          const v = x < w / 2 ? 255 : 0;
          data[idx] = v; data[idx + 1] = v; data[idx + 2] = v; data[idx + 3] = 255;
        }
      }
      return data;
    });
    const bitmap = { width: 4, height: 4 };
    const { result } = renderHook(() => useThresholdHistogram(bitmap, { bins: 4 }));
    await waitFor(() => expect(result.current).toBeTruthy());
    const total = Array.from(result.current.hist).reduce((a, b) => a + b, 0);
    expect(total).toBe(16);
    // Black pixels in bin 0, white pixels in bin 3 (last bin).
    expect(result.current.hist[0]).toBeGreaterThan(0);
    expect(result.current.hist[3]).toBeGreaterThan(0);
  });
});
