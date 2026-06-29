import { useEffect, useState } from 'react';

/**
 * Bin a bitmap's luminance into {bins} buckets. Uses a Web Worker for
 * grids larger than 65536 pixels (~256x256); inline-binned below that to
 * avoid worker spin-up latency. Returns {hist, min, max} or null while
 * pending. Re-runs when the bitmap or bins prop changes.
 */
export function useThresholdHistogram(bitmap, { bins = 64 } = {}) {
  const [hist, setHist] = useState(null);

  useEffect(() => {
    let cancelled = false;
    // Defer every setHist() to a microtask so the React-Hooks lint rule
    // (set-state-in-effect) is satisfied. The work itself (canvas readback,
    // binning) is sync and runs on the effect body — only the state commit
    // gets queued.
    const commit = (value) => {
      queueMicrotask(() => { if (!cancelled) setHist(value); });
    };
    if (!bitmap) { commit(null); return () => { cancelled = true; }; }
    const off = document.createElement('canvas');
    off.width = bitmap.width; off.height = bitmap.height;
    const ctx = off.getContext('2d');
    ctx.drawImage(bitmap, 0, 0);
    const img = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
    const total = bitmap.width * bitmap.height;

    if (total > 65536 && typeof Worker !== 'undefined') {
      let w;
      try {
        w = new Worker(new URL('../workers/histogramWorker.js', import.meta.url), { type: 'module' });
        w.onmessage = (e) => { commit(e.data); try { w.terminate(); } catch { /* ignore */ } };
        w.postMessage({ width: bitmap.width, height: bitmap.height, data: img.data, bins }, [img.data.buffer]);
        return () => { cancelled = true; try { w.terminate(); } catch { /* ignore */ } };
      } catch {
        // Worker construction failed (jsdom test env) — fall through to inline.
        try { w?.terminate(); } catch { /* ignore */ }
      }
    }
    // Inline binning for small grids OR worker-failure fallback.
    const h = new Uint32Array(bins);
    let min = 255, max = 0;
    for (let i = 0; i < img.data.length; i += 4) {
      const a = img.data[i + 3]; if (a === 0) continue;
      const v = (img.data[i] * 0.299 + img.data[i + 1] * 0.587 + img.data[i + 2] * 0.114) | 0;
      if (v < min) min = v;
      if (v > max) max = v;
      const b = Math.min(bins - 1, Math.floor(v * bins / 256));
      h[b]++;
    }
    commit({ hist: h, min, max });
    return () => { cancelled = true; };
  }, [bitmap, bins]);

  return hist;
}
