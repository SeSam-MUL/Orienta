// Pure worker: bin luminance of an ImageData into N bins.
// Skips fully-transparent pixels. Posts {hist: Uint32Array, min, max}.
self.onmessage = (e) => {
  const { data, bins = 64 } = e.data;
  const hist = new Uint32Array(bins);
  let min = 255, max = 0;
  for (let i = 0; i < data.length; i += 4) {
    const a = data[i + 3];
    if (a === 0) continue;
    const v = (data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114) | 0;
    if (v < min) min = v;
    if (v > max) max = v;
    const b = Math.min(bins - 1, Math.floor(v * bins / 256));
    hist[b]++;
  }
  self.postMessage({ hist, min, max });
};
