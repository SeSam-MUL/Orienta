/**
 * One image to lay under the phase map, so a region can be placed against the
 * chemistry or the topography it came from.
 *
 * Deliberately ONE image rather than a layer stack. The page already has two
 * layer stacks; a third would have to be told about display modes, dataset
 * keys and cache invalidation, and its LRU keys on layer id — which would serve
 * a stale bitmap after every merge, split or paint. What this needs is smaller
 * than that: pick a background, set an opacity.
 *
 * Alignment is by field of view, not by resampling. Measured on a real file:
 * the scan raster is 0.5 um x 120 px = 60.00 um and the electron image is
 * 0.0590 um x 1024 px = 60.39 um — the same field, so drawing both with
 * `objectFit: contain` in one box registers them. `fovRatio` below reports that
 * agreement so the UI can warn instead of quietly showing a shifted overlay.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { edsApi, h5Api } from '../../services/api';

export const BG_NONE = 'none';

/** `element:Al Ka1` / `electron:SE/Elektronenbild 1` -> { kind, name }. */
export function parseBackgroundId(id) {
  if (!id || id === BG_NONE) return null;
  const cut = id.indexOf(':');
  if (cut < 0) return null;
  const kind = id.slice(0, cut);
  const name = id.slice(cut + 1);
  if (!name) return null;
  if (kind !== 'element' && kind !== 'electron') return null;
  return { kind, name };
}

/**
 * How well two rasters cover the same field of view, or null when unknown.
 *
 * 1.0 means the same field; drawing both to fit is then correct. Anything far
 * from 1 means the overlay would be shifted or scaled, and the user must be
 * told rather than shown a confident lie.
 */
export function fovRatio(pixelSizes, mapCols, imageWidth) {
  const scan = pixelSizes?.ebsd || pixelSizes?.eds;
  const img = pixelSizes?.electron_image;
  if (!scan?.x || !img?.x || !mapCols || !imageWidth) return null;
  const fovScan = scan.x * mapCols;
  const fovImg = img.x * imageWidth;
  if (!(fovScan > 0)) return null;
  return fovImg / fovScan;
}

export default function useMapBackground({ displayMode = 'at_pct', datasetKey } = {}) {
  const [backgroundId, setBackgroundId] = useState(BG_NONE);
  const [opacity, setOpacity] = useState(0.5);
  const [image, setImage] = useState(null);       // base64 png
  const [shape, setShape] = useState(null);       // [rows, cols] of the background
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const token = useRef(0);

  const parsed = useMemo(() => parseBackgroundId(backgroundId), [backgroundId]);

  useEffect(() => {
    if (!parsed) { setImage(null); setShape(null); setError(null); return undefined; }
    const mine = ++token.current;
    setLoading(true); setError(null);
    const req = parsed.kind === 'element'
      // Grey, not the element's colour: this sits UNDER a categorical map, and
      // two colour schemes stacked read as a third one that means nothing.
      ? edsApi.getMap(parsed.name, displayMode, 'gray')
      : h5Api.getElectronImage(parsed.name, 'dataset');
    req.then((res) => {
      if (mine !== token.current) return;
      setImage(res.data?.image || null);
      setShape(res.data?.shape || null);
    }).catch((e) => {
      if (mine !== token.current) return;
      setError(e.response?.data?.detail || String(e));
      setImage(null); setShape(null);
    }).finally(() => {
      if (mine === token.current) setLoading(false);
    });
    return () => { token.current += 1; };
  }, [parsed?.kind, parsed?.name, displayMode, datasetKey]);

  const clear = useCallback(() => setBackgroundId(BG_NONE), []);

  return {
    backgroundId, setBackgroundId, opacity, setOpacity,
    image, shape, error, loading, clear,
    isSet: !!parsed,
  };
}
