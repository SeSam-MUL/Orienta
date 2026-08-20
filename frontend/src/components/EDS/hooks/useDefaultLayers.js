import { useEffect, useState } from 'react';
import { edsApi, ebsdApi, h5Api } from '../../../services/api';
import { defaultLayersFor, pickPreferredElectronImage } from '../edsLayerSources';
import { DATASET_UNKNOWN } from './useActiveDatasetKey';

/**
 * Probe hasBC / electron-image list / EDS-element list and seed the initial
 * EDS-page layer stack via {@link defaultLayersFor}. Three probes run in
 * parallel via Promise.allSettled — a single failure does not block the rest.
 *
 * Returns { ready, layers, elements, electronImages } where:
 *   - ready=false while probing or when no file is open
 *   - layers is the seeded stack (may be empty)
 *   - elements is the full normalised element list (more than the 3 seeded)
 *   - electronImages is the full list of electron-image names (used by the
 *     "+ Add Layer" dropdown so every detector is mountable, not just the
 *     SE-preferred default).
 *
 * `filePath` is in the effect deps so a file *switch* (isFileOpen stays true,
 * only the path changes) re-runs the probe and re-seeds the layer stack for
 * the new file's elements / electron images.
 *
 * `datasetKey` is the NAME of the active EBSD dataset ('Scan1',
 * 'Scan1_crop1', ...). It is in the deps for the same reason the path is, and
 * it is not redundant with it: cropping changes WHICH DATASET the maps come
 * from without changing the file. Both probes below read the active dataset —
 * `edsApi.elements()` and `getElectronList('dataset')` — so a crop moves them
 * both while `isFileOpen` and `filePath` sit perfectly still. Without this the
 * page kept serving the PARENT's full-scan maps while every probe coordinate
 * (hover, linescan, region) was resolved on the crop's grid.
 *
 * {@link DATASET_UNKNOWN} means "the caller does not know the dataset yet" and
 * suppresses the probe, so a file-open does not probe once on the unknown key
 * and again on the resolved one — the second probe would flush and refetch
 * every bitmap the first had just decoded. `null` (no named dataset) is a
 * legitimate key and does probe.
 */
export function useDefaultLayers(isFileOpen, filePath, datasetKey = null) {
  const [state, setState] = useState({ ready: false, layers: [], elements: [], electronImages: [] });

  useEffect(() => {
    if (!isFileOpen || datasetKey === DATASET_UNKNOWN) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- file-close reset must clear state synchronously; the alternative (derived useMemo) breaks the async probe flow.
      setState({ ready: false, layers: [], elements: [], electronImages: [] });
      return;
    }
    let cancelled = false;
    (async () => {
      const [elemsRes, electronRes, bcRes] = await Promise.allSettled([
        edsApi.elements(),
        // 'dataset': this list feeds the EDS layer stack, which shows the
        // ACTIVE dataset — cropped or not.
        h5Api.getElectronList('dataset'),
        ebsdApi.bandContrast('gray'),
      ]);
      if (cancelled) return;

      if (elemsRes.status === 'rejected')    console.warn('[useDefaultLayers] elements() failed:', elemsRes.reason);
      if (electronRes.status === 'rejected') console.warn('[useDefaultLayers] getElectronList() failed:', electronRes.reason);
      if (bcRes.status === 'rejected')       console.warn('[useDefaultLayers] bandContrast() failed:', bcRes.reason);

      const rawElements = elemsRes.status === 'fulfilled' ? (elemsRes.value.data?.elements || []) : [];
      const elements = rawElements
        .map(e => typeof e === 'string' ? e : (e?.name || e?.element || ''))
        .filter(Boolean);

      const electronImages = electronRes.status === 'fulfilled' ? (electronRes.value.data?.images || []) : [];

      // label === 'fallback' indicates no native BC was stored — the backend
      // synthesised a placeholder. Treat as "no BC" for default-layer purposes.
      const hasBC = bcRes.status === 'fulfilled' && bcRes.value.data?.label !== 'fallback';

      const electronImageName = pickPreferredElectronImage(electronImages);
      const layers = defaultLayersFor({
        hasSE: !!electronImageName,
        hasBC,
        electronImageName,
        elements,
      });
      setState({ ready: true, layers, elements, electronImages });
    })();
    return () => { cancelled = true; };
  }, [isFileOpen, filePath, datasetKey]);

  return state;
}
