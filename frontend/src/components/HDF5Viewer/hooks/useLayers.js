/**
 * useLayers — fetches layer catalog on file open and resolves the active
 * layer to a base64 PNG via the right backend endpoint per layer.kind.
 */
import { useEffect, useState, useCallback, useRef } from 'react';
import i18n from '../../../i18n';
import { h5Api } from '../../../services/api';
import useCockpitStore from '../../../stores/useCockpitStore';

export function useLayers(isFileOpen) {
  const [catalog, setCatalog] = useState(null);
  const [activeImage, setActiveImage] = useState(null);
  const [activeMeta, setActiveMeta] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fetchSeq = useRef(0);

  const setLayerCatalog = useCockpitStore((s) => s.setLayerCatalog);
  const setActiveLayer = useCockpitStore((s) => s.setActiveLayer);
  const activeLayerId = useCockpitStore((s) => s.activeLayerId);

  // Fetch catalog on file open
  useEffect(() => {
    if (!isFileOpen) {
      setCatalog(null);
      setActiveImage(null);
      setLayerCatalog(null);
      setActiveLayer(null);
      return;
    }
    let cancelled = false;
    h5Api.getLayers()
      .then((res) => {
        if (cancelled) return;
        setCatalog(res.data);
        setLayerCatalog(res.data);
        // Auto-select Band Contrast as default
        const defaultLayer = res.data.quality?.find((l) => l.id === 'ebsd:Band Contrast');
        if (defaultLayer) setActiveLayer(defaultLayer.id);
      })
      .catch((err) => {
        if (cancelled) return;
        console.warn('[useLayers] getLayers failed', err);
        setError(i18n.t('hdf5viewer:errors.layerCatalogFailed'));
      });
    return () => { cancelled = true; };
  }, [isFileOpen, setLayerCatalog, setActiveLayer]);

  // Fetch active layer image when ID or catalog changes
  const fetchActiveLayer = useCallback(async () => {
    if (!catalog || !activeLayerId) {
      setActiveImage(null);
      return;
    }
    const allLayers = [
      ...(catalog.quality ?? []),
      ...(catalog.indexing ?? []),
      ...(catalog.geometry ?? []),
      ...(catalog.eds ?? []),
      ...(catalog.electron ?? []),
    ];
    const layer = allLayers.find((l) => l.id === activeLayerId);
    if (!layer) return;

    const seq = ++fetchSeq.current;
    setLoading(true);
    setError(null);
    try {
      let res;
      if (layer.kind === 'scalar') {
        res = await h5Api.getScalarMap(layer.path);
      } else if (layer.kind === 'phase') {
        res = await h5Api.getPhaseMap();
      } else if (layer.kind === 'ipf') {
        res = await h5Api.getIPFMap(layer.direction);
      } else if (layer.kind === 'eds_element') {
        res = await h5Api.getEDSMap(layer.name);
      } else if (layer.kind === 'electron') {
        res = await h5Api.getElectronImage(layer.name);
      } else {
        throw new Error(`Unknown layer kind: ${layer.kind}`);
      }
      if (seq !== fetchSeq.current) return;
      setActiveImage(res.data.image);
      setActiveMeta({ ...layer, ...res.data });
    } catch (err) {
      if (seq !== fetchSeq.current) return;
      console.warn(`[useLayers] active layer fetch failed for ${activeLayerId}`, err);
      setError(err.response?.data?.detail || err.message);
      setActiveImage(null);
    } finally {
      if (seq === fetchSeq.current) setLoading(false);
    }
  }, [catalog, activeLayerId]);

  useEffect(() => { fetchActiveLayer(); }, [fetchActiveLayer]);

  return { catalog, activeImage, activeMeta, loading, error };
}
