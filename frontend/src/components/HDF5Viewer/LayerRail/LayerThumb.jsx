import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { h5Api } from '../../../services/api';
import { colors, alpha } from '../../../theme/components';
import useEdsColorStore from '../../../stores/useEdsColorStore';

const THUMB_W = 110;
const THUMB_H = 70;

export default function LayerThumb({ layer, active, onSelect }) {
  const { t } = useTranslation('hdf5viewer');
  const [thumb, setThumb] = useState(null);
  const [loading, setLoading] = useState(true);

  // EDS element thumbnails are backend-rendered with the colour BAKED IN, so a
  // colour edit must refetch. Subscribe to this layer's colour reactively and
  // key the fetch on it (effect dep below). Non-EDS layers get '' and never
  // refetch on colour changes. (colors[sym] || '#bd93f9' == the old getColor().)
  const sym = layer.kind === 'eds_element'
    ? ((layer.name || '').replace(/^Window Integral\s*/i, '').trim().split(/\s/)[0] || layer.name)
    : null;
  const edsColor = useEdsColorStore((s) => (sym ? (s.colors[sym] || '#bd93f9') : ''));

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    (async () => {
      try {
        let res;
        if (layer.kind === 'scalar') {
          res = await h5Api.getScalarMap(layer.path, 'gray');
        } else if (layer.kind === 'phase') {
          res = await h5Api.getPhaseMap();
        } else if (layer.kind === 'ipf') {
          res = await h5Api.getIPFMap(layer.direction);
        } else if (layer.kind === 'eds_element') {
          res = await h5Api.getEDSMap(layer.name, 'hot', edsColor);
        } else if (layer.kind === 'electron') {
          res = await h5Api.getElectronImage(layer.name);
        }
        if (!cancelled && res) setThumb(res.data.image);
      } catch (err) {
        console.warn(`[LayerThumb] failed for ${layer.id}`, err);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [layer.id, edsColor]);

  return (
    <div
      onClick={() => onSelect(layer.id)}
      title={layer.name}
      style={{
        cursor: 'pointer', padding: 4, marginBottom: 4,
        border: `2px solid ${active ? colors.purple : 'transparent'}`,
        borderRadius: 3,
        background: active ? alpha(colors.purple, 16) : 'transparent',
      }}
    >
      <div style={{
        width: THUMB_W, height: THUMB_H, background: colors.bg,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        overflow: 'hidden', border: `1px solid ${colors.border}`,
      }}>
        {loading ? (
          <div className="skeleton-shimmer" style={{ width: '100%', height: '100%' }} />
        ) : thumb ? (
          <img
            src={`data:image/png;base64,${thumb}`}
            alt={layer.name}
            style={{ width: '100%', height: '100%', objectFit: 'cover', imageRendering: 'pixelated' }}
          />
        ) : (
          <span style={{ fontSize: '8pt', color: colors.red }}>{t('layerRail.missing')}</span>
        )}
      </div>
      <div style={{
        fontSize: '8pt', color: active ? colors.purple : colors.textSecondary,
        textAlign: 'center', marginTop: 2, fontWeight: active ? 600 : 400,
      }}>
        {layer.name}
      </div>
    </div>
  );
}
