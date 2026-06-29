import { useTranslation } from 'react-i18next';
import useCockpitStore from '../../../stores/useCockpitStore';
import { colors } from '../../../theme/components';
import { CATEGORY_ORDER } from './layerCatalog';
import LayerThumb from './LayerThumb';

export default function LayerRail({ isFileOpen }) {
  const { t } = useTranslation('hdf5viewer');
  const catalog = useCockpitStore((s) => s.layerCatalog);
  const activeLayerId = useCockpitStore((s) => s.activeLayerId);
  const setActiveLayer = useCockpitStore((s) => s.setActiveLayer);

  if (!isFileOpen || !catalog) {
    return (
      <div style={{
        height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: colors.textSecondary, fontSize: '9pt', padding: 8,
      }}>
        {isFileOpen ? t('layerRail.loadingLayers') : t('layerRail.noFileOpen')}
      </div>
    );
  }

  return (
    <div className="thin-scrollbar" style={{
      height: '100%', overflowY: 'auto', padding: 6,
      background: colors.bgSecondary, borderRight: `1px solid ${colors.border}`,
    }}>
      {CATEGORY_ORDER.map((cat) => {
        const layers = catalog[cat] ?? [];
        if (layers.length === 0) return null;
        return (
          <div key={cat} style={{ marginBottom: 8 }}>
            <div style={{
              fontSize: '8pt', fontWeight: 700, color: colors.accent,
              padding: '4px 2px', textTransform: 'uppercase', letterSpacing: 1,
              borderBottom: `1px solid ${colors.border}`, marginBottom: 4,
            }}>
              {t(`layerRail.categories.${cat}`)}
            </div>
            {layers.map((layer) => (
              <LayerThumb
                key={layer.id}
                layer={layer}
                active={activeLayerId === layer.id}
                onSelect={setActiveLayer}
              />
            ))}
          </div>
        );
      })}
    </div>
  );
}
