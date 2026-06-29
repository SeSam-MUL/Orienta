import { useTranslation } from 'react-i18next';
import { colors, Button } from '../../theme/components';

export default function CrystalToolbar({
  onBuildDatabase, onViewDatabase, onViewDWF, onSync,
  building, buildProgress,
  syncAvailable, syncing,
}) {
  const { t } = useTranslation('crystaldatabase');
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 6,
      padding: '8px 0 10px',
      borderBottom: `1px solid ${colors.border}`,
      marginBottom: 10,
      flexWrap: 'wrap',
    }}>
      <Button onClick={onBuildDatabase} variant="primary" disabled={building} style={{ fontSize: 11 }} title={t('toolbar.buildTooltip')}>
        {building ? t('toolbar.building') : t('toolbar.build')}
      </Button>
      <Button onClick={onViewDatabase} style={{ fontSize: 11 }} title={t('toolbar.viewDatabaseTooltip')}>
        {t('toolbar.viewDatabase')}
      </Button>
      <Button onClick={onViewDWF} style={{ fontSize: 11 }} title={t('toolbar.viewDwfTooltip')}>
        {t('toolbar.viewDwf')}
      </Button>

      {syncAvailable && (
        <>
          <div style={{ width: 1, height: 20, background: colors.border, margin: '0 4px' }} />
          <Button onClick={onSync} disabled={syncing} style={{ fontSize: 11 }} title={t('toolbar.syncTooltip')}>
            {syncing ? t('toolbar.syncing') : t('toolbar.sync')}
          </Button>
        </>
      )}

      {/* Build progress bar — only shown during building */}
      {building && buildProgress && (
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 8, minWidth: 120 }}>
          <div style={{
            flex: 1, height: 3, background: colors.border, borderRadius: 2, overflow: 'hidden'
          }}>
            <div style={{
              height: '100%',
              width: buildProgress.total > 0
                ? `${(buildProgress.progress / buildProgress.total) * 100}%`
                : '0%',
              background: colors.cyan,
              transition: 'width 0.3s ease',
              borderRadius: 2,
            }} />
          </div>
          <span style={{ fontSize: 10, color: colors.yellow, whiteSpace: 'nowrap' }}>
            {buildProgress.total > 0
              ? `${buildProgress.progress}/${buildProgress.total}`
              : buildProgress.message}
          </span>
        </div>
      )}
    </div>
  );
}
