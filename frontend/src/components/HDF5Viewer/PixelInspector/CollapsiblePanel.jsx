import { useTranslation } from 'react-i18next';
import { colors } from '../../../theme/components';
import useCockpitStore from '../../../stores/useCockpitStore';

export default function CollapsiblePanel({ id, title, children }) {
  const { t } = useTranslation('hdf5viewer');
  const open = useCockpitStore((s) => s.panels[id]);
  const toggle = useCockpitStore((s) => s.togglePanel);
  return (
    <div style={{ borderBottom: `1px solid ${colors.border}` }}>
      <div
        onClick={() => toggle(id)}
        title={t('inspector.panelTooltip')}
        style={{
          padding: '6px 10px', cursor: 'pointer',
          display: 'flex', justifyContent: 'space-between',
          background: colors.bgTertiary,
          fontSize: '9pt', fontWeight: 600, color: colors.accent,
          userSelect: 'none',
        }}
      >
        <span>{title}</span>
        <span>{open ? '▾' : '▸'}</span>
      </div>
      {open && <div style={{ padding: 8 }}>{children}</div>}
    </div>
  );
}
