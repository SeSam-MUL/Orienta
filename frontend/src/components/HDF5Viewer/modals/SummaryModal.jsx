import { useTranslation } from 'react-i18next';
import Modal from './Modal';
import useDataStore from '../../../stores/useDataStore';
import { colors, alpha } from '../../../theme/components';

export default function SummaryModal({ onClose }) {
  const { t } = useTranslation('hdf5viewer');
  const {
    filePath, formatType, gridShape, patternCount, patternShape,
    hasEDS, hasElectronImages, edsElements, electronImages,
    hasPatterns, hasRawPatterns,
  } = useDataStore();
  const filename = filePath?.split(/[\\/]/).pop() ?? t('summary.dash');
  const rows = [
    { label: t('summary.file'), value: filename, color: colors.cyan },
    { label: t('summary.format'), value: formatType ?? t('summary.dash'), color: colors.text },
    { label: t('summary.grid'), value: t('summary.gridValue', { rows: gridShape[0], cols: gridShape[1] }), color: colors.text },
    { label: t('summary.totalPatterns'), value: String(patternCount), color: colors.orange },
    { label: t('summary.patternSize'), value: t('summary.patternSizeValue', { height: patternShape[0], width: patternShape[1] }), color: colors.text },
    { label: t('summary.processedPatterns'), value: hasPatterns ? t('summary.yes') : t('summary.no'), color: hasPatterns ? colors.green : colors.red },
    { label: t('summary.rawPatterns'), value: hasRawPatterns ? t('summary.yes') : t('summary.no'), color: hasRawPatterns ? colors.green : colors.textSecondary },
    { label: t('summary.edsData'), value: hasEDS ? t('summary.yesWith', { items: edsElements.join(', ') }) : t('summary.no'), color: hasEDS ? colors.green : colors.textSecondary },
    { label: t('summary.electronImages'), value: hasElectronImages ? t('summary.yesWith', { items: electronImages.join(', ') }) : t('summary.no'), color: hasElectronImages ? colors.green : colors.textSecondary },
  ];
  return (
    <Modal title={t('summary.title')} onClose={onClose}>
      {rows.map(({ label, value, color }) => (
        <div key={label} style={{
          display: 'flex', gap: 12, padding: '8px 0',
          borderBottom: `1px solid ${alpha(colors.border, 19)}`,
        }}>
          <span style={{ fontSize: '10pt', color: colors.textSecondary, width: 160 }}>{label}</span>
          <span style={{ fontSize: '10pt', color, fontWeight: 500 }}>{value}</span>
        </div>
      ))}
    </Modal>
  );
}
