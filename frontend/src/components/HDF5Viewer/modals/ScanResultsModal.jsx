/**
 * ScanResultsModal — renders the quality OR defect scan results dialog.
 *
 * Props:
 *   open       — boolean, render nothing when false
 *   kind       — 'quality' | 'defects'
 *   result     — null | 'scanning' | { sampled, best?, worst?, counts?, error? }
 *   onClose    — callback to close the dialog
 *   onGoTo     — (index) => void, used by the quality dialog "Go to" buttons.
 *                Called BEFORE onClose so the parent can update the current
 *                index and then close the dialog.
 */
import { useTranslation } from 'react-i18next';
import { colors, alpha, Button } from '../../../theme/components';
import Modal from './Modal';

const DEFECT_CATEGORIES = [
  { key: 'ok',         labelKey: 'scan.categoryOk',        colorKey: 'green' },
  { key: 'low_signal', labelKey: 'scan.categoryLowSignal', colorKey: 'yellow' },
  { key: 'saturated',  labelKey: 'scan.categorySaturated', colorKey: 'orange' },
  { key: 'hot_pixels', labelKey: 'scan.categoryHotPixels', colorKey: 'pink' },
  { key: 'beam_off',   labelKey: 'scan.categoryBeamOff',   colorKey: 'red' },
];

function NoData() {
  const { t } = useTranslation('hdf5viewer');
  return (
    <div style={{ color: colors.textSecondary, fontSize: '10pt', textAlign: 'center', padding: '16px 0' }}>
      <div style={{ fontSize: '18pt', opacity: 0.25, marginBottom: 4 }}>{'⌓'}</div>
      {t('scan.noData')}
    </div>
  );
}

function QualityBody({ result, onGoTo, onClose }) {
  const { t } = useTranslation('hdf5viewer');
  if (result === 'scanning') {
    return (
      <div style={{ color: colors.textSecondary, fontSize: '10pt', textAlign: 'center', padding: '20px 0' }}>
        {t('scan.qualityScanning')}
      </div>
    );
  }
  if (result?.error) {
    return <div style={{ color: colors.red, fontSize: '10pt' }}>{result.error}</div>;
  }
  if (result) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <p style={{ fontSize: '9pt', color: colors.textSecondary, margin: 0 }}>
          {t('scan.qualitySampled', { count: result.sampled })}
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          <div style={{ background: colors.bgTertiary, border: `1px solid ${alpha(colors.green, 25)}`, borderRadius: 5, padding: 12 }}>
            <div style={{ color: colors.green, fontSize: '10pt', fontWeight: 600, marginBottom: 6 }}>{t('scan.bestPattern')}</div>
            <div style={{ fontFamily: 'monospace', fontSize: '11pt', color: colors.text }}>
              {t('scan.indexValue', { index: result.best.index })}
            </div>
            <div style={{ fontSize: '9pt', color: colors.textSecondary }}>
              {t('scan.stdDev', { value: result.best.score.toFixed(1) })}
            </div>
            <Button
              small
              style={{ marginTop: 8 }}
              onClick={() => { onGoTo(result.best.index); onClose(); }}
              title={t('scan.goToTooltip')}
            >
              {t('scan.goTo')}
            </Button>
          </div>
          <div style={{ background: colors.bgTertiary, border: `1px solid ${alpha(colors.red, 25)}`, borderRadius: 5, padding: 12 }}>
            <div style={{ color: colors.red, fontSize: '10pt', fontWeight: 600, marginBottom: 6 }}>{t('scan.worstPattern')}</div>
            <div style={{ fontFamily: 'monospace', fontSize: '11pt', color: colors.text }}>
              {t('scan.indexValue', { index: result.worst.index })}
            </div>
            <div style={{ fontSize: '9pt', color: colors.textSecondary }}>
              {t('scan.stdDev', { value: result.worst.score.toFixed(1) })}
            </div>
            <Button
              small
              style={{ marginTop: 8 }}
              onClick={() => { onGoTo(result.worst.index); onClose(); }}
              title={t('scan.goToTooltip')}
            >
              {t('scan.goTo')}
            </Button>
          </div>
        </div>
      </div>
    );
  }
  return <NoData />;
}

function DefectsBody({ result }) {
  const { t } = useTranslation('hdf5viewer');
  if (result === 'scanning') {
    return (
      <div style={{ color: colors.textSecondary, fontSize: '10pt', textAlign: 'center', padding: '20px 0' }}>
        {t('scan.defectScanning')}
      </div>
    );
  }
  if (result) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <p style={{ fontSize: '9pt', color: colors.textSecondary, margin: 0 }}>
          {t('scan.defectSampled', { count: result.sampled })}
        </p>
        {DEFECT_CATEGORIES.map(({ key, labelKey, colorKey }) => {
          const color = colors[colorKey];
          const count = result.counts[key] ?? 0;
          const pct = result.sampled > 0
            ? ((count / result.sampled) * 100).toFixed(1)
            : '0.0';
          return (
            <div key={key} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <div style={{ width: 10, height: 10, borderRadius: '50%', background: color, flexShrink: 0 }} />
              <span style={{ fontSize: '10pt', color: colors.text, width: 100 }}>{t(labelKey)}</span>
              <div style={{
                flex: 1, height: 8,
                background: colors.bgTertiary,
                borderRadius: 3, overflow: 'hidden',
              }}>
                <div style={{ width: `${pct}%`, height: '100%', background: color, transition: 'width 0.3s' }} />
              </div>
              <span style={{ fontSize: '9pt', color: colors.textSecondary, width: 64, textAlign: 'right' }}>
                {t('scan.categoryCount', { count, pct })}
              </span>
            </div>
          );
        })}
      </div>
    );
  }
  return <NoData />;
}

export default function ScanResultsModal({ open, kind, result, onClose, onGoTo }) {
  const { t } = useTranslation('hdf5viewer');
  if (!open) return null;
  const title = kind === 'quality' ? t('scan.qualityTitle') : t('scan.defectTitle');
  return (
    <Modal title={title} onClose={onClose}>
      {kind === 'quality'
        ? <QualityBody result={result} onGoTo={onGoTo} onClose={onClose} />
        : <DefectsBody result={result} />}
    </Modal>
  );
}
