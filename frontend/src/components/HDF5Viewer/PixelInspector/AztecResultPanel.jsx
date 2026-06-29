import { useTranslation } from 'react-i18next';
import { colors, alpha } from '../../../theme/components';
import { useAztecPixel } from '../hooks/useAztecPixel';

function madColor(mad) {
  if (mad == null) return colors.textSecondary;
  if (mad < 1) return colors.green;
  if (mad < 2) return colors.yellow;
  return colors.red;
}

const Row = ({ label, value, color = colors.text }) => (
  <div style={{
    display: 'flex', justifyContent: 'space-between', padding: '2px 0',
    fontSize: '9pt', borderBottom: `1px solid ${alpha(colors.border, 19)}`,
  }}>
    <span style={{ color: colors.textSecondary }}>{label}</span>
    <span style={{ color, fontFamily: 'monospace' }}>{value ?? '—'}</span>
  </div>
);

export default function AztecResultPanel({ isFileOpen, currentRow, currentCol }) {
  const { t } = useTranslation('hdf5viewer');
  const { data, loading } = useAztecPixel(isFileOpen, currentRow, currentCol);
  if (!isFileOpen) return <div style={{ color: colors.textSecondary }}>{t('aztec.noFile')}</div>;
  if (loading || !data) return <div style={{ color: colors.textSecondary }}>{t('aztec.loading')}</div>;
  const e = data.euler_deg;
  return (
    <div>
      <Row label={t('aztec.phase')}
        value={data.phase_name ? (
          <>
            <span style={{
              width: 8, height: 8, display: 'inline-block', marginRight: 4,
              background: data.phase_color ? `rgb(${data.phase_color.join(',')})` : '#888',
            }} />
            {data.phase_name}
          </>
        ) : null}
      />
      <Row label={t('aztec.euler')} value={e ? `${e[0].toFixed(1)}° / ${e[1].toFixed(1)}° / ${e[2].toFixed(1)}°` : null} />
      <Row label={t('aztec.mad')} value={data.mad_deg != null ? `${data.mad_deg.toFixed(2)}°` : null} color={madColor(data.mad_deg)} />
      <Row label={t('aztec.bandContrast')} value={data.band_contrast} />
      <Row label={t('aztec.bandSlope')} value={data.band_slope} />
      <Row label={t('aztec.bands')} value={data.bands} />
      <Row label={t('aztec.patternQuality')} value={data.pattern_quality?.toFixed(2)} />
      <Row label={t('aztec.error')} value={data.error_code} />
      <Row label={t('aztec.pc')} value={data.pc ? data.pc.map((v) => v?.toFixed(3)).join(' / ') : null} />
      <Row label={t('aztec.liveRealTime')} value={
        data.live_time_s != null ? `${(data.live_time_s * 1000).toFixed(1)} / ${(data.real_time_s * 1000).toFixed(1)} ms` : null
      } />
      <Row label={t('aztec.beamPosition')} value={data.beam_position_um?.[0] != null ? `${data.beam_position_um[0].toFixed(1)} / ${data.beam_position_um[1].toFixed(1)} µm` : null} />
    </div>
  );
}
