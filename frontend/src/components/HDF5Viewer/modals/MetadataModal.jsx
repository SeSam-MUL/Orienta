import { useTranslation } from 'react-i18next';
import Modal from './Modal';
import { colors, alpha } from '../../../theme/components';
import { useHeaders } from '../hooks/useHeaders';

const Section = ({ title, children }) => (
  <div style={{ marginBottom: 16 }}>
    <div style={{ fontSize: '10pt', fontWeight: 700, color: colors.accent, marginBottom: 6 }}>{title}</div>
    {children}
  </div>
);

const KV = ({ k, v }) => (
  <div style={{
    display: 'grid', gridTemplateColumns: '160px 1fr',
    padding: '3px 0', fontSize: '9pt',
    borderBottom: `1px solid ${alpha(colors.border, 19)}`,
  }}>
    <span style={{ color: colors.textSecondary }}>{k}</span>
    <span style={{ color: colors.text, fontFamily: 'monospace' }}>
      {v == null ? '—' : Array.isArray(v) ? v.map((x) => typeof x === 'number' ? x.toFixed(3) : x).join(' / ') : String(v)}
    </span>
  </div>
);

export default function MetadataModal({ isFileOpen, onClose }) {
  const { t } = useTranslation('hdf5viewer');
  const { ebsd, eds } = useHeaders(isFileOpen);
  if (!isFileOpen) return null;
  return (
    <Modal title={t('metadata.title')} onClose={onClose}>
      {!ebsd && <div style={{ color: colors.textSecondary }}>{t('metadata.loading')}</div>}
      {ebsd && (
        <>
          <Section title={t('metadata.sectionGrid')}>
            <KV k={t('metadata.xyCells')} v={[ebsd.grid?.x_cells, ebsd.grid?.y_cells]} />
            <KV k={t('metadata.xyStep')} v={[ebsd.grid?.x_step_um, ebsd.grid?.y_step_um]} />
            <KV k={t('metadata.scanningRotation')} v={ebsd.grid?.scanning_rotation_angle_deg} />
          </Section>
          <Section title={t('metadata.sectionPattern')}>
            <KV k={t('metadata.size')} v={[ebsd.pattern?.height, ebsd.pattern?.width]} />
            <KV k={t('metadata.acquiredSize')} v={[ebsd.pattern?.acquired_height, ebsd.pattern?.acquired_width]} />
            <KV k={t('metadata.acquisitionSpeed')} v={ebsd.pattern?.acquisition_speed} />
            <KV k={t('metadata.framesAveraged')} v={ebsd.pattern?.frames_averaged} />
          </Section>
          <Section title={t('metadata.sectionMicroscope')}>
            <KV k={t('metadata.beamVoltage')} v={ebsd.microscope?.beam_voltage_kV} />
            <KV k={t('metadata.workingDistance')} v={ebsd.microscope?.working_distance_mm} />
            <KV k={t('metadata.magnification')} v={ebsd.microscope?.magnification} />
            <KV k={t('metadata.sampleTilt')} v={ebsd.microscope?.sample_tilt_deg} />
          </Section>
          <Section title={t('metadata.sectionDetector')}>
            <KV k={t('metadata.orientationEuler')} v={ebsd.detector?.orientation_euler_deg} />
            <KV k={t('metadata.insertionDistance')} v={ebsd.detector?.insertion_distance_mm} />
            <KV k={t('metadata.cameraMode')} v={ebsd.detector?.mode} />
          </Section>
          <Section title={t('metadata.sectionPhases')}>
            {(ebsd.phases ?? []).map((p) => (
              <div key={p.id} style={{ borderLeft: `3px solid rgb(${p.color?.join(',')})`, paddingLeft: 8, marginBottom: 8 }}>
                <KV k={t('metadata.name')} v={p.name} />
                <KV k={t('metadata.lattice')} v={p.lattice_dimensions} />
                <KV k={t('metadata.angles')} v={p.lattice_angles} />
                <KV k={t('metadata.spaceGroup')} v={p.space_group} />
                <KV k={t('metadata.reflectors')} v={p.n_reflectors} />
              </div>
            ))}
          </Section>
        </>
      )}
      {eds && (
        <Section title={t('metadata.sectionEDS')}>
          <KV k={t('metadata.channelWidth')} v={eds.channel_width_eV} />
          <KV k={t('metadata.numberOfChannels')} v={eds.number_channels} />
          <KV k={t('metadata.energyRange')} v={eds.energy_range_keV} />
          <KV k={t('metadata.beamVoltage')} v={eds.beam_voltage_kV} />
          <KV k={t('metadata.detectorAzimuthElevation')} v={[eds.detector?.azimuth, eds.detector?.elevation]} />
          <KV k={t('metadata.detectorSerial')} v={eds.detector?.serial} />
          <KV k={t('metadata.windowType')} v={eds.window_type} />
          <KV k={t('metadata.strobeFwhm')} v={eds.strobe_fwhm_eV} />
        </Section>
      )}
    </Modal>
  );
}
