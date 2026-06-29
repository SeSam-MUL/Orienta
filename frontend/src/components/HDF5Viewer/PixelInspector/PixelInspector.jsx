import { useTranslation } from 'react-i18next';
import { colors } from '../../../theme/components';
import CollapsiblePanel from './CollapsiblePanel';
import PatternPanel from './PatternPanel';
import SpectrumPanel from './SpectrumPanel';
import EDSCountsPanel from './EDSCountsPanel';
import AztecResultPanel from './AztecResultPanel';

export default function PixelInspector(props) {
  const { t } = useTranslation('hdf5viewer');
  return (
    <div className="thin-scrollbar" style={{
      height: '100%', overflowY: 'auto',
      background: colors.bgSecondary, borderLeft: `1px solid ${colors.border}`,
    }}>
      <CollapsiblePanel id="pattern" title={t('inspector.panelTitlePattern')}>
        <PatternPanel {...props} />
      </CollapsiblePanel>
      <CollapsiblePanel id="spectrum" title={t('inspector.panelTitleSpectrum')}>
        <SpectrumPanel {...props} />
      </CollapsiblePanel>
      <CollapsiblePanel id="counts" title={t('inspector.panelTitleCounts')}>
        <EDSCountsPanel {...props} />
      </CollapsiblePanel>
      <CollapsiblePanel id="aztec" title={t('inspector.panelTitleAztec')}>
        <AztecResultPanel {...props} />
      </CollapsiblePanel>
    </div>
  );
}
