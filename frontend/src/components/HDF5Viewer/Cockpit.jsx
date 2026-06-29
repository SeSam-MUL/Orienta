/**
 * Cockpit — 5-zone layout shell for the H5OINA viewer.
 *
 * Zones: TopBar | LayerRail | MapCanvas | PixelInspector | BottomNav.
 * Each zone is a child component; this file owns only layout + splitter wiring.
 *
 * NOTE: MapCanvas receives onNavigate via props.onNavigateMap (the parent
 * HDF5Viewer maps its own row/col handler into onNavigateMap).
 */
import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/components';
import useCockpitStore from '../../stores/useCockpitStore';
import TopBar from './TopBar';
import BottomNav from './BottomNav';
import LayerRail from './LayerRail/LayerRail';
import MapCanvas from './MapCanvas/MapCanvas';
import PixelInspector from './PixelInspector/PixelInspector';

export default function Cockpit(props) {
  const splitters = useCockpitStore((s) => s.splitters);
  const setSplitter = useCockpitStore((s) => s.setSplitter);

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', height: '100%',
      background: colors.bg, color: colors.text, overflow: 'hidden',
    }}>
      <TopBar {...props} />

      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>
        <div style={{ width: splitters.rail, flexShrink: 0, overflow: 'hidden' }}>
          <LayerRail {...props} />
        </div>
        <Splitter onResize={(d) => setSplitter('rail', Math.max(56, splitters.rail + d))} />

        <div style={{ flex: 1, minWidth: 360, overflow: 'hidden' }}>
          <MapCanvas {...props} onNavigate={props.onNavigateMap} />
        </div>
        <Splitter onResize={(d) => setSplitter('inspector', Math.max(280, splitters.inspector - d))} />

        <div style={{ width: splitters.inspector, flexShrink: 0, overflow: 'hidden' }}>
          <PixelInspector {...props} />
        </div>
      </div>

      <BottomNav {...props} />
    </div>
  );
}

function Splitter({ onResize }) {
  const { t } = useTranslation('hdf5viewer');
  const onMouseDown = (e) => {
    e.preventDefault();
    let lastX = e.clientX;
    const onMove = (ev) => {
      onResize(ev.clientX - lastX);
      lastX = ev.clientX;
    };
    const onUp = () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  };
  return (
    <div
      onMouseDown={onMouseDown}
      title={t('hoverTips.splitterResize')}
      aria-label={t('hoverTips.splitterResize')}
      style={{ width: 4, background: colors.border, cursor: 'ew-resize', flexShrink: 0 }}
    />
  );
}
