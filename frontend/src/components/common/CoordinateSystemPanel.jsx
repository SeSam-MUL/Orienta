// frontend/src/components/common/CoordinateSystemPanel.jsx
// Shared collapsible coordinate-system editor — used in PhaseMapPage AND the
// detached pole-figure window. Reads/writes useFrameStore (same backend frame),
// so editing in one window updates the other via the storage event + backend.
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import useFrameStore from '../../stores/useFrameStore';
import { CollapsibleGroup, Label } from '../../theme/components';
import { colors, spacing } from '../../theme/tokens';
import OrientationHelperSvg from './OrientationHelperSvg';
import { PRESETS, X_DIRECTIONS, PROJECTIONS, HEMISPHERES } from './coordinateSystemPresets';

// theme/tokens.js has no `md`/`sm` spacing keys; map to the nearest existing
// scale entries (groupSpacing=10, innerSpacing=6) so we never invent tokens.
const GAP_MD = spacing.groupSpacing;
const GAP_SM = spacing.innerSpacing;

export default function CoordinateSystemPanel({ defaultCollapsed = false }) {
  const { t } = useTranslation('polefigure');
  const spec = useFrameStore((s) => s.spec);
  const loaded = useFrameStore((s) => s.loaded);
  const setSpec = useFrameStore((s) => s.setSpec);

  const patch = useCallback((mut) => {
    if (!spec) return;
    const next = JSON.parse(JSON.stringify(spec));
    mut(next);
    setSpec(next);
  }, [spec, setSpec]);

  if (!loaded || !spec) {
    return (
      <CollapsibleGroup title={t('coordSys.title')} defaultCollapsed={defaultCollapsed}>
        <div style={{ fontSize: '9pt', color: colors.textSecondary }}>{t('coordSys.noDataset')}</div>
      </CollapsibleGroup>
    );
  }
  const rot = spec.rotation, plot = spec.plot;

  return (
    <CollapsibleGroup title={t('coordSys.title')} defaultCollapsed={defaultCollapsed}>
      <div style={{ display: 'flex', gap: GAP_MD, alignItems: 'flex-start' }}>
        <OrientationHelperSvg plot={plot} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <Label htmlFor="cs-mode">{t('coordSys.rotationMode')}</Label>
          <select id="cs-mode" aria-label={t('coordSys.rotationMode')} value={rot.mode}
                  onChange={(e) => patch((n) => { n.rotation.mode = e.target.value; })}
                  style={selStyle}>
            <option value="preset">{t('coordSys.modeOpt.preset')}</option>
            <option value="axis_angle">{t('coordSys.modeOpt.axis_angle')}</option>
            <option value="euler">{t('coordSys.modeOpt.euler')}</option>
          </select>

          {rot.mode === 'preset' && (
            <>
              <Label htmlFor="cs-preset">{t('coordSys.preset')}</Label>
              <select id="cs-preset" aria-label={t('coordSys.preset')} value={rot.preset}
                      onChange={(e) => patch((n) => { n.rotation.preset = e.target.value; })}
                      style={selStyle}>
                {PRESETS.map((p) => <option key={p.id} value={p.id}>{t(`coordSys.presetOpt.${p.id}`, { defaultValue: p.label })}</option>)}
              </select>
            </>
          )}

          {rot.mode === 'axis_angle' && (
            <div style={{ display: 'flex', gap: 6, alignItems: 'flex-end', flexWrap: 'wrap' }}>
              {['x', 'y', 'z'].map((ax, i) => (
                <label key={ax} style={{ fontSize: '8pt', color: colors.textSecondary }}>
                  {t('coordSys.axis', { ax })}
                  <input type="number" step="1" value={rot.axis[i]} aria-label={t('coordSys.axis', { ax })}
                         onChange={(e) => patch((n) => { n.rotation.axis[i] = parseFloat(e.target.value) || 0; })}
                         style={numStyle} />
                </label>
              ))}
              <label style={{ fontSize: '8pt', color: colors.textSecondary }}>
                {t('coordSys.angle')}°
                <input type="number" step="1" value={rot.angle_deg} aria-label={t('coordSys.angle')}
                       onChange={(e) => patch((n) => { n.rotation.angle_deg = parseFloat(e.target.value) || 0; })}
                       style={numStyle} />
              </label>
            </div>
          )}

          {rot.mode === 'euler' && (
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {['φ1', 'Φ', 'φ2'].map((lbl, i) => (
                <label key={i} style={{ fontSize: '8pt', color: colors.textSecondary }}>
                  {lbl}°
                  <input type="number" step="1" value={rot.euler_deg[i]} aria-label={lbl}
                         onChange={(e) => patch((n) => { n.rotation.euler_deg[i] = parseFloat(e.target.value) || 0; })}
                         style={numStyle} />
                </label>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Plot convention */}
      <div style={{ marginTop: GAP_SM, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
        <div>
          <Label htmlFor="cs-x">{t('coordSys.xDirection')}</Label>
          <select id="cs-x" aria-label={t('coordSys.xDirection')} value={plot.x_direction}
                  onChange={(e) => patch((n) => { n.plot.x_direction = e.target.value; })} style={selStyle}>
            {X_DIRECTIONS.map((o) => <option key={o.id} value={o.id}>{t(`coordSys.xDirOpt.${o.id}`, { defaultValue: o.label })}</option>)}
          </select>
        </div>
        <div>
          <Label htmlFor="cs-proj">{t('coordSys.projection')}</Label>
          <select id="cs-proj" aria-label={t('coordSys.projection')} value={plot.projection}
                  onChange={(e) => patch((n) => { n.plot.projection = e.target.value; })} style={selStyle}>
            {PROJECTIONS.map((o) => <option key={o.id} value={o.id}>{t(`coordSys.projOpt.${o.id}`, { defaultValue: o.label })}</option>)}
          </select>
        </div>
        <div>
          <Label htmlFor="cs-hemi">{t('coordSys.hemisphere')}</Label>
          <select id="cs-hemi" aria-label={t('coordSys.hemisphere')} value={plot.hemisphere}
                  onChange={(e) => patch((n) => { n.plot.hemisphere = e.target.value; })} style={selStyle}>
            {HEMISPHERES.map((o) => <option key={o.id} value={o.id}>{t(`coordSys.hemiOpt.${o.id}`, { defaultValue: o.label })}</option>)}
          </select>
        </div>
        <label style={{ fontSize: '9pt', color: colors.text, display: 'flex', alignItems: 'center', gap: 6, marginTop: 18 }}
               title={t('hoverTips.zIntoPlane')}>
          <input type="checkbox" checked={plot.z_into_plane}
                 onChange={(e) => patch((n) => { n.plot.z_into_plane = e.target.checked; })} />
          {t('coordSys.zIntoPlane')}
        </label>
      </div>

      <label style={{ fontSize: '9pt', color: colors.text, display: 'flex', alignItems: 'center', gap: 6, marginTop: GAP_SM }}
             title={t('hoverTips.applyToExport')}>
        <input type="checkbox" checked={spec.apply_to_export}
               onChange={(e) => patch((n) => { n.apply_to_export = e.target.checked; })} />
        {t('coordSys.applyToExports')}
      </label>
    </CollapsibleGroup>
  );
}

const selStyle = {
  width: '100%', background: colors.bg, color: colors.text,
  border: `1px solid ${colors.border}`, borderRadius: 3, padding: '3px 6px',
  fontSize: '9pt', marginBottom: 6,
};
const numStyle = {
  width: 52, background: colors.bg, color: colors.text,
  border: `1px solid ${colors.border}`, borderRadius: 3, padding: '2px 4px', fontSize: '9pt',
};
