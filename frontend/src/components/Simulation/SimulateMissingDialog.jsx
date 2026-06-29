import { useState, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, alpha, Button, Input } from '../../theme/components';

const SIZE_COLOR = { ok: colors.green, moderate: colors.yellow, large: colors.orange, unknown: colors.textSecondary };
const SIZE_LABEL_KEY = { ok: 'sizeLevelOk', moderate: 'sizeLevelModerate', large: 'sizeLevelLarge', unknown: 'sizeLevelUnknown' };

/**
 * Pre-launch settings for "Simulate All Missing".
 *
 * Per phase: cell size, an adaptive recommended dmin (small cells keep the fine
 * default, large cells get coarser so the reflection range stays ~15), an
 * include checkbox, and a DISORDER warning. Structures with partially-occupied
 * / mixed sites (occupancy_level "severe", occ < 0.5) can hang the EMsoft
 * master-pattern run — they are flagged red and EXCLUDED by default. On launch
 * it returns the included phases with their per-phase dmin.
 *
 * The whole batch routes to a single pipeline chosen by the manual engine
 * switch — `engine` is 'ours' (default) or 'emsoft'. There is no per-phase
 * EMsoft-vs-ours routing, so no per-phase engine column is shown.
 */
export default function SimulateMissingDialog({ missing, defaultDmin, onLaunch, onClose, engine = 'ours' }) {
  const { t } = useTranslation(['simulation', 'common']);
  const useOurs = engine !== 'emsoft';
  const [dmins, setDmins] = useState(() => {
    const init = {};
    for (const m of missing) init[m.stem] = m.recommended_dmin != null ? m.recommended_dmin : defaultDmin;
    return init;
  });
  const [include, setInclude] = useState(() => {
    const init = {};
    for (const m of missing) init[m.stem] = m.occupancy_level !== 'severe'; // severe disorder off by default
    return init;
  });
  const [globalVal, setGlobalVal] = useState('');

  const largeCount = useMemo(() => missing.filter(m => m.dmin_level === 'large').length, [missing]);
  const disorderCount = useMemo(
    () => missing.filter(m => m.occupancy_level === 'severe' || m.occupancy_level === 'mild').length, [missing]);
  const includedCount = useMemo(() => missing.filter(m => include[m.stem]).length, [missing, include]);

  const setOne = (stem, v) => setDmins(prev => ({ ...prev, [stem]: v }));
  const toggle = (stem) => setInclude(prev => ({ ...prev, [stem]: !prev[stem] }));

  const applyRecommended = () => setDmins(prev => {
    const next = { ...prev };
    for (const m of missing) if (m.recommended_dmin != null) next[m.stem] = m.recommended_dmin;
    return next;
  });
  const applyGlobal = () => {
    const val = parseFloat(globalVal);
    if (!isFinite(val) || val <= 0) return;
    const next = {}; for (const m of missing) next[m.stem] = val; setDmins(next);
  };

  const launch = () => {
    const sel = missing.filter(m => include[m.stem]).map(m => {
      const v = parseFloat(dmins[m.stem]);
      return { stem: m.stem, xtal_path: m.xtal_path, dmin: isFinite(v) && v > 0 ? v : defaultDmin };
    });
    onLaunch(sel);
  };

  const rangeAtChosen = (m) => {
    const v = parseFloat(dmins[m.stem]);
    if (m.a_max_nm == null || !isFinite(v) || v <= 0) return null;
    return m.a_max_nm / v;
  };

  const th = (extra = {}) => ({
    padding: '7px 10px', textAlign: 'left', color: colors.textSecondary, fontWeight: 600,
    borderBottom: `2px solid ${colors.border}`, whiteSpace: 'nowrap',
    position: 'sticky', top: 0, background: colors.bgSecondary, zIndex: 1, ...extra,
  });

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', zIndex: 1100,
      display: 'flex', alignItems: 'center', justifyContent: 'center', backdropFilter: 'blur(2px)' }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div style={{ background: colors.bg, border: `1px solid ${colors.border}`, borderRadius: 8,
        width: 760, maxHeight: '82vh', display: 'flex', flexDirection: 'column',
        boxShadow: '0 12px 40px rgba(0,0,0,0.4)', animation: 'fadeSlideIn 0.2s ease-out' }}>

        {/* Header */}
        <div style={{ padding: '12px 16px', borderBottom: `1px solid ${colors.border}`, flexShrink: 0 }}>
          <div style={{ fontSize: 14, fontWeight: 700, color: colors.accent, display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <span>{t('missingDialog.titlePhases', { count: missing.length })}</span>
            <span style={{
              fontSize: 10, fontWeight: 600, padding: '2px 8px', borderRadius: 10,
              background: alpha(useOurs ? colors.cyan : colors.purple, 12),
              color: useOurs ? colors.cyan : colors.purple,
              border: `1px solid ${alpha(useOurs ? colors.cyan : colors.purple, 27)}`,
            }}>
              {useOurs ? t('missingDialog.pipelineOurs') : t('missingDialog.pipelineEmsoft')}
            </span>
          </div>
          <div style={{ fontSize: 11, color: colors.textSecondary, marginTop: 3, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            {largeCount > 0 && <span><span style={{ color: colors.orange, fontWeight: 600 }}>{t('missingDialog.largeSummary', { count: largeCount })}</span>{t('missingDialog.largeSummarySuffix')}</span>}
            {disorderCount > 0 && <span><span style={{ color: colors.red, fontWeight: 600 }}>{t('missingDialog.disorderSummary', { count: disorderCount })}</span>{t('missingDialog.disorderSummarySuffix')}</span>}
          </div>
        </div>

        {/* Toolbar */}
        <div style={{ padding: '8px 16px', borderBottom: `1px solid ${colors.border}`, flexShrink: 0,
          display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          <Button onClick={applyRecommended} style={{ fontSize: 11 }} title={t('missingDialog.useRecommendedTooltip')}>{t('missingDialog.useRecommended')}</Button>
          <span style={{ width: 1, height: 18, background: colors.border }} />
          <span style={{ fontSize: 11, color: colors.textSecondary }}>{t('missingDialog.setAllTo')}</span>
          <Input value={globalVal} onChange={(e) => setGlobalVal(e.target.value)} placeholder={t('missingDialog.setAllPlaceholder')}
            title={t('hoverTips.globalDmin')}
            style={{ width: 70, fontSize: 11, height: 26, padding: '2px 8px' }} />
          <Button onClick={applyGlobal} disabled={!globalVal} style={{ fontSize: 11 }} title={t('hoverTips.applyGlobalDmin')}>{t('common:apply')}</Button>
        </div>

        {/* Table */}
        <div className="thin-scrollbar" style={{ flex: 1, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr>
                <th style={th({ width: 34, textAlign: 'center' })}>{t('missingDialog.colInclude')}</th>
                <th style={th()}>{t('missingDialog.colPhase')}</th>
                <th style={th({ textAlign: 'right' })}>{t('missingDialog.colA')}</th>
                <th style={th({ textAlign: 'right' })}>{t('missingDialog.colDmin')}</th>
                <th style={th({ textAlign: 'right' })}>{t('missingDialog.colRange')}</th>
                <th style={th()}>{t('missingDialog.colStatus')}</th>
              </tr>
            </thead>
            <tbody>
              {missing.map((m) => {
                const rc = rangeAtChosen(m);
                const inc = include[m.stem];
                const sev = m.occupancy_level === 'severe';
                const mild = m.occupancy_level === 'mild';
                return (
                  <tr key={m.stem} className="table-row-hover"
                    style={{ borderBottom: `1px solid ${alpha(colors.border, 13)}`,
                      background: sev ? alpha(colors.red, 6) : 'transparent', opacity: inc ? 1 : 0.5 }}>
                    <td style={{ padding: '5px 10px', textAlign: 'center' }}>
                      <input type="checkbox" checked={!!inc} onChange={() => toggle(m.stem)}
                        title={inc ? t('missingDialog.included') : t('missingDialog.excluded')} />
                    </td>
                    <td style={{ padding: '5px 10px', color: colors.cyan, whiteSpace: 'nowrap',
                      maxWidth: 240, overflow: 'hidden', textOverflow: 'ellipsis' }} title={m.stem}>{m.stem}</td>
                    <td style={{ padding: '5px 10px', textAlign: 'right', color: colors.text }}>
                      {m.a_max_nm != null ? (m.a_max_nm * 10).toFixed(2) : '—'}</td>
                    <td style={{ padding: '3px 8px', textAlign: 'right' }}>
                      <input value={dmins[m.stem] ?? ''} onChange={(e) => setOne(m.stem, e.target.value)}
                        title={t('hoverTips.perPhaseDmin')}
                        style={{ width: 62, background: colors.bgSecondary, border: `1px solid ${colors.border}`,
                          borderRadius: 3, color: colors.text, fontSize: 12, padding: '2px 6px', textAlign: 'right', outline: 'none' }} /></td>
                    <td style={{ padding: '5px 10px', textAlign: 'right', color: rc != null && rc > 18 ? colors.red : colors.text }}>
                      {rc != null ? rc.toFixed(1) : '—'}</td>
                    <td style={{ padding: '5px 10px', whiteSpace: 'nowrap' }}>
                      <span style={{ color: SIZE_COLOR[m.dmin_level] || colors.textSecondary, fontWeight: 600, fontSize: 11 }}>
                        {SIZE_LABEL_KEY[m.dmin_level] ? t(`missingDialog.${SIZE_LABEL_KEY[m.dmin_level]}`) : m.dmin_level}</span>
                      {(sev || mild) && (
                        <span style={{ marginLeft: 8, color: sev ? colors.red : colors.yellow, fontWeight: 600, fontSize: 11 }}
                          title={t('missingDialog.disorderTooltip')}>
                          {sev ? t('missingDialog.disorderedSevere') : t('missingDialog.disorderedMild')} {m.min_occupancy != null ? t('missingDialog.disorderedOcc', { pct: Math.round(m.min_occupancy * 100) }) : ''}
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Footer */}
        <div style={{ padding: '10px 16px', borderTop: `1px solid ${colors.border}`, flexShrink: 0,
          display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 11, color: colors.textSecondary }}>
            {t('missingDialog.footerSelected', { included: includedCount, total: missing.length })}
          </span>
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
            <Button onClick={onClose} style={{ fontSize: 12 }} title={t('hoverTips.missingCancel')}>{t('common:cancel')}</Button>
            <Button onClick={launch} variant="primary" disabled={includedCount === 0} style={{ fontSize: 12 }} title={t('hoverTips.missingLaunch')}>
              {t('missingDialog.launch', { count: includedCount })}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
