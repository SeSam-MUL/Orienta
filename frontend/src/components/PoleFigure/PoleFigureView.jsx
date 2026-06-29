// frontend/src/components/PoleFigure/PoleFigureView.jsx
import { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { indexApi } from '../../services/api';
import useFrameStore from '../../stores/useFrameStore';
import CoordinateSystemPanel from '../common/CoordinateSystemPanel';
import { useStateVersionPoll } from './useStateVersionPoll';
import { usePoleFigure } from './usePoleFigure';
import { colors, spacing } from '../../theme/tokens';

// NOTE: theme/tokens.js `spacing` has no `md`/`sm` keys. Per Task 11's
// substitution we use spacing.groupSpacing (≈10) for the brief's `spacing.md`.
const PAD = spacing.groupSpacing;

// Normalise a result's `phases` to `{ id, name }`. The backend returns objects
// (`{ id, name }`, unindexed -1 already filtered) but we defensively handle the
// legacy string shape too — same approach as PhaseMapPage/IndexingPage.
function normalisePhases(rawPhases) {
  return (rawPhases || []).map((p, i) => ({
    id: typeof p === 'object' && p !== null ? p.id : i,
    name: (typeof p === 'object' && p !== null ? p.name : p) ?? `Phase ${i}`,
  }));
}

export default function PoleFigureView() {
  const { t } = useTranslation('polefigure');
  const [phases, setPhases] = useState([]);
  // Keep null until phases load — usePoleFigure skips the request while null,
  // so we never send a stale/hardcoded id (which would 400 or render the wrong
  // phase). Synced to the active result's first phase id once phases arrive.
  const [phaseId, setPhaseId] = useState(null);
  const [hkl, setHkl] = useState('100,110,111');
  const [mode, setMode] = useState('both');   // scatter | density | both
  const [refreshKey, setRefreshKey] = useState(0);
  const loadForFile = useFrameStore((s) => s.loadForFile);
  const frameSig = useFrameStore((s) => s.frameSig);

  const refreshPhases = useCallback(async (activeSourceFile) => {
    try {
      const { data } = await indexApi.listResults();
      // Backend keys the result id as `id` (GET /api/indexing/results); fall
      // back to results[0] only when there's no active match.
      const active = (data.results || []).find((r) => r.id === data.active_id) || (data.results || [])[0];
      const next = normalisePhases(active?.phases);
      setPhases(next);
      // Sync phaseId to the active result's first phase. Reset whenever the
      // current phaseId isn't in the new list (result switch) so a stale id
      // from the previous result isn't sent to the backend.
      setPhaseId((prev) => {
        if (next.length === 0) return null;
        if (prev != null && next.some((p) => p.id === prev)) return prev;
        return next[0].id;
      });
    } catch { setPhases([]); setPhaseId(null); }
    if (activeSourceFile !== undefined) loadForFile(activeSourceFile);
  }, [loadForFile]);

  // Live: poll state-version. On any change, reload phases + frame and bump the
  // pole-figure refresh key (which re-fetches the image).
  useStateVersionPoll((snap) => {
    refreshPhases(snap.active_source_file);
    setRefreshKey((k) => k + 1);
  }, { intervalMs: 1500 });

  // Initial phase load.
  useEffect(() => { refreshPhases(undefined); }, [refreshPhases]);
  // Frame change in THIS window also re-renders the figure (frameSig changes).
  useEffect(() => { setRefreshKey((k) => k + 1); }, [frameSig]);

  const { image, loading, error } = usePoleFigure({ phaseId, hkl, mode, refreshKey });

  return (
    <div style={{ display: 'flex', height: '100%', background: colors.bg, color: colors.text }}>
      <div style={{ width: 300, padding: PAD, borderRight: `1px solid ${colors.border}`, overflow: 'auto' }}>
        <label style={{ fontSize: '9pt', color: colors.textSecondary }}>{t('polefigure:controls.phaseLabel')}</label>
        <select aria-label={t('polefigure:controls.phaseAria')} title={t('polefigure:controls.phaseTooltip')} value={phaseId ?? ''} onChange={(e) => { setPhaseId(Number(e.target.value)); setRefreshKey((k) => k + 1); }}
                style={{ width: '100%', marginBottom: 8, background: colors.bg, color: colors.text, border: `1px solid ${colors.border}`, borderRadius: 3, padding: '3px 6px' }}>
          {phases.length === 0 && <option value="">{t('polefigure:controls.noPhases')}</option>}
          {phases.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
        </select>
        <label style={{ fontSize: '9pt', color: colors.textSecondary }}>{t('polefigure:controls.hklLabel')}</label>
        <input aria-label={t('polefigure:controls.hklAria')} title={t('polefigure:controls.hklTooltip')} value={hkl} onChange={(e) => setHkl(e.target.value)}
               onBlur={() => setRefreshKey((k) => k + 1)}
               style={{ width: '100%', marginBottom: 8, background: colors.bg, color: colors.text, border: `1px solid ${colors.border}`, borderRadius: 3, padding: '3px 6px' }} />
        <label style={{ fontSize: '9pt', color: colors.textSecondary }}>{t('polefigure:controls.styleLabel')}</label>
        <select aria-label={t('polefigure:controls.styleAria')} title={t('polefigure:controls.styleTooltip')} value={mode} onChange={(e) => { setMode(e.target.value); setRefreshKey((k) => k + 1); }}
                style={{ width: '100%', marginBottom: 12, background: colors.bg, color: colors.text, border: `1px solid ${colors.border}`, borderRadius: 3, padding: '3px 6px' }}>
          <option value="both">{t('polefigure:controls.styleBoth')}</option>
          <option value="density">{t('polefigure:controls.styleDensity')}</option>
          <option value="scatter">{t('polefigure:controls.styleScatter')}</option>
        </select>
        <CoordinateSystemPanel />
      </div>
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: PAD }}>
        {error && <div style={{ color: colors.red, fontSize: '10pt' }}>{String(error)}</div>}
        {!error && loading && <div style={{ color: colors.textSecondary }}>{t('polefigure:view.rendering')}</div>}
        {!error && image && (
          <img alt={t('polefigure:view.imageAlt')} src={`data:image/png;base64,${image}`}
               style={{ maxWidth: '100%', maxHeight: '100%', background: '#fff', borderRadius: 6 }} />
        )}
        {!error && !loading && !image && <div style={{ color: colors.textSecondary }}>{t('polefigure:view.noResult')}</div>}
      </div>
    </div>
  );
}
