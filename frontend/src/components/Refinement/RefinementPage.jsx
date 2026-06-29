import { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, spacing } from '../../theme/tokens';
import { Button, GroupBox, Input } from '../../theme/components';
import useRefinementStore from '../../stores/useRefinementStore';
import PhaseMap, { PHASE_PALETTE, phaseColor } from './PhaseMap';
import PixelInspector from './PixelInspector';
import OverrideTools from './OverrideTools';

// ---------------------------------------------------------------------------
// Phase legend strip
// ---------------------------------------------------------------------------
function PhaseLegend({ phaseNames }) {
  if (!phaseNames || phaseNames.length === 0) return null;
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 6 }}>
      {phaseNames.map((name, i) => (
        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '9pt' }}>
          <div style={{
            width: 12, height: 12, borderRadius: 2,
            background: phaseColor(i + 1),
            flexShrink: 0,
          }} />
          <span style={{ color: colors.text }}>{name}</span>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Stats panel — shows summary statistics from refinement summary
// ---------------------------------------------------------------------------
function StatsPanel({ summary }) {
  const { t } = useTranslation('refinement');
  if (!summary) return null;
  const stats = [
    { label: t('refinement:stats.totalPixels'), value: summary.total_pixels },
    { label: t('refinement:stats.overridden'),  value: summary.overridden_count },
    { label: t('refinement:stats.avgCi'),       value: summary.mean_ci != null ? summary.mean_ci.toFixed(3) : null },
    { label: t('refinement:stats.lowCi'),       value: summary.low_ci_count },
  ].filter((s) => s.value != null);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      {stats.map((s) => (
        <div key={s.label} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9pt' }}>
          <span style={{ color: colors.textSecondary }}>{s.label}</span>
          <span style={{ color: colors.text, fontFamily: 'monospace' }}>{s.value}</span>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// RefinementPage
// ---------------------------------------------------------------------------
export default function RefinementPage({ onNavigate, isActive }) {
  const { t } = useTranslation(['refinement', 'common']);
  const {
    fileStem, searchDir, summary,
    selectedPixel, ciThreshold, mode,
    loadFile, selectPixel, overridePixels, reset,
  } = useRefinementStore();

  const [fileStemInput, setFileStemInput] = useState('');
  const [searchDirInput, setSearchDirInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showUncertainty, setShowUncertainty] = useState(true);

  // Sync inputs when store loads a file externally (e.g. from BatchDashboard)
  useEffect(() => {
    if (fileStem) setFileStemInput(fileStem);
    if (searchDir) setSearchDirInput(searchDir);
  }, [fileStem, searchDir]);

  // Batch-Dashboard handoff: stem + dir written to sessionStorage by the
  // "Open in Refinement" button. Check every time the page becomes active
  // so a user can run multiple batches in one session and each subsequent
  // handoff still triggers the load.
  useEffect(() => {
    if (!isActive) return;
    let stem, dir;
    try {
      stem = sessionStorage.getItem('refinement_handoff_stem');
      dir  = sessionStorage.getItem('refinement_handoff_dir');
    } catch { return; }
    if (!stem) return;
    try {
      sessionStorage.removeItem('refinement_handoff_stem');
      sessionStorage.removeItem('refinement_handoff_dir');
    } catch { /* ignore */ }
    setFileStemInput(stem);
    if (dir) setSearchDirInput(dir);
    setTimeout(() => loadFile(stem, dir || null).catch(() => {}), 100);
  }, [isActive, loadFile]);

  const handleLoad = useCallback(async () => {
    if (!fileStemInput.trim()) return;
    setLoading(true);
    setError(null);
    try {
      await loadFile(fileStemInput.trim(), searchDirInput.trim() || null);
    } catch (e) {
      setError(e.response?.data?.detail ?? e.message ?? t('refinement:errors.loadFailed'));
    } finally {
      setLoading(false);
    }
  }, [fileStemInput, searchDirInput, loadFile, t]);

  const handlePixelClick = useCallback(async (row, col) => {
    if (mode !== 'click' && mode !== 'auto') return;
    setLoading(true);
    try {
      await selectPixel(row, col);
    } catch { /* ignore */ }
    finally {
      setLoading(false);
    }
  }, [mode, selectPixel]);

  const handleAssignPhase = useCallback(async ({ row, col, phaseId, name }) => {
    setLoading(true);
    try {
      await overridePixels([{ row, col, phase_id: phaseId }], `manual:${name}`);
    } catch (e) {
      setError(e.message ?? t('refinement:errors.overrideFailed'));
    } finally {
      setLoading(false);
    }
  }, [overridePixels, t]);

  const handleExport = useCallback(() => {
    if (!summary?.export_url) return;
    window.open(summary.export_url, '_blank');
  }, [summary]);

  const phaseMapData = summary?.phase_map ?? null;
  const ciMap        = showUncertainty ? (summary?.ci_map ?? null) : null;
  const gridShape    = summary?.grid_shape ?? [0, 0];
  const phaseNames   = summary?.phase_names ?? [];

  return (
    <div style={{
      display: isActive ? 'flex' : 'none',
      flexDirection: 'column',
      height: '100%',
      overflow: 'hidden',
    }}>
      {/* Top bar — file selector */}
      <div style={{
        padding: `${spacing.innerSpacing}px ${spacing.innerMargin}px`,
        borderBottom: `1px solid ${colors.border}`,
        display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 8,
        background: colors.bgSecondary,
      }}>
        <span style={{ fontSize: '12pt', fontWeight: 700, color: colors.accent, marginRight: 4 }}>
          {t('refinement:page.title')}
        </span>
        <Input
          value={fileStemInput}
          onChange={(e) => setFileStemInput(e.target.value)}
          placeholder={t('refinement:page.fileStemPlaceholder')}
          title={t('refinement:tooltips.fileStem')}
          style={{ width: 220 }}
        />
        <Input
          value={searchDirInput}
          onChange={(e) => setSearchDirInput(e.target.value)}
          placeholder={t('refinement:page.searchDirPlaceholder')}
          title={t('refinement:tooltips.searchDir')}
          style={{ width: 240 }}
        />
        <Button
          onClick={handleLoad}
          variant="primary"
          disabled={loading || !fileStemInput.trim()}
          title={t('refinement:tooltips.load')}
        >
          {loading ? t('common:loading') : t('common:load')}
        </Button>
        {fileStem && (
          <Button onClick={reset} variant="ghost" small title={t('refinement:tooltips.clear')}>
            {t('refinement:page.clear')}
          </Button>
        )}
        {error && (
          <span style={{ fontSize: '9pt', color: colors.red }}>{error}</span>
        )}

        {/* Uncertainty overlay toggle */}
        <label
          title={t('refinement:hoverTips.showUncertainty')}
          style={{
            display: 'flex', alignItems: 'center', gap: 6,
            fontSize: '9pt', color: colors.textSecondary, cursor: 'pointer', marginLeft: 'auto',
          }}
        >
          <input
            type="checkbox"
            checked={showUncertainty}
            onChange={(e) => setShowUncertainty(e.target.checked)}
            style={{ accentColor: colors.accent }}
          />
          {t('refinement:page.uncertaintyOverlay')}
        </label>
      </div>

      {/* Main body: left = map, right = inspector + stats */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden', minHeight: 0 }}>

        {/* Left — phase map + legend */}
        <div style={{
          flex: 1, overflow: 'auto',
          padding: spacing.innerMargin,
          display: 'flex', flexDirection: 'column', gap: spacing.groupSpacing,
        }}>
          {phaseMapData ? (
            <>
              <PhaseMap
                phaseMapData={phaseMapData}
                gridShape={gridShape}
                onPixelClick={handlePixelClick}
                selectedPixel={selectedPixel}
                uncertaintyOverlay={ciMap}
                ciThreshold={ciThreshold}
              />
              <PhaseLegend phaseNames={phaseNames} />
            </>
          ) : (
            <div style={{
              flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
              color: colors.textSecondary, fontSize: '10pt', textAlign: 'center',
            }}>
              {fileStem
                ? t('refinement:page.loadingPhaseMap')
                : t('refinement:page.emptyPrompt')}
            </div>
          )}
        </div>

        {/* Right sidebar — pixel inspector + stats */}
        <div style={{
          width: 280, flexShrink: 0,
          borderLeft: `1px solid ${colors.border}`,
          display: 'flex', flexDirection: 'column',
          overflow: 'hidden',
        }}>
          <div style={{ flex: 1, overflow: 'auto', padding: spacing.innerMargin }}>
            <GroupBox title={t('refinement:inspector.title')} style={{ marginBottom: spacing.groupSpacing }}>
              <PixelInspector
                selectedPixel={selectedPixel}
                onAssignPhase={handleAssignPhase}
              />
            </GroupBox>

            {summary && (
              <GroupBox title={t('refinement:stats.title')}>
                <StatsPanel summary={summary} />
              </GroupBox>
            )}
          </div>
        </div>
      </div>

      {/* Bottom toolbar */}
      <OverrideTools onExport={handleExport} />
    </div>
  );
}
