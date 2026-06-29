/**
 * Quality Check (Mode 3).
 *
 * Compares the indexed phase at each pixel with the symmetry detected
 * directly from the experimental pattern. Pixels where the detected
 * n-fold is INCOMPATIBLE with the indexed phase's crystal system are
 * flagged as likely misindexing.
 *
 * Requires an active indexing result (run Indexing first).
 */

import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/tokens';
import useDataStore from '../../stores/useDataStore';
import useResultStore from '../../stores/useResultStore';
import { crystalHintApi } from '../../services/api';

const S = {
  layout: {
    display: 'grid',
    gridTemplateColumns: '320px 1fr',
    gap: 16,
    height: '100%',
  },
  leftPanel: { display: 'flex', flexDirection: 'column', gap: 12 },
  rightPanel: { display: 'flex', flexDirection: 'column', gap: 12, overflow: 'auto' },
  card: {
    padding: 12,
    background: colors.bgSecondary,
    border: `1px solid ${colors.border}`,
    borderRadius: 6,
  },
  cardTitle: { fontSize: 13, fontWeight: 600, color: colors.accent, marginBottom: 8 },
  row: { display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 },
  label: { fontSize: 11, color: colors.textSecondary, minWidth: 70 },
  input: {
    background: colors.bg, color: colors.text,
    border: `1px solid ${colors.border}`,
    borderRadius: 4, padding: '3px 6px',
    fontSize: 11, width: 64,
  },
  button: {
    background: colors.accent, color: '#fff', border: 'none',
    borderRadius: 4, padding: '6px 14px', fontSize: 12, fontWeight: 600,
    cursor: 'pointer', width: '100%', marginTop: 8,
  },
  buttonDisabled: { opacity: 0.5, cursor: 'not-allowed' },
  hint: { fontSize: 10, color: colors.textSecondary, marginTop: 4 },
  error: {
    background: `${colors.red}22`, color: colors.red,
    border: `1px solid ${colors.red}88`,
    borderRadius: 4, padding: '8px 10px',
    fontSize: 11, marginTop: 8,
  },
  metric: { display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 4 },
  metricLabel: { fontSize: 11, color: colors.textSecondary },
  metricValue: { fontSize: 14, fontWeight: 600, color: colors.text },
  phaseRow: {
    display: 'grid',
    gridTemplateColumns: '1fr auto auto auto',
    gap: 8, padding: '4px 0',
    fontSize: 11, alignItems: 'baseline',
    borderBottom: `1px solid ${colors.border}`,
  },
  sampleRow: {
    padding: '6px 8px',
    background: colors.bg,
    border: `1px solid ${colors.border}`,
    borderRadius: 4,
    fontSize: 11,
    marginBottom: 4,
    display: 'flex', alignItems: 'baseline', gap: 6,
  },
  mismatchBadge: {
    display: 'inline-block', padding: '1px 6px',
    background: `${colors.red}33`, color: colors.red,
    border: `1px solid ${colors.red}66`, borderRadius: 3,
    fontSize: 9, fontWeight: 600, textTransform: 'uppercase',
  },
};

export default function QualityCheckMode({ isActive }) {
  const { t } = useTranslation('crystalhint');
  // Either path (EBSD Viewer's `ebsd/load` or HDF5 Viewer's `h5/open`)
  // sets up the backend h5_session — both gate Crystal Hint.
  const fileLoaded = useDataStore(s => s.ebsdLoaded || s.isFileOpen);
  const indexingResult = useResultStore(s => s.indexingResult);
  const [maxPixels, setMaxPixels] = useState(256);
  const [avgRadius, setAvgRadius] = useState(1);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = useCallback(async () => {
    setError(null);
    setLoading(true);
    setResult(null);
    try {
      const res = await crystalHintApi.qualityCheck({
        roiType: 'whole', maxPixels, avgRadius,
      });
      setResult(res.data);
    } catch (err) {
      const msg = err?.response?.data?.detail || err?.message || 'unknown error';
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [maxPixels, avgRadius]);

  const phaseRows = result
    ? Object.entries(result.mismatch_by_phase || {})
        .map(([pid, info]) => ({ pid: parseInt(pid, 10), ...info }))
        .sort((a, b) => b.mismatch_count - a.mismatch_count)
    : [];

  return (
    <div style={S.layout}>
      <div style={S.leftPanel}>
        <div style={S.card}>
          <div style={S.cardTitle}>{t('quality.title')}</div>
          <div style={S.hint}>
            {t('quality.intro')}
          </div>
        </div>

        <div style={S.card}>
          <div style={S.cardTitle}>{t('quality.sampling')}</div>
          <div style={S.row}>
            <span style={S.label}>{t('quality.maxPixels')}</span>
            <input
              type="number"
              min={16} max={4096} step={32}
              style={S.input}
              value={maxPixels}
              onChange={e => setMaxPixels(parseInt(e.target.value, 10) || 256)}
              title={t('quality.maxPixelsTooltip')}
            />
          </div>
          <div style={S.hint}>
            {t('quality.maxPixelsHint')}
          </div>
          <div style={{ ...S.row, marginTop: 8 }}>
            <span style={S.label}>{t('quality.avgRadius')}</span>
            <select
              style={S.input}
              value={avgRadius}
              onChange={e => setAvgRadius(parseInt(e.target.value, 10))}
              title={t('quality.avgRadiusTooltip')}
            >
              <option value={0}>{t('quality.avg0')}</option>
              <option value={1}>{t('quality.avg1')}</option>
              <option value={2}>{t('quality.avg2')}</option>
            </select>
          </div>
          <div style={S.hint}>
            {t('quality.avgRadiusHint')}
          </div>
        </div>

        <button
          style={{ ...S.button, ...((!fileLoaded || !indexingResult || loading) ? S.buttonDisabled : {}) }}
          disabled={!fileLoaded || !indexingResult || loading}
          onClick={handleAnalyze}
          title={t('quality.runTooltip')}
        >
          {loading ? t('quality.checking') : t('quality.run')}
        </button>

        {!fileLoaded && <div style={S.hint}>{t('quality.fileHint')}</div>}
        {fileLoaded && !indexingResult && (
          <div style={S.hint}>
            {t('quality.noIndexingHint')}
          </div>
        )}
        {error && <div style={S.error}>{t('quality.error', { msg: error })}</div>}
      </div>

      <div style={S.rightPanel}>
        {!result && !loading && (
          <div style={{
            ...S.card,
            color: colors.textSecondary,
            textAlign: 'center',
            padding: 32,
          }}>
            {t('quality.placeholder')}
          </div>
        )}

        {result && (
          <>
            <div style={S.card}>
              <div style={S.cardTitle}>{t('quality.summary')}</div>
              <div style={S.metric}>
                <span style={S.metricLabel}>{t('quality.pixelsAnalyzed')}</span>
                <span style={S.metricValue}>{result.n_pixels_analyzed}</span>
              </div>
              <div style={S.metric}>
                <span style={S.metricLabel}>{t('quality.indexed')}</span>
                <span style={S.metricValue}>{result.n_pixels_indexed}</span>
              </div>
              <div style={S.metric}>
                <span style={S.metricLabel}>{t('quality.unindexed')}</span>
                <span style={S.metricValue}>{result.n_pixels_unindexed}</span>
              </div>
              <div style={S.metric}>
                <span style={S.metricLabel}>{t('quality.match')}</span>
                <span style={{ ...S.metricValue, color: colors.green }}>
                  {result.n_pixels_match}
                </span>
              </div>
              <div style={S.metric}>
                <span style={S.metricLabel}>{t('quality.mismatch')}</span>
                <span style={{ ...S.metricValue, color: colors.red }}>
                  {result.n_pixels_mismatch}
                </span>
              </div>
              <div style={S.metric}>
                <span style={S.metricLabel}>{t('quality.noDetection')}</span>
                <span style={S.metricValue}>{result.n_pixels_no_detection}</span>
              </div>
              <div style={S.metric}>
                <span style={S.metricLabel}>{t('quality.overallMismatchRate')}</span>
                <span style={{
                  ...S.metricValue,
                  color: result.mismatch_rate > 0.30 ? colors.red
                       : result.mismatch_rate > 0.10 ? colors.orange
                       : colors.green,
                }}>
                  {(result.mismatch_rate * 100).toFixed(1)}%
                </span>
              </div>
              <div style={S.hint}>{t('quality.elapsed', { seconds: result.elapsed_seconds.toFixed(1) })}</div>
            </div>

            {phaseRows.length > 0 && (
              <div style={S.card}>
                <div style={S.cardTitle}>{t('quality.perPhaseTitle')}</div>
                <div style={S.phaseRow}>
                  <span style={{ fontWeight: 600 }}>{t('quality.colPhase')}</span>
                  <span style={{ fontWeight: 600 }}>{t('quality.colMatch')}</span>
                  <span style={{ fontWeight: 600 }}>{t('quality.colMismatch')}</span>
                  <span style={{ fontWeight: 600 }}>{t('quality.colRate')}</span>
                </div>
                {phaseRows.map(p => (
                  <div key={p.pid} style={S.phaseRow}>
                    <span>
                      {p.phase_name}
                      <span style={{ color: colors.textSecondary, fontSize: 10 }}>
                        {' '}({p.indexed_system})
                      </span>
                    </span>
                    <span style={{ color: colors.green, fontFamily: 'monospace' }}>{p.match_count}</span>
                    <span style={{ color: colors.red, fontFamily: 'monospace' }}>{p.mismatch_count}</span>
                    <span style={{
                      color: p.mismatch_rate > 0.30 ? colors.red
                           : p.mismatch_rate > 0.10 ? colors.orange
                           : colors.green,
                      fontFamily: 'monospace',
                    }}>
                      {(p.mismatch_rate * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            )}

            {result.sample_mismatches?.length > 0 && (
              <div style={S.card}>
                <div style={S.cardTitle}>
                  {t('quality.examplesTitle', { count: result.sample_mismatches.length })}
                </div>
                {result.sample_mismatches.map((m, i) => (
                  <div key={i} style={S.sampleRow}>
                    <span style={S.mismatchBadge}>{t('quality.mismatchBadge')}</span>
                    <span style={{ fontFamily: 'monospace' }}>
                      ({m.row}, {m.col})
                    </span>
                    <span>
                      <strong>{t('quality.detectedFold', { n: m.detected_n_fold })}</strong>
                      &nbsp;({(m.compatible_systems || []).join('/')})
                    </span>
                    <span style={{ color: colors.textSecondary }}>
                      {t('quality.indexedAs')} <strong>{m.indexed_phase_name}</strong> ({m.indexed_system})
                    </span>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
