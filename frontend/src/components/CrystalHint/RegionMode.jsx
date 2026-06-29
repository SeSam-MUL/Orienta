/**
 * Region / Whole-Scan Mode (Mode 2).
 *
 * User picks an ROI (whole scan / rectangle), clicks Analyse, gets back
 * statistical histograms of symmetry + lattice across the region, plus
 * a ranked list of "missing phase" clusters — signatures present in
 * multiple pixels but with no library match.
 */

import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/tokens';
import useDataStore from '../../stores/useDataStore';
import { crystalHintApi } from '../../services/api';
import HistogramBars from './HistogramBars';
import MissingPhaseClusters from './MissingPhaseClusters';

const S = {
  layout: {
    display: 'grid',
    gridTemplateColumns: '340px 1fr',
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
  select: {
    background: colors.bg, color: colors.text,
    border: `1px solid ${colors.border}`,
    borderRadius: 4, padding: '4px 8px',
    fontSize: 12, flex: 1,
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
};

export default function RegionMode({ elements, presetKey, presetEntry, isActive }) {
  const { t } = useTranslation('crystalhint');
  // Either path (EBSD Viewer's `ebsd/load` or HDF5 Viewer's `h5/open`)
  // sets up the backend h5_session — both gate Crystal Hint.
  const fileLoaded = useDataStore(s => s.ebsdLoaded || s.isFileOpen);
  const [roiType, setRoiType] = useState('whole');
  const [rect, setRect] = useState({ rMin: 0, rMax: 50, cMin: 0, cMax: 50 });
  const [maxPixels, setMaxPixels] = useState(256);
  const [avgRadius, setAvgRadius] = useState(1);  // 3x3 by default — same as Mode 1
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = useCallback(async () => {
    setError(null);
    setLoading(true);
    setResult(null);
    try {
      const payload = {
        elements,
        presetKey: presetKey || null,
        roiType,
        rect: roiType === 'rect'
          ? [rect.rMin, rect.rMax, rect.cMin, rect.cMax]
          : null,
        polygon: null,
        maxPixels,
        avgRadius,
      };
      const res = await crystalHintApi.analyzeRegion(payload);
      setResult(res.data);
    } catch (err) {
      const msg = err?.response?.data?.detail || err?.message || 'unknown error';
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [elements, presetKey, roiType, rect, maxPixels, avgRadius]);

  return (
    <div style={S.layout}>
      <div style={S.leftPanel}>
        <div style={S.card}>
          <div style={S.cardTitle}>{t('region.roi')}</div>
          <div style={S.row}>
            <span style={S.label}>{t('region.typeLabel')}</span>
            <select
              style={S.select}
              value={roiType}
              onChange={e => setRoiType(e.target.value)}
              title={t('region.roiTooltip')}
            >
              <option value="whole">{t('region.wholeScan')}</option>
              <option value="rect">{t('region.rectangle')}</option>
            </select>
          </div>
          {roiType === 'rect' && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginTop: 8 }}>
              <label style={{ fontSize: 10, color: colors.textSecondary }} title={t('region.rectTooltip')}>
                {t('region.rowMin')}
                <input type="number" min={0} style={{ ...S.input, width: '100%', marginTop: 2 }}
                  value={rect.rMin} onChange={e => setRect({ ...rect, rMin: parseInt(e.target.value, 10) || 0 })} />
              </label>
              <label style={{ fontSize: 10, color: colors.textSecondary }} title={t('region.rectTooltip')}>
                {t('region.rowMax')}
                <input type="number" min={0} style={{ ...S.input, width: '100%', marginTop: 2 }}
                  value={rect.rMax} onChange={e => setRect({ ...rect, rMax: parseInt(e.target.value, 10) || 0 })} />
              </label>
              <label style={{ fontSize: 10, color: colors.textSecondary }} title={t('region.rectTooltip')}>
                {t('region.colMin')}
                <input type="number" min={0} style={{ ...S.input, width: '100%', marginTop: 2 }}
                  value={rect.cMin} onChange={e => setRect({ ...rect, cMin: parseInt(e.target.value, 10) || 0 })} />
              </label>
              <label style={{ fontSize: 10, color: colors.textSecondary }} title={t('region.rectTooltip')}>
                {t('region.colMax')}
                <input type="number" min={0} style={{ ...S.input, width: '100%', marginTop: 2 }}
                  value={rect.cMax} onChange={e => setRect({ ...rect, cMax: parseInt(e.target.value, 10) || 0 })} />
              </label>
            </div>
          )}
          <div style={S.hint}>
            {t('region.roiHint')}
          </div>
        </div>

        <div style={S.card}>
          <div style={S.cardTitle}>{t('region.sampling')}</div>
          <div style={S.row}>
            <span style={S.label}>{t('region.maxPixels')}</span>
            <input
              type="number"
              min={16}
              max={4096}
              step={32}
              style={S.input}
              value={maxPixels}
              onChange={e => setMaxPixels(parseInt(e.target.value, 10) || 256)}
              title={t('region.maxPixelsTooltip')}
            />
          </div>
          <div style={S.hint}>
            {t('region.maxPixelsHint')}
          </div>
          <div style={{ ...S.row, marginTop: 8 }}>
            <span style={S.label}>{t('region.avgRadius')}</span>
            <select
              style={S.input}
              value={avgRadius}
              onChange={e => setAvgRadius(parseInt(e.target.value, 10))}
              title={t('region.avgRadiusTooltip')}
            >
              <option value={0}>{t('region.avg0')}</option>
              <option value={1}>{t('region.avg1')}</option>
              <option value={2}>{t('region.avg2')}</option>
            </select>
          </div>
          <div style={S.hint}>
            {t('region.avgRadiusHint')}
          </div>
        </div>

        <button
          style={{ ...S.button, ...((!fileLoaded || loading) ? S.buttonDisabled : {}) }}
          disabled={!fileLoaded || loading}
          onClick={handleAnalyze}
          title={t('region.analyseTooltip')}
        >
          {loading ? t('region.analysing') : t('region.analyse')}
        </button>

        {!fileLoaded && (
          <div style={S.hint}>{t('region.fileHint')}</div>
        )}
        {error && <div style={S.error}>{t('region.error', { msg: error })}</div>}
      </div>

      <div style={S.rightPanel}>
        {!result && !loading && (
          <div style={{
            ...S.card,
            color: colors.textSecondary,
            textAlign: 'center',
            padding: 32,
          }}>
            {t('region.placeholderPre')}<b>{t('region.placeholderBold')}</b>{t('region.placeholderPost')}
          </div>
        )}

        {result && (
          <>
            <div style={S.card}>
              <div style={S.cardTitle}>{t('region.summary')}</div>
              <div style={S.metric}>
                <span style={S.metricLabel}>{t('region.pixelsInRoi')}</span>
                <span style={S.metricValue}>{result.n_pixels_in_roi}</span>
              </div>
              <div style={S.metric}>
                <span style={S.metricLabel}>{t('region.pixelsAnalyzed')}</span>
                <span style={S.metricValue}>{result.n_pixels_analyzed}</span>
              </div>
              <div style={S.metric}>
                <span style={S.metricLabel}>{t('region.noSymmetry')}</span>
                <span style={S.metricValue}>
                  {result.n_pixels_no_symmetry}
                  {result.n_pixels_analyzed > 0 && (
                    <span style={{ fontSize: 11, color: colors.textSecondary, fontWeight: 400 }}>
                      &nbsp;({Math.round(100 * result.n_pixels_no_symmetry / result.n_pixels_analyzed)}%)
                    </span>
                  )}
                </span>
              </div>
              <div style={S.metric}>
                <span style={S.metricLabel}>{t('region.libraryMatchRate')}</span>
                <span style={{
                  ...S.metricValue,
                  color: result.library_match_rate > 0.5 ? colors.green
                       : result.library_match_rate > 0.2 ? colors.orange
                       : colors.red,
                }}>
                  {Math.round(100 * result.library_match_rate)}%
                </span>
              </div>
              <div style={S.hint}>
                {t('region.elapsed', { seconds: result.elapsed_seconds.toFixed(1) })}
              </div>
            </div>

            <HistogramBars
              title={t('region.histNFold')}
              data={result.n_fold_histogram}
              labelKey={k => t('region.nFoldLabel', { n: k })}
              total={result.n_pixels_analyzed - result.n_pixels_no_symmetry}
            />

            <HistogramBars
              title={t('region.histLatticeA')}
              data={result.lattice_a_histogram}
              labelKey={k => t('region.latticeALabel', { value: k })}
              total={Object.values(result.lattice_a_histogram).reduce((a, b) => a + b, 0)}
              orderKeys={['<3.5', '3.5-5.0', '5.0-7.0', '7.0-10.0', '10.0-15.0', '>15']}
            />

            <HistogramBars
              title={t('region.histCrystalSystem')}
              data={result.crystal_system_histogram}
              total={Object.values(result.crystal_system_histogram).reduce((a, b) => a + b, 0)}
            />

            <MissingPhaseClusters
              clusters={result.missing_phase_clusters}
              presetEntry={presetEntry}
            />
          </>
        )}
      </div>
    </div>
  );
}
