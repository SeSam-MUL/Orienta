/**
 * AnalysisPage.jsx
 *
 * EBSD Analysis — MTEX-equivalent post-processing.
 * Mirrors gui/analysis_gui.py (EBSDAnalysisPage) exactly:
 *   Tab 1: Data & Preprocessing
 *   Tab 2: Grain Analysis
 *   Tab 3: Deformation
 *   Tab 4: Texture
 *   Tab 5: Recrystallization
 *   Tab 6: Batch & Export
 *
 * Each tab owns its own map viewer area (right side) + controls (left side),
 * matching the per-tab AnalysisMapWidget instances in the PyQt5 original.
 *
 * Bottom bar: "Run Complete Analysis" button + status label, always visible.
 */

import { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { analysisApi, indexApi, edsApi } from '../../services/api';
import useResultStore from '../../stores/useResultStore';
import { toast } from '../../stores/useToastStore';
import {
  colors, alpha,
  Button,
  Label,
  Input,
  FormRow,
  ProgressBar,
} from '../../theme/components';

// ---------------------------------------------------------------------------
// Page-specific UI atoms not covered by shared components
// ---------------------------------------------------------------------------

function BrowseBtn({ onClick, disabled, title }) {
  const { t } = useTranslation('common');
  return (
    <Button
      onClick={onClick}
      disabled={disabled}
      title={title}
      style={{ padding: '4px 10px', fontSize: 11 }}
      small
    >
      {t('common:browse')}
    </Button>
  );
}

function FieldRow({ label, children, style }) {
  return (
    <div style={{ marginBottom: 10, ...style }}>
      {label && <Label secondary small style={{ display: 'block', marginBottom: 3 }}>{label}</Label>}
      {children}
    </div>
  );
}

function SectionTitle({ children, color = colors.cyan }) {
  return (
    <div style={{
      fontSize:       12,
      fontWeight:     700,
      color,
      marginBottom:   10,
      marginTop:      4,
      paddingBottom:  6,
      borderBottom:   `1px solid ${colors.border}`,
      letterSpacing:  0.3,
    }}>
      {children}
    </div>
  );
}

// Matches QGroupBox in PyQt5 — supports colored title via color prop
function ColoredGroupBox({ title, color = colors.purple, children, style }) {
  return (
    <div className="group-focus" style={{
      border:       `1px solid ${colors.border}`,
      borderRadius:  5,
      marginBottom:  10,
      transition:   'border-color 0.15s, box-shadow 0.15s',
      ...style,
    }}>
      {title && (
        <div style={{
          fontSize:    11,
          fontWeight:  700,
          color,
          padding:     '4px 10px',
          borderBottom:`1px solid ${colors.border}`,
          background:  colors.bgSecondary,
          borderRadius:'5px 5px 0 0',
        }}>
          {title}
        </div>
      )}
      <div style={{ padding: '10px 12px' }}>
        {children}
      </div>
    </div>
  );
}

function InfoBadge({ label, value, color = colors.cyan }) {
  return (
    <div
      style={{
        display:        'inline-flex',
        flexDirection:  'column',
        background:     colors.bgSecondary,
        border:         `1px solid ${colors.border}`,
        borderRadius:   5,
        padding:        '5px 10px',
        minWidth:       70,
        transition:     'transform 0.12s, border-color 0.15s',
      }}
      onMouseEnter={(e) => { e.currentTarget.style.transform = 'scale(1.04)'; e.currentTarget.style.borderColor = color; }}
      onMouseLeave={(e) => { e.currentTarget.style.transform = 'scale(1)'; e.currentTarget.style.borderColor = colors.border; }}
    >
      <span style={{ fontSize: 9, color: colors.textSecondary, textTransform: 'uppercase', letterSpacing: 0.5 }}>
        {label}
      </span>
      <span style={{ fontSize: 14, fontWeight: 700, color, marginTop: 1 }}>
        {value}
      </span>
    </div>
  );
}

function StatusMsg({ msg, isError }) {
  if (!msg) return null;
  return (
    <div role={isError ? 'alert' : 'status'} style={{
      marginTop:   8,
      padding:     '5px 10px',
      borderRadius: 4,
      background:  isError ? alpha(colors.red, 13) : alpha(colors.green, 8),
      border:      `1px solid ${isError ? colors.red : colors.green}`,
      color:       isError ? colors.red : colors.green,
      fontSize:    11,
      animation:   'fadeSlideIn 0.2s ease-out',
    }}>
      {msg}
    </div>
  );
}

// Read-only text area (matches QTextEdit read-only in PyQt5)
function ReadonlyText({ text, placeholder, minHeight = 90, style }) {
  return (
    <div style={{
      background:    colors.bgSecondary,
      border:        `1px solid ${colors.border}`,
      borderRadius:   4,
      padding:        '6px 10px',
      fontSize:       11,
      color:          text ? colors.text : colors.textSecondary,
      fontFamily:     'monospace',
      whiteSpace:     'pre-wrap',
      minHeight,
      overflowY:      'auto',
      ...style,
    }}>
      {text || placeholder}
    </div>
  );
}

// Divider
function Div() {
  return <div style={{ borderTop: `1px solid ${colors.border}`, margin: '12px 0' }} />;
}

// ---------------------------------------------------------------------------
// Map viewer — base64 image with layer/cmap controls
// Mirrors AnalysisMapWidget (analysis_map_widget.py)
// ---------------------------------------------------------------------------
// Layer label/tip are resolved at render time via t('analysis:map.layers.*').
const MAP_LAYER_OPTIONS = [
  { value: 'bc' },
  { value: 'ipf' },
  { value: 'ipf_x' },
  { value: 'ipf_y' },
  { value: 'kam' },
  { value: 'gos' },
  { value: 'ecd' },
  { value: 'rx' },
  { value: 'texture_component' },
  { value: 'grain_boundaries' },
];

const CMAP_OPTIONS = [
  { value: 'gray',      label: 'gray' },
  { value: 'viridis',   label: 'viridis' },
  { value: 'plasma',    label: 'plasma' },
  { value: 'inferno',   label: 'inferno' },
  { value: 'magma',     label: 'magma' },
  { value: 'cividis',   label: 'cividis' },
  { value: 'jet',       label: 'jet' },
  { value: 'hot',       label: 'hot' },
  { value: 'coolwarm',  label: 'coolwarm' },
  { value: 'RdYlBu_r',  label: 'RdYlBu_r' },
];

function MapViewer({ defaultLayer = 'bc', analysisLoaded, style }) {
  const { t } = useTranslation(['analysis', 'common']);
  const layerLabel = (v) => t(`analysis:map.layers.${v}`);
  const layerTip   = (v) => t(`analysis:map.layers.${v}Tip`);
  const [layer, setLayer]   = useState(defaultLayer);
  const [cmap, setCmap]     = useState('gray');
  const [image, setImage]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [info, setInfo]     = useState(t('analysis:map.noMapLoaded'));

  const handleRefresh = useCallback(async () => {
    if (!analysisLoaded) return;
    setLoading(true);
    try {
      const res = await analysisApi.getMap(layer, cmap);
      const img = res.data?.image || null;
      setImage(img);
      setInfo(img ? `${layerLabel(layer)} | ${cmap}` : t('analysis:map.noDataReturned'));
    } catch (err) {
      setInfo(err.response?.data?.detail || t('analysis:map.failedToLoad'));
    } finally {
      setLoading(false);
    }
  }, [analysisLoaded, layer, cmap]);

  return (
    <div style={{
      display:        'flex',
      flexDirection:  'column',
      flex:           1,
      minWidth:       0,
      ...style,
    }}>
      {/* Map controls bar — mirrors QGroupBox "Map Controls" */}
      <div style={{
        display:      'flex',
        gap:          8,
        alignItems:   'center',
        flexWrap:     'wrap',
        padding:      '6px 10px',
        background:   colors.bgSecondary,
        border:       `1px solid ${colors.border}`,
        borderRadius: '5px 5px 0 0',
        flexShrink:   0,
      }}>
        <span style={{ fontSize: 10, fontWeight: 700, color: colors.purple }}>{t('analysis:map.controls')}</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ fontSize: 10, color: colors.textSecondary }}>{t('analysis:map.layerLabel')}</span>
          <select
            value={layer}
            onChange={(e) => setLayer(e.target.value)}
            title={layerTip(layer)}
            style={{
              background: colors.bg, border: `1px solid ${colors.border}`, borderRadius: 3,
              color: colors.text, fontSize: 11, padding: '2px 4px', cursor: 'pointer',
            }}
          >
            {MAP_LAYER_OPTIONS.map(o => (
              <option key={o.value} value={o.value} title={layerTip(o.value)}>{layerLabel(o.value)}</option>
            ))}
          </select>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ fontSize: 10, color: colors.textSecondary }}>{t('analysis:map.colormapLabel')}</span>
          <select
            value={cmap}
            onChange={(e) => setCmap(e.target.value)}
            title={t('analysis:map.colormapTooltip')}
            style={{
              background: colors.bg, border: `1px solid ${colors.border}`, borderRadius: 3,
              color: colors.text, fontSize: 11, padding: '2px 4px', cursor: 'pointer',
            }}
          >
            {CMAP_OPTIONS.map(o => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </div>
        <Button
          onClick={handleRefresh}
          disabled={loading || !analysisLoaded}
          title={t('analysis:map.refreshTooltip')}
          style={{ padding: '3px 10px', fontSize: 11, background: colors.cyan, color: colors.textOnAccent }}
          small
        >
          {loading ? <span className="btn-loading">{t('analysis:map.loading')}</span> : t('analysis:map.refresh')}
        </Button>
        <span style={{ fontSize: 10, color: colors.textSecondary, marginLeft: 'auto' }}>{info}</span>
      </div>

      {/* Canvas area */}
      <div className="map-container" style={{
        flex:           1,
        background:     colors.bg,
        border:         `1px solid ${colors.border}`,
        borderTop:      'none',
        borderRadius:   '0 0 5px 5px',
        display:        'flex',
        alignItems:     'center',
        justifyContent: 'center',
        minHeight:      260,
        overflow:       'hidden',
        position:       'relative',
      }}>
        {image ? (
          <img
            className="image-reveal map-image"
            src={`data:image/png;base64,${image}`}
            alt={t('analysis:map.imageAlt')}
            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
          />
        ) : (
          <span style={{ fontSize: 12, color: colors.textSecondary }}>
            {analysisLoaded ? t('analysis:map.selectLayerHint') : t('analysis:map.loadDatasetFirst')}
          </span>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Per-tab layout: left = scrollable controls (300 px), right = map viewer
// ---------------------------------------------------------------------------
function TabLayout({ controls, defaultLayer = 'bc', analysisLoaded }) {
  return (
    <div style={{ display: 'flex', gap: 14, height: '100%', overflow: 'hidden' }}>
      {/* Left controls panel */}
      <div className="thin-scrollbar" style={{
        width:         300,
        minWidth:      300,
        overflowY:     'auto',
        paddingRight:  4,
        paddingBottom: 8,
      }}>
        {controls}
      </div>
      {/* Right map viewer */}
      <MapViewer defaultLayer={defaultLayer} analysisLoaded={analysisLoaded} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// EDS element table — used in Data & Preprocessing tab
// ---------------------------------------------------------------------------
function EDSElementTable({ elements }) {
  const { t } = useTranslation('analysis');
  if (!elements || elements.length === 0) return null;
  return (
    <div style={{
      background:    colors.bgSecondary,
      border:        `1px solid ${colors.border}`,
      borderRadius:  4,
      overflow:      'hidden',
      marginBottom:  10,
      fontSize:      11,
    }}>
      <div>
        <div style={{ display: 'grid', gridTemplateColumns: '60px 1fr 1fr 1fr' }}>
          {[
            t('analysis:edsTable.element'),
            t('analysis:edsTable.counts'),
            t('analysis:edsTable.wtPct'),
            t('analysis:edsTable.atPct'),
          ].map(h => (
            <div key={h} style={{
              padding: '4px 8px', background: colors.border,
              color: colors.textSecondary, fontWeight: 700,
              textTransform: 'uppercase', letterSpacing: 0.3,
            }}>
              {h}
            </div>
          ))}
        </div>
        {elements.map((el, i) => (
          <div key={`row-${i}`} className="table-row-hover" style={{
            display: 'grid', gridTemplateColumns: '60px 1fr 1fr 1fr',
          }}>
            <div style={{ padding: '4px 8px', color: colors.cyan, fontWeight: 600 }}>
              {el.symbol || el.name || el}
            </div>
            <div style={{ padding: '4px 8px', color: colors.text }}>
              {el.counts !== undefined ? el.counts.toFixed(0) : '—'}
            </div>
            <div style={{ padding: '4px 8px', color: colors.orange }}>
              {el.wt_pct !== undefined ? el.wt_pct.toFixed(2) : '—'}
            </div>
            <div style={{ padding: '4px 8px', color: colors.green }}>
              {el.at_pct !== undefined ? el.at_pct.toFixed(2) : '—'}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// TAB 1: Data & Preprocessing
// Mirrors _create_data_tab() in analysis_gui.py
// ---------------------------------------------------------------------------
function DataTab({ status, onLoad, onComplete, analysisLoaded }) {
  const { t } = useTranslation(['analysis', 'common']);
  const [xmapPath,         setXmapPath]         = useState('');
  const [loading,          setLoading]           = useState(false);
  const [msg,              setMsg]               = useState(null);
  const [isError,          setIsError]           = useState(false);

  // Session results selector (ResultStore / last indexing)
  const [sessionResults,   setSessionResults]    = useState([]);
  const [selectedSession,  setSelectedSession]   = useState('');
  const [sessionLoading,   setSessionLoading]    = useState(false);

  // Preprocessing controls — mirrors QFormLayout in preproc_group
  const [tiltAngle,        setTiltAngle]         = useState('70');
  const [zRotation,        setZRotation]         = useState('0');
  const [xRotation,        setXRotation]         = useState('0');
  const [bcThreshold,      setBcThreshold]       = useState('28');
  const [preprocessMsg,    setPreprocessMsg]     = useState(null);
  const [preprocessError,  setPreprocessError]   = useState(false);
  const [preprocessing,    setPreprocessing]     = useState(false);

  // Quality filter — wires the BC + Bands thresholds onto the live dataset
  // via /api/analysis/quality-filter. Independent from the (still TODO) tilt
  // / rotation preprocessing because this is the only one that actually
  // affects downstream statistics today.
  const [qfBcMin,        setQfBcMin]        = useState('28');
  const [qfBandsMin,     setQfBandsMin]     = useState('5');
  const [qfApplying,     setQfApplying]     = useState(false);
  const [qfStats,        setQfStats]        = useState(null);  // pixel-distribution from /quality-stats
  const [qfMsg,          setQfMsg]          = useState(null);
  const [qfMsgError,     setQfMsgError]     = useState(false);

  // Crop controls (optional)
  const [cropRow0,  setCropRow0]  = useState('');
  const [cropRow1,  setCropRow1]  = useState('');
  const [cropCol0,  setCropCol0]  = useState('');
  const [cropCol1,  setCropCol1]  = useState('');

  // EDS phase composition
  const [edsElements,    setEdsElements]    = useState([]);
  const [edsLoading,     setEdsLoading]     = useState(false);
  const [suggestMsg,     setSuggestMsg]     = useState(null);
  const [suggestedPhases,setSuggestedPhases]= useState([]);

  // Aztec-vs-Ours comparison
  const [azCmpLoading,   setAzCmpLoading]   = useState(false);
  const [azCmpImage,     setAzCmpImage]     = useState(null);
  const [azCmpStats,     setAzCmpStats]     = useState(null);
  const [azCmpError,     setAzCmpError]     = useState(null);

  // On mount: populate session results from last indexing
  useEffect(() => {
    indexApi.getLastResult().then((res) => {
      const d = res.data;
      if (d && d.available !== false) {
        const label = t('analysis:data.sessionResultLabel', { method: d.method || '?', count: d.n_indexed || '?' });
        setSessionResults([{ value: '__last_indexing__', label }]);
      }
    }).catch(() => {});
  }, []);

  const handleLoad = async (pathOverride) => {
    const path = (pathOverride || xmapPath).trim();
    if (!path) return;
    setLoading(true);
    setMsg(null);
    try {
      const res = await analysisApi.load(path);
      const d = res.data;
      onLoad(d);
      const shape = d.shape?.length ? d.shape.join(' × ') : t('analysis:data.unknownShape');
      const phases = d.phases ? t('analysis:data.loadedPhasesSuffix', { count: d.phases.length }) : '';
      setMsg(t('analysis:data.loadedMsg', { shape, phases }));
      setIsError(false);
      onComplete('data');
    } catch (err) {
      setMsg(err.response?.data?.detail || err.message || t('analysis:data.loadFailed'));
      setIsError(true);
    } finally {
      setLoading(false);
    }
  };

  const handleSessionSelect = async (e) => {
    const path = e.target.value;
    setSelectedSession(path);
    if (!path) return;
    setXmapPath(path);
    setSessionLoading(true);
    await handleLoad(path);
    setSessionLoading(false);
  };

  const handleApplyPreprocessing = async () => {
    setPreprocessMsg(t('analysis:data.preprocNotImplemented'));
    setPreprocessError(true);
  };

  // Fetch BC / Bands distribution stats so the threshold inputs can show
  // sensible reference values (5th / 50th / 95th percentile). Refreshed on
  // every analysis-load. Older light h5 without quality fields just returns
  // null — UI hides the box in that case.
  useEffect(() => {
    if (!analysisLoaded) {
      setQfStats(null);
      return;
    }
    analysisApi.qualityStats()
      .then(r => setQfStats(r.data))
      .catch(() => setQfStats(null));
  }, [analysisLoaded]);

  const handleAztecComparison = async () => {
    setAzCmpLoading(true);
    setAzCmpError(null);
    setAzCmpImage(null);
    setAzCmpStats(null);
    try {
      const r = await analysisApi.aztecComparison();
      setAzCmpImage(r.data?.image || null);
      setAzCmpStats(r.data || null);
    } catch (err) {
      setAzCmpError(err.response?.data?.detail || err.message || t('analysis:data.comparisonFailed'));
    } finally {
      setAzCmpLoading(false);
    }
  };

  const handleApplyQualityFilter = async () => {
    setQfApplying(true);
    setQfMsg(null);
    try {
      const bc = Number(qfBcMin) || 0;
      const bands = Number(qfBandsMin) || 0;
      const r = await analysisApi.applyQualityFilter(bc, bands);
      const d = r.data || {};
      const pct = (d.filtered_fraction || 0) * 100;
      setQfMsg(t('analysis:data.qualityFilterResult', {
        total: d.filtered_total,
        pct: pct.toFixed(2),
        byBc: d.filtered_by_bc,
        byBands: d.filtered_by_bands,
        after: d.indexed_after,
        before: d.indexed_before,
      }));
      setQfMsgError(false);
    } catch (err) {
      setQfMsg(err.response?.data?.detail || err.message || t('analysis:data.qualityFilterFailed'));
      setQfMsgError(true);
    } finally {
      setQfApplying(false);
    }
  };

  const handleFetchEDS = async () => {
    setEdsLoading(true);
    setSuggestMsg(null);
    try {
      const res = await edsApi.elements();
      const els = res.data?.elements || [];
      setEdsElements(els.map(e => ({ symbol: e })));
    } catch {
      setEdsElements([]);
    } finally {
      setEdsLoading(false);
    }
  };

  const handleSuggestPhases = async () => {
    setSuggestMsg(t('analysis:data.analysingEds'));
    setSuggestedPhases([]);
    try {
      const res = await edsApi.suggestPhases(0, 0);
      const phases = res.data?.suggested_phases || [];
      setSuggestedPhases(phases);
      setSuggestMsg(t('analysis:data.phasesSuggested', { count: phases.length }));
    } catch (err) {
      setSuggestMsg(err.response?.data?.detail || t('analysis:data.suggestionFailed'));
    }
  };

  return (
    <TabLayout defaultLayer="bc" analysisLoaded={analysisLoaded} controls={
      <>
        {/* Session Results — matches session_group in PyQt5 */}
        {sessionResults.length > 0 && (
          <ColoredGroupBox title={t('analysis:data.sessionTitle')} color={colors.green} style={{ animation: 'fadeSlideIn 0.2s ease-out' }}>
            <FieldRow label={t('analysis:data.sessionSelectLabel')}>
              <div style={{ display: 'flex', gap: 6 }}>
                <select
                  value={selectedSession}
                  onChange={handleSessionSelect}
                  disabled={sessionLoading || loading}
                  title={t('analysis:hoverTips.sessionSelect')}
                  style={{
                    flex: 1, background: colors.bgSecondary, border: `1px solid ${colors.border}`,
                    borderRadius: 4, color: colors.text, fontSize: 12, padding: '4px 8px',
                  }}
                >
                  <option value="">{t('analysis:data.sessionPlaceholder')}</option>
                  {sessionResults.map(o => (
                    <option key={o.value} value={o.value}>{o.label}</option>
                  ))}
                </select>
              </div>
            </FieldRow>
            {sessionLoading && (
              <div style={{ fontSize: 11, color: colors.cyan }}>{t('analysis:data.sessionLoading')}</div>
            )}
          </ColoredGroupBox>
        )}

        {/* Load CrystalMap — matches load_group */}
        <ColoredGroupBox title={t('analysis:data.loadTitle')}>
          <FieldRow label={t('analysis:data.xmapFileLabel')}>
            <div style={{ display: 'flex', gap: 6 }}>
              <Input
                value={xmapPath}
                onChange={(e) => setXmapPath(e.target.value)}
                disabled={loading}
                placeholder={t('analysis:data.xmapPlaceholder')}
                title={t('analysis:hoverTips.xmapPath')}
                style={{ flex: 1 }}
              />
              <BrowseBtn
                disabled={loading}
                title={t('analysis:data.browseFileTooltip')}
                onClick={async () => {
                  if (window.electronAPI?.openFile) {
                    const selected = await window.electronAPI.openFile({
                      filters: [
                        { name: t('analysis:data.fileFilterCrystalMap'), extensions: ['h5', 'hdf5', 'h5oina', 'ang', 'ctf'] },
                        { name: t('analysis:data.fileFilterAll'), extensions: ['*'] },
                      ],
                    });
                    if (selected) { setXmapPath(selected); handleLoad(selected); }
                  }
                }}
              />
            </div>
          </FieldRow>
          <Button
            onClick={() => handleLoad()}
            disabled={loading || !xmapPath.trim()}
            title={t('analysis:data.loadTooltip')}
            style={{ width: '100%', background: colors.green, color: colors.textOnAccent }}
          >
            {loading ? <span className="btn-loading">{t('analysis:data.loading')}</span> : t('analysis:data.load')}
          </Button>
          <StatusMsg msg={msg} isError={isError} />

          {/* Dataset info — mirrors data_info_label */}
          {status?.loaded && (
            <div style={{
              marginTop: 10, padding: '6px 8px',
              background: colors.bgSecondary, borderRadius: 4,
              fontSize: 11, color: colors.text,
            }}>
              {status.shape?.length > 0 && <div>{t('analysis:data.infoShape', { shape: status.shape.join(' × ') })}</div>}
              {status.phases?.length > 0 && <div>{t('analysis:data.infoPhases', { phases: status.phases.join(', ') })}</div>}
              {status.step_size > 0 && <div>{t('analysis:data.infoStep', { step: Number(status.step_size).toFixed(3) })}</div>}
            </div>
          )}
        </ColoredGroupBox>

        {/* Quality Filter — live, destructive: marks low-quality pixels
            as unindexed so grain reconstruction, KAM/GOS, texture and
            Excel export skip them. Only shown when the dataset actually
            carries BC / Bands (post-quality-field-copy light h5). */}
        {status?.loaded && qfStats && (qfStats.bc || qfStats.bands) && (
          <ColoredGroupBox title={t('analysis:data.qualityTitle')} color={colors.orange}>
            <div style={{ fontSize: 11, color: colors.textSecondary, marginBottom: 8 }}>
              {t('analysis:data.qualityDesc')}
            </div>

            {qfStats.bc && (
              <FormRow label={t('analysis:data.minBandContrast')}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <Input
                    type="number" value={qfBcMin}
                    onChange={(e) => setQfBcMin(e.target.value)}
                    min={0} max={Math.ceil(qfStats.bc.max)} step={1}
                    disabled={qfApplying}
                    title={t('analysis:data.minBandContrastTooltip')}
                    style={{ width: 70 }}
                  />
                  <span style={{ fontSize: 10, color: colors.textSecondary }}>
                    {t('analysis:data.percentileHint', { p5: qfStats.bc.p05.toFixed(0), p50: qfStats.bc.p50.toFixed(0), p95: qfStats.bc.p95.toFixed(0) })}
                  </span>
                </div>
              </FormRow>
            )}

            {qfStats.bands && (
              <FormRow label={t('analysis:data.minBands')}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <Input
                    type="number" value={qfBandsMin}
                    onChange={(e) => setQfBandsMin(e.target.value)}
                    min={0} max={Math.ceil(qfStats.bands.max)} step={1}
                    disabled={qfApplying}
                    title={t('analysis:data.minBandsTooltip')}
                    style={{ width: 70 }}
                  />
                  <span style={{ fontSize: 10, color: colors.textSecondary }}>
                    {t('analysis:data.percentileHint', { p5: qfStats.bands.p05.toFixed(0), p50: qfStats.bands.p50.toFixed(0), p95: qfStats.bands.p95.toFixed(0) })}
                  </span>
                </div>
              </FormRow>
            )}

            <Button
              onClick={handleApplyQualityFilter}
              disabled={qfApplying}
              title={t('analysis:data.applyQualityFilterTooltip')}
              style={{ width: '100%', marginTop: 6, background: colors.orange, color: colors.textOnAccent }}
            >
              {qfApplying ? t('analysis:data.applying') : t('analysis:data.applyQualityFilter')}
            </Button>
            <StatusMsg msg={qfMsg} isError={qfMsgError} />
          </ColoredGroupBox>
        )}

        {/* Preprocessing — tilt/rotation/smooth/crop + Apply are NO-OP on the
            backend (handleApplyPreprocessing only sets a "not implemented"
            message). Hidden for launch. The Quality Filter group above is the
            ONLY preprocessing that actually affects downstream statistics and
            stays visible. Do not delete — re-enable when the backend lands. */}
        {false && status?.loaded && (
          <ColoredGroupBox title={t('analysis:data.preprocTitle')}>
            <FormRow label={t('analysis:data.tiltCorrection')}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <Input
                  type="number" value={tiltAngle}
                  onChange={(e) => setTiltAngle(e.target.value)}
                  min={0} max={90} step={1}
                  disabled={preprocessing}
                  style={{ width: 70 }}
                />
                <span style={{ fontSize: 11, color: colors.textSecondary }}>{t('analysis:grain.unitDegrees')}</span>
              </div>
            </FormRow>
            <FormRow label={t('analysis:data.zAxisRotation')}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <Input
                  type="number" value={zRotation}
                  onChange={(e) => setZRotation(e.target.value)}
                  min={-180} max={180} step={1}
                  disabled={preprocessing}
                  style={{ width: 70 }}
                />
                <span style={{ fontSize: 11, color: colors.textSecondary }}>{t('analysis:grain.unitDegrees')}</span>
              </div>
            </FormRow>
            <FormRow label={t('analysis:data.xAxisRotation')}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <Input
                  type="number" value={xRotation}
                  onChange={(e) => setXRotation(e.target.value)}
                  min={-180} max={180} step={1}
                  disabled={preprocessing}
                  style={{ width: 70 }}
                />
                <span style={{ fontSize: 11, color: colors.textSecondary }}>{t('analysis:grain.unitDegrees')}</span>
              </div>
            </FormRow>
            <FormRow label={t('analysis:data.minBcThreshold')}>
              <Input
                type="number" value={bcThreshold}
                onChange={(e) => setBcThreshold(e.target.value)}
                min={0} max={255} step={1}
                disabled={preprocessing}
                style={{ width: 70 }}
              />
            </FormRow>

            {/* Crop (optional) */}
            <Div />
            <div style={{ fontSize: 11, color: colors.textSecondary, marginBottom: 6 }}>
              {t('analysis:data.cropRegionLabel')}
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 10 }}>
              <div>
                <Label secondary small style={{ display: 'block', marginBottom: 3 }}>{t('analysis:data.rowStart')}</Label>
                <Input type="number" value={cropRow0} onChange={(e) => setCropRow0(e.target.value)} min={0} />
              </div>
              <div>
                <Label secondary small style={{ display: 'block', marginBottom: 3 }}>{t('analysis:data.rowEnd')}</Label>
                <Input type="number" value={cropRow1} onChange={(e) => setCropRow1(e.target.value)} min={0} />
              </div>
              <div>
                <Label secondary small style={{ display: 'block', marginBottom: 3 }}>{t('analysis:data.colStart')}</Label>
                <Input type="number" value={cropCol0} onChange={(e) => setCropCol0(e.target.value)} min={0} />
              </div>
              <div>
                <Label secondary small style={{ display: 'block', marginBottom: 3 }}>{t('analysis:data.colEnd')}</Label>
                <Input type="number" value={cropCol1} onChange={(e) => setCropCol1(e.target.value)} min={0} />
              </div>
            </div>

            <Button
              onClick={handleApplyPreprocessing}
              disabled={preprocessing}
              style={{ width: '100%', background: colors.orange, color: colors.textOnAccent }}
            >
              {preprocessing ? t('analysis:data.applying') : t('analysis:data.applyPreprocessing')}
            </Button>
            <StatusMsg msg={preprocessMsg} isError={preprocessError} />
          </ColoredGroupBox>
        )}

        {/* Aztec-vs-Ours diagnostic — reads Aztec Phase + Euler from the
            source h5oina (via SourceReference) and confronts them with the
            in-memory analysis dataset. Shows where Aztec's Hough indexing
            and our (re)indexing disagree on phase and on orientation.
            Useful for sanity checking and frame-convention diagnosis. */}
        {status?.loaded && (
          <ColoredGroupBox title={t('analysis:data.aztecTitle')} color={colors.purple}>
            <div style={{ fontSize: 11, color: colors.textSecondary, marginBottom: 8 }}>
              {t('analysis:data.aztecDesc')}
            </div>
            <Button
              onClick={handleAztecComparison}
              disabled={azCmpLoading}
              title={t('analysis:data.compareWithAztecTooltip')}
              style={{ width: '100%', background: colors.purple, color: colors.textOnAccent, marginBottom: 6 }}
            >
              {azCmpLoading ? t('analysis:data.computing') : t('analysis:data.compareWithAztec')}
            </Button>
            {azCmpError && (
              <div style={{
                fontSize: 11, color: colors.red,
                background: alpha(colors.red, 7),
                border: `1px solid ${alpha(colors.red, 27)}`,
                borderRadius: 3, padding: '3px 6px', marginBottom: 6,
              }}>
                {azCmpError}
              </div>
            )}
            {azCmpImage && (
              <>
                <img
                  src={`data:image/png;base64,${azCmpImage}`}
                  alt={t('analysis:data.aztecImageAlt')}
                  style={{ width: '100%', borderRadius: 3, border: `1px solid ${colors.border}`, marginBottom: 6 }}
                />
                {azCmpStats && (
                  <div style={{ fontSize: 11, color: colors.textSecondary, fontFamily: 'monospace', lineHeight: 1.5 }}>
                    <div>{t('analysis:data.phaseAgreement', { pct: azCmpStats.agree_pct?.toFixed(1), agree: azCmpStats.n_agree_phase, both: azCmpStats.n_pixels_both_indexed })}</div>
                    <div>{t('analysis:data.misorientationLabel')}</div>
                    <div style={{ paddingLeft: 8 }}>{t('analysis:data.misorientationStats', { mean: azCmpStats.misorientation_deg?.mean?.toFixed(2), median: azCmpStats.misorientation_deg?.median?.toFixed(2), p95: azCmpStats.misorientation_deg?.p95?.toFixed(2) })}</div>
                    {azCmpStats.misorientation_deg?.mean > 5 && (
                      <div style={{ color: colors.orange, marginTop: 4 }}>
                        {t('analysis:data.misorientationNote', { mean: azCmpStats.misorientation_deg.mean.toFixed(0) })}
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </ColoredGroupBox>
        )}

        {/* EDS Phase Composition — matches eds_group (hidden until data loaded) */}
        {status?.loaded && (
          <ColoredGroupBox title={t('analysis:data.edsTitle')} color={colors.cyan}>
            <Button
              onClick={handleFetchEDS}
              disabled={edsLoading}
              title={t('analysis:data.loadEdsElementsTooltip')}
              style={{ width: '100%', marginBottom: 8, background: colors.cyan, color: colors.textOnAccent }}
            >
              {edsLoading ? t('analysis:data.fetchingEds') : t('analysis:data.loadEdsElements')}
            </Button>
            <EDSElementTable elements={edsElements} />
            {edsElements.length > 0 && (
              <Button
                onClick={handleSuggestPhases}
                disabled={edsLoading}
                title={t('analysis:data.suggestFromEdsTooltip')}
                style={{ width: '100%', marginBottom: 8, background: colors.green, color: colors.textOnAccent }}
              >
                {t('analysis:data.suggestFromEds')}
              </Button>
            )}
            {suggestMsg && (
              <div style={{ fontSize: 11, color: colors.textSecondary, marginBottom: 6 }}>{suggestMsg}</div>
            )}
            {suggestedPhases.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                {suggestedPhases.map((p, i) => (
                  <div key={i} style={{
                    fontSize: 11, color: colors.green,
                    background: alpha(colors.green, 7), border: `1px solid ${alpha(colors.green, 27)}`,
                    borderRadius: 4, padding: '3px 8px',
                  }}>
                    {p}
                  </div>
                ))}
              </div>
            )}
          </ColoredGroupBox>
        )}
      </>
    } />
  );
}

// ---------------------------------------------------------------------------
// TAB 2: Grain Analysis
// Mirrors _create_grain_tab() in analysis_gui.py
// Parameters: GB Threshold (4°), Min Grain Size (3 px), Min Intercept (1.5 µm)
// ---------------------------------------------------------------------------
function GrainTab({ analysisLoaded, hasGrains, onGrainsReconstructed, onComplete }) {
  const { t } = useTranslation('analysis');
  const [gbThreshold,  setGbThreshold]  = useState('4.0');
  const [minPixels,    setMinPixels]    = useState('3');
  const [minIntercept, setMinIntercept] = useState('1.5');

  const [loading,      setLoading]      = useState(false);
  const [taskId,       setTaskId]       = useState(null);
  const [progress,     setProgress]     = useState(null);
  const [msg,          setMsg]          = useState(null);
  const [isError,      setIsError]      = useState(false);

  const [grainStats,   setGrainStats]   = useState(null);
  const [analyzingSize,setAnalyzingSize]= useState(false);

  // Poll reconstruction task
  useEffect(() => {
    if (!taskId) return;
    const iv = setInterval(async () => {
      try {
        const res = await analysisApi.getTaskStatus(taskId);
        const d = res.data;
        if (d.progress !== undefined) setProgress(d.progress);
        if (d.status === 'done' || d.status === 'completed') {
          clearInterval(iv);
          setTaskId(null);
          setLoading(false);
          setProgress(null);
          setGrainStats({ count: d.grain_count, meanEcd: d.mean_ecd, medianEcd: d.median_ecd, stdEcd: d.std_ecd, meanAr: d.mean_aspect_ratio });
          setMsg(t('analysis:grain.reconstructedMsg', { count: d.grain_count }));
          setIsError(false);
          onGrainsReconstructed();
          onComplete('grains');
        } else if (d.status === 'error') {
          clearInterval(iv);
          setTaskId(null);
          setLoading(false);
          setProgress(null);
          setMsg(d.message || t('analysis:grain.reconstructionFailed'));
          setIsError(true);
        }
      } catch { /* keep polling */ }
    }, 1500);
    return () => clearInterval(iv);
  }, [taskId, onGrainsReconstructed, onComplete, t]);

  const handleReconstruct = async () => {
    setLoading(true);
    setProgress(0);
    setMsg(null);
    setGrainStats(null);
    try {
      const res = await analysisApi.reconstructGrains(
        parseFloat(gbThreshold),
        parseInt(minPixels, 10),
        parseFloat(minIntercept),
      );
      const d = res.data;
      if (d.task_id) {
        setTaskId(d.task_id);
        setMsg(t('analysis:grain.reconstructionRunning'));
        setIsError(false);
      } else if (d.grain_count !== undefined) {
        setLoading(false);
        setProgress(null);
        setGrainStats({ count: d.grain_count, meanEcd: d.mean_ecd, medianEcd: d.median_ecd, stdEcd: d.std_ecd, meanAr: d.mean_aspect_ratio });
        setMsg(t('analysis:grain.reconstructedMsg', { count: d.grain_count }));
        setIsError(false);
        toast.success(t('analysis:grain.reconstructionCompleteToast', { count: d.grain_count }));
        onGrainsReconstructed();
        onComplete('grains');
      }
    } catch (err) {
      setLoading(false);
      setProgress(null);
      setMsg(err.response?.data?.detail || err.message || t('analysis:grain.failed'));
      setIsError(true);
    }
  };

  const handleGrainSizeAnalysis = async () => {
    setAnalyzingSize(true);
    try {
      const res = await analysisApi.grainSize();
      const d = res.data || {};
      if (d.mean_ecd !== undefined) {
        setGrainStats(prev => ({
          ...prev,
          meanEcd:   d.mean_ecd,
          medianEcd: d.median_ecd,
          stdEcd:    d.std_ecd,
          meanAr:    d.mean_aspect_ratio,
        }));
      }
    } catch { /* ignore */ } finally {
      setAnalyzingSize(false);
    }
  };

  return (
    <TabLayout defaultLayer="ecd" analysisLoaded={analysisLoaded} controls={
      <>
        {/* Grain Reconstruction — matches recon_group with QFormLayout */}
        <ColoredGroupBox title={t('analysis:grain.reconstructionTitle')}>
          <FormRow label={t('analysis:grain.gbThreshold')}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <Input
                type="number" value={gbThreshold}
                onChange={(e) => setGbThreshold(e.target.value)}
                min={1.0} max={15.0} step={0.5}
                disabled={loading || !analysisLoaded}
                title={t('analysis:grain.gbThresholdTooltip')}
                style={{ width: 70 }}
              />
              <span style={{ fontSize: 11, color: colors.textSecondary }}>{t('analysis:grain.unitDegrees')}</span>
            </div>
          </FormRow>
          <FormRow label={t('analysis:grain.minGrainSize')}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <Input
                type="number" value={minPixels}
                onChange={(e) => setMinPixels(e.target.value)}
                min={1} max={100} step={1}
                disabled={loading || !analysisLoaded}
                title={t('analysis:grain.minGrainSizeTooltip')}
                style={{ width: 70 }}
              />
              <span style={{ fontSize: 11, color: colors.textSecondary }}>{t('analysis:grain.unitPixels')}</span>
            </div>
          </FormRow>
          <FormRow label={t('analysis:grain.minIntercept')}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <Input
                type="number" value={minIntercept}
                onChange={(e) => setMinIntercept(e.target.value)}
                min={0.1} max={10.0} step={0.1}
                disabled={loading || !analysisLoaded}
                title={t('analysis:grain.minInterceptTooltip')}
                style={{ width: 70 }}
              />
              <span style={{ fontSize: 11, color: colors.textSecondary }}>{t('analysis:grain.unitMicrometres')}</span>
            </div>
          </FormRow>
          <Button
            onClick={handleReconstruct}
            disabled={loading || !analysisLoaded}
            title={t('analysis:grain.reconstructGrainsTooltip')}
            style={{ width: '100%', marginTop: 4, background: colors.green, color: colors.textOnAccent }}
          >
            {loading ? <span className="btn-loading">{t('analysis:grain.reconstructing')}</span> : t('analysis:grain.reconstructGrains')}
          </Button>

          {loading && progress !== null && (
            <div style={{ marginTop: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                <Label secondary small>{t('analysis:grain.progress')}</Label>
                <Label secondary small>{Math.round(progress)}%</Label>
              </div>
              <ProgressBar value={progress} color={colors.green} />
            </div>
          )}
          <StatusMsg msg={msg} isError={isError} />
        </ColoredGroupBox>

        {/* Grain Statistics — matches stats_group */}
        {grainStats && (
          <ColoredGroupBox title={t('analysis:grain.statisticsTitle')}>
            <ReadonlyText
              minHeight={70}
              text={[
                grainStats.count    !== undefined ? t('analysis:grain.statTotalGrains', { count: grainStats.count.toLocaleString() }) : null,
                grainStats.meanEcd  !== undefined ? t('analysis:grain.statMeanEcd', { value: grainStats.meanEcd.toFixed(2) }) : null,
                grainStats.medianEcd !== undefined ? t('analysis:grain.statMedianEcd', { value: grainStats.medianEcd.toFixed(2) }) : null,
                grainStats.stdEcd   !== undefined ? t('analysis:grain.statStdEcd', { value: grainStats.stdEcd.toFixed(2) }) : null,
                grainStats.meanAr   !== undefined ? t('analysis:grain.statMeanAr', { value: grainStats.meanAr.toFixed(3) }) : null,
              ].filter(Boolean).join('\n')}
              placeholder={t('analysis:grain.statisticsPlaceholder')}
            />
          </ColoredGroupBox>
        )}

        {/* Grain Size Analysis — only after grains are available */}
        {hasGrains && !loading && (
          <ColoredGroupBox title={t('analysis:grain.sizeAnalysisTitle')}>
            <Button
              onClick={handleGrainSizeAnalysis}
              disabled={analyzingSize}
              title={t('analysis:grain.analyzeSizeTooltip')}
              style={{ width: '100%', background: colors.cyan, color: colors.textOnAccent }}
            >
              {analyzingSize ? t('analysis:grain.analysing') : t('analysis:grain.analyzeSize')}
            </Button>
          </ColoredGroupBox>
        )}
      </>
    } />
  );
}

// ---------------------------------------------------------------------------
// TAB 3: Deformation
// Mirrors _create_deformation_tab() in analysis_gui.py
// Parameters: KAM Order (1-3), KAM Threshold (0.55°), HAGB Threshold (10°)
// ---------------------------------------------------------------------------
function DeformationTab({ analysisLoaded, hasGrains, onComplete }) {
  const { t } = useTranslation('analysis');
  const [kamOrder,     setKamOrder]     = useState('1');
  const [kamThreshold, setKamThreshold] = useState('0.55');
  const [hagbThreshold,setHagbThreshold]= useState('10.0');

  const [loading,      setLoading]      = useState(false);
  const [gmmRunning,   setGmmRunning]   = useState(false);
  const [msg,          setMsg]          = useState(null);
  const [isError,      setIsError]      = useState(false);
  const [results,      setResults]      = useState(null);

  const handleDeformation = async () => {
    setLoading(true);
    setMsg(null);
    setResults(null);
    try {
      const res = await analysisApi.deformation();
      const d = res.data || {};
      setResults(d);
      setMsg(t('analysis:deformation.complete'));
      setIsError(false);
      onComplete('deformation');
    } catch (err) {
      setMsg(err.response?.data?.detail || t('analysis:deformation.failed'));
      setIsError(true);
    } finally {
      setLoading(false);
    }
  };

  const handleFitGMM = async () => {
    setGmmRunning(true);
    setMsg(null);
    try {
      const res = await analysisApi.bcGmm();
      const d = res.data;
      const comps = d?.components;
      if (comps) {
        const parts = [
          t('analysis:deformation.gmmLowBc', { label: comps.low_bc.label, center: comps.low_bc.center.toFixed(1), frac: (comps.low_bc.fraction * 100).toFixed(1) }),
          t('analysis:deformation.gmmMidBc', { label: comps.mid_bc.label, center: comps.mid_bc.center.toFixed(1), frac: (comps.mid_bc.fraction * 100).toFixed(1) }),
          t('analysis:deformation.gmmHighBc', { label: comps.high_bc.label, center: comps.high_bc.center.toFixed(1), frac: (comps.high_bc.fraction * 100).toFixed(1) }),
        ];
        const state = d.converged ? t('analysis:deformation.gmmConverged') : t('analysis:deformation.gmmNotConverged');
        setMsg(t('analysis:deformation.gmmFitResult', { state, parts: parts.join('\n') }));
      } else {
        setMsg(t('analysis:deformation.gmmComplete'));
      }
      setIsError(false);
    } catch (err) {
      setMsg(err.response?.data?.detail || t('analysis:deformation.gmmFailed'));
      setIsError(true);
    } finally {
      setGmmRunning(false);
    }
  };

  // Build results text to match deformation_results_text in PyQt5
  const resultsText = results ? [
    results.mean_kam  !== undefined ? t('analysis:deformation.metricMeanKam', { value: Number(results.mean_kam).toFixed(3) }) : null,
    results.hagb_length_um !== undefined ? t('analysis:deformation.metricHagbLength', { value: Number(results.hagb_length_um).toFixed(1) }) : null,
    results.sagb_length_um !== undefined ? t('analysis:deformation.metricSagbLength', { value: Number(results.sagb_length_um).toFixed(1) }) : null,
    results.sphericity !== undefined ? t('analysis:deformation.metricSphericity', { value: Number(results.sphericity).toFixed(3) }) : null,
    results.gb_classification?.hagb !== undefined
      ? t('analysis:deformation.metricHagbFraction', { value: (results.gb_classification.hagb * 100).toFixed(1) }) : null,
    results.gb_classification?.sagb !== undefined
      ? t('analysis:deformation.metricSagbFraction', { value: (results.gb_classification.sagb * 100).toFixed(1) }) : null,
  ].filter(Boolean).join('\n') : null;

  return (
    <TabLayout defaultLayer="kam" analysisLoaded={analysisLoaded} controls={
      <>
        {/* Deformation Parameters — matches params_group with QFormLayout */}
        <ColoredGroupBox title={t('analysis:deformation.parametersTitle')}>
          <FormRow label={t('analysis:deformation.kamOrder')}>
            <Input
              type="number" value={kamOrder}
              onChange={(e) => setKamOrder(e.target.value)}
              min={1} max={3} step={1}
              disabled={loading || !analysisLoaded}
              title={t('analysis:deformation.kamOrderTooltip')}
              style={{ width: 60 }}
            />
          </FormRow>
          <FormRow label={t('analysis:deformation.kamThreshold')}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <Input
                type="number" value={kamThreshold}
                onChange={(e) => setKamThreshold(e.target.value)}
                min={0.1} max={10.0} step={0.05}
                disabled={loading || !analysisLoaded}
                title={t('analysis:deformation.kamThresholdTooltip')}
                style={{ width: 70 }}
              />
              <span style={{ fontSize: 11, color: colors.textSecondary }}>{t('analysis:deformation.unitDegrees')}</span>
            </div>
          </FormRow>
          <FormRow label={t('analysis:deformation.hagbThreshold')}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <Input
                type="number" value={hagbThreshold}
                onChange={(e) => setHagbThreshold(e.target.value)}
                min={5.0} max={30.0} step={1.0}
                disabled={loading || !analysisLoaded}
                title={t('analysis:deformation.hagbThresholdTooltip')}
                style={{ width: 70 }}
              />
              <span style={{ fontSize: 11, color: colors.textSecondary }}>{t('analysis:deformation.unitDegrees')}</span>
            </div>
          </FormRow>
          <Button
            onClick={handleDeformation}
            disabled={loading || !analysisLoaded || !hasGrains}
            title={t('analysis:deformation.analyzeTooltip')}
            style={{ width: '100%', marginTop: 4, background: colors.green, color: colors.textOnAccent }}
          >
            {loading ? t('analysis:deformation.computing') : t('analysis:deformation.analyze')}
          </Button>
          <StatusMsg msg={msg} isError={isError} />
        </ColoredGroupBox>

        {/* Deformation Metrics — matches results_group */}
        <ColoredGroupBox title={t('analysis:deformation.metricsTitle')}>
          <ReadonlyText
            text={resultsText}
            placeholder={t('analysis:deformation.metricsPlaceholder')}
            minHeight={90}
          />
        </ColoredGroupBox>

        {/* BC Histogram + 3-Gaussian GMM */}
        <ColoredGroupBox title={t('analysis:deformation.gmmTitle')}>
          <div style={{ fontSize: 11, color: colors.textSecondary, marginBottom: 8 }}>
            {t('analysis:deformation.gmmDesc')}
          </div>
          <Button
            onClick={handleFitGMM}
            disabled={gmmRunning || !analysisLoaded}
            title={t('analysis:deformation.fitGmmTooltip')}
            style={{ width: '100%', background: colors.purple, color: colors.textOnAccent }}
          >
            {gmmRunning ? t('analysis:deformation.fittingGmm') : t('analysis:deformation.fitGmm')}
          </Button>
        </ColoredGroupBox>
      </>
    } />
  );
}

// ---------------------------------------------------------------------------
// TAB 4: Texture
// Mirrors _create_texture_tab() in analysis_gui.py
// Parameters: Texture Preset, Tolerance Angle (10°)
// Results: 14 components + surface planes {111}/{100}/{110} + Texture Index/Entropy
// ---------------------------------------------------------------------------
function TextureTab({ analysisLoaded, hasGrains, onComplete }) {
  const { t } = useTranslation('analysis');
  const [preset,       setPreset]       = useState('FCC_Rolling');
  const [tolerance,    setTolerance]    = useState('10.0');
  const [components,   setComponents]   = useState([]);
  const [loading,      setLoading]      = useState(false);
  const [odfRunning,   setOdfRunning]   = useState(false);
  const [msg,          setMsg]          = useState(null);
  const [isError,      setIsError]      = useState(false);
  const [textureIndex, setTextureIndex] = useState(null);
  const [entropy,      setEntropy]      = useState(null);
  const [surfacePlanes,setSurfacePlanes]= useState(null);   // { p111, p100, p110 }

  const PRESET_OPTIONS = [
    { value: 'FCC_Rolling', label: t('analysis:texture.presetFcc') },
    { value: 'BCC_Rolling', label: t('analysis:texture.presetBcc') },
  ];

  const handlePresetChange = async (e) => {
    const p = e.target.value;
    setPreset(p);
    setComponents([]);
    if (!analysisLoaded) return;
    setLoading(true);
    try {
      const res = await analysisApi.textureComponents(p);
      setComponents(res.data?.components || []);
    } catch { setComponents([]); } finally { setLoading(false); }
  };

  const handleLoadComponents = async () => {
    if (!analysisLoaded) return;
    setLoading(true);
    try {
      const res = await analysisApi.textureComponents(preset);
      setComponents(res.data?.components || []);
    } catch (err) {
      setMsg(err.response?.data?.detail || t('analysis:texture.loadComponentsFailed'));
      setIsError(true);
    } finally { setLoading(false); }
  };

  const handleODF = async () => {
    setOdfRunning(true);
    setMsg(null);
    try {
      const res = await analysisApi.texture();
      const d = res.data || {};
      if (d.components)             setComponents(d.components);
      if (d.texture_index !== undefined) setTextureIndex(d.texture_index);
      if (d.texture_entropy !== undefined) setEntropy(d.texture_entropy);
      if (d.plane_111_fraction !== undefined) {
        setSurfacePlanes({
          p111: d.plane_111_fraction,
          p100: d.plane_100_fraction,
          p110: d.plane_110_fraction,
        });
      }
      setMsg(t('analysis:texture.complete'));
      setIsError(false);
      onComplete('texture');
    } catch (err) {
      setMsg(err.response?.data?.detail || t('analysis:texture.failed'));
      setIsError(true);
    } finally { setOdfRunning(false); }
  };

  return (
    <TabLayout defaultLayer="ipf" analysisLoaded={analysisLoaded} controls={
      <>
        {/* Texture Parameters — matches params_group */}
        <ColoredGroupBox title={t('analysis:texture.parametersTitle')}>
          <FormRow label={t('analysis:texture.presetLabel')}>
            <select
              value={preset}
              onChange={handlePresetChange}
              disabled={odfRunning}
              title={t('analysis:texture.presetTooltip')}
              style={{
                background: colors.bgSecondary, border: `1px solid ${colors.border}`,
                borderRadius: 4, color: colors.text, fontSize: 12, padding: '4px 6px',
                width: '100%',
              }}
            >
              {PRESET_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </FormRow>
          <FormRow label={t('analysis:texture.toleranceAngle')}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <Input
                type="number" value={tolerance}
                onChange={(e) => setTolerance(e.target.value)}
                min={1.0} max={30.0} step={1.0}
                disabled={odfRunning || !analysisLoaded}
                title={t('analysis:texture.toleranceTooltip')}
                style={{ width: 70 }}
              />
              <span style={{ fontSize: 11, color: colors.textSecondary }}>{t('analysis:texture.unitDegrees')}</span>
            </div>
          </FormRow>
          <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
            <Button
              onClick={handleLoadComponents}
              disabled={loading || !analysisLoaded}
              title={t('analysis:texture.loadComponentsTooltip')}
              style={{ flex: 1, background: colors.cyan, color: colors.textOnAccent }}
            >
              {loading ? <span className="btn-loading">{t('analysis:texture.loading')}</span> : t('analysis:texture.loadComponents')}
            </Button>
            <Button
              onClick={handleODF}
              disabled={odfRunning || !analysisLoaded || !hasGrains}
              title={t('analysis:texture.analyzeTooltip')}
              style={{ flex: 1, background: colors.purple, color: colors.textOnAccent }}
            >
              {odfRunning ? t('analysis:texture.computing') : t('analysis:texture.analyze')}
            </Button>
          </div>
          <StatusMsg msg={msg} isError={isError} />
        </ColoredGroupBox>

        {/* Texture Metrics — Texture Index + Entropy */}
        {(textureIndex !== null || entropy !== null) && (
          <ColoredGroupBox title={t('analysis:texture.metricsTitle')}>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {textureIndex !== null && (
                <InfoBadge label={t('analysis:texture.textureIndexBadge')} value={Number(textureIndex).toFixed(3)} color={colors.purple} />
              )}
              {entropy !== null && (
                <InfoBadge label={t('analysis:texture.entropyBadge')} value={Number(entropy).toFixed(3)} color={colors.yellow} />
              )}
            </div>
          </ColoredGroupBox>
        )}

        {/* Surface Plane Fractions {111}/{100}/{110} */}
        {surfacePlanes && (
          <ColoredGroupBox title={t('analysis:texture.surfacePlanesTitle')}>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <InfoBadge label="{111}" value={`${Number(surfacePlanes.p111).toFixed(1)}%`} color={colors.cyan} />
              <InfoBadge label="{100}" value={`${Number(surfacePlanes.p100).toFixed(1)}%`} color={colors.orange} />
              <InfoBadge label="{110}" value={`${Number(surfacePlanes.p110).toFixed(1)}%`} color={colors.green} />
            </div>
          </ColoredGroupBox>
        )}

        {/* Texture Components table — 14 for FCC_Rolling */}
        {components.length > 0 && (
          <ColoredGroupBox title={t('analysis:texture.componentsTitle', { count: components.length })}>
            {/* Column headers */}
            <div style={{
              display: 'grid', gridTemplateColumns: '1fr 70px',
              background: colors.border, borderRadius: '3px 3px 0 0',
              padding: '3px 8px', marginBottom: 0,
            }}>
              <span style={{ fontSize: 10, color: colors.textSecondary, fontWeight: 700, textTransform: 'uppercase' }}>{t('analysis:texture.columnComponent')}</span>
              <span style={{ fontSize: 10, color: colors.textSecondary, fontWeight: 700, textTransform: 'uppercase', textAlign: 'right' }}>{t('analysis:texture.columnVolPct')}</span>
            </div>
            <div style={{
              maxHeight: 220, overflowY: 'auto',
              border: `1px solid ${colors.border}`, borderTop: 'none',
              borderRadius: '0 0 3px 3px', marginBottom: 4,
            }}>
              {components.map((c, i) => (
                <div key={i} className="table-row-hover" style={{
                  display: 'grid', gridTemplateColumns: '1fr 70px',
                  alignItems: 'center',
                  background: i % 2 === 0 ? colors.bgSecondary : colors.bg,
                  padding: '4px 8px', fontSize: 12,
                  borderBottom: i < components.length - 1 ? `1px solid ${colors.border}` : 'none',
                }}>
                  <span style={{ color: colors.text }}>{c.name || c}</span>
                  {c.fraction !== undefined && (
                    <span style={{ color: colors.orange, fontWeight: 600, textAlign: 'right' }}>
                      {typeof c.fraction === 'number'
                        ? (c.fraction > 1 ? c.fraction.toFixed(1) : (c.fraction * 100).toFixed(1))
                        : '—'}%
                    </span>
                  )}
                </div>
              ))}
            </div>
          </ColoredGroupBox>
        )}
      </>
    } />
  );
}

// ---------------------------------------------------------------------------
// TAB 5: Recrystallization
// Mirrors _create_rx_tab() in analysis_gui.py
// Parameters: GOS Threshold (1.25°), gBC Fraction (0.7), gKAM Threshold (0.55°), Min Grain Radius (0.4 µm)
// ---------------------------------------------------------------------------
function RXTab({ analysisLoaded, hasGrains, onComplete }) {
  const { t } = useTranslation('analysis');
  const [gosThreshold,    setGosThreshold]    = useState('1.25');
  const [gbcFraction,     setGbcFraction]     = useState('0.7');
  const [gkamThreshold,   setGkamThreshold]   = useState('0.55');
  const [minGrainRadius,  setMinGrainRadius]  = useState('0.4');

  const [loading,    setLoading]    = useState(false);
  const [msg,        setMsg]        = useState(null);
  const [isError,    setIsError]    = useState(false);
  const [rxResults,  setRxResults]  = useState(null);

  const handleRX = async () => {
    setLoading(true);
    setMsg(null);
    setRxResults(null);
    try {
      // analysisApi does not yet expose separate RX params — pass via rx analysis endpoint
      const res = await analysisApi.getMap('rx', 'RdYlBu_r');
      const d = res.data || {};
      setRxResults({
        rxFraction:       d.rx_fraction,
        highBcFraction:   d.high_bc_fraction,
        lowGkamFraction:  d.low_gkam_fraction,
        lowKamFraction:   d.low_kam_fraction,
        rxGrainCount:     d.rx_grain_count,
        totalGrains:      d.total_grains,
      });
      setMsg(t('analysis:rx.complete'));
      setIsError(false);
      onComplete('rx');
    } catch (err) {
      setMsg(err.response?.data?.detail || t('analysis:rx.failed'));
      setIsError(true);
    } finally {
      setLoading(false);
    }
  };

  // Build results text to match rx_results_text in PyQt5
  const resultsText = rxResults ? [
    rxResults.rxFraction      !== undefined ? t('analysis:rx.metricRxFraction', { value: Number(rxResults.rxFraction).toFixed(1) }) : null,
    rxResults.highBcFraction  !== undefined ? t('analysis:rx.metricHighBcFraction', { value: Number(rxResults.highBcFraction).toFixed(1) }) : null,
    rxResults.lowGkamFraction !== undefined ? t('analysis:rx.metricLowGkamFraction', { value: Number(rxResults.lowGkamFraction).toFixed(1) }) : null,
    rxResults.lowKamFraction  !== undefined ? t('analysis:rx.metricLowKamFraction', { value: Number(rxResults.lowKamFraction).toFixed(1) }) : null,
    (rxResults.rxGrainCount !== undefined && rxResults.totalGrains !== undefined)
      ? t('analysis:rx.metricRxGrains', { rx: rxResults.rxGrainCount, total: rxResults.totalGrains }) : null,
  ].filter(Boolean).join('\n') : null;

  return (
    <TabLayout defaultLayer="rx" analysisLoaded={analysisLoaded} controls={
      <>
        {/* RX Classification Parameters — matches params_group with QFormLayout */}
        <ColoredGroupBox title={t('analysis:rx.parametersTitle')}>
          <FormRow label={t('analysis:rx.gosThreshold')}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <Input
                type="number" value={gosThreshold}
                onChange={(e) => setGosThreshold(e.target.value)}
                min={0.1} max={5.0} step={0.05}
                disabled={loading || !analysisLoaded}
                title={t('analysis:rx.gosThresholdTooltip')}
                style={{ width: 70 }}
              />
              <span style={{ fontSize: 11, color: colors.textSecondary }}>{t('analysis:rx.unitDegrees')}</span>
            </div>
          </FormRow>
          <FormRow label={t('analysis:rx.gbcFraction')}>
            <Input
              type="number" value={gbcFraction}
              onChange={(e) => setGbcFraction(e.target.value)}
              min={0.1} max={1.0} step={0.05}
              disabled={loading || !analysisLoaded}
              title={t('analysis:rx.gbcFractionTooltip')}
              style={{ width: 70 }}
            />
          </FormRow>
          <FormRow label={t('analysis:rx.gkamThreshold')}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <Input
                type="number" value={gkamThreshold}
                onChange={(e) => setGkamThreshold(e.target.value)}
                min={0.1} max={2.0} step={0.05}
                disabled={loading || !analysisLoaded}
                title={t('analysis:rx.gkamThresholdTooltip')}
                style={{ width: 70 }}
              />
              <span style={{ fontSize: 11, color: colors.textSecondary }}>{t('analysis:rx.unitDegrees')}</span>
            </div>
          </FormRow>
          <FormRow label={t('analysis:rx.minGrainRadius')}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <Input
                type="number" value={minGrainRadius}
                onChange={(e) => setMinGrainRadius(e.target.value)}
                min={0.1} max={5.0} step={0.1}
                disabled={loading || !analysisLoaded}
                title={t('analysis:rx.minGrainRadiusTooltip')}
                style={{ width: 70 }}
              />
              <span style={{ fontSize: 11, color: colors.textSecondary }}>{t('analysis:rx.unitMicrometres')}</span>
            </div>
          </FormRow>
          <Button
            onClick={handleRX}
            disabled={loading || !analysisLoaded || !hasGrains}
            title={t('analysis:rx.classifyTooltip')}
            style={{ width: '100%', marginTop: 4, background: colors.green, color: colors.textOnAccent }}
          >
            {loading ? t('analysis:rx.classifying') : t('analysis:rx.classify')}
          </Button>
          <StatusMsg msg={msg} isError={isError} />
        </ColoredGroupBox>

        {/* RX Metrics — matches rx_results_text */}
        <ColoredGroupBox title={t('analysis:rx.metricsTitle')}>
          <ReadonlyText
            text={resultsText}
            placeholder={t('analysis:rx.metricsPlaceholder')}
            minHeight={90}
          />
        </ColoredGroupBox>

        {/* RX fraction badges */}
        {rxResults?.rxFraction !== undefined && (
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 10, animation: 'fadeSlideIn 0.2s ease-out' }}>
            <InfoBadge label={t('analysis:rx.badgeRxFraction')}   value={`${Number(rxResults.rxFraction).toFixed(1)}%`}   color={colors.green} />
            {rxResults.highBcFraction !== undefined && (
              <InfoBadge label={t('analysis:rx.badgeHighBc')}      value={`${Number(rxResults.highBcFraction).toFixed(1)}%`}  color={colors.cyan} />
            )}
            {rxResults.lowGkamFraction !== undefined && (
              <InfoBadge label={t('analysis:rx.badgeLowGkam')}     value={`${Number(rxResults.lowGkamFraction).toFixed(1)}%`} color={colors.orange} />
            )}
          </div>
        )}
      </>
    } />
  );
}

// ---------------------------------------------------------------------------
// TAB 6: Batch & Export
// Mirrors _create_batch_tab() in analysis_gui.py
// ---------------------------------------------------------------------------
function BatchTab({ analysisLoaded, onComplete }) {
  const { t } = useTranslation('analysis');
  const [folderPath,   setFolderPath]   = useState('');
  const [outputPath,   setOutputPath]   = useState('');
  const [datasetName,  setDatasetName]  = useState('Dataset');
  const [batchRunning, setBatchRunning] = useState(false);
  const [exportRunning,setExportRunning]= useState(false);
  const [batchTaskId,  setBatchTaskId]  = useState(null);
  const [batchProgress,setBatchProgress]= useState(null);
  const [batchLog,     setBatchLog]     = useState('');
  const [batchMsg,     setBatchMsg]     = useState(null);
  const [batchError,   setBatchError]   = useState(false);
  const [exportMsg,    setExportMsg]    = useState(null);
  const [exportError,  setExportError]  = useState(false);

  // Separate export output path — matches batch_output_edit in PyQt5
  const [exportOutputPath, setExportOutputPath] = useState('');

  // Poll batch task
  useEffect(() => {
    if (!batchTaskId) return;
    const iv = setInterval(async () => {
      try {
        const res = await analysisApi.getTaskStatus(batchTaskId);
        const d = res.data;
        if (d.progress !== undefined) setBatchProgress(d.progress);
        if (d.message) setBatchLog(prev => prev + '\n' + d.message);
        if (d.status === 'done' || d.status === 'completed') {
          clearInterval(iv);
          setBatchTaskId(null);
          setBatchRunning(false);
          setBatchMsg(d.message || t('analysis:batch.complete'));
          setBatchError(false);
          setBatchProgress(null);
          onComplete('batch');
        } else if (d.status === 'error') {
          clearInterval(iv);
          setBatchTaskId(null);
          setBatchRunning(false);
          setBatchMsg(d.message || t('analysis:batch.failedMsg'));
          setBatchError(true);
          setBatchProgress(null);
        }
      } catch { /* keep polling */ }
    }, 2000);
    return () => clearInterval(iv);
  }, [batchTaskId, onComplete, t]);

  const handleBatch = async () => {
    if (!folderPath.trim() || !outputPath.trim()) return;
    setBatchRunning(true);
    setBatchLog(t('analysis:batch.logHeader', { folder: folderPath, output: outputPath }));
    setBatchMsg(t('analysis:batch.startingMsg'));
    setBatchError(false);
    try {
      const res = await analysisApi.batch(folderPath.trim(), outputPath.trim());
      if (res.data?.task_id) {
        setBatchTaskId(res.data.task_id);
      } else {
        setBatchRunning(false);
        setBatchMsg(res.data?.message || t('analysis:batch.completeShort'));
        setBatchError(false);
        onComplete('batch');
      }
    } catch (err) {
      setBatchRunning(false);
      setBatchMsg(err.response?.data?.detail || t('analysis:batch.failedShort'));
      setBatchError(true);
    }
  };

  const handleExport = async () => {
    setExportRunning(true);
    setExportMsg(null);
    try {
      // Browser mode: trigger native file download via the streaming endpoint
      if (!window.electronAPI?.saveFile) {
        const url = analysisApi.downloadExcel(datasetName);
        window.open(url, '_blank');
        setExportMsg(t('analysis:batch.downloadStarted'));
        setExportError(false);
        return;
      }
      // Electron mode: save to a user-chosen path
      const path = exportOutputPath.trim() || outputPath.trim();
      if (!path) {
        setExportMsg(t('analysis:batch.chooseOutputFirst'));
        setExportError(true);
        return;
      }
      const res = await analysisApi.exportExcel(path, datasetName);
      setExportMsg(res.data?.message || t('analysis:batch.exportedTo', { path }));
      setExportError(false);
    } catch (err) {
      setExportMsg(err.response?.data?.detail || t('analysis:batch.exportFailed'));
      setExportError(true);
    } finally {
      setExportRunning(false);
    }
  };

  return (
    <TabLayout defaultLayer="bc" analysisLoaded={analysisLoaded} controls={
      <>
        {/* Batch Processing — matches batch_group */}
        <ColoredGroupBox title={t('analysis:batch.processingTitle')} color={colors.purple}>
          <FieldRow label={t('analysis:batch.inputFolderLabel')}>
            <div style={{ display: 'flex', gap: 6 }}>
              <Input
                value={folderPath}
                onChange={(e) => setFolderPath(e.target.value)}
                disabled={batchRunning}
                placeholder={t('analysis:batch.inputFolderPlaceholder')}
                style={{ flex: 1 }}
              />
              <BrowseBtn
                disabled={batchRunning}
                title={t('analysis:batch.browseFolderTooltip')}
                onClick={async () => {
                  if (window.electronAPI?.openFolder) {
                    const selected = await window.electronAPI.openFolder();
                    if (selected) {
                      setFolderPath(selected);
                      // Auto-suggest output file if not set (matches _browse_batch_folder in PyQt5)
                      if (!outputPath) setOutputPath(selected + '/Documentation.xlsx');
                    }
                  }
                }}
              />
            </div>
          </FieldRow>
          <FieldRow label={t('analysis:batch.outputLabel')}>
            <div style={{ display: 'flex', gap: 6 }}>
              <Input
                value={outputPath}
                onChange={(e) => setOutputPath(e.target.value)}
                disabled={batchRunning}
                placeholder={t('analysis:batch.outputPlaceholder')}
                style={{ flex: 1 }}
              />
              <BrowseBtn
                disabled={batchRunning}
                title={t('analysis:batch.browseOutputTooltip')}
                onClick={async () => {
                  if (window.electronAPI?.saveFile) {
                    const selected = await window.electronAPI.saveFile({
                      filters: [{ name: t('analysis:batch.fileFilterExcel'), extensions: ['xlsx'] }, { name: t('analysis:batch.fileFilterAll'), extensions: ['*'] }],
                    });
                    if (selected) setOutputPath(selected);
                  }
                }}
              />
            </div>
          </FieldRow>
          <Button
            onClick={handleBatch}
            disabled={batchRunning || !folderPath.trim() || !outputPath.trim()}
            title={t('analysis:batch.processFolderTooltip')}
            style={{ width: '100%', padding: '8px 14px', background: colors.purple, color: colors.textOnAccent }}
          >
            {batchRunning ? <span className="btn-loading">{t('analysis:batch.processing')}</span> : t('analysis:batch.processFolder')}
          </Button>
          {batchRunning && batchProgress !== null && (
            <div style={{ marginTop: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                <Label secondary small>{t('analysis:batch.progress')}</Label>
                <Label secondary small>{Math.round(batchProgress)}%</Label>
              </div>
              <ProgressBar value={batchProgress} color={colors.purple} />
            </div>
          )}
          <StatusMsg msg={batchMsg} isError={batchError} />

          {/* Batch processing log — matches batch_log QTextEdit */}
          {batchLog && (
            <div style={{
              marginTop: 8, padding: '6px 8px',
              background: colors.bgSecondary, border: `1px solid ${colors.border}`,
              borderRadius: 4, fontSize: 10, color: colors.textSecondary,
              fontFamily: 'monospace', whiteSpace: 'pre-wrap',
              maxHeight: 100, overflowY: 'auto',
            }}>
              {batchLog}
            </div>
          )}
        </ColoredGroupBox>

        {/* Single-file export — matches export_group */}
        <ColoredGroupBox title={t('analysis:batch.exportTitle')}>
          <div style={{ fontSize: 11, color: colors.textSecondary, marginBottom: 10 }}>
            {t('analysis:batch.exportDesc')}
          </div>
          <FormRow label={t('analysis:batch.datasetNameLabel')}>
            <Input
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              disabled={exportRunning}
            />
          </FormRow>
          <FieldRow label={t('analysis:batch.outputPathLabel')}>
            <div style={{ display: 'flex', gap: 6 }}>
              <Input
                value={exportOutputPath}
                onChange={(e) => setExportOutputPath(e.target.value)}
                disabled={exportRunning}
                placeholder={outputPath || t('analysis:batch.outputPathPlaceholder')}
                style={{ flex: 1 }}
              />
              <BrowseBtn
                disabled={exportRunning}
                title={t('analysis:batch.browseExportTooltip')}
                onClick={async () => {
                  if (window.electronAPI?.saveFile) {
                    const selected = await window.electronAPI.saveFile({
                      filters: [{ name: t('analysis:batch.fileFilterExcel'), extensions: ['xlsx'] }, { name: t('analysis:batch.fileFilterAll'), extensions: ['*'] }],
                    });
                    if (selected) setExportOutputPath(selected);
                  }
                }}
              />
            </div>
          </FieldRow>
          <Button
            onClick={handleExport}
            disabled={exportRunning || !analysisLoaded}
            title={t('analysis:batch.downloadExcelTooltip')}
            style={{ width: '100%', background: colors.green, color: colors.textOnAccent }}
          >
            {exportRunning ? <span className="btn-loading">{t('analysis:batch.exporting')}</span> : t('analysis:batch.downloadExcel')}
          </Button>
          <StatusMsg msg={exportMsg} isError={exportError} />
        </ColoredGroupBox>
      </>
    } />
  );
}

// ---------------------------------------------------------------------------
// Tab configuration (matches TABS order in analysis_gui.py)
// ---------------------------------------------------------------------------
// Tab label/tip are resolved at render time via t('analysis:tabs.*').
const TABS = [
  { id: 'data' },
  { id: 'grains' },
  { id: 'deformation' },
  { id: 'texture' },
  { id: 'rx' },
  { id: 'batch' },
];

// ---------------------------------------------------------------------------
// Main AnalysisPage
// Mirrors EBSDAnalysisPage._build_ui() — includes bottom bar with
// "Run Complete Analysis" button and status label, always visible.
// ---------------------------------------------------------------------------
export default function AnalysisPage({ onNavigate, isActive }) {
  const { t } = useTranslation('analysis');
  const [activeTab,  setActiveTab]  = useState('data');
  const [hoveredTab, setHoveredTab] = useState(null);
  const [datasetStatus, setDatasetStatus] = useState(null);

  // Step completion tracking (mirrors _step_complete dict in PyQt5)
  const [completionFlags, setCompletionFlags] = useState({
    data: false, grains: false, deformation: false, texture: false, rx: false, batch: false,
  });

  const { analysisLoaded, hasGrains, setAnalysis } = useResultStore();

  // Global status label text — mirrors status_label in PyQt5
  const [statusMsg, setStatusMsg] = useState(() => t('analysis:status.ready'));

  // "Run Complete Analysis" state
  const [runAllBusy, setRunAllBusy] = useState(false);
  const [runAllStep, setRunAllStep] = useState(0);
  const [runAllTotal, setRunAllTotal] = useState(0);

  const markComplete = useCallback((tabId) => {
    setCompletionFlags(prev => ({ ...prev, [tabId]: true }));
    const labels = {
      data:        t('analysis:status.dataLoaded'),
      grains:      t('analysis:status.grainsDone'),
      deformation: t('analysis:status.deformationDone'),
      texture:     t('analysis:status.textureDone'),
      rx:          t('analysis:status.rxDone'),
      batch:       t('analysis:status.batchDone'),
    };
    setStatusMsg(labels[tabId] || t('analysis:status.tabComplete', { tab: tabId }));
  }, [t]);

  // On mount + when page becomes active: check backend for already-loaded xmap
  useEffect(() => {
    if (!isActive) return;
    analysisApi.status().then(res => {
      const d = res.data;
      if (d?.loaded) {
        setAnalysis(d);
        setDatasetStatus(d);
        setCompletionFlags(prev => ({ ...prev, data: true, grains: d.has_grains || false }));
        const detail = d.shape?.length ? t('analysis:page.pixelsLoadedPlain', { shape: d.shape.join(' × ') }) : t('analysis:status.readyForAnalysis');
        setStatusMsg(t('analysis:status.datasetLoaded', { detail }));
      }
    }).catch(() => {});
  }, [setAnalysis, isActive, t]);

  // Batch-Dashboard handoff: clicked "Open in Analysis" after a batch run.
  // The .ang path is in sessionStorage. Load it via the public API —
  // this populates the backend _dataset and triggers the status effect above.
  //
  // Why at page level (not in DataTab): DataTab remounts inconsistently, and
  // sessionStorage only fires on mount. Hooking to isActive means the handoff
  // is picked up every time the user navigates here after a batch, not just
  // the first time in a session.
  useEffect(() => {
    if (!isActive) return;
    let handoff;
    try { handoff = sessionStorage.getItem('batch_handoff_path'); }
    catch { return; }
    if (!handoff) return;
    try {
      sessionStorage.removeItem('batch_handoff_path');
      sessionStorage.removeItem('batch_handoff_source');
    } catch { /* ignore */ }
    analysisApi.load(handoff)
      .then((res) => {
        const d = res.data;
        setAnalysis(d);
        setDatasetStatus(d);
        setCompletionFlags(prev => ({ ...prev, data: true }));
        const shape = d.shape?.length ? t('analysis:page.pixelsLoadedPlain', { shape: d.shape.join(' × ') }) : t('analysis:status.dataset');
        setStatusMsg(t('analysis:status.loadedFromBatch', { shape }));
      })
      .catch((err) => {
        const msg = err.response?.data?.detail || err.message || t('analysis:status.handoffFailed');
        setStatusMsg(t('analysis:status.batchHandoffFailed', { msg }));
      });
  }, [isActive, setAnalysis, t]);

  const handleLoaded = useCallback((data) => {
    setAnalysis(data);
    setDatasetStatus(data);
    const shape = data.shape?.length ? t('analysis:page.pixelsLoadedPlain', { shape: data.shape.join(' × ') }) : t('analysis:status.dataset');
    setStatusMsg(t('analysis:status.loadedShape', { shape }));
  }, [setAnalysis, t]);

  const handleGrainsReconstructed = useCallback(() => {
    useResultStore.setState({ hasGrains: true });
  }, []);

  // "Run Complete Analysis" — chains Grains → Deformation → Texture → RX
  // Grain reconstruction is async (returns task_id), so we poll until done.
  const handleRunAll = async () => {
    if (!analysisLoaded) return;
    setRunAllBusy(true);

    // Helper: wait for async task to complete
    const waitForTask = async (taskId) => {
      for (let i = 0; i < 300; i++) { // max 5 min (300 × 1s)
        await new Promise(r => setTimeout(r, 1000));
        const res = await analysisApi.getTaskStatus(taskId);
        if (res.data?.status === 'done') return res.data;
        if (res.data?.status === 'error') throw new Error(res.data?.error || t('analysis:runSteps.taskFailed'));
      }
      throw new Error(t('analysis:runSteps.taskTimedOut'));
    };

    const steps = [
      {
        label: t('analysis:runSteps.grainReconstruction'),
        fn: async () => {
          const res = await analysisApi.reconstructGrains(5.0, 5);
          const taskId = res.data?.task_id;
          if (taskId) await waitForTask(taskId);
        },
      },
      { label: t('analysis:runSteps.deformationAnalysis'), fn: () => analysisApi.deformation() },
      { label: t('analysis:runSteps.textureAnalysis'),     fn: () => analysisApi.texture() },
      { label: t('analysis:runSteps.rxClassification'),    fn: () => analysisApi.getMap('rx', 'RdYlBu_r') },
    ];
    setRunAllTotal(steps.length);
    for (let i = 0; i < steps.length; i++) {
      setRunAllStep(i + 1);
      setStatusMsg(t('analysis:status.runningStep', { step: i + 1, total: steps.length, label: steps[i].label }));
      try {
        await steps[i].fn();
      } catch (err) {
        setStatusMsg(t('analysis:status.failedAtStep', { step: i + 1, total: steps.length, label: steps[i].label, error: err.message || '' }));
        setRunAllBusy(false);
        setRunAllStep(0);
        return;
      }
    }
    setCompletionFlags({ data: true, grains: true, deformation: true, texture: true, rx: true, batch: false });
    useResultStore.setState({ hasGrains: true });
    setStatusMsg(t('analysis:status.completeFinished'));
    setRunAllBusy(false);
    setRunAllStep(0);
  };

  return (
    <div style={{
      height:         '100%',
      display:        'flex',
      flexDirection:  'column',
      color:          colors.text,
      fontFamily:     "'Segoe UI', system-ui, sans-serif",
      gap:            0,
    }}>
      {/* Page header — mirrors header + subtitle labels in _build_ui() */}
      <div style={{ flexShrink: 0, marginBottom: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <h1 style={{ fontSize: '18pt', fontWeight: 700, color: colors.accent, margin: 0 }}>
            {t('analysis:page.title')}
          </h1>
          {analysisLoaded && (() => {
            const done = Object.values(completionFlags).filter(Boolean).length;
            const total = Object.keys(completionFlags).length;
            return done > 0 ? (
              <span style={{
                fontSize: '8pt', fontWeight: 600, padding: '2px 8px', borderRadius: 10,
                background: done === total ? `${colors.green}1a` : `${colors.orange}1a`,
                color: done === total ? colors.green : colors.orange,
              }}>
                {t('analysis:page.doneBadge', { done, total })}
              </span>
            ) : null;
          })()}
        </div>
        <p style={{ fontSize: '10pt', color: colors.textSecondary, margin: '3px 0 0' }}>
          {t('analysis:page.subtitle')}
          {analysisLoaded && datasetStatus?.shape?.length > 0 && (
            <span style={{ color: colors.cyan, marginLeft: 8 }}>
              {t('analysis:page.pixelsLoaded', { shape: datasetStatus.shape.join(' × ') })}
            </span>
          )}
        </p>
      </div>

      {/* Empty state guidance when no data is loaded */}
      {!analysisLoaded && (
        <div style={{
          background: `${colors.accent}11`, border: `1px solid ${colors.accent}33`,
          borderRadius: 6, padding: '14px 18px', marginBottom: 10,
          fontSize: '10pt', color: colors.textSecondary,
          display: 'flex', alignItems: 'center', gap: 12,
          animation: 'pageFadeIn 0.3s ease-out',
        }}>
          <span style={{ fontSize: '22pt', opacity: 0.6, flexShrink: 0 }}>{'\u2261'}</span>
          <div>
            <strong style={{ color: colors.accent }}>{t('analysis:page.emptyTitle')}</strong>{' '}
            {t('analysis:page.emptyHint')}
          </div>
        </div>
      )}

      {/* Tab bar — mirrors QTabWidget styling */}
      <div style={{
        display:       'flex',
        gap:           2,
        borderBottom:  `2px solid ${colors.border}`,
        marginBottom:  10,
        flexShrink:    0,
        flexWrap:      'wrap',
      }}>
        {TABS.map(tab => {
          const isActive  = activeTab === tab.id;
          const isHovered = hoveredTab === tab.id;
          const isDone    = completionFlags[tab.id];
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              onMouseEnter={() => setHoveredTab(tab.id)}
              onMouseLeave={() => setHoveredTab(null)}
              title={t(`analysis:tabs.${tab.id}Tip`)}
              style={{
                background:  isActive ? colors.purple : isHovered ? colors.border : 'transparent',
                border:      'none',
                borderRadius: '4px 4px 0 0',
                color:       isActive ? colors.bg : isHovered ? colors.text : colors.textSecondary,
                cursor:      'pointer',
                fontSize:    11,
                fontWeight:  isActive ? 700 : 400,
                padding:     '7px 14px',
                transition:  'all 0.12s',
                display:     'flex',
                alignItems:  'center',
                gap:          4,
                boxShadow:   isActive ? `0 2px 0 ${colors.purple}` : 'none',
                position:    'relative',
              }}
            >
              {isDone && (
                <span style={{ color: isActive ? colors.bg : colors.green, fontSize: 12 }} title={t('analysis:tabs.completedTooltip', { label: t(`analysis:tabs.${tab.id}`) })}>
                  ✓
                </span>
              )}
              {t(`analysis:tabs.${tab.id}`)}
            </button>
          );
        })}
      </div>

      {/* Tab content — flex: 1 to fill remaining height */}
      <div style={{ flex: 1, overflow: 'hidden' }}>
        {activeTab === 'data' && (
          <DataTab
            status={datasetStatus}
            onLoad={handleLoaded}
            onComplete={markComplete}
            analysisLoaded={analysisLoaded}
          />
        )}
        {activeTab === 'grains' && (
          <GrainTab
            analysisLoaded={analysisLoaded}
            hasGrains={hasGrains}
            onGrainsReconstructed={handleGrainsReconstructed}
            onComplete={markComplete}
          />
        )}
        {activeTab === 'deformation' && (
          <DeformationTab
            analysisLoaded={analysisLoaded}
            hasGrains={hasGrains}
            onComplete={markComplete}
          />
        )}
        {activeTab === 'texture' && (
          <TextureTab
            analysisLoaded={analysisLoaded}
            hasGrains={hasGrains}
            onComplete={markComplete}
          />
        )}
        {activeTab === 'rx' && (
          <RXTab
            analysisLoaded={analysisLoaded}
            hasGrains={hasGrains}
            onComplete={markComplete}
          />
        )}
        {activeTab === 'batch' && (
          <BatchTab
            analysisLoaded={analysisLoaded}
            onComplete={markComplete}
          />
        )}
      </div>

      {/* Bottom bar — mirrors bottom_bar layout in _build_ui() */}
      {/* "Run Complete Analysis" button + status label, always visible */}
      <div style={{
        display:       'flex',
        gap:           10,
        alignItems:    'center',
        flexShrink:    0,
        paddingTop:    8,
        borderTop:     `1px solid ${colors.border}`,
        marginTop:     6,
      }}>
        <Button
          onClick={handleRunAll}
          disabled={runAllBusy || !analysisLoaded}
          style={{ flexShrink: 0, background: colors.pink, color: colors.textOnAccent }}
          title={t('analysis:page.runCompleteTooltip')}
        >
          {runAllBusy ? t('analysis:page.runCompleteBusy', { step: runAllStep, total: runAllTotal }) : t('analysis:page.runComplete')}
        </Button>
        {/* Step completion counter with mini progress */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 4, flexShrink: 0 }}>
          <div style={{
            width: 40, height: 4, borderRadius: 2, background: colors.border, overflow: 'hidden',
          }}>
            <div style={{
              width: `${(Object.values(completionFlags).filter(Boolean).length / Object.keys(completionFlags).length) * 100}%`,
              height: '100%', borderRadius: 2, background: colors.green,
              transition: 'width 0.3s',
            }} />
          </div>
          <span style={{ fontSize: 10, color: colors.textSecondary, fontWeight: 600 }}>
            {Object.values(completionFlags).filter(Boolean).length}/{Object.keys(completionFlags).length}
          </span>
        </div>
        <div style={{
          flex:         1,
          padding:      '6px 10px',
          background:   colors.bgSecondary,
          borderRadius:  4,
          fontSize:      11,
          color:         colors.textSecondary,
        }}>
          {statusMsg}
        </div>
      </div>
    </div>
  );
}
