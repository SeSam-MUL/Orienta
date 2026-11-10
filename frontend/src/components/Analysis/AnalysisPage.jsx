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

import { useState, useEffect, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { analysisApi, indexApi, edsApi } from '../../services/api';
import useResultStore, { EMPTY_BATCH_RUN, EMPTY_EXPORT_RUN } from '../../stores/useResultStore';
import useDataStore from '../../stores/useDataStore';
import { toast } from '../../stores/useToastStore';
import { codedProblemText } from '../../services/problemText';
import {
  grainStatsFrom, mergeGrainSize, edgeSummary, reconstructArgs, batchReconstruction, taskOutcome, pollErrorOutcome,
  errorMessageFrom, statsPhaseLabel, warningsFromHeader, samePath, progressPercent, DEFAULT_GRAIN_STATS_SETTINGS,
  datasetNameFromFile, batchStatsFromGrain,
} from './grainStats';
import GrainStatsSettings from './GrainStatsSettings';
import CitationsPanel from '../PhaseMap/CitationsPanel';
import { MAX_TEXTURE_TOLERANCE, textureBatchArgs, textureToleranceOf } from './textureSettings';
import { rxCriteriaArgs, rxCriteriaLabels } from './rxCriteria';
import { deformationArgs, deformationBody } from './deformationSettings';
import { useReconstructionWatch } from './useReconstructionWatch';
import {
  colors, alpha, alpha as colorAlpha,   // GrainTab has a state called `alpha`
  Button,
  Label,
  Input,
  FormRow,
  ProgressBar,
} from '../../theme/components';

// A refusal with a stable code (a pattern source a step cannot read) in the reader's language;
// null for anything else, so each caller keeps the sentence it showed before.
const codedError = (err, t) => codedProblemText(err, t, 'analysis');

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

function StatusMsg({ msg, isError, isWarn = false }) {
  if (!msg) return null;
  const tone = isError ? colors.red : isWarn ? colors.orange : colors.green;
  return (
    <div role={isError ? 'alert' : 'status'} style={{
      marginTop:   8,
      padding:     '5px 10px',
      borderRadius: 4,
      background:  isError ? alpha(colors.red, 13) : isWarn ? alpha(colors.orange, 13) : alpha(colors.green, 8),
      border:      `1px solid ${tone}`,
      color:       tone,
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

// The map types whose route raises "Reconstruct grains first" (routes/analysis.py get_map):
// they cannot be drawn before grains exist, so they are not requested then.
const GRAIN_SIZE_DEBOUNCE_MS = 250;
const GRAIN_LAYERS = ['gos', 'ecd', 'rx', 'grain_boundaries'];

/**
 * Ask the backend what it holds (GET /status) and hand the answer to the store. An answer that was overtaken
 * by a reconstruction or a load made here while it was on its way is older than what the page knows and is
 * not applied (`applied: false`). Throws when the request fails.
 */
async function syncFromBackend() {
  const { grainsVersion: versionAsked, grainSizeEpoch: epochAsked, hasGrains: hadGrains } = useResultStore.getState();
  const res = await analysisApi.status();
  const d = res.data;
  if (!d || typeof d !== 'object') return null;
  const now = useResultStore.getState();
  if (now.grainsVersion !== versionAsked || now.grainSizeEpoch !== epochAsked) return { d, applied: false };
  now.applyBackendStatus(d);
  // Grains that appeared unseen (a reconstruction that finished while nobody looked, a map with grains loaded
  // elsewhere) come without statistics: ask for them once, with the settings the page shows.
  if (d.loaded && d.has_grains && !hadGrains) fetchStatisticsOnce();
  return { d, applied: true };
}

// A coded refusal (a pattern source a step cannot read) in the reader's language first, as the page's own
// handlers do; otherwise the backend's sentence, then the fallback under `fallbackKey`.
async function refusedRequest(err, t, fallbackKey) {
  if (err?.response?.status === 409) {
    try { await syncFromBackend(); } catch { /* the message below says enough */ }
  }
  return new Error(codedError(err, t) || errorMessageFrom(err, t(fallbackKey)));
}

/**
 * The Deformation parameters that are typed and not run yet (`deformationEdited`), sent with a deformation run: the RX
 * analysis's KAM and the workbook follow them. Nothing pending, or no grains to work on: nothing is sent and null comes
 * back; else the numbers that were sent. Throws a readable message when they are not valid or the backend refuses.
 */
async function applyPendingDeformation(t) {
  const store = useResultStore.getState();
  if (!store.hasGrains || !store.deformationEdited) return null;
  const args = deformationArgs(store.deformationParams);
  if (!args) throw new Error(t('analysis:deformation.invalidParams'));
  try {
    const res = await analysisApi.deformation(args);
    useResultStore.getState().adoptDeformationUsed(res.data?.deformation_params ?? deformationBody(args));
    useResultStore.getState().bumpGrainMaps();
    return args;
  } catch (err) {
    throw await refusedRequest(err, t, 'analysis:deformation.failed');
  }
}

/**
 * The settings the user chose on the Deformation, Texture and Recrystallization tabs and did not send yet
 * (`deformationEdited`, `textureEdited`, `rxCriteriaEdited`) are the current choice, so an output of the open map must
 * use them: they are sent first, in the order the analyses depend on each other (the RX analysis's KAM uses the
 * Deformation settings). Nothing pending: nothing is sent; the Deformation and RX settings also wait for grains to work on, the texture does not need any.
 * Returns what was sent, `{ deformation, texture, rx }` (each null when nothing was sent): numbers for deformation and
 * RX, `{ preset, tolerance, phase }` for texture. Throws, with a readable message, when a pending setting is not valid or
 * the backend refuses; a 409 (the map changed meanwhile) asks the backend what it holds first.
 */
async function applyTypedSettings(t) {
  const applied = { deformation: null, texture: null, rx: null };
  // the texture needs no grains; the Deformation and RX analyses work on them
  const hasGrains = useResultStore.getState().hasGrains;
  const refused = refusedRequest;
  applied.deformation = await applyPendingDeformation(t);
  if (useResultStore.getState().textureEdited) {
    const { texturePhase: phase, texturePreset: preset, textureTolerance: tolerance } = useResultStore.getState();
    const toleranceDeg = textureToleranceOf(tolerance);
    if (toleranceDeg === undefined) throw new Error(t('analysis:texture.toleranceInvalid'));
    try {
      await analysisApi.texture(phase, preset, toleranceDeg);
      useResultStore.getState().markTextureApplied({ phase, preset, tolerance });
      useResultStore.getState().bumpGrainMaps();
      applied.texture = { preset, tolerance: toleranceDeg, phase };
    } catch (err) {
      throw await refused(err, t, 'analysis:texture.failed');
    }
  }
  if (hasGrains && useResultStore.getState().rxCriteriaEdited) {
    const args = rxCriteriaArgs(useResultStore.getState().rxCriteria);
    if (!args) throw new Error(t('analysis:rx.invalidCriteria'));
    try {
      const res = await analysisApi.rxClassify(args, useResultStore.getState().rxPhase);
      useResultStore.getState().adoptRxCriteriaUsed(res.data);
      useResultStore.getState().bumpGrainMaps();
      applied.rx = args;
    } catch (err) {
      throw await refused(err, t, 'analysis:rx.failed');
    }
  }
  return applied;
}

async function fetchStatisticsOnce() {
  const store = useResultStore.getState();
  if (store.grainStats !== null || store.runAllBusy || store.reconTaskId !== null) return;
  const epoch = store.grainSizeEpoch;
  const { edgeRule, ecdBasis, statsPhase } = store.grainStatsSettings;
  try {
    const res = await analysisApi.grainSize(edgeRule, ecdBasis, statsPhase ?? null);
    const d = res?.data;
    const now = useResultStore.getState();
    if (!d || d.mean_ecd === undefined || now.grainSizeEpoch !== epoch || now.grainStats !== null) return;
    now.setGrainStats(mergeGrainSize(null, d));
  } catch { /* the statistics stay empty; "Analyze Grain Size" asks again */ }
}

// The fixed colours of the RX map (routes/analysis.py _RX_CLASS_COLOURS): red is "recrystallized by the RX analysis"
// and nothing else; the other colours are the GOS bands of the remaining grains.
const RX_LEGEND_COLOURS = {
  rx:       'rgb(215, 48, 39)',
  lowGos:   'rgb(166, 206, 227)',
  sub:      'rgb(254, 224, 139)',
  deformed: 'rgb(140, 86, 40)',
  noGrain:  'rgb(128, 128, 128)',
};

// The colour scale under a scalar map: every such map is drawn with its colormap stretched between the smallest and
// the largest value, so the bar says which values the two ends are (`range` = { min, max, unit, stops, basis }).
// Numbers are shown with three significant digits.
function formatSig3(value, language) {
  try {
    return new Intl.NumberFormat(language, { maximumSignificantDigits: 3, useGrouping: false }).format(value);
  } catch {
    return String(Number(Number(value).toPrecision(3)));
  }
}

function MapScaleBar({ range, layer }) {
  const { t, i18n } = useTranslation('analysis');
  if (!range) return null;
  const min = formatSig3(range.min, i18n.language);
  const max = formatSig3(range.max, i18n.language);
  const unit = range.unit || '';
  const line = (layer === 'ecd' && range.basis
    ? t('analysis:map.rangeEcd', { basis: t(`analysis:grain.ecdBasisShort.${range.basis}`), min, max, unit })
    : t('analysis:map.range', { min, max, unit })).trim();
  return (
    <div data-testid="map-scale" style={{ padding: '4px 10px 2px', flexShrink: 0 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <span data-testid="map-scale-min" style={{ fontSize: 10, color: colors.textSecondary }}>{min}</span>
        <div
          role="img"
          aria-label={t('analysis:map.rangeBarLabel')}
          data-testid="map-scale-bar"
          style={{
            flex: 1, height: 8, borderRadius: 2, border: `1px solid ${colors.border}`,
            background: `linear-gradient(to right, ${range.stops.join(', ')})`,
          }}
        />
        <span data-testid="map-scale-max" style={{ fontSize: 10, color: colors.textSecondary }}>{`${max} ${unit}`.trim()}</span>
      </div>
      <div data-testid="map-scale-line" style={{ fontSize: 10, color: colors.textSecondary, textAlign: 'center', marginTop: 2 }}>
        {line}
      </div>
    </div>
  );
}

/** What the colours of the RX map mean. ``rx`` is the answer of the map route: ``rx_drawn`` (red is drawn),
 * ``rx_needs_bc`` (the map has no native band contrast, so the analysis did not run). */
function RxLegend({ rx }) {
  const { t } = useTranslation('analysis');
  const drawn = rx?.rx_drawn === true;
  const rows = [
    drawn && ['rx', 'legendRx'],
    ['lowGos', drawn ? 'legendLowGos' : 'legendLowGosOnly'],
    ['sub', 'legendSub'],
    ['deformed', 'legendDeformed'],
    ['noGrain', 'legendNoGrain'],
  ].filter(Boolean);
  return (
    <div data-testid="rx-legend" style={{
      padding: '6px 10px', background: colors.bgSecondary, border: `1px solid ${colors.border}`,
      borderTop: 'none', borderRadius: '0 0 5px 5px', fontSize: 11, color: colors.textSecondary,
    }}>
      <div style={{ fontWeight: 700, color: colors.purple, marginBottom: 3 }}>{t('analysis:rx.legendTitle')}</div>
      {!drawn && (
        <div data-testid="rx-legend-no-red" style={{ color: colors.orange, fontWeight: 600, marginBottom: 3 }}>
          {rx?.rx_needs_bc ? t('analysis:rx.legendNoRedBc') : t('analysis:rx.legendNoRedFailed')}
        </div>
      )}
      <div style={{ display: 'flex', flexWrap: 'wrap', columnGap: 14, rowGap: 2 }}>
        {rows.map(([key, text]) => (
          <span key={key} style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
            <span style={{ width: 11, height: 11, background: RX_LEGEND_COLOURS[key], border: `1px solid ${colors.border}`, flexShrink: 0 }} />
            {t(`analysis:rx.${text}`)}
          </span>
        ))}
      </div>
      {drawn && <div style={{ marginTop: 3 }}>{t('analysis:rx.legendBands')}</div>}
    </div>
  );
}

/** What the colours of the Texture Components map mean. ``info`` is the answer of the map route: ``legend``
 * ([{ name, color }], one colour per component), the colours of "no component" and "not of the texture phase", and
 * ``texture_phase`` ({ id, name }). */
function TextureLegend({ info }) {
  const { t } = useTranslation('analysis');
  const rows = [
    ...(info.legend || []).map((e) => [e.name, e.color, e.name]),
    ['none', info.no_component_color, t('analysis:map.textureLegendNone')],
    ['outside', info.outside_color, t('analysis:map.textureLegendOutside')],
  ];
  return (
    <div data-testid="texture-legend" style={{
      padding: '6px 10px', background: colors.bgSecondary, border: `1px solid ${colors.border}`,
      borderTop: 'none', borderRadius: '0 0 5px 5px', fontSize: 11, color: colors.textSecondary,
    }}>
      <div style={{ fontWeight: 700, color: colors.purple, marginBottom: 3 }}>
        {t('analysis:map.textureLegendTitle', {
          phase: info.texture_phase?.name ?? '', preset: info.texture_preset ?? '', tolerance: info.texture_tolerance_deg ?? '',
        })}
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', columnGap: 14, rowGap: 2 }}>
        {rows.map(([key, color, text]) => (
          <span key={key} data-testid={`texture-legend-${key}`} style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
            <span style={{ width: 11, height: 11, background: color, border: `1px solid ${colors.border}`, flexShrink: 0 }} />
            {text}
          </span>
        ))}
      </div>
    </div>
  );
}

function MapViewer({ defaultLayer = 'bc', analysisLoaded, hasGrains = false, grainsVersion = 0, style }) {
  const { t } = useTranslation(['analysis', 'common']);
  const layerLabel = (v) => t(`analysis:map.layers.${v}`);
  const layerTip   = (v) => t(`analysis:map.layers.${v}Tip`);
  // A map without band contrast of its own cannot draw the BC layer (the backend answers 400): the tabs whose
  // default is BC start on IPF-Z then. What the user picks stays; null = nothing picked yet.
  const noBc = useResultStore((state) => state.hasNativeBc) === false;
  const [picked, setPicked] = useState(null);
  const layer = picked ?? (defaultLayer === 'bc' && noBc ? 'ipf' : defaultLayer);
  const [cmap, setCmap]     = useState('gray');
  const [image, setImage]   = useState(null);
  const [rxInfo, setRxInfo] = useState(null);     // what the RX map says about its red (rx_drawn, rx_needs_bc)
  const [textureInfo, setTextureInfo] = useState(null);   // the legend of the Texture Components map
  const [range, setRange]   = useState(null);     // the values the scalar map's colours stand for (value_range, colormap_stops)
  const [loading, setLoading] = useState(false);
  const [info, setInfo]     = useState(t('analysis:map.noMapLoaded'));

  // IPF colours are a code for crystal directions, not a value range: the
  // backend ignores `cmap` for these layers entirely. Offering the selector
  // anyway suggests an effect that does not exist — every reviewer of the
  // user report called that out first.
  const isOrientationLayer = layer.startsWith('ipf');
  // the RX map has fixed colours as well: red is "RX by the analysis", a colormap would paint over it
  const hasFixedColours = isOrientationLayer || layer === 'rx' || layer === 'texture_component';
  const [error, setError] = useState(false);
  const needsGrains = GRAIN_LAYERS.includes(layer) && !hasGrains;
  const requestId = useRef(0);   // a slow answer for an older request must not overwrite a newer one

  const handleRefresh = useCallback(async () => {
    if (!analysisLoaded) return;
    const mine = ++requestId.current;
    if (needsGrains) {
      // Neutral hint, no request: the backend would answer 400 and the red line would stay.
      setImage(null);
      setRxInfo(null);
      setRange(null);
      setTextureInfo(null);
      setError(false);
      setLoading(false);
      setInfo(t('analysis:map.needsGrainsHint'));
      return;
    }
    setLoading(true);
    // the "needs grains" hint must not stay on the line while the map is on its way
    setInfo((prev) => (prev === t('analysis:map.needsGrainsHint') ? t('analysis:map.loading') : prev));
    try {
      const res = await analysisApi.getMap(layer, cmap);
      if (mine !== requestId.current) return;
      const img = res.data?.image || null;
      setImage(img);
      setRxInfo(layer === 'rx' && img ? res.data : null);
      setTextureInfo(layer === 'texture_component' && img ? res.data : null);
      const vr = res.data?.value_range;
      setRange(img && vr && Array.isArray(res.data?.colormap_stops) && res.data.colormap_stops.length > 1
        ? { ...vr, stops: res.data.colormap_stops, basis: res.data.ecd_basis ?? null }
        : null);
      setError(!img);
      setInfo(img
        ? (hasFixedColours ? layerLabel(layer) : `${layerLabel(layer)} | ${cmap}`)
        : t('analysis:map.noDataReturned'));
    } catch (err) {
      if (mine !== requestId.current) return;
      // The backend says exactly what is missing ("Reconstruct grains first
      // for grain boundary map"). Show it, and show it as a failure.
      setImage(null);
      setRxInfo(null);
      setRange(null);
      setTextureInfo(null);
      setError(true);
      setInfo(codedError(err, t) || errorMessageFrom(err, t('analysis:map.failedToLoad')));
    } finally {
      if (mine === requestId.current) setLoading(false);
    }
  }, [analysisLoaded, layer, cmap, needsGrains, hasFixedColours, t]);

  // Changing the layer used to do nothing until "Refresh" was pressed — the
  // reported "grain boundaries draws nothing, it just does nothing at all".
  // Picking a layer IS the request to see it. A finished reconstruction (grainsVersion)
  // replaces the grains under every layer, so every layer is fetched again.
  useEffect(() => {
    if (!analysisLoaded) return;
    handleRefresh();
  }, [layer, cmap, analysisLoaded, handleRefresh, grainsVersion]);

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
            onChange={(e) => setPicked(e.target.value)}
            aria-label={t('analysis:map.layerLabel')}
            title={layerTip(layer)}
            style={{
              background: colors.bg, border: `1px solid ${colors.border}`, borderRadius: 3,
              color: colors.text, fontSize: 11, padding: '2px 4px', cursor: 'pointer',
            }}
          >
            {MAP_LAYER_OPTIONS.map(o => (
              <option key={o.value} value={o.value} title={layerTip(o.value)}>
                {o.value === 'bc' && noBc ? t('analysis:map.layers.bcMissing') : layerLabel(o.value)}
              </option>
            ))}
          </select>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{
            fontSize: 10,
            color: colors.textSecondary,
            opacity: hasFixedColours ? 0.5 : 1,
          }}>{t('analysis:map.colormapLabel')}</span>
          <select
            value={cmap}
            disabled={hasFixedColours}
            onChange={(e) => setCmap(e.target.value)}
            aria-label={t('analysis:map.colormapLabel')}
            title={isOrientationLayer
              ? t('analysis:map.colormapIpfTooltip')
              : layer === 'rx' ? t('analysis:map.colormapRxTooltip')
                : layer === 'texture_component' ? t('analysis:map.colormapTextureTooltip') : t('analysis:map.colormapTooltip')}
            style={{
              background: colors.bg, border: `1px solid ${colors.border}`, borderRadius: 3,
              color: colors.text, fontSize: 11, padding: '2px 4px',
              cursor: hasFixedColours ? 'not-allowed' : 'pointer',
              opacity: hasFixedColours ? 0.5 : 1,
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
        {/* A failure must not read like a status line: the backend names the
            missing step, and that sentence is the whole answer to "it does
            nothing". */}
        <span style={{
          fontSize: 10, marginLeft: 'auto', maxWidth: '46%', textAlign: 'right',
          color: error ? colors.red : colors.textSecondary,
          fontWeight: error ? 600 : 400,
        }}>
          {error ? `⚠ ${info}` : info}
        </span>
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
            {!analysisLoaded
              ? t('analysis:map.loadDatasetFirst')
              : needsGrains ? t('analysis:map.needsGrainsHint') : t('analysis:map.selectLayerHint')}
          </span>
        )}
      </div>
      {range && image && <MapScaleBar range={range} layer={layer} />}
      {layer === 'rx' && rxInfo && <RxLegend rx={rxInfo} />}
      {layer === 'texture_component' && textureInfo && <TextureLegend info={textureInfo} />}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Per-tab layout: left = scrollable controls (300 px), right = map viewer
// ---------------------------------------------------------------------------
function TabLayout({ controls, defaultLayer = 'bc', analysisLoaded, hasGrains = false, grainsVersion = 0 }) {
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
      <MapViewer
        defaultLayer={defaultLayer}
        analysisLoaded={analysisLoaded}
        hasGrains={hasGrains}
        grainsVersion={grainsVersion}
      />
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
/** A file the Analysis page can read as a CrystalMap (what its own Browse filter offers). */
export function isCrystalMapPath(p) {
  return typeof p === 'string' && /\.(h5oina|h5|hdf5|ang|ctf)$/i.test(p.trim());
}

const fileNameOf = (p) => String(p ?? '').split(/[\\/]/).filter(Boolean).pop() || '';

function DataTab({ status, onLoad, onComplete, analysisLoaded, hasGrains, grainsVersion }) {
  const { t } = useTranslation(['analysis', 'common']);
  const [xmapPath,         setXmapPath]         = useState('');
  const [loading,          setLoading]           = useState(false);
  const [msg,              setMsg]               = useState(null);
  const [isError,          setIsError]           = useState(false);
  // The path this tab last loaded, to tell whether the viewer's file is already the one analysed.
  const [loadedPath,       setLoadedPath]        = useState('');

  // The file open in the EBSD Viewer. The page used to ignore it: a user who had just opened a
  // scan there found "No indexed data loaded" and had to browse for the same file again
  // It is offered, not loaded on its own: after an indexing run
  // the map to analyse may be the result, not the vendor's map in that file.
  const viewerFile = useDataStore((s) => s.filePath);
  const viewerFileUsable = isCrystalMapPath(viewerFile);
  const viewerName = viewerFileUsable ? fileNameOf(viewerFile) : '';
  // A map is here when the page holds one: the store's flag, set by a load (whose answer says `success`,
  // not `loaded`) and by GET /status alike.
  const mapHere = Boolean(analysisLoaded);
  // Is the map here the viewer's file? The file name the backend reports for the map (`source_name`,
  // from /status and from a load) is the authority: the map also changes without this tab (Batch
  // "Open in Analysis", a load on another page). The full path this tab loaded only refines it
  // (a file of the same name in another folder), and only while it still names that map. A map
  // without a file name (an indexing result held in memory) is not the viewer's file.
  const sourceName = status?.source_name || '';
  const windowsNames = /^[A-Za-z]:[\\/]/.test(viewerFile || '') || (viewerFile || '').includes('\\');
  const sameName = (a, b) => (windowsNames ? a.toLowerCase() === b.toLowerCase() : a === b);
  const loadedPathCurrent = Boolean(loadedPath && sourceName && sameName(fileNameOf(loadedPath), sourceName));
  const sameAsViewer = loadedPathCurrent
    ? samePath(loadedPath, viewerFile)
    : Boolean(sourceName) && sameName(sourceName, viewerName);
  const offerViewerFile = viewerFileUsable && (!mapHere || !sameAsViewer);

  useEffect(() => {
    if (viewerFileUsable) setXmapPath((current) => (current.trim() ? current : viewerFile));
  }, [viewerFile, viewerFileUsable]);

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
      setLoadedPath(isCrystalMapPath(path) ? path : '');   // the session selector loads '__last_indexing__'
      onLoad(d);
      const shape = d.shape?.length ? d.shape.join(' × ') : t('analysis:data.unknownShape');
      const phases = d.phases ? t('analysis:data.loadedPhasesSuffix', { count: d.phases.length }) : '';
      setMsg(t('analysis:data.loadedMsg', { shape, phases }));
      setIsError(false);
      onComplete('data');
    } catch (err) {
      setMsg(codedError(err, t) || errorMessageFrom(err, t('analysis:data.loadFailed')));
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
      setAzCmpError(errorMessageFrom(err, t('analysis:data.comparisonFailed')));
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
      // grains made before are still the old ones (the backend leaves them): say so, and draw the maps again (the indexed
      // pixels changed)
      const grainsOld = useResultStore.getState().hasGrains;
      useResultStore.getState().bumpGrainMaps();
      setQfMsg(t('analysis:data.qualityFilterResult', {
        total: d.filtered_total,
        pct: pct.toFixed(2),
        byBc: d.filtered_by_bc,
        byBands: d.filtered_by_bands,
        after: d.indexed_after,
        before: d.indexed_before,
      }) + (grainsOld ? ` ${t('analysis:data.qualityFilterGrainsNote')}` : ''));
      setQfMsgError(false);
    } catch (err) {
      setQfMsg(codedError(err, t) || errorMessageFrom(err, t('analysis:data.qualityFilterFailed')));
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
      setSuggestMsg(errorMessageFrom(err, t('analysis:data.suggestionFailed')));
    }
  };

  return (
    <TabLayout defaultLayer="bc" analysisLoaded={analysisLoaded} hasGrains={hasGrains} grainsVersion={grainsVersion} controls={
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
          {offerViewerFile && (
            <Button
              onClick={() => { setXmapPath(viewerFile); handleLoad(viewerFile); }}
              disabled={loading}
              title={viewerFile}
              // A scan's name is long: cut with an ellipsis inside the button, the full path is the tooltip.
              // The ellipsis needs a block of its own: Chromium clips a <button>'s text without one.
              // Button is a one-row inline-flex of fixed height: a column of auto height puts the name below.
              style={{
                width: '100%', marginBottom: 8, textAlign: 'left',
                flexDirection: 'column', alignItems: 'stretch', height: 'auto', gap: 3, lineHeight: 1.3,
              }}
            >
              <span style={{ display: 'block' }}>{t('analysis:data.useViewerFile')}</span>
              {/* the name on a line of its own: after the label it was cut away entirely */}
              <span style={{
                display: 'block', minWidth: 0, fontSize: 11, opacity: 0.85,
                whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
              }}>
                {viewerName}
              </span>
            </Button>
          )}
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
                    aria-label={t('analysis:data.minBandContrast')}
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
                    aria-label={t('analysis:data.minBands')}
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
// Parameters: GB Threshold (4°), Min Grain Size (3 px), Min Intercept (1.5 µm),
// Close holes alpha (3.1 px, MTEX default), Fill holes in maps (on)
// ---------------------------------------------------------------------------
// `statsSettings` (the reconstruction parameters, `edgeRule`, `ecdBasis`) and `onStatsSettingsChange(patch)`
// are held by the page: the backend keeps the statistics settings after the tab is left, so a tab-local
// copy would show the wrong ones, and "Run Complete Analysis" reads the same object.
// `onGrainMapsStale()` says the maps drawn from the grains changed although the grains did not.
export function GrainTab({
  analysisLoaded, hasGrains, grainsVersion, onGrainsReconstructed, onComplete,
  statsSettings = DEFAULT_GRAIN_STATS_SETTINGS, onStatsSettingsChange = () => {}, onGrainMapsStale = () => {},
}) {
  const { t } = useTranslation('analysis');
  const { gbThreshold, minPixels, minIntercept, alpha, fill, edgeRule, ecdBasis } = statsSettings;
  const statsPhase = statsSettings.statsPhase ?? null;

  // A reconstruction the backend is running (started here, or found by the page's /status sync) is in the store
  // (`reconTaskId`) and polled by the page (`useReconstructionWatch`), whichever tab is open; this tab shows it.
  const reconTaskId = useResultStore((state) => state.reconTaskId);
  const progress = useResultStore((state) => state.reconProgress);
  const outcome = useResultStore((state) => state.reconOutcome);
  const [starting,     setStarting]     = useState(false);     // the request that starts it is on its way
  const loading = starting || reconTaskId !== null;
  const [msg,          setMsg]          = useState(null);
  const [isError,      setIsError]      = useState(false);
  useEffect(() => { if (outcome) setMsg(null); }, [outcome]);     // the end of a task says more than an earlier detail

  // The statistics block, the epoch of /grain-size answers and the busy flag of "Run Complete Analysis" are
  // in the store: the block survives leaving the tab, and the chain can fill it and stop the tab.
  const grainStats = useResultStore((state) => state.grainStats);
  const runAllBusy = useResultStore((state) => state.runAllBusy);
  const grainSizeEpoch = useResultStore((state) => state.grainSizeEpoch);
  const phaseOptions = useResultStore((state) => state.phaseOptions);
  const setGrainStats = useResultStore.getState().setGrainStats;
  const busy = loading || runAllBusy;      // a reconstruction is running, from this tab or from the chain
  // "Run Complete Analysis" waits for this tab's own reconstruction; a task that runs on keeps the flag when the tab
  // is left, the page's watch ends it.
  useEffect(() => {
    useResultStore.getState().setGrainReconBusy(loading);
  }, [loading]);
  useEffect(() => () => {
    const state = useResultStore.getState();
    if (state.reconTaskId === null) state.setGrainReconBusy(false);     // a task that runs on keeps the flag
  }, []);
  const [analyzingSize,setAnalyzingSize]= useState(false);
  // /grain-size: the backend keeps the settings of the request that FINISHES last, so at most one is in
  // flight, and when it returns and the selection has moved on, one more is sent with the current one.
  const sizeBusy = useRef(false);    // a /grain-size request is in flight
  const sizeWanted = useRef(null);   // { rule, basis, phase } the statistics should be for
  const sizeTimer = useRef(null);    // debounce of the selects
  const sizeTimerPending = useRef(false);   // a pick is waiting out the pause
  const statsUpdateFailed = useRef(false);   // the shown message is that of a failed recompute
  const statsRef = useRef(null);     // the statistics on screen, for the answer that arrives later
  statsRef.current = grainStats;
  const currentEpoch = () => useResultStore.getState().grainSizeEpoch;
  const runGrainSizeRef = useRef(null);
  // A reconstruction started somewhere (this tab, the chain) or a map was loaded: a pick still waiting is void.
  useEffect(() => {
    sizeWanted.current = null;
    sizeTimerPending.current = false;
    clearTimeout(sizeTimer.current);
  }, [grainSizeEpoch]);
  // Leaving the tab inside the pause must not lose the pick: send it once.
  useEffect(() => () => {
    if (sizeTimerPending.current) {
      sizeTimerPending.current = false;
      clearTimeout(sizeTimer.current);
      runGrainSizeRef.current?.();
    }
  }, []);

  const handleReconstruct = async () => {
    const args = reconstructArgs(statsSettings);
    if (!args) {
      // An empty or non-numeric field would reach the API as null and come back as a 422.
      setMsg(t('analysis:grain.reconstructionFailed'));
      setIsError(true);
      return;
    }
    setStarting(true);
    setMsg(null);
    statsUpdateFailed.current = false;
    // the statistics of the old grains go, and a /grain-size answer for them must not land on the new ones
    useResultStore.getState().beginReconstruction();
    sizeWanted.current = null;
    sizeTimerPending.current = false;
    clearTimeout(sizeTimer.current);
    try {
      const res = await analysisApi.reconstructGrains(...args);
      const d = res.data;
      if (d.task_id) {
        useResultStore.getState().setReconTaskId(d.task_id);      // the page's watch polls it from here
        setStarting(false);
        setIsError(false);
      } else if (d.grain_count !== undefined) {
        setStarting(false);
        setGrainStats(grainStatsFrom(d));
        setMsg(t('analysis:grain.reconstructedMsg', { count: d.grain_count }));
        setIsError(false);
        toast.success(t('analysis:grain.reconstructionCompleteToast', { count: d.grain_count }));
        onGrainsReconstructed();
        onComplete('grains');
      }
    } catch (err) {
      setStarting(false);
      const detail = errorMessageFrom(err, t('analysis:grain.failed'));
      if (err?.response?.status === 409) {
        // one is already running (started elsewhere): ask the backend which, the page's watch polls it
        try { await syncFromBackend(); } catch { /* the detail below says enough */ }
      }
      setMsg(detail);
      setIsError(true);
    }
  };

  // Sends /grain-size for `sizeWanted`; if the selection has moved on when the answer is back, once more.
  const runGrainSize = async () => {
    if (sizeBusy.current || !sizeWanted.current) return;
    const epoch = currentEpoch();
    const sent = sizeWanted.current;
    sizeBusy.current = true;
    setAnalyzingSize(true);
    try {
      const res = await analysisApi.grainSize(sent.rule, sent.basis, sent.phase);
      if (epoch !== currentEpoch()) return;
      const d = res.data || {};
      if (d.mean_ecd !== undefined) {
        // the ECD map is drawn on the basis the backend now uses: draw it again if that changed
        if (statsRef.current?.ecdBasis !== (d.ecd_basis ?? sent.basis)) onGrainMapsStale();
        setGrainStats(mergeGrainSize(useResultStore.getState().grainStats, d));
        // only the message of an earlier recompute goes; a reconstruction error stays
        if (statsUpdateFailed.current) { setMsg(null); setIsError(false); statsUpdateFailed.current = false; }
      }
    } catch (err) {
      if (epoch !== currentEpoch()) return;
      setMsg(errorMessageFrom(err, t('analysis:grain.statsUpdateFailed')));
      setIsError(true);
      statsUpdateFailed.current = true;
    } finally {
      sizeBusy.current = false;
      const want = sizeWanted.current;
      // A pick wanted after a reconstruction may have found this request still in flight (and was not
      // sent); a want from before the reconstruction was voided. So: resend when a want exists and either
      // the epoch moved on or the selection differs from what was sent.
      if (want && (epoch !== currentEpoch() || want.rule !== sent.rule || want.basis !== sent.basis
          || want.phase !== sent.phase)) {
        runGrainSize();
      } else {
        setAnalyzingSize(false);
      }
    }
  };

  runGrainSizeRef.current = runGrainSize;

  // The button: the current selection, now.
  const handleGrainSizeAnalysis = () => {
    sizeTimerPending.current = false;
    clearTimeout(sizeTimer.current);
    sizeWanted.current = { rule: edgeRule, basis: ecdBasis, phase: statsPhase };
    runGrainSize();
  };

  // Picking an option IS the request: with grains present the statistics are recomputed (after a short
  // pause, so stepping through the options with the keyboard sends the last one, not each).
  const changeStatsSettings = (patch) => {
    onStatsSettingsChange(patch);
    if (hasGrains && !busy && ('edgeRule' in patch || 'ecdBasis' in patch || 'statsPhase' in patch)) {
      const next = { ...statsSettings, ...patch };
      sizeWanted.current = { rule: next.edgeRule, basis: next.ecdBasis, phase: next.statsPhase ?? null };
      clearTimeout(sizeTimer.current);
      sizeTimerPending.current = true;
      sizeTimer.current = setTimeout(() => { sizeTimerPending.current = false; runGrainSize(); }, GRAIN_SIZE_DEBOUNCE_MS);
    }
  };

  // What the edge-grain rule did, in words (the safeguard gets its own visible notice below).
  const edge = edgeSummary(grainStats);
  const edgeValues = edge.kind && edge.kind !== 'keep_all'
    ? { rule: t(`analysis:grain.edgeRuleShort.${edge.rule}`), edge: edge.edge, total: edge.total, used: edge.used, left: edge.left }
    : {};
  // grains exist but none entered the statistics (every one left out, or none in the chosen set): zeros
  // are not measurements, so no means are printed
  const noneUsed = !!grainStats && grainStats.nGrainsUsed === 0 && grainStats.nGrainsTotal > 0;
  // a chosen phase has no grain in this map (other phases do): one line, no means
  const phaseLabel = statsPhaseLabel(grainStats, phaseOptions);
  const phased = !!grainStats && grainStats.statsPhase !== null && grainStats.statsPhase !== undefined;
  const phaseName = phaseLabel ?? (phased ? t('analysis:grain.statsPhaseUnknown', { id: grainStats.statsPhase }) : null);
  const noGrainOfPhase = phased && !grainStats.statsError && grainStats.nGrainsTotal === 0 && grainStats.count !== 0;
  // no mean where nothing was measured: zeros are not measurements
  const noMeans = noneUsed || noGrainOfPhase || grainStats?.nGrainsTotal === 0;
  let totalLine = null;
  if (grainStats && noGrainOfPhase) {
    totalLine = t('analysis:grain.statNoGrainOfPhase', { phase: phaseName });
  } else if (grainStats && phased && grainStats.nGrainsTotal > 0 && !grainStats.statsError) {
    totalLine = grainStats.count !== undefined
      ? t('analysis:grain.statPhaseGrains', { phase: phaseName, count: grainStats.nGrainsTotal.toLocaleString(), all: grainStats.count.toLocaleString() })
      : t('analysis:grain.statPhaseGrainsOnly', { phase: phaseName, count: grainStats.nGrainsTotal.toLocaleString() });
  } else if (grainStats && grainStats.count !== undefined) {
    totalLine = t('analysis:grain.statTotalGrains', { count: grainStats.count.toLocaleString() });
  }
  // the number the means are made of, next to the total (which is before any edge grain is left out)
  const usedLine = grainStats && !grainStats.statsError && !noGrainOfPhase && !noneUsed
    && grainStats.nGrainsTotal !== 0 && grainStats.nGrainsUsed !== undefined
    ? t('analysis:grain.statGrainsUsed', { used: grainStats.nGrainsUsed.toLocaleString() })
    : null;
  let edgeLine = null;
  if (edge.kind === 'removed') edgeLine = t('analysis:grain.statEdgeRemoved', edgeValues);
  else if (edge.kind === 'keep_all') edgeLine = t('analysis:grain.statEdgeKeepAll');
  else if (edge.kind === 'safeguard') edgeLine = t('analysis:grain.statEdgeSafeguardLine', edgeValues);
  // the step of the map the numbers are made on (µm); with the rule, the basis and the phase it says what they were made with
  const stepSize = useResultStore((state) => state.stepSize);
  const stepLine = Number.isFinite(stepSize) && stepSize > 0
    ? t('analysis:grain.statStep', { step: String(Number(stepSize.toPrecision(6))) })
    : null;

  return (
    <TabLayout defaultLayer="ecd" analysisLoaded={analysisLoaded} hasGrains={hasGrains} grainsVersion={grainsVersion} controls={
      <>
        {/* Grain Reconstruction — matches recon_group with QFormLayout */}
        <ColoredGroupBox title={t('analysis:grain.reconstructionTitle')}>
          <FormRow label={t('analysis:grain.gbThreshold')}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <Input
                type="number" value={gbThreshold}
                onChange={(e) => onStatsSettingsChange({ gbThreshold: e.target.value })}
                min={1.0} max={15.0} step={0.5}
                disabled={busy || !analysisLoaded}
                title={t('analysis:grain.gbThresholdTooltip')}
                aria-label={t('analysis:grain.gbThreshold')}
                style={{ width: 70 }}
              />
              <span style={{ fontSize: 11, color: colors.textSecondary }}>{t('analysis:grain.unitDegrees')}</span>
            </div>
          </FormRow>
          <FormRow label={t('analysis:grain.minGrainSize')}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <Input
                type="number" value={minPixels}
                onChange={(e) => onStatsSettingsChange({ minPixels: e.target.value })}
                min={1} max={100} step={1}
                disabled={busy || !analysisLoaded}
                title={t('analysis:grain.minGrainSizeTooltip')}
                aria-label={t('analysis:grain.minGrainSize')}
                style={{ width: 70 }}
              />
              <span style={{ fontSize: 11, color: colors.textSecondary }}>{t('analysis:grain.unitPixels')}</span>
            </div>
          </FormRow>
          <FormRow label={t('analysis:grain.minIntercept')}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <Input
                type="number" value={minIntercept}
                onChange={(e) => onStatsSettingsChange({ minIntercept: e.target.value })}
                min={0.1} max={10.0} step={0.1}
                disabled={busy || !analysisLoaded}
                title={t('analysis:grain.minInterceptTooltip')}
                aria-label={t('analysis:grain.minIntercept')}
                style={{ width: 70 }}
              />
              <span style={{ fontSize: 11, color: colors.textSecondary }}>{t('analysis:grain.unitMicrometres')}</span>
              <span style={{ fontSize: 10, color: colors.textSecondary, fontStyle: 'italic' }}>{t('analysis:grain.minInterceptNote')}</span>
            </div>
          </FormRow>
          <FormRow label={t('analysis:grain.alpha')}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <Input
                type="number" value={alpha}
                onChange={(e) => onStatsSettingsChange({ alpha: e.target.value })}
                min={0} max={20} step={0.1}
                disabled={busy || !analysisLoaded}
                title={t('analysis:grain.alphaTooltip')}
                aria-label={t('analysis:grain.alpha')}
                style={{ width: 70 }}
              />
              <span style={{ fontSize: 11, color: colors.textSecondary }}>{t('analysis:grain.unitPixelSpacing')}</span>
            </div>
          </FormRow>
          <label
            style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: busy || !analysisLoaded ? 'default' : 'pointer', fontSize: 11, color: colors.text, margin: '4px 0' }}
            title={t('analysis:grain.fillTooltip')}
          >
            <input
              type="checkbox" checked={fill}
              onChange={(e) => onStatsSettingsChange({ fill: e.target.checked })}
              disabled={busy || !analysisLoaded}
              style={{ accentColor: colors.accent }}
            />
            {t('analysis:grain.fill')}
          </label>
          <GrainStatsSettings
            settings={statsSettings}
            onChange={changeStatsSettings}
            disabled={busy || !analysisLoaded}
            phaseOptions={phaseOptions}
          />
          <Button
            onClick={handleReconstruct}
            disabled={busy || !analysisLoaded}
            title={t('analysis:grain.reconstructGrainsTooltip')}
            style={{ width: '100%', marginTop: 4, background: colors.green, color: colors.textOnAccent }}
          >
            {loading ? <span className="btn-loading">{t('analysis:grain.reconstructing')}</span> : t('analysis:grain.reconstructGrains')}
          </Button>

          {loading && progress !== null && progress !== undefined && (
            <div style={{ marginTop: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                <Label secondary small>{t('analysis:grain.progress')}</Label>
                <Label secondary small>{Math.round(progress)}%</Label>
              </div>
              <ProgressBar value={progress} color={colors.green} />
            </div>
          )}
          <StatusMsg
            msg={msg ?? (loading ? t('analysis:grain.reconstructionRunning') : (outcome?.text ?? null))}
            isError={msg ? isError : (!loading && !!outcome?.error)}
          />
        </ColoredGroupBox>

        {/* Grain Statistics — matches stats_group; only while grains exist (a map loaded elsewhere has none) */}
        {hasGrains && grainStats && (
          <ColoredGroupBox title={t('analysis:grain.statisticsTitle')}>
            <div aria-live="polite">
            <ReadonlyText
              minHeight={70}
              text={[
                totalLine,
                usedLine,
                noneUsed ? t('analysis:grain.statNoneUsed') : null,
                !noMeans && grainStats.meanEcd  !== undefined ? t('analysis:grain.statMeanEcd', { value: grainStats.meanEcd.toFixed(2) }) : null,
                !noMeans && grainStats.medianEcd !== undefined ? t('analysis:grain.statMedianEcd', { value: grainStats.medianEcd.toFixed(2) }) : null,
                !noMeans && grainStats.stdEcd   !== undefined ? t('analysis:grain.statStdEcd', { value: grainStats.stdEcd.toFixed(2) }) : null,
                !noMeans && grainStats.meanAr   !== undefined ? t('analysis:grain.statMeanAr', { value: grainStats.meanAr.toFixed(3) }) : null,
                edgeLine,
                // no basis line where no numbers exist: the statistics failed, or there are no grains
                grainStats.ecdBasis && !grainStats.statsError && grainStats.nGrainsTotal !== 0 ? t('analysis:grain.statEcdBasis', { basis: t(`analysis:grain.ecdBasisShort.${grainStats.ecdBasis}`) }) : null,
                grainStats.nGrainsTotal !== 0 ? stepLine : null,
              ].filter(Boolean).join('\n')}
              placeholder={t('analysis:grain.statisticsPlaceholder')}
            />
            {edge.kind === 'safeguard' && (
              // an information about what happened, not a warning: neutral colours
              <div role="status" style={{
                marginTop: 6, padding: '5px 10px', borderRadius: 4, fontSize: 11,
                background: colorAlpha(colors.cyan, 13), border: `1px solid ${colors.cyan}`, color: colors.cyan,
              }}>
                {t({
                  few: phased ? 'analysis:grain.statEdgeSafeguardFewPhase' : 'analysis:grain.statEdgeSafeguardFew',
                  share: phased ? 'analysis:grain.statEdgeSafeguardSharePhase' : 'analysis:grain.statEdgeSafeguardShare',
                  many: phased ? 'analysis:grain.statEdgeSafeguardManyPhase' : 'analysis:grain.statEdgeSafeguardMany',
                }[edge.reason], { ...edgeValues, phase: phaseName })}
              </div>
            )}
            {grainStats.statsError && (
              <div role="alert" style={{
                marginTop: 6, padding: '5px 10px', borderRadius: 4, fontSize: 11,
                background: colorAlpha(colors.red, 13), border: `1px solid ${colors.red}`, color: colors.red,
              }}>
                {t('analysis:grain.statsError', { message: grainStats.statsError })}
              </div>
            )}
            </div>
          </ColoredGroupBox>
        )}

        {/* What to cite for these grains (the same panel as on the Phase Map page): the method and the settings
            they were reconstructed with. The key changes with the grains and with the statistics settings, so
            the methods text always describes the numbers shown above. */}
        {hasGrains && grainStats && !busy && (
          <CitationsPanel
            source="analysis"
            refreshKey={[grainsVersion, grainStats.edgeRule, grainStats.ecdBasis, grainStats.statsPhase,
              grainStats.nEdgeGrains, grainStats.edgeSafeguardApplied].join('|')}
          />
        )}

        {/* Grain Size Analysis — only after grains are available */}
        {hasGrains && !busy && (
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
export function DeformationTab({ analysisLoaded, hasGrains, grainsVersion, onComplete }) {
  const { t } = useTranslation('analysis');
  // The three parameters live in the store (the Batch tab and "Run Complete Analysis" read them too); "Analyze
  // Deformation" sends them to the backend, which remembers them for the KAM map, the RX analysis, the Excel export and /status.
  const params = useResultStore((state) => state.deformationParams);
  const paramsPending = useResultStore((state) => state.deformationEdited) && hasGrains;   // typed, not sent yet
  const setDeformationParams = useResultStore.getState().setDeformationParams;
  const args = deformationArgs(params);

  const [loading,      setLoading]      = useState(false);
  const [gmmRunning,   setGmmRunning]   = useState(false);
  const [msg,          setMsg]          = useState(null);
  const [isError,      setIsError]      = useState(false);
  const [results,      setResults]      = useState(null);

  const handleDeformation = async () => {
    if (!args) {                                   // an empty field or a number outside the backend's range
      setMsg(t('analysis:deformation.invalidParams'));
      setIsError(true);
      return;
    }
    setLoading(true);
    setMsg(null);
    setResults(null);
    try {
      const res = await analysisApi.deformation(args);
      const d = res.data || {};
      // the backend holds these parameters now; the maps drawn from them are drawn again
      useResultStore.getState().adoptDeformationUsed(d.deformation_params ?? deformationBody(args));
      useResultStore.getState().bumpGrainMaps();
      setResults({ ...d, used: args });
      setMsg(t('analysis:deformation.complete'));
      setIsError(false);
      onComplete('deformation');
    } catch (err) {
      if (err?.response?.status === 409) {
        try { await syncFromBackend(); } catch { /* the message says what to do */ }
      }
      setMsg(codedError(err, t) || errorMessageFrom(err, t('analysis:deformation.failed')));
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
      setMsg(codedError(err, t) || errorMessageFrom(err, t('analysis:deformation.gmmFailed')));
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
  // the parameters the numbers above were made with (what was sent, not what is typed now)
  const usedLine = results?.used
    ? t('analysis:deformation.paramsUsed', { order: results.used.kamOrder, threshold: results.used.kamThreshold, hagb: results.used.hagbMin })
    : null;

  return (
    <TabLayout defaultLayer="kam" analysisLoaded={analysisLoaded} hasGrains={hasGrains} grainsVersion={grainsVersion} controls={
      <>
        {/* Deformation Parameters — matches params_group with QFormLayout */}
        <ColoredGroupBox title={t('analysis:deformation.parametersTitle')}>
          <FormRow label={t('analysis:deformation.kamOrder')}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <Input
                type="number" value={params.kamOrder}
                onChange={(e) => setDeformationParams({ kamOrder: e.target.value })}
                min={1} max={5} step={1}
                disabled={loading || !analysisLoaded}
                title={t('analysis:deformation.kamOrderTooltip')}
                aria-label={t('analysis:deformation.kamOrder')}
                style={{ width: 60 }}
              />
              <span style={{ fontSize: 11, color: colors.textSecondary }}>{t('analysis:deformation.rangeOrder')}</span>
            </div>
          </FormRow>
          <FormRow label={t('analysis:deformation.kamThreshold')}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <Input
                type="number" value={params.kamThreshold}
                onChange={(e) => setDeformationParams({ kamThreshold: e.target.value })}
                min={0} max={30} step={0.5}
                disabled={loading || !analysisLoaded}
                title={t('analysis:deformation.kamThresholdTooltip')}
                aria-label={t('analysis:deformation.kamThreshold')}
                style={{ width: 70 }}
              />
              <span style={{ fontSize: 11, color: colors.textSecondary }}>{t('analysis:deformation.unitDegrees')}</span>
            </div>
          </FormRow>
          <FormRow label={t('analysis:deformation.hagbThreshold')}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <Input
                type="number" value={params.hagbMin}
                onChange={(e) => setDeformationParams({ hagbMin: e.target.value })}
                min={3} max={90} step={1}
                disabled={loading || !analysisLoaded}
                title={t('analysis:deformation.hagbThresholdTooltip')}
                aria-label={t('analysis:deformation.hagbThreshold')}
                style={{ width: 70 }}
              />
              <span style={{ fontSize: 11, color: colors.textSecondary }}>{t('analysis:deformation.unitDegrees')}</span>
            </div>
          </FormRow>
          <div data-testid="deformation-sagb-help" style={{ fontSize: 10, color: colors.textSecondary, lineHeight: 1.35, margin: '-2px 0 4px' }}>
            {t('analysis:deformation.sagbHelp')}
          </div>
          {/* what is typed is the current choice, but the maps, the numbers and the export follow the last run */}
          {/* the live region is always there; its text arrives later (a region inserted with its text is often not read) */}
          <div data-testid="deformation-not-applied" role="status" aria-live="polite" style={{ fontSize: 10, color: colors.orange, lineHeight: 1.35, marginTop: 2 }}>
            {paramsPending ? t('analysis:deformation.notApplied') : null}
          </div>
          {!args && (
            <div data-testid="deformation-invalid" style={{ fontSize: 10, color: colors.red, marginTop: 4 }}>
              {t('analysis:deformation.invalidParams')}
            </div>
          )}
          <Button
            onClick={handleDeformation}
            disabled={loading || !analysisLoaded || !hasGrains || !args}
            title={!analysisLoaded ? t('analysis:map.loadDatasetFirst')
              : !hasGrains ? t('analysis:page.needsGrainsFirst') : t('analysis:deformation.analyzeTooltip')}
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
          {usedLine && (
            <div data-testid="deformation-params-used" style={{ marginTop: 6, fontSize: 10, color: colors.textSecondary, lineHeight: 1.35 }}>
              {usedLine}
            </div>
          )}
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
// The three plane families of the surface-plane numbers: the key of the answer, the direction the MATLAB script
// measures for it, and the colour of the number.
const SURFACE_PLANES = [
  { key: '{111}', direction: '1 1 1', color: colors.cyan },
  { key: '{100}', direction: '1 0 0', color: colors.orange },
  { key: '{110}', direction: '1 1 0', color: colors.green },
];

export function TextureTab({ analysisLoaded, hasGrains, grainsVersion, onComplete }) {
  const { t } = useTranslation('analysis');
  // The preset and the tolerance angle (the text of the field) live in the store with the phase: they are sent with every
  // run, and the backend's choice comes back with the status unless a pick is pending.
  const preset = useResultStore((state) => state.texturePreset);
  const tolerance = useResultStore((state) => state.textureTolerance);
  const toleranceDeg = textureToleranceOf(tolerance);
  const [components,   setComponents]   = useState([]);
  const [loading,      setLoading]      = useState(false);
  const [odfRunning,   setOdfRunning]   = useState(false);
  const [msg,          setMsg]          = useState(null);
  const [isError,      setIsError]      = useState(false);
  const [textureIndex, setTextureIndex] = useState(null);
  const [entropy,      setEntropy]      = useState(null);
  const [surfacePlanes,setSurfacePlanes]= useState(null);   // { '{111}': number, ... }: the script's single direction
  const [planeFamilies,setPlaneFamilies]= useState(null);   // the same planes, any plane of the family
  const [resultPhase,  setResultPhase]  = useState(null);   // the name of the phase the numbers are of
  const [resultChoice, setResultChoice] = useState(null);   // { preset, tolerance }: what the numbers were made with
  const choicePending = useResultStore((state) => state.textureEdited);   // chosen or typed, not run yet
  // The phase of the texture: 'most_measured' or a phase id of the map. It lives in the store (the backend's
  // choice comes back with every status answer) and is sent with every run.
  const texturePhase = useResultStore((state) => state.texturePhase);
  const phaseOptions = useResultStore((state) => state.phaseOptions);

  const PRESET_OPTIONS = [
    { value: 'FCC_Rolling', label: t('analysis:texture.presetFcc') },
    { value: 'BCC_Rolling', label: t('analysis:texture.presetBcc') },
  ];

  const handlePresetChange = async (e) => {
    const p = e.target.value;
    useResultStore.getState().setTexturePreset(p);
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
      setMsg(errorMessageFrom(err, t('analysis:texture.loadComponentsFailed')));
      setIsError(true);
    } finally { setLoading(false); }
  };

  const handleODF = async () => {
    setOdfRunning(true);
    setMsg(null);
    try {
      const res = await analysisApi.texture(texturePhase, preset, toleranceDeg);
      useResultStore.getState().markTextureApplied({ phase: texturePhase, preset, tolerance });
      useResultStore.getState().bumpGrainMaps();      // the Texture Components map on screen was drawn with the earlier choice
      const d = res.data || {};
      if (d.components)             setComponents(d.components);
      setTextureIndex(d.texture_index ?? null);
      setEntropy(d.texture_entropy ?? null);
      setSurfacePlanes(d.surface_planes ?? null);
      setPlaneFamilies(d.surface_planes_family ?? null);
      setResultPhase(d.texture_phase?.name ?? null);
      setResultChoice({ preset: d.texture_preset ?? preset, tolerance: d.texture_tolerance_deg ?? toleranceDeg });
      setMsg(t('analysis:texture.complete'));
      setIsError(false);
      onComplete('texture');
    } catch (err) {
      setMsg(errorMessageFrom(err, t('analysis:texture.failed')));
      setIsError(true);
    } finally { setOdfRunning(false); }
  };

  return (
    <TabLayout defaultLayer="ipf" analysisLoaded={analysisLoaded} hasGrains={hasGrains} grainsVersion={grainsVersion} controls={
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
          <FormRow label={t('analysis:texture.phaseLabel')}>
            <select
              value={String(texturePhase)}
              onChange={(e) => useResultStore.getState().setTexturePhase(
                e.target.value === 'most_measured' ? 'most_measured' : parseInt(e.target.value, 10))}
              disabled={odfRunning || !analysisLoaded}
              aria-label={t('analysis:texture.phaseLabel')}
              title={t('analysis:texture.phaseHelp')}
              style={{
                background: colors.bgSecondary, border: `1px solid ${colors.border}`,
                borderRadius: 4, color: colors.text, fontSize: 12, padding: '4px 6px',
                width: '100%',
              }}
            >
              <option value="most_measured">{t('analysis:texture.phaseMostMeasured')}</option>
              {phaseOptions.map((p) => (
                <option key={p.id} value={String(p.id)} disabled={p.cubic === false}>
                  {p.cubic === false ? t('analysis:texture.phaseNotCubic', { name: p.name }) : p.name}
                </option>
              ))}
            </select>
          </FormRow>
          <div style={{ fontSize: 10, color: colors.textSecondary, margin: '-2px 0 6px' }}>
            {t('analysis:texture.phaseHelp')}
          </div>
          <FormRow label={t('analysis:texture.toleranceAngle')}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <Input
                type="number" value={tolerance}
                onChange={(e) => useResultStore.getState().setTextureTolerance(e.target.value)}
                min={0.1} max={MAX_TEXTURE_TOLERANCE} step={1.0}
                disabled={odfRunning || !analysisLoaded}
                title={t('analysis:texture.toleranceTooltip')}
                aria-label={t('analysis:texture.toleranceAngle')}
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
              disabled={odfRunning || !analysisLoaded || toleranceDeg === undefined}
              title={!analysisLoaded ? t('analysis:map.loadDatasetFirst')
                : toleranceDeg === undefined ? t('analysis:texture.toleranceInvalid') : t('analysis:texture.analyzeTooltip')}
              style={{ flex: 1, background: colors.purple, color: colors.textOnAccent }}
            >
              {odfRunning ? t('analysis:texture.computing') : t('analysis:texture.analyze')}
            </Button>
          </div>
          {toleranceDeg === undefined && (
            <div data-testid="texture-tolerance-invalid" role="alert" style={{ fontSize: 10, color: colors.red, marginTop: 4 }}>
              {t('analysis:texture.toleranceInvalid')}
            </div>
          )}
          {/* what is picked is the current choice, but the numbers, the map and the export follow the last run */}
          <div data-testid="texture-not-applied" role="status" aria-live="polite" style={{ fontSize: 10, color: colors.orange, lineHeight: 1.35, marginTop: 4 }}>
            {choicePending ? t('analysis:texture.notApplied') : null}
          </div>
          <StatusMsg msg={msg} isError={isError} />
        </ColoredGroupBox>

        {/* Texture Metrics — Texture Index + Entropy */}
        {resultPhase && (
          <div data-testid="texture-phase" style={{ fontSize: 12, fontWeight: 600, color: colors.text, margin: '0 2px 8px' }}>
            {t('analysis:texture.phaseResult', { phase: resultPhase })}
          </div>
        )}
        {resultPhase && resultChoice && (
          <div data-testid="texture-made-with" style={{ fontSize: 10, color: colors.textSecondary, margin: '-6px 2px 8px' }}>
            {t('analysis:texture.resultMadeWith', resultChoice)}
          </div>
        )}
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
            {SURFACE_PLANES.map(({ key, direction, color }) => {
              const script = surfacePlanes[key];
              const family = planeFamilies?.[key];
              return (
                <div key={key} data-testid={`plane-${key}`} style={{ marginBottom: 8 }}>
                  <div style={{ fontSize: 10, color: colors.textSecondary }}>
                    {t('analysis:texture.planeFamily', { plane: key })}
                  </div>
                  <div style={{ fontSize: 14, fontWeight: 700, color }}>
                    {family !== undefined ? `${Number(family).toFixed(1)} %` : '—'}
                  </div>
                  <div title={t('analysis:texture.planeScriptTooltip')} style={{ fontSize: 10, color: colors.textSecondary, cursor: 'help' }}>
                    {typeof script === 'number'
                      ? t('analysis:texture.planeScript', { direction, value: script.toFixed(1) })
                      : t('analysis:texture.planeScriptNone', { direction })}
                  </div>
                </div>
              );
            })}
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
                      {typeof c.fraction === 'number' ? `${c.fraction.toFixed(1)}%` : '—'}
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
// The four inputs live in the store (the Batch tab and "Run Complete Analysis" read them too); "Classify" sends
// them to the backend, which remembers them for the RX map, the Excel export and the status.
// ---------------------------------------------------------------------------
export function RXTab({ analysisLoaded, hasGrains, grainsVersion, onComplete }) {
  const { t } = useTranslation('analysis');
  const criteria = useResultStore((state) => state.rxCriteria);
  const criteriaPending = useResultStore((state) => state.rxCriteriaEdited) && hasGrains;   // typed, not sent with Classify yet
  const setRxCriteria = useResultStore.getState().setRxCriteria;
  // the phase the RX analysis works on ('most_measured' or a phase id): chosen with the criteria, sent with them
  const rxPhase = useResultStore((state) => state.rxPhase);
  const phaseOptions = useResultStore((state) => state.phaseOptions);

  const [loading,    setLoading]    = useState(false);
  const [msg,        setMsg]        = useState(null);
  const [isError,    setIsError]    = useState(false);
  const [rxResults,  setRxResults]  = useState(null);

  const handleRX = async () => {
    const args = rxCriteriaArgs(criteria);
    if (!args) {                                   // an empty field or a number outside the backend's range
      setMsg(t('analysis:rx.invalidCriteria'));
      setIsError(true);
      return;
    }
    setLoading(true);
    setMsg(null);
    setRxResults(null);
    try {
      // the KAM of the criteria follows the Deformation parameters: a value typed there and not run yet is run first
      await applyPendingDeformation(t);
      const res = await analysisApi.rxClassify(args, rxPhase);
      const d = res.data || {};
      // the backend holds these criteria now; the map and the other maps are drawn again with them
      useResultStore.getState().adoptRxCriteriaUsed(d);
      useResultStore.getState().bumpGrainMaps();
      setRxResults({
        rxFraction:       d.rx_fraction,
        highBcFraction:   d.high_bc_fraction,
        lowGkamFraction:  d.low_gkam_fraction,
        lowKamFraction:   d.low_kam_fraction,
        rxGrainCount:     d.rx_grain_count,
        totalGrains:      d.total_grains,
        needsBc:          d.rx_needs_bc === true,     // no native band contrast: there is no RX number at all
        used:             args,                       // what was sent: the criteria the numbers belong to
        phaseName:        d.rx_phase?.name ?? null,   // the phase the numbers are of
      });
      setMsg(t('analysis:rx.complete'));
      setIsError(false);
      onComplete('rx');
    } catch (err) {
      if (err?.response?.status === 409) {
        // the map or its grains were replaced while classifying: the backend did not store the criteria
        try { await syncFromBackend(); } catch { /* the message says what to do */ }
      }
      setMsg(codedError(err, t) || errorMessageFrom(err, t('analysis:rx.failed')));
      setIsError(true);
    } finally {
      setLoading(false);
    }
  };

  // Build results text to match rx_results_text in PyQt5
  // a number the backend did not give is "not available", with the reason the page knows (never 0, never nothing)
  const rxReason = rxResults?.needsBc ? t('analysis:rx.reasonNeedsBc') : t('analysis:rx.reasonNoPixels');
  const rxMetric = (value, key, nameKey) => (value === undefined || value === null
    ? t('analysis:rx.metricNotAvailable', { name: t(`analysis:rx.${nameKey}`), reason: rxReason })
    : t(`analysis:rx.${key}`, { value: Number(value).toFixed(1) }));
  const resultsText = rxResults ? [
    rxMetric(rxResults.rxFraction, 'metricRxFraction', 'nameRxFraction'),
    rxMetric(rxResults.highBcFraction, 'metricHighBcFraction', 'nameHighBcFraction'),
    rxMetric(rxResults.lowGkamFraction, 'metricLowGkamFraction', 'nameLowGkamFraction'),
    rxMetric(rxResults.lowKamFraction, 'metricLowKamFraction', 'nameLowKamFraction'),
    (rxResults.rxGrainCount !== undefined && rxResults.totalGrains !== undefined)
      ? t('analysis:rx.metricRxGrains', { rx: rxResults.rxGrainCount, total: rxResults.totalGrains }) : null,
  ].filter(Boolean).join('\n') : null;

  return (
    <TabLayout defaultLayer="rx" analysisLoaded={analysisLoaded} hasGrains={hasGrains} grainsVersion={grainsVersion} controls={
      <>
        {/* RX Classification Parameters — matches params_group with QFormLayout */}
        <ColoredGroupBox title={t('analysis:rx.parametersTitle')}>
          <FormRow label={t('analysis:rx.phaseLabel')}>
            <select
              value={String(rxPhase)}
              onChange={(e) => useResultStore.getState().setRxPhase(
                e.target.value === 'most_measured' ? 'most_measured' : parseInt(e.target.value, 10))}
              disabled={loading || !analysisLoaded}
              aria-label={t('analysis:rx.phaseLabel')}
              title={t('analysis:rx.phaseHelp')}
              style={{
                background: colors.bgSecondary, border: `1px solid ${colors.border}`,
                borderRadius: 4, color: colors.text, fontSize: 12, padding: '4px 6px',
                width: '100%',
              }}
            >
              <option value="most_measured">{t('analysis:rx.phaseMostMeasured')}</option>
              {phaseOptions.map((p) => (
                <option key={p.id} value={String(p.id)}>{p.name}</option>
              ))}
            </select>
          </FormRow>
          <div style={{ fontSize: 10, color: colors.textSecondary, margin: '-2px 0 6px' }}>
            {t('analysis:rx.phaseHelp')}
          </div>
          <FormRow label={t('analysis:rx.gosThreshold')}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <Input
                type="number" value={criteria.gosThreshold}
                onChange={(e) => setRxCriteria({ gosThreshold: e.target.value })}
                min={0} max={20} step={0.05}
                disabled={loading || !analysisLoaded}
                title={t('analysis:rx.gosThresholdTooltip')}
                aria-label={t('analysis:rx.gosThreshold')}
                style={{ width: 70 }}
              />
              <span style={{ fontSize: 11, color: colors.textSecondary }}>{t('analysis:rx.unitDegrees')}</span>
            </div>
          </FormRow>
          <FormRow label={t('analysis:rx.gbcFraction')}>
            <Input
              type="number" value={criteria.gbcFraction}
              onChange={(e) => setRxCriteria({ gbcFraction: e.target.value })}
              min={0} max={1} step={0.05}
              disabled={loading || !analysisLoaded}
              title={t('analysis:rx.gbcFractionTooltip')}
              aria-label={t('analysis:rx.gbcFraction')}
              style={{ width: 70 }}
            />
          </FormRow>
          <FormRow label={t('analysis:rx.gkamThreshold')}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <Input
                type="number" value={criteria.gkamThreshold}
                onChange={(e) => setRxCriteria({ gkamThreshold: e.target.value })}
                min={0} max={10} step={0.05}
                disabled={loading || !analysisLoaded}
                title={t('analysis:rx.gkamThresholdTooltip')}
                aria-label={t('analysis:rx.gkamThreshold')}
                style={{ width: 70 }}
              />
              <span style={{ fontSize: 11, color: colors.textSecondary }}>{t('analysis:rx.unitDegrees')}</span>
            </div>
          </FormRow>
          <FormRow label={t('analysis:rx.minGrainRadius')}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <Input
                type="number" value={criteria.minRadius}
                onChange={(e) => setRxCriteria({ minRadius: e.target.value })}
                min={0} max={100} step={0.1}
                disabled={loading || !analysisLoaded}
                title={t('analysis:rx.minGrainRadiusTooltip')}
                aria-label={t('analysis:rx.minGrainRadius')}
                style={{ width: 70 }}
              />
              <span style={{ fontSize: 11, color: colors.textSecondary }}>{t('analysis:rx.unitMicrometres')}</span>
            </div>
          </FormRow>
          {/* what is typed is the current choice, but the map, the numbers and the export follow the last Classify */}
          {/* the live region is always there; its text arrives later (a region inserted with its text is often not read) */}
          <div data-testid="rx-not-applied" role="status" aria-live="polite" style={{ fontSize: 10, color: colors.orange, lineHeight: 1.35, marginTop: 2 }}>
            {criteriaPending ? t('analysis:rx.notApplied') : null}
          </div>
          <Button
            onClick={handleRX}
            disabled={loading || !analysisLoaded || !hasGrains}
            title={!analysisLoaded ? t('analysis:map.loadDatasetFirst')
              : !hasGrains ? t('analysis:page.needsGrainsFirst') : t('analysis:rx.classifyTooltip')}
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
          {rxResults?.phaseName && (
            <div data-testid="rx-phase-result" style={{ marginTop: 6, fontSize: 11, fontWeight: 600, color: colors.text }}>
              {t('analysis:rx.phaseResult', { phase: rxResults.phaseName })}
            </div>
          )}
          {/* the criteria the numbers above were made with (what was sent, not what is typed now) */}
          {rxResults && !rxResults.needsBc && rxResults.used && (
            <div data-testid="rx-criteria-used" style={{ marginTop: 6, fontSize: 10, color: colors.textSecondary, lineHeight: 1.35 }}>
              <div>{t('analysis:rx.criteriaUsed', rxCriteriaLabels(rxResults.used))}</div>
              <div style={{ marginTop: 2 }}>{t('analysis:rx.fixedFractionsNote')}</div>
            </div>
          )}
        </ColoredGroupBox>

        {/* RX fraction badges */}
        {rxResults?.rxFraction !== undefined && rxResults.rxFraction !== null && (
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 10, animation: 'fadeSlideIn 0.2s ease-out' }}>
            <InfoBadge label={t('analysis:rx.badgeRxFraction')}   value={`${Number(rxResults.rxFraction).toFixed(1)}%`}   color={colors.green} />
            {rxResults.highBcFraction !== undefined && rxResults.highBcFraction !== null && (
              <InfoBadge label={t('analysis:rx.badgeHighBc')}      value={`${Number(rxResults.highBcFraction).toFixed(1)}%`}  color={colors.cyan} />
            )}
            {rxResults.lowGkamFraction !== undefined && rxResults.lowGkamFraction !== null && (
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
const EXPORT_POLL_MS = 500;
const LIST_PREVIEW = 10;          // a list longer than this is cut, with a toggle for the rest

const baseName = (path) => String(path ?? '').split(/[\\/]/).filter(Boolean).pop() || String(path ?? '');

/** The text of a Blob (a failed blob request carries the JSON error as a blob). */
function readBlobText(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsText(blob);
  });
}

/** `errorMessageFrom` for a request made with responseType 'blob': the `detail` is inside the blob. */
async function blobErrorMessage(err, fallback) {
  const data = err?.response?.data;
  if (typeof Blob !== 'undefined' && data instanceof Blob) {
    try {
      const detail = JSON.parse(await readBlobText(data))?.detail;
      if (typeof detail === 'string' && detail) return detail;
    } catch { /* not JSON: the fallback */ }
    return fallback;
  }
  return errorMessageFrom(err, fallback);
}

/** Hand a blob to the browser as a file download. */
function saveBlob(blob, fileName) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * The batch's "Processing file 2/5: name" as one line ("File 2 of 5: name"); a message of another shape is shown as it is.
 */
function currentFileText(message, t) {
  const m = /^Processing file (\d+)\/(\d+):\s*(.*)$/.exec(String(message));
  return m ? t('analysis:batch.currentFile', { current: m[1], total: m[2], name: m[3] }) : String(message);
}

/** A compact list that shows its first entries and the rest on request. */
function CompactList({ title, items, color = colors.textSecondary }) {
  const { t } = useTranslation('analysis');
  const [showAll, setShowAll] = useState(false);
  if (!items.length) return null;
  const shown = showAll ? items : items.slice(0, LIST_PREVIEW);
  return (
    <div style={{ marginTop: 6 }}>
      <div style={{ fontSize: 10, fontWeight: 700, color, marginBottom: 2 }}>{title}</div>
      <ul style={{ margin: 0, paddingLeft: 16, fontSize: 10, color: colors.textSecondary, lineHeight: 1.4 }}>
        {shown.map((item, i) => (
          <li key={i} style={{ overflowWrap: 'anywhere' }}>
            {/* focusable, so the full path (title) is reachable by keyboard and named for a screen reader */}
            <span tabIndex={0} title={item.title} aria-label={item.title ? `${item.text} (${item.title})` : item.text}>{item.text}</span>
          </li>
        ))}
      </ul>
      {items.length > LIST_PREVIEW && (
        <button
          type="button"
          aria-expanded={showAll}
          onClick={() => setShowAll((v) => !v)}
          style={{ background: 'none', border: 'none', color: colors.cyan, cursor: 'pointer', fontSize: 10, padding: '2px 0' }}
        >
          {showAll ? t('analysis:batch.showFewer') : t('analysis:batch.showAll', { count: items.length })}
        </button>
      )}
    </div>
  );
}

/**
 * The files where the edge rule kept all grains, in one line: the names are plain text (no tab stop for each),
 * a long list is cut with a toggle for the rest, like the lists below it.
 */
function EdgeSafeguardFiles({ names, processed }) {
  const { t } = useTranslation('analysis');
  const [showAll, setShowAll] = useState(false);
  if (!names.length) return null;
  const shown = showAll ? names : names.slice(0, LIST_PREVIEW);
  const list = shown.join(', ') + (shown.length < names.length ? ', …' : '');
  return (
    <div data-testid="batch-safeguard-files" style={{ marginTop: 6, fontSize: 10, color: colors.textSecondary, lineHeight: 1.4, overflowWrap: 'anywhere' }}>
      {t('analysis:batch.edgeSafeguardFiles', { n: names.length, processed, names: list })}
      {names.length > LIST_PREVIEW && (
        <button
          type="button"
          aria-expanded={showAll}
          onClick={() => setShowAll((v) => !v)}
          style={{ background: 'none', border: 'none', color: colors.cyan, cursor: 'pointer', fontSize: 10, padding: '0 0 0 6px' }}
        >
          {showAll ? t('analysis:batch.showFewer') : t('analysis:batch.showAll', { count: names.length })}
        </button>
      )}
    </div>
  );
}

/** What a batch did, file by file: the lists of the task's `result`, with the reasons. */
function BatchResultLists({ result }) {
  const { t } = useTranslation('analysis');
  if (!result || typeof result !== 'object') return null;
  const listOf = (list) => (Array.isArray(list) ? list : []);
  const withReason = (list) => listOf(list).map((e) => ({ text: `${baseName(e?.file)}: ${e?.reason ?? ''}`, title: String(e?.file ?? '') }));
  const processed = listOf(result.processed).map((f) => ({ text: baseName(f), title: String(f) }));
  const failed = withReason(result.failed);
  const skipped = withReason(result.skipped);
  // the step size note is not an analysis step that was left out: it has its own list ("step_size: ..." in the reason)
  const isStepNote = (e) => typeof e?.reason === 'string' && e.reason.startsWith('step_size:');
  const stepNotes = listOf(result.warnings).filter(isStepNote).map((e) => ({ text: baseName(e?.file), title: String(e?.file ?? '') }));
  const warnings = withReason(listOf(result.warnings).filter((e) => !isStepNote(e)));
  const safeguard = listOf(result.edge_safeguard_files).filter((name) => typeof name === 'string' && name);
  const processedCount = Number.isFinite(result.n_processed) ? result.n_processed : processed.length;
  return (
    <div data-testid="batch-result" style={{ marginTop: 8 }}>
      <CompactList title={t('analysis:batch.resultProcessed', { count: processed.length })} items={processed} color={colors.green} />
      <CompactList title={t('analysis:batch.resultFailed', { count: failed.length })} items={failed} color={colors.red} />
      <CompactList title={t('analysis:batch.resultSkipped', { count: skipped.length })} items={skipped} color={colors.orange} />
      <CompactList title={t('analysis:batch.resultWarnings', { count: warnings.length })} items={warnings} color={colors.orange} />
      <CompactList title={t('analysis:batch.resultStepSize', { count: stepNotes.length })} items={stepNotes} color={colors.orange} />
      <EdgeSafeguardFiles names={safeguard} processed={processedCount} />
    </div>
  );
}

export function BatchTab({ analysisLoaded, hasGrains, grainsVersion, onComplete }) {
  const { t } = useTranslation('analysis');
  const [folderPath,   setFolderPath]   = useState('');
  const [outputPath,   setOutputPath]   = useState('');
  // The name of the dataset in the workbook: what the user typed, else the loaded file's name without folder and
  // extension, else "Dataset". `null` = nothing typed (the field follows the loaded file); an emptied field falls back.
  const sourceName = useResultStore((state) => state.sourceName);
  const defaultDatasetName = datasetNameFromFile(sourceName) || 'Dataset';
  const [typedDatasetName, setTypedDatasetName] = useState(null);
  const datasetName = (typedDatasetName ?? '').trim() || defaultDatasetName;
  // Same settings as in the Grain Analysis tab, here for every file of the folder. The phase is "all phases"
  // or the phase with the most measured pixels of each file (a phase id of the loaded map means nothing for
  // other files). Its own copy, kept in the store so that it survives leaving the tab: a batch runs on other
  // files, not on the map the Grain tab works on; it starts at what the Grain tab has the first time the tab is
  // used (a single phase there becomes all phases), then goes its own way and is not reset per map.
  const storedStats = useResultStore((state) => state.batchStatsSettings);
  const statsSettings = storedStats ?? batchStatsFromGrain(useResultStore.getState().grainStatsSettings);
  useEffect(() => { useResultStore.getState().ensureBatchStatsSettings(); }, []);
  // The reconstruction of every file is the Grain tab's (one place to set it), read from the store.
  const reconSettings = useResultStore((state) => state.grainStatsSettings);
  const recon = batchReconstruction(reconSettings);
  // The RX criteria of every file are the Recrystallization tab's (one place to set them), read from the store.
  const rxSettings = useResultStore((state) => state.rxCriteria);
  const rxPending = useResultStore((state) => state.rxCriteriaEdited);   // typed, not applied to the open map yet
  const rx = rxCriteriaArgs(rxSettings);
  // The texture step of every file uses the Texture tab's preset and tolerance angle (the tolerance is left out, and the
  // batch uses its own 10 degrees, when the field holds no usable number)
  const texturePreset = useResultStore((state) => state.texturePreset);
  const textureTolerance = useResultStore((state) => state.textureTolerance);
  const texture = textureBatchArgs(texturePreset, textureTolerance);
  // The deformation parameters of every file are the Deformation tab's (KAM order, KAM threshold, HAGB limit).
  const deformationTyped = useResultStore((state) => state.deformationParams);
  const deformation = deformationArgs(deformationTyped);
  // What the batch and the export show lives in the store: it survives leaving the tab, and the poll of a
  // task that is still running starts again when the tab is opened.
  const batchRun = useResultStore((state) => state.batchRun);
  const exportRun = useResultStore((state) => state.exportRun);
  const setBatchRun = useResultStore.getState().setBatchRun;
  const setExportRun = useResultStore.getState().setExportRun;
  const {
    taskId: batchTaskId, running: batchRunning, progress: batchProgress, log: batchLog, msg: batchMsg,
    error: batchError, warn: batchWarn, result: batchResult, settings: batchSettings, currentFile: batchCurrentFile,
  } = batchRun;
  const {
    taskId: exportTaskId, running: exportRunning, msg: exportMsg, error: exportError, warn: exportWarn, warnings: exportWarnings,
    renamed: exportRenamed, rxApplied: exportRxApplied, textureApplied: exportTextureApplied,
    deformationApplied: exportDeformationApplied,
  } = exportRun;
  const phaseOptions = useResultStore((state) => state.phaseOptions);

  // Separate export output path — matches batch_output_edit in PyQt5
  const [exportOutputPath, setExportOutputPath] = useState('');

  // Poll batch task
  useEffect(() => {
    if (!batchTaskId) return;
    const iv = setInterval(async () => {
      try {
        const res = await analysisApi.getTaskStatus(batchTaskId);
        const d = res.data;
        const percent = progressPercent(d.progress);
        if (percent !== undefined) setBatchRun({ progress: percent });
        setBatchRun({ currentFile: typeof d.current_file === 'string' && d.current_file ? d.current_file : null });
        if (d.message) setBatchRun((prev) => ({ log: prev.log + '\n' + d.message }));
        // the backend writes 'completed' / 'failed' with the text in `error`; taskOutcome knows both spellings
        const out = taskOutcome(d);
        if (out.done) {
          clearInterval(iv);
          const result = d.result ?? null;
          const trouble = ['failed', 'skipped', 'warnings'].some((k) => Array.isArray(result?.[k]) && result[k].length > 0);
          setBatchRun({
            taskId: null, running: false, progress: null, error: false, warn: trouble, result,
            msg: d.message || t('analysis:batch.complete'),
          });
          onComplete('batch');
        } else if (out.failed) {
          clearInterval(iv);
          setBatchRun({
            taskId: null, running: false, progress: null, error: true, warn: false,
            msg: out.message || t('analysis:batch.failedMsg'),
            result: d.result ?? null,      // a batch with no file processed still says what happened to each
          });
        }
      } catch (err) {
        if (pollErrorOutcome(err).stop) {      // the backend forgot the task: nothing to wait for
          clearInterval(iv);
          // not "failed": a task that is gone may have finished (the registry keeps the last 50), or the server restarted
          setBatchRun({ taskId: null, running: false, progress: null, error: false, warn: true, msg: t('analysis:batch.statusUnknown') });
        }
        // other errors (network blip): keep polling
      }
    }, 2000);
    return () => clearInterval(iv);
  }, [batchTaskId, onComplete, t, setBatchRun]);

  const handleBatch = async () => {
    if (!folderPath.trim() || !outputPath.trim() || !recon || !rx || !deformation) return;
    setBatchRun({
      ...EMPTY_BATCH_RUN, running: true,
      settings: { edgeRule: statsSettings.edgeRule, ecdBasis: statsSettings.ecdBasis, statsPhase: statsSettings.statsPhase, recon, rx, rxPending, deformation },
      log: t('analysis:batch.logHeader', { folder: folderPath, output: outputPath }),
      msg: t('analysis:batch.startingMsg'),
    });
    try {
      const res = await analysisApi.batch(
        folderPath.trim(), outputPath.trim(), statsSettings.edgeRule, statsSettings.ecdBasis, statsSettings.statsPhase, recon, rx,
        texture, deformation,
      );
      if (res.data?.task_id) {
        setBatchRun({ taskId: res.data.task_id });
      } else {
        setBatchRun({ running: false, msg: res.data?.message || t('analysis:batch.completeShort'), error: false });
        onComplete('batch');
      }
    } catch (err) {
      setBatchRun({ running: false, msg: errorMessageFrom(err, t('analysis:batch.failedShort')), error: true });
    }
  };

  // Poll the export task (Electron path): its answer carries the file that was written and the steps that failed
  useEffect(() => {
    if (!exportTaskId) return;
    const fail = (msg) => setExportRun({ taskId: null, running: false, msg, error: true });
    const iv = setInterval(async () => {
      try {
        const res = await analysisApi.getTaskStatus(exportTaskId);
        const out = taskOutcome(res.data);
        if (out.done) {
          clearInterval(iv);
          const result = res.data.result && typeof res.data.result === 'object' ? res.data.result : {};
          const path = result.path || useResultStore.getState().exportRun.path;
          const requested = result.requested_path;
          setExportRun({
            taskId: null, running: false, error: false,
            msg: t('analysis:batch.exportedTo', { path }),
            warnings: Array.isArray(result.warnings) ? result.warnings.filter((w) => typeof w === 'string') : [],
            // the file existed: the backend wrote beside it and did not replace it
            renamed: requested && result.path && !samePath(requested, result.path, window.electronAPI?.platform)
              ? { requested, path: result.path } : null,
          });
          return;
        }
        if (out.failed) {
          clearInterval(iv);
          fail(out.message || t('analysis:batch.exportFailed'));
          return;
        }
      } catch (err) {
        if (pollErrorOutcome(err).stop) {
          // the task is gone: the file may have been written (or the server restarted); not "failed"
          clearInterval(iv);
          setExportRun({ taskId: null, running: false, msg: t('analysis:batch.exportStatusUnknown'), error: false, warn: true });
          return;
        }
        // other errors (network blip): ask again; a running export is polled until it ends or is gone
      }
    }, EXPORT_POLL_MS);
    return () => clearInterval(iv);
  }, [exportTaskId, t, setExportRun]);

  const handleExport = async () => {
    setExportRun({ ...EMPTY_EXPORT_RUN, running: true });
    try {
      // Electron saves to a path: without one nothing is exported, so nothing is reclassified either
      const electronPath = window.electronAPI?.saveFile ? (exportOutputPath.trim() || outputPath.trim()) : null;
      if (window.electronAPI?.saveFile && !electronPath) {
        setExportRun({ running: false, msg: t('analysis:batch.chooseOutputFirst'), error: true });
        return;
      }
      // the workbook of the open map must be made with the deformation parameters, the texture choice and the RX criteria
      // that are typed, not with the last ones sent
      const applied = await applyTypedSettings(t);
      const texturePhaseText = applied.texture
        ? (applied.texture.phase === 'most_measured' ? t('analysis:batch.phaseMostMeasuredShort')
          : (phaseOptions.find((p) => p.id === applied.texture.phase)?.name ?? String(applied.texture.phase)))
        : null;
      setExportRun({
        ...(applied.rx ? { rxApplied: rxCriteriaLabels(applied.rx) } : {}),
        ...(applied.texture ? { textureApplied: { preset: applied.texture.preset, tolerance: applied.texture.tolerance, phase: texturePhaseText } } : {}),
        ...(applied.deformation ? {
          deformationApplied: { order: applied.deformation.kamOrder, threshold: applied.deformation.kamThreshold, hagb: applied.deformation.hagbMin },
        } : {}),
      });
      // Browser mode: the workbook comes as a blob, so that the warnings in its header can be read
      if (!window.electronAPI?.saveFile) {
        let res;
        try {
          res = await analysisApi.downloadExcel(datasetName);
        } catch (err) {
          throw new Error(await blobErrorMessage(err, t('analysis:batch.exportFailed')));
        }
        const { warnings, unreadable } = warningsFromHeader(res.headers?.['x-orienta-warnings']);
        saveBlob(res.data, 'EBSD_Analysis.xlsx');
        setExportRun({
          running: false, msg: t('analysis:batch.downloadStarted'), error: false,
          warnings: unreadable ? [t('analysis:batch.exportWarningsUnreadable')] : warnings,
        });
        return;
      }
      // Electron mode: save to a user-chosen path
      const path = electronPath;
      const res = await analysisApi.exportExcel(path, datasetName);
      const taskId = res.data?.task_id;
      if (!taskId) {
        setExportRun({ running: false, msg: res.data?.message || t('analysis:batch.exportedTo', { path }), error: false });
        return;
      }
      setExportRun({ taskId, path });       // the poll above takes it from here
    } catch (err) {
      setExportRun({ running: false, msg: errorMessageFrom(err, t('analysis:batch.exportFailed')), error: true });
    }
  };

  return (
    <TabLayout defaultLayer="bc" analysisLoaded={analysisLoaded} hasGrains={hasGrains} grainsVersion={grainsVersion} controls={
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
          <GrainStatsSettings
            settings={statsSettings}
            onChange={(patch) => useResultStore.getState().setBatchStatsSettings(patch)}
            disabled={batchRunning}
            phaseOptions={[]}
            allowMostMeasured
          />
          {/* its own copy of the Grain tab's settings: they start equal, and a change on one side stays there */}
          <div data-testid="batch-stats-own" style={{ fontSize: 10, color: colors.textSecondary, lineHeight: 1.35, margin: '2px 0 8px' }}>
            {t('analysis:batch.statsSettingsOwn')}
          </div>
          {/* the texture phase of every file: not a setting here, the rule of the texture step */}
          <div data-testid="batch-texture-line" style={{ fontSize: 10, color: colors.textSecondary, lineHeight: 1.35, margin: '2px 0 8px' }}>
            {t('analysis:batch.textureLine')}
          </div>
          <div data-testid="batch-texture-settings-line" style={{ fontSize: 10, color: colors.textSecondary, lineHeight: 1.35, margin: '2px 0 8px' }}>
            {t('analysis:batch.textureSettingsLine', { preset: texture.preset, tolerance: texture.toleranceDeg ?? 10 })}
          </div>
          {/* the reconstruction of every file: the Grain tab's, said here because it is set there */}
          <div
            data-testid="batch-recon-line"
            style={{ fontSize: 10, color: recon ? colors.textSecondary : colors.red, lineHeight: 1.35, margin: '2px 0 8px' }}
          >
            {recon
              ? t(recon.minIntercept === undefined ? 'analysis:batch.reconLineNoIntercept' : 'analysis:batch.reconLine', {
                threshold: String(reconSettings.gbThreshold).trim(),
                minPixels: String(reconSettings.minPixels).trim(),
                alpha: String(reconSettings.alpha).trim(),
                fill: recon.fill ? t('analysis:batch.reconFillOn') : t('analysis:batch.reconFillOff'),
                minIntercept: String(reconSettings.minIntercept).trim(),
              })
              : t('analysis:batch.reconInvalid')}
          </div>
          {/* the RX criteria of every file: the Recrystallization tab's, said here because it is set there */}
          <div
            data-testid="batch-rx-line"
            style={{ fontSize: 10, color: rx ? colors.textSecondary : colors.red, lineHeight: 1.35, margin: '2px 0 8px' }}
          >
            {rx
              ? `${t(rxPending ? 'analysis:batch.rxLinePending' : 'analysis:batch.rxLine', rxCriteriaLabels(rx))} ${t('analysis:batch.rxPhaseRule')}`
              : t('analysis:batch.rxInvalid')}
          </div>
          {/* the deformation parameters of every file: the Deformation tab's, said here because it is set there */}
          <div
            data-testid="batch-deformation-line"
            style={{ fontSize: 10, color: deformation ? colors.textSecondary : colors.red, lineHeight: 1.35, margin: '2px 0 8px' }}
          >
            {deformation
              ? t('analysis:batch.deformationLine', { order: deformation.kamOrder, threshold: deformation.kamThreshold, hagb: deformation.hagbMin })
              : t('analysis:batch.deformationInvalid')}
          </div>
          <Button
            onClick={handleBatch}
            disabled={batchRunning || !folderPath.trim() || !outputPath.trim() || !recon || !rx || !deformation}
            title={t('analysis:batch.processFolderTooltip')}
            style={{ width: '100%', padding: '8px 14px', background: colors.purple, color: colors.textOnAccent }}
          >
            {batchRunning ? <span className="btn-loading">{t('analysis:batch.processing')}</span> : t('analysis:batch.processFolder')}
          </Button>
          {/* why the button is disabled, when the only reason is a path that is still empty */}
          {!batchRunning && recon && rx && deformation && (!folderPath.trim() || !outputPath.trim()) && (
            <div data-testid="batch-start-hint" style={{ marginTop: 4, fontSize: 10, color: colors.textSecondary, lineHeight: 1.35 }}>
              {t('analysis:batch.startHint')}
            </div>
          )}
          <div data-testid="batch-records" style={{ marginTop: 4, fontSize: 10, color: colors.textSecondary, lineHeight: 1.35 }}>
            {t('analysis:batch.workbookRecords')}
          </div>
          {batchRunning && batchProgress !== null && (
            <div style={{ marginTop: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                <Label secondary small>{t('analysis:batch.progress')}</Label>
                <Label secondary small>{Math.round(batchProgress)}%</Label>
              </div>
              <ProgressBar value={batchProgress} color={colors.purple} />
            </div>
          )}
          {batchRunning && batchCurrentFile && (
            <div data-testid="batch-current-file" role="status" aria-live="polite" style={{ marginTop: 4, fontSize: 10, color: colors.textSecondary, overflowWrap: 'anywhere' }}>
              {currentFileText(batchCurrentFile, t)}
            </div>
          )}
          <StatusMsg msg={batchMsg} isError={batchError} isWarn={batchWarn} />
          <div aria-live="polite">
          {batchSettings && !batchRunning && (batchResult || batchMsg) && (
            <div data-testid="batch-ran-with" style={{ marginTop: 6, fontSize: 10, color: colors.textSecondary, lineHeight: 1.35 }}>
              {t('analysis:batch.ranWith', {
                edge: t(`analysis:grain.edgeRuleShort.${batchSettings.edgeRule}`),
                basis: t(`analysis:grain.ecdBasisShort.${batchSettings.ecdBasis}`),
                phase: batchSettings.statsPhase === 'most_measured'
                  ? t('analysis:grain.statsPhaseMostMeasured') : t('analysis:grain.statsPhaseAll'),
                threshold: batchSettings.recon.threshold,
                minPixels: batchSettings.recon.minPixels,
                alpha: batchSettings.recon.alpha,
                fill: batchSettings.recon.fill ? t('analysis:batch.reconFillOn') : t('analysis:batch.reconFillOff'),
              })}
              {batchSettings.rx && ` · ${t(batchSettings.rxPending ? 'analysis:batch.ranWithRxPending' : 'analysis:batch.ranWithRx', rxCriteriaLabels(batchSettings.rx))}`}
              {batchSettings.deformation && ` · ${t('analysis:batch.ranWithDeformation', { order: batchSettings.deformation.kamOrder, threshold: batchSettings.deformation.kamThreshold, hagb: batchSettings.deformation.hagbMin })}`}
            </div>
          )}
          <BatchResultLists result={batchResult} />
          </div>

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
          <div data-testid="export-records" style={{ fontSize: 10, color: colors.textSecondary, lineHeight: 1.35, marginBottom: 10 }}>
            {t('analysis:batch.workbookRecordsExport')}
          </div>
          <FormRow label={t('analysis:batch.datasetNameLabel')}>
            <Input
              value={typedDatasetName ?? defaultDatasetName}
              onChange={(e) => setTypedDatasetName(e.target.value)}
              placeholder={defaultDatasetName}
              aria-label={t('analysis:batch.datasetNameLabel')}
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
          <StatusMsg msg={exportMsg} isError={exportError} isWarn={exportWarn} />
          {/* the live region is always there; its text arrives later (a region inserted with its text is often not read) */}
          <div role="status" aria-live="polite" data-testid="export-rx-applied-region">
            {exportDeformationApplied && (
              <div data-testid="export-deformation-applied" style={{ marginTop: 6, fontSize: 11, color: colors.textSecondary, overflowWrap: 'anywhere' }}>
                {t('analysis:batch.exportDeformationApplied', exportDeformationApplied)}
              </div>
            )}
            {exportTextureApplied && (
              <div data-testid="export-texture-applied" style={{ marginTop: 6, fontSize: 11, color: colors.textSecondary, overflowWrap: 'anywhere' }}>
                {t('analysis:batch.exportTextureApplied', exportTextureApplied)}
              </div>
            )}
            {exportRxApplied && (
              <div data-testid="export-rx-applied" style={{ marginTop: 6, fontSize: 11, color: colors.textSecondary, overflowWrap: 'anywhere' }}>
                {t('analysis:batch.exportRxApplied', exportRxApplied)}
              </div>
            )}
            {exportRenamed && (
              <div data-testid="export-renamed" style={{
                marginTop: 6, padding: '5px 10px', borderRadius: 4, fontSize: 11,
                background: colorAlpha(colors.orange, 13), border: `1px solid ${colors.orange}`, color: colors.orange,
                overflowWrap: 'anywhere',
              }}>
                {t('analysis:batch.exportRenamed', { requested: exportRenamed.requested, path: exportRenamed.path })}
              </div>
            )}
          </div>
          {exportWarnings.length > 0 && (
            <div role="alert" data-testid="export-warnings" style={{
              marginTop: 6, padding: '5px 10px', borderRadius: 4, fontSize: 11,
              background: colorAlpha(colors.orange, 13), border: `1px solid ${colors.orange}`, color: colors.orange,
            }}>
              <div>{t('analysis:batch.exportWarningsTitle')}</div>
              <ul style={{ margin: '3px 0 0', paddingLeft: 16 }}>
                {exportWarnings.map((w, i) => <li key={i} style={{ overflowWrap: 'anywhere' }}>{w}</li>)}
              </ul>
            </div>
          )}
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

  // grainsVersion counts finished reconstructions (and basis changes): the map viewers fetch again
  // when it changes. It and hasGrains are set in one store update.
  const {
    analysisLoaded, hasGrains, grainsVersion, grainStatsSettings, setAnalysis,
    markGrainsReconstructed, bumpGrainMaps, setGrainStatsSettings, resetGrainStatsSettings,
  } = useResultStore();
  const grainReconBusy = useResultStore((state) => state.grainReconBusy);   // the Grain tab's own reconstruction
  // The map the page shows was replaced or lost: what the tabs kept of the old one (results in their own state, the ticks
  // of the tab bar) goes. The tabs that keep results are keyed on it.
  const mapEpoch = useResultStore((state) => state.mapEpoch);

  // Global status label text — mirrors status_label in PyQt5
  const [statusMsg, setStatusMsg] = useState(() => t('analysis:status.ready'));

  // "Run Complete Analysis" state
  const runAllBusy = useResultStore((state) => state.runAllBusy);   // in the store: the Grain tab stops with it
  const setRunAllBusy = useResultStore.getState().setRunAllBusy;
  const [runAllStep, setRunAllStep] = useState(0);
  const [runAllTotal, setRunAllTotal] = useState(0);

  useEffect(() => {
    setCompletionFlags((prev) => ({
      ...prev, grains: useResultStore.getState().hasGrains, deformation: false, texture: false, rx: false,
    }));
  }, [mapEpoch]);

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

  // Ask the backend what it holds when the page becomes active and whenever the tab changes: a reconstruction may
  // have finished while the Grain tab was away (a tab that is left stops polling), a running one is picked up
  // again by the Grain tab, a map may have been loaded or swapped on another page (EBSD viewer, PhaseMap "Add
  // file", a reload of the window), and the backend may have lost its map (restart). The answer is applied by
  // the store (`applyBackendStatus`); one that was overtaken by something done here is dropped.
  useReconstructionWatch({ onComplete: markComplete });
  const announced = useRef(false);
  useEffect(() => {
    if (!isActive) { announced.current = false; return; }
    syncFromBackend().then((outcome) => {
      if (!outcome || !outcome.applied) return;
      const { d } = outcome;
      if (!d.loaded) {
        setDatasetStatus(null);
        setCompletionFlags({ data: false, grains: false, deformation: false, texture: false, rx: false, batch: false });
        setStatusMsg(t('analysis:status.ready'));
        return;
      }
      setDatasetStatus(d);
      setCompletionFlags(prev => ({ ...prev, data: true, grains: d.has_grains || false }));
      if (!announced.current) {      // the status line is rewritten when the page is entered, not at every tab click
        announced.current = true;
        const detail = d.shape?.length ? t('analysis:page.pixelsLoadedPlain', { shape: d.shape.join(' × ') }) : t('analysis:status.readyForAnalysis');
        setStatusMsg(t('analysis:status.datasetLoaded', { detail }));
      }
    }).catch(() => {});
  }, [isActive, activeTab, t]);

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
        resetGrainStatsSettings();   // a new map: the backend is back at its defaults
        bumpGrainMaps();             // and the maps on screen are the old map's
        setDatasetStatus(d);
        setCompletionFlags(prev => ({ ...prev, data: true }));
        const shape = d.shape?.length ? t('analysis:page.pixelsLoadedPlain', { shape: d.shape.join(' × ') }) : t('analysis:status.dataset');
        setStatusMsg(t('analysis:status.loadedFromBatch', { shape }));
      })
      .catch((err) => {
        const msg = errorMessageFrom(err, t('analysis:status.handoffFailed'));
        setStatusMsg(t('analysis:status.batchHandoffFailed', { msg }));
      });
  }, [isActive, setAnalysis, resetGrainStatsSettings, bumpGrainMaps, t]);

  const handleLoaded = useCallback((data) => {
    setAnalysis(data);
    resetGrainStatsSettings();   // a new map: the backend is back at its defaults
    bumpGrainMaps();             // the layers that need no grains (BC, IPF, KAM) are the old map's: fetch them again
    setDatasetStatus(data);
    const shape = data.shape?.length ? t('analysis:page.pixelsLoadedPlain', { shape: data.shape.join(' × ') }) : t('analysis:status.dataset');
    setStatusMsg(t('analysis:status.loadedShape', { shape }));
  }, [setAnalysis, resetGrainStatsSettings, bumpGrainMaps, t]);

  const handleGrainsReconstructed = markGrainsReconstructed;

  // "Run Complete Analysis" — chains Grains → Deformation → Texture → RX
  // Grain reconstruction is async (returns task_id), so we poll until done.
  const handleRunAll = async () => {
    if (!analysisLoaded) return;
    setRunAllBusy(true);

    // Helper: wait for async task to complete. The backend writes 'completed' / 'failed' (older code
    // also 'done' / 'error'); taskOutcome knows both and a task the backend forgot (404) ends the wait.
    const waitForTask = async (taskId) => {
      for (let i = 0; i < 300; i++) { // max 5 min (300 × 1s)
        await new Promise(r => setTimeout(r, 1000));
        let res;
        try {
          res = await analysisApi.getTaskStatus(taskId);
        } catch (err) {
          if (pollErrorOutcome(err).stop) throw new Error(t('analysis:grain.reconstructionFailed'));
          continue;   // network blip: ask again
        }
        const out = taskOutcome(res.data);
        if (out.done) return res.data;
        if (out.failed) throw new Error(out.message || t('analysis:runSteps.taskFailed'));
      }
      throw new Error(t('analysis:runSteps.taskTimedOut'));
    };

    // the texture and RX steps come after the reconstruction: a tolerance or a criterion that cannot be sent is found
    // before anything has run
    if (textureToleranceOf(useResultStore.getState().textureTolerance) === undefined) {
      setStatusMsg(t('analysis:texture.toleranceInvalid'));
      setRunAllBusy(false);
      return;
    }
    if (!rxCriteriaArgs(useResultStore.getState().rxCriteria)) {
      setStatusMsg(t('analysis:rx.invalidCriteria'));
      setRunAllBusy(false);
      return;
    }
    if (!deformationArgs(useResultStore.getState().deformationParams)) {
      setStatusMsg(t('analysis:deformation.invalidParams'));
      setRunAllBusy(false);
      return;
    }

    const steps = [
      {
        label: t('analysis:runSteps.grainReconstruction'),
        fn: async () => {
          // the Grain tab's parameters and statistics settings, as they are now (the store, not a copy)
          const args = reconstructArgs(useResultStore.getState().grainStatsSettings);
          if (!args) throw new Error(t('analysis:grain.reconstructionFailed'));
          useResultStore.getState().beginReconstruction();   // old statistics and pending answers go
          try {
            const res = await analysisApi.reconstructGrains(...args);
            const taskId = res.data?.task_id;
            const answer = taskId ? await waitForTask(taskId) : res.data;
            // the grains exist now: maps may be drawn and the Grain tab shows this run's statistics
            useResultStore.getState().setGrainStats(grainStatsFrom(answer));
            markGrainsReconstructed();
          } finally {
            // a tab click during the chain may have put the chain's task in the store: it is over, whatever
            // way it ended (the page's watch stays out of the chain's way while it runs)
            useResultStore.getState().setReconTaskId(null);
          }
        },
      },
      {
        label: t('analysis:runSteps.deformationAnalysis'),
        fn: async () => {
          // the Deformation tab's parameters, as they are now (the store, not the backend's last ones)
          const args = deformationArgs(useResultStore.getState().deformationParams);
          if (!args) throw new Error(t('analysis:deformation.invalidParams'));
          const res = await analysisApi.deformation(args);
          useResultStore.getState().adoptDeformationUsed(res.data?.deformation_params ?? deformationBody(args));
          useResultStore.getState().bumpGrainMaps();
          return res;
        },
      },
      { label: t('analysis:runSteps.textureAnalysis'),     fn: async () => {
        const { texturePhase: phase, texturePreset: preset, textureTolerance: tolerance } = useResultStore.getState();
        const res = await analysisApi.texture(phase, preset, textureToleranceOf(tolerance));
        useResultStore.getState().markTextureApplied({ phase, preset, tolerance });
        useResultStore.getState().bumpGrainMaps();
        return res;
      } },
      {
        label: t('analysis:runSteps.rxClassification'),
        fn: async () => {
          // the Recrystallization tab's criteria, as they are now (the store, not the backend's last ones)
          const args = rxCriteriaArgs(useResultStore.getState().rxCriteria);
          if (!args) throw new Error(t('analysis:rx.invalidCriteria'));
          const res = await analysisApi.rxClassify(args, useResultStore.getState().rxPhase);
          useResultStore.getState().adoptRxCriteriaUsed(res.data);
          useResultStore.getState().bumpGrainMaps();
        },
      },
    ];
    setRunAllTotal(steps.length);
    for (let i = 0; i < steps.length; i++) {
      setRunAllStep(i + 1);
      setStatusMsg(t('analysis:status.runningStep', { step: i + 1, total: steps.length, label: steps[i].label }));
      try {
        await steps[i].fn();
      } catch (err) {
        setStatusMsg(t('analysis:status.failedAtStep', { step: i + 1, total: steps.length, label: steps[i].label, error: errorMessageFrom(err, err.message || '') }));
        setRunAllBusy(false);
        setRunAllStep(0);
        return;
      }
    }
    setCompletionFlags({ data: true, grains: true, deformation: true, texture: true, rx: true, batch: false });
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
            hasGrains={hasGrains}
            grainsVersion={grainsVersion}
          />
        )}
        {activeTab === 'grains' && (
          <GrainTab
            analysisLoaded={analysisLoaded}
            hasGrains={hasGrains}
            grainsVersion={grainsVersion}
            onGrainsReconstructed={handleGrainsReconstructed}
            statsSettings={grainStatsSettings}
            onStatsSettingsChange={setGrainStatsSettings}
            onGrainMapsStale={bumpGrainMaps}
            onComplete={markComplete}
          />
        )}
        {activeTab === 'deformation' && (
          <DeformationTab
            key={mapEpoch}
            analysisLoaded={analysisLoaded}
            hasGrains={hasGrains}
            grainsVersion={grainsVersion}
            onComplete={markComplete}
          />
        )}
        {activeTab === 'texture' && (
          <TextureTab
            key={mapEpoch}
            analysisLoaded={analysisLoaded}
            hasGrains={hasGrains}
            grainsVersion={grainsVersion}
            onComplete={markComplete}
          />
        )}
        {activeTab === 'rx' && (
          <RXTab
            key={mapEpoch}
            analysisLoaded={analysisLoaded}
            hasGrains={hasGrains}
            grainsVersion={grainsVersion}
            onComplete={markComplete}
          />
        )}
        {activeTab === 'batch' && (
          <BatchTab
            analysisLoaded={analysisLoaded}
            hasGrains={hasGrains}
            grainsVersion={grainsVersion}
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
          disabled={runAllBusy || grainReconBusy || !analysisLoaded}
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
        <div role="status" aria-live="polite" style={{
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
