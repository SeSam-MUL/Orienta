/**
 * AnomalyBrowserDrawer - right-side slide-out browser.
 *
 * Composes the HistogramPanel + GalleryList plus tab/sort/N controls into a
 * single right-side drawer. The drawer is mounted by PhaseMapPage and driven
 * by its `browserOpen` state.
 *
 * Hover/click callbacks emitted from GalleryItem rows are bubbled up so the
 * parent page can paint a transient marker on the phase-map canvas (hover)
 * or open the PatternMatchesDialog at that pixel (click).
 */
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, Select } from '../../../theme/components';
import HistogramPanel from './HistogramPanel';
import GalleryList from './GalleryList';
import {
  useDiagnosticsSummary,
  useDiagnosticsBrowser,
  clearDiagnosticsCache,
} from './useDiagnosticsQuery';

// Default-sort keys are stable identifiers (not user-facing); labels are
// translated at render time.
const METRIC_TABS = [
  { key: 'ncc',              labelKey: 'tabNcc',      defaultSort: 'ncc_asc' },
  { key: 'local_anomaly',    labelKey: 'tabAnomaly',  defaultSort: 'local_anomaly_desc' },
  { key: 'pc_sensitivity',   labelKey: 'tabPcSens',   defaultSort: 'pc_sensitivity_desc' },
  { key: 'pattern_residual', labelKey: 'tabResidual', defaultSort: 'pattern_residual_desc' },
];

const N_OPTIONS = [25, 50, 100, 200];
const W = 480;

/**
 * GalleryListWrapper - measures available height for the virtualized
 * GalleryList. react-window's FixedSizeList requires explicit pixel height.
 */
function GalleryListWrapper(props) {
  const ref = useRef(null);
  const [h, setH] = useState(400);
  useLayoutEffect(() => {
    if (!ref.current) return;
    const observer = new ResizeObserver(([entry]) => {
      setH(Math.max(100, Math.floor(entry.contentRect.height)));
    });
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  return (
    <div ref={ref} style={{ flex: 1, minHeight: 0 }}>
      <GalleryList {...props} height={h} />
    </div>
  );
}

export default function AnomalyBrowserDrawer({
  open,
  onClose,
  resultId,
  onSelectPixel,
  onHoverPixel,
  onLeavePixel,
}) {
  const { t } = useTranslation('phasemap');
  const SORT_OPTIONS = {
    ncc: [
      { value: 'ncc_asc',  label: t('phasemap:anomalyBrowser.sortNccAsc')  },
      { value: 'ncc_desc', label: t('phasemap:anomalyBrowser.sortNccDesc') },
    ],
    local_anomaly: [
      { value: 'local_anomaly_desc', label: t('phasemap:anomalyBrowser.sortAnomalyDesc') },
      { value: 'local_anomaly_asc',  label: t('phasemap:anomalyBrowser.sortAnomalyAsc')  },
    ],
    pc_sensitivity: [
      { value: 'pc_sensitivity_desc', label: t('phasemap:anomalyBrowser.sortPcSensDesc') },
      { value: 'pc_sensitivity_asc',  label: t('phasemap:anomalyBrowser.sortPcSensAsc')  },
    ],
    pattern_residual: [
      { value: 'pattern_residual_desc', label: t('phasemap:anomalyBrowser.sortResidualDesc') },
      { value: 'pattern_residual_asc',  label: t('phasemap:anomalyBrowser.sortResidualAsc')  },
    ],
  };
  const [metricKey, setMetricKey] = useState('ncc');
  const [range, setRange] = useState(null);
  const [sort, setSort] = useState('ncc_asc');
  const [n, setN] = useState(50);

  // Reset range + restore the metric's default sort when the user switches tabs.
  useEffect(() => {
    setRange(null);
    const def = METRIC_TABS.find((t) => t.key === metricKey)?.defaultSort;
    if (def) setSort(def);
  }, [metricKey]);

  // Re-fetch diagnostics every time the drawer opens. A prior open may have
  // 404'd (compute not finished yet); bumping `refreshKey` on each open — and
  // dropping any stale cache entry for this result — guarantees a fresh fetch
  // so a completed compute is always picked up.
  const [refreshKey, setRefreshKey] = useState(0);
  useEffect(() => {
    if (open) {
      clearDiagnosticsCache(resultId);
      setRefreshKey((k) => k + 1);
    }
  }, [open, resultId]);

  const { summary, loading: summaryLoading } = useDiagnosticsSummary(resultId, refreshKey);
  const browserOpts = useMemo(
    () => ({
      sort,
      n,
      range_min: range ? range[0] : undefined,
      range_max: range ? range[1] : undefined,
    }),
    [sort, n, range],
  );
  const { data } = useDiagnosticsBrowser(resultId, browserOpts);

  if (!open) return null;
  return (
    <div
      style={{
        position: 'fixed',
        right: 0,
        top: 0,
        bottom: 0,
        width: W,
        background: colors.bg,
        borderLeft: `1px solid ${colors.border}`,
        boxShadow: '-4px 0 16px rgba(0,0,0,0.4)',
        display: 'flex',
        flexDirection: 'column',
        zIndex: 100,
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          padding: 12,
          borderBottom: `1px solid ${colors.border}`,
        }}
      >
        <div style={{ flex: 1, fontWeight: 600 }}>{t('phasemap:anomalyBrowser.title')}</div>
        <button
          onClick={onClose}
          aria-label={t('phasemap:anomalyBrowser.closeAria')}
          style={{
            background: 'none',
            border: 'none',
            color: colors.text,
            fontSize: 18,
            cursor: 'pointer',
          }}
        >
          ×
        </button>
      </div>

      {/* Metric tabs */}
      <div style={{ display: 'flex', borderBottom: `1px solid ${colors.border}` }}>
        {METRIC_TABS.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setMetricKey(tab.key)}
            title={t('phasemap:hoverTips.anomalyTab')}
            style={{
              flex: 1,
              padding: 8,
              background: tab.key === metricKey ? colors.bgTertiary : 'transparent',
              color: colors.text,
              border: 'none',
              cursor: 'pointer',
              borderBottom: tab.key === metricKey ? `2px solid ${colors.accent}` : 'none',
            }}
          >
            {t(`phasemap:anomalyBrowser.${tab.labelKey}`)}
          </button>
        ))}
      </div>

      {/* Histogram with brush */}
      <HistogramPanel
        metricKey={metricKey}
        summary={summary}
        loading={summaryLoading}
        range={range}
        onChange={setRange}
      />

      {/* Sort / Top N controls */}
      <div
        style={{
          display: 'flex',
          gap: 8,
          padding: 8,
          alignItems: 'center',
          borderTop: `1px solid ${colors.border}`,
        }}
      >
        <span style={{ fontSize: 12 }}>{t('phasemap:anomalyBrowser.sort')}</span>
        <Select
          value={sort}
          onChange={(e) => setSort(e.target.value)}
          options={SORT_OPTIONS[metricKey]}
          title={t('phasemap:hoverTips.anomalySort')}
        />
        <span style={{ fontSize: 12 }}>{t('phasemap:anomalyBrowser.top')}</span>
        <Select
          value={String(n)}
          onChange={(e) => setN(Number(e.target.value))}
          options={N_OPTIONS.map((v) => ({ value: String(v), label: String(v) }))}
          title={t('phasemap:hoverTips.anomalyTopN')}
        />
        <div style={{ marginLeft: 'auto', fontSize: 11, color: colors.textSecondary }}>
          {data ? t('phasemap:anomalyBrowser.ofTotal', { shown: data.items.length, total: data.total_matching }) : ''}
        </div>
      </div>

      {/* Virtualized gallery (fills remaining height) */}
      <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
        <GalleryListWrapper
          items={data?.items}
          resultId={resultId}
          width={W}
          onClick={onSelectPixel}
          onHover={onHoverPixel}
          onLeave={onLeavePixel}
        />
      </div>
    </div>
  );
}
