import Plot from 'react-plotly.js';
import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/components';

const LAYER_COLOR = {
  bc:       '#888888',
  vbse:     '#cccccc',
  phase:    '#a0f',
  'ipf-z':  '#0ff',
  'ipf-x':  '#0fa',
  'ipf-y':  '#0af',
  ci:       '#f80',
};

function defaultColorFor(layerId) {
  if (LAYER_COLOR[layerId]) return LAYER_COLOR[layerId];
  if (layerId.startsWith('eds-')) {
    // hashed-bright colour per element — stable across renders.
    const h = [...layerId].reduce((a, c) => (a * 31 + c.charCodeAt(0)) | 0, 0);
    const hue = Math.abs(h) % 360;
    return `hsl(${hue}, 70%, 60%)`;
  }
  return '#7a7';
}

/**
 * Renders per-layer linescan traces using Plotly. Each layer becomes one
 * scatter trace with x = sample index (0..n_samples-1).
 */
export default function LinescanProfilePlot({ data, layers }) {
  const { t } = useTranslation('eds');
  if (!data?.series) return null;
  const visibleLayerIds = (layers || []).filter(l => l.visible).map(l => l.id);
  const orderedSeries = visibleLayerIds
    .filter((id) => Array.isArray(data.series[id]))
    .map((id) => ({
      x: data.series[id].map((_, i) => i),
      y: data.series[id],
      name: id,
      type: 'scatter',
      mode: 'lines',
      line: { color: defaultColorFor(id), width: 2 },
    }));

  return (
    <Plot
      data={orderedSeries}
      layout={{
        autosize: true,
        height: 200,
        margin: { l: 50, r: 10, t: 8, b: 30 },
        paper_bgcolor: colors.bgSecondary,
        plot_bgcolor: colors.bg,
        font: { color: colors.text, size: 10 },
        xaxis: { gridcolor: colors.border, zerolinecolor: colors.border, title: t('linescan.axisSample') },
        yaxis: { gridcolor: colors.border, zerolinecolor: colors.border, title: t('linescan.axisValue') },
        legend: { orientation: 'h', y: 1.18 },
      }}
      style={{ width: '100%' }}
      config={{ displayModeBar: false, responsive: true }}
    />
  );
}
