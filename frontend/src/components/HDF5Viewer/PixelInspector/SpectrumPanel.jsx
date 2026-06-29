import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, Label } from '../../../theme/components';
import { useEDSSpectrum } from '../hooks/useEDSSpectrum';
import { buildPolyline, xToEnergy, energyToX } from '../utils/spectrumPlot';
import { getElementLines } from '../utils/edsLines';
import useDataStore from '../../../stores/useDataStore';
import useEdsColorStore from '../../../stores/useEdsColorStore';

const PLOT_W = 380;
const PLOT_H = 140;

export default function SpectrumPanel({ isFileOpen, currentRow, currentCol }) {
  const { t } = useTranslation('hdf5viewer');
  const hasEDS = useDataStore((s) => s.hasEDS);
  const { spectrum, loading, error } = useEDSSpectrum(isFileOpen, hasEDS, currentRow, currentCol);
  const [logScale, setLogScale] = useState(false);
  const [hover, setHover] = useState(null);
  const edsElements = useDataStore((s) => s.edsElements);
  const [showSeries, setShowSeries] = useState({ K: true, L: true, M: false });
  const [roi, setRoi] = useState(null); // { x0, x1, count?, eLo?, eHi? }
  const [dragging, setDragging] = useState(false);
  // Subscribe to the element→colour map so an "EDS Colors" edit recolours the
  // line markers live. Reading via getState().getColor() was a non-reactive
  // snapshot. Must be before the early returns below (rules of hooks).
  const elementColors = useEdsColorStore((s) => s.colors);

  if (!isFileOpen) return <div style={{ color: colors.textSecondary }}>{t('spectrum.noFile')}</div>;
  if (!hasEDS) return (
    <div style={{ color: colors.textSecondary, fontSize: '8pt' }}>
      {t('spectrum.noEDS')}
    </div>
  );
  if (loading || !spectrum) return <div style={{ color: colors.textSecondary }}>{t('spectrum.loading')}</div>;
  if (error) return <div style={{ color: colors.red }}>{error}</div>;

  const eMin = 0;
  const eMax = spectrum.energy_axis_keV[spectrum.energy_axis_keV.length - 1];
  const polyline = buildPolyline(
    spectrum.counts, spectrum.energy_axis_keV,
    eMin, eMax, PLOT_W, PLOT_H, logScale
  );

  // Build element-line markers from edsElements (e.g. ["Fe Kα1", "Al Kα1", ...])
  const symbols = (edsElements ?? []).map((e) => (e || '').replace(/^Window Integral\s*/i, '').trim().split(/\s/)[0]);
  const uniqueSymbols = [...new Set(symbols)];
  const markers = [];
  for (const sym of uniqueSymbols) {
    const color = elementColors[sym] || '#bd93f9';
    for (const { line, energy } of getElementLines(sym)) {
      if (energy < eMin || energy > eMax) continue;
      if (line.startsWith('K') && !showSeries.K) continue;
      if (line.startsWith('L') && !showSeries.L) continue;
      if (line.startsWith('M') && !showSeries.M) continue;
      markers.push({ sym, line, energy, color, x: energyToX(energy, eMin, eMax, PLOT_W) });
    }
  }

  const handleMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    setHover({ x, energy: xToEnergy(x, eMin, eMax, PLOT_W) });
    if (dragging) {
      setRoi((r) => r ? { ...r, x1: x } : null);
    }
  };

  const handleDown = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    setDragging(true);
    setRoi({ x0: x, x1: x, count: null });
  };
  const handleUp = () => {
    if (!dragging) return;
    setDragging(false);
    if (!roi || !spectrum) return;
    const eLo = xToEnergy(Math.min(roi.x0, roi.x1), eMin, eMax, PLOT_W);
    const eHi = xToEnergy(Math.max(roi.x0, roi.x1), eMin, eMax, PLOT_W);
    let total = 0;
    for (let i = 0; i < spectrum.counts.length; i++) {
      if (spectrum.energy_axis_keV[i] >= eLo && spectrum.energy_axis_keV[i] <= eHi) {
        total += spectrum.counts[i];
      }
    }
    setRoi((r) => ({ ...r, count: total, eLo, eHi }));
  };

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
        <Label small secondary>{t('spectrum.title')}</Label>
        <label style={{ fontSize: '9pt', color: colors.text }} title={t('spectrum.logYTooltip')}>
          <input type="checkbox" checked={logScale} onChange={(e) => setLogScale(e.target.checked)} /> {t('spectrum.logY')}
        </label>
        {['K', 'L', 'M'].map((s) => (
          <label key={s} style={{ fontSize: '9pt', color: colors.text }} title={t('spectrum.linesTooltip')}>
            <input type="checkbox" checked={showSeries[s]}
              onChange={(e) => setShowSeries((p) => ({ ...p, [s]: e.target.checked }))} /> {t('spectrum.lines', { series: s })}
          </label>
        ))}
        <div style={{ flex: 1 }} />
        {hover && (
          <span style={{ fontSize: '8pt', color: colors.cyan, fontFamily: 'monospace' }}>
            {t('spectrum.hoverEnergy', { value: hover.energy.toFixed(2) })}
          </span>
        )}
      </div>
      <svg
        width={PLOT_W} height={PLOT_H}
        onMouseDown={handleDown}
        onMouseMove={handleMove}
        onMouseUp={handleUp}
        onMouseLeave={() => { setHover(null); setDragging(false); }}
        style={{ background: '#0a0a0a', display: 'block', userSelect: 'none', cursor: 'crosshair' }}
      >
        <title>{t('spectrum.plotTooltip')}</title>
        <polyline points={polyline} fill="none" stroke={colors.cyan} strokeWidth="1" />
        {markers.map((m, i) => (
          <g key={i}>
            <line x1={m.x} y1={0} x2={m.x} y2={PLOT_H}
              stroke={m.color} strokeWidth="1" strokeDasharray="2 2" opacity="0.6" />
            <text x={m.x + 2} y={10} fill={m.color} fontSize="7" fontFamily="monospace">
              {m.sym} {m.line}
            </text>
          </g>
        ))}
        {roi && (
          <rect x={Math.min(roi.x0, roi.x1)} y={0}
            width={Math.abs(roi.x1 - roi.x0)} height={PLOT_H}
            fill={colors.purple} opacity="0.15" />
        )}
        {hover && (
          <line x1={hover.x} y1={0} x2={hover.x} y2={PLOT_H} stroke={colors.purple} strokeWidth="1" />
        )}
      </svg>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '8pt', color: colors.textSecondary, fontFamily: 'monospace' }}>
        <span>{t('spectrum.axisStart')}</span>
        <span>{t('spectrum.axisEnd', { value: eMax.toFixed(1) })}</span>
      </div>
      {roi?.count != null && (
        <div style={{ fontSize: '8pt', color: colors.purple, fontFamily: 'monospace', marginTop: 4 }}>
          {t('spectrum.roiResult', { lo: roi.eLo.toFixed(2), hi: roi.eHi.toFixed(2), count: roi.count.toFixed(0) })}
        </div>
      )}
    </div>
  );
}
