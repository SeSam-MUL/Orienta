/**
 * PCRefinement — matches gui/pcrefinement_gui.py layout exactly.
 *
 * Layout:
 *   Header: phase label | phase info | stretch | CI | Global CI
 *   ResizableSplitter (horizontal):
 *     Left  — pattern canvas + hide-lines + compare + ? + slider +
 *             selected-patterns list + remove button
 *     Right — ScrollArea:
 *               progress bar (hidden until running) + cancel
 *               Load Phase from CIF
 *               Index Pattern
 *               Global PC Refine
 *               PC Drift Analysis
 *               GroupBox: Pixel-wise PC Correction
 *               GroupBox: Indexing Settings
 *               GroupBox: Detector Settings
 *                 sub-GroupBox: Pixel Size & Binning
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { pcApi, ebsdApi, calibrationApi, indexApi } from '../../services/api';
import useDataStore from '../../stores/useDataStore';
import PhaseDropdown from '../Indexing/PhaseDropdown';
import { useImageExport, exportStem } from '../common/useImageExport';
import { buildPanelSheet } from '../common/imageExport';
import FloatingPhasePanel from '../Indexing/FloatingPhasePanel';
import {
  colors,
  alpha,
  spacing,
  Button,
  Input,
  NumberInput,
  Select,
  GroupBox,
  ResizableSplitter,
  FormRow,
  Label,
  ProgressBar,
  Separator,
  usePrompt,
  PromptDialog,
} from '../../theme/components';

// ---------------------------------------------------------------------------
// Small inline helpers
// ---------------------------------------------------------------------------

/** CI color coding matching PyQt5: green>=0.3, yellow>=0.15, orange>=0.05, red<0.05 */
function ciColor(ci) {
  if (ci == null) return colors.textSecondary;
  if (ci >= 0.3) return colors.green;
  if (ci >= 0.15) return colors.yellow;
  if (ci >= 0.05) return colors.orange;
  return colors.red;
}

function StatusMsg({ msg, isError }) {
  if (!msg) return null;
  return (
    <div
      role={isError ? 'alert' : 'status'}
      style={{
        marginTop: 4,
        padding: '4px 8px',
        borderRadius: 3,
        background: isError ? alpha(colors.red, 10) : alpha(colors.green, 7),
        border: `1px solid ${isError ? alpha(colors.red, 33) : alpha(colors.green, 33)}`,
        color: isError ? colors.red : colors.green,
        fontSize: '9pt',
        wordBreak: 'break-word',
        animation: 'fadeSlideIn 0.2s ease-out',
        transition: 'background 0.2s, border-color 0.2s, color 0.2s',
      }}
    >
      {msg}
    </div>
  );
}

function WarnLabel({ children }) {
  return (
    <div
      style={{
        fontSize: '9pt',
        color: colors.orange,
        background: alpha(colors.orange, 7),
        border: `1px solid ${alpha(colors.orange, 27)}`,
        borderRadius: 3,
        padding: '2px 6px',
        marginTop: 3,
        animation: 'fadeSlideIn 0.2s ease-out',
      }}
    >
      {children}
    </div>
  );
}

// Read-only mono display — matches QPlainTextEdit(readOnly)
function MonoDisplay({ value, rows = 3, title }) {
  return (
    <textarea
      readOnly
      value={value || ''}
      rows={rows}
      title={title}
      style={{
        width: '100%',
        background: colors.bg,
        border: `1px solid ${colors.border}`,
        borderRadius: 3,
        color: colors.textSecondary,
        fontSize: '9pt',
        fontFamily: "'Courier New', monospace",
        padding: '4px 6px',
        resize: 'vertical',
        boxSizing: 'border-box',
        outline: 'none',
      }}
    />
  );
}

// ---------------------------------------------------------------------------
// Detect an EDAX-style circular aperture: all four corner blocks near-black.
// Mirrors backend tools.pattern_comparison.detect_circular_aperture so the red
// overlay is clipped to the inscribed disc for EDAX but left full-frame for
// Oxford (whose corners carry real signal).
function cornersAreBlack(ctx, w, h) {
  if (w < 16 || h < 16) return false;
  const blk = 8;
  try {
    const corners = [
      ctx.getImageData(0, 0, blk, blk),
      ctx.getImageData(w - blk, 0, blk, blk),
      ctx.getImageData(0, h - blk, blk, blk),
      ctx.getImageData(w - blk, h - blk, blk, blk),
    ];
    for (const s of corners) {
      const d = s.data;
      let sum = 0;
      for (let i = 0; i < d.length; i += 4) sum += (d[i] + d[i + 1] + d[i + 2]) / 3;
      // ~5% of 255 (the pattern PNG is contrast-stretched to 0–255).
      if (sum / (d.length / 4) >= 12) return false;
    }
    return true;
  } catch {
    return false; // getImageData may throw on a tainted canvas → no clip
  }
}

// Pattern canvas with PC crosshair overlay
// ---------------------------------------------------------------------------
function PatternCanvas({ patternBase64, pcx, pcy, hideLines, segments, exportName }) {
  const { t } = useTranslation('pcrefinement');
  const canvasRef = useRef(null);
  const imageExport = useImageExport();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    if (!patternBase64) {
      canvas.width = 300;
      canvas.height = 300;
      ctx.fillStyle = colors.bg;
      ctx.fillRect(0, 0, 300, 300);
      ctx.fillStyle = colors.textSecondary;
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(t('pcrefinement:canvas.noPatternLoaded'), 150, 150);
      return;
    }

    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      if (!hideLines) {
        // For EDAX circular-aperture patterns, clip the overlay to the
        // inscribed disc so the Kikuchi lines + crosshair don't draw across
        // the black corners. Oxford full-frame patterns are left unclipped.
        const circular = cornersAreBlack(ctx, canvas.width, canvas.height);
        if (circular) {
          ctx.save();
          ctx.beginPath();
          ctx.arc(canvas.width / 2, canvas.height / 2,
                  Math.min(canvas.width, canvas.height) / 2, 0, Math.PI * 2);
          ctx.clip();
        }

        // Draw Kikuchi band lines (from indexing simulation)
        if (segments && segments.length > 0) {
          ctx.save();
          ctx.strokeStyle = '#ff5555';
          ctx.lineWidth = 1;
          ctx.globalAlpha = 0.85;
          for (const [[x0, y0], [x1, y1]] of segments) {
            ctx.beginPath();
            ctx.moveTo(x0, y0);
            ctx.lineTo(x1, y1);
            ctx.stroke();
          }
          ctx.restore();
        }

        // Draw PC crosshair
        const cx = parseFloat(pcx);
        const cy = parseFloat(pcy);
        if (!isNaN(cx) && !isNaN(cy)) {
          const px = cx * img.width;
          const py = cy * img.height;
          const r = Math.max(6, img.width * 0.025);
          ctx.save();
          ctx.strokeStyle = '#ff5555';
          ctx.lineWidth = 1.5;
          ctx.globalAlpha = 0.9;
          ctx.beginPath();
          ctx.moveTo(px - r * 2, py);
          ctx.lineTo(px + r * 2, py);
          ctx.moveTo(px, py - r * 2);
          ctx.lineTo(px, py + r * 2);
          ctx.stroke();
          ctx.beginPath();
          ctx.arc(px, py, r, 0, Math.PI * 2);
          ctx.stroke();
          ctx.restore();
        }

        if (circular) ctx.restore();
      }
    };
    img.src = `data:image/png;base64,${patternBase64}`;
  }, [patternBase64, pcx, pcy, hideLines, segments, t]);

  return (
    <>
    <canvas
      ref={canvasRef}
      aria-label={t('pcrefinement:canvas.ariaLabel')}
      onContextMenu={(e) => {
        if (!patternBase64) return;
        imageExport.openMenu(e, {
          // Read the CANVAS, not the source PNG: the band lines and the PC
          // crosshair are drawn here, so exporting the base64 would silently
          // drop exactly what the user is looking at. Its backing store is the
          // pattern's own pixel grid, so this is the data resolution — enlarge
          // it in the dialog if a figure needs more.
          build: () => canvasRef.current?.toDataURL('image/png'),
          name: exportName || 'pattern',
          label: exportName || 'pattern',
        });
      }}
      style={{
        width: '100%',
        height: 'auto',
        display: 'block',
        border: `1px solid ${colors.border}`,
        borderRadius: 3,
        imageRendering: 'pixelated',
        background: colors.bg,
      }}
    />
    {imageExport.node}
    </>
  );
}

// ---------------------------------------------------------------------------
// Left preview panel — matches _create_preview_panel()
// ---------------------------------------------------------------------------
function PreviewPanel({
  patterns,
  selectedIdx,
  onSelectIdx,
  onRemovePattern,
  sliderMax,
  sliderValue,
  onSliderChange,
  patternBase64,
  pcx,
  pcy,
  segments,
  // --- Forward-sim preview props ---
  simulatedB64 = null,
  simulatedNcc = null,
  simulatedLoading = false,
  simulatedError = null,
  simulatedOrientation = null,
  simulatedOrientationSource = null,
  shtChoices = [],
  shtPath = null,
  onShtPathChange = null,
  previewBandwidth = 128,
  onPreviewBandwidthChange = null,
}) {
  const { t } = useTranslation(['pcrefinement', 'common']);
  const [hideLines, setHideLines] = useState(false);
  const simExport = useImageExport();
  // The two forward-simulation panels: each on its own, or both side by side —
  // the pair is what makes the comparison readable in a document.
  const simExportMenu = (which) => {
    const exp = patternBase64 ? `data:image/png;base64,${patternBase64}` : null;
    const sim = simulatedB64 ? `data:image/png;base64,${simulatedB64}` : null;
    const items = [];
    if (which === 'experimental' && exp) {
      items.push({
        id: 'panel',
        menuLabel: t('imageexport:menuExportThisPanel'),
        build: () => exp,
        name: 'experimental',
        label: t('pcrefinement:preview.experimental'),
      });
    }
    if (which === 'simulated' && sim) {
      items.push({
        id: 'panel',
        menuLabel: t('imageexport:menuExportThisPanel'),
        build: () => sim,
        name: 'simulated',
        label: t('pcrefinement:preview.simulated'),
      });
    }
    if (exp && sim) {
      items.push({
        id: 'both',
        menuLabel: t('imageexport:menuExportBothPanels'),
        build: () => buildPanelSheet([
          { label: t('pcrefinement:preview.experimental'), src: exp },
          { label: t('pcrefinement:preview.simulated'), src: sim },
        ]).then((c) => c.toDataURL('image/png')),
        name: 'experimental-vs-simulated',
        label: t('pcrefinement:preview.forwardSimTitle'),
      });
    }
    return items;
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        padding: spacing.outerMargin,
        boxSizing: 'border-box',
        overflow: 'hidden',
      }}
    >
      {simExport.node}

      {/* Canvas */}
      <div style={{ flexShrink: 0 }}>
        <PatternCanvas
          patternBase64={patternBase64}
          pcx={pcx}
          pcy={pcy}
          hideLines={hideLines}
          segments={segments}
          exportName={t('pcrefinement:canvas.exportName')}
        />
      </div>

      {/* Hide Lines checkbox + Compare button + ? button */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: spacing.innerSpacing,
          marginTop: spacing.innerSpacing,
          flexShrink: 0,
        }}
      >
        <label
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 4,
            fontSize: '9pt',
            color: colors.text,
            cursor: 'pointer',
            userSelect: 'none',
          }}
          title={t('pcrefinement:preview.hideLinesTooltip')}
        >
          <input
            type="checkbox"
            checked={hideLines}
            onChange={(e) => setHideLines(e.target.checked)}
            style={{ accentColor: colors.accent }}
          />
          {t('pcrefinement:preview.hideLines')}
        </label>

        <Button
          small
          title={t('pcrefinement:preview.compareTooltip')}
          style={{ width: 70 }}
        >
          {t('pcrefinement:preview.compare')}
        </Button>

        <Button
          small
          title={t('pcrefinement:preview.helpTooltip')}
          style={{ width: 28, padding: '3px 0', textAlign: 'center' }}
        >
          {t('pcrefinement:preview.help')}
        </Button>
      </div>

      {/* Forward-Simulated preview — independent geometry check. Tracks
          the trial PC/sample-tilt/det-tilt values and re-renders via
          /api/pc/render-preview. NCC vs experimental is shown live so
          the user can sweep geometry parameters and converge visually +
          numerically. Closes the PC/tilt degeneracy that Hough alone
          cannot break. */}
      {(shtChoices.length > 0 || shtPath || simulatedB64 || simulatedLoading || simulatedError) && (
        <div style={{
          flexShrink: 0,
          marginTop: spacing.groupSpacing,
          background: colors.bgSecondary,
          border: `1px solid ${colors.border}`,
          borderRadius: 4,
          padding: '8px 10px',
        }}>
          <div style={{
            display: 'flex', alignItems: 'center', gap: 6,
            fontSize: '9pt', color: colors.accent, fontWeight: 700,
            marginBottom: 6,
          }}>
            <span>{t('pcrefinement:preview.forwardSimTitle')}</span>
            {simulatedLoading && (
              <span style={{ fontSize: '8pt', color: colors.textSecondary, fontWeight: 400 }}>
                {t('pcrefinement:preview.rendering')}
              </span>
            )}
            {!simulatedLoading && typeof simulatedNcc === 'number' && (
              <span style={{
                fontSize: '9pt', fontWeight: 700, marginLeft: 'auto',
                color: simulatedNcc >= 0.3 ? '#50fa7b' :
                       simulatedNcc >= 0.15 ? '#ffb86c' : '#ff5555',
              }}>
                {t('pcrefinement:preview.ncc', { value: simulatedNcc.toFixed(4) })}
              </span>
            )}
          </div>

          {/* SHT picker + bandwidth */}
          <div style={{ display: 'flex', gap: 6, marginBottom: 6, alignItems: 'center', flexWrap: 'wrap' }}>
            <label style={{ fontSize: '8.5pt', color: colors.textSecondary, whiteSpace: 'nowrap' }}>
              {t('pcrefinement:preview.shtLabel')}
            </label>
            <select
              value={shtPath || ''}
              disabled={shtChoices.length === 0}
              onChange={e => onShtPathChange?.(e.target.value)}
              title={t('pcrefinement:preview.shtTooltip')}
              style={{
                flex: 1, minWidth: 120,
                background: colors.bg, color: colors.text,
                border: `1px solid ${colors.border}`, borderRadius: 3,
                padding: '2px 6px', fontSize: '8.5pt',
              }}
            >
              {shtChoices.length === 0 && <option value="">{t('pcrefinement:preview.noShtFiles')}</option>}
              {shtChoices.map(f => (
                <option key={f.path} value={f.path}>
                  {f.display_label || f.filename || f.path}
                </option>
              ))}
            </select>
            <label style={{ fontSize: '8.5pt', color: colors.textSecondary, whiteSpace: 'nowrap' }}>
              {t('pcrefinement:preview.bandwidthLabel')}
            </label>
            <select
              value={previewBandwidth}
              onChange={e => onPreviewBandwidthChange?.(Number(e.target.value))}
              style={{
                background: colors.bg, color: colors.text,
                border: `1px solid ${colors.border}`, borderRadius: 3,
                padding: '2px 6px', fontSize: '8.5pt',
              }}
              title={t('pcrefinement:preview.bandwidthTooltip')}
            >
              <option value={128}>{t('pcrefinement:preview.bandwidthFast')}</option>
              <option value={256}>{t('pcrefinement:preview.bandwidthStandard')}</option>
              <option value={384}>{t('pcrefinement:preview.bandwidthSharp')}</option>
            </select>
          </div>

          {/* Side-by-side experimental | simulated */}
          <div style={{ display: 'flex', gap: 6, justifyContent: 'space-between' }}>
            <div style={{ flex: 1, textAlign: 'center' }}>
              {patternBase64 ? (
                <img
                  alt={t('pcrefinement:preview.experimentalAlt')}
                  src={`data:image/png;base64,${patternBase64}`}
                  onContextMenu={(e) => simExport.openMenu(e, simExportMenu('experimental'))}
                  style={{
                    width: '100%', maxHeight: 130, objectFit: 'contain',
                    border: `1px solid ${colors.border}`, borderRadius: 3,
                    background: '#000',
                  }}
                />
              ) : (
                <div style={{
                  height: 130, display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '8pt', color: colors.textSecondary,
                  border: `1px solid ${colors.border}`, borderRadius: 3, background: colors.bg,
                }}>
                  {t('pcrefinement:preview.noPattern')}
                </div>
              )}
              <div style={{ fontSize: '8pt', color: colors.textSecondary, marginTop: 2 }}>
                {t('pcrefinement:preview.experimental')}
              </div>
            </div>
            <div style={{ flex: 1, textAlign: 'center' }}>
              {simulatedB64 ? (
                <img
                  alt={t('pcrefinement:preview.simulatedAlt')}
                  src={`data:image/png;base64,${simulatedB64}`}
                  onContextMenu={(e) => simExport.openMenu(e, simExportMenu('simulated'))}
                  style={{
                    width: '100%', maxHeight: 130, objectFit: 'contain',
                    border: `1px solid ${colors.border}`, borderRadius: 3,
                    background: '#000',
                  }}
                />
              ) : (
                <div style={{
                  height: 130, display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '8pt', color: simulatedError ? '#ff5555' : colors.textSecondary,
                  border: `1px solid ${colors.border}`, borderRadius: 3, background: colors.bg,
                  padding: 6,
                }}>
                  {simulatedError ? simulatedError : (simulatedLoading ? '…' : t('pcrefinement:preview.pickSht'))}
                </div>
              )}
              <div style={{ fontSize: '8pt', color: colors.textSecondary, marginTop: 2 }}>
                {t('pcrefinement:preview.simulated')}
              </div>
            </div>
          </div>

          {simulatedOrientation && (
            <div style={{
              fontSize: '7.5pt', color: colors.textSecondary,
              marginTop: 4, fontFamily: 'monospace',
              display: 'flex', gap: 6, alignItems: 'baseline', flexWrap: 'wrap',
            }}>
              <span>{t('pcrefinement:preview.euler', { values: simulatedOrientation.map(v => v.toFixed(1)).join(', ') })}</span>
              {simulatedOrientationSource && (
                <span
                  title="Orientation source: hough = Hough indexing (the reliable path here); supplied = your orientation; spherical = SHT-spherical (under investigation); identity = no index found"
                  style={{
                    color: simulatedOrientationSource === 'hough' || simulatedOrientationSource === 'supplied'
                      ? '#50fa7b'
                      : simulatedOrientationSource === 'spherical' ? '#ffb86c' : '#ff5555',
                  }}
                >
                  · {simulatedOrientationSource}
                  {simulatedOrientationSource === 'identity' ? ' ⚠' : ''}
                </span>
              )}
            </div>
          )}
        </div>
      )}

      {/* Slider steps the SELECTED patterns list (not the whole dataset) */}
      <div style={{ flexShrink: 0, marginTop: spacing.innerSpacing }}>
        <input
          type="range"
          min={0}
          max={sliderMax}
          value={sliderValue}
          disabled={patterns.length === 0}
          onChange={(e) => onSliderChange(parseInt(e.target.value, 10))}
          style={{ width: '100%', accentColor: colors.accent, opacity: patterns.length === 0 ? 0.4 : 1 }}
          title={t('pcrefinement:preview.sliderTooltip')}
        />
        <div style={{ fontSize: '8pt', color: colors.textSecondary, textAlign: 'center', marginTop: 2 }}>
          {patterns.length === 0
            ? t('pcrefinement:preview.noPatternsSelected')
            : t('pcrefinement:preview.patternCounter', { current: (selectedIdx ?? 0) + 1, total: patterns.length })}
        </div>
      </div>

      {/* Selected patterns list label */}
      <div
        style={{
          fontSize: '9pt',
          fontWeight: 700,
          color: colors.textSecondary,
          marginTop: spacing.groupSpacing,
          marginBottom: 2,
          flexShrink: 0,
        }}
      >
        {t('pcrefinement:preview.selectedPatterns')}
      </div>

      {/* QListWidget equivalent */}
      <div
        className="thin-scrollbar"
        style={{
          flex: 1,
          minHeight: 80,
          background: colors.bg,
          border: `1px solid ${colors.border}`,
          borderRadius: 3,
          overflowY: 'auto',
        }}
      >
        {patterns.length === 0 ? (
          <div
            style={{
              padding: 8,
              fontSize: '9pt',
              color: colors.textSecondary,
              textAlign: 'center',
              paddingTop: 20,
            }}
          >
            <div style={{ fontSize: '18pt', opacity: 0.3, marginBottom: 4 }}>{'\u2316'}</div>
            {t('pcrefinement:preview.noPatternsAdded')}
            <div style={{ fontSize: '8pt', marginTop: 4, opacity: 0.6 }}>
              {t('pcrefinement:preview.noPatternsHint')}
            </div>
          </div>
        ) : (
          patterns.map((p, i) => (
            <div
              key={i}
              className="list-item-interactive"
              onClick={() => onSelectIdx(i === selectedIdx ? null : i)}
              title={p.ci != null
                ? t('pcrefinement:preview.patternItemTooltipCi', { row: p.row, col: p.col, ci: p.ci.toFixed(3) })
                : t('pcrefinement:preview.patternItemTooltip', { row: p.row, col: p.col })}
              style={{
                padding: '4px 8px',
                fontSize: '9pt',
                cursor: 'pointer',
                background: selectedIdx === i ? alpha(colors.purple, 20) : 'transparent',
                color: selectedIdx === i ? colors.purple : colors.text,
                borderBottom:
                  i < patterns.length - 1 ? `1px solid ${alpha(colors.border, 27)}` : 'none',
                borderLeft: selectedIdx === i ? `3px solid ${colors.purple}` : '3px solid transparent',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                transition: 'background 0.1s, border-left-color 0.15s',
              }}
            >
              <span>{t('pcrefinement:preview.patternItemLabel', { row: p.row, col: p.col })}</span>
              {p.ci != null && (
                <span style={{
                  color: ciColor(p.ci),
                  fontFamily: 'monospace',
                  fontSize: '8pt',
                  fontWeight: 600,
                }}>
                  {t('pcrefinement:preview.patternItemCi', { ci: p.ci.toFixed(3) })}
                </span>
              )}
            </div>
          ))
        )}
      </div>

      {/* Remove button — patterns are added from the EBSD Viewer */}
      <div
        style={{
          display: 'flex',
          gap: spacing.innerSpacing,
          marginTop: spacing.innerSpacing,
          flexShrink: 0,
        }}
      >
        <Button
          small
          variant="danger"
          disabled={selectedIdx === null}
          style={{ flex: 1 }}
          title={t('pcrefinement:preview.removeTooltip')}
          onClick={onRemovePattern}
        >
          {t('common:remove')}
        </Button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Pixel-wise PC Correction group — matches _create_pixelwise_pc_group()
// ---------------------------------------------------------------------------
function PixelwisePCGroup({ canRun }) {
  const { t } = useTranslation('pcrefinement');
  const [mode, setMode] = useState('Global (single PC)');
  const [gridStep, setGridStep] = useState(10);
  const [calibrated, setCalibrated] = useState(false);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [pcResult, setPcResult] = useState(null);
  const [showMap, setShowMap] = useState(false);
  const [msg, setMsg] = useState('');

  const gs = useDataStore.getState().gridShape || [90, 120];
  const estimatedPoints = gridStep > 0
    ? Math.floor((gs[0] / gridStep) * (gs[1] / gridStep))
    : 0;

  const handleAutoStep = () => {
    const totalPixels = gs[0] * gs[1];
    const suggested = Math.max(2, Math.round(Math.sqrt(totalPixels / 50)));
    setGridStep(suggested);
  };

  const handleRunCalibration = async () => {
    const backendMode = mode === 'Extrapolate from Points' ? 'extrapolate' : 'plane';
    setRunning(true);
    setProgress(0);
    setPcResult(null);
    setShowMap(false);
    setMsg(t('pcrefinement:pixelwise.refiningMsg', { count: estimatedPoints }));
    try {
      const { data } = await pcApi.runGridCalibration(gridStep, backendMode);
      const taskId = data.task_id;
      // Poll the background task until it completes/fails.
      // eslint-disable-next-line no-constant-condition
      while (true) {
        await new Promise((r) => setTimeout(r, 1000));
        const { data: st } = await pcApi.getOptimizeStatus(taskId);
        if (typeof st.progress === 'number') setProgress(st.progress);
        if (st.status === 'completed') {
          setCalibrated(true);
          setPcResult(st.result);
          const s = st.result?.pc_stats;
          setMsg(
            t('pcrefinement:pixelwise.doneMsg', {
              valid: st.result?.n_valid_points,
              total: st.result?.n_calibration_points,
            }) +
            (s ? t('pcrefinement:pixelwise.spanMsg', {
              x: s.pcx.span.toFixed(4), y: s.pcy.span.toFixed(4), z: s.pcz.span.toFixed(4),
            }) : '')
          );
          break;
        }
        if (st.status === 'failed') {
          setMsg(t('pcrefinement:pixelwise.errorMsg', { detail: st.error || t('pcrefinement:pixelwise.calibrationFailed') }));
          break;
        }
      }
    } catch (err) {
      setMsg(t('pcrefinement:pixelwise.errorMsg', { detail: err.response?.data?.detail || err.message }));
    } finally {
      setRunning(false);
    }
  };

  const handleShowPcMap = () => {
    setShowMap((v) => !v);
  };

  return (
    <GroupBox title={t('pcrefinement:pixelwise.title')}>
      <FormRow label={t('pcrefinement:pixelwise.modeLabel')}>
        <Select
          value={mode}
          onChange={(e) => setMode(e.target.value)}
          options={[
            { value: 'Global (single PC)', label: t('pcrefinement:pixelwise.modeGlobal') },
            { value: 'Grid Calibration (fit_pc)', label: t('pcrefinement:pixelwise.modeGrid') },
            { value: 'Extrapolate from Points', label: t('pcrefinement:pixelwise.modeExtrapolate') },
          ]}
          title={t('pcrefinement:pixelwise.modeTooltip')}
        />
      </FormRow>

      <FormRow label={t('pcrefinement:pixelwise.gridStepLabel')}>
        <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
          <NumberInput
            value={gridStep}
            onChange={(e) => setGridStep(parseInt(e.target.value, 10) || 2)}
            min={2}
            max={100}
            step={1}
            style={{ width: 60 }}
            title={t('pcrefinement:pixelwise.gridStepTooltip')}
          />
          <Button
            small
            style={{ width: 50 }}
            onClick={handleAutoStep}
            title={t('pcrefinement:pixelwise.autoTooltip')}
          >
            {t('pcrefinement:pixelwise.auto')}
          </Button>
        </div>
      </FormRow>

      <div
        style={{
          fontSize: '8pt',
          color: colors.textSecondary,
          fontStyle: 'italic',
          marginBottom: spacing.innerSpacing,
          marginLeft: 84,
        }}
      >
        {t('pcrefinement:pixelwise.pointsEstimate', { count: estimatedPoints, step: gridStep })}
      </div>

      <FormRow label={t('pcrefinement:pixelwise.statusLabel')}>
        <Label
          secondary
          style={{
            color: calibrated ? colors.green : colors.red,
            fontSize: '9pt',
          }}
        >
          {calibrated ? t('pcrefinement:pixelwise.calibrated') : t('pcrefinement:pixelwise.notCalibrated')}
        </Label>
      </FormRow>

      <Button
        disabled={mode === 'Global (single PC)' || running || !canRun}
        onClick={handleRunCalibration}
        title={t('pcrefinement:pixelwise.runCalibrationTooltip')}
        style={{ width: '100%', marginBottom: spacing.innerSpacing }}
      >
        {running ? t('pcrefinement:pixelwise.calibrating', { percent: Math.round(progress * 100) }) : t('pcrefinement:pixelwise.runCalibration')}
      </Button>

      {running && (
        <div
          style={{
            height: 6,
            background: colors.bgSecondary,
            border: `1px solid ${colors.border}`,
            borderRadius: 3,
            marginBottom: spacing.innerSpacing,
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              height: '100%',
              width: `${Math.round(progress * 100)}%`,
              background: colors.accent || colors.green,
              transition: 'width 0.3s ease',
            }}
          />
        </div>
      )}

      <Button
        disabled={!calibrated}
        onClick={handleShowPcMap}
        title={t('pcrefinement:pixelwise.pcMapTooltip')}
        style={{ width: '100%', marginBottom: spacing.innerSpacing }}
      >
        {showMap ? t('pcrefinement:pixelwise.hidePcMap') : t('pcrefinement:pixelwise.showPcMap')}
      </Button>

      {showMap && pcResult?.pc_stats && (
        <div
          style={{
            fontSize: '8.5pt',
            color: colors.textSecondary,
            padding: '5px 7px',
            background: colors.bgSecondary,
            border: `1px solid ${colors.border}`,
            borderRadius: 3,
            marginBottom: spacing.innerSpacing,
            fontFamily: 'monospace',
          }}
        >
          <div style={{ color: colors.text, marginBottom: 3 }}>
            {t('pcrefinement:pixelwise.pcFieldHeader', {
              rows: pcResult.grid_shape?.[0],
              cols: pcResult.grid_shape?.[1],
              mode: pcResult.mode,
            })}
          </div>
          {['pcx', 'pcy', 'pcz'].map((k) => {
            const s = pcResult.pc_stats[k];
            return (
              <div key={k}>
                {t('pcrefinement:pixelwise.pcFieldRow', {
                  component: k,
                  min: s.min.toFixed(4),
                  max: s.max.toFixed(4),
                  mean: s.mean.toFixed(4),
                  span: s.span.toFixed(4),
                })}
              </div>
            );
          })}
        </div>
      )}

      {msg && (
        <div
          style={{
            fontSize: '9pt',
            color: colors.textSecondary,
            padding: '3px 6px',
            background: colors.bgSecondary,
            border: `1px solid ${colors.border}`,
            borderRadius: 3,
            wordBreak: 'break-word',
          }}
        >
          {msg}
        </div>
      )}
    </GroupBox>
  );
}

// ---------------------------------------------------------------------------
// Indexing Settings group — matches _create_indexing_settings_group()
// ---------------------------------------------------------------------------
function IndexingSettingsGroup({ onParamsChange }) {
  const { t } = useTranslation('pcrefinement');
  const [minD, setMinD] = useState(1.0);
  const [fThreshold, setFThreshold] = useState(0.1);
  const [maxReflectors, setMaxReflectors] = useState(65);
  const [nBands, setNBands] = useState(12);
  const [method, setMethod] = useState('Nelder-Mead');
  const [searchLimit, setSearchLimit] = useState(0.05);

  const notify = useCallback(() => {
    onParamsChange({ minD, fThreshold, maxReflectors, nBands, method, searchLimit });
  }, [minD, fThreshold, maxReflectors, nBands, method, searchLimit, onParamsChange]);

  useEffect(() => { notify(); }, [notify]);

  return (
    <GroupBox title={t('pcrefinement:indexingSettings.title')}>
      <FormRow label={t('pcrefinement:indexingSettings.minDLabel')}>
        <NumberInput
          value={minD}
          onChange={(e) => setMinD(parseFloat(e.target.value))}
          min={0.1}
          max={10.0}
          step={0.1}
          title={t('pcrefinement:indexingSettings.minDTooltip')}
        />
      </FormRow>

      <FormRow label={t('pcrefinement:indexingSettings.fThresholdLabel')}>
        <NumberInput
          value={fThreshold}
          onChange={(e) => setFThreshold(parseFloat(e.target.value))}
          min={0.0}
          max={1.0}
          step={0.05}
          title={t('pcrefinement:indexingSettings.fThresholdTooltip')}
        />
      </FormRow>

      <FormRow label={t('pcrefinement:indexingSettings.maxReflectorsLabel')}>
        <NumberInput
          value={maxReflectors}
          onChange={(e) => setMaxReflectors(parseInt(e.target.value, 10))}
          min={1}
          max={100}
          step={1}
          title={t('pcrefinement:indexingSettings.maxReflectorsTooltip')}
        />
      </FormRow>

      <FormRow label={t('pcrefinement:indexingSettings.nBandsLabel')}>
        <NumberInput
          value={nBands}
          onChange={(e) => setNBands(parseInt(e.target.value, 10))}
          min={1}
          max={50}
          step={1}
          title={t('pcrefinement:indexingSettings.nBandsTooltip')}
        />
      </FormRow>

      <Separator />

      <FormRow label={t('pcrefinement:indexingSettings.methodLabel')}>
        <Select
          value={method}
          onChange={(e) => setMethod(e.target.value)}
          options={['Nelder-Mead', 'PSO']}
          title={t('pcrefinement:indexingSettings.methodTooltip')}
        />
      </FormRow>

      {method === 'PSO' && (
        <WarnLabel>{t('pcrefinement:indexingSettings.psoWarn')}</WarnLabel>
      )}

      <FormRow label={t('pcrefinement:indexingSettings.searchLimitLabel')}>
        <NumberInput
          value={searchLimit}
          onChange={(e) => setSearchLimit(parseFloat(e.target.value))}
          min={0.01}
          max={1.0}
          step={0.01}
          title={t('pcrefinement:indexingSettings.searchLimitTooltip')}
        />
      </FormRow>
    </GroupBox>
  );
}

// ---------------------------------------------------------------------------
// Pixel Size & Binning sub-group — matches _create_pixel_binning_group()
// ---------------------------------------------------------------------------
function PixelBinningGroup({ binning, setBinning, detWidthMm, setDetWidthMm, shapeW }) {
  const { t } = useTranslation('pcrefinement');
  const binVal = parseInt(binning, 10) || 1;
  const unbinnedW = (parseInt(shapeW, 10) || 0) * binVal;
  const detWidthMmVal = parseFloat(detWidthMm) || 0;
  const binnedPx = unbinnedW > 0 ? (detWidthMmVal * 1000 / unbinnedW).toFixed(2) : '—';

  const handleApplyPixelSize = async () => {
    try {
      await pcApi.updateParams({
        binning: binVal,
        detector_width_mm: detWidthMmVal,
      });
    } catch { /* ignore */ }
  };

  return (
    <GroupBox title={t('pcrefinement:pixelBinning.title')} style={{ marginBottom: spacing.innerSpacing }}>
      <FormRow label={t('pcrefinement:pixelBinning.binningLabel')}>
        <NumberInput
          value={binning}
          onChange={(e) => setBinning(e.target.value)}
          min={1}
          max={100}
          step={1}
          title={t('pcrefinement:pixelBinning.binningTooltip')}
        />
      </FormRow>

      <FormRow label={t('pcrefinement:pixelBinning.unbinnedWLabel')}>
        <Label secondary style={{ fontSize: '9pt', fontFamily: 'monospace' }}>
          {unbinnedW || '—'}
        </Label>
      </FormRow>

      <FormRow label={t('pcrefinement:pixelBinning.detectorWLabel')}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <NumberInput
            value={detWidthMm}
            onChange={(e) => setDetWidthMm(e.target.value)}
            min={0.1}
            max={1000}
            step={0.001}
            title={t('pcrefinement:pixelBinning.detectorWTooltip')}
          />
          <Label secondary style={{ fontSize: '9pt', flexShrink: 0 }}>{t('pcrefinement:pixelBinning.mm')}</Label>
        </div>
      </FormRow>

      <FormRow label={t('pcrefinement:pixelBinning.binnedPxLabel')}>
        <Label secondary style={{ fontSize: '9pt', fontFamily: 'monospace' }}>
          {binnedPx}
        </Label>
      </FormRow>

      <Button
        style={{ width: '100%' }}
        onClick={handleApplyPixelSize}
        title={t('pcrefinement:pixelBinning.applyPixelSizeTooltip')}
      >
        {t('pcrefinement:pixelBinning.applyPixelSize')}
      </Button>
    </GroupBox>
  );
}

// ---------------------------------------------------------------------------
// Detector Settings group — matches _create_detector_settings_group()
// ---------------------------------------------------------------------------
function DetectorSettingsGroup({ onDetectorApplied, onPcChanged, onTiltChanged, externalPc }) {
  const { t } = useTranslation('pcrefinement');
  const [shapeH, setShapeH] = useState('');
  const [shapeW, setShapeW] = useState('');
  const [pcx, setPcx] = useState('0.5');
  const [pcy, setPcy] = useState('0.5');
  const [pcz, setPcz] = useState('0.5');
  const [sampleTilt, setSampleTilt] = useState('70.0');
  const [detTilt, setDetTilt] = useState('0.0');
  const [azimuthal, setAzimuthal] = useState('0.0');
  const [binning, setBinning] = useState('1');
  const [detWidthMm, setDetWidthMm] = useState('0.1');
  const [detectorText, setDetectorText] = useState('');
  const [axesText, setAxesText] = useState('');
  const [msg, setMsg] = useState(null);
  const [isError, setIsError] = useState(false);

  // Debounce timers for reactive PC/tilt changes
  const pcTimerRef = useRef(null);
  const tiltTimerRef = useRef(null);
  const initialLoadRef = useRef(true);

  // Reactive PC change — mirrors PyQt5 apply_detector_pc()
  // Debounced 500ms so rapid spinbox clicks don't flood the backend
  const triggerPcChange = useCallback((x, y, z) => {
    if (pcTimerRef.current) clearTimeout(pcTimerRef.current);
    pcTimerRef.current = setTimeout(() => {
      const px = parseFloat(x), py = parseFloat(y), pz = parseFloat(z);
      if (!isNaN(px) && !isNaN(py) && !isNaN(pz) && onPcChanged) {
        onPcChanged(px, py, pz);
      }
    }, 500);
  }, [onPcChanged]);

  // Reactive tilt change — mirrors PyQt5 apply_tilt_settings()
  const triggerTiltChange = useCallback((st, dt, az) => {
    if (tiltTimerRef.current) clearTimeout(tiltTimerRef.current);
    tiltTimerRef.current = setTimeout(() => {
      const s = parseFloat(st), d = parseFloat(dt), a = parseFloat(az);
      if (!isNaN(s) && !isNaN(d) && !isNaN(a) && onTiltChanged) {
        onTiltChanged(s, d, a);
      }
    }, 500);
  }, [onTiltChanged]);

  // PC setters that also trigger reactive update
  const handlePcxChange = (e) => { setPcx(e.target.value); triggerPcChange(e.target.value, pcy, pcz); };
  const handlePcyChange = (e) => { setPcy(e.target.value); triggerPcChange(pcx, e.target.value, pcz); };
  const handlePczChange = (e) => { setPcz(e.target.value); triggerPcChange(pcx, pcy, e.target.value); };

  // Tilt setters that also trigger reactive update
  const handleSampleTiltChange = (e) => { setSampleTilt(e.target.value); triggerTiltChange(e.target.value, detTilt, azimuthal); };
  const handleDetTiltChange = (e) => { setDetTilt(e.target.value); triggerTiltChange(sampleTilt, e.target.value, azimuthal); };
  const handleAzimuthalChange = (e) => { setAzimuthal(e.target.value); triggerTiltChange(sampleTilt, detTilt, e.target.value); };

  // Load from file metadata on mount AND on file change. Depending on
  // `filePath` (not just `ebsdLoaded`) is what makes the second file
  // refresh the displayed detector values; with `[ebsdLoaded]` alone
  // the flag stayed true between loads and the effect never re-fired,
  // so the panel kept showing the previous file's sample_tilt + tilt
  // (bug fix 2026-05-22).
  const { ebsdLoaded, filePath } = useDataStore();
  useEffect(() => {
    initialLoadRef.current = true;
    ebsdApi.getDetector().then((res) => {
      const d = res.data;
      if (!d) return;
      if (d.shape) { setShapeH(String(d.shape[0])); setShapeW(String(d.shape[1])); }
      if (d.pc) { setPcx(String(d.pc[0])); setPcy(String(d.pc[1])); setPcz(String(d.pc[2])); }
      if (d.sample_tilt != null) setSampleTilt(String(Math.round(d.sample_tilt * 100) / 100));
      if (d.camera_tilt != null) setDetTilt(String(d.camera_tilt));
      if (d.azimuthal != null) setAzimuthal(String(d.azimuthal));
      if (d.binning != null) setBinning(String(d.binning));
      // Compute detector width from pixel_size (µm) and shape
      if (d.pixel_size != null && d.shape && d.shape.length >= 2) {
        const bin = d.binning || 1;
        const unbinnedW = d.shape[1] * bin;
        const widthMm = (d.pixel_size * unbinnedW) / 1000;
        setDetWidthMm(String(widthMm.toFixed(3)));
      }
      if (d.repr) setDetectorText(d.repr);
      if (d.axes_repr) setAxesText(d.axes_repr);
      // Notify parent of initial PC values. Pass camera_tilt too so the
      // forward-sim preview uses the REAL detector tilt instead of its 0.0
      // default — otherwise the preview is rotated by exactly camera_tilt
      // relative to the Single-Pixel Phase Test, which always uses the loaded
      // detector tilt (root-caused 2026-06-27).
      if (d.pc) {
        onDetectorApplied(d.pc, d.sample_tilt || 70, d.camera_tilt);
      }
      initialLoadRef.current = false;
    }).catch(() => { initialLoadRef.current = false; });
  }, [ebsdLoaded, filePath]); // eslint-disable-line react-hooks/exhaustive-deps

  // Update PC spinboxes when optimization changes PC externally
  useEffect(() => {
    if (externalPc && externalPc.length === 3) {
      setPcx(String(externalPc[0]));
      setPcy(String(externalPc[1]));
      setPcz(String(externalPc[2]));
    }
  }, [externalPc]);

  const sampleTiltNum = parseFloat(sampleTilt);
  const detTiltNum = parseFloat(detTilt);
  const showSampleTiltWarn = !isNaN(sampleTiltNum) && Math.abs(sampleTiltNum - 70) > 0.5;
  const showDetTiltWarn = !isNaN(detTiltNum) && (detTiltNum < 0 || detTiltNum > 20);

  const handleApply = async () => {
    setMsg(null);
    try {
      const shape = [parseInt(shapeH, 10), parseInt(shapeW, 10)];
      const pc = [parseFloat(pcx), parseFloat(pcy), parseFloat(pcz)];
      const res = await pcApi.setDetector(shape, pc, {
        sampleTilt: parseFloat(sampleTilt),
        cameraTilt: parseFloat(detTilt),
        binning: parseInt(binning, 10),
        detectorTilt: parseFloat(detTilt),
        azimuthal: parseFloat(azimuthal),
      });
      setMsg(t('pcrefinement:detector.detectorSet'));
      setIsError(false);
      if (res.data?.repr) setDetectorText(res.data.repr);
      if (res.data?.axes_repr) setAxesText(res.data.axes_repr);
      onDetectorApplied(pc, parseFloat(sampleTilt), parseFloat(detTilt));
    } catch (err) {
      setMsg(err.response?.data?.detail || err.message || t('pcrefinement:detector.failed'));
      setIsError(true);
    }
  };

  return (
    <GroupBox title={t('pcrefinement:detector.title')}>
      <FormRow label={t('pcrefinement:detector.pcXLabel')}>
        <NumberInput
          value={pcx}
          onChange={handlePcxChange}
          min={0.0} max={1.0} step={0.01}
          title={t('pcrefinement:detector.pcXTooltip')}
        />
      </FormRow>

      <FormRow label={t('pcrefinement:detector.pcYLabel')}>
        <NumberInput
          value={pcy}
          onChange={handlePcyChange}
          min={0.0} max={1.0} step={0.01}
          title={t('pcrefinement:detector.pcYTooltip')}
        />
      </FormRow>

      <FormRow label={t('pcrefinement:detector.pcZLabel')}>
        <NumberInput
          value={pcz}
          onChange={handlePczChange}
          min={0.0} max={1.0} step={0.01}
          title={t('pcrefinement:detector.pcZTooltip')}
        />
      </FormRow>

      {/* Pixel Size & Binning sub-group */}
      <PixelBinningGroup
        binning={binning}
        setBinning={setBinning}
        detWidthMm={detWidthMm}
        setDetWidthMm={setDetWidthMm}
        shapeW={shapeW}
      />

      {/* Sample tilt */}
      <FormRow label={t('pcrefinement:detector.sampleTiltLabel')}>
        <div>
          <NumberInput
            value={sampleTilt}
            onChange={handleSampleTiltChange}
            min={0} max={90} step={0.5}
            title={t('pcrefinement:detector.sampleTiltTooltip')}
          />
          {showSampleTiltWarn && (
            <WarnLabel>{t('pcrefinement:detector.sampleTiltWarn')}</WarnLabel>
          )}
        </div>
      </FormRow>

      {/* Detector tilt */}
      <FormRow label={t('pcrefinement:detector.detTiltLabel')}>
        <div>
          <NumberInput
            value={detTilt}
            onChange={handleDetTiltChange}
            min={0.0} max={45.0} step={0.01}
            title={t('pcrefinement:detector.detTiltTooltip')}
          />
          {showDetTiltWarn && (
            <WarnLabel>{t('pcrefinement:detector.detTiltWarn')}</WarnLabel>
          )}
        </div>
      </FormRow>

      {/* Azimuthal */}
      <FormRow label={t('pcrefinement:detector.azimuthalLabel')}>
        <NumberInput
          value={azimuthal}
          onChange={handleAzimuthalChange}
          min={0.0} max={360.0} step={1.0}
          title={t('pcrefinement:detector.azimuthalTooltip')}
        />
      </FormRow>

      {/* Detector display (read-only) */}
      <FormRow label={t('pcrefinement:detector.detectorLabel')}>
        <MonoDisplay value={detectorText} rows={3} title={t('pcrefinement:hoverTips.detectorRepr')} />
      </FormRow>

      <div
        style={{
          fontSize: '9pt',
          color: colors.textSecondary,
          marginBottom: 2,
        }}
      >
        {t('pcrefinement:detector.axesManagerLabel')}
      </div>
      <MonoDisplay value={axesText} rows={3} title={t('pcrefinement:hoverTips.axesRepr')} />

      <Button
        variant="default"
        style={{ width: '100%', marginTop: spacing.innerSpacing }}
        onClick={handleApply}
        title={t('pcrefinement:detector.applyDetectorTooltip')}
      >
        {t('pcrefinement:detector.applyDetector')}
      </Button>

      <StatusMsg msg={msg} isError={isError} />
    </GroupBox>
  );
}

// ---------------------------------------------------------------------------
// Right controls panel — matches _create_controls_panel()
// ---------------------------------------------------------------------------
function ControlsPanel({
  phaseLoaded,
  detectorReady,
  patterns,
  currentPatternIdx,
  onLoadPhase,
  onPhaseLoaded,
  onDetectorApplied,
  onIndexingParamsChange,
  ciValue,
  setCiValue,
  globalCi,
  setGlobalCi,
  onSegmentsUpdate,
  onPatternsIndexed,
  onPcChanged,
  onTiltChanged,
  externalPc,
  onOptimizedPc,
  parentName,
  activeDatasetName,
  propagated,
  setPropagated,
}) {
  const { t } = useTranslation(['pcrefinement', 'common']);
  // Progress / cancel
  const [progress, setProgress] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [taskId, setTaskId] = useState(null);
  const pollRef = useRef(null);

  // Phase state
  const [cifPath, setCifPath] = useState('');
  const [phaseInfo, setPhaseInfo] = useState(null);
  const [phaseMsg, setPhaseMsg] = useState(null);
  const [phaseError, setPhaseError] = useState(false);
  const [phaseLoading, setPhaseLoading] = useState(false);

  // Phase library picker (reuses the Hough indexing CIF picker)
  const [phasePickerOpen, setPhasePickerOpen] = useState(false);
  const [discoveredFiles, setDiscoveredFiles] = useState([]);
  const [discoveredGroups, setDiscoveredGroups] = useState([]);

  // Index pattern state
  const [indexMsg, setIndexMsg] = useState(null);
  const [indexError, setIndexError] = useState(false);

  // Global refine state
  const [globalMsg, setGlobalMsg] = useState(null);
  const [globalError, setGlobalError] = useState(false);

  // Drift state — calibration-pattern based (existing) + source-h5oina based (new)
  const [driftMsg, setDriftMsg] = useState(null);
  const [sourceDriftLoading, setSourceDriftLoading] = useState(false);
  const [sourceDriftImage, setSourceDriftImage] = useState(null);
  const [sourceDriftStats, setSourceDriftStats] = useState(null);
  const [sourceDriftError, setSourceDriftError] = useState(null);
  const [seedApplying, setSeedApplying] = useState(false);
  const [seedMsg, setSeedMsg] = useState(null);
  const [seedError, setSeedError] = useState(false);

  // Results
  const [result, setResult] = useState(null);
  const [resultMsg, setResultMsg] = useState(null);
  const [pcCopied, setPcCopied] = useState(false);

  // Indexing params forwarded from IndexingSettingsGroup
  const paramsRef = useRef({ method: 'Nelder-Mead', searchLimit: 0.05 });
  const handleParamsChange = useCallback((p) => {
    paramsRef.current = p;
    onIndexingParamsChange(p);
  }, [onIndexingParamsChange]);

  const stopPolling = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  };

  const pollFailCountRef = useRef(0);
  useEffect(() => {
    if (!taskId) return;
    pollFailCountRef.current = 0;
    pollRef.current = setInterval(async () => {
      try {
        const res = await pcApi.getOptimizeStatus(taskId);
        pollFailCountRef.current = 0; // reset on success
        const d = res.data;
        if (d.progress !== undefined) setProgress(d.progress);
        if (d.status === 'done' || d.status === 'completed') {
          stopPolling();
          setIsRunning(false);
          setTaskId(null);
          setProgress(100);
          setResult(d.result || d);
          setResultMsg(t('pcrefinement:controls.optimizationComplete'));
          if (d.result?.ci !== undefined) { setCiValue(d.result.ci); setGlobalCi(d.result.ci); }
          // Update segments + detector PC spinboxes with optimized values
          if (d.result?.segments && onSegmentsUpdate) onSegmentsUpdate(d.result.segments);
          if (d.result?.mean_pc?.length === 3 && onOptimizedPc) onOptimizedPc(d.result.mean_pc);
        } else if (d.status === 'error' || d.status === 'failed') {
          stopPolling();
          setIsRunning(false);
          setTaskId(null);
          setResultMsg(d.error || d.message || t('pcrefinement:controls.optimizationFailed'));
          setGlobalError(true);
        }
      } catch {
        pollFailCountRef.current += 1;
        // If backend is unreachable for 5+ consecutive polls, abort
        if (pollFailCountRef.current >= 5) {
          stopPolling();
          setIsRunning(false);
          setTaskId(null);
          setGlobalMsg(t('pcrefinement:controls.backendConnectionLost'));
          setGlobalError(true);
        }
      }
    }, 1000);
    return stopPolling;
  }, [taskId, setCiValue, setGlobalCi, t]);

  // Load Phase from CIF
  const handleLoadPhase = async (pathOverride) => {
    const path = (pathOverride || cifPath || '').trim();
    if (!path) {
      // open file dialog or prompt
      let selected = null;
      if (window.electronAPI?.openFile) {
        selected = await window.electronAPI.openFile({
          filters: [{ name: 'CIF Files', extensions: ['cif'] }, { name: 'All Files', extensions: ['*'] }],
        });
      } else {
        askPrompt({
          title: t('pcrefinement:prompt.loadCifTitle'),
          message: t('pcrefinement:prompt.loadCifMessage'),
          defaultValue: cifPath,
          placeholder: t('pcrefinement:prompt.loadCifPlaceholder'),
          submitLabel: t('pcrefinement:prompt.loadCifSubmit'),
          onSubmit: (p) => {
            if (p.trim()) { setCifPath(p.trim()); handleLoadPhase(p.trim()); }
          },
        });
        return;
      }
      if (selected && selected.trim()) {
        setCifPath(selected.trim());
        // Auto-load immediately after browse
        return handleLoadPhase(selected.trim());
      }
      return;
    }
    setCifPath(path);
    setPhaseLoading(true);
    setPhaseMsg(null);
    try {
      const res = await pcApi.loadPhase(path);
      setPhaseInfo(res.data);
      setPhaseMsg(t('pcrefinement:controls.phaseLoaded', { name: res.data.phase_name || res.data.name || t('pcrefinement:controls.phaseInfoUnknown') }));
      setPhaseError(false);
      onPhaseLoaded(res.data);
    } catch (err) {
      setPhaseMsg(err.response?.data?.detail || err.message || t('pcrefinement:controls.phaseLoadFailed'));
      setPhaseError(true);
    } finally {
      setPhaseLoading(false);
    }
  };

  // ── Phase library picker (same picker component as Hough indexing) ──
  const openPhasePicker = useCallback(async () => {
    setPhasePickerOpen(true);
    try {
      // 'hough' discovery == the CIF library; the PC hint only matters for
      // dictionary ranking, so pass null for CIFs.
      const r = await indexApi.discoverFiles('hough', '', null);
      setDiscoveredFiles(r.data?.files || []);
      setDiscoveredGroups(r.data?.groups || []);
    } catch {
      setDiscoveredFiles([]);
      setDiscoveredGroups([]);
    }
  }, []);

  // PC Refinement loads exactly ONE phase, so a click loads that CIF and
  // closes the picker. `null` is PhaseDropdown's "+ Add file manually…" signal.
  const handlePickPhase = (file) => {
    setPhasePickerOpen(false);
    if (!file) { handleLoadPhase(); return; }   // manual native/prompt fallback
    setCifPath(file.path);
    handleLoadPhase(file.path);
  };

  // The shared picker's "All" button is multi-select; collapse to single-load.
  const handleSetAllPhases = (paths) => {
    if (!paths || paths.length === 0) return;
    const p = paths[paths.length - 1];
    handlePickPhase(discoveredFiles.find((d) => d.path === p) || { path: p });
  };

  // Index All Patterns (Hough index all calibration patterns — mirrors PyQt5 IndexAllWorker)
  const handleIndexAll = async () => {
    if (patterns.length === 0) {
      setIndexMsg(t('pcrefinement:controls.addPatternsFirst'));
      setIndexError(true);
      return;
    }
    setIsRunning(true);
    setProgress(0);
    setIndexMsg(t('pcrefinement:controls.indexingN', { count: patterns.length }));
    setIndexError(false);
    try {
      const res = await pcApi.indexAll();
      const d = res.data;
      setIsRunning(false);
      setProgress(100);

      // Update global CI
      if (d.global_ci != null) {
        setGlobalCi(d.global_ci);
      }

      // Update per-pattern CI in pattern list
      if (d.results && onPatternsIndexed) {
        onPatternsIndexed(d.results);
      }

      // Show overlay for currently selected pattern (or first if none selected)
      if (d.results && d.results.length > 0) {
        const targetIdx = currentPatternIdx != null ? currentPatternIdx : 0;
        const currentResult = d.results.find(r => r.index === targetIdx) || d.results[0];
        if (currentResult.ci != null) {
          setCiValue(currentResult.ci);
        }
        if (currentResult.segments && onSegmentsUpdate) {
          onSegmentsUpdate(currentResult.segments);
        }
      }

      const nFailed = (d.results?.length || 0) - (d.n_indexed || 0);
      const failNote = nFailed > 0 ? t('pcrefinement:controls.indexFailedNote', { count: nFailed }) : '';
      setIndexMsg(t('pcrefinement:controls.indexedAllResult', {
        count: d.n_indexed,
        failNote,
        globalCi: d.global_ci?.toFixed(4) || t('pcrefinement:header.notAvailable'),
      }));
    } catch (err) {
      setIsRunning(false);
      const detail = err.response?.data?.detail;
      const msg = typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : err.message || t('pcrefinement:controls.indexFailed');
      setIndexMsg(msg);
      setIndexError(true);
    }
  };

  // Index single pattern (selected calibration pattern from list)
  const handleIndexPattern = async () => {
    // Use the selected calibration pattern, not the slider position
    const pat = currentPatternIdx != null ? patterns[currentPatternIdx] : null;
    if (!pat) {
      setIndexMsg(t('pcrefinement:controls.selectPatternFirst'));
      setIndexError(true);
      return;
    }
    setIsRunning(true);
    setProgress(0);
    setIndexMsg(t('pcrefinement:controls.indexing'));
    setIndexError(false);
    try {
      const res = await pcApi.indexPattern(pat.row, pat.col);
      const d = res.data;
      setIsRunning(false);
      setProgress(100);

      if (d.ci != null) {
        setCiValue(d.ci);
        setIndexMsg(t('pcrefinement:controls.indexedResult', {
          ci: d.ci.toFixed(4),
          phase: d.phase_name ? ` — ${d.phase_name}` : '',
          bands: d.n_bands,
        }));
      } else {
        setIndexMsg(t('pcrefinement:controls.indexComplete'));
      }

      if (d.segments && onSegmentsUpdate) {
        onSegmentsUpdate(d.segments);
      }

      // Update CI in the pattern list (mirrors handlePatternsIndexed for Index All)
      if (d.ci != null && currentPatternIdx != null && onPatternsIndexed) {
        onPatternsIndexed([{ index: patterns[currentPatternIdx].index, ci: d.ci }]);
      }
    } catch (err) {
      setIsRunning(false);
      const detail = err.response?.data?.detail;
      const msg = typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : err.message || t('pcrefinement:controls.indexFailed');
      setIndexMsg(msg);
      setIndexError(true);
    }
  };

  // Global PC Refine
  const handleGlobalRefine = async () => {
    if (patterns.length === 0) {
      setGlobalMsg(t('pcrefinement:controls.addPatternsFirst'));
      setGlobalError(true);
      return;
    }
    setIsRunning(true);
    setProgress(0);
    setGlobalMsg(t('pcrefinement:controls.runningGlobalRefine'));
    setGlobalError(false);
    const p = paramsRef.current;
    const gs = useDataStore.getState().gridShape || [1, 1];
    const nCols = gs[1] || 1;
    const flatIndices = patterns.map((pt) => pt.row * nCols + pt.col);
    try {
      const res = await pcApi.optimize(flatIndices, p.method || 'Nelder-Mead', p.searchLimit || 0.05);
      const d = res.data;
      if (d.task_id) {
        setTaskId(d.task_id);
        setGlobalMsg(t('pcrefinement:controls.globalRefineStarted', { count: patterns.length }));
      } else {
        setIsRunning(false);
        setProgress(100);
        setResult(d.result || d);
        setGlobalMsg(t('pcrefinement:controls.globalRefineComplete'));
        if (d.result?.ci !== undefined) { setCiValue(d.result.ci); setGlobalCi(d.result.ci); }
        // Update segments + detector PC spinboxes with optimized values
        const r = d.result || d;
        if (r?.segments && onSegmentsUpdate) onSegmentsUpdate(r.segments);
        if (r?.mean_pc?.length === 3 && onOptimizedPc) onOptimizedPc(r.mean_pc);
      }
    } catch (err) {
      setIsRunning(false);
      const detail = err.response?.data?.detail;
      const gMsg = typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : err.message || t('pcrefinement:controls.globalRefineFailed');
      setGlobalMsg(gMsg);
      setGlobalError(true);
    }
  };

  // PC Drift Analysis (calibration-pattern based — existing flow)
  const handleDriftAnalysis = async () => {
    setDriftMsg(t('pcrefinement:controls.analyzingDrift'));
    try {
      const indices = patterns.map((p) => p.index);
      const res = await pcApi.analyzeDrift(indices);
      setDriftMsg(t('pcrefinement:controls.driftResult', { data: JSON.stringify(res.data) }));
    } catch (err) {
      setDriftMsg(t('pcrefinement:pixelwise.errorMsg', { detail: err.response?.data?.detail || err.message }));
    }
  };

  // Source PC Drift — visualize Aztec's per-pixel PC from the loaded
  // analysis dataset (xmap.prop.pc_x/pc_y/dd, populated from the source
  // h5oina at light-h5 export). Different signal than the calibration-
  // pattern drift above: this is the raw measurement, not a refined fit.
  const handleSourceDrift = async () => {
    setSourceDriftLoading(true);
    setSourceDriftError(null);
    setSourceDriftImage(null);
    setSourceDriftStats(null);
    try {
      // Lazy import — keep PC Refinement bundle light if user never clicks.
      const { analysisApi } = await import('../../services/api');
      const res = await analysisApi.pcDriftImage();
      setSourceDriftImage(res.data?.image || null);
      setSourceDriftStats(res.data?.stats || null);
    } catch (err) {
      setSourceDriftError(
        err.response?.data?.detail || err.message
        || t('pcrefinement:controls.sourceDriftFetchFailed')
      );
    } finally {
      setSourceDriftLoading(false);
    }
  };

  // Push the per-pixel PC into the calibration store as a "seed". The
  // currently active EBSD viewer dataset is the implicit target — typical
  // workflow: open h5oina in EBSD viewer (registers it in the store) →
  // open the matching light h5 in Analysis → click here to copy Aztec's
  // per-pixel PC over so subsequent indexing / refinement starts from it.
  const handlePushAsSeed = async () => {
    setSeedApplying(true);
    setSeedMsg(null);
    setSeedError(false);
    try {
      const { analysisApi } = await import('../../services/api');
      const res = await analysisApi.pcDriftToSeed(null);
      const d = res.data || {};
      const mean = d.mean_pc || [];
      setSeedMsg(
        t('pcrefinement:controls.seedPushed', {
          mean: mean.map((v) => v.toFixed(4)).join(', '),
          target: d.target_dataset,
        })
      );
    } catch (err) {
      setSeedMsg(err.response?.data?.detail || err.message || t('pcrefinement:controls.seedPushFailed'));
      setSeedError(true);
    } finally {
      setSeedApplying(false);
    }
  };

  const handleCancel = () => {
    stopPolling();
    setIsRunning(false);
    setTaskId(null);
    setProgress(0);
  };

  const canRun = detectorReady && phaseLoaded && !isRunning;
  const driftExport = useImageExport();

  return (
    <div
      className="thin-scrollbar"
      style={{
        overflowY: 'auto',
        height: '100%',
        padding: spacing.outerMargin,
        boxSizing: 'border-box',
        display: 'flex',
        flexDirection: 'column',
        gap: spacing.groupSpacing,
      }}
    >
      {driftExport.node}

      {/* Progress bar + Cancel — initially hidden (visible when running) */}
      {isRunning && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, animation: 'fadeSlideIn 0.2s ease-out' }}>
          <div style={{ flex: 1 }}>
            <ProgressBar value={progress} max={100} />
          </div>
          <Button
            small
            variant="danger"
            style={{ width: 60, flexShrink: 0 }}
            onClick={handleCancel}
            title={t('pcrefinement:hoverTips.cancelRun')}
          >
            {t('common:cancel')}
          </Button>
        </div>
      )}

      {/* Load Phase from CIF — opens the same library picker as Hough indexing */}
      <div>
        <Button
          variant="default"
          disabled={phaseLoading}
          style={{ width: '100%' }}
          onClick={openPhasePicker}
          title={t('pcrefinement:controls.loadPhaseTooltip')}
        >
          {phaseLoading ? <span className="btn-loading">{t('pcrefinement:controls.loadPhaseLoading')}</span> : t('pcrefinement:controls.loadPhase')}
        </Button>

        {/* Manual fallback for CIFs that are not in the library */}
        <div
          style={{
            display: 'flex',
            gap: 4,
            marginTop: 4,
            alignItems: 'center',
          }}
        >
          <Input
            value={cifPath}
            onChange={(e) => setCifPath(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && cifPath.trim() && !phaseLoading) handleLoadPhase(); }}
            placeholder={t('pcrefinement:controls.cifPathPlaceholder')}
            disabled={phaseLoading}
            style={{ flex: 1 }}
            title={t('pcrefinement:hoverTips.cifPathInput')}
          />
          <Button
            small
            disabled={phaseLoading}
            title={t('pcrefinement:controls.browseTooltip')}
            onClick={async () => {
              let p = null;
              if (window.electronAPI?.openFile) {
                p = await window.electronAPI.openFile({
                  filters: [{ name: 'CIF Files', extensions: ['cif'] }],
                });
              } else {
                askPrompt({
                  title: t('pcrefinement:prompt.loadCifTitle'),
                  message: t('pcrefinement:prompt.loadCifMessage'),
                  defaultValue: cifPath,
                  placeholder: t('pcrefinement:prompt.loadCifPlaceholder'),
                  submitLabel: t('pcrefinement:prompt.loadCifSubmit'),
                  onSubmit: (val) => {
                    if (val.trim()) { setCifPath(val.trim()); handleLoadPhase(val.trim()); }
                  },
                });
                return;
              }
              if (p && p.trim()) {
                setCifPath(p.trim());
                handleLoadPhase(p.trim());
              }
            }}
          >
            {t('pcrefinement:controls.browse')}
          </Button>
        </div>
        <StatusMsg msg={phaseMsg} isError={phaseError} />

        {/* Phase library picker — same component the Hough indexing page uses */}
        <FloatingPhasePanel
          open={phasePickerOpen}
          onClose={() => setPhasePickerOpen(false)}
          title={t('pcrefinement:controls.phasePickerTitle')}
          defaultX={typeof window !== 'undefined' ? window.innerWidth - 440 : 100}
          defaultY={120}
        >
          <PhaseDropdown
            discoveredFiles={discoveredFiles}
            groups={discoveredGroups}
            selectedPaths={cifPath ? [cifPath] : []}
            onTogglePath={handlePickPhase}
            onSetAll={handleSetAllPhases}
            method="hough"
            open={true}
            onClose={() => setPhasePickerOpen(false)}
          />
        </FloatingPhasePanel>

        {phaseInfo && (
          <div
            style={{
              marginTop: 4,
              padding: '4px 6px',
              background: colors.bgSecondary,
              border: `1px solid ${colors.border}`,
              borderRadius: 3,
              fontSize: '9pt',
              animation: 'fadeSlideIn 0.2s ease-out',
            }}
          >
            <span style={{ color: colors.accent }}>
              {phaseInfo.phase_name || phaseInfo.name || t('pcrefinement:controls.phaseInfoUnknown')}
            </span>
            {phaseInfo.space_group && (
              <span style={{ color: colors.textSecondary, marginLeft: 8 }}>
                {phaseInfo.space_group}
              </span>
            )}
          </div>
        )}
      </div>

      <Separator />

      {/* Index Pattern (single current pattern) */}
      <Button
        disabled={!canRun}
        style={{ width: '100%' }}
        onClick={handleIndexPattern}
        title={t('pcrefinement:controls.indexPatternTooltip')}
      >
        {t('pcrefinement:controls.indexPattern')}
      </Button>

      {/* Index All Patterns — mirrors PyQt5 IndexAllWorker */}
      <Button
        disabled={!canRun || patterns.length === 0}
        style={{ width: '100%' }}
        variant="purple"
        onClick={handleIndexAll}
        title={t('pcrefinement:controls.indexAllTooltip')}
      >
        {t('pcrefinement:controls.indexAll', { count: patterns.length })}
      </Button>
      <StatusMsg msg={indexMsg} isError={indexError} />

      {/* Global PC Refine */}
      <Button
        disabled={!canRun || patterns.length === 0}
        style={{ width: '100%' }}
        variant="primary"
        onClick={handleGlobalRefine}
        title={t('pcrefinement:controls.globalRefineTooltip')}
      >
        {t('pcrefinement:controls.globalRefine')}
      </Button>
      <StatusMsg msg={globalMsg} isError={globalError} />

      {/* Source PC Drift — render Aztec's per-pixel PC across the whole
          scan (read from xmap.prop.pc_x/pc_y/dd of the active analysis
          dataset). Complements the calibration-pattern drift below: this
          is the *raw* per-pixel PC the indexing was given, before any
          refinement. Useful to see whether a global PC was a fair
          assumption — large drift = global PC throws away signal. */}
      <GroupBox title={t('pcrefinement:controls.sourceDriftTitle')} style={{ animation: 'fadeSlideIn 0.2s ease-out' }}>
        <div style={{ fontSize: '9pt', color: colors.textSecondary, marginBottom: 6 }}>
          {t('pcrefinement:controls.sourceDriftDesc')}
        </div>
        <Button
          small
          style={{ width: '100%', marginBottom: 6 }}
          onClick={handleSourceDrift}
          disabled={sourceDriftLoading}
          title={t('pcrefinement:controls.showSourceDriftTooltip')}
        >
          {sourceDriftLoading ? t('pcrefinement:controls.showSourceDriftRendering') : t('pcrefinement:controls.showSourceDrift')}
        </Button>
        {sourceDriftError && (
          <div style={{
            fontSize: '9pt', color: colors.red,
            background: alpha(colors.red, 7),
            border: `1px solid ${alpha(colors.red, 27)}`,
            borderRadius: 3, padding: '3px 6px',
          }}>
            {sourceDriftError}
          </div>
        )}
        {sourceDriftImage && (
          <>
            <img
              src={`data:image/png;base64,${sourceDriftImage}`}
              alt={t('pcrefinement:controls.sourceDriftImageAlt')}
              onContextMenu={(e) => driftExport.openMenu(e, {
                build: () => `data:image/png;base64,${sourceDriftImage}`,
                name: 'pc-drift',
                label: t('pcrefinement:controls.sourceDriftTitle'),
              })}
              style={{ width: '100%', borderRadius: 3, border: `1px solid ${colors.border}` }}
            />
            {sourceDriftStats && (
              <div style={{
                marginTop: 6, fontSize: '9pt', color: colors.textSecondary,
                fontFamily: 'monospace', lineHeight: 1.5,
              }}>
                {Object.entries(sourceDriftStats).map(([k, v]) => (
                  <div key={k}>
                    <span style={{ color: colors.text }}>{k}</span>: {t('pcrefinement:controls.sourceDriftStatRow', {
                      range: v.range.toFixed(5),
                      mean: v.mean.toFixed(4),
                      drift: v.drift_pct.toFixed(1),
                    })}
                  </div>
                ))}
              </div>
            )}
            <Button
              small
              variant="primary"
              disabled={seedApplying}
              style={{ width: '100%', marginTop: 6 }}
              onClick={handlePushAsSeed}
              title={t('pcrefinement:controls.useAsSeedTooltip')}
            >
              {seedApplying ? t('pcrefinement:controls.useAsSeedPushing') : t('pcrefinement:controls.useAsSeed')}
            </Button>
            <StatusMsg msg={seedMsg} isError={seedError} />
          </>
        )}
      </GroupBox>

      {/* PC Drift Analysis */}
      <Button
        disabled={!canRun || patterns.length < 2}
        style={{ width: '100%' }}
        onClick={handleDriftAnalysis}
        title={t('pcrefinement:controls.driftAnalysisTooltip')}
      >
        {t('pcrefinement:controls.driftAnalysis')}
      </Button>
      {driftMsg && (
        <div
          style={{
            fontSize: '9pt',
            color: colors.textSecondary,
            padding: '3px 6px',
            background: colors.bgSecondary,
            border: `1px solid ${colors.border}`,
            borderRadius: 3,
            wordBreak: 'break-word',
            animation: 'fadeSlideIn 0.2s ease-out',
          }}
        >
          {driftMsg}
        </div>
      )}

      {/* Result display */}
      {result && (
        <GroupBox title={t('pcrefinement:controls.resultsTitle')} style={{ animation: 'fadeSlideIn 0.2s ease-out' }}>
          {[
            { key: 'PCx', label: t('pcrefinement:controls.pcxLabel') },
            { key: 'PCy', label: t('pcrefinement:controls.pcyLabel') },
            { key: 'PCz', label: t('pcrefinement:controls.pczLabel') },
          ].map(({ key, label }, i) => {
            const keys = ['pc_x', 'pc_y', 'pc_z'];
            const val = result[keys[i]] ?? (result.mean_pc ? result.mean_pc[i] : result.pc?.[i]);
            return (
              <FormRow key={key} label={label}>
                <Label style={{ fontFamily: 'monospace', color: colors.green }}>
                  {val !== undefined ? val.toFixed(5) : '—'}
                </Label>
              </FormRow>
            );
          })}
          {result.mean_pc && (
            <FormRow label={t('pcrefinement:controls.meanPcLabel')}>
              <Label style={{ fontFamily: 'monospace', color: colors.accent, fontSize: '9pt' }}>
                {result.mean_pc.map((v) => v.toFixed(4)).join(', ')}
              </Label>
            </FormRow>
          )}
          {/* Reliability guard: Hough-based PC refine is unreliable on small
              (low-res) patterns and can drift far from the vendor/.osc PC. */}
          {result.pc_warning && (
            <div style={{
              marginTop: 6, marginBottom: 6, padding: '8px 10px',
              background: 'rgba(224,160,32,0.12)',
              border: '1px solid #e0a020', borderRadius: 4,
              color: colors.text, fontSize: '8.5pt', lineHeight: 1.4,
            }}>
              ⚠ {result.pc_warning}
            </div>
          )}
          <Button
            small
            style={{ width: '100%', marginBottom: 4 }}
            title={t('pcrefinement:hoverTips.copyPcValues')}
            onClick={() => {
              const keys = ['pc_x', 'pc_y', 'pc_z'];
              const vals = keys.map((k, i) => {
                const v = result[k] ?? (result.mean_pc ? result.mean_pc[i] : result.pc?.[i]);
                return v !== undefined ? v.toFixed(5) : '—';
              });
              const text = `PCx: ${vals[0]}\nPCy: ${vals[1]}\nPCz: ${vals[2]}`;
              navigator.clipboard.writeText(text).then(() => {
                setPcCopied(true);
                setTimeout(() => setPcCopied(false), 1200);
              }).catch(() => {});
            }}
          >
            {pcCopied ? t('pcrefinement:controls.copied') : t('pcrefinement:controls.copyPcValues')}
          </Button>
          {parentName && activeDatasetName && (
            <Button
              small
              disabled={propagated}
              style={{ width: '100%', marginBottom: 4, marginTop: 4 }}
              title={t('pcrefinement:controls.applyPcToParentTooltip', { parent: parentName })}
              onClick={async () => {
                try {
                  const res = await calibrationApi.propagateToParent(activeDatasetName);
                  if (res.data?.success) {
                    setPropagated(true);
                    setResultMsg(t('pcrefinement:controls.pcAppliedMsg', { parent: parentName }));
                    setTimeout(() => setPropagated(false), 3000);
                  }
                } catch (err) {
                  console.error('PC propagation failed:', err);
                  setResultMsg(t('pcrefinement:controls.pcPropagationFailed'));
                }
              }}
            >
              {propagated
                ? t('pcrefinement:controls.pcAppliedToParent', { parent: parentName })
                : t('pcrefinement:controls.applyPcToParent', { parent: parentName })}
            </Button>
          )}
          <StatusMsg msg={resultMsg} isError={false} />
          <Separator />
          <Button
            small
            disabled={!canRun || patterns.length < 2}
            style={{ width: '100%', marginBottom: 4 }}
            onClick={handleDriftAnalysis}
            title={t('pcrefinement:hoverTips.driftAnalysisShort')}
          >
            {t('pcrefinement:controls.driftAnalysisShort')}
          </Button>
          <Button
            small
            disabled
            title={t('pcrefinement:controls.gridCalibrationTooltip')}
            style={{ width: '100%' }}
          >
            {t('pcrefinement:controls.gridCalibration')}
          </Button>
        </GroupBox>
      )}

      <Separator />

      {/* Pixel-wise PC Correction */}
      <PixelwisePCGroup canRun={canRun} />

      {/* Indexing Settings */}
      <IndexingSettingsGroup onParamsChange={handleParamsChange} />

      {/* Detector Settings */}
      <DetectorSettingsGroup
        onDetectorApplied={onDetectorApplied}
        onPcChanged={onPcChanged}
        onTiltChanged={onTiltChanged}
        externalPc={externalPc}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main PCRefinement component
// ---------------------------------------------------------------------------
export default function PCRefinement({ onNavigate }) {
  const { t } = useTranslation('pcrefinement');
  const [askPrompt, promptProps] = usePrompt();
  // Header state
  const [phaseLabel, setPhaseLabel] = useState(t('pcrefinement:header.noPhaseLoaded'));
  const [phaseInfoText, setPhaseInfoText] = useState('');
  const [ciValue, setCiValue] = useState(null);
  const [globalCi, setGlobalCi] = useState(null);

  // Readiness flags
  const [detectorReady, setDetectorReady] = useState(false);
  const [phaseLoaded, setPhaseLoaded] = useState(false);

  // Current PC (for crosshair in canvas)
  const [currentPcx, setCurrentPcx] = useState(0.5);
  const [currentPcy, setCurrentPcy] = useState(0.5);

  // Pattern preview (image of the currently selected calibration pattern)
  const [patternBase64, setPatternBase64] = useState(null);

  // Calibration patterns list
  const [patterns, setPatterns] = useState([]);
  const [selectedPatternIdx, setSelectedPatternIdx] = useState(null);

  // Kikuchi line segments from indexing (for overlay)
  const [kikuchiSegments, setKikuchiSegments] = useState([]);

  // --- Forward-simulation preview (closes the PC/sample_tilt/det_tilt
  // degeneracy that Hough band-fitting alone can't break) ---
  const [trialGeometry, setTrialGeometry] = useState({
    pcx: 0.5, pcy: 0.5, pcz: 0.5,
    sampleTilt: 70.0, detTilt: 0.0, azimuthal: 0.0,
    binning: 1, pixelSize: null,
  });
  const [shtPath, setShtPath] = useState(null);
  const [shtChoices, setShtChoices] = useState([]);
  const [simulatedB64, setSimulatedB64] = useState(null);
  const [simulatedNcc, setSimulatedNcc] = useState(null);
  const [simulatedLoading, setSimulatedLoading] = useState(false);
  const [simulatedError, setSimulatedError] = useState(null);
  const [simulatedOrientation, setSimulatedOrientation] = useState(null);
  const [simulatedOrientationSource, setSimulatedOrientationSource] = useState(null);
  const [previewBandwidth, setPreviewBandwidth] = useState(128);

  // Optimized PC values (passed to DetectorSettingsGroup to update spinboxes)
  const [optimizedPc, setOptimizedPc] = useState(null);

  // Parent dataset info for PC propagation
  const [parentName, setParentName] = useState(null);
  const [activeDatasetName, setActiveDatasetName] = useState(null);
  const [propagated, setPropagated] = useState(false);

  // Indexing params (forwarded from IndexingSettingsGroup)
  const indexingParamsRef = useRef({});

  // Check backend status on mount and sync patterns + PC values
  useEffect(() => {
    pcApi.status().then((res) => {
      const d = res.data;
      if (d?.has_detector || d?.detector_set) setDetectorReady(true);
      if (d?.has_phase || d?.phase_loaded) {
        setPhaseLoaded(true);
        if (d?.phase_name) setPhaseLabel(d.phase_name);
      }
      // Sync patterns from backend
      if (d?.patterns && d.patterns.length > 0) {
        setPatterns(d.patterns);
      }
      // Sync PC values for crosshair and spinboxes
      if (d?.pc?.length === 3) {
        setCurrentPcx(d.pc[0]);
        setCurrentPcy(d.pc[1]);
        setOptimizedPc(d.pc);  // Feed to DetectorSettingsGroup spinboxes
      }
    }).catch(() => {});
  }, []);

  // Fetch calibration entry to check if active dataset has a parent
  useEffect(() => {
    ebsdApi.info().then((res) => {
      const dsName = res.data?.active_dataset;
      if (!dsName) return;
      setActiveDatasetName(dsName);
      calibrationApi.getEntry(dsName).then((calRes) => {
        if (calRes.data?.parent_name) {
          setParentName(calRes.data.parent_name);
        }
      }).catch(() => {}); // No calibration entry — not a derived dataset
    }).catch(() => {});
  }, []);

  // Sync patterns from backend (called on mount and can be re-triggered)
  const syncPatterns = useCallback(() => {
    pcApi.status().then((res) => {
      const d = res.data;
      if (d?.patterns && d.patterns.length > 0) {
        setPatterns(d.patterns); // patterns now include CI from backend cache
        // Auto-select last pattern and load its image for preview
        const last = d.patterns[d.patterns.length - 1];
        setSelectedPatternIdx(d.patterns.length - 1);
        if (last) {
          ebsdApi.getPattern(last.row, last.col).then((pRes) => {
            const img = pRes.data?.image || pRes.data?.image_base64 || null;
            setPatternBase64(img);
          }).catch(() => {});
        }
      }
      if (d?.has_detector) setDetectorReady(true);
      if (d?.has_phase) {
        setPhaseLoaded(true);
        if (d?.phase_name) setPhaseLabel(d.phase_name);
      }
      if (d?.global_ci != null) setGlobalCi(d.global_ci);
      if (d?.pc?.length === 3) {
        setCurrentPcx(d.pc[0]);
        setCurrentPcy(d.pc[1]);
      }
    }).catch(() => {});
  }, []);

  // Re-sync when PC Refinement page becomes visible OR when patterns are updated
  const containerRef = useRef(null);
  useEffect(() => {
    const onPageChange = (e) => {
      if (e.detail?.page === 'pcrefinement') syncPatterns();
    };
    const onPatternsUpdated = () => syncPatterns();
    window.addEventListener('page-changed', onPageChange);
    window.addEventListener('pc-patterns-updated', onPatternsUpdated);
    return () => {
      window.removeEventListener('page-changed', onPageChange);
      window.removeEventListener('pc-patterns-updated', onPatternsUpdated);
    };
  }, [syncPatterns]);

  const handleRemovePattern = async () => {
    if (selectedPatternIdx === null) return;
    try {
      await pcApi.removePattern(selectedPatternIdx);
    } catch {
      // Backend removal failed — still remove locally to stay responsive
    }
    setPatterns((prev) => {
      const next = prev.filter((_, i) => i !== selectedPatternIdx);
      return next.map((p, i) => ({ ...p, index: i }));
    });
    setSelectedPatternIdx(null);
  };

  const handlePhaseLoaded = (info) => {
    setPhaseLoaded(true);
    const name = info.phase_name || info.name || t('pcrefinement:phaseInfo.unknown');
    setPhaseLabel(name);
    const lines = [
      t('pcrefinement:phaseInfo.nameLine', { name }),
      info.space_group ? t('pcrefinement:phaseInfo.spaceGroupLine', { spaceGroup: info.space_group }) : '',
      info.lattice ? `a=${info.lattice.a?.toFixed(3)} b=${info.lattice.b?.toFixed(3)} c=${info.lattice.c?.toFixed(3)}` : '',
    ].filter(Boolean).join('\n');
    setPhaseInfoText(lines);
  };

  const handleDetectorApplied = (pc, sampleTilt, detTilt) => {
    setDetectorReady(true);
    if (pc) {
      setCurrentPcx(parseFloat(pc[0]) || 0.5);
      setCurrentPcy(parseFloat(pc[1]) || 0.5);
      // Keep trialGeometry in sync so the forward-sim preview reflects
      // whatever was just applied via the Apply button. detTilt MUST be
      // carried through too — otherwise it stays at the 0.0 default and the
      // preview is rotated by camera_tilt relative to the Phase Test, which
      // uses the loaded detector tilt (root-caused 2026-06-27).
      setTrialGeometry((g) => ({
        ...g,
        pcx: parseFloat(pc[0]) || 0.5,
        pcy: parseFloat(pc[1]) || 0.5,
        pcz: parseFloat(pc[2]) || 0.5,
        ...(sampleTilt != null ? { sampleTilt: parseFloat(sampleTilt) || 70.0 } : {}),
        ...(detTilt != null && !isNaN(parseFloat(detTilt)) ? { detTilt: parseFloat(detTilt) } : {}),
      }));
    }
  };

  // Called when Index All completes — update pattern list with CI values
  const handlePatternsIndexed = useCallback((results) => {
    setPatterns((prev) =>
      prev.map((p) => {
        const r = results.find((res) => res.index === p.index);
        if (r) return { ...p, ci: r.ci };
        return p;
      })
    );
  }, []);

  // Reactive PC change — mirrors PyQt5 apply_detector_pc()
  // Updates detector, clears caches, re-indexes current pattern, re-overlays
  const handlePcChanged = useCallback(async (pcx, pcy, pcz) => {
    setCurrentPcx(pcx);
    setCurrentPcy(pcy);
    // Update trialGeometry so the forward-sim preview re-renders. This
    // is the geometry knob most users will sweep when calibrating.
    setTrialGeometry((g) => ({ ...g, pcx, pcy, pcz }));
    try {
      const res = await pcApi.updatePC(pcx, pcy, pcz, selectedPatternIdx);
      const d = res.data;
      if (d?.ci != null) setCiValue(d.ci);
      if (d?.global_ci != null) setGlobalCi(d.global_ci);
      if (d?.segments) setKikuchiSegments(d.segments);
    } catch {
      // Ignore errors during reactive update
    }
  }, [selectedPatternIdx, setCiValue, setGlobalCi]);

  // Sample / detector tilt + azimuthal changes — keep trialGeometry in
  // sync. The existing handleTiltChanged callback below posts to the
  // backend; we just piggyback to update the local trial state.
  const handleTrialTiltChanged = useCallback((sampleTilt, detTilt, azimuthal) => {
    setTrialGeometry((g) => ({
      ...g,
      sampleTilt: parseFloat(sampleTilt) || 70.0,
      detTilt: parseFloat(detTilt) || 0.0,
      azimuthal: parseFloat(azimuthal) || 0.0,
    }));
  }, []);

  // Reactive tilt change — mirrors PyQt5 apply_tilt_settings()
  const handleTiltChanged = useCallback(async (sampleTilt, detTilt, azimuthal) => {
    // Mirror the values into trialGeometry so the forward-sim preview
    // refreshes whenever the user nudges a tilt — that's the parameter
    // most likely to be miscalibrated (header values are unreliable).
    handleTrialTiltChanged(sampleTilt, detTilt, azimuthal);
    try {
      const res = await pcApi.updateTilt(sampleTilt, detTilt, azimuthal, selectedPatternIdx);
      const d = res.data;
      if (d?.ci != null) setCiValue(d.ci);
      if (d?.segments) setKikuchiSegments(d.segments);
    } catch {
      // Ignore errors during reactive update
    }
  }, [selectedPatternIdx, setCiValue, handleTrialTiltChanged]);

  // --- Auto-discover SHT files when phase is loaded ---
  // The forward-sim preview needs an SHT master pattern. We list all
  // available SHT files from the spherical-indexing files endpoint
  // and offer them in a small dropdown next to the preview pane.
  useEffect(() => {
    if (!phaseLoaded) {
      setShtChoices([]);
      setShtPath(null);
      return;
    }
    // Direct fetch — the existing api.js doesn't currently expose a
    // typed wrapper for the indexing/files endpoint.
    fetch('/api/indexing/files/spherical')
      .then(r => r.json())
      .then(d => {
        const files = Array.isArray(d?.files) ? d.files : [];
        setShtChoices(files);
        if (files.length > 0 && !shtPath) setShtPath(files[0].path);
      })
      .catch(() => { setShtChoices([]); });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phaseLoaded]);

  // --- Debounced fetch of the simulated preview ---
  // Watches the trial geometry + selected pattern + SHT + bandwidth and
  // posts to /api/pc/render-preview. Uses the cancellation-flag pattern
  // we already proved fixed similar race conditions in the Pattern
  // Match dialog: rapid parameter sweeps would otherwise show stale
  // response data while the controls display the new values.
  useEffect(() => {
    if (selectedPatternIdx == null || !shtPath || !phaseLoaded) {
      setSimulatedB64(null);
      setSimulatedNcc(null);
      setSimulatedError(null);
      return;
    }
    let cancelled = false;
    const timer = setTimeout(() => {
      setSimulatedLoading(true);
      setSimulatedError(null);
      pcApi.renderPreview({
        patternIdx: selectedPatternIdx,
        shtPath,
        pc: [trialGeometry.pcx, trialGeometry.pcy, trialGeometry.pcz],
        sampleTilt: trialGeometry.sampleTilt,
        detectorTilt: trialGeometry.detTilt,
        azimuthal: trialGeometry.azimuthal,
        binning: trialGeometry.binning,
        pixelSize: trialGeometry.pixelSize,
        maxBandwidth: previewBandwidth,
      })
        .then(r => {
          if (cancelled) return;
          setSimulatedB64(r.data?.simulated_b64 || null);
          setSimulatedNcc(typeof r.data?.ncc === 'number' ? r.data.ncc : null);
          setSimulatedOrientation(r.data?.orientation_euler_deg || null);
          setSimulatedOrientationSource(r.data?.orientation_source || null);
        })
        .catch(err => {
          if (cancelled) return;
          setSimulatedB64(null);
          setSimulatedNcc(null);
          setSimulatedError(err?.response?.data?.detail || err?.message || 'render failed');
        })
        .finally(() => { if (!cancelled) setSimulatedLoading(false); });
    }, 500);  // debounce 500ms
    return () => { cancelled = true; clearTimeout(timer); };
  }, [
    selectedPatternIdx, shtPath, phaseLoaded, previewBandwidth,
    trialGeometry.pcx, trialGeometry.pcy, trialGeometry.pcz,
    trialGeometry.sampleTilt, trialGeometry.detTilt, trialGeometry.azimuthal,
    trialGeometry.binning, trialGeometry.pixelSize,
  ]);

  // Select a pattern from the list — load image + cached overlay
  // Mirrors PyQt5 on_pattern_changed() with cache-first logic
  const handleSelectPatternIdx = (idx) => {
    setSelectedPatternIdx(idx);
    const pat = patterns[idx];
    if (pat) {
      // Load the STORED pattern image (captured at add-time, not current active signal)
      // This is critical for DeepCopy/FrameAvg workflows where the active signal may differ
      pcApi.getPatternImage(idx).then((res) => {
        const img = res.data?.image || null;
        setPatternBase64(img);
      }).catch(() => {
        // Fallback: load from active signal if stored image endpoint fails
        const r = pat.row ?? 0;
        const c = pat.col ?? 0;
        ebsdApi.getPattern(r, c).then((res2) => {
          const img = res2.data?.image || res2.data?.image_base64 || null;
          setPatternBase64(img);
        }).catch(() => {});
      });
      // Load cached overlay (auto-indexes if not cached)
      if (idx != null) {
        pcApi.getPatternResult(idx).then((res) => {
          const d = res.data;
          if (d?.success && d.segments) {
            setKikuchiSegments(d.segments);
            if (d.ci != null) setCiValue(d.ci);
          } else {
            setKikuchiSegments([]);
            // Fall back to local CI when backend cache is lost
            if (patterns[idx]?.ci != null) setCiValue(patterns[idx].ci);
          }
        }).catch(() => {
          setKikuchiSegments([]);
          // Fall back to local CI when backend cache is lost
          if (patterns[idx]?.ci != null) setCiValue(patterns[idx].ci);
        });
      }
    }
  };

  return (
    <div
      ref={containerRef}
      style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        background: colors.bg,
        color: colors.text,
        fontFamily: "'Segoe UI', system-ui, sans-serif",
        fontSize: '10pt',
      }}
    >
      {/* Page title */}
      <div style={{ padding: `${spacing.outerMargin}px ${spacing.outerMargin}px 0`, flexShrink: 0 }}>
        <h1 style={{ margin: 0, fontSize: '18pt', fontWeight: 700, color: colors.accent }}>{t('pcrefinement:page.title')}</h1>
        <div style={{ fontSize: '10pt', color: colors.textSecondary, marginTop: 2 }}>
          {t('pcrefinement:page.subtitle')}
        </div>
      </div>
      {/* Workflow steps indicator */}
      {!detectorReady && !phaseLoaded && patterns.length === 0 && (
        <div style={{
          display: 'flex', gap: 0, alignItems: 'center',
          padding: `8px ${spacing.outerMargin}px`,
          fontSize: '9pt', color: colors.textSecondary,
          animation: 'pageFadeIn 0.3s ease-out',
        }}>
          {[
            { n: '1', label: t('pcrefinement:workflow.step1'), done: detectorReady },
            { n: '2', label: t('pcrefinement:workflow.step2'), done: phaseLoaded },
            { n: '3', label: t('pcrefinement:workflow.step3'), done: patterns.length > 0 },
          ].map((step, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center' }}>
              {i > 0 && (
                <div style={{
                  width: 28, height: 2, margin: '0 6px',
                  background: step.done
                    ? `linear-gradient(90deg, ${colors.green}, ${colors.green})`
                    : `linear-gradient(90deg, ${colors.border}, ${colors.border}88)`,
                  borderRadius: 1,
                  transition: 'background 0.3s',
                }} />
              )}
              <span style={{
                display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                width: 20, height: 20, borderRadius: '50%', fontSize: '8pt', fontWeight: 700,
                marginRight: 5,
                background: step.done ? colors.green : colors.border,
                color: step.done ? colors.bg : colors.textSecondary,
                transition: 'all 0.3s',
                boxShadow: step.done ? `0 0 6px ${colors.green}44` : 'none',
              }}>
                {step.done ? '\u2713' : step.n}
              </span>
              <span style={{
                color: step.done ? colors.green : colors.textSecondary,
                fontWeight: step.done ? 600 : 400,
                transition: 'color 0.3s',
              }}>
                {step.label}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Phase info | CI labels | Global CI                                  */}
      {/* ------------------------------------------------------------------ */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: spacing.outerSpacing,
          padding: `${spacing.innerMargin}px ${spacing.outerMargin}px 0`,
          flexShrink: 0,
        }}
      >
        {/* Phase label */}
        <span
          style={{
            fontSize: '10pt',
            color: phaseLoaded ? colors.accent : colors.textSecondary,
            flexShrink: 0,
          }}
        >
          {phaseLabel}
        </span>

        {/* Phase info (read-only textarea) */}
        <div style={{ flex: 1, maxWidth: 320 }}>
          <MonoDisplay value={phaseInfoText} rows={2} title={t('pcrefinement:hoverTips.phaseInfoDisplay')} />
        </div>

        {/* Stretch */}
        <div style={{ flex: 1 }} />

        {/* CI label */}
        <span
          style={{
            fontSize: '10pt',
            color: ciValue !== null ? colors.cyan : colors.textSecondary,
            fontFamily: 'monospace',
            background: colors.bgSecondary,
            border: `1px solid ${colors.border}`,
            borderRadius: 3,
            padding: '2px 8px',
          }}
        >
          {t('pcrefinement:header.ci')} {ciValue !== null ? ciValue.toFixed(4) : t('pcrefinement:header.notAvailable')}
        </span>

        {/* Global CI label */}
        <span
          style={{
            fontSize: '10pt',
            color: globalCi !== null ? colors.green : colors.textSecondary,
            fontFamily: 'monospace',
            background: colors.bgSecondary,
            border: `1px solid ${colors.border}`,
            borderRadius: 3,
            padding: '2px 8px',
          }}
        >
          {t('pcrefinement:header.globalCi')} {globalCi !== null ? globalCi.toFixed(4) : t('pcrefinement:header.notAvailable')}
        </span>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Main horizontal splitter                                            */}
      {/* Left  — preview panel                                               */}
      {/* Right — scrollable controls                                         */}
      {/* ------------------------------------------------------------------ */}
      <div style={{ flex: 1, overflow: 'hidden', marginTop: spacing.outerSpacing }}>
        <ResizableSplitter
          defaultLeftWidth={340}
          minLeftWidth={220}
          maxLeftWidth={600}
          left={
            <PreviewPanel
              patterns={patterns}
              selectedIdx={selectedPatternIdx}
              onSelectIdx={handleSelectPatternIdx}
              onRemovePattern={handleRemovePattern}
              sliderMax={Math.max(0, patterns.length - 1)}
              sliderValue={selectedPatternIdx ?? 0}
              onSliderChange={handleSelectPatternIdx}
              patternBase64={patternBase64}
              pcx={currentPcx}
              pcy={currentPcy}
              segments={kikuchiSegments}
              // --- Forward-Sim preview ---
              simulatedB64={simulatedB64}
              simulatedNcc={simulatedNcc}
              simulatedLoading={simulatedLoading}
              simulatedError={simulatedError}
              simulatedOrientation={simulatedOrientation}
              simulatedOrientationSource={simulatedOrientationSource}
              shtChoices={shtChoices}
              shtPath={shtPath}
              onShtPathChange={setShtPath}
              previewBandwidth={previewBandwidth}
              onPreviewBandwidthChange={setPreviewBandwidth}
            />
          }
          right={
            <ControlsPanel
              phaseLoaded={phaseLoaded}
              detectorReady={detectorReady}
              patterns={patterns}
              currentPatternIdx={selectedPatternIdx}
              onPhaseLoaded={handlePhaseLoaded}
              onDetectorApplied={handleDetectorApplied}
              onIndexingParamsChange={(p) => { indexingParamsRef.current = p; }}
              ciValue={ciValue}
              setCiValue={setCiValue}
              globalCi={globalCi}
              setGlobalCi={setGlobalCi}
              onSegmentsUpdate={setKikuchiSegments}
              onPatternsIndexed={handlePatternsIndexed}
              onPcChanged={handlePcChanged}
              onTiltChanged={handleTiltChanged}
              externalPc={optimizedPc}
              parentName={parentName}
              activeDatasetName={activeDatasetName}
              propagated={propagated}
              setPropagated={setPropagated}
              onOptimizedPc={(pc) => {
                setOptimizedPc(pc);
                if (pc.length === 3) {
                  setCurrentPcx(pc[0]);
                  setCurrentPcy(pc[1]);
                }
              }}
            />
          }
          style={{ height: '100%' }}
        />
      </div>
      <PromptDialog {...promptProps} />
    </div>
  );
}
