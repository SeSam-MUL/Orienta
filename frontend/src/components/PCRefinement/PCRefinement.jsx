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
 *               GroupBox: Pixel-wise PC Correction
 *               GroupBox: Indexing Settings
 *               GroupBox: Detector Settings
 *                 sub-GroupBox: Pixel Size & Binning
 */

import { localiseParams } from '../../i18n/formatDateTime';
import { useState, useEffect, useRef, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { pcApi, ebsdApi, calibrationApi, indexApi, pcCalibrationsApi } from '../../services/api';
import useDataStore from '../../stores/useDataStore';
import useLoadedFilesStore from '../../stores/useLoadedFilesStore';
import PhaseDropdown from '../Indexing/PhaseDropdown';
import { useImageExport, exportStem } from '../common/useImageExport';
import { buildPanelSheet } from '../common/imageExport';
import FloatingPhasePanel from '../Indexing/FloatingPhasePanel';
import {
  checkPixelSize, plausibleWidthMm, TYPICAL_MIN_UM, TYPICAL_MAX_UM,
} from './pixelSizeCheck';
import { pickShtForPhase } from './previewSht';
import { createSelectionGuard, createLatestRequest } from './selectionGuard';
import { problemText, problemCode } from '../../services/problemText';
import { pcMapInfoOf, pcMapActionOf, pcMapNote } from './pcMapNote';
import PcFileHead from './PcFileHead';
import PcFields, { PcBadge } from './PcFields';
import { preselectedConvention } from './pcConvention';
import { PcUndoDone } from './PcGrant';
import CalibrationPanel from './Calibration/CalibrationPanel';
import SaveCalibrationDialog from './Calibration/SaveCalibrationDialog';
import ApplyToFilesDialog from './Calibration/ApplyToFilesDialog';
import FollowerQuestion from './Calibration/FollowerQuestion';
import { questionKey } from './Calibration/calibrationText';
import OlderResults from './Calibration/OlderResults';
import useApplyFlow from './Calibration/applyFlow';
import PcResult from './PcResult';
import { formatPc, samePc } from './pcFormat';
import { PhaseLoadedContext } from './phaseContext';
import { needsDecision } from './pcResultFacts';
import { phaseFitsTooltip } from './phaseFits';
import { isPcJobRunning, pcJobRunningText } from './jobRunning';
import { toast } from '../../stores/useToastStore';
import {
  MAX_PC_PHASES, phaseLabelFor, loadedPaths, planPhaseSync, loadedPhaseFor, sameNameOtherFile,
  summarisePatternPhases, previewPhaseName, createSerialQueue, shortSpaceGroup,
} from './phaseSet';
import {
  phaseStem, pathErrorFrom, samePath, asRecord, resolvePhasePath,
} from '../Indexing/phasePath';
import ReflectorFamilies from '../Indexing/ReflectorFamilies';
import HoughRefusal, { refusalFrom } from '../Indexing/HoughRefusal';
import { loadedPhaseAdapter } from '../Indexing/reflectorSpec';
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

// "Find PC" needs this many calibration patterns on the list (the backend refuses fewer); below it, nine
// are picked from the scan first. NORDIF's calibration set has exactly five.
const FIND_MIN_PATTERNS = 5;

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
        whiteSpace: 'pre-line',   // the lines of a Find PC run stand one under the other
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
function PatternCanvas({ patternBase64, pcx, pcy, hideLines, segments, exportName, testId = 'pc-pattern-canvas' }) {
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
      data-testid={testId}
      data-segment-count={segments ? segments.length : 0}
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

// The pattern the page shows (the selected calibration pattern) with the bands drawn from the PC in use:
// the same canvas and the same drawing as the preview on the left, beside a result so that it can be
// judged where it is given, with a switch to take the bands off the pattern.
function BandsPane({ patternBase64, pcx, pcy, segments, caption, exportName }) {
  const { t } = useTranslation('pcrefinement');
  const [on, setOn] = useState(true);
  return (
    <div data-testid="pc-bands-pane" style={{ display: 'flex', flexDirection: 'column', gap: 4, maxWidth: 320 }}>
      <PatternCanvas
        patternBase64={patternBase64}
        pcx={pcx}
        pcy={pcy}
        segments={segments}
        hideLines={!on}
        exportName={exportName}
        testId="pc-bands-canvas"
      />
      <label style={{ fontSize: '9pt', color: colors.text }}>
        <input type="checkbox" checked={on} onChange={(e) => setOn(e.target.checked)} />{' '}
        {t('decide.bandsSwitch')}
      </label>
      <div style={{ fontSize: '9pt', color: colors.textSecondary }}>{caption}</div>
    </div>
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
  simulatedNccReason = null,
  simulatedLoading = false,
  simulatedError = null,
  simulatedOrientation = null,
  simulatedOrientationSource = null,
  shtChoices = [],
  shtPath = null,
  onShtPathChange = null,
  previewBandwidth = 128,
  onPreviewBandwidthChange = null,
  // --- Several phases ---
  multiPhase = false,
  simulatedIndexedPhase = null,
  // The phase the preview is for; named when the library has no master of its own.
  previewPhase = null,
}) {
  const { t } = useTranslation(['pcrefinement', 'common']);
  const [hideLines, setHideLines] = useState(false);
  // No master is this phase's own (pickShtForPhase gave none): say so instead of
  // letting the dropdown show its first entry as if it had been chosen.
  const noOwnMaster = shtChoices.length > 0 && !shtPath && !!previewPhase;
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
    // The column scrolls as a whole. It used to clip (`overflow: hidden`): on a
    // laptop window the canvas and the forward preview filled it, and the
    // selected-pattern list and the Remove button fell below the edge with no
    // way to reach them.
    <div
      className="thin-scrollbar"
      data-testid="pc-preview-column"
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        minHeight: 0,
        padding: spacing.outerMargin,
        boxSizing: 'border-box',
        overflowY: 'auto',
        overflowX: 'hidden',
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
            {/* No orientation could be indexed here: a pattern rendered from the
                fallback orientation has no NCC worth showing. Say why instead. */}
            {!simulatedLoading && simulatedNcc == null && simulatedNccReason === 'no_orientation' && (
              <span
                data-testid="pc-ncc-none"
                style={{ fontSize: '8pt', marginLeft: 'auto', color: '#ffb86c' }}
              >
                {t('pcrefinement:preview.nccNoOrientation')}
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
              {noOwnMaster && (
                <option value="" data-testid="pc-sht-no-own-master">
                  {t('pcrefinement:preview.noOwnMaster', { phase: previewPhase })}
                </option>
              )}
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
                  {simulatedError ? simulatedError : (simulatedLoading ? '…'
                    : (noOwnMaster
                      ? t('pcrefinement:preview.noOwnMaster', { phase: previewPhase })
                      : t('pcrefinement:preview.pickSht')))}
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
              {multiPhase && simulatedIndexedPhase && (
                <span>· {t('pcrefinement:preview.indexedPhase', { name: simulatedIndexedPhase })}</span>
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
              title={multiPhase && p.ci != null && p.phase
                ? t('pcrefinement:preview.patternItemTooltipPhase', { row: p.row, col: p.col, ci: p.ci.toFixed(3), phase: p.phase })
                : p.ci != null
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
              <span>
                {t('pcrefinement:preview.patternItemLabel', { row: p.row, col: p.col })}
                {/* Several phases: which one this pattern was indexed as. */}
                {multiPhase && p.phase && (
                  <span
                    data-testid="pattern-phase"
                    title={phaseFitsTooltip(p.phaseFits, p.phase, t) || undefined}
                    style={{ marginLeft: 8, fontSize: '8pt', color: colors.accent, fontWeight: 600 }}
                  >
                    {p.phase}
                  </span>
                )}
              </span>
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
function PixelwisePCGroup({ canRun, onCalibrated }) {
  const { t } = useTranslation('pcrefinement');
  const [mode, setMode] = useState('Global (single PC)');
  const [gridStep, setGridStep] = useState(10);
  const [calibrated, setCalibrated] = useState(false);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [pcResult, setPcResult] = useState(null);
  const [showMap, setShowMap] = useState(false);
  const [msg, setMsg] = useState('');
  // The coded refusal of an oversized Hough library, shown with its one-click answer.
  const [refusal, setRefusal] = useState(null);

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
    setRefusal(null);
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
        if (st.status === 'completed' && st.result?.dataset_changed) {
          setMsg(t('pcrefinement:pixelwise.doneOtherFile', { dataset: st.result.dataset }));
          break;
        }
        if (st.status === 'completed') {
          setCalibrated(true);
          if (onCalibrated) onCalibrated();     // the map is now in the controller
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
        if (st.status === 'cancelled') {
          // The run can be cancelled through the backend's cancel route; without
          // this the loop never ended and the panel stayed "running".
          setMsg(t('pcrefinement:controls.optimizationCancelled'));
          break;
        }
        if (st.status === 'failed') {
          const coded = refusalFrom(st);
          if (coded) {
            // A refused Hough library: worded by the notice below with its one-click
            // answer, not by the English text of the backend.
            setRefusal(coded);
            setMsg('');
            break;
          }
          // The file changed under the search: nothing was written; the words are ours, not the backend's English.
          const detail = st.code === 'fileChanged'
            ? t('pcrefinement:controls.searchFileChanged')
            : (st.error || t('pcrefinement:pixelwise.calibrationFailed'));
          setMsg(t('pcrefinement:pixelwise.errorMsg', { detail }));
          break;
        }
      }
    } catch (err) {
      setMsg(t('pcrefinement:pixelwise.errorMsg', { detail: problemText(err, t, 'pcrefinement') || err.message }));
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
      {refusal && (
        <HoughRefusal
          problem={refusal}
          fit={(name) => pcApi.fitPhaseReflectors(name)}
          onFitted={() => setRefusal(null)}
        />
      )}
    </GroupBox>
  );
}

// ---------------------------------------------------------------------------
// Indexing Settings group — matches _create_indexing_settings_group()
// ---------------------------------------------------------------------------
function IndexingSettingsGroup({
  onParamsChange, phases = [], datasetKey = '', onIndexingChanged,
}) {
  const { t } = useTranslation('pcrefinement');
  const [nBands, setNBands] = useState(12);
  const [method, setMethod] = useState('Nelder-Mead');
  const [searchLimit, setSearchLimit] = useState(0.05);

  // The numbers the page itself uses for a run.
  const notify = useCallback(() => {
    onParamsChange({ nBands, method, searchLimit });
  }, [nBands, method, searchLimit, onParamsChange]);

  useEffect(() => { notify(); }, [notify]);

  // nBands is the one parameter of the Hough indexer that lives on the backend
  // controller (per file), so it has to be sent. It is sent only after the user
  // changed it, and again when another file's session is shown: an untouched
  // page leaves the backend at its own default and nothing changes.
  const nBandsTouched = useRef(false);
  const lastSent = useRef(null);
  useEffect(() => {
    if (!nBandsTouched.current) return;
    if (!Number.isFinite(nBands) || nBands < 1) return;
    const sig = `${datasetKey}|${nBands}`;
    if (lastSent.current === sig) return;
    lastSent.current = sig;
    pcApi.updateParams({ n_bands: nBands })
      .then(() => onIndexingChanged?.())
      .catch(() => {});
  }, [nBands, datasetKey, onIndexingChanged]);

  return (
    <GroupBox title={t('pcrefinement:indexingSettings.title')}>
      <FormRow
        label={t('pcrefinement:indexingSettings.nBandsLabel')}
        tip={t('pcrefinement:indexingSettings.nBandsTooltip')}
      >
        <NumberInput
          value={Number.isFinite(nBands) ? nBands : ''}
          onChange={(e) => {
            nBandsTouched.current = true;
            setNBands(parseInt(e.target.value, 10));
          }}
          min={1}
          max={50}
          step={1}
          title={t('pcrefinement:indexingSettings.nBandsTooltip')}
        />
      </FormRow>

      {/* Which reflector families Hough uses, per loaded phase. Replaces the
          three numbers (min_d, f_threshold, max_reflectors) that were shown here
          but never reached the backend: their meaning now lives in the rule of
          each phase's table. */}
      <FormRow
        label={t('pcrefinement:indexingSettings.reflectorsLabel')}
        tip={t('pcrefinement:indexingSettings.reflectorsTooltip')}
      >
        <div data-testid="pc-reflectors">
          {phases.length === 0 && (
            <span style={{ fontSize: '8.5pt', color: colors.textSecondary }}>
              {t('pcrefinement:indexingSettings.reflectorsNoPhase')}
            </span>
          )}
          {phases.map((p) => (
            <div key={p.name} style={{ marginBottom: 4 }}>
              {phases.length > 1 && (
                <div style={{ fontSize: '8.5pt', color: colors.text, fontWeight: 600 }}>{p.name}</div>
              )}
              <ReflectorFamilies
                key={p.name}
                api={loadedPhaseAdapter(p.name, nBands)}
                onChanged={onIndexingChanged}
              />
            </div>
          ))}
        </div>
      </FormRow>

      <Separator />

      <FormRow
        label={t('pcrefinement:indexingSettings.methodLabel')}
        tip={t('pcrefinement:indexingSettings.methodTooltip')}
      >
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

      <FormRow
        label={t('pcrefinement:indexingSettings.searchLimitLabel')}
        tip={t('pcrefinement:indexingSettings.searchLimitTooltip')}
      >
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
function PixelBinningGroup({ binning, setBinning, detWidthMm, setDetWidthMm, shapeW,
                             detWidthFromFile = false, onApplied }) {
  const { t } = useTranslation('pcrefinement');
  const binVal = parseInt(binning, 10) || 1;
  const unbinnedW = (parseInt(shapeW, 10) || 0) * binVal;
  const detWidthMmVal = parseFloat(detWidthMm) || 0;
  const umPerPx = unbinnedW > 0 ? (detWidthMmVal * 1000 / unbinnedW) : null;
  const binnedPx = umPerPx != null ? umPerPx.toFixed(2) : '—';
  const pixelCheck = checkPixelSize(umPerPx);
  const suggestedWidth = pixelCheck.level === 'warn' ? plausibleWidthMm(unbinnedW) : null;

  const handleApplyPixelSize = async () => {
    try {
      await pcApi.updateParams({
        binning: binVal,
        detector_width_mm: detWidthMmVal,
      });
      // The forward preview renders with the same pixel size and binning.
      if (onApplied && umPerPx != null) onApplied({ binning: binVal, pixelSize: umPerPx });
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
        <Label secondary style={{
          fontSize: '9pt', fontFamily: 'monospace',
          color: pixelCheck.level === 'warn' ? colors.red : undefined,
        }}>
          {binnedPx}
        </Label>
      </FormRow>

      {/* Say where the number came from. An estimate that looks like a
          measurement is the problem this whole check exists for. */}
      {detWidthMmVal > 0 && pixelCheck.level !== 'warn' && (
        <Label secondary style={{ fontSize: '8pt', opacity: 0.8, display: 'block', marginBottom: 4 }}>
          {detWidthFromFile
            ? t('pcrefinement:pixelBinning.fromFile')
            : t('pcrefinement:pixelBinning.estimated')}
        </Label>
      )}

      {/* A wrong detector width produces maps that still look right, so this
          has to be loud. See pixelSizeCheck.js. */}
      {pixelCheck.level === 'warn' && (
        <div style={{
          margin: '2px 0 6px', padding: '6px 8px',
          border: `1px solid ${colors.red}`, borderRadius: 4,
          background: `${colors.red}18`, color: colors.text,
          fontSize: '8.5pt', lineHeight: 1.5,
        }}>
          <div style={{ fontWeight: 600, color: colors.red }}>
            {t('pcrefinement:pixelBinning.implausibleTitle')}
          </div>
          <div>
            {t('pcrefinement:pixelBinning.implausibleBody', {
              value: binnedPx, min: TYPICAL_MIN_UM, max: TYPICAL_MAX_UM,
            })}
          </div>
          {suggestedWidth && (
            <div style={{ marginTop: 4 }}>
              {t('pcrefinement:pixelBinning.implausibleHint', {
                width: suggestedWidth.toFixed(1),
              })}
            </div>
          )}
        </div>
      )}

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
function DetectorSettingsGroup({ onDetectorApplied, onPcChanged, onTiltChanged, externalPc,
                                 onPixelSizeApplied, onPcMapAction, datasetName, usedPc, pcInUse,
                                 onUndoPc, onGoToIndexing, renderBands, onSavePc = null }) {
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
  // '' until the file (or the estimate below) fills it — the old default of
  // 0.1 mm was not a possible detector at any pattern size.
  const [detWidthMm, setDetWidthMm] = useState('');
  const [detWidthFromFile, setDetWidthFromFile] = useState(false);
  const [detectorText, setDetectorText] = useState('');
  const [axesText, setAxesText] = useState('');
  const [msg, setMsg] = useState(null);
  const [isError, setIsError] = useState(false);

  // Debounce timer for reactive tilt changes
  const tiltTimerRef = useRef(null);
  const initialLoadRef = useRef(true);
  // Nothing typed into the PC fields applies by itself: "Use these numbers" (PcFields) is the only way a typed
  // PC reaches the backend, and it comes back here with the answer. `pcx..pcz` below are the PC in use as this
  // group last knew it (Bruker's), which Apply Detector Settings sends along.
  const [pcBadge, setPcBadge] = useState(null);
  // Apply Detector Settings was pressed while typed numbers were waiting for "Use these numbers": nothing typed
  // went out, and the page says so for as long as the numbers are still waiting.
  const [typedLeftOut, setTypedLeftOut] = useState(false);
  useEffect(() => { if (pcBadge !== 'unused') setTypedLeftOut(false); }, [pcBadge]);

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

  // "Use these numbers": the typed PC (already Bruker's) goes to the backend. What the group keeps for Apply
  // follows only when the backend took it.
  const typedPressRef = useRef(0);
  const useTypedPc = async (pc) => {
    const mine = ++typedPressRef.current;
    const forFile = openKeyRef.current;
    const answer = onPcChanged ? await onPcChanged(pc[0], pc[1], pc[2]) : { error: t('pcrefinement:detector.failed') };
    if (answer?.data && mine === typedPressRef.current && forFile === openKeyRef.current) { setPcx(String(pc[0])); setPcy(String(pc[1])); setPcz(String(pc[2])); }
    return answer;
  };

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
  //
  // The path alone does not tell: two pattern sources of one file have the same path and a PC each.
  // The name of the loaded entry says the variant.
  const { ebsdLoaded, filePath } = useDataStore();
  const loadedName = useLoadedFilesStore((s) => s.files.find((f) => f.active)?.name || '');
  // Which file the group shows now (the fields start again with it); a press of "Use these numbers" is for the
  // file it was pressed on.
  const openKeyRef = useRef('');
  openKeyRef.current = `${filePath || ''}|${loadedName}|${datasetName || ''}`;
  useEffect(() => {
    initialLoadRef.current = true;
    setMsg(null);
    setTypedLeftOut(false);
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
        setDetWidthFromFile(true);
      } else if (d.shape && d.shape.length >= 2) {
        // The file does not carry the detector geometry. The old fallback was
        // a fixed 0.1 mm, which is not a detector at any pattern size — a user
        // reported exactly this value and asked whether they were supposed to
        // know it. Start from something physically possible instead, and say
        // that it is an assumption.
        const bin = d.binning || 1;
        const est = plausibleWidthMm(d.shape[1] * bin);
        if (est) setDetWidthMm(est.toFixed(3));
        setDetWidthFromFile(false);
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
  }, [ebsdLoaded, filePath, loadedName]); // eslint-disable-line react-hooks/exhaustive-deps

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
      setTypedLeftOut(pcBadge === 'unused');
      if (res.data?.repr) setDetectorText(res.data.repr);
      if (res.data?.axes_repr) setAxesText(res.data.axes_repr);
      // What Apply did to a per-pixel PC map: replaced by the PC shown (it was
      // changed), or kept (it was pressed for a tilt, the binning, ...).
      if (onPcMapAction) onPcMapAction(res.data?.pc_map_action);
      onDetectorApplied(pc, parseFloat(sampleTilt), parseFloat(detTilt));
    } catch (err) {
      setMsg(err.response?.data?.detail || err.message || t('pcrefinement:detector.failed'));
      setIsError(true);
    }
  };

  return (
    <GroupBox title={<>{t('pcrefinement:detector.title')}<PcBadge state={pcBadge} /></>}>
      <PcFields
        fileName={datasetName}
        resetKey={`${filePath || ''}|${loadedName}|${datasetName || ''}`}
        rows={shapeH}
        cols={shapeW}
        usedPc={usedPc}
        fileConvention={preselectedConvention(pcInUse)}
        onBadge={setPcBadge}
        onUse={useTypedPc}
        onUndo={onUndoPc}
        onGoToIndexing={onGoToIndexing}
        renderBands={renderBands}
        onSave={onSavePc}
      />

      {/* Pixel Size & Binning sub-group */}
      <PixelBinningGroup
        binning={binning}
        setBinning={setBinning}
        detWidthMm={detWidthMm}
        detWidthFromFile={detWidthFromFile}
        setDetWidthMm={setDetWidthMm}
        shapeW={shapeW}
        onApplied={onPixelSizeApplied}
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
      {msg && !isError && typedLeftOut && pcBadge === 'unused' && (
        <div data-testid="pc-typed-not-used" role="status" style={{ fontSize: '9pt', color: colors.orange, margin: '2px 0', wordBreak: 'break-word' }}>
          {t('pcrefinement:detector.typedNotUsed')}
        </div>
      )}
    </GroupBox>
  );
}

// ---------------------------------------------------------------------------
// Right controls panel — matches _create_controls_panel()
// ---------------------------------------------------------------------------
/** A failed phase operation ({code, message, params}) in the user's language. */
function phaseOpErrorText(t, error) {
  const code = error?.code || 'generic';
  if (code === 'generic' && error?.message) return error.message;
  return t(`pcrefinement:phaseErrors.${code}`, {
    ...(error?.params || {}),
    defaultValue: error?.message || t('pcrefinement:controls.phaseLoadFailed'),
  });
}

function ControlsPanel({
  phaseLoaded,
  // Counts the visits on which the page found the session without a phase.
  sessionReset = 0,
  phases = [],
  onPhasesChanged,
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
  getSelectedIdx,
  onPatternsIndexed,
  onSelectPattern,
  onPcChanged,
  onTiltChanged,
  pcMapInfo,
  pcSource,
  pcMapAction,
  onPcMapAction,
  onGridCalibrated,
  onPixelSizeApplied,
  externalPc,
  onOptimizedPc,
  parentName,
  activeDatasetName,
  propagated,
  setPropagated,
  usedPc,          // the PC in use for the open file (Bruker x, y, z), null when the file has none
  pcStatusSeen,    // the backend has said what the PC is (before that the head shows nothing)
  usedPcIsMap,     // the PC in use is a per-pixel map: usedPc is its mean
  typedPcNotStored, // a typed PC the calibration store did not take: indexing will not use it
  pcInUse,         // `pc_in_use` of /api/pc/status: where the PC came from and the open file's convention
  onNavigate,      // the app's own page switch ('indexing', ...); null: the page cannot switch
  onPcRestored,    // the PC changed behind the page's back (a step back): load it again from the backend
  patternBase64,
  segments,
  pcx,
  pcy,
}) {
  const { t } = useTranslation(['pcrefinement', 'common']);
  // This panel calls askPrompt() for the browser fallback of "Load CIF";
  // the hook lived only in the parent, so that path threw ReferenceError.
  const [askPrompt, promptProps] = usePrompt();
  // "Take PC from..." (a saved calibration or another file): opened from the hint of a file without a PC and from
  // the row of Pick patterns; the window itself is mounted by the calibration panel.
  const [takePcOpen, setTakePcOpen] = useState(false);
  // The file picker for "Another file" is the one the page uses for a CIF: the app's own dialog in the desktop
  // app, the themed prompt for a path in the browser. `done(path)` is called with the chosen path.
  const pickPcFile = (current, done) => {
    if (window.electronAPI?.openFile) {
      Promise.resolve(window.electronAPI.openFile({
        filters: [{ name: t('pcrefinement:takePc.fileFilter'), extensions: ['ang', 'osc', 'h5', 'h5oina', 'hdf5'] }],
      })).then((p) => { if (typeof p === 'string' && p.trim()) done(p.trim()); }).catch(() => {});
      return;
    }
    askPrompt({
      title: t('pcrefinement:takePc.promptTitle'),
      message: t('pcrefinement:takePc.promptMessage'),
      defaultValue: current,
      placeholder: t('pcrefinement:takePc.promptPlaceholder'),
      submitLabel: t('pcrefinement:takePc.promptSubmit'),
      onSubmit: (val) => { if (val.trim()) done(val.trim()); },
    });
  };
  // Progress / cancel
  const [progress, setProgress] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [taskId, setTaskId] = useState(null);
  const [cancelling, setCancelling] = useState(false);   // Cancel pressed, the run is winding down
  const pollRef = useRef(null);
  // Find PC has steps before the job exists (picking patterns, the request that starts it): Cancel pressed
  // in one of them cannot stop a job yet, so it is noted here and the step in flight ends the run itself.
  const inFlightRef = useRef(null);                       // 'pick' | 'start' | null
  const cancelRequestedRef = useRef(false);
  const [picking, setPicking] = useState(false);          // patterns are being picked from the scan
  // The lines the page says about a Find PC run (picking, searching, finished), one under the other.
  const statusLinesRef = useRef([]);
  const [msgScope, setMsgScope] = useState('refine');     // which button the status line belongs to

  // Phase state
  const [cifPath, setCifPath] = useState('');
  const [phaseInfo, setPhaseInfo] = useState(null);
  const [phaseMsg, setPhaseMsg] = useState(null);
  const [phaseError, setPhaseError] = useState(false);
  const [phaseLoading, setPhaseLoading] = useState(false);

  // Phase library picker (reuses the Hough indexing CIF picker)
  const [phasePickerOpen, setPhasePickerOpen] = useState(false);
  const [discoveredFiles, setDiscoveredFiles] = useState([]);
  const discoveredFilesRef = useRef([]);
  discoveredFilesRef.current = discoveredFiles;
  const [discoveredGroups, setDiscoveredGroups] = useState([]);

  // Several phases (e.g. austenite + ferrite of a duplex steel). The picker
  // ticks and unticks phases; each change is one request, and requests run one
  // after another so a quick second click does not race the first.
  const phaseNames = phases.map((p) => p.name);
  const phaseNamesRef = useRef(phaseNames);
  phaseNamesRef.current = phaseNames;
  // The loaded phases WITH their files: a phase is identified by its file, not by
  // its name (two different files called Al.cif are two files).
  const phasesRef = useRef(phases);
  phasesRef.current = phases;
  const [addedPaths, setAddedPaths] = useState([]);   // typed-in paths we loaded
  const phaseQueueRef = useRef(null);
  if (phaseQueueRef.current === null) phaseQueueRef.current = createSerialQueue();
  const patternsRef = useRef(patterns);
  patternsRef.current = patterns;

  // Index pattern state
  const [indexMsg, setIndexMsg] = useState(null);
  const [indexError, setIndexError] = useState(false);
  // The coded refusal of an oversized Hough library, shown with its one-click answer.
  const [refusal, setRefusal] = useState(null);
  // One place for what a failed Index Pattern / Index All says: a refused Hough library
  // is worded by the notice below with its one-click answer, anything else is the
  // detail of the backend as before.
  const reportIndexFailure = (err) => {
    const coded = refusalFrom(err);
    if (coded) {
      setRefusal(coded);
      setIndexMsg(null);
      setIndexError(false);
      return;
    }
    setRefusal(null);
    const detail = err.response?.data?.detail;
    const msg = typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : err.message || t('pcrefinement:controls.indexFailed');
    setIndexMsg(msg);
    setIndexError(true);
  };

  // Global refine state
  const [globalMsg, setGlobalMsg] = useState(null);
  const [globalError, setGlobalError] = useState(false);

  // Source PC drift: the Aztec per-pixel PC map of the loaded dataset
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
  const [bandsShown, setBandsShown] = useState(false);    // "Show the bands over a pattern" is open
  const [keptFor, setKeptFor] = useState(null);           // "Keep this PC anyway" was pressed for this file
  // Where the patterns of the shown result came from ({kind: 'pick' | 'all' | 'user', noQuality}), and the
  // pick the page made last: its positions as flat indices and what the pick said about itself.
  const [resultFrom, setResultFrom] = useState({ kind: 'user' });
  const searchFromRef = useRef({ kind: 'user' });
  const pickedRef = useRef(null);
  // The positions "Find PC again with other patterns" has given up so far, so a third press does not
  // come back to the first list. A new Find PC or Pick patterns starts the count again.
  const givenUpRef = useRef([]);
  const [inConvention, setInConvention] = useState(null); // the result's PC in the file's own convention
  const [undone, setUndone] = useState(null);             // the step back was made: {file, previous, notRestored}

  // The page found the session without a phase on a visit (a processed copy of
  // the data is a new dataset with a session of its own): "Phase loaded: Al",
  // its info card, the last refine's result card and the run messages describe
  // the old session and must not stay on screen. Not tied to phaseLoaded itself:
  // a phase the user removes here keeps its "removed" message.
  useEffect(() => {
    if (!sessionReset) return;
    setPhaseInfo(null);
    setPhaseMsg(null);
    setPhaseError(false);
    setIndexMsg(null);
    setIndexError(false);
    setGlobalMsg(null);
    setGlobalError(false);
    setResult(null);
    setResultMsg(null);
    setBandsShown(false);
    setKeptFor(null);
    setUndone(null);
    setSeedMsg(null);
    setSeedError(false);
    setRefusal(null);
  }, [sessionReset]);
  const [pcCopied, setPcCopied] = useState(false);

  // Indexing params forwarded from IndexingSettingsGroup
  const paramsRef = useRef({ method: 'Nelder-Mead', searchLimit: 0.05 });
  const handleParamsChange = useCallback((p) => {
    paramsRef.current = p;
    onIndexingParamsChange(p);
  }, [onIndexingParamsChange]);

  // A refine indexed every calibration pattern at the refined PC: the Global
  // CI, and the CI and phase of each pattern, are the ones at that PC. An
  // answer without them (an older backend) leaves no stale number behind.
  const applyRefinedCis = (r) => {
    if (onPcMapAction) onPcMapAction(r?.pc_map_action);
    setGlobalCi(r?.global_ci ?? null);
    setIndexMsg(null);  // "Indexed N patterns — Global CI: x" belongs to the old PC
    if (Array.isArray(r?.pattern_results) && onPatternsIndexed) onPatternsIndexed(r.pattern_results);
  };

  // A refine indexed every calibration pattern at the refined PC. Draw the lines
  // of the pattern that is selected NOW (the run takes a while, the user may
  // have picked another one), not those of the first pattern.
  const showRefinedSegments = (r) => {
    if (!onSegmentsUpdate) return;
    const sel = getSelectedIdx ? getSelectedIdx() : null;
    const entry = sel != null && Array.isArray(r?.pattern_results)
      ? r.pattern_results.find((e) => e.index === sel) : null;
    if (entry && Array.isArray(entry.segments)) onSegmentsUpdate(entry.segments, sel);
    else if (sel == null && r?.segments) onSegmentsUpdate(r.segments);
  };

  // The CI that goes with the lines drawn by showRefinedSegments: the selected
  // pattern's. The top-level `ci` of a refine result is pattern 0's, so with
  // pattern 3 selected the header showed pattern 0's CI next to pattern 3's lines.
  const refinedCiOf = (r) => {
    const sel = getSelectedIdx ? getSelectedIdx() : null;
    if (sel == null) return r?.ci ?? null;
    const entry = Array.isArray(r?.pattern_results)
      ? r.pattern_results.find((e) => e.index === sel) : null;
    return entry?.ci ?? null;
  };

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
        // the backend reports a fraction of the run, the bar counts percent
        if (typeof d.progress === 'number') setProgress(d.progress * 100);
        if (d.status === 'done' || d.status === 'completed') {
          stopPolling();
          setIsRunning(false);
          setCancelling(false);    // Cancel came too late: the run had finished and its result stands
          cancelRequestedRef.current = false;
          setTaskId(null);
          setProgress(100);
          setGlobalMsg(null);      // "Cancelling..." / "running" no longer describe it
          if (d.result?.dataset_changed) {
            // The file was switched while the refine ran: its PC went to ITS dataset (the
            // store), and is not shown here as the PC of the file in front of the user.
            // A PC the store did not take (pc_stored false) is said not to be stored.
            setGlobalMsg(t(d.result.pc_stored === false
              ? 'pcrefinement:controls.optimizationOtherFileNotStored'
              : 'pcrefinement:controls.optimizationOtherFile', { dataset: d.result.dataset }));
            setGlobalError(false);
            return;
          }
          setResult(d.result || d);
          setResultFrom(searchFromRef.current);
          setBandsShown(false);
          setKeptFor(null);
          setUndone(null);
          if (d.result?.mode === 'find') {
            setGlobalMsg([...statusLinesRef.current, t('pcrefinement:find.finished')].join('\n'));
          }
          setResultMsg(d.result?.mode === 'find' ? null : t('pcrefinement:controls.optimizationComplete'));
          setCiValue(refinedCiOf(d.result));
          // A Find PC result, and one that asks what to do, shows a pattern with its bands: one must be
          // selected for that.
          if ((d.result?.mode === 'find' || needsDecision(d.result)) && onSelectPattern
              && getSelectedIdx && getSelectedIdx() == null && patternsRef.current.length > 0) {
            onSelectPattern(0);
          }
          // Update segments + detector PC spinboxes with optimized values
          showRefinedSegments(d.result);
          if (d.result?.mean_pc?.length === 3 && onOptimizedPc) onOptimizedPc(d.result.mean_pc, d.result.mode);
          // Several phases: which one each refined pattern is indexed as. The
          // result lists them in the order the patterns were sent.
          if (Array.isArray(d.result?.pattern_phases) && onPatternsIndexed) {
            onPatternsIndexed(d.result.pattern_phases.flatMap((e) => {
              const index = patternsRef.current[e.index]?.index;
              return index == null ? [] : [{
                index, phase_name: e.phase_name, phase_fits: e.phase_fits,
                ...(e.ci != null ? { ci: e.ci } : {}),
              }];
            }));
          }
          applyRefinedCis(d.result);
        } else if (d.status === 'cancelled') {
          stopPolling();
          setIsRunning(false);
          setTaskId(null);
          setProgress(0);
          setCancelling(false);
          cancelRequestedRef.current = false;
          setGlobalMsg(t('pcrefinement:controls.optimizationCancelled'));
          setGlobalError(false);
        } else if (d.status === 'error' || d.status === 'failed') {
          stopPolling();
          setIsRunning(false);
          setTaskId(null);
          setCancelling(false);
          cancelRequestedRef.current = false;
          const coded = refusalFrom(d);
          if (coded) {
            // A refused Hough library: worded by the notice below with its one-click
            // answer, not by the English text of the backend.
            setRefusal(coded);
            setGlobalMsg(null);
            setGlobalError(false);
            return;
          }
          // The file was replaced under the same name while the search ran: nothing was written.
          const failure = d.code === 'fileChanged'
            ? t('pcrefinement:controls.searchFileChanged')
            : (d.error || d.message || t('pcrefinement:controls.optimizationFailed'));
          setResultMsg(failure);
          // The result panel is only there after a result: a failed search must be said where the run was started.
          setGlobalMsg(failure);
          setGlobalError(true);
        }
      } catch {
        pollFailCountRef.current += 1;
        // If backend is unreachable for 5+ consecutive polls, abort
        if (pollFailCountRef.current >= 5) {
          stopPolling();
          setIsRunning(false);
          setTaskId(null);
          setCancelling(false);
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
      onPhaseLoaded({ ...res.data, path });
    } catch (err) {
      const detail = err.response?.data?.detail;
      setPhaseMsg(detail && typeof detail === 'object'
        ? phaseOpErrorText(t, pathErrorFrom(err))
        : (detail || err.message || t('pcrefinement:controls.phaseLoadFailed')));
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

  // ── Several phases ──
  // The core operations below talk to the backend and keep the page's phase
  // list in step; they return a result instead of showing it, so the picker's
  // path field (which shows its own message) and the tick boxes (which show
  // the status line) can both use them.
  // `target` is a path or a record ({path, real_path}): the resolved path is what
  // says that two spellings are one file.
  const addPhaseCore = async (target) => {
    const rec = typeof target === 'string' ? asRecord(target, discoveredFilesRef.current) : target;
    const path = rec.path;
    const stem = phaseStem(path);
    const same = loadedPhaseFor(phasesRef.current, rec);
    if (same) return { ok: true, already: true, name: same.name };
    // Another FILE with the same name: refused (the backend would too), and said
    // so, instead of looking like success or silently swapping the phase.
    const clash = sameNameOtherFile(phasesRef.current, rec);
    if (clash) {
      return {
        ok: false,
        error: {
          code: 'phase_same_name', message: '',
          params: { name: clash.name, loaded_path: clash.path || '', path },
        },
      };
    }
    if (phaseNamesRef.current.length >= MAX_PC_PHASES) {
      return { ok: false, error: { code: 'phase_limit', message: '', params: { max: MAX_PC_PHASES } } };
    }
    try {
      const res = await pcApi.addPhase(path);
      phaseNamesRef.current = res.data.phases.map((p) => p.name);
      setAddedPaths((prev) => (prev.some((p) => samePath(p, path)) ? prev : [...prev, path]));
      setPhaseInfo(res.data);
      clearPerPatternPhases();
      onPhasesChanged(res.data.phases);
      return { ok: true, name: res.data.phase_name || stem, count: res.data.phases.length };
    } catch (err) {
      return { ok: false, error: pathErrorFrom(err) };
    }
  };

  // "Phase per pattern" belongs to the phases it was computed with: when the set
  // changes it is no longer true of the patterns.
  const clearPerPatternPhases = () => {
    setResult((prev) => (prev && (prev.pattern_phases || prev.phase_names)
      ? { ...prev, pattern_phases: null, phase_names: null } : prev));
  };

  const removePhaseCore = async (name) => {
    const gone = phasesRef.current.find((p) => p.name === name);
    try {
      const res = await pcApi.removePhase(name);
      phaseNamesRef.current = res.data.phases.map((p) => p.name);
      setAddedPaths((prev) => prev.filter((p) => !(gone?.path ? samePath(p, gone.path)
        : phaseStem(p) === name)));
      setPhaseInfo(res.data.phases[0] ? { phase_name: res.data.phases[0].name, ...res.data.phases[0] } : null);
      clearPerPatternPhases();
      onPhasesChanged(res.data.phases);
      return { ok: true };
    } catch (err) {
      return { ok: false, error: pathErrorFrom(err) };
    }
  };

  // Show the outcome of a tick / untick / remove in the status line.
  const reportPhaseOp = (res, okText) => {
    if (res.ok) {
      setPhaseMsg(okText);
      setPhaseError(false);
    } else {
      setPhaseMsg(phaseOpErrorText(t, res.error));
      setPhaseError(true);
    }
  };

  const runPhaseJob = (job) => phaseQueueRef.current(async () => {
    setPhaseLoading(true);
    try { return await job(); } finally { setPhaseLoading(false); }
  });

  // Tick or untick one phase of the picker.
  const togglePhase = (file) => runPhaseJob(async () => {
    // Ticked means THIS file is loaded; unticking removes exactly that phase and
    // never one that only shares its name.
    const loaded = loadedPhaseFor(phasesRef.current, file);
    if (loaded) {
      reportPhaseOp(await removePhaseCore(loaded.name),
                    t('pcrefinement:controls.phaseRemoved', { name: loaded.name }));
    } else {
      const res = await addPhaseCore(file);
      reportPhaseOp(res, t('pcrefinement:controls.phaseAdded', { name: res.name, count: res.count }));
    }
  });

  // The path field of the picker: a phase that is not in the library.
  const addPhaseByPath = (path) => runPhaseJob(async () => {
    // A path that is a library file (possibly spelled differently from the
    // listing) selects the library's entry instead of adding a second one.
    const { record, entry } = await resolvePhasePath({
      method: 'hough', rawPath: path, files: discoveredFilesRef.current,
      check: indexApi.phaseFromPath,
    });
    const res = await addPhaseCore(entry || record || path);
    if (entry && res.ok) res.inLibrary = true;
    if (res.ok && !res.already) {
      setPhaseMsg(t('pcrefinement:controls.phaseAdded', { name: res.name, count: res.count }));
      setPhaseError(false);
    }
    return res;
  });

  // `null` is PhaseDropdown's "+ Add file manually…" signal (the picker below
  // has a path field, so it no longer sends it; kept for any other caller).
  const handlePickPhase = (file) => {
    if (!file) { setPhasePickerOpen(false); handleLoadPhase(); return; }
    togglePhase(file);
  };

  // The picker's All / None buttons hand over the whole wanted set at once.
  const handleSetAllPhases = (paths) => {
    const plan = planPhaseSync(phasesRef.current, paths, discoveredFilesRef.current);
    const resulting = phaseNamesRef.current.length - plan.remove.length + plan.add.length;
    if (resulting > MAX_PC_PHASES) {
      setPhaseMsg(t('pcrefinement:controls.phaseLimit', { max: MAX_PC_PHASES }));
      setPhaseError(true);
      return;
    }
    runPhaseJob(async () => {
      let last = { ok: true };
      for (const name of plan.remove) last = await removePhaseCore(name);
      for (const path of plan.add) {
        last = await addPhaseCore(path);
        if (!last.ok) break;
      }
      reportPhaseOp(last, t('pcrefinement:controls.phasesHeader', { count: phaseNamesRef.current.length }));
    });
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
    setRefusal(null);
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
        // The pattern selected when the run FINISHES (it takes a while).
        const selNow = getSelectedIdx ? getSelectedIdx() : currentPatternIdx;
        const targetIdx = selNow != null ? selNow : 0;
        const currentResult = d.results.find(r => r.index === targetIdx) || d.results[0];
        if (currentResult.ci != null) {
          setCiValue(currentResult.ci);
        }
        if (currentResult.segments && onSegmentsUpdate) {
          onSegmentsUpdate(currentResult.segments, selNow != null ? currentResult.index : undefined);
        }
        // Nothing was selected: the overlay shown is the first pattern's, so
        // select it. The picture, the CI and the forward-sim preview (which needs
        // a selected pattern, and follows its winning phase) belong to it.
        if (selNow == null && onSelectPattern) {
          const pos = patterns.findIndex((p) => p.index === currentResult.index);
          if (pos >= 0) onSelectPattern(pos);
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
      reportIndexFailure(err);
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
    setRefusal(null);
    const forPatternIdx = currentPatternIdx;   // the pattern this request is about
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
        onSegmentsUpdate(d.segments, forPatternIdx);
      }

      // Update CI in the pattern list (mirrors handlePatternsIndexed for Index All)
      if (d.ci != null && currentPatternIdx != null && onPatternsIndexed) {
        onPatternsIndexed([{ index: patterns[currentPatternIdx].index, ci: d.ci, phase_name: d.phase_name, phase_fits: d.phase_fits }]);
      }
    } catch (err) {
      setIsRunning(false);
      reportIndexFailure(err);
    }
  };

  // The Hough indexer was rebuilt (other reflector families, other nBands): the
  // bands and the CI shown for the selected pattern belong to the old one, so
  // index it again. Stable identity; reads the latest page state when called.
  const indexingChangedRef = useRef(null);
  indexingChangedRef.current = () => {
    if (currentPatternIdx != null && patterns[currentPatternIdx]) handleIndexPattern();
  };
  const handleIndexingChanged = useCallback(() => indexingChangedRef.current?.(), []);

  // The flat scan indices of calibration patterns (the job takes positions as row * columns + column).
  const flatIndices = (list) => {
    const gs = useDataStore.getState().gridShape || [1, 1];
    const nCols = gs[1] || 1;
    return list.map((pt) => pt.row * nCols + pt.col);
  };

  // What the page says about a Find PC run so far: its lines, one under the other.
  const sayLines = (lines) => {
    statusLinesRef.current = lines;
    setGlobalMsg(lines.join('\n'));
    setGlobalError(false);
  };

  // One run of the background search for Find PC and for Global PC Refine: the same job, the same progress
  // bar, the same result panel. `mode` is 'find' or 'refine'; `lines` are the lines already said (Find PC).
  const runSearch = async (mode, indices, count, lines = []) => {
    inFlightRef.current = 'start';
    setIsRunning(true);
    setCancelling(cancelRequestedRef.current);
    setProgress(0);
    setRefusal(null);
    setUndone(null);
    setKeptFor(null);
    setMsgScope(mode === 'find' ? 'find' : 'refine');
    // where the patterns come from: the page's own pick (same positions), or a list somebody made
    const picked = pickedRef.current;
    const same = !!picked && picked.key === [...indices].sort((a, b) => a - b).join(',');
    searchFromRef.current = same && picked.blocks
      ? { kind: 'pick', noQuality: picked.noQuality, excluded: picked.excluded }
      : same ? { kind: 'all' } : { kind: 'user' };
    if (mode === 'find') sayLines([...lines, t('pcrefinement:find.searching', { count })]);
    else { setGlobalMsg(t('pcrefinement:controls.runningGlobalRefine')); setGlobalError(false); }
    const p = paramsRef.current;
    try {
      const res = await pcApi.optimize(indices, p.method || 'Nelder-Mead', p.searchLimit || 0.05, mode);
      const d = res.data;
      if (d.task_id) {
        setTaskId(d.task_id);
        if (cancelRequestedRef.current) {
          // Cancel was pressed while the job was being started: stop it now that it has a name. The
          // polling reports "cancelled" when the run has ended.
          setCancelling(true);
          setGlobalMsg(t('pcrefinement:controls.optimizationCancelling'));
          try { await pcApi.cancelOptimize(d.task_id); } catch { /* the polling tells how it ended */ }
        } else if (mode === 'find') {
          // the "Searching ..." line stands
        } else {
          setGlobalMsg(t('pcrefinement:controls.globalRefineStarted', { count }));
        }
      } else {
        setIsRunning(false);
        setProgress(100);
        setResult(d.result || d);
        setResultFrom(searchFromRef.current);
        setBandsShown(false);
        setGlobalMsg(t('pcrefinement:controls.globalRefineComplete'));
        const r = d.result || d;
        setCiValue(refinedCiOf(r));
        applyRefinedCis(r);
        // Update segments + detector PC spinboxes with optimized values
        showRefinedSegments(r);
        if (r?.mean_pc?.length === 3 && onOptimizedPc) onOptimizedPc(r.mean_pc, r.mode);
      }
    } catch (err) {
      setIsRunning(false);
      setCancelling(false);
      const detail = err.response?.data?.detail;
      // a coded refusal ("Find PC needs 5 patterns") in the reader's language, else the backend's sentence
      const gMsg = isPcJobRunning(err) ? pcJobRunningText(t)
        : (typeof detail === 'string' ? problemText(err, t, 'pcrefinement') : '')
          || (detail ? (typeof detail === 'string' ? detail : JSON.stringify(detail))
            : err.message || t('pcrefinement:controls.globalRefineFailed'));
      setGlobalMsg(gMsg);
      setGlobalError(true);
    } finally {
      inFlightRef.current = null;
    }
  };

  // Global PC Refine: starts at the PC the page holds, over the calibration patterns on the list.
  const handleGlobalRefine = async () => {
    setMsgScope('refine');
    if (patterns.length === 0) {
      setGlobalMsg(t('pcrefinement:controls.addPatternsFirst'));
      setGlobalError(true);
      return;
    }
    cancelRequestedRef.current = false;
    await runSearch('refine', flatIndices(patterns), patterns.length);
  };

  // The open file's name, asked of the backend: a file switch during the seconds a pick takes would
  // otherwise search the new file at positions picked in the old one.
  const openFileName = async () => {
    try {
      return (await ebsdApi.info()).data?.active_dataset ?? null;
    } catch {
      return null;
    }
  };

  // Calibration patterns from the whole scan, one per ninth; they replace the list. Returns the picked
  // positions ({row, col, ...}). The page's list follows through the 'pc-patterns-updated' event.
  // `lines` are the lines already said by a Find PC run; the answer replaces the "Picking" line.
  const pickPatterns = async (lines = [], exclude = []) => {
    setPicking(true);
    setMsgScope('find');
    sayLines([...lines, t('pcrefinement:find.picking')]);
    try {
      const d = (await (exclude.length ? pcApi.autoPick(true, exclude) : pcApi.autoPick(true))).data || {};
      // what this pick was, to word where a later search's patterns came from: a scan of nine patterns or
      // fewer is used completely, a larger one is split into 3 x 3 blocks
      const notes = Array.isArray(d.notes) ? d.notes : [];
      pickedRef.current = {
        key: flatIndices(d.patterns || []).sort((a, b) => a - b).join(','),
        blocks: Number(d.n_sampled) > 9, noQuality: notes.includes('blockWithoutQuality'),
        // "other patterns": the positions of earlier lists were left out, so these are the next best ones
        excluded: exclude.length > 0,
      };
      window.dispatchEvent(new Event('pc-patterns-updated'));
      const how = { mean: 'backgroundMean', file: 'backgroundFile', none: 'backgroundNone' }[d.background] || 'backgroundNone';
      const noteLines = notes
        .map((code) => t(`pcrefinement:find.pickNote.${code}`, { defaultValue: '' }))
        .filter(Boolean);
      sayLines([...lines, [
        t('pcrefinement:find.picked', { count: d.n_patterns, background: t(`pcrefinement:find.${how}`) }),
        ...noteLines,
      ].join(' ')]);
      return d.patterns || [];
    } catch (err) {
      // a search is running (the backend refuses to change the list under it): said in words
      setGlobalMsg(isPcJobRunning(err) ? pcJobRunningText(t) : (problemText(err, t, 'pcrefinement') || err.message));
      setGlobalError(true);
      throw err;
    } finally {
      setPicking(false);
    }
  };

  // Find PC: needs no starting value. With fewer than FIND_MIN_PATTERNS patterns on the list, nine
  // are picked from the scan first (and stay on the list, where they can be looked at and removed).
  // `exclude` ([[row, col], ...]) is "other patterns": the pick leaves those positions out, always.
  const startFind = async (exclude) => {
    cancelRequestedRef.current = false;
    statusLinesRef.current = [];
    let list = patterns;
    let lines = [];
    if (patterns.length < FIND_MIN_PATTERNS || exclude.length > 0) {
      inFlightRef.current = 'pick';
      setIsRunning(true);
      setCancelling(false);
      setProgress(0);
      setRefusal(null);
      setUndone(null);
      setKeptFor(null);
      try {
        const before = await openFileName();
        list = await pickPatterns([], exclude);
        lines = statusLinesRef.current;
        const after = await openFileName();
        if (cancelRequestedRef.current) {
          cancelRequestedRef.current = false;
          setCancelling(false);
          setIsRunning(false);
          sayLines([...lines, t('pcrefinement:find.cancelledBeforeSearch')]);
          return;
        }
        if (after !== before) {
          setIsRunning(false);
          setGlobalMsg(t('pcrefinement:find.fileChanged'));
          setGlobalError(true);
          return;
        }
      } catch {
        cancelRequestedRef.current = false;
        setCancelling(false);
        setIsRunning(false);
        return;
      } finally {
        inFlightRef.current = null;
      }
    }
    await runSearch('find', flatIndices(list), list.length, lines);
  };
  const handleFindPc = () => { givenUpRef.current = []; return startFind([]); };
  // "Find PC again with other patterns": the pick is deterministic, so the positions of the list (and of the
  // lists given up before it) are left out and the next best pattern of each block comes.
  const handleFindAgain = () => {
    const seen = new Set(givenUpRef.current.map(([r, c]) => `${r},${c}`));
    patterns.forEach((pt) => {
      if (!seen.has(`${pt.row},${pt.col}`)) givenUpRef.current.push([pt.row, pt.col]);
    });
    return startFind(givenUpRef.current.map((pos) => [...pos]));
  };

  // One step back for the file the result was given to: the backend puts the PC of before back, the page
  // loads it again from there. Returns what PcUndoLine needs to show a refusal.
  // `{data}` when the step was taken back (the page then loads the PC again from the backend), `{gone: true}` when
  // there is no step any more, `{error}` for a refusal in words.
  const putBackFile = async (file) => {
    try {
      const d = (await pcApi.putBack(file)).data || {};
      if (onPcRestored) onPcRestored();
      return { data: d };
    } catch (err) {
      if (problemCode(err) === 'nothingToPutBack') return { gone: true };
      if (isPcJobRunning(err)) return { error: pcJobRunningText(t) };
      return { error: problemText(err, t, 'pcrefinement') || err?.message || String(err) };
    }
  };
  const handleUndo = async (file) => {
    const answer = await putBackFile(file);
    if (!answer.data) return answer;
    const d = answer.data;
    setUndone({
      file,
      previous: { pc: d.pc ?? null, had_pc: !!d.had_pc },
      notRestored: Array.isArray(d.not_restored) ? d.not_restored : [],
    });
    setBandsShown(false);
    setKeptFor(null);
    return undefined;
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
      // An open page works on its own copy of the PC: show what the backend gave it.
      if (d.pc_page_updated) onGridCalibrated?.();
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

  // Cancel asks the backend to stop the run. It used to stop only the polling:
  // the run went on, wrote its PC later unseen, and every new run or phase
  // change got a 409 until it ended. The page now keeps polling until the
  // backend reports "cancelled" (the run stops after its current pattern and
  // writes nothing). A run the backend no longer knows is ended here.
  const handleCancel = async () => {
    if (!taskId && inFlightRef.current) {
      // Find PC is still picking patterns, or the request that starts the job is on its way: there is no
      // job to stop yet. Note it; the step in flight ends the run (or stops the job as soon as it has a name).
      cancelRequestedRef.current = true;
      setCancelling(true);
      setGlobalMsg(t('pcrefinement:controls.optimizationCancelling'));
      setGlobalError(false);
      return;
    }
    if (!taskId) {
      stopPolling(); setIsRunning(false); setProgress(0);
      return;
    }
    setCancelling(true);
    setGlobalMsg(t('pcrefinement:controls.optimizationCancelling'));
    setGlobalError(false);
    try {
      await pcApi.cancelOptimize(taskId);
    } catch (err) {
      if (err?.response?.status === 404) {
        stopPolling(); setIsRunning(false); setTaskId(null); setProgress(0); setCancelling(false);
        return;
      }
      // The request did not arrive: the run goes on; let the user try again.
      setCancelling(false);
      setGlobalMsg(t('pcrefinement:controls.runningGlobalRefine'));
      toast.warning(err?.response?.data?.detail?.message || err?.message || String(err));
    }
  };

  const canRun = detectorReady && phaseLoaded && !isRunning && !picking;
  const driftExport = useImageExport();
  // The result is the PC in use only while nothing else has changed it since (a typed PC, another file,
  // a step back): otherwise "PC in use for ..." would say something that is no longer true.
  const resultPc = result && Array.isArray(result.mean_pc) && result.mean_pc.length === 3 ? result.mean_pc : null;
  // ...and only when the calibration store took it: a PC that lives on this page only is not "in use".
  const resultInUse = !!resultPc && samePc(resultPc, usedPc) && result.pc_stored !== false;
  // The result in the convention of the open file (Oxford, EDAX/TSL): the backend converts. A Bruker file, or
  // one whose format does not say, gets no such line.
  const fileConvention = pcInUse?.convention_known && ['oxford', 'tsl'].includes(pcInUse.convention)
    ? pcInUse.convention : null;
  const patternSize = Array.isArray(result?.pattern_size) && result.pattern_size.length === 2
    ? result.pattern_size : null;
  const resultKey = resultPc ? resultPc.join(',') : '';
  const sizeKey = patternSize ? patternSize.join('x') : '';
  useEffect(() => {
    setInConvention(null);                    // nothing stale: the old numbers go before the new ones come
    if (!resultKey || !resultInUse || !fileConvention || !patternSize) return undefined;
    let live = true;
    pcApi.convert(resultKey.split(',').map(Number), 'bruker', fileConvention, patternSize).then((res) => {
      const out = res.data?.pc;
      if (live && Array.isArray(out) && out.length === 3) setInConvention({ convention: fileConvention, pc: out });
    }).catch(() => {});
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resultKey, resultInUse, fileConvention, sizeKey]);

  // ---- Save as calibration, Apply to other files, the question after a search, "Compare with a saved calibration"
  const [saveFor, setSaveFor] = useState(null);          // {dataset, pc, quality, fromFar, forQuestion}
  const [savedFar, setSavedFar] = useState(null);        // {dataset, name}: the far file's result was saved
  const [asked, setAsked] = useState(null);              // {key, state}: the question after a search was answered
  const [hasCalibrations, setHasCalibrations] = useState(false);
  // "Only this file" leaves the calibration for good: its card is not offered to this file again until the file is
  // loaded again (the mark of the load is the one the calibration panel uses for the offers put off with "Not now")
  const loadMark = useDataStore((s) => s.ebsdInfo?.request_id ?? s.ebsdInfo?.file_path ?? null);
  const [leftCalibration, setLeftCalibration] = useState(null);   // {mark, dataset, id}
  const silencedIds = leftCalibration && leftCalibration.mark === loadMark && leftCalibration.dataset === activeDatasetName
    ? [leftCalibration.id] : null;
  const applyFlow = useApplyFlow((st) => st.flow);
  const openSave = (opts = {}) => setSaveFor({
    dataset: opts.dataset || activeDatasetName, pc: opts.pc || usedPc, quality: opts.quality ?? null,
    fromFar: !!opts.fromFar, forQuestion: !!opts.forQuestion,
  });
  const openTakePc = () => setTakePcOpen(true);
  const openApplyFlow = (calibration, origin) => useApplyFlow.getState().open(calibration, origin);
  const asking = questionKey(result);
  const questionOpen = !!asking && asked?.key !== asking;
  // One main button at a time: while the result panel shows the PC with its own main button, Global PC Refine and
  // Index All take the plain style
  const resultHasMainButton = !!result && !undone && !!resultPc && !!result.dataset && resultInUse;
  // the decision box offers the saved calibrations only when there is one
  const needsDecisionNow = !!result && needsDecision(result);
  useEffect(() => {
    if (!needsDecisionNow) { setHasCalibrations(false); return undefined; }
    let live = true;
    pcCalibrationsApi.list().then((res) => {
      if (live) setHasCalibrations((res.data?.calibrations || []).length > 0);
    }).catch(() => {});
    return () => { live = false; };
  }, [needsDecisionNow, result]);

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

      {/* Which file is open and which PC is in use for it: on every state of the page */}
      {pcStatusSeen && <PcFileHead file={activeDatasetName} pc={usedPc} perPixel={usedPcIsMap} origin={pcInUse} showConvention />}

      {/* A saved calibration is offered for the open file, and "Take PC from..." opens from here. Nothing
          changes without a button. The seams for what comes next (grantActions, renderAfterUse) are empty. */}
      <CalibrationPanel
        dataset={activeDatasetName}
        usedPc={pcStatusSeen ? usedPc : undefined}
        pcInUse={pcInUse}
        takePcOpen={takePcOpen}
        onTakePcOpenChange={setTakePcOpen}
        onPcChanged={onPcRestored}
        onUsePc={(pc) => onPcChanged(pc[0], pc[1], pc[2])}
        onUndo={putBackFile}
        onGoToIndexing={onNavigate ? () => onNavigate('indexing') : null}
        renderBands={(caption) => (
          <BandsPane
            patternBase64={patternBase64} pcx={pcx} pcy={pcy} segments={segments}
            caption={caption} exportName="pc-bands"
          />
        )}
        onFindPc={handleFindPc}
        canFindPc={canRun}
        pickFile={pickPcFile}
        grantActions={[{ key: 'save', label: t('pcrefinement:grant.saveAsCalibration'), onClick: () => openSave() }]}
        onApplyToOthers={(calibration) => openApplyFlow(calibration, activeDatasetName)}
        silenced={silencedIds}
        renderAfterUse={({ offer }) => (
          <div style={{ marginTop: 4 }}>
            <Button small onClick={() => openApplyFlow(offer, activeDatasetName)}>{t('pcrefinement:calibration.applyToOthers')}</Button>
          </div>
        )}
      />
      {pcInUse?.kind === 'calibration' && pcInUse.calibration?.id && (
        <OlderResults
          calibrationId={pcInUse.calibration.id}
          refreshKey={String(pcInUse.calibration.version)}
          onGoToIndexing={onNavigate ? () => onNavigate('indexing') : null}
        />
      )}

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
            disabled={cancelling}
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
            selectedPaths={loadedPaths(discoveredFiles, phases, addedPaths)}
            onTogglePath={handlePickPhase}
            onSetAll={handleSetAllPhases}
            onAddPath={addPhaseByPath}
            method="hough"
            open={true}
            onClose={() => setPhasePickerOpen(false)}
          />
        </FloatingPhasePanel>

        {/* Several phases: the list, each removable, and what it means. */}
        {phaseNames.length > 1 && (
          <div
            data-testid="pc-phase-list"
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
            <div style={{ color: colors.textSecondary, marginBottom: 2 }}>
              {t('pcrefinement:controls.phasesHeader', { count: phaseNames.length })}
            </div>
            {phases.map((p) => (
              <div key={p.name} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '1px 0' }}>
                <span style={{ color: colors.accent, flex: 1, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {p.name}
                </span>
                {p.space_group && (
                  <span style={{ color: colors.textSecondary }}>{shortSpaceGroup(p.space_group)}</span>
                )}
                <button
                  type="button"
                  disabled={phaseLoading}
                  title={t('pcrefinement:controls.removePhaseTooltip', { name: p.name })}
                  aria-label={t('pcrefinement:controls.removePhaseTooltip', { name: p.name })}
                  onClick={() => runPhaseJob(async () => {
                    reportPhaseOp(await removePhaseCore(p.name),
                                  t('pcrefinement:controls.phaseRemoved', { name: p.name }));
                  })}
                  style={{
                    background: 'none', border: 'none', color: colors.textSecondary,
                    cursor: 'pointer', fontSize: '11pt', lineHeight: 1, padding: '0 2px',
                  }}
                >
                  ×
                </button>
              </div>
            ))}
            <div style={{ color: colors.textSecondary, fontSize: '8pt', marginTop: 2 }}>
              {t('pcrefinement:controls.phasesMultiNote')}
            </div>
          </div>
        )}

        {phaseNames.length <= 1 && phaseInfo && (
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
        variant={resultHasMainButton ? 'default' : 'purple'}
        onClick={handleIndexAll}
        title={t('pcrefinement:controls.indexAllTooltip')}
      >
        {t('pcrefinement:controls.indexAll', { count: patterns.length })}
      </Button>
      <StatusMsg msg={indexMsg} isError={indexError} />
      {refusal && (
        <HoughRefusal
          problem={refusal}
          fit={(name) => pcApi.fitPhaseReflectors(name)}
          onFitted={() => { setRefusal(null); indexingChangedRef.current?.(); }}
        />
      )}

      {/* A file that carries no PC and none was set: the fields show a placeholder */}
      {pcSource === 'missing' && (
        <div data-testid="pc-missing-hint" role="alert" style={{ margin: '4px 2px 2px' }}>
          <div style={{ fontSize: '9.5pt', color: colors.red }}><b>{t('pcrefinement:pcMissingTitle')}</b></div>
          <div style={{ fontSize: '9pt', color: colors.textSecondary }}>{t('pcrefinement:pcMissingHint')}</div>
          <Button small style={{ marginTop: 4 }} onClick={() => setTakePcOpen(true)}>{t('pcrefinement:takePc.button')}</Button>
        </div>
      )}

      {/* Find PC: no starting value needed */}
      <div style={{ fontSize: '9pt', color: colors.textSecondary }}>{t('pcrefinement:find.conventionNote')}</div>
      {phaseNames.length > 0 && (
        <div data-testid="pc-searches-with" style={{ fontSize: '9pt', color: colors.textSecondary }}>
          {t(phaseNames.length > 1 ? 'pcrefinement:find.searchesWithMany' : 'pcrefinement:find.searchesWithOne',
            { names: phaseNames.join(', ') })}
        </div>
      )}
      <Button
        disabled={!canRun}
        style={{ width: '100%' }}
        variant="primary"
        onClick={handleFindPc}
        title={t('pcrefinement:find.buttonTooltip')}
      >
        {t('pcrefinement:find.button')}
      </Button>
      <div style={{ fontSize: '9pt', color: colors.textSecondary }}>{t('pcrefinement:find.note')}</div>
      {!phaseLoaded && (
        <div data-testid="pc-find-needs-phase" role="status" style={{ fontSize: '9pt', color: colors.orange }}>
          {t('pcrefinement:find.needPhase')}
        </div>
      )}
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        <Button
          small
          disabled={isRunning || picking}
          onClick={() => { givenUpRef.current = []; pickPatterns().catch(() => {}); }}
          title={t('pcrefinement:find.pickTooltip')}
        >
          {t('pcrefinement:find.pickButton')}
        </Button>
        <Button small onClick={() => setTakePcOpen(true)} title={t('pcrefinement:takePc.tooltip')}>
          {t('pcrefinement:takePc.button')}
        </Button>
      </div>
      {patterns.length > 0 && (
        <div data-testid="pc-patterns-used" style={{ fontSize: '9pt', color: colors.textSecondary }}>
          <b>{t('pcrefinement:find.patternsUsed')}</b>
          <ul style={{ margin: '4px 0', paddingLeft: 18 }}>
            {patterns.slice(0, 2).map((pt) => <li key={`${pt.row},${pt.col}`}>({pt.row}, {pt.col})</li>)}
            {patterns.length > 2 && <li>{t('pcrefinement:find.andMore', { count: patterns.length - 2 })}</li>}
          </ul>
        </div>
      )}
      <StatusMsg msg={msgScope === 'find' ? globalMsg : null} isError={globalError} />

      {/* Result: how the search ended, the PC with the facts to judge it, what the page does with it. After a
          step back the panel says what happened instead of the numbers. */}
      {result && undone && (
        <GroupBox title={t('pcrefinement:result.title')} style={{ animation: 'fadeSlideIn 0.2s ease-out' }}>
          <PcUndoDone file={undone.file} previous={undone.previous} notRestored={undone.notRestored} />
        </GroupBox>
      )}
      {result && !undone && (
        <GroupBox title={t('pcrefinement:result.title')} style={{ animation: 'fadeSlideIn 0.2s ease-out' }}>
          <FollowerQuestion
            key={asking || 'none'}
            result={result}
            savedNew={asked?.key === asking && asked?.state === 'saveNew'}
            onAnswered={(state) => setAsked({ key: asking, state })}
            onOnlyThis={(calibration) => setLeftCalibration({ mark: loadMark, dataset: result.dataset, id: calibration.id })}
            onSaveNew={() => openSave({ dataset: result.dataset, pc: resultPc, quality: result.quality, forQuestion: true })}
            onApplyToOthers={(made) => openApplyFlow(made, result.dataset)}
            onPcChanged={onPcRestored}
            onUndo={putBackFile}
          />
          {resultPc && (
            <PcResult
              result={result}
              file={result.dataset}
              pc={resultPc}
              inUse={!!result.dataset && resultInUse}
              inConvention={inConvention}
              patternsFrom={resultFrom}
              mapAction={pcMapActionOf(result.pc_map_action)}
              busy={isRunning || picking}
              bandsShown={bandsShown}
              onToggleBands={() => setBandsShown((v) => !v)}
              renderBands={(caption) => (
                <BandsPane
                  patternBase64={patternBase64} pcx={pcx} pcy={pcy} segments={segments}
                  caption={caption} exportName="pc-bands"
                />
              )}
              onUndo={handleUndo}
              onFindAgain={handleFindAgain}
              onKeep={() => setKeptFor(result.dataset)}
              keptFor={keptFor}
              onGoToIndexing={onNavigate ? () => onNavigate('indexing') : null}
              warningText={pcWarningText(result, t)}
              warningItems={pcWarningItems(result, t)}
              onSave={questionOpen ? null : () => openSave({ dataset: result.dataset, pc: resultPc, quality: result.quality })}
              onCompare={hasCalibrations ? openTakePc : null}
            />
          )}
          {/* The PC in use for the file the search was run for is the line every way of giving a PC has */}
          {result.dataset && resultPc && !resultInUse && result.pc_stored !== false && (
            <div data-testid="pc-result-not-in-use" style={{ fontSize: '9pt', color: colors.textSecondary, margin: '4px 0' }}>
              {t('pcrefinement:find.notInUse')}
            </div>
          )}
          {/* Several phases: how the calibration patterns split between them. */}
          {Array.isArray(result.pattern_phases) && result.pattern_phases.length > 0 && (
            <FormRow label={t('pcrefinement:controls.phasePerPattern')}>
              <Label style={{ color: colors.accent, fontSize: '9pt' }}>
                <span data-testid="pc-phase-summary">
                  {summarisePatternPhases(result.pattern_phases, t('pcrefinement:controls.phaseNone'))
                    .map((s) => `${s.name} ×${s.count}`).join(', ')}
                </span>
              </Label>
            </FormRow>
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
            disabled
            title={t('pcrefinement:controls.gridCalibrationTooltip')}
            style={{ width: '100%' }}
          >
            {t('pcrefinement:controls.gridCalibration')}
          </Button>
        </GroupBox>
      )}

      {/* Global PC Refine */}
      <Button
        disabled={!canRun || patterns.length === 0}
        style={{ width: '100%' }}
        variant={resultHasMainButton ? 'default' : 'primary'}
        onClick={handleGlobalRefine}
        title={t('pcrefinement:controls.globalRefineTooltip')}
      >
        {t('pcrefinement:controls.globalRefine')}
      </Button>
      <StatusMsg msg={msgScope === 'find' ? null : globalMsg} isError={globalError} />

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


      <Separator />

      {/* Pixel-wise PC Correction */}
      <PixelwisePCGroup canRun={canRun} onCalibrated={onGridCalibrated} />

      {/* Indexing Settings */}
      <IndexingSettingsGroup
        onParamsChange={handleParamsChange}
        phases={phases}
        datasetKey={activeDatasetName || ''}
        // A change of the Hough indexer (reflector families, nBands) makes the
        // bands of the shown pattern stale: index it again with the new one.
        onIndexingChanged={handleIndexingChanged}
      />

      {/* A per-pixel PC map: what the fields show, and what the last edit did to it */}
      {(() => {
        const note = pcMapNote(pcMapInfo, pcMapAction);
        return note ? (
          <div data-testid="pc-map-note" role="status"
               style={{ fontSize: '9pt', color: colors.textSecondary, margin: '4px 2px 6px' }}>
            {t(`pcrefinement:pixelwise.${note.key}`, note.params)}
          </div>
        ) : null;
      })()}

      {/* A typed PC the calibration store did not take: indexing would use the file's, not this one */}
      {typedPcNotStored && (
        <div data-testid="pc-not-stored" role="status"
             style={{ fontSize: '9pt', color: colors.orange, margin: '4px 2px 6px' }}>
          {t('pcrefinement:warnings.pcNotStored')}
        </div>
      )}

      {/* Detector Settings */}
      <DetectorSettingsGroup
        onDetectorApplied={onDetectorApplied}
        onPcChanged={onPcChanged}
        onTiltChanged={onTiltChanged}
        externalPc={externalPc}
        onPixelSizeApplied={onPixelSizeApplied}
        onPcMapAction={onPcMapAction}
        datasetName={activeDatasetName}
        usedPc={pcStatusSeen ? usedPc : undefined}
        pcInUse={pcInUse}
        onUndoPc={putBackFile}
        onGoToIndexing={onNavigate ? () => onNavigate('indexing') : null}
        renderBands={(caption) => (
          <BandsPane
            patternBase64={patternBase64} pcx={pcx} pcy={pcy} segments={segments}
            caption={caption} exportName="pc-bands"
          />
        )}
        onSavePc={(pc) => openSave({ pc })}
      />
      {saveFor && (
        <SaveCalibrationDialog
          dataset={saveFor.dataset}
          pc={saveFor.pc}
          quality={saveFor.quality}
          perPixel={!!pcInUse?.per_pixel}
          onClose={() => setSaveFor(null)}
          onSaved={(created) => {
            if (saveFor.fromFar) setSavedFar({ dataset: saveFor.dataset, name: created.name });
            if (saveFor.forQuestion) setAsked({ key: asking, state: 'saveNew' });
          }}
          onApplyToOthers={saveFor.fromFar ? null : (created) => { setSaveFor(null); openApplyFlow(created, saveFor.dataset); }}
        />
      )}
      {applyFlow && (
        <ApplyToFilesDialog
          onClose={() => useApplyFlow.getState().close()}
          onPcChanged={onPcRestored}
          onUndo={putBackFile}
          onSave={openSave}
          onFindPc={handleFindPc}
          canFindPc={canRun}
          pageReady={pcStatusSeen && activeDatasetName != null}
          findStatus={{ running: isRunning || picking, message: msgScope === 'find' ? globalMsg : null, error: globalError }}
          findResult={result}
          savedFar={savedFar}
        />
      )}
      <PromptDialog {...promptProps} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main PCRefinement component
// ---------------------------------------------------------------------------
/** A value no translation can equal, so a missing key is detectable.
 *  i18next here has returnEmptyString:false, which makes '' unusable. */
// The ESCAPE, not a raw NUL byte: the byte makes git treat this file as
// binary, so it gets no readable diff at all. Same value either way.
const I18N_MISS = '\u0000miss';

/**
 * The reliability warning after a PC refine, in the reader's language.
 *
 * The backend sends `pc_warning_codes` (a stable code plus its numbers) and
 * its English sentence in `pc_warning`. Translate on the codes; fall back to
 * the sentence, so an older backend or a code this build has no text for
 * still says something true.
 */
export function pcWarningText(result, t) {
  const codes = result?.pc_warning_codes;
  if (Array.isArray(codes) && codes.length) {
    const parts = codes
      .map((c) => t(`warnings.${c.code}`, { ...localiseParams(c.params), defaultValue: I18N_MISS }))
      .filter((s) => s && s !== I18N_MISS);
    if (parts.length === codes.length) return parts.join('  ');
  }
  return result?.pc_warning || '';
}

/**
 * Every warning of a result worded on its own: `[{code, text}]`, in the order the backend sent them. A code
 * this build has no text for is left out (its English sentence is only available joined with the others).
 */
export function pcWarningItems(result, t) {
  const codes = result?.pc_warning_codes;
  if (!Array.isArray(codes)) return [];
  return codes
    .map((c) => ({ code: c.code, text: t(`warnings.${c.code}`, { ...localiseParams(c.params), defaultValue: I18N_MISS }) }))
    .filter((w) => w.text && w.text !== I18N_MISS);
}

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
  // syncPatterns is created once; it reads whether a phase WAS loaded through this ref.
  const phaseLoadedRef = useRef(phaseLoaded);
  phaseLoadedRef.current = phaseLoaded;
  const [sessionReset, setSessionReset] = useState(0);
  // syncPatterns is created once: the current translator, and a counter that
  // voids status answers asked before a newer change (see syncPatterns).
  const tRef = useRef(t);
  tRef.current = t;
  const syncSeqRef = useRef(0);
  // The loaded phases ({name, space_group?, lattice?}); several when calibrating
  // a multi-phase sample. `phaseLabel` is their joined name for the header.
  const [phases, setPhases] = useState([]);
  const phaseNames = phases.map((p) => p.name);
  const multiPhase = phaseNames.length > 1;

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

  // Which pattern is selected, and which late answers may still be drawn on it
  // (see selectionGuard.js). Updated synchronously by the handlers, not by the
  // render, so an answer that arrives between a click and its re-render is
  // already judged against the new selection.
  const selectionRef = useRef(null);
  if (selectionRef.current === null) selectionRef.current = createSelectionGuard();
  // Re-index requests (PC / tilt edits) of the selected pattern: only the newest draws.
  const reindexRef = useRef(null);
  if (reindexRef.current === null) reindexRef.current = createLatestRequest();
  // The latest list / phase count / list-update handler, for callbacks that must
  // keep a stable identity (showPattern is called from effects and event listeners).
  const patternsRef = useRef([]);
  patternsRef.current = patterns;
  const multiPhaseRef = useRef(false);
  multiPhaseRef.current = multiPhase;
  const handlePatternsIndexedRef = useRef(null);
  const pcEditSeqRef = useRef(0);      // the newest "Use these numbers" (see handlePcChanged)

  // --- Forward-simulation preview (closes the PC/sample_tilt/det_tilt
  // degeneracy that Hough band-fitting alone can't break) ---
  const [trialGeometry, setTrialGeometry] = useState({
    pcx: 0.5, pcy: 0.5, pcz: 0.5,
    sampleTilt: 70.0, detTilt: 0.0, azimuthal: 0.0,
    binning: 1, pixelSize: null,
  });
  const [shtPath, setShtPath] = useState(null);
  const [shtChoices, setShtChoices] = useState([]);
  const [shtRefresh, setShtRefresh] = useState(0);   // bumped on every visit: re-list the masters
  const shtPhaseRef = useRef(null);                  // the phase the master list was last picked for
  const [simulatedB64, setSimulatedB64] = useState(null);
  const [simulatedNcc, setSimulatedNcc] = useState(null);
  const [simulatedNccReason, setSimulatedNccReason] = useState(null);
  const [simulatedLoading, setSimulatedLoading] = useState(false);
  const [simulatedError, setSimulatedError] = useState(null);
  const [simulatedOrientation, setSimulatedOrientation] = useState(null);
  const [simulatedOrientationSource, setSimulatedOrientationSource] = useState(null);
  const [simulatedIndexedPhase, setSimulatedIndexedPhase] = useState(null);
  const [previewBandwidth, setPreviewBandwidth] = useState(128);

  // Optimized PC values (passed to DetectorSettingsGroup to update spinboxes)
  const [optimizedPc, setOptimizedPc] = useState(null);

  // A per-pixel PC map (grid calibration): {rows, cols}, or null. The fields show
  // its mean. And what the last write of one PC did to it ("shifted" | "replaced").
  const [pcMapInfo, setPcMapInfo] = useState(null);
  // Where the active dataset's PC comes from ("missing": the file carries none and
  // nobody set one, the fields show a placeholder).
  const [pcSource, setPcSource] = useState(null);
  // The PC in use for the open file as the backend reports it (null for a file without one), and whether
  // the backend has answered yet: the head of the page says nothing before that rather than a wrong "none".
  const [usedPc, setUsedPc] = useState(null);
  const [pcStatusSeen, setPcStatusSeen] = useState(false);
  // The PC in use is a per-pixel map (usedPc is then its mean): the head of the page must not read as one PC.
  const [usedPcIsMap, setUsedPcIsMap] = useState(false);
  // `pc_in_use` of the status: where that PC came from (file, Find PC, refine, typed, grid, a calibration) and
  // the convention of the open file's vendor. The head of the page names the origin on every state.
  const [pcInUse, setPcInUse] = useState(null);
  // A typed PC the calibration store did not take (the backend answered pc_stored false): said on the page.
  const [typedPcNotStored, setTypedPcNotStored] = useState(false);
  const noteStatusPc = useCallback((d) => {
    setPcStatusSeen(true);
    const has = d?.pc_source !== 'missing' && Array.isArray(d?.pc) && d.pc.length === 3;
    setUsedPc(has ? d.pc.map(Number) : null);
    // the head says what the FILE uses (the store's PC), not what the page's own detector holds
    setUsedPcIsMap(has && !!d?.pc_in_use?.per_pixel);
    setPcInUse(d?.pc_in_use ?? null);
  }, []);
  const refreshPcSource = useCallback(() => {
    pcApi.status().then((res) => {
      setPcSource(res.data?.pc_source ?? null);
      noteStatusPc(res.data);
    }).catch(() => {});
  }, [noteStatusPc]);
  const [pcMapAction, setPcMapAction] = useState(null);
  const handlePcMapAction = useCallback((value) => {
    const action = pcMapActionOf(value);
    if (action === 'replaced') setPcMapInfo(null);   // one PC for the whole scan now
    if (action) setPcMapAction(action);
  }, []);
  const refreshPcMap = useCallback(() => {
    pcApi.status().then((res) => {
      setPcMapInfo(pcMapInfoOf(res.data));
      setPcSource(res.data?.pc_source ?? null);
      noteStatusPc(res.data);
      setPcMapAction(null);
      const pc = res.data?.pc;
      if (Array.isArray(pc) && pc.length === 3) {
        setOptimizedPc(pc);   // the fields: the map mean
        // ...and the forward preview, whose NCC is the page's own check of that PC.
        setTrialGeometry((g) => ({ ...g, pcx: pc[0], pcy: pc[1], pcz: pc[2] }));
      }
    }).catch(() => {});
  }, [noteStatusPc]);

  // The status bar at the bottom shows the PC of the data store's detector, which only a load fills. Every way of
  // giving a file a PC, or of putting one back, ends in a new `usedPc` here; when that is not the PC the bar shows,
  // the bar is given the file's metadata again (the backend answers it from the same store as the status).
  const usedPcKey = usedPc ? usedPc.join(',') : '';
  useEffect(() => {
    if (!pcStatusSeen) return undefined;
    const store = useDataStore.getState();
    if (!store.isFileOpen) return undefined;
    const shown = store.detector;
    if (usedPc ? (shown?.has_detector && samePc(shown.pc, usedPc)) : !(shown?.has_detector && shown.pc?.length === 3)) return undefined;
    let live = true;
    const forFile = store.filePath;
    ebsdApi.getMetadata().then((res) => {
      if (live && res?.data?.detector && useDataStore.getState().filePath === forFile) useDataStore.getState().setMetadata(res.data);
    }).catch(() => {});
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [usedPcKey, pcStatusSeen]);

  // Parent dataset info for PC propagation
  const [parentName, setParentName] = useState(null);
  const [activeDatasetName, setActiveDatasetName] = useState(null);
  const [propagated, setPropagated] = useState(false);
  // "This PC is on this page only" was said about the file typed for: another file does not carry it.
  useEffect(() => { setTypedPcNotStored(false); }, [activeDatasetName]);
  // Which file is open now, for the answer of a "Use these numbers" that was pressed on another one.
  const openFilePath = useDataStore((s) => s.filePath);
  const openLoadedName = useLoadedFilesStore((s) => s.files.find((f) => f.active)?.name || '');
  const openFileRef = useRef('');
  openFileRef.current = `${openFilePath || ''}|${openLoadedName}|${activeDatasetName || ''}`;

  // Indexing params (forwarded from IndexingSettingsGroup)
  const indexingParamsRef = useRef({});

  // Check backend status on mount and sync patterns + PC values
  useEffect(() => {
    // Void if anything newer happened before it answers (a sync, a Remove, a
    // phase change): those bump syncSeqRef. This one does not bump it, so it
    // never voids a sync of its own.
    const asked = syncSeqRef.current;
    pcApi.status().then((res) => {
      if (asked !== syncSeqRef.current) return;
      const d = res.data;
      if (d?.has_detector || d?.detector_set) setDetectorReady(true);
      if (d?.has_phase || d?.phase_loaded) {
        setPhaseLoaded(true);
        applyStatusPhases(d);
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
      setPcMapInfo(pcMapInfoOf(d));
      setPcSource(d?.pc_source ?? null);
      noteStatusPc(d);
    }).catch(() => {});
  }, [noteStatusPc]);

  // Which dataset is active and whether it has a parent ("Apply PC to parent").
  // Asked again on every visit: processing in the viewer can make a new active
  // dataset while this page stays mounted, and "apply to parent" must not
  // target the dataset that was active when the page first opened.
  const refreshDatasetInfo = useCallback(() => {
    ebsdApi.info().then((res) => {
      const dsName = res.data?.active_dataset || null;
      setActiveDatasetName(dsName);
      if (!dsName) { setParentName(null); return; }
      calibrationApi.getEntry(dsName).then((calRes) => {
        setParentName(calRes.data?.parent_name || null);
      }).catch(() => setParentName(null)); // No calibration entry — not a derived dataset
    }).catch(() => {});
  }, []);
  useEffect(() => { refreshDatasetInfo(); }, [refreshDatasetInfo]);

  // Show calibration pattern `idx` of `list`: its stored image and its cached
  // (or freshly computed) overlay. The ONE place a selection starts, so every
  // way of selecting - the list, the slider, Index All, a re-sync of the page -
  // clears the lines of the pattern shown before at once and ignores whatever
  // that pattern's requests answer later.
  const showPattern = useCallback((idx, list) => {
    const guard = selectionRef.current;
    const token = guard.select(idx);
    reindexRef.current.next();           // a re-index of the previous pattern is void too
    setSelectedPatternIdx(idx);
    setKikuchiSegments([]);              // those lines belong to the previous pattern
    const pat = idx == null ? null : (list ?? patternsRef.current)[idx];
    if (!pat) {
      // Nothing selected (a removal, a deselect, an empty list): the picture and
      // the CI of the pattern shown before must go with its lines.
      setPatternBase64(null);
      setCiValue(null);
      return;
    }
    // Load the STORED pattern image (captured at add-time, not current active signal)
    // This is critical for DeepCopy/FrameAvg workflows where the active signal may differ
    pcApi.getPatternImage(idx).then((res) => {
      if (!guard.isCurrent(token)) return;
      setPatternBase64(res.data?.image || null);
    }).catch(() => {
      if (!guard.isCurrent(token)) return;
      // Fallback: load from active signal if stored image endpoint fails
      ebsdApi.getPattern(pat.row ?? 0, pat.col ?? 0).then((res2) => {
        if (!guard.isCurrent(token)) return;
        setPatternBase64(res2.data?.image || res2.data?.image_base64 || null);
      }).catch(() => {});
    });
    // Load cached overlay (auto-indexes if not cached)
    pcApi.getPatternResult(idx).then((res) => {
      if (!guard.isCurrent(token)) return;
      const d = res.data;
      if (d?.success && d.segments) {
        setKikuchiSegments(d.segments);
        if (d.ci != null) setCiValue(d.ci);
        // Several phases: remember which one this pattern was indexed as
        // (the list shows it, and the preview picks its master by it).
        if (multiPhaseRef.current && d.phase_name) {
          handlePatternsIndexedRef.current?.([{ index: pat.index, phase_name: d.phase_name, phase_fits: d.phase_fits }]);
        }
      } else {
        setKikuchiSegments([]);
        // Fall back to local CI when backend cache is lost
        if (pat.ci != null) setCiValue(pat.ci);
      }
    }).catch(() => {
      if (!guard.isCurrent(token)) return;
      setKikuchiSegments([]);
      // Fall back to local CI when backend cache is lost
      if (pat.ci != null) setCiValue(pat.ci);
    });
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  // Sync with the backend (on every visit and after the viewer adds a pattern).
  // The backend's answer is the truth, also when it is EMPTY: the PC session
  // belongs to the active dataset, and processing in the viewer (deep copy on)
  // makes a new dataset with a session of its own - no patterns, no phase. The
  // page used to keep the old list and "phase loaded", and every action then
  // failed against the empty session, which read as a hang.
  const syncPatterns = useCallback(() => {
    refreshDatasetInfo();
    // An answer to an older question never overwrites a newer state: a status
    // asked before a Remove or a phase change (both bump this) is dropped.
    const asked = ++syncSeqRef.current;
    pcApi.status().then((res) => {
      if (asked !== syncSeqRef.current) return;
      const tt = tRef.current;   // syncPatterns is created once; the language may have changed
      const d = res.data || {};
      const list = Array.isArray(d.patterns) ? d.patterns : [];
      setPatterns(list); // patterns include CI from the backend cache
      // Auto-select the last pattern: its stored image and ITS lines (not those
      // of whatever was shown when the page was left); none when the list is empty.
      showPattern(list.length > 0 ? list.length - 1 : null, list);
      setDetectorReady(!!(d.has_detector || d.detector_set));
      const hasPhase = !!(d.has_phase || d.phase_loaded);
      setPhaseLoaded(hasPhase);
      if (hasPhase) applyStatusPhases(d);
      else {
        setPhases([]);
        setPhaseLabel(tt('pcrefinement:header.noPhaseLoaded'));
        setPhaseInfoText('');           // the header card of the old session's phase
        if (phaseLoadedRef.current) setSessionReset((n) => n + 1);   // it had one before
      }
      setGlobalCi(d?.global_ci ?? null);
      if (d?.pc?.length === 3) {
        // The PC of THIS session everywhere it is shown: the crosshair, the PC
        // fields and the forward preview (a processed copy may carry another PC;
        // the fields and the preview used to keep the old one, and the preview's
        // NCC then judged a good PC as bad).
        setCurrentPcx(d.pc[0]);
        setCurrentPcy(d.pc[1]);
        setOptimizedPc(d.pc);
        setTrialGeometry((g) => ({ ...g, pcx: d.pc[0], pcy: d.pc[1], pcz: d.pc[2] }));
      }
      setPcMapInfo(pcMapInfoOf(d));
      setPcSource(d?.pc_source ?? null);
      noteStatusPc(d);
    }).catch(() => {});
  }, [noteStatusPc]);

  // The PC came back from the backend after a step back: what the last write did to a map is history now
  // (the restored map is in use again), and a typed PC that was not stored is gone with the step.
  const handlePcRestored = useCallback(() => {
    setPcMapAction(null);
    setTypedPcNotStored(false);
    syncPatterns();
  }, [syncPatterns]);

  // Re-sync when PC Refinement page becomes visible OR when patterns are updated
  const containerRef = useRef(null);
  useEffect(() => {
    const onPageChange = (e) => {
      if (e.detail?.page !== 'pcrefinement') return;
      syncPatterns();
      setShtRefresh((n) => n + 1);
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
    syncSeqRef.current += 1;   // a status asked before this removal must not bring the pattern back
    try {
      await pcApi.removePattern(selectedPatternIdx);
    } catch (err) {
      // The backend kept the pattern (e.g. a refine is running, or the list
      // changed under us): show its list, never a local one that differs from
      // it - the indices of every later action would point at other patterns.
      // And say why, instead of a Remove that silently did nothing.
      const detail = err?.response?.data?.detail;
      toast.warning(isPcJobRunning(err) ? pcJobRunningText(t)
        : (typeof detail === 'string' ? detail : detail?.message || err?.message || String(err)));
      syncPatterns();
      return;
    }
    setPatterns((prev) => {
      const next = prev.filter((_, i) => i !== selectedPatternIdx);
      return next.map((p, i) => ({ ...p, index: i }));
    });
    setGlobalCi(null);       // it was the mean over a set that included this pattern
    showPattern(null);       // nothing selected: its picture, lines and CI go, its pending answers are void
  };

  // The names a status reply carries: with several phases the header shows them
  // all; with one it is the single `phase_name`, exactly as before.
  function applyStatusPhases(d) {
    const names = Array.isArray(d?.phase_names) ? d.phase_names : [];
    setPhases(names.map((name) => ({ name })));
    if (names.length > 1) setPhaseLabel(phaseLabelFor(names, ''));
    else if (d?.phase_name) setPhaseLabel(d.phase_name);
  }

  // Phases added or removed in the picker: `list` is the backend's loaded set.
  // Everything computed for the old set (CI, per-pattern phase, overlay) is
  // stale — the backend dropped its caches — so clear it here too.
  const handlePhasesChanged = (list) => {
    syncSeqRef.current += 1;   // a status asked before this change must not undo it
    setPhases(list);
    setPhaseLoaded(list.length > 0);
    setPhaseLabel(phaseLabelFor(list.map((p) => p.name), t('pcrefinement:header.noPhaseLoaded')));
    setPhaseInfoText(list.map((p) => [
      t('pcrefinement:phaseInfo.nameLine', { name: p.name }),
      p.space_group ? t('pcrefinement:phaseInfo.spaceGroupLine', { spaceGroup: p.space_group }) : '',
      p.lattice?.a != null ? `a=${p.lattice.a.toFixed(3)} b=${p.lattice.b.toFixed(3)} c=${p.lattice.c.toFixed(3)}` : '',
    ].filter(Boolean).join(' · ')).join('\n'));
    setPatterns((prev) => prev.map((p) => ({ ...p, ci: null, phase: null, phaseFits: null })));
    setKikuchiSegments([]);
    setCiValue(null);
    setGlobalCi(null);
  };

  const handlePhaseLoaded = (info) => {
    syncSeqRef.current += 1;   // a status asked before this load must not mark the phase gone
    setPhaseLoaded(true);
    const name = info.phase_name || info.name || t('pcrefinement:phaseInfo.unknown');
    setPhaseLabel(name);
    setPhases([{ name, space_group: info.space_group, lattice: info.lattice, path: info.path }]);
    setPatterns((prev) => prev.map((p) => ({ ...p, phase: null, phaseFits: null })));
    const lines = [
      t('pcrefinement:phaseInfo.nameLine', { name }),
      info.space_group ? t('pcrefinement:phaseInfo.spaceGroupLine', { spaceGroup: info.space_group }) : '',
      info.lattice ? `a=${info.lattice.a?.toFixed(3)} b=${info.lattice.b?.toFixed(3)} c=${info.lattice.c?.toFixed(3)}` : '',
    ].filter(Boolean).join('\n');
    setPhaseInfoText(lines);
  };

  const handleDetectorApplied = (pc, sampleTilt, detTilt) => {
    setDetectorReady(true);
    refreshPcSource();
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
        if (r) {
          return {
            ...p,
            ci: 'ci' in r ? r.ci : p.ci,
            phase: r.phase_name || r.phase || p.phase,
            // What every phase made of this pattern (why the winner won);
            // a result that carries none leaves the earlier one alone.
            phaseFits: 'phase_fits' in r ? r.phase_fits : p.phaseFits,
          };
        }
        return p;
      })
    );
  }, []);

  handlePatternsIndexedRef.current = handlePatternsIndexed;

  // "Use these numbers": the typed PC (Bruker's numbers) becomes the PC of the open file. Updates the detector,
  // clears caches, re-indexes the current pattern, re-overlays. Typing alone never gets here.
  // Answers `{data}` when the backend took it and `{error: text}` when not, so the fields can say it; the
  // crosshair and the forward preview follow the PC only then, never a typed number that was refused.
  const handlePcChanged = useCallback(async (pcx, pcy, pcz) => {
    // The answer describes the pattern that was selected NOW, at THIS request:
    // another pick (or removal), or a newer edit, before it returns voids it.
    const forIdx = selectedPatternIdx;
    const seq = reindexRef.current.next();
    // Two presses of "Use these numbers" can overlap: the answer of the older one must not put its PC on the page
    // after the newer one has (the status below then corrects whatever the backend ended with).
    const mine = ++pcEditSeqRef.current;
    // The press is for the file that was open when it was made: if another file is open when the answer comes,
    // the old file's PC does not go into the crosshair, the preview or the head (the backend keeps it for its
    // own dataset; the next status shows the PC of the file in front of the person).
    const forFile = openFileRef.current;
    const newest = () => mine === pcEditSeqRef.current && forFile === openFileRef.current;
    try {
      const res = await pcApi.updatePC(pcx, pcy, pcz, forIdx);
      const d = res.data;
      if (newest()) {
        setCurrentPcx(pcx);
        setCurrentPcy(pcy);
        // The forward-sim preview renders at the PC now in use (its NCC is the check of that PC).
        setTrialGeometry((g) => ({ ...g, pcx, pcy, pcz }));
      }
      handlePcMapAction(d?.pc_map_action);
      if (newest()) setTypedPcNotStored(d?.pc_stored === false);   // on this page only: say so, never silently
      if (newest() && d?.pc_stored !== false) {
        // until the status below confirms it: the head names this PC and where it came from
        setPcStatusSeen(true);
        setUsedPc([pcx, pcy, pcz]);
        setPcInUse((prev) => ({ ...(prev || {}), kind: 'typed', calibration: null }));
      }
      refreshPcSource();                       // a typed PC is a PC: the "no PC" hint goes
      // Only the selected pattern was indexed at the new PC: no global value
      // until all of them are (Index All, Global PC Refine).
      setGlobalCi(d?.global_ci ?? null);
      // (picking another pattern also starts a new "request": showPattern)
      const stillShown = reindexRef.current.isLatest(seq) && forFile === openFileRef.current;
      if (stillShown) {
        if (d?.ci != null) setCiValue(d.ci);
        if (d?.segments) setKikuchiSegments(d.segments);
      }
      return { data: d || {} };
    } catch (err) {
      // The backend did not take the PC (a refine is running, the file changed, ...): nothing on the page moved,
      // and the person is told, where the fields used to stay silent about everything but a running refine.
      if (isPcJobRunning(err)) {
        const text = t('pcrefinement:controls.pcEditNotApplied', { reason: pcJobRunningText(t) });
        toast.warning(text);
        refreshPcMap();
        return { error: text };
      }
      return { error: problemText(err, t, 'pcrefinement') || err?.message || t('pcrefinement:detector.failed') };
    }
  }, [selectedPatternIdx, setCiValue, setGlobalCi, handlePcMapAction, refreshPcSource, refreshPcMap, t]);

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
    const forIdx = selectedPatternIdx;
    const seq = reindexRef.current.next();
    try {
      const res = await pcApi.updateTilt(sampleTilt, detTilt, azimuthal, forIdx);
      const d = res.data;
      setGlobalCi(d?.global_ci ?? null);  // other patterns were not re-indexed
      // (picking another pattern also starts a new "request": showPattern)
      const stillShown = reindexRef.current.isLatest(seq);
      if (stillShown) {
        if (d?.ci != null) setCiValue(d.ci);
        if (d?.segments) setKikuchiSegments(d.segments);
      }
    } catch (err) {
      // Refused while a refine runs: say so rather than look applied. The tilt
      // fields keep their own state, so the text tells the user to enter it again.
      if (isPcJobRunning(err)) {
        toast.warning(t('pcrefinement:controls.tiltEditNotApplied', { reason: pcJobRunningText(t) }));
      }
      // Other errors during a reactive update stay quiet, as before.
    }
  }, [selectedPatternIdx, setCiValue, setGlobalCi, handleTrialTiltChanged, t]);

  // Which phase's master the preview uses: the loaded phase, or with several
  // phases the one the selected pattern was indexed as (see phaseSet.js).
  const previewPhase = previewPhaseName({
    phaseNames, phaseLabel, patterns, selectedIdx: selectedPatternIdx,
  });
  const previewPhaseRef = useRef(previewPhase);
  previewPhaseRef.current = previewPhase;

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
    // A later phase change must not be overwritten by an earlier, slower reply.
    let stale = false;
    // Asked again on every visit too (shtRefresh): a master simulated meanwhile
    // on another page must appear, since a phase without one gets no stand-in.
    // A visit keeps a master the user chose while it is still listed; a phase
    // change picks the new phase's own.
    // Noted only when the answer is applied: a run voided by a newer one must not
    // mark the phase as handled, or the next run keeps the old phase's master.
    const phaseChanged = shtPhaseRef.current !== phaseLabel;
    fetch('/api/indexing/files/spherical')
      .then(r => r.json())
      .then(d => {
        if (stale) return;
        shtPhaseRef.current = phaseLabel;
        const files = Array.isArray(d?.files) ? d.files : [];
        setShtChoices(files);
        // The master of the LOADED phase: the first file was Al while Ni was
        // being calibrated, and the preview reported a good PC as NCC 0.03.
        setShtPath((current) => (!phaseChanged && current && files.some((f) => f.path === current)
          ? current : pickShtForPhase(files, previewPhaseRef.current)));
      })
      .catch(() => { if (!stale) setShtChoices([]); });
    return () => { stale = true; };
  }, [phaseLoaded, phaseLabel, shtRefresh]);

  // Several phases: the master must be the one of the phase the SELECTED
  // pattern was indexed as, so follow the selection. (With one phase the effect
  // above is the only one that sets it, exactly as before.)
  useEffect(() => {
    if (!multiPhase || shtChoices.length === 0) return;
    setShtPath(pickShtForPhase(shtChoices, previewPhaseRef.current));
  }, [multiPhase, shtChoices, previewPhase]);

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
      setSimulatedNccReason(null);
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
          setSimulatedNccReason(r.data?.ncc_reason || null);
          setSimulatedOrientation(r.data?.orientation_euler_deg || null);
          setSimulatedOrientationSource(r.data?.orientation_source || null);
          setSimulatedIndexedPhase(r.data?.indexed_phase || null);
        })
        .catch(err => {
          if (cancelled) return;
          setSimulatedB64(null);
          setSimulatedNcc(null);
          setSimulatedNccReason(null);
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
  const handleSelectPatternIdx = (idx) => showPattern(idx, patterns);

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
          data-testid="pc-ci"
          title={t('pcrefinement:header.ciTooltip')}
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
          data-testid="pc-global-ci"
          title={t('pcrefinement:header.globalCiTooltip')}
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
              simulatedNccReason={simulatedNccReason}
              simulatedLoading={simulatedLoading}
              simulatedError={simulatedError}
              simulatedOrientation={simulatedOrientation}
              simulatedOrientationSource={simulatedOrientationSource}
              multiPhase={multiPhase}
              simulatedIndexedPhase={simulatedIndexedPhase}
              shtChoices={shtChoices}
              shtPath={shtPath}
              previewPhase={previewPhase}
              onShtPathChange={setShtPath}
              previewBandwidth={previewBandwidth}
              onPreviewBandwidthChange={setPreviewBandwidth}
            />
          }
          right={
            <PhaseLoadedContext.Provider value={phaseLoaded}>
            <ControlsPanel
              phaseLoaded={phaseLoaded}
              sessionReset={sessionReset}
              phases={phases}
              onPhasesChanged={handlePhasesChanged}
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
              onSegmentsUpdate={(segs, forIdx) => {
                // Lines for a pattern that is no longer the selected one are not drawn.
                if (forIdx == null || selectionRef.current.isSelected(forIdx)) setKikuchiSegments(segs);
              }}
              getSelectedIdx={() => selectionRef.current.selected()}
              onPatternsIndexed={handlePatternsIndexed}
              onSelectPattern={handleSelectPatternIdx}
              onPcChanged={handlePcChanged}
              onTiltChanged={handleTiltChanged}
              pcMapInfo={pcMapInfo}
              pcSource={pcSource}
              pcMapAction={pcMapAction}
              onPcMapAction={handlePcMapAction}
              onGridCalibrated={refreshPcMap}
              onPixelSizeApplied={({ binning, pixelSize }) =>
                setTrialGeometry((g) => ({ ...g, binning, pixelSize }))}
              externalPc={optimizedPc}
              parentName={parentName}
              activeDatasetName={activeDatasetName}
              propagated={propagated}
              setPropagated={setPropagated}
              usedPc={usedPc}
              pcStatusSeen={pcStatusSeen}
              onPcRestored={handlePcRestored}
              usedPcIsMap={usedPcIsMap}
              pcInUse={pcInUse}
              onNavigate={onNavigate}
              typedPcNotStored={typedPcNotStored}
              patternBase64={patternBase64}
              segments={kikuchiSegments}
              pcx={currentPcx}
              pcy={currentPcy}
              onOptimizedPc={(pc, mode) => {
                setOptimizedPc(pc);
                if (pc.length === 3) {   // until the status below confirms it
                  setUsedPc(pc.map(Number));
                  setUsedPcIsMap(false);   // a run gives ONE PC
                  // ...and its origin is the run's own, not the file's
                  setPcInUse((prev) => ({ ...(prev || {}), kind: mode === 'find' ? 'find' : 'refine', calibration: null }));
                  setTypedPcNotStored(false);   // the run's own answer says what the store took
                }
                refreshPcSource();
                if (pc.length === 3) {
                  setCurrentPcx(pc[0]);
                  setCurrentPcy(pc[1]);
                  // The forward preview renders at the refined PC too: its NCC is
                  // the check of that PC, not of the one before the refine.
                  setTrialGeometry((g) => ({ ...g, pcx: pc[0], pcy: pc[1], pcz: pc[2] }));
                }
              }}
            />
            </PhaseLoadedContext.Provider>
          }
          style={{ height: '100%' }}
        />
      </div>
      <PromptDialog {...promptProps} />
    </div>
  );
}
