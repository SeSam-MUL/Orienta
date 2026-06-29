/**
 * PatternPreview — shows the experimental EBSD pattern with overlay
 * markers for detected zone-axis + n-fold rotation symmetry arrows.
 *
 * Uses an HTML canvas overlay on top of an <img> so the markers redraw
 * cheaply when the symmetry report changes (no canvas re-rasterise of
 * the pattern itself).
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/tokens';
import InfoTooltip from '../common/InfoTooltip';

const S = {
  card: {
    padding: 12,
    background: colors.bgSecondary,
    border: `1px solid ${colors.border}`,
    borderRadius: 6,
  },
  title: {
    fontSize: 13,
    fontWeight: 600,
    color: colors.accent,
    marginBottom: 8,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  patternBox: {
    position: 'relative',
    // Cap at 360 px so the preview fits in the right-panel column without
    // pushing the candidate list off-screen. Use maxWidth + aspectRatio
    // for browser compatibility (CSS min() doesn't always work in React
    // inline styles consistently across cached + fresh builds).
    maxWidth: 360,
    width: '100%',
    aspectRatio: '1 / 1',
    margin: '0 auto',
    background: colors.bg,
    borderRadius: 4,
    overflow: 'hidden',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  patternImg: {
    width: '100%',
    height: '100%',
    objectFit: 'contain',
    display: 'block',
  },
  overlay: {
    position: 'absolute',
    inset: 0,
    pointerEvents: 'none',
    // Force explicit transparent. Some Chromium builds render a fresh
    // <canvas> with a WHITE default surface for a brief flash before
    // the first ctx.clearRect call — that flash is what's masking the
    // pattern. background: transparent guarantees no flash, and any
    // arrow drawing still works because ctx draws above transparent.
    backgroundColor: 'transparent',
    background: 'transparent',
  },
  legend: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 10,
    marginTop: 8,
    fontSize: 11,
    color: colors.textSecondary,
  },
  legendItem: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 4,
  },
  legendDot: (color) => ({
    width: 10,
    height: 10,
    borderRadius: '50%',
    background: color,
    display: 'inline-block',
  }),
};

export default function PatternPreview({
  patternB64,
  patternShape,
  symmetry,
}) {
  const { t } = useTranslation('crystalhint');
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  // Magnifier state: pointer-relative within the container [0..1, 0..1] or null.
  const [magnifier, setMagnifier] = useState(null);
  const [magnifierEnabled, setMagnifierEnabled] = useState(true);
  // Circular detector mask — clips the white detector edges that appear
  // outside the inscribed circle on EDAX (and most other vendor) data.
  // Default on, matching the EBSD Viewer's circular-mask default.
  const [circularMask, setCircularMask] = useState(true);
  const handleMouseMove = useCallback((e) => {
    if (!magnifierEnabled) return;
    const r = containerRef.current?.getBoundingClientRect();
    if (!r) return;
    const x = (e.clientX - r.left) / r.width;
    const y = (e.clientY - r.top) / r.height;
    if (x < 0 || x > 1 || y < 0 || y > 1) {
      setMagnifier(null);
      return;
    }
    setMagnifier({ x, y });
  }, [magnifierEnabled]);
  const handleMouseLeave = useCallback(() => setMagnifier(null), []);
  // Track the canvas's rendered size so we redraw whenever the flex
  // container actually has dimensions. Without this, the first paint can
  // hit a 0×0 canvas (image hasn't sized the parent yet) and nothing draws.
  const [canvasSize, setCanvasSize] = useState({ w: 0, h: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let ro;
    if (typeof ResizeObserver !== 'undefined') {
      ro = new ResizeObserver(entries => {
        for (const entry of entries) {
          const r = entry.contentRect;
          setCanvasSize(prev =>
            (prev.w !== r.width || prev.h !== r.height)
              ? { w: r.width, h: r.height }
              : prev
          );
        }
      });
      ro.observe(canvas);
    } else {
      // Fallback for older browsers: one-shot read of current size
      const r = canvas.getBoundingClientRect();
      setCanvasSize({ w: r.width, h: r.height });
    }
    return () => { if (ro) ro.disconnect(); };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !patternShape || patternShape.length < 2) return;
    if (canvasSize.w === 0 || canvasSize.h === 0) return;  // not laid out yet

    // Use device pixel ratio for crisp lines on Retina/HiDPI
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvasSize.w * dpr;
    canvas.height = canvasSize.h * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, canvasSize.w, canvasSize.h);
    const rect = canvasSize;

    const [patternH, patternW] = patternShape;
    // Map pattern-coordinate (row, col) to canvas-coordinate
    const scaleX = rect.w / patternW;
    const scaleY = rect.h / patternH;

    if (!symmetry) return;
    const zoneYx = symmetry.zone_axis_yx;
    const nFold = symmetry.detected_n_fold;
    const confidence = symmetry.confidence;

    if (!zoneYx || nFold == null) return;
    const [zoneY, zoneX] = zoneYx;
    const cx = zoneX * scaleX;
    const cy = zoneY * scaleY;

    // Color by confidence
    const markerColor = {
      high: '#4ade80',     // green
      medium: '#facc15',   // amber
      low: '#fb923c',      // orange
      none: '#9ca3af',     // gray
    }[confidence] || '#9ca3af';

    // Draw zone-axis crosshair
    ctx.strokeStyle = markerColor;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(cx - 12, cy); ctx.lineTo(cx + 12, cy);
    ctx.moveTo(cx, cy - 12); ctx.lineTo(cx, cy + 12);
    ctx.stroke();
    // Circle around the zone axis
    ctx.beginPath();
    ctx.arc(cx, cy, 8, 0, Math.PI * 2);
    ctx.stroke();

    // Draw n-fold rotation arrows: n arrows radiating from the zone axis,
    // rotated by 2π/n around it
    const armLen = Math.min(rect.w, rect.h) * 0.30;
    ctx.lineWidth = 2;
    for (let i = 0; i < nFold; i++) {
      const angle = (2 * Math.PI * i) / nFold - Math.PI / 2; // start at top
      const ex = cx + armLen * Math.cos(angle);
      const ey = cy + armLen * Math.sin(angle);
      // Translucent line from zone axis to arrow tip
      ctx.strokeStyle = `${markerColor}aa`;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(ex, ey);
      ctx.stroke();
      // Arrowhead
      const headLen = 8;
      const a1 = angle + Math.PI - 0.35;
      const a2 = angle + Math.PI + 0.35;
      ctx.strokeStyle = markerColor;
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(ex + headLen * Math.cos(a1), ey + headLen * Math.sin(a1));
      ctx.moveTo(ex, ey);
      ctx.lineTo(ex + headLen * Math.cos(a2), ey + headLen * Math.sin(a2));
      ctx.stroke();
    }

    // Centre label
    ctx.fillStyle = markerColor;
    ctx.font = 'bold 11px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`${nFold}-fold`, cx + 12, cy - 12);
  }, [patternShape, symmetry, canvasSize]);

  if (!patternB64) {
    return (
      <div style={S.card}>
        <div style={S.title}>
          <span>{t('pattern.title')}</span>
        </div>
        <div style={{
          ...S.patternBox,
          color: colors.textSecondary,
          fontSize: 12,
        }}>
          {t('pattern.noData')}
        </div>
      </div>
    );
  }

  return (
    <div style={S.card}>
      <div style={S.title}>
        <span>{t('pattern.title')}</span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <label style={{ fontSize: 10, color: colors.textSecondary, cursor: 'pointer' }} title={t('pattern.circularMaskTooltip')}>
            <input
              type="checkbox"
              checked={circularMask}
              onChange={e => setCircularMask(e.target.checked)}
              style={{ marginRight: 4, verticalAlign: 'middle' }}
            />
            {t('pattern.circularMask')}
          </label>
          <label style={{ fontSize: 10, color: colors.textSecondary, cursor: 'pointer' }} title={t('pattern.magnifierTooltip')}>
            <input
              type="checkbox"
              checked={magnifierEnabled}
              onChange={e => setMagnifierEnabled(e.target.checked)}
              style={{ marginRight: 4, verticalAlign: 'middle' }}
            />
            {t('pattern.magnifier')}
          </label>
          {patternShape && (
            <span style={{ fontSize: 11, fontWeight: 400, color: colors.textSecondary }}>
              {t('pattern.pxSize', { h: patternShape[0], w: patternShape[1] })}
            </span>
          )}
        </span>
      </div>
      <div
        ref={containerRef}
        style={S.patternBox}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      >
        {/* Pattern body: plain <div> with backgroundImage. Same technique
            the working Magnifier uses. Removed role="img"+aria-label
            because some browsers render a broken-image placeholder for
            divs with that role even when the backgroundImage works
            (which was confusing users into thinking the render failed). */}
        <div
          data-testid="pattern-preview-body"
          style={{
            position: 'absolute', inset: 0,
            // Explicit dark backdrop so a slow/failed backgroundImage
            // shows dark, not the white page background bleeding through.
            background: '#000',
            backgroundImage: patternB64
              ? `url(data:image/png;base64,${patternB64})`
              : 'none',
            backgroundSize: 'contain',
            backgroundRepeat: 'no-repeat',
            backgroundPosition: 'center',
            imageRendering: 'pixelated',
            // Circular detector mask — clips the white square corners
            // that appear when EDAX/Bruker store the pattern as a square
            // but the actual detector is a disc.
            clipPath: circularMask ? 'circle(48% at 50% 50%)' : 'none',
          }}
        />
        {/* Diagnostic overlay if no pattern data arrived. Shows in red
            so a regression is obvious instead of a silent white box. */}
        {!patternB64 && (
          <div style={{
            position: 'absolute', inset: 0,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: colors.red, fontSize: 12,
          }}>
            {t('pattern.noDataResponse')}
          </div>
        )}
        {/* Canvas overlay for zone-axis arrows. Only mounted when there
            IS actually a zone axis to draw — keeps it out of the DOM
            entirely otherwise so it can't flash white on first render. */}
        {symmetry?.zone_axis_yx && symmetry?.detected_n_fold && (
          <canvas
            ref={canvasRef}
            style={S.overlay}
          />
        )}
        {magnifier && patternB64 && (() => {
          // 3× zoom lens centred at the cursor, clamped to the container.
          // background-image trick: same PNG, larger size, offset so the
          // pointer position maps to the lens centre.
          const ZOOM = 3;
          const LENS_PX = 120;
          const containerR = containerRef.current?.getBoundingClientRect();
          if (!containerR) return null;
          const cx = magnifier.x * containerR.width;
          const cy = magnifier.y * containerR.height;
          // Clamp lens so it stays inside the container
          const lensX = Math.max(0, Math.min(containerR.width - LENS_PX, cx - LENS_PX / 2));
          const lensY = Math.max(0, Math.min(containerR.height - LENS_PX, cy - LENS_PX / 2));
          // Background offset: pointer-x at scaled coords minus lens-half
          const bgW = containerR.width * ZOOM;
          const bgH = containerR.height * ZOOM;
          const bgX = -(magnifier.x * bgW - LENS_PX / 2);
          const bgY = -(magnifier.y * bgH - LENS_PX / 2);
          return (
            <div
              style={{
                position: 'absolute',
                left: lensX, top: lensY,
                width: LENS_PX, height: LENS_PX,
                borderRadius: '50%',
                border: `2px solid ${colors.accent}`,
                boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
                backgroundImage: `url(data:image/png;base64,${patternB64})`,
                backgroundRepeat: 'no-repeat',
                backgroundSize: `${bgW}px ${bgH}px`,
                backgroundPosition: `${bgX}px ${bgY}px`,
                imageRendering: 'pixelated',
                pointerEvents: 'none',
              }}
            />
          );
        })()}
      </div>
      {symmetry?.detected_n_fold && (
        <div style={S.legend}>
          <InfoTooltip inline content={(
            <>
              <div style={{ fontWeight: 700, marginBottom: 4, color: colors.accent }}>
                {t('pattern.confidenceTitle')}
              </div>
              <p style={{ margin: '4px 0' }}>
                {t('pattern.confidenceIntro')}
              </p>
              <ul style={{ margin: '4px 0', paddingLeft: 18 }}>
                <li><b style={{ color: '#4ade80' }}>{t('pattern.confidenceHigh')}</b>{t('pattern.confidenceHighBody')}</li>
                <li><b style={{ color: '#facc15' }}>{t('pattern.confidenceMedium')}</b>{t('pattern.confidenceMediumBody')}</li>
                <li><b style={{ color: '#fb923c' }}>{t('pattern.confidenceLow')}</b>{t('pattern.confidenceLowBody')}</li>
              </ul>
              <p style={{ margin: '4px 0', fontSize: 10,
                color: colors.textSecondary }}>
                {t('pattern.confidenceNote')}
              </p>
            </>
          )}>
            <span style={{ ...S.legendItem, cursor: 'help' }}>
              <span style={S.legendDot('#4ade80')} /> {t('pattern.legendHigh')}
              <span style={S.legendDot('#facc15')} /> {t('pattern.legendMedium')}
              <span style={S.legendDot('#fb923c')} /> {t('pattern.legendLow')}
              <span style={{ fontSize: 10, color: colors.accent, marginLeft: 4 }}>
                {t('pattern.whatsThis')}
              </span>
            </span>
          </InfoTooltip>
          <span style={{ marginLeft: 'auto', fontSize: 10 }}>
            {t('pattern.overlayLabel', { n: symmetry.detected_n_fold })}
          </span>
        </div>
      )}
    </div>
  );
}
