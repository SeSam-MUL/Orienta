/**
 * OverviewCanvas — scan-position minimap matching PyQt5 minimap_canvas.
 *
 * Renders a small grid with current-position crosshair (cyan) and bookmark
 * diamonds (purple). Click navigates to the underlying (row, col).
 *
 * Props:
 *   rows, cols        scan grid dimensions
 *   currentRow, currentCol  current cursor position
 *   bookmarks         [{ row, col, ... }, ...]
 *   onNavigate        (row, col) => void  — click handler
 *   minimapImage      base64 PNG | null   — optional real minimap to draw under the overlay
 *   width, height     optional canvas dimensions (default 180x120)
 */

import { useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { colors } from '../../../theme/components';

const DEFAULT_W = 180;
const DEFAULT_H = 120;

export default function OverviewCanvas({
  rows,
  cols,
  currentRow,
  currentCol,
  bookmarks,
  onNavigate,
  minimapImage,
  width = DEFAULT_W,
  height = DEFAULT_H,
}) {
  const { t } = useTranslation('hdf5viewer');
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, width, height);

    // Background
    ctx.fillStyle = colors.bgSecondary;
    ctx.fillRect(0, 0, width, height);

    // If we have a real minimap image, draw it first
    if (minimapImage) {
      const img = new Image();
      img.onload = () => {
        ctx.drawImage(img, 0, 0, width, height);
        drawOverlay(ctx, rows, cols, currentRow, currentCol, bookmarks, width, height);
      };
      img.src = `data:image/png;base64,${minimapImage}`;
      return;
    }

    // Fallback: draw grid pattern
    const cellW = width / Math.max(cols, 1);
    const cellH = height / Math.max(rows, 1);

    if (cols <= 80 && rows <= 50) {
      ctx.strokeStyle = 'rgba(100,100,130,0.3)';
      ctx.lineWidth = 0.5;
      for (let c = 0; c <= cols; c++) {
        ctx.beginPath();
        ctx.moveTo(c * cellW, 0);
        ctx.lineTo(c * cellW, height);
        ctx.stroke();
      }
      for (let r = 0; r <= rows; r++) {
        ctx.beginPath();
        ctx.moveTo(0, r * cellH);
        ctx.lineTo(width, r * cellH);
        ctx.stroke();
      }
    }

    drawOverlay(ctx, rows, cols, currentRow, currentCol, bookmarks, width, height);
  }, [rows, cols, currentRow, currentCol, bookmarks, minimapImage, width, height]);

  const handleClick = (e) => {
    if (!onNavigate) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const col = Math.max(0, Math.min(cols - 1, Math.floor((x / width) * cols)));
    const row = Math.max(0, Math.min(rows - 1, Math.floor((y / height) * rows)));
    onNavigate(row, col);
  };

  return (
    <canvas
      ref={canvasRef}
      aria-label={t('overview.ariaLabel')}
      width={width}
      height={height}
      onClick={handleClick}
      style={{
        display: 'block',
        borderRadius: 3,
        border: `1px solid ${colors.border}`,
        cursor: 'crosshair',
      }}
      title={t('overview.tooltip')}
    />
  );
}

function drawOverlay(ctx, rows, cols, currentRow, currentCol, bookmarks, width, height) {
  const cellW = width / Math.max(cols, 1);
  const cellH = height / Math.max(rows, 1);

  // Bookmarks — purple diamonds
  (bookmarks || []).forEach(({ row, col }) => {
    const cx = (col + 0.5) * cellW;
    const cy = (row + 0.5) * cellH;
    const s = Math.max(2, Math.min(cellW, cellH, 5));
    ctx.fillStyle = colors.purple;
    ctx.beginPath();
    ctx.moveTo(cx, cy - s);
    ctx.lineTo(cx + s, cy);
    ctx.lineTo(cx, cy + s);
    ctx.lineTo(cx - s, cy);
    ctx.closePath();
    ctx.fill();
  });

  // Current position — cyan crosshair + filled dot
  if (rows > 0 && cols > 0) {
    const cx = (currentCol + 0.5) * cellW;
    const cy = (currentRow + 0.5) * cellH;

    // Crosshair lines
    ctx.strokeStyle = 'rgba(139,233,253,0.8)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, 0); ctx.lineTo(cx, height);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, cy); ctx.lineTo(width, cy);
    ctx.stroke();

    // Dot
    const r = Math.max(3, Math.min(cellW, cellH, 6));
    ctx.fillStyle = colors.cyan;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
  }
}
