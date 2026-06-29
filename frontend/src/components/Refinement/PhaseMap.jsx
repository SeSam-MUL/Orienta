import { useRef, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/tokens';

// Scale factor: pixels per EBSD pixel on the canvas
const SCALE = 4;

// Distinct colors for phase indices 0..N.
// Index 0 is conventionally "not indexed" / background.
const PHASE_PALETTE = [
  '#444444', // 0 — not indexed / background
  '#4EC9FF', // 1
  '#FF6B6B', // 2
  '#A8FF78', // 3
  '#FFD93D', // 4
  '#C77DFF', // 5
  '#FF9F43', // 6
  '#48DBFB', // 7
  '#FF6FD8', // 8
];

function phaseColor(index) {
  return PHASE_PALETTE[index % PHASE_PALETTE.length] ?? '#888888';
}

// ---------------------------------------------------------------------------
// PhaseMap
// ---------------------------------------------------------------------------
export default function PhaseMap({
  phaseMapData,   // flat Int32Array or Array of phase indices, row-major
  gridShape,      // [rows, cols]
  onPixelClick,   // (row, col) => void
  selectedPixel,  // { row, col } | null
  uncertaintyOverlay, // flat Float32Array of CI values (same shape) | null
  ciThreshold,    // number — pixels below this CI get hatched
}) {
  const { t } = useTranslation('refinement');
  const canvasRef = useRef(null);

  const [rows, cols] = gridShape ?? [0, 0];
  const canvasWidth  = cols * SCALE;
  const canvasHeight = rows * SCALE;

  // Draw whenever data changes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !phaseMapData || rows === 0 || cols === 0) return;

    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(canvasWidth, canvasHeight);
    const data = imageData.data;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const srcIdx = r * cols + c;
        const phaseIdx = phaseMapData[srcIdx] ?? 0;
        const hex = phaseColor(phaseIdx);

        // Parse hex to RGB
        const ri = parseInt(hex.slice(1, 3), 16);
        const gi = parseInt(hex.slice(3, 5), 16);
        const bi = parseInt(hex.slice(5, 7), 16);

        // Uncertainty overlay: dim pixels below CI threshold
        let alpha = 255;
        if (uncertaintyOverlay) {
          const ci = uncertaintyOverlay[srcIdx] ?? 1;
          if (ci < (ciThreshold ?? 0.3)) alpha = 80;
        }

        // Fill SCALE x SCALE block
        for (let dy = 0; dy < SCALE; dy++) {
          for (let dx = 0; dx < SCALE; dx++) {
            const px = ((r * SCALE + dy) * canvasWidth + (c * SCALE + dx)) * 4;
            data[px]     = ri;
            data[px + 1] = gi;
            data[px + 2] = bi;
            data[px + 3] = alpha;
          }
        }
      }
    }

    ctx.putImageData(imageData, 0, 0);

    // Draw selected pixel highlight
    if (selectedPixel) {
      const { row: sr, col: sc } = selectedPixel;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.strokeRect(sc * SCALE, sr * SCALE, SCALE, SCALE);
    }
  }, [phaseMapData, rows, cols, canvasWidth, canvasHeight, selectedPixel, uncertaintyOverlay, ciThreshold]);

  const handleClick = useCallback((e) => {
    if (!onPixelClick || rows === 0 || cols === 0) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    // Scale from display pixels back to canvas pixels
    const scaleX = canvasWidth / rect.width;
    const scaleY = canvasHeight / rect.height;
    const canvasX = x * scaleX;
    const canvasY = y * scaleY;
    const col = Math.floor(canvasX / SCALE);
    const row = Math.floor(canvasY / SCALE);
    if (row >= 0 && row < rows && col >= 0 && col < cols) {
      onPixelClick(row, col);
    }
  }, [onPixelClick, rows, cols, canvasWidth, canvasHeight]);

  if (rows === 0 || cols === 0) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: 200, color: colors.textSecondary, fontSize: '10pt',
      }}>
        {t('refinement:phaseMap.noData')}
      </div>
    );
  }

  return (
    <canvas
      ref={canvasRef}
      width={canvasWidth}
      height={canvasHeight}
      onClick={handleClick}
      style={{
        cursor: onPixelClick ? 'crosshair' : 'default',
        display: 'block',
        maxWidth: '100%',
        imageRendering: 'pixelated',
        border: `1px solid ${colors.border}`,
        borderRadius: 4,
      }}
      title={onPixelClick ? t('refinement:tooltips.phaseMapCanvas') : undefined}
      aria-label={t('refinement:phaseMap.ariaLabel')}
    />
  );
}

// Export palette so consumers can build a legend
export { PHASE_PALETTE, phaseColor };
