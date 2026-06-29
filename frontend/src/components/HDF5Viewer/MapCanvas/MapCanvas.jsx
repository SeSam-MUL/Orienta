/**
 * MapCanvas — large center panel showing the active map layer.
 * Click navigates to the corresponding pixel.
 */
import { useRef, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors } from '../../../theme/components';
import { useLayers } from '../hooks/useLayers';
import ColorLegend from './ColorLegend';
import LassoOverlay from './LassoOverlay';
import useCockpitStore from '../../../stores/useCockpitStore';

export default function MapCanvas({ isFileOpen, gridShape, currentRow, currentCol, onNavigate }) {
  const { t } = useTranslation('hdf5viewer');
  const { activeImage, activeMeta, loading, error } = useLayers(isFileOpen);
  const canvasRef = useRef(null);
  const [selectMode, setSelectMode] = useState(false);
  const selection = useCockpitStore((s) => s.selection);
  const clearSelection = useCockpitStore((s) => s.clearSelection);

  // Render map + crosshair
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !activeImage) return;
    const img = new Image();
    img.onload = () => {
      const ctx = canvas.getContext('2d');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, 0, 0);
      // Crosshair
      ctx.strokeStyle = '#ff79c6';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(currentCol + 0.5, 0);
      ctx.lineTo(currentCol + 0.5, canvas.height);
      ctx.moveTo(0, currentRow + 0.5);
      ctx.lineTo(canvas.width, currentRow + 0.5);
      ctx.stroke();
      // Render selection mask as a purple overlay
      if (selection?.mask && gridShape) {
        const [nRows, nCols] = gridShape;
        ctx.fillStyle = 'rgba(189, 147, 249, 0.4)';
        for (let r = 0; r < nRows; r++) {
          for (let c = 0; c < nCols; c++) {
            if (selection.mask[r * nCols + c]) ctx.fillRect(c, r, 1, 1);
          }
        }
      }
    };
    img.src = `data:image/png;base64,${activeImage}`;
  }, [activeImage, currentRow, currentCol, selection, gridShape]);

  const handleClick = (e) => {
    const canvas = canvasRef.current;
    if (!canvas || !gridShape) return;
    const rect = canvas.getBoundingClientRect();
    const xRatio = canvas.width / rect.width;
    const yRatio = canvas.height / rect.height;
    const col = Math.floor((e.clientX - rect.left) * xRatio);
    const row = Math.floor((e.clientY - rect.top) * yRatio);
    if (row >= 0 && row < gridShape[0] && col >= 0 && col < gridShape[1]) {
      onNavigate?.(row, col);
    }
  };

  return (
    <div style={{
      height: '100%', display: 'flex', flexDirection: 'column',
      background: '#000', position: 'relative', overflow: 'hidden',
    }}>
      {/* Title bar */}
      <div style={{
        flexShrink: 0, padding: '4px 10px',
        background: colors.bgSecondary, borderBottom: `1px solid ${colors.border}`,
        fontSize: '9pt', color: colors.text, display: 'flex', alignItems: 'center', gap: 8,
      }}>
        <span>{activeMeta?.name ?? t('mapCanvas.noLayer')}</span>
        <div style={{ flex: 1 }} />
        <button
          onClick={() => setSelectMode((s) => !s)}
          title={t('mapCanvas.selectTooltip')}
          style={{
            fontSize: '8pt', padding: '2px 8px',
            background: selectMode ? colors.purple : colors.bgTertiary,
            color: selectMode ? '#fff' : colors.text,
            border: `1px solid ${colors.border}`,
            borderRadius: 3,
            cursor: 'pointer',
          }}
        >
          {selectMode ? t('mapCanvas.selecting') : t('mapCanvas.selectMode')}
        </button>
        {selection?.mask && (
          <button
            onClick={clearSelection}
            title={t('mapCanvas.clearSelectionTooltip')}
            style={{
              fontSize: '8pt', padding: '2px 8px',
              background: colors.bgTertiary, color: colors.red,
              border: `1px solid ${colors.border}`,
              borderRadius: 3,
              cursor: 'pointer',
            }}
          >
            {t('mapCanvas.clearSelection')}
          </button>
        )}
        {activeMeta?.min != null && (
          <span style={{ color: colors.textSecondary, fontFamily: 'monospace' }}>
            {activeMeta.min.toFixed(2)} / {activeMeta.max.toFixed(2)}
          </span>
        )}
      </div>

      {/* Canvas area */}
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden', position: 'relative' }}>
        {loading && <div style={{ color: colors.textSecondary }}>{t('mapCanvas.loadingLayer')}</div>}
        {error && <div style={{ color: colors.red }}>{error}</div>}
        {activeImage && (
          <canvas
            ref={canvasRef}
            onClick={selectMode ? null : handleClick}
            title={t('mapCanvas.canvasTooltip')}
            style={{
              maxWidth: '100%', maxHeight: '100%',
              imageRendering: 'pixelated', cursor: 'crosshair',
            }}
          />
        )}
        <LassoOverlay canvasRef={canvasRef} gridShape={gridShape} enabled={selectMode} />
        <ColorLegend
          kind={activeMeta?.kind}
          min={activeMeta?.min}
          max={activeMeta?.max}
          phases={activeMeta?.phases}
        />
      </div>
    </div>
  );
}
