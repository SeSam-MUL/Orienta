/**
 * PatternCanvas — renders an EBSD pattern with brightness/contrast/gamma/CLAHE.
 *
 * Matches PyQt5 pattern_canvas (FigureCanvas with enhancement pipeline):
 *   - brightness/contrast/invert via CSS filter (cheap, GPU-accelerated)
 *   - gamma + CLAHE via per-pixel ImageData passes (applyGammaLUT / applyHistEq)
 *
 * Props:
 *   base64        string | null — either a `blob:`/`data:` URL (preferred,
 *                                 used by the navigation hook) or a bare
 *                                 base64 PNG payload (legacy form).
 *   enhancement   { brightness, contrast, gamma, invert, clahe }
 */

import { useEffect, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { applyGammaLUT, applyHistEq, patternToImgSrc } from '../utils/imageOps';

export default function PatternCanvas({ base64, enhancement }) {
  const { t } = useTranslation('hdf5viewer');
  const canvasRef = useRef(null);
  const imgRef = useRef(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !img.complete || !img.naturalWidth) return;

    const container = canvas.parentElement;
    if (!container) return;
    const maxW = container.clientWidth  || 500;
    const maxH = container.clientHeight || 500;

    const aspect = img.naturalWidth / img.naturalHeight;
    let w = Math.min(maxW, img.naturalWidth);
    let h = w / aspect;
    if (h > maxH) { h = maxH; w = h * aspect; }

    canvas.width  = Math.round(w);
    canvas.height = Math.round(h);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    if (enhancement.clahe || Math.abs(enhancement.gamma - 100) > 1) {
      const id = ctx.getImageData(0, 0, canvas.width, canvas.height);
      if (Math.abs(enhancement.gamma - 100) > 1) applyGammaLUT(id, enhancement.gamma);
      if (enhancement.clahe) applyHistEq(id);
      ctx.putImageData(id, 0, 0);
    }
  }, [enhancement]);

  useEffect(() => {
    if (!base64) {
      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
      return;
    }
    const img = new Image();
    img.onload = draw;
    img.src = patternToImgSrc(base64);
    imgRef.current = img;
  }, [base64, draw]);

  const cssFilter = buildCssFilter(enhancement);

  return (
    <canvas
      ref={canvasRef}
      aria-label={t('pattern.canvasAriaLabel')}
      style={{
        maxWidth: '100%',
        maxHeight: '100%',
        imageRendering: 'pixelated',
        display: 'block',
        filter: cssFilter,
      }}
    />
  );
}

function buildCssFilter({ brightness, contrast, invert }) {
  const b = 1 + brightness / 100;
  const c = contrast / 100;
  const parts = [`brightness(${b.toFixed(3)})`, `contrast(${c.toFixed(3)})`];
  if (invert) parts.push('invert(1)');
  return parts.join(' ');
}
