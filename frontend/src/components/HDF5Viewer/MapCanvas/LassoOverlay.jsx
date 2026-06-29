import { useState, useEffect } from 'react';
import { colors } from '../../../theme/components';
import useCockpitStore from '../../../stores/useCockpitStore';

export default function LassoOverlay({ canvasRef, gridShape, enabled }) {
  const [start, setStart] = useState(null);
  const [end, setEnd] = useState(null);
  const setSelection = useCockpitStore((s) => s.setSelection);

  useEffect(() => {
    if (!enabled || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const onDown = (e) => {
      const rect = canvas.getBoundingClientRect();
      setStart({ x: e.clientX - rect.left, y: e.clientY - rect.top });
      setEnd({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    };
    const onMove = (e) => {
      if (!start) return;
      const rect = canvas.getBoundingClientRect();
      setEnd({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    };
    const onUp = () => {
      if (!start || !end || !gridShape) { setStart(null); setEnd(null); return; }
      const rect = canvas.getBoundingClientRect();
      const xRatio = canvas.width / rect.width;
      const yRatio = canvas.height / rect.height;
      const r0 = Math.floor(Math.min(start.y, end.y) * yRatio);
      const r1 = Math.ceil(Math.max(start.y, end.y) * yRatio);
      const c0 = Math.floor(Math.min(start.x, end.x) * xRatio);
      const c1 = Math.ceil(Math.max(start.x, end.x) * xRatio);
      const [nRows, nCols] = gridShape;
      const mask = new Uint8Array(nRows * nCols);
      let count = 0;
      for (let r = r0; r < r1; r++) {
        for (let c = c0; c < c1; c++) {
          if (r >= 0 && r < nRows && c >= 0 && c < nCols) {
            mask[r * nCols + c] = 1;
            count++;
          }
        }
      }
      if (count > 0) setSelection(mask, 'rect');
      setStart(null);
      setEnd(null);
    };
    canvas.addEventListener('mousedown', onDown);
    canvas.addEventListener('mousemove', onMove);
    canvas.addEventListener('mouseup', onUp);
    return () => {
      canvas.removeEventListener('mousedown', onDown);
      canvas.removeEventListener('mousemove', onMove);
      canvas.removeEventListener('mouseup', onUp);
    };
  }, [enabled, canvasRef, gridShape, start, end, setSelection]);

  if (!start || !end) return null;
  const left = Math.min(start.x, end.x);
  const top = Math.min(start.y, end.y);
  const width = Math.abs(start.x - end.x);
  const height = Math.abs(start.y - end.y);
  return (
    <div style={{
      position: 'absolute',
      left: `${(left / (canvasRef.current?.getBoundingClientRect().width || 1)) * 100}%`,
      top: `${(top / (canvasRef.current?.getBoundingClientRect().height || 1)) * 100}%`,
      width: `${(width / (canvasRef.current?.getBoundingClientRect().width || 1)) * 100}%`,
      height: `${(height / (canvasRef.current?.getBoundingClientRect().height || 1)) * 100}%`,
      border: `1px dashed ${colors.purple}`,
      background: 'rgba(189, 147, 249, 0.15)',
      pointerEvents: 'none',
    }} />
  );
}
