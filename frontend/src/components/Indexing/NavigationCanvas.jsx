/**
 * NavigationCanvas.jsx
 *
 * Interactive canvas for EBSD navigation map in IndexingPage.
 * Renders layered: base image → EDS overlays → mask dimming → region rectangle
 * Supports drag-to-select for region mode with bidirectional NumberInput sync.
 */

import { useRef, useEffect, useCallback, useState, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { colors as C } from '../../theme/tokens';
import { regionFromDrag, regionFrame, maskToRaster } from './hexNavMap';

export default function NavigationCanvas({
  navImage,
  nRows,
  nCols,
  region,
  savedRegions = [],
  selectionMode,
  onRegionChange,
  edsLayers = [],
  maskData = null,
  hexMap = null,   // makeHexMap() of a hexagonal scan: the picture is then the display raster; null = square scan
  notice = null,   // why there is no picture: 'loading' (answer on its way), 'mismatch', 'failed'; null = no data
}) {
  const { t } = useTranslation('indexing');
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const baseImgRef = useRef(null);
  const [isSelecting, setIsSelecting] = useState(false);
  const [dragStart, setDragStart] = useState(null);
  const [dragEnd, setDragEnd] = useState(null);
  const [canvasSize, setCanvasSize] = useState({ w: 400, h: 300 });
  const [imgVersion, setImgVersion] = useState(0); // triggers redraw when image loads

  // Load base image when navImage changes
  useEffect(() => {
    if (!navImage) { baseImgRef.current = null; setImgVersion(v => v + 1); return; }
    const img = new Image();
    img.onload = () => { baseImgRef.current = img; setImgVersion(v => v + 1); };
    img.src = `data:image/png;base64,${navImage}`;
  }, [navImage]);

  // Fit canvas to container
  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver(entries => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          setCanvasSize({ w: Math.floor(width), h: Math.floor(height) });
        }
      }
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  // Compute image draw rect (object-fit: contain logic)
  const getImageRect = useCallback(() => {
    const canvas = canvasRef.current;
    const img = baseImgRef.current;
    if (!canvas || !img) return null;
    const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
    const dw = img.width * scale;
    const dh = img.height * scale;
    const dx = (canvas.width - dw) / 2;
    const dy = (canvas.height - dh) / 2;
    return { dx, dy, dw, dh, scale };
  }, []);

  // Convert canvas pixel to grid coordinate
  const canvasToGrid = useCallback((cx, cy) => {
    const ir = getImageRect();
    if (!ir || nRows <= 0 || nCols <= 0) return null;
    const { dx, dy, dw, dh } = ir;
    const rx = (cx - dx) / dw;
    const ry = (cy - dy) / dh;
    if (rx < 0 || rx > 1 || ry < 0 || ry > 1) return null;
    return {
      col: Math.min(nCols - 1, Math.max(0, Math.floor(rx * nCols))),
      row: Math.min(nRows - 1, Math.max(0, Math.floor(ry * nRows))),
    };
  }, [nRows, nCols, getImageRect]);

  // Convert canvas pixel to fractions {rx, ry} of the picture (null outside it)
  const canvasToFraction = useCallback((cx, cy) => {
    const ir = getImageRect();
    if (!ir) return null;
    const rx = (cx - ir.dx) / ir.dw;
    const ry = (cy - ir.dy) / ir.dh;
    if (!(rx >= 0 && rx <= 1 && ry >= 0 && ry <= 1)) return null;
    return { rx, ry };
  }, [getImageRect]);

  // Convert grid coordinate to canvas pixel
  const gridToCanvas = useCallback((row, col) => {
    const ir = getImageRect();
    if (!ir || nRows <= 0 || nCols <= 0) return null;
    const { dx, dy, dw, dh } = ir;
    return {
      x: dx + (col / nCols) * dw,
      y: dy + (row / nRows) * dh,
    };
  }, [nRows, nCols, getImageRect]);

  // The canvas rectangle {x, y, w, h} of a region: between its grid lines on a square scan, around its hexagons
  // on a hexagonal one.
  const regionRect = useCallback((r) => {
    if (hexMap) {
      const ir = getImageRect();
      if (!ir) return null;
      const f = regionFrame(hexMap, r);
      return {
        x: ir.dx + f.x0 * ir.dw, y: ir.dy + f.y0 * ir.dh,
        w: (f.x1 - f.x0) * ir.dw, h: (f.y1 - f.y0) * ir.dh,
      };
    }
    const tl = gridToCanvas(r.rowStart, r.colStart);
    const br = gridToCanvas(r.rowEnd, r.colEnd);
    return tl && br ? { x: tl.x, y: tl.y, w: br.x - tl.x, h: br.y - tl.y } : null;
  }, [hexMap, getImageRect, gridToCanvas]);

  // The chemistry mask on the raster, once per mask (a hexagonal picture has more pixels than the grid has cells)
  const drawnMask = useMemo(
    () => (hexMap && maskData?.mask2d ? maskToRaster(hexMap, maskData) : maskData),
    [hexMap, maskData],
  );

  // Get mouse position relative to canvas
  const getCanvasPos = useCallback((e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const scaleX = canvasRef.current.width / rect.width;
    const scaleY = canvasRef.current.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  }, []);

  // --- Mouse handlers for region drawing ---
  const handleMouseDown = useCallback((e) => {
    if (selectionMode !== 'region') return;
    e.preventDefault();
    const pos = getCanvasPos(e);
    const grid = hexMap ? canvasToFraction(pos.x, pos.y) : canvasToGrid(pos.x, pos.y);
    if (!grid) return;
    setIsSelecting(true);
    setDragStart(pos);
    setDragEnd(pos);
  }, [selectionMode, hexMap, getCanvasPos, canvasToGrid, canvasToFraction]);

  const handleMouseMove = useCallback((e) => {
    if (!isSelecting) return;
    e.preventDefault();
    setDragEnd(getCanvasPos(e));
  }, [isSelecting, getCanvasPos]);

  const handleMouseUp = useCallback((e) => {
    if (!isSelecting) return;
    e.preventDefault();
    setIsSelecting(false);
    const endPos = getCanvasPos(e);
    if (hexMap) {
      // The dragged rectangle is a rectangle on the sample: the box of the cells whose centres it contains.
      const a = canvasToFraction(dragStart.x, dragStart.y);
      const b = canvasToFraction(endPos.x, endPos.y);
      const picked = a && b ? regionFromDrag(hexMap, a, b) : null;
      if (picked) onRegionChange(picked);
      setDragStart(null);
      setDragEnd(null);
      return;
    }
    const startGrid = canvasToGrid(dragStart.x, dragStart.y);
    const endGrid = canvasToGrid(endPos.x, endPos.y);
    if (!startGrid || !endGrid) { setDragStart(null); setDragEnd(null); return; }
    const rowStart = Math.min(startGrid.row, endGrid.row);
    const rowEnd = Math.max(startGrid.row, endGrid.row) + 1;
    const colStart = Math.min(startGrid.col, endGrid.col);
    const colEnd = Math.max(startGrid.col, endGrid.col) + 1;
    onRegionChange({ rowStart, rowEnd, colStart, colEnd });
    setDragStart(null);
    setDragEnd(null);
  }, [isSelecting, dragStart, hexMap, getCanvasPos, canvasToGrid, canvasToFraction, onRegionChange]);

  const handleMouseLeave = useCallback(() => {
    if (isSelecting) {
      setIsSelecting(false);
      setDragStart(null);
      setDragEnd(null);
    }
  }, [isSelecting]);

  // --- DRAWING ---
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = canvasSize.w;
    canvas.height = canvasSize.h;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = C.bgSecondary;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const img = baseImgRef.current;
    if (!img) {
      ctx.fillStyle = C.textSecondary;
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      if (notice === 'loading') {
        // the data is loaded and the picture is coming: not the hint that nothing is
        ctx.fillText(t('ebsdviewer:overview.computing'), canvas.width / 2, canvas.height / 2);
      } else if (notice === 'mismatch') {
        drawWrapped(ctx, t('ebsdviewer:overview.rasterMismatch'), canvas.width / 2, canvas.height / 2, canvas.width - 24);
      } else if (notice === 'failed') {
        ctx.fillText(t('ebsdviewer:overview.failed'), canvas.width / 2, canvas.height / 2);
      } else {
        ctx.fillText(t('navMap.noImage'), canvas.width / 2, canvas.height / 2);
        ctx.font = '10px sans-serif';
        ctx.fillText(t('navMap.loadFirst'), canvas.width / 2, canvas.height / 2 + 18);
      }
      return;
    }

    // Draw base image (contain)
    const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
    const dw = img.width * scale;
    const dh = img.height * scale;
    const dx = (canvas.width - dw) / 2;
    const dy = (canvas.height - dh) / 2;
    ctx.drawImage(img, dx, dy, dw, dh);

    // Draw EDS overlay layers (their images are index-grid pictures: never laid over a display raster)
    for (const layer of hexMap ? [] : edsLayers) {
      if (layer.image) {
        ctx.save();
        ctx.globalAlpha = layer.opacity;
        ctx.drawImage(layer.image, dx, dy, dw, dh);
        ctx.restore();
      }
    }

    // Draw mask dimming (A+C): darken non-masked pixels
    if (drawnMask && drawnMask.mask2d && selectionMode === 'mask') {
      drawMaskOverlay(ctx, drawnMask, dx, dy, dw, dh);
    }

    // Draw region selection overlay
    if (selectionMode === 'region') {
      // Draw saved regions
      for (const sr of savedRegions) {
        // an empty region has no hexagons to frame (its frame would run across the whole picture)
        if (hexMap && !(sr.rowEnd > sr.rowStart && sr.colEnd > sr.colStart)) continue;
        const box = regionRect(sr);
        if (box) {
          ctx.save();
          ctx.strokeStyle = 'rgba(100,150,255,0.6)';
          ctx.lineWidth = 1;
          ctx.setLineDash([4, 4]);
          ctx.strokeRect(box.x, box.y, box.w, box.h);
          ctx.setLineDash([]);
          ctx.restore();
        }
      }

      if (isSelecting && dragStart && dragEnd) {
        // Live drag preview with dimming
        const x1 = Math.min(dragStart.x, dragEnd.x);
        const y1 = Math.min(dragStart.y, dragEnd.y);
        const rw = Math.abs(dragEnd.x - dragStart.x);
        const rh = Math.abs(dragEnd.y - dragStart.y);
        drawDimmedRegion(ctx, { x: x1, y: y1, w: rw, h: rh }, dx, dy, dw, dh, img, hexMap ? [] : edsLayers);
      } else if (region.rowEnd > region.rowStart && region.colEnd > region.colStart) {
        // Draw from NumberInput values
        const rect = regionRect(region);
        if (rect) drawDimmedRegion(ctx, rect, dx, dy, dw, dh, img, hexMap ? [] : edsLayers);
      }
    }

    // Grid size label
    if (nCols > 0 && nRows > 0) {
      ctx.save();
      ctx.fillStyle = 'rgba(0,0,0,0.7)';
      const label = t('navMap.gridSize', { cols: nCols, rows: nRows });
      ctx.font = '10px monospace';
      const tw = ctx.measureText(label).width;
      ctx.fillRect(dx + 2, dy + 2, tw + 10, 18);
      ctx.fillStyle = C.text;
      ctx.fillText(label, dx + 7, dy + 14);
      ctx.restore();
    }
  }, [canvasSize, edsLayers, drawnMask, selectionMode, region, savedRegions, isSelecting, dragStart, dragEnd, nRows, nCols, regionRect, hexMap, imgVersion, notice, t]);

  // Redraw on any state change
  useEffect(() => { drawCanvas(); }, [drawCanvas]);

  return (
    <div
      ref={containerRef}
      style={{
        flex: 1,
        position: 'relative',
        overflow: 'hidden',
        background: C.bgSecondary,
        border: `1px solid ${C.border}`,
        borderRadius: 4,
        minHeight: 180,
      }}
    >
      <canvas
        ref={canvasRef}
        aria-label={t('navMap.dragToSelect')}
        style={{
          display: 'block',
          width: '100%',
          height: '100%',
          cursor: selectionMode === 'region' ? 'crosshair' : 'default',
          imageRendering: 'pixelated',
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        draggable={false}
        onDragStart={e => e.preventDefault()}
      />
    </div>
  );
}

/* ---- Drawing helpers ---- */

// A line of text broken at spaces to fit `maxWidth`, centred on x, the first line at y.
function drawWrapped(ctx, text, x, y, maxWidth) {
  const lines = [];
  let line = '';
  for (const word of String(text).split(' ')) {
    const next = line ? `${line} ${word}` : word;
    if (line && ctx.measureText(next).width > maxWidth) { lines.push(line); line = word; } else line = next;
  }
  lines.push(line);
  lines.forEach((l, k) => ctx.fillText(l, x, y + k * 16));
}

function drawDimmedRegion(ctx, rect, dx, dy, dw, dh, baseImg, edsLayers) {
  // Dim entire image area
  ctx.save();
  ctx.fillStyle = 'rgba(0,0,0,0.5)';
  ctx.fillRect(dx, dy, dw, dh);

  // Clip to selection rect, redraw bright content
  ctx.save();
  ctx.beginPath();
  ctx.rect(rect.x, rect.y, rect.w, rect.h);
  ctx.clip();
  ctx.clearRect(rect.x, rect.y, rect.w, rect.h);
  ctx.drawImage(baseImg, dx, dy, dw, dh);
  for (const layer of edsLayers) {
    if (layer.image) {
      ctx.globalAlpha = layer.opacity;
      ctx.drawImage(layer.image, dx, dy, dw, dh);
      ctx.globalAlpha = 1.0;
    }
  }
  ctx.restore();

  // Dashed white border
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([6, 4]);
  ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
  ctx.setLineDash([]);
  ctx.restore();
}

function drawMaskOverlay(ctx, maskData, dx, dy, dw, dh) {
  const { mask2d, shape, skip } = maskData;       // skip: pixels that are not part of the scan (hexagonal raster)
  const [rows, cols] = shape;

  // --- Layer 1: Excluded pixels get red-tinted diagonal hatching ---
  const offscreen = document.createElement('canvas');
  offscreen.width = cols;
  offscreen.height = rows;
  const octx = offscreen.getContext('2d');
  const imgData = octx.createImageData(cols, rows);

  for (let i = 0; i < rows * cols; i++) {
    const masked = mask2d[i];
    if (!masked && !(skip && skip[i])) {
      const r = Math.floor(i / cols);
      const c = i % cols;
      // Diagonal hatching: every 3rd diagonal gets a red line
      const onHatch = (r + c) % 3 === 0;
      imgData.data[i * 4 + 0] = onHatch ? 200 : 0;   // R
      imgData.data[i * 4 + 1] = 0;                     // G
      imgData.data[i * 4 + 2] = 0;                     // B
      imgData.data[i * 4 + 3] = onHatch ? 140 : 100;  // A: hatch lines brighter
    }
    // masked pixels: fully transparent (no overlay)
  }
  octx.putImageData(imgData, 0, 0);
  ctx.drawImage(offscreen, dx, dy, dw, dh);

  // --- Layer 2: Selected pixels get subtle green tint ---
  const selCanvas = document.createElement('canvas');
  selCanvas.width = cols;
  selCanvas.height = rows;
  const sctx = selCanvas.getContext('2d');
  const selData = sctx.createImageData(cols, rows);

  for (let i = 0; i < rows * cols; i++) {
    if (mask2d[i]) {
      selData.data[i * 4 + 0] = 80;   // R
      selData.data[i * 4 + 1] = 250;  // G
      selData.data[i * 4 + 2] = 123;  // B
      selData.data[i * 4 + 3] = 30;   // very subtle green tint
    }
  }
  sctx.putImageData(selData, 0, 0);
  ctx.drawImage(selCanvas, dx, dy, dw, dh);

  // --- Layer 3: Cyan border at mask edges ---
  const borderCanvas = document.createElement('canvas');
  borderCanvas.width = cols;
  borderCanvas.height = rows;
  const bctx = borderCanvas.getContext('2d');
  const borderData = bctx.createImageData(cols, rows);

  // A pixel no hexagon covers (`skip`, hexagonal raster only) is not an excluded one: it is no mask edge.
  const outside = (i) => !mask2d[i] && !(skip && skip[i]);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const idx = r * cols + c;
      if (!mask2d[idx]) continue;
      const isEdge = (
        (r > 0 && outside((r - 1) * cols + c)) ||
        (r < rows - 1 && outside((r + 1) * cols + c)) ||
        (c > 0 && outside(r * cols + c - 1)) ||
        (c < cols - 1 && outside(r * cols + c + 1))
      );
      if (isEdge) {
        borderData.data[idx * 4 + 0] = 139;  // cyan
        borderData.data[idx * 4 + 1] = 233;
        borderData.data[idx * 4 + 2] = 253;
        borderData.data[idx * 4 + 3] = 220;
      }
    }
  }
  bctx.putImageData(borderData, 0, 0);
  ctx.drawImage(borderCanvas, dx, dy, dw, dh);
}
