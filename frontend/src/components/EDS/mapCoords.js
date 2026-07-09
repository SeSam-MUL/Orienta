/**
 * Contain-fit a box of the given aspect ratio into (boxW, boxH). Returns the
 * displayed content size plus the top-left letterbox offset. This is the
 * object-fit:contain math used both for sizing the canvas inside its
 * container (letterbox 1) and for placing a zoomed bbox inside the canvas
 * buffer (letterbox 2).
 */
function containFit(boxW, boxH, aspect) {
  const boxAspect = boxW / boxH;
  let w, h;
  if (boxAspect > aspect) {
    h = boxH;
    w = h * aspect;
  } else {
    w = boxW;
    h = w / aspect;
  }
  return { w, h, offX: (boxW - w) / 2, offY: (boxH - h) / 2 };
}

/**
 * Given the native buffer dims and a content bbox (native map pixels), return
 * the rectangle (in native buffer pixels) that the bbox occupies after being
 * drawn object-fit:contain into the buffer. Mirrors the exact drawImage math
 * in LayeredCanvas (including the Math.round on the destination rect) so the
 * pointer mapping lands on the same pixel the user sees.
 */
export function bboxContentRect(natW, natH, bbox) {
  const bboxAspect = bbox.w / bbox.h;
  const bufAspect = natW / natH;
  let dstW, dstH;
  if (bboxAspect > bufAspect) {
    dstW = natW;
    dstH = natW / bboxAspect;
  } else {
    dstH = natH;
    dstW = natH * bboxAspect;
  }
  return {
    x: Math.round((natW - dstW) / 2),
    y: Math.round((natH - dstH) / 2),
    w: Math.round(dstW),
    h: Math.round(dstH),
  };
}

/**
 * Compose both letterboxes and return the CSS-pixel rectangle (relative to the
 * container top-left) actually occupied by the drawn map content, plus the
 * native-pixel source rect it represents.
 *
 * Without a bbox the content fills the letterboxed canvas (single letterbox by
 * full-shape aspect — identical to legacy behaviour). With a bbox the canvas
 * CSS box is still sized by the FULL shape aspect (letterbox 1), and the bbox
 * is drawn object-fit:contain INSIDE that canvas buffer (letterbox 2); the two
 * letterboxes compose here.
 */
export function displayContentRect(rect, shape, contentBbox = null) {
  const [natH, natW] = shape;
  const fit = containFit(rect.width, rect.height, natW / natH); // letterbox 1
  if (!contentBbox) {
    return {
      x: fit.offX, y: fit.offY, w: fit.w, h: fit.h,
      srcX: 0, srcY: 0, srcW: natW, srcH: natH,
    };
  }
  // Uniform buffer→CSS scale (fit.w/natW === fit.h/natH because the canvas box
  // shares the full-shape aspect with the native buffer).
  const s = fit.w / natW;
  const dst = bboxContentRect(natW, natH, contentBbox); // letterbox 2 (buffer px)
  return {
    x: fit.offX + dst.x * s,
    y: fit.offY + dst.y * s,
    w: dst.w * s,
    h: dst.h * s,
    srcX: contentBbox.x, srcY: contentBbox.y, srcW: contentBbox.w, srcH: contentBbox.h,
  };
}

/**
 * Convert a pointer event coordinate to a (row, col) pixel inside a letterboxed
 * canvas of given image shape. Returns null when the pointer is outside the
 * displayed content (i.e. in the letterbox bars).
 *
 * `contentBbox` (native map pixels {x, y, w, h}) is optional. When set the
 * canvas is auto-zoomed to that bbox, so the mapping composes both letterboxes
 * and returns coordinates INSIDE the bbox. Passing no bbox (default null) is
 * byte-for-byte compatible with the legacy single-letterbox behaviour.
 */
export function pointerToRowCol(ev, rect, shape, contentBbox = null) {
  const [natH, natW] = shape;
  if (!natW || !natH) return null;
  const x = ev.clientX - rect.left;
  const y = ev.clientY - rect.top;

  if (contentBbox) {
    const c = displayContentRect(rect, shape, contentBbox);
    const fx = (x - c.x) / c.w;
    const fy = (y - c.y) / c.h;
    if (fx < 0 || fy < 0 || fx >= 1 || fy >= 1) return null;
    return {
      row: Math.floor(c.srcY + fy * c.srcH),
      col: Math.floor(c.srcX + fx * c.srcW),
    };
  }

  // Legacy single-letterbox path — kept verbatim for byte-for-byte parity.
  const containerAspect = rect.width / rect.height;
  const imageAspect = natW / natH;
  let dispW, dispH, offX, offY;
  if (containerAspect > imageAspect) {
    dispH = rect.height;
    dispW = rect.height * imageAspect;
    offX = (rect.width - dispW) / 2;
    offY = 0;
  } else {
    dispW = rect.width;
    dispH = rect.width / imageAspect;
    offX = 0;
    offY = (rect.height - dispH) / 2;
  }
  const px = (x - offX) / dispW * natW;
  const py = (y - offY) / dispH * natH;
  if (px < 0 || py < 0 || px >= natW || py >= natH) return null;
  return { row: Math.floor(py), col: Math.floor(px) };
}

/**
 * Inverse: given (row, col) on an image of given shape and a container rect,
 * return CSS pixel offsets relative to the container's top-left.
 * Returns null when shape has a zero dimension.
 *
 * `contentBbox` (optional) mirrors pointerToRowCol — when set, positions the
 * point through the composed double-letterbox so overlays (crosshair) land on
 * the auto-zoomed content.
 */
export function rowColToContainerPx(row, col, rect, shape, contentBbox = null) {
  const [natH, natW] = shape;
  if (!natW || !natH) return null;

  if (contentBbox) {
    const c = displayContentRect(rect, shape, contentBbox);
    const fx = (col + 0.5 - c.srcX) / c.srcW;
    const fy = (row + 0.5 - c.srcY) / c.srcH;
    return { x: c.x + fx * c.w, y: c.y + fy * c.h };
  }

  // Legacy single-letterbox path — kept verbatim for byte-for-byte parity.
  const containerAspect = rect.width / rect.height;
  const imageAspect = natW / natH;
  let dispW, dispH, offX, offY;
  if (containerAspect > imageAspect) {
    dispH = rect.height;
    dispW = rect.height * imageAspect;
    offX = (rect.width - dispW) / 2;
    offY = 0;
  } else {
    dispW = rect.width;
    dispH = rect.width / imageAspect;
    offX = 0;
    offY = (rect.height - dispH) / 2;
  }
  return {
    x: offX + (col + 0.5) / natW * dispW,
    y: offY + (row + 0.5) / natH * dispH,
  };
}
