/**
 * Convert a pointer event coordinate to a (row, col) pixel inside a letterboxed
 * canvas of given image shape. Returns null when the pointer is outside the
 * image area (i.e. in the letterbox bars).
 */
export function pointerToRowCol(ev, rect, shape) {
  const [natH, natW] = shape;
  if (!natW || !natH) return null;
  const x = ev.clientX - rect.left;
  const y = ev.clientY - rect.top;
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
 */
export function rowColToContainerPx(row, col, rect, shape) {
  const [natH, natW] = shape;
  if (!natW || !natH) return null;
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
