/**
 * Where the content of a layer actually is, in canvas pixels.
 *
 * The auto-zoom draws the map inside this box, so a small region run fills the
 * frame instead of sitting in the corner of an otherwise black canvas. The box
 * used to be min/max over every pixel carrying alpha, which is the one
 * statistic a handful of stray pixels can move all the way to the edge of the
 * grid: a 608-px region inside a ~168x150 scan came out as a thin bright strip
 * at the top of an empty canvas, and because the box then counted as "covers
 * everything" the zoom abstained altogether (BUG-PM-03).
 *
 * So the frame follows the BODY of the content. Per axis, the occupied lines
 * are grouped into contiguous runs and an outer run is dropped while it is a
 * speckle AND costs far more frame than it carries content. A fringe that
 * doubles the height of the picture to show three pixels is a finding, not a
 * region; a second block — even a much smaller one — the thin tip rows of a
 * round selection, and any run of more than a couple of dozen pixels are the
 * region and stay. What is dropped is not drawn at all (the caller hard-crops
 * to this box), so the rule has to be strict about what it calls a speckle.
 *
 * Returns null when the box would cover (essentially) the whole canvas — the
 * documented "this layer does not constrain the frame" answer, which is what
 * lets a full-scan layer abstain instead of cancelling another layer's crop.
 */

// A fringe may keep at most this share of the frame per share of content it
// carries: costing 20 % of the span buys it at most 10 % of the pixels.
const FRINGE_SPAN_FACTOR = 0.5;
// ...and whatever the geometry says, a cluster holding this much of the
// content is a region. Two blocks far apart are both the result.
const FRINGE_MASS_FRAC = 0.15;

// Size, not share: a fringe is a SPECKLE, not merely something small.
//
// Those two ratios alone put the cliff at ~10-17 % of a layer's mass, which is
// an arbitrary pixel count — measured on a 200x200 layer, a genuine 900 px and
// 100 px pair came back as the 900 alone (the 100-px region is then not drawn
// at all: the caller hard-crops to this box with drawImage, and zoom and pan
// sit after the crop, so nothing can bring it back). Worse for the abstention
// this module owes its caller: a full-width layer with 5 indexed rows at the
// top and a gap before rows 100-199 used to span the whole canvas and answer
// null — "I do not constrain the frame" — and instead cropped away 1000 real
// pixels.
//
// So a run is only ever a fringe while it is smaller than BOTH of these: a
// couple of dozen pixels in absolute terms (below that it cannot be a region
// anyone wants framed — the speckles that started this were one pixel each),
// and a fiftieth of the biggest run, which is what keeps the rule working on
// a layer where a real region is hundreds of thousands of pixels.
const FRINGE_ABS_MAX = 24;
const FRINGE_OF_LARGEST = 0.02;

// Contiguous groups of occupied lines: [{ start, end, mass }], in order.
function runsOf(counts) {
  const runs = [];
  let cur = null;
  for (let i = 0; i < counts.length; i++) {
    if (counts[i] > 0) {
      if (cur) { cur.end = i; cur.mass += counts[i]; }
      else cur = { start: i, end: i, mass: counts[i] };
    } else if (cur) {
      runs.push(cur);
      cur = null;
    }
  }
  if (cur) runs.push(cur);
  return runs;
}

/**
 * The extent of one axis after dropping outer fringe runs.
 * Returns null when the axis holds nothing.
 */
function bodyExtent(counts) {
  const runs = runsOf(counts);
  if (runs.length === 0) return null;

  let mass = runs.reduce((s, r) => s + r.mass, 0);
  // Measured against the biggest run there IS, not against what is left after
  // trimming, so dropping speckles one by one can never promote the next one
  // into being a fringe.
  const largest = runs.reduce((m, r) => Math.max(m, r.mass), 0);
  const speckleMax = Math.max(FRINGE_ABS_MAX, FRINGE_OF_LARGEST * largest);

  let dropped = true;
  while (runs.length > 1 && dropped) {
    dropped = false;
    // Measured once per pass, so the second end of this pass judges itself
    // against the span before the first end was dropped. Deliberate: a stale
    // (larger) span makes spanFrac smaller and the test harder to pass, so the
    // error is always towards keeping content, and the next pass re-measures.
    const span = runs[runs.length - 1].end - runs[0].start + 1;
    for (const end of ['first', 'last']) {
      if (runs.length < 2) break;
      const run = end === 'first' ? runs[0] : runs[runs.length - 1];
      const next = end === 'first' ? runs[1] : runs[runs.length - 2];
      // What dropping this run takes off the span: the run itself plus the
      // gap between it and the body it is separated from.
      const removed = end === 'first'
        ? next.start - runs[0].start
        : runs[runs.length - 1].end - next.end;
      const massFrac = run.mass / mass;
      const spanFrac = removed / span;
      if (run.mass < speckleMax
          && massFrac < FRINGE_MASS_FRAC
          && massFrac < FRINGE_SPAN_FACTOR * spanFrac) {
        if (end === 'first') runs.shift(); else runs.pop();
        mass -= run.mass;
        dropped = true;
      }
    }
  }
  return { start: runs[0].start, end: runs[runs.length - 1].end };
}

/**
 * Bounding box of the body of a layer, from raw RGBA data.
 * `data` is an ImageData buffer for a `w` x `h` canvas.
 */
export function bboxFromAlpha(data, w, h) {
  if (w <= 0 || h <= 0 || !data) return null;

  const rowCounts = new Int32Array(h);
  for (let y = 0; y < h; y++) {
    const rowOff = y * w * 4;
    let n = 0;
    for (let x = 0; x < w; x++) {
      if (data[rowOff + x * 4 + 3] > 0) n++;
    }
    rowCounts[y] = n;
  }
  const rows = bodyExtent(rowCounts);
  if (!rows) return null;

  // Columns are counted over the rows that survived, so a speckle already
  // ruled out vertically does not get a second vote horizontally.
  const colCounts = new Int32Array(w);
  for (let y = rows.start; y <= rows.end; y++) {
    const rowOff = y * w * 4;
    for (let x = 0; x < w; x++) {
      if (data[rowOff + x * 4 + 3] > 0) colCounts[x]++;
    }
  }
  const cols = bodyExtent(colCounts);
  if (!cols) return null;

  const bbox = {
    x: cols.start,
    y: rows.start,
    w: cols.end - cols.start + 1,
    h: rows.end - rows.start + 1,
  };
  // Covers (essentially) the whole canvas: nothing to crop to.
  if (bbox.w >= w - 1 && bbox.h >= h - 1) return null;
  return bbox;
}

/**
 * Same, read off a 2D context. Returns null when the canvas is tainted or
 * otherwise unreadable — the caller then keeps the frame it had.
 */
export function computeAlphaBbox(ctx, w, h) {
  if (w <= 0 || h <= 0) return null;
  let imgData;
  try {
    imgData = ctx.getImageData(0, 0, w, h);
  } catch {
    return null;
  }
  return bboxFromAlpha(imgData.data, w, h);
}
