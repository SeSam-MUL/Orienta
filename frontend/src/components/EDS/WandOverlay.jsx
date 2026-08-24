import { useEffect, useRef } from 'react';

/**
 * Preview layer for the seeded selection.
 *
 * Draws on a <canvas>, not the SVG the rectangle/polygon use: an SVG cannot
 * paint 485 000 arbitrary pixels.
 *
 * It DIMS everything outside the selection rather than tinting the inside.
 * The phase palette is a golden-ratio walk through HSV, so any fixed
 * highlight colour eventually lands on a phase — a cyan selection rectangle
 * was once invisible on a teal phase, which is how that lesson was learned.
 * Dimming is palette-agnostic, it leaves the selected phase's own colour
 * readable (the user is judging chemistry, not decoration), and it makes the
 * selection the only fully saturated thing on screen. A white boundary over
 * a dark casing marks the edge.
 */
export default function WandOverlay({ mask, shape, seed }) {
  const ref = useRef(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv || !mask || !shape) return;
    const { n_rows: R, n_cols: C } = shape;
    cv.width = C;
    cv.height = R;
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    const img = ctx.createImageData(C, R);
    const d = img.data;

    for (let i = 0, o = 0; i < R * C; i++, o += 4) {
      if (mask[i]) {
        d[o] = 0; d[o + 1] = 0; d[o + 2] = 0; d[o + 3] = 0;   // untouched
      } else {
        d[o] = 8; d[o + 1] = 10; d[o + 2] = 16; d[o + 3] = 150; // dimmed
      }
    }
    // Boundary: a selected pixel with an unselected 4-neighbour.
    for (let r = 0; r < R; r++) {
      for (let c = 0; c < C; c++) {
        const i = r * C + c;
        if (!mask[i]) continue;
        const edge =
          r === 0 || r === R - 1 || c === 0 || c === C - 1 ||
          !mask[i - 1] || !mask[i + 1] || !mask[i - C] || !mask[i + C];
        if (edge) {
          const o = i * 4;
          d[o] = 255; d[o + 1] = 255; d[o + 2] = 255; d[o + 3] = 235;
        }
      }
    }
    ctx.putImageData(img, 0, 0);

    // Seed marker, so the user can see what the selection grew from.
    if (seed) {
      ctx.fillStyle = 'rgba(0,0,0,0.85)';
      ctx.fillRect(seed.col - 1, seed.row - 1, 3, 3);
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(seed.col, seed.row, 1, 1);
    }
  }, [mask, shape, seed]);

  if (!mask || !shape) return null;
  return (
    <canvas
      ref={ref}
      style={{
        position: 'absolute', inset: 0, width: '100%', height: '100%',
        objectFit: 'contain', imageRendering: 'pixelated',
        pointerEvents: 'none',
      }}
    />
  );
}
