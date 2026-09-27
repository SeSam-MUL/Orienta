/**
 * Open the map export WITH a scale bar when the picture has a scale.
 *
 * The phase map owns its bar as an annotation, and the export dialog's
 * scale-bar controls edit that annotation rather than adding one of their
 * own ("one bar, two places"). So a bar that should be on by default has to
 * be seeded here, in the owner — the dialog's own default is bypassed on this
 * path. The M5 tester's export opened with no bar because nobody had placed
 * one on the map.
 *
 * Pure: returns the same array when nothing is to be added.
 */
import { niceScalebar } from '../../common/imageExport';
import { applyPatch, withAdded } from './exportAnnotEdits';

/**
 * @param annotations  the dialog's copy of the map annotations
 * @param umPerPx      micrometres per OUTPUT pixel of the composed picture
 * @param mapWidthPx   width of the map inside that picture, in output pixels
 */
export function seedScalebar(annotations, { umPerPx, mapWidthPx } = {}) {
  const list = annotations || [];
  if (list.some((a) => a?.type === 'scalebar')) return list;
  const upp = Number(umPerPx);
  // A 1/2/5 length covering about a quarter of the map, like the dialog's own
  // bar would; the DEFAULTS' 5 µm is right for one scan and absurd for another.
  // No length computable (no scale, or no map width) -> no bar: a bar whose
  // length is a guess is worse than none.
  const bar = upp > 0 ? niceScalebar(upp, mapWidthPx, 0.25) : null;
  if (!bar) return list;
  const { annotations: next, id } = withAdded(list, 'scalebar');
  if (!id) return list;
  return applyPatch(next, id, { props: { lengthUm: bar.lengthUnits } });
}
