/**
 * Which kind of "region composition" a number is.
 *
 * The backend says it on every region composition it sends, in
 * `composition_method` (`eds_wand.COMPOSITION_SUMMED_NETS` /
 * `COMPOSITION_MEAN_OF_PIXELS`):
 *
 *  - `"summed_net_counts"` -- spectrum source: the region's net counts summed
 *    over its pixels and quantified ONCE. No pixel is clipped, so a weak
 *    element is not pushed up by the noise of its empty pixels;
 *  - `"mean_of_pixel_at_pct"` -- window source: the mean of the pixels' at%.
 *
 * Returns the `eds` i18n key of the sentence that says which, or null for a
 * value this build does not know (and for `null`, which the backend sends
 * while compositions are left out) -- a label that guessed would be worse
 * than none.
 */
export const COMPOSITION_SUMMED_NETS = 'summed_net_counts';
export const COMPOSITION_MEAN_OF_PIXELS = 'mean_of_pixel_at_pct';

export function compositionMethodKey(method) {
  if (method === COMPOSITION_SUMMED_NETS) return 'composition.summed';
  if (method === COMPOSITION_MEAN_OF_PIXELS) return 'composition.mean';
  return null;
}
