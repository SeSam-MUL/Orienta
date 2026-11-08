/**
 * The window sizes the backend's frame averaging accepts on a hexagonal scan.
 *
 * The neighbourhood of a hexagonal point grows by whole rings (3 = the point and its 6 neighbours,
 * 5 = the point and the 18 within two steps (19 points), ...), so an even size has no ring of its own, and the backend
 * stops at 49. It answers anything else with a 400. A square scan takes 1 to 50 as before.
 */
export const HEX_MAX_WINDOW = 49;

/** True when the backend takes this window size on a hexagonal scan. */
export const hexWindowOk = (size) => (
  Number.isInteger(size) && size >= 1 && size <= HEX_MAX_WINDOW && size % 2 === 1
);
