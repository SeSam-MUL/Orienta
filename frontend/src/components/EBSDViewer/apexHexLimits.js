/**
 * What a hexagonal scan cannot do yet, from the backend's own list.
 *
 * `GET /api/scan-geometry/hex-limits` is built from the table that enforces the limits, so the
 * dialog's hexagonal line and its Details cannot say something the application does not do. The
 * dialog never writes a list of its own: it takes the keys and categories the backend returns and
 * looks up one short name per key, per category, per affected kind and per cleanup parameter in
 * `hexLimits.*` of the `ebsdviewer` namespace (a test holds the locales against the backend's list).
 */

/**
 * The limits grouped for the dialog.
 *
 * Several items can share a key (the three phase-map cleanup routes are one limit to the user):
 * they are merged, their `affects` and `parameters` united. Categories come in the backend's order
 * and only the ones that have a limit are returned.
 *
 * @param {object} data the answer of GET /hex-limits
 * @returns {{categories: {category: string, items: {key: string, partial: boolean,
 *            affects: string[], parameters: string[]}[]}[],
 *            works: string[]}|null} null when the answer is not usable
 */
export function groupHexLimits(data) {
  if (!data || typeof data !== 'object' || !data.groups || !Array.isArray(data.categories)) return null;
  const byKey = new Map();
  for (const items of Object.values(data.groups)) {
    for (const item of Array.isArray(items) ? items : []) {
      if (!item || !item.key) continue;
      const have = byKey.get(item.key) || {
        key: item.key, category: item.category, partial: false, affects: [], parameters: [],
      };
      have.partial = have.partial || item.partial === true;
      for (const a of item.affects || []) if (!have.affects.includes(a)) have.affects.push(a);
      for (const p of item.parameters || []) if (!have.parameters.includes(p)) have.parameters.push(p);
      byKey.set(item.key, have);
    }
  }
  const categories = data.categories
    .map((category) => ({ category, items: [...byKey.values()].filter((i) => i.category === category) }))
    .filter((c) => c.items.length > 0);
  return {
    categories,
    works: (Array.isArray(data.not_limited) ? data.not_limited : []).map((n) => n.key).filter(Boolean),
  };
}
