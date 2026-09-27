/**
 * Pure logic behind "Sammlung übernehmen" (Indexing's one-click adopt) and
 * the resolved-paths fetch that narrows Spherical/Dictionary's phase
 * dropdown. Extracted out of IndexingPage.jsx — which is too large to mount
 * in a test — so this can be unit-tested directly instead of through a
 * source-text regex, the same treatment `collectionFilter.js` already gets.
 */
import { keyForPath } from './collectionFilter';

/**
 * Maps a `GET /phase-collections/resolve` response onto the two arrays
 * IndexingPage keeps index-aligned (`phases`/`phaseFiles`), preferring the
 * already-discovered file object (it carries display_label/formula/
 * crystal_system) and falling back to a derived object when the discovery
 * scan hasn't (yet) seen a path the library index resolved. `missing` is
 * carried through unchanged — nothing here may drop it.
 *
 * `files`/`phaseFiles` stay in `resolveResponse.paths` order by construction
 * (both are produced by the same single `.map` over that array), so a caller
 * that does `setPhases(plan.files); setPhaseFiles(plan.phaseFiles);` cannot
 * end up with the two arrays out of step.
 */
export function planCollectionAdoption(resolveResponse, discoveredFiles) {
  const paths = resolveResponse?.paths || [];
  const missing = resolveResponse?.missing || [];
  const byPath = new Map((discoveredFiles || []).map((f) => [f.path, f]));
  const files = paths.map((p) => byPath.get(p)
    || { path: p, filename: String(p).split(/[\\/]/).pop(), formula: keyForPath(p) });
  return { files, phaseFiles: files.map((f) => f.path), missing };
}

/**
 * `missing[].reason` -> the translation key for its human label, matching
 * `GET /resolve` exactly (`backend/api/routes/phase_collections.py#resolve`):
 * `no_cif` for Hough, `no_sht` for Spherical, `no_master` for Dictionary,
 * `not_in_library` when the key resolves to no library entry at all. These
 * are four different problems with four different fixes — "run a
 * simulation" is a different day's work from "this phase is not in your
 * library" — which is the whole reason `formatMissingLogMessage` groups by
 * this instead of only counting.
 */
const REASON_KEYS = {
  no_cif: 'collections:counts.reasonNoCif',
  no_sht: 'collections:counts.reasonNoSht',
  no_master: 'collections:counts.reasonNoMaster',
  not_in_library: 'collections:counts.reasonNotInLibrary',
};

/**
 * The log line for phases the collection named but could not resolve for
 * this method. `null` when there is nothing to report — the caller must not
 * log an empty line. `t` is whatever translation function is in scope
 * (namespaced, as IndexingPage's `t('collections:counts.notUsableHere', …)`
 * already is); this function does not import i18n itself so it can be
 * tested with a plain spy.
 *
 * Grouped by reason, in first-seen order — an earlier version of this
 * function joined every key into one flat list regardless of WHY each one
 * was missing, which said nothing a user could act on: "6 phases could not
 * be used: A, B, C, D, E, F" does not tell you that three of those need a
 * simulated master and three are not in your library at all. A reason this
 * map does not recognise (or a member with no `reason` field) falls back to
 * `not_in_library`'s label rather than being dropped or crashing on an
 * unmatched `t()` key.
 */
export function formatMissingLogMessage(t, missing) {
  if (!missing || missing.length === 0) return null;
  const prefix = t('collections:counts.notUsableHere', { count: missing.length });
  const order = [];
  const byReason = new Map();
  for (const m of missing) {
    const reason = m.reason || 'not_in_library';
    if (!byReason.has(reason)) { byReason.set(reason, []); order.push(reason); }
    byReason.get(reason).push(m.key);
  }
  const groups = order.map((reason) => {
    const label = t(REASON_KEYS[reason] || REASON_KEYS.not_in_library);
    return `${label}: ${byReason.get(reason).join(', ')}`;
  });
  return prefix + ': ' + groups.join(' · ');
}

/**
 * The allow-list for Spherical/Dictionary's phase dropdown. `null` means "no
 * filter" (no active collection, or Hough — which filters by
 * `collectionKeys`/stem instead, never by this). A `resolveFn` that rejects
 * must still produce an empty `Set`, never `null`: `null` reads downstream
 * as "no filter", which would silently widen the picker back to the whole
 * library exactly when the collection couldn't be resolved.
 */
export async function resolveAllowedPaths(activeName, method, resolveFn) {
  if (!activeName || method === 'hough') return null;
  try {
    const { data } = await resolveFn(activeName, method);
    return new Set(data.paths);
  } catch {
    return new Set();
  }
}
