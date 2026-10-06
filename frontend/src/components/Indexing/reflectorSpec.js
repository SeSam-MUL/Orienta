/**
 * The logic of the reflector-family table, apart from the table itself.
 *
 * Two pages show the same table for the same per-phase choice (the Indexing page
 * names a phase by its CIF path, PC Refinement by the phase it has loaded), so
 * the component is given an `api` of four calls and these pure helpers do the
 * thinking: which families a change would leave ticked, what an error says, how
 * a size is written.
 */
import { indexApi, pcApi } from '../../services/api';
import { pathErrorFrom, phaseStem } from './phasePath';

/** Registry key of a phase: the lower-case CIF stem (the backend's own rule). */
export function specKey(pathOrName) {
  return phaseStem(pathOrName).trim().toLowerCase();
}

/** The four calls for a phase named by its CIF path (Indexing page). */
export function pathAdapter(cifPath, nBands = 12) {
  return {
    key: specKey(cifPath),
    load: (extended = false) => indexApi.houghReflectors(cifPath, extended),
    change: (spec, extended = false) => indexApi.setHoughReflectors(cifPath, spec, extended),
    validate: (hkl, spec) => indexApi.validateHoughReflector(cifPath, hkl, spec),
    cost: () => indexApi.houghReflectorsCost(cifPath, nBands),
    stored: () => indexApi.houghReflectorSpecs(),
  };
}

/** The same four for a phase that is loaded on the PC Refinement page. */
export function loadedPhaseAdapter(phaseName, nBands = 12) {
  return {
    key: specKey(phaseName),
    load: (extended = false) => pcApi.phaseReflectors(phaseName, extended),
    change: (spec, extended = false) => pcApi.setPhaseReflectors(phaseName, spec, extended),
    validate: (hkl, spec) => pcApi.validatePhaseReflector(phaseName, hkl, spec),
    cost: () => pcApi.phaseReflectorCost(phaseName, nBands),
    stored: () => indexApi.houghReflectorSpecs(),
  };
}

/** `[h, k, l]` of every family that is ticked now. */
export function tickedFamilies(table) {
  return (table?.families || []).filter((f) => f.selected).map((f) => f.hkl);
}

/**
 * The spec that results from clicking one family's box.
 *
 * Any hand change turns the choice into an explicit list of families (the
 * default list and a rule are both written out as the ticked families), so the
 * list the user sees is the list that is stored.
 */
export function specAfterToggle(table, hkl) {
  const key = hkl.join(',');
  const current = tickedFamilies(table);
  const has = current.some((h) => h.join(',') === key);
  const families = has ? current.filter((h) => h.join(',') !== key) : [...current, hkl];
  return { mode: 'custom', families };
}

/** The ticked families plus one more (a validated hand-typed family). */
export function specAfterAdd(table, hkl) {
  const current = tickedFamilies(table);
  if (current.some((h) => h.join(',') === hkl.join(','))) return { mode: 'custom', families: current };
  return { mode: 'custom', families: [...current, hkl] };
}

/** Spec the table is working on, for the server's duplicate check. */
export function currentSpec(table) {
  if (!table) return null;
  return { mode: 'custom', families: tickedFamilies(table) };
}

/** Text for the translated error of a failed call. */
export function reflectorError(t, err) {
  const e = err && err.code ? err : pathErrorFrom(err);
  const params = { ...(e.params || {}) };
  if (Array.isArray(params.hkl)) params.hkl = `{${params.hkl.join(' ')}}`;
  if (typeof params.d === 'number') params.d = params.d.toFixed(2);
  // A plain-text error (a string detail, a network failure) says its own thing.
  if (e.code === 'generic' && e.message) return e.message;
  const key = `reflectorFamilies.errors.${e.code}`;
  const known = t(key, { ...params, defaultValue: '' });
  if (known) return known;
  return e.message || t('reflectorFamilies.errors.generic');
}

const GiB = 1024 ** 3;

/** "1.45 GiB" / "12 MiB" / null for "negligible". */
export function formatSize(bytes) {
  if (!bytes) return null;
  if (bytes >= GiB / 10) return `${(bytes / GiB).toFixed(2)} GiB`;
  return `${Math.max(1, Math.round(bytes / 1024 ** 2))} MiB`;
}

/** Which of the three headline states a table is in. */
export function summaryKey(mode) {
  if (mode === 'custom') return 'summaryCustom';
  if (mode === 'auto') return 'summaryAuto';
  return 'summaryDefault';
}
