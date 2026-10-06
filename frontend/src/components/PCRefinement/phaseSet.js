/**
 * Pure helpers for PC Refinement with SEVERAL phases (duplex steel: austenite
 * and ferrite). Kept out of the 2800-line page so they can be tested.
 */
import { phaseStem, samePath, sameFile, asRecord } from '../Indexing/phasePath';

/** Mirrors `PCController.MAX_PHASES` on the backend (which is what enforces it). */
export const MAX_PC_PHASES = 8;

/** The header text for the loaded phases: one name, or "A + B". */
export function phaseLabelFor(names, fallback) {
  if (!names || names.length === 0) return fallback;
  return names.join(' + ');
}

/**
 * The loaded phase that IS the given file, if there is one.
 *
 * ``target`` is a record ``{path, real_path?}`` or a bare path (looked up in
 * ``files`` for its resolved location when given). Phases are identified by FILE,
 * not by name: two different files called Al.cif are two files, and one file
 * reached through a link and directly is one. A phase whose file is not known
 * (loaded through the single-phase path before the page knew files) falls back
 * to the file's name.
 *
 * @param {{name: string, path?: string, real_path?: string}[]} phases
 */
export function loadedPhaseFor(phases, target, files = []) {
  const rec = typeof target === 'string' ? asRecord(target, files) : target;
  const stem = phaseStem(rec.path).toLowerCase();
  return (phases || []).find((p) => (p.path
    ? sameFile(p, rec)
    : String(p.name).toLowerCase() === stem));
}

/**
 * A loaded phase with the same NAME as the given file but another file.
 * The backend refuses it (a name is how a phase, and a reflector selection, is
 * kept), so the page can say so before asking.
 */
export function sameNameOtherFile(phases, target, files = []) {
  const rec = typeof target === 'string' ? asRecord(target, files) : target;
  const stem = phaseStem(rec.path).toLowerCase();
  const mine = loadedPhaseFor(phases, rec);
  return (phases || []).find((p) => p !== mine && String(p.name).toLowerCase() === stem);
}

/**
 * Paths of the picker entries that are currently loaded: the library entries and
 * the typed-in paths (`added`) that are the very file of a loaded phase. An entry
 * that only shares a name with a loaded phase is NOT loaded.
 */
export function loadedPaths(files, phases, added = []) {
  const out = [];
  const entries = [...(files || []), ...added.map((p) => ({ path: p }))];
  for (const e of entries) {
    if (loadedPhaseFor(phases, e) && !out.some((q) => samePath(q, e.path))) out.push(e.path);
  }
  return out;
}

/**
 * What to do so that exactly `wanted` (paths) are loaded: `{add: paths to add,
 * remove: names to remove}`. The picker's All / None buttons hand over the whole
 * desired set at once.
 */
export function planPhaseSync(phases, wanted, files = []) {
  const list = phases || [];
  const add = (wanted || []).filter((p) => !loadedPhaseFor(list, p, files));
  const remove = list
    .filter((ph) => !(wanted || []).some((p) => loadedPhaseFor([ph], p, files)))
    .map((ph) => ph.name);
  return { add, remove };
}

/**
 * `[{name, count}]` from the per-pattern phases of a refine, in order of first
 * appearance; a pattern no phase fitted counts under `noneName`.
 */
export function summarisePatternPhases(patternPhases, noneName) {
  const order = [];
  const counts = new Map();
  for (const e of patternPhases || []) {
    const name = e?.phase_name ?? noneName;
    if (!counts.has(name)) { counts.set(name, 0); order.push(name); }
    counts.set(name, counts.get(name) + 1);
  }
  return order.map((name) => ({ name, count: counts.get(name) }));
}

/**
 * The phase whose master pattern the forward-simulation preview should use.
 *
 * One loaded phase: its name, exactly as before. Several: the phase the
 * selected pattern was indexed as (a pattern of ferrite compared against the
 * austenite master would read as a bad PC), else the first loaded phase.
 */
export function previewPhaseName({ phaseNames, phaseLabel, patterns, selectedIdx }) {
  if (!phaseNames || phaseNames.length <= 1) return phaseLabel;
  const own = selectedIdx != null ? patterns?.[selectedIdx]?.phase : null;
  return own || phaseNames[0];
}

/** Runs async jobs one after another: a click on a second phase waits for the first. */
export function createSerialQueue() {
  let tail = Promise.resolve();
  return (job) => {
    const run = tail.then(job, job);
    tail = run.catch(() => {});
    return run;
  };
}

/**
 * A space group for the phase list: "Fm-3m" out of orix's long description
 * ("SpaceGroup #225 (Fm-3m, Cubic). Symmetry matrices: 192, ..."). A string
 * that does not look like that is returned as it is.
 */
export function shortSpaceGroup(text) {
  const m = /\(([^,)]+)[,)]/.exec(String(text || ''));
  return m ? m[1].trim() : String(text || '');
}
