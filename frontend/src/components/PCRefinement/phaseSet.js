/**
 * Pure helpers for PC Refinement with SEVERAL phases (duplex steel: austenite
 * and ferrite). Kept out of the 2800-line page so they can be tested.
 */
import { phaseStem } from '../Indexing/phasePath';

/** Mirrors `PCController.MAX_PHASES` on the backend (which is what enforces it). */
export const MAX_PC_PHASES = 8;

/** The header text for the loaded phases: one name, or "A + B". */
export function phaseLabelFor(names, fallback) {
  if (!names || names.length === 0) return fallback;
  return names.join(' + ');
}

/**
 * Paths of the picker entries that are currently loaded.
 *
 * The backend names a phase by its file's stem, so a library entry counts as
 * loaded when its stem is one of the loaded names. `added` holds the paths the
 * user typed in (they are not in the library listing).
 */
export function loadedPaths(files, names, added = []) {
  const loaded = new Set(names || []);
  const out = [];
  for (const f of files || []) {
    if (loaded.has(phaseStem(f.path)) && !out.includes(f.path)) out.push(f.path);
  }
  for (const p of added) {
    if (loaded.has(phaseStem(p)) && !out.includes(p)) out.push(p);
  }
  return out;
}

/**
 * What to do so that exactly `wanted` (paths) are loaded, given the loaded
 * names: `{add: paths to add, remove: names to remove}`. The picker's All /
 * None buttons hand over the whole desired set at once.
 */
export function planPhaseSync(loadedNames, wanted) {
  const have = new Set(loadedNames || []);
  const wantStems = new Map((wanted || []).map((p) => [phaseStem(p), p]));
  return {
    add: [...wantStems].filter(([stem]) => !have.has(stem)).map(([, p]) => p),
    remove: [...have].filter((n) => !wantStems.has(n)),
  };
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
