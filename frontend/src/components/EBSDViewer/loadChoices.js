/**
 * Where a picked EBSD path leads: straight to the ordinary load, or to a dialog first.
 *
 * `GET /api/ebsd/inspect` answers for every HDF5 EBSD file, not only for EDAX APEX metadata files:
 * `kind: 'apex'` (its measurements, each paired with a pattern file), `kind: 'choices'` (a file that
 * offers more than one measurement x pattern source), `kind: 'unreadable'` (an HDF5 name that cannot
 * be opened) or `kind: 'plain'` (one variant: load it as ever). The backend decides what is a
 * choice; nothing here counts variants itself.
 */

/** A picked file whose backend answer can be a dialog: the HDF5 names of EBSD files. */
export const isInspectable = (path) => /\.(h5|hdf5|h5oina|h5ebsd)$/i.test(String(path || '').trim());

/**
 * What a picked path leads to. Only an HDF5 name is asked about (`inspect`); the answer decides:
 * an APEX metadata file or a file with several variants opens a dialog (`dialog`; `kind` says which),
 * a file that cannot be opened as HDF5 says why first (`unreadable`, and the user may insist on the
 * ordinary load), anything else loads exactly as before (`load`). A name that cannot be a choice
 * is not asked about, and a question that fails for any reason loads the ordinary way, which then
 * reports its own error.
 *
 * @param {string} path
 * @param {(path: string) => Promise<{data?: object}>} inspect `ebsdApi.inspect`
 * @returns {Promise<{action: 'load'|'dialog'|'unreadable', kind?: 'apex'|'choices', inspection?: object,
 *                    dialog?: object, problem?: string, detail?: string}>}
 *   `dialog` is `{h5Path, inspection}` for `apex` and `{path, inspection}` for `choices`.
 */
export async function routeLoad(path, inspect) {
  if (!isInspectable(path)) return { action: 'load' };
  const picked = String(path).trim();
  try {
    const answer = (await inspect(picked))?.data || {};
    if (answer.kind === 'apex') {
      return { action: 'dialog', kind: 'apex', inspection: answer, dialog: { h5Path: picked, inspection: answer } };
    }
    if (answer.kind === 'choices' && Array.isArray(answer.measurements) && answer.measurements.length > 0) {
      return { action: 'dialog', kind: 'choices', inspection: answer, dialog: { path: picked, inspection: answer } };
    }
    if (answer.kind === 'unreadable') {
      return { action: 'unreadable', problem: answer.problem || 'apexFileUnreadable', detail: answer.problem_detail || '' };
    }
  } catch { /* not inspectable: the ordinary load says what is wrong */ }
  return { action: 'load' };
}

// ---------------------------------------------------------------------------------------------
// The generic load dialog: the rows of a `choices` answer, the variant preselected, the job a
// choice becomes, and the last choice remembered per file. A variant is one measurement with one
// pattern source; a file is open as one variant at a time.
// ---------------------------------------------------------------------------------------------

/** The width x height of one pattern from a `pattern_shape` [rows, cols], or null. */
const sizeFromShape = (shape) => {
  if (!Array.isArray(shape) || shape.length !== 2) return null;
  const [rows, cols] = shape.map(Number);
  return rows > 0 && cols > 0 ? { width: cols, height: rows } : null;
};

/**
 * The rows of the dialog: one per measurement of a `choices` answer, each with the sources it offers.
 * @returns {{name: string, label: string, nRows: number, nCols: number, sources: object[]}[]}
 */
export function choiceRows(inspection) {
  const list = Array.isArray(inspection?.measurements) ? inspection.measurements : [];
  return list.map((m) => ({
    name: String(m.name),
    label: String(m.label ?? m.name),
    nRows: Number(m.n_rows),
    nCols: Number(m.n_cols),
    sources: (Array.isArray(m.sources) ? m.sources : []).map((s) => ({
      key: s.key,
      nPatterns: Number(s.n_patterns),
      size: sizeFromShape(s.pattern_shape),
      dtype: s.dtype || '',
      hasBackground: s.has_background === true,
      loadable: s.loadable === true,
      problem: s.problem || null,
    })),
  }));
}

/**
 * What the bit-depth line of a source says. It follows the data type the backend measured, never the
 * name of the source: a 16-bit integer source is "16-bit counts" (kind `unprocessed`, the text of that
 * key); 8-bit patterns of the source called `processed` are "8-bit, background corrected by the
 * acquisition software" (kind `processed`: the one case where the file's own naming is the evidence
 * of a correction); any other depth is named as a plain number of bits; a source without a usable
 * data type says nothing.
 * @returns {{kind: 'processed'|'unprocessed'|'none'} | {kind: 'bits', bits: number}}
 */
export function sourceBits(source) {
  const dtype = String(source?.dtype || '');
  if (dtype === 'int16' || dtype === 'uint16') return { kind: 'unprocessed' };
  if (dtype === 'uint8' && source?.key === 'processed') return { kind: 'processed' };
  const hit = /^u?int(8|16|32|64)$|^float(16|32|64)$/.exec(dtype);
  return hit ? { kind: 'bits', bits: Number(hit[1] || hit[2]) } : { kind: 'none' };
}

const findSource = (inspection, variant) => {
  if (!variant) return null;
  const m = (inspection?.measurements || []).find((x) => String(x.name) === String(variant.measurement));
  return (m?.sources || []).find((s) => s.key === variant.pattern_source) || null;
};

const isLoadable = (inspection, variant) => findSource(inspection, variant)?.loadable === true;

/**
 * The variant preselected when the dialog opens: the last one chosen for this file when the file
 * still offers it and it can be opened, else the one a cropped file says it was cut from, else the
 * backend's default (the first measurement, processed patterns: what a plain load takes). `remembered`
 * says it came from memory. null when no variant can be opened.
 * @returns {{measurement: string, pattern_source: string, remembered: boolean}|null}
 */
export function defaultChoice(inspection, remembered) {
  const as = (v, memory) => ({ measurement: String(v.measurement), pattern_source: v.pattern_source, remembered: memory });
  if (isLoadable(inspection, remembered)) return as(remembered, true);
  for (const v of [inspection?.suggested, inspection?.default_variant]) {
    if (isLoadable(inspection, v)) return as(v, false);
  }
  for (const m of inspection?.measurements || []) {
    const s = (m.sources || []).find((x) => x.loadable === true);
    if (s) return as({ measurement: m.name, pattern_source: s.key }, false);
  }
  return null;
}

/**
 * The loads a choice stands for: one `POST /api/ebsd/load` carrying the variant. Nothing for a
 * variant the file does not offer or cannot open.
 * @returns {{path: string, label: string, variant: {measurement: string, pattern_source: string}}[]}
 */
export function choiceJobs(path, inspection, choice) {
  if (!choice || !isLoadable(inspection, choice)) return [];
  const row = choiceRows(inspection).find((r) => r.name === String(choice.measurement));
  return [{
    path,
    label: row ? row.label : String(choice.measurement),
    variant: { measurement: String(choice.measurement), pattern_source: choice.pattern_source },
  }];
}

/**
 * A queue that runs at most `max` tasks at a time, in the order they were asked for. `run(task, signal)`
 * answers with what the task answers. A task whose `signal` is aborted before its turn never starts
 * (its answer is an AbortError); a running one that is aborted gives its place up at once, whether or
 * not its own answer ever comes. The dialog asks for one small picture per source this way, so a file
 * with many measurements does not put dozens of reads on the backend at the same moment.
 * @param {number} max
 * @returns {(task: () => Promise<any>, signal?: AbortSignal) => Promise<any>}
 */
export function limitConcurrent(max) {
  let active = 0;
  const waiting = [];
  const aborted = () => Object.assign(new Error('aborted'), { name: 'AbortError' });
  const pump = () => {
    while (active < max && waiting.length > 0) {
      const job = waiting.shift();
      if (job.signal?.aborted) { job.reject(aborted()); continue; }
      active += 1;
      let freed = false;
      const free = () => {
        if (freed) return;
        freed = true;
        active -= 1;
        job.signal?.removeEventListener('abort', free);
        // Not at once: whoever aborts one task often aborts the ones behind it in the same breath,
        // and those must find themselves aborted when their turn comes.
        Promise.resolve().then(pump);
      };
      job.signal?.addEventListener('abort', free);
      Promise.resolve().then(job.task).then(job.resolve, job.reject).finally(free);
    }
  };
  return (task, signal) => new Promise((resolve, reject) => {
    waiting.push({ task, signal, resolve, reject });
    pump();
  });
}

/** One spelling for the paths of one file (slashes, case), as the key of what is remembered. */
export const canonicalPath = (p) => String(p || '').trim().replace(/\\/g, '/').toLowerCase();

export const variantStorageKey = (path) => `orienta.loadVariant.${canonicalPath(path)}`;

/** The variant last opened for this file, or null (never chosen, unreadable, not a variant, no storage). */
export function readRememberedVariant(path) {
  try {
    const raw = localStorage.getItem(variantStorageKey(path));
    if (!raw) return null;
    const v = JSON.parse(raw);
    return v && typeof v.measurement === 'string' && typeof v.pattern_source === 'string'
      ? { measurement: v.measurement, pattern_source: v.pattern_source } : null;
  } catch {
    return null;
  }
}

/** Remember the variant a load of this file succeeded with. Never throws; false when it could not. */
export function rememberVariant(path, variant) {
  try {
    localStorage.setItem(variantStorageKey(path), JSON.stringify({
      measurement: String(variant.measurement), pattern_source: variant.pattern_source }));
    return true;
  } catch {
    return false;
  }
}
