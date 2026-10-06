/**
 * Helpers for adding a phase by file path (Indexing page and PC Refinement).
 *
 * Pure, so they can be tested without mounting a ~4000-line page.
 */

/** A path in a form two spellings of the same file agree on. */
function normKey(p) {
  const s = String(p || '').replace(/\\/g, '/');
  // Windows drive paths are case-insensitive; POSIX ones are not.
  return /^[a-zA-Z]:\//.test(s) ? s.toLowerCase() : s;
}

/** True when both are non-empty and name the same file. */
export function samePath(a, b) {
  if (!a || !b) return false;
  return normKey(a) === normKey(b);
}

/**
 * A pasted path without the quotes "Copy as path" (Windows), a shell, a chat or a
 * word processor put around it: straight, typographic (English, German) or
 * guillemets. Mirrors `phase_path._clean` on the server, which the Indexing page
 * relies on; the PC Refinement page's endpoint reads the path as given.
 */
const QUOTE_PAIRS = {
  '"': '"', "'": "'",
  '“': '”“', '‘': '’‘',
  '„': '“”', '‚': '‘’',
  '«': '»', '»': '«',
};
export function cleanPastedPath(raw) {
  let text = String(raw ?? '').trim();
  while (text.length >= 2 && (QUOTE_PAIRS[text[0]] || '').includes(text[text.length - 1])) {
    text = text.slice(1, -1).trim();
  }
  return text;
}

/** The entry of `files` ({path}) that is the file at `path`, if any. */
export function findByPath(files, path) {
  return (files || []).find((f) => samePath(f.path, path));
}

/**
 * Do two phase files name the same file on disk?
 *
 * Each is a record `{path, real_path?}` or a bare path. `real_path` is the
 * resolved location the server reports: the library is often reached through a
 * link, so the listing's spelling and a pasted or stored path can differ for one
 * file. When both sides have it, it decides; otherwise the spelling does.
 */
export function sameFile(a, b) {
  const ra = typeof a === 'string' ? { path: a } : (a || {});
  const rb = typeof b === 'string' ? { path: b } : (b || {});
  if (ra.real_path && rb.real_path) return normKey(ra.real_path) === normKey(rb.real_path);
  return samePath(ra.path, rb.path);
}

/** The entry of `files` that is the same file as `file` (a record or a path). */
export function findByFile(files, file) {
  return (files || []).find((f) => sameFile(f, file));
}

/** A path as a record, with its resolved location when `files` knows it. */
export function asRecord(path, files) {
  return findByPath(files, path) || { path };
}

/**
 * Move a selection from the own files that collapsed into library entries onto
 * those entries: `{phaseFiles, phases}` with each collapsed path replaced by the
 * library entry's path / record, nothing selected twice.
 */
export function remapCollapsed(phaseFiles, phases, collapses) {
  if (!collapses.length) return { phaseFiles, phases };
  const to = new Map(collapses.map((c) => [normKey(c.from), c.to]));
  const seen = new Set();
  const outPaths = [];
  for (const p of phaseFiles) {
    const lib = to.get(normKey(p));
    const next = lib ? lib.path : p;
    if (!seen.has(normKey(next))) { seen.add(normKey(next)); outPaths.push(next); }
  }
  const seenRec = new Set();
  const outPhases = [];
  for (const f of phases) {
    const lib = to.get(normKey(f.path));
    const next = lib || f;
    if (!seenRec.has(normKey(next.path))) { seenRec.add(normKey(next.path)); outPhases.push(next); }
  }
  return { phaseFiles: outPaths, phases: outPhases };
}

/**
 * The library listing plus the phase files the user added by path.
 *
 * A listing replaces the page's file list every time it is fetched (method
 * change, retry), so the user's own files must be merged back in each time or
 * a refresh would silently drop a phase they had just added. Returns `files`
 * itself when there is nothing to add.
 */
export function mergeUserAdded(files, extra) {
  const add = (extra || []).filter((e) => !findByFile(files, e));
  return add.length ? [...files, ...add] : files;
}

/**
 * The own files that turned out to be library files: `[{from: path, to: entry}]`.
 *
 * A file added by path before the listing knew it (or before the server reported
 * resolved paths) can be the very file of a library entry under another spelling.
 * `mergeUserAdded` drops it; this says what it collapsed into, so a selection
 * that holds the old path can be moved onto the library entry.
 */
export function collapseUserAdded(files, extra) {
  const out = [];
  for (const e of extra || []) {
    const lib = findByFile(files, e);
    if (lib && !samePath(lib.path, e.path)) out.push({ from: e.path, to: lib });
  }
  return out;
}

/**
 * The `{code, message, params}` of a failed from-path request.
 *
 * The server answers a bad path with a structured detail; anything else (a
 * string detail, a network error) becomes a generic error so the caller always
 * has one shape to word.
 */
export function pathErrorFrom(err) {
  const d = err?.response?.data?.detail;
  if (d && typeof d === 'object' && typeof d.code === 'string') {
    return { code: d.code, message: d.message || '', params: d.params || {} };
  }
  const message = typeof d === 'string' ? d : (err?.message || '');
  return { code: 'generic', message, params: {} };
}

/** A CIF's phase name as the backend gives it: file name without `.cif`. */
export function phaseStem(path) {
  return String(path || '').split(/[\\/]/).pop().replace(/\.cif$/i, '');
}

/**
 * Ask the server about `rawPath`: `{record, entry}` where `record` is what it says
 * about the file and `entry` the library listing entry that is the same file (by
 * resolved path, or the `library_path` the server names), or null.
 *
 * For a page that adds the path itself (PC Refinement) and wants to use the
 * library's entry rather than a second one under another spelling. Any refusal or
 * failure of the check gives `{record: null, entry: null}`: the caller then goes
 * on with the path as typed and reports what its own request says.
 */
export async function resolvePhasePath({ method, rawPath, files, check }) {
  try {
    const record = (await check(method, rawPath)).data?.file;
    if (!record) return { record: null, entry: null };
    const entry = findByPath(files, record.library_path) || findByFile(files, record) || null;
    return { record, entry };
  } catch {
    return { record: null, entry: null };
  }
}

/** The library listing entry that `rawPath` is, or null (see `resolvePhasePath`). */
export async function libraryEntryForPath(args) {
  const { record, entry } = await resolvePhasePath(args);
  return record?.in_library || entry ? entry : null;
}

/**
 * Check a phase path with the server and, if it is good, select that phase.
 *
 * The page's logic for "add by path", kept out of the 4000-line page so it can
 * be tested. It runs after an await, so it asks `latest()` for the page's
 * state NOW rather than relying on the render it was started in.
 *
 * @param {object} o
 * @param {string} o.method  the method the path is for
 * @param {string} o.rawPath
 * @param {(method: string, path: string) => Promise<{data: {file: object}}>} o.check
 * @param {() => {method: string, discoveredFiles: object[], phaseFiles: string[],
 *                select: (file: object) => void}} o.latest
 * @param {(record: object) => void} o.remember  keep a file that is not in the
 *        library, so a later listing does not drop it
 * @returns {Promise<{ok: true, name: string, already?: true, inLibrary?: true}
 *                 | {ok: false, error: {code: string, message: string, params: object}}>}
 */
export async function addPhaseFromPath({ method, rawPath, check, latest, remember }) {
  let record;
  try {
    record = (await check(method, rawPath)).data?.file;
  } catch (err) {
    return { ok: false, error: pathErrorFrom(err) };
  }
  const now = latest();
  if (!record || now.method !== method) {
    // The user switched method while the file was being checked: the file is
    // for the other method's list, so it does not go into this one.
    return { ok: false, error: { code: 'generic', message: '', params: {} } };
  }
  // The same file may already be in the library: use that entry, not a twin.
  // The listing may spell the library folder differently from the pasted path
  // (a link, a `..`), so the server names the listing's entry (`library_path`)
  // when the file is the library's; plain path equality covers the rest.
  const inLibrary = findByPath(now.discoveredFiles, record.library_path)
    || findByFile(now.discoveredFiles, record);
  const file = inLibrary || record;
  const name = file.display_label || file.formula || file.filename;
  // Only a file the listing really has is "in the library": the page can then
  // say it selected the library's entry.
  const flags = inLibrary && record.in_library ? { inLibrary: true } : {};
  if (now.phaseFiles.some((p) => sameFile(asRecord(p, now.discoveredFiles), file))) {
    return { ok: true, already: true, ...flags, name };
  }
  // Another FILE with the same name is already selected. A phase is identified by
  // its file name (a reflector selection is stored under it as well), so the two
  // would be one phase to the app: say so instead of adding it.
  const stem = phaseStem(file.path).toLowerCase();
  const clash = now.phaseFiles.find((p) => phaseStem(p).toLowerCase() === stem
    && !sameFile(asRecord(p, now.discoveredFiles), file));
  if (clash) {
    return {
      ok: false,
      error: { code: 'phase_same_name', message: '',
               params: { name: phaseStem(clash), loaded_path: clash, path: file.path } },
    };
  }
  if (!inLibrary) remember(record);
  now.select(file);
  return { ok: true, ...flags, name };
}
