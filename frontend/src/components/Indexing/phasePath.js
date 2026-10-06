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
 * The library listing plus the phase files the user added by path.
 *
 * A listing replaces the page's file list every time it is fetched (method
 * change, retry), so the user's own files must be merged back in each time or
 * a refresh would silently drop a phase they had just added. Returns `files`
 * itself when there is nothing to add.
 */
export function mergeUserAdded(files, extra) {
  const add = (extra || []).filter((e) => !findByPath(files, e.path));
  return add.length ? [...files, ...add] : files;
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
 * @returns {Promise<{ok: true, name: string, already?: true}
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
  const inLibrary = findByPath(now.discoveredFiles, record.path);
  const file = inLibrary || record;
  const name = file.display_label || file.formula || file.filename;
  if (now.phaseFiles.some((p) => samePath(p, file.path))) {
    return { ok: true, already: true, name };
  }
  // Another FILE with the same name is already selected. A phase is identified by
  // its file name (a reflector selection is stored under it as well), so the two
  // would be one phase to the app: say so instead of adding it.
  const stem = phaseStem(file.path).toLowerCase();
  const clash = now.phaseFiles.find((p) => phaseStem(p).toLowerCase() === stem);
  if (clash) {
    return {
      ok: false,
      error: { code: 'phase_same_name', message: '',
               params: { name: phaseStem(clash), loaded_path: clash, path: file.path } },
    };
  }
  if (!inLibrary) remember(record);
  now.select(file);
  return { ok: true, name };
}
