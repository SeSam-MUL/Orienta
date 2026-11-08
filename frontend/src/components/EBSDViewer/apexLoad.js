/**
 * Pure helpers of the "open measurements" dialog for an EDAX APEX metadata file.
 *
 * An APEX `.h5` lists measurements (`OIM Map n` below `Area n`) and holds no patterns; each
 * measurement's patterns are in a `.up1`/`.up2` file. `GET /api/ebsd/inspect` answers with the
 * measurements and, for each, whether it can be opened (`loadable`), why not (`problem`), and the
 * pattern files of the folder that name it; the pattern files that name no measurement come
 * separately (`unmatched_pattern_files`, each with the measurements it `fits`).
 *
 * The backend decides what can be opened. Nothing here re-derives a verdict from the numbers: a
 * row is tickable exactly when the backend says `loadable` (or the user chose a file the backend
 * has described as fitting), and every reason shown is the backend's code. Rows are keyed on
 * `group` (unique within the file), never on `id`: a measurement whose group could not be read
 * has no `id`.
 */

import { apexMeasurementName, isNonDefaultVariant, loadedFileLabel } from '../../services/loadedFileLabel';

/** A picked file that may be an APEX metadata file and so is worth asking the backend about. */
export const isApexCandidate = (path) => /\.(h5|hdf5)$/i.test(String(path || '').trim());

/** `routeLoad` lives in loadChoices.js: it routes every HDF5 EBSD file, APEX being one kind of answer. */
export { routeLoad } from './loadChoices';

/** "Area 1 / OIM Map 1" (the pieces that exist, for a group that could not be read). */
export function measurementTitle(m) {
  if (!m) return '';
  if (m.area && m.name) return `${m.area} / ${m.name}`;
  return m.name || m.area || m.group || '';
}

/** The key a row, its tick and a pattern file chosen for it are kept under. */
export const rowKey = (m) => m.group;

/**
 * The DATE the scan ended, "2026-03-14", or null. APEX stores the end of the scan as a tick count
 * without a time zone, so only the date is shown (a time of day could be an hour or a day off for
 * the reader) and never a start time. Read from the ISO string by position: no `Date` object, which
 * would convert it to the viewer's zone.
 */
export function scanEndDate(m) {
  if (!m || m.measured_at_kind !== 'scan_end') return null;
  const hit = /^(\d{4}-\d{2}-\d{2})(?:[T ]|$)/.exec(String(m.measured_at || ''));
  return hit ? hit[1] : null;
}

const three = (v) => {
  const n = Number(v);
  return Number.isFinite(n) && n > 0 ? Number(n.toPrecision(3)) : null;
};

/**
 * The step along a row and, when the rows are a different distance apart (every hexagonal scan, and
 * a square one whose two steps differ), the spacing between rows. Both in micrometres, three
 * significant digits, as NUMBERS (the dialog writes them in the language of the page: 0,26 in
 * German); `step` is null when the file's step is not a positive number.
 * @returns {{step: number|null, rowStep: number|null}}
 */
export function stepParts(m) {
  const step = three(m?.step_x);
  const rows = three(m?.step_y);
  return { step, rowStep: step && rows && rows !== step ? rows : null };
}

/** The pattern size [width, height] in pixels from a pattern file's `pattern_shape` [rows, cols]. */
export function patternSize(file) {
  const shape = file?.pattern_shape;
  if (!Array.isArray(shape) || shape.length !== 2) return null;
  const [rows, cols] = shape.map(Number);
  return rows > 0 && cols > 0 ? { width: cols, height: rows } : null;
}

/** The size of one pattern, `{width, height}`, from a `pattern_shape` [rows, cols] of a measurement or a file. */
export const sizeOf = (thing) => patternSize({ pattern_shape: thing?.pattern_shape });

/**
 * The two pattern sizes when a pattern file and the measurement both state one and they differ, else
 * null (an unknown size on either side proves nothing). Compared as [rows, cols], so a transposed
 * size is a different size.
 * @returns {{have: {width: number, height: number}, need: {width: number, height: number}}|null}
 */
export function sizeMismatch(m, file) {
  const need = sizeOf(m);
  const have = sizeOf(file);
  if (!need || !have) return null;
  return need.width === have.width && need.height === have.height ? null : { have, need };
}

/** A pattern file that carries no measurement ID: only its pattern count (and size) tie it to a row. */
const countOnly = (f) => !f?.names_measurement;

/**
 * Where a measurement stands, from the backend's answer and what the user chose.
 *
 * `ctx.chosen` maps a row key to a pattern file the user took with "Choose file..." or "Use ...",
 * only after the backend described it as fitting: `{state: 'ok', path, name, entry}`. A choice that
 * is still being checked or was refused is an ATTEMPT (`ctx.attempts`, `{state: 'checking' |
 * 'refused' | 'error', path, name, entry?, message?}`): it never changes the state of the row, so a
 * refused attempt cannot take away a file the row already has. `ctx.unmatched` is the
 * `unmatched_pattern_files` of the folders searched.
 *
 * `state`:
 *   `blocked`  the measurement itself cannot be opened whatever file it gets (`problem` = the code)
 *   `chosen`   the user chose a file and the backend proved it
 *   `ready`    a file that carries the measurement's ID was found and fits (`loadable` is the backend's)
 *   `wrong`    files carry its ID and none fits (`file` is the first, `problem` its code)
 *   `folder`   the folder could not be listed, so no file was looked for
 *   `missing`  no file carries its ID
 * `candidates` are the pattern files of the folders that carry no measurement ID and fit this one by
 * their pattern count; each says in `usedFor` (a row key) when another row already took it. When the
 * measurement's own pattern size is known, a file of another size is no candidate: it is listed in
 * `sizeRejected` instead (a count that fits is not enough when the patterns are of another size).
 * `alternatives` are files of a folder searched later that could replace the file the row has.
 */
export function rowOf(m, ctx = {}) {
  const files = Array.isArray(m.pattern_files) ? m.pattern_files : [];
  const fits = files.filter((f) => f.matches_grid);
  const choice = ctx.chosen?.[rowKey(m)] || null;
  const attempt = ctx.attempts?.[rowKey(m)] || null;
  const taken = new Map();
  for (const [key, c] of Object.entries(ctx.chosen || {})) {
    if (key !== rowKey(m) && c?.state === 'ok' && c.path) taken.set(c.path, key);
  }
  const mark = (f) => ({ ...f, usedFor: taken.get(f.path) ?? null });
  const fitting = m.id
    ? (ctx.unmatched || []).filter((u) => Array.isArray(u.fits) && u.fits.includes(m.id))
    : [];
  const sizeRejected = fitting.filter((u) => sizeMismatch(m, u)).map(mark);
  const candidates = fitting.filter((u) => !sizeMismatch(m, u)).map(mark);
  const alternatives = (Array.isArray(m.alternatives) ? m.alternatives : [])
    .filter((f) => !(countOnly(f) && sizeMismatch(m, f))).map(mark);
  const row = {
    key: rowKey(m), state: 'missing', loadable: false, file: null, extra: 0, others: [],
    problem: null, attempt, candidates, alternatives, sizeRejected,
  };
  if (m.measurement_problem) {
    return { ...row, state: 'blocked', problem: m.measurement_problem, candidates: [], alternatives: [],
             sizeRejected: [], attempt: null };
  }
  if (choice && choice.state === 'ok' && choice.entry?.matches_grid) {
    return { ...row, state: 'chosen', loadable: true, file: choice.entry,
             others: files.filter((f) => f.path !== choice.path) };
  }
  if (fits.length > 0) {
    return { ...row, state: 'ready', loadable: m.loadable === true, file: fits[0],
             extra: fits.length - 1, others: fits.slice(1) };
  }
  if (files.length > 0) {
    return { ...row, state: 'wrong', file: files[0], problem: files[0].problem || m.problem || null,
             others: files.slice(1) };
  }
  if (m.problem === 'apexFolderUnreadable') {
    return { ...row, state: 'folder', problem: m.problem };
  }
  return { ...row, state: 'missing', problem: m.problem || null };
}

/** The ctx `rowOf` reads, from the dialog's state. */
export const rowContext = (chosen, unmatched, attempts) => ({
  chosen: chosen || {}, unmatched: unmatched || [], attempts: attempts || {},
});

/**
 * The listing after "Search another folder...": it only ever ADDS. A row that can already be opened
 * (a file found by its ID, or one the user chose) keeps that file and its answer from the backend;
 * the files the new folder holds for it are offered as `alternatives` (`alternativesFolder` says
 * where they are). A row without a usable file takes what the new folder found. A row for which the
 * new folder found no file at all keeps what it had (a file that carries its ID but does not fit
 * stays visible). The unmatched files of both folders are kept.
 *
 * @param {{measurements: object[], unmatched: object[]}} listing what the dialog shows now
 * @param {{measurements?: object[], unmatched_pattern_files?: object[], folder?: string}} data the
 *        answer of GET /api/ebsd/apex/pattern-files
 * @param {object} chosen the accepted choices (`ctx.chosen`)
 */
export function mergeFolderSearch(listing, data, chosen = {}) {
  const fresh = new Map((data?.measurements || []).map((m) => [rowKey(m), m]));
  const newLoose = data?.unmatched_pattern_files || [];
  const ctx = rowContext(chosen, listing.unmatched);
  const measurements = listing.measurements.map((old) => {
    const next = fresh.get(rowKey(old));
    if (!next) return old;
    const nextFiles = Array.isArray(next.pattern_files) ? next.pattern_files : [];
    const loose = next.id ? newLoose.filter((u) => Array.isArray(u.fits) && u.fits.includes(next.id)) : [];
    if (rowOf(old, ctx).loadable) {
      const inUse = new Set([rowOf(old, ctx).file?.path]);
      const seen = new Set();
      const alternatives = [...nextFiles.filter((f) => f.matches_grid), ...loose].filter((f) => {
        if (inUse.has(f.path) || seen.has(f.path)) return false;
        seen.add(f.path);
        return true;
      });
      return { ...old, alternatives, alternativesFolder: data?.folder || '' };
    }
    if (nextFiles.length === 0 && (old.pattern_files || []).length > 0) return old;
    return next;
  });
  const seenLoose = new Set();
  const unmatched = [...(listing.unmatched || []), ...newLoose].filter((u) => {
    if (seenLoose.has(u.path)) return false;
    seenLoose.add(u.path);
    return true;
  });
  return { measurements, unmatched };
}

/**
 * The path a loaded file is recorded under in Recent, the path box and the Dashboard. A measurement
 * of an APEX file is loaded from its pattern file, but opening that pattern file again by itself
 * would skip the measurement dialog and load without the metadata (a hexagonal scan is then
 * refused, a square one gets the placeholder pattern centre). So it is recorded under the APEX
 * `.h5`, and opening that goes through the dialog again.
 */
export const loadedAs = (path, extra) => extra?.apex?.h5_path || path;

/**
 * The path to ask about when a pattern file that was loaded with an APEX pairing is opened again
 * from somewhere that still holds the pattern file's own path: the metadata file of its entry in
 * the list of loaded files, else the path itself.
 */
const canon = (p) => String(p || '').replace(/\\/g, '/').toLowerCase();

export function reopenPath(path, loadedFiles) {
  const entry = (loadedFiles || []).find((f) => canon(f.path) === canon(path));
  return entry?.apex?.h5_path || path;
}

/**
 * The active entry of the loaded files when it is a measurement of the APEX file whose path is in
 * the path box (`filePath`, which a dialog load sets to the metadata file), else null. Typing
 * another path into the box is not the open measurement, so it gets no measurement name.
 */
export function activeApexEntry(loadedFiles, filePath) {
  const here = canon(String(filePath || '').trim());
  if (!here) return null;
  return (loadedFiles || []).find(
    (f) => f.active && apexMeasurementName(f) && canon(f.apex.h5_path) === here) || null;
}

const basename = (p) => String(p || '').split(/[\\/]/).pop();

/**
 * What the status bar names: the file in the path box, as ever; for a measurement of an APEX file
 * the same label the list of loaded files shows (the measurement, then its pattern file), with the
 * pattern file's path as the tooltip, because the metadata file is shared by all its measurements.
 * @returns {{text: string, title: string}}
 */
export function statusFileName(filePath, loadedFiles) {
  const entry = activeApexEntry(loadedFiles, filePath);
  return entry ? { text: loadedFileLabel(entry), title: entry.path }
    : { text: basename(filePath), title: filePath };
}

/** A name that Windows accepts as (part of) a file name: "Area 1 / OIM Map 1" -> "Area 1_OIM Map 1". */
export function fileNameSafe(text) {
  const printable = Array.from(String(text || '')).filter((c) => c.charCodeAt(0) >= 32).join('');
  return printable.replace(/\s*[<>:"/\\|?*]+\s*/g, '_').replace(/[. ]+$/, '').trim();
}

/**
 * The stem of the suggested name of an exported image: the file's name without its extension, as
 * ever; for a measurement of an APEX file `<metadata file stem>_<measurement>`, so two measurements
 * of one file do not suggest the same name.
 */
export function exportStem(filePath, activeDataset, loadedFiles) {
  const raw = filePath || activeDataset || 'ebsd';
  const stem = basename(raw).replace(/\.[^.]+$/, '') || 'ebsd';
  const entry = activeApexEntry(loadedFiles, filePath);
  const measurement = entry ? fileNameSafe(apexMeasurementName(entry)) : '';
  return measurement ? `${stem}_${measurement}` : stem;
}


/**
 * What is ticked when the dialog opens: the one measurement if the FILE holds exactly one and it
 * can be opened, otherwise nothing. With several measurements the choice is the user's; loading
 * several multi-gigabyte scans is a decision, not a default.
 */
export function defaultSelection(measurements, ctx = {}) {
  const list = measurements || [];
  if (list.length !== 1) return [];
  return rowOf(list[0], ctx).loadable ? [rowKey(list[0])] : [];
}

/**
 * The loads to run, in the order the measurements are listed: one `POST /api/ebsd/load` each,
 * carrying the pattern file's path and the pairing the user saw (`apex`). A ticked row that cannot
 * be opened any more (a new folder search took its file away) is left out.
 */
export function buildLoadJobs(h5Path, measurements, selected, ctx = {}) {
  const wanted = new Set(selected || []);
  const jobs = [];
  for (const m of measurements || []) {
    if (!wanted.has(rowKey(m))) continue;
    const row = rowOf(m, ctx);
    if (!row.loadable || !row.file?.path) continue;
    jobs.push({
      path: row.file.path,
      label: measurementTitle(m),
      apex: { h5_path: h5Path, measurement_id: m.id },
    });
  }
  return jobs;
}

/**
 * The summary line's numbers: how many measurements, how many can be opened now, how many need a
 * pattern file (none found), how many have a pattern file that does not fit, how many cannot be
 * opened at all.
 */
export function summaryOf(measurements, ctx = {}) {
  const rows = (measurements || []).map((m) => rowOf(m, ctx));
  const ready = rows.filter((r) => r.loadable).length;
  const blocked = rows.filter((r) => r.state === 'blocked').length;
  // a pattern file was found and does not fit: its own kind, as the row says "cannot open it with this"
  const wrongFile = rows.filter((r) => !r.loadable && r.state === 'wrong').length;
  return { total: rows.length, ready, blocked, wrongFile, needFile: rows.length - ready - blocked - wrongFile };
}

/** The "problem" of a chosen pattern file that the dialog itself refused: another pattern size. Not a backend code. */
export const SIZE_MISMATCH = 'patternSizeMismatch';

/**
 * Why a pattern file the backend described is not usable, in the shape the dialog words:
 * `tooFew` / `tooMany` (the pattern count against the grid, with the numbers), `size` (a file chosen
 * without proof by ID whose pattern size differs from the measurement's), `otherMeasurement`
 * (its header names another measurement; `other` is its title when that measurement is one of this
 * file's, else null), or `code` (any other code of the backend, worded by `errors.<code>`).
 * "Incomplete" is said only of a file with too few patterns.
 * @returns {{kind: string, [k: string]: any}|null} null for a file with no problem
 */
export function refusalOf(entry, m, measurements = []) {
  if (!entry || !entry.problem) return null;
  if (entry.problem === SIZE_MISMATCH) {
    const both = sizeMismatch(m, entry);
    return both ? { kind: 'size', ...both } : { kind: 'code', code: entry.problem };
  }
  if (entry.problem === 'apexPatternCountMismatch' && entry.n_patterns != null
      && (entry.count === 'too_few' || entry.count === 'too_many')) {
    return {
      kind: entry.count === 'too_few' ? 'tooFew' : 'tooMany',
      have: entry.n_patterns, need: Number(m.n_rows) * Number(m.n_cols),
      rows: m.n_rows, cols: m.n_cols,
    };
  }
  if (entry.problem === 'apexGuidMismatch') {
    const owner = entry.names_group
      ? (measurements || []).find((x) => x.group === entry.names_group) : null;
    return { kind: 'otherMeasurement', other: owner ? measurementTitle(owner) : null };
  }
  return { kind: 'code', code: entry.problem };
}

/** The entry for a pattern file of a folder, taken for a row: one that carries no measurement ID
 *  is paired by its pattern count, one that carries the row's ID keeps its pairing by ID. */
export function candidateAsChoice(file) {
  return {
    state: 'ok', path: file.path, name: file.name,
    entry: { ...file, pairing: file.pairing || 'count', count: 'fits', matches_grid: true, problem: null },
  };
}

/**
 * The sentence for a switch back to a file whose APEX pairing could not be used again (the metadata
 * file moved, was rewritten, or cannot be read), or null when the pairing was used or there was none.
 * The load itself went ahead without the pairing, so the user has to be told that the pattern centre
 * and the grid come from somewhere else now.
 * @param {object} data the answer of POST /api/ebsd/switch-file
 * @param {(key: string, options?: object) => string} tt translates a key of the `ebsdviewer` namespace
 */
export function pairingDroppedMessage(data, tt) {
  if (!data || data.apex_pairing_dropped !== true) return null;
  const code = data.apex_pairing_dropped_reason || '';
  return tt('apexDialog.pairingDropped', { reason: tt(`errors.${code}`, { defaultValue: code }) });
}

/**
 * Say that a switch had to drop the remembered pairing: to the log and as a warning toast, in the
 * language of the page. Every place that switches the loaded file calls this with the answer of
 * switch-file. Returns the sentence, or null when there was nothing to say.
 * @param {object} data the answer of POST /api/ebsd/switch-file
 * @param {(key: string, options?: object) => string} tt translates a key of the `ebsdviewer` namespace
 * @param {{log?: (m: string) => void, warn: (m: string) => void}} sinks
 */
export function announcePairingDropped(data, tt, sinks) {
  const message = pairingDroppedMessage(data, tt);
  if (!message) return null;
  sinks.log?.(message);
  sinks.warn(message);
  return message;
}

/**
 * `fn` that runs one call at a time: while a call is pending, another returns `busy` at once. A
 * second Enter or click on the same .h5 must not open a second dialog.
 */
export function singleFlight(fn, busy) {
  let pending = false;
  return async (...args) => {
    if (pending) return busy;
    pending = true;
    try {
      return await fn(...args);
    } finally {
      pending = false;
    }
  };
}

/**
 * The dialog's run of loads: one after another, so each lands in the list of loaded files and the
 * last is the active dataset. The first failure stops the run (its error window is open, and
 * loading on would stack another on top of it).
 * @param {{path: string, label: string, apex: object}[]} jobs
 * @param {(job: object) => Promise<boolean>} load true when the file loaded
 * @param {(index: number, job: object) => void} [onStart]
 * @returns {Promise<{done: number, stopped: {failed: string, notOpened: string[]}|null}>}
 */
export async function runLoads(jobs, load, onStart) {
  let done = 0;
  for (let i = 0; i < jobs.length; i += 1) {
    onStart?.(i, jobs[i]);
    const ok = await load(jobs[i]);
    if (!ok) break;
    done += 1;
  }
  return { done, stopped: stoppedRun(jobs, done) };
}

/**
 * The labels of the loaded files in the viewer's list, in the order of the list. A measurement of
 * an APEX file is named by its measurement ("Area 1 / OIM Map 1 (a)"), which already tells it
 * apart; the other files go through `shorten`, which strips what all their names share.
 * @param {object[]} loadedFiles entries of GET /loaded-files
 * A file opened as a variant other than its default (`scan · 2 · unprocessed`) keeps its whole name,
 * which carries the variant: shortening would cut the very part that tells it from its sibling.
 * @param {(names: string[]) => string[]} shorten `disambiguateNames`
 * @param {(key: string, options?: object) => string} [tt] translates a key of the `ebsdviewer` namespace
 */
export function sidebarLabels(loadedFiles, shorten, tt) {
  const files = loadedFiles || [];
  const whole = (f) => !!apexMeasurementName(f) || isNonDefaultVariant(f);
  const plain = files.filter((f) => !whole(f));
  const shortened = shorten(plain.map((f) => f.name || ''));
  let next = 0;
  return files.map((f) => (whole(f) ? loadedFileLabel(f, tt) : (shortened[next++] || f.name || '')));
}

/**
 * The line that says which measurements stayed closed when a run of loads stopped at its first
 * failure, as `{failed, notOpened[]}` (titles); null when everything was opened.
 */
export function stoppedRun(jobs, doneCount) {
  if (!Array.isArray(jobs) || doneCount >= jobs.length) return null;
  return {
    failed: jobs[doneCount].label,
    notOpened: jobs.slice(doneCount + 1).map((j) => j.label),
  };
}
