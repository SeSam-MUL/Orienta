// What the calibration offer and the "Take PC from..." window read out of the backend's answers.
// Pure functions, no page state, no words: they return i18n keys and their parameters (or plain values), and
// the components call `t`. Every rule here is pinned by calibrationText.test.js.
import { warningOf, contrastOf } from '../pcResultFacts';

const finite = (v) => typeof v === 'number' && Number.isFinite(v);

/** The settings the table lists, in the order the page shows them. `pattern_shape` is always compared. */
export const SETTING_KEYS = [
  'pattern_shape', 'binning', 'camera_tilt', 'sample_tilt', 'working_distance_mm', 'detector_insertion_mm',
];

// the i18n name (under `calibration.settings.names`) of each setting
const NAMES = {
  pattern_shape: 'patternSize', binning: 'binning', camera_tilt: 'cameraTilt', sample_tilt: 'sampleTilt',
  working_distance_mm: 'workingDistance', detector_insertion_mm: 'detectorInsertion',
};
export const settingNameKey = (key) => `calibration.settings.names.${NAMES[key] || key}`;

const trimmed = (v, digits) => String(Number(v.toFixed(digits)));

/**
 * One setting as the table writes it: "128 × 156" (rows × columns), "8", "5.0°", "20 mm".
 * `null` for a setting that is not stated (the page writes "not stated").
 */
export function formatSetting(key, value) {
  if (key === 'pattern_shape') {
    return Array.isArray(value) && value.length === 2 && value.every(finite) ? `${value[0]} × ${value[1]}` : null;
  }
  if (!finite(value)) return null;
  if (key === 'binning') return String(Math.round(value));
  if (key === 'camera_tilt' || key === 'sample_tilt') return `${value.toFixed(1)}°`;
  return `${trimmed(value, 2)} mm`;
}

/**
 * The rows of the settings table of an offer (a suggestion of the backend): for each setting the value of the
 * file and of the calibration (`null`: not stated) and the check, `'same' | 'different' | 'notStated'`.
 * The values come from `settings` (also the ones that agree); a backend without it still gives the ones that differ.
 */
export function settingRows(offer) {
  const differences = new Map((offer?.differences || []).map((d) => [d.key, d]));
  const unknown = new Set(offer?.unknown || []);
  const file = offer?.settings?.file || {};
  const calibration = offer?.settings?.calibration || {};
  return SETTING_KEYS.map((key) => {
    const diff = differences.get(key);
    const check = diff ? 'different' : (unknown.has(key) ? 'notStated' : 'same');
    return {
      key,
      file: formatSetting(key, key in file ? file[key] : diff?.file),
      calibration: formatSetting(key, key in calibration ? calibration[key] : diff?.calibration),
      check,
    };
  });
}

/** "All N settings same": N is the number of settings really compared, the pattern size included. */
export function sameCount(offer) {
  return Array.isArray(offer?.checked) ? offer.checked.length : 0;
}

/**
 * The lines "Other working distance: 15 mm, calibration 20 mm": one per setting that differs, as
 * `{key, file, calibration}` (formatted).
 */
export function differenceLines(offer) {
  return (offer?.differences || []).map((d) => ({
    key: d.key,
    file: formatSetting(d.key, d.file),
    calibration: formatSetting(d.key, d.calibration),
    // a pattern shape of another height-to-width ratio: the PC does not carry over (see `otherAspect`)
    aspect: d.key === 'pattern_shape' && otherAspect(d.file, d.calibration),
  }));
}

/** The largest relative difference of two height-to-width ratios that is still the same ratio (a binned pattern
 *  is rounded to whole pixels: 1244 x 1024 binned by 8 is 155 x 128, ratios 1.2148 and 1.2109). */
export const ASPECT_TOLERANCE = 0.02;

/**
 * True when two pattern shapes `[rows, columns]` have another height-to-width ratio. The same ratio at another size
 * (binning) is not: a PC is a fraction of the pattern and carries over. A shape that is not two positive numbers
 * says nothing, so false.
 */
export function otherAspect(a, b) {
  const ok = (s) => Array.isArray(s) && s.length === 2 && s.every((v) => finite(v) && v > 0);
  if (!ok(a) || !ok(b)) return false;
  const ra = a[0] / a[1];
  const rb = b[0] / b[1];
  return Math.abs(ra - rb) / Math.max(ra, rb) > ASPECT_TOLERANCE;
}

/** How an offer stands to the file: 'fits' | 'no' (a setting differs) | 'unstated' (it cannot be told). */
export function fitState(offer) {
  if (offer?.match === true) return 'fits';
  if ((offer?.differences || []).length > 0) return 'no';
  return 'unstated';
}

/**
 * Which of the three kinds of offer a suggestion is: `fits` (A), `differs` (B: the file used this
 * calibration before, and a setting differs), `unstated` (C: the settings cannot be compared).
 */
export function offerKind(offer) {
  const state = fitState(offer);
  return state === 'fits' ? 'fits' : (state === 'no' ? 'differs' : 'unstated');
}

/** "Version 2, updated 2026-10-20 from scan_x.h5oina" / "Version 1, saved 2026-10-14", as a key and parameters. */
export function versionMeta(item) {
  const version = Number(item?.version);
  const created = typeof item?.version_created === 'string' ? item.version_created : '';
  const date = /^\d{4}-\d{2}-\d{2}/.test(created) ? created.slice(0, 10) : '';
  const source = typeof item?.source_name === 'string' && item.source_name ? item.source_name : '';
  if (!date) return { key: 'versionOnly', params: { version } };
  if (version <= 1) return { key: 'versionSaved', params: { version, date } };
  if (source) return { key: 'versionUpdated', params: { version, date, source } };
  return { key: 'versionUpdatedNoSource', params: { version, date } };
}

/** The date ("2026-10-21") of a file, or ''. */
export function isoDate(value) {
  return typeof value === 'string' && /^\d{4}-\d{2}-\d{2}/.test(value) ? value.slice(0, 10) : '';
}

/** The size of a PC change in pixels: x counts in columns, y and z in rows; the length of that vector. */
export function changeInPixels(from, to, shape) {
  const ok = (p) => Array.isArray(p) && p.length === 3 && p.every(finite);
  if (!ok(from) || !ok(to) || !Array.isArray(shape) || shape.length !== 2 || !shape.every((v) => finite(v) && v > 0)) {
    return null;
  }
  const [rows, cols] = shape;
  return Math.hypot((to[0] - from[0]) * cols, (to[1] - from[1]) * rows, (to[2] - from[2]) * rows);
}

// what an offer replaces, by the kind of the PC in use (`pc_in_use.kind` of the backend)
const REPLACES = {
  file: 'file', find: 'find', refine: 'refine', typed: 'typed', calibration: 'calibration', grid: 'grid',
};

/**
 * What taking the offer would change: `{kind, pixels}` with `kind` the i18n name of what is replaced ("the PC from
 * the file") and `pixels` the size of the change (null when it cannot be told, and for a per-pixel map, whose mean
 * says nothing about the points of the map). `null` when the file has no PC in use: nothing is replaced.
 */
export function replacesOf(offer) {
  const r = offer?.replaces;
  const kind = REPLACES[r?.kind];
  if (!kind || !Array.isArray(r?.pc)) return null;
  const pixels = kind === 'grid' ? null : changeInPixels(r.pc, offer.pc, offer.settings?.file?.pattern_shape);
  return { kind, pixels };
}

/** The pixels as the page writes them: one decimal ("0.6"). */
export const formatPixels = (pixels) => pixels.toFixed(1);

/** A change that one decimal would write as "0.0": the page says "under 0.1 pixels" instead (no "about 0.0"). */
export const isTinyChange = (pixels) => pixels < 0.05;

/** The warning a saved calibration carries: 'edge' | 'notAMinimum' | 'secondPosition' | null. */
export function savedWarning(quality) {
  return warningOf({ quality: quality || null })?.kind ?? null;
}

/** The fit contrast of a saved calibration, one decimal, or null. */
export function savedContrast(quality) {
  const c = contrastOf({ quality: quality || null });
  return c == null ? null : c.toFixed(1);
}

/** The base name of a path typed or picked by the user. */
export function baseName(path) {
  return String(path || '').split(/[\\/]/).filter(Boolean).pop() || '';
}

// the kind of file a PC was read from, as the "In the ..." line names it
const SOURCE_KINDS = ['ang', 'osc', 'edax_h5', 'oxford_h5', 'orienta_export'];
export const sourceKindOf = (kind) => (SOURCE_KINDS.includes(kind) ? kind : 'other');

/**
 * The i18n key (under `takePc.notes`) and parameters of a note of `from-file`'s `note_items` `{code, params, text}`.
 * A code this build has no words for gives `null`: the caller shows the backend's own `text`.
 */
export function noteWords(item, context = {}) {
  const p = item?.params || {};
  switch (item?.code) {
    case 'assumedShape':
      if (!finite(p.rows) || !finite(p.cols) || !['tsl', 'oxford'].includes(p.convention)) return null;
      return { key: `assumedShape.${p.convention}`, params: { kind: context.kind, file: context.file, rows: p.rows, cols: p.cols } };
    case 'outsideDetectorRange':
      return Array.isArray(p.axes) ? { key: 'outsideDetectorRange', params: { axes: p.axes.join(', '), limits: p.limits } } : null;
    case 'pixelsLeftOut':
      return finite(p.left_out) && finite(p.total) ? { key: 'pixelsLeftOut', params: { left: p.left_out, total: p.total } } : null;
    case 'meanOfMap':
      return { key: 'meanOfMap', params: {} };
    case 'stdOverMap':
      return finite(p.x) && finite(p.y) && finite(p.z)
        ? { key: 'stdOverMap', params: { x: Number(p.x.toPrecision(3)), y: Number(p.y.toPrecision(3)), z: Number(p.z.toPrecision(3)) } }
        : null;
    case 'orientaExport':
      return { key: 'orientaExport', params: {} };
    default:
      return null;
  }
}

/** "+0.0007" / "-0.0029": a change of one PC axis with its sign. */
export const formatSigned = (v) => {
  const digits = Math.abs(v).toFixed(4);
  // a number that rounds to zero has no sign: "-0.0000" says a direction the four decimals cannot show
  if (Number(digits) === 0) return digits;
  return `${v < 0 ? '-' : '+'}${digits}`;
};

/**
 * What a refine changed for the files that stay on the calibration: the change of each axis (`to - from`) and its
 * size in pixels (x counts in columns, y and z in rows, the length of that vector). `null` when a PC is missing.
 * `pixels` is `null` when the pattern size is not known.
 */
export function pcChange(from, to) {
  const ok = (p) => Array.isArray(p) && p.length === 3 && p.every(finite);
  if (!ok(from) || !ok(to)) return null;
  return { delta: [to[0] - from[0], to[1] - from[1], to[2] - from[2]] };
}

/** One decimal for a fit contrast, or null. */
export const contrastText = (v) => (finite(v) ? v.toFixed(1) : null);

/**
 * What a finished search leaves to ask: `result.calibration` is `{calibration: {id, name, version, latest},
 * diverged, followers}`. A question is due only when the file left a calibration that other files are still on.
 */
export function followerQuestion(block) {
  if (!block || !block.calibration || !block.diverged) return null;
  if (!Array.isArray(block.followers) || block.followers.length === 0) return null;
  // a follower is `{dataset, per_pixel, map_kind}`; `map_kind`: "grid" (a version replaces that map) | "header" (it moves)
  const entries = block.followers.map((f) => (typeof f === 'string' ? { dataset: f, per_pixel: false, map_kind: null } : f))
    .filter((f) => f && typeof f.dataset === 'string');
  if (entries.length === 0) return null;
  // a map of a kind the page does not know is told as the milder one: the version moves it, it does not drop it
  const withMap = (isGrid) => entries
    .filter((f) => f.per_pixel === true && (f.map_kind === 'grid') === isGrid).map((f) => f.dataset);
  return {
    calibration: block.calibration,
    followers: entries.map((f) => f.dataset),
    maps: { grid: withMap(true), header: withMap(false) },
  };
}

/** What identifies one question: the file, the calibration version it left and the PC it moved to. */
export function questionKey(result) {
  const q = followerQuestion(result?.calibration);
  return q ? `${result.dataset}|${q.calibration.id}@${q.calibration.version}|${(result.mean_pc || []).join(',')}` : null;
}

/** The stored results of `GET /{id}/results` that were computed at an older version (the only ones the page shows). */
export function olderResults(results) {
  return (Array.isArray(results) ? results : []).filter((r) => r && r.state === 'older');
}

/**
 * The rows of the window "Apply to other files": `candidates` is the answer of `GET /{id}/candidates`. The file the
 * calibration was made on (`origin`) is not a row. `fit`: the settings agree; `others`: they differ (`differs`, with
 * the lines of what differs) or cannot be told (`unstated`). A file that fits and is already on this version is
 * listed with "no change" and is not ticked; everything else that fits is.
 */
export function applyRows(candidates, origin) {
  const fit = [];
  const others = [];
  (Array.isArray(candidates) ? candidates : []).forEach((c) => {
    if (!c || c.dataset === origin) return;
    const wd = c.settings?.file?.working_distance_mm;
    const wdUnknown = (c.unknown || []).includes('working_distance_mm') || !finite(wd);
    const diffWd = (c.differences || []).find((d) => d.key === 'working_distance_mm');
    const row = {
      dataset: c.dataset, path: c.path || null, pc: Array.isArray(c.pc) ? c.pc : null, ref: c.ref || null,
      wd: formatSetting('working_distance_mm', wd),
      wdCheck: wdUnknown ? 'notStated' : (diffWd ? 'different' : 'same'),
      differences: differenceLines(c),
    };
    if (c.match === true) {
      const same = c.on_this_version === true;
      fit.push({ ...row, kind: 'fits', ticked: !same, why: same ? 'sameVersion' : '' });
    } else if (c.on_this_version === true) {
      // already given this version, by a second naming: nothing would change, so it cannot be ticked
      others.push({ ...row, kind: 'same', ticked: false, why: 'sameVersion' });
    } else {
      others.push({ ...row, kind: row.differences.length > 0 ? 'differs' : 'unstated', ticked: false,
        why: row.differences.length > 0 ? 'differs' : 'unstated' });
    }
  });
  return { fit, others };
}

/**
 * The open dataset when a calibration was applied to the FILE it is a copy of and the copy was not written (it has a
 * PC of its own and keeps it), else null. `answer` is one result of the apply: `dataset` is the file that was named,
 * `pc_map_actions` lists every dataset written.
 */
export function copyKeptOf(answer, openDataset) {
  if (!answer || !openDataset || !answer.dataset || answer.dataset === openDataset) return null;
  const written = answer.pc_map_actions && typeof answer.pc_map_actions === 'object' ? answer.pc_map_actions : {};
  return Object.prototype.hasOwnProperty.call(written, openDataset) ? null : openDataset;
}
