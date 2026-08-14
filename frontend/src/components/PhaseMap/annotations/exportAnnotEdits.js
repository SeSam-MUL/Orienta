/**
 * The same edits `useAnnotations` performs, on a plain array.
 *
 * The export dialog works on a COPY of the map's annotations, so it cannot use
 * that hook (which owns state and persists to localStorage). Keeping the rules
 * here — rather than a second inline implementation — means a property patch
 * behaves identically in both places, in particular the shallow merge of
 * `props` that lets the toolbar send a single key.
 */

import { DEFAULTS, SCALE_TYPES, makeId } from './useAnnotations';

export function applyPatch(annotations, id, patch) {
  return (annotations || []).map((a) => (
    a.id === id
      ? { ...a, ...patch, props: patch.props ? { ...a.props, ...patch.props } : a.props }
      : a
  ));
}

export function withAdded(annotations, type) {
  const def = DEFAULTS[type];
  if (!def) return { annotations, id: null };
  const annot = { id: makeId(type), type, ...def, props: { ...def.props } };
  return { annotations: [...(annotations || []), annot], id: annot.id };
}

export function withRemoved(annotations, id) {
  return (annotations || []).filter((a) => a.id !== id);
}

// ---------------------------------------------------------------------------
// Scale bodies (IPF colour key, value scales)
// ---------------------------------------------------------------------------

/**
 * An `entry` names one scale the figure COULD show:
 *   { type: 'colorkey' }                      — there is only ever one
 *   { type: 'valuescale', layerId: 'bc' }     — one per layer that has a scale
 *
 * The checkbox in the export dialog is not a stored flag: it IS whether the
 * body exists. One source of truth, so removing a body with its own × keeps
 * the box in step.
 */
export function findScaleBody(annotations, entry) {
  if (!entry?.type) return null;
  return (annotations || []).find((a) => (
    a.type === entry.type
    && (entry.type !== 'valuescale' || a.props?.layerId === entry.layerId)
  )) || null;
}

// The value scales share a column to the LEFT of the map, the colour key
// stands to its RIGHT — the arrangement a reader expects, and the one the
// welded-on columns used to produce. All of it is outside [0..1], i.e. beside
// the data rather than covering it; the export dialog's border is sized to
// hold it (see `scaleMargins`).
const LEFT_COLUMN = { top: 0.02, bottom: 0.98, gap: 0.03 };

/** Still in its column — as opposed to dragged onto the map by the user. */
function inLeftColumn(a) {
  return a.type === 'valuescale' && a.x < 0;
}

/**
 * Share the left column's height between the value scales that are still in
 * it, so two bars read like one legend instead of overlapping.
 *
 * A bar the user has dragged somewhere else keeps where it was put: it is no
 * longer part of the column, and moving it back would undo their arrangement.
 */
export function respaceLeftColumn(annotations) {
  const list = annotations || [];
  const col = list.filter(inLeftColumn);
  if (col.length === 0) return list;
  const span = LEFT_COLUMN.bottom - LEFT_COLUMN.top;
  const h = (span - LEFT_COLUMN.gap * (col.length - 1)) / col.length;
  const placed = new Map(col.map((a, i) => [a, { y: LEFT_COLUMN.top + i * (h + LEFT_COLUMN.gap), h }]));
  return list.map((a) => (placed.has(a) ? { ...a, ...placed.get(a) } : a));
}

/**
 * Add or remove the body for `entry`. Idempotent in both directions, so the
 * caller can simply pass the checkbox state.
 */
export function withScaleBody(annotations, entry, present) {
  const list = annotations || [];
  const existing = findScaleBody(list, entry);
  if (!present) {
    return {
      annotations: existing ? respaceLeftColumn(list.filter((a) => a !== existing)) : list,
      id: null,
    };
  }
  if (existing) return { annotations: list, id: existing.id };
  const def = DEFAULTS[entry.type];
  if (!def) return { annotations: list, id: null };
  const annot = {
    id: makeId(entry.type),
    type: entry.type,
    ...def,
    props: { ...def.props, ...(entry.type === 'valuescale' ? { layerId: entry.layerId } : null) },
  };
  return { annotations: respaceLeftColumn([...list, annot]), id: annot.id };
}

/**
 * How much border the figure needs so nothing beside the map is cut off.
 *
 * Fractions of the map on each axis, which is exactly what the export dialog's
 * border takes. Without this a key placed beside the map would show in the
 * preview and be missing from the file.
 */
export function scaleMargins(annotations, pad = 0.02) {
  const out = { top: 0, right: 0, bottom: 0, left: 0 };
  for (const a of annotations || []) {
    if (!SCALE_TYPES.includes(a.type)) continue;
    out.left = Math.max(out.left, -a.x);
    out.right = Math.max(out.right, a.x + a.w - 1);
    out.top = Math.max(out.top, -a.y);
    out.bottom = Math.max(out.bottom, a.y + a.h - 1);
  }
  // MAX_MARGIN_FRACTION in the export dialog: half the image per side. A body
  // pushed further out than that is the user's business — they can widen the
  // border by hand.
  const fit = (v) => (v > 0 ? Math.min(0.5, v + pad) : 0);
  return { top: fit(out.top), right: fit(out.right), bottom: fit(out.bottom), left: fit(out.left) };
}
