/**
 * The same edits `useAnnotations` performs, on a plain array.
 *
 * The export dialog works on a COPY of the map's annotations, so it cannot use
 * that hook (which owns state and persists to localStorage). Keeping the rules
 * here — rather than a second inline implementation — means a property patch
 * behaves identically in both places, in particular the shallow merge of
 * `props` that lets the toolbar send a single key.
 */

import { DEFAULTS, makeId } from './useAnnotations';

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
