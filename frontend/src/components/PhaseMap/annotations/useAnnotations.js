/**
 * useAnnotations — React state hook for canvas annotations
 * (Legend, Scalebar, Title, North-Arrow) on the Phase Maps page.
 *
 * State shape:
 *   annotations: [
 *     { id: 'legend-1', type: 'legend',  x, y, w, h, rotation, props },
 *     { id: 'sb-1',     type: 'scalebar', x, y, w, h, rotation, props },
 *     { id: 'title-1',  type: 'title',    x, y, w, h, rotation, props },
 *     { id: 'arrow-1',  type: 'arrow',    x, y, w, h, rotation, props },
 *   ]
 *
 * (x, y) are normalised [0..1] relative to the canvas; (w, h) are
 * also normalised. ``rotation`` is degrees (CW positive).
 * ``props`` is type-specific config (font size, text, length-um, etc).
 *
 * Persists to localStorage keyed by resultId so re-opening the page
 * restores the user's layout.
 */
import { useState, useEffect, useCallback, useMemo } from 'react';

const STORAGE_PREFIX = 'phaseMapAnnotations:';

// Default templates for each type. x/y/w/h are normalised canvas coords.
const DEFAULTS = {
  legend: {
    x: 0.02, y: 0.65, w: 0.28, h: 0.33, rotation: 0,
    props: { fontSize: 11, background: 'rgba(20,22,30,0.85)' },
  },
  scalebar: {
    x: 0.6, y: 0.93, w: 0.3, h: 0.05, rotation: 0,
    props: { lengthUm: 5, fontSize: 12, barColor: '#ffffff', textColor: '#ffffff' },
  },
  title: {
    x: 0.04, y: 0.02, w: 0.5, h: 0.06, rotation: 0,
    props: { text: 'Phase Map', fontSize: 16, color: '#ffffff' },
  },
  arrow: {
    x: 0.88, y: 0.05, w: 0.1, h: 0.1, rotation: 0,
    props: { label: 'ND', color: '#ffb86c', fontSize: 11 },
  },
};

function makeId(type) {
  return `${type}-${Date.now().toString(36)}-${Math.floor(Math.random() * 1e4).toString(36)}`;
}

export function useAnnotations(resultId) {
  const storageKey = resultId
    ? `${STORAGE_PREFIX}${String(resultId)}`
    : null;

  const [annotations, setAnnotations] = useState([]);

  // Load on mount / resultId change
  useEffect(() => {
    if (!storageKey) {
      setAnnotations([]);
      return;
    }
    try {
      const raw = localStorage.getItem(storageKey);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) {
          setAnnotations(parsed);
          return;
        }
      }
    } catch {
      // ignore
    }
    setAnnotations([]);
  }, [storageKey]);

  // Persist on change
  useEffect(() => {
    if (!storageKey) return;
    try {
      localStorage.setItem(storageKey, JSON.stringify(annotations));
    } catch {
      // localStorage quota / unavailable; not a hard error
    }
  }, [storageKey, annotations]);

  const addAnnotation = useCallback((type) => {
    if (!DEFAULTS[type]) return;
    const def = DEFAULTS[type];
    const annot = {
      id: makeId(type),
      type,
      ...def,
      props: { ...def.props },
    };
    setAnnotations((prev) => [...prev, annot]);
  }, []);

  const removeAnnotation = useCallback((id) => {
    setAnnotations((prev) => prev.filter((a) => a.id !== id));
  }, []);

  const updateAnnotation = useCallback((id, patch) => {
    setAnnotations((prev) =>
      prev.map((a) =>
        a.id === id
          ? {
              ...a,
              ...patch,
              props: patch.props ? { ...a.props, ...patch.props } : a.props,
            }
          : a,
      ),
    );
  }, []);

  const clearAnnotations = useCallback(() => {
    setAnnotations([]);
  }, []);

  return useMemo(
    () => ({
      annotations,
      addAnnotation,
      removeAnnotation,
      updateAnnotation,
      clearAnnotations,
    }),
    [annotations, addAnnotation, removeAnnotation, updateAnnotation, clearAnnotations],
  );
}

export default useAnnotations;
