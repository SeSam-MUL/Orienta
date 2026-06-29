import { useState, useRef, useCallback } from 'react';
import { renumber, hitTestMarker } from './markerGeometry';

// Shared state for one Pattern-Match dialog instance: the transient hover point
// and the persistent numbered markers, mirrored across the detector-frame panels
// (Experimental / Simulated / NCC-difference).
const HIT_RADIUS = 0.03; // normalized — click within this of a marker deletes it

export function useLinkedPatternMarkers() {
  const [hover, setHover] = useState(null);   // {x,y}|null in content-norm
  const [markers, setMarkers] = useState([]); // [{id,n,x,y}]
  const counter = useRef(0);

  const addMarker = useCallback((norm) => {
    setMarkers((prev) => renumber([...prev, { id: `m${++counter.current}`, n: 0, x: norm.x, y: norm.y }]));
  }, []);
  const removeMarker = useCallback((id) => {
    setMarkers((prev) => renumber(prev.filter((m) => m.id !== id)));
  }, []);
  const clearMarkers = useCallback(() => setMarkers([]), []);

  // A panel click either deletes an existing marker it landed on, or adds a new one.
  const handlePanelClick = useCallback((norm) => {
    setMarkers((prev) => {
      const hit = hitTestMarker(prev, norm, HIT_RADIUS);
      if (hit) return renumber(prev.filter((m) => m.id !== hit));
      return renumber([...prev, { id: `m${++counter.current}`, n: 0, x: norm.x, y: norm.y }]);
    });
  }, []);

  return { hover, setHover, markers, addMarker, removeMarker, clearMarkers, handlePanelClick };
}
