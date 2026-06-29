/**
 * usePlayback — auto-advances the pattern index at a configurable rate.
 *
 * Reads the live current index from useDataStore.getState() inside the
 * interval callback (rather than closing over it) so the loop sees the
 * latest value even when the user scrubs manually mid-playback.
 *
 * Auto-stops when isOpen becomes false.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import useDataStore from '../../../stores/useDataStore';

export function usePlayback({ navigateByIndex, maxIndex, isOpen }) {
  const [playing, setPlaying] = useState(false);
  const [speed,   setSpeed]   = useState(5); // patterns per second
  const intervalRef = useRef(null);

  // Tick loop: bump current index by 1, wrap at maxIndex.
  useEffect(() => {
    if (!playing) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return undefined;
    }
    const ms = Math.round(1000 / Math.max(1, speed));
    intervalRef.current = setInterval(() => {
      const cur = useDataStore.getState().currentIndex;
      navigateByIndex(cur >= maxIndex ? 0 : cur + 1);
    }, ms);
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  // The interval lives only as long as `playing` is true. Reading the live
  // currentIndex via useDataStore.getState() (rather than closing over it)
  // avoids a stale-closure bug that would freeze playback at the index the
  // interval started on. The effect restarts whenever any of these deps
  // change so the bound `navigateByIndex`, target `maxIndex`, and tick
  // period (derived from `speed`) stay current.
  }, [playing, speed, maxIndex, navigateByIndex]);

  // Auto-stop on file close.
  useEffect(() => {
    if (!isOpen) setPlaying(false);
  }, [isOpen]);

  const start  = useCallback(() => setPlaying(true), []);
  const stop   = useCallback(() => setPlaying(false), []);
  const toggle = useCallback(() => setPlaying((p) => !p), []);

  return { playing, speed, setSpeed, start, stop, toggle };
}

export default usePlayback;
