/**
 * A copy of a drawn path that only updates once the drawing has stopped.
 *
 * WHY THIS EXISTS. Building a lasso mask costs O(rows·cols·points): the
 * even-odd fill in `lassoMask` tests every cell of the bounding box against
 * every edge of the path. Measured on a 12-core desktop, one rebuild takes
 * 31 ms for a 136×169 selection with 400 sampled points, 392 ms at 361×461
 * with 1200, and 3.0 s at 693×923 with 2000. That work runs on the very thread
 * carrying the drag, so doing it per mouse-move turns a lasso over a large scan
 * into a frozen tab.
 *
 * WHY IT IS NOT A DEBOUNCE ON THE POINTS. The obvious shape — "rebuild when no
 * new point has arrived for N ms" — is shaped for the wrong drag. It suppresses
 * the rebuild during a FAST drag, but the expensive drag is the slow, careful
 * one: tracing a grain boundary crosses a new pixel every 150–250 ms, so a gap
 * short enough to feel live is exceeded on most moves and the timer fires again
 * and again MID-drag. Worse, each firing costs more than the last, because n
 * grows as the path lengthens — the stutter arrives exactly when the drag is
 * longest. No choice of N fixes that; it only moves the hand speed at which it
 * bites.
 *
 * So the settle is keyed on the DRAG, not on the gap between points. While
 * `drawing` holds, the reported value is frozen however long the hand
 * hesitates, and it is the live value again the moment the drag ends — not
 * after a delay, but structurally, because an idle hook is a passthrough.
 * Nothing needs the mask before then: it feeds the selected-pixel count and the
 * crop payload, neither of which can be read or acted on while the button is
 * down.
 *
 * THE TIMER IS ONLY A BACKSTOP, AND IT FIRES AT MOST ONCE PER DRAG. A drag can
 * end where we never hear about it — released over another application, or the
 * mouse-up swallowed. The caller is expected to end the drag on a WINDOW-level
 * mouseup, which already covers a release anywhere on the page; the timer is
 * for what even that misses. It is scheduled once, when the drag starts, and
 * deliberately NOT restarted by new points — restarting is precisely what turns
 * a backstop back into the per-pause debounce this hook exists to avoid. So the
 * cost of a drag is bounded: at most one rebuild before it ends, whatever the
 * hand does. A drag shorter than `backstopMs` pays nothing at all.
 *
 * @param {*} value           the live value (a path); returned as-is when idle
 * @param {boolean} drawing   whether a drag is currently producing it
 * @param {number} backstopMs how long an abandoned drag may hold a stale value
 * @returns {*} the settled value
 */
import { useEffect, useRef, useState } from 'react';

export function useSettledPath(value, drawing, backstopMs) {
  // Only ever read while `drawing` — see the return below.
  const [frozen, setFrozen] = useState(value);
  const [wasDrawing, setWasDrawing] = useState(drawing);

  // Adjusting state during render when an input flips, rather than in an
  // effect: React re-runs the component before committing, so the very first
  // render of a drag already reports that drag. Through an effect it would show
  // the PREVIOUS drag's shape for one frame.
  if (drawing !== wasDrawing) {
    setWasDrawing(drawing);
    // A drag has just begun. The frozen value still describes the last one — a
    // shape no longer on screen, which would otherwise be reported against this
    // drag's box for as long as this drag lasts. Adopt the starting point
    // instead; it is one point, so it costs nothing to build.
    if (drawing) setFrozen(value);
  }

  // The backstop fires long after any render, so it must read the path as it is
  // THEN, not as it was when the drag started.
  const latest = useRef(value);
  useEffect(() => { latest.current = value; }, [value]);

  // One backstop for the whole drag. `value` is deliberately absent from the
  // dependencies — see above.
  useEffect(() => {
    if (!drawing) return undefined;
    const id = setTimeout(() => setFrozen(latest.current), backstopMs);
    return () => clearTimeout(id);
  }, [drawing, backstopMs]);

  // Idle is a pure passthrough: there is no window in which a finished drag
  // reports anything but its own final path.
  return drawing ? frozen : value;
}

export default useSettledPath;
