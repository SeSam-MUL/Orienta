import { useCallback, useState } from 'react';
import { edsApi } from '../../../services/api';
import { normalizeRect } from '../../../services/rect';

/**
 * Region Average state for the EDS page.
 *
 * Two entry points, and the difference between them is the whole point:
 *
 *   - ``quantify()``            — the user pressed "Compute", so the rectangle
 *                                 is whatever the four number fields hold.
 *   - ``onRegionSelected(rect)``— the user shift-dragged one, so the rectangle
 *                                 is the one they just drew, and it travels as
 *                                 an ARGUMENT.
 *
 * It used to write the fields and then fire the compute on a
 * ``setTimeout(..., 0)`` "so the state updates have a chance to land". They do
 * land — but a timeout defers WHEN a function runs, not WHICH function: the
 * one it held was bound in the render before the drag and read the pre-drag
 * values. So the first drag of a session averaged the 0,0,0,0 default and
 * every later one averaged the PREVIOUS rectangle, while the fields and the
 * dashed outline showed the new one. If the fields happened to hold a reversed
 * pair typed by hand, that is the HTTP 400 recorded on 2026-09-10 13:15:51.
 *
 * The fields still follow the drag, because the user must be able to read and
 * adjust the numbers — but they are a display of the rectangle, no longer the
 * channel it is sent through.
 */
export function useRegionAverage(displayMode, t) {
  const [rowStart, setRowStart] = useState(0);
  const [rowEnd, setRowEnd] = useState(0);
  const [colStart, setColStart] = useState(0);
  const [colEnd, setColEnd] = useState(0);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  /**
   * Average a rectangle. With no argument, the one in the number fields.
   * (`rect` is checked for a coordinate rather than truthiness, so wiring this
   * straight to an onClick cannot pass a DOM event off as a rectangle.)
   */
  const quantify = useCallback(async (rect) => {
    const r = rect && rect.rowStart !== undefined
      ? rect
      // Typed by hand, so possibly bottom-right to top-left. The request is
      // ordered anyway (services/rect.js); the fields are ordered too, so the
      // panel shows the rectangle that was actually measured instead of one
      // the numbers deny.
      : normalizeRect({ rowStart, rowEnd, colStart, colEnd });
    if (!rect) {
      setRowStart(r.rowStart); setRowEnd(r.rowEnd);
      setColStart(r.colStart); setColEnd(r.colEnd);
    }
    setLoading(true); setError(null); setData(null);
    try {
      const res = await edsApi.regionQuantify(
        r.rowStart, r.rowEnd, r.colStart, r.colEnd, displayMode,
      );
      setData(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || t('regionAvg.error'));
    } finally { setLoading(false); }
  }, [rowStart, rowEnd, colStart, colEnd, displayMode, t]);

  /** Shift+drag ROI → show the rectangle in the fields and average it. */
  const onRegionSelected = useCallback((rect) => {
    setRowStart(rect.rowStart);
    setRowEnd(rect.rowEnd);
    setColStart(rect.colStart);
    setColEnd(rect.colEnd);
    quantify(rect);
  }, [quantify]);

  return {
    rowStart, setRowStart,
    rowEnd, setRowEnd,
    colStart, setColStart,
    colEnd, setColEnd,
    data, loading, error,
    quantify, onRegionSelected,
  };
}
