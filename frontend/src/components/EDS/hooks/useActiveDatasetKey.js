import { useEffect, useState } from 'react';
import { ebsdApi } from '../../../services/api';

/**
 * "The active dataset is not known yet."
 *
 * A Symbol rather than `undefined` or `null` because both of those are already
 * spoken for: a default parameter value silently replaces `undefined`, so a
 * consumer could not tell "not passed" from "not known", and `null` is the
 * legitimate answer for a session with no named dataset. A Symbol cannot
 * collide with a dataset name either.
 */
export const DATASET_UNKNOWN = Symbol('active dataset not resolved yet');

/**
 * The NAME of the active EBSD dataset ('Scan1', 'Scan1_crop1', …).
 *
 * Why the EDS page needs one at all: cropping produces a new dataset from the
 * SAME file. `isFileOpen` and `filePath` do not move, so nothing the page
 * keyed on moved either — and every page in this app stays mounted (App.jsx
 * toggles display), so the EDS bitmaps are already warm long before the user
 * draws a crop. The page went on compositing the PARENT's full-scan element
 * maps while `POST /api/eds/probe`, `/linescan` and `/region-stats` resolved
 * the very same (row, col) on the CROP's grid: a hover inside the crop
 * returned another pixel's chemistry, with no error anywhere.
 *
 * `isActive` is in the deps because that is when the answer can have changed:
 * the crop is drawn on the EBSD viewer while this page is hidden, so the page
 * re-asks as it comes back into view. Re-setting the same name is a no-op for
 * React, so an unchanged dataset costs one request and nothing else.
 *
 * Returns {@link DATASET_UNKNOWN} while the answer is in flight — callers must
 * treat that as "do not probe yet" rather than as "no dataset". Without it the
 * page probes once on the unknown key and again on the resolved one, and the
 * second probe flushes and refetches every bitmap it just decoded. A FAILED
 * read degrades to `null`, which DOES probe: losing the crop-awareness is bad,
 * but a permanently blank EDS page is worse, and the warning says so.
 */
export function useActiveDatasetKey(isFileOpen, filePath, isActive) {
  const [key, setKey] = useState(DATASET_UNKNOWN);

  useEffect(() => {
    if (!isFileOpen) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- mirrors useDefaultLayers' file-close reset; must clear synchronously.
      setKey(DATASET_UNKNOWN);
      return undefined;
    }
    let cancelled = false;
    ebsdApi.datasets()
      .then((res) => { if (!cancelled) setKey(res.data?.active ?? null); })
      .catch((err) => {
        if (cancelled) return;
        console.warn('[useActiveDatasetKey] datasets() failed — the EDS page '
          + 'cannot tell a crop from its parent until this succeeds:', err);
        setKey(null);
      });
    return () => { cancelled = true; };
  }, [isFileOpen, filePath, isActive]);

  return key;
}
