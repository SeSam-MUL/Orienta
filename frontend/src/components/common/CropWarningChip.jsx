/**
 * "This layer could not follow the crop" — one line, in the layer's own row.
 *
 * Electron images are a separate acquisition area on their own, finer grid.
 * Two ways that can go wrong, and both have to reach the user rather than only
 * the log, because neither is visible in the picture:
 *
 *   cropped: false — the backend refused to place the window (no step size on
 *     one of the areas, or the file says the two areas start at different
 *     points) and returned the FULL image.
 *   exact: false — the window ran past the edge of that area, so the cut-out
 *     had to be clamped. It is real data, but no longer the window's aspect
 *     ratio, and `LayeredCanvas` stretches every layer to the composite size —
 *     so every feature on it moves.
 *
 * Renders nothing when the layer followed the crop, which is the normal case
 * and the case for every layer when no crop is active.
 */
import React from 'react';
import { useTranslation } from 'react-i18next';

/** Is this verdict worth showing? Exported so the pages' one-line insertion
 *  carries no logic of its own — see CropWarningChip.test.jsx. */
export function isCropWarning(status) {
  if (!status) return false;
  return status.cropped === false || status.exact === false;
}

/** The verdict for `layerId`, or null when there is nothing to say.
 *  `statuses` is the Map both layer-stack hooks expose. */
export function cropWarningFor(statuses, layerId) {
  const status = statuses?.get?.(layerId);
  return isCropWarning(status) ? status : null;
}

export default function CropWarningChip({ status }) {
  const { t } = useTranslation();
  if (!isCropWarning(status)) return null;
  const key = status.cropped === false
    ? 'ebsdviewer:crop.electronImageNotCropped'
    : 'ebsdviewer:crop.electronImageClamped';
  return (
    <div
      role="status"
      style={{
        fontSize: '7.5pt',
        color: '#fbbf24',
        display: 'flex',
        alignItems: 'flex-start',
        gap: 4,
        padding: '1px 2px',
      }}
      // The backend's reason is English prose written for a log; it is the
      // precise answer, so it belongs in the tooltip rather than in place of
      // the translated sentence.
      title={status.reason || undefined}
    >
      <span aria-hidden="true">⚠</span>
      <span>{t(key)}</span>
    </div>
  );
}
