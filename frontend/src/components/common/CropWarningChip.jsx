/**
 * "This layer could not follow the crop" — one line, in the layer's own row.
 *
 * Electron images are a separate acquisition area on their own, finer grid.
 * Where a file does not place both areas in microns the backend refuses to
 * guess a cut-out and returns the FULL image; without this chip the user sees
 * a whole-sample image stacked under a cropped map and has no way to tell.
 *
 * Renders nothing when the layer followed the crop, which is the normal case
 * and the case for every layer when no crop is active.
 */
import React from 'react';
import { useTranslation } from 'react-i18next';

export default function CropWarningChip({ status }) {
  const { t } = useTranslation();
  if (!status || status.cropped !== false) return null;
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
      <span>{t('ebsdviewer:crop.electronImageNotCropped')}</span>
    </div>
  );
}
