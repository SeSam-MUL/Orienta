/**
 * What the spectrum fit found that the page would otherwise not say.
 *
 * Two things, both from the `fit` block of `GET /api/eds/spectrum-fit/status`
 * (`eds_spectrum_service._summary`):
 *
 *  - `detected_not_exported` -- elements the SPECTRUM shows (a K line at
 *    z >= 5) that AZtec did not export as a map. They are NOT part of the
 *    composition (a sum-level detection is
 *    counting noise per pixel, and per-pixel at% would clip it into a
 *    positive bias on every pixel). Said here with the family, the
 *    significance, its at% in the whole map were it added
 *    (`at_pct_map_sum`), and a way to put its net-count map on the overlay;
 *  - `quality_warnings` -- lines the fit leaves a residual of more than 5 % of
 *    the line's net area at. A warning, never a block.
 *
 * Chip style of `SuggestLibrarySkipped`. Renders nothing when the fit has
 * nothing to report: a box after every clean fit teaches the eye to skip it.
 */
import React from 'react';
import { useTranslation } from 'react-i18next';

import { colors, alpha } from '../../theme/components';

/** The label the backend gives a fitted family AZtec did not export
 *  (`_GENERATED_LABEL` in eds_spectrum_service). */
const GENERATED = { K: 'Kα1', L: 'Lα1,2', M: 'Mα' };

/**
 * The map label of a detected element: the one the backend sends with it
 * (`label`), else the line in `fit.lines`, else the label the backend
 * generates for the family. It is one of `/api/eds/elements`' `detected_only`.
 */
export function detectedLabel(d, lines) {
  if (typeof d?.label === 'string' && d.label) return d.label;
  const fromLines = lines && typeof lines === 'object' ? lines[d?.element]?.label : null;
  if (fromLines) return fromLines;
  return `${d?.element} ${GENERATED[d?.family] || GENERATED.K}`;
}

/** "Ti K (5σ, 0.12 at%)" -- family, the significance the detection rule
 *  (z >= 5) was applied to, and its share of the whole map were it added. */
export function describeDetected(d) {
  const head = d?.family ? `${d.element} ${d.family}` : String(d?.element);
  const parts = [];
  const z = d?.z;
  if (typeof z === 'number' && Number.isFinite(z)) parts.push(`${Math.round(z)}σ`);
  const at = d?.at_pct_map_sum;
  if (typeof at === 'number' && Number.isFinite(at)) parts.push(`${at.toFixed(2)} at%`);
  return parts.length ? `${head} (${parts.join(', ')})` : head;
}

function chip(color) {
  return {
    fontSize: '8pt',
    color,
    background: alpha(color, 10),
    border: `1px solid ${alpha(color, 25)}`,
    borderRadius: 3,
    padding: '1px 6px',
    fontWeight: 600,
  };
}

export default function SpectrumFitNotice({ fit, onAddMap }) {
  const { t } = useTranslation('eds');
  const detected = Array.isArray(fit?.detected_not_exported)
    ? fit.detected_not_exported.filter((d) => d && d.element) : [];
  const quality = Array.isArray(fit?.quality_warnings)
    ? fit.quality_warnings.filter((q) => q && q.family) : [];
  if (detected.length === 0 && quality.length === 0) return null;

  return (
    <div
      data-spectrum-fit-notice
      role="status"
      style={{ display: 'flex', flexDirection: 'column', gap: 4, flexShrink: 0 }}
    >
      {detected.length > 0 && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
          <span style={chip(colors.cyan)}>
            {t('spectrumFit.detected', {
              list: detected.map(describeDetected).join(', '),
            })}
          </span>
          {typeof onAddMap === 'function' && detected.map((d) => {
            const label = detectedLabel(d, fit?.lines);
            return (
              <button
                key={label}
                type="button"
                data-add-map={label}
                onClick={() => onAddMap(label)}
                title={t('spectrumFit.addMapTooltip', { element: label })}
                style={{
                  ...chip(colors.cyan),
                  cursor: 'pointer',
                  background: 'transparent',
                }}
              >
                {t('spectrumFit.addMap')}: {d.element}
              </button>
            );
          })}
        </div>
      )}
      {quality.map((q) => (
        <div
          key={q.family}
          data-fit-quality={q.family}
          style={{ ...chip(colors.orange), alignSelf: 'flex-start' }}
        >
          {t('spectrumFit.quality', {
            pct: Math.abs(100 * Number(q.residual_fraction || 0)).toFixed(1),
            line: q.family,
          })}
        </div>
      ))}
    </div>
  );
}
