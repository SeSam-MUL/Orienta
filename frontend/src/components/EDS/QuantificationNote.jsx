import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, alpha } from '../../theme/components';

/**
 * One line saying how provisional a displayed composition is.
 *
 * The backend has always known. `eds_utils` attaches a `.provenance` to every
 * wt%/at% result — which k-factor model produced it, what it was calibrated
 * on, that there is NO ZAF matrix correction, which lines sit outside the
 * calibrated response range, which windows are known to overlap. Until
 * 2026-09-12 the routes pulled the numbers out with `float(...)` and dropped
 * all of it, so the panel showed a composition with nothing to say it is
 * semi-quantitative. That is the recurring shape in this project: a warning
 * the backend builds correctly and nobody ever reads.
 *
 * Deliberately small and grey, with the details collapsed. The caveat is
 * permanent — it applies to every number this page shows — so a banner would
 * be noise within a day and get ignored, which is the same as not having it.
 * Two short lines stay open: where the intensities came from, and that
 * nothing here is calibrated on a measured standard. It only raises its voice
 * (amber) when there is something specific to say: a spectral overlap, an
 * extrapolated line, a beam voltage the calibration does not cover.
 */
/**
 * Where the intensities came from: `"spectrum"` (net line intensities from
 * the spectrum fit), `"window"` (AZtec's window integrals, background
 * included) or null for a provenance from before the source was recorded.
 *
 * The short `source` field first; the sentence `source_data`
 * (`eds_utils.quantification_provenance`) as the fallback, read by its
 * opening words.
 */
export function quantSourceOf(provenance) {
  const src = provenance?.source;
  if (src === 'spectrum' || src === 'window') return src;
  const data = String(provenance?.source_data || '');
  if (data.startsWith('spectrum fit')) return 'spectrum';
  if (data.startsWith('vendor')) return 'window';
  return null;
}

/** An angle for the caveat: one decimal, trailing .0 dropped (70, 81, 9.5). */
function fmtDeg(v) {
  const x = Number(v);
  return Number.isFinite(x) ? String(Math.round(x * 10) / 10) : '?';
}

export default function QuantificationNote({ provenance }) {
  const { t } = useTranslation('eds');
  const [open, setOpen] = useState(false);
  if (!provenance) return null;
  const source = quantSourceOf(provenance);
  // Lines the spectrum fit leaves a residual of more than 5 % of their area
  // at: a specific caveat about THESE numbers, so it counts like one.
  const quality = source === 'spectrum'
    ? (provenance.fit?.quality_warnings || []).filter((q) => q && q.family)
    : [];

  const assumedKv = provenance.beam_kv_assumed === true;
  // The backend puts the assumed-voltage note in `warnings` too, so that log
  // readers and API consumers see it; here it has a translated line of its
  // own, so drop the raw one rather than saying it twice. The prefix is the
  // backend's (`_note_assumed_voltage` in routes/eds.py).
  const warnings = (provenance.warnings || [])
    .filter((w) => !(assumedKv && typeof w === 'string'
      && w.startsWith('beam voltage ')));
  const overlaps = provenance.spectral_overlaps || [];
  const matrix = provenance.matrix_correction && typeof provenance.matrix_correction === 'object'
    ? provenance.matrix_correction : null;
  const hasSpecifics = warnings.length > 0 || overlaps.length > 0 || assumedKv
    || quality.length > 0;
  const tone = hasSpecifics ? colors.yellow : colors.textSecondary;

  return (
    <div
      style={{
        marginTop: 6,
        fontSize: '8pt',
        color: tone,
        borderTop: `1px solid ${alpha(colors.border, 40)}`,
        paddingTop: 5,
      }}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        style={{
          background: 'none',
          border: 'none',
          padding: 0,
          color: tone,
          cursor: 'pointer',
          font: 'inherit',
          textAlign: 'left',
        }}
      >
        {hasSpecifics ? '⚠ ' : 'ⓘ '}
        {t('quantNote.headline', {
          count: warnings.length + overlaps.length + (assumedKv ? 1 : 0)
            + quality.length,
        })}
        <span style={{ marginLeft: 6, opacity: 0.7 }}>{open ? '▴' : '▾'}</span>
      </button>

      {/* Where the numbers come from, always on screen rather than behind
          the click: a file without
          spectra gets a visible label (its numbers include the background), and the spectrum
          source is what makes a negative net count or a changed at% make
          sense. */}
      {source && (
        <div
          data-quant-source={source}
          style={{ marginTop: 3, color: colors.textSecondary, lineHeight: 1.4 }}
        >
          {t(source === 'spectrum' ? 'quantNote.sourceSpectrum' : 'quantNote.sourceWindow')}
        </div>
      )}

      {/* Also always on screen: no path here was calibrated on a measured
          standard, and the one fitted step (the detector response) belongs
          to one detector. Behind the click, nobody who quotes a number from
          this panel would know that. Grey, like the source line: it is true
          of every number, so it is not a warning. */}
      <div
        data-quant-uncalibrated
        style={{ marginTop: 3, color: colors.textSecondary, lineHeight: 1.4 }}
      >
        {t('quantNote.uncalibrated')}
      </div>

      {open && (
        <div style={{ marginTop: 5, lineHeight: 1.5 }}>
          {/* The standing caveat, always — it is true of every number here.
              Which one is the backend's call: the matrix-corrected spectrum
              source sends a `matrix_correction` block (eds_utils
              ._matrix_provenance), everything else the no-ZAF method. */}
          <div style={{ color: colors.textSecondary }}>
            {matrix
              ? t('quantNote.matrixCorrected', {
                tilt: fmtDeg(matrix.tilt_deg), psi: fmtDeg(matrix.exit_angle_deg),
              })
              : t('quantNote.noZaf')}
          </div>
          {provenance.calibration?.standard && (
            <div style={{ color: colors.textSecondary, marginTop: 3 }}>
              {matrix
                ? t('quantNote.calibratedAfterMatrix', {
                  standard: provenance.calibration.standard,
                  kv: provenance.calibration.beam_kv,
                })
                : t('quantNote.calibratedOn', {
                  matrix: provenance.calibration.matrix,
                  kv: provenance.calibration.beam_kv,
                })}
            </div>
          )}
          {provenance.calibration?.detector && (
            <div style={{ color: colors.textSecondary, marginTop: 3 }}>
              {t('quantNote.detectorOnly', { detector: provenance.calibration.detector })}
            </div>
          )}
          {/* Loudest of the specifics: every k-factor's overvoltage term
              rides on this number, and an assumed 20 kV that happens to be
              right is indistinguishable from a measured one until it is not. */}
          {assumedKv && (
            <div style={{ marginTop: 3, color: colors.yellow }}>
              {t('quantNote.beamAssumed', { kv: provenance.beam_kv })}
            </div>
          )}
          {overlaps.map((o) => (
            <div key={`${o.window?.join?.('-')}`} style={{ marginTop: 3 }}>
              {t('quantNote.overlap', {
                window: (o.window || []).join(' '),
                interferer: (o.interferer || []).join(' '),
              })}
            </div>
          ))}
          {quality.map((q) => (
            <div key={`fit-${q.family}`} style={{ marginTop: 3 }}>
              {t('spectrumFit.quality', {
                pct: Math.abs(100 * Number(q.residual_fraction || 0)).toFixed(1),
                line: q.family,
              })}
            </div>
          ))}
          {warnings.map((w) => (
            <div key={w} style={{ marginTop: 3 }}>• {w}</div>
          ))}
        </div>
      )}
    </div>
  );
}
