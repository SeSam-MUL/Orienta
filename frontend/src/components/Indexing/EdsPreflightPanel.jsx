/**
 * EdsPreflightPanel.jsx
 *
 * "Can the EDS chemistry of the loaded file actually be used for these phases?"
 * Renders the POST /api/indexing/eds-preflight response.
 *
 * Two things this panel exists to say, that a plain `has_eds` boolean could not:
 *
 *  1. A grid mismatch between the EDS map and the EBSD navigation grid is
 *     BLOCKING — every pattern would be paired with the wrong pixel's chemistry.
 *     The caller disables Start on it.
 *  2. `max_area_pct` is an UPPER BOUND on a phase's area, never a prediction.
 *     The mask behind it is "chemistry does not veto this phase here"; the true
 *     area can only be smaller. For a single-element phase (`weak_bound`) the
 *     bound is trivially weak — trace amounts of that element sit everywhere —
 *     so those numbers are de-emphasised instead of read as a result.
 */

import { useTranslation } from 'react-i18next';
import { colors as C } from '../../theme/tokens';

const CHECK_ORDER = ['eds_present', 'grid', 'signal', 'coverage'];

const SEVERITY_COLOR = { ok: C.green, warn: C.yellow, block: C.red };
const SEVERITY_GLYPH = { ok: '✓', warn: '⚠', block: '✕' };

function severityColor(sev) {
  return SEVERITY_COLOR[sev] || C.textSecondary;
}

/** Small ok/warn/block dot in front of a row. */
function SeverityMark({ severity }) {
  const color = severityColor(severity);
  return (
    <span
      aria-hidden="true"
      style={{
        color,
        fontSize: '9pt',
        fontWeight: 700,
        lineHeight: 1.3,
        flexShrink: 0,
        width: 12,
        textAlign: 'center',
      }}
    >
      {SEVERITY_GLYPH[severity] || '·'}
    </span>
  );
}

/** "Fe 8, Mn 8, Al 75, Si 10" from the backend's `defining` map. */
function formatDefining(defining) {
  const entries = Object.entries(defining || {});
  if (entries.length === 0) return null;
  return entries
    .map(([el, pct]) => `${el} ${Number(pct).toFixed(0)}`)
    .join(', ');
}

export default function EdsPreflightPanel({
  result = null,
  loading = false,
  error = null,
  phaseLabels = {},
}) {
  const { t } = useTranslation('indexing');

  // Fetch failed → say so and get out of the way. A broken check must never
  // stop the user from indexing (the backend still guards the run itself).
  if (error && !result) {
    return (
      <div style={boxStyle(C.border)}>
        <div style={{ fontSize: '8.5pt', color: C.textSecondary }}>
          {t('edsPrior.unavailable', { error })}
        </div>
      </div>
    );
  }

  if (loading && !result) {
    return (
      <div style={boxStyle(C.border)}>
        <div style={{ fontSize: '8.5pt', color: C.textSecondary }}>
          {t('edsPrior.checking')}
        </div>
      </div>
    );
  }

  if (!result) return null;

  const checks = result.checks || {};
  const phases = result.phases || [];
  const blocked = result.can_index === false;
  const anyWeak = phases.some((p) => p.weak_bound && p.max_area_pct != null);

  return (
    <div style={{ ...boxStyle(blocked ? C.red : C.border), opacity: loading ? 0.6 : 1 }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: 5,
        }}
      >
        <span style={{ fontSize: '8.5pt', fontWeight: 600, color: C.text }}>
          {t('edsPrior.preflightTitle')}
        </span>
        {loading && (
          <span style={{ fontSize: '8pt', color: C.textSecondary }}>
            {t('edsPrior.checking')}
          </span>
        )}
      </div>

      {/* Blocking banner — prominent, and it names the consequence. */}
      {blocked && (
        <div
          role="alert"
          style={{
            fontSize: '8.5pt',
            color: C.red,
            background: `${C.red}14`,
            border: `1px solid ${C.red}55`,
            borderRadius: 4,
            padding: '5px 8px',
            marginBottom: 6,
            lineHeight: 1.4,
          }}
        >
          <strong>{t('edsPrior.blocked')}</strong>
          <div style={{ color: C.text, marginTop: 2 }}>{t('edsPrior.blockedHint')}</div>
        </div>
      )}

      {/* ---- The four checks ---- */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {CHECK_ORDER.filter((key) => checks[key]).map((key) => {
          const c = checks[key];
          return (
            <div
              key={key}
              style={{ display: 'flex', alignItems: 'flex-start', gap: 5, fontSize: '8.5pt' }}
            >
              <SeverityMark severity={c.severity} />
              <span style={{ color: C.text, flexShrink: 0, minWidth: 92 }}>
                {t(`edsPrior.checks.${key}`)}
              </span>
              <span style={{ color: severityColor(c.severity) === C.green ? C.textSecondary : severityColor(c.severity), lineHeight: 1.35 }}>
                {c.detail}
                {key === 'grid' && c.severity === 'block' && c.eds_shape && c.ebsd_shape && (
                  <span style={{ color: C.textSecondary }}>
                    {' '}
                    ({t('edsPrior.gridShapes', {
                      eds: c.eds_shape.join('×'),
                      ebsd: c.ebsd_shape.join('×'),
                    })})
                  </span>
                )}
              </span>
            </div>
          );
        })}
      </div>

      {/* ---- Per-phase table ---- */}
      {phases.length > 0 && (
        <div style={{ marginTop: 7 }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '8pt' }}>
            <thead>
              <tr style={{ color: C.textSecondary, textAlign: 'left' }}>
                <th style={thStyle}>{t('edsPrior.colPhase')}</th>
                <th style={thStyle}>{t('edsPrior.colDefining')}</th>
                <th style={{ ...thStyle, textAlign: 'right' }}>{t('edsPrior.colArea')}</th>
              </tr>
            </thead>
            <tbody>
              {phases.map((p, i) => {
                const label = phaseLabels[p.path] || p.name || '—';
                const defining = formatDefining(p.defining);
                const weak = !!p.weak_bound;
                const hasBound = p.max_area_pct != null;
                return (
                  <tr key={p.path || i} style={{ borderTop: `1px solid ${C.border}` }}>
                    <td style={{ ...tdStyle, color: C.text }} title={p.detail || ''}>
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                        <SeverityMark severity={p.severity} />
                        <span
                          style={{
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                            maxWidth: 150,
                            display: 'inline-block',
                          }}
                        >
                          {label}
                        </span>
                      </span>
                    </td>
                    <td style={{ ...tdStyle, color: C.textSecondary, fontFamily: 'monospace' }}>
                      {defining || '—'}
                      {(p.missing_elements || []).length > 0 && (
                        <span style={{ color: C.yellow }}>
                          {' '}
                          {t('edsPrior.notMeasured', {
                            elements: p.missing_elements.join(', '),
                          })}
                        </span>
                      )}
                    </td>
                    <td
                      style={{
                        ...tdStyle,
                        textAlign: 'right',
                        whiteSpace: 'nowrap',
                        // A single-element phase's bound is trivially weak —
                        // show it, but never at the same visual weight as a
                        // multi-element intermetallic's bound.
                        color: !hasBound
                          ? C.textSecondary
                          : weak
                            ? C.textSecondary
                            : p.max_area_pct <= 0
                              ? C.yellow
                              : C.text,
                        opacity: weak ? 0.65 : 1,
                        fontStyle: weak ? 'italic' : 'normal',
                      }}
                      title={weak ? t('edsPrior.weakBoundTip') : p.detail || ''}
                    >
                      {hasBound
                        ? t('edsPrior.atMost', { pct: p.max_area_pct.toFixed(2) })
                        : t('edsPrior.notJudgeable')}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          <div style={{ fontSize: '7.5pt', color: C.textSecondary, marginTop: 4, lineHeight: 1.4 }}>
            {t('edsPrior.upperBoundNote')}
            {anyWeak && <> {t('edsPrior.weakBoundNote')}</>}
          </div>
        </div>
      )}

      {phases.length === 0 && (
        <div style={{ fontSize: '8pt', color: C.textSecondary, marginTop: 6 }}>
          {t('edsPrior.noPhases')}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

function boxStyle(borderColor) {
  return {
    marginTop: 6,
    padding: '7px 9px',
    border: `1px solid ${borderColor}`,
    borderRadius: 4,
    background: 'rgba(255,255,255,0.03)',
  };
}

const thStyle = { padding: '0 4px 3px 0', fontWeight: 600 };
const tdStyle = { padding: '3px 4px 3px 0', verticalAlign: 'top' };
