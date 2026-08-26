/**
 * What one region actually is.
 *
 * Sits under the map rather than in the right rail because the three things
 * worth reading — the full composition, who it touches, and which phases come
 * close — are columns, and stacking them in 300 px turns each into a scroll
 * box. The rail keeps the tools that change the grouping.
 *
 * Everything here is measured, nothing is decided:
 *
 * * enrichment is the factor over the map's OWN background, the same quantity
 *   the classifier gates on, so the readout and the classifier cannot disagree;
 * * spread says whether the region is homogeneous — one that is not is
 *   either two things or a gradient;
 * * connected pieces is how you see three concentric rings as one particle;
 * * the neighbour gap is the merge decision in a number.
 */
import React from 'react';
import { useTranslation } from 'react-i18next';

import { Button, Label, colors as C, alpha } from '../../theme/components';

/** Bar width for an enrichment factor, on a log scale clamped to [1/16, 16]. */
export function enrichmentBarPct(factor) {
  if (factor == null || !Number.isFinite(factor) || factor <= 0) return 50;
  const clamped = Math.max(1 / 16, Math.min(16, factor));
  // log2 in [-4, 4] -> [0, 100], so 1x sits exactly in the middle and
  // depletion reads as clearly as enrichment.
  return ((Math.log2(clamped) + 4) / 8) * 100;
}

/**
 * How far from the map's own background counts as "concentrated here".
 *
 * The depletion side is the RECIPROCAL, so a factor and its inverse are
 * judged alike: 1.3x more is as notable as 1.3x less. Exported because the
 * legend has to state the same numbers the colours use — it said 0.8 while
 * the code used 0.77, and a legend that disagrees with what it explains is
 * worse than no legend.
 */
export const ENRICHED_AT = 1.3;
export const DEPLETED_AT = Number((1 / ENRICHED_AT).toFixed(2));   // 0.77

/** Enriched, depleted, or neither — for colouring, not for judging. */
export function enrichmentKind(factor) {
  if (factor == null || !Number.isFinite(factor)) return 'unknown';
  if (factor >= ENRICHED_AT) return 'enriched';
  if (factor <= DEPLETED_AT) return 'depleted';
  return 'flat';
}

function Column({ title, hint, children, grow = 1 }) {
  return (
    <div style={{ flex: `${grow} 1 220px`, minWidth: 0 }}>
      <Label secondary small style={{ display: 'block' }}>
        {title}
      </Label>
      {/* Always visible. A tooltip only helps someone who already
          suspects there is something to learn. */}
      {hint && (
        <div style={{ fontSize: '7.5pt', color: C.textSecondary,
                      marginBottom: 3, lineHeight: 1.3 }}>
          {hint}
        </div>
      )}
      {children}
    </div>
  );
}

export default function RegionInspector({
  detail, loading, error, phaseIndex, onAssign, onSelectRegion, busy,
  onDefineFromRegion,
}) {
  const { t } = useTranslation('eds');

  if (error) {
    return <div style={{ fontSize: '8.5pt', color: C.red }}>{error}</div>;
  }
  if (loading && !detail) {
    return <Label secondary small>{t('inspector.loading')}</Label>;
  }
  if (!detail) {
    return <Label secondary small>{t('inspector.empty')}</Label>;
  }

  const els = detail.elements || [];
  const neighbours = detail.neighbours || [];
  const candidates = (detail.candidates || []).slice(0, 6);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {/* --- headline: size and how it is put together ------------------- */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8,
                    flexWrap: 'wrap' }}>
        <span style={{ width: 14, height: 14, borderRadius: 3, flexShrink: 0,
                       background: detail.color,
                       border: `1px solid ${C.border}` }} />
        <span style={{ fontSize: '9pt', color: C.text }}>
          {t('inspector.headline', {
            px: detail.n_pixels.toLocaleString(),
            pct: detail.percentage,
          })}
        </span>
        {detail.n_pieces != null && (
          <span
            style={{ fontSize: '8.5pt', color: C.textSecondary }}
            title={t('inspector.piecesTooltip')}
          >
            {t('inspector.pieces', { count: detail.n_pieces })}
            {detail.pieces?.length > 1 && (
              <span style={{ marginLeft: 4, opacity: 0.75 }}>
                ({detail.pieces.slice(0, 5).join(' · ')}
                {detail.pieces.length > 5 ? ' …' : ''})
              </span>
            )}
          </span>
        )}
      </div>

      <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap',
                    alignItems: 'flex-start' }}>
        {/* --- composition ---------------------------------------------- */}
        <Column title={t('inspector.composition')}
                hint={t('inspector.compositionHint')} grow={2}>
          <table style={{ width: '100%', borderCollapse: 'collapse',
                          fontSize: '8.5pt', fontVariantNumeric: 'tabular-nums' }}>
            <thead>
              <tr style={{ color: C.textSecondary }}>
                <th style={{ textAlign: 'left', fontWeight: 400 }}>
                  {t('inspector.element')}
                </th>
                <th style={{ textAlign: 'right', fontWeight: 400 }}>at%</th>
                <th style={{ textAlign: 'right', fontWeight: 400 }}
                    title={t('inspector.spreadTooltip')}>
                  ±
                </th>
                <th style={{ textAlign: 'left', fontWeight: 400, paddingLeft: 8 }}
                    title={t('inspector.enrichmentTooltip')}>
                  {t('inspector.enrichment')}
                </th>
              </tr>
            </thead>
            <tbody>
              {els.map((e) => {
                const kind = enrichmentKind(e.enrichment);
                const colour = kind === 'enriched' ? C.green
                  : kind === 'depleted' ? C.purple : C.textSecondary;
                return (
                  <tr key={e.element}>
                    <td style={{ color: C.text }}>{e.element}</td>
                    <td style={{ textAlign: 'right', color: C.text }}>
                      {e.at_pct.toFixed(1)}
                    </td>
                    <td style={{ textAlign: 'right', color: C.textSecondary }}>
                      {e.spread.toFixed(1)}
                    </td>
                    <td style={{ paddingLeft: 8, minWidth: 96 }}>
                      {e.enrichment == null ? (
                        <span style={{ color: C.textSecondary }}>—</span>
                      ) : (
                        <span style={{ display: 'flex', alignItems: 'center',
                                       gap: 5 }}>
                          {/* The 1x mark is fixed at the middle, so a glance
                              separates enrichment from depletion. */}
                          <span style={{ position: 'relative', flex: 1,
                                         height: 6, borderRadius: 3,
                                         background: alpha(C.border, 60) }}>
                            <span style={{
                              position: 'absolute', left: '50%', top: -2,
                              width: 1, height: 10,
                              background: alpha(C.textSecondary, 70),
                            }} />
                            <span style={{
                              position: 'absolute', top: 0, height: 6,
                              borderRadius: 3, background: colour,
                              left: `${Math.min(50, enrichmentBarPct(e.enrichment))}%`,
                              width: `${Math.abs(enrichmentBarPct(e.enrichment) - 50)}%`,
                            }} />
                          </span>
                          <span style={{ color: colour, minWidth: 40,
                                         textAlign: 'right' }}>
                            {e.enrichment.toFixed(2)}×
                          </span>
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div style={{ fontSize: '7.5pt', color: C.textSecondary,
                        marginTop: 3, lineHeight: 1.35 }}>
            {t('inspector.scaleLegend')}
          </div>
        </Column>

        {/* --- neighbours ----------------------------------------------- */}
        <Column title={t('inspector.neighbours')}
                hint={t('inspector.neighboursHint')}>
          {neighbours.length === 0 ? (
            <Label secondary small>{t('inspector.noNeighbours')}</Label>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {neighbours.map((n) => (
                <button
                  key={n.region_id}
                  type="button"
                  onClick={() => onSelectRegion?.(n.region_id)}
                  title={t('inspector.neighbourTooltip', {
                    gap: n.gap_at_pct.toFixed(1), edge: n.shared_edge_px,
                  })}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 6,
                    fontSize: '8.5pt', padding: '2px 4px', borderRadius: 3,
                    background: 'transparent', color: C.text,
                    border: '1px solid transparent', cursor: 'pointer',
                    textAlign: 'left',
                  }}
                >
                  <span style={{ width: 10, height: 10, borderRadius: 2,
                                 flexShrink: 0, background: n.color,
                                 border: `1px solid ${C.border}` }} />
                  <span style={{ flex: 1, fontVariantNumeric: 'tabular-nums' }}>
                    {t('inspector.neighbourGap', { gap: n.gap_at_pct.toFixed(1) })}
                  </span>
                  <span style={{ color: C.textSecondary }}>
                    {n.shared_edge_px} px
                  </span>
                </button>
              ))}
            </div>
          )}
        </Column>

        {/* --- candidates ----------------------------------------------- */}
        <Column title={t('inspector.candidates')}
                hint={t('inspector.candidatesHint')} grow={1.4}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            {candidates.map((c) => {
              const current = c.phase_index === phaseIndex;
              return (
                <button
                  key={c.phase_index}
                  type="button"
                  disabled={busy}
                  onClick={() => onAssign?.(c.phase_index)}
                  title={t('inspector.candidateTooltip', {
                    name: c.cif_filename, gap: c.gap_at_pct.toFixed(1),
                  })}
                  style={{
                    display: 'flex', alignItems: 'baseline', gap: 6,
                    fontSize: '8.5pt', padding: '2px 4px', borderRadius: 3,
                    background: current ? alpha(C.green, 20) : 'transparent',
                    color: C.text, border: '1px solid transparent',
                    cursor: busy ? 'wait' : 'pointer', textAlign: 'left',
                  }}
                >
                  <span style={{ flex: 1, overflow: 'hidden',
                                 textOverflow: 'ellipsis',
                                 whiteSpace: 'nowrap' }}>
                    {c.cif_filename}
                  </span>
                  <span style={{ color: C.textSecondary,
                                 fontVariantNumeric: 'tabular-nums' }}>
                    {c.gap_at_pct.toFixed(1)}
                  </span>
                </button>
              );
            })}
            {phaseIndex >= 0 && (
              <Button
                variant="secondary"
                disabled={busy}
                onClick={() => onAssign?.(-1)}
                style={{ marginTop: 3 }}
              >
                {t('inspector.clearName')}
              </Button>
            )}
          </div>
        </Column>
      </div>

      {/* The user's own suggestion was to make regions definable "in the
          Region Inspector". The editor is map-wide so it lives in its own
          tab, but the journey starts here: you are looking at the region
          the grouping got wrong, and this turns it into a window you can
          adjust. Without it the path is pick region -> change tab -> find
          the seed button, and nothing on this panel says so. */}
      {onDefineFromRegion && (
        <Button
          variant="secondary"
          disabled={busy}
          onClick={() => onDefineFromRegion(detail)}
          title={t('inspector.defineTooltip')}
          style={{ marginTop: 6, width: '100%' }}
        >
          {t('inspector.define')}
        </Button>
      )}
    </div>
  );
}
