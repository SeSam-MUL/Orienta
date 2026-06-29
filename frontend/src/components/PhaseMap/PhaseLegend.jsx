import React, { useEffect, useState } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { phaseMapApi } from '../../services/api';
import { colors, spacing, CollapsibleGroup } from '../../theme/components';

/**
 * Phase Legend panel for the Phase Maps page (rebuilt 2026-05-26).
 *
 * Filters to PRESENT phases only (`pixel_count >= 1`) and decorates each
 * row with pixel-count, area-%, median CI. Phases flagged as `suspect`
 * by the backend (median CI < 0.5 AND area-% > 5%) get a warning icon
 * so the user can spot likely misindex candidates at a glance.
 *
 * Replaces the older "list every PhaseList entry regardless of whether
 * any pixel is indexed there" behaviour, which produced a 12-row legend
 * for a 3-phase result on a 12-phase input list.
 *
 * @param {string|number|null} resultId  Active indexing result id. The
 *   legend re-fetches whenever this changes; when null the empty state
 *   is shown.
 */
function PhaseLegend({ resultId }) {
  const { t } = useTranslation('phasemap');
  const [phases, setPhases] = useState([]);
  const [totals, setTotals] = useState({ total_indexed: 0, unindexed_count: 0 });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  // When ON, also list phases that were INPUT to indexing but ended up
  // with zero indexed pixels. Default off (the user's "I just want what
  // was identified" baseline).
  const [showEmpty, setShowEmpty] = useState(false);

  useEffect(() => {
    if (resultId == null) {
      setPhases([]);
      setTotals({ total_indexed: 0, unindexed_count: 0 });
      setError(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    phaseMapApi
      .phaseStats(showEmpty)
      .then((res) => {
        if (cancelled) return;
        setPhases(res.data?.phases || []);
        setTotals({
          total_indexed: res.data?.total_indexed || 0,
          unindexed_count: res.data?.unindexed_count || 0,
        });
      })
      .catch((err) => {
        if (cancelled) return;
        setError(
          err?.response?.data?.detail || err?.message || t('phasemap:phaseLegend.loadFailed'),
        );
        setPhases([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [resultId, showEmpty]);

  const Swatch = ({ hex }) => (
    <span
      style={{
        display: 'inline-block',
        width: 12,
        height: 12,
        flex: '0 0 12px',
        borderRadius: 2,
        background: hex,
        border: `1px solid ${colors.border}`,
      }}
    />
  );

  // One legend row. `suspect` adds a ⚠ icon + amber border-left to mark
  // likely-misindex candidates (high area but low CI). Empty rows
  // (pixel_count = 0, only shown when showEmpty is on) are dimmed.
  const Row = ({ phase }) => (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '4px 6px',
        fontSize: '9pt',
        color: phase.pixel_count > 0 ? colors.text : colors.textSecondary,
        borderLeft: phase.suspect
          ? `2px solid #f0b429`
          : '2px solid transparent',
        background: phase.suspect ? 'rgba(240, 180, 41, 0.06)' : 'transparent',
        marginBottom: 1,
        opacity: phase.pixel_count > 0 ? 1 : 0.5,
      }}
      title={
        phase.suspect
          ? t('phasemap:phaseLegend.suspectTooltip', { area: phase.area_pct, ci: phase.median_ci })
          : t('phasemap:phaseLegend.rowTooltip', {
              count: phase.pixel_count.toLocaleString(),
              area: phase.area_pct,
              ci: phase.median_ci != null
                ? t('phasemap:phaseLegend.rowTooltipCi', { ci: phase.median_ci })
                : '',
            })
      }
    >
      <Swatch hex={phase.color_hex} />
      <span
        style={{
          flex: 1,
          minWidth: 0,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
      >
        {phase.name}
        {phase.suspect && (
          <span style={{ color: '#f0b429', marginLeft: 6 }}>⚠</span>
        )}
      </span>
      <span
        style={{
          flex: '0 0 auto',
          color: colors.textSecondary,
          fontVariantNumeric: 'tabular-nums',
          fontSize: '8.5pt',
        }}
      >
        {phase.area_pct}%
      </span>
      {phase.median_ci != null && (
        <span
          style={{
            flex: '0 0 auto',
            color: colors.textSecondary,
            fontVariantNumeric: 'tabular-nums',
            fontSize: '8.5pt',
            minWidth: 56,
            textAlign: 'right',
          }}
        >
          Ø {phase.median_ci.toFixed(2)}
        </span>
      )}
    </div>
  );

  let body;
  if (resultId == null) {
    body = (
      <div style={{ fontSize: '8.5pt', color: colors.textSecondary }}>
        {t('phasemap:phaseLegend.loadPrompt')}
      </div>
    );
  } else if (loading) {
    body = (
      <div style={{ fontSize: '8.5pt', color: colors.textSecondary }}>
        {t('phasemap:phaseLegend.loading')}
      </div>
    );
  } else if (error) {
    body = <div style={{ fontSize: '8.5pt', color: colors.red }}>{error}</div>;
  } else if (phases.length === 0) {
    body = (
      <div style={{ fontSize: '8.5pt', color: colors.textSecondary }}>
        {t('phasemap:phaseLegend.noIndexedPhases')}
      </div>
    );
  } else {
    const suspectCount = phases.filter((p) => p.suspect).length;
    body = (
      <div>
        <div
          style={{
            fontSize: '8pt',
            color: colors.textSecondary,
            padding: '0 2px 4px',
            display: 'flex',
            justifyContent: 'space-between',
          }}
        >
          <span>
            {t('phasemap:phaseLegend.present', {
              count: phases.length,
              px: totals.total_indexed.toLocaleString(),
            })}
          </span>
          {suspectCount > 0 && (
            <span style={{ color: '#f0b429' }}>
              {t('phasemap:phaseLegend.suspectCount', { count: suspectCount })}
            </span>
          )}
        </div>
        <div
          style={{
            maxHeight: 320,
            overflowY: 'auto',
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {phases.map((p) => (
            <Row key={p.phase_id} phase={p} />
          ))}
        </div>
        <div
          style={{
            fontSize: '7.5pt',
            color: colors.textSecondary,
            lineHeight: 1.45,
            padding: '6px 2px 0',
            marginTop: 4,
            borderTop: `1px solid ${colors.border}`,
          }}
        >
          <Trans
            i18nKey="phasemap:phaseLegend.footnote"
            components={{ 1: <strong />, 3: <em />, 5: <em />, 7: <strong />, 9: <strong /> }}
          />
        </div>
      </div>
    );
  }

  return (
    <CollapsibleGroup title={t('phasemap:phaseLegend.title')}>
      <div style={{ marginTop: spacing.innerSpacing }}>
        {resultId != null && (
          <label
            style={{
              display: 'flex', alignItems: 'center', gap: 6,
              fontSize: '8.5pt', color: colors.textSecondary,
              cursor: 'pointer', marginBottom: 6,
            }}
            title={t('phasemap:phaseLegend.showEmptyTooltip')}
          >
            <input
              type="checkbox"
              checked={showEmpty}
              onChange={(e) => setShowEmpty(e.target.checked)}
              style={{ cursor: 'pointer' }}
            />
            {t('phasemap:phaseLegend.showEmpty')}
          </label>
        )}
        {body}
      </div>
    </CollapsibleGroup>
  );
}

export default PhaseLegend;
