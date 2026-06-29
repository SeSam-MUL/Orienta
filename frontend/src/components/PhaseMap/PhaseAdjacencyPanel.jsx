import React, { useEffect, useState, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { phaseMapApi } from '../../services/api';
import { colors, spacing, CollapsibleGroup } from '../../theme/components';

/**
 * Phase Adjacency / Confusion Matrix panel (2026-05-26).
 *
 * For every pair of indexed phases (a, b) shows how often they share
 * a 4-neighbour pixel border. High counts on chemically-similar phase
 * pairs (e.g. Al4FeSi ↔ Al3Fe2Si) are a hallmark of cubic-lattice
 * degeneracy + likely misindex rather than a real phase interface.
 *
 * Rendering: NxN cell grid coloured by log1p(count). Diagonal greyed
 * out. Cell hover shows: "A ↔ B: N borders (P% of A's borders)".
 *
 * Refetches whenever the active indexing result changes (driven by
 * `resultId` prop — null disables the panel).
 */
function PhaseAdjacencyPanel({ resultId }) {
  const { t } = useTranslation('phasemap');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hoverCell, setHoverCell] = useState(null);

  useEffect(() => {
    if (resultId == null) {
      setData(null);
      setError(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    phaseMapApi
      .phaseAdjacency()
      .then((res) => {
        if (cancelled) return;
        setData(res.data || null);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(
          err?.response?.data?.detail || err?.message || t('phasemap:adjacency.loadFailed'),
        );
        setData(null);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [resultId]);

  // log1p-normalised colour scale: max value -> deep red, 0 -> background.
  // Uses HSL so high counts pop. Symmetric matrix so colour map is shared.
  const { maxLog, cellHsl } = useMemo(() => {
    if (!data?.adjacency || data.adjacency.length === 0) {
      return { maxLog: 0, cellHsl: (() => null) };
    }
    let maxVal = 0;
    for (const row of data.adjacency) {
      for (const v of row) {
        if (v > maxVal) maxVal = v;
      }
    }
    const mLog = Math.log1p(maxVal);
    return {
      maxLog: mLog,
      cellHsl: (v) => {
        if (v <= 0 || mLog <= 0) return null;
        const t = Math.log1p(v) / mLog;
        // Cold (low) -> hot (high): blue -> cyan -> yellow -> red
        // Simple lerp through HSL hue 240 -> 0
        const hue = 240 - 240 * t;
        const sat = 75;
        const light = 65 - 25 * t; // darker for higher values, more readable
        return `hsl(${hue.toFixed(0)}, ${sat}%, ${light.toFixed(0)}%)`;
      },
    };
  }, [data]);

  let body;
  if (resultId == null) {
    body = (
      <div style={{ fontSize: '8.5pt', color: colors.textSecondary }}>
        {t('phasemap:adjacency.loadPrompt')}
      </div>
    );
  } else if (loading) {
    body = (
      <div style={{ fontSize: '8.5pt', color: colors.textSecondary }}>
        {t('phasemap:adjacency.loading')}
      </div>
    );
  } else if (error) {
    body = <div style={{ fontSize: '8.5pt', color: colors.red }}>{error}</div>;
  } else if (!data?.phases || data.phases.length < 2) {
    body = (
      <div style={{ fontSize: '8.5pt', color: colors.textSecondary }}>
        {t('phasemap:adjacency.needTwoPhases')}
      </div>
    );
  } else {
    const phases = data.phases;
    const adj = data.adjacency;
    const rowSums = data.border_total_per_phase || [];
    const n = phases.length;
    // Truncate row/col headers so 12-phase labels don't take 60% of the panel
    const truncate = (s, len) =>
      s == null ? '?' : (s.length > len ? s.slice(0, len - 1) + '…' : s);

    body = (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <div style={{ fontSize: '8pt', color: colors.textSecondary }}>
          {t('phasemap:adjacency.summary', { borders: data.total_borders.toLocaleString(), phases: n })}
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ borderCollapse: 'collapse', fontSize: '7.5pt' }}>
            <thead>
              <tr>
                <th style={{ padding: 2 }} />
                {phases.map((p) => (
                  <th
                    key={p.id}
                    title={p.name}
                    style={{
                      writingMode: 'vertical-rl',
                      transform: 'rotate(180deg)',
                      padding: '4px 2px',
                      color: colors.textSecondary,
                      fontWeight: 400,
                      maxHeight: 70,
                    }}
                  >
                    {truncate(p.name, 14)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {phases.map((rowPhase, i) => (
                <tr key={rowPhase.id}>
                  <th
                    title={rowPhase.name}
                    style={{
                      padding: '2px 6px',
                      textAlign: 'right',
                      color: colors.textSecondary,
                      fontWeight: 400,
                      maxWidth: 90,
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                    }}
                  >
                    {truncate(rowPhase.name, 14)}
                  </th>
                  {phases.map((colPhase, j) => {
                    const v = adj[i]?.[j] ?? 0;
                    const isDiag = i === j;
                    const isHover =
                      hoverCell && hoverCell.row === i && hoverCell.col === j;
                    const bg = isDiag
                      ? '#2a2d3a'
                      : (cellHsl(v) || '#1f2330');
                    const pct =
                      rowSums[i] > 0 && !isDiag ? (v / rowSums[i]) * 100 : 0;
                    return (
                      <td
                        key={colPhase.id}
                        onMouseEnter={() =>
                          setHoverCell({ row: i, col: j, v, pct })
                        }
                        onMouseLeave={() => setHoverCell(null)}
                        title={
                          isDiag
                            ? t('phasemap:adjacency.diagonalTooltip', { name: rowPhase.name })
                            : t('phasemap:adjacency.cellTooltip', { a: rowPhase.name, b: colPhase.name, count: v.toLocaleString() }) +
                              (rowSums[i] > 0
                                ? t('phasemap:adjacency.cellTooltipPct', { pct: pct.toFixed(1), name: rowPhase.name })
                                : '')
                        }
                        style={{
                          background: bg,
                          width: 22,
                          height: 22,
                          minWidth: 22,
                          textAlign: 'center',
                          color: isDiag ? '#3a3f4f' : '#0b0d12',
                          border: isHover
                            ? `1px solid ${colors.accent}`
                            : `1px solid #11131a`,
                          fontVariantNumeric: 'tabular-nums',
                          fontSize: '7pt',
                          cursor: isDiag ? 'default' : 'pointer',
                        }}
                      >
                        {isDiag ? '·' : v > 0 ? v : ''}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {hoverCell && hoverCell.row !== hoverCell.col && (
          <div
            style={{
              fontSize: '8.5pt',
              color: colors.text,
              background: colors.bg,
              border: `1px solid ${colors.border}`,
              borderRadius: 3,
              padding: '4px 8px',
              marginTop: 4,
            }}
          >
            <strong>{phases[hoverCell.row].name}</strong>{' ↔ '}
            <strong>{phases[hoverCell.col].name}</strong>:{' '}
            {t('phasemap:adjacency.hoverBorders', { count: hoverCell.v.toLocaleString() })}
            {rowSums[hoverCell.row] > 0 && (
              <span style={{ color: colors.textSecondary, marginLeft: 6 }}>
                {t('phasemap:adjacency.hoverPct', { pct: hoverCell.pct.toFixed(1), name: phases[hoverCell.row].name })}
              </span>
            )}
          </div>
        )}
        <div
          style={{
            fontSize: '7.5pt',
            color: colors.textSecondary,
            fontStyle: 'italic',
          }}
        >
          {t('phasemap:adjacency.tip')}
        </div>
      </div>
    );
  }

  return (
    <CollapsibleGroup title={t('phasemap:adjacency.title')}>
      <div style={{ marginTop: spacing.innerSpacing }}>{body}</div>
    </CollapsibleGroup>
  );
}

export default PhaseAdjacencyPanel;
