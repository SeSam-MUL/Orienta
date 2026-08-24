/**
 * "11 structures → 4 phases", with the phases themselves.
 *
 * The structure list names a phase once per STRUCTURE, so a phase that covers
 * four structures is written four times. That reads as a bug — the user's
 * words: "die tauchen dann ca 100 mal in der Liste auf" — when it is actually
 * the point: many regions, few phases. This makes the collapse visible, and
 * makes the phase view discoverable, which is the layer where two neighbouring
 * regions of one phase become one uninterrupted area.
 */
import React, { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { Label, colors as C, alpha } from '../../theme/components';

/**
 * Fold the structure list down to the phases it carries.
 *
 * Keyed on the phase INDEX, not the name: two library entries can share a
 * display name, and merging them here would claim they are one phase when the
 * map treats them as two.
 */
export function rollUpPhases(structures, allPhases) {
  const byIndex = new Map();
  (structures || []).forEach((s) => {
    const idx = s.phase_index;
    if (idx == null || idx < 0) return;
    const prev = byIndex.get(idx);
    if (prev) {
      prev.n_structures += 1;
      prev.n_pixels += s.n_pixels || 0;
    } else {
      byIndex.set(idx, {
        phase_index: idx,
        cif_filename: s.cif_filename,
        n_structures: 1,
        n_pixels: s.n_pixels || 0,
      });
    }
  });
  // The colour has to come from the phase list, not from a structure: the
  // structure palette is a different scheme on purpose.
  const colourOf = new Map((allPhases || []).map((p) => [p.phase_index, p.color]));
  return [...byIndex.values()]
    .map((p) => ({ ...p, color: colourOf.get(p.phase_index) || '#3c3c3c' }))
    .sort((a, b) => b.n_pixels - a.n_pixels);
}

/** How many structures carry no phase yet. */
export function countUnnamed(structures) {
  return (structures || []).filter((s) => s.phase_index == null || s.phase_index < 0).length;
}

export default function PhaseRollup({ structures, allPhases, onShowPhases, totalPixels }) {
  const { t } = useTranslation('eds');
  const phases = useMemo(() => rollUpPhases(structures, allPhases),
    [structures, allPhases]);
  const unnamed = countUnnamed(structures);

  if (!structures?.length) return null;

  return (
    <div style={{ marginTop: 6, padding: '5px 6px', borderRadius: 4,
                  border: `1px solid ${alpha(C.cyan, 30)}` }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 6,
                    flexWrap: 'wrap' }}>
        <span style={{ fontSize: '9pt', color: C.text }}>
          {t('rollup.headline', {
            structures: structures.length,
            phases: phases.length,
          })}
        </span>
        {unnamed > 0 && (
          <span style={{ fontSize: '8pt', color: C.orange || '#f0b429' }}
                title={t('rollup.unnamedTooltip')}>
            {t('rollup.unnamed', { count: unnamed })}
          </span>
        )}
      </div>

      <Label secondary small style={{ display: 'block', marginTop: 2 }}>
        {t('rollup.explain')}
      </Label>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 1,
                    marginTop: 4 }}>
        {phases.map((p) => (
          <div
            key={p.phase_index}
            style={{ display: 'flex', alignItems: 'center', gap: 6,
                     fontSize: '8.5pt' }}
            title={t('rollup.rowTooltip', {
              name: p.cif_filename,
              structures: p.n_structures,
              px: p.n_pixels.toLocaleString(),
            })}
          >
            <span style={{ width: 12, height: 12, borderRadius: 2,
                           flexShrink: 0, background: p.color,
                           border: `1px solid ${C.border}` }} />
            <span style={{ flex: 1, minWidth: 0, color: C.text,
                           overflow: 'hidden', textOverflow: 'ellipsis',
                           whiteSpace: 'nowrap' }}>
              {p.cif_filename}
            </span>
            <span style={{ color: C.textSecondary, flexShrink: 0,
                           fontVariantNumeric: 'tabular-nums' }}>
              {t('rollup.fromStructures', { count: p.n_structures })}
            </span>
            {totalPixels > 0 && (
              <span style={{ color: C.textSecondary, flexShrink: 0, minWidth: 42,
                             textAlign: 'right',
                             fontVariantNumeric: 'tabular-nums' }}>
                {(p.n_pixels / totalPixels * 100).toFixed(1)}%
              </span>
            )}
          </div>
        ))}
      </div>

      <button
        type="button"
        onClick={onShowPhases}
        title={t('rollup.showPhasesTooltip')}
        style={{ marginTop: 5, width: '100%', fontSize: '8.5pt',
                 padding: '3px 6px', borderRadius: 3, cursor: 'pointer',
                 background: alpha(C.cyan, 15), color: C.text,
                 border: `1px solid ${alpha(C.cyan, 45)}` }}
      >
        {t('rollup.showPhases')}
      </button>
    </div>
  );
}
