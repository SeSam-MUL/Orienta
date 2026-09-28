/**
 * The band view: one band per element, a phase in every band it belongs to.
 *
 * SELECTION IS KEYED ON THE PHASE, NOT ON THE ROW, and that is the whole
 * reason this component exists rather than three nested loops in the page.
 * An Al-Fe-Si phase is on screen three times; ticking it in the Al band and
 * finding it unticked in the Fe band would say the two cards are different
 * phases. They are one phase in three systems -- that is what the view is
 * for -- so the tick follows the phase.
 *
 * Bands collapse, and collapsing is per band and remembered while the page
 * lives. A collapsed band still shows its count: the count is the reason to
 * open it.
 *
 * What a selection is FOR arrives with the groups -- adding to one, using it
 * for a run. Until then it is its own visible, undoable state and nothing
 * else, which is better than a control that looks like it does something.
 */
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, spacing } from '../../theme/components';
import PhaseRow, { phaseLabel } from './PhaseRow';
import IdentityLine from './IdentityLine';

const S = {
  collapseAll: {
    background: 'transparent', border: 'none', color: colors.purple,
    cursor: 'pointer', fontSize: '8pt', padding: '0 0 6px', marginLeft: 'auto',
    display: 'block',
  },
  band: { marginBottom: spacing.groupMargin },
  head: { display: 'flex', alignItems: 'baseline', gap: spacing.innerSpacing,
          width: '100%', textAlign: 'left', padding: '4px 6px',
          background: 'transparent', border: 'none', borderBottom: `1px solid ${colors.border}`,
          color: colors.text, fontSize: '10pt', fontWeight: 600, cursor: 'pointer' },
  caret: { width: 10, color: colors.textSecondary, fontSize: '8pt' },
  bandCount: { marginLeft: 'auto', fontSize: '8pt', color: colors.textSecondary },
};

export default function PhaseBands({
  bands, selected, onToggleSelect, whyFor, onReveal = null, loadCard = null,
  dragPayload = null, onNamed = null, siblingsFor = null, groupsFor = null,
}) {
  const { t } = useTranslation('phaselibrary');
  const [collapsed, setCollapsed] = useState(() => new Set());

  const toggleBand = (element) => setCollapsed((cur) => {
    const next = new Set(cur);
    if (next.has(element)) next.delete(element); else next.add(element);
    return next;
  });

  const allCollapsed = bands.length > 0 && bands.every((b) => collapsed.has(b.element));

  return (
    <div data-testid="phase-bands">
      {/* MEASURED BY A KEYBOARD USER, and it was the worst thing on the page:
          reaching the Mg band from the top costs 65 Tab stops, and crossing
          the whole list about 110 for 36 phases -- the duplication that makes
          the view useful triples the cost of leaving it. Collapsing every
          band turns the list into a nine-line table of contents, which he
          called "genuinely good design and I'd use it every session". It
          took him 18 keystrokes and nothing said it was possible. */}
      {bands.length > 1 && (
        <button
          type="button"
          style={S.collapseAll}
          onClick={() => setCollapsed(allCollapsed
            ? new Set()
            : new Set(bands.map((b) => b.element)))}
        >
          {t(allCollapsed ? 'bands.expandAll' : 'bands.collapseAll')}
        </button>
      )}
      {bands.map((band) => {
        const isOpen = !collapsed.has(band.element);
        return (
          <section key={band.element} style={S.band} data-band={band.element}>
            <button
              type="button"
              style={S.head}
              aria-expanded={isOpen}
              // "Al 28" is on this page TWICE -- as this band header and as
              // the element chip that filters by aluminium. Same name, same
              // number, two different actions; from a screen reader's list
              // of buttons they were indistinguishable. The accessible name
              // says which this is, the visible text stays short.
              aria-label={t('bands.header', {
                element: band.element, count: band.phases.length,
              })}
              onClick={() => toggleBand(band.element)}
            >
              {/* aria-hidden: `aria-expanded` already carries the state, so
                  the glyph is either redundant or read out as a symbol. */}
              <span style={S.caret} aria-hidden="true">{isOpen ? '▾' : '▸'}</span>
              <span>{band.element}</span>
              {/* Shown while collapsed too: the count is the reason to open. */}
              <span style={S.bandCount}>{band.phases.length}</span>
            </button>

            {isOpen && band.phases.map((p) => (
              <PhaseRow
                key={p.key}
                phase={p}
                label={phaseLabel(p)}
                why={(whyFor && whyFor(p.key)) || []}
                checked={selected.has(p.key)}
                onToggle={onToggleSelect}
                // The band is in the name: a phase in three systems has
                // three checkboxes, and from a checkbox list they were three
                // identical entries -- meeting the second one already ticked
                // reads as a double-press rather than as the same phase.
                selectLabel={t('selectPhaseInBand', {
                  name: phaseLabel(p), band: band.element,
                })}
                whyText={(k) => t(`why.${k}`)}
                detail={<IdentityLine phase={p} compact />}
                withCard
                onReveal={onReveal}
                loadCard={loadCard}
                dragPayload={dragPayload}
                onNamed={onNamed}
                siblingsFor={siblingsFor}
                groupsFor={groupsFor}
                indented
              />
            ))}
          </section>
        );
      })}
    </div>
  );
}
