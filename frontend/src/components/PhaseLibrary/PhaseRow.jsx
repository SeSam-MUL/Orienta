/**
 * One phase, wherever it is shown.
 *
 * ONE COMPONENT FOR BOTH VIEWS, because the first version had two and they
 * disagreed: a phase was selectable in the band view and not in the flat
 * list. The same phase, two behaviours, decided by which button the user had
 * pressed at the top of the page -- and the selection is supposed to survive
 * that switch (spec task 8, "Auswahlzustand über Bänder hinweg synchron"),
 * which it cannot do if one of the views has no way to make one.
 *
 * It is a `<label>` wrapping a real checkbox, so it is in the tab order, it
 * responds to Space, and a screen reader says what it is and whether it is
 * ticked. Spec §2.9: the mockups had no keyboard path at all, and a row that
 * is a clickable `<div>` is the usual way that happens.
 *
 * THE NAME IS A BUTTON INSIDE THAT LABEL, and that is deliberate rather than
 * careless. HTML does not forward a label's activation when the click landed
 * on interactive content inside it, so pressing the name opens the card and
 * does NOT tick the box -- but that is a rule worth a test rather than a
 * reading of the specification, and there is one.
 */
import { useEffect, useRef, useState } from 'react';
import { colors, spacing } from '../../theme/components';
import Popover from './Popover';
import PhaseCard from './PhaseCard';
import { bindPhaseDraggable } from './dragBinding';
import CapabilityMarks from './CapabilityMarks';

const S = {
  row: {
    display: 'flex', alignItems: 'baseline', gap: spacing.innerSpacing,
    padding: '4px 8px', fontSize: '10pt', cursor: 'pointer',
    borderBottom: `1px solid ${colors.border}`,
  },
  indented: { paddingLeft: 20 },
  why: { fontSize: '8pt', color: colors.textSecondary },
  nameButton: {
    background: 'transparent', border: 'none', padding: 0, cursor: 'pointer',
    color: colors.text, font: 'inherit', textAlign: 'left',
    textDecorationLine: 'underline', textDecorationStyle: 'dotted',
    textUnderlineOffset: 3,
  },
  stack: { display: 'flex', flexDirection: 'column', gap: 2, minWidth: 0 },
  // While a row is in the air it stays where it was, faded: removing it
  // would reflow the list under the pointer and move the drop target the
  // person is aiming at.
  lifted: { opacity: 0.4 },
  head: { display: 'flex', alignItems: 'baseline', gap: spacing.innerSpacing },
};

export default function PhaseRow({
  phase, label, why = [], checked, onToggle, selectLabel, indented = false,
  whyText = (k) => k, detail = null, withCard = false, loadCard = null,
  onReveal = null, dragPayload = null, onNamed = null, siblingsFor = null,
  groupsFor = null,
}) {
  const ref = useRef(null);
  const [lifted, setLifted] = useState(false);

  // `dragPayload` is a function so the payload is read at PICK-UP: the
  // selection can change between render and grab, and a payload captured at
  // render would file yesterday's selection.
  useEffect(() => {
    const el = ref.current;
    if (!el || !dragPayload) return undefined;
    return bindPhaseDraggable(el, () => dragPayload(phase.key), setLifted);
  }, [phase.key, dragPayload]);

  return (
    <label
      ref={ref}
      style={{
        ...(indented ? { ...S.row, ...S.indented } : S.row),
        ...(lifted ? S.lifted : null),
      }}
      data-phase-key={phase.key}
      data-draggable={dragPayload ? 'yes' : undefined}
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={() => onToggle(phase.key)}
        aria-label={selectLabel}
      />
      <span style={S.stack}>
        <span style={S.head}>
          {/* The NAME opens the card, and it is a button so the keyboard
              reaches it -- the row around it is a label for the checkbox,
              which is a different job. A `span` with an onClick would have
              put the card out of reach of every reader who does not use a
              mouse (§2.9). */}
          {withCard ? (
            <Popover
              label={label}
              trigger={(
                <button type="button" style={S.nameButton}>{label}</button>
              )}
            >
              {({ close }) => (
                <PhaseCard
                  phaseKey={phase.key}
                  load={loadCard}
                  onNamed={onNamed}
                  siblings={siblingsFor ? siblingsFor(phase.key) : []}
                  inGroups={groupsFor ? groupsFor(phase.key) : []}
                  onReveal={onReveal && ((category, name) => {
                    // Close first: the popover would otherwise stay
                    // open over a page the reader has just left.
                    close();
                    onReveal(category, name);
                  })}
                />
              )}
            </Popover>
          ) : (
            <span>{label}</span>
          )}
          {/* Which of the three methods this phase can be indexed with.
              Both first-time readers asked for it here: it decides the
              run, and it was two clicks deep, per entry, with no way to
              compare. */}
          <CapabilityMarks capabilities={phase.capabilities} />
          {/* Every row says WHY it is here: a hit with no visible reason
              reads as a bug, which a tester reported in as many words. */}
          {why.length > 0 && (
            <span style={S.why}>{why.map(whyText).join(' · ')}</span>
          )}
        </span>
        {/* The identity line, where there is room for it. */}
        {detail}
      </span>
    </label>
  );
}

// Re-exported, not re-implemented: `phaseSearch.js` owns the rule, because
// the name a row shows and the name the search ranks on have to be the same
// name. They were two functions until a mutation run showed they could
// disagree.
export { phaseLabel } from './phaseSearch';
