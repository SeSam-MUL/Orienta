/**
 * "There is another entry that looks like this one" — on the card.
 *
 * The change a reader asked for in as many words, after being given the
 * ordinary task "use the alpha phase" and getting it right only by
 * accident: "Put a sibling line on the card, directly under the name,
 * naming the other entry and the one fact that differs. It solves both
 * halves at once: it makes the pair discoverable no matter which one I
 * land on first, and it puts the deciding fact in front of me without a
 * second click."
 *
 * ON BOTH CARDS, because a reader who types `Al Fe Si` gets one of the two
 * first and the other fifteenth, and either is the one they will open.
 *
 * IT DOES NOT PICK A WINNER. Which structure model describes an alloy
 * better is a crystallographic judgement this program has no business
 * making from a citation year. It names the differences and stops -- the
 * same rule the card already follows where the label and the structure
 * disagree.
 */
import { useTranslation } from 'react-i18next';
import { colors, spacing } from '../../theme/components';

const S = {
  box: {
    border: `1px solid ${colors.border}`, borderRadius: 3,
    padding: `4px ${spacing.innerSpacing}px`, marginTop: 4,
    fontSize: '8.5pt', color: colors.textSecondary, lineHeight: 1.4,
  },
  name: { color: colors.text, fontWeight: 600 },
  list: { margin: '2px 0 0', paddingLeft: 16 },
  open: {
    background: 'transparent', border: 'none', color: colors.purple,
    cursor: 'pointer', font: 'inherit', fontSize: '8.5pt', padding: 0,
  },
};

/** One difference, as a sentence. */
function tell(t, d) {
  const names = (ms) => ms.map((m) => t(`facets.method.${m}`)).join(', ');
  switch (d.kind) {
    case 'alsoIndexableWith':
      return t('sibling.alsoIndexableWith', { methods: names(d.methods) });
    case 'notIndexableWith':
      return t('sibling.notIndexableWith', { methods: names(d.methods) });
    case 'composition':
      return t('sibling.composition', { elements: d.theirs || '—' });
    case 'reference':
      return t('sibling.reference', { reference: d.theirs });
    case 'cellA':
      return t('sibling.cellA', { mine: d.mine, theirs: d.theirs });
    default:
      return null;
  }
}

export default function SiblingNote({ siblings, onOpen = null }) {
  const { t } = useTranslation('phaselibrary');
  if (!siblings || !siblings.length) return null;

  return (
    <div style={S.box} data-testid="sibling-note">
      <span>{t('sibling.intro', { count: siblings.length })}</span>
      {siblings.map((s) => (
        <div key={s.key} data-sibling-key={s.key}>
          {onOpen ? (
            <button type="button" style={{ ...S.open, ...S.name }}
                    onClick={() => onOpen(s.key)}>
              {s.display}
            </button>
          ) : (
            <span style={S.name}>{s.display}</span>
          )}
          {/* The key as well as the name: the two in this library are
              called `α-Al(Fe,Mn)Si` and `α-Al(Fe,Mn)`, which differ by two
              characters and are exactly what somebody misreads. */}
          <span> ({s.key})</span>
          <ul style={S.list}>
            {s.differences.map((d, i) => {
              const text = tell(t, d);
              return text
                ? <li key={`${d.kind}-${i}`} data-difference={d.kind}>{text}</li>
                : null;
            })}
          </ul>
        </div>
      ))}
    </div>
  );
}
