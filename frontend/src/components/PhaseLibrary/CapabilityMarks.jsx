/**
 * H S D on a row: which of the three methods this phase can be indexed with.
 *
 * Both first-time readers landed on the same complaint, independently.
 * "The list shows no file availability at all. That is the fact that
 * decides this question, and it is invisible until I open a card." And:
 * "a 'has SHT / has Dictionary' marker belongs on the row" -- said while
 * comparing two entries that are identical on every field the list shows
 * and differ only in that one of them cannot be used with two of the three
 * methods.
 *
 * THREE LETTERS AND NOT THREE WORDS. There are 93 rows; "Hough · Spherical
 * · Dictionary" on each would be longer than the phase name. The letters
 * are unexplained on their own, so every one of them carries the full
 * sentence as a tooltip and an accessible label -- and the sidebar counts
 * above the list already name the three methods in full, three lines up.
 *
 * WHAT IS MISSING IS SHOWN, not omitted. A row with only "H" tells you
 * nothing about whether Spherical was considered; "H s d" tells you the
 * other two were looked for and are not there. That distinction is the
 * whole point for the twins: one has all three, the other Hough alone.
 */
import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/components';

const S = {
  wrap: { display: 'inline-flex', gap: 2, fontSize: '7.5pt',
          fontFamily: 'monospace', letterSpacing: '0.04em' },
  yes: { color: colors.green || '#7fd17f', fontWeight: 700 },
  no: { color: colors.textSecondary, opacity: 0.45 },
};

/** The three, in the order the sidebar counts them. */
export const METHODS = ['hough', 'spherical', 'dictionary'];

/** `{hough: true, spherical: false}` -> [{method, letter, has}] */
export function marksFor(capabilities, letters) {
  const caps = capabilities || {};
  return METHODS.map((m) => ({
    method: m,
    letter: letters[m],
    has: Boolean(caps[m]),
  }));
}

export default function CapabilityMarks({ capabilities }) {
  const { t } = useTranslation('phaselibrary');
  const letters = {
    hough: t('capability.letter.hough'),
    spherical: t('capability.letter.spherical'),
    dictionary: t('capability.letter.dictionary'),
  };
  const marks = marksFor(capabilities, letters);
  const title = marks
    .map((m) => t(m.has ? 'capability.can' : 'capability.cannot',
      { method: t(`facets.method.${m.method}`) }))
    .join(' · ');

  return (
    <span style={S.wrap} title={title} aria-label={title} data-testid="capability-marks">
      {marks.map((m) => (
        <span
          key={m.method}
          style={m.has ? S.yes : S.no}
          data-capability={m.method}
          data-has={m.has ? 'yes' : 'no'}
          aria-hidden="true"
        >
          {m.has ? m.letter : m.letter.toLowerCase()}
        </span>
      ))}
    </span>
  );
}
