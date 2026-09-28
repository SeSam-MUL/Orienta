/**
 * The glossary: a term where it is used, and all of them in one place.
 *
 * `GlossaryTerm` marks a word in the interface and carries its explanation.
 * `GlossaryLegend` lists every term in a `<details>`, which is collapsible,
 * focusable and operable from the keyboard without a line of JavaScript --
 * and is the reason the tooltip is allowed to be a tooltip: someone who
 * cannot hover still has the whole glossary one Tab away (§2.9).
 */
import { useTranslation } from 'react-i18next';
import { colors, spacing } from '../../theme/components';
import { GLOSSARY, glossaryEntry } from './glossary';

const S = {
  term: {
    borderBottom: `1px dotted ${colors.textSecondary}`,
    cursor: 'help',
    textDecoration: 'none',
  },
  details: { marginTop: spacing.outerMargin, fontSize: '9pt' },
  summary: { cursor: 'pointer', color: colors.textSecondary, fontWeight: 600 },
  list: { margin: `${spacing.innerSpacing}px 0 0`, padding: 0, listStyle: 'none' },
  item: { marginBottom: spacing.innerSpacing, lineHeight: 1.5 },
  name: { fontWeight: 600 },
  text: { color: colors.textSecondary },
};

/** One term, marked and explained where it stands. */
export function GlossaryTerm({ termKey, children }) {
  const { t } = useTranslation('phaselibrary');
  const entry = glossaryEntry(termKey);
  const explanation = t(`glossary.${termKey}.text`);
  const Tag = entry.abbr ? 'abbr' : 'span';
  return (
    <Tag style={S.term} title={explanation} data-glossary={termKey}>
      {children ?? t(`glossary.${termKey}.name`)}
    </Tag>
  );
}

export default function GlossaryLegend() {
  const { t } = useTranslation('phaselibrary');
  return (
    <details style={S.details} data-testid="glossary">
      <summary style={S.summary}>{t('glossary.title')}</summary>
      <ul style={S.list}>
        {GLOSSARY.map((g) => (
          <li key={g.key} style={S.item} data-glossary-entry={g.key}>
            <span style={S.name}>{t(`glossary.${g.key}.name`)}</span>
            {' — '}
            <span style={S.text}>{t(`glossary.${g.key}.text`)}</span>
          </li>
        ))}
      </ul>
    </details>
  );
}
