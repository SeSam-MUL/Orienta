/**
 * The element chips.
 *
 * NAMED FOR THE FILESYSTEM, not for elegance. `ElementFacets.jsx` beside
 * `elementFacets.js` differs only in case, and Windows resolves both spellings
 * to one file: `import ElementFacets from './ElementFacets'` handed back the
 * LOGIC module, which has no default export, and every test that rendered the
 * page died on "Element type is invalid ... got: undefined". It would have
 * worked on Linux and in CI.
 *
 * All the deciding happens in `elementFacets.js`; this draws it. Two things
 * here are not decoration:
 *
 *   - A chip is a real toggle button with `aria-pressed`, not a styled div.
 *     Spec §2.9: the mockups had no keyboard path at all.
 *   - A chip at zero is disabled, visibly, and still there. Hiding it takes
 *     away the information that the element exists in this library -- and in
 *     the state where every count is zero, a facet column that empties itself
 *     is the §2.8 trap with no way out.
 */
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, spacing } from '../../theme/components';
import { ELEMENT_NAMES } from './elementNames';
import { filterChips, CHIP_SEARCH_THRESHOLD } from './facets';

const S = {
  head: { display: 'flex', alignItems: 'baseline', gap: spacing.innerSpacing,
          marginBottom: spacing.compactMargin },
  title: { fontSize: '9pt', fontWeight: 600, color: colors.textSecondary },
  reset: { marginLeft: 'auto', background: 'transparent', border: 'none',
           color: colors.purple, cursor: 'pointer', fontSize: '8pt', padding: 0 },
  find: { width: '100%', marginBottom: spacing.compactMargin, padding: '3px 6px',
          fontSize: '9pt', background: colors.bgSecondary, color: colors.text,
          border: `1px solid ${colors.border}`, borderRadius: 3 },
  chips: { display: 'flex', flexWrap: 'wrap', gap: 4 },
  chip: (r) => ({
    display: 'inline-flex', alignItems: 'baseline', gap: 4,
    padding: '2px 8px', borderRadius: 10, fontSize: '9pt',
    border: `1px solid ${r.selected ? colors.purple : colors.border}`,
    background: r.selected ? `${colors.purple}33` : 'transparent',
    color: r.disabled ? colors.textSecondary : colors.text,
    opacity: r.disabled ? 0.45 : 1,
    cursor: r.disabled ? 'default' : 'pointer',
  }),
  count: { fontSize: '8pt', color: colors.textSecondary },
};

export default function ElementFacetPanel({ rows, selected, onToggle, onReset }) {
  const { t } = useTranslation('phaselibrary');
  const [find, setFind] = useState('');
  const shown = filterChips(rows, find, (s) => ELEMENT_NAMES[s] || []);

  return (
    <section aria-labelledby="pl-elements">
      <div style={S.head}>
        <span style={S.title} id="pl-elements">{t('facets.elements')}</span>
        {selected.length > 0 && (
          <button
            type="button"
            style={S.reset}
            onClick={onReset}
            // The word is on the BUTTON, not only in an aria-label. The
            // first repair gave the two resets distinct accessible names and
            // left both reading "Clear"; two users then pressed one, saw the
            // list not move, and concluded the page was broken. One wrote
            // "that poisons everything after it", the other "I pressed Clear
            // and literally nothing changed". A name only a screen reader
            // hears does not help the eye that is looking at two identical
            // buttons.
          >
            {t('facets.resetElements')}
          </button>
        )}
      </div>

      {rows.length > CHIP_SEARCH_THRESHOLD && (
        <input
          style={S.find}
          type="search"
          value={find}
          onChange={(e) => setFind(e.target.value)}
          placeholder={t('facets.findElement')}
          aria-label={t('facets.findElement')}
        />
      )}

      <div style={S.chips}>
        {shown.map((r) => (
          <button
            key={r.symbol}
            type="button"
            aria-pressed={r.selected}
            disabled={r.disabled && !r.selected}
            style={S.chip(r)}
            onClick={() => onToggle(r.symbol)}
            /* EVERY spelling the search knows, not the first one.
               `ELEMENT_NAMES` is the search's synonym list, not a
               translation table: index 0 is the US English name from
               pymatgen and index 1 is whatever else was added by hand --
               `aluminium` for Al, but `eisen` for Fe. Taking [0] put
               "aluminum" and "silicon" on the chip in all four
               languages, and taking [1] would have put a German word in
               front of an English reader. Listing them says what it
               actually is: what you can type to find this element. */
            title={(ELEMENT_NAMES[r.symbol] || []).length
              ? t('facets.elementNames',
                  { names: (ELEMENT_NAMES[r.symbol] || []).join(', ') })
              : undefined}
          >
            <span>{r.symbol}</span>
            <span style={S.count}>{r.count}</span>
          </button>
        ))}
      </div>
    </section>
  );
}
