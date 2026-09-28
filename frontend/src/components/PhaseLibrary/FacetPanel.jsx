/**
 * The filter column: elements, then the three methods.
 *
 * The methods come SECOND on purpose. "Which phases contain iron" is the
 * question people arrive with; "which of them can I actually index" is the
 * one they ask next, and putting it first would make the library look like a
 * list of what the software can do rather than a list of what exists.
 *
 * A method row shows ONE number, the same one an element chip shows: what
 * you would get by clicking it now. The library-wide fact -- 29 of 36 phases
 * can be indexed spherically -- is true while you filter and is therefore in
 * the tooltip, not beside a number measured on something else. The first
 * draft printed both and read "Spherical 29 of 29" on an unfiltered library,
 * which is a sentence that answers nothing.
 */
import { useTranslation } from 'react-i18next';
import { colors, spacing } from '../../theme/components';
import ElementFacetPanel from './ElementFacetPanel';

const S = {
  section: { marginTop: spacing.outerMargin },
  head: { display: 'flex', alignItems: 'baseline', gap: spacing.innerSpacing,
          marginBottom: spacing.compactMargin },
  title: { fontSize: '9pt', fontWeight: 600, color: colors.textSecondary },
  reset: { marginLeft: 'auto', background: 'transparent', border: 'none',
           color: colors.purple, cursor: 'pointer', fontSize: '8pt', padding: 0 },
  row: (r) => ({
    display: 'flex', alignItems: 'baseline', gap: spacing.innerSpacing,
    width: '100%', textAlign: 'left', padding: '3px 6px', marginBottom: 2,
    border: `1px solid ${r.selected ? colors.purple : 'transparent'}`,
    background: r.selected ? `${colors.purple}22` : 'transparent',
    color: r.disabled && !r.selected ? colors.textSecondary : colors.text,
    opacity: r.disabled && !r.selected ? 0.45 : 1,
    borderRadius: 3, fontSize: '9pt',
    cursor: r.disabled && !r.selected ? 'default' : 'pointer',
  }),
  counts: { marginLeft: 'auto', fontSize: '8pt', color: colors.textSecondary,
            whiteSpace: 'nowrap' },
  note: { marginTop: spacing.innerSpacing, fontSize: '8pt',
          color: colors.textSecondary, lineHeight: 1.5 },
};

export default function FacetPanel({
  elementRows, selectedElements, onToggleElement, onResetElements,
  capabilityRows, selectedCapabilities, onToggleCapability, onResetCapabilities,
  unassignedMasters = [],
}) {
  const { t } = useTranslation('phaselibrary');

  return (
    // Tagged so a test can say "no facet counts yet" about the FACETS
    // rather than about the whole column -- the groups live in the same
    // column and are not an answer to the request in flight.
    <div data-testid="facet-counts">
      <ElementFacetPanel
        rows={elementRows}
        selected={selectedElements}
        onToggle={onToggleElement}
        onReset={onResetElements}
      />

      <section style={S.section} aria-labelledby="pl-methods">
        <div style={S.head}>
          <span style={S.title} id="pl-methods">{t('facets.methods')}</span>
          {selectedCapabilities.length > 0 && (
            <button
              type="button"
              style={S.reset}
              onClick={onResetCapabilities}
            >
              {t('facets.resetMethods')}
            </button>
          )}
        </div>

        {capabilityRows.map((r) => (
          <button
            key={r.capability}
            type="button"
            aria-pressed={r.selected}
            disabled={r.disabled && !r.selected}
            style={S.row(r)}
            onClick={() => onToggleCapability(r.capability)}
            // In the ACCESSIBLE NAME, not only in `title`: Chromium shows
            // no tooltip on keyboard focus, so for a keyboard user the
            // sentence that says what a method needs -- the entire point of
            // this column -- did not exist. All he heard was "Hough 35,
            // button". Thirty-five what?
            aria-label={`${t(`facets.method.${r.capability}`)}: ${r.count} — `
                        + `${t(`facets.methodNeeds.${r.capability}`)} — `
                        + t('facets.methodLibraryFact',
                            { capable: r.capableInLibrary, total: r.libraryTotal })}
            title={`${t(`facets.methodNeeds.${r.capability}`)} — `
                   + t('facets.methodLibraryFact',
                       { capable: r.capableInLibrary, total: r.libraryTotal })}
          >
            <span>{t(`facets.method.${r.capability}`)}</span>
            <span style={S.counts}>{r.count}</span>
          </button>
        ))}

        {unassignedMasters.length > 0 && (
          // Shown rather than concealed (spec §2.7): one of these four is the
          // S-phase's master, sitting under a stem the library does not map.
          // A card that says "no master" about a phase whose master is on the
          // disk is worse than an untidy list.
          <p style={S.note}>
            {/* Reworded after the user loop. The first version -- "N
                simulated masters on disk belong to no phase in this library"
                -- was read as a fault report by two of four readers, one of
                whom would have screenshotted it and asked a postdoc, before
                touching anything. It is a fact about the folder, not an
                error, and it now says what it means for the reader. */}
            {t('facets.unassignedMasters', { count: unassignedMasters.length })}
          </p>
        )}
      </section>
    </div>
  );
}
