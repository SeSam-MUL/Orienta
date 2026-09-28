/**
 * The identity line, drawn (spec §2.2).
 *
 *     [synonym] · formula · Pearson · space group · source      (key, small)
 *
 * WHERE IT IS SHOWN, and why not everywhere: in the flat list and on the
 * profile card, not on a band row. A band row is already indented under its
 * element and there are 93 of them; five more fields on each would bury the
 * name. The band view answers "which phases are in this system", the list
 * answers "which phase is this" -- same data, different question. That is a
 * difference in density, not in what a row can do: both rows are the same
 * component underneath and both are selectable.
 *
 * The source is the last part and is deliberately allowed to be absent.
 * Three phases in this library have neither a citation nor a DOI, and for
 * them there is NOTHING there -- no "unknown", no dash. §2.2: "fehlt ein
 * Feld, steht dort nichts", and that covers rubbish as well as emptiness.
 */
import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/components';
import { identityParts, compactIdentityParts } from './identityParts';
import { GlossaryTerm } from './GlossaryLegend';

const S = {
  line: { display: 'flex', flexWrap: 'wrap', alignItems: 'baseline', gap: 6,
          fontSize: '8.5pt', color: colors.textSecondary },
  formula: { color: colors.text },
  guess: { fontStyle: 'italic' },
  sep: { opacity: 0.4 },
  key: { fontSize: '7.5pt', opacity: 0.7 },
  source: { maxWidth: 380, overflow: 'hidden', textOverflow: 'ellipsis',
            whiteSpace: 'nowrap' },
  link: { color: colors.purple },
};

export default function IdentityLine({ phase, synonym = null, compact = false }) {
  const { t } = useTranslation('phaselibrary');
  // In a band row too, in short form. It was list-view-only, and both rows
  // called `Mn0.5Fe0.5Al5Si0.68` sit in the BAND view -- where a reader had
  // nothing at all to tell them apart.
  const parts = compact ? compactIdentityParts(phase) : identityParts(phase, { synonym });

  return (
    <span style={S.line} data-testid="identity-line" data-phase={phase.key}>
      {parts.map((part, i) => (
        <span key={part.kind + i} style={{ display: 'inline-flex', gap: 6 }}>
          {i > 0 && <span style={S.sep}>·</span>}
          {renderPart(part, t)}
        </span>
      ))}
      {/* The key, small and last: it is how the file is named, not what the
          phase is called, and putting it first is what made people search
          for filenames in the first place. The compact line already ends
          with it. */}
      {!compact && <span style={S.key}>{phase.key}</span>}
    </span>
  );
}

function renderPart(part, t) {
  switch (part.kind) {
    case 'synonym':
      return <strong>{part.text}</strong>;

    case 'formula':
      return (
        <span
          style={part.provenance === 'stem'
            ? { ...S.formula, ...S.guess } : S.formula}
          // Three different things can stand here and they must not read
          // alike: a recorded label, a formula computed from the structure,
          // or the bare file name.
          title={t(`identity.provenance.${part.provenance}`)}
          data-provenance={part.provenance}
        >
          {part.text}
        </span>
      );

    case 'pearson':
      return (
        <span data-part="pearson">
          <GlossaryTerm termKey="pearson">{part.text}</GlossaryTerm>
          {part.system && part.centring && (
            <span> ({t(`identity.centring.${part.centring}`)}{' '}
              {t(`identity.system.${part.system}`)})</span>
          )}
        </span>
      );

    case 'spaceGroup':
      return (
        <span data-part="spaceGroup">
          {part.text}
          {part.it && (
            <span> (<GlossaryTerm termKey="itNumber">{t('identity.it')}</GlossaryTerm>
              {' '}{part.it})</span>
          )}
        </span>
      );

    case 'cellA':
      return <span data-part="cellA">{part.text}</span>;

    case 'key':
      return <span data-part="key" style={S.key}>{part.text}</span>;

    case 'reference':
      return <span style={S.source} data-part="reference" title={part.text}>{part.text}</span>;

    case 'doi':
      // The identifier is stored bare; the link is built here, once.
      return (
        <a
          style={{ ...S.source, ...S.link }}
          data-part="doi"
          href={`https://doi.org/${part.text}`}
          target="_blank"
          rel="noreferrer"
        >
          {part.text}
        </a>
      );

    default:
      return <span>{part.text}</span>;
  }
}
