/**
 * Giving a phase a name people actually use.
 *
 * The library is named after files. Sebastian, holding the shipped app: "ich
 * bin generell damit unzufrieden wie wir phasen wo anzeigen und nennen", and
 * the phases in question are called `sd_0302719` and
 * `Al2Cu_mp-985806_conventional_standard` while the group in the corridor
 * calls them "die Alpha" and "Theta". Both names are true; only one of them
 * is in the program.
 *
 * A NAME STANDS IN FRONT OF THE KEY, NEVER INSTEAD OF IT. The key is still
 * in the row, still indexed, still searchable, and the card still shows it.
 * The complaint this page answers was "ich musste ewig suchen um das cif zu
 * finden" -- a rename that hid the filename would bring it straight back.
 *
 * SEARCH TERMS ARE A SEPARATE FIELD from the display name, because they do
 * different jobs. The display name is what the list should read; a search
 * term is a spelling nobody wants to see but everybody types -- "s phase",
 * "s-phase", "Al2CuMg", "die klebrige". Making one field do both would
 * force a choice between a readable list and a findable library.
 *
 * WHAT IS WRITTEN IS WHAT IS SHOWN. The editor renders the answer the
 * endpoint gives back, not the text that was typed: the store trims, drops
 * blanks and records the author, so echoing the form would show a name the
 * library does not have.
 */
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, spacing } from '../../theme/components';
import { phaseLibraryApi } from '../../services/api';
import useGroups from './useGroups';

const S = {
  form: { display: 'flex', flexDirection: 'column', gap: 4 },
  row: { display: 'flex', alignItems: 'baseline', gap: spacing.innerSpacing },
  label: { fontSize: '8pt', color: colors.textSecondary, minWidth: 92 },
  input: {
    flex: 1, minWidth: 0, padding: '3px 6px', fontSize: '9pt',
    background: colors.bgSecondary, color: colors.text,
    border: `1px solid ${colors.border}`, borderRadius: 3,
  },
  hint: { fontSize: '8pt', color: colors.textSecondary, lineHeight: 1.35 },
  problem: { fontSize: '8pt', color: colors.red || '#ff6b6b', lineHeight: 1.35 },
  buttons: { display: 'flex', gap: spacing.innerSpacing, marginTop: 2 },
  button: {
    background: 'transparent', border: `1px solid ${colors.border}`,
    color: colors.purple, cursor: 'pointer', font: 'inherit', fontSize: '9pt',
    padding: '2px 8px', borderRadius: 3,
  },
  open: {
    background: 'transparent', border: 'none', color: colors.purple,
    cursor: 'pointer', font: 'inherit', fontSize: '9pt', padding: 0,
    textAlign: 'left',
  },
  said: { fontSize: '8pt', color: colors.textSecondary },
};

/** `a, b ,, c` -> ['a','b','c'] -- people type commas and spaces. */
export function parseTerms(text) {
  return String(text || '').split(',').map((s) => s.trim()).filter(Boolean);
}

export default function NameEditor({ card, onSaved, save = null }) {
  const { t } = useTranslation('phaselibrary');
  const author = useGroups((st) => st.author);
  const setAuthor = useGroups((st) => st.setAuthor);
  const [open, setOpen] = useState(false);
  const [name, setName] = useState(card.display_name || '');
  const [terms, setTerms] = useState((card.search_terms || []).join(', '));
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState(null);

  const put = save || ((body) => phaseLibraryApi.setNames(body)
    .then((r) => r.data));

  const submit = (e) => {
    e.preventDefault();
    setBusy(true);
    setProblem(null);
    Promise.resolve()
      .then(() => put({
        key: card.key,
        // An empty field is a DECISION -- "take the name away" -- and the
        // empty string is how the service is told. `|| null` here meant
        // "leave it alone", so clearing the field saved nothing and the
        // old name came straight back into the box.
        displayName: name.trim(),
        searchTerms: parseTerms(terms),
        author,
      }))
      .then((stored) => {
        setBusy(false);
        setOpen(false);
        // What came back, not what was typed.
        setName(stored.display_name || '');
        setTerms((stored.search_terms || []).join(', '));
        if (onSaved) onSaved(stored);
      })
      .catch((err) => {
        setBusy(false);
        // Named, not swallowed: the store lives in the library folder, which
        // may be read-only or on a share that is not there -- and a name
        // that was not written is worse than one that was refused, because
        // the screen would go on showing it.
        const detail = err && err.response && err.response.data
          && err.response.data.detail;
        setProblem(detail || String(err && err.message ? err.message : err));
      });
  };

  if (!open) {
    return (
      <div>
        <button type="button" style={S.open} onClick={() => setOpen(true)}
                data-testid="name-editor-open">
          {card.display_name ? t('name.change') : t('name.give')}
        </button>
        {card.name_author && (
          <div style={S.said} data-testid="name-author">
            {t('name.by', { author: card.name_author,
              when: (card.name_updated || '').slice(0, 10) })}
          </div>
        )}
        {problem && <p style={S.problem} data-testid="name-problem">{problem}</p>}
      </div>
    );
  }

  return (
    <form style={S.form} onSubmit={submit} data-testid="name-editor">
      <div style={S.row}>
        <label style={S.label} htmlFor={`nm-${card.key}`}>{t('name.display')}</label>
        <input
          id={`nm-${card.key}`} style={S.input} value={name} autoFocus
          placeholder={t('name.displayPlaceholder')}
          onChange={(e) => setName(e.target.value)}
        />
      </div>
      <div style={S.row}>
        <label style={S.label} htmlFor={`tm-${card.key}`}>{t('name.terms')}</label>
        <input
          id={`tm-${card.key}`} style={S.input} value={terms}
          placeholder={t('name.termsPlaceholder')}
          onChange={(e) => setTerms(e.target.value)}
        />
      </div>
      <div style={S.row}>
        <label style={S.label} htmlFor={`au-${card.key}`}>{t('name.author')}</label>
        <input
          id={`au-${card.key}`} style={S.input} value={author}
          placeholder={t('name.authorPlaceholder')}
          onChange={(e) => setAuthor(e.target.value)}
        />
      </div>
      <p style={S.hint}>{t('name.hint', { key: card.key })}</p>
      {problem && <p style={S.problem} data-testid="name-problem">{problem}</p>}
      <div style={S.buttons}>
        <button type="submit" style={S.button} disabled={busy}>
          {busy ? t('name.saving') : t('name.save')}
        </button>
        <button type="button" style={S.button} onClick={() => setOpen(false)}>
          {t('name.cancel')}
        </button>
      </div>
    </form>
  );
}
