/**
 * One line for the CIFs that are on disk and never became phases.
 *
 * `_entry_from_cif` has always survived a file it cannot read — one broken
 * download must not cost the rest of the library — but it survived it
 * SILENTLY: the file sits in `Database/CIF_Library` where the user put it,
 * the suggestion panel simply does not offer that phase, and even the
 * "why is there no match" explanation cannot name it, because it reasons
 * about library entries and this file never became one. The report reads
 * "my phase is in the folder and the program ignores it".
 *
 * The backend returns `library_skipped: [{file, reason, code?}]` from the same
 * load that built the library, and it carries TWO kinds:
 *
 *   no code               the file never became a phase — unreadable, empty,
 *                         a duplicate filename, or its data blocks disagree;
 *   `suspect_listed_row`  the file IS a phase, supplied by a
 *                         `crystal_database.xlsx` row, and that row's
 *                         composition does not match the CIF's own atom-site
 *                         table. Counting it as "not read" would send the user
 *                         looking for a missing phase that is in the list.
 *
 * One chip each, with the files and their reasons in the tooltip — the panel
 * has room for a line, and the reason is a backend sentence carrying measured
 * numbers, which is why it is not in the locale files. Shown by the suggestion
 * panel and by the phase map, which is where those compositions are USED.
 */
import React from 'react';
import { useTranslation } from 'react-i18next';

import { colors, alpha } from '../../theme/components';

function Chip({ testId, label, detail }) {
  return (
    <div
      data-testid={testId}
      title={detail}
      style={{
        fontSize: '8pt',
        color: colors.orange,
        background: alpha(colors.orange, 10),
        border: `1px solid ${alpha(colors.orange, 25)}`,
        borderRadius: 3,
        padding: '1px 6px',
        fontWeight: 600,
        cursor: 'help',
      }}
    >
      {label}
    </div>
  );
}

const detailOf = (list) =>
  list.map((s) => `${s?.file ?? '?'} — ${s?.reason ?? ''}`).join('\n');

export default function SuggestLibrarySkipped({ skipped }) {
  const { t } = useTranslation('eds');
  const list = Array.isArray(skipped) ? skipped : [];
  if (list.length === 0) return null;

  // TWO DIFFERENT SENTENCES, NOT ONE COUNT. "Not read" is wrong for a
  // `suspect_listed_row`: that phase IS offered — the database spreadsheet
  // supplies it — but from a row written by the same parse the CIF reader now
  // refuses, so its composition is not to be trusted. Counting it as "not
  // read" would tell the user to look for a missing phase that is right there
  // in the list.
  const suspect = list.filter((s) => s?.code === 'suspect_listed_row');
  const unread = list.filter((s) => s?.code !== 'suspect_listed_row');

  return (
    <>
      {unread.length > 0 && (
        <Chip
          testId="suggest-library-skipped"
          label={t('suggest.librarySkipped', { count: unread.length })}
          detail={detailOf(unread)}
        />
      )}
      {suspect.length > 0 && (
        <Chip
          testId="suggest-library-suspect"
          label={t('suggest.librarySuspect', { count: suspect.length })}
          detail={detailOf(suspect)}
        />
      )}
    </>
  );
}
