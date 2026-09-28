/**
 * The profile card (spec §2.3): one phase, one card, the same everywhere.
 *
 * It loads when the popover opens and not before -- `Popover` calls its
 * children as a function for exactly this reason, so no card is fetched for
 * a phase nobody looked at. Thirty-six of these at once would read
 * thirty-six namelists.
 *
 * The four states are the page's four states again, and for the same
 * reason: a request that failed must not look like a phase with no data.
 */
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { phaseLibraryApi } from '../../services/api';
import { colors, spacing } from '../../theme/components';
import { GlossaryTerm } from './GlossaryLegend';
import IdentityLine from './IdentityLine';
import { phaseLabel } from './phaseSearch';
import NameEditor from './NameEditor';
import SiblingNote from './SiblingNote';
import {
  cellRows, compositionRows, fileRows, masterParameters, shtParameters,
  shtProvenance, notices,
} from './phaseCardRows';

const S = {
  groups: { fontSize: '8.5pt', color: colors.textSecondary, marginTop: 3 },
  groupNames: { color: colors.text },
  head: { marginBottom: spacing.groupMargin },
  name: { fontSize: '11pt', fontWeight: 600, color: colors.text },
  section: { marginTop: spacing.groupMargin },
  label: { fontSize: '8pt', fontWeight: 600, color: colors.textSecondary,
           textTransform: 'uppercase', letterSpacing: '0.04em' },
  grid: { display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '2px 10px',
          fontSize: '9pt', marginTop: 4 },
  dim: { color: colors.textSecondary },
  notice: { marginTop: spacing.innerSpacing, padding: '4px 8px', fontSize: '8.5pt',
            background: `${colors.yellow}1a`, borderLeft: `2px solid ${colors.yellow}`,
            lineHeight: 1.5 },
  fileRow: { display: 'flex', gap: 6, alignItems: 'baseline', fontSize: '8.5pt' },
  tick: { fontWeight: 700, width: 12, display: 'inline-block' },
  rel: { color: colors.textSecondary, fontSize: '8pt', wordBreak: 'break-all' },
  disagree: { color: colors.yellow },
  message: { fontSize: '9pt', color: colors.textSecondary },
  link: { background: 'transparent', border: 'none', padding: 0,
          color: colors.purple, cursor: 'pointer', font: 'inherit' },
};

export default function PhaseCard({
  phaseKey, load, onReveal = null, onNamed = null, saveNames = null,
  siblings = [], onOpenSibling = null, inGroups = [],
}) {
  const { t } = useTranslation('phaselibrary');
  const [state, setState] = useState('loading');
  const [card, setCard] = useState(null);

  useEffect(() => {
    let cancelled = false;
    const fetchIt = load
      ? () => load(phaseKey)
      : () => phaseLibraryApi.getPhase(phaseKey).then((r) => r.data);
    setState('loading');
    Promise.resolve().then(fetchIt).then(
      (data) => { if (!cancelled) { setCard(data); setState('loaded'); } },
      (err) => {
        if (cancelled) return;
        // 404 is a different sentence from "could not ask": the first means
        // the library has no such phase, the second that we do not know.
        setState(err && err.response && err.response.status === 404
          ? 'missing' : 'error');
      },
    );
    return () => { cancelled = true; };
  }, [phaseKey, load]);

  if (state === 'loading') {
    return <p style={S.message} data-testid="card-loading">{t('card.loading')}</p>;
  }
  if (state === 'missing') {
    return <p style={S.message} data-testid="card-missing">{t('card.missing')}</p>;
  }
  if (state === 'error' || !card) {
    return <p style={S.message} data-testid="card-error">{t('card.error')}</p>;
  }

  // The card keeps what the endpoint stored, so the name in its heading and
  // the name in the list are the same name, without a second request for a
  // record we were just handed.
  //
  // THE TWO SIDES SPELL THE BYLINE DIFFERENTLY, and a plain spread let the
  // old one survive: the card reads `name_author`/`name_updated` (that is
  // what `build_phase_detail` emits), while `PUT /names` answers with
  // `author`/`updated`. So after Bob renamed a phase Alice had named, the
  // heading showed Bob's name credited to Alice, until the page was read
  // again. Mapped explicitly rather than spread, because the mismatch is
  // invisible at the spread and obvious here.
  const named = (stored) => {
    setCard((cur) => ({
      ...cur,
      ...stored,
      name_author: stored.author ?? cur.name_author,
      name_updated: stored.updated ?? cur.name_updated,
    }));
    if (onNamed) onNamed(stored);
  };

  const composition = compositionRows(card);
  const master = masterParameters(card);
  const sht = shtParameters(card);
  const prov = shtProvenance(card);

  return (
    <div data-testid="phase-card" data-card-key={card.key}>
      <div style={S.head}>
        <div style={S.name}>{phaseLabel(card)}</div>
        <IdentityLine phase={card} synonym={card.display_name} />
        {/* Naming lives on the card and not in the list, because naming a
            phase is something you do once you have looked at it: the card
            is where the cell, the composition and the citation are, and
            those are what tell you WHICH alpha this is. */}
        <NameEditor card={card} onSaved={named} save={saveNames} />
        {/* Directly under the name, which is where the reader who needed
            it said it belonged. */}
        <SiblingNote siblings={siblings} onOpen={onOpenSibling} />
        {/* WHICH GROUPS HOLD THIS PHASE. The tag model creates this
            question and had no answer: a returning user, who under the old
            folder model never needed to ask, put it plainly -- "with tags
            I need one badly", because a plain drag now ADDS and a phase
            can quietly be in three places. Shown only when it is in one,
            so a library nobody has arranged says nothing. */}
        {inGroups.length > 0 && (
          <div style={S.groups} data-testid="card-groups">
            {t('card.inGroups', { count: inGroups.length })}
            {' '}
            <span style={S.groupNames}>{inGroups.join(' · ')}</span>
          </div>
        )}
      </div>

      {notices(card).map((n) => (
        <p key={n.kind} style={S.notice} data-notice={n.kind}>
          {t(`card.notice.${n.kind}`, { detail: n.detail, shown: n.shown })}
        </p>
      ))}

      <Section label={t('card.composition')}>
        {/* Twice, side by side, with the difference COMPUTED -- c1's idea,
            and the right one: a curated list of known discrepancies goes
            stale the day someone adds a CIF. */}
        <div style={S.grid} data-testid="card-composition">
          <span style={S.dim}>{t('card.fromStructure')}</span>
          <span>{composition.filter((r) => r.inStructure)
            .map((r) => r.element).join(' ') || '—'}</span>
          <span style={S.dim}>{t('card.fromLabel')}</span>
          <span style={card.elements_disagree ? S.disagree : undefined}>
            {composition.filter((r) => r.inLabel).map((r) => r.element).join(' ') || '—'}
          </span>
        </div>
      </Section>

      <Section label={t('card.lattice')}>
        <div style={S.grid} data-testid="card-cell">
          {cellRows(card.cell).map((r) => (
            <Row key={r.key} k={t(`card.cellAxis.${r.key}`)} v={r.text} />
          ))}
          {card.cell_setting && (
            <Row k={t('card.setting')} v={card.cell_setting} />
          )}
          {card.n_atoms != null && <Row k={t('card.atoms')} v={card.n_atoms} />}
        </div>
      </Section>

      <Section label={t('card.files')}>
        <div data-testid="card-files">
          {fileRows(card).map((f) => (
            <div key={f.slot} style={S.fileRow} data-file={f.slot}>
              {/* A tick or a cross, not a colour: red alone reads as an
                  error, and a phase with no dictionary is not an error. */}
              <span style={S.tick} aria-hidden="true">{f.present ? '✓' : '✗'}</span>
              <span>
                <GlossaryTerm termKey={glossaryFor(f.slot)}>
                  {t(`card.file.${f.slot}`)}
                </GlossaryTerm>
              </span>
              {f.present ? (
                <span style={S.rel}>
                  {f.name} · {f.date}
                  {/* The repo-relative path only. The endpoint also sends an
                      absolute one; this project has already put a local path
                      into a manuscript. */}
                  {f.rel && <> · {f.rel}</>}
                  {f.category && onReveal && (
                    <>
                      {' · '}
                      <button type="button" style={S.link}
                              onClick={() => onReveal(f.category, f.name)}>
                        {t('card.showInBrowser')}
                      </button>
                    </>
                  )}
                </span>
              ) : (
                /* "Absent", unless the row above says Dictionary is
                   possible -- which it does whenever a master pattern is
                   here, because a dictionary is generated from one. Both
                   statements were true and the card showed only the
                   emptier of the two, so a reader comparing the H S D
                   letters with the file list found them contradicting
                   each other and had no way to tell which was wrong. */
                <span style={S.dim}>
                  {f.slot === 'dictionary' && (card.capabilities || {}).dictionary
                    ? t('card.dictionaryFromMaster')
                    : t('card.absent')}
                </span>
              )}
            </div>
          ))}
        </div>
      </Section>

      {master.length > 0 && (
        <Section label={t('card.masterParameters')}>
          <div style={S.grid} data-testid="card-master">
            {master.map((p) => (
              <Row key={p.key} k={t(`card.param.${p.key}`)}
                   v={`${p.value}${p.unit ? ' ' + p.unit : ''}`} />
            ))}
          </div>
        </Section>
      )}

      {sht.length > 0 && (
        <Section label={t('card.shtParameters')}>
          <div style={S.grid} data-testid="card-sht">
            {sht.map((p) => (
              <Row key={p.key} k={t(`card.param.${p.key}`)}
                   v={`${p.value}${p.unit ? ' ' + p.unit : ''}`} />
            ))}
            {prov.sources.map((s) => (
              <Row
                key={s.slot}
                k={t(`card.builtFrom.${s.slot}`)}
                // `found` is measured locally, so it answers "is the file
                // this was built from still here" -- which a path does not.
                //
                // And when it is NOT here, whether the sidecar recorded a
                // path from outside this project: that separates "the
                // file was deleted" from "this master was built against
                // somebody else's library", which is the difference
                // between a tidy-up and a result you cannot reproduce.
                // The server keeps only the boolean, on purpose -- the
                // path names an account.
                v={`${s.name} ${s.found ? '✓' : '✗'}`
                   + (s.elsewhere ? ` — ${t('card.builtElsewhere')}` : '')}
              />
            ))}
          </div>
        </Section>
      )}
    </div>
  );
}

function Section({ label, children }) {
  return (
    <div style={S.section}>
      <div style={S.label}>{label}</div>
      {children}
    </div>
  );
}

function Row({ k, v }) {
  return (
    <>
      <span style={S.dim}>{k}</span>
      <span>{v}</span>
    </>
  );
}

function glossaryFor(slot) {
  return slot === 'dictionary' ? 'master' : slot;
}
