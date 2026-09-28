/**
 * "Suggest groups" — an arrangement proposed from the library itself.
 *
 * Sebastian, on the feature this ports: "das mit den auto suggest sachen ist
 * zwar cool". It is the only thing in the program that turns 36 unsorted
 * phases into a starting order without somebody doing it by hand, and it is
 * what a person opening a colleague's library wants first.
 *
 * NOTHING IS WRITTEN UNTIL YOU ACCEPT. The proposal comes back from
 * `POST /suggest` and lives on this screen only; the names you leave ticked
 * go to `POST /suggest/apply`, and only then does a file appear. That
 * division is the backend's and it is kept visible here, because "suggest"
 * that quietly created nine groups would be the kind of surprise people
 * never forgive a tool for.
 *
 * A NAME THAT ALREADY EXISTS IS SHOWN, NOT DROPPED. The endpoint answers
 * with `skipped`, and a proposal that silently did nothing for three of its
 * nine entries reads as a bug in the suggestion, not as "you already have
 * those".
 */
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { collectionsApi } from '../../services/api';
import { colors, spacing } from '../../theme/components';

const S = {
  wrap: { display: 'flex', flexDirection: 'column', gap: 4 },
  button: {
    background: 'transparent', border: 'none', color: colors.purple,
    cursor: 'pointer', font: 'inherit', fontSize: '9pt', padding: 0,
    textAlign: 'left',
  },
  box: {
    border: `1px solid ${colors.border}`, borderRadius: 3, padding: 6,
    display: 'flex', flexDirection: 'column', gap: 2,
  },
  row: { display: 'flex', alignItems: 'center', gap: spacing.innerSpacing,
         fontSize: '9pt', cursor: 'pointer' },
  count: { color: colors.textSecondary, fontVariantNumeric: 'tabular-nums' },
  note: { fontSize: '8pt', color: colors.textSecondary, lineHeight: 1.35 },
  problem: { fontSize: '8pt', color: colors.red || '#ff6b6b', lineHeight: 1.35 },
  good: { fontSize: '8pt', color: colors.green || '#7fd17f', lineHeight: 1.35 },
  actions: { display: 'flex', gap: spacing.innerSpacing, marginTop: 2 },
  act: {
    background: 'transparent', border: `1px solid ${colors.border}`,
    color: colors.purple, cursor: 'pointer', font: 'inherit', fontSize: '9pt',
    padding: '2px 8px', borderRadius: 3,
  },
};

/** The message an axios failure should show a person. */
export function reasonOf(err) {
  const detail = err && err.response && err.response.data
    && err.response.data.detail;
  return detail || String(err && err.message ? err.message : err);
}

/**
 * How many phases the whole proposal would leave unfiled.
 *
 * A reader added the four numbers up -- 2 + 23 + 6 + 1 = 32 against a
 * library of 36 -- and could not tell whether the other four would end up
 * ungrouped or be dropped. "The box does not mention them." Counted here
 * over the union, because a phase may appear in more than one proposal and
 * summing the rows would be wrong.
 */
export function untouchedBy(proposals, libraryTotal) {
  const covered = new Set();
  for (const s of proposals || []) for (const k of s.keys || []) covered.add(k);
  if (!libraryTotal) return null;
  return Math.max(0, libraryTotal - covered.size);
}

export default function SuggestGroups({
  onApplied, api = collectionsApi, libraryTotal = 0, labelFor = (k) => k,
}) {
  const { t } = useTranslation('phaselibrary');
  //: null = not run, [] = ran and proposed nothing
  const [proposals, setProposals] = useState(null);
  const [ticked, setTicked] = useState({});
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState(null);
  const [result, setResult] = useState(null);
  //: the proposal whose phases are open, by name
  const [showing, setShowing] = useState(null);

  const run = () => {
    setBusy(true); setProblem(null); setResult(null);
    Promise.resolve().then(() => api.suggest())
      .then((r) => {
        const list = (r.data && r.data.suggestions) || [];
        setProposals(list);
        // Everything ticked to begin with: the proposal is the answer to
        // "what would you do", and making somebody tick nine boxes to accept
        // an answer they asked for is a toll, not a safeguard. Untick what
        // you do not want.
        setTicked(Object.fromEntries(list.map((s) => [s.name, true])));
      })
      .catch((e) => setProblem(reasonOf(e)))
      .finally(() => setBusy(false));
  };

  const apply = () => {
    const accepted = (proposals || []).filter((s) => ticked[s.name])
      .map((s) => s.name);
    if (!accepted.length) { setProposals(null); return; }
    setBusy(true); setProblem(null);
    Promise.resolve().then(() => api.applySuggest(accepted))
      .then((r) => {
        const d = r.data || {};
        setResult({ created: (d.created || []).length,
          skipped: d.skipped || [] });
        setProposals(null);
        if (onApplied) onApplied();
      })
      .catch((e) => setProblem(reasonOf(e)))
      .finally(() => setBusy(false));
  };

  const nTicked = (proposals || []).filter((s) => ticked[s.name]).length;

  return (
    <div style={S.wrap} data-testid="suggest-groups">
      {proposals === null && (
        <button type="button" style={S.button} onClick={run} disabled={busy}
                data-testid="suggest-run">
          {busy ? t('suggest.running') : t('suggest.run')}
        </button>
      )}

      {result && (
        <>
          <p style={S.good} data-testid="suggest-created">
            {t('suggest.created', { count: result.created })}
          </p>
          {result.skipped.length > 0 && (
            // Shown, never dropped: three of nine doing nothing silently
            // reads as a broken suggestion.
            <p style={S.note} data-testid="suggest-skipped">
              {t('suggest.skipped', { count: result.skipped.length,
                names: result.skipped.join(', ') })}
            </p>
          )}
        </>
      )}

      {problem && <p style={S.problem} data-testid="suggest-problem">{problem}</p>}

      {proposals !== null && (
        <div style={S.box}>
          {proposals.length === 0 ? (
            <p style={S.note} data-testid="suggest-empty">{t('suggest.empty')}</p>
          ) : (
            <>
              <p style={S.note}>{t('suggest.nothingYet')}</p>
              {proposals.map((s) => (
                <div key={s.name}>
                  <label style={S.row}>
                    <input
                      type="checkbox"
                      checked={Boolean(ticked[s.name])}
                      onChange={() => setTicked(
                        (cur) => ({ ...cur, [s.name]: !cur[s.name] }))}
                    />
                    <span style={{ flex: 1 }}>{s.name}</span>
                    {/* WHICH phases. One proposal here is 23 of 36, and a
                        reader put it plainly: "I am being asked to accept a
                        grouping of two thirds of my library sight-unseen. A
                        tick box next to a number is not enough to decide." */}
                    <button
                      type="button" style={S.act}
                      onClick={(e) => {
                        e.preventDefault();
                        setShowing(showing === s.name ? null : s.name);
                      }}
                      aria-expanded={showing === s.name}
                      data-testid={`suggest-peek-${s.name}`}
                    >
                      {t('suggest.members', { count: (s.keys || []).length })}
                    </button>
                  </label>
                  {showing === s.name && (
                    <p style={S.note} data-testid={`suggest-phases-${s.name}`}>
                      {(s.keys || []).map(labelFor).join(' · ')}
                    </p>
                  )}
                </div>
              ))}
              {/* And what the whole proposal does NOT touch. */}
              {untouchedBy(proposals, libraryTotal) > 0 && (
                <p style={S.note} data-testid="suggest-untouched">
                  {t('suggest.untouched',
                    { count: untouchedBy(proposals, libraryTotal) })}
                </p>
              )}
            </>
          )}
          <div style={S.actions}>
            <button type="button" style={S.act} onClick={apply} disabled={busy}>
              {t('suggest.apply', { count: nTicked })}
            </button>
            <button type="button" style={S.act}
                    onClick={() => { setProposals(null); setProblem(null); }}>
              {t('suggest.cancel')}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
