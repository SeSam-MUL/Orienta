/**
 * HoughRefusal.jsx - why a Hough run was refused, and the one click that fixes it.
 *
 * The band-triplet library of a reflector list that does not fit the memory budget
 * is refused before it is asked for (backend: `HoughLibraryTooBig`). The backend
 * sends it as a coded problem ({ code: 'hough_library_too_big', params }) and this
 * component words it in the user's language, on both pages that run Hough
 * (Indexing, PC Refinement).
 *
 * Why here and not only in the "Reflector families" card: that card is collapsed
 * by default, so the user who just saw a run fail would never see the reason or
 * the answer. The button sits where the failure is shown. It stores exactly what
 * the card's button stores (the largest default list that fits, an ordinary rule of
 * the phase), through `fit(phaseName)`; a phase with a selection of families of its
 * own gets no button (the backend never replaces a hand selection).
 */
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors as C } from '../../theme/tokens';
import { formatSize, reflectorError } from './reflectorSpec';

export const REFUSAL_CODE = 'hough_library_too_big';

/** The coded problem inside an HTTP error or a task status, or null. */
export function refusalFrom(source) {
  const p = source?.response?.data?.detail ?? source?.problem ?? source;
  return p && p.code === REFUSAL_CODE ? p : null;
}

const btn = (disabled) => ({
  background: 'transparent', color: disabled ? C.textSecondary : C.accent,
  border: `1px solid ${C.border}`, borderRadius: 4, fontSize: '8.5pt',
  padding: '1px 8px', cursor: disabled ? 'default' : 'pointer', opacity: disabled ? 0.6 : 1,
});

export default function HoughRefusal({ problem, fit, onFitted }) {
  const { t } = useTranslation('indexing');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [done, setDone] = useState(null);
  if (!problem || problem.code !== REFUSAL_CODE) return null;
  const params = problem.params || {};
  const phases = params.phases || [];
  const budget = formatSize(params.budget_bytes);

  const onFit = async (phase) => {
    setBusy(true);
    setError(null);
    try {
      const res = await fit(phase.name);
      const found = res?.data?.fit || {};
      setDone(t('reflectorFamilies.fitDone', {
        rows: found.rows ?? phase.fit_rows,
        size: formatSize(found.predicted_bytes) ?? t('reflectorFamilies.costTiny'),
      }));
      onFitted?.(phase.name, res?.data);
    } catch (err) {
      setError(reflectorError(t, err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      role="alert" data-testid="hough-refusal"
      style={{
        marginTop: 6, padding: '6px 8px', borderRadius: 4, fontSize: '8.5pt',
        border: `1px solid ${C.yellow}`, background: 'rgba(255,203,107,0.08)', color: C.text,
      }}
    >
      <div style={{ color: C.yellow, fontWeight: 600 }}>{t('houghRefusal.title')}</div>
      {phases.map((p) => (
        <div key={p.name} data-testid={`hough-refusal-${p.name}`} style={{ marginTop: 4 }}>
          <div>
            {t('houghRefusal.phase', {
              name: p.name, rows: p.rows ?? '?',
              size: formatSize(p.predicted_bytes) ?? '?', budget: budget ?? '?',
            })}
          </div>
          {p.own_selection ? (
            <div>{t('houghRefusal.own', { card: t('reflectorFamilies.title') })}</div>
          ) : p.fit_rows ? (
            <button
              type="button" data-testid={`hough-refusal-fit-${p.name}`} disabled={busy || !!done}
              onClick={() => onFit(p)}
              title={t('reflectorFamilies.fitTip', { rows: p.fit_rows })}
              style={btn(busy || !!done)}
            >
              {t('reflectorFamilies.fitButton', { rows: p.fit_rows })}
            </button>
          ) : p.fit_reason ? (
            <div data-testid={`hough-refusal-none-${p.name}`}>
              {p.fit_reason === 'low_symmetry'
                ? t('houghRefusal.lowSymmetry', { card: t('reflectorFamilies.title') })
                : t('reflectorFamilies.fitNone')}
            </div>
          ) : null}
        </div>
      ))}
      <div style={{ marginTop: 4, color: C.textSecondary }}>
        {t('houghRefusal.other', { card: t('reflectorFamilies.title') })}
      </div>
      {done && <div role="status" style={{ marginTop: 4, color: C.green }}>{done} {t('houghRefusal.runAgain')}</div>}
      {error && <div style={{ marginTop: 4, color: C.red }}>{error}</div>}
    </div>
  );
}
