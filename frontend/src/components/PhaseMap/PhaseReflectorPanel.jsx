/**
 * PhaseReflectorPanel.jsx — the reflector control, where the failure appears.
 *
 * Phase Verification and the pseudo-symmetry step that follows a spherical run
 * both build their own Hough indexers. When one of them refuses a phase for
 * memory, this page is where the user is standing — and until now the only
 * place to do anything about it was the Indexing page, which they may not have
 * used at all for this result.
 *
 * Collapsed by default, and it fetches NOTHING until opened: measuring the
 * cost curve probes a build per reflector count and takes 3-5 s per phase
 * (measured), so doing it for every phase on page load would spend ten seconds
 * of background work on a question nobody asked.
 */
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors as C } from '../../theme/tokens';
import { indexApi } from '../../services/api';
import ReflectorBudget from '../Indexing/ReflectorBudget';

export default function PhaseReflectorPanel({ resultId = null }) {
  const { t } = useTranslation(['phasemap', 'indexing']);
  const [open, setOpen] = useState(false);
  const [phases, setPhases] = useState(null);
  const [error, setError] = useState(null);
  const [limits, setLimits] = useState({});

  const load = useCallback(async () => {
    setError(null);
    try {
      const res = await indexApi.phaseCheckPhases(resultId);
      const list = res.data?.phases || [];
      setPhases(list);
      // Seed from what the backend already remembers, so a limit set earlier
      // (here, or on the Indexing page, or in a previous session) shows as the
      // current value instead of silently reading "all".
      setLimits(Object.fromEntries(
        list.filter((p) => p.max_reflectors).map((p) => [p.cif_path, p.max_reflectors]),
      ));
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || 'error');
    }
  }, [resultId]);

  const toggle = () => {
    const next = !open;
    setOpen(next);
    if (next && phases === null) load();
  };

  return (
    <div style={{ marginTop: 8 }}>
      <button
        type="button"
        onClick={toggle}
        aria-expanded={open}
        title={t('indexing:hoverTips.reflectorCount')}
        style={{
          background: 'transparent', border: 'none', padding: 0,
          color: C.textSecondary, fontSize: '8.5pt', cursor: 'pointer',
          textAlign: 'left',
        }}
      >
        {open ? '▼' : '▶'} {t('phasemap:reflectors.disclosure')}
      </button>

      {open && (
        <div style={{ marginTop: 6 }}>
          {error && (
            <div style={{ fontSize: '8pt', color: C.textSecondary }}>
              {t('indexing:reflectors.unavailable')}
            </div>
          )}
          {!error && phases === null && (
            <div style={{ fontSize: '8pt', color: C.textSecondary }}>
              {t('indexing:reflectors.measuring')}
            </div>
          )}
          {!error && phases !== null && phases.length === 0 && (
            <div style={{ fontSize: '8pt', color: C.textSecondary }}>
              {t('phasemap:reflectors.noCifs')}
            </div>
          )}
          {(phases || []).map((p) => (
            <div
              key={p.cif_path}
              style={{
                marginBottom: 8, paddingLeft: 8,
                borderLeft: `2px solid ${C.border}`,
              }}
            >
              <div style={{ fontSize: '8.5pt', color: C.text }}>
                {p.name}
                {p.point_group && (
                  <span style={{ color: C.textSecondary }}> ({p.point_group})</span>
                )}
              </div>
              <ReflectorBudget
                cifPath={p.cif_path}
                value={limits[p.cif_path] ?? null}
                onChange={(n) => {
                  setLimits((prev) => {
                    const next = { ...prev };
                    if (n == null) delete next[p.cif_path];
                    else next[p.cif_path] = n;
                    return next;
                  });
                  // Straight to the backend: this panel has no "run" to carry
                  // the value, and the builders that need it (Phase
                  // Verification, the pseudo-symmetry resolver) read the
                  // registry, not a request.
                  indexApi.setHoughReflectorLimit(p.cif_path, n).catch(() => {});
                }}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
