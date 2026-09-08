/**
 * ReflectorBudget.jsx — how many reflector families one phase contributes to
 * the Hough band-triplet library, with what each choice costs.
 *
 * Why this control exists at all
 * ------------------------------
 * PyEBSDIndex sizes that library from the family count and it grows as
 * O(npoles · nangs³), so the SAME 70 families cost a few MiB for a cubic phase
 * and 44.7 GiB for one whose CIF carries no symmetry. On Windows every one of
 * those pages is charged against the commit limit the moment it is asked for,
 * so the ask itself is the damage — on a laptop it is the machine.
 *
 * Why it is not automatic
 * -----------------------
 * Because trimming silently changes the answer. Measured on Ni (m-3m, 58
 * families, 800 real patterns): 40 and 32 families are bit-identical to the
 * full set (0.000° median deviation, the same 495/800 indexed), 24 collapses to
 * 192/800 at 119.7°, and 16 returns a median fit of 180° — handing back
 * orientations that look like data. A machine cannot tell a good trim from a
 * bad one without indexing, so it refuses what it cannot afford, shows the
 * price of each option, and leaves the choice here.
 */
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors as C } from '../../theme/tokens';
import { indexApi } from '../../services/api';

const GiB = 1024 ** 3;

/** "1.45 GiB" / "under 46 MiB" — a size the user can compare at a glance. */
function formatBytes(bytes, t) {
  if (!bytes) return t('reflectors.tiny');
  if (bytes >= GiB / 10) return `${(bytes / GiB).toFixed(2)} GiB`;
  return `${Math.round(bytes / 1024 ** 2)} MiB`;
}

export default function ReflectorBudget({ cifPath, value = null, onChange, nBands = 12 }) {
  const { t } = useTranslation('indexing');
  const [info, setInfo] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    if (!cifPath) { setInfo(null); return undefined; }
    setError(null);
    (async () => {
      try {
        const res = await indexApi.houghReflectorCost(cifPath, nBands);
        if (!cancelled) setInfo(res.data);
      } catch (err) {
        // A cost lookup that fails must cost the user nothing: the phase card
        // keeps working and simply says the figure is unavailable.
        if (!cancelled) setError(err?.response?.data?.detail || err?.message || 'error');
      }
    })();
    return () => { cancelled = true; };
  }, [cifPath, nBands]);

  if (error) {
    return (
      <div style={{ fontSize: '8pt', color: C.textSecondary, marginTop: 4 }}>
        {t('reflectors.unavailable')}
      </div>
    );
  }
  // Measuring the curve takes seconds (it probes a build per count), and
  // seconds of nothing reads as broken — so say what is happening.
  if (!info) {
    return (
      <div style={{ fontSize: '8pt', color: C.textSecondary, marginTop: 4 }}>
        {t('reflectors.measuring')}
      </div>
    );
  }

  const rows = info.table || [];
  const nFull = info.reflectors_available;
  const current = value ?? nFull;
  const chosen = rows.find((r) => r.reflectors === current);
  // The full set is the honest default and stays selectable even when it does
  // not fit: the run then fails with a message that says why, which is far
  // better than the UI quietly pretending the choice was never available.
  const fullRow = rows.find((r) => r.reflectors === nFull);
  const fullFits = fullRow ? fullRow.fits : true;

  return (
    <div style={{ marginTop: 6 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
        <label
          htmlFor={`refl-${cifPath}`}
          title={t('hoverTips.reflectorCount')}
          style={{ fontSize: '8.5pt', color: C.textSecondary, cursor: 'help' }}
        >
          {t('reflectors.label')}
        </label>
        <select
          id={`refl-${cifPath}`}
          value={current}
          onChange={(e) => {
            const n = Number(e.target.value);
            onChange?.(n === nFull ? null : n);
          }}
          title={t('hoverTips.reflectorCount')}
          style={{
            background: C.bg, color: C.text, border: `1px solid ${C.border}`,
            borderRadius: 4, fontSize: '8.5pt', padding: '1px 4px',
            // A select is as wide as its longest option and will not shrink as
            // a flex child unless told to (min-width defaults to auto).
            flexGrow: 0, flexShrink: 1, flexBasis: 'auto', minWidth: 0,
          }}
        >
          {rows.map((r) => (
            <option key={r.reflectors} value={r.reflectors}>
              {r.reflectors === nFull
                ? t('reflectors.allOption', { n: r.reflectors })
                : r.reflectors}
              {' — '}
              {formatBytes(r.bytes, t)}
              {r.fits ? '' : ` ${t('reflectors.tooBigTag')}`}
            </option>
          ))}
        </select>
        {chosen && (
          <span
            style={{
              fontSize: '8pt', fontFamily: 'monospace',
              color: chosen.fits ? C.textSecondary : C.red,
            }}
          >
            {formatBytes(chosen.bytes, t)}
          </span>
        )}
      </div>

      {/* The one fact that turns "bad luck" into something fixable at source. */}
      {info.no_symmetry && (
        <div
          title={t('hoverTips.reflectorNoSymmetry')}
          style={{ fontSize: '8pt', color: C.yellow ?? '#ffcb6b', marginTop: 3, cursor: 'help' }}
        >
          {t('reflectors.noSymmetry', { sg: info.space_group || 'P 1' })}
        </div>
      )}

      {!fullFits && (
        <div style={{ fontSize: '8pt', color: C.textSecondary, marginTop: 3 }}>
          {t('reflectors.budgetNote', {
            budget: formatBytes(info.budget_bytes, t),
          })}
        </div>
      )}

      {value != null && value < nFull && (
        <div style={{ fontSize: '8pt', color: C.textSecondary, marginTop: 3 }}>
          {t('reflectors.checkResult')}
        </div>
      )}
    </div>
  );
}
