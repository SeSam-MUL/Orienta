/**
 * ReflectorFamilies.jsx - which crystal planes Hough indexing uses for a phase.
 *
 * Hough indexing matches the detected bands against the angles between a short
 * list of plane FAMILIES. The default list is built from a rule (d >= 1 A,
 * |F| above a tenth of the strongest, at most 70 rows), and PyEBSDIndex then
 * ignores any family whose pole is a multiple of an earlier one. This table
 * shows those families - spacing, relative structure factor, multiplicity,
 * whether PyEBSDIndex keeps it - and lets the user leave some out, ask for the
 * N strongest, add one of their own, or change the rule.
 *
 * The choice is stored per phase on the backend, so it reaches every Hough build
 * of that phase (the Indexing run, PC Refinement, the pseudo-symmetry resolver,
 * the phase check). Nothing here is sent with a run. A phase without a choice is
 * built exactly as before.
 *
 * Both pages use this component with their own `api` (see reflectorSpec.js);
 * `onChanged` lets the page refresh whatever was computed with the old list. Give
 * it `key={api.key}` so that another phase starts from a clean state.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors as C } from '../../theme/tokens';
import {
  currentSpec, formatSize, reflectorError, specAfterAdd, specAfterToggle, summaryKey,
} from './reflectorSpec';

const small = { fontSize: '8.5pt', color: C.textSecondary };
const inputStyle = {
  background: C.bg, color: C.text, border: `1px solid ${C.border}`, borderRadius: 4,
  fontSize: '8.5pt', padding: '1px 4px', minWidth: 0,
};
const btnStyle = (disabled) => ({
  background: 'transparent', color: disabled ? C.textSecondary : C.accent,
  border: `1px solid ${C.border}`, borderRadius: 4, fontSize: '8.5pt',
  padding: '1px 8px', cursor: disabled ? 'default' : 'pointer',
  opacity: disabled ? 0.6 : 1,
});
const th = { textAlign: 'left', fontWeight: 600, padding: '1px 4px', color: C.textSecondary, cursor: 'help' };
const td = { padding: '1px 4px', whiteSpace: 'nowrap' };

export default function ReflectorFamilies({
  api, nBands = 12, onChanged, defaultOpen = false,
}) {
  const { t } = useTranslation('indexing');
  const [open, setOpen] = useState(defaultOpen);
  const [table, setTable] = useState(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [note, setNote] = useState(null);
  const [cost, setCost] = useState(null);       // null | 'loading' | 'error' | {…}
  const [storedMode, setStoredMode] = useState(null);
  const [topN, setTopN] = useState(4);
  const [addText, setAddText] = useState('');
  const [minD, setMinD] = useState('1.0');
  const [fThr, setFThr] = useState('0.1');
  const [ruleOpen, setRuleOpen] = useState(false);
  // A late answer for a phase or an older request must not overwrite a newer one.
  const seq = useRef(0);
  const apiRef = useRef(api);
  apiRef.current = api;

  const subject = api.key;

  const showTable = useCallback((data) => {
    setTable(data);
    setMinD(String(data?.rule?.min_d ?? '1.0'));
    setFThr(String(data?.rule?.f_threshold ?? '0.1'));
    setStoredMode(data?.mode ?? 'default');
  }, []);

  const loadCost = useCallback(async () => {
    const mine = ++seq.current;
    setCost('loading');
    try {
      const res = await apiRef.current.cost();
      if (mine === seq.current) setCost(res.data);
    } catch {
      if (mine === seq.current) setCost('error');
    }
  }, []);

  // The headline mode of the closed card: one instant call, no table.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await apiRef.current.stored();
        if (!cancelled) setStoredMode(res.data?.specs?.[subject]?.mode ?? 'default');
      } catch { /* the badge is a convenience */ }
    })();
    return () => { cancelled = true; };
  }, [subject]);

  // The table itself is computed only when somebody opens the card (for a large
  // unit cell it takes seconds), and again at every opening: the choice is shared
  // with the other page, which may have changed it since.
  const haveTable = useRef(false);
  haveTable.current = !!table;
  useEffect(() => {
    if (!open) return undefined;
    let cancelled = false;
    if (!haveTable.current) setLoading(true);
    setError(null);
    (async () => {
      try {
        const res = await apiRef.current.load();
        if (cancelled) return;
        showTable(res.data);
        loadCost();
      } catch (err) {
        if (!cancelled) setError(reflectorError(t, err));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, subject]);

  const change = useCallback(async (spec, after) => {
    setBusy(true);
    setError(null);
    setNote(null);
    try {
      const res = await apiRef.current.change(spec);
      showTable(res.data);
      onChanged?.(res.data);
      loadCost();
      if (after) after(res.data);
      return res.data;
    } catch (err) {
      setError(reflectorError(t, err));
      return null;
    } finally {
      setBusy(false);
    }
  }, [loadCost, onChanged, showTable, t]);

  const onAdd = async () => {
    const text = addText.trim();
    if (!text || !table) return;
    setBusy(true);
    setError(null);
    setNote(null);
    try {
      const res = await apiRef.current.validate(text, currentSpec(table));
      const info = res.data;
      setBusy(false);
      const done = await change(specAfterAdd(table, info.hkl));
      if (done) {
        setAddText('');
        const other = (info.parallel_with || [])[0];
        setNote(other
          ? t('reflectorFamilies.addedParallel', { label: info.label, other })
          : t('reflectorFamilies.added', { label: info.label }));
      }
    } catch (err) {
      setBusy(false);
      setError(reflectorError(t, err));
    }
  };

  const onApplyRule = () => {
    const rule = { min_d: Number(minD), f_threshold: Number(fThr) };
    if (!Number.isFinite(rule.min_d) || !Number.isFinite(rule.f_threshold)) return;
    change({ mode: 'auto', rule });
  };

  const mode = table?.mode ?? storedMode ?? 'default';
  const used = table?.n_effective;
  const summary = table
    ? t(`reflectorFamilies.${summaryKey(mode)}`, { used })
    : (mode === 'custom' ? t('reflectorFamilies.summaryCustom', { used: '…' })
      : mode === 'auto' ? t('reflectorFamilies.summaryAuto', { used: '…' }) : null);
  const staleError = table?.spec_error;

  const rows = table?.families || [];
  const placeholder = table?.hexagonal
    ? t('reflectorFamilies.addPlaceholderHex') : t('reflectorFamilies.addPlaceholder');

  return (
    <div style={{ marginTop: 6 }} data-testid="reflector-families">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        title={t('reflectorFamilies.toggleTip')}
        aria-expanded={open}
        style={{
          background: 'transparent', border: 'none', padding: 0, cursor: 'pointer',
          color: C.textSecondary, fontSize: '8.5pt', textAlign: 'left',
        }}
      >
        {open ? '▼' : '▶'} {t('reflectorFamilies.title')}
        {summary && (
          <span style={{ marginLeft: 6, color: mode === 'default' ? C.textSecondary : C.accent }}>
            ({summary})
          </span>
        )}
      </button>

      {open && (
        <div style={{ marginTop: 4 }}>
          {loading && <div style={small}>{t('reflectorFamilies.loading')}</div>}
          {!loading && !table && error && (
            <div role="alert" style={{ ...small, color: C.red }}>{error}</div>
          )}

          {table && (
            <>
              <div style={{ ...small, marginBottom: 4 }}>{t('reflectorFamilies.intro')}</div>

              {staleError && (
                <div role="alert" style={{ ...small, color: C.yellow ?? '#ffcb6b', marginBottom: 4 }}>
                  {reflectorError(t, staleError)}
                </div>
              )}

              <div style={{ maxHeight: 190, overflowY: 'auto', border: `1px solid ${C.border}`, borderRadius: 4 }}>
                <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: '8.5pt', color: C.text }}>
                  <thead>
                    <tr>
                      <th style={th} title={t('reflectorFamilies.tipColUse')}>{t('reflectorFamilies.colUse')}</th>
                      <th style={th} title={t('reflectorFamilies.tipColFamily')}>{t('reflectorFamilies.colFamily')}</th>
                      <th style={th} title={t('reflectorFamilies.tipColD')}>{t('reflectorFamilies.colD')}</th>
                      <th style={th} title={t('reflectorFamilies.tipColF')}>{t('reflectorFamilies.colF')}</th>
                      <th style={th} title={t('reflectorFamilies.tipColMult')}>{t('reflectorFamilies.colMult')}</th>
                      <th style={th} />
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((f) => {
                      const ignored = f.selected && !f.effective;
                      const label = table.hexagonal && f.hkl4 ? labelFor(f) : f.label;
                      return (
                        <tr
                          key={f.hkl.join(',')}
                          data-testid={`family-${f.label}`}
                          style={{ opacity: ignored ? 0.55 : 1 }}
                        >
                          <td style={td}>
                            <input
                              type="checkbox"
                              checked={!!f.selected}
                              disabled={busy}
                              aria-label={f.label}
                              title={t('reflectorFamilies.tipColUse')}
                              onChange={() => change(specAfterToggle(table, f.hkl))}
                            />
                          </td>
                          <td style={{ ...td, fontFamily: 'monospace' }}>{label}</td>
                          <td style={td}>{f.d.toFixed(3)}</td>
                          <td style={td}>{f.rel_f.toFixed(2)}</td>
                          <td style={td}>{f.mult}</td>
                          <td style={{ ...td, color: C.textSecondary }}>
                            {ignored
                              ? t('reflectorFamilies.ignored', { label: f.dropped_by || f.parallel_with[0] })
                              : (!f.selected && f.parallel_with.length > 0
                                ? t('reflectorFamilies.samePole', { label: f.parallel_with[0] })
                                : '')}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <div style={{ ...small, marginTop: 2 }}>
                {t('reflectorFamilies.counts', { selected: table.n_selected, used: table.n_effective })}
                {table.truncated && (
                  <span> - {t('reflectorFamilies.truncated', { shown: rows.length, total: table.n_total })}</span>
                )}
              </div>

              <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap', marginTop: 6 }}>
                <span style={small} title={t('reflectorFamilies.tipTopN')}>{t('reflectorFamilies.topNLabel')}</span>
                <input
                  type="number" min={2} max={50} value={topN}
                  onChange={(e) => setTopN(Number(e.target.value))}
                  title={t('reflectorFamilies.tipTopN')}
                  aria-label={t('reflectorFamilies.topNLabel')}
                  style={{ ...inputStyle, width: 44 }}
                />
                <span style={small}>{t('reflectorFamilies.topNUnit')}</span>
                <button
                  type="button" disabled={busy || !(topN >= 2)}
                  onClick={() => change({ mode: 'top_n', n: topN })}
                  title={t('reflectorFamilies.tipTopN')}
                  style={btnStyle(busy || !(topN >= 2))}
                >
                  {t('reflectorFamilies.topNButton')}
                </button>
                <button
                  type="button" disabled={busy || mode === 'default'}
                  onClick={() => change(null)}
                  title={t('reflectorFamilies.tipReset')}
                  style={btnStyle(busy || mode === 'default')}
                >
                  {t('reflectorFamilies.reset')}
                </button>
              </div>

              <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap', marginTop: 6 }}>
                <span style={small} title={t('reflectorFamilies.tipAdd')}>{t('reflectorFamilies.addLabel')}</span>
                <input
                  type="text" value={addText} placeholder={placeholder}
                  onChange={(e) => setAddText(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') onAdd(); }}
                  title={t('reflectorFamilies.tipAdd')}
                  aria-label={t('reflectorFamilies.addLabel')}
                  style={{ ...inputStyle, width: 130 }}
                />
                <button
                  type="button" disabled={busy || !addText.trim()}
                  onClick={onAdd}
                  title={t('reflectorFamilies.tipAdd')}
                  style={btnStyle(busy || !addText.trim())}
                >
                  {t('reflectorFamilies.addButton')}
                </button>
              </div>

              <div style={{ marginTop: 6 }}>
                <button
                  type="button"
                  onClick={() => setRuleOpen((v) => !v)}
                  aria-expanded={ruleOpen}
                  style={{ background: 'transparent', border: 'none', padding: 0, cursor: 'pointer', ...small }}
                >
                  {ruleOpen ? '▼' : '▶'} {t('reflectorFamilies.ruleTitle')}
                </button>
                {ruleOpen && (
                  <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap', marginTop: 3 }}>
                    <span style={small} title={t('reflectorFamilies.tipRuleMinD')}>{t('reflectorFamilies.ruleMinD')}</span>
                    <input
                      type="number" step="0.1" min="0.3" value={minD}
                      onChange={(e) => setMinD(e.target.value)}
                      title={t('reflectorFamilies.tipRuleMinD')}
                      aria-label={t('reflectorFamilies.ruleMinD')}
                      style={{ ...inputStyle, width: 56 }}
                    />
                    <span style={small} title={t('reflectorFamilies.tipRuleF')}>{t('reflectorFamilies.ruleF')}</span>
                    <input
                      type="number" step="0.05" min="0" max="1" value={fThr}
                      onChange={(e) => setFThr(e.target.value)}
                      title={t('reflectorFamilies.tipRuleF')}
                      aria-label={t('reflectorFamilies.ruleF')}
                      style={{ ...inputStyle, width: 56 }}
                    />
                    <button
                      type="button" disabled={busy}
                      onClick={onApplyRule}
                      title={t('reflectorFamilies.tipRuleApply')}
                      style={btnStyle(busy)}
                    >
                      {t('reflectorFamilies.ruleApply')}
                    </button>
                  </div>
                )}
              </div>

              <div style={{ ...small, marginTop: 6 }} title={t('reflectorFamilies.tipCost')}>
                <CostLine t={t} cost={cost} />
              </div>
              <div style={{ ...small, marginTop: 3 }}>{t('reflectorFamilies.rowLimitNote')}</div>
              {note && <div role="status" style={{ ...small, marginTop: 3, color: C.green ?? C.text }}>{note}</div>}
              {error && <div role="alert" style={{ ...small, marginTop: 3, color: C.red }}>{error}</div>}
            </>
          )}
        </div>
      )}
    </div>
  );
}

/** {hkil} text of a hexagonal family from its four indices. */
function labelFor(f) {
  const idx = f.hkl4;
  return idx.every((x) => Math.abs(x) < 10) ? `{${idx.join('')}}` : `{${idx.join(' ')}}`;
}

function CostLine({ t, cost }) {
  if (cost == null) return null;
  if (cost === 'loading') return <span>{t('reflectorFamilies.costMeasuring')}</span>;
  if (cost === 'error' || cost.bytes == null) return <span>{t('reflectorFamilies.costUnavailable')}</span>;
  const size = formatSize(cost.bytes) ?? t('reflectorFamilies.costTiny');
  return (
    <span style={{ color: cost.fits === false ? C.red : undefined }}>
      {t('reflectorFamilies.cost', { size })}
      {' - '}
      {cost.fits === false ? t('reflectorFamilies.costTooBig') : t('reflectorFamilies.costFits')}
    </span>
  );
}
