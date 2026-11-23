import { useState, useRef, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { indexApi } from '../../services/api';
import { problemText } from '../../services/problemText';
import {
  nextParticleName, addParticle, discSize, particleRequest, fmtNcc, phaseRow,
} from './particleNcc';

const C = {
  bg: '#282a36', bgSecondary: '#21222c', border: '#44475a', text: '#f8f8f2',
  textSecondary: '#6272a4', accent: '#bd93f9', green: '#50fa7b',
  orange: '#ffb86c', red: '#ff5555',
};
const btn = {
  background: C.bgSecondary, border: `1px solid ${C.border}`, color: C.text,
  borderRadius: 5, padding: '4px 10px', fontSize: 11, cursor: 'pointer',
};
const cell = { padding: '3px 6px', borderBottom: `1px solid ${C.border}`, textAlign: 'right' };
const cellL = { ...cell, textAlign: 'left' };

/**
 * Phase comparison per particle, inside the single-pixel phase test.
 *
 * A particle is the pixel under the crosshair plus its neighbours within a
 * radius. Every pattern of every particle runs through the same compare engine
 * as the single-pixel test (same phases, bandwidth, background removal, mask),
 * and the backend reports per particle the NCC mean ± s over the best patterns
 * of each phase, NCC_max, the next-best phase and ΔNCC — plus the two tables as
 * LaTeX and CSV.
 */
export default function ParticleNccPanel({
  pixelIndex, nRows, nCols, phases, selectedKeys, settings, blocked,
}) {
  const { t, i18n } = useTranslation(['indexing']);
  const [particles, setParticles] = useState([]);
  const [name, setName] = useState('P1');
  const [radius, setRadius] = useState(2);
  const [topN, setTopN] = useState(10);
  const [progress, setProgress] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(null);
  const pollRef = useRef(null);
  const jobRef = useRef(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  }, []);
  useEffect(() => stopPolling, [stopPolling]);

  const running = progress?.status === 'running';
  const lang = i18n.language;
  const phaseCount = (phases || []).filter((p) => selectedKeys.has(p.key)).length;

  const onAdd = () => {
    const out = addParticle(particles, { name, pixelIndex, radius });
    if (out.error) { setError(t(`phaseTest.particle.${out.error}`)); return; }
    setError(null);
    setParticles(out.particles);
    setName(nextParticleName(out.particles));
  };

  const onRun = async () => {
    setError(null); setResult(null); stopPolling();
    try {
      const body = particleRequest({ particles, phases, selectedKeys, topN, settings });
      const res = await indexApi.particleNccStart(body);
      const jid = res.data.job_id;
      jobRef.current = jid;
      setProgress({ status: 'running', done: 0, total: res.data.total, current: null });
      pollRef.current = setInterval(async () => {
        try {
          const p = (await indexApi.particleNccProgress(jid)).data;
          if (jobRef.current !== jid) return;
          setProgress(p);
          if (p.status === 'done') { stopPolling(); setResult(p.result); }
          else if (p.status === 'error') { stopPolling(); setError(p.error || t('phaseTest.failed')); }
          else if (p.status === 'cancelled') { stopPolling(); setError(t('phaseTest.cancelled')); }
        } catch (err) {
          stopPolling();
          setProgress(null);
          setError(problemText(err, t, 'indexing') || err?.message || t('phaseTest.failed'));
        }
      }, 700);
    } catch (err) {
      setProgress(null);
      setError(problemText(err, t, 'indexing') || err?.response?.data?.detail || err?.message
        || t('phaseTest.failed'));
    }
  };

  const onCancel = async () => {
    if (jobRef.current) {
      try { await indexApi.particleNccCancel(jobRef.current); } catch { /* the poll reports it */ }
    }
  };

  const copy = async (which, text) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(which);
      setTimeout(() => setCopied(null), 1800);
    } catch {
      setError(t('phaseTest.particle.copyFailed'));
    }
  };

  const saveCsv = () => {
    if (!result?.csv) return;
    const blob = new Blob([String.fromCharCode(0xfeff), result.csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'phasenvergleich_partikel.csv';
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  };

  const label = (key) => result?.phase_info?.[key]?.label || key;
  const sg = (key) => result?.phase_info?.[key]?.space_group || '';
  const row = Math.floor(pixelIndex / Math.max(nCols, 1));
  const col = pixelIndex % Math.max(nCols, 1);

  return (
    <div data-testid="particle-ncc-panel" style={{
      marginTop: 12, border: `1px solid ${C.border}`, borderRadius: 6, padding: 10,
      background: C.bgSecondary,
    }}>
      <div style={{ color: C.accent, fontWeight: 700, fontSize: 12, marginBottom: 4 }}>
        {t('phaseTest.particle.title')}
      </div>
      <div style={{ fontSize: 10, color: C.textSecondary, marginBottom: 8 }}>
        {t('phaseTest.particle.intro', { n: topN })}
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center', fontSize: 11 }}>
        <label style={{ color: C.textSecondary }}>{t('phaseTest.particle.name')}
          <input value={name} onChange={(e) => setName(e.target.value)} maxLength={64}
            aria-label={t('phaseTest.particle.name')}
            style={{ marginLeft: 4, width: 80, background: C.bg, color: C.text, border: `1px solid ${C.border}`, borderRadius: 4, padding: '2px 4px' }} />
        </label>
        <label style={{ color: C.textSecondary }} title={t('phaseTest.particle.radiusTip')}>
          {t('phaseTest.particle.radius')}
          <input type="number" min={0} max={10} value={radius}
            onChange={(e) => setRadius(Math.max(0, Math.min(10, Number(e.target.value) || 0)))}
            aria-label={t('phaseTest.particle.radius')}
            style={{ marginLeft: 4, width: 44, background: C.bg, color: C.text, border: `1px solid ${C.border}`, borderRadius: 4, padding: '2px 4px' }} />
        </label>
        <button type="button" style={btn} onClick={onAdd} disabled={running}
          title={t('phaseTest.particle.addTip')}>
          {t('phaseTest.particle.add', {
            row, col, count: discSize(pixelIndex, radius, nRows, nCols),
          })}
        </button>
      </div>

      {particles.length > 0 && (
        <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {particles.map((p) => {
            const r0 = Math.floor(p.center_index / Math.max(nCols, 1));
            const c0 = p.center_index % Math.max(nCols, 1);
            return (
              <span key={p.name} style={{ fontSize: 10, border: `1px solid ${C.border}`, borderRadius: 10, padding: '2px 8px', color: C.text }}>
                {t('phaseTest.particle.chip', {
                  name: p.name, row: r0, col: c0, radius: p.radius_px,
                  count: discSize(p.center_index, p.radius_px, nRows, nCols),
                })}
                <button type="button" disabled={running}
                  onClick={() => setParticles(particles.filter((q) => q.name !== p.name))}
                  aria-label={t('phaseTest.particle.remove', { name: p.name })}
                  title={t('phaseTest.particle.remove', { name: p.name })}
                  style={{ marginLeft: 4, background: 'transparent', border: 'none', color: C.red, cursor: 'pointer', fontSize: 11 }}>×</button>
              </span>
            );
          })}
        </div>
      )}

      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 8, fontSize: 11 }}>
        <label style={{ color: C.textSecondary }} title={t('phaseTest.particle.topNTip')}>
          {t('phaseTest.particle.topN')}
          <input type="number" min={1} max={100} value={topN}
            onChange={(e) => setTopN(Math.max(1, Math.min(100, Number(e.target.value) || 1)))}
            aria-label={t('phaseTest.particle.topN')}
            style={{ marginLeft: 4, width: 44, background: C.bg, color: C.text, border: `1px solid ${C.border}`, borderRadius: 4, padding: '2px 4px' }} />
        </label>
        <button type="button" onClick={onRun}
          disabled={running || blocked || particles.length === 0 || phaseCount === 0}
          style={{ ...btn, background: C.accent, color: C.bg, fontWeight: 700 }}
          title={t('phaseTest.particle.runTip')}>
          {running ? t('phaseTest.testing')
            : t('phaseTest.particle.run', { count: particles.length, phases: phaseCount })}
        </button>
        {running && (
          <button type="button" style={btn} onClick={onCancel}>{t('phaseTest.cancel')}</button>
        )}
        {running && (
          <span style={{ color: C.textSecondary, fontSize: 10 }}>
            {t('phaseTest.particle.progress', { done: progress?.done ?? 0, total: progress?.total ?? 0 })}
            {progress?.current ? ` · ${progress.current}` : ''}
          </span>
        )}
      </div>

      {error && <div style={{ color: C.red, fontSize: 11, marginTop: 6 }}>{t('phaseTest.errorPrefix', { error })}</div>}

      {result && (
        <div style={{ marginTop: 10 }} data-testid="particle-ncc-result">
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11, color: C.text }}>
            <thead>
              <tr style={{ color: C.textSecondary }}>
                <th style={cellL}>{t('phaseTest.particle.colParticle')}</th>
                <th style={cellL}>{t('phaseTest.particle.colPhase')}</th>
                <th style={cell}>{t('phaseTest.particle.colMean', { n: result.top_n })}</th>
                <th style={cell}>{t('phaseTest.particle.colMax')}</th>
                <th style={cellL}>{t('phaseTest.particle.colNext')}</th>
                <th style={cell}>{t('phaseTest.particle.colDelta')}</th>
              </tr>
            </thead>
            <tbody>
              {result.particles.map((p) => {
                const a = phaseRow(p, p.assigned);
                const b = phaseRow(p, p.next_best);
                return (
                  <tr key={p.name}>
                    <td style={cellL}>{p.name}</td>
                    <td style={cellL}>
                      {a ? `${label(p.assigned)}${sg(p.assigned) ? ` (${sg(p.assigned)})` : ''}` : '—'}
                      {p.indistinguishable && (
                        <span title={t('phaseTest.particle.indistinguishableTip', {
                          a: label(p.assigned), b: label(p.next_best),
                        })} style={{ color: C.orange, marginLeft: 4 }}>
                          {t('phaseTest.particle.indistinguishable')}
                        </span>
                      )}
                    </td>
                    <td style={cell}>
                      {a ? `${fmtNcc(a.mean, lang)}${a.std != null ? ` ± ${fmtNcc(a.std, lang)}` : ''}` : '—'}
                      {a && a.n < result.top_n && (
                        <span title={t('phaseTest.particle.fewPatternsTip', { n: a.n })} style={{ color: C.orange }}> *</span>
                      )}
                    </td>
                    <td style={cell}>{fmtNcc(a?.max, lang)}</td>
                    <td style={cellL}>{b ? `${label(p.next_best)} (${fmtNcc(b.mean, lang)})` : '—'}</td>
                    <td style={{ ...cell, color: p.indistinguishable ? C.orange : C.text }}>{fmtNcc(p.delta, lang)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          <div style={{ fontSize: 10, color: C.textSecondary, margin: '10px 0 4px' }}>
            {t('phaseTest.particle.matrixTitle', { n: result.top_n })}
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11, color: C.text }}>
            <thead>
              <tr style={{ color: C.textSecondary }}>
                <th style={cellL}>{t('phaseTest.particle.colPhase')}</th>
                <th style={cellL}>{t('phaseTest.particle.colSpaceGroup')}</th>
                {result.particles.map((p) => <th key={p.name} style={cell}>{p.name}</th>)}
              </tr>
            </thead>
            <tbody>
              {result.phase_order.map((key) => (
                <tr key={key}>
                  <td style={cellL}>{label(key)}</td>
                  <td style={cellL}>{sg(key)}</td>
                  {result.particles.map((p) => {
                    const v = result.matrix?.[key]?.[p.name];
                    const best = result.best_per_particle?.[p.name] === key;
                    return (
                      <td key={p.name} style={{ ...cell, fontWeight: best ? 700 : 400, color: best ? C.green : C.text }}>
                        {fmtNcc(v, lang)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>

          <div style={{ display: 'flex', gap: 6, marginTop: 8, flexWrap: 'wrap' }}>
            <button type="button" style={btn} onClick={() => copy('zuordnung', result.latex.zuordnung)}
              title={t('phaseTest.particle.copyLatexTip')}>
              {copied === 'zuordnung' ? t('phaseTest.particle.copied') : t('phaseTest.particle.copyAssignment')}
            </button>
            <button type="button" style={btn} onClick={() => copy('kandidaten', result.latex.kandidaten)}
              title={t('phaseTest.particle.copyLatexTip')}>
              {copied === 'kandidaten' ? t('phaseTest.particle.copied') : t('phaseTest.particle.copyCandidates')}
            </button>
            <button type="button" style={btn} onClick={saveCsv}>{t('phaseTest.particle.saveCsv')}</button>
          </div>
        </div>
      )}
    </div>
  );
}
