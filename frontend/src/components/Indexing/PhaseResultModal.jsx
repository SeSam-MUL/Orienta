/**
 * PhaseResultModal — Shows per-phase indexing results in a ranked table.
 *
 * Props:
 *   open: boolean
 *   onClose: () => void
 *   stats: array of { name, ci, pixels, fraction }
 *   method: string
 *   totalPixels: number
 *   elapsed: string
 *   onGoPhaseMap: () => void
 *   onGoAnalysis: () => void
 */

import { useTranslation } from 'react-i18next';
import { colors as C } from '../../theme/tokens';

const PHASE_COLORS = ['#5ff77a', '#82aaff', '#f78c6c', '#c892ea', '#ffcb6b', '#89ddff', '#ff5370', '#c3e88d'];

function ciColor(ci) {
  if (ci == null) return C.textSecondary;
  if (ci >= 0.15) return '#5ff77a';
  if (ci >= 0.05) return '#ffcb6b';
  return '#ff5370';
}

export default function PhaseResultModal({
  open, onClose, stats = [], method = '', totalPixels = 0, elapsed = '',
  onGoPhaseMap, onGoAnalysis,
}) {
  const { t } = useTranslation('indexing');
  if (!open) return null;

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 1000,
        background: 'rgba(0,0,0,0.6)', display: 'flex',
        alignItems: 'center', justifyContent: 'center',
      }}
    >
      <div
        onClick={e => e.stopPropagation()}
        style={{
          background: C.bgSecondary, borderRadius: 8, padding: 0,
          border: `1px solid ${C.border}`, minWidth: 480, maxWidth: 640,
          boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
        }}
      >
        {/* Header */}
        <div style={{
          padding: '12px 16px', borderBottom: `1px solid ${C.border}`,
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        }}>
          <div>
            <div style={{ color: C.green, fontWeight: 700, fontSize: '11pt' }}>
              {t('phaseResultModal.title')}
            </div>
            <div style={{ color: C.textSecondary, fontSize: '9pt', marginTop: 2 }}>
              {t('phaseResultModal.subtitle', { method, pixels: totalPixels, elapsed })}
            </div>
          </div>
          <button
            onClick={onClose}
            title={t('hoverTips.phaseResultClose')}
            aria-label={t('hoverTips.phaseResultClose')}
            style={{
              background: 'none', border: 'none', color: C.textSecondary,
              fontSize: '14pt', cursor: 'pointer', padding: '0 4px',
            }}
          >
            ✕
          </button>
        </div>

        {/* Table */}
        <div style={{ padding: '8px 0' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '10pt' }}>
            <thead>
              <tr style={{ color: C.textSecondary, fontSize: '8pt', textTransform: 'uppercase' }}>
                <th style={{ padding: '4px 16px', textAlign: 'left' }}>{t('phaseResultModal.rank')}</th>
                <th style={{ padding: '4px 16px', textAlign: 'left' }}>{t('phaseResultModal.phase')}</th>
                <th style={{ padding: '4px 16px', textAlign: 'right' }}>{t('phaseResultModal.ciScore')}</th>
                <th style={{ padding: '4px 16px', textAlign: 'right' }}>{t('phaseResultModal.pixel')}</th>
                <th style={{ padding: '4px 16px', textAlign: 'right' }}>{t('phaseResultModal.fraction')}</th>
              </tr>
            </thead>
            <tbody>
              {stats.map((s, i) => (
                <tr key={i} style={{ borderTop: `1px solid ${C.border}33` }}>
                  <td style={{ padding: '6px 16px' }}>
                    <span style={{
                      display: 'inline-block', width: 10, height: 10,
                      borderRadius: 2, background: PHASE_COLORS[i % PHASE_COLORS.length],
                      marginRight: 6, verticalAlign: 'middle',
                    }} />
                    {i + 1}
                  </td>
                  <td style={{
                    padding: '6px 16px', fontWeight: 700,
                    color: PHASE_COLORS[i % PHASE_COLORS.length],
                  }}>
                    {s.name}
                  </td>
                  <td style={{
                    padding: '6px 16px', textAlign: 'right',
                    fontWeight: 600, color: ciColor(s.ci),
                    fontFamily: 'monospace',
                  }}>
                    {s.ci != null ? s.ci.toFixed(4) : '—'}
                  </td>
                  <td style={{ padding: '6px 16px', textAlign: 'right', color: C.text }}>
                    {s.pixels}
                  </td>
                  <td style={{ padding: '6px 16px', textAlign: 'right', color: C.text }}>
                    {s.fraction}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Actions */}
        <div style={{
          padding: '12px 16px', borderTop: `1px solid ${C.border}`,
          display: 'flex', gap: 8, justifyContent: 'flex-end',
        }}>
          {onGoPhaseMap && (
            <button onClick={() => { onGoPhaseMap(); onClose(); }} title={t('hoverTips.phaseResultGoPhaseMap')} style={{
              padding: '6px 16px', background: C.green, color: C.bg,
              border: 'none', borderRadius: 4, fontWeight: 600,
              fontSize: '9pt', cursor: 'pointer',
            }}>
              {t('phaseResultModal.goPhaseMap')}
            </button>
          )}
          {onGoAnalysis && (
            <button onClick={() => { onGoAnalysis(); onClose(); }} title={t('hoverTips.phaseResultGoAnalysis')} style={{
              padding: '6px 16px', background: C.bg, color: C.text,
              border: `1px solid ${C.border}`, borderRadius: 4,
              fontSize: '9pt', cursor: 'pointer',
            }}>
              {t('phaseResultModal.goAnalysis')}
            </button>
          )}
          <button onClick={onClose} title={t('hoverTips.phaseResultClose')} style={{
            padding: '6px 16px', background: 'none', color: C.textSecondary,
            border: `1px solid ${C.border}`, borderRadius: 4,
            fontSize: '9pt', cursor: 'pointer',
          }}>
            {t('phaseResultModal.close')}
          </button>
        </div>
      </div>
    </div>
  );
}
