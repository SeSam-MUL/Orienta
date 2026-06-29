/**
 * SyncUploadDialog — pick which categories to upload to the server, then push
 * them with a live progress bar. "Sync All" only moved CIF/XTAL; this uploads
 * any chosen categories (SHT / MC h5 / Master / CIF / XTAL).
 *
 * Selection + overwrite are owned here; the actual upload loop (with progress)
 * runs in the parent, which feeds `running` / `progress` / `result` back in.
 */
import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, alpha, Button } from '../../theme/components';

export default function SyncUploadDialog({
  open, categories = [], running = false, progress = null, result = null,
  onStart, onCancel, onClose,
}) {
  const { t } = useTranslation(['databasebrowser', 'common']);
  const [selected, setSelected] = useState(() => new Set());
  const [overwrite, setOverwrite] = useState(false);

  // Default-select every category that has something to upload — EXCEPT those
  // flagged defaultOff (dictionaries): still checkable, just not auto-included.
  useEffect(() => {
    if (open) {
      setSelected(new Set(
        categories.filter(c => c.uploadable > 0 && !c.defaultOff).map(c => c.id),
      ));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  if (!open) return null;

  const toggle = (id) => setSelected(prev => {
    const next = new Set(prev);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });

  const totalSelected = categories
    .filter(c => selected.has(c.id))
    .reduce((s, c) => s + c.uploadable, 0);

  const pct = progress && progress.total
    ? Math.round((progress.current / progress.total) * 100) : 0;

  return (
    <div
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', zIndex: 1100,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        backdropFilter: 'blur(2px)',
      }}
      onClick={(e) => { if (e.target === e.currentTarget && !running) onClose(); }}
    >
      <div style={{
        background: colors.bg, border: `1px solid ${colors.border}`, borderRadius: 8,
        width: 560, maxHeight: '78vh', display: 'flex', flexDirection: 'column',
        boxShadow: '0 12px 40px rgba(0,0,0,0.4)', animation: 'fadeSlideIn 0.2s ease-out',
      }}>
        {/* Header */}
        <div style={{
          padding: '12px 16px', borderBottom: `1px solid ${colors.border}`,
          display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0,
        }}>
          <div style={{ flex: 1, fontSize: 13, fontWeight: 700, color: colors.accent }}>
            {t('databasebrowser:syncUpload.title')}
          </div>
          <Button onClick={onClose} disabled={running} style={{ fontSize: 11 }}>
            {t('common:close')}
          </Button>
        </div>

        {/* Body */}
        <div className="thin-scrollbar" style={{ flex: 1, overflowY: 'auto', padding: '14px 16px' }}>
          <div style={{ fontSize: 11, color: colors.textSecondary, marginBottom: 12 }}>
            {t('databasebrowser:syncUpload.subtitle')}
          </div>

          {/* Category rows */}
          {categories.map(cat => {
            const isOn = selected.has(cat.id);
            const disabled = running || cat.uploadable === 0;
            return (
              <label
                key={cat.id}
                title={cat.uploadable === 0 ? t('databasebrowser:syncUpload.nothingInCategory') : ''}
                style={{
                  display: 'flex', alignItems: 'center', gap: 10,
                  padding: '8px 10px', marginBottom: 6, borderRadius: 6,
                  border: `1px solid ${isOn ? alpha(colors.cyan, 0.35) : colors.border}`,
                  background: isOn ? alpha(colors.cyan, 0.06) : colors.bgSecondary,
                  cursor: disabled ? 'default' : 'pointer',
                  opacity: cat.uploadable === 0 ? 0.5 : 1,
                }}
              >
                <input
                  type="checkbox"
                  checked={isOn}
                  disabled={disabled}
                  onChange={() => toggle(cat.id)}
                />
                <span style={{ flex: 1, fontSize: 12, fontWeight: 600, color: colors.text }}>
                  {cat.label}
                  {cat.defaultOff && (
                    <span style={{ marginLeft: 6, fontSize: 10, fontWeight: 400, color: colors.textSecondary }}>
                      {t('databasebrowser:syncUpload.optIn')}
                    </span>
                  )}
                </span>
                <span style={{ fontSize: 11, color: colors.textSecondary }}>
                  {t('databasebrowser:syncUpload.catCounts', {
                    upload: cat.uploadable, onServer: cat.both,
                  })}
                </span>
              </label>
            );
          })}

          {/* Overwrite toggle */}
          <label style={{
            display: 'flex', alignItems: 'center', gap: 8, marginTop: 10,
            fontSize: 11, color: colors.textSecondary, cursor: running ? 'default' : 'pointer',
          }}>
            <input
              type="checkbox"
              checked={overwrite}
              disabled={running}
              onChange={(e) => setOverwrite(e.target.checked)}
            />
            {t('databasebrowser:syncUpload.overwrite')}
          </label>

          {/* Progress */}
          {running && progress && (
            <div style={{ marginTop: 16 }}>
              <div style={{ fontSize: 11, color: colors.textSecondary, marginBottom: 6 }}>
                {t('databasebrowser:syncUpload.progress', {
                  current: progress.current, total: progress.total, name: progress.name || '',
                })}
              </div>
              <div style={{
                height: 10, borderRadius: 5, background: colors.bgTertiary,
                border: `1px solid ${colors.border}`, overflow: 'hidden',
              }}>
                <div style={{
                  height: '100%', width: `${pct}%`, background: colors.cyan,
                  transition: 'width 0.2s ease-out',
                }} />
              </div>
              <div style={{ fontSize: 10, color: colors.textSecondary, marginTop: 4, textAlign: 'right' }}>
                {pct}%
              </div>
            </div>
          )}

          {/* Result */}
          {!running && result && (
            <div style={{
              marginTop: 16, padding: '10px 14px', borderRadius: 6,
              background: colors.bgSecondary, border: `1px solid ${colors.border}`,
              display: 'flex', flexWrap: 'wrap', gap: 14, fontSize: 12,
            }}>
              <span style={{ color: colors.green }}>
                <strong>{result.uploaded}</strong> {t('databasebrowser:syncUpload.resUploaded')}
              </span>
              <span style={{ color: colors.textSecondary }}>
                <strong>{result.upToDate}</strong> {t('databasebrowser:syncUpload.resUpToDate')}
              </span>
              {result.conflicts > 0 && (
                <span style={{ color: colors.yellow }}>
                  <strong>{result.conflicts}</strong> {t('databasebrowser:syncUpload.resConflicts')}
                </span>
              )}
              {result.errors > 0 && (
                <span style={{ color: colors.red }}>
                  <strong>{result.errors}</strong> {t('databasebrowser:syncUpload.resErrors')}
                </span>
              )}
              {result.cancelled && (
                <span style={{ color: colors.textSecondary }}>
                  {t('databasebrowser:syncUpload.resCancelled')}
                </span>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div style={{
          padding: '10px 16px', borderTop: `1px solid ${colors.border}`,
          display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0,
        }}>
          <span style={{ fontSize: 11, color: colors.textSecondary }}>
            {t('databasebrowser:syncUpload.selectedSummary', { count: totalSelected })}
          </span>
          {running ? (
            <Button variant="danger" onClick={onCancel} style={{ fontSize: 11 }}>
              {t('databasebrowser:syncUpload.cancel')}
            </Button>
          ) : (
            <Button
              variant="primary"
              onClick={() => onStart(Array.from(selected), overwrite)}
              disabled={totalSelected === 0}
              style={{ fontSize: 11, background: totalSelected > 0 ? colors.cyan : undefined }}
            >
              {t('databasebrowser:syncUpload.start', { count: totalSelected })}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
