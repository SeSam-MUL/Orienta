/**
 * SyncUploadDialog — pick which categories to transfer between local and server,
 * then run it with a live progress bar. Works in two directions via `mode`:
 *   - mode="upload"   : local -> server  (counts uploadable = local|both)
 *   - mode="download" : server -> local  (counts downloadable = server|both)
 * "Sync All" only moves CIF/XTAL; this covers any chosen categories
 * (SHT / MC h5 / Master / CIF / XTAL / Dictionary) in EITHER direction.
 *
 * Selection + overwrite are owned here; the actual transfer loop (with progress)
 * runs in the parent, which feeds `running` / `progress` / `result` back in.
 */
import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, alpha, Button } from '../../theme/components';

export default function SyncUploadDialog({
  open, categories = [], mode = 'upload', running = false, progress = null, result = null,
  onStart, onCancel, onClose,
}) {
  const { t } = useTranslation(['databasebrowser', 'common']);
  const [selected, setSelected] = useState(() => new Set());
  const [overwrite, setOverwrite] = useState(false);

  // Namespace + per-direction helpers. Upload counts local|both files;
  // download counts server|both files.
  const ns = mode === 'download' ? 'syncDownload' : 'syncUpload';
  const countOf = (c) => (mode === 'download' ? (c.downloadable || 0) : (c.uploadable || 0));
  // A category is "off by default" if flagged defaultOff (both directions, e.g.
  // dictionaries) OR downloadOff in download mode (e.g. MC h5 — big intermediate).
  const isOff = (c) => c.defaultOff || (mode === 'download' && c.downloadOff);

  // Default-select every category that has something to transfer — EXCEPT the
  // off-by-default ones (still checkable, just not auto-included).
  useEffect(() => {
    if (open) {
      setSelected(new Set(
        categories.filter(c => countOf(c) > 0 && !isOff(c)).map(c => c.id),
      ));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, mode]);

  if (!open) return null;

  const toggle = (id) => setSelected(prev => {
    const next = new Set(prev);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });

  const totalSelected = categories
    .filter(c => selected.has(c.id))
    .reduce((s, c) => s + countOf(c), 0);

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
            {t(`databasebrowser:${ns}.title`)}
          </div>
          <Button onClick={onClose} disabled={running} style={{ fontSize: 11 }}>
            {t('common:close')}
          </Button>
        </div>

        {/* Body */}
        <div className="thin-scrollbar" style={{ flex: 1, overflowY: 'auto', padding: '14px 16px' }}>
          <div style={{ fontSize: 11, color: colors.textSecondary, marginBottom: 12 }}>
            {t(`databasebrowser:${ns}.subtitle`)}
          </div>

          {/* Category rows */}
          {categories.map(cat => {
            const isOn = selected.has(cat.id);
            const count = countOf(cat);
            const disabled = running || count === 0;
            return (
              <label
                key={cat.id}
                title={count === 0 ? t(`databasebrowser:${ns}.nothingInCategory`) : ''}
                style={{
                  display: 'flex', alignItems: 'center', gap: 10,
                  padding: '8px 10px', marginBottom: 6, borderRadius: 6,
                  border: `1px solid ${isOn ? alpha(colors.cyan, 0.35) : colors.border}`,
                  background: isOn ? alpha(colors.cyan, 0.06) : colors.bgSecondary,
                  cursor: disabled ? 'default' : 'pointer',
                  opacity: count === 0 ? 0.5 : 1,
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
                  {isOff(cat) && (
                    <span style={{ marginLeft: 6, fontSize: 10, fontWeight: 400, color: colors.textSecondary }}>
                      {t(`databasebrowser:${ns}.optIn`)}
                    </span>
                  )}
                </span>
                <span style={{ fontSize: 11, color: colors.textSecondary }}>
                  {mode === 'download'
                    ? t('databasebrowser:syncDownload.catCounts', { download: count, onLocal: cat.both })
                    : t('databasebrowser:syncUpload.catCounts', { upload: count, onServer: cat.both })}
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
            {t(`databasebrowser:${ns}.overwrite`)}
          </label>

          {/* Progress */}
          {running && progress && (
            <div style={{ marginTop: 16 }}>
              <div style={{ fontSize: 11, color: colors.textSecondary, marginBottom: 6 }}>
                {t(`databasebrowser:${ns}.progress`, {
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
                <strong>{result.transferred}</strong> {t(`databasebrowser:${ns}.resTransferred`)}
              </span>
              <span style={{ color: colors.textSecondary }}>
                <strong>{result.upToDate}</strong> {t(`databasebrowser:${ns}.resUpToDate`)}
              </span>
              {result.conflicts > 0 && (
                <span style={{ color: colors.yellow }}>
                  <strong>{result.conflicts}</strong> {t(`databasebrowser:${ns}.resConflicts`)}
                </span>
              )}
              {result.errors > 0 && (
                <span style={{ color: colors.red }}>
                  <strong>{result.errors}</strong> {t(`databasebrowser:${ns}.resErrors`)}
                </span>
              )}
              {result.cancelled && (
                <span style={{ color: colors.textSecondary }}>
                  {t(`databasebrowser:${ns}.resCancelled`)}
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
            {t(`databasebrowser:${ns}.selectedSummary`, { count: totalSelected })}
          </span>
          {running ? (
            <Button variant="danger" onClick={onCancel} style={{ fontSize: 11 }}>
              {t(`databasebrowser:${ns}.cancel`)}
            </Button>
          ) : (
            <Button
              variant="primary"
              onClick={() => onStart(Array.from(selected), overwrite)}
              disabled={totalSelected === 0}
              style={{ fontSize: 11, background: totalSelected > 0 ? colors.cyan : undefined }}
            >
              {t(`databasebrowser:${ns}.start`, { count: totalSelected })}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
