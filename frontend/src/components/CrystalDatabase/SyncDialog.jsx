import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, alpha, Button } from '../../theme/components';

function formatDate(isoString) {
  if (!isoString) return '—';
  try {
    return new Date(isoString).toLocaleString(undefined, {
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit',
    });
  } catch {
    return isoString;
  }
}

function formatSize(bytes) {
  if (bytes == null) return '—';
  if (bytes < 1024) return `${bytes} B`;
  return `${(bytes / 1024).toFixed(1)} KB`;
}

function RadioOption({ label, value, selected, onChange, title }) {
  return (
    <label title={title} style={{
      display: 'flex', alignItems: 'center', gap: 7,
      cursor: 'pointer', userSelect: 'none',
      fontSize: 12, color: selected ? colors.text : colors.textSecondary,
    }}>
      <div
        onClick={() => onChange(value)}
        style={{
          width: 14, height: 14, borderRadius: '50%', flexShrink: 0,
          border: `2px solid ${selected ? colors.cyan : colors.border}`,
          background: 'transparent',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          cursor: 'pointer',
          transition: 'border-color 0.12s',
        }}
      >
        {selected && (
          <div style={{
            width: 6, height: 6, borderRadius: '50%',
            background: colors.cyan,
          }} />
        )}
      </div>
      <span onClick={() => onChange(value)}>{label}</span>
    </label>
  );
}

function ConflictCard({ conflict, resolution, onResolve }) {
  const { t } = useTranslation('crystaldatabase');
  const { filename, file_type, local_mtime, server_mtime, local_size, server_size } = conflict;

  // Compare mtimes so we can annotate which side is newer. Helps the user
  // pick the right resolution instead of guessing (the wrong click here
  // overwrites real work — see "alles futsch" user report).
  let localNewer = false, serverNewer = false;
  try {
    const l = new Date(local_mtime).getTime();
    const s = new Date(server_mtime).getTime();
    if (Number.isFinite(l) && Number.isFinite(s)) {
      if (l > s) localNewer = true;
      else if (s > l) serverNewer = true;
    }
  } catch { /* ignore */ }

  return (
    <div style={{
      background: colors.bgSecondary,
      border: `1px solid ${resolution ? colors.border : alpha(colors.yellow, 0.35)}`,
      borderRadius: 6,
      padding: '10px 14px',
      marginBottom: 8,
      transition: 'border-color 0.15s',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
        <span style={{ fontSize: 13, fontWeight: 600, color: colors.text }}>{filename}</span>
        <span style={{
          fontSize: 10, color: colors.textSecondary,
          background: colors.bgTertiary, borderRadius: 3,
          padding: '1px 6px', border: `1px solid ${colors.border}`,
        }}>{file_type}</span>
        {!resolution && (
          <span style={{ fontSize: 10, color: colors.yellow, marginLeft: 'auto' }}>
            {t('sync.needsResolution')}
          </span>
        )}
        {resolution && (
          <span style={{ fontSize: 10, color: colors.green, marginLeft: 'auto' }}>
            {resolution === 'keep_local' ? t('sync.keepLocal')
              : resolution === 'keep_server' ? t('sync.keepServer')
              : resolution === 'keep_both' ? t('sync.keepBoth')
              : t('sync.skip')}
          </span>
        )}
      </div>

      <div style={{
        display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4,
        fontSize: 11, color: colors.textSecondary, marginBottom: 10,
      }}>
        <div>
          <span style={{ color: colors.cyan }}>{t('sync.local')}</span>
          {formatDate(local_mtime)}
          <span style={{ color: colors.border }}> · </span>
          {formatSize(local_size)}
          {localNewer && (
            <span style={{ color: colors.green, marginLeft: 6, fontWeight: 600 }}>
              {t('sync.newer')}
            </span>
          )}
        </div>
        <div>
          <span style={{ color: colors.purple }}>{t('sync.server')}</span>
          {formatDate(server_mtime)}
          <span style={{ color: colors.border }}> · </span>
          {formatSize(server_size)}
          {serverNewer && (
            <span style={{ color: colors.green, marginLeft: 6, fontWeight: 600 }}>
              {t('sync.newer')}
            </span>
          )}
        </div>
      </div>

      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 11 }}>
        <RadioOption
          label={t('sync.optKeepLocal')}
          value="keep_local"
          selected={resolution === 'keep_local'}
          onChange={onResolve}
          title={t('hoverTips.syncResolveKeepLocal')}
        />
        <RadioOption
          label={t('sync.optKeepServer')}
          value="keep_server"
          selected={resolution === 'keep_server'}
          onChange={onResolve}
          title={t('hoverTips.syncResolveKeepServer')}
        />
        <RadioOption
          label={t('sync.optKeepBoth')}
          value="keep_both"
          selected={resolution === 'keep_both'}
          onChange={onResolve}
          title={t('hoverTips.syncResolveKeepBoth')}
        />
        <RadioOption
          label={t('sync.optSkip')}
          value="skip"
          selected={resolution === 'skip'}
          onChange={onResolve}
          title={t('hoverTips.syncResolveSkip')}
        />
      </div>
    </div>
  );
}

function CollapsibleSection({ title, color, children, defaultOpen = false, tip }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div style={{ marginBottom: 12 }}>
      <button
        onClick={() => setOpen(o => !o)}
        title={tip}
        style={{
          background: 'none', border: 'none', cursor: 'pointer',
          display: 'flex', alignItems: 'center', gap: 6,
          padding: '0 0 6px', color: color || colors.textSecondary, fontSize: 12,
          fontWeight: 600, width: '100%', textAlign: 'left',
        }}
      >
        <span style={{ fontSize: 10, opacity: 0.7 }}>{open ? '▼' : '▶'}</span>
        {title}
      </button>
      {open && children}
    </div>
  );
}

export default function SyncDialog({ syncResult, onResolve, onClose, resolving }) {
  const { t } = useTranslation(['crystaldatabase', 'common']);
  const [resolutions, setResolutions] = useState({});

  // Smart-default every conflict based on which side is newer. Local edit
  // → "keep_local", server update → "keep_server". Users can override
  // each one; this just stops the common footgun where someone picks
  // "All: Keep Server" without reading, and vapourises their DOI/reference
  // edits.
  const conflicts = syncResult?.conflicts || [];
  useEffect(() => {
    if (conflicts.length === 0) return;
    const next = {};
    conflicts.forEach((c) => {
      try {
        const l = new Date(c.local_mtime).getTime();
        const s = new Date(c.server_mtime).getTime();
        if (Number.isFinite(l) && Number.isFinite(s)) {
          next[c.filename] = l >= s ? 'keep_local' : 'keep_server';
        }
      } catch { /* leave unset */ }
    });
    setResolutions((prev) => ({ ...next, ...prev }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [syncResult]);

  if (!syncResult) return null;

  const { uploaded = [], downloaded = [], up_to_date = [], errors = [] } = syncResult;

  const resolvedCount = Object.keys(resolutions).length;
  const totalConflicts = conflicts.length;
  const allResolved = totalConflicts > 0 && resolvedCount === totalConflicts;

  const handleSetResolution = (filename, action) => {
    setResolutions(prev => ({ ...prev, [filename]: action }));
  };

  const handleResolveAll = () => {
    const resolutionList = conflicts.map(c => ({
      filename: c.filename,
      file_type: c.file_type,
      action: resolutions[c.filename],
    }));
    onResolve(resolutionList);
  };

  return (
    <div
      style={{
        position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
        background: 'rgba(0,0,0,0.6)', zIndex: 1100,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        backdropFilter: 'blur(2px)',
      }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div style={{
        background: colors.bg,
        border: `1px solid ${colors.border}`,
        borderRadius: 8,
        width: 600,
        maxHeight: '75vh',
        display: 'flex',
        flexDirection: 'column',
        boxShadow: '0 12px 40px rgba(0,0,0,0.4)',
        animation: 'fadeSlideIn 0.2s ease-out',
      }}>
        {/* Header */}
        <div style={{
          padding: '12px 16px',
          borderBottom: `1px solid ${colors.border}`,
          display: 'flex', alignItems: 'center', gap: 10,
          flexShrink: 0,
        }}>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: colors.accent }}>
              {t('sync.title')}
            </div>
          </div>
          <Button onClick={onClose} style={{ fontSize: 11 }} title={t('hoverTips.syncClose')}>{t('common:close')}</Button>
        </div>

        {/* Scrollable body */}
        <div className="thin-scrollbar" style={{ flex: 1, overflowY: 'auto', padding: '14px 16px' }}>
          {/* Summary row */}
          <div style={{
            display: 'flex', flexWrap: 'wrap', gap: 16,
            padding: '10px 14px', marginBottom: 14,
            background: colors.bgSecondary,
            border: `1px solid ${colors.border}`,
            borderRadius: 6,
            fontSize: 12,
          }}>
            <span style={{ color: colors.green }}>
              <strong>{uploaded.length}</strong> {t('sync.summaryUploaded')}
            </span>
            <span style={{ color: colors.cyan }}>
              <strong>{downloaded.length}</strong> {t('sync.summaryDownloaded')}
            </span>
            <span style={{ color: colors.textSecondary }}>
              <strong>{up_to_date.length}</strong> {t('sync.summaryUpToDate')}
            </span>
            {errors.length > 0 && (
              <span style={{ color: colors.red }}>
                <strong>{errors.length}</strong> {t('sync.summaryErrors', { count: errors.length })}
              </span>
            )}
            {totalConflicts > 0 && (
              <span style={{ color: colors.yellow }}>
                <strong>{totalConflicts}</strong> {t('sync.summaryConflicts', { count: totalConflicts })}
              </span>
            )}
          </div>

          {/* Errors section */}
          {errors.length > 0 && (
            <CollapsibleSection
              title={t('sync.errorsSection', { count: errors.length })}
              color={colors.red}
              defaultOpen={true}
              tip={t('hoverTips.syncErrorsToggle')}
            >
              <div style={{
                background: alpha(colors.red, 0.08),
                border: `1px solid ${alpha(colors.red, 0.25)}`,
                borderRadius: 5, padding: '8px 12px', marginBottom: 8,
              }}>
                {errors.map((err, i) => (
                  <div key={i} style={{
                    fontSize: 11, color: colors.red,
                    padding: '2px 0',
                    borderBottom: i < errors.length - 1 ? `1px solid ${alpha(colors.red, 0.15)}` : 'none',
                  }}>
                    {err}
                  </div>
                ))}
              </div>
            </CollapsibleSection>
          )}

          {/* Conflicts section */}
          {totalConflicts > 0 && (
            <div>
              <div style={{
                fontSize: 12, fontWeight: 600, color: colors.yellow,
                marginBottom: 8, display: 'flex', alignItems: 'center', gap: 8,
              }}>
                <span>{t('sync.conflictsHeading', { count: totalConflicts })}</span>
                <span style={{ fontSize: 10, color: colors.textSecondary, fontWeight: 400 }}>
                  {t('sync.conflictsHint')}
                </span>
              </div>

              {/* Bulk actions */}
              <div style={{
                display: 'flex', gap: 6, marginBottom: 10, flexWrap: 'wrap',
              }}>
                {[
                  { label: t('sync.bulkKeepLocal'), action: 'keep_local', color: colors.cyan, tip: t('hoverTips.syncBulkKeepLocal') },
                  { label: t('sync.bulkKeepServer'), action: 'keep_server', color: colors.purple, tip: t('hoverTips.syncBulkKeepServer') },
                  { label: t('sync.bulkKeepBoth'), action: 'keep_both', color: colors.green, tip: t('hoverTips.syncBulkKeepBoth') },
                  { label: t('sync.bulkSkip'), action: 'skip', color: colors.textSecondary, tip: t('hoverTips.syncBulkSkip') },
                ].map(({ label, action, color, tip }) => (
                  <button
                    key={action}
                    onClick={() => {
                      const bulk = {};
                      conflicts.forEach(c => { bulk[c.filename] = action; });
                      setResolutions(bulk);
                    }}
                    title={tip}
                    style={{
                      background: alpha(color, 10),
                      border: `1px solid ${alpha(color, 30)}`,
                      borderRadius: 4, padding: '3px 10px',
                      fontSize: 10, fontWeight: 600, color,
                      cursor: 'pointer',
                      transition: 'background 0.15s',
                    }}
                    onMouseEnter={e => { e.currentTarget.style.background = alpha(color, 20); }}
                    onMouseLeave={e => { e.currentTarget.style.background = alpha(color, 10); }}
                  >
                    {label}
                  </button>
                ))}
              </div>

              {conflicts.map(conflict => (
                <ConflictCard
                  key={conflict.filename}
                  conflict={conflict}
                  resolution={resolutions[conflict.filename]}
                  onResolve={(action) => handleSetResolution(conflict.filename, action)}
                />
              ))}
            </div>
          )}

          {/* Nothing to do */}
          {totalConflicts === 0 && errors.length === 0 && (
            <div style={{
              textAlign: 'center', color: colors.textSecondary,
              fontSize: 12, padding: '20px 0',
            }}>
              {t('sync.noConflicts')}
            </div>
          )}
        </div>

        {/* Footer */}
        {totalConflicts > 0 && (
          <div style={{
            padding: '10px 16px',
            borderTop: `1px solid ${colors.border}`,
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            flexShrink: 0,
          }}>
            <span style={{ fontSize: 11, color: colors.textSecondary }}>
              {t('sync.resolvedCount', { resolved: resolvedCount, total: totalConflicts })}
            </span>
            <Button
              onClick={handleResolveAll}
              disabled={!allResolved || resolving}
              variant="primary"
              style={{ fontSize: 11, background: allResolved ? colors.purple : undefined }}
              title={t('hoverTips.syncResolveAll')}
            >
              {resolving ? t('sync.resolving') : t('sync.resolveAll')}
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
