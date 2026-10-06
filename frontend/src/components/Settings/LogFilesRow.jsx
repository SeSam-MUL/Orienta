/**
 * "Show log files" - where Orienta writes its logs, for bug reports.
 *
 * The packaged app has no console, and the log folder lives inside the runtime
 * directory, a place a user has never been told about. In the desktop app the
 * button opens that folder in the file manager; in a browser there is no way to
 * open a folder on the user's machine, so the path is shown as text to copy.
 *
 * The path comes from the running backend (GET /api/system/log-info), because
 * it is the backend that knows where its handler writes. The desktop shell
 * opens the folder it works out itself and sends nothing from the page, so the
 * channel cannot be pointed at another folder; when the backend does not answer
 * (the case where the logs matter most) the path the shell reports is shown.
 */

import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, Label } from '../../theme/components';
import { getLogInfo } from '../../services/api';

export default function LogFilesRow() {
  const { t } = useTranslation('settings');
  const [info, setInfo] = useState(null);          // backend answer, or null
  const [infoFailed, setInfoFailed] = useState(false);
  const [shellPath, setShellPath] = useState(null); // what the desktop shell opened
  const [revealed, setRevealed] = useState(false);
  const [openError, setOpenError] = useState('');
  const [copied, setCopied] = useState(false);

  const bridge = typeof window !== 'undefined' ? window.electronAPI : null;
  const canOpen = Boolean(bridge && typeof bridge.openLogFolder === 'function');

  useEffect(() => {
    let cancelled = false;
    getLogInfo()
      .then((data) => { if (!cancelled) { setInfo(data); setInfoFailed(false); } })
      .catch(() => { if (!cancelled) setInfoFailed(true); });
    return () => { cancelled = true; };
  }, []);

  const handleClick = async () => {
    setRevealed(true);
    setOpenError('');
    // The backend may have come up since the page loaded.
    if (!info) {
      getLogInfo().then((d) => { setInfo(d); setInfoFailed(false); }).catch(() => setInfoFailed(true));
    }
    if (!canOpen) return;
    try {
      const result = await bridge.openLogFolder();
      if (result && result.path) setShellPath(result.path);
      if (result && !result.ok) setOpenError(result.error || 'unknown');
    } catch (err) {
      setOpenError(err && err.message ? err.message : String(err));
    }
  };

  const path = (info && info.log_dir) || shellPath;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(path);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // No clipboard permission: the path is in a selectable field right there.
    }
  };

  const buttonStyle = {
    padding: '6px 14px', background: colors.border, border: 'none',
    borderRadius: 4, color: colors.text, fontSize: '12px', cursor: 'pointer',
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div>
        <button onClick={handleClick} title={t('settings:logs.tooltip')} style={buttonStyle}>
          {t('settings:logs.showFiles')}
        </button>
      </div>
      {revealed && (
        <>
          {path ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <span style={{ fontSize: '11px', color: colors.textSecondary }}>
                {t('settings:logs.folderLabel')}
              </span>
              <input
                readOnly
                value={path}
                onFocus={(e) => e.target.select()}
                aria-label={t('settings:logs.folderLabel')}
                style={{
                  flex: 1, minWidth: 240, fontFamily: 'monospace', fontSize: '11px',
                  background: colors.bgTertiary, color: colors.text,
                  border: `1px solid ${colors.border}`, borderRadius: 3, padding: '3px 6px',
                }}
              />
              <button onClick={handleCopy} style={buttonStyle}>
                {copied ? t('settings:logs.copied') : t('settings:logs.copyPath')}
              </button>
            </div>
          ) : (
            infoFailed && (
              <span style={{ fontSize: '11px', color: colors.yellow }}>
                {t('settings:logs.unavailable')}
              </span>
            )
          )}
          {!canOpen && path && (
            <Label secondary style={{ fontSize: '10px', opacity: 0.8 }}>
              {t('settings:logs.browserHint')}
            </Label>
          )}
          {openError && (
            <span style={{ fontSize: '11px', color: colors.yellow }}>
              {t('settings:logs.openFailed', { error: openError })}
            </span>
          )}
          <Label secondary style={{ fontSize: '10px', opacity: 0.7 }}>
            {t('settings:logs.hint')}
          </Label>
        </>
      )}
    </div>
  );
}
