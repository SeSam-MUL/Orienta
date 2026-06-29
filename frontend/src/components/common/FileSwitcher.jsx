import { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { ebsdApi } from '../../services/api';
import useDataStore from '../../stores/useDataStore';
import useLoadedFilesStore from '../../stores/useLoadedFilesStore';
import { colors } from '../../theme/components';

/**
 * Dropdown to switch between loaded H5OINA files.
 *
 * Switching re-loads the chosen file on the backend (a few seconds for
 * large files) and then refreshes the global data store via
 * syncFromBackend() so every page picks up the new file's EDS elements,
 * electron images, grid shape, etc.
 *
 * Renders nothing when fewer than 2 files are loaded — a switcher with
 * one option is just noise.
 */
export default function FileSwitcher({ onSwitched }) {
  const { t } = useTranslation('shell');
  const [switching, setSwitching] = useState(false);
  const [error, setError] = useState(null);
  const filePath = useDataStore((s) => s.filePath);
  const isFileOpen = useDataStore((s) => s.isFileOpen);
  const syncFromBackend = useDataStore((s) => s.syncFromBackend);
  // Shared registry mirror — the sidebar "Loaded Files" panel and the
  // Indexing dataset dropdown read the same store, so pruning the list
  // anywhere updates this dropdown immediately (no more stale entries).
  const files = useLoadedFilesStore((s) => s.files);
  const refresh = useLoadedFilesStore((s) => s.refresh);

  // Refresh the list whenever the active file changes (a load or a switch).
  useEffect(() => {
    if (isFileOpen) refresh();
  }, [isFileOpen, filePath, refresh]);

  // The backend's `active` flag is authoritative for which file is open.
  // Relying on the store's `filePath` to drive the <select> value breaks
  // after a page reload (the store rehydrates a beat later, or with a
  // different path spelling), leaving the dropdown defaulted to the FIRST
  // option — i.e. showing the WRONG file as selected while a different file
  // is actually open. Derive the selected value from `active` instead, with
  // `filePath` only as a fallback before the list has loaded.
  const activePath = files.find((f) => f.active)?.path || filePath || '';

  const handleSwitch = useCallback(async (path) => {
    if (path === activePath || switching) return;
    setSwitching(true);
    setError(null);
    try {
      await ebsdApi.switchFile(path);
      await syncFromBackend();   // refresh global store → all pages re-read
      await refresh();           // shared registry → all switcher views re-read
      onSwitched?.(path);
    } catch (e) {
      setError(e?.response?.data?.detail || e?.message || 'Switch failed');
    } finally {
      setSwitching(false);
    }
  }, [activePath, switching, syncFromBackend, refresh, onSwitched]);

  // Fewer than 2 files → nothing to switch between.
  if (files.length < 2) return null;

  return (
    <div data-file-switcher style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <label style={{ fontSize: '9pt', color: colors.textSecondary }}>File:</label>
      <select
        aria-label={t('hoverTips.fileSwitcher')}
        title={t('hoverTips.fileSwitcher')}
        value={activePath}
        disabled={switching}
        onChange={(e) => handleSwitch(e.target.value)}
        style={{
          background: colors.bg, color: colors.text,
          border: `1px solid ${colors.border}`, borderRadius: 4,
          padding: '3px 8px', fontSize: '9pt', maxWidth: 280,
          cursor: switching ? 'wait' : 'pointer',
        }}
      >
        {files.map((f) => (
          <option key={f.path} value={f.path}>{f.name}</option>
        ))}
      </select>
      {switching && <span style={{ fontSize: '8.5pt', color: colors.cyan }}>switching…</span>}
      {error && <span style={{ fontSize: '8.5pt', color: colors.red }}>{error}</span>}
    </div>
  );
}
