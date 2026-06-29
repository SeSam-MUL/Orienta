/**
 * Crystal Database — Redesigned orchestrator.
 *
 * Layout:
 *   Toolbar: Build Database, View Database, View DWF Table
 *   Splitter: Left (CIF list with checkboxes + XTAL badges) | Right (Crystal Parameters)
 *   Action Bar: Convert / Batch Convert / Delete Selected
 *   Status Bar: colored status + CIF/XTAL counts
 */

import { useState, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { dbApi } from '../../services/api';
import {
  colors,
  Button, ResizableSplitter,
  usePrompt, PromptDialog,
} from '../../theme/components';

import CrystalToolbar from './CrystalToolbar';
import CifFileList from './CifFileList';
import CrystalParametersPanel from './CrystalParametersPanel';
import ExportBar from './ExportBar';
import DatabaseViewer from './DatabaseViewer';
import DwfDialog from './DwfDialog';
import SyncDialog from './SyncDialog';

// ---------------------------------------------------------------------------
// Status bar color logic
// ---------------------------------------------------------------------------
function statusColor(status) {
  if (!status) return colors.textSecondary;
  if (/fail|error/i.test(status)) return colors.red;
  if (/convert|build|scan|pars/i.test(status)) return colors.yellow;
  if (/complet|added|convert|loaded|sync/i.test(status)) return colors.green;
  return colors.textSecondary;
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function CrystalDatabasePage({ onNavigate, isActive = false }) {
  const { t } = useTranslation('crystaldatabase');

  // --- CIF files state ---
  const [cifFiles, setCifFiles] = useState([]);
  // Each entry: { path, name, checked: false, hasXtal: false, location, parsedInfo: null }

  const [selectedIdx, setSelectedIdx] = useState(-1);
  const [crystalInfo, setCrystalInfo] = useState(null);
  const [infoLoading, setInfoLoading] = useState(false);
  const [xtalCount, setXtalCount] = useState(0);

  // --- Status ---
  const [status, setStatus] = useState(() => t('status.ready'));

  // --- Toolbar state ---
  const [building, setBuilding] = useState(false);
  const [buildProgress, setBuildProgress] = useState(null);
  const [dbViewerEntries, setDbViewerEntries] = useState(null);
  const [dbViewerRich, setDbViewerRich] = useState(false);
  const [showDwf, setShowDwf] = useState(false);

  // --- Convert state ---
  const [converting, setConverting] = useState(false);
  const [batchConverting, setBatchConverting] = useState(false);

  // --- Sync state ---
  const [syncAvailable, setSyncAvailable] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [syncResult, setSyncResult] = useState(null); // null = dialog closed
  const [resolving, setResolving] = useState(false);

  // --- Prompt (for text input dialogs) ---
  const [askPrompt, promptProps] = usePrompt();

  // --- Filter ---
  const [cifFilter, setCifFilter] = useState('');

  // =========================================================================
  // Auto-load CIF files when page becomes active
  // =========================================================================
  useEffect(() => {
    if (!isActive) return;
    if (cifFiles.length > 0) return; // already loaded
    refreshCifList();
    // Check if server sync is available
    dbApi.syncStatus().then(res => {
      setSyncAvailable(!!res.data?.available);
    }).catch(() => {});
  }, [isActive]); // eslint-disable-line react-hooks/exhaustive-deps

  const refreshCifList = useCallback(async () => {
    try {
      const res = await dbApi.browse('cif', '');
      const entries = res.data?.entries || [];
      const files = entries.map((e) => ({
        path: e.path || e.name,
        name: (e.path || e.name || '').split(/[/\\]/).pop(),
        checked: false,
        hasXtal: !!e.has_xtal,
        location: e.location || 'local',
        parsedInfo: null,
      }));
      setCifFiles(files);
      setStatus(t('status.loadedCount', { count: files.length }));

      // Count XTAL files
      const xtalRes = await dbApi.browse('xtal', '');
      setXtalCount((xtalRes.data?.entries || []).length);

      // Batch-parse all CIF files in background
      if (files.length > 0) {
        setStatus(t('status.parsingCount', { count: files.length }));
        try {
          const batchRes = await dbApi.batchCifInfo(files.map(f => f.name));
          const results = batchRes.data?.results || [];
          setCifFiles(prev => prev.map(f => {
            const info = results.find(r => r.filename === f.name);
            return info ? { ...f, parsedInfo: info, hasXtal: !!info.has_xtal } : f;
          }));
          setStatus(t('status.readyLoaded', { count: files.length }));
        } catch {
          setStatus(t('status.loadedBatchUnavailable', { count: files.length }));
        }
      }
    } catch {
      setStatus(t('status.loadFailed'));
    }
  }, [t]);

  // =========================================================================
  // CIF Selection — show crystal parameters (from cache or single-parse)
  // =========================================================================
  const handleSelectCif = useCallback(async (idx) => {
    if (idx === selectedIdx) {
      setSelectedIdx(-1);
      setCrystalInfo(null);
      return;
    }
    setSelectedIdx(idx);
    const file = cifFiles[idx];
    if (!file) return;

    // Use cached parsedInfo if available
    if (file.parsedInfo) {
      setCrystalInfo(file.parsedInfo);
      return;
    }

    // Otherwise parse via API
    setInfoLoading(true);
    setCrystalInfo(null);
    try {
      const res = await dbApi.cifInfo(file.name);
      const info = res.data;
      setCrystalInfo(info);
      setCifFiles(prev => prev.map((f, i) => i === idx ? { ...f, parsedInfo: info } : f));
    } catch (err) {
      setStatus(t('status.parseFailed', { error: err.response?.data?.detail || err.message }));
    } finally {
      setInfoLoading(false);
    }
  }, [selectedIdx, cifFiles, t]);

  // =========================================================================
  // Checkbox toggling
  // =========================================================================
  const handleToggleCheck = useCallback((idx) => {
    setCifFiles(prev => prev.map((f, i) => i === idx ? { ...f, checked: !f.checked } : f));
  }, []);

  const handleCheckAll = useCallback((checked) => {
    setCifFiles(prev => prev.map(f => ({ ...f, checked })));
  }, []);

  const checkedCount = cifFiles.filter(f => f.checked).length;

  // =========================================================================
  // Load files — auto-copy to database
  // =========================================================================
  const handleLoadFiles = async () => {
    if (window.electronAPI?.openFile) {
      const path = await window.electronAPI.openFile({
        filters: [
          { name: 'CIF Files', extensions: ['cif', 'CIF'] },
          { name: 'All', extensions: ['*'] },
        ],
        multiple: true,
      });
      if (path) {
        const paths = Array.isArray(path) ? path : [path];
        await importCifFiles(paths);
      }
    } else {
      const input = await askPrompt({ message: t('prompts.enterCifPaths') });
      if (input?.trim()) {
        const paths = input.split(',').map(p => p.trim()).filter(Boolean);
        await importCifFiles(paths);
      }
    }
  };

  const handleLoadFolder = async () => {
    if (window.electronAPI?.openFolder) {
      const folder = await window.electronAPI.openFolder();
      if (folder) {
        setStatus(t('status.scanning', { folder }));
        // The backend browse will pick up any CIF files in the cif_library
        // For external folders, we'd need to scan and import each file
        // For now, refresh to pick up any manually placed files
        await refreshCifList();
      }
    } else {
      const folder = await askPrompt({ message: t('prompts.enterFolderPath') });
      if (folder?.trim()) {
        setStatus(t('status.folderNoted', { folder: folder.trim() }));
      }
    }
  };

  const importCifFiles = async (paths) => {
    let added = 0;
    let duplicates = 0;
    let failed = 0;
    setStatus(t('status.importing', { count: paths.length }));
    for (const p of paths) {
      try {
        const res = await dbApi.addCif(p);
        if (res.data?.duplicate) {
          duplicates++;
          const name = p.split(/[/\\]/).pop();
          if (res.data.different_content) {
            setStatus(t('status.alreadyDifferentContent', { name }));
          }
        } else {
          added++;
        }
      } catch {
        failed++;
      }
    }
    const parts = [];
    if (added) parts.push(t('status.addedCount', { count: added }));
    if (duplicates) parts.push(t('status.alreadyInDb', { count: duplicates }));
    if (failed) parts.push(t('status.failedCount', { count: failed }));
    setStatus(parts.join(', ') || t('status.noFilesImported'));
    await refreshCifList();
  };

  // =========================================================================
  // Convert to .xtal
  // =========================================================================
  const handleConvert = async () => {
    const file = cifFiles[selectedIdx];
    if (!file) return;
    setConverting(true);
    setStatus(t('status.converting', { name: file.name }));
    try {
      const res = await dbApi.convertCifToXtal(file.path);
      const data = res.data;
      setStatus(t('status.converted', { name: data.name, spaceGroup: data.space_group, atoms: data.n_atoms }));
      // Update hasXtal for this file
      setCifFiles(prev => prev.map((f, i) => i === selectedIdx ? { ...f, hasXtal: true } : f));
      setXtalCount(c => c + 1);
    } catch (err) {
      setStatus(t('status.conversionFailed', { error: err.response?.data?.detail || err.message }));
    } finally {
      setConverting(false);
    }
  };

  const handleBatchConvert = async () => {
    const checked = cifFiles.filter(f => f.checked);
    if (checked.length === 0) return;
    // Skip files that are not fit for xtal conversion
    const skipped = checked.filter(f => f.parsedInfo?.fit_for_xtal === false);
    const convertible = checked.filter(f => f.parsedInfo?.fit_for_xtal !== false);
    if (skipped.length > 0) {
      setStatus(t('status.skippingUnfit', { count: skipped.length }));
    }
    if (convertible.length === 0) {
      setStatus(t('status.noneFit', { count: skipped.length }));
      return;
    }
    setBatchConverting(true);
    let ok = 0, fail = 0;
    for (const file of convertible) {
      try {
        await dbApi.convertCifToXtal(file.path);
        ok++;
        setStatus(t('status.convertingProgress', {
          done: ok + fail,
          total: convertible.length,
          skipped: skipped.length ? t('status.convertingSkippedSuffix', { count: skipped.length }) : '',
        }));
      } catch {
        fail++;
      }
    }
    const parts = [t('status.convertedCount', { count: ok })];
    if (fail) parts.push(t('status.failedCount', { count: fail }));
    if (skipped.length) parts.push(t('status.skippedNotFit', { count: skipped.length }));
    setStatus(t('status.batchComplete', { parts: parts.join(', ') }));
    setBatchConverting(false);
    // Update hasXtal for converted files
    setCifFiles(prev => prev.map(f => {
      if (f.checked && f.parsedInfo?.fit_for_xtal !== false) return { ...f, hasXtal: true, checked: false };
      if (f.checked) return { ...f, checked: false };
      return f;
    }));
    setXtalCount(c => c + ok);
  };

  // =========================================================================
  // Delete selected
  // =========================================================================
  const handleDelete = async (deleteFromServer = false) => {
    const checked = cifFiles.filter(f => f.checked);
    if (checked.length === 0) return;
    const xtalCount = checked.filter(f => f.hasXtal).length;
    const where = deleteFromServer ? t('status.whereLocalServer') : '';
    setStatus(t('status.deleting', {
      cif: checked.length,
      xtal: xtalCount > 0 ? t('status.deletingXtalSuffix', { count: xtalCount }) : '',
      where,
    }));
    try {
      // Delete CIF files
      const cifEntries = checked.map(f => ({ path: f.path, category: 'cif' }));
      // Also delete matching XTAL files (same stem)
      const xtalEntries = checked
        .filter(f => f.hasXtal)
        .map(f => {
          const stem = f.name.replace(/\.cif$/i, '');
          return { path: `${stem}.xtal`, category: 'xtal' };
        });
      const res = await dbApi.deleteFiles([...cifEntries, ...xtalEntries], { deleteFromServer });
      const xtalMsg = xtalEntries.length > 0 ? t('status.deletedXtalSuffix', { count: xtalEntries.length }) : '';
      const serverMsg = res.data?.server_deleted?.length ? t('status.deletedServerSuffix', { count: res.data.server_deleted.length }) : '';
      setStatus(t('status.deleted', { cif: checked.length, xtal: xtalMsg, server: serverMsg }));
      // If selected file was deleted, clear selection
      if (selectedIdx >= 0 && cifFiles[selectedIdx]?.checked) {
        setSelectedIdx(-1);
        setCrystalInfo(null);
      }
      await refreshCifList();
    } catch (err) {
      setStatus(t('status.deleteFailed', { error: err.response?.data?.detail || err.message }));
    }
  };

  // =========================================================================
  // Delete single .xtal (from badge click)
  // =========================================================================
  const handleDeleteXtal = useCallback(async (idx) => {
    const file = cifFiles[idx];
    if (!file || !file.hasXtal) return;
    const stem = file.name.replace(/\.cif$/i, '');
    setStatus(t('status.deletingXtal', { name: `${stem}.xtal` }));
    try {
      await dbApi.deleteFiles([{ path: `${stem}.xtal`, category: 'xtal' }]);
      setCifFiles(prev => prev.map((f, i) => i === idx ? { ...f, hasXtal: false } : f));
      setXtalCount(c => Math.max(0, c - 1));
      setStatus(t('status.deletedXtal', { name: `${stem}.xtal` }));
    } catch (err) {
      setStatus(t('status.deleteFailed', { error: err.response?.data?.detail || err.message }));
    }
  }, [cifFiles, t]);

  // =========================================================================
  // Sync handlers
  // =========================================================================
  const handleSync = async () => {
    setSyncing(true);
    setStatus(t('status.syncingServer'));
    try {
      const res = await dbApi.sync();
      const data = res.data;
      setSyncResult(data);
      const total = (data.uploaded?.length || 0) + (data.downloaded?.length || 0);
      const conflicts = data.conflicts?.length || 0;
      if (conflicts > 0) {
        setStatus(t('status.syncConflicts', { total, conflicts }));
      } else if (total > 0) {
        setStatus(t('status.syncComplete', { total }));
      } else {
        setStatus(t('status.syncUpToDate'));
      }
      await refreshCifList();
    } catch (err) {
      setStatus(t('status.syncFailed', { error: err.response?.data?.detail || err.message }));
    } finally {
      setSyncing(false);
    }
  };

  const handleResolveConflicts = async (resolutions) => {
    setResolving(true);
    try {
      await dbApi.resolveConflicts(resolutions);
      setStatus(t('status.resolved', { count: resolutions.length }));
      setSyncResult(null);
      await refreshCifList();
    } catch (err) {
      setStatus(t('status.resolveFailed', { error: err.response?.data?.detail || err.message }));
    } finally {
      setResolving(false);
    }
  };

  // =========================================================================
  // Toolbar handlers
  // =========================================================================
  const handleBuildDatabase = async () => {
    setBuilding(true);
    setBuildProgress({ progress: 0, total: 0, message: t('status.buildStarting') });
    setStatus(t('status.buildingDatabase'));
    try {
      const res = await dbApi.build(true);
      const taskId = res.data?.task_id;
      if (!taskId) throw new Error('No task ID');

      const poll = setInterval(async () => {
        try {
          const st = await dbApi.buildStatus(taskId);
          const d = st.data;
          setBuildProgress({ progress: d.progress || 0, total: d.total || 0, message: d.message || '' });
          setStatus(t('status.buildingProgress', { message: d.message || '...', progress: d.progress || 0, total: d.total || '?' }));

          if (d.status === 'completed') {
            clearInterval(poll);
            setBuilding(false);
            setBuildProgress(null);
            setStatus(t('status.databaseBuilt', { count: d.entry_count || '?' }));
            await refreshCifList();
          } else if (d.status === 'failed') {
            clearInterval(poll);
            setBuilding(false);
            setBuildProgress(null);
            setStatus(t('status.buildFailed', { error: d.message || d.error || t('status.unknown') }));
          }
        } catch {
          clearInterval(poll);
          setBuilding(false);
          setBuildProgress(null);
          setStatus(t('status.buildPollFailed'));
        }
      }, 1000);
    } catch (err) {
      setBuilding(false);
      setBuildProgress(null);
      setStatus(t('status.buildFailed', { error: err.response?.data?.detail || err.message }));
    }
  };

  const handleViewDatabase = async () => {
    setStatus(t('status.loadingDatabase'));
    try {
      const richRes = await dbApi.entries();
      const richEntries = richRes.data?.entries || [];
      if (richEntries.length > 0) {
        setDbViewerEntries(richEntries);
        setDbViewerRich(true);
        setStatus(t('status.databaseEntries', { count: richEntries.length }));
        return;
      }
    } catch { /* fall through */ }

    try {
      const res = await dbApi.browse('all', '');
      const entries = res.data?.entries || [];
      setDbViewerEntries(entries);
      setDbViewerRich(false);
      setStatus(t('status.fileListing', { count: entries.length }));
    } catch (err) {
      setStatus(t('status.failed', { error: err.response?.data?.detail || err.message }));
    }
  };

  const handleDbViewerLoadCif = (path) => {
    importCifFiles([path]);
    setDbViewerEntries(null);
  };

  const handleDbViewerSave = async (editArray) => {
    try {
      await dbApi.updateEntries(editArray);
      setStatus(t('status.savedChanges', { count: editArray.length }));
      const refreshed = await dbApi.entries();
      if (refreshed.data?.entries?.length) setDbViewerEntries(refreshed.data.entries);
    } catch (err) {
      setStatus(t('status.saveFailed', { error: err.response?.data?.detail || err.message }));
    }
  };

  // =========================================================================
  // Render
  // =========================================================================
  const selectedFile = selectedIdx >= 0 ? cifFiles[selectedIdx] : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', color: colors.text, fontFamily: "'Segoe UI', system-ui, sans-serif" }}>

      {/* Header */}
      <div style={{ marginBottom: 4 }}>
        <h1 style={{ margin: 0, fontSize: '18pt', fontWeight: 700, color: colors.accent }}>{t('page.title')}</h1>
        <p style={{ margin: '2px 0 0', fontSize: 12, color: colors.textSecondary }}>
          {t('page.subtitle')}
        </p>
      </div>

      {/* Toolbar */}
      <CrystalToolbar
        onBuildDatabase={handleBuildDatabase}
        onViewDatabase={handleViewDatabase}
        onViewDWF={() => setShowDwf(true)}
        onSync={handleSync}
        building={building}
        buildProgress={buildProgress}
        syncAvailable={syncAvailable}
        syncing={syncing}
      />

      {/* Modals */}
      {dbViewerEntries != null && (
        <DatabaseViewer
          entries={dbViewerEntries}
          rich={dbViewerRich}
          onClose={() => { setDbViewerEntries(null); setDbViewerRich(false); }}
          onLoadCif={handleDbViewerLoadCif}
          onSave={handleDbViewerSave}
        />
      )}
      {showDwf && <DwfDialog onClose={() => setShowDwf(false)} />}
      {syncResult && (syncResult.conflicts?.length > 0 || syncResult.errors?.length > 0) && (
        <SyncDialog
          syncResult={syncResult}
          onResolve={handleResolveConflicts}
          onClose={() => setSyncResult(null)}
          resolving={resolving}
        />
      )}

      {/* Main splitter */}
      <ResizableSplitter
        defaultLeftWidth={280}
        minLeftWidth={220}
        maxLeftWidth={500}
        style={{ flex: 1, minHeight: 0 }}
        left={
          <CifFileList
            cifFiles={cifFiles}
            selectedIdx={selectedIdx}
            onSelect={handleSelectCif}
            onToggleCheck={handleToggleCheck}
            onCheckAll={handleCheckAll}
            onDeleteXtal={handleDeleteXtal}
            onLoadFiles={handleLoadFiles}
            onLoadFolder={handleLoadFolder}
            filter={cifFilter}
            onFilterChange={setCifFilter}
          />
        }
        right={
          <CrystalParametersPanel
            crystalInfo={crystalInfo}
            loading={infoLoading}
            selectedFileName={selectedFile?.name}
            hasXtal={!!selectedFile?.hasXtal}
            onInfoUpdated={async () => {
              if (!selectedFile) return;
              try {
                const res = await dbApi.cifInfo(selectedFile.name);
                const info = res.data;
                setCrystalInfo(info);
                setCifFiles(prev => prev.map((f, i) =>
                  i === selectedIdx ? { ...f, parsedInfo: info, hasXtal: !!info.has_xtal } : f
                ));
              } catch { /* ignore — will be stale until next click */ }
            }}
          />
        }
      />

      {/* Action bar */}
      <ExportBar
        selectedFile={selectedFile}
        selectedFit={crystalInfo?.fit_for_xtal}
        checkedCount={checkedCount}
        checkedXtalCount={cifFiles.filter(f => f.checked && f.hasXtal).length}
        serverAvailable={syncAvailable}
        converting={converting}
        batchConverting={batchConverting}
        onConvert={handleConvert}
        onBatchConvert={handleBatchConvert}
        onDelete={handleDelete}
      />

      {/* Status bar */}
      <div style={{
        padding: '4px 10px',
        background: colors.bgSecondary,
        borderTop: `1px solid ${colors.border}`,
        fontSize: 11,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexShrink: 0,
        marginTop: 4,
      }}>
        <span style={{ color: statusColor(status) }}>{status}</span>
        <span style={{ color: colors.textSecondary, fontSize: 10 }}>
          {t('counts.cifXtal', { cif: cifFiles.length, xtal: xtalCount })}
        </span>
      </div>

      <PromptDialog {...promptProps} />
    </div>
  );
}
