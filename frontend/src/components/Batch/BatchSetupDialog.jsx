import { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, spacing } from '../../theme/tokens';
import { Button, GroupBox, Input } from '../../theme/components';
import { batchV2Api, indexApi } from '../../services/api';
import useBatchStore from '../../stores/useBatchStore';
import PreFlightReport from './PreFlightReport';

// ---------------------------------------------------------------------------
// Chip toggle button for phase selection
// ---------------------------------------------------------------------------
function PhaseChip({ label, detail, selected, onToggle }) {
  return (
    <button
      onClick={onToggle}
      style={{
        padding: '4px 10px',
        borderRadius: 12,
        fontSize: '10pt',
        cursor: 'pointer',
        border: `1px solid ${selected ? colors.accent : colors.border}`,
        background: selected ? colors.accent : colors.bgSecondary,
        color: selected ? colors.textOnAccent : colors.text,
        transition: 'all 0.12s',
        whiteSpace: 'nowrap',
        display: 'flex',
        alignItems: 'center',
        gap: 4,
      }}
    >
      {label}
      {detail && (
        <span style={{ fontSize: '8pt', opacity: 0.7 }}>{detail}</span>
      )}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Method options
// ---------------------------------------------------------------------------
const METHODS = [
  { value: 'hough', labelKey: 'methods.houghLong', fileHintKey: 'setupDialog.fileHintCif' },
  { value: 'spherical', labelKey: 'methods.sphericalLong', fileHintKey: 'setupDialog.fileHintSht' },
  { value: 'dictionary', labelKey: 'methods.dictionaryLong', fileHintKey: 'setupDialog.fileHintMaster' },
];

// ---------------------------------------------------------------------------
// BatchSetupDialog
// ---------------------------------------------------------------------------
export default function BatchSetupDialog({ open, onClose, onStarted }) {
  const { t } = useTranslation(['batch', 'common']);
  const { createBatch, startBatch, batchId, preflight, totalJobs } = useBatchStore();

  // --- Files state ---
  const [folderPath, setFolderPath] = useState('');
  const [pastedPaths, setPastedPaths] = useState('');
  const [files, setFiles] = useState([]);          // resolved EBSD file paths
  const [scanning, setScanning] = useState(false);

  // --- Method & Phase state ---
  const [method, setMethod] = useState('hough');
  const [discoveredPhases, setDiscoveredPhases] = useState([]);  // full phase objects from discover API
  const [selectedPhaseIndices, setSelectedPhaseIndices] = useState(new Set());
  const [loadingPhases, setLoadingPhases] = useState(false);

  // --- Options ---
  const [autoExport, setAutoExport] = useState(true);
  const [exportDir, setExportDir] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [preflightChecks, setPreflightChecks] = useState([]);

  // --- Load available phases when method changes ---
  useEffect(() => {
    if (!open) return;
    setLoadingPhases(true);
    setDiscoveredPhases([]);
    setSelectedPhaseIndices(new Set());
    indexApi.discoverFiles(method)
      .then((res) => {
        const phaseFiles = res.data?.files ?? [];
        setDiscoveredPhases(phaseFiles);
      })
      .catch(() => setDiscoveredPhases([]))
      .finally(() => setLoadingPhases(false));
  }, [open, method]);

  // --- Build the phases array for the backend ---
  const buildPhases = useCallback(() => {
    return [...selectedPhaseIndices].map((idx) => {
      const p = discoveredPhases[idx];
      return {
        name: p.formula || p.name || p.filename,
        path: p.path,
        method,
      };
    });
  }, [selectedPhaseIndices, discoveredPhases, method]);

  // --- Refresh pre-flight whenever files or phases change ---
  useEffect(() => {
    const phases = buildPhases();
    if (files.length === 0 || phases.length === 0) {
      setPreflightChecks([]);
      return;
    }
    setLoading(true);
    createBatch(files, phases, { auto_export: autoExport, export_dir: exportDir })
      .then((data) => {
        setPreflightChecks(data.preflight?.checks ?? []);
        setError(null);
      })
      .catch((e) => {
        setError(e.message || t('setupDialog.preflightFailedMsg'));
      })
      .finally(() => setLoading(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [files, selectedPhaseIndices, autoExport, exportDir]);

  // --- Add folder: scan for EBSD files ---
  const handleAddFolder = async () => {
    let target = folderPath.trim();

    // Try native Electron folder dialog
    if (window.electronAPI?.openFolder) {
      const selected = await window.electronAPI.openFolder();
      if (!selected) return;
      target = selected;
    } else if (!target) {
      return;
    }

    setScanning(true);
    setError(null);
    try {
      const res = await batchV2Api.scanFolder(target);
      const found = res.data?.files ?? [];
      if (found.length === 0) {
        setError(t('setupDialog.noEbsdFiles', { target }));
      } else {
        setFiles((prev) => [...new Set([...prev, ...found])]);
      }
    } catch (e) {
      setError(e.response?.data?.detail || e.message || t('setupDialog.folderScanFailed'));
    } finally {
      setScanning(false);
      setFolderPath('');
    }
  };

  const handleRemoveFile = (path) => {
    setFiles((prev) => prev.filter((f) => f !== path));
  };

  const handleClearFiles = () => setFiles([]);

  const togglePhase = (idx) => {
    setSelectedPhaseIndices((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  };

  const preflightOk =
    preflightChecks.length > 0 && preflightChecks.every((c) => c.status !== 'fail');

  const handleStart = async () => {
    if (!preflightOk) return;
    setLoading(true);
    try {
      await startBatch();
      onStarted?.();
      onClose?.();
    } catch (e) {
      setError(e.message || t('setupDialog.startFailedMsg'));
    } finally {
      setLoading(false);
    }
  };

  if (!open) return null;

  const methodInfo = METHODS.find((m) => m.value === method);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={t('setupDialog.ariaLabel')}
      style={{
        position: 'fixed', inset: 0,
        background: 'rgba(0,0,0,0.65)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        zIndex: 1000,
      }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose?.(); }}
    >
      <div style={{
        background: colors.bgSecondary,
        border: `1px solid ${colors.border}`,
        borderRadius: 8,
        padding: spacing.outerMargin,
        width: 640,
        maxHeight: '90vh',
        overflow: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: spacing.groupSpacing,
      }}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: '13pt', fontWeight: 700, color: colors.accent }}>
            {t('setupDialog.title')}
          </span>
          <button
            onClick={onClose}
            style={{ background: 'none', border: 'none', color: colors.textSecondary, cursor: 'pointer', fontSize: 18 }}
            aria-label={t('common:close')}
          >{'\u00D7'}</button>
        </div>

        {/* ---- STEP 1: EBSD Files ---- */}
        <GroupBox title={t('setupDialog.filesTitle')}>
          <div style={{ display: 'flex', gap: 6, marginBottom: 6 }}>
            <Input
              value={folderPath}
              onChange={(e) => setFolderPath(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleAddFolder()}
              placeholder={t('setupDialog.folderPlaceholder')}
              style={{ flex: 1 }}
            />
            <Button onClick={handleAddFolder} variant="default" small disabled={scanning}>
              {scanning ? t('setupDialog.scanning') : t('setupDialog.addFolder')}
            </Button>
          </div>
          {files.length > 0 && (
            <>
              <div style={{
                maxHeight: 120, overflow: 'auto',
                display: 'flex', flexDirection: 'column', gap: 2, marginBottom: 6,
                border: `1px solid ${colors.border}`, borderRadius: 4, padding: 4,
                background: colors.bg,
              }}>
                {files.map((f) => (
                  <div key={f} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '9pt', color: colors.text }}>
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }} title={f}>
                      {f.split(/[/\\]/).pop()}
                    </span>
                    <button
                      onClick={() => handleRemoveFile(f)}
                      style={{
                        background: 'none', border: 'none', color: colors.red,
                        cursor: 'pointer', fontSize: 14, padding: '0 4px', marginLeft: 4,
                      }}
                    >{'\u00D7'}</button>
                  </div>
                ))}
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '9pt', color: colors.textSecondary }}>
                  {t('setupDialog.filesFound', { count: files.length })}
                </span>
                <Button onClick={handleClearFiles} variant="danger" small>{t('setupDialog.clearAll')}</Button>
              </div>
            </>
          )}
          {/* Paste fallback */}
          <details style={{ marginTop: 6 }}>
            <summary style={{ fontSize: '9pt', color: colors.textSecondary, cursor: 'pointer' }}>
              {t('setupDialog.pasteSummary')}
            </summary>
            <textarea
              value={pastedPaths}
              onChange={(e) => {
                setPastedPaths(e.target.value);
                // Auto-add pasted paths to files
                const newPaths = e.target.value.split('\n').map((l) => l.trim()).filter(Boolean);
                if (newPaths.length > 0) {
                  setFiles((prev) => [...new Set([...prev, ...newPaths])]);
                }
              }}
              placeholder={t('setupDialog.pastePlaceholder')}
              rows={3}
              style={{
                width: '100%', boxSizing: 'border-box', marginTop: 4,
                background: colors.bg, border: `1px solid ${colors.border}`,
                borderRadius: 4, color: colors.text, fontSize: '9pt',
                padding: '4px 8px', resize: 'vertical',
              }}
            />
          </details>
        </GroupBox>

        {/* ---- STEP 2: Method ---- */}
        <GroupBox title={t('setupDialog.methodTitle')}>
          <div style={{ display: 'flex', gap: 6 }}>
            {METHODS.map((m) => (
              <button
                key={m.value}
                onClick={() => setMethod(m.value)}
                style={{
                  flex: 1,
                  padding: '6px 10px',
                  borderRadius: 6,
                  fontSize: '10pt',
                  cursor: 'pointer',
                  border: `1px solid ${method === m.value ? colors.accent : colors.border}`,
                  background: method === m.value ? colors.accent : 'transparent',
                  color: method === m.value ? colors.textOnAccent : colors.text,
                  transition: 'all 0.12s',
                }}
              >
                {t(m.labelKey)}
              </button>
            ))}
          </div>
        </GroupBox>

        {/* ---- STEP 3: Phase Selection ---- */}
        <GroupBox title={t('setupDialog.phasesTitle', { hint: methodInfo ? t(methodInfo.fileHintKey) : '' })}>
          {loadingPhases ? (
            <span style={{ fontSize: '9pt', color: colors.textSecondary }}>
              {t('setupDialog.discovering')}
            </span>
          ) : discoveredPhases.length === 0 ? (
            <span style={{ fontSize: '9pt', color: colors.textSecondary }}>
              {t('setupDialog.noPhaseFilesForMethod', { method: methodInfo ? t(methodInfo.labelKey) : '' })}
            </span>
          ) : (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {discoveredPhases.map((p, idx) => (
                <PhaseChip
                  key={p.path}
                  label={p.formula || p.name || p.filename}
                  detail={p.space_group || null}
                  selected={selectedPhaseIndices.has(idx)}
                  onToggle={() => togglePhase(idx)}
                />
              ))}
            </div>
          )}
          {selectedPhaseIndices.size > 0 && (
            <div style={{ marginTop: 6, fontSize: '9pt', color: colors.textSecondary }}>
              {t('setupDialog.phasesSelected', { count: selectedPhaseIndices.size })}
              {' \u2014 '}
              {[...selectedPhaseIndices].map((i) => discoveredPhases[i]?.formula || discoveredPhases[i]?.name).join(', ')}
            </div>
          )}
        </GroupBox>

        {/* ---- STEP 4: Options ---- */}
        <GroupBox title={t('setupDialog.optionsTitle')}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '10pt', color: colors.text, cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={autoExport}
              onChange={(e) => setAutoExport(e.target.checked)}
              style={{ accentColor: colors.accent }}
            />
            {t('setupDialog.autoExportResults')}
          </label>
          {autoExport && (
            <div style={{ marginTop: 8 }}>
              <Input
                value={exportDir}
                onChange={(e) => setExportDir(e.target.value)}
                placeholder={t('setupDialog.exportDirPlaceholder')}
              />
            </div>
          )}
        </GroupBox>

        {/* Pre-flight report */}
        {(preflightChecks.length > 0 || loading) && (
          <GroupBox title={t('setupDialog.preflightTitle')}>
            {loading ? (
              <span style={{ fontSize: '10pt', color: colors.textSecondary }}>{t('setupDialog.checking')}</span>
            ) : (
              <PreFlightReport checks={preflightChecks} />
            )}
            {totalJobs > 0 && (
              <div style={{ marginTop: 8, fontSize: '9pt', color: colors.textSecondary }}>
                {t('setupDialog.jobsPlanned', { count: totalJobs, files: files.length, phases: selectedPhaseIndices.size })}
              </div>
            )}
          </GroupBox>
        )}

        {error && (
          <div style={{ color: colors.red, fontSize: '10pt', padding: '4px 0' }}>{error}</div>
        )}

        {/* Actions */}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <Button onClick={onClose} variant="default">{t('common:cancel')}</Button>
          <Button
            onClick={handleStart}
            variant="primary"
            disabled={!preflightOk || loading}
          >
            {t('setupDialog.startBatch', { files: files.length, phases: selectedPhaseIndices.size })}
          </Button>
        </div>
      </div>
    </div>
  );
}
