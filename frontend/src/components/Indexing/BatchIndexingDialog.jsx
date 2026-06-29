/**
 * BatchIndexingDialog.jsx
 *
 * Modal dialog for batch indexing multiple EBSD files.
 * Supports: file selection, method + phase config per dataset,
 * batch queue, auto-export, and progress tracking.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { indexApi } from '../../services/api';
import { colors as C } from '../../theme/tokens';
import { Button, GroupBox, NumberInput, Select, ProgressBar } from '../../theme/components';

export default function BatchIndexingDialog({ open, onClose }) {
  const { t } = useTranslation('indexing');
  const METHODS = [
    { value: 'hough', label: t('batchDialog.methodHough') },
    { value: 'dictionary', label: t('batchDialog.methodDictionary') },
    { value: 'spherical', label: t('batchDialog.methodSpherical') },
  ];

  const [files, setFiles] = useState([]);
  const [globalMethod, setGlobalMethod] = useState('hough');
  const [globalCifs, setGlobalCifs] = useState('');
  const [globalMasterH5, setGlobalMasterH5] = useState('');
  const [autoExport, setAutoExport] = useState(true);
  const [exportDir, setExportDir] = useState('');
  const [cleanupMemory, setCleanupMemory] = useState(true);

  // Batch state
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState(null);
  const pollRef = useRef(null);

  // Discovered CIF/H5 files from database
  const [availableCifs, setAvailableCifs] = useState([]);
  const [availableH5s, setAvailableH5s] = useState([]);

  useEffect(() => {
    if (!open) return;
    // Load available CIF files for phase selection
    indexApi.discoverFiles('hough').then(r => {
      setAvailableCifs((r.data?.files || []).map(f => f.path));
    }).catch(() => {});
    indexApi.discoverFiles('dictionary').then(r => {
      setAvailableH5s((r.data?.files || []).map(f => f.path));
    }).catch(() => {});
  }, [open]);

  // Add files via text input (one path per line)
  const [fileInput, setFileInput] = useState('');

  const addFiles = useCallback(() => {
    const paths = fileInput.split('\n').map(p => p.trim()).filter(Boolean);
    const newFiles = paths.map(p => ({
      path: p,
      name: p.split(/[/\\]/).pop().replace(/\.[^.]+$/, ''),
      method: globalMethod,
      cifs: globalCifs,
      masterH5: globalMasterH5,
      selectionMode: 'full',
      rowStart: 0, rowEnd: -1, colStart: 0, colEnd: -1,
    }));
    setFiles(prev => [...prev, ...newFiles]);
    setFileInput('');
  }, [fileInput, globalMethod, globalCifs, globalMasterH5]);

  const removeFile = useCallback((idx) => {
    setFiles(prev => prev.filter((_, i) => i !== idx));
  }, []);

  const updateFile = useCallback((idx, updates) => {
    setFiles(prev => prev.map((f, i) => i === idx ? { ...f, ...updates } : f));
  }, []);

  // Apply global settings to all files
  const applyGlobalSettings = useCallback(() => {
    setFiles(prev => prev.map(f => ({
      ...f,
      method: globalMethod,
      cifs: globalCifs,
      masterH5: globalMasterH5,
    })));
  }, [globalMethod, globalCifs, globalMasterH5]);

  // Start batch
  const startBatch = useCallback(async () => {
    if (files.length === 0) return;

    const datasets = files.map(f => ({
      file_path: f.path,
      method: f.method,
      cif_paths: f.cifs ? f.cifs.split(',').map(s => s.trim()).filter(Boolean) : [],
      master_h5_paths: f.masterH5 ? [f.masterH5.trim()] : [],
      sht_paths: [],
      selection_mode: f.selectionMode,
      row_start: f.rowStart,
      row_end: f.rowEnd,
      col_start: f.colStart,
      col_end: f.colEnd,
    }));

    try {
      await indexApi.batchStart(datasets, autoExport, exportDir, cleanupMemory);
      setRunning(true);
    } catch (err) {
      alert(t('batchDialog.startFailed', { error: err.response?.data?.detail || err.message }));
    }
  }, [files, autoExport, exportDir, cleanupMemory]);

  // Poll status
  useEffect(() => {
    if (!running) {
      if (pollRef.current) clearInterval(pollRef.current);
      return;
    }
    const poll = async () => {
      try {
        const res = await indexApi.batchStatus();
        setStatus(res.data);
        if (!res.data.running) {
          setRunning(false);
        }
      } catch { /* ignore */ }
    };
    poll();
    pollRef.current = setInterval(poll, 2000);
    return () => clearInterval(pollRef.current);
  }, [running]);

  const stopBatch = useCallback(async () => {
    try { await indexApi.batchStop(); } catch { /* ignore */ }
  }, []);

  if (!open) return null;

  const progress = status ? (status.completed / Math.max(status.total, 1)) * 100 : 0;

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 1000,
      background: 'rgba(0,0,0,0.6)', display: 'flex',
      alignItems: 'center', justifyContent: 'center',
    }}>
      <div style={{
        background: C.bgSecondary, border: `1px solid ${C.border}`,
        borderRadius: 8, width: '90%', maxWidth: 900, maxHeight: '85vh',
        overflow: 'hidden', display: 'flex', flexDirection: 'column',
      }}>
        {/* Header */}
        <div style={{
          padding: '12px 16px', borderBottom: `1px solid ${C.border}`,
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <h2 style={{ margin: 0, color: C.purple, fontSize: '14pt' }}>{t('batchDialog.title')}</h2>
          <button onClick={onClose} title={t('hoverTips.batchClose')} aria-label={t('hoverTips.batchClose')} style={{
            background: 'none', border: 'none', color: C.text,
            fontSize: '16pt', cursor: 'pointer',
          }}>x</button>
        </div>

        {/* Content */}
        <div style={{ flex: 1, overflow: 'auto', padding: 16, display: 'flex', flexDirection: 'column', gap: 12 }}>

          {/* Global settings */}
          <GroupBox title={t('batchDialog.globalSettings')}>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center', fontSize: '10pt' }}>
              <span style={{ color: C.textSecondary }}>{t('batchDialog.method')}</span>
              <select value={globalMethod} onChange={e => setGlobalMethod(e.target.value)}
                title={t('hoverTips.batchGlobalMethod')}
                style={{ background: C.bg, color: C.text, border: `1px solid ${C.border}`, borderRadius: 4, padding: '2px 4px' }}>
                {METHODS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
              </select>
              <span style={{ color: C.textSecondary }}>{t('batchDialog.cif')}</span>
              <select
                value=""
                onChange={e => { if (e.target.value) setGlobalCifs(prev => prev ? `${prev}, ${e.target.value}` : e.target.value); }}
                title={t('hoverTips.batchPickCif')}
                style={{ background: C.bg, color: C.text, border: `1px solid ${C.border}`, borderRadius: 4, fontSize: '9pt', maxWidth: 160 }}
              >
                <option value="">{t('batchDialog.pickCif')}</option>
                {availableCifs.map(p => <option key={p} value={p}>{p.split(/[/\\]/).pop()}</option>)}
              </select>
              <input type="text" value={globalCifs} onChange={e => setGlobalCifs(e.target.value)}
                placeholder={t('batchDialog.cifPlaceholder')}
                title={t('hoverTips.batchCifInput')}
                style={{ flex: 1, minWidth: 120, background: C.bg, color: C.text, border: `1px solid ${C.border}`, borderRadius: 4, padding: '2px 6px', fontSize: '9pt' }} />
              <span style={{ color: C.textSecondary }}>{t('batchDialog.dictH5')}</span>
              <select
                value=""
                onChange={e => { if (e.target.value) setGlobalMasterH5(e.target.value); }}
                title={t('hoverTips.batchPickH5')}
                style={{ background: C.bg, color: C.text, border: `1px solid ${C.border}`, borderRadius: 4, fontSize: '9pt', maxWidth: 160 }}
              >
                <option value="">{t('batchDialog.pickH5')}</option>
                {availableH5s.map(p => <option key={p} value={p}>{p.split(/[/\\]/).pop()}</option>)}
              </select>
              <input type="text" value={globalMasterH5} onChange={e => setGlobalMasterH5(e.target.value)}
                placeholder={t('batchDialog.dictH5Placeholder')}
                title={t('hoverTips.batchH5Input')}
                style={{ flex: 1, minWidth: 120, background: C.bg, color: C.text, border: `1px solid ${C.border}`, borderRadius: 4, padding: '2px 6px', fontSize: '9pt' }} />
              <button onClick={applyGlobalSettings} title={t('hoverTips.batchApplyToAll')} style={{
                background: C.purple, color: '#fff', border: 'none', borderRadius: 4,
                padding: '3px 10px', cursor: 'pointer', fontSize: '9pt',
              }}>{t('batchDialog.applyToAll')}</button>
            </div>
          </GroupBox>

          {/* File list */}
          <GroupBox title={t('batchDialog.datasets', { count: files.length })}>
            <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
              <textarea
                value={fileInput}
                onChange={e => setFileInput(e.target.value)}
                placeholder={t('batchDialog.filePlaceholder')}
                title={t('hoverTips.batchFilePaths')}
                rows={3}
                style={{
                  flex: 1, background: C.bg, color: C.text, border: `1px solid ${C.border}`,
                  borderRadius: 4, padding: '4px 8px', fontSize: '9pt', fontFamily: 'monospace', resize: 'vertical',
                }}
              />
              <button onClick={addFiles} title={t('hoverTips.batchAddFiles')} style={{
                background: C.green, color: '#000', border: 'none', borderRadius: 4,
                padding: '4px 12px', cursor: 'pointer', fontWeight: 'bold', alignSelf: 'flex-start',
              }}>{t('batchDialog.addFiles')}</button>
            </div>

            {files.length === 0 ? (
              <div style={{ color: C.textSecondary, fontSize: '10pt', textAlign: 'center', padding: 12 }}>
                {t('batchDialog.noFiles')}
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                {files.map((f, i) => (
                  <div key={i} style={{
                    display: 'flex', gap: 6, alignItems: 'center',
                    background: C.bg, borderRadius: 4, padding: '4px 8px', fontSize: '9pt',
                  }}>
                    <span style={{ color: C.accent, fontWeight: 'bold', width: 20 }}>{i + 1}</span>
                    <span style={{ flex: 1, color: C.text, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                      title={f.path}>{f.name}</span>
                    <select value={f.method} onChange={e => updateFile(i, { method: e.target.value })}
                      title={t('hoverTips.batchFileMethod')}
                      style={{ background: C.bgSecondary, color: C.text, border: `1px solid ${C.border}`, borderRadius: 3, padding: '1px 3px', fontSize: '9pt' }}>
                      {METHODS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
                    </select>
                    <select value={f.selectionMode} onChange={e => updateFile(i, { selectionMode: e.target.value })}
                      title={t('hoverTips.batchFileSelection')}
                      style={{ background: C.bgSecondary, color: C.text, border: `1px solid ${C.border}`, borderRadius: 3, padding: '1px 3px', fontSize: '9pt' }}>
                      <option value="full">{t('batchDialog.selFull')}</option>
                      <option value="region">{t('batchDialog.selRegion')}</option>
                    </select>
                    {/* Status indicator */}
                    {status?.results?.find(r => r.file_path === f.path) && (
                      <span style={{
                        color: status.results.find(r => r.file_path === f.path).error ? C.red : C.green,
                        fontWeight: 'bold',
                      }}>
                        {status.results.find(r => r.file_path === f.path).error ? t('batchDialog.statusFail') : t('batchDialog.statusCi', { ci: status.results.find(r => r.file_path === f.path).mean_ci?.toFixed(3) || '?' })}
                      </span>
                    )}
                    <button onClick={() => removeFile(i)} title={t('hoverTips.batchRemoveFile')} aria-label={t('hoverTips.batchRemoveFile')} style={{
                      background: 'none', border: 'none', color: C.red, cursor: 'pointer', fontWeight: 'bold',
                    }}>X</button>
                  </div>
                ))}
              </div>
            )}
          </GroupBox>

          {/* Export options */}
          <GroupBox title={t('batchDialog.exportMemory')}>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', fontSize: '10pt' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: 4, color: C.text, cursor: 'pointer' }} title={t('hoverTips.batchAutoExport')}>
                <input type="checkbox" checked={autoExport} onChange={e => setAutoExport(e.target.checked)} style={{ accentColor: C.green }} />
                {t('batchDialog.autoExport')}
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: 4, color: C.text, cursor: 'pointer' }} title={t('hoverTips.batchFreeMemory')}>
                <input type="checkbox" checked={cleanupMemory} onChange={e => setCleanupMemory(e.target.checked)} style={{ accentColor: C.cyan }} />
                {t('batchDialog.freeMemory')}
              </label>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span style={{ color: C.textSecondary }}>{t('batchDialog.exportDir')}</span>
                <input type="text" value={exportDir} onChange={e => setExportDir(e.target.value)}
                  placeholder={t('batchDialog.exportDirPlaceholder')}
                  title={t('hoverTips.batchExportDir')}
                  style={{ width: 200, background: C.bg, color: C.text, border: `1px solid ${C.border}`, borderRadius: 4, padding: '2px 6px', fontSize: '9pt' }} />
              </div>
            </div>
          </GroupBox>

          {/* Progress */}
          {status && (
            <GroupBox title={t('batchDialog.progress')}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10pt' }}>
                  <span style={{ color: C.text }}>
                    {status.running ? t('batchDialog.processing', { name: status.current_dataset }) : t('batchDialog.batchComplete')}
                  </span>
                  <span style={{ color: C.accent }}>{status.completed}/{status.total}</span>
                </div>
                <ProgressBar value={progress} />
                <div style={{
                  background: C.bg, borderRadius: 4, padding: 8,
                  maxHeight: 150, overflow: 'auto', fontFamily: 'monospace', fontSize: '9pt', color: C.textSecondary,
                }}>
                  {(status.log || []).slice(-20).map((line, i) => (
                    <div key={i} style={{ color: line.includes('ERROR') ? C.red : line.includes('Exported') ? C.green : C.textSecondary }}>
                      {line}
                    </div>
                  ))}
                </div>
                {/* Results summary */}
                {status.results?.length > 0 && (
                  <div style={{ fontSize: '9pt' }}>
                    <span style={{ color: C.green }}>
                      {t('batchDialog.succeeded', { count: status.results.filter(r => !r.error).length })}
                    </span>
                    {status.results.some(r => r.error) && (
                      <span style={{ color: C.red, marginLeft: 8 }}>
                        {t('batchDialog.failed', { count: status.results.filter(r => r.error).length })}
                      </span>
                    )}
                  </div>
                )}
              </div>
            </GroupBox>
          )}
        </div>

        {/* Footer */}
        <div style={{
          padding: '10px 16px', borderTop: `1px solid ${C.border}`,
          display: 'flex', gap: 8, justifyContent: 'flex-end',
        }}>
          {running ? (
            <button onClick={stopBatch} title={t('hoverTips.batchStop')} style={{
              background: C.red, color: '#fff', border: 'none', borderRadius: 4,
              padding: '6px 16px', cursor: 'pointer', fontWeight: 'bold',
            }}>{t('batchDialog.stopBatch')}</button>
          ) : (
            <button onClick={startBatch} disabled={files.length === 0} title={t('hoverTips.batchStart')} style={{
              background: files.length > 0 ? C.green : C.border,
              color: files.length > 0 ? '#000' : C.textSecondary,
              border: 'none', borderRadius: 4,
              padding: '6px 16px', cursor: files.length > 0 ? 'pointer' : 'not-allowed',
              fontWeight: 'bold',
            }}>{t('batchDialog.startBatch', { count: files.length })}</button>
          )}
          <button onClick={onClose} title={t('hoverTips.batchCloseFooter')} style={{
            background: C.border, color: C.text, border: 'none', borderRadius: 4,
            padding: '6px 16px', cursor: 'pointer',
          }}>{t('batchDialog.close')}</button>
        </div>
      </div>
    </div>
  );
}
