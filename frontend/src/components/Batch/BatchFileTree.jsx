import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, spacing } from '../../theme/tokens';
import useBatchConfigStore from '../../stores/useBatchConfigStore';
import { batchV2Api } from '../../services/api';

// -----------------------------------------------------------------------
// PC status indicators
// -----------------------------------------------------------------------

const PC_STATUS_ICON = {
  refined:   '🟢',
  inherited: '🔵',
  header:    '⚪',
  manual:    '🟡',
  none:      '❌',
};

// Maps a backend pc_status value to its i18n key under "pcStatus".
const PC_STATUS_KEY = {
  refined:   'pcStatus.refined',
  inherited: 'pcStatus.inherited',
  header:    'pcStatus.header',
  manual:    'pcStatus.manual',
  none:      'pcStatus.none',
};

// -----------------------------------------------------------------------
// Small status badge: tick or dash
// -----------------------------------------------------------------------

function StatusBadge({ ok, label }) {
  return (
    <span
      style={{
        fontSize: '10px',
        color: ok ? colors.green : colors.textSecondary,
        marginRight: 4,
      }}
      title={label}
    >
      {label}: {ok ? '✅' : '—'}
    </span>
  );
}

// -----------------------------------------------------------------------
// Single file row
// -----------------------------------------------------------------------

function FileRow({ entry, index, isActive, isChecked, onSelect, onCheck, onRemove }) {
  const { t } = useTranslation('batch');
  const pcIcon  = PC_STATUS_ICON[entry.pc_status]  ?? '❌';
  const pcStatusLabel = t(PC_STATUS_KEY[entry.pc_status] ?? 'pcStatus.none');
  const hasPc   = entry.pc_status !== 'none';
  const pcText  = entry.pc_value
    ? `(${entry.pc_value.map((v) => v.toFixed(3)).join(', ')})`
    : null;

  const rowStyle = {
    display:       'flex',
    flexDirection: 'column',
    gap:           2,
    padding:       `${spacing.compactMargin}px ${spacing.innerMargin}px`,
    marginBottom:  4,
    borderRadius:  4,
    border:        isActive
      ? `1px solid ${colors.accent}`
      : `1px solid ${colors.border}`,
    backgroundColor: colors.bgSecondary,
    cursor:          'pointer',
    userSelect:      'none',
    transition:      'border-color 0.15s',
  };

  return (
    <div style={rowStyle} onClick={() => onSelect(index)} title={t('fileTree.fileRowTooltip')}>
      {/* Row 1: checkbox + name + PC icon + remove */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <input
          type="checkbox"
          checked={isChecked}
          title={t('fileTree.fileCheckboxTooltip')}
          onChange={(e) => {
            e.stopPropagation();
            onCheck(index, e.target.checked);
          }}
          onClick={(e) => e.stopPropagation()}
          style={{ margin: 0, flexShrink: 0, accentColor: colors.accent }}
        />

        {/* File name — truncated, full path on hover */}
        <span
          title={entry.file_path}
          style={{
            flex:         1,
            overflow:     'hidden',
            textOverflow: 'ellipsis',
            whiteSpace:   'nowrap',
            fontSize:     '12px',
            color:        isActive ? colors.accent : colors.text,
            fontWeight:   isActive ? 600 : 400,
          }}
        >
          {entry.file_name}
        </span>

        {/* PC status indicator */}
        <span
          title={t('fileTree.pcStatusTooltip', { status: pcStatusLabel })}
          style={{ fontSize: '12px', flexShrink: 0 }}
        >
          {pcIcon}
        </span>

        {/* Remove file (X) — visible on every row */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            onRemove(index);
          }}
          title={t('fileTree.removeFromBatchTooltip')}
          style={{
            flexShrink: 0,
            width: 18,
            height: 18,
            padding: 0,
            border: 'none',
            background: 'transparent',
            color: colors.textSecondary,
            cursor: 'pointer',
            fontSize: '14px',
            lineHeight: '18px',
            borderRadius: 3,
          }}
          onMouseEnter={(e) => { e.currentTarget.style.color = colors.red; e.currentTarget.style.background = `${colors.red}22`; }}
          onMouseLeave={(e) => { e.currentTarget.style.color = colors.textSecondary; e.currentTarget.style.background = 'transparent'; }}
        >
          ✕
        </button>
      </div>

      {/* Row 2: PC value (when available) */}
      {pcText && (
        <div
          style={{
            fontSize:    '10px',
            color:       colors.textSecondary,
            paddingLeft: 22,
          }}
        >
          PC {pcText}
        </div>
      )}

      {/* Row 3: status icons */}
      <div style={{ display: 'flex', alignItems: 'center', paddingLeft: 22 }}>
        <StatusBadge ok={hasPc}                       label={t('fileTree.badgePc')}  />
        <StatusBadge ok={entry.preprocessing_done}    label={t('fileTree.badgePre')} />
        <StatusBadge ok={entry.phases_configured}     label={t('fileTree.badgePhase')} />

        {/* Ready label */}
        <span
          style={{
            marginLeft:  'auto',
            fontSize:    '10px',
            fontWeight:  600,
            color:       entry.ready ? colors.green : '#ff5555',
          }}
        >
          {entry.ready ? t('fileTree.ready') : t('fileTree.notConfigured')}
        </span>
      </div>
    </div>
  );
}

// -----------------------------------------------------------------------
// Main component
// -----------------------------------------------------------------------

export default function BatchFileTree() {
  const { t } = useTranslation(['batch', 'common']);
  const files          = useBatchConfigStore((s) => s.files);
  const activeFileIndex = useBatchConfigStore((s) => s.activeFileIndex);
  const setActiveFile  = useBatchConfigStore((s) => s.setActiveFile);
  const removeFile     = useBatchConfigStore((s) => s.removeFile);

  // Local checkbox state: Set of checked indices
  const [checkedSet, setCheckedSet] = useState(/** @type {Set<number>} */ (new Set()));

  // Manual PC entry dialog state
  const [pcDialog, setPcDialog] = useState({ open: false, x: '0.500', y: '0.500', z: '0.500' });

  // Inline error/success feedback for action buttons
  const [actionMsg, setActionMsg] = useState(null);  // { kind: 'ok'|'err', text: string }

  // ---- checkbox helpers --------------------------------------------------

  const handleCheck = (index, checked) => {
    setCheckedSet((prev) => {
      const next = new Set(prev);
      if (checked) next.add(index);
      else next.delete(index);
      return next;
    });
  };

  const handleRemove = (index) => {
    // Clear checkbox state for this row, shift higher indices down
    setCheckedSet((prev) => {
      const next = new Set();
      [...prev].forEach((i) => {
        if (i < index) next.add(i);
        else if (i > index) next.add(i - 1);
      });
      return next;
    });
    removeFile(index);
  };

  const checkedIndices = [...checkedSet];

  // ---- action: apply PC to selected -------------------------------------

  const handleApplyPC = async () => {
    if (activeFileIndex < 0 || checkedIndices.length === 0) return;

    const activeFile  = files[activeFileIndex];
    const checkedFiles = checkedIndices.map((i) => files[i]).filter(Boolean);

    // Optimistic local update
    useBatchConfigStore
      .getState()
      .applyConfigToSelected(activeFileIndex, checkedIndices, 'pc');

    // Backend call — surface failures so user knows what to fix
    try {
      const res = await batchV2Api.copyPC(
        activeFile.file_name,
        checkedFiles.map((f) => f.file_name),
      );
      const results = res.data?.results || [];
      const failed = results.filter((r) => !r.success);
      if (failed.length > 0) {
        setActionMsg({
          kind: 'err',
          text: t('fileTree.msgPcCopyPartial', { failed: failed.length, total: results.length, error: failed[0].error || t('fileTree.seeConsole') }),
        });
        console.warn('[BatchFileTree] copyPC partial failure:', failed);
      } else {
        setActionMsg({ kind: 'ok', text: t('fileTree.msgPcCopied', { count: results.length }) });
      }
    } catch (err) {
      const detail = err.response?.data?.detail || err.message || t('fileTree.errUnknown');
      setActionMsg({ kind: 'err', text: t('fileTree.msgCopyPcFailed', { detail }) });
      console.error('[BatchFileTree] copyPC failed:', err);
    }
  };

  // ---- action: apply everything to selected -----------------------------

  const handleApplyAll = async () => {
    if (activeFileIndex < 0 || checkedIndices.length === 0) return;

    const activeFile   = files[activeFileIndex];
    const checkedFiles = checkedIndices.map((i) => files[i]).filter(Boolean);

    useBatchConfigStore
      .getState()
      .applyConfigToSelected(activeFileIndex, checkedIndices, 'all');

    try {
      const res = await batchV2Api.copyPC(
        activeFile.file_name,
        checkedFiles.map((f) => f.file_name),
      );
      const results = res.data?.results || [];
      const failed = results.filter((r) => !r.success);
      if (failed.length > 0) {
        setActionMsg({ kind: 'err', text: t('fileTree.msgApplyAllLackedPc', { failed: failed.length, total: results.length }) });
      } else {
        setActionMsg({ kind: 'ok', text: t('fileTree.msgAppliedTo', { count: results.length }) });
      }
    } catch (err) {
      const detail = err.response?.data?.detail || err.message || t('fileTree.errUnknown');
      setActionMsg({ kind: 'err', text: t('fileTree.msgApplyAllFailed', { detail }) });
      console.error('[BatchFileTree] copyPC (all) failed:', err);
    }
  };

  // ---- action: open manual PC dialog ------------------------------------

  const openPcDialog = () => {
    if (activeFileIndex < 0) return;
    const v = files[activeFileIndex].pc_value;
    setPcDialog({
      open: true,
      x: v ? String(v[0].toFixed(4)) : '0.5000',
      y: v ? String(v[1].toFixed(4)) : '0.5000',
      z: v ? String(v[2].toFixed(4)) : '0.5000',
    });
  };

  const submitManualPC = () => {
    const x = parseFloat(pcDialog.x);
    const y = parseFloat(pcDialog.y);
    const z = parseFloat(pcDialog.z);
    if (![x, y, z].every((v) => Number.isFinite(v))) {
      setActionMsg({ kind: 'err', text: t('fileTree.msgInvalidPc') });
      return;
    }
    if (activeFileIndex < 0) return;
    useBatchConfigStore
      .getState()
      .updateFilePCStatus(activeFileIndex, {
        pc_value: [x, y, z],
        pc_status: 'manual',
        pc_source_file: null,
      });
    setPcDialog({ ...pcDialog, open: false });
    setActionMsg({ kind: 'ok', text: t('fileTree.msgManualPcSet') });
  };

  // ---- action: open active file in EBSD Viewer --------------------------
  // Uses the existing handoff mechanism (Dashboard + HDF5Viewer use the same):
  // stash the path in sessionStorage; EBSDViewer reads it on mount and
  // auto-loads via /api/ebsd/load.

  const openInEBSDViewer = () => {
    if (activeFileIndex < 0) return;
    const path = files[activeFileIndex].file_path;
    try { sessionStorage.setItem('ebsd_preload_path', path); } catch { /* unavailable */ }
    window.dispatchEvent(new CustomEvent('navigate-to', { detail: { page: 'ebsdviewer' } }));
  };

  // -----------------------------------------------------------------------
  // Styles
  // -----------------------------------------------------------------------

  const containerStyle = {
    width:          260,
    height:         '100%',
    display:        'flex',
    flexDirection:  'column',
    backgroundColor: colors.bg,
    borderRight:    `1px solid ${colors.border}`,
    overflow:       'hidden',
  };

  const headerStyle = {
    padding:      `${spacing.innerMargin}px ${spacing.innerMargin}px`,
    borderBottom: `1px solid ${colors.border}`,
    fontSize:     '13px',
    fontWeight:   600,
    color:        colors.accent,
    flexShrink:   0,
  };

  const listStyle = {
    flex:       1,
    overflowY:  'auto',
    padding:    spacing.compactMargin,
    scrollbarWidth: 'thin',
    scrollbarColor: `${colors.border} transparent`,
  };

  const footerStyle = {
    padding:     spacing.compactMargin,
    borderTop:   `1px solid ${colors.border}`,
    display:     'flex',
    flexDirection: 'column',
    gap:         spacing.compactSpacing,
    flexShrink:  0,
  };

  const btnBase = {
    width:           '100%',
    padding:         '6px 8px',
    borderRadius:    4,
    border:          `1px solid ${colors.border}`,
    cursor:          'pointer',
    fontSize:        '11px',
    fontWeight:      500,
    backgroundColor: colors.bgSecondary,
    color:           colors.text,
    transition:      'border-color 0.15s, color 0.15s',
  };

  const btnDisabled = checkedIndices.length === 0 || activeFileIndex < 0;

  const btnActiveStyle = btnDisabled
    ? { ...btnBase, opacity: 0.45, cursor: 'not-allowed' }
    : { ...btnBase, borderColor: colors.accent, color: colors.accent };

  // -----------------------------------------------------------------------

  return (
    <div style={containerStyle}>
      {/* Header */}
      <div style={headerStyle}>
        {t('fileTree.header', { count: files.length })}
      </div>

      {/* Scrollable file list */}
      <div style={listStyle}>
        {files.length === 0 && (
          <div
            style={{
              color:     colors.textSecondary,
              fontSize:  '12px',
              textAlign: 'center',
              marginTop: 24,
            }}
          >
            {t('fileTree.noFiles')}
          </div>
        )}

        {files.map((entry, index) => (
          <FileRow
            key={entry.file_path}
            entry={entry}
            index={index}
            isActive={index === activeFileIndex}
            isChecked={checkedSet.has(index)}
            onSelect={setActiveFile}
            onCheck={handleCheck}
            onRemove={handleRemove}
          />
        ))}
      </div>

      {/* Action buttons */}
      <div style={footerStyle}>
        <button
          style={activeFileIndex < 0 ? { ...btnBase, opacity: 0.45, cursor: 'not-allowed' } : { ...btnBase, borderColor: colors.accent, color: colors.accent }}
          disabled={activeFileIndex < 0}
          onClick={openInEBSDViewer}
          title={t('fileTree.openInViewerTooltip')}
        >
          {t('fileTree.openInViewer')}
        </button>
        <button
          style={activeFileIndex < 0 ? { ...btnBase, opacity: 0.45, cursor: 'not-allowed' } : { ...btnBase, borderColor: colors.accent, color: colors.accent }}
          disabled={activeFileIndex < 0}
          onClick={openPcDialog}
          title={t('fileTree.setPcManuallyTooltip')}
        >
          {t('fileTree.setPcManually')}
        </button>
        <button
          style={btnActiveStyle}
          disabled={btnDisabled}
          onClick={handleApplyPC}
          title={t('fileTree.applyPcSelectedTooltip')}
        >
          {t('fileTree.applyPcSelected')}
        </button>
        <button
          style={btnActiveStyle}
          disabled={btnDisabled}
          onClick={handleApplyAll}
          title={t('fileTree.applyAllSelectedTooltip')}
        >
          {t('fileTree.applyAllSelected')}
        </button>

        {/* Inline action feedback */}
        {actionMsg && (
          <div
            onClick={() => setActionMsg(null)}
            title={t('fileTree.dismissTooltip')}
            style={{
              fontSize: '10px',
              padding: '4px 6px',
              borderRadius: 3,
              cursor: 'pointer',
              color: actionMsg.kind === 'ok' ? colors.green : colors.red,
              background: actionMsg.kind === 'ok' ? `${colors.green}11` : `${colors.red}11`,
              border: `1px solid ${actionMsg.kind === 'ok' ? colors.green : colors.red}33`,
              wordBreak: 'break-word',
            }}
          >
            {actionMsg.text}
          </div>
        )}
      </div>

      {/* Manual PC dialog */}
      {pcDialog.open && (
        <div
          onClick={() => setPcDialog({ ...pcDialog, open: false })}
          style={{
            position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              background: colors.bg,
              border: `1px solid ${colors.accent}`,
              borderRadius: 6,
              padding: 16,
              minWidth: 280,
              boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
            }}
          >
            <div style={{ fontSize: '12px', fontWeight: 600, color: colors.accent, marginBottom: 8 }}>
              {t('fileTree.manualPcTitle')}
            </div>
            <div style={{ fontSize: '10px', color: colors.textSecondary, marginBottom: 10 }}>
              {t('fileTree.manualPcFile', { name: activeFileIndex >= 0 ? files[activeFileIndex].file_name : '—' })}<br/>
              {t('fileTree.manualPcFractional')}
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '40px 1fr', gap: '6px 8px', fontSize: '11px', alignItems: 'center' }}>
              <span>{t('fileTree.pcx')}</span>
              <input
                type="number" step="0.001" value={pcDialog.x}
                title={t('hoverTips.pcXInput')}
                onChange={(e) => setPcDialog({ ...pcDialog, x: e.target.value })}
                style={{ background: colors.bgSecondary, color: colors.text, border: `1px solid ${colors.border}`, borderRadius: 3, padding: '3px 6px', fontSize: '11px' }}
              />
              <span>{t('fileTree.pcy')}</span>
              <input
                type="number" step="0.001" value={pcDialog.y}
                title={t('hoverTips.pcYInput')}
                onChange={(e) => setPcDialog({ ...pcDialog, y: e.target.value })}
                style={{ background: colors.bgSecondary, color: colors.text, border: `1px solid ${colors.border}`, borderRadius: 3, padding: '3px 6px', fontSize: '11px' }}
              />
              <span>{t('fileTree.pcz')}</span>
              <input
                type="number" step="0.001" value={pcDialog.z}
                title={t('hoverTips.pcZInput')}
                onChange={(e) => setPcDialog({ ...pcDialog, z: e.target.value })}
                style={{ background: colors.bgSecondary, color: colors.text, border: `1px solid ${colors.border}`, borderRadius: 3, padding: '3px 6px', fontSize: '11px' }}
              />
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 6, marginTop: 12 }}>
              <button
                onClick={() => setPcDialog({ ...pcDialog, open: false })}
                title={t('hoverTips.pcDialogCancel')}
                style={{ ...btnBase, width: 'auto', padding: '4px 12px' }}
              >
                {t('common:cancel')}
              </button>
              <button
                onClick={submitManualPC}
                title={t('hoverTips.pcDialogApply')}
                style={{ ...btnBase, width: 'auto', padding: '4px 12px', borderColor: colors.accent, color: colors.accent }}
              >
                {t('common:apply')}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
