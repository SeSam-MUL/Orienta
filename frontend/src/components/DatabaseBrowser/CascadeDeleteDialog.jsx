/**
 * CascadeDeleteDialog — modal for confirming deletion of simulation files
 * with cascade dependency display.
 *
 * Props:
 *   open        — boolean, whether the dialog is visible
 *   cascadeData — { selected: [], associated: [], has_local: bool, has_server: bool }
 *   deleting    — boolean, show loading state while delete is in progress
 *   onConfirm   — (deleteFrom: string, files: array) => void
 *   onCancel    — () => void
 */

import { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, alpha, spacing, Button } from '../../theme/components';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function formatBytes(bytes) {
  if (bytes == null || bytes === '') return '—';
  const n = Number(bytes);
  if (isNaN(n) || n === 0) return n === 0 ? '0 B' : String(bytes);
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let v = n;
  let ui = 0;
  while (v >= 1024 && ui < units.length - 1) { v /= 1024; ui++; }
  return `${v.toFixed(1)} ${units[ui]}`;
}

function locationLabel(loc, t) {
  if (!loc) return '—';
  if (loc === 'both') return t('databasebrowser:cascade.locationBoth');
  return loc;
}

// ---------------------------------------------------------------------------
// FileItem — one row in selected or associated list
// ---------------------------------------------------------------------------
function FileItem({ file, checked, onToggle, showCheckbox }) {
  const { t } = useTranslation('databasebrowser');
  return (
    <div style={{
      display: 'flex',
      alignItems: 'flex-start',
      gap: 8,
      padding: '5px 0',
      borderBottom: `1px solid ${alpha(colors.border, 30)}`,
    }}>
      {showCheckbox && (
        <input
          type="checkbox"
          checked={checked}
          disabled={file.required}
          onChange={() => onToggle && onToggle(file.name)}
          title={t('databasebrowser:cascade.associatedTooltip')}
          style={{ marginTop: 3, cursor: file.required ? 'not-allowed' : 'pointer', flexShrink: 0 }}
        />
      )}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 6, flexWrap: 'wrap' }}>
          <span style={{ fontSize: 9, fontWeight: 700, color: colors.text, wordBreak: 'break-all' }}>
            {file.name}
          </span>
          {file.required && (
            <span style={{ fontSize: 8, color: colors.orange, flexShrink: 0 }}>{t('databasebrowser:cascade.required')}</span>
          )}
        </div>
        <div style={{ fontSize: 8, color: colors.textSecondary, marginTop: 1, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {file.material && <span>{file.material}</span>}
          {file.category && <span>{file.category}</span>}
          {file.location && <span>{locationLabel(file.location, t)}</span>}
          {file.size != null && <span>{formatBytes(file.size)}</span>}
        </div>
        {file.reason && (
          <div style={{ fontSize: 8, color: colors.textSecondary, marginTop: 1 }}>
            → {file.reason}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// CascadeDeleteDialog
// ---------------------------------------------------------------------------
export default function CascadeDeleteDialog({ open, cascadeData, deleting, onConfirm, onCancel }) {
  const { t } = useTranslation(['databasebrowser', 'common']);
  // Track which associated files are checked (by name)
  const [checkedNames, setCheckedNames] = useState(new Set());

  // Reset checkboxes whenever cascadeData changes
  useEffect(() => {
    if (!cascadeData) return;
    const all = new Set((cascadeData.associated || []).map((f) => f.name));
    setCheckedNames(all);
  }, [cascadeData]);

  const handleToggle = useCallback((name) => {
    setCheckedNames((prev) => {
      const next = new Set(prev);
      if (next.has(name)) {
        next.delete(name);
      } else {
        next.add(name);
      }
      return next;
    });
  }, []);

  const handleOverlayClick = useCallback(() => {
    if (!deleting) onCancel?.();
  }, [deleting, onCancel]);

  const handleModalClick = useCallback((e) => {
    e.stopPropagation();
  }, []);

  if (!open || !cascadeData) return null;

  const selected = cascadeData.selected || [];
  const associated = cascadeData.associated || [];
  const { has_local, has_server } = cascadeData;

  // Files that will be deleted on confirm
  const checkedAssociated = associated.filter((f) => checkedNames.has(f.name) || f.required);

  // Total size = selected + checked associated
  const totalBytes = [...selected, ...checkedAssociated].reduce((acc, f) => {
    const n = Number(f.size);
    return acc + (isNaN(n) ? 0 : n);
  }, 0);

  // Build the file list for onConfirm
  const buildFileList = () =>
    [...selected, ...checkedAssociated].map((f) => ({
      path: f.name,
      category: f.category,
      material: f.material,
    }));

  const handleConfirm = (scope) => {
    onConfirm?.(scope, buildFileList());
  };

  // ---------------------------------------------------------------------------
  // Styles
  // ---------------------------------------------------------------------------
  const overlay = {
    position: 'fixed',
    inset: 0,
    background: alpha(colors.bg, 80),
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1200,
  };

  const modal = {
    background: colors.bg,
    border: `1px solid ${colors.border}`,
    borderRadius: 8,
    boxShadow: `0 8px 32px ${alpha(colors.border, 40)}`,
    width: 480,
    maxWidth: '92vw',
    maxHeight: '80vh',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  };

  const sectionTitle = {
    fontSize: 8,
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.08em',
    color: colors.textSecondary,
    margin: '10px 0 4px',
  };

  const scrollArea = {
    overflowY: 'auto',
    flex: 1,
    padding: `0 ${spacing.innerMargin}px`,
  };

  const footer = {
    padding: `${spacing.innerMargin}px`,
    borderTop: `1px solid ${colors.border}`,
    display: 'flex',
    gap: 8,
    justifyContent: 'flex-end',
    flexWrap: 'wrap',
    alignItems: 'center',
    flexShrink: 0,
  };

  const serverOnlyStyle = {
    background: alpha(colors.red, 15),
    color: colors.red,
    border: `1px solid ${alpha(colors.red, 30)}`,
  };

  return (
    <div style={overlay} onClick={handleOverlayClick}>
      <div style={modal} onClick={handleModalClick}>
        {/* Header */}
        <div style={{
          padding: `${spacing.innerMargin}px`,
          borderBottom: `1px solid ${colors.border}`,
          flexShrink: 0,
        }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: colors.red }}>
            {t('databasebrowser:cascade.deleteCount', { count: selected.length + checkedAssociated.length })}
          </div>
          {totalBytes > 0 && (
            <div style={{ fontSize: 9, color: colors.textSecondary, marginTop: 3 }}>
              {t('databasebrowser:cascade.totalSize', { size: formatBytes(totalBytes) })}
            </div>
          )}
        </div>

        {/* Body */}
        <div style={scrollArea}>
          {/* Selected — always deleted */}
          {selected.length > 0 && (
            <>
              <div style={sectionTitle}>{t('databasebrowser:cascade.selectedSection', { count: selected.length })}</div>
              {selected.map((f) => (
                <FileItem key={f.name} file={f} showCheckbox={false} />
              ))}
            </>
          )}

          {/* Associated — opt-out with checkboxes */}
          {associated.length > 0 && (
            <>
              <div style={sectionTitle}>{t('databasebrowser:cascade.associatedSection', { count: associated.length })}</div>
              {associated.map((f) => (
                <FileItem
                  key={f.name}
                  file={f}
                  showCheckbox
                  checked={checkedNames.has(f.name) || !!f.required}
                  onToggle={f.required ? undefined : handleToggle}
                />
              ))}
            </>
          )}

          {/* Spacer so last item isn't right against the footer */}
          <div style={{ height: 8 }} />
        </div>

        {/* Footer buttons */}
        <div style={footer}>
          {has_local && !has_server && (
            <Button variant="danger" disabled={deleting} onClick={() => handleConfirm('local')}
              title={t('databasebrowser:cascade.scopeTooltip')}>
              {deleting ? t('databasebrowser:cascade.deleting') : t('databasebrowser:cascade.localOnly')}
            </Button>
          )}

          {has_server && !has_local && (
            <Button
              disabled={deleting}
              onClick={() => handleConfirm('server')}
              title={t('databasebrowser:cascade.scopeTooltip')}
              style={serverOnlyStyle}
            >
              {deleting ? t('databasebrowser:cascade.deleting') : t('databasebrowser:cascade.serverOnly')}
            </Button>
          )}

          {has_local && has_server && (
            <>
              <Button variant="danger" disabled={deleting} onClick={() => handleConfirm('local')}
                title={t('databasebrowser:cascade.scopeTooltip')}>
                {deleting ? t('databasebrowser:cascade.deleting') : t('databasebrowser:cascade.localOnly')}
              </Button>
              <Button
                disabled={deleting}
                onClick={() => handleConfirm('server')}
                title={t('databasebrowser:cascade.scopeTooltip')}
                style={serverOnlyStyle}
              >
                {deleting ? t('databasebrowser:cascade.deleting') : t('databasebrowser:cascade.serverOnly')}
              </Button>
              <Button variant="danger" disabled={deleting} onClick={() => handleConfirm('everywhere')}
                title={t('databasebrowser:cascade.scopeTooltip')}>
                {deleting ? t('databasebrowser:cascade.deleting') : t('databasebrowser:cascade.everywhere')}
              </Button>
            </>
          )}

          <Button disabled={deleting} onClick={onCancel} title={t('databasebrowser:cascade.cancelTooltip')}>
            {t('common:cancel')}
          </Button>
        </div>
      </div>
    </div>
  );
}
