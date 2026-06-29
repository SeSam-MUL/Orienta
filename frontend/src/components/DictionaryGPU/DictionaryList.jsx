/**
 * DictionaryList — table of existing GPU-generated dictionaries with delete buttons.
 *
 * Backend `/api/dictionary-gpu/list` returns `{ dictionaries: [{ name, path,
 * size_mb, n_patterns, created_at, ... }, ...] }`. We render whatever subset
 * of those fields is present — extra fields are ignored, missing fields show
 * a dash.
 */

import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, spacing, Button, GroupBox } from '../../theme/components';

function fmtSize(mb) {
  if (mb == null) return '—';
  if (mb >= 1024) return `${(mb / 1024).toFixed(2)} GB`;
  return `${mb.toFixed(1)} MB`;
}

function fmtDate(iso) {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

export default function DictionaryList({ dictionaries, loading, onRefresh, onDelete }) {
  const { t } = useTranslation(['dictionarygpu', 'common']);
  const [busyName, setBusyName] = useState(null);

  const handleDelete = async (name) => {
    if (!onDelete) return;
    if (!window.confirm(t('list.deleteConfirm', { name }))) return;
    setBusyName(name);
    try {
      await onDelete(name);
    } finally {
      setBusyName(null);
    }
  };

  const list = Array.isArray(dictionaries) ? dictionaries : [];

  return (
    <GroupBox title={t('list.groupTitle')}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: spacing.innerSpacing }}>
        <span style={{ fontSize: '9pt', color: colors.textSecondary }}>
          {loading ? t('common:loading') : t('list.count', { count: list.length })}
        </span>
        <Button small onClick={onRefresh} disabled={loading} title={t('list.refreshTooltip')}>
          {t('list.refresh')}
        </Button>
      </div>

      {list.length === 0 && !loading && (
        <div
          style={{
            fontSize: '9pt',
            color: colors.textSecondary,
            padding: '12px 8px',
            textAlign: 'center',
            border: `1px dashed ${colors.border}`,
            borderRadius: 4,
            opacity: 0.7,
          }}
        >
          {t('list.empty')}
        </div>
      )}

      {list.length > 0 && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '9pt' }}>
            <thead>
              <tr style={{ color: colors.textSecondary, textAlign: 'left', borderBottom: `1px solid ${colors.border}` }}>
                <th style={{ padding: '4px 6px' }}>{t('list.colName')}</th>
                <th style={{ padding: '4px 6px', textAlign: 'right' }}>{t('list.colPatterns')}</th>
                <th style={{ padding: '4px 6px', textAlign: 'right' }}>{t('list.colSize')}</th>
                <th style={{ padding: '4px 6px' }}>{t('list.colCreated')}</th>
                <th style={{ padding: '4px 6px', width: 80 }}></th>
              </tr>
            </thead>
            <tbody>
              {list.map((d) => (
                <tr key={d.name || d.path} style={{ borderBottom: `1px solid ${colors.border}` }}>
                  <td style={{ padding: '4px 6px', color: colors.text, fontFamily: "'Courier New', monospace", fontSize: '8.5pt' }}>
                    {d.name || d.path}
                  </td>
                  <td style={{ padding: '4px 6px', textAlign: 'right', color: colors.text, fontFamily: 'monospace' }}>
                    {d.n_patterns ?? '—'}
                  </td>
                  <td style={{ padding: '4px 6px', textAlign: 'right', color: colors.text, fontFamily: 'monospace' }}>
                    {fmtSize(d.size_mb)}
                  </td>
                  <td style={{ padding: '4px 6px', color: colors.textSecondary }}>
                    {fmtDate(d.created_at)}
                  </td>
                  <td style={{ padding: '4px 6px', textAlign: 'right' }}>
                    <Button
                      small
                      variant="danger"
                      onClick={() => handleDelete(d.name)}
                      disabled={busyName === d.name || !d.name}
                      title={t('list.deleteTooltip')}
                    >
                      {busyName === d.name ? '…' : t('list.delete')}
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </GroupBox>
  );
}
