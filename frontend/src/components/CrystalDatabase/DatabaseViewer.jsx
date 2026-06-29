import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, alpha, Button, Input } from '../../theme/components';

// DatabaseViewer — spreadsheet-style viewer matching PyQt5 ExcelViewerDialog
// Search, sort by column header click, double-click to load CIF
// Supports two modes: rich (crystal_database.xlsx, 13 cols) and simple (file browse, 5 cols)
export default function DatabaseViewer({ entries, rich, onClose, onLoadCif, onSave }) {
  const { t } = useTranslation(['crystaldatabase', 'common']);
  const [search, setSearch] = useState('');
  const [sortCol, setSortCol] = useState(null);
  const [sortAsc, setSortAsc] = useState(true);
  // edits: Map of "<entryIndex>|Phase Name" -> new value
  const [edits, setEdits] = useState({});
  // editingCell: { rowKey, col } | null
  const [editingCell, setEditingCell] = useState(null);
  const [editingValue, setEditingValue] = useState('');
  const [saving, setSaving] = useState(false);

  // Rich columns from crystal_database.xlsx (matches PyQt5 ExcelViewerDialog — all 13 cols)
  const richColumns = [
    { key: 'Phase Name', label: t('databaseViewer.richCols.phaseName'), flex: 1.5, editable: true },
    { key: 'Composition', label: t('databaseViewer.richCols.composition'), flex: 1.5 },
    { key: 'Space Group', label: t('databaseViewer.richCols.spaceGroup'), flex: 1 },
    { key: 'Crystal System', label: t('databaseViewer.richCols.crystalSystem'), flex: 1 },
    { key: 'Lattice Parameters (Å)', label: t('databaseViewer.richCols.latticeParams'), flex: 2 },
    { key: 'DOI', label: t('databaseViewer.richCols.doi'), flex: 1.5, editable: true },
    { key: 'Reference Text', label: t('databaseViewer.richCols.referenceText'), flex: 2, editable: true },
    { key: 'Fit for .xtal', label: t('databaseViewer.richCols.fitForXtal'), flex: 0.7 },
    { key: 'Structure Fingerprint', label: t('databaseViewer.richCols.fingerprint'), flex: 1.5 },
    { key: 'Warnings', label: t('databaseViewer.richCols.warnings'), flex: 1.5 },
    { key: 'File Path', label: t('databaseViewer.richCols.filePath'), flex: 2 },
    { key: 'Possible Duplicates', label: t('databaseViewer.richCols.duplicates'), flex: 1.5 },
    { key: 'CIF File Name', label: t('databaseViewer.richCols.cifFileName'), flex: 2 },
  ];

  // Simple columns from file browse
  const simpleColumns = [
    { key: 'name', label: t('databaseViewer.simpleCols.name'), flex: 3 },
    { key: 'category', label: t('databaseViewer.simpleCols.type'), flex: 1 },
    { key: 'material', label: t('databaseViewer.simpleCols.material'), flex: 1.5 },
    { key: 'size', label: t('databaseViewer.simpleCols.size'), flex: 1, align: 'right', format: (v) => v ? `${(v / 1024).toFixed(1)} KB` : '—' },
    { key: 'location', label: t('databaseViewer.simpleCols.location'), flex: 1 },
  ];

  const columns = rich ? richColumns : simpleColumns;

  // Filter — search across all columns (including edited Phase Name values)
  const searchLower = search.toLowerCase();
  const filtered = search
    ? entries.filter((e, i) => columns.some((c) => {
        const val = (c.key === 'Phase Name' && edits[`${i}|Phase Name`] != null)
          ? edits[`${i}|Phase Name`]
          : String(e[c.key] || '');
        return val.toLowerCase().includes(searchLower);
      }))
    : entries;

  // Attach original index so we can map back after sort
  const filteredWithIdx = filtered.map((e) => {
    const origIdx = entries.indexOf(e);
    return { entry: e, origIdx };
  });

  // Sort (sort index col offset by 1 because of the # column)
  const dataColIdx = sortCol != null ? sortCol - 1 : null;
  const sortedWithIdx = dataColIdx != null && dataColIdx >= 0
    ? [...filteredWithIdx].sort((a, b) => {
        const colKey = columns[dataColIdx].key;
        const va = (colKey === 'Phase Name' && edits[`${a.origIdx}|Phase Name`] != null)
          ? edits[`${a.origIdx}|Phase Name`]
          : (a.entry[colKey] ?? '');
        const vb = (colKey === 'Phase Name' && edits[`${b.origIdx}|Phase Name`] != null)
          ? edits[`${b.origIdx}|Phase Name`]
          : (b.entry[colKey] ?? '');
        const cmp = typeof va === 'number' && typeof vb === 'number'
          ? va - vb
          : String(va).localeCompare(String(vb));
        return sortAsc ? cmp : -cmp;
      })
    : filteredWithIdx;

  const handleHeaderClick = (colIdx) => {
    // colIdx 0 is the # column — not sortable
    if (colIdx === 0) return;
    if (sortCol === colIdx) {
      setSortAsc(!sortAsc);
    } else {
      setSortCol(colIdx);
      setSortAsc(true);
    }
  };

  // Inline edit helpers (Phase Name only)
  const startEdit = (rowKey, currentValue) => {
    setEditingCell(rowKey);
    setEditingValue(currentValue);
  };
  const commitEdit = (rowKey) => {
    if (editingCell !== rowKey) return;
    setEdits((prev) => ({ ...prev, [rowKey]: editingValue }));
    setEditingCell(null);
  };
  const cancelEdit = () => {
    setEditingCell(null);
  };

  const modifiedCount = Object.keys(edits).length;

  const handleSave = async () => {
    if (!onSave || modifiedCount === 0) return;
    setSaving(true);
    const editArray = Object.entries(edits).map(([key, value]) => {
      const [rowStr, column] = key.split('|');
      return { row: parseInt(rowStr, 10), column, value };
    });
    try {
      await onSave(editArray);
    } finally {
      setSaving(false);
    }
  };

  // Summary stats
  const counts = {};
  if (rich) {
    // Count by Crystal System
    entries.forEach((e) => {
      const cs = e['Crystal System'] || 'Unknown';
      counts[cs] = (counts[cs] || 0) + 1;
    });
  } else {
    entries.forEach((e) => { counts[e.category] = (counts[e.category] || 0) + 1; });
  }
  const fitCount = rich ? entries.filter((e) => e['Fit for .xtal'] === 'Yes').length : null;

  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
      background: 'rgba(0,0,0,0.6)', zIndex: 1000,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
    }}
    onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div style={{
        background: colors.bg, border: `1px solid ${colors.border}`, borderRadius: 8,
        width: rich ? '96%' : '85%', maxWidth: rich ? 1600 : 1000, height: '80vh',
        display: 'flex', flexDirection: 'column', boxShadow: '0 12px 40px rgba(0,0,0,0.4)',
        animation: 'fadeSlideIn 0.2s ease-out',
      }}>
        {/* Header */}
        <div style={{ padding: '12px 16px', borderBottom: `1px solid ${colors.border}`, display: 'flex', alignItems: 'center', gap: 12, flexShrink: 0 }}>
          <span style={{ fontSize: 14, fontWeight: 700, color: colors.accent }}>{t('databaseViewer.title')}</span>
          <span style={{ fontSize: 11, color: colors.textSecondary }}>
            {t('databaseViewer.summary', { parts: Object.entries(counts).map(([k, v]) => `${v} ${k}`).join(' · '), total: entries.length })}
            {fitCount != null && <> · <span style={{ color: colors.green }}>{t('databaseViewer.fitForXtalCount', { count: fitCount })}</span></>}
          </span>
          <div style={{ flex: 1 }} />
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder={t('databaseViewer.searchPlaceholder')}
            title={t('hoverTips.dbSearch')}
            style={{ width: 260, fontSize: 11, height: 28, padding: '2px 10px' }}
          />
          {rich && (
            <Button
              onClick={handleSave}
              disabled={modifiedCount === 0 || saving}
              variant={modifiedCount > 0 ? 'primary' : undefined}
              style={{ fontSize: 11 }}
              title={modifiedCount === 0 ? t('databaseViewer.saveTooltipNone') : t('databaseViewer.saveTooltipCount', { count: modifiedCount })}
            >
              {saving ? t('databaseViewer.saving') : t('common:save')}
            </Button>
          )}
          <Button onClick={onClose} style={{ fontSize: 11 }} title={t('hoverTips.dbClose')}>{t('common:close')}</Button>
        </div>

        {/* Table */}
        <div className="thin-scrollbar" style={{ flex: 1, overflowY: 'auto', overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11, minWidth: rich ? 1100 : 600 }}>
            <thead>
              <tr style={{ position: 'sticky', top: 0, background: colors.bgSecondary, zIndex: 1 }}>
                {/* Row number column */}
                <th style={{
                  padding: '6px 8px',
                  textAlign: 'right',
                  color: colors.textSecondary,
                  fontWeight: 600,
                  borderBottom: `2px solid ${colors.border}`,
                  userSelect: 'none',
                  whiteSpace: 'nowrap',
                  width: 36,
                  minWidth: 36,
                }}>#</th>
                {columns.map((col, ci) => {
                  const thIdx = ci + 1; // offset by 1 for # column
                  return (
                    <th
                      key={col.key}
                      onClick={() => handleHeaderClick(thIdx)}
                      title={t('hoverTips.dbSortColumn')}
                      style={{
                        padding: '6px 10px',
                        textAlign: col.align || 'left',
                        color: sortCol === thIdx ? colors.cyan : colors.textSecondary,
                        fontWeight: 600,
                        borderBottom: `2px solid ${sortCol === thIdx ? colors.cyan : colors.border}`,
                        cursor: 'pointer',
                        userSelect: 'none',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {col.label}
                      {col.editable && <span style={{ marginLeft: 3, fontSize: 9, opacity: 0.5 }} title={t('databaseViewer.headerEditableTooltip')}>✎</span>}
                      {sortCol === thIdx && (
                        <span style={{ marginLeft: 4, fontSize: 9 }}>{sortAsc ? '▲' : '▼'}</span>
                      )}
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {sortedWithIdx.length === 0 ? (
                <tr>
                  <td colSpan={columns.length + 1} style={{ padding: 24, textAlign: 'center', color: colors.textSecondary }}>
                    {search ? t('databaseViewer.noMatchingEntries') : t('databaseViewer.noEntries')}
                  </td>
                </tr>
              ) : sortedWithIdx.map(({ entry, origIdx }, ri) => (
                <tr
                  key={entry.path || origIdx}
                  className="table-row-hover"
                  style={{
                    borderBottom: `1px solid ${alpha(colors.border, 13)}`,
                    cursor: entry.category === 'cif' ? 'pointer' : 'default',
                  }}
                  onDoubleClick={(e) => {
                    // Only trigger load-CIF on non-editable cells
                    if (e.target.tagName === 'INPUT') return;
                    const cifPath = rich ? entry['File Path'] : entry.path;
                    const isCif = rich || entry.category === 'cif';
                    if (isCif && cifPath && onLoadCif) onLoadCif(cifPath);
                  }}
                  title={rich ? t('databaseViewer.rowTooltipRich', { path: entry['File Path'] || '' }) : (entry.category === 'cif' ? t('databaseViewer.rowTooltipCif') : entry.path)}
                >
                  {/* Row number cell */}
                  <td style={{
                    padding: '4px 8px',
                    textAlign: 'right',
                    color: colors.textSecondary,
                    fontSize: 10,
                    userSelect: 'none',
                    whiteSpace: 'nowrap',
                    width: 36,
                    minWidth: 36,
                  }}>
                    {ri + 1}
                  </td>
                  {columns.map((col) => {
                    const editKey = `${origIdx}|${col.key}`;
                    const rawVal = entry[col.key];
                    const editedVal = edits[editKey];
                    const val = editedVal != null ? editedVal : rawVal;
                    const display = col.format ? col.format(val) : (val || '—');
                    const isEdited = editedVal != null;
                    const isCurrentlyEditing = editingCell === editKey;

                    let cellColor = colors.text;
                    if (col.key === 'name' || col.key === 'CIF File Name') cellColor = colors.cyan;
                    else if (col.key === 'Composition') cellColor = colors.green;
                    else if (col.key === 'Fit for .xtal') cellColor = val === 'Yes' ? colors.green : colors.red;
                    else if (col.key === 'Warnings' && val) cellColor = colors.yellow;
                    else if (col.key === 'Phase Name') cellColor = isEdited ? colors.yellow : colors.text;

                    if (col.editable && isCurrentlyEditing) {
                      return (
                        <td
                          key={col.key}
                          style={{
                            padding: '2px 6px',
                            background: alpha(colors.yellow, 8),
                          }}
                        >
                          <input
                            autoFocus
                            value={editingValue}
                            onChange={(e) => setEditingValue(e.target.value)}
                            onBlur={() => commitEdit(editKey)}
                            title={t('hoverTips.dbEditCell')}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') commitEdit(editKey);
                              else if (e.key === 'Escape') cancelEdit();
                            }}
                            style={{
                              width: '100%',
                              background: colors.bgSecondary,
                              border: `1px solid ${colors.yellow}`,
                              borderRadius: 3,
                              color: colors.yellow,
                              fontSize: 11,
                              padding: '2px 6px',
                              outline: 'none',
                              boxSizing: 'border-box',
                            }}
                          />
                        </td>
                      );
                    }

                    return (
                      <td
                        key={col.key}
                        style={{
                          padding: '4px 10px',
                          textAlign: col.align || 'left',
                          color: cellColor,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          maxWidth: (col.key === 'name' || col.key === 'Lattice Parameters (Å)') ? 350 : 200,
                          background: isEdited ? alpha(colors.yellow, 5) : 'transparent',
                          cursor: col.editable ? 'text' : 'inherit',
                        }}
                        title={col.editable ? t('databaseViewer.cellEditableTooltip', { value: String(rawVal || '') }) : String(rawVal || '')}
                        onDoubleClick={col.editable ? (e) => {
                          e.stopPropagation();
                          startEdit(editKey, val != null ? String(val) : '');
                        } : undefined}
                      >
                        {display}
                        {isEdited && <span style={{ marginLeft: 4, fontSize: 9, opacity: 0.7, color: colors.yellow }}>*</span>}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Footer */}
        <div style={{ padding: '6px 16px', borderTop: `1px solid ${colors.border}`, fontSize: 10, color: colors.textSecondary, flexShrink: 0 }}>
          {t('databaseViewer.footerEntries', { total: entries.length, shown: sortedWithIdx.length, filtered: search ? t('databaseViewer.footerFiltered') : '' })}
          {rich && <>{t('databaseViewer.footerModifiedLabel')}<span style={{ color: modifiedCount > 0 ? colors.yellow : colors.textSecondary }}>{modifiedCount}</span></>}
          {rich && <span style={{ marginLeft: 12, opacity: 0.6 }}>{t('databaseViewer.footerEditHint')}</span>}
        </div>
      </div>
    </div>
  );
}
