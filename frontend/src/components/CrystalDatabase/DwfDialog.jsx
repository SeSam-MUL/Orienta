import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { dbApi } from '../../services/api';
import { colors, alpha, Button, Input } from '../../theme/components';

/**
 * Debye-Waller factor (DWF) table editor.
 *
 * Shows the per-element B(300 K) values used during CIF → .xtal conversion.
 * Each row carries a literature Reference (provenance). Values and references
 * are editable inline, and NEW elements can be added (e.g. Cu, which was
 * missing and silently fell back to the generic 0.005). Save upserts via
 * PATCH /api/database/dwf.
 */
const cellInput = {
  width: '100%', background: colors.bgSecondary, border: `1px solid ${colors.border}`,
  borderRadius: 3, color: colors.text, fontSize: 12, padding: '3px 6px',
  outline: 'none', boxSizing: 'border-box',
};

export default function DwfDialog({ onClose }) {
  const { t } = useTranslation(['crystaldatabase', 'common']);
  const [rows, setRows] = useState([]);        // {element, dwb, reference, isNew}
  const [original, setOriginal] = useState({}); // element -> {dwb, reference}
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [search, setSearch] = useState('');
  const [msg, setMsg] = useState('');
  const [newEl, setNewEl] = useState('');
  const [newDwb, setNewDwb] = useState('');
  const [newRef, setNewRef] = useState('');

  useEffect(() => {
    dbApi.getDwf()
      .then(res => {
        const entries = res.data?.entries || [];
        const rs = entries.map(e => ({
          element: String(e['Element'] ?? e.element ?? ''),
          dwb: e['DWB 300 K'] ?? e.dwb_300k ?? '',
          reference: String(e['Reference'] ?? e.reference ?? ''),
          isNew: false,
        }));
        setRows(rs);
        const orig = {};
        rs.forEach(r => { orig[r.element] = { dwb: r.dwb, reference: r.reference }; });
        setOriginal(orig);
      })
      .catch(() => setMsg(t('dwf.loadFailed')))
      .finally(() => setLoading(false));
  }, [t]);

  const firstToken = (s) => String(s).trim().split(/\s+/)[0] || '';

  const isChanged = (r) => {
    if (!r.element.trim()) return false;
    const o = original[r.element];
    if (!o) return true; // newly added element
    return String(o.dwb) !== String(r.dwb) || String(o.reference) !== String(r.reference);
  };
  const changed = rows.filter(isChanged);
  const modifiedCount = changed.length;

  const setCell = (idx, field, value) =>
    setRows(prev => prev.map((r, i) => (i === idx ? { ...r, [field]: value } : r)));

  const addRow = () => {
    const el = newEl.trim();
    const v = parseFloat(newDwb);
    if (!el) { setMsg(t('dwf.enterElement')); return; }
    if (!isFinite(v) || v <= 0) { setMsg(t('dwf.enterValue')); return; }
    if (rows.some(r => firstToken(r.element) === el)) {
      setMsg(t('dwf.alreadyInTable', { element: el })); return;
    }
    setRows(prev => [...prev, { element: el, dwb: v, reference: newRef.trim(), isNew: true }]);
    setNewEl(''); setNewDwb(''); setNewRef(''); setMsg('');
  };

  const handleSave = async () => {
    if (modifiedCount === 0) return;
    setSaving(true);
    const updates = changed
      .map(r => ({ element: firstToken(r.element), dwb_300k: parseFloat(r.dwb), reference: r.reference || '' }))
      .filter(u => u.element && isFinite(u.dwb_300k));
    try {
      const res = await dbApi.updateDwf(updates);
      const orig = { ...original };
      rows.forEach(r => { if (r.element.trim()) orig[r.element] = { dwb: r.dwb, reference: r.reference }; });
      setOriginal(orig);
      setRows(prev => prev.map(r => ({ ...r, isNew: false })));
      const added = res.data?.added?.length || 0;
      const upd = res.data?.updated?.length || 0;
      setMsg(t('dwf.saved', { updated: upd, added }));
    } catch {
      setMsg(t('dwf.saveFailedMsg'));
    } finally {
      setSaving(false);
    }
  };

  const sl = search.toLowerCase();
  const visible = rows
    .map((r, i) => ({ r, i }))
    .filter(({ r }) => !sl || r.element.toLowerCase().includes(sl) || r.reference.toLowerCase().includes(sl));

  const th = {
    padding: '7px 12px', textAlign: 'left', color: colors.textSecondary,
    fontWeight: 600, borderBottom: `2px solid ${colors.border}`, whiteSpace: 'nowrap',
    position: 'sticky', top: 0, background: colors.bgSecondary, zIndex: 1,
  };

  return (
    <div
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', zIndex: 1100,
        display: 'flex', alignItems: 'center', justifyContent: 'center', backdropFilter: 'blur(2px)',
      }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div style={{
        background: colors.bg, border: `1px solid ${colors.border}`, borderRadius: 8,
        width: 660, maxHeight: '78vh', display: 'flex', flexDirection: 'column',
        boxShadow: '0 12px 40px rgba(0,0,0,0.4)', animation: 'fadeSlideIn 0.2s ease-out',
      }}>
        {/* Header */}
        <div style={{ padding: '12px 16px', borderBottom: `1px solid ${colors.border}`,
                      display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0 }}>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: colors.accent }}>
              {t('dwf.title')}
            </div>
            <div style={{ fontSize: 10, color: colors.textSecondary, marginTop: 2 }}>
              {t('dwf.subtitle')}
            </div>
          </div>
          <Input value={search} onChange={(e) => setSearch(e.target.value)} placeholder={t('dwf.searchPlaceholder')}
                 title={t('hoverTips.dwfSearch')}
                 style={{ width: 120, fontSize: 11, height: 26, padding: '2px 8px', flexShrink: 0 }} />
          <Button onClick={handleSave} disabled={modifiedCount === 0 || saving}
                  variant={modifiedCount > 0 ? 'primary' : undefined}
                  style={{ fontSize: 11, flexShrink: 0 }}
                  title={modifiedCount === 0 ? t('dwf.saveTooltipNone') : t('dwf.saveTooltipCount', { count: modifiedCount })}>
            {saving ? t('dwf.saving') : t('common:save')}
          </Button>
          <Button onClick={onClose} style={{ fontSize: 11, flexShrink: 0 }} title={t('hoverTips.dwfClose')}>{t('common:close')}</Button>
        </div>

        {/* Table */}
        <div className="thin-scrollbar" style={{ flex: 1, overflowY: 'auto' }}>
          {loading ? (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center',
                          height: 120, color: colors.textSecondary, fontSize: 12 }}>{t('dwf.loading')}</div>
          ) : (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr>
                  <th style={{ ...th, width: '22%' }}>{t('dwf.colElement')}</th>
                  <th style={{ ...th, width: '26%', textAlign: 'right' }}>{t('dwf.colDwb')}</th>
                  <th style={th}>{t('dwf.colReference')}</th>
                </tr>
              </thead>
              <tbody>
                {visible.length === 0 ? (
                  <tr><td colSpan={3} style={{ padding: 24, textAlign: 'center', color: colors.textSecondary }}>
                    {search ? t('dwf.noMatchingEntries') : t('dwf.noEntries')}
                  </td></tr>
                ) : visible.map(({ r, i }) => (
                  <tr key={i} className="table-row-hover"
                      style={{ borderBottom: `1px solid ${alpha(colors.border, 13)}`,
                               background: isChanged(r) ? alpha(colors.yellow, 5) : 'transparent' }}>
                    <td style={{ padding: '4px 12px', whiteSpace: 'nowrap' }}>
                      {r.isNew ? (
                        <input value={r.element} onChange={(e) => setCell(i, 'element', e.target.value)}
                               title={t('hoverTips.dwfElementCell')}
                               style={{ ...cellInput, color: colors.cyan }} placeholder={t('dwf.elementPlaceholder')} />
                      ) : (
                        <span style={{ color: colors.cyan, fontWeight: 500 }}>{r.element}</span>
                      )}
                    </td>
                    <td style={{ padding: '4px 8px', textAlign: 'right' }}>
                      <input value={r.dwb} onChange={(e) => setCell(i, 'dwb', e.target.value)}
                             title={t('dwf.valueTooltip')}
                             style={{ ...cellInput, textAlign: 'right' }} />
                    </td>
                    <td style={{ padding: '4px 12px' }}>
                      <input value={r.reference} onChange={(e) => setCell(i, 'reference', e.target.value)}
                             title={t('dwf.referenceTooltip')}
                             style={cellInput} placeholder={t('dwf.referencePlaceholder')} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Add row */}
        <div style={{ padding: '8px 12px', borderTop: `1px solid ${colors.border}`, flexShrink: 0,
                      display: 'flex', alignItems: 'center', gap: 8 }}>
          <input value={newEl} onChange={(e) => setNewEl(e.target.value)} placeholder={t('dwf.addElement')}
                 onKeyDown={(e) => { if (e.key === 'Enter') addRow(); }}
                 title={t('hoverTips.dwfNewElement')}
                 style={{ ...cellInput, width: 90, color: colors.cyan }} />
          <input value={newDwb} onChange={(e) => setNewDwb(e.target.value)} placeholder={t('dwf.addValue')}
                 onKeyDown={(e) => { if (e.key === 'Enter') addRow(); }}
                 title={t('hoverTips.dwfNewValue')}
                 style={{ ...cellInput, width: 90, textAlign: 'right' }} />
          <input value={newRef} onChange={(e) => setNewRef(e.target.value)} placeholder={t('dwf.addReference')}
                 onKeyDown={(e) => { if (e.key === 'Enter') addRow(); }}
                 title={t('hoverTips.dwfNewReference')}
                 style={{ ...cellInput, flex: 1 }} />
          <Button onClick={addRow} style={{ fontSize: 11, flexShrink: 0 }} title={t('dwf.addTooltip')}>{t('dwf.addButton')}</Button>
        </div>

        {/* Footer */}
        <div style={{ padding: '6px 16px', borderTop: `1px solid ${colors.border}`, fontSize: 10,
                      color: colors.textSecondary, flexShrink: 0, display: 'flex', alignItems: 'center', gap: 8 }}>
          <span>{t('dwf.entries', { count: rows.length })}</span>
          <span style={{ opacity: 0.4 }}>|</span>
          <span>{t('dwf.modifiedLabel')}<span style={{ color: modifiedCount > 0 ? colors.yellow : colors.textSecondary }}>{modifiedCount}</span></span>
          {msg && <span style={{ marginLeft: 'auto', color: colors.textSecondary }}>{msg}</span>}
        </div>
      </div>
    </div>
  );
}
