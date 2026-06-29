/**
 * CrystalPickerModal — searchable modal for selecting .xtal files
 * with dynamic MC/Master/SHT pipeline status at current kV.
 */
import { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { simApi } from '../../services/api';
import { colors as C, spacing } from '../../theme/tokens';
import { Button, GroupBox } from '../../theme/components';

const CHECK = '\u2713';
const DASH = '\u2014';

function StatusIcon({ ok }) {
  return (
    <span style={{
      color: ok ? C.green : C.textSecondary,
      fontWeight: ok ? 'bold' : 'normal',
      fontSize: '11pt',
    }}>
      {ok ? CHECK : DASH}
    </span>
  );
}

export default function CrystalPickerModal({ open, onClose, onSelect, currentKv }) {
  const { t } = useTranslation('simulation');
  const [crystals, setCrystals] = useState([]);
  const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState('');
  const [selectedIdx, setSelectedIdx] = useState(-1);

  // Load crystal list when modal opens or kV changes
  useEffect(() => {
    if (!open) return;
    setLoading(true);
    setSelectedIdx(-1);
    simApi.crystalPicker(Math.round(currentKv || 20))
      .then(res => setCrystals(res.data || []))
      .catch(() => setCrystals([]))
      .finally(() => setLoading(false));
  }, [open, currentKv]);

  // Filter by search
  const filtered = crystals.filter(c =>
    !search || c.stem.toLowerCase().includes(search.toLowerCase())
  );

  const handleSelect = useCallback(() => {
    if (selectedIdx >= 0 && selectedIdx < filtered.length) {
      onSelect(filtered[selectedIdx].path);
      onClose();
    }
  }, [selectedIdx, filtered, onSelect, onClose]);

  const handleRowClick = useCallback((idx) => {
    setSelectedIdx(idx);
  }, []);

  const handleRowDoubleClick = useCallback((idx) => {
    if (idx >= 0 && idx < filtered.length) {
      onSelect(filtered[idx].path);
      onClose();
    }
  }, [filtered, onSelect, onClose]);

  if (!open) return null;

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 1000,
      background: 'rgba(0,0,0,0.6)', display: 'flex',
      alignItems: 'center', justifyContent: 'center',
    }}>
      <div style={{
        background: C.bgSecondary, border: `1px solid ${C.border}`,
        borderRadius: 8, width: '80%', maxWidth: 720, maxHeight: '75vh',
        overflow: 'hidden', display: 'flex', flexDirection: 'column',
      }}>
        {/* Header */}
        <div style={{
          padding: '12px 16px', borderBottom: `1px solid ${C.border}`,
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <h2 style={{ margin: 0, color: C.purple, fontSize: '13pt' }}>{t('crystalPicker.title')}</h2>
          <button onClick={onClose} title={t('hoverTips.pickerClose')} aria-label={t('hoverTips.pickerClose')} style={{
            background: 'none', border: 'none', color: C.text,
            fontSize: '16pt', cursor: 'pointer',
          }}>x</button>
        </div>

        {/* Search + kV info */}
        <div style={{
          padding: '10px 16px', display: 'flex', gap: 12, alignItems: 'center',
          borderBottom: `1px solid ${C.border}`,
        }}>
          <input
            type="text"
            value={search}
            onChange={e => { setSearch(e.target.value); setSelectedIdx(-1); }}
            placeholder={t('crystalPicker.searchPlaceholder')}
            title={t('hoverTips.pickerSearch')}
            autoFocus
            style={{
              flex: 1, background: C.bg, color: C.text,
              border: `1px solid ${C.border}`, borderRadius: 4,
              padding: '6px 10px', fontSize: '10pt', outline: 'none',
            }}
          />
          <span style={{ color: C.accent, fontSize: '10pt', whiteSpace: 'nowrap' }}>
            {t('crystalPicker.matchingLabel')} <b>{t('crystalPicker.matchingKv', { kv: Math.round(currentKv || 20) })}</b>
          </span>
        </div>

        {/* Table */}
        <div style={{ flex: 1, overflow: 'auto', padding: '0 16px' }}>
          {loading ? (
            <div style={{ padding: 24, textAlign: 'center', color: C.textSecondary }}>{t('crystalPicker.loading')}</div>
          ) : filtered.length === 0 ? (
            <div style={{ padding: 24, textAlign: 'center', color: C.textSecondary }}>
              {search ? t('crystalPicker.noMatch') : t('crystalPicker.noFiles')}
            </div>
          ) : (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '10pt' }}>
              <thead>
                <tr style={{ color: C.textSecondary, borderBottom: `1px solid ${C.border}` }}>
                  <th style={{ textAlign: 'left', padding: '8px 10px', fontWeight: 600 }}>{t('crystalPicker.colCrystal')}</th>
                  <th style={{ textAlign: 'center', padding: '8px 6px', fontWeight: 600, width: 50 }}>{t('crystalPicker.colMc')}</th>
                  <th style={{ textAlign: 'center', padding: '8px 6px', fontWeight: 600, width: 60 }}>{t('crystalPicker.colMaster')}</th>
                  <th style={{ textAlign: 'center', padding: '8px 6px', fontWeight: 600, width: 50 }}>{t('crystalPicker.colSht')}</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((c, i) => {
                  const isSelected = i === selectedIdx;
                  const hasNothing = !c.has_mc && !c.has_master && !c.has_sht;
                  return (
                    <tr
                      key={c.name}
                      onClick={() => handleRowClick(i)}
                      onDoubleClick={() => handleRowDoubleClick(i)}
                      title={t('hoverTips.pickerRow')}
                      style={{
                        cursor: 'pointer',
                        background: isSelected ? C.purple + '22' : 'transparent',
                        borderBottom: `1px solid ${C.border}22`,
                        opacity: hasNothing ? 0.5 : 1,
                        transition: 'background 0.1s',
                      }}
                    >
                      <td style={{
                        padding: '7px 10px',
                        color: isSelected ? C.purple : C.text,
                        fontWeight: isSelected ? 'bold' : 'normal',
                      }}>
                        {c.stem}
                      </td>
                      <td style={{ textAlign: 'center', padding: '7px 6px' }}>
                        <StatusIcon ok={c.has_mc} />
                      </td>
                      <td style={{ textAlign: 'center', padding: '7px 6px' }}>
                        <StatusIcon ok={c.has_master} />
                      </td>
                      <td style={{ textAlign: 'center', padding: '7px 6px' }}>
                        <StatusIcon ok={c.has_sht} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>

        {/* Selection info + actions */}
        <div style={{
          padding: '10px 16px', borderTop: `1px solid ${C.border}`,
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <div style={{ fontSize: '9pt', color: C.textSecondary, flex: 1, overflow: 'hidden' }}>
            {selectedIdx >= 0 && selectedIdx < filtered.length ? (
              <>
                <span style={{ color: C.accent }}>{filtered[selectedIdx].name}</span>
                <span style={{ marginLeft: 8, opacity: 0.6 }} title={filtered[selectedIdx].path}>
                  {filtered[selectedIdx].path}
                </span>
              </>
            ) : (
              <span>{t('crystalPicker.selectionHint')}</span>
            )}
          </div>
          <div style={{ display: 'flex', gap: 8, flexShrink: 0 }}>
            <Button onClick={onClose} title={t('hoverTips.pickerCancel')}>{t('crystalPicker.cancel')}</Button>
            <Button
              variant="primary"
              disabled={selectedIdx < 0}
              onClick={handleSelect}
              title={t('hoverTips.pickerSelect')}
            >
              {t('crystalPicker.select')}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
