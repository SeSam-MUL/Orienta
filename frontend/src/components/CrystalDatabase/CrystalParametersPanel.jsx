import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, alpha, GroupBox, LoadingOverlay, Button, Input } from '../../theme/components';
import { dbApi } from '../../services/api';

// ---------------------------------------------------------------------------
// Shared sub-components
// ---------------------------------------------------------------------------

function InfoRow({ label, value, valueColor }) {
  return (
    <div className="table-row-hover" style={{
      display: 'flex', justifyContent: 'space-between',
      padding: '3px 4px', borderBottom: `1px solid ${alpha(colors.border, 13)}`, borderRadius: 2
    }}>
      <span style={{ fontSize: 12, color: colors.textSecondary }}>{label}</span>
      <span style={{ fontSize: 12, color: valueColor || colors.text, fontWeight: 500, maxWidth: '60%', textAlign: 'right', wordBreak: 'break-all' }}>
        {value || '—'}
      </span>
    </div>
  );
}

const LATTICE_PARAMS = [
  { key: 'a', label: 'a', unit: 'Å' },
  { key: 'b', label: 'b', unit: 'Å' },
  { key: 'c', label: 'c', unit: 'Å' },
  { key: 'alpha', label: 'α', unit: '°' },
  { key: 'beta',  label: 'β', unit: '°' },
  { key: 'gamma', label: 'γ', unit: '°' },
];

function LatticeCard({ label, value, unit }) {
  return (
    <div
      style={{
        background: colors.bg, border: `1px solid ${colors.border}`,
        borderRadius: 4, padding: '6px 8px', textAlign: 'center',
        transition: 'border-color 0.15s',
      }}
      onMouseEnter={e => { e.currentTarget.style.borderColor = colors.cyan; }}
      onMouseLeave={e => { e.currentTarget.style.borderColor = colors.border; }}
    >
      <div style={{ fontSize: 10, color: colors.textSecondary }}>{label}</div>
      <div style={{ fontSize: 13, fontWeight: 600, color: colors.cyan }}>
        {value != null ? `${Number(value).toFixed(4)} ${unit}` : '—'}
      </div>
    </div>
  );
}

function AtomTable({ atoms, dwfPrecision = 4 }) {
  const { t } = useTranslation('crystaldatabase');
  if (!atoms || atoms.length === 0) return null;
  const atomCols = [
    { key: 'element', label: t('parameters.cols.element') },
    { key: 'x', label: t('parameters.cols.x') },
    { key: 'y', label: t('parameters.cols.y') },
    { key: 'z', label: t('parameters.cols.z') },
    { key: 'occ', label: t('parameters.cols.occ') },
    { key: 'dwf', label: t('parameters.cols.dwf') },
  ];
  return (
    <div>
      <div style={{ fontSize: 11, color: colors.textSecondary, marginBottom: 4, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
        {t('parameters.asymmetricUnit')}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr>
              {atomCols.map((col, i) => (
                <th key={col.key} style={{
                  padding: '3px 6px', color: colors.textSecondary, fontWeight: 600,
                  textAlign: i === 0 ? 'left' : 'right',
                  borderBottom: `1px solid ${colors.border}`, whiteSpace: 'nowrap',
                }}>{col.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {atoms.map((atom, idx) => (
              <tr key={idx} className="table-row-hover">
                <td style={{ padding: '3px 6px', color: colors.cyan, fontWeight: 600, textAlign: 'left' }}>
                  {atom.element || atom.symbol || '?'}
                </td>
                <td style={{ padding: '3px 6px', color: colors.text, textAlign: 'right' }}>
                  {atom.x != null ? Number(atom.x).toFixed(4) : '—'}
                </td>
                <td style={{ padding: '3px 6px', color: colors.text, textAlign: 'right' }}>
                  {atom.y != null ? Number(atom.y).toFixed(4) : '—'}
                </td>
                <td style={{ padding: '3px 6px', color: colors.text, textAlign: 'right' }}>
                  {atom.z != null ? Number(atom.z).toFixed(4) : '—'}
                </td>
                <td style={{ padding: '3px 6px', color: colors.text, textAlign: 'right' }}>
                  {(atom.occ ?? atom.occupancy) != null ? Number(atom.occ ?? atom.occupancy).toFixed(3) : '—'}
                </td>
                <td style={{ padding: '3px 6px', color: colors.text, textAlign: 'right' }}>
                  {atom.dwf != null ? Number(atom.dwf).toFixed(dwfPrecision) : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab switcher (document-style tabs)
// ---------------------------------------------------------------------------

function DocTabs({ tabs, activeTab, onTabChange }) {
  const { t } = useTranslation('crystaldatabase');
  return (
    <div style={{ display: 'flex', gap: 0, marginBottom: 12 }}>
      {tabs.map(({ id, label, disabled }) => {
        const isActive = activeTab === id;
        return (
          <button
            key={id}
            onClick={() => !disabled && onTabChange(id)}
            disabled={disabled}
            title={t('parameters.tabTooltip')}
            style={{
              padding: '6px 16px',
              fontSize: 12, fontWeight: 600,
              color: disabled ? colors.border : isActive ? colors.accent : colors.textSecondary,
              background: isActive ? alpha(colors.accent, 10) : 'transparent',
              border: `1px solid ${isActive ? alpha(colors.accent, 30) : colors.border}`,
              borderBottom: isActive ? `2px solid ${colors.accent}` : `1px solid ${colors.border}`,
              borderRadius: '6px 6px 0 0',
              cursor: disabled ? 'default' : 'pointer',
              transition: 'all 0.15s',
              opacity: disabled ? 0.4 : 1,
            }}
          >
            {label}
          </button>
        );
      })}
      <div style={{ flex: 1, borderBottom: `1px solid ${colors.border}` }} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// CIF tab content
// ---------------------------------------------------------------------------

function FitBadge({ fit, warnings }) {
  const { t } = useTranslation('crystaldatabase');
  if (fit == null) return null;
  const color = fit ? colors.green : colors.red;
  const label = fit ? t('parameters.fitForXtal') : t('parameters.notFitForXtal');
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{
        display: 'inline-flex', alignItems: 'center', gap: 6,
        padding: '3px 10px', borderRadius: 6, fontSize: 11, fontWeight: 600,
        color, background: alpha(color, 8), border: `1px solid ${alpha(color, 30)}`,
      }}>
        {fit ? '\u2713' : '\u2717'} {label}
      </div>
      {warnings && warnings.length > 0 && (
        <div style={{
          fontSize: 11, color: colors.yellow, background: alpha(colors.yellow, 8),
          border: `1px solid ${alpha(colors.yellow, 30)}`,
          borderRadius: 4, padding: '4px 8px', marginTop: 6,
        }}>
          {warnings.map((w, i) => <div key={i}>{w}</div>)}
        </div>
      )}
    </div>
  );
}

function EditableRow({ label, value, placeholder, onSave }) {
  const { t } = useTranslation(['crystaldatabase', 'common']);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value || '');
  useEffect(() => { setDraft(value || ''); }, [value]);

  if (editing) {
    return (
      <div style={{ padding: '3px 4px', borderBottom: `1px solid ${alpha(colors.border, 13)}` }}>
        <div style={{ fontSize: 10, color: colors.textSecondary, marginBottom: 2 }}>{label}</div>
        <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
          <Input value={draft} onChange={(e) => setDraft(e.target.value)} placeholder={placeholder}
            style={{ flex: 1, fontSize: 11, height: 24, padding: '2px 6px' }} autoFocus
            title={t('hoverTips.editFieldInput')}
            onKeyDown={(e) => {
              if (e.key === 'Enter') { onSave(draft); setEditing(false); }
              if (e.key === 'Escape') { setDraft(value || ''); setEditing(false); }
            }} />
          <Button small style={{ fontSize: 10, padding: '2px 8px' }}
            title={t('hoverTips.editFieldSave')}
            onClick={() => { onSave(draft); setEditing(false); }}>{t('common:save')}</Button>
          <Button small variant="ghost" style={{ fontSize: 10, padding: '2px 6px' }}
            title={t('hoverTips.editFieldCancel')}
            onClick={() => { setDraft(value || ''); setEditing(false); }}>{t('common:cancel')}</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="table-row-hover" style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '3px 4px', borderBottom: `1px solid ${alpha(colors.border, 13)}`,
      borderRadius: 2, cursor: 'pointer',
    }} onClick={() => setEditing(true)} title={t('parameters.clickToEdit')}>
      <span style={{ fontSize: 12, color: colors.textSecondary, display: 'flex', alignItems: 'center', gap: 4 }}>
        {label}<span style={{ fontSize: 9, opacity: 0.5 }}>{'\u270E'}</span>
      </span>
      <span style={{
        fontSize: 12, color: value ? colors.text : colors.textSecondary,
        fontWeight: 500, maxWidth: '60%', textAlign: 'right',
        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
      }}>{value || t('parameters.clickToAdd')}</span>
    </div>
  );
}

function CifTabContent({ crystalInfo, selectedFileName, onInfoUpdated }) {
  const { t } = useTranslation('crystaldatabase');
  const [saving, setSaving] = useState(false);
  const atoms = crystalInfo.atoms || crystalInfo.asymmetric_unit || [];

  const handleSaveMeta = async (field, value) => {
    if (!selectedFileName) return;
    setSaving(true);
    try {
      // Only send the field the user just edited — don't pass an empty
      // string for the other one. The backend already skips empty-string
      // fields, but this keeps the intent explicit so any future change
      // to that rule doesn't accidentally wipe the untouched field.
      const payload = field === 'doi' ? { doi: value } : { reference: value };
      await dbApi.updateCifMeta(selectedFileName, payload);
      if (onInfoUpdated) onInfoUpdated();
    } catch (err) {
      console.error('Failed to save CIF metadata:', err);
    } finally { setSaving(false); }
  };

  return (
    <>
      <FitBadge fit={crystalInfo.fit_for_xtal} warnings={crystalInfo.fit_warnings} />

      {crystalInfo.parse_warning && (
        <div style={{
          fontSize: 11, color: colors.yellow, background: alpha(colors.yellow, 8),
          border: `1px solid ${alpha(colors.yellow, 30)}`,
          borderRadius: 4, padding: '4px 8px', marginBottom: 10,
        }}>{crystalInfo.parse_warning}</div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6, marginBottom: 14 }}>
        {LATTICE_PARAMS.map(({ key, label, unit }) => (
          <LatticeCard key={key} label={label} value={crystalInfo.lattice?.[key] ?? crystalInfo[key]} unit={unit} />
        ))}
      </div>

      <div style={{ marginBottom: 14 }}>
        <InfoRow label={t('parameters.cifFile')} value={selectedFileName} />
        <InfoRow label={t('parameters.asymmetricUnitSites')} value={atoms.length > 0 ? String(atoms.length) : null} valueColor={colors.green} />
        <EditableRow label={t('parameters.doi')} value={crystalInfo.doi} placeholder={t('parameters.doiPlaceholder')}
          onSave={(v) => handleSaveMeta('doi', v)} />
        <EditableRow label={t('parameters.reference')} value={crystalInfo.reference} placeholder={t('parameters.referencePlaceholder')}
          onSave={(v) => handleSaveMeta('reference', v)} />
      </div>

      {saving && <div style={{ fontSize: 10, color: colors.cyan, marginBottom: 6 }}>{t('parameters.saving')}</div>}

      <AtomTable atoms={atoms} dwfPrecision={4} />
    </>
  );
}

// ---------------------------------------------------------------------------
// XTAL tab content
// ---------------------------------------------------------------------------

function XtalTabContent({ xtalInfo, loading }) {
  const { t } = useTranslation('crystaldatabase');
  if (loading) return <LoadingOverlay message={t('parameters.xtal.reading')} />;

  if (!xtalInfo) return (
    <div style={{ textAlign: 'center', color: colors.textSecondary, padding: 24, fontSize: 12 }}>
      {t('parameters.xtal.notFound')}
    </div>
  );

  if (xtalInfo.error) return (
    <div style={{ color: colors.red, fontSize: 12, padding: 8 }}>{xtalInfo.error}</div>
  );

  const latticeNm = xtalInfo.lattice;
  const latticeA = xtalInfo.lattice_angstrom;

  return (
    <>
      {/* Header info */}
      <div style={{
        padding: '8px 12px', background: alpha(colors.green, 8),
        borderRadius: 6, borderLeft: `3px solid ${colors.green}`, marginBottom: 12,
      }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: colors.text, marginBottom: 2 }}>
          {xtalInfo.filename}
        </div>
        <div style={{ fontSize: 11, color: colors.textSecondary }}>
          {xtalInfo.crystal_system_name || '—'} · SG #{xtalInfo.SpaceGroupNumber || '?'}
          {xtalInfo.SpaceGroupSetting === 2 ? t('parameters.xtal.originSetting2') : ''}
        </div>
      </div>

      {/* Lattice — show nm (EMsoft native) and Å */}
      {latticeNm && (
        <div style={{ marginBottom: 14 }}>
          <div style={{ fontSize: 11, color: colors.textSecondary, marginBottom: 4, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            {t('parameters.xtal.latticeParameters')}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6 }}>
            {['a', 'b', 'c'].map(k => (
              <LatticeCard key={k} label={k} value={latticeA?.[k]} unit="Å" />
            ))}
            {['alpha', 'beta', 'gamma'].map(k => (
              <LatticeCard key={k} label={k === 'alpha' ? 'α' : k === 'beta' ? 'β' : 'γ'} value={latticeNm[k]} unit="°" />
            ))}
          </div>
          <div style={{ fontSize: 10, color: colors.textSecondary, marginTop: 4, opacity: 0.6 }}>
            {t('parameters.xtal.storedAsNm', { a: latticeNm.a, b: latticeNm.b, c: latticeNm.c })}
          </div>
        </div>
      )}

      {/* Metadata */}
      <div style={{ marginBottom: 14 }}>
        <InfoRow label={t('parameters.xtal.creator')} value={xtalInfo.Creator} />
        <InfoRow label={t('parameters.xtal.program')} value={xtalInfo.ProgramName} />
        <InfoRow label={t('parameters.xtal.created')} value={
          xtalInfo.CreationDate ? `${xtalInfo.CreationDate} ${xtalInfo.CreationTime || ''}` : null
        } />
        <InfoRow label={t('parameters.xtal.reference')} value={xtalInfo.useReference} />
        <InfoRow label={t('parameters.xtal.file')} value={xtalInfo.filename} />
        <InfoRow label={t('parameters.xtal.size')} value={xtalInfo.size ? t('parameters.xtal.kb', { value: (xtalInfo.size / 1024).toFixed(1) }) : null} />
      </div>

      {/* Atom table with higher DWF precision for xtal */}
      <AtomTable atoms={xtalInfo.atoms} dwfPrecision={6} />
    </>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function CrystalParametersPanel({ crystalInfo, loading, selectedFileName, hasXtal, onInfoUpdated }) {
  const { t } = useTranslation('crystaldatabase');
  const [tab, setTab] = useState('cif');
  const [xtalInfo, setXtalInfo] = useState(null);
  const [xtalLoading, setXtalLoading] = useState(false);

  // Reset tab when file changes
  useEffect(() => { setTab('cif'); setXtalInfo(null); }, [selectedFileName]);

  // Load xtal info when switching to xtal tab
  useEffect(() => {
    if (tab !== 'xtal' || !selectedFileName || !hasXtal) return;
    if (xtalInfo?.filename === selectedFileName.replace(/\.cif$/i, '.xtal')) return;
    let cancelled = false;
    setXtalLoading(true);
    const stem = selectedFileName.replace(/\.cif$/i, '');
    dbApi.xtalInfo(stem)
      .then(res => { if (!cancelled) setXtalInfo(res.data); })
      .catch(() => { if (!cancelled) setXtalInfo(null); })
      .finally(() => { if (!cancelled) setXtalLoading(false); });
    return () => { cancelled = true; };
  }, [tab, selectedFileName, hasXtal]); // eslint-disable-line react-hooks/exhaustive-deps

  if (loading) {
    return (
      <GroupBox title={t('parameters.title')}>
        <LoadingOverlay message={t('parameters.parsing')} />
      </GroupBox>
    );
  }

  if (!crystalInfo) {
    return (
      <GroupBox title={t('parameters.title')} style={{ flex: 1, overflow: 'auto' }}>
        <div style={{ textAlign: 'center', color: colors.textSecondary, padding: 24 }}>
          <div style={{ fontSize: '24pt', opacity: 0.15, marginBottom: 6 }}>{'\u2B21'}</div>
          {t('parameters.selectPrompt')}
        </div>
      </GroupBox>
    );
  }

  const tabs = [
    { id: 'cif', label: t('parameters.tabCif') },
    { id: 'xtal', label: t('parameters.tabXtal'), disabled: !hasXtal },
  ];

  return (
    <GroupBox title={t('parameters.title')} style={{ flex: 1, overflow: 'auto' }}>
      {/* Phase header */}
      <div style={{
        padding: '10px 12px', background: alpha(colors.accent, 8),
        borderRadius: 6, borderLeft: `3px solid ${colors.accent}`, marginBottom: 12,
      }}>
        <div style={{ fontSize: 15, fontWeight: 700, color: colors.text, marginBottom: 2 }}>
          {crystalInfo.phase_name || crystalInfo.name || '—'}
        </div>
        <div style={{ fontSize: 12, color: colors.textSecondary }}>
          {crystalInfo.crystal_system || '—'} · {crystalInfo.space_group || '—'} (#{crystalInfo.space_group_number || '?'})
        </div>
      </div>

      {/* Document tabs */}
      <DocTabs tabs={tabs} activeTab={tab} onTabChange={setTab} />

      {/* Tab content */}
      {tab === 'cif' && (
        <CifTabContent crystalInfo={crystalInfo} selectedFileName={selectedFileName} onInfoUpdated={onInfoUpdated} />
      )}
      {tab === 'xtal' && (
        <XtalTabContent xtalInfo={xtalInfo} loading={xtalLoading} />
      )}
    </GroupBox>
  );
}
