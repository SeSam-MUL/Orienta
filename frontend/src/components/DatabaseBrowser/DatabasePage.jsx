/**
 * Database Browser — React port of gui/database_browser_gui.py (DatabaseBrowserPage).
 *
 * Layout: 5 tabs (SHT / MC h5 / Master H5 / CIF / XTAL) with search, material filter,
 *         table of files, location/size/action columns.  Preview panel on the right.
 * Bottom bar: cache stats, Clear Local Cache button.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { dbApi, h5Api } from '../../services/api';
import CascadeDeleteDialog from './CascadeDeleteDialog';
import MasterSphereViewer from './MasterSphereViewer';
import CrystalStructureViewer from './CrystalStructureViewer';
import SyncDialog from '../CrystalDatabase/SyncDialog';
import SyncUploadDialog from './SyncUploadDialog';
import {
  colors, alpha, spacing,
  Button, Input, Select, Tabs, TabPanel, GroupBox,
  ResizableSplitter, FormRow, Label, Separator, Card, StatusDot,
  useConfirm, ConfirmDialog,
} from '../../theme/components';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
// Tabs map 1:1 onto the backend `/browse` categories. The `.h5` files are
// already split by the backend's _classify_h5 into `master` (filename contains
// `_master`) and `h5` (the residual = Monte-Carlo). Each tab matches exactly one
// category, so the per-tab "Type" sub-filter is no longer needed.
export const TAB_DEFS = [
  {
    id: 'sht',
    label: 'SHT',
    tip: 'Spherical Harmonic Transform master files for spherical indexing',
    fileTypes: new Set(['sht']),
    typeFilterItems: null,
  },
  {
    id: 'mc',
    label: 'MC h5',
    tip: 'Monte-Carlo HDF5 files (electron scattering, …_E20kV_sig70_n501_o0.h5)',
    fileTypes: new Set(['h5']),
    typeFilterItems: null,
  },
  {
    id: 'master',
    label: 'Master H5',
    tip: 'Master-pattern HDF5 files for dictionary indexing (…_master_…_npx….h5)',
    fileTypes: new Set(['master']),
    typeFilterItems: null,
  },
  {
    id: 'cif',
    label: 'CIF',
    tip: 'Crystallographic Information Files — crystal structure definitions',
    fileTypes: new Set(['cif']),
    typeFilterItems: null,
  },
  {
    id: 'xtal',
    label: 'XTAL',
    tip: 'EMsoft crystal structure files — converted from CIF for simulation',
    fileTypes: new Set(['xtal']),
    typeFilterItems: null,
  },
  {
    id: 'dict',
    label: 'Dictionary',
    tip: 'Dictionary index .h5 files (derived from masters) — excluded from bulk upload by default',
    fileTypes: new Set(['dictionary']),
    typeFilterItems: null,
  },
];

/** True if `entry` belongs in `tabDef` (matches on file_type → type → category). */
export function entryMatchesTab(entry, tabDef) {
  const ft = (entry.file_type || entry.type || entry.category || '').toLowerCase();
  return tabDef.fileTypes.has(ft);
}

const LOCATION_META = {
  local:       { labelKey: 'location.local',       color: colors.green },
  server:      { labelKey: 'location.server',      color: colors.purple },
  both:        { labelKey: 'location.both',        color: colors.cyan },
  downloading: { labelKey: 'location.downloading', color: colors.yellow },
};

/** Resolve location meta. `t` is the databasebrowser translator. */
function getLocationMeta(loc, t) {
  const key = (loc || '').toLowerCase();
  const meta = LOCATION_META[key];
  if (meta) return { label: t(`databasebrowser:${meta.labelKey}`), color: meta.color };
  return { label: loc || '—', color: colors.textSecondary };
}

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

// ---------------------------------------------------------------------------
// File table (one per tab)
// ---------------------------------------------------------------------------
function FileTable({ tabDef, entries, onRowClick, loading, selectedFiles, onToggleSelect, onSelectAll, onDownload }) {
  const { t } = useTranslation('databasebrowser');
  const [search, setSearch] = useState('');
  const [typeFilter, setTypeFilter] = useState('All');
  const [materialFilter, setMaterialFilter] = useState('All');
  const [selectedRow, setSelectedRow] = useState(-1);

  // Derive available materials from entries
  const materials = ['All', ...Array.from(
    new Set(entries.map((e) => e.material || '').filter(Boolean))
  ).sort()];

  // Filter entries
  const filtered = entries.filter((e) => {
    const ft = (e.file_type || e.type || e.category || '').toLowerCase();
    if (!entryMatchesTab(e, tabDef)) return false;

    if (typeFilter && typeFilter !== 'All') {
      if (ft !== typeFilter.toLowerCase()) return false;
    }
    if (materialFilter && materialFilter !== 'All') {
      if ((e.material || '') !== materialFilter) return false;
    }
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      if (!(e.filename || e.name || '').toLowerCase().includes(q)) return false;
    }
    return true;
  });

  const handleRowClick = (idx) => {
    setSelectedRow(idx);
    if (onRowClick) onRowClick(filtered[idx]);
  };

  const headerStyle = {
    padding: '6px 8px',
    textAlign: 'left',
    fontSize: '9pt',
    fontWeight: 600,
    color: colors.textSecondary,
    borderBottom: `1px solid ${colors.border}`,
    background: colors.bgTertiary,
    userSelect: 'none',
    whiteSpace: 'nowrap',
  };

  const cellStyle = (isSelected, isFirst) => ({
    padding: '5px 8px',
    fontSize: '9pt',
    borderBottom: `1px solid ${alpha(colors.border, 13)}`,
    background: isSelected ? colors.sidebarActive : 'transparent',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    maxWidth: 200,
    ...(isFirst ? { borderLeft: isSelected ? `3px solid ${colors.purple}` : '3px solid transparent' } : {}),
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: spacing.outerSpacing }}>

      {/* Filter bar */}
      <div style={{ display: 'flex', gap: spacing.outerSpacing, alignItems: 'center', flexWrap: 'wrap', padding: `${spacing.outerMargin}px ${spacing.outerMargin}px 0` }}>
        <Input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder={t('databasebrowser:filter.searchPlaceholder')}
          title={t('databasebrowser:filter.searchTooltip')}
          style={{ width: 220 }}
        />

        {tabDef.typeFilterItems && (
          <>
            <Label secondary small>{t('databasebrowser:filter.typeLabel')}</Label>
            <Select
              value={typeFilter}
              onChange={(e) => setTypeFilter(e.target.value)}
              options={tabDef.typeFilterItems}
              title={t('databasebrowser:filter.typeTooltip')}
              style={{ width: 110 }}
            />
          </>
        )}

        <Label secondary small>{t('databasebrowser:filter.materialLabel')}</Label>
        <Select
          value={materialFilter}
          onChange={(e) => setMaterialFilter(e.target.value)}
          options={materials}
          title={t('databasebrowser:filter.materialTooltip')}
          style={{ width: 120 }}
        />
      </div>

      {/* Table */}
      <div style={{ flex: 1, overflow: 'auto', paddingBottom: spacing.outerMargin }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', tableLayout: 'auto' }}>
          <thead>
            <tr>
              <th style={{ ...headerStyle, width: 32, textAlign: 'center' }}>
                <input
                  type="checkbox"
                  checked={filtered.length > 0 && filtered.every(e => selectedFiles?.has(e.name || e.filename))}
                  onChange={() => onSelectAll?.(filtered)}
                  title={t('databasebrowser:table.selectAllTooltip')}
                />
              </th>
              <th style={headerStyle} title={t('databasebrowser:table.colLocationTooltip')}>{t('databasebrowser:table.colLocation')}</th>
              <th style={{ ...headerStyle, width: '40%' }} title={t('databasebrowser:table.colFilenameTooltip')}>{t('databasebrowser:table.colFilename')}</th>
              <th style={headerStyle} title={t('databasebrowser:table.colMaterialTooltip')}>{t('databasebrowser:table.colMaterial')}</th>
              <th style={headerStyle} title={t('databasebrowser:table.colTypeTooltip')}>{t('databasebrowser:table.colType')}</th>
              <th style={{ ...headerStyle, textAlign: 'right' }} title={t('databasebrowser:table.colSizeTooltip')}>{t('databasebrowser:table.colSize')}</th>
              <th style={headerStyle}>{t('databasebrowser:table.colAction')}</th>
            </tr>
          </thead>
          <tbody>
            {loading && filtered.length === 0 ? (
              Array.from({ length: 4 }).map((_, i) => (
                <tr key={`skel-${i}`}>
                  {Array.from({ length: 7 }).map((__, j) => (
                    <td key={j} style={{ padding: '8px' }}>
                      <div style={{
                        height: 12, borderRadius: 3,
                        background: colors.border, opacity: 0.4,
                        width: j === 1 ? '80%' : j === 4 ? '40%' : '60%',
                        animation: 'pulse 1.5s ease-in-out infinite',
                      }} />
                    </td>
                  ))}
                </tr>
              ))
            ) : filtered.length === 0 ? (
              <tr>
                <td colSpan={7} style={{ padding: '32px', textAlign: 'center', color: colors.textSecondary, fontSize: '10pt' }}>
                  <div style={{ fontSize: '20pt', opacity: 0.3, marginBottom: 6 }}>{'\u2637'}</div>
                  <div>{t('databasebrowser:table.emptyTitle')}</div>
                  <div style={{ fontSize: '8pt', marginTop: 4, opacity: 0.6 }}>
                    {t('databasebrowser:table.emptyHint')}
                  </div>
                </td>
              </tr>
            ) : filtered.map((entry, i) => {
              const isSelected = i === selectedRow;
              const locMeta = getLocationMeta(entry.location, t);
              const ft = (entry.file_type || entry.type || entry.category || '').toUpperCase();
              const isDownloadable = (entry.location || '').toLowerCase() === 'server';
              const isOpenable = ['local', 'both'].includes((entry.location || '').toLowerCase());

              return (
                <tr
                  key={entry.filename || entry.name || i}
                  onClick={() => handleRowClick(i)}
                  className="list-item-interactive"
                  title={t('databasebrowser:table.rowClickTooltip')}
                  style={{ cursor: 'pointer', transition: 'background 0.1s' }}
                  onMouseEnter={(e) => !isSelected && (e.currentTarget.style.background = colors.bgTertiary)}
                  onMouseLeave={(e) => !isSelected && (e.currentTarget.style.background = 'transparent')}
                >
                  <td style={{ ...cellStyle(isSelected), width: 32, textAlign: 'center' }}
                      onClick={(e) => e.stopPropagation()}>
                    <input
                      type="checkbox"
                      checked={selectedFiles?.has(entry.name || entry.filename) || false}
                      onChange={() => onToggleSelect?.(entry.name || entry.filename)}
                      title={t('databasebrowser:table.rowSelectTooltip')}
                    />
                  </td>
                  <td style={cellStyle(isSelected, true)}>
                    <span style={{
                      color: locMeta.color, fontWeight: 600, fontSize: '8pt',
                      padding: '1px 6px', borderRadius: 8,
                      background: alpha(locMeta.color, 10),
                      border: `1px solid ${alpha(locMeta.color, 20)}`,
                    }}>
                      {locMeta.label}
                    </span>
                  </td>
                  <td style={{ ...cellStyle(isSelected), maxWidth: 300 }} title={entry.filename || entry.name}>
                    {entry.filename || entry.name || '—'}
                  </td>
                  <td style={cellStyle(isSelected)}>
                    {entry.material || '—'}
                  </td>
                  <td style={cellStyle(isSelected)}>
                    <span style={{ color: colors.textSecondary }}>{ft}</span>
                  </td>
                  <td style={{ ...cellStyle(isSelected), textAlign: 'right' }}>
                    {formatBytes(entry.size_bytes ?? entry.size)}
                  </td>
                  <td style={cellStyle(isSelected)}>
                    {isDownloadable && (
                      <span
                        role="button"
                        tabIndex={0}
                        onClick={(e) => { e.stopPropagation(); onDownload?.(entry); }}
                        title={t('databasebrowser:table.downloadTooltip')}
                        style={{
                        color: colors.orange, fontWeight: 600, fontSize: '8pt', cursor: 'pointer',
                        padding: '2px 8px', borderRadius: 4,
                        background: alpha(colors.orange, 8),
                        transition: 'background 0.15s',
                      }}>
                        {'\u2B07'} {t('databasebrowser:table.download')}
                      </span>
                    )}
                    {isOpenable && (
                      <span
                        role="button"
                        tabIndex={0}
                        onClick={(e) => {
                          e.stopPropagation();
                          const p = entry.path || entry.name || '';
                          if (/\.(h5|h5oina|hdf5)$/i.test(p)) {
                            // h5/master file \u2192 open in the in-app HDF5 viewer
                            try { sessionStorage.setItem('h5_preload_path', p); } catch { /* unavailable */ }
                            window.dispatchEvent(new CustomEvent('navigate-to', { detail: { page: 'h5viewer' } }));
                          } else {
                            // CIF/XTAL \u2192 show the structure preview panel
                            handleRowClick(i);
                          }
                        }}
                        title={t('databasebrowser:table.openTooltip')}
                        style={{
                        color: colors.cyan, fontWeight: 600, fontSize: '8pt', cursor: 'pointer',
                        padding: '2px 8px', borderRadius: 4,
                        background: alpha(colors.cyan, 8),
                        transition: 'background 0.15s',
                      }}>
                        {'\u2197'} {t('databasebrowser:table.open')}
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Thumbnail preview (async loading from backend)
// ---------------------------------------------------------------------------
function ThumbnailPreview({ filename, isLocal, fileType }) {
  const { t } = useTranslation('databasebrowser');
  const [thumb, setThumb] = useState(null);
  const [loading, setLoading] = useState(false);
  const ft = (fileType || '').toUpperCase();

  useEffect(() => {
    if (!isLocal || !['SHT', 'H5', 'MASTER'].includes(ft) || !filename) {
      setThumb(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    dbApi.preview(filename)
      .then(r => { if (!cancelled) setThumb(r.data); })
      .catch(() => { if (!cancelled) setThumb(null); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [filename, isLocal, ft]);

  const boxStyle = {
    height: 160, background: colors.bgSecondary,
    border: `1px solid ${colors.border}`, borderRadius: 4,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    color: colors.textSecondary, fontSize: '9pt',
  };

  if (!isLocal || !['SHT', 'H5', 'MASTER'].includes(ft)) {
    return <div style={boxStyle}>{ft === 'CIF' || ft === 'XTAL' ? t('databasebrowser:preview.thumbNoPreviewForType', { type: ft }) : t('databasebrowser:preview.thumbSelectLocal')}</div>;
  }
  if (loading) return <div style={boxStyle}>{t('databasebrowser:preview.thumbLoading')}</div>;
  if (thumb?.image) {
    return (
      <>
        <img src={`data:image/png;base64,${thumb.image}`} alt={t('databasebrowser:preview.title')}
          style={{ width: '100%', height: 160, objectFit: 'contain', background: '#282a36', borderRadius: 4, border: `1px solid ${colors.border}` }} />
        {thumb.metadata && Object.keys(thumb.metadata).length > 0 && (
          <GroupBox title={t('databasebrowser:preview.patternInfo')} style={{ marginTop: spacing.outerSpacing }}>
            {Object.entries(thumb.metadata).map(([k, v]) => (
              <div key={k} className="table-row-hover" style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 4px', fontSize: '9pt', borderBottom: `1px solid ${alpha(colors.border, 13)}`, borderRadius: 2 }}>
                <span style={{ color: colors.purple, fontWeight: 600 }}>{k}:</span>
                <span style={{ color: colors.text }}>{String(v)}</span>
              </div>
            ))}
          </GroupBox>
        )}
      </>
    );
  }
  return <div style={boxStyle}>{t('databasebrowser:preview.thumbNone')}</div>;
}

// ---------------------------------------------------------------------------
// SHT provenance + simulation parameters (File Info extension)
// ---------------------------------------------------------------------------
export function ShtInfoBlocks({ filename, isLocal }) {
  const { t } = useTranslation('databasebrowser');
  const [info, setInfo] = useState(null);
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (!filename || !isLocal) { setInfo(null); return; }
    let cancelled = false;
    dbApi.shtInfo(filename)
      .then((r) => { if (!cancelled) setInfo(r.data); })
      .catch(() => { if (!cancelled) setInfo(null); });
    return () => { cancelled = true; };
  }, [filename, isLocal]);
  if (!info) return null;

  const P = info.provenance || {};
  const params = info.parameters || {};
  const row = (label, value) => (
    <div key={label} style={{ display: 'flex', justifyContent: 'space-between',
      padding: '2px 4px', fontSize: '9pt' }}>
      <span style={{ color: colors.purple, fontWeight: 600 }}>{label}:</span>
      <span style={{ color: value ? colors.text : colors.textSecondary,
        maxWidth: '60%', textAlign: 'right', wordBreak: 'break-all' }}>
        {value || t('databasebrowser:shtInfo.unknown')}
      </span>
    </div>
  );
  const paramRows = [
    ['dmin', params.dmin], ['npx', params.npx], [t('databasebrowser:shtInfo.paramVoltage'), params.voltage_kV != null ? `${params.voltage_kV} kV` : null],
    [t('databasebrowser:shtInfo.paramTilt'), params.sig != null ? `${params.sig}°` : null], [t('databasebrowser:shtInfo.paramElectrons'), params.electrons ?? params.totnum_el],
    [t('databasebrowser:shtInfo.paramBandwidth'), params.bandwidth], [t('databasebrowser:shtInfo.paramBethe'), Array.isArray(params.bethe) && params.bethe.length ? params.bethe.join(', ') : null],
  ].filter(([, v]) => v != null && v !== '');

  return (
    <>
      <GroupBox title={t('databasebrowser:shtInfo.provenanceTitle')}>
        {row(t('databasebrowser:shtInfo.parentXtal'), P.source_xtal && P.source_xtal.found ? P.source_xtal.name : null)}
        {row(t('databasebrowser:shtInfo.sourceCif'), P.source_cif && P.source_cif.found ? P.source_cif.name : null)}
        {row(t('databasebrowser:shtInfo.reference'), P.reference)}
      </GroupBox>
      <GroupBox title={t('databasebrowser:shtInfo.parametersTitle')}>
        <div style={{ cursor: 'pointer', fontSize: '9pt', color: colors.accent, padding: '2px 4px' }}
             onClick={() => setOpen((o) => !o)}
             title={t('databasebrowser:shtInfo.parametersToggleTooltip')}>
          {open ? t('databasebrowser:shtInfo.hide') : t('databasebrowser:shtInfo.show')} ({paramRows.length})
        </div>
        {open && paramRows.map(([label, value]) => row(label, String(value)))}
      </GroupBox>
    </>
  );
}

// ---------------------------------------------------------------------------
// Preview panel (right side)
// ---------------------------------------------------------------------------
function PreviewPanel({ entry, onOpenViewer }) {
  const { t } = useTranslation('databasebrowser');
  if (!entry) {
    return (
      <div style={{
        padding: spacing.outerMargin,
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        gap: spacing.outerSpacing,
      }}>
        <div style={{ fontSize: '12pt', fontWeight: 'bold', color: colors.accent }}>{t('databasebrowser:preview.title')}</div>
        <div style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          color: colors.textSecondary,
          fontSize: '10pt',
          gap: 8,
        }}>
          <span style={{ fontSize: '28pt', opacity: 0.2 }}>{'\u2190'}</span>
          <span>{t('databasebrowser:preview.emptyTitle')}</span>
          <span style={{ fontSize: '8pt', opacity: 0.5 }}>{t('databasebrowser:preview.emptyHint')}</span>
        </div>
      </div>
    );
  }

  const locMeta = getLocationMeta(entry.location, t);
  const ft = (entry.file_type || entry.type || entry.category || '').toUpperCase();
  const isLocal = ['local', 'both'].includes((entry.location || '').toLowerCase());

  const rows = [
    { label: t('databasebrowser:preview.fieldFilename'), value: entry.filename || entry.name },
    { label: t('databasebrowser:preview.fieldType'), value: ft },
    { label: t('databasebrowser:preview.fieldMaterial'), value: entry.material },
    { label: t('databasebrowser:preview.fieldLocation'), value: locMeta.label, color: locMeta.color },
    { label: t('databasebrowser:preview.fieldSize'), value: formatBytes(entry.size_bytes ?? entry.size) },
  ].filter((r) => r.value);

  return (
    <div style={{
      padding: spacing.outerMargin,
      display: 'flex',
      flexDirection: 'column',
      gap: spacing.outerSpacing,
      height: '100%',
      overflow: 'auto',
    }}>
      <div style={{ fontSize: '12pt', fontWeight: 'bold', color: colors.accent }}>{t('databasebrowser:preview.title')}</div>

      {/* SHT → interactive rotatable master-pattern sphere; CIF/XTAL →
          interactive 3D crystal structure; everything else → the existing async
          2D thumbnail (SHT has no working 2D thumbnail anyway: the preview
          endpoint opens files with h5py and .sht is not HDF5). */}
      {ft === 'SHT' ? (
        <MasterSphereViewer filename={entry.filename || entry.name} isLocal={isLocal} />
      ) : (ft === 'CIF' || ft === 'XTAL') ? (
        <CrystalStructureViewer filename={entry.filename || entry.name} isLocal={isLocal} fileType={ft} />
      ) : (
        <ThumbnailPreview
          filename={entry.filename || entry.name}
          isLocal={isLocal}
          fileType={ft}
        />
      )}

      <GroupBox title={t('databasebrowser:preview.fileInfo')}>
        {rows.map(({ label, value, color }) => (
          <div key={label} className="table-row-hover" style={{
            display: 'flex',
            justifyContent: 'space-between',
            padding: '2px 4px',
            fontSize: '9pt',
            borderBottom: `1px solid ${alpha(colors.border, 13)}`,
            borderRadius: 2,
          }}>
            <span style={{ color: colors.purple, fontWeight: 600 }}>{label}:</span>
            <span style={{ color: color || colors.text, maxWidth: '60%', textAlign: 'right', wordBreak: 'break-all' }}>
              {value}
            </span>
          </div>
        ))}
      </GroupBox>

      {ft === 'SHT' && (
        <ShtInfoBlocks filename={entry.filename || entry.name} isLocal={isLocal} />
      )}

      <Button
        variant="default"
        disabled={!isLocal || !ft || !['H5', 'MASTER'].includes(ft)}
        style={{ width: '100%' }}
        onClick={() => onOpenViewer && onOpenViewer(entry)}
        title={t('databasebrowser:preview.openInHdf5ViewerTooltip')}
      >
        {t('databasebrowser:preview.openInHdf5Viewer')}
      </Button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function DatabasePage({ onNavigate, isActive = false }) {
  const { t } = useTranslation(['databasebrowser', 'common']);
  const [activeTab, setActiveTab] = useState('sht');
  const [allEntries, setAllEntries] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [offlineMode, setOfflineMode] = useState(false);
  const [cacheStats, setCacheStats] = useState(null);
  const [selectedEntry, setSelectedEntry] = useState(null);
  const [statusMsg, setStatusMsg] = useState(null);
  const [lastRefreshed, setLastRefreshed] = useState(null);
  const [askConfirm, confirmProps] = useConfirm();
  const [selectedFiles, setSelectedFiles] = useState(new Set());
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [cascadeData, setCascadeData] = useState(null);
  const [deleting, setDeleting] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [syncResult, setSyncResult] = useState(null);   // null = conflict dialog closed
  const [resolving, setResolving] = useState(false);
  const [showSyncUpload, setShowSyncUpload] = useState(false);
  const [syncMode, setSyncMode] = useState('upload');   // 'upload' (local->server) | 'download' (server->local)
  const [syncRunning, setSyncRunning] = useState(false);
  const [syncProgress, setSyncProgress] = useState(null);     // { current, total, name }
  const [syncSummary, setSyncSummary] = useState(null);       // { uploaded, upToDate, conflicts, errors, cancelled }
  const syncCancelRef = useRef(false);
  const syncAbortRef = useRef(null);   // AbortController for the in-flight upload

  // ---------------------------------------------------------------------------
  // Load data
  // ---------------------------------------------------------------------------
  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await dbApi.browse('all', '');
      const entries = res.data?.entries || res.data?.files || res.data || [];
      setAllEntries(Array.isArray(entries) ? entries : []);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || t('databasebrowser:status.loadFailed'));
    } finally {
      setLoading(false);
      setLastRefreshed(new Date());
    }
  }, []);

  const loadCacheStats = useCallback(async () => {
    try {
      const res = await dbApi.cacheStatus();
      setCacheStats(res.data);
    } catch {
      // ignore — stats are non-critical
    }
  }, []);

  useEffect(() => {
    if (!isActive) return;
    loadData();
    loadCacheStats();

    // Auto-refresh every 5 s when page is visible (matches PyQt5 parity)
    const timer = setInterval(() => {
      if (!offlineMode) {
        loadData();
        loadCacheStats();
      }
    }, 5000);
    return () => clearInterval(timer);
  }, [isActive, loadData, loadCacheStats, offlineMode]);

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------
  const handleRefresh = () => {
    loadData();
    loadCacheStats();
  };

  const handleSyncAll = async () => {
    // Use the real bidirectional sync endpoint. The previous code looped over
    // server-only files calling addCif() \u2014 the wrong API (addCif imports an
    // external CIF into the library); it never uploaded local files and never
    // detected conflicts. /sync handles upload + download + hash-based
    // conflict detection atomically for CIF/XTAL.
    setStatusMsg({ text: t('databasebrowser:status.syncing'), type: 'info' });
    try {
      const { data } = await dbApi.sync();
      await loadData();
      await loadCacheStats();
      const n = (k) => (data?.[k] || []).length;
      const parts = [];
      if (n('downloaded')) parts.push(t('databasebrowser:status.syncDownloaded', { count: n('downloaded') }));
      if (n('uploaded')) parts.push(t('databasebrowser:status.syncUploaded', { count: n('uploaded') }));
      if (n('up_to_date')) parts.push(t('databasebrowser:status.syncUpToDate', { count: n('up_to_date') }));
      if (n('conflicts')) parts.push(t('databasebrowser:status.syncConflicts', { count: n('conflicts') }));
      if (n('errors')) parts.push(t('databasebrowser:status.syncErrors', { count: n('errors') }));
      const type = n('errors') ? 'error' : (n('conflicts') ? 'info' : 'success');
      const suffix = n('conflicts') ? t('databasebrowser:status.syncConflictSuffix') : '';
      setStatusMsg({ text: t('databasebrowser:status.syncComplete', { parts: parts.join(', ') || t('databasebrowser:status.syncNothing'), suffix }), type });
      setTimeout(() => setStatusMsg(null), 9000);
      // Open the conflict-resolution dialog so the "review conflicting files"
      // hint is actionable on this page (not just on the Crystal Database page).
      if (n('conflicts')) setSyncResult(data);
    } catch (err) {
      setStatusMsg({ text: t('databasebrowser:status.syncError', { error: err.response?.data?.detail || err.message || err }), type: 'error' });
      setTimeout(() => setStatusMsg(null), 9000);
    }
  };

  // Resolve CIF/XTAL conflicts surfaced by Sync All (keep_local/keep_server/…).
  const handleResolveConflicts = async (resolutions) => {
    setResolving(true);
    try {
      await dbApi.resolveConflicts(resolutions);
      setSyncResult(null);
      await loadData();
      await loadCacheStats();
      setStatusMsg({ text: t('databasebrowser:status.conflictsResolved', { count: resolutions.length }), type: 'success' });
      setTimeout(() => setStatusMsg(null), 8000);
    } catch (err) {
      setStatusMsg({ text: t('databasebrowser:status.resolveFailed', { error: err.response?.data?.detail || err.message }), type: 'error' });
      setTimeout(() => setStatusMsg(null), 8000);
    } finally {
      setResolving(false);
    }
  };

  // Push local files to the server (any category). Differing server copies are
  // reported as conflicts; we ask once, then re-send with overwrite — this is
  // what resolves the "16 conflicts" in favour of the local versions.
  const doUpload = async (entries, overwrite) => {
    setUploading(true);
    setStatusMsg({ text: t('databasebrowser:status.uploading', { count: entries.length }), type: 'info' });
    try {
      const { data } = await dbApi.upload(entries, { overwrite });
      const nUp = (data?.uploaded || []).length;
      const nSkip = (data?.up_to_date || []).length;
      const nConf = (data?.conflicts || []).length;
      const nErr = (data?.errors || []).length;
      await loadData();
      await loadCacheStats();
      setStatusMsg({
        text: t('databasebrowser:status.uploadDone', { uploaded: nUp, upToDate: nSkip, conflicts: nConf, errors: nErr }),
        type: nErr ? 'error' : 'success',
      });
      setTimeout(() => setStatusMsg(null), 9000);

      if (!overwrite && nConf > 0) {
        // useConfirm is callback-based (not a promise): re-upload from onConfirm.
        askConfirm({
          title: t('databasebrowser:upload.overwriteTitle'),
          message: t('databasebrowser:upload.overwriteMessage', { count: nConf }),
          confirmLabel: t('databasebrowser:upload.overwriteConfirm'),
          variant: 'danger',
          onConfirm: () => { doUpload(entries, true); },
        });
      } else {
        setSelectedFiles(new Set());
      }
    } catch (err) {
      setStatusMsg({ text: t('databasebrowser:status.uploadFailed', { error: err.response?.data?.detail || err.message }), type: 'error' });
      setTimeout(() => setStatusMsg(null), 9000);
    } finally {
      setUploading(false);
    }
  };

  const handleUploadSelected = () => {
    const entries = allEntries
      .filter(e => selectedFiles.has(e.name || e.filename))
      // Only files present locally can be uploaded.
      .filter(e => ['local', 'both'].includes((e.location || '').toLowerCase()))
      .map(e => ({
        name: e.name || e.filename,
        category: (e.file_type || e.type || e.category || '').toLowerCase(),
        material: e.material || '',
      }));
    if (entries.length === 0) {
      setStatusMsg({ text: t('databasebrowser:status.uploadNothingLocal'), type: 'info' });
      setTimeout(() => setStatusMsg(null), 6000);
      return;
    }
    doUpload(entries, false);
  };

  // Pull a single server-only file into the local cache (any category).
  // Replaces the old behaviour where the per-row download button misfired
  // into Sync All (CIF/XTAL only — useless for SHT/h5).
  const handleDownload = async (entry) => {
    if (!entry) return;
    const file = {
      name: entry.name || entry.filename,
      category: (entry.file_type || entry.type || entry.category || '').toLowerCase(),
      material: entry.material || '',
    };
    setStatusMsg({ text: t('databasebrowser:status.downloading', { name: file.name }), type: 'info' });
    try {
      const { data } = await dbApi.download([file]);
      const ok = (data?.downloaded || []).length > 0;
      const skipped = (data?.up_to_date || []).length > 0;
      setStatusMsg({
        text: ok ? t('databasebrowser:status.downloaded', { name: file.name })
                 : skipped ? t('databasebrowser:status.downloadUpToDate', { name: file.name })
                 : t('databasebrowser:status.downloadFailed', { error: (data?.errors?.[0]?.error) || '' }),
        type: ok || skipped ? 'success' : 'error',
      });
      setTimeout(() => setStatusMsg(null), 8000);
      await loadData();
      await loadCacheStats();
    } catch (err) {
      setStatusMsg({ text: t('databasebrowser:status.downloadFailed', { error: err.response?.data?.detail || err.message }), type: 'error' });
      setTimeout(() => setStatusMsg(null), 8000);
    }
  };

  // --- Category-select bulk upload (with progress bar) ---------------------
  // Build per-category upload candidates from the current listing. Files are
  // "uploadable" when they exist locally (location local or both); 'both' means
  // they also exist on the server (only changed if differing + overwrite).
  const buildSyncCategories = () => {
    // Labels are file-format acronyms — identical in every language (match TAB_DEFS).
    // Dictionaries are OFF by default in BOTH directions (defaultOff): includable
    // on demand, never auto-pushed to / pulled from the shared server. MC h5 is
    // OFF by default only for DOWNLOAD (downloadOff) — it's a large intermediate a
    // fresh user rarely needs; SHT/Master/CIF/XTAL are what indexing consumes.
    const defs = [
      { id: 'sht',        label: 'SHT' },
      { id: 'h5',         label: 'MC h5', downloadOff: true },
      { id: 'master',     label: 'Master H5' },
      { id: 'cif',        label: 'CIF' },
      { id: 'xtal',       label: 'XTAL' },
      { id: 'dictionary', label: 'Dictionary', defaultOff: true },
    ];
    return defs.map(d => {
      const inCat = allEntries.filter(e => (e.file_type || e.type || e.category || '').toLowerCase() === d.id);
      const local = inCat.filter(e => (e.location || '').toLowerCase() === 'local');
      const server = inCat.filter(e => (e.location || '').toLowerCase() === 'server');
      const both = inCat.filter(e => (e.location || '').toLowerCase() === 'both');
      // uploadable = files present locally (local|both); downloadable = files
      // present on the server (server|both). 'both' files are counted on both
      // sides but resolve to "up-to-date" no-ops unless they differ + overwrite.
      return {
        ...d,
        uploadable: local.length + both.length,
        downloadable: server.length + both.length,
        both: both.length,
      };
    });
  };

  const handleOpenSyncUpload = () => {
    setSyncMode('upload');
    setSyncSummary(null);
    setSyncProgress(null);
    setShowSyncUpload(true);
  };

  const handleOpenSyncDownload = () => {
    setSyncMode('download');
    setSyncSummary(null);
    setSyncProgress(null);
    setShowSyncUpload(true);
  };

  const handleCancelSync = () => {
    syncCancelRef.current = true;
    // Abort the request that's currently in flight so a big file (master .h5 is
    // ~32 MB) doesn't have to finish before the loop stops.
    syncAbortRef.current?.abort();
  };

  const handleCloseSyncUpload = () => {
    if (syncRunning) return;          // can't close mid-run; use Cancel
    setShowSyncUpload(false);
    setSyncSummary(null);
    setSyncProgress(null);
  };

  // Transfer all files in the chosen categories, one request per file so the
  // progress bar advances per file (the big master .h5 are ~32 MB each).
  // direction 'upload' pushes local|both files to the server; 'download' pulls
  // server|both files to the local cache. The dialog's onStart routes here with
  // the active syncMode. Totals use a neutral `transferred` count (the dialog
  // labels it "uploaded"/"downloaded" via its mode-aware i18n namespace).
  const runCategoryTransfer = async (catIds, overwrite, direction) => {
    const isDownload = direction === 'download';
    const locWanted = isDownload ? ['server', 'both'] : ['local', 'both'];
    const wanted = new Set(catIds);
    const files = allEntries
      .filter(e => wanted.has((e.file_type || e.type || e.category || '').toLowerCase()))
      .filter(e => locWanted.includes((e.location || '').toLowerCase()))
      .map(e => ({
        name: e.name || e.filename,
        category: (e.file_type || e.type || e.category || '').toLowerCase(),
        material: e.material || '',
      }));
    if (files.length === 0) return;

    syncCancelRef.current = false;
    setSyncRunning(true);
    setSyncSummary(null);
    const totals = { transferred: 0, upToDate: 0, conflicts: 0, errors: 0, cancelled: false };
    setSyncProgress({ current: 0, total: files.length, name: '' });

    for (let i = 0; i < files.length; i++) {
      if (syncCancelRef.current) { totals.cancelled = true; break; }
      const f = files[i];
      setSyncProgress({ current: i, total: files.length, name: f.name });
      const controller = new AbortController();
      syncAbortRef.current = controller;
      try {
        const { data } = isDownload
          ? await dbApi.download([f], { overwrite, signal: controller.signal })
          : await dbApi.upload([f], { overwrite, signal: controller.signal });
        totals.transferred += ((isDownload ? data?.downloaded : data?.uploaded) || []).length;
        totals.upToDate += (data?.up_to_date || []).length;
        totals.conflicts += (data?.conflicts || []).length;
        totals.errors += (data?.errors || []).length;
      } catch {
        // Cancelled (aborted) mid-request, or a real failure.
        if (syncCancelRef.current || controller.signal.aborted) {
          totals.cancelled = true;
          break;
        }
        totals.errors += 1;
      } finally {
        syncAbortRef.current = null;
      }
      setSyncProgress({ current: i + 1, total: files.length, name: f.name });
    }

    setSyncRunning(false);
    setSyncSummary(totals);
    await loadData();
    await loadCacheStats();
  };

  // Dialog onStart — runs in the currently active direction (syncMode).
  const handleStartCategorySync = (catIds, overwrite) =>
    runCategoryTransfer(catIds, overwrite, syncMode);

  const handleClearCache = () => {
    // Honest behaviour: there is NO separate cache directory — the "cache"
    // counted by /cache/status is the actual CIF/XTAL/SHT/H5 library on disk,
    // and CIF/XTAL are user-authored (not re-downloadable). A blanket delete
    // would destroy curated data, so we don't offer one. The previous handler
    // only called loadData()/loadCacheStats() (a refresh) while claiming to
    // "clear all files" — a no-op that misrepresented what happened. Direct the
    // user to the safe, explicit per-file delete instead, and refresh stats.
    loadData();
    loadCacheStats();
    setStatusMsg({
      text: t('databasebrowser:status.cacheClearDisabled'),
      type: 'info',
    });
    setTimeout(() => setStatusMsg(null), 9000);
  };

  const handleToggleSelect = useCallback((name) => {
    setSelectedFiles(prev => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }, []);

  const handleSelectAll = useCallback((filteredEntries) => {
    setSelectedFiles(prev => {
      const allNames = new Set(filteredEntries.map(e => e.name || e.filename));
      const allSelected = filteredEntries.every(e => prev.has(e.name || e.filename));
      if (allSelected) return new Set(); // deselect all
      return allNames;
    });
  }, []);

  const handleDeleteSelected = async () => {
    const entries = allEntries.filter(e => selectedFiles.has(e.name || e.filename));
    const cascadeFiles = entries.map(e => ({
      name: e.name || e.filename,
      category: (e.file_type || e.type || e.category || '').toLowerCase(),
      material: e.material || '',
    }));

    setStatusMsg({ text: t('databasebrowser:status.resolvingDeps'), type: 'info' });
    try {
      const res = await dbApi.resolveCascade(cascadeFiles);
      setCascadeData(res.data);
      setShowDeleteDialog(true);
      setStatusMsg(null);
    } catch (err) {
      setStatusMsg({ text: t('databasebrowser:status.resolveDepsFailed', { error: err.response?.data?.detail || err.message }), type: 'error' });
      setTimeout(() => setStatusMsg(null), 8000);
    }
  };

  const handleConfirmDelete = async (deleteFrom, files) => {
    setDeleting(true);
    try {
      const res = await dbApi.deleteFiles(files, { deleteFrom });
      const count = (res.data?.deleted?.length || 0) + (res.data?.server_deleted?.length || 0);
      const errCount = res.data?.errors?.length || 0;
      const msg = errCount > 0
        ? t('databasebrowser:status.deletedWithErrors', { count, errors: errCount })
        : t('databasebrowser:status.deleted', { count });
      setStatusMsg({ text: msg, type: errCount > 0 ? 'error' : 'success' });
      setTimeout(() => setStatusMsg(null), 8000);
      setShowDeleteDialog(false);
      setCascadeData(null);
      setSelectedFiles(new Set());
      loadData();
      loadCacheStats();
    } catch (err) {
      setStatusMsg({ text: t('databasebrowser:status.deleteFailed', { error: err.response?.data?.detail || err.message }), type: 'error' });
      setTimeout(() => setStatusMsg(null), 8000);
    } finally {
      setDeleting(false);
    }
  };

  const handleCancelDelete = () => {
    setShowDeleteDialog(false);
    setCascadeData(null);
  };

  // ---------------------------------------------------------------------------
  // Cache stats display
  // ---------------------------------------------------------------------------
  let statsText = t('databasebrowser:cache.loading');
  if (cacheStats) {
    const used = cacheStats.used_bytes ?? cacheStats.size_bytes ?? cacheStats.total_bytes ?? 0;
    const max = cacheStats.max_bytes ?? 20 * 1024 * 1024 * 1024;
    const count = cacheStats.file_count ?? cacheStats.count ?? cacheStats.n_files ?? 0;
    const usedGb = (used / (1024 ** 3)).toFixed(2);
    const maxGb = (max / (1024 ** 3)).toFixed(1);
    statsText = t('databasebrowser:cache.stats', { count, used: usedGb, max: maxGb });
  }

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  const TAB_TIP_KEYS = {
    sht: 'tabs.shtTip', mc: 'tabs.mcTip', master: 'tabs.masterTip',
    cif: 'tabs.cifTip', xtal: 'tabs.xtalTip', dict: 'tabs.dictTip',
  };
  const tabs = TAB_DEFS.map((tab) => {
    const count = allEntries.filter((e) => entryMatchesTab(e, tab)).length;
    return {
      id: tab.id,
      label: count > 0 ? `${tab.label} (${count})` : tab.label,
      tip: TAB_TIP_KEYS[tab.id] ? t(`databasebrowser:${TAB_TIP_KEYS[tab.id]}`) : tab.tip,
    };
  });

  const tableArea = (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>

      {/* Header */}
      <div style={{ padding: `${spacing.outerMargin}px ${spacing.outerMargin}px 0` }}>
        <h1 style={{ margin: 0, fontSize: '18pt', fontWeight: 700, color: colors.accent }}>
          {t('databasebrowser:title')}
        </h1>
        <div style={{ fontSize: '10pt', color: colors.textSecondary, marginBottom: spacing.outerSpacing }}>
          {t('databasebrowser:subtitle')}
        </div>

        {/* Top controls */}
        <div style={{ display: 'flex', gap: spacing.outerSpacing, alignItems: 'center', marginBottom: spacing.outerSpacing }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}
                 title={t('databasebrowser:controls.offlineModeTooltip')}>
            <input
              type="checkbox"
              checked={offlineMode}
              onChange={(e) => setOfflineMode(e.target.checked)}
              title={t('databasebrowser:controls.offlineModeTooltip')}
            />
            <Label>{t('databasebrowser:controls.offlineMode')}</Label>
          </label>
          <div style={{ flex: 1 }} />
          {lastRefreshed && (
            <span style={{ fontSize: '8pt', color: colors.textSecondary, opacity: 0.6 }}>
              {t('databasebrowser:controls.updated', { time: lastRefreshed.toLocaleTimeString() })}
            </span>
          )}
          <Button onClick={handleRefresh} disabled={loading} title={t('databasebrowser:controls.refreshTooltip')}>
            {loading ? t('common:loading') : t('databasebrowser:controls.refresh')}
          </Button>
          <Button
            onClick={handleSyncAll}
            title={t('databasebrowser:controls.syncAllTooltip')}
          >
            {t('databasebrowser:controls.syncAll')}
          </Button>
          <Button
            onClick={handleOpenSyncDownload}
            title={t('databasebrowser:controls.downloadAllTooltip')}
          >
            {'⬇'} {t('databasebrowser:controls.downloadAll')}
          </Button>
          <Button
            onClick={handleOpenSyncUpload}
            title={t('databasebrowser:controls.uploadAllTooltip')}
          >
            {'⬆'} {t('databasebrowser:controls.uploadAll')}
          </Button>
          {selectedFiles.size > 0 && (
            <Button
              onClick={handleUploadSelected}
              disabled={uploading}
              title={t('databasebrowser:controls.uploadSelectedTooltip')}
            >
              {'⬆'} {t('databasebrowser:controls.uploadSelected', { count: selectedFiles.size })}
            </Button>
          )}
          {selectedFiles.size > 0 && (
            <Button
              variant="danger"
              onClick={handleDeleteSelected}
              title={t('databasebrowser:controls.deleteSelectedTooltip')}
            >
              {t('databasebrowser:controls.deleteSelected', { count: selectedFiles.size })}
            </Button>
          )}
        </div>

        {error && (
          <div role="alert" style={{
            padding: '6px 10px',
            marginBottom: spacing.outerSpacing,
            background: alpha(colors.red, 8),
            border: `1px solid ${colors.red}`,
            borderRadius: 4,
            fontSize: '9pt',
            color: colors.red,
            animation: 'fadeSlideIn 0.2s ease-out',
          }}>
            {'\u26A0'} {error}
          </div>
        )}

        {statusMsg && (
          <div role={statusMsg.type === 'error' ? 'alert' : 'status'} style={{
            padding: '6px 10px',
            marginBottom: spacing.outerSpacing,
            background: statusMsg.type === 'error' ? alpha(colors.red, 8)
                       : statusMsg.type === 'success' ? alpha(colors.green, 8)
                       : alpha(colors.cyan, 8),
            border: `1px solid ${statusMsg.type === 'error' ? colors.red
                                : statusMsg.type === 'success' ? colors.green
                                : colors.cyan}`,
            borderRadius: 4,
            fontSize: '9pt',
            color: statusMsg.type === 'error' ? colors.red
                  : statusMsg.type === 'success' ? colors.green
                  : colors.cyan,
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            animation: 'fadeSlideIn 0.2s ease-out',
          }}>
            <span>{statusMsg.text}</span>
            <button
              onClick={() => setStatusMsg(null)}
              title={t('databasebrowser:status.dismissTooltip')}
              style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', fontSize: 14, padding: '0 4px' }}
            >&times;</button>
          </div>
        )}
      </div>

      {/* Tabs */}
      <div style={{ padding: `0 ${spacing.outerMargin}px` }}>
        <Tabs tabs={tabs} activeTab={activeTab} onTabChange={setActiveTab} />
      </div>

      {/* Tab content */}
      <div style={{ flex: 1, overflow: 'hidden' }}>
        {TAB_DEFS.map((tabDef) => (
          <TabPanel
            key={tabDef.id}
            visible={activeTab === tabDef.id}
            style={{ padding: 0, height: '100%', overflow: 'hidden' }}
          >
            <FileTable
              tabDef={tabDef}
              entries={allEntries}
              onRowClick={setSelectedEntry}
              loading={loading}
              selectedFiles={selectedFiles}
              onToggleSelect={handleToggleSelect}
              onSelectAll={handleSelectAll}
              onDownload={handleDownload}
            />
          </TabPanel>
        ))}
      </div>

      {/* Bottom stats bar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: spacing.outerSpacing,
        padding: `${spacing.innerMargin}px ${spacing.outerMargin}px`,
        borderTop: `1px solid ${colors.border}`,
        background: colors.bgSecondary,
        flexShrink: 0,
      }}>
        <Label secondary small style={{ flex: 1 }}>{statsText}</Label>
        <Button small onClick={handleClearCache} title={t('databasebrowser:cache.refreshStatsTooltip')}>
          {t('databasebrowser:cache.refreshStats')}
        </Button>
      </div>
    </div>
  );

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      color: colors.text,
      fontFamily: "'Segoe UI', system-ui, sans-serif",
    }}>
      <ResizableSplitter
        left={tableArea}
        right={<PreviewPanel entry={selectedEntry} onOpenViewer={(entry) => {
          const path = entry.path || entry.name;
          if (path) {
            h5Api.open(path).then(() => {
              if (onNavigate) onNavigate('h5viewer');
            }).catch((err) => {
              const detail = err.response?.data?.detail || err.message || t('databasebrowser:status.unknownError');
              setStatusMsg({ text: t('databasebrowser:status.openFailed', { name: entry.name || path, detail }), type: 'error' });
            });
          }
        }} />}
        defaultLeftWidth={700}
        minLeftWidth={400}
        maxLeftWidth={1200}
        style={{ flex: 1 }}
      />
      <ConfirmDialog {...confirmProps} />
      <CascadeDeleteDialog
        open={showDeleteDialog}
        cascadeData={cascadeData}
        deleting={deleting}
        onConfirm={handleConfirmDelete}
        onCancel={handleCancelDelete}
      />
      {syncResult && (
        <SyncDialog
          syncResult={syncResult}
          resolving={resolving}
          onResolve={handleResolveConflicts}
          onClose={() => setSyncResult(null)}
        />
      )}
      <SyncUploadDialog
        open={showSyncUpload}
        mode={syncMode}
        categories={showSyncUpload ? buildSyncCategories() : []}
        running={syncRunning}
        progress={syncProgress}
        result={syncSummary}
        onStart={handleStartCategorySync}
        onCancel={handleCancelSync}
        onClose={handleCloseSyncUpload}
      />
    </div>
  );
}
