/**
 * Settings Page — system configuration, server mode, and manual path overrides.
 */

import { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { simApi, settingsApi } from '../../services/api';
import InstallWizardSection from './InstallWizardSection';
import ApiKeysSection from './ApiKeysSection';
import AboutSection from './AboutSection';
import {
  colors, alpha, spacing,
  Button, Input, GroupBox, Label,
} from '../../theme/components';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function StatusDot({ status }) {
  const colorMap = {
    ok:      colors.green,
    error:   colors.red,
    warning: colors.yellow,
    idle:    colors.textSecondary,
  };
  return (
    <span style={{
      display: 'inline-block',
      width: 8,
      height: 8,
      borderRadius: '50%',
      background: colorMap[status] ?? colors.textSecondary,
      flexShrink: 0,
    }} />
  );
}

function Toggle({ enabled, onChange }) {
  const { t } = useTranslation('settings');
  return (
    <button
      type="button"
      onClick={() => onChange(!enabled)}
      aria-label={enabled ? t('settings:serverMode.toggleDisableLabel') : t('settings:serverMode.toggleEnableLabel')}
      title={t('settings:serverMode.toggleTooltip')}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        width: 40,
        height: 22,
        borderRadius: 11,
        background: enabled ? colors.cyan : colors.border,
        border: 'none',
        cursor: 'pointer',
        padding: 2,
        transition: 'background 0.15s',
        flexShrink: 0,
      }}
    >
      <span style={{
        display: 'block',
        width: 18,
        height: 18,
        borderRadius: '50%',
        background: colors.text,
        transform: enabled ? 'translateX(18px)' : 'translateX(0)',
        transition: 'transform 0.15s',
      }} />
    </button>
  );
}

// ---------------------------------------------------------------------------
// Section 1 — System Status
// ---------------------------------------------------------------------------

function SystemStatusSection() {
  const { t } = useTranslation('settings');
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchStatus = useCallback(async (opts = {}) => {
    setLoading(true);
    setError(null);
    try {
      // The Re-Check button should bypass the 5-min backend cache so users
      // see the actual current state after fixing a dependency.
      const res = await simApi.systemStatus({ force: !!opts.force });
      setStatus(res.data);
    } catch (err) {
      setError(err?.response?.data?.detail ?? err.message ?? t('settings:systemStatus.fetchError'));
    } finally {
      setLoading(false);
    }
  }, [t]);

  useEffect(() => { fetchStatus(); }, [fetchStatus]);

  const rows = status ? [
    {
      label: t('settings:systemStatus.rows.wsl'),
      value: status.wsl_distro || t('settings:systemStatus.values.dash'),
      dotStatus: status.wsl_installed ? 'ok' : 'error',
    },
    {
      label: t('settings:systemStatus.rows.emsoft'),
      value: status.emsoft_path || (status.emsoft_available ? t('settings:systemStatus.values.detected') : t('settings:systemStatus.values.notFound')),
      dotStatus: status.emsoft_available ? 'ok' : 'error',
    },
    {
      label: t('settings:systemStatus.rows.emsphinx'),
      value: status.emsphinx_path || (status.emsphinx_available ? t('settings:systemStatus.values.detected') : t('settings:systemStatus.values.notFound')),
      dotStatus: status.emsphinx_available ? 'ok' : 'error',
    },
    {
      label: t('settings:systemStatus.rows.opencl'),
      value: status.opencl_available
        ? (status.gpu_name || t('settings:systemStatus.values.available'))
        : t('settings:systemStatus.values.notAvailable'),
      dotStatus: status.opencl_available ? 'ok' : 'warning',
    },
    {
      label: t('settings:systemStatus.rows.gpuMemory'),
      value: status.has_gpu && status.gpu_memory_gb != null
        ? `${status.gpu_memory_gb.toFixed(1)} GB`
        : t('settings:systemStatus.values.dash'),
      dotStatus: null,
    },
    {
      label: t('settings:systemStatus.rows.cpuCores'),
      value: status.cpu_count != null ? String(status.cpu_count) : t('settings:systemStatus.values.dash'),
      dotStatus: null,
    },
    {
      label: t('settings:systemStatus.rows.mode'),
      value: status.recommended_mode
        ? t('settings:systemStatus.values.modeRecommended', {
            mode: status.recommended_mode.charAt(0).toUpperCase() + status.recommended_mode.slice(1),
          })
        : t('settings:systemStatus.values.dash'),
      dotStatus: null,
    },
  ] : [];

  return (
    <GroupBox title={t('settings:systemStatus.title')}>
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: spacing.groupSpacing }}>
        <Button onClick={() => fetchStatus({ force: true })} disabled={loading} title={t('settings:systemStatus.recheckTooltip')}>
          {loading ? t('settings:systemStatus.checking') : t('settings:systemStatus.recheck')}
        </Button>
      </div>

      {error && (
        <div style={{
          color: colors.red,
          fontSize: '11px',
          marginBottom: spacing.innerSpacing,
          padding: '6px 10px',
          background: alpha(colors.red, 10),
          borderRadius: 4,
          border: `1px solid ${alpha(colors.red, 30)}`,
        }}>
          {error}
        </div>
      )}

      {loading && !status && (
        <Label secondary style={{ fontSize: '11px' }}>{t('settings:systemStatus.loading')}</Label>
      )}

      {rows.length > 0 && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{
            width: '100%',
            borderCollapse: 'collapse',
            fontSize: '12px',
            color: colors.text,
          }}>
            <thead>
              <tr style={{ borderBottom: `1px solid ${colors.border}` }}>
                {[t('settings:systemStatus.columnComponent'), t('settings:systemStatus.columnValue'), t('settings:systemStatus.columnStatus')].map((h) => (
                  <th key={h} style={{
                    textAlign: 'left',
                    padding: '4px 10px 6px',
                    color: colors.textSecondary,
                    fontWeight: 600,
                    fontSize: '11px',
                    whiteSpace: 'nowrap',
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={row.label} style={{
                  background: i % 2 === 0 ? 'transparent' : alpha(colors.border, 20),
                }}>
                  <td style={{ padding: '5px 10px', color: colors.textSecondary, whiteSpace: 'nowrap' }}>
                    {row.label}
                  </td>
                  <td style={{ padding: '5px 10px', fontFamily: 'monospace', fontSize: '11px' }}>
                    {row.value}
                  </td>
                  <td style={{ padding: '5px 10px' }}>
                    {row.dotStatus ? (
                      <StatusDot status={row.dotStatus} />
                    ) : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {status?.errors?.length > 0 && (
        <div style={{ marginTop: spacing.groupSpacing }}>
          {status.errors.map((msg, i) => (
            <div key={i} style={{ color: colors.red, fontSize: '11px', marginTop: 4 }}>
              {t('settings:systemStatus.errorPrefix', { message: msg })}
            </div>
          ))}
        </div>
      )}

      {status?.warnings?.length > 0 && (
        <div style={{ marginTop: status?.errors?.length > 0 ? 4 : spacing.groupSpacing }}>
          {status.warnings.map((msg, i) => (
            <div key={i} style={{ color: colors.yellow, fontSize: '11px', marginTop: 4 }}>
              {t('settings:systemStatus.warningPrefix', { message: msg })}
            </div>
          ))}
        </div>
      )}
    </GroupBox>
  );
}

// ---------------------------------------------------------------------------
// Section 2 — Manual Paths
// ---------------------------------------------------------------------------

function ManualPathsSection() {
  const { t } = useTranslation('settings');
  const [emsoftBin, setEmsoftBin] = useState('');
  const [emsphinxDir, setEmsphinxDir] = useState('');
  const [originalPaths, setOriginalPaths] = useState({ emsoft_bin: '', emsphinx_dir: '' });
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState(null);

  useEffect(() => {
    settingsApi.get()
      .then((res) => {
        const paths = res.data?.paths ?? {};
        setEmsoftBin(paths.emsoft_bin ?? '');
        setEmsphinxDir(paths.emsphinx_dir ?? '');
        setOriginalPaths({ emsoft_bin: paths.emsoft_bin ?? '', emsphinx_dir: paths.emsphinx_dir ?? '' });
      })
      .catch(() => {});
  }, []);

  const dirty = emsoftBin !== originalPaths.emsoft_bin || emsphinxDir !== originalPaths.emsphinx_dir;

  const handleSave = useCallback(async () => {
    setSaving(true);
    setSaveMsg(null);
    try {
      await settingsApi.savePaths({ emsoft_bin: emsoftBin, emsphinx_dir: emsphinxDir });
      setOriginalPaths({ emsoft_bin: emsoftBin, emsphinx_dir: emsphinxDir });
      setSaveMsg({ type: 'ok', text: t('settings:manualPaths.saved') });
    } catch (err) {
      setSaveMsg({ type: 'error', text: err?.response?.data?.detail ?? t('settings:manualPaths.saveFailed') });
    } finally {
      setSaving(false);
      setTimeout(() => setSaveMsg(null), 3000);
    }
  }, [emsoftBin, emsphinxDir, t]);

  return (
    <GroupBox title={t('settings:manualPaths.title')}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: spacing.innerSpacing }}>
        <div>
          <Label style={{ display: 'block', marginBottom: 4, fontSize: '11px', color: colors.textSecondary }}>
            {t('settings:manualPaths.emsoftBinLabel')}
          </Label>
          <Input
            value={emsoftBin}
            onChange={(e) => setEmsoftBin(e.target.value)}
            placeholder={t('settings:manualPaths.emsoftBinPlaceholder')}
            title={t('settings:manualPaths.emsoftBinTooltip')}
          />
        </div>

        <div>
          <Label style={{ display: 'block', marginBottom: 4, fontSize: '11px', color: colors.textSecondary }}>
            {t('settings:manualPaths.emsphinxDirLabel')}
          </Label>
          <Input
            value={emsphinxDir}
            onChange={(e) => setEmsphinxDir(e.target.value)}
            placeholder={t('settings:manualPaths.emsphinxDirPlaceholder')}
            title={t('settings:manualPaths.emsphinxDirTooltip')}
          />
        </div>

        <Label secondary style={{ fontSize: '11px' }}>
          {t('settings:manualPaths.hint')}
        </Label>

        <div style={{ display: 'flex', alignItems: 'center', gap: spacing.innerSpacing }}>
          <Button variant="primary" onClick={handleSave} disabled={!dirty || saving} title={t('settings:manualPaths.savePathsTooltip')}>
            {saving ? t('settings:manualPaths.saving') : t('settings:manualPaths.savePaths')}
          </Button>
          {saveMsg && (
            <span style={{
              fontSize: '11px',
              color: saveMsg.type === 'ok' ? colors.green : colors.red,
            }}>
              {saveMsg.text}
            </span>
          )}
        </div>
      </div>
    </GroupBox>
  );
}

// ---------------------------------------------------------------------------
// Section 3 — Server Mode
// ---------------------------------------------------------------------------

// Keys must match backend SERVER_SUBDIRS keys: cif, xtal, sht, h5
const DISCOVERY_ITEMS = [
  { key: 'sht',  labelKey: 'settings:serverMode.discovery.sht' },
  { key: 'h5',   labelKey: 'settings:serverMode.discovery.h5' },
  { key: 'cif',  labelKey: 'settings:serverMode.discovery.cif' },
  { key: 'xtal', labelKey: 'settings:serverMode.discovery.xtal' },
];

function DiscoveryRow({ label, result }) {
  const { t } = useTranslation('settings');
  // result: null (not tested), { exists: bool, count: number } | undefined
  if (!result) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '12px', padding: '3px 0' }}>
        <StatusDot status="idle" />
        <span style={{ color: colors.textSecondary, minWidth: 120 }}>{label}</span>
        <span style={{ color: colors.textSecondary }}>{t('settings:serverMode.discovery.untested')}</span>
      </div>
    );
  }
  if (result.exists) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '12px', padding: '3px 0' }}>
        <StatusDot status="ok" />
        <span style={{ color: colors.text, minWidth: 120 }}>{label}</span>
        <span style={{ color: colors.green }}>{t('settings:serverMode.discovery.exists', { count: result.file_count ?? result.count ?? 0 })}</span>
      </div>
    );
  }
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '12px', padding: '3px 0' }}>
      <StatusDot status="warning" />
      <span style={{ color: colors.text, minWidth: 120 }}>{label}</span>
      <span style={{ color: colors.yellow }}>{t('settings:serverMode.discovery.notFound')}</span>
    </div>
  );
}

function ServerModeSection() {
  const { t } = useTranslation('settings');
  const [enabled, setEnabled] = useState(false);
  const [dbRoot, setDbRoot] = useState('');
  const [originalConfig, setOriginalConfig] = useState({ enabled: false, database_root: '' });
  const [discovery, setDiscovery] = useState(null);
  const [connStatus, setConnStatus] = useState('idle'); // 'idle' | 'ok' | 'error'
  const [testing, setTesting] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState(null);

  useEffect(() => {
    settingsApi.get()
      .then((res) => {
        const srv = res.data?.server ?? {};
        const isEnabled = srv.enabled ?? false;
        const root = srv.database_root ?? '';
        setEnabled(isEnabled);
        setDbRoot(root);
        setOriginalConfig({ enabled: isEnabled, database_root: root });
        // Auto-test connection if server mode is enabled and root is set
        if (isEnabled && root) {
          setTesting(true);
          settingsApi.testServer(root)
            .then((testRes) => {
              setDiscovery(testRes.data?.directories ?? null);
              setConnStatus(testRes.data?.reachable ? 'ok' : 'error');
            })
            .catch(() => setConnStatus('error'))
            .finally(() => setTesting(false));
        }
      })
      .catch(() => {});
  }, []);

  const dirty = enabled !== originalConfig.enabled || dbRoot !== originalConfig.database_root;

  const handleTestConnection = useCallback(async () => {
    setTesting(true);
    setDiscovery(null);
    setConnStatus('idle');
    try {
      const res = await settingsApi.testServer(dbRoot);
      setDiscovery(res.data?.directories ?? null);
      setConnStatus(res.data?.reachable ? 'ok' : 'error');
    } catch {
      setConnStatus('error');
    } finally {
      setTesting(false);
    }
  }, [dbRoot]);

  const handleSave = useCallback(async () => {
    setSaving(true);
    setSaveMsg(null);
    try {
      await settingsApi.saveServer({ enabled, database_root: dbRoot });
      setOriginalConfig({ enabled, database_root: dbRoot });
      setSaveMsg({ type: 'ok', text: t('settings:serverMode.saved') });
    } catch (err) {
      setSaveMsg({ type: 'error', text: err?.response?.data?.detail ?? t('settings:serverMode.saveFailed') });
    } finally {
      setSaving(false);
      setTimeout(() => setSaveMsg(null), 3000);
    }
  }, [enabled, dbRoot, t]);

  const connDotStatus = { idle: 'idle', ok: 'ok', error: 'error' }[connStatus];

  return (
    <GroupBox
      title={
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span>{t('settings:serverMode.title')}</span>
            <StatusDot status={connDotStatus} />
          </div>
          <Toggle enabled={enabled} onChange={setEnabled} />
        </div>
      }
    >
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: spacing.innerSpacing,
        opacity: enabled ? 1 : 0.6,
        transition: 'opacity 0.15s',
      }}>
        <div>
          <Label style={{ display: 'block', marginBottom: 4, fontSize: '11px', color: colors.textSecondary }}>
            {t('settings:serverMode.databaseRootLabel')}
          </Label>
          <Input
            value={dbRoot}
            onChange={(e) => setDbRoot(e.target.value)}
            placeholder={t('settings:serverMode.databaseRootPlaceholder')}
            disabled={!enabled}
            title={t('settings:serverMode.databaseRootTooltip')}
          />
        </div>

        <div style={{
          background: alpha(colors.border, 30),
          borderRadius: 4,
          padding: '8px 12px',
          display: 'flex',
          flexDirection: 'column',
          gap: 2,
        }}>
          {DISCOVERY_ITEMS.map(({ key, labelKey }) => (
            <DiscoveryRow
              key={key}
              label={t(labelKey)}
              result={discovery ? discovery[key] : null}
            />
          ))}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: spacing.innerSpacing, flexWrap: 'wrap' }}>
          <Button onClick={handleTestConnection} disabled={!enabled || testing || !dbRoot} title={t('settings:serverMode.testConnectionTooltip')}>
            {testing ? t('settings:serverMode.testing') : t('settings:serverMode.testConnection')}
          </Button>
          {/* Show "Create Missing" button if some dirs don't exist */}
          {discovery && Object.values(discovery).some(d => !d.exists) && (
            <Button
              onClick={async () => {
                try {
                  await settingsApi.createServerDirs(dbRoot);
                  // Re-test to refresh discovery
                  handleTestConnection();
                } catch { /* ignore */ }
              }}
              disabled={!enabled}
              style={{ fontSize: 11 }}
              title={t('settings:serverMode.createMissingTooltip')}
            >
              {t('settings:serverMode.createMissing')}
            </Button>
          )}
          {connStatus === 'ok' && (
            <span style={{ fontSize: '11px', color: colors.green, display: 'flex', alignItems: 'center', gap: 5 }}>
              <StatusDot status="ok" /> {t('settings:serverMode.connected')}
            </span>
          )}
          {connStatus === 'error' && (
            <span style={{ fontSize: '11px', color: colors.red, display: 'flex', alignItems: 'center', gap: 5 }}>
              <StatusDot status="error" /> {t('settings:serverMode.notReachable')}
            </span>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: spacing.innerSpacing, borderTop: `1px solid ${colors.border}`, paddingTop: spacing.innerSpacing }}>
          <Button variant="primary" onClick={handleSave} disabled={!dirty || saving} title={t('settings:serverMode.saveTooltip')}>
            {saving ? t('settings:serverMode.saving') : t('settings:serverMode.save')}
          </Button>
          {saveMsg && (
            <span style={{
              fontSize: '11px',
              color: saveMsg.type === 'ok' ? colors.green : colors.red,
            }}>
              {saveMsg.text}
            </span>
          )}
        </div>
      </div>
    </GroupBox>
  );
}

// ---------------------------------------------------------------------------
// Section — Dashboard Background
// ---------------------------------------------------------------------------

const BG_OPTIONS = [
  { value: 'clean', labelKey: 'settings:dashboardBackground.clean.label', descKey: 'settings:dashboardBackground.clean.desc', tooltipKey: 'settings:dashboardBackground.clean.tooltip' },
  { value: 'crystal', labelKey: 'settings:dashboardBackground.crystal.label', descKey: 'settings:dashboardBackground.crystal.desc', tooltipKey: 'settings:dashboardBackground.crystal.tooltip' },
  { value: 'fog', labelKey: 'settings:dashboardBackground.fog.label', descKey: 'settings:dashboardBackground.fog.desc', tooltipKey: 'settings:dashboardBackground.fog.tooltip' },
];

function DashboardBackgroundSection() {
  const { t } = useTranslation('settings');
  const [selected, setSelected] = useState(() => {
    try { return localStorage.getItem('kikuchipy_dashboard_bg') || 'clean'; }
    catch { return 'clean'; }
  });

  const handleChange = useCallback((value) => {
    setSelected(value);
    try { localStorage.setItem('kikuchipy_dashboard_bg', value); }
    catch { /* ignore */ }
    // Notify Dashboard immediately
    window.dispatchEvent(new CustomEvent('dashboard-bg-changed'));
  }, []);

  return (
    <GroupBox title={t('settings:dashboardBackground.title')}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {BG_OPTIONS.map((opt) => (
          <label
            key={opt.value}
            title={t(opt.tooltipKey)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              padding: '8px 12px',
              borderRadius: 8,
              cursor: 'pointer',
              background: selected === opt.value
                ? alpha(colors.accent, 8)
                : 'transparent',
              border: `1px solid ${selected === opt.value
                ? alpha(colors.accent, 20)
                : 'rgba(255,255,255,0.04)'}`,
              transition: 'all 0.15s',
            }}
          >
            <input
              type="radio"
              name="dashboard-bg"
              value={opt.value}
              checked={selected === opt.value}
              onChange={() => handleChange(opt.value)}
              title={t(opt.tooltipKey)}
              aria-label={t(opt.labelKey)}
              style={{ accentColor: 'var(--accent-cyan)' }}
            />
            <div>
              <div style={{
                fontSize: 13,
                fontWeight: 600,
                color: selected === opt.value ? colors.accent : colors.text,
              }}>
                {t(opt.labelKey)}
              </div>
              <div style={{
                fontSize: 11,
                color: colors.textSecondary,
                marginTop: 2,
              }}>
                {t(opt.descKey)}
              </div>
            </div>
          </label>
        ))}
      </div>
      <div style={{
        fontSize: 10,
        color: colors.textSecondary,
        opacity: 0.6,
        marginTop: 8,
      }}>
        {t('settings:dashboardBackground.footnote')}
      </div>
    </GroupBox>
  );
}

// ---------------------------------------------------------------------------
// Page root
// ---------------------------------------------------------------------------

export default function SettingsPage({ isActive = false, onNavigate }) {
  // isActive is used by parent to mount/unmount; we expose it here for
  // potential future re-fetch triggers.
  void onNavigate;
  const { t } = useTranslation('settings');

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: 16,
      height: '100%',
      overflow: 'auto',
      padding: '0 4px',
    }}>
      <header style={{ marginBottom: 4 }}>
        <h1 style={{
          fontSize: '18px',
          fontWeight: 600,
          color: colors.accent,
          margin: '0 0 6px',
        }}>
          {t('settings:page.title')}
        </h1>
        <p style={{ color: colors.textSecondary, fontSize: '12px', margin: 0 }}>
          {t('settings:page.subtitle')}
        </p>
      </header>

      <DashboardBackgroundSection />
      <SystemStatusSection />
      <InstallWizardSection />
      <ApiKeysSection />
      <ManualPathsSection />
      <ServerModeSection />
      <AboutSection />
    </div>
  );
}
