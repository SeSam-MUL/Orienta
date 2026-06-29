/**
 * InstallWizardSection — 3-step WSL + EMsoft installation wizard.
 *
 * Step 1: WSL Status — detect / install / repair
 * Step 2: User Setup — create WSL user or reset password
 * Step 3: EMsoft Install — stream install log via WebSocket
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { installApi } from '../../services/api';
import {
  colors, alpha, spacing,
  Button, Input, GroupBox, Label,
} from '../../theme/components';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function StatusBadge({ ok, warn, text }) {
  const color = ok ? colors.green : warn ? colors.yellow : colors.red;
  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 5,
      fontSize: 11,
      color,
      fontWeight: 500,
    }}>
      <span style={{
        width: 7,
        height: 7,
        borderRadius: '50%',
        background: color,
        flexShrink: 0,
        display: 'inline-block',
      }} />
      {text}
    </span>
  );
}

function StepHeader({ number, title, status }) {
  // status: null | { ok, warn, text }
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
      <div style={{
        width: 24,
        height: 24,
        borderRadius: '50%',
        background: alpha(colors.accent, 20),
        border: `1px solid ${alpha(colors.accent, 40)}`,
        color: colors.accent,
        fontSize: 12,
        fontWeight: 700,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
      }}>
        {number}
      </div>
      <span style={{ fontSize: 13, fontWeight: 600, color: colors.text, flex: 1 }}>
        {title}
      </span>
      {status && <StatusBadge {...status} />}
    </div>
  );
}

function StepContainer({ children, dimmed = false }) {
  return (
    <div style={{
      border: `1px solid ${alpha(colors.border, 60)}`,
      borderRadius: 8,
      padding: '14px 16px',
      opacity: dimmed ? 0.4 : 1,
      pointerEvents: dimmed ? 'none' : 'auto',
      transition: 'opacity 0.2s',
    }}>
      {children}
    </div>
  );
}

const DISTROS = [
  'Ubuntu-22.04',
  'Ubuntu-20.04',
  'Ubuntu-24.04',
  'Debian',
];

// ---------------------------------------------------------------------------
// Step 0 — Pre-Flight System Check
// ---------------------------------------------------------------------------

function Step0PreFlight({ wslStatus, loading }) {
  const { t } = useTranslation('settings');
  const nvidia = wslStatus?.nvidia;
  const wslVer = wslStatus?.wsl_version;
  const isInstalled = wslStatus?.installed === true;
  const plat = wslStatus?.platform;

  if (loading || !wslStatus) return null;

  const warnings = [];
  const infos = [];

  // Platform info
  const osName = plat?.os === 'windows'
    ? t('settings:install.preflight.osWindows')
    : plat?.os === 'macos'
      ? t('settings:install.preflight.osMacos')
      : plat?.os === 'linux'
        ? t('settings:install.preflight.osLinux')
        : t('settings:install.preflight.osUnknown');
  infos.push(t('settings:install.preflight.platform', { os: osName, arch: plat?.arch || 'unknown' }));

  // Architecture info
  if (plat?.arch && !['x86_64', 'amd64', 'arm64', 'aarch64'].includes(plat.arch)) {
    warnings.push(t('settings:install.preflight.archUntested', { arch: plat.arch }));
  }

  // macOS info
  if (plat?.os === 'macos') {
    infos.push(t('settings:install.preflight.macosNative'));
    if (plat?.arch === 'arm64') {
      infos.push(t('settings:install.preflight.appleSilicon'));
    } else {
      infos.push(t('settings:install.preflight.intelMac'));
    }
  }

  // Linux native info
  if (plat?.os === 'linux') {
    infos.push(t('settings:install.preflight.linuxNative'));
  }

  // NVIDIA driver info
  if (nvidia?.available) {
    infos.push(t('settings:install.preflight.gpuDetected', { name: nvidia.gpu_name || t('settings:install.preflight.gpuFallbackName') }));
    infos.push(
      nvidia.cuda_version
        ? t('settings:install.preflight.driverInfoCuda', { driver: nvidia.driver_version, cuda: nvidia.cuda_version })
        : t('settings:install.preflight.driverInfo', { driver: nvidia.driver_version })
    );
    if (plat?.os === 'windows' && !nvidia.wsl2_ready) {
      warnings.push(t('settings:install.preflight.driverTooOld', { driver: nvidia.driver_version }));
    }
  } else if (plat?.os === 'windows') {
    warnings.push(t('settings:install.preflight.noNvidia'));
  }

  // WSL version (Windows only)
  if (plat?.os === 'windows') {
    if (isInstalled && wslVer === 1) {
      warnings.push(t('settings:install.preflight.wsl1Warning'));
    } else if (isInstalled && wslVer === 2) {
      infos.push(t('settings:install.preflight.wsl2Info'));
    }
  }

  if (infos.length === 0 && warnings.length === 0) return null;

  return (
    <div style={{
      border: `1px solid ${alpha(colors.border, 60)}`,
      borderRadius: 8,
      padding: '12px 16px',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
        <span style={{ fontSize: 13, fontWeight: 600, color: colors.text }}>
          {t('settings:install.preflight.title')}
        </span>
        {warnings.length > 0
          ? <StatusBadge ok={false} warn text={t('settings:install.preflight.warningCount', { count: warnings.length })} />
          : <StatusBadge ok warn={false} text={t('settings:install.preflight.ready')} />
        }
      </div>

      {infos.length > 0 && (
        <div style={{
          fontSize: 11,
          color: colors.textSecondary,
          lineHeight: 1.7,
          marginBottom: warnings.length > 0 ? 8 : 0,
        }}>
          {infos.map((info, i) => (
            <div key={i}>{info}</div>
          ))}
        </div>
      )}

      {warnings.map((w, i) => (
        <div key={i} style={{
          fontSize: 11,
          color: colors.yellow,
          background: alpha(colors.yellow, 8),
          border: `1px solid ${alpha(colors.yellow, 20)}`,
          borderRadius: 4,
          padding: '6px 10px',
          marginTop: i > 0 ? 6 : 0,
          lineHeight: 1.5,
        }}>
          {w}
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Step 1 — WSL Status
// ---------------------------------------------------------------------------

function Step1Wsl({ wslStatus, loading, onRefresh }) {
  const { t } = useTranslation('settings');
  const [selectedDistro, setSelectedDistro] = useState('Ubuntu-22.04');
  const [installing, setInstalling] = useState(false);
  const [installMsg, setInstallMsg] = useState(null);

  const isInstalled = wslStatus?.installed === true;
  const isCorrupted = wslStatus?.corrupted === true;
  const hasUser = wslStatus?.has_user === true;
  const distroName = wslStatus?.distro ?? '';

  const statusProps = loading
    ? { ok: false, warn: true, text: t('settings:install.step1.checking') }
    : isInstalled && !isCorrupted
      ? { ok: true, warn: false, text: distroName ? t('settings:install.step1.ready', { distro: distroName }) : t('settings:install.step1.readyFallback') }
      : isCorrupted
        ? { ok: false, warn: true, text: t('settings:install.step1.corrupted') }
        : { ok: false, warn: false, text: t('settings:install.step1.notInstalled') };

  const handleInstall = useCallback(async () => {
    setInstalling(true);
    setInstallMsg(null);
    try {
      await installApi.installWsl(selectedDistro, isCorrupted, isCorrupted ? distroName : '');
      setInstallMsg({ ok: true, text: t('settings:install.step1.installStarted') });
      setTimeout(onRefresh, 5000);
    } catch (err) {
      setInstallMsg({ ok: false, text: err?.response?.data?.detail ?? t('settings:install.step1.installFailed') });
    } finally {
      setInstalling(false);
    }
  }, [selectedDistro, isCorrupted, distroName, onRefresh, t]);

  return (
    <StepContainer>
      <StepHeader number={1} title={t('settings:install.step1.title')} status={statusProps} />

      {!isInstalled || isCorrupted ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div>
            <Label style={{ display: 'block', marginBottom: 4, fontSize: 11, color: colors.textSecondary }}>
              {t('settings:install.step1.distributionLabel')}
            </Label>
            <select
              value={selectedDistro}
              onChange={(e) => setSelectedDistro(e.target.value)}
              title={t('settings:install.step1.distributionTooltip')}
              style={{
                background: colors.background ?? '#282a36',
                color: colors.text,
                border: `1px solid ${colors.border}`,
                borderRadius: 4,
                padding: '5px 10px',
                fontSize: 12,
                width: '100%',
                maxWidth: 240,
                cursor: 'pointer',
              }}
            >
              {DISTROS.map((d) => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <Button
              variant="primary"
              onClick={handleInstall}
              disabled={installing}
              title={t('settings:install.step1.installWslTooltip')}
            >
              {installing
                ? t('settings:install.step1.installing')
                : isCorrupted
                  ? t('settings:install.step1.repairWsl')
                  : t('settings:install.step1.installWsl')}
            </Button>
            <Button onClick={onRefresh} disabled={loading || installing} title={t('settings:install.step1.refreshStatusTooltip')}>
              {t('settings:install.step1.refreshStatus')}
            </Button>
          </div>

          {installMsg && (
            <span style={{ fontSize: 11, color: installMsg.ok ? colors.green : colors.red }}>
              {installMsg.text}
            </span>
          )}
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {wslStatus?.wsl_version === 1 && (
            <div style={{
              fontSize: 11,
              color: colors.red,
              background: alpha(colors.red, 8),
              border: `1px solid ${alpha(colors.red, 20)}`,
              borderRadius: 4,
              padding: '6px 10px',
              marginBottom: 6,
            }}>
              {t('settings:install.step1.wsl1Detected')}{' '}
              <code style={{ color: colors.accent }}>wsl --set-version {distroName} 2</code>
            </div>
          )}
          {!hasUser && (
            <div style={{
              fontSize: 11,
              color: colors.yellow,
              background: alpha(colors.yellow, 8),
              border: `1px solid ${alpha(colors.yellow, 20)}`,
              borderRadius: 4,
              padding: '6px 10px',
            }}>
              {t('settings:install.step1.noUserDetected')}
            </div>
          )}
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <Button onClick={onRefresh} disabled={loading} title={t('settings:install.step1.refreshStatusTooltip')}>
              {loading ? t('settings:install.step1.checking') : t('settings:install.step1.refreshStatus')}
            </Button>
          </div>
        </div>
      )}
    </StepContainer>
  );
}

// ---------------------------------------------------------------------------
// Step 2 — User Setup
// ---------------------------------------------------------------------------

function Step2User({ wslStatus, onRefresh }) {
  const { t } = useTranslation(['settings', 'common']);
  const isInstalled = wslStatus?.installed === true;
  const isCorrupted = wslStatus?.corrupted === true;
  const hasUser = wslStatus?.has_user === true;
  const existingUser = wslStatus?.username ?? '';

  const stepAvailable = isInstalled && !isCorrupted;
  const needsUser = stepAvailable && !hasUser;

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [creating, setCreating] = useState(false);
  const [createMsg, setCreateMsg] = useState(null);

  const [showReset, setShowReset] = useState(false);
  const [resetPw, setResetPw] = useState('');
  const [resetConfirm, setResetConfirm] = useState('');
  const [resetting, setResetting] = useState(false);
  const [resetMsg, setResetMsg] = useState(null);

  const passwordsMatch = password === confirm && password.length > 0;
  const resetMatch = resetPw === resetConfirm && resetPw.length > 0;

  const statusProps = !stepAvailable
    ? null
    : hasUser
      ? { ok: true, warn: false, text: t('settings:install.step2.userConfigured', { name: existingUser }) }
      : { ok: false, warn: true, text: t('settings:install.step2.noUser') };

  const handleCreate = useCallback(async () => {
    setCreating(true);
    setCreateMsg(null);
    try {
      await installApi.createUser(username, password);
      setCreateMsg({ ok: true, text: t('settings:install.step2.userCreated') });
      setUsername('');
      setPassword('');
      setConfirm('');
      setTimeout(onRefresh, 1500);
    } catch (err) {
      setCreateMsg({ ok: false, text: err?.response?.data?.detail ?? t('settings:install.step2.createFailed') });
    } finally {
      setCreating(false);
    }
  }, [username, password, onRefresh, t]);

  const handleReset = useCallback(async () => {
    setResetting(true);
    setResetMsg(null);
    try {
      await installApi.resetPassword(existingUser, resetPw);
      setResetMsg({ ok: true, text: t('settings:install.step2.passwordReset') });
      setResetPw('');
      setResetConfirm('');
      setShowReset(false);
    } catch (err) {
      setResetMsg({ ok: false, text: err?.response?.data?.detail ?? t('settings:install.step2.resetFailed') });
    } finally {
      setResetting(false);
    }
  }, [existingUser, resetPw, t]);

  return (
    <StepContainer dimmed={!stepAvailable}>
      <StepHeader number={2} title={t('settings:install.step2.title')} status={statusProps} />

      {stepAvailable && needsUser && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <FieldRow label={t('settings:install.step2.usernameLabel')}>
            <Input
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder={t('settings:install.step2.usernamePlaceholder')}
              title={t('settings:install.step2.usernameTooltip')}
            />
          </FieldRow>

          <FieldRow label={t('settings:install.step2.passwordLabel')}>
            <Input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={t('settings:install.step2.passwordPlaceholder')}
              title={t('settings:install.step2.passwordTooltip')}
            />
          </FieldRow>

          <FieldRow label={t('settings:install.step2.confirmLabel')}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Input
                type="password"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                placeholder={t('settings:install.step2.confirmPlaceholder')}
                title={t('settings:hoverTips.confirmPassword')}
              />
              {confirm.length > 0 && (
                <span style={{ fontSize: 11, color: passwordsMatch ? colors.green : colors.red, whiteSpace: 'nowrap' }}>
                  {passwordsMatch ? t('settings:install.step2.matches') : t('settings:install.step2.noMatch')}
                </span>
              )}
            </div>
          </FieldRow>

          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <Button
              variant="primary"
              onClick={handleCreate}
              disabled={creating || !username || !passwordsMatch}
              title={t('settings:install.step2.createUserTooltip')}
            >
              {creating ? t('settings:install.step2.creating') : t('settings:install.step2.createUser')}
            </Button>
          </div>

          {createMsg && (
            <span style={{ fontSize: 11, color: createMsg.ok ? colors.green : colors.red }}>
              {createMsg.text}
            </span>
          )}
        </div>
      )}

      {stepAvailable && hasUser && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ fontSize: 12, color: colors.text }}>
            {t('settings:install.step2.userConfiguredPrefix')} <code style={{ color: colors.accent }}>{existingUser}</code> {t('settings:install.step2.userConfiguredSuffix')}{' '}
            <button
              type="button"
              onClick={() => setShowReset((v) => !v)}
              title={t('settings:install.step2.resetPasswordToggleTooltip')}
              style={{
                background: 'none',
                border: 'none',
                color: colors.cyan ?? colors.accent,
                cursor: 'pointer',
                fontSize: 12,
                textDecoration: 'underline',
                padding: 0,
              }}
            >
              {showReset ? t('common:cancel') : t('settings:install.step2.resetPassword')}
            </button>
          </div>

          {showReset && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 4 }}>
              <FieldRow label={t('settings:install.step2.newPasswordLabel')}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Input
                    type="password"
                    value={resetPw}
                    onChange={(e) => setResetPw(e.target.value)}
                    placeholder={t('settings:install.step2.newPasswordPlaceholder')}
                    title={t('settings:hoverTips.newPassword')}
                  />
                </div>
              </FieldRow>
              <FieldRow label={t('settings:install.step2.confirmLabel')}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Input
                    type="password"
                    value={resetConfirm}
                    onChange={(e) => setResetConfirm(e.target.value)}
                    placeholder={t('settings:install.step2.confirmPlaceholder')}
                    title={t('settings:hoverTips.confirmNewPassword')}
                  />
                  {resetConfirm.length > 0 && (
                    <span style={{ fontSize: 11, color: resetMatch ? colors.green : colors.red, whiteSpace: 'nowrap' }}>
                      {resetMatch ? t('settings:install.step2.matches') : t('settings:install.step2.noMatch')}
                    </span>
                  )}
                </div>
              </FieldRow>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <Button
                  variant="primary"
                  onClick={handleReset}
                  disabled={resetting || !resetMatch}
                  title={t('settings:install.step2.resetButtonTooltip')}
                >
                  {resetting ? t('settings:install.step2.resetting') : t('settings:install.step2.resetButton')}
                </Button>
              </div>
              {resetMsg && (
                <span style={{ fontSize: 11, color: resetMsg.ok ? colors.green : colors.red }}>
                  {resetMsg.text}
                </span>
              )}
            </div>
          )}
        </div>
      )}
    </StepContainer>
  );
}

// ---------------------------------------------------------------------------
// Step 3 — EMsoft Install
// ---------------------------------------------------------------------------

function Step3Emsoft({ wslStatus }) {
  const { t } = useTranslation('settings');
  const isWindows = !wslStatus?.platform || wslStatus.platform.os === 'windows';
  // On Windows: need WSL installed + user created. On Linux/Mac: always ready.
  const isReady = isWindows
    ? (wslStatus?.installed === true && !wslStatus?.corrupted && wslStatus?.has_user === true)
    : true;

  const [password, setPassword] = useState('');
  const [installing, setInstalling] = useState(false);
  const [logs, setLogs] = useState([]);
  const [finalStatus, setFinalStatus] = useState(null); // null | { ok, text }
  const wsRef = useRef(null);
  const logEndRef = useRef(null);
  const installingRef = useRef(false);
  const finalStatusRef = useRef(null);

  // Auto-scroll log to bottom
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // Cleanup WS on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const handleInstall = useCallback(async () => {
    if (!password) return;

    setInstalling(true);
    installingRef.current = true;
    setLogs([]);
    setFinalStatus(null);
    finalStatusRef.current = null;

    // Validate password first
    try {
      await installApi.validatePassword(password);
    } catch (err) {
      setFinalStatus({ ok: false, text: err?.response?.data?.detail ?? t('settings:install.step3.passwordValidationFailed') });
      setInstalling(false);
      return;
    }

    // Open WebSocket for streaming install
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host || 'localhost:8000';
    const wsUrl = `${protocol}//${host}/api/install/ws/install-emsoft`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify({ password }));
    };

    ws.onmessage = (event) => {
      let data;
      try {
        data = JSON.parse(event.data);
      } catch {
        setLogs((prev) => [...prev, String(event.data)]);
        return;
      }

      if (data.type === 'log' && data.line != null) {
        setLogs((prev) => [...prev, data.line]);
      } else if (data.type === 'done') {
        const status = {
          ok: data.success === true,
          text: data.message ?? (data.success ? t('settings:install.step3.installComplete') : t('settings:install.step3.installFailed')),
        };
        finalStatusRef.current = status;
        setFinalStatus(status);
        installingRef.current = false;
        setInstalling(false);
        ws.close();
      }
    };

    ws.onerror = () => {
      finalStatusRef.current = { ok: false, text: t('settings:install.step3.wsError') };
      setFinalStatus(finalStatusRef.current);
      installingRef.current = false;
      setInstalling(false);
    };

    ws.onclose = (evt) => {
      if (installingRef.current && !finalStatusRef.current) {
        // Unexpected close
        setFinalStatus({ ok: false, text: t('settings:install.step3.connectionClosed', { code: evt.code }) });
        setInstalling(false);
        installingRef.current = false;
      }
    };
  }, [password, t]);

  const statusProps = isReady
    ? finalStatus
      ? { ok: finalStatus.ok, warn: false, text: finalStatus.ok ? t('settings:install.step3.installed') : t('settings:install.step3.failed') }
      : null
    : null;

  return (
    <StepContainer dimmed={!isReady}>
      <StepHeader number={3} title={t('settings:install.step3.title')} status={statusProps} />

      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <div>
          <Label style={{ display: 'block', marginBottom: 4, fontSize: 11, color: colors.textSecondary }}>
            {t('settings:install.step3.sudoPasswordLabel')}
          </Label>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <Input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={t('settings:install.step3.sudoPasswordPlaceholder')}
              title={t('settings:install.step3.sudoPasswordTooltip')}
              style={{ maxWidth: 260 }}
            />
            <Button
              variant="primary"
              onClick={handleInstall}
              disabled={installing || !password}
              title={t('settings:install.step3.runInWslTooltip')}
            >
              {installing ? t('settings:install.step3.installing') : t('settings:install.step3.runInWsl')}
            </Button>
          </div>
        </div>

        {(logs.length > 0 || installing) && (
          <div style={{
            background: '#1a1b23',
            border: `1px solid ${alpha(colors.border, 50)}`,
            borderRadius: 6,
            padding: '10px 12px',
            maxHeight: 300,
            overflowY: 'auto',
            fontFamily: 'monospace',
            fontSize: 11,
            lineHeight: 1.5,
            color: '#f8f8f2',
          }}>
            {logs.length === 0 && installing && (
              <span style={{ color: colors.textSecondary }}>{t('settings:install.step3.starting')}</span>
            )}
            {logs.map((line, i) => (
              <div key={i} style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                {line}
              </div>
            ))}
            <div ref={logEndRef} />
          </div>
        )}

        {finalStatus && (
          <div style={{
            fontSize: 12,
            color: finalStatus.ok ? colors.green : colors.red,
            background: alpha(finalStatus.ok ? colors.green : colors.red, 8),
            border: `1px solid ${alpha(finalStatus.ok ? colors.green : colors.red, 20)}`,
            borderRadius: 4,
            padding: '6px 10px',
          }}>
            {finalStatus.text}
          </div>
        )}
      </div>
    </StepContainer>
  );
}

// ---------------------------------------------------------------------------
// Shared field row helper
// ---------------------------------------------------------------------------

function FieldRow({ label, children }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <Label style={{ fontSize: 11, color: colors.textSecondary }}>{label}</Label>
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main export
// ---------------------------------------------------------------------------

export default function InstallWizardSection() {
  const { t } = useTranslation('settings');
  const [wslStatus, setWslStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [fetchError, setFetchError] = useState(null);

  const fetchStatus = useCallback(async () => {
    setLoading(true);
    setFetchError(null);
    try {
      const res = await installApi.wslStatus();
      setWslStatus(res.data);
    } catch (err) {
      setFetchError(err?.response?.data?.detail ?? err.message ?? t('settings:install.fetchError'));
    } finally {
      setLoading(false);
    }
  }, [t]);

  useEffect(() => { fetchStatus(); }, [fetchStatus]);

  return (
    <GroupBox title={t('settings:install.title')}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {fetchError && (
          <div style={{
            color: colors.red,
            fontSize: 11,
            padding: '6px 10px',
            background: alpha(colors.red, 10),
            borderRadius: 4,
            border: `1px solid ${alpha(colors.red, 30)}`,
          }}>
            {fetchError}
          </div>
        )}

        <Step0PreFlight
          wslStatus={wslStatus}
          loading={loading}
        />
        {/* WSL steps only shown on Windows */}
        {(!wslStatus?.platform || wslStatus.platform.os === 'windows') && (
          <>
            <Step1Wsl
              wslStatus={wslStatus}
              loading={loading}
              onRefresh={fetchStatus}
            />
            <Step2User
              wslStatus={wslStatus}
              onRefresh={fetchStatus}
            />
          </>
        )}
        {/* On Linux/macOS, show a simpler info message instead of WSL steps */}
        {wslStatus?.platform && wslStatus.platform.os !== 'windows' && (
          <StepContainer>
            <StepHeader number={1} title={t('settings:install.direct.title')} status={{ ok: true, warn: false, text: t('settings:install.direct.noWslNeeded') }} />
            <div style={{ fontSize: 12, color: colors.textSecondary, lineHeight: 1.6 }}>
              {wslStatus.platform.os === 'macos'
                ? t('settings:install.direct.infoMacos')
                : t('settings:install.direct.infoLinux')}
              {wslStatus.emsoft_installed && (
                <div style={{
                  marginTop: 8,
                  color: colors.green,
                  fontSize: 11,
                }}>
                  {t('settings:install.direct.alreadyInstalled')}
                </div>
              )}
            </div>
          </StepContainer>
        )}
        <Step3Emsoft
          wslStatus={wslStatus}
        />
      </div>
    </GroupBox>
  );
}
