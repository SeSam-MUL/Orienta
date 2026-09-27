/**
 * StatusBar — matches PyQt5 main_hub.py status bar + status_indicator.py exactly.
 *
 * PyQt5 spec:
 *   Left side:  File loaded indicator + filename + grid shape + pattern count
 *   Center:     Active task count
 *   Right side: System status dots (EMsoft, WSL, EMSphinx, GPU) + Backend status
 */

import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, layout } from '../../theme/tokens';
import useDataStore from '../../stores/useDataStore';
import useProgressStore from '../../stores/useProgressStore';
import { simApi } from '../../services/api';

function StatusDot({ color, label }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }} title={label}>
      <span style={{
        width: 8, height: 8, borderRadius: '50%',
        background: color, display: 'inline-block', flexShrink: 0,
        transition: 'background 0.5s ease',
      }} aria-hidden="true" />
      <span style={{ fontSize: '8pt', color: colors.textSecondary, whiteSpace: 'nowrap' }}>
        {label}
      </span>
    </span>
  );
}

/**
 * The three states App.jsx puts on the wire, in the reader's language.
 *
 * The namespace is written out. Sidebar calls this with a `t` bound to `nav`
 * (useTranslation(['nav', 'shell'])), and react-i18next binds to the FIRST
 * namespace, so a bare 'sidebar.connected' resolved against nav, missed, and
 * printed the key into the status strip in every language.
 */
export function backendStateLabel(t, status) {
  const key = { connected: 'connected', disconnected: 'disconnected', checking: 'checking' }[status];
  return key ? t(`shell:sidebar.${key}`) : status;
}

export default function StatusBar({ backendStatus, onNavigate }) {
  const { t } = useTranslation('shell');
  const { isFileOpen, filePath, gridShape, patternCount, detector, stepSize, edsElements } = useDataStore();
  const { tasks } = useProgressStore();
  const [sysStatus, setSysStatus] = useState(null);

  const activeTasks = Object.values(tasks).filter(t => t.status === 'running');

  // Fetch system status on mount and whenever backend reconnects
  useEffect(() => {
    if (backendStatus !== 'connected') return;
    simApi.systemStatus()
      .then(res => setSysStatus(res.data))
      .catch(() => setSysStatus(null));
  }, [backendStatus]);

  // Derive system dots matching status_indicator.py COLORS
  const dots = [];
  if (sysStatus && !sysStatus.error) {
    // EMsoft
    dots.push({
      color: sysStatus.emsoft_available ? colors.green : colors.red,
      label: sysStatus.emsoft_available ? t('shell:status.emsoftOk') : t('shell:status.emsoftMissing'),
    });
    // WSL — a Windows feature. On macOS and Linux the backend reports
    // wsl_distro = "native-linux", and the status line used to show
    // "WSL: native-linux" on a Mac, which names a thing that cannot exist
    // there. Off Windows the dot is simply not shown.
    const wslOk = sysStatus.wsl_installed || sysStatus.wsl_available;
    const onWindows = (sysStatus.platform_os || 'windows') === 'windows';
    if (onWindows) {
      dots.push({
        color: wslOk ? colors.green : colors.red,
        label: wslOk
          ? (sysStatus.wsl_distro ? t('shell:status.wslDistro', { distro: sysStatus.wsl_distro }) : t('shell:status.wslOk'))
          : t('shell:status.wslMissing'),
      });
    }
    // EMSphinx
    const sphinxOk = sysStatus.emsphinx_available || (sysStatus.emsoft_available && wslOk);
    dots.push({
      color: sphinxOk ? colors.green : colors.red,
      label: sphinxOk ? t('shell:status.sphinxOk') : t('shell:status.sphinxMissing'),
    });
    // GPU
    if (sysStatus.opencl_available && sysStatus.gpu_name) {
      const name = sysStatus.gpu_name.length > 20 ? sysStatus.gpu_name.slice(0, 18) + '\u2026' : sysStatus.gpu_name;
      dots.push({ color: colors.green, label: name });
    } else if (sysStatus.opencl_available) {
      dots.push({ color: colors.yellow, label: t('shell:status.gpuOpencl') });
    } else {
      dots.push({ color: colors.yellow, label: t('shell:status.gpuNone') });
    }
  }

  // Backend dot
  const backendDot = {
    color: backendStatus === 'connected' ? colors.green : colors.red,
    label: t('shell:status.backend', { state: backendStateLabel(t, backendStatus) }),
  };

  const Separator = () => (
    <span style={{ width: 1, height: 14, background: colors.border, flexShrink: 0, opacity: 0.5 }} />
  );

  return (
    <div role="status" aria-label={t('hoverTips.appStatus')} style={{
      height: layout.statusBarHeight,
      minHeight: layout.statusBarHeight,
      background: colors.bgSecondary,
      borderTop: `1px solid ${colors.border}`,
      display: 'flex',
      alignItems: 'center',
      padding: '0 12px',
      gap: 10,
      fontSize: '8.5pt',
      color: colors.textSecondary,
      flexShrink: 0,
      transition: 'background 0.3s ease, border-color 0.3s ease, color 0.3s ease',
    }}>
      {/* Left: file info */}
      {isFileOpen ? (
        <>
          <StatusDot color={colors.green} label="" />
          <span title={filePath || undefined} style={{
            maxWidth: 220, overflow: 'hidden',
            textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            fontWeight: 500,
            color: colors.text,
          }}>
            {filePath?.split(/[\\/]/).pop()}
          </span>
          {gridShape?.[0] > 0 && (
            <>
              <Separator />
              {/* The escape must sit inside an expression: text between JSX
                  tags is literal, so a bare \u00d7 renders as those 6 chars. */}
              <span>{gridShape[0]}{'\u00d7'}{gridShape[1]}</span>
            </>
          )}
          {detector?.has_detector && detector.pc?.length === 3 && (
            <>
              <Separator />
              <span>PC ({detector.pc[0]?.toFixed(3)}, {detector.pc[1]?.toFixed(3)}, {detector.pc[2]?.toFixed(3)})</span>
            </>
          )}
          {/* A step of exactly 1.0 used to be hidden here, on the assumption
              that it meant "no step size in the file". It does not: the
              backend answers `null` for that (see `_extract_step_size`), and
              1.0 \u00b5m is an ordinary scan step \u2014 both EDAX .osc files in
              Test_data report exactly 1.0. Hiding it took the one number that
              scales every figure off the screen for those scans. */}
          {stepSize?.x > 0 && (
            <>
              <Separator />
              <span>{stepSize.x.toFixed(2)} {stepSize.units || '\u00b5m'}</span>
            </>
          )}
          {edsElements?.length > 0 && (
            <>
              <Separator />
              <span title={edsElements.join(', ')}>{t('shell:status.edsElements', { count: edsElements.length })}</span>
            </>
          )}
        </>
      ) : (
        <span style={{ fontStyle: 'italic', opacity: 0.6 }}>{'\u2014'} {t('shell:status.noFile')}</span>
      )}

      {/* Center: active tasks */}
      {activeTasks.length > 0 && (
        <>
          <Separator />
          <span style={{ color: colors.orange, fontWeight: 500, display: 'inline-flex', alignItems: 'center', gap: 4 }}>
            <span style={{ animation: 'spin 2s linear infinite', display: 'inline-block' }}>{'\u27f3'}</span>
            {t('shell:status.tasks', { count: activeTasks.length })}
          </span>
        </>
      )}

      {/* Right: system status (clickable → Settings) + backend */}
      <div
        style={{
          marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 10,
          cursor: onNavigate ? 'pointer' : 'default',
          borderRadius: 4, padding: '2px 6px', margin: '-2px -6px -2px auto',
          transition: 'background 0.15s',
        }}
        onClick={() => onNavigate && onNavigate('settings')}
        onMouseEnter={e => { if (onNavigate) e.currentTarget.style.background = 'var(--bg-hover, rgba(255,255,255,0.05))'; }}
        onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; }}
        title={t('hoverTips.openSettings')}
      >
        {dots.map((dot, i) => (
          <StatusDot key={i} color={dot.color} label={dot.label} />
        ))}
        <Separator />
        <StatusDot color={backendDot.color} label={backendDot.label} />
      </div>
    </div>
  );
}
