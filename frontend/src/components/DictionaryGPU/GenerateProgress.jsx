/**
 * GenerateProgress — progress bar + status message for the current generation.
 *
 * Pure display component. Parent owns the polling loop and feeds us the latest
 * `{ status, progress, message, result: { output_path, n_orientations }, error }`
 * blob. The backend only populates `result` when status == 'done'.
 * When progress is null we render nothing — keeps the page compact.
 */

import { useTranslation, Trans } from 'react-i18next';
import { colors, alpha, spacing, ProgressBar } from '../../theme/components';

export default function GenerateProgress({ progress }) {
  const { t } = useTranslation('dictionarygpu');
  if (!progress) return null;

  const { status, progress: pct, message, result, error } = progress;
  const n_patterns = result?.n_orientations;
  const output_path = result?.output_path;
  // Throughput of the finished run. Kept on screen (and in the sidecar next to
  // the .h5) so a CPU and a GPU run can be compared and the figure written down.
  const rate = result?.patterns_per_second;
  const elapsed = result?.elapsed_s;
  const device = result?.device;
  const isError = status === 'error';
  const isDone = status === 'done';
  const isRunning = status === 'running' || status === 'pending';

  const tone = isError
    ? { fg: colors.red, bg: alpha(colors.red, 10), border: alpha(colors.red, 33) }
    : isDone
      ? { fg: colors.green, bg: alpha(colors.green, 7), border: alpha(colors.green, 33) }
      : { fg: colors.textSecondary, bg: alpha(colors.accent, 5), border: alpha(colors.accent, 27) };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: spacing.innerSpacing,
        padding: '10px 12px',
        marginBottom: spacing.outerSpacing,
        background: tone.bg,
        border: `1px solid ${tone.border}`,
        borderRadius: 6,
      }}
    >
      <ProgressBar
        value={Math.max(0, Math.min(100, (pct ?? 0) * 100))}
        max={100}
        label={t('progress.statusLabel', { status: status || t('progress.statusIdle') })}
        color={isError ? colors.red : isDone ? colors.green : colors.accent}
      />
      {(message || error) && (
        <div style={{ fontSize: '9pt', color: tone.fg, wordBreak: 'break-word' }}>
          {error || message}
        </div>
      )}
      {isDone && (n_patterns != null || output_path) && (
        <div style={{ fontSize: '9pt', color: colors.textSecondary }}>
          {n_patterns != null && (
            <span>
              <Trans
                i18nKey="progress.generatedPatterns"
                ns="dictionarygpu"
                values={{ count: n_patterns }}
                components={[<b style={{ color: colors.text }} />]}
              />
            </span>
          )}
          {rate > 0 && (
            <div data-throughput style={{ marginTop: 2 }}>
              <b style={{ color: colors.text }}>
                {t('progress.throughput', {
                  rate: Math.round(rate).toLocaleString(),
                  seconds: Number(elapsed).toFixed(1),
                })}
              </b>
              {device && (
                <span style={{ opacity: 0.75 }}>{t('progress.onDevice', { device })}</span>
              )}
            </div>
          )}
          {output_path && (
            <div
              style={{
                fontFamily: "'Courier New', monospace",
                fontSize: '8pt',
                marginTop: 2,
                wordBreak: 'break-all',
                opacity: 0.85,
              }}
            >
              {output_path}
            </div>
          )}
        </div>
      )}
      {isRunning && (
        <div style={{ fontSize: '8pt', color: colors.textSecondary, opacity: 0.7 }}>
          {t('progress.polling')}
        </div>
      )}
    </div>
  );
}
