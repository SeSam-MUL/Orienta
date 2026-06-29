/**
 * LoadProgressModal — stage breadcrumb + ticking elapsed time for EBSD load.
 *
 * Renders nothing when isOpen is false. When open, shows the four named
 * stages (reading metadata → building lazy signal → detecting features →
 * finalising) with the current one marked aria-current="step". Below the
 * breadcrumb: human-readable message + wall-clock elapsed (e.g. "2.3 s").
 *
 * Stage names are driven by the backend `stage` field; the elapsed time is
 * driven by the backend `elapsed_seconds` field (which is wall-clock since
 * the load started, computed server-side on each progress read). We do NOT
 * estimate or extrapolate — that would be a lie. If a stage hangs, the
 * stage name stays put and the elapsed counter keeps ticking; the user
 * sees exactly where it is stuck.
 *
 * Error state: when stage === 'error', shows the error message in red and
 * keeps the modal open until the user clicks Close. The Close button is
 * also shown on stage === 'complete'; during in-progress stages there is
 * no dismiss control, because cancelling mid-load would leave the backend
 * in an indeterminate state.
 *
 * Theme: uses `colors` from theme/components. We avoid hard-coded hex
 * values so the modal respects the active theme. The "active stage" tint
 * is derived from `colors.accent` via `alpha(...)` so it adapts when the
 * user switches theme. Likewise the "done" tint is derived from
 * `colors.green`. The slow-hint warning band uses `colors.orange` for the
 * same reason.
 *
 * A11y:
 *   - Focus moves to the Close button when we enter a terminal state
 *     (error/complete) so keyboard users can dismiss without mouse.
 *   - The progress-message div carries aria-live="polite"+aria-atomic so
 *     screen readers announce each stage transition.
 *   - Escape always dismisses the modal. During in-progress stages this
 *     does NOT cancel the backend load — it just hides the UI ("Hide"
 *     semantics). The toast notification on success/error still fires.
 *     This escape hatch was added 2026-05-27 after a session where a
 *     dead backend left the modal frozen at "Starting… 0.0 s elapsed"
 *     with no way out (the previous design assumed the load would
 *     always either complete or error within the 5-min axios timeout,
 *     which is not true when sockets stall on a crashed backend).
 */
import { useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, alpha } from '../../theme/components';

const STAGE_DEFS = [
  { id: 'reading_metadata',   labelKey: 'loadModal.stageReadingMetadata' },
  { id: 'building_signal',    labelKey: 'loadModal.stageBuildingSignal' },
  { id: 'detecting_features', labelKey: 'loadModal.stageDetectingFeatures' },
  { id: 'finalising',         labelKey: 'loadModal.stageFinalising' },
];

export default function LoadProgressModal({ isOpen, progressState, onClose }) {
  const { t } = useTranslation('ebsdviewer');
  // Hooks must run on every render — they cannot be conditional. The early
  // `if (!isOpen) return null;` therefore lives BELOW the hook declarations.
  const closeButtonRef = useRef(null);

  const stage = progressState?.stage || 'reading_metadata';
  const elapsedSec = progressState?.elapsed_seconds ?? 0;
  const message = progressState?.message || '';
  const isError = stage === 'error';
  const isComplete = stage === 'complete';
  const isTerminal = isError || isComplete;
  const errorMsg = progressState?.error || message;
  const stageIdx = STAGE_DEFS.findIndex((s) => s.id === stage);

  // A11y-1: focus the Close button on entering a terminal state. Pulls
  // focus away from whatever spawned the load so keyboard users can press
  // Enter/Space to dismiss without mouse.
  useEffect(() => {
    if (isOpen && isTerminal) {
      closeButtonRef.current?.focus();
    }
  }, [isOpen, isTerminal]);

  // A11y-3: Escape-to-close on ALL states. During in-progress stages
  // this is "Hide" semantics — the backend load continues, only the UI
  // is dismissed. Toast on success/error still fires. See the file-level
  // docstring for why we no longer guard this with isTerminal.
  useEffect(() => {
    if (!isOpen) return;
    const handler = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      style={{
        position: 'fixed', inset: 0,
        background: 'rgba(0,0,0,0.6)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        zIndex: 9999,
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={t('loadModal.ariaLabel')}
        style={{
          background: colors.bgSecondary,
          border: `1px solid ${colors.border}`,
          borderRadius: 8,
          padding: 24,
          minWidth: 420,
          maxWidth: 560,
          boxShadow: '0 8px 40px rgba(0,0,0,0.7)',
        }}
      >
        <div style={{
          fontSize: '12pt', fontWeight: 600, color: colors.text, marginBottom: 16,
        }}>
          {isError ? t('loadModal.titleFailed') : t('loadModal.titleLoading')}
        </div>

        {!isError && (
          <ul
            role="list"
            style={{
              display: 'flex', gap: 8, marginBottom: 16,
              listStyle: 'none', padding: 0, margin: '0 0 16px 0',
            }}
          >
            {STAGE_DEFS.map((s, myIdx) => {
              const isActive = s.id === stage;
              // A stage is "done" if the current stage is later in the list,
              // OR if the load has completed (all stages are done).
              const isDone = isComplete || (stageIdx >= 0 && stageIdx > myIdx);
              return (
                <li
                  key={s.id}
                  role="listitem"
                  aria-current={isActive ? 'step' : undefined}
                  style={{
                    flex: 1, padding: '8px 6px', borderRadius: 4,
                    background: isActive
                      ? alpha(colors.accent, 20)
                      : isDone
                      ? alpha(colors.green, 15)
                      : colors.bg,
                    border: `1px solid ${isActive ? colors.accent : colors.border}`,
                    fontSize: '9pt', textAlign: 'center',
                    color: isActive ? colors.text : colors.textSecondary,
                    fontWeight: isActive ? 600 : 400,
                  }}
                >
                  {t(s.labelKey)}
                </li>
              );
            })}
          </ul>
        )}

        {isError ? (
          <div
            data-testid="progress-error"
            style={{ color: colors.red, fontSize: '10pt', marginBottom: 12 }}
          >
            {errorMsg}
          </div>
        ) : (
          <div
            data-testid="progress-message"
            aria-live="polite"
            aria-atomic="true"
            style={{ color: colors.textSecondary, fontSize: '10pt', marginBottom: 8 }}
          >
            {message}
          </div>
        )}

        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        }}>
          <div style={{
            fontSize: '9pt',
            color: colors.textSecondary,
            fontVariantNumeric: 'tabular-nums',
          }}>
            {t('loadModal.elapsed', { seconds: elapsedSec.toFixed(1) })}
          </div>
          <button
            ref={closeButtonRef}
            onClick={onClose}
            aria-label={isTerminal ? t('loadModal.close') : t('loadModal.hide')}
            title={isTerminal
              ? t('loadModal.closeTooltip')
              : t('loadModal.hideTooltip')}
            style={{
              background: isError
                ? colors.red
                : isComplete
                ? colors.accent
                : 'transparent',
              color: isTerminal ? colors.textOnAccent : colors.textSecondary,
              border: isTerminal ? 'none' : `1px solid ${colors.border}`,
              borderRadius: 4,
              padding: '6px 14px',
              cursor: 'pointer',
              fontSize: '10pt',
            }}
          >
            {isTerminal ? t('loadModal.close') : t('loadModal.hide')}
          </button>
        </div>

        {!isError && elapsedSec > 30 && (
          <div
            data-testid="progress-slow-hint"
            style={{
              marginTop: 12,
              padding: 8,
              background: alpha(colors.orange, 20),
              border: `1px solid ${colors.orange}`,
              borderRadius: 4,
              color: colors.textSecondary,
              fontSize: '9pt',
            }}
          >
            {t('loadModal.slowHint')}
          </div>
        )}
      </div>
    </div>
  );
}
