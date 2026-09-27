/**
 * Removing Orienta's data, where the operating system cannot ask.
 *
 * Windows asks two questions while uninstalling: remove the downloaded Python
 * and program files, and — only then — delete the crystal library. macOS and
 * Linux have nowhere to ask them: the user drags the app to the Trash or
 * deletes an AppImage, and two to eight gigabytes stay behind with the
 * library inside. So the application asks, in the same order, with the same
 * preselections, and shows what it found before it asks anything at all.
 *
 * Deliberately not offered on Windows: there the uninstaller owns this, and a
 * second way to delete the same gigabytes is a second way to get it wrong.
 */
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, spacing, GroupBox, Label } from '../../theme/components';

const api = () => (typeof window !== 'undefined' ? window.electronAPI : null);

/** Windows has its own uninstaller; a browser session has no folder to remove. */
export function shouldOffer(platform, hasApi) {
  if (!hasApi) return false;
  return platform === 'darwin' || platform === 'linux';
}

export default function RemoveDataSection({ platform }) {
  const { t } = useTranslation('settings');
  const [plan, setPlan] = useState(null);
  const [stage, setStage] = useState('idle');   // idle | confirm | library | busy | done
  const [result, setResult] = useState(null);

  // From the preload, not from `process`: the renderer has no `process`, so
  // the old fallback made this section render nowhere at all — on the two
  // platforms it exists for.
  const running = platform ?? api()?.platform ?? '';
  const offered = shouldOffer(running, Boolean(api()));

  useEffect(() => {
    if (!offered) return;
    let alive = true;
    api().planDataRemoval()
      .then((p) => { if (alive) setPlan(p); })
      .catch(() => { if (alive) setPlan({ ok: false, reason: 'error' }); });
    return () => { alive = false; };
  }, [offered]);

  if (!offered) return null;

  const start = () => setStage('confirm');

  const confirmFiles = () => {
    // The library question is only asked when there is a library, and only
    // after yes to the first — exactly as the Windows uninstaller does it.
    if (plan?.library) setStage('library');
    else run(false);
  };

  const run = async (removeLibrary) => {
    setStage('busy');
    try {
      setResult(await api().removeData({ removeLibrary }));
    } catch (err) {
      setResult({ ok: false, reason: 'error', error: String(err?.message || err) });
    }
    setStage('done');
  };

  return (
    <GroupBox title={t('settings:removeData.title')}>
      <div style={{ fontSize: 12, color: colors.textSecondary, lineHeight: 1.6, marginBottom: spacing.sm }}>
        {t('settings:removeData.intro')}
      </div>

      {plan && !plan.ok && (
        <div style={{ fontSize: 12, color: colors.yellow }}>
          {t(`settings:removeData.refused.${plan.reason}`, { defaultValue: t('settings:removeData.refused.notOurs') })}
        </div>
      )}

      {plan?.ok && stage === 'idle' && (
        <>
          <Label>{t('settings:removeData.folder')}</Label>
          <div style={{ fontFamily: 'monospace', fontSize: 11, marginBottom: spacing.sm }}>{plan.home}</div>
          {plan.foreign?.length > 0 && (
            <div style={{ fontSize: 11, color: colors.textSecondary, marginBottom: spacing.sm }}>
              {t('settings:removeData.keepsForeign', { count: plan.foreign.length })}
            </div>
          )}
          <button type="button" onClick={start} style={buttonStyle}>
            {t('settings:removeData.button')}
          </button>
        </>
      )}

      {stage === 'confirm' && (
        <Question
          text={t('settings:removeData.askFiles')}
          path={plan?.home}
          confirmLabel={t('settings:removeData.yes')}
          cancelLabel={t('settings:removeData.no')}
          onConfirm={confirmFiles}
          onCancel={() => setStage('idle')}
          defaultYes
        />
      )}

      {stage === 'library' && (
        <Question
          text={t('settings:removeData.askLibrary')}
          path={plan?.library}
          confirmLabel={t('settings:removeData.yesLibrary')}
          cancelLabel={t('settings:removeData.keepLibrary')}
          onConfirm={() => run(true)}
          onCancel={() => run(false)}
        />
      )}

      {stage === 'busy' && <div style={{ fontSize: 12 }}>{t('settings:removeData.busy')}</div>}

      {stage === 'done' && (
        <div style={{ fontSize: 12, color: doneColour(result) }}>
          {!result?.ok && t(`settings:removeData.refused.${result?.reason || 'error'}`,
            { defaultValue: t('settings:removeData.failed') })}
          {result?.ok && result.finished && t('settings:removeData.done')}
          {result?.ok && !result.finished && t('settings:removeData.partly')}
          {/* Each of these is a thing the Windows uninstaller says out loud,
              and each is a case where a plain "done" would be untrue. */}
          {result?.failed?.length > 0 && (
            <div style={{ color: colors.yellow, marginTop: 4 }}>
              {t('settings:removeData.someLeft', { count: result.failed.length })}
            </div>
          )}
          {result?.libraryWasLink && (
            <div style={{ color: colors.yellow, marginTop: 4 }}>
              {t('settings:removeData.libraryWasLink')}
            </div>
          )}
          {result?.ok && result.kept?.length > 0 && (
            <div style={{ color: colors.textSecondary, marginTop: 4 }}>
              {t('settings:removeData.keptList', { list: result.kept.join(', ') })}
            </div>
          )}
        </div>
      )}
    </GroupBox>
  );
}

function doneColour(result) {
  if (!result?.ok) return colors.red;
  if (!result.finished || result.failed?.length) return colors.yellow;
  return colors.green;
}

const buttonStyle = {
  background: colors.bgTertiary,
  border: `1px solid ${colors.border}`,
  color: colors.text,
  borderRadius: 4,
  padding: '6px 12px',
  fontSize: 12,
  cursor: 'pointer',
};

function Question({ text, path, confirmLabel, cancelLabel, onConfirm, onCancel, defaultYes = false }) {
  return (
    <div>
      <div style={{ fontSize: 12, marginBottom: spacing.sm }}>{text}</div>
      {/* The folder is in front of the user while they answer, as it is in
          the Windows uninstaller: approving 2 to 8 GB without seeing which
          folder is not a decision. */}
      {path && (
        <div style={{ fontFamily: 'monospace', fontSize: 11, color: colors.textSecondary, marginBottom: spacing.sm }}>
          {path}
        </div>
      )}
      <div style={{ display: 'flex', gap: 8 }}>
        {/* The preselected answer is the first button and carries the focus
            ring, so Enter does what the Windows uninstaller's Enter does. */}
        {defaultYes ? (
          <>
            <button type="button" onClick={onConfirm} style={buttonStyle} autoFocus>{confirmLabel}</button>
            <button type="button" onClick={onCancel} style={buttonStyle}>{cancelLabel}</button>
          </>
        ) : (
          <>
            <button type="button" onClick={onCancel} style={buttonStyle} autoFocus>{cancelLabel}</button>
            <button type="button" onClick={onConfirm} style={buttonStyle}>{confirmLabel}</button>
          </>
        )}
      </div>
    </div>
  );
}
