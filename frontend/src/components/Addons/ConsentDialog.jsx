/**
 * What a user is agreeing to before an add-on's code runs in this process.
 *
 * Shown once per add-on, before the FIRST enable. Not a formality: an add-on
 * is third-party code that runs with the user's permissions, and the only
 * moment anyone can weigh that is before it starts. The dialog therefore
 * states what the runtime actually allows, not a softened version of it.
 *
 * Every field comes from the manifest and is rendered VERBATIM — including
 * the display name, which is the author's text.
 */
import { useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { Button, colors, spacing } from '../../theme/components';

export default function ConsentDialog({ addon, onAccept, onCancel }) {
  const { t } = useTranslation('addons');
  const cancelRef = useRef(null);

  // Declaring aria-modal and then doing nothing about focus is a claim the
  // dialog does not honour: measured, the focus stayed on BODY and Escape did
  // nothing. Follows ProblemReportDialog, the house precedent.
  //
  // Focus lands on CANCEL, not on the accept button: this dialog exists to
  // make someone decide whether to run a stranger's code, and a Return key
  // pressed out of habit must not be the decision.
  useEffect(() => {
    if (!addon) return undefined;
    cancelRef.current?.focus();
    const onKey = (e) => { if (e.key === 'Escape') onCancel?.(); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [addon, onCancel]);

  if (!addon) return null;

  const authors = (addon.authors || []).join(', ');

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={t('consent.title', { name: addon.display_name })}
      style={{
        position: 'fixed', inset: 0, zIndex: 1000,
        background: 'rgba(0,0,0,0.55)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
    >
      <div style={{
        background: colors.panel || colors.bg,
        border: `1px solid ${colors.border}`,
        borderRadius: 6,
        padding: spacing.panelPadding || 16,
        maxWidth: 560,
        color: colors.text,
      }}>
        <h3 style={{ margin: '0 0 8px' }}>
          {t('consent.title', { name: addon.display_name })}
        </h3>
        <p style={{ color: colors.textSecondary, fontSize: '10pt' }}>
          {t('consent.body')}
        </p>
        <dl style={{ fontSize: '10pt', margin: '12px 0' }}>
          <dt style={{ color: colors.textSecondary }}>{t('list.version')}</dt>
          <dd style={{ margin: '0 0 6px' }}>{addon.version}</dd>
          <dt style={{ color: colors.textSecondary }}>{t('list.authors')}</dt>
          <dd style={{ margin: '0 0 6px' }}>{authors}</dd>
          <dt style={{ color: colors.textSecondary }}>DOI</dt>
          {/* An empty DOI and a missing one look identical in a snapshot, and
              "no DOI" is a fact a citing user needs, so it is written out. */}
          <dd style={{ margin: '0 0 6px' }}>{addon.doi || t('list.noDoi')}</dd>
          <dt style={{ color: colors.textSecondary }}>{t('list.source')}</dt>
          <dd style={{ margin: 0, wordBreak: 'break-all' }}>
            {addon.source_path}
          </dd>
        </dl>
        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
          <Button ref={cancelRef} onClick={onCancel}>
            {t('consent.cancel')}
          </Button>
          <Button variant="primary" onClick={onAccept}>
            {t('consent.accept')}
          </Button>
        </div>
      </div>
    </div>
  );
}
