import { useEffect, useId, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, alpha, Button } from '../../theme/components';
import { splitDoiOrCitation, MAX_CITATION_CHARS } from './cifWarnings';

const MAX_NAMES = 5;

/**
 * Asked before converting CIF files that carry no DOI and no reference text.
 *
 * Nothing is converted with an invented "unknown". One file: type a DOI or a
 * citation (saved into the CIF), or declare the data unpublished / own data
 * (the .xtal then says so, and is flagged as a declaration). Several files:
 * one question for all of them, declare or skip. Cancel always backs out.
 *
 * `files`: [{ name }] — the files that have no reference.
 */
export default function ReferenceDialog({
  files,
  onSubmitReference,
  onDeclareUnpublished,
  onSkip,
  onCancel,
}) {
  const { t } = useTranslation(['crystaldatabase', 'common']);
  const [text, setText] = useState('');
  const inputRef = useRef(null);
  const cancelRef = useRef(null);
  const dialogRef = useRef(null);
  const titleId = useId();
  const messageId = useId();
  const single = files.length === 1;
  const parsed = splitDoiOrCitation(text);
  const canSubmit = !!(parsed.doi || parsed.reference);

  useEffect(() => {
    const onKey = (e) => {
      // Escape during an IME composition cancels the composition, not the dialog.
      if (e.key === 'Escape') {
        if (!(e.isComposing || e.keyCode === 229)) onCancel?.();
        return;
      }
      if (e.key !== 'Tab') return;
      // Keep Tab inside the dialog: the page behind it must not take focus.
      const focusable = dialogRef.current?.querySelectorAll('input, button:not([disabled])');
      if (!focusable || focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey && (document.activeElement === first || !dialogRef.current.contains(document.activeElement))) {
        e.preventDefault(); last.focus();
      } else if (!e.shiftKey && (document.activeElement === last || !dialogRef.current.contains(document.activeElement))) {
        e.preventDefault(); first.focus();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onCancel]);

  // Give the focus back to whatever opened the dialog (the Convert button) when
  // it closes; otherwise it falls to the top of the page.
  useEffect(() => {
    const opener = document.activeElement;
    return () => {
      if (opener && typeof opener.focus === 'function' && document.contains(opener)) opener.focus();
    };
  }, []);

  // One file: the text box. Several files there is no box, so Cancel, the one
  // button that changes nothing, takes the focus.
  useEffect(() => {
    (single ? inputRef.current : cancelRef.current)?.focus();
  }, [single]);

  // A click beside the dialog closes it, unless something was typed: that
  // would throw the text away.
  const onBackdrop = () => { if (!text.trim()) onCancel?.(); };

  const submit = () => { if (canSubmit) onSubmitReference?.(parsed); };
  // Enter that confirms an IME candidate (Japanese, Chinese) is not a submit:
  // keyCode 229 is what browsers send for it when isComposing is not set.
  const onTextKeyDown = (e) => {
    if (e.key !== 'Enter' || e.nativeEvent?.isComposing || e.keyCode === 229) return;
    submit();
  };

  const names = files.slice(0, MAX_NAMES).map(f => f.name);
  const more = files.length - names.length;

  return (
    <div
      onClick={onBackdrop}
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 10000,
      }}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={messageId}
        onClick={(e) => e.stopPropagation()}
        style={{
          background: colors.bg, border: `1px solid ${colors.border}`, borderRadius: 8,
          padding: '20px 24px', width: 460, maxWidth: '92vw',
          boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
        }}
      >
        <div id={titleId} style={{ fontSize: '11pt', fontWeight: 600, color: colors.text, marginBottom: 8 }}>
          {t('referenceDialog.title')}
        </div>

        <div id={messageId} style={{ fontSize: '10pt', color: colors.textSecondary, lineHeight: 1.5, marginBottom: 10 }}>
          {single
            ? t('referenceDialog.messageOne', { name: files[0].name })
            : t('referenceDialog.messageMany', { count: files.length })}
        </div>

        {!single && (
          <div style={{
            fontSize: 11, color: colors.text, background: alpha(colors.border, 30),
            borderRadius: 4, padding: '6px 10px', marginBottom: 10, wordBreak: 'break-all',
          }}>
            {names.map(n => <div key={n}>{n}</div>)}
            {more > 0 && <div style={{ color: colors.textSecondary }}>{t('referenceDialog.andMore', { count: more })}</div>}
          </div>
        )}

        {single && (
          <>
            <input
              ref={inputRef}
              type="text"
              value={text}
              aria-label={t('referenceDialog.inputLabel')}
              placeholder={t('referenceDialog.inputPlaceholder')}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={onTextKeyDown}
              maxLength={MAX_CITATION_CHARS}
              style={{
                width: '100%', boxSizing: 'border-box', padding: '6px 10px', fontSize: '10pt',
                background: colors.bgSecondary, color: colors.text,
                border: `1px solid ${colors.border}`, borderRadius: 4, outline: 'none',
              }}
            />
            <div style={{
              fontSize: 11, color: colors.textSecondary, margin: '6px 0 14px',
              display: 'flex', justifyContent: 'space-between', gap: 8,
            }}>
              <span>
                {canSubmit
                  ? (parsed.doi ? t('referenceDialog.willSaveDoi') : t('referenceDialog.willSaveCitation'))
                  : t('referenceDialog.inputHint')}
              </span>
              <span data-testid="citation-counter" style={{ whiteSpace: 'nowrap', color: text.length >= MAX_CITATION_CHARS ? colors.red : undefined }}>
                {t('referenceDialog.counter', { count: text.length, max: MAX_CITATION_CHARS })}
              </span>
            </div>
          </>
        )}

        <div style={{ fontSize: 11, color: colors.textSecondary, marginBottom: 14, lineHeight: 1.5 }}>
          {t('referenceDialog.unpublishedNote')}
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, flexWrap: 'wrap' }}>
          <Button ref={cancelRef} variant="ghost" onClick={onCancel}>{t('common:cancel')}</Button>
          {!single && (
            <Button onClick={onSkip} title={t('referenceDialog.skipTooltip')}>
              {t('referenceDialog.skip', { count: files.length })}
            </Button>
          )}
          <Button onClick={onDeclareUnpublished} title={t('referenceDialog.declareTooltip')}>
            {single ? t('referenceDialog.declareOne') : t('referenceDialog.declareMany', { count: files.length })}
          </Button>
          {single && (
            <Button variant="primary" onClick={submit} disabled={!canSubmit}>
              {t('referenceDialog.saveAndConvert')}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
