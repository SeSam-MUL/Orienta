import { useState, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { colors, alpha, Button } from '../../theme/components';

// ---------------------------------------------------------------------------
// Spinner — small CSS-animated circle
// ---------------------------------------------------------------------------

const spinnerKeyframes = `
@keyframes exportbar-spin {
  to { transform: rotate(360deg); }
}
`;

function ensureSpinnerStyles() {
  if (typeof document !== 'undefined' && !document.getElementById('exportbar-spinner-styles')) {
    const style = document.createElement('style');
    style.id = 'exportbar-spinner-styles';
    style.textContent = spinnerKeyframes;
    document.head.appendChild(style);
  }
}

function Spinner() {
  ensureSpinnerStyles();
  return (
    <span style={{
      display: 'inline-block',
      width: 11, height: 11,
      borderRadius: '50%',
      border: `2px solid ${alpha(colors.text, 25)}`,
      borderTopColor: colors.text,
      animation: 'exportbar-spin 0.7s linear infinite',
      flexShrink: 0,
    }} />
  );
}

// ---------------------------------------------------------------------------
// Delete confirm inline pattern
// ---------------------------------------------------------------------------

const CONFIRM_TIMEOUT_MS = 5000;

function DeleteButton({ checkedCount, checkedXtalCount = 0, serverAvailable = false, onDelete }) {
  const { t } = useTranslation(['crystaldatabase', 'common']);
  const [confirming, setConfirming] = useState(false);
  const timerRef = useRef(null);

  const startConfirm = () => {
    setConfirming(true);
    timerRef.current = setTimeout(() => setConfirming(false), CONFIRM_TIMEOUT_MS);
  };

  const cancelConfirm = () => {
    clearTimeout(timerRef.current);
    setConfirming(false);
  };

  const handleDelete = (fromServer) => {
    clearTimeout(timerRef.current);
    setConfirming(false);
    onDelete(fromServer);
  };

  // Clean up timer on unmount
  useEffect(() => () => clearTimeout(timerRef.current), []);

  // Reset confirm state when selection changes away
  useEffect(() => {
    if (checkedCount === 0) cancelConfirm();
  }, [checkedCount]);

  const fileDesc = t('exportBar.deleteDesc', {
    count: checkedCount,
    xtal: checkedXtalCount > 0 ? t('exportBar.deleteDescXtal', { count: checkedXtalCount }) : '',
  });

  if (confirming) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 11, color: colors.red, whiteSpace: 'nowrap' }}>
          {t('exportBar.deletePrompt', { desc: fileDesc })}
        </span>
        <Button
          variant="danger"
          small
          onClick={() => handleDelete(false)}
          style={{ fontSize: 11 }}
          title={t('exportBar.localOnlyTooltip')}
        >
          {t('exportBar.localOnly')}
        </Button>
        {serverAvailable && (
          <Button
            small
            onClick={() => handleDelete(true)}
            style={{
              fontSize: 11,
              background: alpha(colors.red, 15),
              color: colors.red,
              border: `1px solid ${colors.red}`,
            }}
            title={t('exportBar.everywhereTooltip')}
          >
            {t('exportBar.everywhere')}
          </Button>
        )}
        <Button
          variant="ghost"
          small
          onClick={cancelConfirm}
          style={{ fontSize: 11 }}
          title={t('hoverTips.deleteCancel')}
        >
          {t('common:cancel')}
        </Button>
      </div>
    );
  }

  return (
    <Button
      variant="danger"
      small
      onClick={startConfirm}
      style={{ fontSize: 11 }}
      title={t('hoverTips.deleteStart')}
    >
      {t('exportBar.deleteSelected', { count: checkedCount })}
    </Button>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function ExportBar({
  selectedFile,
  selectedFit,
  checkedCount = 0,
  checkedXtalCount = 0,
  serverAvailable = false,
  converting = false,
  batchConverting = false,
  onConvert,
  onBatchConvert,
  onDelete,
}) {
  const { t } = useTranslation('crystaldatabase');
  const canConvertSingle = !!selectedFile && !converting && selectedFit !== false;

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 8,
      padding: '8px 0 4px',
      borderTop: `1px solid ${colors.border}`,
      flexShrink: 0,
      flexWrap: 'wrap',
    }}>
      {/* Single file convert */}
      <Button
        small
        variant="primary"
        onClick={onConvert}
        disabled={!canConvertSingle}
        style={{ fontSize: 11 }}
        title={!selectedFile ? t('exportBar.convertSelectFirst')
          : selectedFit === false ? t('exportBar.convertNotFit')
          : t('exportBar.convertTooltip', { name: selectedFile.name })}
      >
        {converting ? (
          <>
            <Spinner />
            {t('exportBar.converting')}
          </>
        ) : (
          t('exportBar.convert')
        )}
      </Button>

      {/* Batch convert — only when checked files exist */}
      {checkedCount > 0 && (
        <Button
          small
          variant="purple"
          onClick={onBatchConvert}
          disabled={batchConverting}
          style={{ fontSize: 11 }}
          title={t('exportBar.batchConvertTooltip', { count: checkedCount })}
        >
          {batchConverting ? (
            <>
              <Spinner />
              {t('exportBar.batchConverting', { count: checkedCount })}
            </>
          ) : (
            t('exportBar.batchConvert', { count: checkedCount })
          )}
        </Button>
      )}

      {/* Delete — only when checked files exist */}
      {checkedCount > 0 && (
        <DeleteButton checkedCount={checkedCount} checkedXtalCount={checkedXtalCount} serverAvailable={serverAvailable} onDelete={onDelete} />
      )}
    </div>
  );
}
