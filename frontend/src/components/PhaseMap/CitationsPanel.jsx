/**
 * "Citations for this result" — BibTeX / methods paragraph / plain reference
 * list, derived by the backend from the pipeline steps the active result
 * actually ran (`GET /api/citations/result/{id}`).
 *
 * The three payload texts (bibtex/methods/plain) are rendered VERBATIM and
 * NEVER translated — they are meant to be pasted into a manuscript, and a
 * "translated" BibTeX entry would be a broken BibTeX entry. Only the panel
 * chrome around them (tab labels, the copy button, the notes) is localised.
 *
 * `warnings` (e.g. "the bibliography could not be loaded") is shown
 * prominently above the tabs, whatever tab is active — a blank BibTeX tab
 * with no explanation is exactly the silent failure this panel exists to
 * avoid. `undeclared` steps (ran, but nothing citable is on file for them)
 * are listed rather than hidden, for the same reason.
 *
 * Refetches on every `resultId` change. Uses the ref-holding-the-latest-id
 * guard from `EDS/hooks/useEdsLayerStack.js`'s stale-mode race: a response
 * that arrives after `resultId` has already moved on must never be applied.
 */
import { useEffect, useRef, useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { citationsApi } from '../../services/api';
import {
  colors, alpha, CollapsibleGroup, Button,
} from '../../theme/components';

const TABS = ['bibtex', 'methods', 'plain'];

function CitationsPanel({ resultId }) {
  const { t } = useTranslation('citations');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('methods');
  const [copied, setCopied] = useState(false);

  // Mirrors the latest requested resultId so an in-flight fetch for an OLDER
  // id can detect, after its await, that it is no longer the current one —
  // same pattern useEdsLayerStack.js uses for displayMode.
  const resultIdRef = useRef(resultId);
  useEffect(() => {
    resultIdRef.current = resultId;
  }, [resultId]);

  useEffect(() => {
    if (!resultId) {
      setData(null);
      setError(null);
      setLoading(false);
      return;
    }
    const startedId = resultId;
    // Clear stale output immediately so a result switch never shows the
    // PREVIOUS result's citations while the new ones are loading.
    setData(null);
    setError(null);
    setLoading(true);
    citationsApi
      .forResult(resultId)
      .then((res) => {
        if (resultIdRef.current !== startedId) return; // superseded — drop it
        setData(res?.data || null);
      })
      .catch((err) => {
        if (resultIdRef.current !== startedId) return;
        setData(null);
        setError(
          err?.response?.data?.detail || err?.message || t('citations:error'),
        );
      })
      .finally(() => {
        if (resultIdRef.current === startedId) setLoading(false);
      });
  }, [resultId, t]);

  // Transient "Copied" label.
  useEffect(() => {
    if (!copied) return undefined;
    const timer = setTimeout(() => setCopied(false), 1500);
    return () => clearTimeout(timer);
  }, [copied]);

  const activeText = data
    ? (activeTab === 'bibtex' ? data.bibtex
      : activeTab === 'methods' ? data.methods
        : data.plain)
    : '';

  const handleCopy = useCallback(() => {
    if (!activeText) return;
    const p = navigator.clipboard?.writeText(activeText);
    if (p?.catch) p.catch(() => {}); // clipboard denial is not this panel's problem
    setCopied(true);
  }, [activeText]);

  if (!resultId) return null;

  return (
    <CollapsibleGroup title={t('citations:title')}>
      {loading && (
        <div style={{ fontSize: '8.5pt', color: colors.textSecondary }}>
          {t('citations:loading')}
        </div>
      )}

      {!loading && error && (
        <div style={{ fontSize: '8.5pt', color: colors.red }}>{error}</div>
      )}

      {!loading && !error && data && (
        <>
          {Array.isArray(data.warnings) && data.warnings.length > 0 && (
            <div
              style={{
                fontSize: '8pt',
                color: colors.orange,
                background: alpha(colors.orange, 10),
                border: `1px solid ${alpha(colors.orange, 25)}`,
                borderRadius: 3,
                padding: '6px 8px',
                display: 'flex',
                flexDirection: 'column',
                gap: 3,
              }}
            >
              {data.warnings.map((w, i) => (
                // eslint-disable-next-line react/no-array-index-key -- backend sentences, no stable id
                <div key={i}>{w}</div>
              ))}
            </div>
          )}

          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            {TABS.map((tab) => {
              const active = activeTab === tab;
              return (
                <button
                  key={tab}
                  type="button"
                  onClick={() => setActiveTab(tab)}
                  style={{
                    fontSize: '8.5pt',
                    padding: '4px 10px',
                    borderRadius: 4,
                    cursor: 'pointer',
                    border: `1px solid ${active ? colors.accent : colors.border}`,
                    background: active ? alpha(colors.accent, 15) : 'transparent',
                    color: active ? colors.accent : colors.textSecondary,
                    fontWeight: active ? 600 : 400,
                  }}
                >
                  {t(`citations:tabs.${tab}`)}
                </button>
              );
            })}
            <div style={{ flex: 1 }} />
            <Button small onClick={handleCopy} disabled={!activeText}>
              {copied ? t('citations:copied') : t('citations:copy')}
            </Button>
          </div>

          {/* The payload text itself — verbatim, never run through t(). */}
          <pre
            style={{
              fontSize: '8pt',
              color: colors.text,
              background: colors.bg,
              border: `1px solid ${colors.border}`,
              borderRadius: 4,
              padding: 8,
              margin: 0,
              maxHeight: 220,
              overflowY: 'auto',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              fontFamily: activeTab === 'bibtex' ? 'monospace' : 'inherit',
            }}
          >
            {activeText || '—'}
          </pre>

          <div style={{ fontSize: '7.5pt', color: colors.textSecondary, fontStyle: 'italic' }}>
            {t('citations:englishOnly')}
          </div>
          <div style={{ fontSize: '7.5pt', color: colors.textSecondary, fontStyle: 'italic' }}>
            {t('citations:scopeNote')}
          </div>

          {Array.isArray(data.undeclared) && data.undeclared.length > 0 && (
            <div style={{ fontSize: '8pt', color: colors.textSecondary }}>
              {t('citations:undeclared', { count: data.undeclared.length })}
              <ul style={{ margin: '4px 0 0 16px', padding: 0 }}>
                {data.undeclared.map((key) => (
                  <li key={key}>{key}</li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </CollapsibleGroup>
  );
}

export default CitationsPanel;
