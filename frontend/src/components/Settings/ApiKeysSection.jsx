/**
 * API Keys section of the Settings page.
 *
 * Stores user API keys (Materials Project, etc.) in the per-machine user-config
 * file. Keys NEVER come back from the server in plain text — only as a
 * masked preview + a `configured: bool` flag. The "Save" button sends the
 * raw key over the API once; "Test" runs a live connectivity check.
 *
 * When the project tree is moved to another PC, this config goes WITH the
 * user (it lives in `%APPDATA%\Kikuchipy\` on Windows, not the project).
 */

import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { settingsApi } from '../../services/api';
import {
  colors, spacing,
  Button, Input, GroupBox, Label,
} from '../../theme/components';

const PROVIDERS = [
  {
    key: 'materials_project',
    labelKey: 'settings:apiKeys.providers.materialsProject.label',
    placeholder: 'mp_…',
    helpUrl: 'https://next-gen.materialsproject.org/api',
    helpTextKey: 'settings:apiKeys.providers.materialsProject.helpText',
  },
];


function MaskedInput({ value, onChange, placeholder, disabled }) {
  const { t } = useTranslation('settings');
  const [revealed, setRevealed] = useState(false);
  return (
    <div style={{ display: 'flex', gap: 6, alignItems: 'stretch', flex: 1 }}>
      <Input
        type={revealed ? 'text' : 'password'}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        disabled={disabled}
        autoComplete="off"
        spellCheck={false}
        title={t('settings:apiKeys.keyTooltip')}
        style={{ flex: 1, fontFamily: 'monospace' }}
      />
      <Button
        type="button"
        variant="secondary"
        onClick={() => setRevealed(v => !v)}
        title={revealed ? t('settings:apiKeys.hideKey') : t('settings:apiKeys.showKey')}
        aria-label={t('settings:apiKeys.showKeyTooltip')}
        style={{ minWidth: 44 }}
      >
        {revealed ? t('settings:apiKeys.hideKey') : t('settings:apiKeys.showKey')}
      </Button>
    </div>
  );
}


function ProviderRow({ provider, status, onSave, onTest, onClear }) {
  const { t } = useTranslation('settings');
  const [draft, setDraft] = useState('');
  const [busy, setBusy] = useState(null);  // 'save' | 'test' | 'clear' | null
  const [feedback, setFeedback] = useState(null);  // { ok, msg }

  const handleSave = useCallback(async () => {
    if (!draft.trim()) return;
    setBusy('save'); setFeedback(null);
    try {
      await onSave(draft);
      setDraft('');
      setFeedback({ ok: true, msg: t('settings:apiKeys.saved') });
    } catch (err) {
      const m = err?.response?.data?.detail || err?.message || t('settings:apiKeys.saveFailed');
      setFeedback({ ok: false, msg: m });
    } finally {
      setBusy(null);
    }
  }, [draft, onSave, t]);

  const handleTest = useCallback(async () => {
    setBusy('test'); setFeedback(null);
    try {
      const res = await onTest();
      const ok = res.data?.success === true;
      const msg = ok
        ? (res.data?.message || t('settings:apiKeys.keyWorks'))
        : (res.data?.error || t('settings:apiKeys.testFailed'));
      setFeedback({ ok, msg });
    } catch (err) {
      const m = err?.response?.data?.detail || err?.message || t('settings:apiKeys.testError');
      setFeedback({ ok: false, msg: m });
    } finally {
      setBusy(null);
    }
  }, [onTest, t]);

  const handleClear = useCallback(async () => {
    setBusy('clear'); setFeedback(null);
    try {
      await onClear();
      setFeedback({ ok: true, msg: t('settings:apiKeys.cleared') });
    } catch (err) {
      const m = err?.response?.data?.detail || err?.message || t('settings:apiKeys.clearFailed');
      setFeedback({ ok: false, msg: m });
    } finally {
      setBusy(null);
    }
  }, [onClear, t]);

  const configured = !!status?.configured;
  const preview = status?.preview;

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: 6,
      padding: spacing.md,
      background: colors.bgSecondary,
      border: `1px solid ${colors.border}`,
      borderRadius: 6,
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'baseline',
        gap: 8,
        justifyContent: 'space-between',
      }}>
        <Label style={{ fontWeight: 600 }}>{t(provider.labelKey)}</Label>
        <span style={{
          fontSize: 11,
          color: configured ? colors.green : colors.textSecondary,
          fontFamily: configured ? 'monospace' : 'inherit',
        }}>
          {configured ? t('settings:apiKeys.configured', { preview }) : t('settings:apiKeys.notConfigured')}
        </span>
      </div>
      <p style={{ fontSize: 11, color: colors.textSecondary, margin: '2px 0 6px' }}>
        {t(provider.helpTextKey)}{' '}
        <a
          href={provider.helpUrl}
          target="_blank"
          rel="noopener noreferrer"
          title={t('settings:apiKeys.getKeyTooltip')}
          style={{ color: colors.cyan }}
        >
          {t('settings:apiKeys.getKey')}
        </a>
      </p>
      <div style={{ display: 'flex', gap: 6 }}>
        <MaskedInput
          value={draft}
          onChange={setDraft}
          placeholder={provider.placeholder}
          disabled={busy !== null}
        />
        <Button
          onClick={handleSave}
          disabled={!draft.trim() || busy !== null}
          variant="primary"
          style={{ minWidth: 70 }}
          title={t('settings:apiKeys.saveTooltip')}
        >
          {busy === 'save' ? t('settings:apiKeys.busy') : t('settings:apiKeys.save')}
        </Button>
        <Button
          onClick={handleTest}
          disabled={!configured || busy !== null}
          variant="secondary"
          style={{ minWidth: 70 }}
          title={configured ? t('settings:apiKeys.testStoredKeyTooltip') : t('settings:apiKeys.saveKeyFirstTooltip')}
        >
          {busy === 'test' ? t('settings:apiKeys.busy') : t('settings:apiKeys.test')}
        </Button>
        <Button
          onClick={handleClear}
          disabled={!configured || busy !== null}
          variant="secondary"
          style={{ minWidth: 70 }}
          title={t('settings:apiKeys.clearTooltip')}
        >
          {busy === 'clear' ? t('settings:apiKeys.busy') : t('settings:apiKeys.clear')}
        </Button>
      </div>
      {feedback && (
        <div style={{
          fontSize: 11,
          color: feedback.ok ? colors.green : colors.red,
          padding: '4px 0',
        }}>
          {feedback.ok ? '✓ ' : '✗ '}{feedback.msg}
        </div>
      )}
    </div>
  );
}


export default function ApiKeysSection() {
  const { t } = useTranslation('settings');
  const [apiKeyStatus, setApiKeyStatus] = useState({});
  const [configPath, setConfigPath] = useState('');
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const res = await settingsApi.get();
      setApiKeyStatus(res.data?.api_keys || {});
      setConfigPath(res.data?.config_path || '');
    } catch (err) {
      console.error('Failed to load settings:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const saveKey = useCallback(async (providerKey, value) => {
    await settingsApi.saveApiKeys({ [providerKey]: value });
    await refresh();
  }, [refresh]);

  const testKey = useCallback(async (providerKey) => {
    return await settingsApi.testApiKey(providerKey);
  }, []);

  const clearKey = useCallback(async (providerKey) => {
    await settingsApi.saveApiKeys({ [providerKey]: '' });
    await refresh();
  }, [refresh]);

  return (
    <GroupBox title={t('settings:apiKeys.title')}>
      <p style={{
        fontSize: 11,
        color: colors.textSecondary,
        margin: '0 0 8px',
      }}>
        {t('settings:apiKeys.storedAt')} <code style={{
          fontFamily: 'monospace',
          background: colors.bgTertiary,
          padding: '1px 6px',
          borderRadius: 3,
          fontSize: 10,
        }}>{configPath || '…'}</code>.{' '}
        {t('settings:apiKeys.storedNote')}
      </p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {PROVIDERS.map(p => (
          <ProviderRow
            key={p.key}
            provider={p}
            status={apiKeyStatus[p.key]}
            onSave={(value) => saveKey(p.key, value)}
            onTest={() => testKey(p.key)}
            onClear={() => clearKey(p.key)}
          />
        ))}
      </div>
      {loading && (
        <div style={{ fontSize: 11, color: colors.textSecondary, marginTop: 6 }}>
          {t('settings:apiKeys.loading')}
        </div>
      )}
    </GroupBox>
  );
}
