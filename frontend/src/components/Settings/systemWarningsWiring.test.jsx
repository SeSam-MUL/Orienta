// @vitest-environment jsdom
/**
 * The connection, not the parts.
 *
 * `translateSystemWarnings` and the four locale files are each covered on
 * their own. This renders the real page against a real i18next instance, so
 * that a wrong argument order, a missing `ns:` prefix or a dropped `exists`
 * binding at the single call site (SettingsPage.jsx) shows up as English text
 * on screen instead of passing every unit test — the defect class this repo
 * keeps logging: both sides valid, only the wire between them wrong.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, cleanup } from '@testing-library/react';
import { I18nextProvider } from 'react-i18next';
import i18n from '../../i18n';

vi.mock('../../services/api', () => ({
  simApi: { systemStatus: vi.fn() },
  settingsApi: {
    get: vi.fn().mockResolvedValue({ data: {} }),
    update: vi.fn().mockResolvedValue({ data: {} }),
  },
}));
// Children that talk to the backend or Electron on mount; not under test here.
vi.mock('./InstallWizardSection', () => ({ default: () => null }));
vi.mock('./ApiKeysSection', () => ({ default: () => null }));
vi.mock('./AboutSection', () => ({ default: () => null }));

import { simApi } from '../../services/api';
import SettingsPage from './SettingsPage';

const CODED = 'GPU has 7.9 GB — using blocksize=8 for EMEBSDmasterOpenCL. Complex crystals (>20 atoms) may still fail.';
const PROSE = 'EMMCOpenCL not found';

const STATUS = {
  warnings: [PROSE, CODED],
  warning_items: [
    { code: 'gpuBlocksizeMinimal', values: { gb: '7.9', blocksize: 8 }, message: CODED },
  ],
  errors: [],
  checks: [],
};

describe('SettingsPage shows the warning in the user language', () => {
  beforeEach(() => {
    simApi.systemStatus.mockResolvedValue({ data: STATUS });
  });
  afterEach(cleanup);

  const renderIn = async (lng) => {
    await i18n.changeLanguage(lng);
    render(<I18nextProvider i18n={i18n}><SettingsPage /></I18nextProvider>);
    await waitFor(() => expect(simApi.systemStatus).toHaveBeenCalled());
  };

  it('German shows the German sentence, not the English one', async () => {
    await renderIn('de');
    await screen.findByText(/Blockgr/);
    expect(document.body.textContent).toContain('7.9 GB');
    expect(document.body.textContent).not.toContain('using blocksize=8');
  });

  it('a warning without a code stays exactly as the backend wrote it', async () => {
    await renderIn('de');
    await waitFor(() => expect(document.body.textContent).toContain(PROSE));
  });

  it('a code this frontend does not know yet shows prose, never a raw key', async () => {
    // A newer backend paired with an older frontend. The user must not read
    // "settings:systemStatus.warnings.somethingNew" on screen.
    const future = {
      code: 'somethingNew',
      values: {},
      message: 'Backend says something new',
    };
    simApi.systemStatus.mockResolvedValue({
      data: { ...STATUS, warnings: [future.message], warning_items: [future] },
    });
    await renderIn('de');
    await waitFor(() => expect(document.body.textContent).toContain(future.message));
    expect(document.body.textContent).not.toContain('systemStatus.warnings');
  });

  it('English gets the English sentence, with the same number', async () => {
    await renderIn('en');
    await waitFor(() => expect(document.body.textContent).toContain('smallest blocksize 8'));
    expect(document.body.textContent).toContain('7.9 GB');
  });
});
