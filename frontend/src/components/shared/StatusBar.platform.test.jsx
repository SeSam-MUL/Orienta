// @vitest-environment jsdom
/**
 * What the status line may claim on each platform.
 *
 * The backend reports wsl_distro = "native-linux" off Windows, so the line
 * used to read "WSL: native-linux" on a Mac — naming a Windows feature that
 * cannot exist there, in green, as if it were installed.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, cleanup, waitFor } from '@testing-library/react';

vi.mock('../../theme/tokens', () => ({
  colors: {
    bg: '#000', bgTertiary: '#111', border: '#444', text: '#fff',
    textSecondary: '#aaa', green: '#0f0', red: '#f00', yellow: '#fd0',
    accent: '#0ff', purple: '#a6f',
  },
  layout: { statusBarHeight: 24 },
}));

const status = { data: {} };
vi.mock('../../services/api', () => ({
  simApi: { systemStatus: () => Promise.resolve(status) },
}));

vi.mock('../../stores/useDataStore', () => ({
  default: () => ({
    isFileOpen: false, filePath: null, gridShape: null, patternCount: 0,
    detector: null, stepSize: null, edsElements: null,
  }),
}));
vi.mock('../../stores/useProgressStore', () => ({ default: () => ({ tasks: {} }) }));

afterEach(cleanup);

import StatusBar from './StatusBar';

const BASE = {
  emsoft_available: true, wsl_installed: true, emsphinx_available: true,
  opencl_available: false, has_gpu: false, cpu_count: 8,
};

async function textFor(payload) {
  status.data = { ...BASE, ...payload };
  const { container } = render(<StatusBar backendStatus="connected" />);
  await waitFor(() => expect(container.textContent).toContain('EMsoft'));
  return container.textContent;
}

describe('the status line and the platform', () => {
  it('does not claim WSL on a Mac', async () => {
    const text = await textFor({ platform_os: 'macos', wsl_distro: 'native-linux' });
    expect(text).not.toContain('WSL');
    expect(text).not.toContain('native-linux');
  });

  it('does not claim WSL on Linux either', async () => {
    const text = await textFor({ platform_os: 'linux', wsl_distro: 'native-linux' });
    expect(text).not.toContain('WSL');
  });

  it('still shows the distribution on Windows', async () => {
    const text = await textFor({ platform_os: 'windows', wsl_distro: 'Ubuntu' });
    expect(text).toContain('WSL: Ubuntu');
  });

  it('assumes Windows when the backend is too old to say', async () => {
    // An older backend has no platform_os; on Windows, where that backend
    // runs, the dot must not disappear.
    const text = await textFor({ wsl_distro: 'Ubuntu' });
    expect(text).toContain('WSL: Ubuntu');
  });
});

/**
 * The status line in the reader's language.
 *
 * The M5 tester ran the app in German and read "EMsoft: Not found",
 * "No GPU" and "Backend: Connected" along the bottom edge (report, section 5,
 * point 4). Every one of those was an English literal in this file.
 */
import i18n from '../../i18n';
import { act } from '@testing-library/react';

describe('the status line and the language', () => {
  afterEach(async () => {
    await act(async () => { await i18n.changeLanguage('en'); });
  });

  it('names the system state in German when the app is in German', async () => {
    await act(async () => { await i18n.changeLanguage('de'); });
    const text = await textFor({
      emsoft_available: false, emsphinx_available: false, wsl_installed: false,
      opencl_available: false, platform_os: 'windows',
    });
    expect(text).toContain('Nicht gefunden');      // EMsoft and EMSphinx
    expect(text).toContain('Nicht installiert');   // WSL
    expect(text).toContain('Keine GPU');
    expect(text).toContain('Backend: Verbunden');
    // and nothing left in English
    expect(text).not.toContain('Not found');
    expect(text).not.toContain('No GPU');
    expect(text).not.toContain('Connected');
  });

  it('still reads correctly in English', async () => {
    await act(async () => { await i18n.changeLanguage('en'); });
    const text = await textFor({
      emsoft_available: false, emsphinx_available: false, wsl_installed: false,
      opencl_available: false, platform_os: 'windows',
    });
    expect(text).toContain('EMsoft: Not found');
    expect(text).toContain('No GPU');
    expect(text).toContain('Backend: Connected');
  });
});
