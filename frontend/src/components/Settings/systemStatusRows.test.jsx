/**
 * What the system-status panel may claim on each platform.
 *
 * The backend reports wsl_distro = "native-linux" off Windows, so this panel
 * showed a green "WSL: native-linux" row on a Mac — naming a Windows feature
 * that cannot exist there and reporting it as installed.
 */
import { describe, it, expect } from 'vitest';
import { systemStatusRows } from './SettingsPage';

const t = (key) => key;
const BASE = {
  wsl_installed: true, wsl_distro: 'native-linux',
  emsoft_available: false, emsphinx_available: false,
  opencl_available: false, has_gpu: false, cpu_count: 8,
};
const labels = (status) => systemStatusRows(status, t).map((r) => r.label);

describe('the system-status rows and the platform', () => {
  it('has no WSL row on a Mac', () => {
    expect(labels({ ...BASE, platform_os: 'macos' }))
      .not.toContain('settings:systemStatus.rows.wsl');
  });

  it('has no WSL row on Linux', () => {
    expect(labels({ ...BASE, platform_os: 'linux' }))
      .not.toContain('settings:systemStatus.rows.wsl');
  });

  it('keeps the WSL row on Windows, with the distribution', () => {
    const rows = systemStatusRows({ ...BASE, platform_os: 'windows', wsl_distro: 'Ubuntu' }, t);
    const wsl = rows.find((r) => r.label === 'settings:systemStatus.rows.wsl');
    expect(wsl).toBeTruthy();
    expect(wsl.value).toBe('Ubuntu');
    expect(wsl.dotStatus).toBe('ok');
  });

  it('assumes Windows when an older backend does not say', () => {
    expect(labels(BASE)).toContain('settings:systemStatus.rows.wsl');
  });

  it('keeps every other row on every platform', () => {
    const win = labels({ ...BASE, platform_os: 'windows' });
    const mac = labels({ ...BASE, platform_os: 'macos' });
    expect(win.length - mac.length).toBe(1);
    for (const label of mac) expect(win).toContain(label);
  });

  it('returns nothing before the status has arrived', () => {
    expect(systemStatusRows(null, t)).toEqual([]);
  });
});

// A red dot means "something is wrong and you should act on it". On macOS
// there is nothing to act on: the app offers no way to install EMsoft there,
// and its own simulation engine does not need it. The 2026-09-25 Mac tester
// saw two red rows plus a red error telling him to "Install EMsoft in WSL" —
// a Windows feature — on a machine where everything he tried worked.
describe('EMsoft on a platform where it is not offered', () => {
  const rowsFor = (os) => {
    const rows = systemStatusRows({ ...BASE, platform_os: os }, t);
    const by = (name) => rows.find((r) => r.label === `settings:systemStatus.rows.${name}`);
    return { emsoft: by('emsoft'), emsphinx: by('emsphinx'), opencl: by('opencl') };
  };

  it('is a state, not a fault, on macOS', () => {
    const { emsoft, emsphinx } = rowsFor('macos');
    expect(emsoft.dotStatus).toBe(null);
    expect(emsphinx.dotStatus).toBe(null);
    expect(emsoft.value).toBe('settings:systemStatus.values.notSetUp');
    expect(emsphinx.value).toBe('settings:systemStatus.values.notSetUp');
  });

  it('does not warn about OpenCL on macOS — it is EMsoft\'s GPU backend', () => {
    expect(rowsFor('macos').opencl.dotStatus).toBe(null);
  });

  it('stays an error on Windows, where the wizard can install it', () => {
    const { emsoft, emsphinx, opencl } = rowsFor('windows');
    expect(emsoft.dotStatus).toBe('error');
    expect(emsphinx.dotStatus).toBe('error');
    expect(opencl.dotStatus).toBe('warning');
    expect(emsoft.value).toBe('settings:systemStatus.values.notFound');
  });

  it('is a state on Linux too, since 2026-09-25', () => {
    // The install script has a Linux branch and the endpoint would run it
    // natively -- but nobody ever has, so the wizard does not offer it, and a
    // red dot would point at a button that is no longer there.
    const { emsoft, emsphinx } = rowsFor('linux');
    expect(emsoft.dotStatus).toBe(null);
    expect(emsphinx.dotStatus).toBe(null);
  });

  it('leaves Windows as the one platform that really installs it', () => {
    // Positive control for the two above.
    expect(rowsFor('windows').emsoft.dotStatus).toBe('error');
  });

  it('shows the path when EMsoft IS present, whatever the platform', () => {
    // The neutral wording must not swallow a real installation.
    const rows = systemStatusRows(
      { ...BASE, platform_os: 'macos', emsoft_available: true, emsoft_path: '/opt/emsoft/Bin' }, t,
    );
    const emsoft = rows.find((r) => r.label === 'settings:systemStatus.rows.emsoft');
    expect(emsoft.value).toBe('/opt/emsoft/Bin');
    expect(emsoft.dotStatus).toBe('ok');
  });
});
