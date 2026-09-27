// @vitest-environment jsdom
/**
 * The app tells the shell what language it is in.
 *
 * The splash screen runs before any page, so it cannot read the app's choice
 * out of localStorage — that belongs to the backend's origin, and the splash
 * is a file:// page. The M5 tester ran Orienta in German on an English Mac
 * and met an English splash at every start.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';
import i18n, { setLanguage } from './index';

afterEach(async () => {
  delete window.electronAPI;
  await i18n.changeLanguage('en');
});

describe('setLanguage', () => {
  it('hands the choice to the Electron shell', () => {
    const spy = vi.fn();
    window.electronAPI = { setLanguage: spy };
    setLanguage('de');
    expect(spy).toHaveBeenCalledWith('de');
    expect(i18n.language).toBe('de');
  });

  it('changes the language in the browser build, where there is no shell', () => {
    // No electronAPI at all: the language must still change.
    setLanguage('ja');
    expect(i18n.language).toBe('ja');
  });

  it('changes the language even if the shell refuses', async () => {
    // A read-only APPDATA, a locked file, a preload without the channel.
    // ipcRenderer.invoke returns a PROMISE, so that is a rejection and not a
    // throw. The first version of this test used a synchronous throw, which
    // the real bridge cannot do, and so proved nothing about the one failure
    // that can happen. An unhandled rejection is not cosmetic either:
    // errorReporter listens for them and files a problem report against the
    // user's own log.
    // Listened for on `process`, not on `window`: vitest runs on node, and
    // jsdom does not emit the window event. Watching the wrong one is how a
    // test like this passes while the rejection escapes — checked by removing
    // the .catch() and confirming this fails.
    const seen = [];
    const onUnhandled = (reason) => { seen.push(reason); };
    process.on('unhandledRejection', onUnhandled);
    window.electronAPI = { setLanguage: () => Promise.reject(new Error('read-only')) };

    expect(() => setLanguage('zh')).not.toThrow();
    expect(i18n.language).toBe('zh');
    // Two turns: one for the rejection, one for node to decide nobody handled it.
    await new Promise((resolve) => setTimeout(resolve, 10));
    process.off('unhandledRejection', onUnhandled);

    expect(seen.map(String)).toEqual([]);
  });

  it('hands the language over on a change made anywhere, not only through setLanguage', async () => {
    // This is what makes it work for the M5 tester. He had already chosen
    // German, so a hand-over that only fired inside setLanguage() would never
    // run for him: his splash would stay English until he re-picked the
    // language he was already using. The hook is on languageChanged, and the
    // module also hands over once at load.
    const spy = vi.fn();
    window.electronAPI = { setLanguage: spy };
    await i18n.changeLanguage('ja');
    expect(spy).toHaveBeenCalledWith('ja');
  });

  it('survives an older shell that has no such channel', () => {
    window.electronAPI = {};   // no setLanguage
    expect(() => setLanguage('de')).not.toThrow();
    expect(i18n.language).toBe('de');
  });
});
