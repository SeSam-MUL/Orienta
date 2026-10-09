/**
 * The text on the page shown while the backend starts, and the clock that ticks
 * on it.
 *
 * WHY A PHASE. The clock used to rewrite title, body and hint every second from
 * the same three strings, while the runtime update painted its own text ONCE.
 * The update's "Updating the program files..." was therefore overwritten within
 * a second of appearing -- never seen on screen, only found by reading the code.
 * A package sync lasts a minute, so its text has to survive: the page now has a
 * current PHASE, set by whoever knows what the shell is doing, and every paint,
 * including the clock's, reads its words from it.
 *
 * The page itself is inert (CSP default-src 'none'), so the main process writes
 * its text; this module only decides WHICH text, and it takes the window and the
 * timer functions as arguments so the behaviour can be tested without Electron.
 */

/** phase -> the two strings (keys in strings.js) that change with it. The title
 *  is the same in every phase: it is the name of the state, "starting". */
const PHASES = Object.freeze({
  starting: { body: 'startingBody', hint: 'startingHint' },
  updating: { body: 'updatingBody', hint: 'updatingHint' },
  syncing: { body: 'syncingBody', hint: 'syncingHint' },
});

/**
 * @param t              strings.t
 * @param shellLanguage  strings.shellLanguage
 * @param getLocale      () -> the locale to speak. Asked ONCE, at the first paint,
 *                       and kept: the clock paints every second, and the answer
 *                       reads the app's language file from disk. The language
 *                       cannot change while this page is up.
 */
function createWaitingPage({
  t, shellLanguage, getLocale,
  now = Date.now,
  setIntervalFn = setInterval,
  clearIntervalFn = clearInterval,
}) {
  let phase = 'starting';
  let locale = null;
  const currentLocale = () => {
    if (locale === null) locale = getLocale();
    return locale;
  };

  /** The page text for the current phase. `seconds` undefined leaves the
   *  counter alone; null clears it; a number shows it. */
  function textFor(seconds) {
    const lang = currentLocale();
    const keys = PHASES[phase];
    const text = {
      lang: shellLanguage(lang),
      title: t(lang, 'startingTitle'),
      body: t(lang, keys.body),
      hint: t(lang, keys.hint),
    };
    if (seconds !== undefined) {
      text.elapsed = seconds === null ? '' : t(lang, 'elapsed', { seconds });
    }
    return text;
  }

  function paint(window, seconds) {
    if (!window || window.isDestroyed()) return;
    const text = textFor(seconds);
    window.webContents
      .executeJavaScript(
        `(() => { const s = ${JSON.stringify(text)};
          document.documentElement.lang = s.lang;
          const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
          set('title', s.title); set('body', s.body); set('hint', s.hint);
          if (s.elapsed !== undefined) set('elapsed', s.elapsed);
        })()`,
      )
      .catch(() => { /* the page may not be loaded yet or already be gone; the phase is kept either way */ });
  }

  return {
    phase: () => phase,

    /** Say what the shell is doing now, immediately. */
    setPhase(window, next) {
      if (!Object.prototype.hasOwnProperty.call(PHASES, next)) {
        throw new Error(`unknown waiting phase: ${next}`);
      }
      phase = next;
      paint(window);
    },

    /** Fill in the page when it has loaded, then tick the counter. Returns the
     *  function that stops it. */
    startClock(window) {
      const started = now();
      window.webContents.once('did-finish-load', () => paint(window, null));
      const timer = setIntervalFn(() => {
        if (!window || window.isDestroyed()) { clearIntervalFn(timer); return; }
        paint(window, Math.round((now() - started) / 1000));
      }, 1000);
      // Stopped by whenReady once the backend answers, and by the window closing.
      window.once('closed', () => clearIntervalFn(timer));
      return () => clearIntervalFn(timer);
    },
  };
}

module.exports = { PHASES, createWaitingPage };
