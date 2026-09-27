/**
 * i18n setup (react-i18next).
 *
 * Locale resources are auto-discovered from src/locales/<lng>/<namespace>.json
 * via Vite's import.meta.glob — to add translations for a page you only drop a
 * JSON file in the right folder; no edit here is needed. Namespaces map 1:1 to
 * UI areas (e.g. `nav`, `shell`, `eds`, `indexing`), plus a shared `common`.
 *
 * Target languages: en (source), de, ja, zh.
 */
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import { resolveStartLanguage } from './startLanguage';

// Eagerly bundle every locale JSON. Path shape: ../locales/<lng>/<ns>.json
const modules = import.meta.glob('../locales/*/*.json', { eager: true });
const resources = {};
for (const path in modules) {
  const m = path.match(/\/locales\/([^/]+)\/([^/]+)\.json$/);
  if (!m) continue;
  const [, lng, ns] = m;
  (resources[lng] ||= {})[ns] = modules[path].default;
}
const namespaces = [...new Set(Object.values(resources).flatMap((r) => Object.keys(r)))];

export const LANGUAGES = [
  { code: 'en', label: 'English', short: 'EN' },
  { code: 'de', label: 'Deutsch', short: 'DE' },
  { code: 'ja', label: '日本語', short: 'JA' },
  { code: 'zh', label: '中文', short: 'ZH' },
];

const STORAGE_KEY = 'app_lang';
function storedLang() {
  try { return localStorage.getItem(STORAGE_KEY); } catch { return null; }
}

// The setup wizard's language, handed over once by the shell (see
// startLanguage.js for the rule). Read, applied, and removed from the address.
function startLanguage() {
  let search = '';
  try { search = window.location.search; } catch { /* no window: tests */ }
  const decided = resolveStartLanguage({
    stored: storedLang(),
    search,
    supported: LANGUAGES.map((l) => l.code),
  });
  if (decided.persist) {
    try { localStorage.setItem(STORAGE_KEY, decided.lang); } catch { /* storage unavailable */ }
  }
  if (decided.hadHandOver) {
    try {
      window.history.replaceState(window.history.state, '',
        window.location.pathname + decided.cleanSearch + window.location.hash);
    } catch { /* cosmetic */ }
  }
  return decided.lang;
}

i18n.use(initReactI18next).init({
  resources,
  lng: startLanguage(),
  fallbackLng: 'en',
  defaultNS: 'common',
  ns: namespaces.length ? namespaces : ['common'],
  interpolation: { escapeValue: false }, // React already escapes
  returnEmptyString: false,
});

/**
 * Tell the Electron shell what language we are in.
 *
 * The shell cannot read the choice itself: it lives in the localStorage of
 * the backend's origin, and the splash screen is a file:// page that runs
 * before any of this. So it is handed over, and the shell remembers it for
 * the next start.
 *
 * On `languageChanged` AND once at load, and the second one is the point. A
 * hand-over only on change would do nothing for the person who reported the
 * bug: he had already chosen German, so the splash would have stayed English
 * until he happened to re-pick the language he was already using. It would
 * also miss every existing user and every wizard install, where the wizard's
 * own hand-off file is consumed on first load and never reaches this one.
 *
 * recordApp() on the other side no-ops when the language is unchanged, so
 * doing this at every start costs nothing.
 */
function handOverToShell(code) {
  if (!code) return;
  try {
    // ipcRenderer.invoke returns a promise: a synchronous try/catch does not
    // catch its rejection, and an unhandled rejection is picked up by
    // errorReporter and filed as a problem report against the user's own log.
    window.electronAPI?.setLanguage?.(code)?.catch?.(() => {});
  } catch { /* not in Electron, or an older preload with no such channel */ }
}

i18n.on('languageChanged', handOverToShell);
handOverToShell(i18n.language);

/** Change the active language and persist the choice. */
export function setLanguage(code) {
  i18n.changeLanguage(code);          // fires languageChanged -> handOverToShell
  try { localStorage.setItem(STORAGE_KEY, code); } catch { /* storage unavailable */ }
}

export default i18n;

// The document's language attribute follows the app's.
//
// Without this it stays at whatever index.html says (en), which makes CSS
// hyphenation apply English rules to German compounds and makes a screen
// reader voice Japanese text with an English voice. Guarded because this
// module is imported in jsdom and in node tests too.
if (typeof document !== 'undefined' && document.documentElement) {
  const applyLang = (lng) => { document.documentElement.lang = lng || 'en'; };
  applyLang(i18n.language);
  i18n.on('languageChanged', applyLang);
}
