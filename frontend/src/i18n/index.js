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

i18n.use(initReactI18next).init({
  resources,
  lng: storedLang() || 'en',
  fallbackLng: 'en',
  defaultNS: 'common',
  ns: namespaces.length ? namespaces : ['common'],
  interpolation: { escapeValue: false }, // React already escapes
  returnEmptyString: false,
});

/** Change the active language and persist the choice. */
export function setLanguage(code) {
  i18n.changeLanguage(code);
  try { localStorage.setItem(STORAGE_KEY, code); } catch { /* storage unavailable */ }
}

export default i18n;
