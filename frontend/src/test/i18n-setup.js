// Vitest global setup: initialize i18n so components that use useTranslation()
// render real strings (English by default) instead of raw `ns:key` keys.
// Keeps existing text-based assertions valid as components migrate to i18n.
import '../i18n';
