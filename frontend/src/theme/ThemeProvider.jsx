/**
 * ThemeProvider — applies a CSS-variable theme to :root at runtime.
 *
 * Usage:
 *   import { useTheme } from './theme/ThemeProvider';
 *   const { theme, setTheme, themes } = useTheme();
 *
 * Theme is persisted in localStorage under key 'kikuchipy-theme'.
 * Default theme: 'dracula'.
 */

import { createContext, useContext, useEffect, useState } from 'react';
import { themes, THEME_NAMES } from './themes';

const STORAGE_KEY = 'kikuchipy-theme';
const DEFAULT_THEME = 'dracula';

const ThemeContext = createContext(null);

/**
 * Apply a theme's CSS variables to document :root.
 */
function applyTheme(themeName) {
  const vars = themes[themeName];
  if (!vars) return;
  const root = document.documentElement;
  Object.entries(vars).forEach(([key, value]) => {
    if (key.startsWith('--')) {
      root.style.setProperty(key, value);
    }
  });
}

export function ThemeProvider({ children }) {
  const [theme, setThemeState] = useState(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      return THEME_NAMES.includes(stored) ? stored : DEFAULT_THEME;
    } catch {
      return DEFAULT_THEME;
    }
  });

  // Apply theme on mount and whenever it changes
  useEffect(() => {
    applyTheme(theme);
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      // localStorage unavailable — ignore
    }
  }, [theme]);

  const setTheme = (name) => {
    if (THEME_NAMES.includes(name)) {
      setThemeState(name);
    }
  };

  return (
    <ThemeContext.Provider value={{ theme, setTheme, themes: THEME_NAMES }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) {
    throw new Error('useTheme must be used inside <ThemeProvider>');
  }
  return ctx;
}

export default ThemeProvider;
