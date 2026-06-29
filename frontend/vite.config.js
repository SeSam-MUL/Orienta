import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  test: {
    // Initialize i18n before each test file so useTranslation() yields real
    // (English-default) strings as components migrate to i18n. Per-file
    // `// @vitest-environment jsdom` directives still control the DOM env.
    setupFiles: ['./src/test/i18n-setup.js'],
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://127.0.0.1:8000',
        ws: true,
      },
    },
  },
})
