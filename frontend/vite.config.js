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
        // The install wizard streams its log over a WebSocket that lives
        // under /api (/api/install/ws/install-emsoft). Without ws:true Vite
        // registers no upgrade handler for this context, so the browser's
        // upgrade request is never answered and the socket dies with a bare
        // "WebSocket connection error" — while every REST call still works.
        ws: true,
      },
      '/ws': {
        target: 'ws://127.0.0.1:8000',
        ws: true,
      },
    },
  },
})
