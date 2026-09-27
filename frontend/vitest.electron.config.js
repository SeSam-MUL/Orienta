import path from 'node:path';
import { defineConfig } from 'vitest/config';

/**
 * A second vitest project for the Electron main-process modules.
 *
 * Separate from vite.config.js on purpose: that config has no `test.include`
 * at all, so adding one there would introduce a glob where none exists and
 * silently change which files the main suite picks up. These tests also need
 * the node environment rather than jsdom, and a root above `frontend/` so the
 * glob does not have to climb out of the vitest root.
 */
export default defineConfig({
  root: path.resolve(import.meta.dirname, '..'),
  test: {
    name: 'electron',
    include: ['tests/electron/**/*.test.js'],
    environment: 'node',
  },
});
