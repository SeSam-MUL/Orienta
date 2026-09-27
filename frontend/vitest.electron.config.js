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
    // Vitest's default is 5 s. These tests create and delete a temporary
    // directory each, and the work inside them is synchronous file reads that
    // take single-digit milliseconds — so a five-second limit never measures
    // the code, it measures the machine. On 2026-09-27 a CI runner stalled long
    // enough for `projectRoot.test.js` to blow it, and the cost of that flake
    // was not the rerun: it failed the job, so the checks that ran after it
    // never ran at all, and a green result was withheld from work that was
    // fine. Twenty seconds still catches anything genuinely hung — an infinite
    // loop blows any limit — while a stalled disk no longer reports a product
    // failure that is not one.
    testTimeout: 20_000,
    hookTimeout: 20_000,
  },
});
