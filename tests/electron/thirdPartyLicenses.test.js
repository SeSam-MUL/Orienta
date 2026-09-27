/**
 * The built interface must carry the notices of what it inlines.
 *
 * Measured before this existed: 316 production packages bundled, zero
 * `@license` banners surviving minification, no notice file in dist/. MIT's
 * one condition is that the copyright and permission notice accompany every
 * copy; the generator below is how the release meets it.
 */
import { mkdtempSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

import {
  assertPermissive,
  buildThirdPartyLicenses,
  collectPackages,
  FRONTEND_DIR,
} from '../../frontend/scripts/third_party_licenses.mjs';

const lockPath = path.join(FRONTEND_DIR, 'package-lock.json');
const nodeModulesDir = path.join(FRONTEND_DIR, 'node_modules');

describe('third-party licence notices of the bundled frontend', () => {
  const packages = collectPackages({ lockPath, nodeModulesDir });
  const text = buildThirdPartyLicenses({ lockPath, nodeModulesDir });

  it('lists the production packages and none of the dev tooling', () => {
    const names = new Set(packages.map((p) => p.name));
    for (const shipped of ['react', 'react-dom', 'plotly.js', 'mapbox-gl', 'axios', 'i18next', 'three', 'vanta']) {
      expect(names.has(shipped), `${shipped} is a runtime dependency and must be listed`).toBe(true);
    }
    for (const dev of ['vitest', 'eslint', 'vite', '@vitejs/plugin-react', 'electron', 'electron-builder']) {
      expect(names.has(dev), `${dev} is dev tooling and must not be listed`).toBe(false);
    }
  });

  it('reproduces the notice text, not just a license name', () => {
    // mapbox-gl is BSD-3: clause 2 requires the notice in the materials
    // provided with a binary distribution.
    expect(text).toMatch(/^mapbox-gl@1\./m);
    expect(text).toMatch(/Mapbox/);
    // MIT permission notice, once per MIT package
    const mitPackages = packages.filter((p) => p.license === 'MIT').length;
    const grants = (text.match(/Permission is hereby granted, free of charge/g) || []).length;
    expect(mitPackages).toBeGreaterThan(100);
    expect(grants).toBeGreaterThanOrEqual(Math.floor(mitPackages * 0.9));
  });

  it('carries a declared license for every package', () => {
    const undeclared = packages.filter((p) => !p.license).map((p) => `${p.name}@${p.version}`);
    expect(undeclared).toEqual([]);
  });

  it('is deterministic', () => {
    expect(buildThirdPartyLicenses({ lockPath, nodeModulesDir })).toBe(text);
    expect(text).not.toMatch(/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/);
  });

  it('refuses a copyleft or non-commercial production dependency', () => {
    const dir = mkdtempSync(path.join(os.tmpdir(), 'orienta-lock-'));
    const fakeLock = path.join(dir, 'package-lock.json');
    writeFileSync(fakeLock, JSON.stringify({
      lockfileVersion: 3,
      packages: {
        '': { name: 'x', version: '0.0.0' },
        'node_modules/fine': { version: '1.0.0', license: 'MIT' },
        'node_modules/viral': { version: '2.0.0', license: 'GPL-3.0-only' },
        'node_modules/tooling': { version: '3.0.0', license: 'AGPL-3.0', dev: true },
      },
    }));
    const packages = collectPackages({ lockPath: fakeLock, nodeModulesDir: path.join(dir, 'node_modules') });
    expect(packages.map((p) => p.name)).toEqual(['fine', 'viral']);
    expect(() => assertPermissive(packages)).toThrow(/viral@2\.0\.0: GPL-3\.0-only/);
    expect(() => buildThirdPartyLicenses({ lockPath: fakeLock, nodeModulesDir: path.join(dir, 'node_modules') }))
      .toThrow(/copyleft/);
  });
});
