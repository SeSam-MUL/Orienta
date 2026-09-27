/**
 * Write the third-party notices of the built user interface.
 *
 * `vite build` inlines every production npm package into `dist/assets/*.js`
 * and drops their license headers on the way (measured: zero `@license`
 * banners survive in 49 bundled files). MIT, ISC and BSD all require the
 * copyright notice and the permission text to accompany every copy, so this
 * script reads `package-lock.json`, takes each production package's own
 * LICENSE file from `node_modules`, and writes them all to
 * `dist/THIRD-PARTY-LICENSES.txt`. NOTICE.md at the repository root points at
 * that file, and the release build refuses a package that lacks it.
 *
 * Deterministic: no timestamp, packages sorted by name, so two builds of the
 * same lock file produce the same bytes.
 *
 *     node scripts/third_party_licenses.mjs            # after `vite build`
 */
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const FRONTEND_DIR = path.resolve(HERE, '..');
export const OUTPUT_NAME = 'THIRD-PARTY-LICENSES.txt';

const LICENSE_FILE = /^(licen[cs]e|copying)(\.|-|$)/i;

/**
 * Packages whose published tarball carries neither a `license` field nor a
 * license file. The license is stated only in the upstream repository; the
 * text is reproduced here so the notice file is complete without it. Keep
 * this list short and every entry sourced.
 */
const UNDECLARED_UPSTREAM = {
  '@mapbox/jsonlint-lines-primitives': {
    license: 'MIT',
    source: 'https://github.com/mapbox/jsonlint (README: "MIT License"; a fork of zaach/jsonlint, MIT)',
    text: [
      'Copyright (C) 2012 Zachary Carter',
      '',
      'Permission is hereby granted, free of charge, to any person obtaining a copy',
      'of this software and associated documentation files (the "Software"), to deal',
      'in the Software without restriction, including without limitation the rights',
      'to use, copy, modify, merge, publish, distribute, sublicense, and/or sell',
      'copies of the Software, and to permit persons to whom the Software is',
      'furnished to do so, subject to the following conditions:',
      '',
      'The above copyright notice and this permission notice shall be included in',
      'all copies or substantial portions of the Software.',
      '',
      'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR',
      'IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,',
      'FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE',
      'AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER',
      'LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,',
      'OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN',
      'THE SOFTWARE.',
    ].join('\n'),
  },
};

/** When a package ships a license file but no `license` field, name the license from the text. */
function inferLicense(text) {
  if (!text) return null;
  if (/Permission is hereby granted, free of charge/i.test(text)) return 'MIT (from the license file)';
  if (/Permission to use, copy, modify,? and\/or distribute/i.test(text)) return 'ISC (from the license file)';
  if (/Redistribution and use in source and binary forms/i.test(text)) return 'BSD (from the license file)';
  return 'see the license text below';
}

function readJson(file) {
  return JSON.parse(readFileSync(file, 'utf8'));
}

/** The license file(s) a package ships, concatenated; null when it ships none. */
function licenseText(pkgDir) {
  if (!existsSync(pkgDir)) return null;
  const files = readdirSync(pkgDir, { withFileTypes: true })
    .filter((d) => d.isFile() && LICENSE_FILE.test(d.name))
    .map((d) => d.name)
    .sort();
  if (files.length === 0) return null;
  return files
    .map((f) => readFileSync(path.join(pkgDir, f), 'utf8').replace(/\r\n/g, '\n').trim())
    .join('\n\n');
}

/**
 * Every package the production bundle can contain: the lock file's non-dev
 * entries. `dev: true` marks devDependencies and their subtree; everything
 * else is reachable from `dependencies` and may be inlined by the bundler.
 */
export function collectPackages({ lockPath, nodeModulesDir }) {
  const lock = readJson(lockPath);
  if (lock.lockfileVersion !== 3) {
    throw new Error(`expected package-lock.json lockfileVersion 3, got ${lock.lockfileVersion}`);
  }
  const out = [];
  for (const [key, entry] of Object.entries(lock.packages)) {
    if (!key || entry.dev) continue;
    const name = key.slice(key.lastIndexOf('node_modules/') + 'node_modules/'.length);
    const pkgDir = path.join(nodeModulesDir, key.replace(/^node_modules\//, ''));
    let declared = entry.license || null;
    if (!declared && existsSync(path.join(pkgDir, 'package.json'))) {
      const pj = readJson(path.join(pkgDir, 'package.json'));
      declared = typeof pj.license === 'string' ? pj.license : (pj.license?.type ?? null);
    }
    let text = licenseText(pkgDir);
    let source = null;
    if (!declared && !text && UNDECLARED_UPSTREAM[name]) {
      ({ license: declared, text, source } = UNDECLARED_UPSTREAM[name]);
    }
    if (!declared) declared = inferLicense(text);
    out.push({
      name,
      version: entry.version ?? '',
      license: declared,
      text,
      source,
      installed: existsSync(pkgDir),
    });
  }
  // code-point order, not localeCompare: the header promises byte-identical
  // output for the same lock file, and collation depends on the Node build
  const cmp = (x, y) => (x < y ? -1 : x > y ? 1 : 0);
  out.sort((a, b) => cmp(a.name, b.name) || cmp(a.version, b.version));
  return out;
}

/**
 * Licenses that would change what the whole may be distributed under, or
 * forbid commercial use. Everything in the bundle today is MIT/ISC/BSD-like;
 * a dependency that brings one of these in must be a deliberate decision,
 * not something the notice file quietly records.
 */
const NON_PERMISSIVE = /\b(A?GPL|LGPL|SSPL|EUPL|OSL|CC[- ]BY[- ]NC|BUSL|Commons Clause)\b/i;

export function assertPermissive(packages) {
  const offenders = packages
    .filter((p) => p.license && NON_PERMISSIVE.test(p.license))
    .map((p) => `${p.name}@${p.version}: ${p.license}`);
  if (offenders.length) {
    throw new Error(
      `production npm packages with a copyleft or non-commercial license:\n  ${offenders.join('\n  ')}\n`
      + 'Decide deliberately (NOTICE.md), then extend the allowance in third_party_licenses.mjs.',
    );
  }
}

export function renderNotices(packages) {
  const rule = '='.repeat(78);
  const lines = [
    'Third-party notices for the Orienta user interface (frontend/dist)',
    '',
    'The built interface inlines npm packages; every package the production',
    'dependency tree can reach is listed below. Each is used under its own',
    'license; the notices their authors require are reproduced here in full.',
    'Generated from frontend/package-lock.json by',
    'frontend/scripts/third_party_licenses.mjs. Orienta itself is GPL-3.0,',
    'see LICENSE and NOTICE.md at the repository root.',
    '',
    `${packages.length} packages`,
    '',
    'Summary (name@version: license as declared by the package)',
    '-'.repeat(78),
  ];
  for (const p of packages) lines.push(`${p.name}@${p.version}: ${p.license ?? 'not declared'}`);
  lines.push('', rule, '');
  for (const p of packages) {
    lines.push(`${p.name}@${p.version}`, `License: ${p.license ?? 'not declared in package.json'}`);
    if (p.source) lines.push(`(the published package states no license; taken from ${p.source})`);
    lines.push('');
    if (p.text) {
      lines.push(p.text);
    } else if (!p.installed) {
      lines.push('(package not present in node_modules at build time; see its package.json on npm)');
    } else {
      lines.push('(the package ships no license file; the license is the one declared above)');
    }
    lines.push('', rule, '');
  }
  return lines.join('\n') + '\n';
}

export function buildThirdPartyLicenses({
  lockPath = path.join(FRONTEND_DIR, 'package-lock.json'),
  nodeModulesDir = path.join(FRONTEND_DIR, 'node_modules'),
} = {}) {
  const packages = collectPackages({ lockPath, nodeModulesDir });
  assertPermissive(packages);
  return renderNotices(packages);
}

function main() {
  const dist = path.join(FRONTEND_DIR, 'dist');
  if (!existsSync(path.join(dist, 'index.html'))) {
    console.error(`${dist} has no index.html: run \`vite build\` first`);
    return 1;
  }
  const text = buildThirdPartyLicenses();
  mkdirSync(dist, { recursive: true });
  const target = path.join(dist, OUTPUT_NAME);
  writeFileSync(target, text, 'utf8');
  const count = (text.match(/^\d+ packages$/m) || ['?'])[0];
  console.log(`${path.relative(FRONTEND_DIR, target)}: ${count}`);
  return 0;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.exitCode = main();
}
