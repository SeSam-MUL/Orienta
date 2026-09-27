/**
 * Does an installer that carries a newer package actually replace the program
 * files?
 *
 * Until 2026-09-27 it did not, and nothing said so. Installing 0.4.6 over 0.4.4
 * gave a 0.4.6 shell, a 0.4.6 `resources/orienta-runtime-v0.4.6.zip`, a registry
 * entry reading 0.4.6 — and `runtime/VERSION` still at v0.4.4, with the backend
 * importing that tree and About reporting it. `installedRuntime()` asks only
 * whether `runtime/VERSION` is a file, never which version it names, and the one
 * thing that unpacked a package was the first-install wizard.
 *
 * What is NOT tested here, on purpose: whether the unpack protects the crystal
 * library. That is `apply_update.py`'s promise and `tests/test_apply_update.py`
 * proves it, among others in
 * `test_the_database_survives_even_if_a_manifest_names_it` and
 * `test_a_package_that_writes_into_database_is_refused_whole`. This layer's job
 * is to DECIDE and to PARK; re-asserting the applier's guarantees here would
 * duplicate them in the copy nobody would update.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';

import {
  tagOf,
  bundledPackage,
  installedTag,
  alreadyParked,
  updateDecision,
  digestFrom,
  parkBundled,
  readVerdict,
} from '../../electron/bundled_update.js';

let tmp;
const at = (...parts) => path.join(tmp, ...parts);

function writePackage(dir, tag, body = 'a package') {
  fs.mkdirSync(dir, { recursive: true });
  const name = `orienta-runtime-${tag}.zip`;
  const zip = path.join(dir, name);
  fs.writeFileSync(zip, body);
  const digest = createHash('sha256').update(body).digest('hex');
  fs.writeFileSync(`${zip}.sha256`, `${digest}  ${name}\n`);
  return { zip, name, digest };
}

beforeEach(() => {
  tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'orienta-bundled-'));
});
afterEach(() => {
  fs.rmSync(tmp, { recursive: true, force: true });
});

describe('finding the package that shipped with the shell', () => {
  it('takes the zip only when its checksum file is there too', () => {
    const res = at('resources');
    writePackage(res, 'v0.4.6');
    expect(bundledPackage(res).tag).toBe('v0.4.6');

    fs.rmSync(path.join(res, 'orienta-runtime-v0.4.6.zip.sha256'));
    // The applier verifies the digest. A package we cannot state one for is one
    // it would refuse, so it is not a package as far as this layer is concerned.
    expect(bundledPackage(res)).toBeNull();
  });

  it('takes the NEWEST of two, not the first the directory lists', () => {
    const res = at('resources');
    writePackage(res, 'v0.9.0');
    writePackage(res, 'v0.10.0');
    // Lexically "v0.10.0" sorts BELOW "v0.9.0"; the version order is the point.
    expect(bundledPackage(res).tag).toBe('v0.10.0');
  });

  it('is null, not an exception, when there is no resources directory', () => {
    expect(bundledPackage(at('nowhere'))).toBeNull();
    expect(bundledPackage(undefined)).toBeNull();
  });

  it('ignores a file that merely looks similar', () => {
    const res = at('resources');
    fs.mkdirSync(res, { recursive: true });
    fs.writeFileSync(path.join(res, 'orienta-runtime.zip'), 'x');
    fs.writeFileSync(path.join(res, 'orienta-runtime.zip.sha256'), 'x');
    expect(tagOf('orienta-runtime.zip')).toBeNull();
    expect(bundledPackage(res)).toBeNull();
  });
});

describe('what the installation on disk says it is', () => {
  it('reads the first line and trims it', () => {
    fs.mkdirSync(at('home', 'runtime'), { recursive: true });
    fs.writeFileSync(at('home', 'runtime', 'VERSION'), 'v0.4.4\r\nignored\n');
    expect(installedTag(at('home'))).toBe('v0.4.4');
  });

  it('is null when there is no runtime, and never throws', () => {
    expect(installedTag(at('home'))).toBeNull();
    expect(installedTag(undefined)).toBeNull();
  });
});

describe('the decision', () => {
  const bundled = { tag: 'v0.4.6' };

  it('parks when the bundle is newer — the bug this exists for', () => {
    const d = updateDecision({ bundled, installed: 'v0.4.4' });
    expect(d.action).toBe('park');
    expect(d.reason).toMatch(/v0\.4\.6.*newer.*v0\.4\.4/);
  });

  it('does nothing when they are the same version', () => {
    expect(updateDecision({ bundled, installed: 'v0.4.6' }).action).toBe('none');
  });

  it('never downgrades, and says why in the log', () => {
    const d = updateDecision({ bundled: { tag: 'v0.4.4' }, installed: 'v0.4.6' });
    expect(d.action).toBe('none');
    expect(d.reason).toMatch(/newer than the bundled/);
    expect(d.reason).toMatch(/leaving it alone/);
  });

  it('respects version order across the 0.9 -> 0.10 boundary', () => {
    expect(updateDecision({ bundled: { tag: 'v0.10.0' }, installed: 'v0.9.0' }).action)
      .toBe('park');
    expect(updateDecision({ bundled: { tag: 'v0.9.0' }, installed: 'v0.10.0' }).action)
      .toBe('none');
  });

  it('leaves a first install to the wizard', () => {
    const d = updateDecision({ bundled, installed: null });
    expect(d.action).toBe('none');
    expect(d.reason).toMatch(/wizard/);
  });

  it('applies what is already parked instead of parking over it', () => {
    // A package parked by something else is the one the user is waiting for;
    // overwriting the record would lose it.
    const d = updateDecision({ bundled, installed: 'v0.4.4', parked: true });
    expect(d.action).toBe('apply');
  });

  it('does nothing when the shell carries no package at all', () => {
    expect(updateDecision({ bundled: null, installed: 'v0.4.4' }).action).toBe('none');
  });
});

describe('parking it for the applier', () => {
  it('copies the zip and writes the record the applier reads', () => {
    const res = at('resources');
    const { digest, name } = writePackage(res, 'v0.4.6', 'the real bytes');
    const pkg = bundledPackage(res);
    fs.mkdirSync(at('home'), { recursive: true });

    const record = parkBundled({ home: at('home'), bundled: pkg });

    expect(record).toMatchObject({ tag: 'v0.4.6', file: name, sha256: digest });
    // Copied, not referenced: the applier deletes what it discards, and the
    // original lives in the installation directory.
    expect(fs.readFileSync(at('home', 'pending', name), 'utf8')).toBe('the real bytes');
    expect(fs.existsSync(path.join(res, name))).toBe(true);

    const onDisk = JSON.parse(fs.readFileSync(at('home', 'pending.json'), 'utf8'));
    expect(onDisk).toEqual({ tag: 'v0.4.6', file: name, sha256: digest });
    expect(alreadyParked(at('home'))).toBe(true);
  });

  it('refuses a checksum file that does not hold a sha256', () => {
    const res = at('resources');
    writePackage(res, 'v0.4.6');
    fs.writeFileSync(path.join(res, 'orienta-runtime-v0.4.6.zip.sha256'), 'not a digest\n');
    expect(() => digestFrom(path.join(res, 'orienta-runtime-v0.4.6.zip.sha256')))
      .toThrow(/sha256/);
  });

  it('reports the digest the file states, lower-cased', () => {
    const f = at('sum');
    fs.writeFileSync(f, `${'A'.repeat(64)}  something.zip\n`);
    expect(digestFrom(f)).toBe('a'.repeat(64));
  });
});

describe('the shell actually does it, and does it first', () => {
  // The decision layer above is worthless if nobody calls it, and that is
  // precisely how the original defect existed: `apply_update.py` had 33 tests
  // and a docstring beginning "Apply a parked runtime package, BEFORE the
  // backend starts" -- and no line in electron/ ever ran it outside the
  // first-install wizard. `pendingUpdate()` was exported and had no caller
  // either. A unit test of a function nobody invokes is a green light on a
  // disconnected wire, so this reads the wire.
  const main = fs.readFileSync(
    path.join(process.cwd(), '..', 'electron', 'main.js'), 'utf8');

  it('awaits the update before it spawns the backend', () => {
    const update = main.indexOf('await applyBundledUpdate(');
    const spawn = main.indexOf('startBackend(plan.python');
    expect(update).toBeGreaterThan(-1);
    expect(spawn).toBeGreaterThan(-1);
    // uvicorn imports `runtime/`. Replacing those files under a running process
    // is not an option, and starting first would serve the old version for the
    // whole session.
    expect(update).toBeLessThan(spawn);
  });

  it('does not start the backend when the tree is left half-updated', () => {
    // `ok: false` means dirty: part one release, part the other. Starting
    // uvicorn against that is the one outcome worse than not starting.
    expect(main).toMatch(/if\s*\(!update\.ok\)\s*\{[\s\S]{0,200}?app\.quit\(\)/);
  });

  it('throws the renderer cache away once the update has been applied', () => {
    // Measured on 2026-09-27: after v0.4.4 -> v0.4.6, About read v0.4.6 from the
    // new backend while the window rendered the 0.4.4 index.html and its hashed
    // chunks from the renderer's HTTP cache. The backend now sends `no-store`,
    // but that only governs responses fetched after it shipped -- the copy an
    // older version already wrote into this cache carries no Cache-Control at
    // all, and Chromium's freshness heuristic serves it without asking. So the
    // upgrade itself is the one case the header cannot reach.
    const applied = main.indexOf('Runtime update: applied');
    const clear = main.indexOf('await clearRendererCache()');
    const load = main.indexOf('mainWindow?.loadURL(`http://127.0.0.1:');
    expect(applied).toBeGreaterThan(-1);
    expect(clear).toBeGreaterThan(-1);
    expect(load).toBeGreaterThan(-1);
    expect(clear).toBeGreaterThan(applied);   // only after a real update
    expect(clear).toBeLessThan(load);         // and before the window loads
  });

  it('clears the cache only after an update, not on every start', () => {
    // A cold start already costs 30-40 s; re-fetching several megabytes of
    // chunks on every launch would buy nothing.
    const calls = main.match(/clearRendererCache\(\)/g) || [];
    // One definition, one call site.
    expect(calls.length).toBe(2);
    const guard = main.indexOf('if (verdict.applied)');
    expect(guard).toBeGreaterThan(-1);
    expect(main.indexOf('await clearRendererCache()')).toBeGreaterThan(guard);
  });

  it('runs the audited applier rather than unpacking the zip itself', () => {
    expect(main).toMatch(/apply_update\.py/);
    // No second extractor in the shell: the member checks, the Database refusal
    // and prune-before-extract live in one file and must stay there.
    expect(main).not.toMatch(/\bextract_only\b/);
  });
});

describe('reading the verdict', () => {
  it('passes the applier three outcomes through unchanged', () => {
    const f = at('v.json');
    fs.writeFileSync(f, JSON.stringify({ applied: true, tag: 'v0.4.6' }));
    expect(readVerdict({ file: f, exitCode: 0 })).toMatchObject({ applied: true, tag: 'v0.4.6' });

    fs.writeFileSync(f, JSON.stringify({ dirty: true, error: 'extraction failed' }));
    expect(readVerdict({ file: f, exitCode: 1 }))
      .toMatchObject({ dirty: true, error: 'extraction failed' });

    fs.writeFileSync(f, JSON.stringify({ terminal: true, error: 'corrupt' }));
    expect(readVerdict({ file: f, exitCode: 0 }))
      .toMatchObject({ terminal: true, dirty: false });
  });

  it('believes exit 0 when no verdict was written', () => {
    // Exit 0 IS the applier's promise that the shell may proceed.
    expect(readVerdict({ file: at('missing.json'), exitCode: 0 }).dirty).toBe(false);
  });

  it('calls a missing verdict after a failure dirty, because we cannot know', () => {
    const v = readVerdict({ file: at('missing.json'), exitCode: 1 });
    expect(v.dirty).toBe(true);
    expect(v.error).toMatch(/no readable verdict/);
  });

  it('treats unreadable JSON after a failure the same way', () => {
    const f = at('broken.json');
    fs.writeFileSync(f, '{ not json');
    expect(readVerdict({ file: f, exitCode: 1 }).dirty).toBe(true);
  });
});
