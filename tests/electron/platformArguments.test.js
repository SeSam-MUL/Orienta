/**
 * Every test that asks a platform question must SAY which platform.
 *
 * `electron/platform.js` and `electron/setup/installer.js` take the platform
 * as an argument that falls back to `process.platform`. That fallback is for
 * production, where the host IS the answer. In a test it means something
 * else: the assertion silently becomes "on whatever machine happens to run
 * this", and a pinned Windows contract stops being pinned anywhere else.
 *
 * Nine such lines were found on 2026-09-23, and only because the suite ran
 * off Windows for the first time — `lockFileFor('gpu')` instead of
 * `lockFileFor('gpu', 'win32')`, `chooseRecommendation(...)` without its
 * platform, `path.isAbsolute` on a Windows path. They did not skip on the
 * macOS runner; they FAILED, and the port would have been blamed. Nothing
 * stopped a tenth from being written the next day. This is that tenth's
 * guard.
 *
 * What it does NOT do: it reads source text, so a call reached through a
 * variable (`const f = lockFileFor; f('gpu')`) is invisible to it, and so is
 * one built inside a template interpolation that this scanner's own string
 * handling mangles. It catches the shape that has actually occurred.
 */
import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

const REPO = path.resolve(import.meta.dirname, '..', '..');
const SOURCES = ['electron/platform.js', 'electron/setup/installer.js'];
const SUITE = path.join(REPO, 'tests', 'electron');
const SELF = 'platformArguments.test.js';

/**
 * Calls that are host-dependent on purpose. An entry has to say why, and it
 * is one line in a diff — the point is that adding one is a decision somebody
 * makes and a reviewer sees, not a thing that happens by itself.
 */
const DELIBERATE = [
  {
    file: 'installerSteps.test.js',
    name: 'interpreterIn',
    why: 'the test is titled "is the same answer the installer uses on this host": '
       + 'it asserts pythonExeIn and interpreterIn agree, and both default to the '
       + 'host, so the claim holds on any machine. Naming a platform would test '
       + 'something else.',
  },
];

// ---------------------------------------------------------------------------
// reading JavaScript without running it
// ---------------------------------------------------------------------------

/**
 * Blank out comments and the text inside string literals.
 *
 * Both are necessary and for opposite reasons. Comments in this suite quote
 * the very calls they document (`// These read \`lockFileFor('gpu')\``), and
 * several tests pin source text by searching for it as a string
 * (`hasLine("if (platform.current() !== 'darwin') {")`) — neither is a call.
 *
 * Two details that cost a round each: a literal is replaced by a single `0`
 * and then blanks, because it must still COUNT as an argument — blank it
 * away entirely and `minDriverDisplay('win32')` reads as a bare call. And
 * `${...}` is kept, because a real call can sit inside an interpolation.
 * Lengths and newlines are preserved so reported line numbers are the real
 * ones.
 */
export function blankOutText(src) {
  const blank = (ch) => (ch === '\n' ? '\n' : ' ');
  let out = '';
  let i = 0;
  while (i < src.length) {
    const ch = src[i];
    if (ch === '/' && src[i + 1] === '/') {
      while (i < src.length && src[i] !== '\n') { out += ' '; i += 1; }
    } else if (ch === '/' && src[i + 1] === '*') {
      out += '  '; i += 2;
      while (i < src.length && !(src[i] === '*' && src[i + 1] === '/')) { out += blank(src[i]); i += 1; }
      out += '  '; i += 2;
    } else if (ch === "'" || ch === '"') {
      out += '0'; i += 1;
      while (i < src.length && src[i] !== ch) {
        if (src[i] === '\\') { out += ' '; i += 1; }
        if (i < src.length) { out += blank(src[i]); i += 1; }
      }
      out += ' '; i += 1;
    } else if (ch === '`') {
      out += '0'; i += 1;
      while (i < src.length && src[i] !== '`') {
        if (src[i] === '\\') {
          out += ' '; i += 1;
          if (i < src.length) { out += blank(src[i]); i += 1; }
          continue;
        }
        if (src[i] === '$' && src[i + 1] === '{') {
          out += '  '; i += 2;
          let depth = 1;
          while (i < src.length && depth > 0) {
            if (src[i] === '{') depth += 1;
            else if (src[i] === '}') {
              depth -= 1;
              if (depth === 0) { out += ' '; i += 1; break; }
            }
            out += src[i]; i += 1;
          }
          continue;
        }
        out += blank(src[i]); i += 1;
      }
      out += ' '; i += 1;
    } else {
      out += src[i]; i += 1;
    }
  }
  return out;
}

/** Index just past the `)` that closes the `(` at `open`, or -1. */
function closingParen(src, open) {
  let depth = 0;
  for (let i = open; i < src.length; i += 1) {
    if (src[i] === '(') depth += 1;
    else if (src[i] === ')') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function splitTopLevel(text) {
  const parts = [];
  let depth = 0;
  let cur = '';
  for (const ch of text) {
    if ('([{'.includes(ch)) depth += 1;
    if (')]}'.includes(ch)) depth -= 1;
    if (ch === ',' && depth === 0) { parts.push(cur); cur = ''; } else cur += ch;
  }
  parts.push(cur);
  return parts;
}

/** name -> { index, file } for every function whose platform falls back. */
export function hostFallbackSignatures(repo = REPO) {
  const found = new Map();
  for (const rel of SOURCES) {
    const src = blankOutText(fs.readFileSync(path.join(repo, rel), 'utf8'));
    for (const m of src.matchAll(/function\s+([\w$]+)\s*\(/g)) {
      const open = m.index + m[0].length - 1;
      const close = closingParen(src, open);
      if (close < 0) continue;
      const params = splitTopLevel(src.slice(open + 1, close));
      const index = params.findIndex((p) => /=\s*process\s*\.\s*platform\b/.test(p));
      if (index >= 0) found.set(m[1], { index, file: rel });
    }
  }
  return found;
}

/** Calls in `src` that stop short of the platform argument. */
export function callsThatLeaveItToTheHost(src, signatures) {
  const clean = blankOutText(src);
  const hits = [];
  for (const [name, sig] of signatures) {
    const re = new RegExp(String.raw`\b${name}\s*\(`, 'g');
    for (const m of clean.matchAll(re)) {
      const open = m.index + m[0].length - 1;
      const close = closingParen(clean, open);
      if (close < 0) continue;
      const argText = clean.slice(open + 1, close);
      // A spread cannot be counted; saying nothing beats saying something wrong.
      if (argText.includes('...')) continue;
      const args = argText.trim() === '' ? [] : splitTopLevel(argText);
      if (args.length <= sig.index) {
        hits.push({ name, line: clean.slice(0, m.index).split('\n').length, needs: sig.index + 1, got: args.length });
      }
    }
  }
  return hits;
}

// ---------------------------------------------------------------------------
// the scanner's own controls
// ---------------------------------------------------------------------------

describe('the scanner itself', () => {
  it('finds the functions whose platform falls back to the host', () => {
    const sigs = hostFallbackSignatures();
    // Positive control. If platform.js is reshaped so these regexes stop
    // matching, this test fails LOUDLY rather than scanning nothing and
    // reporting a clean suite — a scan that cannot find a thing reports the
    // same as a scan that found nothing.
    for (const [name, index] of [
      ['canUseCuda', 0], ['lockFileFor', 1], ['interpreterIn', 1],
      ['driverMeetsCuda12', 1], ['killTree', 1], ['parseNvidiaSmi', 1],
      ['chooseRecommendation', 3],
    ]) {
      expect(sigs.has(name), `${name} is no longer recognised as taking a platform`).toBe(true);
      expect(sigs.get(name).index, `${name}: platform moved to another position`).toBe(index);
    }
    expect(sigs.size).toBeGreaterThanOrEqual(15);
  });

  it('bites when a call stops short of the platform', () => {
    const sigs = hostFallbackSignatures();
    const hits = callsThatLeaveItToTheHost("expect(lockFileFor('gpu')).toBe('x');", sigs);
    expect(hits.map((h) => h.name)).toEqual(['lockFileFor']);
    expect(hits[0]).toMatchObject({ needs: 2, got: 1 });
  });

  it('is quiet when the call names its platform', () => {
    const sigs = hostFallbackSignatures();
    expect(callsThatLeaveItToTheHost("expect(lockFileFor('gpu', 'win32')).toBe('x');", sigs)).toEqual([]);
  });

  it('does not mistake quoted source text for a call', () => {
    // The shape that made the first draft of this file cry wolf four times:
    // tests that pin source by searching for it as a string.
    const sigs = hostFallbackSignatures();
    const src = [
      '// These read `lockFileFor(\'gpu\')` and pinned the Windows answer.',
      'expect(hasLine("if (platform.current() !== \'darwin\') {")).toBe(true);',
      "expect(body).toContain('mode === \\'gpu\\' ? GPU : cpuBytesNeeded()');",
    ].join('\n');
    expect(callsThatLeaveItToTheHost(src, sigs)).toEqual([]);
  });

  it('still sees a call hidden in a template interpolation', () => {
    const sigs = hostFallbackSignatures();
    const hits = callsThatLeaveItToTheHost('const s = `lock is ${lockFileFor(mode)}`;', sigs);
    expect(hits.map((h) => h.name)).toEqual(['lockFileFor']);
  });

  it('counts a string argument as an argument', () => {
    // Blanking a literal away entirely made `minDriverDisplay('win32')` read
    // as a bare call and produced sixteen false alarms.
    const sigs = hostFallbackSignatures();
    expect(callsThatLeaveItToTheHost("expect(minDriverDisplay('win32')).toBe('x');", sigs)).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// the suite
// ---------------------------------------------------------------------------

describe('the electron suite', () => {
  it('names a platform everywhere it asks a platform question', () => {
    const sigs = hostFallbackSignatures();
    const files = fs.readdirSync(SUITE).filter((f) => f.endsWith('.test.js') && f !== SELF);
    expect(files.length).toBeGreaterThan(5);

    const unexplained = [];
    for (const file of files) {
      const src = fs.readFileSync(path.join(SUITE, file), 'utf8');
      for (const hit of callsThatLeaveItToTheHost(src, sigs)) {
        const allowed = DELIBERATE.some((d) => d.file === file && d.name === hit.name);
        if (!allowed) unexplained.push(`${file}:${hit.line}  ${hit.name}(...) — ${hit.got} argument(s), the platform is #${hit.needs}`);
      }
    }

    expect(unexplained, [
      'These tests ask a platform question and let the host answer it.',
      'Pass the platform you mean — platform.js takes it as an argument for',
      'exactly this reason. If a call is host-dependent on purpose, add it to',
      'DELIBERATE in this file with the reason.',
      '',
      ...unexplained,
    ].join('\n')).toEqual([]);
  });

  it('has no stale entry in the deliberate list', () => {
    // An allowlist that outlives its reason is a hole nobody can see. If the
    // call is fixed or gone, the entry has to go too.
    const sigs = hostFallbackSignatures();
    for (const entry of DELIBERATE) {
      const src = fs.readFileSync(path.join(SUITE, entry.file), 'utf8');
      const hits = callsThatLeaveItToTheHost(src, sigs).filter((h) => h.name === entry.name);
      expect(hits.length, `${entry.file}: ${entry.name} no longer leaves the platform to the host — drop the DELIBERATE entry`).toBeGreaterThan(0);
      expect(entry.why.length, `${entry.file}: ${entry.name} needs a reason`).toBeGreaterThan(40);
    }
  });
});
