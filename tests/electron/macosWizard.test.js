/**
 * What the setup wizard shows a Mac.
 *
 * The M5 tester's first screen on 2026-09-25, on a German Mac:
 *
 *   - a choice between "Fassung für die Grafikkarte" (greyed out) and
 *     "Fassung für den Prozessor", on a computer that cannot have an NVIDIA
 *     card at all,
 *   - the explanation under it in English — "Orienta uses the processor on
 *     this Mac (7.0 GB)..." — under a German heading,
 *   - that option labelled "about 2.5 GB" while the sentence below it said
 *     7.0 GB: two numbers for the same install, on the same screen,
 *   - and a step in the progress list called "Grafikkarte wird geprüft",
 *     which a CPU install never runs.
 *
 * The real folder afterwards was 9.89 GB.
 */
import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const requireCjs = createRequire(import.meta.url);
const installer = requireCjs('../../electron/setup/installer.js');
const strings = requireCjs('../../electron/strings.js');
const REPO = path.resolve(import.meta.dirname, '..', '..');

const GB = 1024 ** 3;
const read = (rel) => fs.readFileSync(path.join(REPO, rel), 'utf8');

// ---------------------------------------------------------------------------
// the number
// ---------------------------------------------------------------------------

describe('how much room the wizard asks for', () => {
  it('uses the measured macOS figure, not the Windows one', () => {
    // 9.89 GB measured in ~/Library/Application Support/Orienta, plus room for
    // the package cache that exists alongside the environment during the
    // transaction.
    expect(installer.cpuBytesNeeded('darwin')).toBe(Math.round(12 * GB));
  });

  it('leaves Windows and Linux where they were', () => {
    expect(installer.cpuBytesNeeded('win32')).toBe(Math.round(2.5 * GB));
    expect(installer.cpuBytesNeeded('linux')).toBe(Math.round(2.5 * GB));
  });

  it('is never below what the tester actually needed', () => {
    // The floor that shipped in v0.4.5 was 7 GB — under the measurement.
    expect(installer.cpuBytesNeeded('darwin')).toBeGreaterThan(9.89 * GB);
  });

  it('is the number the main process hands the wizard', () => {
    // The label said 2.5 GB because main.js shipped the bare Windows constant
    // while the sentence below it called the function. Same shape as
    // chooseRecommendation answering for the host instead of its argument.
    const main = read('electron/main.js');
    expect(main).toMatch(/cpuBytesNeeded\(process\.platform\)/);
    // Positive control: the bare constant must not be what gets exported to
    // the renderer any more.
    expect(main).not.toMatch(/require\('\.\/setup\/installer'\);[\s\S]{0,200}?CPU_BYTES_NEEDED\s*\}/);
  });
});

// ---------------------------------------------------------------------------
// the sentence
// ---------------------------------------------------------------------------

describe('the reason the wizard gives', () => {
  const onMac = (freeGb) => installer.chooseRecommendation(
    { present: false }, freeGb * GB, true, 'darwin',
  );

  it('carries a code, so it can be said in the user language', () => {
    const r = onMac(500);
    expect(r.reasonCode).toBe('macosCpuOnly');
    expect(r.reasonValues.size).toBe('12.0');
  });

  it('keeps the English prose as a fallback', () => {
    // An older renderer, and the log, still get a sentence.
    expect(onMac(500).reason).toMatch(/NVIDIA/);
  });

  it('codes the out-of-space refusal too', () => {
    const r = onMac(3);
    expect(r.blocked).toBe(true);
    expect(r.reasonCode).toBe('tooLittleSpace');
    expect(r.reasonValues).toMatchObject({ needed: '12.0' });
  });

  it('codes the unreadable-disk refusal', () => {
    const r = installer.chooseRecommendation({ present: false }, NaN, false, 'darwin');
    expect(r.reasonCode).toBe('freeSpaceUnknown');
  });

  it('has every code it can emit translated into all four languages', () => {
    // The defect this project keeps meeting: two sides each correct alone.
    // A code with no string renders as the key itself, in front of a user.
    const src = read('electron/setup/installer.js');
    const codes = new Set();
    for (const m of src.matchAll(/reasonCode:\s*(.+)/g)) {
      for (const q of m[1].matchAll(/'([a-zA-Z][\w]*)'/g)) codes.add(q[1]);
    }
    expect(codes.size).toBeGreaterThan(0);          // the scan must find something
    const locales = JSON.parse(read('electron/setup/locales.json'));
    for (const lang of ['en', 'de', 'ja', 'zh']) {
      for (const code of codes) {
        expect(locales[lang].reasons?.[code], `${lang} is missing reasons.${code}`)
          .toBeTruthy();
      }
    }
  });
});

// ---------------------------------------------------------------------------
// the screen
// ---------------------------------------------------------------------------

describe('the choice, and the steps', () => {
  it('offers no choice where the other option cannot exist', () => {
    const setup = read('electron/setup/setup.js');
    expect(setup).toMatch(/const noChoice = probe\.reasonCode === 'macosCpuOnly'/);
    expect(setup).toMatch(/group\.hidden = noChoice/);
    // The radios stay in the DOM: chosenMode() reads them.
    const html = read('electron/setup/index.html');
    expect(html).toMatch(/id="optCpu"/);
    expect(html).toMatch(/id="singleOption"/);
  });

  it('does not list a graphics-card check for an install that has none', () => {
    const setup = read('electron/setup/setup.js');
    expect(setup).toMatch(/function stepsFor\(mode\)/);
    expect(setup).toMatch(/filter\(\(s\) => s !== 'verify'\)/);
    // And the renderer must actually use it — the list was a fixed array.
    expect(setup).toMatch(/stepsFor\(lastMode\)/);
  });
});

// ---------------------------------------------------------------------------
// the language
// ---------------------------------------------------------------------------

describe('which language the wizard opens in', () => {
  it('believes the system over the bundle', () => {
    // app.getLocale() negotiates against the localizations the .app declares,
    // and Orienta declares none — so it answers "en" on a German Mac.
    expect(strings.preferredLocale(['de-AT', 'en-US'], 'en-US')).toBe('de-AT');
  });

  it('falls back when the system list is empty or absent', () => {
    expect(strings.preferredLocale([], 'ja-JP')).toBe('ja-JP');
    expect(strings.preferredLocale(null, 'zh-CN')).toBe('zh-CN');
    expect(strings.preferredLocale(undefined, undefined)).toBe('en');
  });

  it('still reduces to a language this shell has strings for', () => {
    expect(strings.shellLanguage(strings.preferredLocale(['de-AT'], 'en'))).toBe('de');
    expect(strings.shellLanguage(strings.preferredLocale(['fr-FR'], 'en'))).toBe('en');
  });

  it('is what main.js asks for', () => {
    const main = read('electron/main.js');
    expect(main).toMatch(/getPreferredSystemLanguages/);
    // The hazard is not how many getLocale() calls the file has — it is
    // uiLocale() calling ITSELF. An earlier edit did exactly that with a
    // blanket replace, and `node --check` passes a stack overflow happily. So
    // read the helper's own body instead of counting across 1500 lines.
    const start = main.indexOf('function uiLocale()');
    expect(start).toBeGreaterThan(-1);
    const helper = main.slice(start, main.indexOf('\n}', start));
    expect(helper).toMatch(/app\.getLocale\(\)/);              // a real fallback
    expect(helper.slice('function uiLocale()'.length)).not.toMatch(/uiLocale\(\)/);
  });
});

// ---------------------------------------------------------------------------
// what the progress line shows
// ---------------------------------------------------------------------------

describe('micromamba boilerplate', () => {
  const WARNING = 'warning  libmamba Security Warning: This transaction includes '
    + 'executing package scripts (pre/post-link/unlink) if present. These scripts '
    + 'can contain arbitrary code. Please ensure you trust the package sources.';

  it('does not go on the wizard screen', () => {
    expect(installer.isSetupNoise(WARNING)).toBe(true);
  });

  it('is still written to the log', () => {
    // The filter sits on emit(), never on record(). Read the handler's body
    // rather than matching an exact layout: the claim is that record() runs
    // unconditionally and only emit() is gated, and that stays true through
    // reindentation or an added comment.
    const src = read('electron/setup/installer.js');
    const handler = src.slice(src.indexOf('onLine: (line) => {'));
    const body = handler.slice(0, handler.indexOf('},'));
    expect(body).toMatch(/record\(line\);/);
    expect(body).toMatch(/if \(!isSetupNoise\(line\)\)[\s\S]*?emit\(/);
    expect(body.indexOf('record(line)')).toBeLessThan(body.indexOf('isSetupNoise'));
  });

  it('lets everything else through, including real trouble', () => {
    for (const line of [
      'error    libmamba Could not solve for environment specs',
      'Downloading pytorch-2.11.0',
      'Transaction finished',
      'warning: something from pip',
    ]) {
      expect(installer.isSetupNoise(line), line).toBe(false);
    }
  });
});

// ---------------------------------------------------------------------------
// which screen the app opens on
// ---------------------------------------------------------------------------

describe('the window after the wizard restarts the app', () => {
  const platform = requireCjs('../../electron/platform.js');
  // A three-monitor desk like the tester's: the laptop screen plus two to the
  // right of it.
  const DISPLAYS = [
    { workArea: { x: 0, y: 0, width: 1800, height: 1100 } },
    { workArea: { x: 1800, y: 0, width: 2560, height: 1400 } },
    { workArea: { x: 4360, y: 0, width: 2560, height: 1400 } },
  ];
  const SIZE = { width: 1400, height: 900 };

  it('opens on the screen the wizard was on', () => {
    const at = platform.windowPositionFor(DISPLAYS, { x: 3000, y: 700 }, SIZE);
    expect(at.x).toBe(1800 + Math.round((2560 - 1400) / 2));
    expect(at.y).toBe(Math.round((1400 - 900) / 2));
  });

  it('centres on the third screen just as well', () => {
    expect(platform.windowPositionFor(DISPLAYS, { x: 5000, y: 100 }, SIZE).x)
      .toBeGreaterThanOrEqual(4360);
  });

  it('lets the OS decide when that monitor is gone', () => {
    // Unplugged between the wizard and the restart: no coordinates, same
    // behaviour as before this existed.
    expect(platform.windowPositionFor(DISPLAYS, { x: 9999, y: 9999 }, SIZE)).toBe(null);
  });

  it('never pushes a window off the top-left of its display', () => {
    const small = [{ workArea: { x: 100, y: 50, width: 800, height: 600 } }];
    const at = platform.windowPositionFor(small, { x: 200, y: 100 }, SIZE);
    expect(at).toEqual({ x: 100, y: 50 });
  });

  it('says nothing when it was told nothing', () => {
    expect(platform.windowPositionFor(DISPLAYS, null, SIZE)).toBe(null);
    expect(platform.windowPositionFor(DISPLAYS, { x: NaN, y: 0 }, SIZE)).toBe(null);
    expect(platform.windowPositionFor(null, { x: 1, y: 1 }, SIZE)).toBe(null);
  });

  it('is carried across the restart as an argument, not a file', () => {
    const main = read('electron/main.js');
    expect(main).toMatch(/--orienta-window-at=/);
    // And the old one is dropped, so restarts do not accumulate them.
    expect(main).toMatch(/filter\(\(a\) => !String\(a\)\.startsWith\('--orienta-window-at='\)\)/);
  });
});
