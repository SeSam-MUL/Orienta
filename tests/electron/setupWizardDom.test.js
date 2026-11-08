/**
 * The setup wizard's page, RUN: index.html and setup.js loaded into jsdom with a
 * scripted `window.setupAPI`, and clicked.
 *
 * Source pins cannot tell whether a button is shown, or whether pressing it
 * lands on the right screen. The cases here are the data-folder ones: what the
 * user reads and can do after a refusal, and after a pick.
 */
import { describe, it, expect, afterEach } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
// jsdom lives in frontend/node_modules, which the tests directory cannot see.
const { JSDOM } = createRequire(
  path.resolve(import.meta.dirname, '..', '..', 'frontend', 'package.json'))('jsdom');

const HERE = path.resolve(import.meta.dirname, '..', '..', 'electron', 'setup');
const HTML = fs.readFileSync(path.join(HERE, 'index.html'), 'utf8')
  .replace(/<script src="setup\.js"><\/script>/, '');
const SCRIPT = fs.readFileSync(path.join(HERE, 'setup.js'), 'utf8');
const LOCALES = JSON.parse(fs.readFileSync(path.join(HERE, 'locales.json'), 'utf8'));

const PROBE = {
  gpu: { present: false }, recommendation: 'cpu', freeBytes: 50e9, freeKnown: true,
  gpuBytesNeeded: 9e9, cpuBytesNeeded: 2.5e9,
};

let dom = null;
afterEach(() => { if (dom) dom.window.close(); dom = null; });

const tick = () => new Promise((resolve) => setTimeout(resolve, 15));
async function settle() { for (let i = 0; i < 6; i++) await tick(); }

/** Start the wizard with the given stubbed API pieces; returns the document. */
async function start({ context = {}, run, pickHome, probe = PROBE } = {}) {
  dom = new JSDOM(HTML, { runScripts: 'outside-only', url: 'http://localhost/' });
  const { window } = dom;
  window.setupAPI = {
    context: async () => ({ reason: 'first-run', message: '', locale: 'en', home: 'C:\\H', releasesUrl: 'u', ...context }),
    locales: async () => LOCALES,
    probe: async () => probe,
    setLanguage: async () => true,
    onProgress: () => () => {},
    run: run || (async () => ({ ok: true })),
    pickHome: pickHome || (async () => null),
    releases: async () => ({ ok: true, releases: [] }),
    quit: async () => true, openLog: async () => true, relaunch: async () => true,
    cancel: async () => true, pickPackage: async () => null, selectRelease: async () => true,
  };
  window.eval(SCRIPT);
  await settle();
  return window.document;
}

const visible = (doc, id) => !doc.getElementById(id).hidden;
const shown = (doc) => ['choose', 'working', 'failed', 'done'].filter((id) => visible(doc, id));

async function failWith(result) {
  const doc = await start({ run: async () => result });
  doc.getElementById('installBtn').click();
  await settle();
  return doc;
}

describe('after a refusal about the data folder', () => {
  const refusal = {
    ok: false, code: 'homeNotEmptyDefault', values: { dir: 'C:\\Users\\a\\Orienta', entries: 'thesis.docx' },
    error: 'x',
  };

  it('shows the sentence with the folder and the entries filled in, not the placeholders', async () => {
    const doc = await failWith(refusal);
    expect(shown(doc)).toEqual(['failed']);
    const text = doc.getElementById('failedMessage').textContent;
    expect(text).toContain('C:\\Users\\a\\Orienta');
    expect(text).toContain('thesis.docx');
    expect(text).not.toContain('{{');
  });

  it.each(['homeNotEmpty', 'homeNotEmptyDefault', 'homeHoldsData'])(
    'offers Back for %s, and Back leads to the first screen where "change…" is', async (code) => {
      const doc = await failWith({ ...refusal, code });
      expect(visible(doc, 'backBtn')).toBe(true);
      expect(doc.getElementById('backBtn').textContent).toBe(LOCALES.en.back);
      doc.getElementById('backBtn').click();
      expect(shown(doc)).toEqual(['choose']);
      expect(visible(doc, 'pickHomeBtn')).toBe(true);
    });

  it('makes Back the main action there, not the Retry that only repeats the refusal', async () => {
    const doc = await failWith(refusal);
    expect(doc.getElementById('backBtn').classList.contains('primary')).toBe(true);
    expect(doc.getElementById('retryBtn').classList.contains('primary')).toBe(false);
  });

  it('is not offered for failures that are not about the folder', async () => {
    for (const code of ['network', 'pip', 'unexpected']) {
      const doc = await failWith({ ok: false, code, error: 'x' });
      expect(visible(doc, 'backBtn'), code).toBe(false);
      expect(doc.getElementById('retryBtn').classList.contains('primary'), code).toBe(true);
      dom.window.close(); dom = null;
    }
  });

  it('is written in the language on screen', async () => {
    const doc = await failWith(refusal);
    const select = doc.getElementById('langSelect');
    select.value = 'de';
    select.dispatchEvent(new dom.window.Event('change'));
    await settle();
    expect(doc.getElementById('backBtn').textContent).toBe(LOCALES.de.back);
    expect(doc.getElementById('failedMessage').textContent).toContain('thesis.docx');
  });
});

describe('a file in use or not allowed', () => {
  // The sentence ends "... or choose another data folder with change": without
  // Back that advice had no way to be followed from the error page.
  it.each(['inUse', 'noAccess'])('offers Back for %s, which leads to the first screen, but keeps Retry as the main action', async (code) => {
    const doc = await failWith({ ok: false, code, error: 'x' });
    expect(visible(doc, 'backBtn')).toBe(true);
    expect(doc.getElementById('retryBtn').classList.contains('primary')).toBe(true);
    expect(doc.getElementById('backBtn').classList.contains('primary')).toBe(false);
    doc.getElementById('backBtn').click();
    expect(shown(doc)).toEqual(['choose']);
  });
});

describe('the blocked page for a data folder that cannot be used', () => {
  async function blocked(kind) {
    return start({
      probe: {
        gpu: { present: false }, recommendation: 'cpu', blocked: true, freeBytes: 0, freeKnown: false,
        reason: 'detail', gpuBytesNeeded: 9e9, cpuBytesNeeded: 2.5e9,
        homeUnreachable: { dir: 'E:\\Orienta', pointerFile: 'C:\\R\\home.txt', kind, why: 'boom' },
      },
    });
  }

  it.each([
    ['drive', 'homeUnreachable', /not connected/],
    ['retire', 'homeRetireFailed', /could not update/],
    ['unreadable', 'homePointerUnreadable', /cannot read/],
    ['check', 'homeCheckFailed', /could not check where/],
  ])('kind %s says its own sentence (%s)', async (kind, code, pattern) => {
    const doc = await blocked(kind);
    const text = doc.getElementById('failedMessage').textContent;
    expect(text).toMatch(pattern);
    if (kind !== 'check') expect(text).toContain('C:\\R\\home.txt');
    if (kind === 'check') expect(text).toContain('boom');
    expect(text).not.toContain('{{');
    // only the drive case claims a drive is missing
    expect(/not connected/.test(text)).toBe(kind === 'drive');
  });
});

describe('the first screen', () => {
  it('says so, once, when the folder chosen earlier was deleted and the default is used', async () => {
    const doc = await start({
      context: { homeNotice: { code: 'staleFallback', dir: 'D:\\gone\\Orienta', previousFile: 'C:\\R\\home.txt.previous' } },
    });
    const note = doc.getElementById('homeNote');
    expect(note.hidden).toBe(false);
    expect(note.textContent).toContain('D:\\gone\\Orienta');
    expect(note.textContent).toContain('home.txt.previous');
    expect(note.textContent).not.toContain('{{');
  });

  it('says so in its own words when the recorded folder WAS the default one', async () => {
    const doc = await start({
      context: { homeNotice: { code: 'staleDefault', dir: 'C:\L\Orienta', previousFile: 'C:\R\home.txt.previous' } },
    });
    const text = doc.getElementById('homeNote').textContent;
    expect(text).toBe(LOCALES.en.homeNotices.staleDefault
      .replace('{{dir}}', 'C:\L\Orienta').replace('{{previous}}', 'C:\R\home.txt.previous'));
    expect(text).not.toMatch(/chosen earlier/);
  });

  it('shows a refused pick from its code, and keeps the folder in use', async () => {
    const doc = await start({
      pickHome: async () => ({
        dir: 'D:\\X\\Orienta', code: 'homeHoldsData', values: { dir: 'D:\\X\\Orienta', entries: 'Orienta' }, error: 'x',
      }),
    });
    doc.getElementById('pickHomeBtn').click();
    await settle();
    const note = doc.getElementById('homeNote');
    expect(note.textContent).toContain('Database');
    expect(note.textContent).toContain('D:\\X\\Orienta');
    expect(doc.getElementById('homeValue').textContent).toBe('C:\\H');
  });

  it('says "found ... and will use it" when a different, deeper folder was adopted', async () => {
    const doc = await start({
      pickHome: async () => ({ dir: 'D:\\X\\Orienta\\Orienta', adopted: true, restartNeeded: false }),
    });
    doc.getElementById('pickHomeBtn').click();
    await settle();
    expect(doc.getElementById('homeNote').textContent).toBe(
      LOCALES.en.homeAdopted.replace('{{dir}}', 'D:\\X\\Orienta\\Orienta'));
    expect(doc.getElementById('homeValue').textContent).toBe('D:\\X\\Orienta\\Orienta');
    expect(doc.getElementById('homeNote').textContent).not.toMatch(/keeps using/);
  });

  it('says BOTH that existing data was found and that a restart is needed', async () => {
    // In the packaged app a restart is nearly always due, so "found" used to be
    // lost behind "saved, restart".
    const doc = await start({
      pickHome: async () => ({ dir: 'D:\\X\\Orienta', adopted: true, restartNeeded: true }),
    });
    doc.getElementById('pickHomeBtn').click();
    await settle();
    const text = doc.getElementById('homeNote').textContent;
    expect(text).toBe(LOCALES.en.homeAdoptedRestart.replace('{{dir}}', 'D:\\X\\Orienta'));
    expect(text).toMatch(/existing Orienta data/);
    expect(text).toMatch(/start it again/);
  });

  it('says only "saved, restart" for an ordinary pick that needs a restart', async () => {
    const doc = await start({
      pickHome: async () => ({ dir: 'D:\\Y\\Orienta', restartNeeded: true }),
    });
    doc.getElementById('pickHomeBtn').click();
    await settle();
    expect(doc.getElementById('homeNote').textContent).toBe(LOCALES.en.homeRestart);
  });

  it('says "keeps using" when the choice is the folder already in use', async () => {
    const doc = await start({ pickHome: async () => ({ dir: 'C:\\H', kept: true, restartNeeded: false }) });
    doc.getElementById('pickHomeBtn').click();
    await settle();
    expect(doc.getElementById('homeNote').textContent).toBe(
      LOCALES.en.homeKept.replace('{{dir}}', 'C:\\H'));
  });
});
