// @vitest-environment jsdom
/**
 * Nothing offers a GPU on a machine that has none.
 *
 * M5 tester report, point 3: a list of GPU/CUDA words on a Mac — the backend
 * preselect "GPU (PyTorch, empfohlen)", the dictionary dialog's preselected
 * "GPU (PyTorch)", "GPU freigeben". Each one is small; together they make a
 * CPU-only machine look misconfigured, and the tester asked what was missing.
 *
 * The rule: the LABEL follows the device, the compute path does not move. On a
 * Mac the PyTorch path is still the right one — it simply runs on the CPU.
 *
 * A review of the first version of this file found it was coupled to NAMES
 * rather than behaviour: renaming a local variable failed three assertions
 * while reverting the actual feature failed none. The behaviour now lives in
 * `hasNoCudaDevice`, which is tested directly, and the dialog's label has a
 * render test in GenerateDictionaryDialog.test.jsx. What is left here as a
 * source check is only the wiring, and is labelled as such.
 */
import { describe, it, expect } from 'vitest';
import en from '../../locales/en/indexing.json';
import de from '../../locales/de/indexing.json';
import ja from '../../locales/ja/indexing.json';
import zh from '../../locales/zh/indexing.json';
import sim from '../../locales/en/simulation.json';
import { hasNoCudaDevice } from './cpuEstimate';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const read = (rel) => fs.readFileSync(path.join(HERE, rel), 'utf8');
const LANGS = [['en', en], ['de', de], ['ja', ja], ['zh', zh]];

describe('hasNoCudaDevice — the one predicate behind every such label', () => {
  it('answers yes only for a stated, non-CUDA device', () => {
    expect(hasNoCudaDevice({ device: 'cpu' })).toBe(true);
    expect(hasNoCudaDevice({ device: 'mps' })).toBe(true);
    expect(hasNoCudaDevice({ device: 'cuda' })).toBe(false);
  });

  it('says no while the probe has not answered', () => {
    // A label that claims CPU before we know is worse than one that corrects
    // itself a second later.
    expect(hasNoCudaDevice(null)).toBe(false);
    expect(hasNoCudaDevice(undefined)).toBe(false);
    expect(hasNoCudaDevice({})).toBe(false);
  });

  it('treats "unknown" as no answer, not as a CPU', () => {
    // /status reports "unknown" when the CUDA probe throws mid-run. Reading it
    // as "no GPU" would relabel a CUDA machine in the middle of a run.
    expect(hasNoCudaDevice({ device: 'unknown' })).toBe(false);
  });
});

describe('every device-dependent label has a no-CUDA twin, in four languages', () => {
  const get = (dict, dotted) => dotted.split('.').reduce((n, k) => n?.[k], dict);

  it.each([
    ['spherical.backendGpu', 'spherical.backendGpuOnCpu'],
    ['genDictDialog.backendGpu', 'genDictDialog.backendGpuOnCpu'],
  ])('%s -> %s', (gpuKey, cpuKey) => {
    for (const [lang, dict] of LANGS) {
      const gpu = get(dict, gpuKey);
      const cpu = get(dict, cpuKey);
      expect(gpu, `${lang}.${gpuKey}`).toBeTruthy();
      expect(cpu, `${lang}.${cpuKey}`).toBeTruthy();
      expect(cpu, `${lang}.${cpuKey} must mention the CPU`).toMatch(/CPU/);
      expect(cpu, `${lang}.${cpuKey} must not promise a GPU`).not.toMatch(/GPU/);
      expect(cpu, `${lang} twins must differ`).not.toBe(gpu);
    }
  });

  it('the twins name the runtime, not the forward simulator', () => {
    // "Orienta Engine" is the built-in FORWARD SIMULATOR (feat/engine-name,
    // simulation.json). The spherical indexer and the dictionary generator are
    // different subsystems; an earlier version of these labels borrowed that
    // name and so made it mean three things — inverting the point of the
    // rename. They name PyTorch, which is what actually runs.
    expect(sim.engine.oursLabel).toMatch(/Orienta Engine/);   // the real owner
    for (const [lang, dict] of LANGS) {
      for (const key of ['spherical.backendGpuOnCpu', 'genDictDialog.backendGpuOnCpu']) {
        const label = get(dict, key);
        expect(label, `${lang}.${key}`).not.toMatch(/Orienta Engine/);
        expect(label, `${lang}.${key}`).toMatch(/PyTorch/);
      }
    }
  });

  it('the cache-release control has a device-neutral name and message', () => {
    for (const [lang, dict] of LANGS) {
      expect(dict.actions.releaseCaches, lang).toBeTruthy();
      expect(dict.actions.releaseCaches, lang).not.toMatch(/GPU/);
      expect(dict.actions.releaseCachesTip, lang).toBeTruthy();
      // it must say what it frees, since on this machine it is host memory
      expect(dict.messages.cachesReleased, lang).toMatch(/\{\{backends\}\}/);
      expect(dict.messages.cachesReleased, lang).toMatch(/\{\{phases\}\}/);
    }
  });
});

describe('the wiring (source checks — the weak kind, listed on purpose)', () => {
  it('both screens use the shared predicate rather than their own copy', () => {
    for (const rel of ['./IndexingPage.jsx', '../Batch/BatchWorkflow.jsx']) {
      const src = read(rel);
      expect(src, rel).toMatch(/hasNoCudaDevice\(runtimeInfo\)/);
      // the expression this replaced, in either file, would be a second copy
      expect(src, rel).not.toMatch(/runtimeInfo\.device !== 'cuda'/);
    }
  });

  it('the dictionary dialog is TOLD, rather than probing a second time', () => {
    // Two sources of truth for "is there a GPU" is how two screens end up
    // claiming different things about the same machine. The dialog's own label
    // behaviour is covered by a render test, not by this.
    const dialog = read('./GenerateDictionaryDialog.jsx');
    expect(dialog).toMatch(/noCudaHere = false,/);
    expect(dialog).not.toMatch(/gpuStatus\(/);
    expect(read('./IndexingPage.jsx')).toMatch(/noCudaHere=\{noCudaHere\}/);
  });

  it('the cache-release button is offered on EVERY machine', () => {
    // It was hidden where there is no CUDA device, on the strength of the
    // message it printed there ("no CUDA device — nothing to do"). The endpoint
    // does three things before it looks for a device, including dropping the
    // SHT renderer's Lambert grids — several GB of HOST memory on a Mac — and
    // this is the only UI path to that.
    const src = read('./IndexingPage.jsx');
    expect(src).not.toMatch(/\{!noCudaHere && \(\s*\n\s*<button/);
    expect(src).toMatch(/t\('actions\.releaseCaches'\)/);
    expect(src).toMatch(/t\('messages\.cachesReleased'/);
  });
});

describe('the compute path is untouched', () => {
  it('no label change moved a default backend', () => {
    // Still the PyTorch backend by default — on a Mac the only alternative for
    // spherical is EMSphInx, which runs through WSL and so not at all there.
    expect(read('./IndexingPage.jsx')).toMatch(/useState\('spherical_gpu'\)/);
    expect(read('./GenerateDictionaryDialog.jsx'))
      .toMatch(/backend: mode === 'cpu' \? 'cpu' : 'gpu'/);
  });
});
