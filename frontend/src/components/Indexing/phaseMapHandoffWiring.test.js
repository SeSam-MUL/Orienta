// @vitest-environment node
//
// A source-level guard, deliberately, and only for wiring (see
// phaseListWiring.test.js for why IndexingPage cannot be mounted and why this
// kind of check is a placeholder): what the hand-off does is tested against
// the real hook in usePhaseMapHandoff.test.jsx; what is checked here is that
// the page hands the hook what it needs and no longer forces Hough.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const SRC = readFileSync(
  fileURLToPath(new URL('./IndexingPage.jsx', import.meta.url)), 'utf8');

describe('the per-phase routing toggle talks about phases, in every language', () => {
  for (const lang of ['en', 'de', 'ja', 'zh']) {
    it(`${lang}: mask.perPhaseRoutingDesc does not say CIF`, () => {
      const json = JSON.parse(readFileSync(
        fileURLToPath(new URL(`../../locales/${lang}/indexing.json`, import.meta.url)), 'utf8'));
      const text = json.mask?.perPhaseRoutingDesc ?? json.perPhaseRoutingDesc
        ?? Object.values(json).map((s) => s?.perPhaseRoutingDesc).find(Boolean);
      expect(text).toBeTruthy();
      expect(text).not.toMatch(/CIF/);
    });
  }
});

/** The text between two markers of the page (the markers must exist). */
function between(text, start, end) {
  const i = text.indexOf(start);
  expect(i, `${start} not found`).toBeGreaterThan(-1);
  const j = text.indexOf(end, i);
  expect(j, `${end} not found after ${start}`).toBeGreaterThan(i);
  return text.slice(i, j);
}

/** The call `usePhaseMapHandoff({ ... })` in the page. */
function handoffCall() {
  const i = SRC.indexOf('usePhaseMapHandoff({');
  expect(i).toBeGreaterThan(-1);
  return SRC.slice(i, SRC.indexOf('\n  });', i));
}

describe('the Indexing page and the phase-map hand-off', () => {
  it('no longer forces Hough when the hand-off arrives', () => {
    expect(SRC).not.toMatch(/setMethod\(\s*'hough'\s*\)/);
  });

  it('tells the hook which method the list on screen belongs to', () => {
    // Without it the hook would pick files out of the previous method's list.
    expect(handoffCall()).toMatch(/discoveredFor/);
    expect(SRC).toMatch(/setDiscoveredFor\(methodRef\.current\)/);
  });

  it('tells the hook whether routing is on, or a method switch cannot redo the selection', () => {
    expect(handoffCall()).toMatch(/routingOn:\s*usePhaseMapRouting/);
  });

  it('selects the files and switches routing on when the hook says so', () => {
    const call = handoffCall();
    expect(call).toMatch(/onSelect:[\s\S]*setPhaseFiles\(matched\.map/);
    expect(call).toMatch(/onSelect:[\s\S]*setPhases\(matched\)/);
    expect(call).toMatch(/onInstall:[\s\S]*setPhaseMapRoutingAvailable\(true\)/);
    expect(call).toMatch(/onInstall:[\s\S]*setUsePhaseMapRouting\(true\)/);
  });

  it('lets the hook see whether a phase is already chosen', () => {
    expect(handoffCall()).toMatch(/hasSelection:[^\n]*phaseFilesRef/);
  });

  it('hands the hook what a Dictionary has to fit: PC, detector shape, geometry', () => {
    expect(handoffCall()).toMatch(/getDictContext:\s*\(\)\s*=>\s*dictContextRef\.current/);
    const assign = SRC.slice(SRC.indexOf('dictContextRef.current = {'));
    expect(assign.slice(0, 200)).toMatch(/currentPc:\s*currentPcVec/);
    expect(assign.slice(0, 200)).toMatch(/detectorShape:\s*genDictDetectorShape/);
    expect(assign.slice(0, 200)).toMatch(/geom:\s*detectorGeom/);
  });

  it('ships exactly the dictionaries the hand-off chose', () => {
    expect(handoffCall()).toMatch(/onSelect:[\s\S]*setSelectedDictPaths\(prev => \(\{ \.\.\.prev, \.\.\.dictPaths \}\)\)/);
  });

  it('knows when the detector context of the LOADED file has arrived', () => {
    expect(handoffCall()).toMatch(
      /dictContextReady:\s*detectorContextFor !== null && detectorContextFor === filePath/);
    // fetchDataState says so only after every part answered, and only for the same file
    const fetch = between(SRC, 'const fetchDataState', '}, [refreshLoadedFiles, t]);');
    expect(fetch).toMatch(/const forFile = useDataStore\.getState\(\)\.filePath/);
    expect(fetch).toMatch(/Promise\.all\(\[pDatasets, pGeom, pPc\]\)/);
    expect(fetch).toMatch(/getState\(\)\.filePath === forFile\) setDetectorContextFor\(forFile\)/);
  });

  describe('while routing is on, a typed or seeded file is never a stand-in for the phases', () => {
    const memo = (name) => between(SRC, `const ${name} = useMemo`, '\n  }, [');

    it('one rule in one place: phaseBase', () => {
      expect(SRC).toMatch(/const routingActive = phaseMapRoutingAvailable && usePhaseMapRouting;/);
      const base = between(SRC, 'const phaseBase = useMemo', '[phaseFiles, fileInput, routingActive]');
      expect(base).toMatch(/!routingActive && fileInput \? \[fileInput\]/);
      expect(SRC).toMatch(/use_phase_map_routing: routingActive,/);
    });

    it('the request paths and the labels are built from it, not from fileInput', () => {
      for (const name of ['effectivePhasePaths', 'edsPhaseLabels']) {
        const body = memo(name);
        expect(body).toContain('phaseBase');
        expect(body).not.toContain('fileInput');
      }
    });

    it('the Generate button does not read it either', () => {
      expect(SRC).toMatch(/const genDictMaster =\s*\(!routingActive && fileInput\) \|\|/);
    });

    it('Start stays disabled, with the reason, when no phase file is selected', () => {
      expect(SRC).toMatch(
        /const routingWithoutPhases = routingActive && effectivePhasePaths\.length === 0;/);
      expect(SRC).toMatch(/&& !phaseListBlocked && !groupMissing && !routingWithoutPhases;/);
      expect(SRC).toMatch(
        /routingWithoutPhases \? t\('actions\.startBlockedRoutingNoPhases'\)/);
    });

    for (const lang of ['en', 'de', 'ja', 'zh']) {
      it(`${lang}: the reason is translated`, () => {
        const json = JSON.parse(readFileSync(
          fileURLToPath(new URL(`../../locales/${lang}/indexing.json`, import.meta.url)), 'utf8'));
        expect(json.actions.startBlockedRoutingNoPhases).toBeTruthy();
      });
    }
  });

  it('declares the hook before the effect that clears the phases on a method change', () => {
    // The hook must run first in the commit that changes the method (it waits
    // for the new list), see the comment in usePhaseMapHandoff.js.
    const hook = SRC.indexOf('usePhaseMapHandoff({');
    const clear = SRC.indexOf('Clear files from previous method');
    expect(hook).toBeGreaterThan(-1);
    expect(clear).toBeGreaterThan(hook);
  });
});
