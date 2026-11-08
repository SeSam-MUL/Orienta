// @vitest-environment node
//
// EBSDViewer cannot be mounted in a test, so the pieces it hands its decisions to are tested as
// functions (apexLoad.test.js: routeLoad, singleFlight, runLoads, sidebarLabels,
// announcePairingDropped), and this pins that the viewer, and the two other places that switch the
// loaded file, really call them: every way of opening a file goes through requestLoad; the only other
// callers of doLoadFile are "open anyway" for an .h5 that cannot be read and the dialog's own run.
import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const SRC_DIR = fileURLToPath(new URL('../../', import.meta.url));
const src = (rel) => readFileSync(fileURLToPath(new URL(rel, import.meta.url)), 'utf8');
const SRC = src('./EBSDViewer.jsx');
const between = (text, from, to) => text.slice(text.indexOf(from), text.indexOf(to, text.indexOf(from)));

describe('how a file is opened', () => {
  it('calls doLoadFile in exactly three places: requestLoad, "open anyway", and the dialog\'s run', () => {
    const calls = [...SRC.matchAll(/doLoadFile\(/g)].map((m) => SRC.slice(Math.max(0, m.index - 30), m.index + 80));
    expect(calls).toHaveLength(3);
    expect(calls.some((c) => c.includes('return doLoadFile(p)'))).toBe(true);
    expect(calls.some((c) => c.includes('onConfirm: () => doLoadFile(p)'))).toBe(true);
    expect(calls.some((c) => c.includes('doLoadFile(job.path, { apex: job.apex, variant: job.variant })'))).toBe(true);
  });

  it('opens every other entry point through requestLoad', () => {
    for (const needle of [
      'requestLoad(preload)', 'requestLoad(selected)', 'requestLoad(path.trim())',
      "if (e.key === 'Enter') requestLoad(filePath)", 'onClick={() => requestLoad(filePath)}',
      'requestLoad(f)', 'requestLoad(path);',
    ]) {
      expect(SRC).toContain(needle);
    }
  });

  it('asks the backend about the path before loading, one question at a time, through routeLoad', () => {
    const body = between(SRC, 'const requestLoad', 'const handleDialogLoad');
    expect(SRC).toContain('useRef(singleFlight(routeLoad, { action: \'busy\' }))');
    expect(body).toContain('await askRoute.current(p, ebsdApi.inspect)');
    expect(body).toContain("route.action === 'busy'");
    expect(SRC).toContain("import { routeLoad, readRememberedVariant, rememberVariant, canonicalPath } from './loadChoices';");
    // an APEX answer opens the APEX dialog, a `choices` answer the generic one (preselecting what was
    // chosen for the file last time); a file with one variant opens neither
    expect(body).toContain("route.action === 'dialog' && route.kind === 'apex'");
    expect(body).toContain('setApexDialog(route.dialog)');
    expect(body).toContain("route.action === 'dialog' && route.kind === 'choices'");
    expect(body).toContain('{ ...route.dialog, remembered: readRememberedVariant(route.dialog.path) }');
    expect(body).toContain("route.action === 'unreadable'");
    expect(body.indexOf('askRoute.current')).toBeLessThan(body.lastIndexOf('return doLoadFile(p)'));
  });

  it('asks about the metadata file when a pattern file of a paired entry is opened again', () => {
    const body = between(SRC, 'const requestLoad', 'const handleDialogLoad');
    expect(body).toContain('reopenPath((path || \'\').trim(), useLoadedFilesStore.getState().files)');
  });

  it('accepts the NORDIF and Oxford formats in the dialog and on drop', () => {
    expect(SRC).toContain("'up1', 'up2', 'dat', 'ebsp', 'txt'");
    expect(SRC).toContain('/\\.(h5oina|h5|hdf5|up1|up2|dat|ebsp|txt)$/i');
  });

  it('runs the dialog\'s loads through runLoads and says which measurements stayed closed', () => {
    const run = between(SRC, 'const handleDialogLoad', 'const choosePatternFile');
    expect(run).toContain('await runLoads(');
    expect(run).toContain('(job) => doLoadFile(job.path, { apex: job.apex, variant: job.variant })');
    expect(run).toContain("t('apexDialog.runNotOpened'");
  });

  it('sends the chosen pairing with the load, and nothing else changes for an ordinary load', () => {
    expect(SRC).toContain('apex: extra?.apex || null,');
  });

  it('records the metadata file, not the pattern file, in Recent and in the path box after a dialog load', () => {
    const load = between(SRC, 'const doLoadFile', 'const requestLoad');
    expect(load).toContain('const recorded = loadedAs(path, extra);');
    expect(load).toContain('setFilePath(recorded);');
    expect(load).toContain('addRecentFile(recorded);');
    expect(load).toContain('const filtered = prev.filter((f) => f !== recorded);');
    expect(load).not.toMatch(/addRecentFile\(path\)/);
  });

  it('hands the dialog identities that do not change on every render, and tells it about open prompts', () => {
    for (const name of ['apexDescribe', 'apexSearchFolder', 'apexLoadPreview', 'apexHexLimits']) {
      expect(SRC).toMatch(new RegExp(`const ${name} = useCallback\\(`));
    }
    for (const prop of [
      'describePatternFile={apexDescribe}', 'searchFolder={apexSearchFolder}',
      'loadPreview={apexLoadPreview}', 'loadHexLimits={apexHexLimits}',
      'choosePatternFile={choosePatternFile}', 'chooseFolder={chooseApexFolder}',
      'escapeBlocked={promptProps.open || confirmProps.open}',
    ]) {
      expect(SRC).toContain(prop);
    }
  });

  it('does not promise the batch that a chosen file works there', () => {
    const en = JSON.parse(src('../../locales/en/ebsdviewer.json'));
    expect(JSON.stringify(en.apexDialog.footer)).not.toMatch(/batch/i);
  });
});

describe('each measurement is recognisable where the loaded files are listed', () => {
  it('names a measurement by its measurement name in the viewer\'s list', () => {
    expect(SRC).toContain('sidebarLabels(loadedFiles, disambiguateNames, (k, o) => t(k, o))');
    expect(SRC).toContain('loadedFileLabels[idx]');
  });
  it('and in the header switcher and the indexing page\'s dataset list', () => {
    expect(src('../common/FileSwitcher.jsx')).toContain('{loadedFileLabel(f, (k, o) => t(`ebsdviewer:${k}`, o))}');
    expect(src('../Indexing/IndexingPage.jsx')).toContain('${loadedFileLabel(f, (k, o) => t(`ebsdviewer:${k}`, o))}');
  });
});

describe('every place that switches the loaded file tells the user when the pairing was dropped', () => {
  const walk = (dir) => readdirSync(dir).flatMap((name) => {
    const full = path.join(dir, name);
    if (statSync(full).isDirectory()) return name === 'node_modules' ? [] : walk(full);
    return /\.(jsx?|tsx?)$/.test(name) && !/\.test\./.test(name) ? [full] : [];
  });

  it('knows all the callers of switchFile: the viewer, the indexing page, and the one sequence the header switcher and the PC page share', () => {
    const callers = walk(SRC_DIR)
      .filter((f) => /\.switchFile\(/.test(readFileSync(f, 'utf8')))
      .map((f) => path.relative(SRC_DIR, f).replace(/\\/g, '/'))
      .sort();
    expect(callers).toEqual([
      'components/EBSDViewer/EBSDViewer.jsx',
      'components/Indexing/IndexingPage.jsx',
      'components/common/switchLoadedFile.js',
    ]);
  });

  it.each([
    ['components/EBSDViewer/EBSDViewer.jsx', 'announcePairingDropped(info, t, { log, warn:'],
    ['components/common/switchLoadedFile.js', 'announcePairingDropped(res?.data,'],
    ['components/Indexing/IndexingPage.jsx', 'announcePairingDropped(switched?.data,'],
  ])('%s announces it right after the switch', (file, call) => {
    const text = readFileSync(path.join(SRC_DIR, file), 'utf8');
    const at = text.indexOf('.switchFile(');
    expect(at).toBeGreaterThan(-1);
    // from the switch to the end of its try block (the shared helper has none: to its last line)
    const end = text.indexOf('} catch', at);
    const after = text.slice(at, end === -1 ? text.length : end);
    expect(after).toContain(call);
  });

  it.each([
    'components/common/FileSwitcher.jsx',
    'components/PCRefinement/Calibration/ApplyToFilesDialog.jsx',
  ])('%s switches through the shared sequence, not a copy of it', (file) => {
    const text = readFileSync(path.join(SRC_DIR, file), 'utf8');
    expect(text).toMatch(/await switchLoadedFile\(/);
    expect(text).not.toMatch(/\.switchFile\(/);
  });
});
