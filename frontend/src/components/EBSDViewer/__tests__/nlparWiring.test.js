/**
 * EBSDViewer.jsx is too big to mount in a test, and that is exactly where an
 * unwired feature hides: every component test is green while the button does
 * nothing. These read the source and pin the few lines that connect NLPAR to
 * the viewer's processing model.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(join(here, '..', 'EBSDViewer.jsx'), 'utf8');

/** The body of doProcessing, from its declaration to the dependency array. */
const doProcessing = (() => {
  const start = source.indexOf('const doProcessing = useCallback(');
  const end = source.indexOf('}, [log, t, nlpar,', start);
  return start > -1 && end > start ? source.slice(start, end) : '';
})();

describe('NLPAR wiring in EBSDViewer.jsx', () => {
  it('doProcessing was found', () => {
    expect(doProcessing.length).toBeGreaterThan(500);
  });

  it('the button runs the op "nlpar" through doProcessing (auto-copy, busy overlay, refresh)', () => {
    expect(source).toMatch(/<NlparControls[\s\S]{0,200}doProcessing\('nlpar'/);
  });

  it('the step runs BEFORE every other step, also in the Recommended Pipeline', () => {
    const nlparAt = doProcessing.indexOf("if (nlparStep === 'run')");
    const firstOp = doProcessing.indexOf("if (op === 'bg_dyn')");
    expect(nlparAt).toBeGreaterThan(-1);
    expect(firstOp).toBeGreaterThan(nlparAt);
  });

  it('a cancelled run stops the operation but still refreshes the dataset list', () => {
    expect(doProcessing).toMatch(/runStep\(\{ timed, withProgress, log \}\)\)\) \{ await fetchDatasets\(\); return; \}/);
  });

  it('the Cancel button is drawn inside the busy overlay, which covers the panel', () => {
    const overlayAt = source.indexOf('Processing overlay for pattern operations');
    const extrasAt = source.indexOf('<NlparOverlayExtras', overlayAt);
    const pageHeaderAt = source.indexOf('Page header', overlayAt);
    expect(extrasAt).toBeGreaterThan(overlayAt);
    expect(extrasAt).toBeLessThan(pageHeaderAt);
    expect(source).toMatch(/<NlparOverlayExtras nlpar=\{nlpar\} phase=\{procProgress\?\.phase\} \/>/);
  });

  it('the hook is told what the viewer knows (loaded file, active dataset, busy flag), after the busy flag exists', () => {
    const call = source.indexOf('useNlpar({ ebsdLoaded, activeDataset, busy: processingBusy })');
    expect(call).toBeGreaterThan(-1);
    expect(call).toBeGreaterThan(source.indexOf('const [processingBusy, setProcessingBusy]'));
  });

  it('an already denoised dataset is turned away BEFORE the auto-copy and before the busy overlay', () => {
    const decide = doProcessing.indexOf('const nlparStep = nlparStepFor(op, nlpar);');
    const guard = doProcessing.indexOf("if (nlparStep === 'refuse') return;");
    expect(decide).toBeGreaterThan(-1);
    expect(guard).toBeGreaterThan(decide);
    expect(guard).toBeLessThan(doProcessing.indexOf('setProcessingBusy(true)'));
    expect(guard).toBeLessThan(doProcessing.indexOf('ebsdApi.deepcopy'));
  });

  it('the Recommended Pipeline decides about NLPAR through the same function, before the copy, and says when it skips', () => {
    expect(doProcessing).toContain('nlparStepFor(op, nlpar)');
    expect(doProcessing).toContain("if (nlparStep === 'skip') log(t('nlpar.skippedInPipeline'));");
    expect(doProcessing.indexOf('nlparStepFor(op, nlpar)')).toBeLessThan(doProcessing.indexOf('ebsdApi.deepcopy'));
    // the old condition, which only looked at "applied", is gone
    expect(doProcessing).not.toContain('nlpar.denoiseFirst && !nlpar.applied');
  });

  it('doProcessing re-creates itself when NLPAR state changes (denoiseFirst is read inside)', () => {
    expect(source).toContain('}, [log, t, nlpar,');
  });
});
