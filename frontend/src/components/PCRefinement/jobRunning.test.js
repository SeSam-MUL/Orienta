import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { isPcJobRunning, PC_JOB_RUNNING } from './jobRunning';

const here = dirname(fileURLToPath(import.meta.url));
const locales = resolve(here, '..', '..', 'locales');
const backend = readFileSync(resolve(here, '..', '..', '..', '..', 'backend', 'api', 'routes', 'pcrefinement.py'), 'utf8');

describe('pc_job_running', () => {
  it('is the code the backend sends', () => {
    // Contract between the two sides: the frontend recognises what the backend raises.
    expect(backend).toContain(`_coded(409, "${PC_JOB_RUNNING}"`);
  });

  it('is recognised only as a 409 with that code', () => {
    const err = (status, code) => ({ response: { status, data: { detail: { code } } } });
    expect(isPcJobRunning(err(409, PC_JOB_RUNNING))).toBe(true);
    expect(isPcJobRunning(err(409, 'phase_limit'))).toBe(false);
    expect(isPcJobRunning(err(400, PC_JOB_RUNNING))).toBe(false);
    expect(isPcJobRunning(new Error('network'))).toBe(false);
  });

  it.each(['en', 'de', 'ja', 'zh'])('has a text in %s, and so do the Cancel messages', (lng) => {
    const j = JSON.parse(readFileSync(resolve(locales, lng, 'pcrefinement.json'), 'utf8'));
    expect(j.errors.pc_job_running).toBeTruthy();
    for (const key of ['optimizationCancelling', 'optimizationCancelled', 'pcEditNotApplied', 'tiltEditNotApplied']) {
      expect(j.controls[key], key).toBeTruthy();
    }
    expect(j.controls.pcEditNotApplied).toContain('{{reason}}');
    expect(j.controls.tiltEditNotApplied).toContain('{{reason}}');
  });
});
