/**
 * The startup plan — what the shell DOES, as data.
 *
 * Every case here corresponds to a defect found by review of the first version,
 * and every one of them lived in `main.js`, which no test in this repository can
 * load. Extracting the decision into a pure function is what makes them
 * testable at all; before that, the four-way page switch, the "keep the
 * existing backend" branch, the hand-off of the interpreter and the root, and
 * the packaged-with-a-checkout case were covered by nothing.
 */
import { describe, it, expect } from 'vitest';

import { startupPlan } from '../../electron/project_root.js';

const DEV = { mode: 'dev', root: 'C:/checkout' };
const RUN = { mode: 'run', python: 'C:/home/python/python.exe', root: 'C:/home/runtime' };

describe('the root and the interpreter always come from the decision', () => {
  it('never re-derives the root', () => {
    // The time bomb: `resolveProjectRoot()` returns the INSTALLED runtime as
    // soon as one exists, and it is not gated on dev mode. A developer who
    // installed Orienta once would have had `npm run electron:dev` silently
    // run the backend out of the installation — edits to backend/ having no
    // effect, with nothing said.
    expect(startupPlan(DEV, { packaged: false }).root).toBe('C:/checkout');
    expect(startupPlan(RUN, { packaged: true }).root).toBe('C:/home/runtime');
  });

  it('passes the decision\'s interpreter through', () => {
    expect(startupPlan(RUN, { packaged: true }).python).toBe(RUN.python);
  });
});

describe('where the window points', () => {
  it('uses the dev server in a checkout', () => {
    const plan = startupPlan(DEV, { packaged: false });
    expect(plan.page).toBe('devServer');
    expect(plan.thenLoad).toBe('devServer');
  });

  it('uses the backend in a packaged build', () => {
    const plan = startupPlan(RUN, { packaged: true });
    expect(plan.page).toBe('waiting');
    expect(plan.thenLoad).toBe('backend');
  });

  it('a PACKAGED shell pointed at a checkout still loads the backend', () => {
    // ORIENTA_PROJECT_ROOT is a documented override and makes the decision
    // `dev` while the deployment is packaged. Keying the page on the decision
    // mode left that window on the spinner permanently, because the URL was
    // only ever loaded for mode === 'run'. What decides is whether the BACKEND
    // SERVES THE UI, which is a fact about the deployment.
    const plan = startupPlan(DEV, { packaged: true });
    expect(plan.thenLoad).toBe('backend');
    expect(plan.page).toBe('waiting');
  });
});

describe('the busy-port question', () => {
  it('is not asked when nothing would be started', () => {
    // Asking first meant a launch that was never going to spawn anything still
    // offered to kill a backend the user had running — with "Stop it and start
    // fresh" as the DEFAULT button, and a 12 h batch behind it.
    for (const mode of ['setup', 'repair']) {
      const plan = startupPlan({ mode, message: 'x' }, { packaged: true });
      expect(plan.askAboutPort).toBe(false);
      expect(plan.spawn).toBe(false);
    }
  });

  it('is asked when we would spawn', () => {
    expect(startupPlan(RUN, { packaged: true }).askAboutPort).toBe(true);
    expect(startupPlan(DEV, { packaged: false }).askAboutPort).toBe(true);
  });

  it('keeping the existing backend still points the window at it', () => {
    // It ANSWERS /api/health — that is how we know it is there, and pointing
    // the window at it is the entire purpose of the "keep it" button. Returning
    // early instead left the spinner turning forever in front of a perfectly
    // healthy backend, with no timeout and no message.
    const plan = startupPlan(RUN, { packaged: true, portBusy: true, keepExisting: true });
    expect(plan.spawn).toBe(false);
    expect(plan.thenLoad).toBe('backend');
  });

  it('keeping the existing backend in dev points at the dev server', () => {
    const plan = startupPlan(DEV, { packaged: false, portBusy: true, keepExisting: true });
    expect(plan.spawn).toBe(false);
    expect(plan.thenLoad).toBe('devServer');
  });
});

describe('the wizard', () => {
  it('shows nothing to load and starts nothing', () => {
    const plan = startupPlan({ mode: 'repair', message: 'the reason' }, { packaged: true });
    expect(plan.page).toBe('wizard');
    expect(plan.thenLoad).toBeNull();
    expect(plan.spawn).toBe(false);
    expect(plan.reason).toBe('the reason');
  });
});

describe('the plan is total', () => {
  it('answers every mode without throwing, in both deployments', () => {
    for (const mode of ['dev', 'run', 'setup', 'repair', 'something-new']) {
      for (const packaged of [true, false]) {
        const plan = startupPlan({ mode }, { packaged });
        expect(plan).toHaveProperty('page');
        expect(plan).toHaveProperty('spawn');
        expect(plan).toHaveProperty('thenLoad');
        // Never claim to spawn without saying what to spawn it from.
        if (plan.spawn) expect('root' in plan).toBe(true);
      }
    }
  });

  it('tolerates a missing context', () => {
    expect(() => startupPlan(RUN)).not.toThrow();
  });
});
