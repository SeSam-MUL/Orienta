import { describe, it, expect } from 'vitest';
import { installOutcome } from './installOutcome';

const t = (k) => `<${k}>`;

describe('installOutcome', () => {
  it('shows the backend message for the feature stage and schedules a refresh', () => {
    const r = installOutcome({ success: true, stage: 'feature', message: 'Restart Windows, then click Install again.' }, t);
    expect(r).toEqual({ ok: true, text: 'Restart Windows, then click Install again.', refresh: true });
  });

  it('shows the backend message for the distro stage', () => {
    const r = installOutcome({ success: true, stage: 'distro', message: 'Installing Ubuntu-22.04 for your account.' }, t);
    expect(r.ok).toBe(true);
    expect(r.text).toMatch(/for your account/);
  });

  it('a denied UAC prompt is an error, not "installation started"', () => {
    // HTTP 200 with success:false used to be shown as success.
    const r = installOutcome({ success: false, stage: 'feature', message: 'the administrator prompt was denied' }, t);
    expect(r).toEqual({ ok: false, text: 'the administrator prompt was denied', refresh: false });
  });

  it('falls back to the i18n strings when the backend sends no message', () => {
    expect(installOutcome({ success: true }, t).text).toBe('<settings:install.step1.installStarted>');
    expect(installOutcome({ success: false }, t).text).toBe('<settings:install.step1.installFailed>');
  });

  it('a missing or garbled answer is not "installation started"', () => {
    expect(installOutcome(undefined, t)).toEqual({ ok: false, text: '<settings:install.step1.installFailed>', refresh: false });
    expect(installOutcome({}, t).ok).toBe(false);
    expect(installOutcome({ success: 'yes' }, t).ok).toBe(false);
  });
});
