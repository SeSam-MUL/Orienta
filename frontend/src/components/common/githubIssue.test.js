import { describe, it, expect } from 'vitest';
import { newIssueUrl, issueTitle, issueBody } from './githubIssue';

const REPO = 'https://github.com/SeSam-MUL/Orienta';
const FIELDS = {
  description: 'Indexing stopped at 40% and the map stayed empty',
  version: 'v0.2.1',
  page: '#indexing',
  fingerprint: 'a1b2c3d4',
  breadcrumbs: '  -6.0s [nav] page → indexing\n  -0.4s [http-error] POST /start → 500',
  environment: 'branch: main',
  hasScreenshot: true,
};

describe('issueTitle', () => {
  it('leads with the error id so duplicates are visible in the list', () => {
    expect(issueTitle(FIELDS)).toBe(
      '[a1b2c3d4] Indexing stopped at 40% and the map stayed empty',
    );
  });

  it('falls back to the page when there is no description', () => {
    expect(issueTitle({ page: '#eds' })).toBe('Problem on #eds');
  });

  it('uses only the first line and stays short', () => {
    const t = issueTitle({ description: 'first line\nsecond line' });
    expect(t).toBe('first line');
    expect(issueTitle({ description: 'x'.repeat(500) }).length).toBeLessThanOrEqual(200);
  });
});

describe('issueBody', () => {
  it('carries description, version, error id and the trail', () => {
    const b = issueBody(FIELDS);
    expect(b).toContain('Indexing stopped at 40%');
    expect(b).toContain('version: v0.2.1');
    expect(b).toContain('error id: a1b2c3d4');
    expect(b).toContain('POST /start → 500');
  });

  it('asks for the zip and mentions the screenshot when there is one', () => {
    expect(issueBody(FIELDS)).toMatch(/drag the report zip/i);
    expect(issueBody(FIELDS)).toMatch(/screenshot of the window/i);
    expect(issueBody({ ...FIELDS, hasScreenshot: false }))
      .not.toMatch(/screenshot of the window/i);
  });

  it('marks an empty description rather than leaving a blank section', () => {
    expect(issueBody({ ...FIELDS, description: '' })).toContain('no description given');
  });
});

describe('newIssueUrl', () => {
  it('builds a pre-filled new-issue URL', () => {
    const url = newIssueUrl(REPO, FIELDS);
    expect(url.startsWith(`${REPO}/issues/new?`)).toBe(true);
    expect(url).toContain('title=');
    expect(url).toContain('body=');
    expect(decodeURIComponent(url)).toContain('a1b2c3d4');
  });

  it('returns null when the repository is unknown', () => {
    expect(newIssueUrl(null, FIELDS)).toBeNull();
    expect(newIssueUrl('', FIELDS)).toBeNull();
  });

  it('tolerates a trailing slash on the repo url', () => {
    expect(newIssueUrl(`${REPO}/`, FIELDS)).toContain(`${REPO}/issues/new?`);
  });

  it('drops the trail first when the URL would be too long', () => {
    const huge = { ...FIELDS, breadcrumbs: 'x'.repeat(20000) };
    const url = newIssueUrl(REPO, huge);
    expect(url.length).toBeLessThanOrEqual(6000);
    // the important parts survive
    expect(decodeURIComponent(url)).toContain('a1b2c3d4');
    expect(decodeURIComponent(url)).toContain('v0.2.1');
  });

  it('still fits when even the description is enormous', () => {
    const url = newIssueUrl(REPO, { ...FIELDS, description: 'y'.repeat(50000) });
    expect(url.length).toBeLessThanOrEqual(6000);
  });
});
