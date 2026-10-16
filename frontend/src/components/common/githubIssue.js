/**
 * Build a pre-filled "new issue" URL.
 *
 * GitHub's REST API cannot attach files to an issue — that is a deliberate
 * restriction, and only a real browser session can do it. So instead of
 * automating the impossible, we hand the user a browser tab where everything
 * machine-readable is already filled in, and they drag the report zip in.
 *
 * A URL has a practical length limit (browsers and servers start rejecting
 * somewhere past ~8 kB), so the body carries a summary and the zip carries
 * the detail.
 */

const MAX_URL = 6000;

export function issueTitle({ fingerprint, description, page }) {
  const first = String(description || '').split('\n')[0].trim();
  const subject = first || (page ? `Problem on ${page}` : 'Problem report');
  const id = fingerprint ? `[${fingerprint}] ` : '';
  return (id + subject).slice(0, 200);
}

/**
 * First line of every body. The issue form on GitHub asks the reporter to
 * confirm that an issue is public, but this pre-filled link opens a blank
 * issue and skips that form, so the warning has to travel in the body. It
 * comes first so that trimming the body from the end never removes it.
 */
export const PUBLIC_NOTICE =
  '> **This issue will be public.** The text below was filled in from your session ' +
  'and can contain file and folder names. Remove anything you do not want to publish, ' +
  'and check the report zip before you attach it.\n\n';

export function issueBody({
  description, version, page, fingerprint, breadcrumbs, environment, hasScreenshot,
}) {
  const parts = [PUBLIC_NOTICE];
  parts.push('### What happened\n');
  parts.push(`${(description || '').trim() || '_(no description given)_'}\n`);
  parts.push('\n### Report file\n');
  parts.push(
    'Please drag the report zip into this issue before submitting — it holds ' +
    'the full logs' + (hasScreenshot ? ' and a screenshot of the window' : '') + '.\n',
  );
  parts.push('\n### Version and environment\n');
  parts.push('```\n');
  parts.push(`version: ${version || 'unknown'}\n`);
  if (fingerprint) parts.push(`error id: ${fingerprint}\n`);
  if (page) parts.push(`page: ${page}\n`);
  if (environment) parts.push(`${environment}\n`);
  parts.push('```\n');
  if (breadcrumbs) {
    parts.push('\n### What happened before\n');
    parts.push('```\n' + breadcrumbs.trim() + '\n```\n');
  }
  return parts.join('');
}

/**
 * @returns {string|null} the URL, or null when the repository is unknown.
 *   The body is trimmed from the end (dropping breadcrumbs first) until the
 *   URL fits; the zip carries anything that falls off.
 */
export function newIssueUrl(repoUrl, fields) {
  if (!repoUrl) return null;
  const base = `${repoUrl.replace(/\/+$/, '')}/issues/new`;
  const title = issueTitle(fields);
  let body = issueBody(fields);

  const build = (b) =>
    `${base}?title=${encodeURIComponent(title)}&body=${encodeURIComponent(b)}`;

  let url = build(body);
  if (url.length > MAX_URL) {
    // Drop the trail first — it is the long part and it is in the zip.
    body = issueBody({ ...fields, breadcrumbs: '' });
    url = build(body);
  }
  while (url.length > MAX_URL && body.length > 200) {
    body = body.slice(0, Math.floor(body.length * 0.8));
    url = build(body + '\n…\n');
  }
  return url;
}
