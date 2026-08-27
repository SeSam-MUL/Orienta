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

export function issueBody({
  description, version, page, fingerprint, breadcrumbs, environment, hasScreenshot,
}) {
  const parts = [];
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
