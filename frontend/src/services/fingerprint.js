/**
 * A stable short id for "this is the same bug".
 *
 * Two people hitting the same crash must produce the same fingerprint, or
 * duplicates can only be spotted by reading every report. Two *different*
 * bugs must not collide, so the normalisation stays conservative: it removes
 * only the parts that vary between runs of the same fault.
 *
 * Normalised away:
 *  - numbers in the message (pixel 123 / pixel 456 are one bug),
 *  - quoted values and paths (they carry the user's file names),
 *  - the content hash in built bundle names (index-a1b2c3.js changes every
 *    release, the fault does not),
 *  - line and column numbers.
 *
 * Kept: the error kind, the shape of the message, and the top frame's
 * function and file — the things that actually identify the fault.
 */

/** FNV-1a, 32 bit. No cryptographic claim — this is a bucket label. */
function hash32(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, '0');
}

export function normalizeMessage(message) {
  return String(message ?? '')
    .replace(/[A-Za-z]:\\[^\s"']+|\/(?:[\w.-]+\/)+[\w.-]+/g, '<path>')
    .replace(/'[^']*'|"[^"]*"/g, '<value>')
    .replace(/\b\d+(?:\.\d+)?\b/g, '#')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 300)
    .toLowerCase();
}

/** The first stack frame that identifies our own code, normalised. */
export function topFrame(stack) {
  const lines = String(stack ?? '').split('\n');
  for (const raw of lines) {
    const line = raw.trim();
    if (!line.startsWith('at ') && !line.includes('@')) continue;
    return line
      .replace(/:\d+:\d+/g, '')                    // line:col
      .replace(/-[A-Za-z0-9_]{6,}\.js/g, '-*.js')  // bundle content hash
      .replace(/https?:\/\/[^/]+/g, '')            // host
      .replace(/\s+/g, ' ')
      .trim()
      .slice(0, 200)
      .toLowerCase();
  }
  return '';
}

/**
 * Short id for one fault. Same fault → same id, on any machine.
 * @returns {string} 8 hex characters
 */
export function errorFingerprint(kind, message, stack) {
  return hash32([
    String(kind ?? '').toLowerCase(),
    normalizeMessage(message),
    topFrame(stack),
  ].join('|'));
}
