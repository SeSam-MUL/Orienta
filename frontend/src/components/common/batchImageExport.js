import { bufferToBase64, downloadBlob } from './imageExport';

/**
 * Writing one export over many pictures.
 *
 * The point of a figure series is that the pictures share their settings, so
 * the batch is the ordinary export dialog applied more than once: the same
 * resolution, border, format and scale-bar style, seen on a real picture
 * before N files are committed to.
 *
 * Only the planning and the run loop live here. What a picture IS, and how it
 * is rendered, stays with the caller — this module never touches a canvas.
 */

/**
 * Characters Windows refuses in a file name, plus the ones that split paths.
 *
 * The control range is deliberate, not a typo: a map label is data, and a
 * stray control character in it would go straight into a file name.
 */
// eslint-disable-next-line no-control-regex -- see above: labels are data
const FORBIDDEN = /[\\/:*?"<>|\u0000-\u001f]+/g;

/**
 * A file name that is safe to write and still says which map it holds.
 *
 * Map labels are not tame: "SE/Elektronenbild 33 (Input1)" carries a slash,
 * and a leading dot would make a hidden file on Unix and read as traversal.
 */
export function batchFileName(stem, label, ext) {
  const clean = (s) => String(s ?? '')
    .replace(FORBIDDEN, '_')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/^[.\s]+/, '')
    .replace(/[.\s]+$/, '');
  const a = clean(stem) || 'export';
  const b = clean(label);
  return `${a}${b ? `_${b}` : ''}.${ext}`;
}

/**
 * Names for a whole series, in order, with collisions broken.
 *
 * Two layers may legitimately carry the same label; without this the second
 * would overwrite the first and the run would report N files where N-1 exist.
 * Compared case-insensitively, because Windows would: writing both would
 * leave one file, not two.
 */
export function planBatchFiles(items, { stem, ext, labelOf = (i) => i?.label }) {
  const used = new Set();
  return (items || []).map((item) => {
    const label = labelOf(item);
    let filename = batchFileName(stem, label, ext);
    for (let n = 2; used.has(filename.toLowerCase()); n += 1) {
      filename = batchFileName(stem, `${label} (${n})`, ext);
    }
    used.add(filename.toLowerCase());
    return { item, filename };
  });
}

/**
 * Render and write each planned file in turn.
 *
 * One at a time on purpose: a sixteen-times series of a large map must not
 * have to sit in memory all at once.
 *
 * A picture that cannot be built or written is recorded and SKIPPED. One
 * layer whose bitmap has not arrived must not cost the user the other sixteen
 * — the contract of a batch is "do what can be done and say what could not",
 * and every failure comes back named.
 */
export async function runImageBatch({ plan, render, write, onProgress, isCancelled }) {
  const written = [];
  const failed = [];
  const list = plan || [];
  let stopped = false;

  for (let i = 0; i < list.length; i += 1) {
    if (isCancelled?.()) { stopped = true; break; }
    const { item, filename } = list[i];
    onProgress?.({ index: i, total: list.length, filename, item });
    try {
      const blob = await render(item, filename);
      if (!blob) throw new Error('nothing to write');
      await write(blob, filename);
      written.push(filename);
    } catch (err) {
      failed.push({ filename, message: err?.message || String(err) });
    }
  }

  onProgress?.({ index: list.length, total: list.length, done: true });
  return { written, failed, cancelled: stopped, total: list.length };
}

/**
 * Where a series of files goes.
 *
 * The folder is chosen ONCE and then written into file by file — a save dialog
 * per picture is exactly what a batch has to avoid. In Electron the main
 * process refuses any folder the user did not pick, so the renderer cannot
 * name a path of its own; in a plain browser the fallback is the ordinary
 * download per file, which is the only thing a page may do.
 */
export function makeFolderWriter(api = (typeof window !== 'undefined' ? window.electronAPI : null)) {
  if (api?.openFolder && api?.writeImageInFolder) {
    let dir = null;
    return {
      kind: 'folder',
      async pick() { dir = await api.openFolder(); return dir; },
      async write(blob, filename) {
        if (!dir) throw new Error('no folder chosen');
        const base64 = bufferToBase64(await blob.arrayBuffer());
        return api.writeImageInFolder({ dir, name: filename, base64 });
      },
      get target() { return dir; },
    };
  }
  return {
    kind: 'download',
    // Nothing to choose: the browser decides where downloads land.
    async pick() { return 'download'; },
    async write(blob, filename) { downloadBlob(blob, filename); return filename; },
    get target() { return null; },
  };
}

/**
 * The caption one picture of a series carries.
 *
 * Two honest answers, and the user picks which — no rule that guesses:
 *
 *   perMap  each file is captioned with its OWN map's name. A series of
 *           element maps where every file says "Fe Ka1" would be mislabelled
 *           figures, so this is the default.
 *   typed   the text in the caption field, on every file. For a caption that
 *           describes the SAMPLE rather than the map.
 *
 * `undefined` means "the dialog's own caption text", which is what
 * `buildSpec` falls back to — so the typed case needs no special path.
 */
export function batchCaption(item, { perMap }) {
  return perMap ? (item?.label ?? undefined) : undefined;
}
