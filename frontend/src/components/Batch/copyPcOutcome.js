/**
 * What the file tree says after a PC copy succeeded.
 *
 * The backend reports a target that is the source file itself as `skipped` (success, nothing
 * written). Counting every result as "copied" said "copied to 3 files" for two writes, and
 * "copied to 1 file" when only the source was ticked. Only the written ones are counted and a
 * skipped one is named.
 *
 * @param {Array<{success?: boolean, skipped?: boolean}>} results  `results` of /copy-pc
 * @param {string} doneKey  the message of the action ("fileTree.msgPcCopied" | "fileTree.msgAppliedTo"), takes {{count}}
 * @param {Function} t  i18next's t
 */
export function copyOutcomeText(results, doneKey, t) {
  const skipped = results.filter((r) => r?.skipped).length;
  const written = results.length - skipped;
  const parts = [];
  if (written > 0 || skipped === 0) parts.push(t(doneKey, { count: written }));
  if (skipped > 0) parts.push(t('fileTree.msgSkippedSource', { count: skipped }));
  return parts.join(' ');
}
