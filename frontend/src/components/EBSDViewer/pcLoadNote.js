// What the viewer says about the pattern centre when a file has been loaded.
//
// The load response reports the calibration STORE's state (the PC indexing will
// use), not only what the file carries: a file without a PC whose PC the user typed
// earlier comes back with that PC, and must not be called "defaulted".

const fmt = (pc) => (Array.isArray(pc) && pc.length === 3
  ? pc.map((v) => Number(v).toFixed(4)).join(', ') : '');

/** {level: 'warning' | 'info' | 'log', key, params} under logMessages, or null. */
export function pcLoadNote(info) {
  if (!info) return null;
  if (info.pc_defaulted) return { level: 'warning', key: 'pcDefaultWarning', params: {} };
  const pc = fmt(info.pc);
  const params = { pc };
  if (info.pc_notice === 'file_pc_wins') {
    return { level: 'info', key: 'pcFileWins', params: { pc, discarded: fmt(info.pc_discarded) } };
  }
  if (info.pc_notice === 'user_pc_kept') return { level: 'info', key: 'pcUserKept', params };
  if (info.pc_notice === 'export_pc_ignored') {
    return { level: 'info', key: 'pcExportIgnored',
             params: { pc, exported: fmt(info.pc_export_ignored) } };
  }
  if (info.pc_source === 'osc') return { level: 'log', key: 'pcFromOsc', params: {} };
  if (info.pc_source === 'apex') return { level: 'log', key: 'pcFromApex', params: {} };
  return null;
}

/**
 * Say it to the user: the toast of the note's level, in the language of the page. For a
 * loader that has no log pane of its own (the Phase Map's "Add file...", which loads an
 * export through import-h5 and gets the same PC fields as a load). The texts are the
 * viewer's (`ebsdviewer:logMessages.*`).
 * @returns the message shown, or null
 */
export function announceImportedPc(info, t, toast) {
  const note = pcLoadNote(info);
  if (!note) return null;
  const msg = t(`ebsdviewer:logMessages.${note.key}`, note.params);
  if (note.level === 'warning') toast.warning(msg, 15000);
  else if (note.level === 'info') toast.info(msg, 15000);
  return msg;
}
