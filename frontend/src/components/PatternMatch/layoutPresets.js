// Named figure-layout presets, persisted in localStorage so a user can reuse a
// composed figure template across sessions / pixels / datasets.
const KEY = 'pm_figure_presets';

function readAll() {
  try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; }
}
function writeAll(o) { localStorage.setItem(KEY, JSON.stringify(o)); }

export function savePreset(name, model) {
  const a = readAll(); a[name] = model; writeAll(a);
}
export function listPresets() {
  return Object.keys(readAll()).map((name) => ({ name }));
}
export function loadPreset(name) {
  const a = readAll();
  return a[name] ? JSON.parse(JSON.stringify(a[name])) : null;
}
export function deletePreset(name) {
  const a = readAll(); delete a[name]; writeAll(a);
}
