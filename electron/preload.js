/**
 * Preload script for Electron.
 * Exposes safe APIs to the renderer process.
 */

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  openFile: (options) => ipcRenderer.invoke('dialog:openFile', options),
  // `{ remember: true }` from an EXPORT picker: that folder becomes the start
  // of the next save dialog. Input pickers (a folder of scans) leave it alone.
  openFolder: (options) => ipcRenderer.invoke('dialog:openFolder', options),
  saveFile: (options) => ipcRenderer.invoke('dialog:saveFile', options),
  // Unlike saveFile (which only returns a path for the BACKEND to write to),
  // this one also writes the bytes — images are produced in the renderer and
  // never touch the backend.
  saveImage: (options) => ipcRenderer.invoke('dialog:saveImage', options),
  // Same reason as saveImage, for text: an add-on's table lives in the page
  // and never reaches the backend, so `saveFile` — which only hands back a
  // path for the backend to write to — cannot save it.
  saveText: (options) => ipcRenderer.invoke('dialog:saveText', options),
  // Batch export: the folder is chosen once through openFolder, then one call
  // per file writes into it. Main refuses any folder the user did not pick.
  writeImageInFolder: (options) => ipcRenderer.invoke('fs:writeImageInFolder', options),
  openPoleFigure: () => ipcRenderer.invoke('window:openPoleFigure'),
  // Which operating system this is. The renderer has no `process`
  // (contextIsolation, no node integration), so a component that asks
  // `process.platform` silently gets nothing and renders nothing.
  platform: process.platform,
  // Removing the data folder on macOS/Linux, where there is no uninstaller.
  planDataRemoval: () => ipcRenderer.invoke('data:plan'),
  removeData: (options) => ipcRenderer.invoke('data:remove', options),
  // Restart the whole app after a self-update, so the new backend and the
  // newly built interface are both loaded.
  relaunch: () => ipcRenderer.invoke('app:relaunch'),
  // Remember the app's language so the NEXT start's splash screen, which
  // runs before any page, speaks it instead of the operating system's.
  setLanguage: (lang) => ipcRenderer.invoke('app:setLanguage', lang),
  // Base64 PNG of the app window, for problem reports.
  captureScreen: () => ipcRenderer.invoke('app:captureScreen'),
  // Opens the folder holding orienta.log / backend-console.log in the file
  // manager. Takes no argument: the main process knows the folder.
  openLogFolder: () => ipcRenderer.invoke('app:openLogFolder'),
});

/**
 * The setup wizard's surface, exposed separately from `electronAPI`.
 *
 * Separate because the two are used at different times by different pages: the
 * wizard runs before the application exists, and the application runs after the
 * wizard is gone. Keeping them apart means the running app carries no channel
 * that can start an 8 GB install, and the wizard carries no file dialogs.
 */
// ONLY in the setup window. There is one preload file, and `main.js` gives it
// to the application window and to every pole-figure window too, so an
// unconditional expose hands `window.setupAPI.run({mode:'gpu'})` to the React
// app -- which could rename <home>/python out from under the backend that is
// running out of it. The main process sets this variable for the setup window
// alone, and the handlers check the sender as well: a flag on its own would be
// a comment, not a boundary.
// `additionalArguments`, not an environment variable: env is inherited by every
// renderer the process ever creates, so a flag set for the wizard would also be
// true for a pole-figure window opened an hour later. This is per window.
if (process.argv.includes('--orienta-setup-window')) {
  contextBridge.exposeInMainWorld('setupAPI', {
  context: () => ipcRenderer.invoke('setup:context'),
  locales: () => ipcRenderer.invoke('setup:locales'),
  probe: () => ipcRenderer.invoke('setup:probe'),
  run: (options) => ipcRenderer.invoke('setup:run', options),
  cancel: () => ipcRenderer.invoke('setup:cancel'),
  openLog: () => ipcRenderer.invoke('setup:openLog'),
  relaunch: () => ipcRenderer.invoke('app:relaunch'),

  /**
   * Subscribe to progress. Returns an UNSUBSCRIBE function.
   *
   * Returning it, rather than offering a removeListener, is what keeps a retry
   * from stacking a second listener on the same channel: each run would then
   * repaint the screen once per previous attempt, and the step list would
   * flicker between two different runs' ideas of where it is.
   */
    onProgress: (handler) => {
      const listener = (_event, payload) => handler(payload);
      ipcRenderer.on('setup:progress', listener);
      return () => ipcRenderer.removeListener('setup:progress', listener);
    },
    quit: () => ipcRenderer.invoke('setup:quit'),
    releases: () => ipcRenderer.invoke('setup:releases'),
    pickPackage: () => ipcRenderer.invoke('setup:pickPackage'),
    pickHome: () => ipcRenderer.invoke('setup:pickHome'),
    setLanguage: (lang, explicit) => ipcRenderer.invoke('setup:setLanguage', lang, explicit),
    selectRelease: (tag) => ipcRenderer.invoke('setup:selectRelease', tag),
  });
}
