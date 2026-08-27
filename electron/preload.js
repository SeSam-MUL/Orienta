/**
 * Preload script for Electron.
 * Exposes safe APIs to the renderer process.
 */

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  openFile: (options) => ipcRenderer.invoke('dialog:openFile', options),
  openFolder: () => ipcRenderer.invoke('dialog:openFolder'),
  saveFile: (options) => ipcRenderer.invoke('dialog:saveFile', options),
  // Unlike saveFile (which only returns a path for the BACKEND to write to),
  // this one also writes the bytes — images are produced in the renderer and
  // never touch the backend.
  saveImage: (options) => ipcRenderer.invoke('dialog:saveImage', options),
  openPoleFigure: () => ipcRenderer.invoke('window:openPoleFigure'),
  // Restart the whole app after a self-update, so the new backend and the
  // newly built interface are both loaded.
  relaunch: () => ipcRenderer.invoke('app:relaunch'),
  // Base64 PNG of the app window, for problem reports.
  captureScreen: () => ipcRenderer.invoke('app:captureScreen'),
});
