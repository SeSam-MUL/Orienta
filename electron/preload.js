/**
 * Preload script for Electron.
 * Exposes safe APIs to the renderer process.
 */

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  openFile: (options) => ipcRenderer.invoke('dialog:openFile', options),
  openFolder: () => ipcRenderer.invoke('dialog:openFolder'),
  saveFile: (options) => ipcRenderer.invoke('dialog:saveFile', options),
  openPoleFigure: () => ipcRenderer.invoke('window:openPoleFigure'),
});
