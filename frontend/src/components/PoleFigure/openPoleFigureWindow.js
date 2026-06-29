// Opens the detached pole-figure window. Electron path spawns a real child
// BrowserWindow; browser-dev path uses window.open to a same-origin popup.
export function openPoleFigureWindow() {
  if (typeof window !== 'undefined' && window.electronAPI?.openPoleFigure) {
    window.electronAPI.openPoleFigure();
    return;
  }
  const url = new URL(window.location.href);
  url.searchParams.set('view', 'polefigure');
  url.hash = '';
  window.open(url.toString(), 'polefigure', 'width=1000,height=820');
}
