// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { openPoleFigureWindow } from './openPoleFigureWindow';

afterEach(() => { delete window.electronAPI; vi.restoreAllMocks(); });

describe('openPoleFigureWindow', () => {
  it('uses electronAPI when present', () => {
    const openPoleFigure = vi.fn();
    window.electronAPI = { openPoleFigure };
    openPoleFigureWindow();
    expect(openPoleFigure).toHaveBeenCalled();
  });

  it('falls back to window.open in the browser', () => {
    const spy = vi.spyOn(window, 'open').mockImplementation(() => ({}));
    openPoleFigureWindow();
    expect(spy).toHaveBeenCalled();
    expect(spy.mock.calls[0][0]).toContain('view=polefigure');
  });
});
