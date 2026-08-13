// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, act } from '@testing-library/react';
import { exportStem, useImageExport } from './useImageExport';

// The dialog pulls in canvas/Electron plumbing that has nothing to do with the
// menu wiring under test; stub it down to something observable.
vi.mock('./ImageExportDialog', () => ({
  default: ({ src, title, defaultBaseName }) => (
    <div data-testid="dialog" data-src={src} data-title={title} data-name={defaultBaseName} />
  ),
}));

afterEach(cleanup);

function Harness({ spec }) {
  const exp = useImageExport();
  return (
    <div>
      <div data-testid="host" onContextMenu={(e) => exp.openMenu(e, spec)}>view</div>
      {exp.node}
    </div>
  );
}

function rightClick(el) {
  act(() => {
    el.dispatchEvent(new MouseEvent('contextmenu', {
      bubbles: true, cancelable: true, clientX: 40, clientY: 60,
    }));
  });
}

const flush = async () => { await act(async () => { await Promise.resolve(); }); };

describe('exportStem', () => {
  it('drops the directory and the extension', () => {
    expect(exportStem('Al_alpha.sht')).toBe('Al_alpha');
    expect(exportStem('C:\\Database\\SHT_Library\\Al\\Al_alpha.sht')).toBe('Al_alpha');
    expect(exportStem('/srv/lib/Al_alpha.cif')).toBe('Al_alpha');
  });

  it('keeps dots that are part of the name', () => {
    expect(exportStem('Mn0.5Fe0.5Al5.sht')).toBe('Mn0.5Fe0.5Al5');
  });

  it('falls back when there is nothing usable', () => {
    expect(exportStem('', 'sphere')).toBe('sphere');
    expect(exportStem(null, 'sphere')).toBe('sphere');
    expect(exportStem('.sht', 'sphere')).toBe('sphere');
    expect(exportStem(undefined)).toBe('export');
  });
});

describe('useImageExport', () => {
  it('opens a one-item menu on right-click', () => {
    render(<Harness spec={{ build: () => 'data:image/png;base64,AAA', name: 'x', label: 'X' }} />);
    expect(document.querySelector('[data-context-menu]')).toBeNull();
    rightClick(screen.getByTestId('host'));
    const items = [...document.querySelectorAll('[data-context-item]')];
    expect(items).toHaveLength(1);
    expect(items[0].dataset.contextItem).toBe('export');
  });

  it('builds the image only when the menu item is chosen', async () => {
    const build = vi.fn(() => 'data:image/png;base64,AAA');
    render(<Harness spec={{ build, name: 'file', label: 'Label' }} />);
    rightClick(screen.getByTestId('host'));
    // Merely opening the menu must not capture anything — on the crystal viewer
    // that would force a WebGL read-back for a menu the user may dismiss.
    expect(build).not.toHaveBeenCalled();

    act(() => { document.querySelector('[data-context-item="export"]').click(); });
    await flush();
    expect(build).toHaveBeenCalledTimes(1);
    const dlg = screen.getByTestId('dialog');
    expect(dlg.dataset.src).toBe('data:image/png;base64,AAA');
    expect(dlg.dataset.name).toBe('file');
    expect(dlg.dataset.title).toBe('Label');
  });

  it('awaits an async build (the Plotly/WebGL read-back case)', async () => {
    const build = vi.fn(async () => 'data:image/png;base64,BBB');
    render(<Harness spec={{ build, name: 'sphere', label: 'Sphere' }} />);
    rightClick(screen.getByTestId('host'));
    act(() => { document.querySelector('[data-context-item="export"]').click(); });
    await flush();
    expect(screen.getByTestId('dialog').dataset.src).toBe('data:image/png;base64,BBB');
  });

  it('reports a failed capture instead of opening an empty dialog', async () => {
    const build = vi.fn(() => { throw new Error('scene not ready'); });
    render(<Harness spec={{ build, name: 'x', label: 'X' }} />);
    rightClick(screen.getByTestId('host'));
    act(() => { document.querySelector('[data-context-item="export"]').click(); });
    await flush();
    expect(screen.queryByTestId('dialog')).toBeNull();
    expect(document.querySelector('[data-image-export-error]').textContent).toContain('scene not ready');
  });

  it('treats an empty result as a failure too', async () => {
    render(<Harness spec={{ build: () => null, name: 'x', label: 'X' }} />);
    rightClick(screen.getByTestId('host'));
    act(() => { document.querySelector('[data-context-item="export"]').click(); });
    await flush();
    expect(screen.queryByTestId('dialog')).toBeNull();
    expect(document.querySelector('[data-image-export-error]')).not.toBeNull();
  });

  it('suppresses the browser menu and does not let it bubble', () => {
    render(<Harness spec={{ build: () => 'data:x', name: 'x', label: 'X' }} />);
    const outer = vi.fn();
    document.addEventListener('contextmenu', outer);
    const ev = new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 1, clientY: 1 });
    act(() => { screen.getByTestId('host').dispatchEvent(ev); });
    expect(ev.defaultPrevented).toBe(true);
    expect(outer).not.toHaveBeenCalled();
    document.removeEventListener('contextmenu', outer);
  });

  it('does nothing without a spec, so a viewer can opt out per state', () => {
    render(<Harness spec={null} />);
    rightClick(screen.getByTestId('host'));
    expect(document.querySelector('[data-context-menu]')).toBeNull();
  });
});
