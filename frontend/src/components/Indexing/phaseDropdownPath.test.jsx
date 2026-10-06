// @vitest-environment jsdom
//
// Adding a phase by path from the phase picker. In a plain browser
// (start_app.py) there is no native file dialog, so a phase outside the library
// can only be added by typing or pasting its path.
import React from 'react';
import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';
import PhaseDropdown from './PhaseDropdown';

const LIB = [
  { path: 'D:/lib/CIF_Library/Al.cif', filename: 'Al.cif', formula: 'Al', element_group: 'Al-Fe-Si' },
];

function mount(props = {}) {
  return render(
    <PhaseDropdown
      discoveredFiles={props.discoveredFiles ?? LIB}
      groups={['Al-Fe-Si']}
      selectedPaths={[]}
      onTogglePath={props.onTogglePath ?? (() => {})}
      onSetAll={() => {}}
      method="hough"
      open
      onClose={() => {}}
      {...props}
    />
  );
}

const pathInput = () => screen.getByPlaceholderText(/path/i);

afterEach(() => {
  cleanup();
  delete window.electronAPI;
});

describe('without onAddPath (PC Refinement before, other callers)', () => {
  it('keeps the old "+ Add file manually…" link and shows no path field', () => {
    const onTogglePath = vi.fn();
    mount({ onTogglePath });
    expect(screen.queryByPlaceholderText(/path/i)).toBeNull();
    fireEvent.click(screen.getByText('+ Add file manually…'));
    expect(onTogglePath).toHaveBeenCalledWith(null);
  });
});

describe('with onAddPath', () => {
  it('shows a path field instead of the manual link, worded for the method', () => {
    mount({ onAddPath: vi.fn() });
    expect(screen.queryByText('+ Add file manually…')).toBeNull();
    expect(pathInput().getAttribute('placeholder')).toMatch(/\.cif/);
    cleanup();
    mount({ onAddPath: vi.fn(), method: 'spherical' });
    expect(pathInput().getAttribute('placeholder')).toMatch(/\.sht/);
    cleanup();
    mount({ onAddPath: vi.fn(), method: 'dictionary' });
    expect(pathInput().getAttribute('placeholder')).toMatch(/\.h5/);
  });

  it('Add sends the typed path, then clears the field and says what was added', async () => {
    const onAddPath = vi.fn().mockResolvedValue({ ok: true, name: 'MyAl' });
    mount({ onAddPath });
    fireEvent.change(pathInput(), { target: { value: '  C:\\data\\MyAl.cif ' } });
    fireEvent.click(screen.getByText('Add'));
    await waitFor(() => expect(screen.getByText('Added MyAl')).toBeTruthy());
    expect(onAddPath).toHaveBeenCalledWith('C:\\data\\MyAl.cif');
    expect(pathInput().value).toBe('');
  });

  it('Enter submits too', async () => {
    const onAddPath = vi.fn().mockResolvedValue({ ok: true, name: 'X' });
    mount({ onAddPath });
    fireEvent.change(pathInput(), { target: { value: '/a/X.cif' } });
    fireEvent.keyDown(pathInput(), { key: 'Enter' });
    await waitFor(() => expect(onAddPath).toHaveBeenCalledWith('/a/X.cif'));
  });

  it('cannot send an empty path', () => {
    const onAddPath = vi.fn();
    mount({ onAddPath });
    expect(screen.getByText('Add').disabled).toBe(true);
    fireEvent.keyDown(pathInput(), { key: 'Enter' });
    expect(onAddPath).not.toHaveBeenCalled();
  });

  it('keeps what was typed and says what is wrong, in words, when the server refuses', async () => {
    const onAddPath = vi.fn().mockResolvedValue({
      ok: false,
      error: { code: 'not_found', message: 'No such file: /a/nope.cif', params: { path: '/a/nope.cif' } },
    });
    mount({ onAddPath });
    fireEvent.change(pathInput(), { target: { value: '/a/nope.cif' } });
    fireEvent.click(screen.getByText('Add'));
    await waitFor(() => expect(screen.getByText('No file at /a/nope.cif.')).toBeTruthy());
    expect(pathInput().value).toBe('/a/nope.cif');   // so it can be corrected
  });

  it.each([
    ['not_absolute', {}, /complete path/i],
    ['not_a_file', { path: '/a' }, /\/a is a folder/],
    ['wrong_extension', { got: '.txt', expected: '.cif' }, /needs a \.cif file.*\.txt/s],
    ['unreadable', { reason: 'no atom sites' }, /no atom sites/],
    ['not_a_phase', {}, /not an EMsoft master pattern/],
  ])('has its own wording for %s', async (code, params, pattern) => {
    const onAddPath = vi.fn().mockResolvedValue({
      ok: false, error: { code, message: 'ENGLISH FALLBACK', params },
    });
    mount({ onAddPath });
    fireEvent.change(pathInput(), { target: { value: '/a/x' } });
    fireEvent.click(screen.getByText('Add'));
    await waitFor(() => expect(screen.getByText(pattern)).toBeTruthy());
    expect(screen.queryByText('ENGLISH FALLBACK')).toBeNull();
  });

  it('falls back to the server sentence for a code it does not know', async () => {
    const onAddPath = vi.fn().mockResolvedValue({
      ok: false, error: { code: 'something_new', message: 'The server says no.', params: {} },
    });
    mount({ onAddPath });
    fireEvent.change(pathInput(), { target: { value: '/a/x' } });
    fireEvent.click(screen.getByText('Add'));
    await waitFor(() => expect(screen.getByText('The server says no.')).toBeTruthy());
  });

  it('shows the words of the server for a generic failure, not a fixed sentence', async () => {
    const onAddPath = vi.fn().mockResolvedValue({
      ok: false, error: { code: 'generic', message: "A phase named 'x' is already loaded.", params: {} },
    });
    mount({ onAddPath });
    fireEvent.change(pathInput(), { target: { value: '/a/x.cif' } });
    fireEvent.click(screen.getByText('Add'));
    await waitFor(() => expect(screen.getByText("A phase named 'x' is already loaded.")).toBeTruthy());
    expect(screen.queryByText('The file could not be added.')).toBeNull();
  });

  it('the field has an accessible name, and what appears under it is announced', async () => {
    mount({ onAddPath: vi.fn().mockResolvedValue({ ok: true, name: 'Al' }) });
    expect(screen.getByLabelText('Path of a phase file')).toBe(pathInput());
    const region = screen.getByTestId('phase-path-note');
    expect(region.getAttribute('role')).toBe('status');
    expect(region.getAttribute('aria-live')).toBe('polite');
    expect(region.textContent).toBe('');                  // in the page before anything is said
    fireEvent.change(pathInput(), { target: { value: 'D:/x/Al.cif' } });
    fireEvent.click(screen.getByText('Add'));
    await waitFor(() => expect(screen.getByTestId('phase-path-note').textContent).toMatch(/Added Al/));
  });

  it('a refusal is announced assertively', async () => {
    mount({ onAddPath: vi.fn().mockResolvedValue({ ok: false, error: { code: 'not_found', message: '', params: { path: 'D:/x' } } }) });
    fireEvent.change(pathInput(), { target: { value: 'D:/x/Al.cif' } });
    fireEvent.click(screen.getByText('Add'));
    await waitFor(() => expect(screen.getByTestId('phase-path-note').getAttribute('role')).toBe('alert'));
  });

  it('strips typographic quotes pasted around the path before sending it', async () => {
    const onAddPath = vi.fn().mockResolvedValue({ ok: true, name: 'Al' });
    mount({ onAddPath });
    fireEvent.change(pathInput(), { target: { value: '“D:/x/Al.cif”' } });
    fireEvent.click(screen.getByText('Add'));
    await waitFor(() => expect(onAddPath).toHaveBeenCalledWith('D:/x/Al.cif'));
  });

  it("words the new refusals in the user's language", async () => {
    const onAddPath = vi.fn()
      .mockResolvedValueOnce({ ok: false, error: { code: 'too_large', message: 'x', params: { size_mb: 210, limit_mb: 20 } } })
      .mockResolvedValueOnce({ ok: false, error: { code: 'phase_same_name', message: 'x',
        params: { name: 'Al', loaded_path: 'C:/lib/Al.cif', path: 'D:/m/Al.cif' } } });
    mount({ onAddPath });
    fireEvent.change(pathInput(), { target: { value: 'D:/x/big.cif' } });
    fireEvent.click(screen.getByText('Add'));
    await screen.findByText(/This file is 210 MB, far too large/);
    fireEvent.change(pathInput(), { target: { value: 'D:/m/Al.cif' } });
    fireEvent.click(screen.getByText('Add'));
    await screen.findByText(/A different file named Al is already in use \(C:\/lib\/Al.cif\)/);
  });

  it('says when the phase was already in the list', async () => {
    const onAddPath = vi.fn().mockResolvedValue({ ok: true, already: true, name: 'Al' });
    mount({ onAddPath });
    fireEvent.change(pathInput(), { target: { value: '/a/Al.cif' } });
    fireEvent.click(screen.getByText('Add'));
    await waitFor(() => expect(screen.getByText('Al is already in the list')).toBeTruthy());
  });

  it('says the library entry was selected when the path was a library file', async () => {
    const onAddPath = vi.fn().mockResolvedValue({ ok: true, inLibrary: true, name: 'Al' });
    mount({ onAddPath });
    fireEvent.change(pathInput(), { target: { value: '/a/Al.cif' } });
    fireEvent.click(screen.getByText('Add'));
    await waitFor(() => expect(screen.getByText('Al is in the library — selected')).toBeTruthy());
  });

  it('offers Browse only where there is a native dialog, and sends its pick', async () => {
    const onAddPath = vi.fn().mockResolvedValue({ ok: true, name: 'P' });
    mount({ onAddPath });
    expect(screen.queryByText('Browse…')).toBeNull();
    cleanup();

    const openFile = vi.fn().mockResolvedValue('C:\\picked\\P.cif');
    window.electronAPI = { openFile };
    mount({ onAddPath });
    fireEvent.click(screen.getByText('Browse…'));
    await waitFor(() => expect(onAddPath).toHaveBeenCalledWith('C:\\picked\\P.cif'));
    expect(openFile.mock.calls[0][0].filters[0].extensions).toEqual(['cif']);
  });

  it('a cancelled Browse sends nothing', async () => {
    const onAddPath = vi.fn();
    window.electronAPI = { openFile: vi.fn().mockResolvedValue(null) };
    mount({ onAddPath });
    fireEvent.click(screen.getByText('Browse…'));
    await Promise.resolve();
    expect(onAddPath).not.toHaveBeenCalled();
  });
});

describe('a phase the user added by path', () => {
  it('stays in the list while a group filter narrows the library', () => {
    const mine = { path: '/mine/X.cif', filename: 'X.cif', formula: 'X', element_group: 'Sonstiges', user_added: true };
    mount({
      discoveredFiles: [...LIB, mine],
      collectionKeys: new Set(['al']),        // a group that does not name X
      method: 'hough',
    });
    expect(screen.getByText('X')).toBeTruthy();
  });
});
