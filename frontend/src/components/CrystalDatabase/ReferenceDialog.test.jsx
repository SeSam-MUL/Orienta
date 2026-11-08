// @vitest-environment jsdom
/**
 * The question asked before a CIF without DOI or reference is converted.
 *
 * One file: type a DOI or a citation, or declare the data unpublished / own
 * data, or cancel. Several files: one question for all, declare or skip.
 */
import '@testing-library/jest-dom/vitest';
import { useState } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import i18n from '../../i18n';
import ReferenceDialog from './ReferenceDialog';

beforeEach(async () => { await act(async () => { await i18n.changeLanguage('en'); }); });
afterEach(() => { cleanup(); vi.clearAllMocks(); });

function setup(files, overrides = {}) {
  const handlers = {
    onSubmitReference: vi.fn(),
    onDeclareUnpublished: vi.fn(),
    onSkip: vi.fn(),
    onCancel: vi.fn(),
    ...overrides,
  };
  render(<ReferenceDialog files={files} {...handlers} />);
  return handlers;
}

describe('one file', () => {
  const files = [{ name: 'Zr.cif' }];

  it('names the file and offers no way to skip', () => {
    setup(files);
    expect(screen.getByRole('dialog')).toHaveTextContent('Zr.cif has no DOI');
    expect(screen.queryByRole('button', { name: /Skip/ })).toBeNull();
  });

  it('cannot save an empty entry', () => {
    setup(files);
    expect(screen.getByRole('button', { name: 'Save and convert' })).toBeDisabled();
  });

  it('sends a DOI as a DOI', () => {
    const h = setup(files);
    fireEvent.change(screen.getByLabelText('DOI or citation'), { target: { value: '10.1000/xyz123' } });
    expect(screen.getByRole('dialog')).toHaveTextContent('as its DOI');
    fireEvent.click(screen.getByRole('button', { name: 'Save and convert' }));
    expect(h.onSubmitReference).toHaveBeenCalledWith({ doi: '10.1000/xyz123', reference: '' });
  });

  it('strips a doi.org link down to the DOI', () => {
    const h = setup(files);
    fireEvent.change(screen.getByLabelText('DOI or citation'), { target: { value: ' https://doi.org/10.1107/S0021889811047258 ' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save and convert' }));
    expect(h.onSubmitReference).toHaveBeenCalledWith({ doi: '10.1107/S0021889811047258', reference: '' });
  });

  it('sends anything else as the reference text, also with Enter', () => {
    const h = setup(files);
    const box = screen.getByLabelText('DOI or citation');
    fireEvent.change(box, { target: { value: 'Smith et al., Acta Mater. 1 (2020) 1-9' } });
    expect(screen.getByRole('dialog')).toHaveTextContent('as its reference text');
    fireEvent.keyDown(box, { key: 'Enter' });
    expect(h.onSubmitReference).toHaveBeenCalledWith({ doi: '', reference: 'Smith et al., Acta Mater. 1 (2020) 1-9' });
  });

  it('declares the data unpublished without needing any text', () => {
    const h = setup(files);
    fireEvent.click(screen.getByRole('button', { name: 'Unpublished / own data' }));
    expect(h.onDeclareUnpublished).toHaveBeenCalledTimes(1);
    expect(h.onSubmitReference).not.toHaveBeenCalled();
  });

  it('says what the declaration writes, so it is never a silent unknown', () => {
    setup(files);
    expect(screen.getByRole('dialog')).toHaveTextContent('unpublished (declared by user)');
  });

  it('cancels with the button and with Escape, converting nothing', () => {
    const h = setup(files);
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(h.onCancel).toHaveBeenCalledTimes(2);
    expect(h.onSubmitReference).not.toHaveBeenCalled();
    expect(h.onDeclareUnpublished).not.toHaveBeenCalled();
  });
});

describe('several files', () => {
  const files = ['A.cif', 'B.cif', 'C.cif', 'D.cif', 'E.cif', 'F.cif', 'G.cif'].map(name => ({ name }));

  it('asks once, lists the first names and counts the rest', () => {
    setup(files);
    const dialog = screen.getByRole('dialog');
    expect(dialog).toHaveTextContent('7 selected CIF files have no DOI');
    expect(dialog).toHaveTextContent('A.cif');
    expect(dialog).toHaveTextContent('E.cif');
    expect(dialog).not.toHaveTextContent('F.cif');
    expect(dialog).toHaveTextContent('and 2 more');
    expect(screen.queryByLabelText('DOI or citation')).toBeNull();
  });

  it('declares all, or skips them, or cancels', () => {
    const h = setup(files);
    fireEvent.click(screen.getByRole('button', { name: 'Declare all 7 unpublished' }));
    fireEvent.click(screen.getByRole('button', { name: 'Skip these 7' }));
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(h.onDeclareUnpublished).toHaveBeenCalledTimes(1);
    expect(h.onSkip).toHaveBeenCalledTimes(1);
    expect(h.onCancel).toHaveBeenCalledTimes(1);
  });
});

describe('keyboard and pointer', () => {
  it('a click beside the dialog closes it when nothing was typed', () => {
    const h = setup([{ name: 'Zr.cif' }]);
    fireEvent.click(screen.getByRole('dialog').parentElement);
    expect(h.onCancel).toHaveBeenCalledTimes(1);
  });

  it('a click beside the dialog does NOT discard typed text', () => {
    const h = setup([{ name: 'Zr.cif' }]);
    fireEvent.change(screen.getByLabelText('DOI or citation'), { target: { value: 'Smith 2020' } });
    fireEvent.click(screen.getByRole('dialog').parentElement);
    expect(h.onCancel).not.toHaveBeenCalled();
    expect(screen.getByLabelText('DOI or citation')).toHaveValue('Smith 2020');
  });

  it('a click inside the dialog never closes it', () => {
    const h = setup([{ name: 'Zr.cif' }]);
    fireEvent.click(screen.getByRole('dialog'));
    expect(h.onCancel).not.toHaveBeenCalled();
  });

  it('single file: the text box has the focus', () => {
    setup([{ name: 'Zr.cif' }]);
    expect(screen.getByLabelText('DOI or citation')).toHaveFocus();
  });

  it('several files: Cancel has the focus, not the button that declares', () => {
    setup([{ name: 'A.cif' }, { name: 'B.cif' }]);
    expect(screen.getByRole('button', { name: 'Cancel' })).toHaveFocus();
  });

  it('Tab wraps from the last button to the first control, Shift+Tab the other way', () => {
    setup([{ name: 'A.cif' }, { name: 'B.cif' }]);
    const buttons = screen.getAllByRole('button');
    const first = buttons[0];
    const last = buttons[buttons.length - 1];
    last.focus();
    fireEvent.keyDown(last, { key: 'Tab' });
    expect(first).toHaveFocus();
    fireEvent.keyDown(first, { key: 'Tab', shiftKey: true });
    expect(last).toHaveFocus();
  });

  it('Tab skips the disabled save button and stays inside', () => {
    setup([{ name: 'Zr.cif' }]);
    const declare = screen.getByRole('button', { name: 'Unpublished / own data' });
    declare.focus();                                     // last enabled control (save is disabled)
    fireEvent.keyDown(declare, { key: 'Tab' });
    expect(screen.getByLabelText('DOI or citation')).toHaveFocus();
  });
});

describe('IME, focus return, accessibility, length', () => {
  const files = [{ name: 'Zr.cif' }];

  it('Enter that confirms an IME candidate does not submit a half-typed citation', () => {
    const h = setup(files);
    const box = screen.getByLabelText('DOI or citation');
    fireEvent.change(box, { target: { value: '山田 (2020' } });
    fireEvent.keyDown(box, { key: 'Enter', isComposing: true });
    fireEvent.keyDown(box, { key: 'Enter', keyCode: 229 });       // what some browsers send instead
    expect(h.onSubmitReference).not.toHaveBeenCalled();
    fireEvent.keyDown(box, { key: 'Enter' });                       // the real Enter, composition over
    expect(h.onSubmitReference).toHaveBeenCalledTimes(1);
  });

  it('gives the focus back to the control that opened it', () => {
    function Host() {
      const [open, setOpen] = useState(false);
      return (
        <>
          <button onClick={() => setOpen(true)}>opener</button>
          {open && <ReferenceDialog files={files} onCancel={() => setOpen(false)} onDeclareUnpublished={() => {}} onSubmitReference={() => {}} />}
        </>
      );
    }
    render(<Host />);
    const opener = screen.getByRole('button', { name: 'opener' });
    opener.focus();
    fireEvent.click(opener);
    expect(screen.getByLabelText('DOI or citation')).toHaveFocus();
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(opener).toHaveFocus();
  });

  it('is labelled by its title and described by its message', () => {
    setup(files);
    const dialog = screen.getByRole('dialog', { name: 'No reference for this crystal structure' });
    expect(dialog).toHaveAccessibleDescription(/Zr\.cif has no DOI or literature reference/);
    expect(dialog).toHaveAttribute('aria-modal', 'true');
  });

  it('caps the input below the SHT header field and shows a counter', () => {
    setup(files);
    const box = screen.getByLabelText('DOI or citation');
    expect(box).toHaveAttribute('maxlength', '2000');
    expect(screen.getByTestId('citation-counter')).toHaveTextContent('0 / 2000 characters');
    fireEvent.change(box, { target: { value: 'Smith 2020' } });
    expect(screen.getByTestId('citation-counter')).toHaveTextContent('10 / 2000 characters');
  });
});

describe('Escape during an IME composition', () => {
  it('cancels the composition, not the dialog', () => {
    const h = setup([{ name: 'Zr.cif' }]);
    const box = screen.getByLabelText('DOI or citation');
    fireEvent.keyDown(box, { key: 'Escape', isComposing: true });
    fireEvent.keyDown(box, { key: 'Escape', keyCode: 229 });
    expect(h.onCancel).not.toHaveBeenCalled();
    fireEvent.keyDown(box, { key: 'Escape' });                      // no composition: it closes
    expect(h.onCancel).toHaveBeenCalledTimes(1);
  });
});
