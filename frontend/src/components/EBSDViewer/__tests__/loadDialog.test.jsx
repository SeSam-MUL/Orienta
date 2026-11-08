// @vitest-environment jsdom
//
// The generic load dialog: one row per measurement, per measurement a choice of pattern source, the
// processed source preselected (or the one remembered for the file). The answer is the backend's own
// (__fixtures__/inspectChoices.json, held against the real route by a backend test). The texts are the
// real ones: i18n is initialised for real (English by default; German once, to see it is not a stand-in).
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor, within, act } from '@testing-library/react';
import LoadDialog from '../LoadDialog';
import i18n from '../../../i18n';
import choices from '../__fixtures__/inspectChoices.json';

const PATH = '/d/two.h5oina';

let urlCounter = 0;
beforeEach(() => {
  urlCounter = 0;
  URL.createObjectURL = vi.fn(() => `blob:preview-${urlCounter++}`);
  URL.revokeObjectURL = vi.fn();
});
afterEach(() => cleanup());

function setup(over = {}) {
  const props = {
    isOpen: true, path: PATH, inspection: choices, remembered: null,
    onClose: vi.fn(), onLoad: vi.fn(), loadPreview: vi.fn(async () => new Blob(['x'])),
    ...over,
  };
  const view = render(<LoadDialog {...props} />);
  return { props, view };
}

const radio = (name) => screen.getByRole('radio', { name });
const openButton = () => screen.getByRole('button', { name: 'Open' });

describe('the rows and the preselection', () => {
  it('shows one row per measurement and the processed source of the first preselected', () => {
    setup();
    expect(screen.getByRole('dialog', { name: 'Open two.h5oina' })).toBeTruthy();
    expect(screen.getAllByTestId(/^measurement-/).map((r) => r.getAttribute('data-testid')))
      .toEqual(['measurement-1', 'measurement-2']);
    expect(screen.getAllByRole('radio')).toHaveLength(3);
    expect(radio('1: Processed patterns').checked).toBe(true);
    expect(radio('1: Unprocessed patterns').checked).toBe(false);
    expect(radio('2: Processed patterns').checked).toBe(false);
    // the sources of a measurement sit inside its row
    const row1 = screen.getByTestId('measurement-1');
    expect(within(row1).getAllByRole('radio')).toHaveLength(2);
    expect(within(screen.getByTestId('measurement-2')).getAllByRole('radio')).toHaveLength(1);
  });

  it('loads exactly the variant chosen: the unprocessed source of measurement 1', () => {
    const { props } = setup();
    fireEvent.click(radio('1: Unprocessed patterns'));
    expect(radio('1: Unprocessed patterns').checked).toBe(true);
    expect(radio('1: Processed patterns').checked).toBe(false);
    fireEvent.click(openButton());
    expect(props.onLoad).toHaveBeenCalledTimes(1);
    expect(props.onLoad).toHaveBeenCalledWith(
      [{ path: PATH, label: '1', variant: { measurement: '1', pattern_source: 'unprocessed' } }]);
  });

  it('is one choice for the whole file: picking measurement 2 clears the pick in measurement 1', () => {
    const { props } = setup();
    fireEvent.click(radio('2: Processed patterns'));
    expect(radio('1: Processed patterns').checked).toBe(false);
    fireEvent.click(openButton());
    expect(props.onLoad.mock.calls[0][0]).toEqual(
      [{ path: PATH, label: '2', variant: { measurement: '2', pattern_source: 'processed' } }]);
  });

  it('preselects the variant remembered for the file, says so, and Enter opens it', () => {
    const { props } = setup({ remembered: { measurement: '1', pattern_source: 'unprocessed' } });
    expect(radio('1: Unprocessed patterns').checked).toBe(true);
    expect(screen.getByText('Preselected because you opened this one last time.')).toBeTruthy();
    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Enter' });
    expect(props.onLoad).toHaveBeenCalledTimes(1);
    expect(props.onLoad.mock.calls[0][0][0].variant).toEqual({ measurement: '1', pattern_source: 'unprocessed' });
  });

  it('says nothing about memory when the default is what is preselected', () => {
    setup();
    expect(screen.queryByText(/Preselected because/)).toBeNull();
  });

  it('starts over for another file: the choice of the old one does not carry over', () => {
    const { view, props } = setup();
    fireEvent.click(radio('1: Unprocessed patterns'));
    const another = { ...choices };
    view.rerender(<LoadDialog {...props} path="/d/other.h5oina" inspection={another} />);
    expect(radio('1: Processed patterns').checked).toBe(true);
    // another path alone, with the very same answer object, starts over as well
    fireEvent.click(radio('1: Unprocessed patterns'));
    expect(radio('1: Unprocessed patterns').checked).toBe(true);
    view.rerender(<LoadDialog {...props} path="/d/third.h5oina" inspection={another} />);
    expect(radio('1: Processed patterns').checked).toBe(true);
  });
});

describe('what each source says about itself', () => {
  it('names count, size, bit depth and background, in the agreed words', () => {
    setup();
    const row = within(screen.getByTestId('measurement-1'));
    expect(row.getAllByText('6 patterns · 5 × 4 pixels')).toHaveLength(2);
    expect(row.getByText('8-bit, background corrected by the acquisition software')).toBeTruthy();
    expect(row.getByText('16-bit counts')).toBeTruthy();
    expect(row.getAllByText('Has its own background image')).toHaveLength(2);
    expect(within(screen.getByTestId('measurement-2')).getByText('No background image')).toBeTruthy();
  });

  it('words the depth by the measured data type: an 8-bit unprocessed source is neither "counts" nor "corrected"', () => {
    const odd = { ...choices, measurements: [{ ...choices.measurements[0], sources: [
      { ...choices.measurements[0].sources[0], key: 'processed', dtype: 'float32' },
      { ...choices.measurements[0].sources[1], dtype: 'uint8' },
    ] }] };
    setup({ inspection: odd });
    expect(screen.getByText('32-bit')).toBeTruthy();
    expect(screen.getByText('8-bit')).toBeTruthy();
    expect(screen.queryByText('16-bit counts')).toBeNull();
    expect(screen.queryByText('8-bit, background corrected by the acquisition software')).toBeNull();
  });

  it('names the grid of a measurement in words', () => {
    setup();
    expect(within(screen.getByTestId('measurement-1')).getByText('2 rows × 3 points')).toBeTruthy();
  });

  it('adds the hint on background only to unprocessed patterns, and does nothing else about them', () => {
    const { props } = setup();
    const hint = 'These patterns are not background corrected, so you may want to remove the background after opening.';
    expect(screen.getAllByText(hint)).toHaveLength(1);
    expect(within(radio('1: Unprocessed patterns').closest('label')).getByText(hint)).toBeTruthy();
    expect(props.onLoad).not.toHaveBeenCalled();
    expect(radio('1: Processed patterns').checked).toBe(true);
  });

  it('says that a typed pattern centre belongs to the patterns opened, in the words of the dialog', () => {
    setup();
    expect(screen.getByText('A pattern centre you enter applies to the patterns you open here. '
      + 'The other kind of patterns keeps its own.')).toBeTruthy();
  });

  it('shows a source that cannot be opened as disabled, with the reason in words', () => {
    const blocked = { ...choices, measurements: choices.measurements.map((m) => (m.name === '2'
      ? { ...m, sources: [{ ...m.sources[0], loadable: false, problem: 'loadNoPatterns' }] } : m)) };
    const { props } = setup({ inspection: blocked });
    expect(radio('2: Processed patterns').disabled).toBe(true);
    expect(screen.getByText(/holds no diffraction patterns/)).toBeTruthy();
    fireEvent.click(radio('2: Processed patterns'));
    fireEvent.click(openButton());
    expect(props.onLoad.mock.calls[0][0][0].variant.measurement).toBe('1');
  });

  it('cannot open anything when no variant can be opened, and says so', () => {
    const none = { ...choices, measurements: [{ ...choices.measurements[1],
      sources: [{ ...choices.measurements[1].sources[0], loadable: false, problem: 'loadNoPatterns' }] }] };
    const { props } = setup({ inspection: none });
    expect(screen.getByText('None of the variants in this file can be opened.')).toBeTruthy();
    expect(openButton().disabled).toBe(true);
    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Enter' });
    expect(props.onLoad).not.toHaveBeenCalled();
  });
});

describe('the preview of each source', () => {
  it('asks for one small pattern per source, by measurement and source', async () => {
    const { props } = setup();
    await waitFor(() => expect(props.loadPreview).toHaveBeenCalledTimes(3));
    expect(props.loadPreview.mock.calls.map((c) => c.slice(0, 2).join('|')).sort())
      .toEqual(['1|processed', '1|unprocessed', '2|processed']);
    await waitFor(() => expect(screen.getByAltText('Pattern from the middle of 1, Unprocessed patterns').tagName)
      .toBe('IMG'));
    expect(document.querySelectorAll('img')).toHaveLength(3);
  });

  it('never blocks the dialog when a preview fails: it says there is none and the load still works', async () => {
    const { props } = setup({ loadPreview: vi.fn(async () => { throw new Error('400 loadVariantUnknown'); }) });
    await waitFor(() => expect(screen.getAllByText('No preview')).toHaveLength(3));
    fireEvent.click(openButton());
    expect(props.onLoad).toHaveBeenCalledTimes(1);
  });

  it('shows no picture of the old file under another file while the new answer is on its way', async () => {
    const { view, props } = setup();
    await waitFor(() => expect(document.querySelectorAll('img')).toHaveLength(3));
    const oldSrc = [...document.querySelectorAll('img')].map((i) => i.getAttribute('src')).sort();
    expect(oldSrc).toEqual(['blob:preview-0', 'blob:preview-1', 'blob:preview-2']);
    // another file with the very same measurement and source names; its previews have not come yet
    const answers = [];
    const slow = vi.fn(() => new Promise((resolve) => { answers.push(resolve); }));
    view.rerender(<LoadDialog {...props} path="/d/other.h5oina" inspection={{ ...choices }} loadPreview={slow} />);
    expect(document.querySelectorAll('img')).toHaveLength(0);
    expect(screen.getAllByText('…').length).toBeGreaterThan(0);
    // the old pictures are released, once each
    expect(URL.revokeObjectURL).toHaveBeenCalledTimes(3);
    for (const u of oldSrc) expect(URL.revokeObjectURL).toHaveBeenCalledWith(u);
    // an answer of the new file then shows, and it is a new picture
    await waitFor(() => expect(answers).toHaveLength(2));   // two at a time
    await act(async () => { answers[0](new Blob(['y'])); });
    await waitFor(() => expect(document.querySelectorAll('img').length).toBeGreaterThan(0));
    for (const i of document.querySelectorAll('img')) expect(oldSrc).not.toContain(i.getAttribute('src'));
  });

  describe('a file with many measurements', () => {
    // six measurements of one source each; every preview waits until the test lets it go
    const many = {
      ...choices,
      measurements: ['a', 'b', 'c', 'd', 'e', 'f'].map((name) => ({
        ...choices.measurements[1], name, label: name })),
    };
    function slowPreviews() {
      const state = { active: 0, peak: 0, calls: [], release: [] };
      state.fn = vi.fn((measurement, source, signal) => new Promise((resolve) => {
        state.active += 1;
        state.peak = Math.max(state.peak, state.active);
        state.calls.push({ measurement, source, signal });
        state.release.push(() => { state.active -= 1; resolve(new Blob(['x'])); });
      }));
      return state;
    }
    const tick = () => act(async () => { await new Promise((r) => setTimeout(r, 0)); });

    it('asks for at most two pictures at a time and shows them all in the end', async () => {
      const p = slowPreviews();
      setup({ inspection: many, loadPreview: p.fn });
      await tick();
      expect(p.calls.map((c) => c.measurement)).toEqual(['a', 'b']);
      for (let i = 0; i < 6; i += 1) {
        await waitFor(() => expect(p.release.length).toBeGreaterThan(i));
        p.release[i]();
        await tick();
      }
      expect(p.calls.map((c) => c.measurement)).toEqual(['a', 'b', 'c', 'd', 'e', 'f']);   // the order of the rows
      expect(p.peak).toBe(2);
      await waitFor(() => expect(document.querySelectorAll('img')).toHaveLength(6));
    });

    it('stops asking when the dialog closes: the running questions are aborted and the waiting ones never start', async () => {
      const p = slowPreviews();
      const { view, props } = setup({ inspection: many, loadPreview: p.fn });
      await tick();
      expect(p.calls).toHaveLength(2);
      expect(p.calls.map((c) => c.signal.aborted)).toEqual([false, false]);
      view.rerender(<LoadDialog {...props} isOpen={false} loadPreview={p.fn} />);
      expect(p.calls.map((c) => c.signal.aborted)).toEqual([true, true]);
      p.release.forEach((r) => r());
      await tick();
      await tick();
      expect(p.calls).toHaveLength(2);               // c to f were never asked
    });
  });

  it('survives a preview function that throws at once, and one that is not given', async () => {
    setup({ loadPreview: () => { throw new Error('sync'); } });
    await waitFor(() => expect(screen.getAllByText('No preview')).toHaveLength(3));
    cleanup();
    setup({ loadPreview: undefined });
    await waitFor(() => expect(screen.getAllByText('No preview')).toHaveLength(3));
  });
});

describe('closing', () => {
  it('Cancel and Escape close without a load', () => {
    const { props } = setup();
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(props.onClose).toHaveBeenCalledTimes(1);
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(props.onClose).toHaveBeenCalledTimes(2);
    expect(props.onLoad).not.toHaveBeenCalled();
  });

  it('renders nothing while closed', () => {
    setup({ isOpen: false });
    expect(screen.queryByRole('dialog')).toBeNull();
  });
});

describe('the keyboard and the focus', () => {
  function withOpener(over = {}) {
    const props = {
      isOpen: false, path: PATH, inspection: choices, remembered: null,
      onClose: vi.fn(), onLoad: vi.fn(), loadPreview: vi.fn(async () => new Blob(['x'])), ...over,
    };
    const tree = (p) => (<><button type="button" data-testid="opener">opener</button><LoadDialog {...p} /></>);
    const view = render(tree(props));
    screen.getByTestId('opener').focus();
    return { props, view, again: (next) => view.rerender(tree({ ...props, ...next })) };
  }
  // A Tab is not a click: the browser moves the focus unless the page prevents it, so a test asks
  // whether the key was prevented and, if not, that nothing in the dialog was meant to handle it.
  const tab = (shiftKey = false) => fireEvent.keyDown(document.activeElement, { key: 'Tab', shiftKey });

  it('puts the focus on the preselected option when it opens and gives it back when it closes', () => {
    const { again } = withOpener();
    expect(document.activeElement).toBe(screen.getByTestId('opener'));
    again({ isOpen: true });
    expect(document.activeElement).toBe(radio('1: Processed patterns'));
    again({ isOpen: false });
    expect(document.activeElement).toBe(screen.getByTestId('opener'));
  });

  it('keeps Tab inside the dialog: from Open to the chosen option, and back with Shift', () => {
    const { again } = withOpener();
    again({ isOpen: true });
    openButton().focus();
    expect(tab()).toBe(false);                       // prevented: the dialog moves the focus itself
    expect(document.activeElement).toBe(radio('1: Processed patterns'));
    expect(tab(true)).toBe(false);
    expect(document.activeElement).toBe(openButton());
    // inside the dialog Tab is left to the browser
    screen.getByRole('button', { name: 'Cancel' }).focus();
    expect(tab()).toBe(true);
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Cancel' }));
  });

  it('counts only the chosen option of the group as a stop: Shift+Tab on it wraps although others precede it', () => {
    const { again } = withOpener({ remembered: { measurement: '1', pattern_source: 'unprocessed' } });
    again({ isOpen: true });
    expect(document.activeElement).toBe(radio('1: Unprocessed patterns'));
    expect(tab(true)).toBe(false);
    expect(document.activeElement).toBe(openButton());
  });

  it('leaves Tab alone while a question stands on top of the dialog', () => {
    const { again } = withOpener();
    again({ isOpen: true, escapeBlocked: true });
    openButton().focus();
    expect(tab()).toBe(true);
    expect(document.activeElement).toBe(openButton());
  });

  it('takes a Tab from outside the dialog back in', () => {
    const { again } = withOpener();
    again({ isOpen: true });
    screen.getByTestId('opener').focus();
    expect(tab()).toBe(false);
    expect(document.activeElement).toBe(radio('1: Processed patterns'));
    screen.getByTestId('opener').focus();
    expect(tab(true)).toBe(false);
    expect(document.activeElement).toBe(openButton());
  });

  it('with nothing to open the focus goes to Cancel, and Escape still closes', () => {
    const none = { ...choices, measurements: [{ ...choices.measurements[1],
      sources: [{ ...choices.measurements[1].sources[0], loadable: false, problem: 'loadNoPatterns' }] }] };
    const { props, again } = withOpener({ inspection: none });
    again({ isOpen: true });
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Cancel' }));
    expect(tab()).toBe(false);                       // the only button: Tab stays on it
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Cancel' }));
    fireEvent.keyDown(document.activeElement, { key: 'Escape' });
    expect(props.onClose).toHaveBeenCalledTimes(1);
  });

  it('ties what a source says about itself to its option, so a reader hears why one is disabled', () => {
    const blocked = { ...choices, measurements: choices.measurements.map((m) => (m.name === '2'
      ? { ...m, sources: [{ ...m.sources[0], loadable: false, problem: 'loadNoPatterns' }] } : m)) };
    setup({ inspection: blocked });
    const described = (el) => el.getAttribute('aria-describedby').split(' ')
      .map((id) => document.getElementById(id)?.textContent).join(' ');
    const raw = described(radio('1: Unprocessed patterns'));
    expect(raw).toContain('6 patterns · 5 × 4 pixels');
    expect(raw).toContain('16-bit counts');
    expect(raw).toContain('Has its own background image');
    expect(raw).toContain('not background corrected');
    const off = described(radio('2: Processed patterns'));
    expect(off).toContain('holds no diffraction patterns');
    expect(off).not.toContain('not background corrected');
    // the name itself is already the label: it is not read twice
    expect(raw).not.toContain('Unprocessed patterns');
  });
});

describe('the real German texts', () => {
  afterEach(async () => { await act(async () => { await i18n.changeLanguage('en'); }); });

  it('words the rows, the bit depths and the button in German, with the same radios', async () => {
    await act(async () => { await i18n.changeLanguage('de'); });
    setup();
    expect(screen.getByRole('dialog', { name: 'two.h5oina öffnen' })).toBeTruthy();
    expect(radio('1: Rohe Muster').checked).toBe(false);
    expect(radio('1: Bearbeitete Muster').checked).toBe(true);
    expect(screen.getByText('16 Bit, Zählwerte')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Öffnen' })).toBeTruthy();
  });
});
