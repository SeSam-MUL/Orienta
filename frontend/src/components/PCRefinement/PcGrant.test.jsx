// @vitest-environment jsdom
//
// The line the page shows whenever it gives a file a PC: which PC, for how long, the disk untouched,
// the PC before and the one step back with its confirmation.
import React from 'react';
import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';
import PcGrant, { PcUndoLine, PcUndoDone } from './PcGrant';

afterEach(() => cleanup());

const PC = [0.5026, 0.3272, 0.8458];
const HAD = { pc: [0.5019, 0.3301, 0.842], had_pc: true };
const NONE = { pc: null, had_pc: false };

describe('PcGrant', () => {
  it('says which PC is in use for which file, until when, and that the disk is untouched', () => {
    render(<PcGrant files={['scan_a']} pc={PC} previous={[HAD]} onUndo={vi.fn()} />);
    expect(screen.getByTestId('pc-grant-text').textContent).toBe(
      'PC in use for scan_a: 0.5026, 0.3272, 0.8458. Kept until you close Orienta. The file on disk is not changed.');
  });

  it('says "files" when it is more than one', () => {
    render(<PcGrant files={['scan_a', 'scan_b']} pc={PC} previous={[HAD, NONE]} onUndo={vi.fn()} />);
    expect(screen.getByTestId('pc-grant-text').textContent).toMatch(
      /^PC in use for scan_a, scan_b: .* The files on disk are not changed\.$/);
    expect(screen.getByText('scan_a: previous PC: 0.5019, 0.3301, 0.8420')).toBeTruthy();
    expect(screen.getByText('scan_b had no PC before.')).toBeTruthy();
  });

  it('offers "Put it back" with the numbers when the file had a PC, and asks before it acts', async () => {
    const onUndo = vi.fn(() => Promise.resolve());
    render(<PcGrant files={['scan_a']} pc={PC} previous={[HAD]} onUndo={onUndo} />);
    expect(screen.getByText('Previous PC: 0.5019, 0.3301, 0.8420')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Put it back' }));
    expect(onUndo).not.toHaveBeenCalled();
    expect(screen.getByText('Put back the previous PC (0.5019, 0.3301, 0.8420) for scan_a?')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Yes, put it back' }));
    await waitFor(() => expect(onUndo).toHaveBeenCalledWith('scan_a'));
  });

  it('offers "Remove this PC" when the file had no PC, and Cancel leaves everything as it was', () => {
    const onUndo = vi.fn();
    render(<PcGrant files={['scan_a']} pc={PC} previous={[NONE]} onUndo={onUndo} />);
    expect(screen.getByText('This file had no PC before.')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Remove this PC' }));
    expect(screen.getByText('Remove the PC of scan_a? It will have no PC again.')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Yes, remove this PC' })).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(screen.getByRole('button', { name: 'Remove this PC' })).toBeTruthy();
    expect(onUndo).not.toHaveBeenCalled();
  });

  it('shows no step back for a file without a previous state, or without a handler', () => {
    const { rerender } = render(<PcGrant files={['scan_a']} pc={PC} previous={[null]} onUndo={vi.fn()} />);
    expect(screen.queryByTestId('pc-undo')).toBeNull();
    rerender(<PcGrant files={['scan_a']} pc={PC} previous={[HAD]} />);
    expect(screen.queryByTestId('pc-undo')).toBeNull();
  });

  it('puts the main button (the bands) first on the left, then Go to indexing, then the extra buttons, and leaves out what is not given', () => {
    const save = vi.fn();
    const toggle = vi.fn();
    const go = vi.fn();
    const { rerender } = render(
      <PcGrant files={['scan_a']} pc={PC} actions={[{ key: 'save', label: 'Save as calibration…', onClick: save }]}
               bands={{ shown: false, onToggle: toggle }} onGoToIndexing={go} />);
    const names = screen.getAllByRole('button').map((b) => b.textContent);
    expect(names).toEqual(['Show the bands over a pattern', 'Go to indexing', 'Save as calibration…']);
    fireEvent.click(screen.getByRole('button', { name: 'Save as calibration…' }));
    fireEvent.click(screen.getByRole('button', { name: 'Show the bands over a pattern' }));
    fireEvent.click(screen.getByRole('button', { name: 'Go to indexing' }));
    expect(save).toHaveBeenCalledTimes(1);
    expect(toggle).toHaveBeenCalledTimes(1);
    expect(go).toHaveBeenCalledTimes(1);
    rerender(<PcGrant files={['scan_a']} pc={PC} />);
    expect(screen.queryAllByRole('button')).toHaveLength(0);          // no dead button
  });

  it('can leave the numbers out of the line when the result shows them in the row above', () => {
    render(<PcGrant files={['scan_a']} pc={PC} variant="nonum" />);
    expect(screen.getByTestId('pc-grant-text').textContent)
      .toBe('PC in use for scan_a. Kept until you close Orienta. The file on disk is not changed.');
    cleanup();
    render(<PcGrant files={['scan_a', 'scan_b']} pc={PC} variant="nonum" />);
    expect(screen.getByTestId('pc-grant-text').textContent)
      .toBe('PC in use for scan_a, scan_b. Kept until you close Orienta. The files on disk are not changed.');
  });

  it('the quiet line is the same words without buttons and without a step back: the decision box has them', () => {
    render(<PcGrant files={['scan_a']} pc={PC} variant="quiet" previous={[HAD]} onUndo={vi.fn()}
                    bands={{ shown: false, onToggle: vi.fn() }} onGoToIndexing={vi.fn()}
                    actions={[{ key: 'x', label: 'Extra', onClick: vi.fn() }]} />);
    expect(screen.getByTestId('pc-grant-text').textContent)
      .toBe('PC in use for scan_a. Kept until you close Orienta. The file on disk is not changed.');
    expect(screen.queryAllByRole('button')).toHaveLength(0);
    expect(screen.queryByTestId('pc-undo')).toBeNull();
  });
});

describe('PcUndoLine', () => {
  it('replaces the question with the sentence about a step that no longer exists', async () => {
    render(<PcUndoLine file="scan_a" previous={HAD} onUndo={() => Promise.resolve({ gone: true })} />);
    fireEvent.click(screen.getByRole('button', { name: 'Put it back' }));
    fireEvent.click(screen.getByRole('button', { name: 'Yes, put it back' }));
    expect(await screen.findByText('This PC can no longer be put back: a file was loaded since.')).toBeTruthy();
    expect(screen.queryAllByRole('button')).toHaveLength(0);          // the button goes
  });

  it('keeps the button and says why when the step could not be made', async () => {
    render(<PcUndoLine file="scan_a" previous={NONE} onUndo={() => Promise.resolve({ error: 'A PC search is running.' })} />);
    fireEvent.click(screen.getByRole('button', { name: 'Remove this PC' }));
    fireEvent.click(screen.getByRole('button', { name: 'Yes, remove this PC' }));
    expect((await screen.findByRole('alert')).textContent).toBe('A PC search is running.');
    expect(screen.getByRole('button', { name: 'Remove this PC' })).toBeTruthy();
  });

  it('turns a rejected promise into a visible message instead of a hang', async () => {
    render(<PcUndoLine file="scan_a" previous={NONE} onUndo={() => Promise.reject(new Error('boom'))} />);
    fireEvent.click(screen.getByRole('button', { name: 'Remove this PC' }));
    fireEvent.click(screen.getByRole('button', { name: 'Yes, remove this PC' }));
    expect((await screen.findByRole('alert')).textContent).toBe('boom');
  });
});

describe('PcUndoDone', () => {
  it('says the previous PC is back', () => {
    render(<PcUndoDone file="scan_a" previous={HAD} />);
    expect(screen.getByRole('status').textContent).toBe(
      'Previous PC put back for scan_a: 0.5019, 0.3301, 0.8420. Kept until you close Orienta. The file on disk is not changed.');
  });

  it('says the file has no PC again, and names the copies that keep the new one', () => {
    render(<PcUndoDone file="scan_a" previous={NONE} notRestored={['scan_a copy', 'scan_a crop']} />);
    expect(screen.getByRole('status').textContent).toBe(
      'PC removed. scan_a has no PC again. Indexing is refused until there is a PC. The file on disk is not changed.');
    expect(screen.getByTestId('pc-not-restored').textContent)
      .toBe('Copies made since keep this PC: scan_a copy, scan_a crop.');
  });

  it('says nothing about copies when there are none', () => {
    render(<PcUndoDone file="scan_a" previous={NONE} notRestored={[]} />);
    expect(screen.queryByTestId('pc-not-restored')).toBeNull();
  });
});
