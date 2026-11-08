// @vitest-environment jsdom
//
// The line "Open file: ... PC in use: ..." that stands on every state of the PC page.
import React from 'react';
import { describe, it, expect, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import PcFileHead from './PcFileHead';
import { formatPc, samePc } from './pcFormat';

afterEach(() => cleanup());

describe('PcFileHead', () => {
  it('names the file and the PC in use with four decimals', () => {
    render(<PcFileHead file="scan_a" pc={[0.50264, 0.32716, 0.8458]} />);
    expect(screen.getByTestId('pc-file-head').textContent)
      .toBe('Open file: scan_a · PC in use: 0.5026, 0.3272, 0.8458');
  });

  it('names the kind of numbers after the PC when asked, before the origin, and not for a file without a PC', () => {
    const { rerender } = render(<PcFileHead file="scan_a" pc={[0.50264, 0.32716, 0.8458]} showConvention origin={{ kind: 'file' }} />);
    expect(screen.getByTestId('pc-file-head').textContent)
      .toBe('Open file: scan_a · PC in use: 0.5026, 0.3272, 0.8458 (Bruker) (from the file)');
    rerender(<PcFileHead file="scan_a" pc={null} showConvention />);
    expect(screen.getByTestId('pc-file-head').textContent).toBe('Open file: scan_a · PC in use: none');
  });

  it('says "none" for a file without a PC, and for no file', () => {
    render(<PcFileHead file={null} pc={null} />);
    expect(screen.getByTestId('pc-file-head').textContent).toBe('Open file: none · PC in use: none');
  });

  it('says the numbers are the mean of a per-pixel map, not one PC', () => {
    render(<PcFileHead file="scan_a" pc={[0.50264, 0.32716, 0.8458]} perPixel />);
    expect(screen.getByTestId('pc-file-head').textContent)
      .toBe('Open file: scan_a · PC in use: per-pixel map, mean 0.5026, 0.3272, 0.8458');
  });

  it.each([
    [{ kind: 'file' }, '(from the file)'],
    [{ kind: 'find' }, '(from Find PC)'],
    [{ kind: 'refine' }, '(from the refine)'],
    [{ kind: 'typed' }, '(typed)'],
    [{ kind: 'grid' }, '(from the grid calibration)'],
    [{ kind: 'calibration', calibration: { id: 'c1', name: 'SEM 2026-10-14', version: 2 } }, '(from SEM 2026-10-14, version 2)'],
  ])('says where the PC came from: %j', (origin, words) => {
    render(<PcFileHead file="scan_a" pc={[0.50264, 0.32716, 0.8458]} origin={origin} />);
    expect(screen.getByTestId('pc-file-head').textContent).toBe(`Open file: scan_a · PC in use: 0.5026, 0.3272, 0.8458 ${words}`);
  });

  it('says the origin of a per-pixel map too', () => {
    render(<PcFileHead file="scan_a" pc={[0.5, 0.3, 0.8]} perPixel origin={{ kind: 'file' }} />);
    expect(screen.getByTestId('pc-file-head').textContent)
      .toBe('Open file: scan_a · PC in use: per-pixel map, mean 0.5000, 0.3000, 0.8000 (from the file)');
  });

  it('names no origin without a PC, without an origin, for kind "none", and for a calibration it cannot name', () => {
    const cases = [[null, null], [{ kind: 'find' }, null], [{ kind: 'none' }, [0.5, 0.3, 0.8]], [null, [0.5, 0.3, 0.8]],
      [{ kind: 'calibration', calibration: null }, [0.5, 0.3, 0.8]], [{ kind: 'unheardOf' }, [0.5, 0.3, 0.8]]];
    cases.forEach(([origin, pc]) => {
      render(<PcFileHead file="scan_a" pc={pc} origin={origin} />);
      expect(screen.queryByTestId('pc-file-head-origin')).toBeNull();
      cleanup();
    });
  });

  it('a per-pixel file without a usable PC still says "none"', () => {
    render(<PcFileHead file="scan_a" pc={null} perPixel />);
    expect(screen.getByTestId('pc-file-head-pc').textContent).toBe('none');
  });

  it('does not show something that is not three numbers as a PC', () => {
    render(<PcFileHead file="scan_a" pc={[0.5, NaN, 0.5]} />);
    expect(screen.getByTestId('pc-file-head-pc').textContent).toBe('none');
  });
});

describe('pcFormat', () => {
  it('writes three numbers and nothing else', () => {
    expect(formatPc([0.5, 0.25, 1])).toBe('0.5000, 0.2500, 1.0000');
    expect(formatPc([0.5, 0.25])).toBe('');
    expect(formatPc(null)).toBe('');
  });

  it('calls two PCs the same when they agree to the four decimals shown', () => {
    expect(samePc([0.5, 0.3, 0.8], [0.50004, 0.3, 0.8])).toBe(true);
    expect(samePc([0.5, 0.3, 0.8], [0.5001, 0.3, 0.8])).toBe(false);
    expect(samePc([0.5, 0.3, 0.8], null)).toBe(false);
  });
});
