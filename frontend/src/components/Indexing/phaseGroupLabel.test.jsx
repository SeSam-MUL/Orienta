// @vitest-environment jsdom
//
// The group headers of the phase pickers. The backend names two groups in
// German ("Reine Elemente", "Sonstiges"); those strings are KEYS (the backend
// sorts and compares on them), so they stay as they are and the picker
// translates them where it displays them. The key strings are read from the
// backend source, so the test cannot agree with a spelling the backend no longer
// uses.
import React from 'react';
import { describe, it, expect, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent } from '@testing-library/react';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import PhaseDropdown from './PhaseDropdown';

const BACKEND = readFileSync(
  resolve(import.meta.dirname, '../../../../phase_metadata.py'), 'utf8');
// The string returned right after `marker` in compute_element_group.
const KEY_OF = (marker) => {
  const at = BACKEND.indexOf(marker, BACKEND.indexOf('def compute_element_group'));
  const ret = BACKEND.indexOf('return "', at);
  if (at < 0 || ret < 0) throw new Error(`no group key after ${marker} in phase_metadata.py`);
  return BACKEND.slice(ret + 8, BACKEND.indexOf('"', ret + 8));
};
const OTHER_KEY = KEY_OF('if not elements:');
const PURE_KEY = KEY_OF('if len(elements) == 1:');

const files = [
  { path: 'D:/lib/CIF_Library/Al.cif', filename: 'Al.cif', formula: 'Al', element_group: PURE_KEY },
  { path: 'D:/lib/CIF_Library/Al7FeCu2.cif', filename: 'Al7FeCu2.cif', formula: 'Al7FeCu2', element_group: 'Al-Cu-Fe' },
  { path: 'D:/lib/CIF_Library/x.cif', filename: 'x.cif', formula: 'x', element_group: OTHER_KEY },
];

afterEach(() => cleanup());

const mount = () => render(
  <PhaseDropdown
    discoveredFiles={files}
    groups={[PURE_KEY, 'Al-Cu-Fe', OTHER_KEY]}
    selectedPaths={[]}
    onTogglePath={() => {}}
    onSetAll={() => {}}
    method="hough"
    open
    onClose={() => {}}
  />,
);

describe('phase picker group headers', () => {
  it('the backend keys are the German strings this test is about', () => {
    expect(PURE_KEY).toBe('Reine Elemente');
    expect(OTHER_KEY).toBe('Sonstiges');
  });

  it('shows the pure-elements group in the language of the page, not the backend key', () => {
    mount();
    expect(screen.getByText('Pure elements')).toBeTruthy();
    expect(screen.queryByText(PURE_KEY)).toBeNull();
  });

  it('shows the catch-all group in the language of the page', () => {
    mount();
    expect(screen.getByText('Other')).toBeTruthy();
    expect(screen.queryByText(OTHER_KEY)).toBeNull();
  });

  it('leaves a chemical system header (not a key) as it is', () => {
    mount();
    expect(screen.getByText('Al-Cu-Fe')).toBeTruthy();
  });

  it('the search finds a group by the name shown for it', () => {
    mount();
    fireEvent.change(screen.getByPlaceholderText('Search phase...'), { target: { value: 'pure' } });
    expect(screen.queryByTitle('Al')).not.toBeNull();
    expect(screen.queryByTitle('Al7FeCu2')).toBeNull();
  });
});
