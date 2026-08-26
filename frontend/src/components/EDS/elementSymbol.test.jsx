/**
 * Line name in, element symbol out.
 *
 * The regression this pins: the rule and region editors were offering
 * "Al Kα1" as an element, while everything the clause is evaluated against
 * is keyed "Al". A clause written that way names an element the dataset does
 * not have, so it blocks instead of matching — silently, from the user's
 * side. Found by driving the real page against a real Oxford file.
 */
import { describe, it, expect } from 'vitest';

import { elementSymbol, elementSymbols } from './elementSymbol';

describe('elementSymbol', () => {
  it('strips the X-ray line', () => {
    expect(elementSymbol('Al Kα1')).toBe('Al');
    expect(elementSymbol('Fe Ka1')).toBe('Fe');
  });

  it('handles the comma-bearing line names Aztec writes', () => {
    expect(elementSymbol('C Kα1,2')).toBe('C');
  });

  it('leaves a bare symbol alone', () => {
    expect(elementSymbol('Si')).toBe('Si');
  });

  it('reads the object shapes the element list arrives in', () => {
    expect(elementSymbol({ symbol: 'Mn' })).toBe('Mn');
    expect(elementSymbol({ name: 'Mn Kα1' })).toBe('Mn');
    expect(elementSymbol({ element: 'Zn' })).toBe('Zn');
  });

  it('prefers an explicit symbol over a line name', () => {
    expect(elementSymbol({ symbol: 'Cu', name: 'Cu Kα1' })).toBe('Cu');
  });

  it('gives an empty string rather than throwing on nothing', () => {
    expect(elementSymbol(null)).toBe('');
    expect(elementSymbol({})).toBe('');
    expect(elementSymbol('   ')).toBe('');
  });
});

describe('elementSymbols', () => {
  it('converts a whole element list', () => {
    expect(elementSymbols([{ name: 'Al Kα1' }, { name: 'Si Kα1' }]))
      .toEqual(['Al', 'Si']);
  });

  it('collapses two lines of the same element into one entry', () => {
    // Offering "Fe" twice in a dropdown is how you get two clauses on one
    // element that silently contradict each other.
    expect(elementSymbols([{ name: 'Fe Kα1' }, { name: 'Fe Lα1' }]))
      .toEqual(['Fe']);
  });

  it('drops entries with nothing in them', () => {
    expect(elementSymbols([{ name: '' }, 'Si', null])).toEqual(['Si']);
  });

  it('survives no list at all', () => {
    expect(elementSymbols(undefined)).toEqual([]);
  });
});
