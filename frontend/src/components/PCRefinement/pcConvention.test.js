// What a person typed as a PC number, the pattern size, and the file's kind of numbers: the small rules
// behind the PC fields.
import { describe, it, expect } from 'vitest';
import { parseNum, fmt4, patternSizeOf, preselectedConvention } from './pcConvention';

describe('parseNum', () => {
  it.each([
    ['0.5026', 0.5026], ['  0.5 ', 0.5], ['0,5026', 0.5026], ['.5', 0.5], ['5.', 5], ['1', 1], ['-0.25', -0.25], ['+2', 2],
  ])('reads %j as %d', (text, value) => {
    expect(parseNum(text)).toBe(value);
  });

  it.each(['', '   ', 'abc', '0.5.5', '1e-3', '0.5x', '0x10', '--1', ',', '.', '1,2,3', '0 5', null, undefined])(
    'does not read %j as a number', (text) => {
      expect(Number.isNaN(parseNum(text))).toBe(true);
    });
});

describe('fmt4', () => {
  it('writes four decimals', () => {
    expect(fmt4(0.5497)).toBe('0.5497');
    expect(fmt4(0.84203)).toBe('0.8420');
    expect(fmt4(1)).toBe('1.0000');
  });
});

describe('patternSizeOf', () => {
  it('gives rows and columns when both are whole numbers above zero', () => {
    expect(patternSizeOf('128', '156')).toEqual([128, 156]);
    expect(patternSizeOf(60, 80)).toEqual([60, 80]);
  });

  it.each([['', ''], ['128', ''], ['0', '156'], ['-1', '5'], ['12.5', '5'], ['a', 'b'], [null, 5]])(
    'gives null for %j x %j', (rows, cols) => {
      expect(patternSizeOf(rows, cols)).toBeNull();
    });
});

describe('preselectedConvention', () => {
  it('is the file\'s own convention when the backend knows it', () => {
    expect(preselectedConvention({ convention: 'oxford', convention_known: true })).toBe('oxford');
    expect(preselectedConvention({ convention: 'tsl', convention_known: true })).toBe('tsl');
    expect(preselectedConvention({ convention: 'bruker', convention_known: true })).toBe('bruker');
  });

  it('is nothing when the backend only said "bruker" for lack of better, or said something unknown', () => {
    expect(preselectedConvention({ convention: 'bruker', convention_known: false })).toBeNull();
    expect(preselectedConvention({ convention: 'oxford' })).toBeNull();
    expect(preselectedConvention({ convention: 'emsoft', convention_known: true })).toBeNull();
    expect(preselectedConvention(null)).toBeNull();
  });
});
