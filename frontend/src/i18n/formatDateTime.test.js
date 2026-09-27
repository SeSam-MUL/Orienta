/**
 * Timestamps follow the app's language, not the browser's.
 *
 * The M5 tester ran Orienta in German on a Mac set to English and read
 * "9/25/2026, 10:10:02 AM" (report, section 5, point 4). Every call site used
 * toLocaleTimeString() with no argument, which asks the browser.
 */
import { describe, it, expect, afterEach } from 'vitest';
import i18n from './index';
import { formatTime, formatDateTime, formatDate, currentLocale, localiseParams } from './formatDateTime';

const WHEN = new Date('2026-09-25T10:10:02Z');

afterEach(async () => { await i18n.changeLanguage('en'); });

describe('formatDateTime', () => {
  it('takes the locale from the app, not from the browser', async () => {
    await i18n.changeLanguage('de');
    expect(currentLocale()).toBe('de');
    // German writes the day first and uses dots; the point is that it differs
    // from the US order, not the exact glyphs a runtime chooses.
    const de = formatDate(WHEN);
    await i18n.changeLanguage('en');
    const en = formatDate(WHEN);
    expect(de).not.toBe(en);
    expect(de).toMatch(/25/);
  });

  it('formats a time and a date-time without throwing', () => {
    expect(formatTime(WHEN)).toMatch(/\d/);
    expect(formatDateTime(WHEN)).toMatch(/\d/);
    expect(formatTime()).toMatch(/\d/);      // defaults to now
  });

  it('accepts an ISO string as well as a Date', () => {
    expect(formatDateTime('2026-09-25T10:10:02Z')).toMatch(/\d/);
  });

  it('gives an em dash for something that is not a date', () => {
    // The history table used to print "Invalid Date" into a cell.
    expect(formatDateTime('not a date')).toBe('—');
    expect(formatDateTime(undefined)).toBe('—');
    expect(formatDate(null)).toBe('—');
  });
});

describe('localiseParams', () => {
  it('formats a number for the sentence it lands in', async () => {
    await i18n.changeLanguage('de');
    const de = localiseParams({ median: 12345 }).median;
    await i18n.changeLanguage('en');
    const en = localiseParams({ median: 12345 }).median;
    // The point: "12,345" in a German sentence reads as twelve point three
    // four five, because a comma is the decimal separator there.
    expect(de).not.toBe(en);
    expect(en).toBe('12,345');
  });

  it('leaves strings and everything else alone', () => {
    expect(localiseParams({ elements: 'Fe, Mn' }).elements).toBe('Fe, Mn');
    expect(localiseParams({ a: null, b: undefined, c: true }))
      .toEqual({ a: null, b: undefined, c: true });
    expect(localiseParams(null)).toEqual({});
  });

  it('does not turn a non-finite number into "NaN"', () => {
    expect(localiseParams({ x: NaN }).x).toBeNaN();
    expect(localiseParams({ x: Infinity }).x).toBe(Infinity);
  });
});
