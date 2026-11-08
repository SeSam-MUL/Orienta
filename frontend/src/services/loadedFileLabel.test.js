import { describe, it, expect, afterEach } from 'vitest';
import { loadedFileLabel, apexMeasurementName, isNonDefaultVariant } from './loadedFileLabel';
import i18n from '../i18n';

describe('loadedFileLabel: a file opened as a variant other than its default', () => {
  const unprocessed = { path: '/d/multi.h5oina', name: 'multi · 2 · unprocessed',
    variant: { measurement: '2', pattern_source: 'unprocessed' } };
  const second = { path: '/d/multi.h5oina', name: 'multi · 2', variant: { measurement: '2', pattern_source: 'processed' } };
  const real = (key, options) => i18n.t(`ebsdviewer:${key}`, options);
  afterEach(async () => { await i18n.changeLanguage('en'); });

  it('keeps the name the backend gave, which already carries the variant', () => {
    expect(loadedFileLabel(unprocessed)).toBe('multi · 2 · unprocessed');
    expect(loadedFileLabel(second)).toBe('multi · 2');
  });

  it('writes the word for the unprocessed source in the language of the page (real texts)', async () => {
    expect(loadedFileLabel(unprocessed, real)).toBe('multi · 2 · unprocessed');
    await i18n.changeLanguage('de');
    expect(loadedFileLabel(unprocessed, real)).toBe('multi · 2 · roh');
    // a name that does not end in the backend's word is left alone, and so is a measurement part
    expect(loadedFileLabel(second, real)).toBe('multi · 2');
  });

  it('knows a non-default variant by the variant and the parts in the name, not by the parts alone', () => {
    expect(isNonDefaultVariant(unprocessed)).toBe(true);
    expect(isNonDefaultVariant(second)).toBe(true);
    expect(isNonDefaultVariant({ name: 'a · b' })).toBe(false);
    expect(isNonDefaultVariant({ name: 'scan', variant: { measurement: '1', pattern_source: 'processed' } })).toBe(false);
    expect(isNonDefaultVariant(undefined)).toBe(false);
  });
});

describe('loadedFileLabel', () => {
  it('leads with the measurement the user ticked, then the pattern file', () => {
    const entry = { path: '/d/a.up1', name: 'a', apex: { measurement_name: 'Area 1 / OIM Map 2', h5_path: '/d/s.h5' } };
    expect(loadedFileLabel(entry)).toBe('Area 1 / OIM Map 2 (a)');
  });
  it('keeps the name of any other file', () => {
    expect(loadedFileLabel({ path: '/d/a.h5oina', name: 'scan A' })).toBe('scan A');
    expect(loadedFileLabel({ name: 'scan A', apex: null })).toBe('scan A');
    expect(loadedFileLabel({ name: 'scan A', apex: { measurement_name: '  ' } })).toBe('scan A');
    expect(loadedFileLabel(undefined)).toBe('');
  });
  it('reads the measurement name only when it is text', () => {
    expect(apexMeasurementName({ apex: { measurement_name: 5 } })).toBe('');
    expect(apexMeasurementName({ apex: { measurement_name: ' Area 1 / OIM Map 1 ' } })).toBe('Area 1 / OIM Map 1');
  });
});
