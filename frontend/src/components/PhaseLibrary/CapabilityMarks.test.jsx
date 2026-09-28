// @vitest-environment jsdom
/**
 * H S D on every row -- and, until now, not one test.
 *
 * WHY THIS FILE EXISTS. The final review mutated `marksFor` to report
 * `has: true` for every method -- every row showing all three capital,
 * claiming each of the 93 placements can be indexed three ways -- and all
 * 3022 frontend tests passed. Nothing in the suite mentioned `marksFor`,
 * `capability-marks`, `data-capability` or `data-has`.
 *
 * WHAT THAT WOULD COST. The manual calls these letters "the fact that most
 * often decides which of two similar phases you use", and it is the whole
 * point of the twin pair: `sd_0302719` can be indexed three ways,
 * `sd_1401510` only with Hough, and nothing else on the row tells them
 * apart. A reader picking the second for a Spherical run would find out
 * when the run failed, and would blame the run.
 *
 * So the letters are asserted here in the two ways they can be wrong:
 * saying a phase can do something it cannot, and saying nothing where it
 * cannot -- because "H" alone and "H s d" mean different things, and the
 * second is the one that says the other two were looked for.
 */
import { describe, it, expect, afterEach, beforeAll } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import { act } from 'react';
import i18n from '../../i18n';
import CapabilityMarks, { marksFor, METHODS } from './CapabilityMarks';

beforeAll(async () => { await act(async () => { await i18n.changeLanguage('en'); }); });
afterEach(cleanup);

const show = (caps) => render(<CapabilityMarks capabilities={caps} />);
const letters = () => [...screen.getByTestId('capability-marks').children]
  .map((el) => `${el.getAttribute('data-capability')}:${el.getAttribute('data-has')}`);

describe('marksFor, on its own', () => {
  it('reports a method that is there and one that is not', () => {
    const marks = marksFor({ hough: true, spherical: false },
      { hough: 'H', spherical: 'S', dictionary: 'D' });
    expect(marks.map((m) => [m.method, m.has])).toEqual([
      ['hough', true], ['spherical', false], ['dictionary', false]]);
  });

  it('never drops a method, so a gap cannot be mistaken for an absence', () => {
    // Three letters always, whatever the record says. A row with only "H"
    // would leave a reader unable to tell "no SHT" from "not checked".
    expect(marksFor({}, { hough: 'H', spherical: 'S', dictionary: 'D' }))
      .toHaveLength(METHODS.length);
    expect(marksFor(undefined, { hough: 'H', spherical: 'S', dictionary: 'D' })
      .every((m) => m.has === false)).toBe(true);
  });

  it('treats a missing key and an explicit false alike', () => {
    const a = marksFor({ hough: true }, { hough: 'H', spherical: 'S', dictionary: 'D' });
    const b = marksFor({ hough: true, spherical: false, dictionary: false },
      { hough: 'H', spherical: 'S', dictionary: 'D' });
    expect(a.map((m) => m.has)).toEqual(b.map((m) => m.has));
  });
});

describe('what the row shows', () => {
  it('the twin that can be indexed three ways', () => {
    show({ hough: true, spherical: true, dictionary: true });
    expect(letters()).toEqual(
      ['hough:yes', 'spherical:yes', 'dictionary:yes']);
    expect(screen.getByTestId('capability-marks').textContent).toBe('HSD');
  });

  it('and the twin that can only be indexed with Hough', () => {
    // The pair this whole feature exists for. If these two rendered the
    // same, the list could not tell them apart at all.
    show({ hough: true, spherical: false, dictionary: false });
    expect(letters()).toEqual(
      ['hough:yes', 'spherical:no', 'dictionary:no']);
    // Lower case for what is missing: still visible, plainly not a yes.
    expect(screen.getByTestId('capability-marks').textContent).toBe('Hsd');
  });

  it('a phase with no files at all says so in three letters', () => {
    show({});
    expect(letters()).toEqual(
      ['hough:no', 'spherical:no', 'dictionary:no']);
  });
});

describe('the letters are unreadable alone, so they carry the sentence', () => {
  it('names each method and whether it can be used', () => {
    show({ hough: true, spherical: false, dictionary: false });
    const title = screen.getByTestId('capability-marks').getAttribute('title');
    for (const word of ['Hough', 'Spherical', 'Dictionary']) {
      expect(title).toContain(word);
    }
    // And the two states read differently, or the tooltip says nothing.
    const [houghPart, sphericalPart] = title.split(' · ');
    expect(houghPart).not.toBe(sphericalPart);
  });

  it('the same sentence is the accessible label', () => {
    show({ hough: true });
    const el = screen.getByTestId('capability-marks');
    expect(el.getAttribute('aria-label')).toBe(el.getAttribute('title'));
    // The letters themselves are decoration once the label carries it.
    expect([...el.children].every((c) => c.getAttribute('aria-hidden') === 'true'))
      .toBe(true);
  });
});
