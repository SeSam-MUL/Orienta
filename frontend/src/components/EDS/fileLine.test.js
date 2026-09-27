/**
 * The "written" list names a picture by its size, a table by its rows.
 *
 * The M5 tester read "phase_map.png — 462 Zeilen" for a 21 x 22 px map: the
 * backend's `rows` on a PNG is the pixel count, and the dialog printed it as
 * rows of something.
 */
import { describe, expect, it } from 'vitest';
import { fileLine } from './writtenFileLine';

const t = (key, vars) => `${key}:${JSON.stringify(vars)}`;

describe('fileLine', () => {
  it('describes a picture by width x height', () => {
    expect(fileLine({ name: 'phase_map.png', rows: 462, bytes: 900, width: 22, height: 21 }, t))
      .toBe('export.doneImage:{"name":"phase_map.png","width":"22","height":"21"}');
  });

  it('describes a table by its rows', () => {
    expect(fileLine({ name: 'particles.csv', rows: 135, bytes: 1 }, t))
      .toBe('export.doneRows:{"name":"particles.csv","rows":"135"}');
  });

  it('treats a picture without a size like before rather than inventing one', () => {
    expect(fileLine({ name: 'old.png', rows: 462, bytes: 1, width: null }, t))
      .toBe('export.doneRows:{"name":"old.png","rows":"462"}');
  });

  it('formats through the caller’s number formatter', () => {
    const fmt = (v) => `<${v}>`;
    expect(fileLine({ name: 'f.png', width: 352, height: 336 }, t, fmt))
      .toContain('"width":"<352>","height":"<336>"');
  });
});
