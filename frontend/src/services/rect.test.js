// @vitest-environment jsdom
/**
 * A rectangle dragged upwards is the same rectangle.
 *
 * Drawing a region from bottom-right to top-left sends row_start > row_end,
 * and the two endpoints that take a rectangle disagree about what that means:
 * /api/eds/region-quantify answers 400 ("row_start/col_start must be <=
 * row_end/col_end" — field names the user never typed), while
 * /api/eds/phase-map/assign-region takes an empty slice and reports 0 pixels
 * painted, which looks like nothing happened at all. Logged 2026-09-10
 * 13:15:51.
 *
 * The rectangle is normalised once, where it leaves the client, so no caller
 * has to remember — and so a rectangle that never came from the drag hook
 * (the four Region Average number fields are typed by hand) is covered too.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';

vi.mock('./errorReporter', () => ({
  reportError: vi.fn(),
  reportUiError: vi.fn(),
  installGlobalErrorReporter: vi.fn(),
}));

import api, { edsApi } from './api';
import { normalizeRect } from './rect';

// Captures the body axios would put on the wire.
let sent;
function capturingAdapter() {
  return (config) => {
    sent = typeof config.data === 'string' ? JSON.parse(config.data) : config.data;
    return Promise.resolve({
      status: 200, data: {}, statusText: 'OK', headers: {}, config,
    });
  };
}

beforeEach(() => {
  sent = undefined;
  api.defaults.adapter = capturingAdapter();
});

describe('normalizeRect', () => {
  it('orders a rectangle dragged bottom-right to top-left', () => {
    expect(normalizeRect({ rowStart: 40, rowEnd: 10, colStart: 90, colEnd: 20 }))
      .toEqual({ rowStart: 10, rowEnd: 40, colStart: 20, colEnd: 90 });
  });

  it('leaves an already ordered rectangle untouched', () => {
    expect(normalizeRect({ rowStart: 10, rowEnd: 40, colStart: 20, colEnd: 90 }))
      .toEqual({ rowStart: 10, rowEnd: 40, colStart: 20, colEnd: 90 });
  });

  it('keeps a single pixel a single pixel', () => {
    expect(normalizeRect({ rowStart: 7, rowEnd: 7, colStart: 3, colEnd: 3 }))
      .toEqual({ rowStart: 7, rowEnd: 7, colStart: 3, colEnd: 3 });
  });

  it('normalises each axis on its own', () => {
    expect(normalizeRect({ rowStart: 5, rowEnd: 9, colStart: 90, colEnd: 20 }))
      .toEqual({ rowStart: 5, rowEnd: 9, colStart: 20, colEnd: 90 });
  });

  it('takes the numbers the number inputs actually hold (strings)', () => {
    expect(normalizeRect({ rowStart: '40', rowEnd: '10', colStart: '2', colEnd: '8' }))
      .toEqual({ rowStart: 10, rowEnd: 40, colStart: 2, colEnd: 8 });
  });

  it('floors fractional coordinates rather than sending them to a pixel index', () => {
    expect(normalizeRect({ rowStart: 3.7, rowEnd: 1.2, colStart: 0, colEnd: 4.9 }))
      .toEqual({ rowStart: 1, rowEnd: 3, colStart: 0, colEnd: 4 });
  });

  it('hands an empty field to the backend rather than reading it as pixel 0', () => {
    // `Number('') || 0` is 0, which turns "nothing typed here" into the corner
    // of the scan and averages a region nobody asked for. Passed through, it
    // is a 422 that names the field.
    expect(normalizeRect({ rowStart: '', rowEnd: '40', colStart: '2', colEnd: '8' }))
      .toEqual({ rowStart: '', rowEnd: '40', colStart: '2', colEnd: '8' });
    expect(normalizeRect({ rowStart: '  ', rowEnd: 40, colStart: 2, colEnd: 8 }))
      .toEqual({ rowStart: '  ', rowEnd: 40, colStart: 2, colEnd: 8 });
  });

  it('hands anything else that is not a number through unchanged too', () => {
    for (const junk of ['abc', NaN, Infinity, null, undefined, {}]) {
      expect(normalizeRect({ rowStart: 40, rowEnd: junk, colStart: 90, colEnd: 20 }))
        .toEqual({ rowStart: 40, rowEnd: junk, colStart: 90, colEnd: 20 });
    }
  });

  it('0 is a coordinate, not a missing value', () => {
    expect(normalizeRect({ rowStart: 0, rowEnd: 0, colStart: '0', colEnd: 0 }))
      .toEqual({ rowStart: 0, rowEnd: 0, colStart: 0, colEnd: 0 });
  });
});

describe('the rectangle endpoints send an ordered rectangle', () => {
  it('region-quantify', async () => {
    await edsApi.regionQuantify(40, 10, 90, 20, 'at_pct');
    expect(sent).toMatchObject({
      row_start: 10, row_end: 40, col_start: 20, col_end: 90, display_mode: 'at_pct',
    });
  });

  it('assign-region — a reversed drag painted nothing at all', async () => {
    await edsApi.assignRegion(40, 10, 90, 20, 3);
    expect(sent).toMatchObject({
      row_start: 10, row_end: 40, col_start: 20, col_end: 90, phase_index: 3,
    });
  });

  it('an ordinary rectangle goes out exactly as before', async () => {
    await edsApi.regionQuantify(10, 40, 20, 90, 'counts');
    expect(sent).toMatchObject({
      row_start: 10, row_end: 40, col_start: 20, col_end: 90, display_mode: 'counts',
    });
  });
});
