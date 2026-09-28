/**
 * What the naming form actually puts on the wire.
 *
 * WHY THIS FILE EXISTS. `NameEditor.test.jsx` asserts against a stubbed
 * `save`, so it can only say what the component MEANT. The service reads
 * `display_name` with three distinct meanings -- a string sets the name,
 * the empty string clears it, `null` leaves it alone (`phase_synonyms.py`:
 * `if display_name is not None:`) -- and the api client collapsed two of
 * them: `displayName || null` turned "" into `null`.
 *
 * So the one documented way to take a name away ("clear the field and
 * save") meant "leave unchanged". Measured against the real service: the
 * name survived both a cleared name field and a cleared everything, and
 * the editor then repopulated the box with the old name. Two green tests
 * covered the two halves and neither crossed the seam -- the component
 * test stubbed the save, the backend test pinned the opposite meaning of
 * `null`, and nothing in between ever looked at the body.
 *
 * This is that missing middle: the real `phaseLibraryApi`, a mocked axios,
 * and an assertion on the bytes.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('axios', () => {
  const inst = {
    get: vi.fn(() => Promise.resolve({ data: {} })),
    put: vi.fn(() => Promise.resolve({ data: {} })),
    post: vi.fn(() => Promise.resolve({ data: {} })),
    patch: vi.fn(() => Promise.resolve({ data: {} })),
    delete: vi.fn(() => Promise.resolve({ data: {} })),
    interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } },
  };
  return { default: { create: () => inst, __inst: inst }, __inst: inst };
});

import axios from 'axios';
import { phaseLibraryApi } from '../../services/api';

const inst = () => axios.__inst;
const body = () => inst().put.mock.calls
  .filter((c) => c[0] === '/api/phase-library/names').at(-1)[1];

beforeEach(() => { vi.clearAllMocks(); });

describe('PUT /names -- the three meanings of display_name', () => {
  it('a name is sent as itself', async () => {
    await phaseLibraryApi.setNames(
      { key: 'sd_1', displayName: 'S-Phase', searchTerms: [], author: 'seb' });
    expect(body().display_name).toBe('S-Phase');
  });

  it('an EMPTIED field is sent as "", which is what clears the name', async () => {
    // The whole point. `''` must survive the api client unchanged.
    await phaseLibraryApi.setNames(
      { key: 'sd_1', displayName: '', searchTerms: ['alpha'], author: 'seb' });
    expect(body().display_name).toBe('');
    // ... and the terms still travel, so clearing a name is not a way to
    // lose them.
    expect(body().search_terms).toEqual(['alpha']);
  });

  it('and a caller that says nothing at all still means "leave it"', async () => {
    // Nothing in the app sends this today; it is the third meaning, and it
    // has to stay reachable or a partial update becomes impossible.
    await phaseLibraryApi.setNames(
      { key: 'sd_1', searchTerms: ['alpha'], author: 'seb' });
    expect(body().display_name).toBe(null);
  });
});
