// @vitest-environment jsdom
/**
 * The page, mounted: a Start that the backend refuses with a load code (HTTP 400 and the
 * `X-Orienta-Code` header, as the measurement readers answer) puts the translated sentence in the
 * log, not the server's English prose and not "Request failed with status code 400".
 *
 * `variantCodeSurfaces.test.jsx` proves the strings and pins the wiring by reading the page's
 * source; this mounts the page with the API stubbed, so renaming the call or losing the handler
 * fails here as well. Everything but the start request answers with an empty body.
 */
import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import i18n from '../../i18n';

const PROSE = 'start_indexing cannot read the processed patterns; English prose from the server';
const refused = (code) => ({
  message: 'Request failed with status code 400',
  response: { status: 400, headers: { 'x-orienta-code': code }, data: { detail: PROSE } },
});

const mocks = vi.hoisted(() => ({ start: vi.fn() }));

vi.mock('../../services/api', async (importOriginal) => {
  const actual = await importOriginal();
  const empty = () => Promise.resolve({ data: {} });
  // every method of an api group answers with an empty body, except the ones the page needs to be
  // "loaded" and the start request under test
  const group = (overrides = {}) => new Proxy({}, {
    get: (_t, name) => (name in overrides ? overrides[name] : empty),
  });
  return {
    ...actual,
    indexApi: group({ start: (...a) => mocks.start(...a) }),
    ebsdApi: group({
      datasets: () => Promise.resolve({ data: { datasets: [{ name: 'scan' }], active: 'scan' } }),
    }),
    pcApi: group(),
    edsApi: group(),
    dictionaryGpuApi: group(),
    phaseMapApi: group(),
    collectionsApi: group(),
    getGpuStatus: empty,
  };
});

// the navigation map is a canvas drawing; it has nothing to do with the start request
vi.mock('./NavigationCanvas', () => ({ default: () => null }));

import IndexingPage from './IndexingPage';

beforeEach(async () => {
  globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
  mocks.start.mockReset();
  await i18n.changeLanguage('en');
});
afterEach(() => cleanup());

const EN = (await import('../../locales/en/common.json')).default;
const DE = (await import('../../locales/de/common.json')).default;

async function startAndWatch(lang, code) {
  await i18n.changeLanguage(lang);
  mocks.start.mockRejectedValue(refused(code));
  render(<IndexingPage isActive />);
  const label = (await import(`../../locales/${lang}/indexing.json`)).default.actions.startIndexing;
  const start = await screen.findByRole('button', { name: label });
  await waitFor(() => expect(start).toBeEnabled(), { timeout: 4000 });
  fireEvent.click(start);
  await waitFor(() => expect(mocks.start).toHaveBeenCalled(), { timeout: 4000 });
}

describe('Indexing page, mounted', () => {
  it('en: a refused start shows the sentence of its code', async () => {
    await startAndWatch('en', 'variantNotSupported');
    const sentence = EN.errors.variantNotSupported;
    await waitFor(() => expect(document.body.textContent).toContain(sentence), { timeout: 4000 });
    expect(document.body.textContent).not.toContain('English prose');
    expect(document.body.textContent).not.toContain('Request failed with status code');
  });

  it('de: in the reader\'s language', async () => {
    await startAndWatch('de', 'variantNotSupported');
    await waitFor(() => expect(document.body.textContent).toContain(DE.errors.variantNotSupported),
      { timeout: 4000 });
    expect(document.body.textContent).not.toContain('English prose');
  });
});
