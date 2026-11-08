// @vitest-environment jsdom
// A failed request of the anomaly browser must not leave the previous tab's gallery on screen: the
// drawer shows why the list could not be loaded (a coded refusal in the language of the page,
// anything else in the server's words) and none of the old rows, nor the old "N of M" count.
import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';

vi.mock('../../../services/api', () => ({
  forwardDiagApi: { summary: vi.fn(), browser: vi.fn(), thumbnail: vi.fn() },
}));
// The real list is virtualised and measures itself; here only which rows are on screen matters.
vi.mock('./GalleryList', () => ({
  default: ({ items }) => (
    <div>{(items || []).map((i) => <div key={i.rank} data-testid="row">{`row ${i.rank}`}</div>)}</div>
  ),
}));
vi.mock('./HistogramPanel', () => ({ default: () => null }));

import AnomalyBrowserDrawer from './AnomalyBrowserDrawer';
import { clearDiagnosticsCache } from './useDiagnosticsQuery';
import { forwardDiagApi } from '../../../services/api';

const GOOD = {
  total_matching: 123,
  items: [{ rank: 1 }, { rank: 2 }],
};

// The NCC tab (sort ncc_asc) answers; the PC-Sens tab (sort pc_sensitivity_desc) is refused.
const refuse = (error) => forwardDiagApi.browser.mockImplementation(async (_id, opts) => {
  if (opts.sort === 'ncc_asc') return GOOD;
  throw error;
});

const coded = (code, message) => Object.assign(new Error(message), { code });

let originalResizeObserver;
beforeEach(() => {
  clearDiagnosticsCache();
  forwardDiagApi.summary.mockResolvedValue(null);
  originalResizeObserver = globalThis.ResizeObserver;
  globalThis.ResizeObserver = class { observe() {} disconnect() {} unobserve() {} };
});
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  globalThis.ResizeObserver = originalResizeObserver;     // do not leak the stub into other files' tests
});

const open = async () => {
  render(<AnomalyBrowserDrawer open onClose={() => {}} resultId="r1" />);
  await waitFor(() => expect(screen.getAllByTestId('row')).toHaveLength(2));
  expect(screen.getByText('2 of 123')).toBeTruthy();
};

describe('the anomaly browser when a request fails', () => {
  it('shows a coded refusal in the language of the page, and none of the previous rows', async () => {
    refuse(coded('hex_neighbourhood_unsupported', 'ENGLISH PROSE FROM THE BACKEND'));
    await open();
    fireEvent.click(screen.getByText('PC-Sens'));
    // the sentence kept once in common:errors (the page has none of its own for this code)
    expect(await screen.findByText(/compares each point with its four neighbours/)).toBeTruthy();
    expect(screen.queryByText(/ENGLISH PROSE/)).toBeNull();
    expect(screen.queryAllByTestId('row')).toHaveLength(0);
    expect(screen.queryByText('2 of 123')).toBeNull();
  });

  it('shows the words of an uncoded failure, and none of the previous rows', async () => {
    refuse(new Error('Network down'));
    await open();
    fireEvent.click(screen.getByText('PC-Sens'));
    expect(await screen.findByText('Network down')).toBeTruthy();
    expect(screen.queryAllByTestId('row')).toHaveLength(0);
    expect(screen.queryByText('2 of 123')).toBeNull();
  });

  it('goes back to the list when the user returns to a tab that works', async () => {
    refuse(new Error('Network down'));
    await open();
    fireEvent.click(screen.getByText('PC-Sens'));
    await screen.findByText('Network down');
    fireEvent.click(screen.getByText('NCC'));
    await waitFor(() => expect(screen.getAllByTestId('row')).toHaveLength(2));
    expect(screen.queryByText('Network down')).toBeNull();
  });
});

describe('the anomaly browser when it is reopened after a failed request', () => {
  it('asks again, because the refusal may have passed in the meantime', async () => {
    forwardDiagApi.browser.mockRejectedValueOnce(new Error('Network down')).mockResolvedValue(GOOD);
    const drawer = (isOpen) => <AnomalyBrowserDrawer open={isOpen} onClose={() => {}} resultId="r1" />;
    const { rerender } = render(drawer(true));
    expect(await screen.findByText('Network down')).toBeTruthy();
    expect(forwardDiagApi.browser).toHaveBeenCalledTimes(1);

    rerender(drawer(false));
    rerender(drawer(true));
    await waitFor(() => expect(screen.getAllByTestId('row')).toHaveLength(2));
    expect(forwardDiagApi.browser).toHaveBeenCalledTimes(2);
    expect(screen.queryByText('Network down')).toBeNull();
  });

  it('does not ask again for a request that worked', async () => {
    forwardDiagApi.browser.mockResolvedValue(GOOD);
    const drawer = (isOpen) => <AnomalyBrowserDrawer open={isOpen} onClose={() => {}} resultId="r1" />;
    const { rerender } = render(drawer(true));
    await waitFor(() => expect(screen.getAllByTestId('row')).toHaveLength(2));
    const calls = forwardDiagApi.browser.mock.calls.length;
    rerender(drawer(false));
    rerender(drawer(true));
    await waitFor(() => expect(screen.getAllByTestId('row')).toHaveLength(2));
    expect(forwardDiagApi.browser.mock.calls.length).toBe(calls);
  });
});
