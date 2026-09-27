// @vitest-environment jsdom
/**
 * The citation refresh — the point of the whole feature.
 *
 * An add-on's run writes its step into the SAME indexing result, so the panel's
 * `resultId` does not change and its effect never re-runs. The add-on's
 * sentence would then be missing from the one place the feature exists to put
 * it, and the methods paragraph on screen would go on describing the run
 * before this one.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, waitFor } from '@testing-library/react';

vi.mock('../../services/api', () => ({
  citationsApi: { forResult: vi.fn() },
}));

import { citationsApi } from '../../services/api';
import CitationsPanel from './CitationsPanel';
import useAddonLayerRequests from '../../stores/useAddonLayerRequests';

afterEach(cleanup);

const payload = (methods) => ({ data: {
  methods, bibtex: '@misc{orienta}', plain: 'Orienta', steps: [],
  undeclared: [] } });

beforeEach(() => {
  vi.clearAllMocks();
  useAddonLayerRequests.setState({ citationTick: 0 });
  citationsApi.forResult.mockResolvedValue(payload('Indexed with Orienta.'));
});

describe('the panel and an add-on run', () => {
  it('re-reads when a run reports, although the result id did not change', async () => {
    render(<CitationsPanel resultId="res-1" />);
    await waitFor(() => expect(citationsApi.forResult).toHaveBeenCalledTimes(1));

    citationsApi.forResult.mockResolvedValue(
      payload('Indexed with Orienta. Band contrast was fitted with 2 Gaussian components.'));
    useAddonLayerRequests.getState().bumpCitations();

    await waitFor(() => expect(citationsApi.forResult).toHaveBeenCalledTimes(2));
    expect(await screen.findByText(/2 Gaussian components/)).toBeTruthy();
  });

  it('does not re-read on an unrelated re-render', async () => {
    // The dependency must be the TRIGGER, not the store object or a fresh
    // selector result: a panel that re-fetched on every render would put a
    // request on the wire for every keystroke elsewhere on the page.
    const { rerender } = render(<CitationsPanel resultId="res-1" />);
    await waitFor(() => expect(citationsApi.forResult).toHaveBeenCalledTimes(1));
    rerender(<CitationsPanel resultId="res-1" />);
    rerender(<CitationsPanel resultId="res-1" />);
    await new Promise((r) => setTimeout(r, 30));
    expect(citationsApi.forResult).toHaveBeenCalledTimes(1);
  });

  it('a result switch still re-reads, as it always did', async () => {
    const { rerender } = render(<CitationsPanel resultId="res-1" />);
    await waitFor(() => expect(citationsApi.forResult).toHaveBeenCalledTimes(1));
    rerender(<CitationsPanel resultId="res-2" />);
    await waitFor(() => expect(citationsApi.forResult).toHaveBeenCalledTimes(2));
    expect(citationsApi.forResult).toHaveBeenLastCalledWith('res-2');
  });
});
