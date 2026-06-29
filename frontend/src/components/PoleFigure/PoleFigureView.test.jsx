// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import '@testing-library/jest-dom/vitest';
import { render, screen, waitFor, cleanup } from '@testing-library/react';

vi.mock('./useStateVersionPoll', () => ({ useStateVersionPoll: () => ({ version: 1, active_result_id: 'a', active_source_file: 'f.h5' }) }));
vi.mock('./usePoleFigure', () => ({ usePoleFigure: () => ({ image: 'BASE64', loading: false, error: null, reload: () => {} }) }));
vi.mock('../common/CoordinateSystemPanel', () => ({ default: () => <div data-testid="cs-panel" /> }));
const loadForFile = vi.fn();
vi.mock('../../stores/useFrameStore', () => ({ default: (sel) => sel({ loadForFile, spec: {}, frameSig: 'aaa' }) }));
// Real GET /api/indexing/results shape: result id is keyed `id` (not
// `result_id`), and `phases` are objects `{ id, name }` with the unindexed
// -1 phase already filtered out. Two results so the active-id lookup (`r.id
// === active_id`) is actually exercised — `results[0]` is NOT the active one.
vi.mock('../../services/api', () => ({
  indexApi: { listResults: () => Promise.resolve({ data: {
    results: [
      { id: 'other', phases: [{ id: 0, name: 'Iron' }] },
      { id: 'a', phases: [{ id: 0, name: 'Al' }, { id: 1, name: 'Si' }] },
    ],
    active_id: 'a',
  } }) },
}));

import PoleFigureView from './PoleFigureView';

afterEach(() => cleanup());

describe('PoleFigureView', () => {
  it('renders the pole-figure image and the CS panel', async () => {
    render(<PoleFigureView />);
    await waitFor(() => expect(screen.getByAltText(/pole figure/i)).toBeInTheDocument());
    expect(screen.getByTestId('cs-panel')).toBeInTheDocument();
    expect(screen.getByAltText(/pole figure/i).src).toContain('BASE64');
  });

  it('renders phase options from the ACTIVE result (objects, not array index)', async () => {
    render(<PoleFigureView />);
    // Phase names from the active result ('a') must render as option labels.
    // Before the fix, rendering an object `{id,name}` as a child crashed React.
    await waitFor(() => expect(screen.getByRole('option', { name: 'Al' })).toBeInTheDocument());
    expect(screen.getByRole('option', { name: 'Si' })).toBeInTheDocument();
    // The non-active result's phase ('Iron') must NOT appear.
    expect(screen.queryByRole('option', { name: 'Iron' })).not.toBeInTheDocument();
    // Option values are the REAL phase ids, not the array index.
    expect(screen.getByRole('option', { name: 'Al' }).value).toBe('0');
    expect(screen.getByRole('option', { name: 'Si' }).value).toBe('1');
  });
});
