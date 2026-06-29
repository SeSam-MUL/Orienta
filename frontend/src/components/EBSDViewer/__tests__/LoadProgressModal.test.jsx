/**
 * Tests for LoadProgressModal — stage breadcrumb, ticking elapsed, error state.
 *
 * The five behaviours under test:
 *   1. Renders nothing when isOpen=false
 *   2. Stage breadcrumb lists all four stage labels and marks the active stage
 *      with aria-current="step"; the backend message is visible.
 *   3. Elapsed seconds are formatted to 1 decimal ("2.3 s").
 *   4. Error stage shows the error text and renders a Close button.
 *   5. After 30 s of elapsed time, a "backend might be slow" hint appears.
 */
// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { act } from 'react';
import { afterEach } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import LoadProgressModal from '../LoadProgressModal';

afterEach(() => cleanup());

describe('LoadProgressModal', () => {
  it('renders nothing when isOpen is false', () => {
    const { container } = render(
      <LoadProgressModal isOpen={false} progressState={null} onClose={() => {}} />
    );
    expect(container.innerHTML).toBe('');
  });

  it('renders a stage breadcrumb with the current stage highlighted', () => {
    render(
      <LoadProgressModal
        isOpen={true}
        progressState={{
          stage: 'building_signal',
          stage_idx: 2,
          stage_total: 4,
          elapsed_seconds: 0.45,
          message: 'Building lazy signal',
        }}
        onClose={() => {}}
      />
    );
    // All four stage names present (use getAllByText because the breadcrumb
    // label "Building lazy signal" may equal the backend `message`, so two
    // elements can contain the same visible text).
    expect(screen.getAllByText(/reading metadata/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/building lazy signal/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/detecting features/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/finalising/i).length).toBeGreaterThan(0);
    // The backend message is visible in the detail line (its own element).
    expect(screen.getByTestId('progress-message')).toHaveTextContent('Building lazy signal');
    // The active stage cell is queryable directly via aria-current="step".
    const active = screen.getByRole('listitem', { current: 'step' });
    expect(active).toHaveAttribute('aria-current', 'step');
    expect(active).toHaveTextContent(/building lazy signal/i);
  });

  it('shows elapsed time formatted as seconds', () => {
    render(
      <LoadProgressModal
        isOpen={true}
        progressState={{
          stage: 'reading_metadata',
          stage_idx: 1, stage_total: 4,
          elapsed_seconds: 2.345,
          message: 'Reading file headers',
        }}
        onClose={() => {}}
      />
    );
    expect(screen.getByText(/2\.3 s/)).toBeInTheDocument();
  });

  it('renders error state with the error message and does not auto-dismiss', () => {
    const onClose = vi.fn();
    render(
      <LoadProgressModal
        isOpen={true}
        progressState={{
          stage: 'error',
          stage_idx: 0, stage_total: 4,
          elapsed_seconds: 0.5,
          message: 'File not found: /bad.h5oina',
          error: 'File not found: /bad.h5oina',
        }}
        onClose={onClose}
      />
    );
    expect(screen.getByText(/file not found/i)).toBeInTheDocument();
    // Close button is rendered and wired
    const close = screen.getByRole('button', { name: /close/i });
    expect(close).toBeInTheDocument();
  });

  it('shows the "backend might be slow" hint when elapsed > 30 s', () => {
    render(
      <LoadProgressModal
        isOpen={true}
        progressState={{
          stage: 'reading_metadata',
          stage_idx: 1, stage_total: 4,
          elapsed_seconds: 35.0,
          message: 'Reading file headers',
        }}
        onClose={() => {}}
      />
    );
    expect(screen.getByText(/backend might be slow/i)).toBeInTheDocument();
  });

  // A11y nits — focus management, aria-live, Escape-to-close

  it('focuses the Close button when entering terminal state (complete)', () => {
    const { rerender } = render(
      <LoadProgressModal
        isOpen={true}
        progressState={{ stage: 'finalising', stage_idx: 4, stage_total: 4, elapsed_seconds: 1.0, message: 'Finalising' }}
        onClose={() => {}}
      />
    );
    rerender(
      <LoadProgressModal
        isOpen={true}
        progressState={{ stage: 'complete', stage_idx: 4, stage_total: 4, elapsed_seconds: 1.2, message: 'Done' }}
        onClose={() => {}}
      />
    );
    const close = screen.getByRole('button', { name: /close/i });
    expect(close).toHaveFocus();
  });

  it('announces stage messages via aria-live', () => {
    render(
      <LoadProgressModal
        isOpen={true}
        progressState={{ stage: 'building_signal', stage_idx: 2, stage_total: 4, elapsed_seconds: 0.3, message: 'Building lazy signal' }}
        onClose={() => {}}
      />
    );
    const msg = screen.getByTestId('progress-message');
    expect(msg).toHaveAttribute('aria-live', 'polite');
    expect(msg).toHaveAttribute('aria-atomic', 'true');
  });

  it('Escape key dismisses the modal in terminal state', () => {
    const onClose = vi.fn();
    render(
      <LoadProgressModal
        isOpen={true}
        progressState={{ stage: 'error', stage_idx: 0, stage_total: 4, elapsed_seconds: 0.5, message: 'oops', error: 'oops' }}
        onClose={onClose}
      />
    );
    act(() => {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('Escape key dismisses during in-progress (Hide semantics — load continues in background)', () => {
    // Updated 2026-05-27: the prior design refused to close the modal while a
    // load was in progress, on the theory that cancelling mid-load would
    // leave the backend in an indeterminate state. That assumption is wrong
    // when the backend has died — the load is then never going to complete
    // OR error within the axios 5-min timeout, and the user is trapped at a
    // frozen "0.0 s elapsed". Escape is now an always-available hide hatch;
    // the backend load (if any) continues in the background, and the user
    // is notified by toast on completion just like always.
    const onClose = vi.fn();
    render(
      <LoadProgressModal
        isOpen={true}
        progressState={{ stage: 'reading_metadata', stage_idx: 1, stage_total: 4, elapsed_seconds: 0.1, message: 'Reading file headers' }}
        onClose={onClose}
      />
    );
    act(() => {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('Hide button is shown during in-progress and dismisses the modal on click', () => {
    const onClose = vi.fn();
    render(
      <LoadProgressModal
        isOpen={true}
        progressState={{ stage: 'reading_metadata', stage_idx: 1, stage_total: 4, elapsed_seconds: 0.1, message: 'Reading file headers' }}
        onClose={onClose}
      />
    );
    const hide = screen.getByRole('button', { name: /hide/i });
    expect(hide).toBeInTheDocument();
    act(() => hide.click());
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
