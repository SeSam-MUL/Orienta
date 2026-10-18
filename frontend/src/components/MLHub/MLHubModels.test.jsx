// @vitest-environment jsdom
/**
 * ML Hub: the model list comes from the stored training runs (/api/ml/models),
 * clicking a run asks the backend to load it, and the learning rate typed in
 * the Training tab is sent with the training request.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';

const mocks = vi.hoisted(() => ({
  status: vi.fn(),
  models: vi.fn(),
  loadModel: vi.fn(),
  train: vi.fn(),
  trainStatus: vi.fn(),
  predict: vi.fn(),
  clearStore: vi.fn(),
}));

vi.mock('../../services/api', () => ({ mlApi: mocks }));

import MLHubPage from './MLHubPage';

const RUN_NEW = {
  run_id: '20261014T120000Z', phase_names: ['Al', 'Si'], n_samples: 300,
  source_file: 'SampleB.h5oina', final_val_loss: 0.1234, loaded: true,
};
const RUN_OLD = {
  run_id: '20261013T090000Z', phase_names: ['Al'], n_samples: 40,
  source_file: 'Older.h5oina', final_val_loss: 0.5, loaded: false,
};

function statusFor(runId) {
  return {
    data: {
      available: true, model_loaded: true, loaded_run_id: runId,
      phase_names: ['Al', 'Si'], store_stats: { total_samples: 10 },
    },
  };
}

beforeEach(() => {
  Object.values(mocks).forEach((m) => m.mockReset());
  mocks.status.mockResolvedValue(statusFor(RUN_NEW.run_id));
  mocks.models.mockResolvedValue({ data: { models: [RUN_NEW, RUN_OLD] } });
  mocks.loadModel.mockResolvedValue({ data: { loaded_run_id: RUN_OLD.run_id } });
});
afterEach(() => cleanup());

describe('ML Hub model list', () => {
  it('lists the stored runs from /models with their metadata', async () => {
    render(<MLHubPage isActive />);
    expect(await screen.findByText(RUN_NEW.run_id)).toBeTruthy();
    expect(screen.getByText(RUN_OLD.run_id)).toBeTruthy();
    expect(screen.getByText(/Al, Si · 300 samples · SampleB\.h5oina/)).toBeTruthy();
    expect(screen.getByText(/val loss 0\.1234/)).toBeTruthy();
    expect(mocks.models).toHaveBeenCalled();
  });

  it('shows which run the backend has loaded', async () => {
    render(<MLHubPage isActive />);
    expect(await screen.findByText(`Loaded run: ${RUN_NEW.run_id}`)).toBeTruthy();
  });

  it('clicking a run calls the load route with its run id and re-reads the list', async () => {
    render(<MLHubPage isActive />);
    fireEvent.click(await screen.findByText(RUN_OLD.run_id));
    await waitFor(() => expect(mocks.loadModel).toHaveBeenCalledWith(RUN_OLD.run_id));
    // after loading, the status and list are asked again so the page shows
    // what the backend really holds (not just the clicked card)
    await waitFor(() => expect(mocks.models.mock.calls.length).toBeGreaterThanOrEqual(2));
    expect(mocks.status.mock.calls.length).toBeGreaterThanOrEqual(2);
  });

  it('says so when the backend refuses to load the run', async () => {
    mocks.loadModel.mockRejectedValue({ response: { data: { detail: 'Unknown model run' } } });
    render(<MLHubPage isActive />);
    fireEvent.click(await screen.findByText(RUN_OLD.run_id));
    expect(await screen.findByText(/Could not load the model: Unknown model run/)).toBeTruthy();
  });
});

describe('ML Hub training request', () => {
  it('sends the learning rate typed in the Training tab', async () => {
    mocks.train.mockResolvedValue({ data: { task_id: 'abc' } });
    mocks.trainStatus.mockResolvedValue({
      data: { status: 'completed', epoch: 1, total: 1, log: [], result: { phase_names: ['Al'], best_val_loss: 0.1 } },
    });
    render(<MLHubPage isActive />);
    await screen.findByText(RUN_NEW.run_id);
    fireEvent.click(screen.getByRole('tab', { name: /training/i }));

    const lr = await screen.findByTitle(/learning rate/i, { selector: 'input' });
    fireEvent.change(lr, { target: { value: '0.0042' } });
    fireEvent.click(screen.getByRole('button', { name: /^start training$/i }));

    await waitFor(() => expect(mocks.train).toHaveBeenCalled());
    expect(mocks.train.mock.calls[0][0].learningRate).toBeCloseTo(0.0042, 6);
  });
});
