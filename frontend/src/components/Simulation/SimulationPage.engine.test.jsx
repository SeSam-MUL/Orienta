// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { afterEach } from 'vitest';
import { cleanup, render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import SimulationPage from './SimulationPage';

afterEach(() => cleanup());

// ---------------------------------------------------------------------------
// Mock heavy dependencies
// ---------------------------------------------------------------------------

// systemStatus is overridable per-test (mockResolvedValueOnce) so we can flip
// emsoft_available on/off. Default = EMsoft available + a GPU detected.
const SYS_DEFAULT = {
  data: { emsoft_available: true, has_gpu: true, gpu_name: 'RTX 4070', gpu_memory_gb: 12, cpu_count: 24 },
};

// Mock the entire api module — we only care about simApi.start / startGpu and
// the batch endpoints.
vi.mock('../../services/api', () => {
  const batchResp = { data: { batch_id: 'b1', job_count: 1, jobs: [{ task_id: 'bt-1', xtal_path: '/x/Al.xtal', status: 'queued', progress: 0, message: '' }] } };
  const start         = vi.fn(() => Promise.resolve({ data: { task_id: 'task-emsoft-1', status: 'queued' } }));
  const startGpu      = vi.fn(() => Promise.resolve({ data: { task_id: 'task-gpu-1',    status: 'queued' } }));
  const batchStart    = vi.fn(() => Promise.resolve(batchResp));
  const batchStartGpu = vi.fn(() => Promise.resolve(batchResp));
  // Should NOT be called any more (auto-routing removed) — kept so the page's
  // import surface is intact; tests assert it stays untouched.
  const recommendEngine = vi.fn(() => Promise.resolve({ data: { engine: 'gpu', reason: 'n/a' } }));
  const systemStatus = vi.fn(() => Promise.resolve(SYS_DEFAULT));
  // vi.fn so tests can override the missing-list per-call (mockResolvedValueOnce).
  const scanMissing = vi.fn(() => Promise.resolve({ data: { missing: [
    { stem: 'Al',  xtal_path: '/x/Al.xtal',  recommended_dmin: 0.05 },
    { stem: 'Tri', xtal_path: '/x/Tri.xtal', recommended_dmin: 0.07 },
  ] } }));
  return {
    simApi: {
      start,
      startGpu,
      batchStart,
      batchStartGpu,
      recommendEngine,
      scanMissing,
      systemStatus,
      getConfig:             () => Promise.resolve({ data: { config: {} } }),
      history:               () => Promise.resolve({ data: { jobs: [] } }),
      getStatus:             () => Promise.resolve({ data: { status: 'queued' } }),
      stop:                  () => Promise.resolve({ data: {} }),
      batchStatus:           () => Promise.resolve({ data: { jobs: [] } }),
      batchCancel:           () => Promise.resolve({ data: {} }),
      clearHistory:          () => Promise.resolve({ data: {} }),
      getNmlTemplate:        () => Promise.resolve({ data: '' }),
      crystalPicker:         () => Promise.resolve({ data: { phases: [] } }),
      serverConfig:          () => Promise.resolve({ data: {} }),
      updateServerConfig:    () => Promise.resolve({ data: {} }),
      testServerConnection:  () => Promise.resolve({ data: {} }),
    },
  };
});

// Mock CrystalPickerModal to avoid deep deps
vi.mock('./CrystalPickerModal', () => ({ default: () => null }));

// Mock SimulateMissingDialog — render a launch button that fires onLaunch with
// the supplied missing phases so we can drive launchMissingBatch from the test.
vi.mock('./SimulateMissingDialog', () => {
  const React = require('react');
  return {
    default: ({ missing, onLaunch }) =>
      React.createElement('button', {
        'data-testid': 'sim-missing-launch',
        onClick: () => onLaunch((missing || []).map(m => ({
          stem: m.stem, xtal_path: m.xtal_path, dmin: m.recommended_dmin,
        }))),
      }, 'Launch missing'),
  };
});

// Mock theme/components to return trivial HTML
vi.mock('../../theme/components', () => {
  const React = require('react');
  const noop = () => {};
  return {
    colors:    { bg: '#1e1e2e', text: '#cdd6f4', textSecondary: '#a6adc8', accent: '#cba6f7', border: '#313244', cyan: '#89dceb', green: '#a6e3a1', red: '#f38ba8', orange: '#fab387', purple: '#cba6f7', yellow: '#f9e2af', bgSecondary: '#181825' },
    spacing:   { outerMargin: 20, outerSpacing: 16, groupSpacing: 12, innerSpacing: 8, buttonSpacing: 8 },
    alpha:     (c, a) => c + Math.round(a).toString(16).padStart(2, '0'),
    Button:    ({ children, onClick, disabled, title, variant }) =>
      React.createElement('button', { onClick, disabled, title, 'data-variant': variant }, children),
    Input:     (props) => React.createElement('input', props),
    NumberInput: (props) => React.createElement('input', { type: 'number', ...props }),
    Select:    ({ value, onChange, options = [], style, title }) =>
      React.createElement('select', { value, onChange, style, title },
        (options || []).map(o => React.createElement('option', { key: o.value, value: o.value }, o.label))
      ),
    Tabs:      ({ tabs, activeTab, onTabChange }) =>
      React.createElement('div', null,
        (tabs || []).map(t =>
          React.createElement('button', { key: t.id, onClick: () => onTabChange(t.id), 'data-active': activeTab === t.id }, t.label)
        )
      ),
    TabPanel:  ({ children, visible }) => visible ? React.createElement('div', null, children) : null,
    GroupBox:  ({ children, title }) => React.createElement('fieldset', null,
      React.createElement('legend', null, title), children),
    CollapsibleGroup: ({ children, title }) => React.createElement('div', null,
      React.createElement('span', null, title), children),
    FormRow:   ({ children, label }) => React.createElement('div', null,
      React.createElement('label', null, label), children),
    Label:     ({ children }) => React.createElement('span', null, children),
    ProgressBar: () => null,
    Separator: () => React.createElement('hr', null),
    useConfirm: () => [noop, {}],
    usePrompt:  () => [noop, {}],
    ConfirmDialog: () => null,
    PromptDialog: () => null,
  };
});

// Mock useDataStore — must support both hook call (selector) and .getState()
vi.mock('../../stores/useDataStore', () => {
  const state = { detector: null, beamEnergy: null, pendingSimXtal: null, setPendingSimXtal: () => {} };
  const useDataStore = (selector) => selector ? selector(state) : state;
  useDataStore.getState = () => state;
  return { default: useDataStore };
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function renderPage() {
  return render(<SimulationPage isActive={true} />);
}

/** Query the engine toggle group specifically, to avoid collision with the
 *  internal Compute Mode toggle that also has an "⚡ GPU" button. */
function getEngineGroup() {
  return document.querySelector('[aria-label="Simulation engine"]');
}
function getEngineButton(label) {
  const group = getEngineGroup();
  return Array.from(group.querySelectorAll('button')).find(b => b.textContent.includes(label));
}
/** The native-GPU engine toggle button (label is now "✨ GPU (native)"). */
function getGpuButton() {
  const group = getEngineGroup();
  return Array.from(group.querySelectorAll('button')).find(b => /GPU \(native\)/.test(b.textContent));
}

/** Wait until the capability probe has resolved and enabled the EMsoft toggle
 *  (default mock reports emsoft_available=true). */
async function waitEmsoftEnabled() {
  await waitFor(() => expect(getEngineButton('EMsoft')).not.toBeDisabled());
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
describe('SimulationPage two-way engine toggle', () => {

  // Reset systemStatus to the EMsoft-available default before each test so a
  // per-test capability override never leaks into the next test's render.
  beforeEach(async () => {
    const { simApi } = await import('../../services/api');
    simApi.systemStatus.mockReset();
    simApi.systemStatus.mockResolvedValue(SYS_DEFAULT);
  });

  it('renders exactly two engine toggle buttons (GPU (native) + EMsoft, no Auto/GPU)', async () => {
    renderPage();
    const group = getEngineGroup();
    const buttons = Array.from(group.querySelectorAll('button'));
    expect(buttons).toHaveLength(2);
    expect(getGpuButton()).toBeTruthy();
    expect(getEngineButton('EMsoft')).toBeTruthy();
    // No leftover 3-way options.
    expect(getEngineButton('Auto')).toBeUndefined();
  });

  it('defaults to GPU (native) — h1 shows "GPU Simulation"', () => {
    renderPage();
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('GPU Simulation');
  });

  it('GPU (native) toggle is pressed by default (aria-pressed=true)', () => {
    renderPage();
    expect(getGpuButton().getAttribute('aria-pressed')).toBe('true');
    expect(getEngineButton('EMsoft').getAttribute('aria-pressed')).toBe('false');
  });

  it('surfaces the auto-selected hardware path on the GPU (native) button + h1', async () => {
    renderPage();
    // Capability probe resolves with a GPU name.
    await waitFor(() => expect(getGpuButton().textContent).toMatch(/RTX 4070/));
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('RTX 4070');
  });

  it('EMsoft is enabled once the capability check reports emsoft_available', async () => {
    renderPage();
    await waitEmsoftEnabled();
  });

  it('EMsoft toggle is DISABLED when emsoft_available=false (optional, tooltip)', async () => {
    const { simApi } = await import('../../services/api');
    simApi.systemStatus.mockResolvedValue({
      data: { emsoft_available: false, has_gpu: false, cpu_count: 24 },
    });
    renderPage();
    // After the probe resolves, EMsoft stays disabled with the "Optional…" tooltip.
    await waitFor(() => {
      const em = getEngineButton('EMsoft');
      expect(em).toBeDisabled();
      expect(em.getAttribute('title')).toMatch(/Optional/i);
    });
    // GPU (native) is always enabled + remains selected.
    expect(getGpuButton()).not.toBeDisabled();
    expect(getGpuButton().getAttribute('aria-pressed')).toBe('true');
  });

  it('clicking the disabled EMsoft option does NOT switch engine', async () => {
    const { simApi } = await import('../../services/api');
    simApi.systemStatus.mockResolvedValue({
      data: { emsoft_available: false, has_gpu: true, gpu_name: 'RTX 4070', cpu_count: 24 },
    });
    renderPage();
    await waitFor(() => expect(getEngineButton('EMsoft')).toBeDisabled());
    fireEvent.click(getEngineButton('EMsoft'));
    expect(getGpuButton().getAttribute('aria-pressed')).toBe('true');
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('GPU Simulation');
  });

  it('switching to EMsoft changes h1 + aria-pressed', async () => {
    renderPage();
    await waitEmsoftEnabled();
    fireEvent.click(getEngineButton('EMsoft'));
    expect(getEngineButton('EMsoft').getAttribute('aria-pressed')).toBe('true');
    expect(getGpuButton().getAttribute('aria-pressed')).toBe('false');
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('EMsoft Simulation');
  });

  it('switching back to GPU (native) restores aria-pressed', async () => {
    renderPage();
    await waitEmsoftEnabled();
    fireEvent.click(getEngineButton('EMsoft'));
    fireEvent.click(getGpuButton());
    expect(getGpuButton().getAttribute('aria-pressed')).toBe('true');
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('GPU Simulation');
  });

  it('GPU (native) engine keeps batch buttons enabled, but disables Download NML', () => {
    renderPage();
    // default engine=ours
    expect(screen.getByRole('button', { name: /Simulate All Missing/i })).not.toBeDisabled();
    expect(screen.getByRole('button', { name: /Batch Select/i })).not.toBeDisabled();
    // Download NML emits EMsoft .nml files — genuinely EMsoft-only.
    expect(screen.getByRole('button', { name: /Download NML/i })).toBeDisabled();
  });

  it('EMsoft engine keeps batch buttons + Download NML enabled', async () => {
    renderPage();
    await waitEmsoftEnabled();
    fireEvent.click(getEngineButton('EMsoft'));
    expect(screen.getByRole('button', { name: /Simulate All Missing/i })).not.toBeDisabled();
    expect(screen.getByRole('button', { name: /Batch Select/i })).not.toBeDisabled();
    expect(screen.getByRole('button', { name: /Download NML/i })).not.toBeDisabled();
  });

  it('Start Simulation is always enabled (when params valid)', async () => {
    renderPage();
    expect(screen.getByRole('button', { name: /Start Simulation/i })).not.toBeDisabled();
    await waitEmsoftEnabled();
    fireEvent.click(getEngineButton('EMsoft'));
    expect(screen.getByRole('button', { name: /Start Simulation/i })).not.toBeDisabled();
  });

  // -------------------------------------------------------------------------
  // Single dispatch routing
  // -------------------------------------------------------------------------
  it('handleStart calls simApi.startGpu when engine=ours (default)', async () => {
    const { simApi } = await import('../../services/api');
    simApi.start.mockClear();
    simApi.startGpu.mockClear();
    simApi.recommendEngine.mockClear();

    renderPage();
    // default engine=ours — fill crystal file
    const inputs = screen.getAllByRole('textbox');
    if (inputs.length > 0) fireEvent.change(inputs[0], { target: { value: '/path/to/Al.xtal' } });

    fireEvent.click(screen.getByRole('button', { name: /Start Simulation/i }));

    await waitFor(() => {
      expect(simApi.startGpu).toHaveBeenCalledTimes(1);
      expect(simApi.start).not.toHaveBeenCalled();
    });
    // No auto-routing any more.
    expect(simApi.recommendEngine).not.toHaveBeenCalled();
  });

  it('handleStart calls simApi.start (not startGpu) when engine=emsoft', async () => {
    const { simApi } = await import('../../services/api');
    simApi.start.mockClear();
    simApi.startGpu.mockClear();
    simApi.recommendEngine.mockClear();

    renderPage();
    await waitEmsoftEnabled();
    fireEvent.click(getEngineButton('EMsoft'));
    const inputs = screen.getAllByRole('textbox');
    if (inputs.length > 0) fireEvent.change(inputs[0], { target: { value: '/path/to/Al.xtal' } });

    fireEvent.click(screen.getByRole('button', { name: /Start Simulation/i }));

    await waitFor(() => {
      expect(simApi.start).toHaveBeenCalledTimes(1);
      expect(simApi.startGpu).not.toHaveBeenCalled();
    });
    expect(simApi.recommendEngine).not.toHaveBeenCalled();
  });

  // -------------------------------------------------------------------------
  // Batch routing (Simulate All Missing -> launchMissingBatch) — NO partition
  // -------------------------------------------------------------------------
  it('launchMissingBatch routes ALL phases to batchStartGpu when engine=ours (GPU (native))', async () => {
    const { simApi } = await import('../../services/api');
    simApi.batchStart.mockClear();
    simApi.batchStartGpu.mockClear();
    simApi.recommendEngine.mockClear();

    renderPage();
    // default engine=ours
    fireEvent.click(screen.getByRole('button', { name: /Simulate All Missing/i }));
    const launch = await screen.findByTestId('sim-missing-launch');
    fireEvent.click(launch);

    await waitFor(() => {
      expect(simApi.batchStartGpu).toHaveBeenCalledTimes(1);
      expect(simApi.batchStart).not.toHaveBeenCalled();
    });
    // Both phases go to the single GPU endpoint — no per-phase recommendEngine.
    expect(simApi.batchStartGpu.mock.calls[0][0].xtal_paths).toEqual(['/x/Al.xtal', '/x/Tri.xtal']);
    expect(simApi.recommendEngine).not.toHaveBeenCalled();
  });

  it('launchMissingBatch routes ALL phases to batchStart when engine=emsoft', async () => {
    const { simApi } = await import('../../services/api');
    simApi.batchStart.mockClear();
    simApi.batchStartGpu.mockClear();
    simApi.recommendEngine.mockClear();

    renderPage();
    await waitEmsoftEnabled();
    fireEvent.click(getEngineButton('EMsoft'));
    fireEvent.click(screen.getByRole('button', { name: /Simulate All Missing/i }));
    const launch = await screen.findByTestId('sim-missing-launch');
    fireEvent.click(launch);

    await waitFor(() => {
      expect(simApi.batchStart).toHaveBeenCalledTimes(1);
      expect(simApi.batchStartGpu).not.toHaveBeenCalled();
    });
    expect(simApi.batchStart.mock.calls[0][0].xtal_paths).toEqual(['/x/Al.xtal', '/x/Tri.xtal']);
    expect(simApi.recommendEngine).not.toHaveBeenCalled();
  });

  // -------------------------------------------------------------------------
  // Batch routing (Batch Select... -> handleBatchSelect) — NO partition
  // -------------------------------------------------------------------------
  it('handleBatchSelect routes to batchStartGpu when engine=ours (GPU (native))', async () => {
    const { simApi } = await import('../../services/api');
    simApi.batchStart.mockClear();
    simApi.batchStartGpu.mockClear();
    simApi.recommendEngine.mockClear();
    // No electronAPI in jsdom -> handleBatchSelect falls back to window.prompt + window.confirm
    const promptSpy  = vi.spyOn(window, 'prompt').mockReturnValue('/x/Em.xtal;/x/Gp.xtal');
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);

    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /Batch Select/i }));

    await waitFor(() => {
      expect(simApi.batchStartGpu).toHaveBeenCalledTimes(1);
      expect(simApi.batchStart).not.toHaveBeenCalled();
    });
    expect(simApi.batchStartGpu.mock.calls[0][0].xtal_paths).toEqual(['/x/Em.xtal', '/x/Gp.xtal']);
    expect(simApi.recommendEngine).not.toHaveBeenCalled();
    promptSpy.mockRestore();
    confirmSpy.mockRestore();
  });

  it('handleBatchSelect routes to batchStart when engine=emsoft', async () => {
    const { simApi } = await import('../../services/api');
    simApi.batchStart.mockClear();
    simApi.batchStartGpu.mockClear();
    const promptSpy  = vi.spyOn(window, 'prompt').mockReturnValue('/x/Al.xtal');
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);

    renderPage();
    await waitEmsoftEnabled();
    fireEvent.click(getEngineButton('EMsoft'));
    fireEvent.click(screen.getByRole('button', { name: /Batch Select/i }));

    await waitFor(() => {
      expect(simApi.batchStart).toHaveBeenCalledTimes(1);
      expect(simApi.batchStartGpu).not.toHaveBeenCalled();
    });
    promptSpy.mockRestore();
    confirmSpy.mockRestore();
  });

  // -------------------------------------------------------------------------
  // The EMsoft Compute-mode (OpenCL/OpenMP) card is dead UI under GPU (native) — hide
  // it there, keep it under EMsoft.
  // -------------------------------------------------------------------------
  it('hides the EMsoft "Compute" card under engine=ours (GPU (native), default)', () => {
    renderPage();
    expect(screen.queryByText('Compute')).toBeNull();
  });

  it('shows the EMsoft "Compute" card under engine=emsoft and hides it again on GPU (native)', async () => {
    renderPage();
    await waitEmsoftEnabled();
    fireEvent.click(getEngineButton('EMsoft'));
    expect(screen.getByText('Compute')).toBeInTheDocument();
    fireEvent.click(getGpuButton());
    expect(screen.queryByText('Compute')).toBeNull();
  });
});
