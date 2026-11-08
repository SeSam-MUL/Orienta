// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach } from 'vitest';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

const sphere = vi.fn();
const jobsList = vi.fn();
vi.mock('../../services/api', () => ({
  dbApi: { sphere: (...a) => sphere(...a) },
  jobsApi: { list: (...a) => jobsList(...a) },
}));

// Plotly is heavy + jsdom-hostile and lazy-loaded; stub it with a marker.
vi.mock('react-plotly.js', () => ({
  default: (props) => (
    <div data-testid="sphere-plot" data-traces={(props.data || []).length} />
  ),
}));

import MasterSphereViewer from './MasterSphereViewer';
import en from '../../locales/en/databasebrowser.json';
import de from '../../locales/de/databasebrowser.json';
import ja from '../../locales/ja/databasebrowser.json';
import zh from '../../locales/zh/databasebrowser.json';

const PAYLOAD = {
  filename: 'Al3FeSi2 (..) [AlFe] {20kV}.sht',
  grid: [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
  phi: [0, 2.094, 4.188],
  theta: [0.4, 1.2, 2.6],
  bandwidth: 88,
  file_bandwidth: 384,
  meta: { formula: 'AlFe', space_group: 140, point_group: '4/mmm', voltage_kV: 20.0 },
};

// Every request carries an AbortSignal, so leaving it can cancel it.
const WITH_SIGNAL = expect.objectContaining({ signal: expect.any(AbortSignal) });

beforeEach(() => { jobsList.mockResolvedValue({ data: { jobs: [] } }); });
afterEach(() => { cleanup(); vi.clearAllMocks(); });

describe('MasterSphereViewer', () => {
  it('renders the sphere plot + metadata on success', async () => {
    sphere.mockResolvedValue({ data: PAYLOAD });
    render(<MasterSphereViewer filename={PAYLOAD.filename} isLocal />);
    // lazy Plot resolves after the fetch
    expect(await screen.findByTestId('sphere-plot')).toBeInTheDocument();
    expect(sphere).toHaveBeenCalledWith(PAYLOAD.filename, {}, WITH_SIGNAL);
    // metadata caption
    expect(screen.getByText(/4\/mmm/)).toBeInTheDocument();
    expect(screen.getByText(/SG 140/)).toBeInTheDocument();
    expect(screen.getByText(/bw 88/)).toBeInTheDocument();
  });

  it('shows an error message when the fetch fails', async () => {
    sphere.mockRejectedValue({ response: { data: { detail: 'could not read this .sht' } } });
    render(<MasterSphereViewer filename={PAYLOAD.filename} isLocal />);
    expect(await screen.findByText(/could not read this .sht/i)).toBeInTheDocument();
    expect(screen.queryByTestId('sphere-plot')).not.toBeInTheDocument();
  });

  it('does NOT fetch for a non-local entry', async () => {
    render(<MasterSphereViewer filename={PAYLOAD.filename} isLocal={false} />);
    expect(screen.getByText(/download this file first/i)).toBeInTheDocument();
    await waitFor(() => expect(sphere).not.toHaveBeenCalled());
  });

  it('re-fetches with higher bandwidth when preset changes to Hoch', async () => {
    sphere.mockResolvedValue({ data: {
      grid: [[0.5]], phi: [0], theta: [1.57], bandwidth: 128, file_bandwidth: 384,
      meta: { formula: 'Ni', space_group: 225, point_group: 'm-3m', voltage_kV: 20 },
    }});
    render(<MasterSphereViewer filename="Ni.sht" isLocal />);
    const select = await screen.findByLabelText(/resolution/i);
    fireEvent.change(select, { target: { value: 'high' } });
    await waitFor(() => {
      expect(sphere).toHaveBeenLastCalledWith(
        'Ni.sht', { max_bandwidth: 256, target_size: 192 }, WITH_SIGNAL);
    });
  });
});

describe('MasterSphereViewer error state at High', () => {
  const OK = { data: {
    grid: [[0.5]], phi: [0], theta: [1.57], bandwidth: 128, file_bandwidth: 384,
    meta: { formula: 'Ni', space_group: 225, point_group: 'm-3m', voltage_kV: 20 },
  }};
  // High/Max fail, Standard (no params) works -- the reported situation.
  const failAtHigh = (detail = 'GPU out of memory') => (file, params) => (
    params?.max_bandwidth ? Promise.reject({ response: { data: { detail } } }) : Promise.resolve(OK)
  );

  it('keeps the resolution selector while the error is shown, and Standard recovers', async () => {
    sphere.mockImplementation(failAtHigh());
    render(<MasterSphereViewer filename="Ni.sht" isLocal />);
    fireEvent.change(await screen.findByLabelText(/resolution/i), { target: { value: 'high' } });

    expect(await screen.findByText(/GPU out of memory/)).toBeInTheDocument();
    // The control that leaves the error state must still be there.
    const select = screen.getByLabelText(/resolution/i);
    expect(select).toBeInTheDocument();
    expect(select.value).toBe('high');

    fireEvent.change(select, { target: { value: 'standard' } });
    expect(await screen.findByTestId('sphere-plot')).toBeInTheDocument();
    expect(screen.queryByText(/GPU out of memory/)).not.toBeInTheDocument();
    expect(sphere).toHaveBeenLastCalledWith('Ni.sht', {}, WITH_SIGNAL);
  });

  it('offers "Use Standard" on the error and that button recovers too', async () => {
    sphere.mockImplementation(failAtHigh());
    render(<MasterSphereViewer filename="Ni.sht" isLocal />);
    fireEvent.change(await screen.findByLabelText(/resolution/i), { target: { value: 'max' } });
    fireEvent.click(await screen.findByRole('button', { name: /use standard/i }));
    expect(await screen.findByTestId('sphere-plot')).toBeInTheDocument();
    expect(screen.getByLabelText(/resolution/i).value).toBe('standard');
  });

  it('offers Retry next to "Use Standard" after a High failure, and Retry asks for High again', async () => {
    let highCalls = 0;
    sphere.mockImplementation((file, params) => {
      if (!params?.max_bandwidth) return Promise.resolve(OK);
      highCalls += 1;
      return highCalls === 1
        ? Promise.reject({ response: { data: { detail: 'taken by another program' } } })
        : Promise.resolve(OK);
    });
    render(<MasterSphereViewer filename="Ni.sht" isLocal />);
    fireEvent.change(await screen.findByLabelText(/resolution/i), { target: { value: 'high' } });
    expect(await screen.findByText(/taken by another program/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /use standard/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /retry/i }));
    expect(await screen.findByTestId('sphere-plot')).toBeInTheDocument();
    expect(highCalls).toBe(2);
    expect(sphere).toHaveBeenLastCalledWith('Ni.sht', { max_bandwidth: 256, target_size: 192 }, WITH_SIGNAL);
    expect(screen.getByLabelText(/resolution/i).value).toBe('high');
  });

  it('says the server is busy when the client gave up waiting, not the raw axios text', async () => {
    sphere.mockRejectedValueOnce({ code: 'ECONNABORTED', message: 'timeout of 300000ms exceeded' });
    sphere.mockResolvedValue(OK);
    render(<MasterSphereViewer filename="Ni.sht" isLocal />);
    expect(await screen.findByText(/still busy with other spheres/i)).toBeInTheDocument();
    expect(screen.queryByText(/timeout of 300000ms/)).not.toBeInTheDocument();
    // and Retry is there to ask again
    fireEvent.click(screen.getByRole('button', { name: /retry/i }));
    expect(await screen.findByTestId('sphere-plot')).toBeInTheDocument();
  });

  it('treats the newer axios timeout code the same way', async () => {
    sphere.mockRejectedValue({ code: 'ETIMEDOUT', message: 'timeout of 300000ms exceeded' });
    render(<MasterSphereViewer filename="Ni.sht" isLocal />);
    expect(await screen.findByText(/still busy with other spheres/i)).toBeInTheDocument();
  });

  it('does not offer "Use Standard" when Standard itself failed', async () => {
    sphere.mockRejectedValue({ response: { data: { detail: 'broken file' } } });
    render(<MasterSphereViewer filename="Ni.sht" isLocal />);
    expect(await screen.findByText(/broken file/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /use standard/i })).not.toBeInTheDocument();
    expect(screen.getByLabelText(/resolution/i)).toBeInTheDocument();
  });

  it('shows the translated text for a coded out-of-memory error, not the English prose', async () => {
    sphere.mockImplementation((file, params) => (params?.max_bandwidth
      ? Promise.reject({
          response: {
            headers: { 'x-orienta-code': 'sphereOutOfMemory' },
            data: { detail: 'PROSE-THAT-MUST-NOT-SHOW' },
          },
        })
      : Promise.resolve(OK)));
    render(<MasterSphereViewer filename="Ni.sht" isLocal />);
    fireEvent.change(await screen.findByLabelText(/resolution/i), { target: { value: 'high' } });
    expect(await screen.findByText(/Not enough memory for the sphere at this resolution/)).toBeInTheDocument();
    expect(screen.queryByText(/PROSE-THAT-MUST-NOT-SHOW/)).not.toBeInTheDocument();
  });

  it('starts every new file at Standard, so one failed High does not follow the next .sht', async () => {
    sphere.mockImplementation(failAtHigh());
    const { rerender } = render(<MasterSphereViewer filename="A.sht" isLocal />);
    fireEvent.change(await screen.findByLabelText(/resolution/i), { target: { value: 'high' } });
    expect(await screen.findByText(/GPU out of memory/)).toBeInTheDocument();

    sphere.mockClear();
    rerender(<MasterSphereViewer filename="B.sht" isLocal />);

    expect(await screen.findByTestId('sphere-plot')).toBeInTheDocument();
    expect(screen.getByLabelText(/resolution/i).value).toBe('standard');
    expect(sphere).toHaveBeenCalledTimes(1);
    expect(sphere).toHaveBeenCalledWith('B.sht', {}, WITH_SIGNAL);
  });

  it('offers Retry when Standard itself failed, and Retry fetches again', async () => {
    sphere.mockRejectedValueOnce({ response: { data: { detail: 'transient failure' } } });
    sphere.mockResolvedValue(OK);
    render(<MasterSphereViewer filename="Ni.sht" isLocal />);
    expect(await screen.findByText(/transient failure/)).toBeInTheDocument();
    // Re-selecting "Standard" in the dropdown fires no change event, so the
    // error box has to carry its own way to run the request again.
    fireEvent.click(screen.getByRole('button', { name: /retry/i }));
    expect(await screen.findByTestId('sphere-plot')).toBeInTheDocument();
    expect(sphere).toHaveBeenCalledTimes(2);
  });

  describe('a response that arrives after the user has moved on', () => {
    const deferred = () => {
      let resolve, reject;
      const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
      return { promise, resolve, reject };
    };
    const payloadWithBw = (bw) => ({ data: { ...OK.data, bandwidth: bw } });

    it('a late High result does not replace the Standard sphere', async () => {
      const high = deferred();
      sphere.mockImplementation((file, params) => (params?.max_bandwidth ? high.promise : Promise.resolve(payloadWithBw(128))));
      render(<MasterSphereViewer filename="Ni.sht" isLocal />);
      const select = await screen.findByLabelText(/resolution/i);
      expect(await screen.findByText(/bw 128/)).toBeInTheDocument();

      fireEvent.change(select, { target: { value: 'high' } });     // request stays pending
      fireEvent.change(select, { target: { value: 'standard' } }); // user changes their mind
      expect(await screen.findByText(/bw 128/)).toBeInTheDocument();

      high.resolve(payloadWithBw(256));                            // the old answer lands late
      await new Promise((r) => setTimeout(r, 50));
      expect(screen.getByText(/bw 128/)).toBeInTheDocument();
      expect(screen.queryByText(/bw 256/)).not.toBeInTheDocument();
    });

    it('a late High FAILURE does not put an error over the Standard sphere', async () => {
      const high = deferred();
      sphere.mockImplementation((file, params) => (params?.max_bandwidth ? high.promise : Promise.resolve(payloadWithBw(128))));
      render(<MasterSphereViewer filename="Ni.sht" isLocal />);
      const select = await screen.findByLabelText(/resolution/i);
      await screen.findByText(/bw 128/);

      fireEvent.change(select, { target: { value: 'high' } });
      fireEvent.change(select, { target: { value: 'standard' } });
      await screen.findByText(/bw 128/);

      high.reject({ response: { data: { detail: 'LATE-ERROR-FROM-HIGH' } } });
      await new Promise((r) => setTimeout(r, 50));
      expect(screen.queryByText(/LATE-ERROR-FROM-HIGH/)).not.toBeInTheDocument();
      expect(screen.getByText(/bw 128/)).toBeInTheDocument();
    });
  });
});

describe('MasterSphereViewer cancels the request it leaves', () => {
  const OKP = { data: {
    grid: [[0.5]], phi: [0], theta: [1.57], bandwidth: 128, file_bandwidth: 384,
    meta: { formula: 'Ni', space_group: 225, point_group: 'm-3m', voltage_kV: 20 },
  }};
  const signalOf = (call) => sphere.mock.calls[call][2].signal;

  it('aborts the previous request when the resolution changes, and only that one', async () => {
    sphere.mockImplementation(() => new Promise(() => {}));        // never answers
    render(<MasterSphereViewer filename="Ni.sht" isLocal />);
    await waitFor(() => expect(sphere).toHaveBeenCalledTimes(1));
    expect(signalOf(0).aborted).toBe(false);

    fireEvent.change(await screen.findByLabelText(/resolution/i), { target: { value: 'high' } });
    await waitFor(() => expect(sphere).toHaveBeenCalledTimes(2));
    expect(signalOf(0).aborted).toBe(true);
    expect(signalOf(1).aborted).toBe(false);
  });

  it('aborts the request when another file is selected', async () => {
    sphere.mockImplementation(() => new Promise(() => {}));
    const { rerender } = render(<MasterSphereViewer filename="A.sht" isLocal />);
    await waitFor(() => expect(sphere).toHaveBeenCalledTimes(1));
    rerender(<MasterSphereViewer filename="B.sht" isLocal />);
    await waitFor(() => expect(sphere).toHaveBeenCalledTimes(2));
    expect(signalOf(0).aborted).toBe(true);
    expect(signalOf(1).aborted).toBe(false);
  });

  it('aborts the request when the viewer goes away', async () => {
    sphere.mockImplementation(() => new Promise(() => {}));
    const { unmount } = render(<MasterSphereViewer filename="Ni.sht" isLocal />);
    await waitFor(() => expect(sphere).toHaveBeenCalledTimes(1));
    unmount();
    expect(signalOf(0).aborted).toBe(true);
  });

  it('an aborted request that rejects shows no error', async () => {
    let rejectFirst;
    sphere.mockImplementationOnce(() => new Promise((_res, rej) => { rejectFirst = rej; }));
    sphere.mockResolvedValue(OKP);
    render(<MasterSphereViewer filename="Ni.sht" isLocal />);
    await waitFor(() => expect(sphere).toHaveBeenCalledTimes(1));
    fireEvent.change(await screen.findByLabelText(/resolution/i), { target: { value: 'high' } });
    expect(await screen.findByTestId('sphere-plot')).toBeInTheDocument();
    rejectFirst({ code: 'ERR_CANCELED', name: 'CanceledError', message: 'canceled' });
    await new Promise((r) => setTimeout(r, 50));
    expect(screen.queryByText(/canceled/i)).not.toBeInTheDocument();
    expect(screen.getByTestId('sphere-plot')).toBeInTheDocument();
  });
});

describe('the clock text of the loading box, per language', () => {
  const texts = { en, de, ja, zh };
  it.each(Object.keys(texts))('%s: one string carrying the seconds, composed in that language', (lang) => {
    const s = texts[lang].sphere.reconstructingElapsed;
    expect(s).toContain('{{s}}');
    if (lang === 'ja' || lang === 'zh') {
      // Full-width brackets after full-width text take no ASCII space before them.
      expect(s).toMatch(/…（\{\{s\}\} 秒）$/);
    } else {
      expect(s).toMatch(/… \(\{\{s\}\} s\)$/);
    }
  });
});

describe('MasterSphereViewer cancel and slow start', () => {
  const OKP = { data: {
    grid: [[0.5]], phi: [0], theta: [1.57], bandwidth: 128, file_bandwidth: 384,
    meta: { formula: 'Ni', space_group: 225, point_group: 'm-3m', voltage_kV: 20 },
  }};
  const SLOW_JOBS = { data: { jobs: [{ job: 'sphere', status: 'starting', message_code: 'workerSlowStart' }] } };
  // A request that stays open until the viewer aborts it, like axios does.
  const cancelable = (_f, _p, cfg) => new Promise((_res, rej) => {
    cfg.signal.addEventListener('abort', () => rej(Object.assign(new Error('canceled'), { code: 'ERR_CANCELED' })));
  });
  // Fake timers with React's act around the clock, so the state it sets is rendered.
  const tick = (ms) => act(async () => { await vi.advanceTimersByTimeAsync(ms); });
  const withFakeTimers = async (body) => {
    vi.useFakeTimers();
    try { await body(); } finally { vi.useRealTimers(); }
  };

  it('shows a Cancel button while computing that aborts the request', async () => {
    let seenSignal;
    sphere.mockImplementation((_f, _p, cfg) => {
      seenSignal = cfg.signal;
      return new Promise((_res, rej) => {
        cfg.signal.addEventListener('abort', () => rej(Object.assign(new Error('canceled'), { code: 'ERR_CANCELED' })));
      });
    });
    render(<MasterSphereViewer filename="Ni.sht" isLocal />);
    const btn = await screen.findByRole('button', { name: /cancel/i });
    fireEvent.click(btn);
    expect(seenSignal.aborted).toBe(true);
    expect(await screen.findByText(/^Cancelled\.$/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
  });

  it('says so when the compute process starts slowly', async () => {
    sphere.mockReturnValue(new Promise(() => {}));
    jobsList.mockResolvedValue({ data: { jobs: [{ job: 'sphere', status: 'starting', message_code: 'workerSlowStart' }] } });
    render(<MasterSphereViewer filename="Ni.sht" isLocal />);
    expect(await screen.findByText(/starting slowly/i, {}, { timeout: 4000 })).toBeInTheDocument();
    expect(jobsList).toHaveBeenCalledWith({ job: 'sphere' });
  });

  it('says nothing about a slow start when the job reports no such code', async () => {
    await withFakeTimers(async () => {
      sphere.mockReturnValue(new Promise(() => {}));
      jobsList.mockResolvedValue({ data: { jobs: [{ job: 'sphere', status: 'starting', message_code: null }] } });
      render(<MasterSphereViewer filename="Ni.sht" isLocal />);
      await tick(4100);   // the first answer and two more polls have landed
      expect(jobsList.mock.calls.length).toBeGreaterThanOrEqual(3);
      expect(screen.getByRole('button', { name: /cancel/i })).toBeInTheDocument();
      expect(screen.queryByText(/starting slowly/i)).not.toBeInTheDocument();
    });
  });

  it('asks about the compute process at once, not only after the first 2 s', async () => {
    sphere.mockReturnValue(new Promise(() => {}));
    render(<MasterSphereViewer filename="Ni.sht" isLocal />);
    expect(jobsList).toHaveBeenCalledTimes(1);
  });

  it('takes the slow-start note back when the job no longer reports it', async () => {
    await withFakeTimers(async () => {
      sphere.mockReturnValue(new Promise(() => {}));
      jobsList.mockResolvedValueOnce(SLOW_JOBS);   // then the default: no jobs
      render(<MasterSphereViewer filename="Ni.sht" isLocal />);
      await tick(0);
      expect(screen.getByText(/starting slowly/i)).toBeInTheDocument();
      await tick(2100);
      expect(screen.queryByText(/starting slowly/i)).not.toBeInTheDocument();
    });
  });

  it('stops asking when the viewer goes away', async () => {
    await withFakeTimers(async () => {
      sphere.mockReturnValue(new Promise(() => {}));
      const { unmount } = render(<MasterSphereViewer filename="Ni.sht" isLocal />);
      expect(jobsList).toHaveBeenCalledTimes(1);
      unmount();
      await tick(10000);
      expect(jobsList).toHaveBeenCalledTimes(1);
    });
  });

  it('stops asking, and drops the Cancel button, once the sphere has arrived', async () => {
    await withFakeTimers(async () => {
      sphere.mockResolvedValue(OKP);
      render(<MasterSphereViewer filename="Ni.sht" isLocal />);
      await tick(0);
      expect(screen.queryByRole('button', { name: /cancel/i })).not.toBeInTheDocument();
      const asked = jobsList.mock.calls.length;
      await tick(10000);
      expect(jobsList.mock.calls.length).toBe(asked);
    });
  });

  it('stops the clock and the polling when the file stops being local mid-load', async () => {
    await withFakeTimers(async () => {
      sphere.mockReturnValue(new Promise(() => {}));
      const { rerender } = render(<MasterSphereViewer filename="Ni.sht" isLocal />);
      expect(jobsList).toHaveBeenCalledTimes(1);
      rerender(<MasterSphereViewer filename="Ni.sht" isLocal={false} />);
      expect(screen.getByText(/download this file first/i)).toBeInTheDocument();
      await tick(10000);
      expect(jobsList).toHaveBeenCalledTimes(1);
      // and coming back starts a fresh request with a fresh clock
      rerender(<MasterSphereViewer filename="Ni.sht" isLocal />);
      expect(sphere).toHaveBeenCalledTimes(2);
      expect(screen.getByText(/\(0 s\)/)).toBeInTheDocument();
    });
  });

  it('counts from zero again, and drops the old slow-start note, when another resolution is asked for', async () => {
    await withFakeTimers(async () => {
      sphere.mockReturnValue(new Promise(() => {}));
      // The first answer says "slow"; every later poll stays unanswered, so only
      // the viewer itself can take the note back.
      jobsList.mockResolvedValueOnce(SLOW_JOBS).mockReturnValue(new Promise(() => {}));
      render(<MasterSphereViewer filename="Ni.sht" isLocal />);
      await tick(0);
      expect(screen.getByText(/starting slowly/i)).toBeInTheDocument();

      fireEvent.change(screen.getByLabelText(/resolution/i), { target: { value: 'high' } });
      await tick(40000);
      expect(screen.getByText(/\(40 s\)/)).toBeInTheDocument();

      fireEvent.change(screen.getByLabelText(/resolution/i), { target: { value: 'max' } });
      expect(screen.getByText(/\(0 s\)/)).toBeInTheDocument();
      expect(screen.queryByText(/starting slowly/i)).not.toBeInTheDocument();
    });
  });

  it('Retry after a cancel starts a new request and the "Cancelled." is gone', async () => {
    sphere.mockImplementationOnce(cancelable);
    sphere.mockResolvedValue(OKP);
    render(<MasterSphereViewer filename="Ni.sht" isLocal />);
    fireEvent.click(await screen.findByRole('button', { name: /cancel/i }));
    expect(await screen.findByText(/^Cancelled\.$/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /retry/i }));
    expect(await screen.findByTestId('sphere-plot')).toBeInTheDocument();
    expect(screen.queryByText(/^Cancelled\.$/)).not.toBeInTheDocument();
    expect(sphere).toHaveBeenCalledTimes(2);
  });

  it('another resolution after a cancel starts a new request and the "Cancelled." is gone', async () => {
    sphere.mockImplementationOnce(cancelable);
    sphere.mockResolvedValue(OKP);
    render(<MasterSphereViewer filename="Ni.sht" isLocal />);
    fireEvent.click(await screen.findByRole('button', { name: /cancel/i }));
    expect(await screen.findByText(/^Cancelled\.$/)).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/resolution/i), { target: { value: 'high' } });
    expect(await screen.findByTestId('sphere-plot')).toBeInTheDocument();
    expect(screen.queryByText(/^Cancelled\.$/)).not.toBeInTheDocument();
    expect(sphere).toHaveBeenLastCalledWith('Ni.sht', { max_bandwidth: 256, target_size: 192 }, WITH_SIGNAL);
  });

  it('a real failure after an earlier cancel is shown as a failure, not as "Cancelled."', async () => {
    sphere.mockImplementationOnce(cancelable);
    sphere.mockRejectedValue({ response: { data: { detail: 'the real failure' } } });
    render(<MasterSphereViewer filename="Ni.sht" isLocal />);
    fireEvent.click(await screen.findByRole('button', { name: /cancel/i }));
    expect(await screen.findByText(/^Cancelled\.$/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /retry/i }));
    expect(await screen.findByText(/the real failure/)).toBeInTheDocument();
    expect(screen.queryByText(/^Cancelled\.$/)).not.toBeInTheDocument();
  });

  it('shows the seconds the sphere has been computing', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      sphere.mockReturnValue(new Promise(() => {}));
      render(<MasterSphereViewer filename="Ni.sht" isLocal />);
      expect(await screen.findByText(/\(0 s\)/)).toBeInTheDocument();
      await vi.advanceTimersByTimeAsync(3100);
      expect(await screen.findByText(/\(3 s\)/)).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it('translates a crashed compute process', async () => {
    sphere.mockRejectedValue({
      response: { status: 422, headers: { 'x-orienta-code': 'workerCrashed' }, data: { detail: 'The compute process ended unexpectedly (exit code 3).' } },
    });
    render(<MasterSphereViewer filename="Ni.sht" isLocal />);
    expect(await screen.findByText(/ended unexpectedly\. Orienta and your data/i)).toBeInTheDocument();
  });

  it('translates a backend that is shutting down', async () => {
    sphere.mockRejectedValue({
      response: { status: 503, headers: { 'x-orienta-code': 'workerStopped' }, data: { detail: 'The compute process manager has stopped.' } },
    });
    render(<MasterSphereViewer filename="Ni.sht" isLocal />);
    expect(await screen.findByText(/shutting down, so the sphere was not computed/i)).toBeInTheDocument();
    expect(screen.queryByText(/compute process manager has stopped/i)).not.toBeInTheDocument();
  });
});

