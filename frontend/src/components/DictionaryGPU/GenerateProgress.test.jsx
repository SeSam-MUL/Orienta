// @vitest-environment jsdom
/**
 * The throughput figure has to stay on screen after the run, not just flash
 * past in the live message — the point is to write it down and compare CPU
 * against GPU.
 */
import { describe, it, expect, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import GenerateProgress from './GenerateProgress';

afterEach(() => cleanup());

const done = (extra = {}) => ({
  status: 'done',
  progress: 1,
  message: 'Saved 6,579 patterns in 31.3 s · 210 pat/s (cuda)',
  result: {
    output_path: 'C:/lib/Ni/x.h5',
    n_orientations: 6579,
    elapsed_s: 31.3,
    patterns_per_second: 210.2,
    device: 'cuda',
    ...extra,
  },
});

describe('throughput readout', () => {
  it('shows patterns/s, elapsed time and the device after a run', () => {
    render(<GenerateProgress progress={done()} />);
    const el = document.querySelector('[data-throughput]');
    expect(el).toBeTruthy();
    expect(el.textContent).toMatch(/210/);
    expect(el.textContent).toMatch(/31\.3/);
    expect(el.textContent).toMatch(/cuda/);
  });

  it('reads cpu runs the same way', () => {
    render(<GenerateProgress progress={done({ patterns_per_second: 18.4, device: 'cpu' })} />);
    const el = document.querySelector('[data-throughput]');
    expect(el.textContent).toMatch(/18/);
    expect(el.textContent).toMatch(/cpu/);
  });

  it('stays quiet for an older run that recorded no rate', () => {
    render(<GenerateProgress progress={done({ patterns_per_second: 0, device: '' })} />);
    expect(document.querySelector('[data-throughput]')).toBeNull();
  });

  it('still shows the live counter while running', () => {
    render(<GenerateProgress progress={{
      status: 'running', progress: 0.4, message: '2560/6579 patterns · 205 pat/s',
    }} />);
    expect(screen.getByText(/2560\/6579 patterns · 205 pat\/s/)).toBeTruthy();
    expect(document.querySelector('[data-throughput]')).toBeNull();
  });
});
