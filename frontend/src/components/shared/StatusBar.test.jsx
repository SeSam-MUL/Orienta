// @vitest-environment jsdom
/**
 * User-reported (Bug#1, 2026-08-24): the status bar showed "150\u00d7201"
 * instead of "150×201". A unicode escape written as bare text between JSX
 * tags is not an escape — JSX prints it verbatim.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, cleanup } from '@testing-library/react';

vi.mock('../../theme/tokens', () => ({
  colors: {
    bg: '#000', bgTertiary: '#111', border: '#444', text: '#fff',
    textSecondary: '#aaa', green: '#0f0', red: '#f00', yellow: '#fd0',
    accent: '#0ff', purple: '#a6f',
  },
  layout: { statusBarHeight: 24 },
}));

vi.mock('../../services/api', () => ({
  simApi: { systemStatus: () => Promise.resolve({ data: {} }) },
}));

// StatusBar calls the stores without a selector: useDataStore()
const dataState = {
  isFileOpen: true,
  filePath: 'D:/scans/SampleB.h5oina',
  gridShape: [150, 201],
  patternCount: 30150,
  detector: null,
  stepSize: null,
  edsElements: null,
};
vi.mock('../../stores/useDataStore', () => ({ default: () => dataState }));
vi.mock('../../stores/useProgressStore', () => ({ default: () => ({ tasks: {} }) }));

afterEach(cleanup);

import StatusBar from './StatusBar';

describe('StatusBar grid shape', () => {
  it('renders the multiplication sign, not its escape sequence', () => {
    const { container } = render(<StatusBar />);
    const text = container.textContent;
    expect(text).toContain('150\u00d7201');
    expect(text).not.toContain('u00d7');   // the reported symptom
    expect(text).not.toContain('\\u');
  });
});

describe('StatusBar step size', () => {
  it('shows a step of exactly 1.0 µm instead of hiding it', () => {
    // The old guard was `stepSize.x !== 1.0`, on the assumption that 1.0 meant
    // "unknown". Unknown is `null` — the backend's `_extract_step_size` returns
    // None rather than a placeholder — and 1.0 µm is a perfectly ordinary
    // scan step: both EDAX .osc files in Test_data report exactly that.
    dataState.stepSize = { x: 1.0, y: 1.0, units: 'um' };
    try {
      const { container } = render(<StatusBar />);
      expect(container.textContent).toContain('1.00 um');
    } finally {
      dataState.stepSize = null;
    }
  });

  it('says nothing when the file carries no step size', () => {
    dataState.stepSize = null;
    const { container } = render(<StatusBar />);
    expect(container.textContent).not.toMatch(/µm|\bum\b/);
  });

  it('ignores a nonsensical step rather than printing it', () => {
    dataState.stepSize = { x: 0, y: 0, units: 'um' };
    try {
      const { container } = render(<StatusBar />);
      expect(container.textContent).not.toContain('0.00');
    } finally {
      dataState.stepSize = null;
    }
  });
});
