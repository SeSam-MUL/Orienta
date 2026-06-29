// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, cleanup, fireEvent } from '@testing-library/react';

// Mock the heavy deps so the file loads. We're only testing ScanPicker
// behaviour exported via SinglePixelMode.
vi.mock('../../theme/tokens', () => ({
  colors: {
    bg: '#0b0b0b', bgSecondary: '#000', border: '#444',
    text: '#fff', textSecondary: '#aaa', accent: '#0ff',
    green: '#0f0', red: '#f00', orange: '#fa3', purple: '#a0a',
  },
}));
vi.mock('../../stores/useDataStore', () => ({
  default: (selector) => selector({
    currentIndex: 0,
    gridShape: [10, 12],
    setPosition: vi.fn(),
    ebsdLoaded: false,
    isFileOpen: false,
  }),
}));
vi.mock('../../services/api', () => ({
  ebsdApi: { overview: vi.fn(() => Promise.resolve({ data: { image: '' } })) },
  crystalHintApi: { analyzePixel: vi.fn(), externalSearch: vi.fn() },
}));
vi.mock('./SymmetryReportPanel', () => ({ default: () => null }));
vi.mock('./LatticeReportPanel', () => ({ default: () => null }));
vi.mock('./CandidateList', () => ({ default: () => null }));
vi.mock('./PatternPreview', () => ({ default: () => null }));

import SinglePixelMode from './SinglePixelMode';

afterEach(() => cleanup());

describe('SinglePixelMode + ScanPicker', () => {
  it('renders an Index number input bounded to the scan size', () => {
    const { container } = render(<SinglePixelMode elements={[]} isActive={true} />);
    const idx = container.querySelector('input[type="number"]');
    expect(idx).toBeTruthy();
    // gridShape = [10, 12] → max = 10*12 - 1 = 119
    expect(idx.getAttribute('max')).toBe('119');
  });

  it('renders the ScanPicker SVG with click handler', () => {
    const { container } = render(<SinglePixelMode elements={[]} isActive={true} />);
    // ScanPicker is a relative div with an SVG inside + crosshair
    const svg = container.querySelector('svg');
    expect(svg).toBeTruthy();
    // Should show a crosshair circle
    const circle = svg.querySelector('circle');
    expect(circle).toBeTruthy();
  });

  it('clicking ScanPicker updates the index input', () => {
    const { container } = render(<SinglePixelMode elements={[]} isActive={true} />);
    // Find the picker container (has cursor: crosshair)
    const picker = [...container.querySelectorAll('div')]
      .find(d => d.style.cursor === 'crosshair');
    expect(picker).toBeTruthy();
    // Simulate click at (50%, 50%) — should land near row 5, col 6 of a 10x12 grid
    Object.defineProperty(picker, 'getBoundingClientRect', {
      value: () => ({ left: 0, top: 0, width: 240, height: 200, right: 240, bottom: 200 }),
    });
    fireEvent.click(picker, { clientX: 120, clientY: 100 });
    const idx = container.querySelector('input[type="number"]');
    // (5 rows * 12 cols + 6) = 66 — but exact value depends on flooring
    const value = parseInt(idx.value, 10);
    expect(value).toBeGreaterThan(50);
    expect(value).toBeLessThan(80);
  });

  it('disables Analyse button when file not loaded', () => {
    const { container } = render(<SinglePixelMode elements={[]} isActive={true} />);
    const btn = [...container.querySelectorAll('button')]
      .find(b => b.textContent.includes('Analyse'));
    expect(btn).toBeTruthy();
    expect(btn.disabled).toBe(true);
  });
});
