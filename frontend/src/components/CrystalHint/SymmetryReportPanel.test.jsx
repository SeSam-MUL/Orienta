// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, cleanup } from '@testing-library/react';
import SymmetryReportPanel from './SymmetryReportPanel';

afterEach(() => cleanup());

vi.mock('../../theme/tokens', () => ({
  colors: {
    bg: '#0b0b0b', bgSecondary: '#000', border: '#444',
    text: '#fff', textSecondary: '#aaa', accent: '#0ff',
    green: '#0f0', red: '#f00', orange: '#fa3', purple: '#a0a',
  },
}));

describe('SymmetryReportPanel', () => {
  it('renders null when symmetry is null', () => {
    const { container } = render(<SymmetryReportPanel symmetry={null} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders detected n-fold prominently', () => {
    const symmetry = {
      detected_n_fold: 4,
      n_fold_scores: { 2: 0.5, 3: 0.4, 4: 0.92, 6: 0.3 },
      confidence: 'high',
      compatible_systems: ['cubic', 'tetragonal'],
      warnings: [],
      zone_axis_yx: [58, 58],
    };
    const { container } = render(<SymmetryReportPanel symmetry={symmetry} />);
    expect(container.textContent).toMatch(/4-fold/);
    expect(container.textContent).toMatch(/high/);
    expect(container.textContent).toMatch(/cubic/);
    expect(container.textContent).toMatch(/tetragonal/);
  });

  it('renders all four n-fold score boxes', () => {
    const symmetry = {
      detected_n_fold: 3,
      n_fold_scores: { 2: 0.7, 3: 0.85, 4: 0.4, 6: 0.5 },
      confidence: 'medium',
      compatible_systems: ['cubic', 'hexagonal'],
      warnings: [],
      zone_axis_yx: [50, 50],
    };
    const { container } = render(<SymmetryReportPanel symmetry={symmetry} />);
    // Each NCC value should appear (3 decimals)
    expect(container.textContent).toMatch(/0\.700/);
    expect(container.textContent).toMatch(/0\.850/);
    expect(container.textContent).toMatch(/0\.400/);
    expect(container.textContent).toMatch(/0\.500/);
  });

  it('shows "—" placeholder when n-fold not detected', () => {
    const symmetry = {
      detected_n_fold: null,
      n_fold_scores: { 2: 0.3, 3: 0.2, 4: 0.25, 6: 0.2 },
      confidence: 'none',
      compatible_systems: [],
      warnings: ['too noisy'],
      zone_axis_yx: null,
    };
    const { container } = render(<SymmetryReportPanel symmetry={symmetry} />);
    expect(container.textContent).toMatch(/—/);
    expect(container.textContent).toMatch(/No compatible crystal system/i);
    expect(container.textContent).toMatch(/too noisy/);
  });

  it('shows the symmetry method tag when method is set', () => {
    const symmetry = {
      detected_n_fold: 4,
      n_fold_scores: { 2: 0.5, 3: 0.4, 4: 0.92, 6: 0.3 },
      confidence: 'high',
      compatible_systems: ['cubic'],
      warnings: [],
      zone_axis_yx: null,
      method: 'spherical',
      n_patterns_averaged: 1,
    };
    const { container } = render(<SymmetryReportPanel symmetry={symmetry} />);
    expect(container.textContent).toMatch(/Method B/);
    expect(container.textContent).toMatch(/spherical/i);
  });

  it('shows averaging-count when n_patterns_averaged > 1', () => {
    const symmetry = {
      detected_n_fold: 4,
      n_fold_scores: { 2: 0.5, 3: 0.4, 4: 0.92, 6: 0.3 },
      confidence: 'high',
      compatible_systems: ['cubic'],
      warnings: [],
      zone_axis_yx: null,
      method: 'spherical_avg9',
      n_patterns_averaged: 9,
    };
    const { container } = render(<SymmetryReportPanel symmetry={symmetry} />);
    expect(container.textContent).toMatch(/avg 9 patterns/);
    expect(container.textContent).toMatch(/n=9/);
  });

  it('falls back gracefully when method is missing', () => {
    const symmetry = {
      detected_n_fold: 4,
      n_fold_scores: { 4: 0.85 },
      confidence: 'high',
      compatible_systems: ['cubic'],
      warnings: [],
      zone_axis_yx: null,
    };
    const { container } = render(<SymmetryReportPanel symmetry={symmetry} />);
    // No crash, method tag absent
    expect(container.textContent).toMatch(/4-fold/);
    expect(container.textContent).not.toMatch(/Method/);
  });

  it('shows "consistent" badge when indexed phase matches detected symmetry', () => {
    const symmetry = {
      detected_n_fold: 4,
      n_fold_scores: { 4: 0.92 },
      confidence: 'high',
      compatible_systems: ['cubic'],
      warnings: [],
      zone_axis_yx: null,
    };
    const indexedPhase = {
      phase_id: 0,
      phase_name: 'Al',
      crystal_system: 'cubic',
      space_group: 'Fm-3m',
      is_unindexed: false,
      consistent_with_symmetry: true,
    };
    const { container } = render(<SymmetryReportPanel symmetry={symmetry} indexedPhase={indexedPhase} />);
    expect(container.textContent).toMatch(/Indexed phase/);
    expect(container.textContent).toMatch(/Al/);
    expect(container.textContent).toMatch(/Fm-3m/);
    expect(container.textContent).toMatch(/✓ consistent/);
  });

  it('shows "mismatch" badge when indexed phase disagrees with detected symmetry', () => {
    const symmetry = {
      detected_n_fold: 6,
      n_fold_scores: { 6: 0.85 },
      confidence: 'high',
      compatible_systems: ['hexagonal'],
      warnings: [],
      zone_axis_yx: null,
    };
    const indexedPhase = {
      phase_id: 2,
      phase_name: 'Al2Cu',
      crystal_system: 'tetragonal',
      space_group: 'I4/mcm',
      is_unindexed: false,
      consistent_with_symmetry: false,
    };
    const { container } = render(<SymmetryReportPanel symmetry={symmetry} indexedPhase={indexedPhase} />);
    expect(container.textContent).toMatch(/Indexed phase/);
    expect(container.textContent).toMatch(/Al2Cu/);
    expect(container.textContent).toMatch(/⚠ symmetry mismatch/);
  });

  it('shows "unindexed" placeholder when phase_id is -1', () => {
    const symmetry = {
      detected_n_fold: 4,
      n_fold_scores: { 4: 0.5 },
      confidence: 'medium',
      compatible_systems: ['cubic'],
      warnings: [],
      zone_axis_yx: null,
    };
    const indexedPhase = {
      phase_id: -1,
      phase_name: '',
      crystal_system: 'unknown',
      space_group: '',
      is_unindexed: true,
      consistent_with_symmetry: null,
    };
    const { container } = render(<SymmetryReportPanel symmetry={symmetry} indexedPhase={indexedPhase} />);
    expect(container.textContent).toMatch(/unindexed/i);
  });

  it('renders without indexedPhase prop (no indexing result)', () => {
    const symmetry = {
      detected_n_fold: 4,
      n_fold_scores: { 4: 0.5 },
      confidence: 'medium',
      compatible_systems: ['cubic'],
      warnings: [],
      zone_axis_yx: null,
    };
    const { container } = render(<SymmetryReportPanel symmetry={symmetry} />);
    expect(container.textContent).not.toMatch(/Indexed phase/);
  });

  it('color-codes the confidence label', () => {
    const high = {
      detected_n_fold: 4, n_fold_scores: { 2: 0, 3: 0, 4: 0.95, 6: 0 },
      confidence: 'high', compatible_systems: ['cubic'], warnings: [],
      zone_axis_yx: [50, 50],
    };
    const { container } = render(<SymmetryReportPanel symmetry={high} />);
    const confSpan = [...container.querySelectorAll('span')]
      .find(s => s.textContent === 'high');
    expect(confSpan).toBeTruthy();
    // green color from the mocked tokens
    expect(confSpan.getAttribute('style')).toMatch(/#0f0|rgb\(0, 255, 0\)/);
  });
});
