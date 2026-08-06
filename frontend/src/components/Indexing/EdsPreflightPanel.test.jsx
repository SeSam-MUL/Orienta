// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest';
import { render, cleanup } from '@testing-library/react';
import EdsPreflightPanel from './EdsPreflightPanel';

afterEach(() => cleanup());

// Shape copied from a real POST /api/indexing/eds-preflight response.
const okResult = {
  checks: {
    eds_present: { severity: 'ok', ok: true, detail: '10 elements', elements: ['Al', 'Si'] },
    grid: {
      severity: 'ok', ok: true, eds_shape: [301, 402], ebsd_shape: [301, 402],
      detail: 'EDS and EBSD grids match',
    },
    signal: {
      severity: 'ok', ok: true, median_counts_per_px: 443614.2,
      detail: 'median 443,614 counts/pixel',
    },
    coverage: { severity: 'ok', ok: true, detail: 'all defining elements are measured' },
  },
  phases: [
    {
      path: 'C:/lib/Si.sht', name: 'Si (Si) [cF8] {20kV}', formula: 'Si',
      defining: { Si: 100.0 }, missing_elements: [], max_area_pct: 28.89,
      threshold_at_pct: 2.0, is_upper_bound: true, weak_bound: true,
      severity: 'ok', detail: 'not ruled out on 28.89 % of the map',
    },
    {
      path: 'C:/lib/alpha.sht', name: 'alpha-AlFeMnSi', formula: 'Al75Fe8Mn8Si10',
      defining: { Fe: 8.0, Mn: 8.0, Al: 75.0, Si: 10.0 }, missing_elements: [],
      max_area_pct: 1.54, threshold_at_pct: 2.0, is_upper_bound: true,
      weak_bound: false, severity: 'ok', detail: 'not ruled out on 1.54 % of the map',
    },
  ],
  blocking: [],
  can_index: true,
  eds_usable: true,
};

const blockedResult = {
  checks: {
    eds_present: { severity: 'ok', ok: true, detail: '10 elements', elements: [] },
    grid: {
      severity: 'block', ok: false, eds_shape: [301, 402], ebsd_shape: [150, 200],
      detail: 'EDS grid does not match the EBSD navigation grid',
    },
  },
  phases: [],
  blocking: ['grid'],
  can_index: false,
  eds_usable: false,
};

describe('EdsPreflightPanel', () => {
  it('renders nothing without a result', () => {
    const { container } = render(<EdsPreflightPanel result={null} />);
    expect(container.firstChild).toBeNull();
  });

  it('shows a loading state before the first result', () => {
    const { getByText } = render(<EdsPreflightPanel result={null} loading />);
    expect(getByText(/Checking/i)).toBeTruthy();
  });

  it('degrades to "check unavailable" on a failed fetch', () => {
    const { getByText } = render(<EdsPreflightPanel result={null} error="boom" />);
    expect(getByText(/unavailable/i)).toBeTruthy();
    expect(getByText(/still allowed/i)).toBeTruthy();
  });

  it('renders the four checks with their details', () => {
    const { getByText } = render(<EdsPreflightPanel result={okResult} />);
    expect(getByText('EDS data')).toBeTruthy();
    expect(getByText('Grid alignment')).toBeTruthy();
    expect(getByText('Signal strength')).toBeTruthy();
    expect(getByText('Element coverage')).toBeTruthy();
    expect(getByText(/10 elements/)).toBeTruthy();
    expect(getByText(/443,614 counts\/pixel/)).toBeTruthy();
  });

  it('labels max_area_pct as an UPPER BOUND, never as an estimate', () => {
    const { getByText, getByTitle } = render(<EdsPreflightPanel result={okResult} />);
    expect(getByText('at most 28.89 %')).toBeTruthy();
    expect(getByText('at most 1.54 %')).toBeTruthy();
    expect(getByText(/upper bound on each phase's area/i)).toBeTruthy();
    // The single-element phase's bound is de-emphasised and says why.
    expect(getByTitle(/Single-element phase/i)).toBeTruthy();
    expect(getByText(/Greyed values come from single-element phases/i)).toBeTruthy();
  });

  it('lists the defining elements per phase', () => {
    const { getByText } = render(<EdsPreflightPanel result={okResult} />);
    expect(getByText(/Fe 8, Mn 8, Al 75, Si 10/)).toBeTruthy();
  });

  it('prefers the caller-supplied phase label over the backend file stem', () => {
    const { getByText, queryByText } = render(
      <EdsPreflightPanel result={okResult} phaseLabels={{ 'C:/lib/Si.sht': 'Si (cF8)' }} />
    );
    expect(getByText('Si (cF8)')).toBeTruthy();
    expect(queryByText('Si (Si) [cF8] {20kV}')).toBeNull();
    // Phases without a supplied label fall back to the backend name.
    expect(getByText('alpha-AlFeMnSi')).toBeTruthy();
  });

  it('shows a blocking verdict prominently, with the way out', () => {
    const { getByRole, getByText } = render(<EdsPreflightPanel result={blockedResult} />);
    const alert = getByRole('alert');
    expect(alert.textContent).toMatch(/cannot be used/i);
    expect(alert.textContent).toMatch(/Switch it off/i);
    expect(getByText(/301×402/)).toBeTruthy();
  });

  it('says so when no phases are selected yet', () => {
    const { getByText } = render(
      <EdsPreflightPanel result={{ ...okResult, phases: [] }} />
    );
    expect(getByText(/No phases selected yet/i)).toBeTruthy();
  });

  // Verbatim POST /api/indexing/eds-preflight response, captured 2026-08-05
  // from the running backend on the ProbeB h5oina with two real SHT phases.
  // Contract guard: if the backend's field names drift, this fails.
  it('renders a verbatim live backend response', () => {
    const live = {
      checks: {
        eds_present: {
          severity: 'ok', ok: true, detail: '10 elements',
          elements: ['Al', 'C', 'Cu', 'Fe', 'Mg', 'Mn', 'Ni', 'O', 'Si', 'Zn'],
        },
        grid: {
          severity: 'ok', ok: true, eds_shape: [301, 402], ebsd_shape: [301, 402],
          detail: 'EDS and EBSD grids match',
        },
        signal: {
          severity: 'ok', ok: true, median_counts_per_px: 443614.2,
          detail: 'median 443,614 counts/pixel',
        },
        coverage: { severity: 'ok', ok: true, detail: 'all defining elements are measured' },
      },
      phases: [
        {
          path: 'E:\\Database\\EBSD_SHT_Database\\Al\\Al (Al) [cF4] {20kV}.sht',
          name: 'Al (Al) [cF4] {20kV}', formula: 'Al', defining: { Al: 100.0 },
          missing_elements: [], max_area_pct: 100.0, threshold_at_pct: 2.0,
          is_upper_bound: true, weak_bound: true, severity: 'ok',
          detail: 'not ruled out on 100.00 % of the map (needs Al >= 2 at%) — upper bound on its area',
        },
        {
          path: 'E:\\Database\\EBSD_SHT_Database\\Al\\Al4FeSi (beta-AlFeSi) {20kV}.sht',
          name: 'Al4FeSi (beta-AlFeSi) {20kV}', formula: 'Al4FeSi',
          defining: { Al: 66.7, Fe: 16.7, Si: 16.7 }, missing_elements: [],
          max_area_pct: 2.19, threshold_at_pct: 2.0, is_upper_bound: true,
          weak_bound: false, severity: 'ok',
          detail: 'not ruled out on 2.19 % of the map (needs Al, Fe, Si >= 2 at%) — upper bound on its area',
        },
      ],
      blocking: [],
      can_index: true,
      eds_usable: true,
    };
    const { getByText } = render(<EdsPreflightPanel result={live} />);
    // Al is single-element: 100 % is a real number but a meaningless bound, and
    // it must never read as "Al covers the whole map".
    expect(getByText('at most 100.00 %')).toBeTruthy();
    expect(getByText('at most 2.19 %')).toBeTruthy();
    expect(getByText(/Al 67, Fe 17, Si 17/)).toBeTruthy();
    expect(getByText(/upper bound on each phase's area/i)).toBeTruthy();
  });

  it('reports a phase whose defining elements were not measured', () => {
    const result = {
      ...okResult,
      phases: [{
        path: 'C:/lib/x.sht', name: 'X', formula: 'Cr2Nb', defining: { Cr: 66, Nb: 33 },
        missing_elements: ['Nb'], max_area_pct: null, severity: 'warn',
        detail: 'not measured: Nb — chemistry is blind to this phase',
      }],
    };
    const { getByText } = render(<EdsPreflightPanel result={result} />);
    expect(getByText(/not measured: Nb/)).toBeTruthy();
    expect(getByText('not judgeable')).toBeTruthy();
  });
});
