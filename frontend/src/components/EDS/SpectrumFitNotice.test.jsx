// @vitest-environment jsdom
/**
 * The spectrum fit's own account, on the EDS page.
 *
 * Two things the fit knows that nothing else on the page can say:
 *  - elements the SPECTRUM shows that AZtec did not export as a map. They are
 *    NOT part of the composition; the notice
 *    names them with family, significance and their at% in the whole map were
 *    they added, with a way to look at the element's own net-count map;
 *  - lines the fit does not reproduce well (a residual above 5 % of the line's
 *    net area). A warning, never a block.
 *
 * The `fit` block is the one `GET /api/eds/spectrum-fit/status` returns
 * (`eds_spectrum_service._summary`).
 */
import React from 'react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';

import en from '../../locales/en/eds.json';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k, o) => (o ? `${k}:${JSON.stringify(o)}` : k) }),
}));

import SpectrumFitNotice, { detectedLabel, describeDetected } from './SpectrumFitNotice';

afterEach(cleanup);

/** What `_summary` serialises for the Oxford reference-scan-like data, plus a detection. */
const FIT = {
  model_version: 'eds-spectrum-fit/1',
  chi2_red: 5.18,
  params: { offset_kev: -0.01, gain: 0.01, fwhm_mnka_kev: 0.128, absorption_a: 1.2 },
  beam_kv: 20,
  lines: {
    Al: { label: 'Al Kα1', family: 'K' },
    Si: { label: 'Si Kα1', family: 'K' },
    Cu: { label: 'Cu Lα1,2', family: 'L' },
  },
  detected_not_exported: [{ element: 'Si', family: 'K', z: 8.2,
    label: 'Si Kα1', at_pct_map_sum: 0.4 }],
  quality_warnings: [],
  exported_not_fitted: [],
};

describe('SpectrumFitNotice', () => {
  it('lists each detected element with family, significance and map-sum at%', () => {
    render(<SpectrumFitNotice fit={FIT} onAddMap={() => {}} />);
    const box = document.querySelector('[data-spectrum-fit-notice]');
    expect(box).toBeTruthy();
    expect(box.textContent).toContain('spectrumFit.detected');
    expect(box.textContent).toContain('"list":"Si K (8σ, 0.40 at%)"');
  });

  it('describes a detection without the optional numbers', () => {
    expect(describeDetected({ element: 'Ti', family: 'K', z: 5.2 })).toBe('Ti K (5σ)');
    expect(describeDetected({ element: 'Ti', family: 'K', z: null,
      at_pct_map_sum: null })).toBe('Ti K');
    expect(describeDetected({ element: 'Ti' })).toBe('Ti');
  });

  it('"Add as map" hands over the element\'s own map label', () => {
    const onAddMap = vi.fn();
    render(<SpectrumFitNotice fit={FIT} onAddMap={onAddMap} />);
    fireEvent.click(screen.getByRole('button', { name: /spectrumFit\.addMap/ }));
    expect(onAddMap).toHaveBeenCalledWith('Si Kα1');
  });

  it('falls back to the label the backend generates when lines lacks the element', () => {
    // `_GENERATED_LABEL` in eds_spectrum_service: K -> "<El> Kα1",
    // L -> "<El> Lα1,2", M -> "<El> Mα".
    expect(detectedLabel({ element: 'Ti', family: 'K' }, {})).toBe('Ti Kα1');
    expect(detectedLabel({ element: 'Zr', family: 'L' }, null)).toBe('Zr Lα1,2');
    expect(detectedLabel({ element: 'W', family: 'M' }, undefined)).toBe('W Mα');
    expect(detectedLabel({ element: 'Si', family: 'K' }, FIT.lines)).toBe('Si Kα1');
  });

  it('prefers the map label the backend sends with the detection', () => {
    expect(detectedLabel({ element: 'Ti', family: 'K', label: 'Ti Kβ1' }, {}))
      .toBe('Ti Kβ1');
  });

  it('one button per detected element, each naming its element', () => {
    const onAddMap = vi.fn();
    const fit = { ...FIT, detected_not_exported: [
      { element: 'Si', family: 'K', z: 8.2 },
      { element: 'Ti', family: 'K', z: 5.04 },
    ] };
    render(<SpectrumFitNotice fit={fit} onAddMap={onAddMap} />);
    const box = document.querySelector('[data-spectrum-fit-notice]');
    expect(box.textContent).toContain('"list":"Si K (8σ), Ti K (5σ)"');
    const buttons = screen.getAllByRole('button', { name: /spectrumFit\.addMap/ });
    expect(buttons).toHaveLength(2);
    fireEvent.click(buttons[1]);
    expect(onAddMap).toHaveBeenCalledWith('Ti Kα1');
  });

  it('shows each quality warning with its residual and line', () => {
    const fit = { ...FIT, detected_not_exported: [],
      quality_warnings: [{ family: 'Fe K', residual_fraction: -0.0724 }] };
    render(<SpectrumFitNotice fit={fit} onAddMap={() => {}} />);
    const warn = document.querySelector('[data-fit-quality]');
    expect(warn).toBeTruthy();
    expect(warn.textContent).toContain('spectrumFit.quality');
    expect(warn.textContent).toContain('"pct":"7.2"');
    expect(warn.textContent).toContain('"line":"Fe K"');
  });

  it('says nothing when the fit has nothing to report', () => {
    const { container } = render(
      <SpectrumFitNotice fit={{ ...FIT, detected_not_exported: [] }} onAddMap={() => {}} />);
    expect(container.innerHTML).toBe('');
    cleanup();
    const r2 = render(<SpectrumFitNotice fit={null} onAddMap={() => {}} />);
    expect(r2.container.innerHTML).toBe('');
  });

  it('no add button without a handler to receive it', () => {
    render(<SpectrumFitNotice fit={FIT} />);
    expect(screen.queryByRole('button')).toBeNull();
  });

  it('the English copy says what the notice must say', () => {
    expect(en.spectrumFit.detected).toBe(
      'Also in the spectrum, not exported by AZtec: {{list}}. Not part of the '
      + 'composition; the at% is its share of the whole map were it added. Add '
      + 'it as a map to see where it is.');
    // never again "included in the composition"
    expect(en.spectrumFit.detected).not.toMatch(/included in the composition/);
    expect(en.spectrumFit.addMap).toBe('Add as map');
    expect(en.spectrumFit.quality).toBe(
      'The fit leaves a residual of {{pct}} % at {{line}}; treat that element with care.');
  });
});
