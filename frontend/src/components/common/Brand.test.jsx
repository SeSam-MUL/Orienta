// @vitest-environment jsdom
/**
 * Brand marks — the in-app SVG must stay identical to the shipped design files.
 *
 * Brand.jsx mirrors the geometry of branding/*.svg by hand (it has to, because
 * the ink colour follows the theme). These tests read the real design files and
 * fail the moment the two drift — the failure mode this guards against is the
 * triangle sliding off the i, which is exactly what happened once already when
 * the coordinates were copied over from another wordmark.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import ThemeProvider from '../../theme/ThemeProvider';
import {
  Wordmark, BrandMark,
  WORDMARK_TRIANGLE, WORDMARK_GRAD_BOX, WORDMARK_VIEWBOX, ICON_TRIANGLE, ICON_VIEWBOX,
} from './Brand';

const HERE = dirname(fileURLToPath(import.meta.url));
const brandingFile = (name) => readFileSync(resolve(HERE, '../../../../branding', name), 'utf8');

/** Collapse whitespace so "M84.5 12 100.5 40" compares equal regardless of spacing. */
const norm = (d) => String(d).replace(/\s+/g, ' ').trim();

const renderThemed = (ui, theme = 'dracula') => {
  localStorage.setItem('kikuchipy-theme', theme);
  return render(<ThemeProvider>{ui}</ThemeProvider>);
};

beforeEach(() => localStorage.clear());
afterEach(cleanup);   // renders are not auto-torn-down here; screen queries would see stale marks

describe('geometry parity with branding/', () => {
  it('wordmark triangle matches branding/wordmark-dark.svg', () => {
    const svg = brandingFile('wordmark-dark.svg');
    const d = /<clipPath[^>]*><path d="([^"]+)"/.exec(svg)?.[1];
    expect(d, 'clipPath not found in wordmark-dark.svg').toBeTruthy();
    expect(norm(d)).toBe(norm(WORDMARK_TRIANGLE));
  });

  it('wordmark triangle matches branding/wordmark-light.svg too', () => {
    const d = /<clipPath[^>]*><path d="([^"]+)"/.exec(brandingFile('wordmark-light.svg'))?.[1];
    expect(norm(d)).toBe(norm(WORDMARK_TRIANGLE));
  });

  it('wordmark gradient box and viewBox match the design file', () => {
    const svg = brandingFile('wordmark-dark.svg');
    const rect = /<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"/.exec(svg);
    expect(rect, 'gradient rect not found').toBeTruthy();
    const [, x, y, width, height] = rect.map(Number);
    expect({ x, y, width, height }).toEqual(WORDMARK_GRAD_BOX);
    expect(/viewBox="([^"]+)"/.exec(svg)[1]).toBe(WORDMARK_VIEWBOX);
  });

  it('icon triangle matches branding/icon.svg', () => {
    const svg = brandingFile('icon.svg');
    const d = /<clipPath[^>]*><path d="([^"]+)"/.exec(svg)?.[1];
    expect(norm(d)).toBe(norm(ICON_TRIANGLE));
    expect(/viewBox="([^"]+)"/.exec(svg)[1]).toBe(ICON_VIEWBOX);
  });

  it('the standalone mark uses the icon triangle as well', () => {
    const d = /<clipPath[^>]*><path d="([^"]+)"/.exec(brandingFile('mark.svg'))?.[1];
    expect(norm(d)).toBe(norm(ICON_TRIANGLE));
  });

  it('the app favicon uses the icon triangle as well', () => {
    const svg = readFileSync(resolve(HERE, '../../../public/favicon.svg'), 'utf8');
    const d = /<clipPath[^>]*><path d="([^"]+)"/.exec(svg)?.[1];
    expect(norm(d)).toBe(norm(ICON_TRIANGLE));
  });
});

describe('Wordmark', () => {
  it('renders the dotless i so the triangle is the only dot', () => {
    const { container } = renderThemed(<Wordmark />);
    const text = container.querySelector('text');
    expect(text.textContent).toBe('orıenta');       // U+0131, not a plain "i"
    expect(text.textContent).not.toContain('i');
  });

  it('keeps the height/width aspect of the design file', () => {
    const { container } = renderThemed(<Wordmark height={86} />);
    const svg = container.querySelector('svg');
    expect(svg.getAttribute('height')).toBe('86');
    expect(svg.getAttribute('width')).toBe('228');
  });

  it('is labelled for screen readers', () => {
    renderThemed(<Wordmark />);
    expect(screen.getByRole('img', { name: 'Orienta' })).toBeTruthy();
  });

  it('screens the IPF hues on a dark theme and multiplies them on a light one', () => {
    const dark = renderThemed(<Wordmark />, 'dracula');
    expect(dark.container.innerHTML).toContain('screen');
    expect(dark.container.innerHTML).not.toContain('multiply');
    dark.unmount();

    const light = renderThemed(<Wordmark />, 'light');
    expect(light.container.innerHTML).toContain('multiply');
    expect(light.container.innerHTML).not.toContain('screen');
  });

  it('honours an explicit variant over the theme', () => {
    const { container } = renderThemed(<Wordmark variant="light" />, 'dracula');
    expect(container.innerHTML).toContain('multiply');
  });

  it('gives each instance its own gradient ids', () => {
    const { container } = renderThemed(<><Wordmark /><Wordmark /></>);
    const ids = [...container.querySelectorAll('clipPath')].map((n) => n.id);
    expect(ids).toHaveLength(2);
    expect(new Set(ids).size).toBe(2);
  });
});

describe('BrandMark', () => {
  it('draws the triangle without a plate by default', () => {
    const { container } = renderThemed(<BrandMark />);
    expect(container.querySelector('rect[rx="27"]')).toBeNull();
    expect(container.querySelector('clipPath path').getAttribute('d')).toBe(ICON_TRIANGLE);
  });

  it('adds the rounded square when asked', () => {
    const { container } = renderThemed(<BrandMark plate />);
    expect(container.querySelector('rect[rx="27"]')).toBeTruthy();
  });
});
