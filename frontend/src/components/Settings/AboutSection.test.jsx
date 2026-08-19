// @vitest-environment jsdom
import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, cleanup } from '@testing-library/react';

vi.mock('../../theme/components', () => ({
  colors: {
    bg: '#000', bgSecondary: '#111', bgTertiary: '#222', border: '#444',
    text: '#fff', textSecondary: '#aaa', accent: '#0ff',
    green: '#0f0', red: '#f00', yellow: '#fd0', cyan: '#0ff',
  },
  alpha: () => '#000',
  spacing: { sm: 4, md: 8, lg: 16, innerSpacing: 8 },
  GroupBox: ({ title, children }) => <fieldset><legend>{title}</legend>{children}</fieldset>,
  Label: ({ children, ...p }) => <label {...p}>{children}</label>,
}));

afterEach(cleanup);

import AboutSection from './AboutSection';

describe('AboutSection', () => {
  it('shows the GPL appropriate legal notices: copyright, no-warranty, license', () => {
    const { getByText } = render(<AboutSection />);
    expect(getByText(/Copyright © 2026/)).toBeTruthy();
    expect(getByText(/ABSOLUTELY NO WARRANTY/)).toBeTruthy();
    expect(getByText(/GNU General Public License, version 3/)).toBeTruthy();
  });

  it('links to the full license text and the source code', () => {
    const { getByText } = render(<AboutSection />);
    const license = getByText(/Full license text/).closest('a');
    expect(license.getAttribute('href')).toBe('https://www.gnu.org/licenses/gpl-3.0.html');
    const source = getByText(/Source code/).closest('a');
    expect(source.getAttribute('href')).toBe('https://github.com/SeSam-MUL/Orienta');
  });

  it('states local-only processing and the backup advice', () => {
    const { getByText } = render(<AboutSection />);
    expect(getByText(/locally on this machine/)).toBeTruthy();
    expect(getByText(/keep backups of your original data/)).toBeTruthy();
  });
});
