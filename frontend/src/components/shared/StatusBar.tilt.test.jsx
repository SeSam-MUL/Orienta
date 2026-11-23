// @vitest-environment jsdom
/**
 * The status bar's tilt distortion chip (.h5oina, backend services/tilt_distortion.py).
 */
import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, cleanup, act } from '@testing-library/react';

vi.mock('../../theme/tokens', () => ({
  colors: {
    bg: '#000', bgTertiary: '#111', border: '#444', text: '#fff',
    textSecondary: '#aaa', green: '#0f0', red: '#f00', yellow: '#fd0',
    accent: '#0ff', purple: '#a6f',
  },
  layout: { statusBarHeight: 24 },
}));
const tiltCheckApi = vi.fn(() => Promise.resolve({ data: { tilt_check: null } }));
vi.mock('../../services/api', () => ({
  simApi: { systemStatus: () => Promise.resolve({ data: {} }) },
  ebsdApi: { tiltCheck: () => tiltCheckApi() },
}));

const AB6 = {
  datei: 'AB6.h5oina', befund: 'X gestaucht', faktor_korrektur: 2.92,
  kippung_header_grad: 70, kippung_wirksam_grad: 70,
  bilder: [{ name: 'SE/Elektronenbild 33', kanal: 'SE', lx_ly_median: 0.34, zielgroesse_px: [2994, 768] }],
  karten: [{ typ: 'EBSD', step_nominal_um: 0.658, erbt_bildverzerrung: true, wahrer_schritt_um: [1.9239, 0.658] }],
};
const setTiltCheck = vi.fn();
let dataState;
vi.mock('../../stores/useDataStore', () => ({ default: () => dataState }));
vi.mock('../../stores/useProgressStore', () => ({ default: () => ({ tasks: {} }) }));

import i18n from '../../i18n';
import StatusBar from './StatusBar';
import { tiltBadge } from './tiltBadge';

const base = {
  isFileOpen: true, filePath: '/data/AB6.h5oina', gridShape: [9, 12], patternCount: 108,
  detector: null, stepSize: null, edsElements: null, scanGeometry: null, hexResampled: null,
  setTiltCheck,
};

afterEach(async () => {
  cleanup();
  tiltCheckApi.mockClear();
  setTiltCheck.mockClear();
  await act(async () => { await i18n.changeLanguage('en'); });
});

describe('tiltBadge', () => {
  it('says nothing for no check, an undistorted file or an untilted unclear one', () => {
    expect(tiltBadge(null)).toBeNull();
    expect(tiltBadge({ befund: 'unverzerrt' })).toBeNull();
    expect(tiltBadge({ befund: 'unklar', kippung_header_grad: 0 })).toBeNull();
  });
  it('names direction, factor and the true map step of a foreshortened file', () => {
    const b = tiltBadge(AB6);
    expect(b.key).toBe('status.tiltDistorted');
    expect(b.params).toEqual({ axis: 'X', factor: '2.92' });
    expect(b.titleKey).toBe('status.tiltDistortedMapTitle');
    expect(b.titleParams.trueStep).toBe('1.924');
    expect(b.titleParams.target).toBe('2994 × 768');
  });
  it('flags a tilted file the image does not confirm', () => {
    expect(tiltBadge({ befund: 'unklar', kippung_header_grad: 70, bilder: [] }).key).toBe('status.tiltUnclear');
  });
});

describe('the status bar chip', () => {
  it('shows the finding of the load answer', async () => {
    dataState = { ...base, tiltCheck: AB6 };
    await act(async () => { await i18n.changeLanguage('de'); });
    const { getByTestId } = render(<StatusBar backendStatus="disconnected" />);
    const chip = getByTestId('tilt-distortion-badge');
    expect(chip.textContent).toBe('Kippverzerrung: X gestaucht ×2,92');
    expect(chip.getAttribute('title')).toContain('1,924 µm statt 0,658 µm');
    expect(tiltCheckApi).not.toHaveBeenCalled();
  });

  it('asks the backend once when an .h5oina came without a check', async () => {
    tiltCheckApi.mockImplementationOnce(() => Promise.resolve({ data: { tilt_check: AB6 } }));
    dataState = { ...base, tiltCheck: null };
    await act(async () => { render(<StatusBar backendStatus="disconnected" />); });
    expect(tiltCheckApi).toHaveBeenCalledTimes(1);
    expect(setTiltCheck).toHaveBeenCalledWith(AB6);
  });

  it('does not ask for other formats', async () => {
    dataState = { ...base, filePath: '/data/scan.up1', tiltCheck: null };
    await act(async () => { render(<StatusBar backendStatus="disconnected" />); });
    expect(tiltCheckApi).not.toHaveBeenCalled();
  });
});
