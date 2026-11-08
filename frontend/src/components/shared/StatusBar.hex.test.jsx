// @vitest-environment jsdom
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
vi.mock('../../services/api', () => ({
  simApi: { systemStatus: () => Promise.resolve({ data: {} }) },
}));

const dataState = {
  isFileOpen: true, filePath: 'scan.up1', gridShape: [6, 5], patternCount: 30,
  detector: null, stepSize: null, edsElements: null, scanGeometry: null, hexResampled: null,
};
vi.mock('../../stores/useDataStore', () => ({ default: () => dataState }));
vi.mock('../../stores/useProgressStore', () => ({ default: () => ({ tasks: {} }) }));

import i18n from '../../i18n';
import enShell from '../../locales/en/shell.json';
import deShell from '../../locales/de/shell.json';
import jaShell from '../../locales/ja/shell.json';
import zhShell from '../../locales/zh/shell.json';
import StatusBar from './StatusBar';

afterEach(async () => {
  cleanup();
  await act(async () => { await i18n.changeLanguage('en'); });
});

const LANGS = ['en', 'de', 'ja', 'zh'];
const shellLocale = (lang) => ({ en: enShell, de: deShell, ja: jaShell, zh: zhShell })[lang];

// What the chip's explanation says in each language, written by hand from its meaning: the maps
// are drawn hexagon-true; the Analysis page is the one place that cannot show the scan yet.
const SAYS = {
  en: ['every second row is offset by half a step', 'as they lie on the sample', 'Analysis page'],
  de: ['jede zweite Zeile ist um einen halben Schritt versetzt', 'so gezeichnet, wie sie auf der Probe liegen', 'Analyse-Seite'],
  ja: ['1行おきに半ステップずれています', '試料上に並ぶとおりに描画されます', '解析ページ'],
  zh: ['每隔一行偏移半个步长', '在样品上的实际排布绘制', '分析页面'],
};
// sentences that would be false now: maps of a hexagonal scan are no longer drawn on the index grid
const STALE = [
  /index grid/i, /hexagon-true/i, /Indexraster/, /nicht hexagonal/,
  /インデックス格子/, /六方配置ではなく/, /索引网格/, /尚未按六方/,
];

describe('the hex chip explanation is true in every language', () => {
  it.each(LANGS)('%s has status.hexGridTitle and not the sentence that went stale', (lang) => {
    const text = shellLocale(lang).status.hexGridTitle;
    expect(typeof text).toBe('string');
    expect(text.length).toBeGreaterThan(20);
    for (const stale of STALE) expect(text).not.toMatch(stale);
  });

  it.each(LANGS)('%s shows it as the title of the chip, through the real i18n', async (lang) => {
    await act(async () => { await i18n.changeLanguage(lang); });
    dataState.scanGeometry = { kind: 'hex', gridShape: [6, 5] };
    const { container } = render(<StatusBar />);
    const chip = [...container.querySelectorAll('[title]')]
      .find((el) => el.textContent === shellLocale(lang).status.hexGrid);
    expect(chip, 'the chip').toBeDefined();
    const title = chip.getAttribute('title');
    for (const part of SAYS[lang]) expect(title).toContain(part);
    expect(title).not.toMatch(/hexGridTitle|status\./);   // a missing key shows its name
  });
});

describe('StatusBar says when the scan is hexagonal', () => {
  it('shows a chip, with its explanation, for a hexagonal scan', () => {
    dataState.scanGeometry = { kind: 'hex', gridShape: [6, 5] };
    const { container } = render(<StatusBar />);
    expect(container.textContent).toContain('hex grid');
    const chip = [...container.querySelectorAll('[title]')]
      .find((el) => el.textContent === 'hex grid');
    expect(chip).toBeDefined();
    expect(chip.getAttribute('title')).toMatch(/every second row is offset by half a step/);
  });

  it('shows nothing extra for a square scan or an unknown one', () => {
    for (const g of [{ kind: 'square', gridShape: [6, 5] }, null]) {
      dataState.scanGeometry = g;
      const { container, unmount } = render(<StatusBar />);
      expect(container.textContent).not.toContain('hex grid');
      unmount();
    }
  });
});

describe('StatusBar says when a square map was resampled from a hexagonal scan', () => {
  const RESAMPLED = 'resampled from a hexagonal scan';

  it('shows a chip, with its explanation, only for hexResampled === true', () => {
    dataState.scanGeometry = { kind: 'square', gridShape: [6, 5] };
    dataState.hexResampled = true;
    const { container } = render(<StatusBar />);
    expect(container.textContent).toContain(RESAMPLED);
    const chip = [...container.querySelectorAll('[title]')].find((el) => el.textContent === RESAMPLED);
    expect(chip).toBeDefined();
    expect(chip.getAttribute('title')).toMatch(/map is square/);
    expect(chip.getAttribute('title')).toMatch(/hexagonal/);
    // it is a square map: the "hex grid" chip is not shown by it
    expect(container.textContent).not.toContain('hex grid');
  });

  it('shows nothing for false, for "the backend did not say" (null) or for undefined', () => {
    dataState.scanGeometry = { kind: 'square', gridShape: [6, 5] };
    for (const flag of [false, null, undefined]) {
      dataState.hexResampled = flag;
      const { container, unmount } = render(<StatusBar />);
      expect(container.textContent).not.toContain(RESAMPLED);
      unmount();
    }
  });
});
