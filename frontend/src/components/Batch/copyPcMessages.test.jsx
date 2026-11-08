// @vitest-environment jsdom
// What the file tree says after "Apply PC" and "Apply All": only the files that were written
// are counted, and a target that is the source file itself (reported as skipped) is named.
// Real BatchFileTree, real store and real locale texts; the only seam is the API.
import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';

const calls = vi.hoisted(() => ({ results: [] }));

vi.mock('../../services/api', () => ({
  batchV2Api: {
    copyPC: vi.fn(() => Promise.resolve({ data: { results: calls.results } })),
  },
}));

import BatchFileTree from './BatchFileTree';
import useBatchConfigStore from '../../stores/useBatchConfigStore';
import { copyOutcomeText } from './copyPcOutcome';

const ok = (name) => ({ dataset: name, success: true });
const skipped = (name) => ({ dataset: name, success: true, skipped: true });

function setup(checked) {
  useBatchConfigStore.setState({ files: [], activeFileIndex: -1 });
  useBatchConfigStore.getState().addFiles(['C:/d/a.up1', 'C:/d/b.up1', 'C:/d/c.up1']);   // a is active
  render(<BatchFileTree />);
  const boxes = screen.getAllByRole('checkbox');
  checked.forEach((i) => fireEvent.click(boxes[i]));
}

beforeEach(() => { cleanup(); calls.results = []; });

describe('Apply PC -> Selected', () => {
  it('counts the written files and names the skipped one', async () => {
    calls.results = [skipped('a.up1'), ok('b.up1'), ok('c.up1')];
    setup([0, 1, 2]);
    fireEvent.click(screen.getByText('Apply PC → Selected'));
    await waitFor(() => expect(document.body.textContent).toMatch(/PC copied to 2 file\(s\)/));
    expect(document.body.textContent).toMatch(/1 skipped \(source = target\)/);
    expect(document.body.textContent).not.toMatch(/copied to 3/);
  });

  it('with only the source ticked it says nothing was copied, not "copied to 1"', async () => {
    calls.results = [skipped('a.up1')];
    setup([0]);
    fireEvent.click(screen.getByText('Apply PC → Selected'));
    await waitFor(() => expect(document.body.textContent).toMatch(/1 skipped \(source = target\)/));
    expect(document.body.textContent).not.toMatch(/PC copied to/);
  });

  it('without a skipped result the message is as before', async () => {
    calls.results = [ok('b.up1'), ok('c.up1')];
    setup([1, 2]);
    fireEvent.click(screen.getByText('Apply PC → Selected'));
    await waitFor(() => expect(document.body.textContent).toMatch(/PC copied to 2 file\(s\)/));
    expect(document.body.textContent).not.toMatch(/skipped/);
  });
});

describe('Apply All -> Selected', () => {
  it('counts the written files and names the skipped one', async () => {
    calls.results = [skipped('a.up1'), ok('b.up1'), ok('c.up1')];
    setup([0, 1, 2]);
    fireEvent.click(screen.getByText('Apply All → Selected'));
    await waitFor(() => expect(document.body.textContent).toMatch(/Applied to 2 file\(s\)/));
    expect(document.body.textContent).toMatch(/1 skipped \(source = target\)/);
  });

  it('without a skipped result the message is as before', async () => {
    calls.results = [ok('b.up1')];
    setup([1]);
    fireEvent.click(screen.getByText('Apply All → Selected'));
    await waitFor(() => expect(document.body.textContent).toMatch(/Applied to 1 file\(s\)/));
    expect(document.body.textContent).not.toMatch(/skipped/);
  });
});

describe('the skipped text in every language', () => {
  it('is worded by each locale', async () => {
    const { default: i18n } = await import('i18next');
    const out = {};
    for (const lng of ['en', 'de', 'ja', 'zh']) {
      out[lng] = copyOutcomeText([skipped('a')], 'fileTree.msgAppliedTo',
        (k, o) => i18n.t(`batch:${k}`, { ...o, lng }));
    }
    expect(out.de).toBe('1 übersprungen (Quelle = Ziel)');
    expect(new Set(Object.values(out)).size).toBe(4);
    for (const text of Object.values(out)) expect(text).not.toMatch(/msgSkippedSource/);
  });
});
