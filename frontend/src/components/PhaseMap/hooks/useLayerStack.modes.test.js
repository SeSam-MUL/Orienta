// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

// Layer fetches resolve immediately here — this file is about what the stack
// REMEMBERS across mode switches, not about fetch timing (that is covered by
// useLayerStack.race.test.js).
vi.mock('../../../services/api', () => ({
  phaseMapApi: { layer: vi.fn(() => Promise.resolve({ data: { image: 'Zm9v' } })) },
  ebsdApi: {}, h5Api: {}, analysisApi: {},
}));
vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({ blob: () => Promise.resolve({}) })));
vi.stubGlobal('createImageBitmap', vi.fn(() => Promise.resolve({ width: 4, height: 4, close: () => {} })));

import { useLayerStack } from './useLayerStack';

const CLEANUP = {};
const tick = () => new Promise((r) => setTimeout(r, 0));

function mount(resetSignal = 'r1') {
  return renderHook((props) => useLayerStack({
    cleanupParams: CLEANUP, frameSig: null, colorOverrides: {}, ...props,
  }), { initialProps: { resetSignal } });
}

describe('useLayerStack — quick modes keep what the user built', () => {
  beforeEach(() => { vi.clearAllMocks(); });

  it('a first visit starts the mode as one layer at full opacity', async () => {
    const { result } = mount();
    await act(async () => { await tick(); });
    await act(async () => { result.current.setSingleLayer('ipf-z'); await tick(); });

    expect(result.current.layers.map((l) => l.id)).toEqual(['ipf-z']);
    expect(result.current.layers[0].opacity).toBe(1);
    expect(result.current.activeMode).toBe('ipf-z');
  });

  it('brings the stack back when the user returns to a mode', async () => {
    // This is the reported behaviour: build something on IPF-Z, look at
    // IPF-X, come back — and find the work still there.
    const { result } = mount();
    await act(async () => { await tick(); });
    await act(async () => { result.current.setSingleLayer('ipf-z'); await tick(); });
    await act(async () => { result.current.addLayer('bc'); await tick(); });
    await act(async () => { result.current.setOpacity('bc', 0.42); await tick(); });
    await act(async () => { result.current.setBlend('bc', 'multiply'); await tick(); });

    const built = result.current.layers.map((l) => ({ id: l.id, o: l.opacity, b: l.blend }));
    expect(built).toHaveLength(2);

    await act(async () => { result.current.setSingleLayer('ipf-x'); await tick(); });
    expect(result.current.layers.map((l) => l.id)).toEqual(['ipf-x']);

    await act(async () => { result.current.setSingleLayer('ipf-z'); await tick(); });
    expect(result.current.layers.map((l) => ({ id: l.id, o: l.opacity, b: l.blend }))).toEqual(built);
    expect(result.current.activeMode).toBe('ipf-z');
  });

  it('keeps each mode separate', async () => {
    const { result } = mount();
    await act(async () => { await tick(); });
    await act(async () => { result.current.setSingleLayer('ipf-z'); await tick(); });
    await act(async () => { result.current.addLayer('bc'); await tick(); });
    await act(async () => { result.current.setSingleLayer('ipf-x'); await tick(); });
    await act(async () => { result.current.addLayer('ci'); await tick(); });

    await act(async () => { result.current.setSingleLayer('ipf-z'); await tick(); });
    expect(result.current.layers.map((l) => l.id)).toEqual(['ipf-z', 'bc']);
    await act(async () => { result.current.setSingleLayer('ipf-x'); await tick(); });
    expect(result.current.layers.map((l) => l.id)).toEqual(['ipf-x', 'ci']);
  });

  it('clicking the mode you are already in does not wipe it', async () => {
    const { result } = mount();
    await act(async () => { await tick(); });
    await act(async () => { result.current.setSingleLayer('ipf-z'); await tick(); });
    await act(async () => { result.current.addLayer('bc'); await tick(); });
    await act(async () => { result.current.setSingleLayer('ipf-z'); await tick(); });
    expect(result.current.layers.map((l) => l.id)).toEqual(['ipf-z', 'bc']);
  });

  it('forgets the parked stacks when the result changes', async () => {
    // Those layers were built on that result's data; carrying them to another
    // result would show numbers from the wrong scan.
    const { result, rerender } = mount('r1');
    await act(async () => { await tick(); });
    await act(async () => { result.current.setSingleLayer('ipf-z'); await tick(); });
    await act(async () => { result.current.addLayer('bc'); await tick(); });

    await act(async () => { rerender({ resetSignal: 'r2' }); await tick(); });
    await act(async () => { result.current.setSingleLayer('ipf-z'); await tick(); });
    expect(result.current.layers.map((l) => l.id)).toEqual(['ipf-z']);
  });
});
