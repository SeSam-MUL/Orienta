// @vitest-environment jsdom
import { describe, it, expect, vi } from 'vitest';
import { useEffect } from 'react';
import { render, act } from '@testing-library/react';
import { CursorSyncProvider, useCursorSync, useCursorPublisher } from './CursorSyncContext';

function Consumer({ onPos }) {
  useCursorSync(onPos);
  return null;
}

function Publisher({ pos }) {
  const pub = useCursorPublisher();
  useEffect(() => { pub(pos); }, [pub, pos]);
  return null;
}

// Exposes the publisher to the test so we can drive it imperatively.
function PublisherHandle({ apiRef }) {
  const pub = useCursorPublisher();
  useEffect(() => { apiRef.current = pub; }, [pub, apiRef]);
  return null;
}

describe('CursorSyncContext', () => {
  it('replays last position to a newly mounted subscriber', () => {
    const cb = vi.fn();
    const { rerender } = render(
      <CursorSyncProvider>
        <Consumer onPos={cb} />
        <Publisher pos={{ row: 5, col: 7, hovering: true }} />
      </CursorSyncProvider>
    );
    expect(cb).toHaveBeenCalledWith({ row: 5, col: 7, hovering: true });
    rerender(
      <CursorSyncProvider>
        <Consumer onPos={cb} />
        <Publisher pos={{ row: 9, col: 3, hovering: true }} />
      </CursorSyncProvider>
    );
    expect(cb).toHaveBeenLastCalledWith({ row: 9, col: 3, hovering: true });
  });

  it('a publish after subscribe is delivered synchronously', () => {
    const cb = vi.fn();
    const apiRef = { current: null };
    render(
      <CursorSyncProvider>
        <Consumer onPos={cb} />
        <PublisherHandle apiRef={apiRef} />
      </CursorSyncProvider>
    );
    cb.mockClear();                              // ignore replay-on-mount
    act(() => apiRef.current({ row: 1, col: 2, hovering: true }));
    expect(cb).toHaveBeenCalledTimes(1);
    expect(cb).toHaveBeenCalledWith({ row: 1, col: 2, hovering: true });
  });

  it('an unhandled throw in one subscriber does not break others', () => {
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const goodCb = vi.fn();
    const apiRef = { current: null };
    render(
      <CursorSyncProvider>
        <Consumer onPos={() => { throw new Error('boom'); }} />
        <Consumer onPos={goodCb} />
        <PublisherHandle apiRef={apiRef} />
      </CursorSyncProvider>
    );
    goodCb.mockClear();
    act(() => apiRef.current({ row: 3, col: 4, hovering: true }));
    expect(goodCb).toHaveBeenCalledWith({ row: 3, col: 4, hovering: true });
    errSpy.mockRestore();
  });
});
