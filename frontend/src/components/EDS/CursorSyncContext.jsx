import { createContext, useContext, useEffect, useRef, useCallback } from 'react';

const CursorSyncContext = createContext(null);

export function CursorSyncProvider({ children }) {
  // Subscribers + last position live in refs so they don't trigger React
  // re-renders. The whole point: hover events at 60 Hz must not thrash
  // the tree.
  const stateRef = useRef({ row: 0, col: 0, hovering: false, screenX: 0, screenY: 0 });
  const subsRef = useRef(new Set());
  const publish = useCallback((pos) => {
    stateRef.current = pos;
    for (const cb of Array.from(subsRef.current)) {
      try { cb(pos); } catch (e) { console.error('[CursorSync] subscriber threw', e); }
    }
  }, []);
  return (
    <CursorSyncContext.Provider value={{ stateRef, subsRef, publish }}>
      {children}
    </CursorSyncContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export function useCursorSync(cb) {
  const ctx = useContext(CursorSyncContext);
  if (!ctx) throw new Error('useCursorSync must be used inside CursorSyncProvider');
  const cbRef = useRef(cb);
  useEffect(() => { cbRef.current = cb; });
  useEffect(() => {
    const stable = (pos) => {
      try { cbRef.current(pos); } catch (e) { console.error('[CursorSync] subscriber threw', e); }
    };
    ctx.subsRef.current.add(stable);
    stable(ctx.stateRef.current);                // replay last known on subscribe
    return () => { ctx.subsRef.current.delete(stable); };
  }, [ctx]);
}

// eslint-disable-next-line react-refresh/only-export-components
export function useCursorPublisher() {
  const ctx = useContext(CursorSyncContext);
  if (!ctx) throw new Error('useCursorPublisher must be used inside CursorSyncProvider');
  return ctx.publish;
}
