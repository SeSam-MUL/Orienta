/**
 * Phase color override store — lets users pin a custom color to a phase name.
 *
 * Default behaviour (no override) is the deterministic hash in backend
 * phase_map.stable_color_idx / frontend stableColorIdx, so "Al" lands on the
 * same Dracula slot across every result. When a user picks a color for a
 * phase name, that override wins — globally, across all gallery entries —
 * and persists through reloads via zustand/middleware persist.
 *
 * Backend applies overrides when /api/phasemap/render receives a
 * ?color_overrides={"Al":"#ff0000",...} JSON query parameter.
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

const usePhaseColorStore = create(
  persist(
    (set, get) => ({
      overrides: {},  // { [phaseName]: '#rrggbb' }
      setColor: (name, color) =>
        set((s) => ({ overrides: { ...s.overrides, [name]: color } })),
      resetColor: (name) =>
        set((s) => {
          const next = { ...s.overrides };
          delete next[name];
          return { overrides: next };
        }),
      resetAll: () => set({ overrides: {} }),
      getColor: (name) => get().overrides[name] || null,
    }),
    { name: 'phase-color-overrides' }
  )
);

export default usePhaseColorStore;
