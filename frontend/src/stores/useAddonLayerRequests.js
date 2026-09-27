/**
 * One-shot requests to show an add-on's map as a layer on the Phase Maps page.
 *
 * A QUEUE, consumed once — not a second copy of the layer stack. The stack is
 * local state inside PhaseMapPage, and mirroring it here is the defect shape
 * this branch has already shipped four times: state held twice with one copy
 * updated. So this holds only what has not been handed over yet, and the page
 * empties it.
 *
 * Each request carries its LABEL as well as its id, because the id cannot
 * produce one: the page builds a layer's label from the id string alone, and
 * the best that can yield is `component_map`. The author's declared label
 * exists only in the run response on the other page — and it is what the
 * value-scale legend prints beside a figure bound for a paper.
 */
import { create } from 'zustand';

const useAddonLayerRequests = create((set, get) => ({
  //: [{id, label}] — pending, in the order they were asked for.
  requests: [],

  request: (id, label) => set((s) => (
    // Asking twice for the same layer before the page has drained is one
    // request, not two: the second would be deduplicated by the stack anyway,
    // and a doubled entry would make the drain report a refusal it invented.
    s.requests.some((r) => r.id === id)
      ? s
      : { requests: [...s.requests, { id, label }] }
  )),

  /** Take everything pending and empty the queue in ONE step.
   *
   * Read and cleared together, synchronously, because the drain runs in an
   * effect and StrictMode runs that effect twice in development: a drain that
   * read from a render snapshot would see the same requests again on the
   * second run and add them twice.
   */
  take: () => {
    const pending = get().requests;
    if (pending.length) set({ requests: [] });
    return pending;
  },

  clear: () => set({ requests: [] }),

  //: Bumped after every successful add-on run. The citations panel keys its
  //: fetch on the RESULT id, and an add-on run mutates the same result — so
  //: without a trigger the panel never re-fetches and the methods paragraph
  //: goes on describing the run before this one. The panel also lives on a
  //: different page than the run, which is why this travels the same way the
  //: layer request does.
  citationTick: 0,
  bumpCitations: () => set((s) => ({ citationTick: s.citationTick + 1 })),

  //: Base ids (everything before the run token) whose layers are no longer
  //: what they say they are. A re-run OVERWRITES the map in the server's
  //: store, which is keyed without the token — so a layer from the previous
  //: run keeps its id, its label and its legend, and the next time its bitmap
  //: is dropped (a quick-mode switch flushes the cache without removing the
  //: layer) it re-fetches and draws the NEW run's numbers under the OLD run's
  //: label. That is a figure that says 3 components while showing 2.
  superseded: [],
  supersede: (baseId) => set((s) => (
    s.superseded.includes(baseId)
      ? s
      : { superseded: [...s.superseded, baseId] })),
  takeSuperseded: () => {
    const pending = get().superseded;
    if (pending.length) set({ superseded: [] });
    return pending;
  },
}));

export default useAddonLayerRequests;
