/**
 * PhaseMapPanel — owns the EDS phase-map state on the EDS page.
 *
 * Responsibilities split into two render slots so the parent decides
 * the page layout:
 *   - ``renderControls`` returns the right-rail controls (auto-classify,
 *     legend, region painting, send-to-indexing).
 *   - ``renderCanvas`` returns the phase-map image + click handler for
 *     the center column. The canvas is also returned wrapped in the
 *     same letterbox-tolerant click conversion the element map uses,
 *     so a click on a 4:3 letterboxed phase map still hits the right
 *     pixel.
 *
 * The two slots share a single state instance, so toggling between
 * the element map and the phase map in the parent does NOT lose the
 * classification or the user's manual edits.
 *
 * Lifecycle:
 *   - on mount + when the file path changes, query GET /phase-map so
 *     the panel reflects whatever the backend already has (e.g. after
 *     a page navigation away-and-back).
 *   - the backend's ``close_file()`` hook clears the store on file
 *     switch, so a stale phase map can never bleed into a new file.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { edsApi } from '../../services/api';
import {
  colors as C, alpha,
  Button, NumberInput, GroupBox, Label,
  ConfirmDialog,
} from '../../theme/components';
import useDataStore from '../../stores/useDataStore';
import usePhaseColorStore from '../../stores/usePhaseColorStore';
import useCollectionStore from '../../stores/useCollectionStore';
import { activeKeySet, activeKeySignature, keyForPath } from '../PhaseCollections/collectionFilter';
import { pointerToRowCol } from './mapCoords';
import {
  IDENTITY_VIEW, isZoomed, viewToTransform, wheelFactor, zoomedRect,
} from './zoomView';
import RegionPanel from './RegionPanel';
import SuggestLibrarySkipped from './SuggestLibrarySkipped';
import QuantificationNote from './QuantificationNote';
import { elementSymbols } from './elementSymbol';

/** Store key for a phase colour.
 *
 * The EBSD phase map names a phase by the file stem, the EDS candidate list
 * carries the full filename. Stripping `.cif` makes them the same key, so a
 * colour picked on either page shows on both. The backend normalises the
 * same way (`phase_map_store._norm_phase_name`).
 */
export function phaseNameKey(cifFilename) {
  const n = String(cifFilename || '').trim();
  return n.toLowerCase().endsWith('.cif') ? n.slice(0, -4) : n;
}
import WandOverlay from './WandOverlay';
import { useEdsWand } from './hooks/useEdsWand';
import PresetBar from './PresetBar';
import ExportDialog from './ExportDialog';

/**
 * How many hand-given region names a re-classification would throw away.
 *
 * The asymmetry this counts is deliberate on the backend and it is the trap
 * three users independently named as the worst on this page: painting a pixel
 * LOCKS it (`assign_mask`), naming a region does NOT (`assign_region_phase`).
 * So re-classifying keeps the paint and discards the names.
 *
 * The count comes from the ids the USER assigned during this session,
 * intersected with the regions that still exist and still carry a phase.
 * The payload has no "named by hand" flag — every region gets a
 * `phase_index` from the classifier — so the session's own record of what the
 * user did is the only honest source. It is deliberately not persisted:
 * after a reload we do not know, and claiming a number we cannot back would
 * be worse than the generic warning.
 */
export function countHandNamedRegions(regions, namedIds) {
  if (!namedIds || !namedIds.size) return 0;
  return (regions || []).filter(
    (r) => namedIds.has(r.region_id) && Number(r.phase_index) >= 0,
  ).length;
}

/**
 * The smoothing width the backend actually used, lifted out of a classify
 * response — or `null` when the response carries no such statement.
 *
 * Deliberately not a spread of the whole response: the fields below are the
 * contract (`_resolve_scale` in `routes/eds.py` builds exactly them and
 * `response.update(scale_report)` flattens them onto the payload), and
 * copying only those keeps a rendered readout from picking up a same-named
 * field that means something else.
 *
 * `scale_source` is a MACHINE CODE — `um` / `pixels` / `pixels_no_step` /
 * `not_applicable`. It gets translated; `scale_note` is the backend's English
 * prose and is the fallback for a code this build does not know, so a new
 * code degrades to a sentence rather than to silence.
 */
export function readScaleReport(data) {
  if (!data || typeof data !== 'object') return null;
  if (!('scale_source' in data) && !('scale_px_used' in data)) return null;
  return {
    scale_px_used: data.scale_px_used ?? null,
    scale_um_used: data.scale_um_used ?? null,
    scale_box_um: data.scale_box_um ?? null,
    step_x_um: data.step_x_um ?? null,
    step_y_um: data.step_y_um ?? null,
    scale_source: data.scale_source ?? null,
    scale_um_requested: data.scale_um_requested ?? null,
    scale_note: data.scale_note ?? null,
  };
}

/** Round for display without pretending to a precision the step size lacks. */
const um1 = (v) => (Number.isFinite(Number(v)) ? Number(v).toFixed(2).replace(/0$/, '') : '');

const DEFAULT_TOLERANCE = 15.0;
const DEFAULT_MIN_SCORE = 0.3;

/**
 * `narrowSelection` from `collectionFilter.js`, translated into the key space
 * this panel actually works in.
 *
 * The active collection stores E2 STEMS (`Al`); `edsApi.cifPhases()` returns
 * E1 keys — CIF filenames (`Al.cif`, `cif_phase_library.py:343`
 * `key=path.name`). `narrowSelection` compares its `allKeys` directly against
 * the collection's keys, so calling it with raw `p.key` values would compare
 * `Al.cif` against `Al` and match nothing — every EDS phase would fall
 * outside every collection. This maps each phase to its stem for the
 * membership test, but returns the ORIGINAL `p.key` (the filename), because
 * that is the key space `selectedPhaseKeys` and the classify request live in.
 */
export function cifNarrowSelection(cifPhases, collectionKeys) {
  const all = cifPhases.map((p) => p.key);
  if (!collectionKeys) {
    return { keys: all.slice(), inCollection: all.length, usableHere: all.length };
  }
  const keys = cifPhases
    .filter((p) => collectionKeys.has(keyForPath(p.key)))
    .map((p) => p.key);
  return { keys, inCollection: collectionKeys.size, usableHere: keys.length };
}

/**
 * usePhaseMap — small custom hook so EDSPage doesn't grow a dozen
 * useState lines just to host the phase map. Returns a stable handle
 * with state + actions; consumers use the helpers, never the setters.
 */
export function usePhaseMap({ onIndexingHandoff } = {}) {
  const { t } = useTranslation('eds');
  const filePath = useDataStore(s => s.filePath);
  // The active phase collection, if any — narrows which CIF phases are
  // pre-ticked below, the same way the toolbar picker and the Phase Tester
  // read it. `data.state` is `{}` until the store's first load resolves, so
  // `?.active` (not `.active`) is required — reading past an undefined
  // `state` would throw, not just read as "no collection".
  const collections = useCollectionStore((s) => s.data.collections);
  const activeName = useCollectionStore((s) => s.data.state?.active || null);

  const [phaseMap, setPhaseMap] = useState(null);   // GET /phase-map response
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [tolerance, setTolerance] = useState(DEFAULT_TOLERANCE);
  const [minScore, setMinScore] = useState(DEFAULT_MIN_SCORE);

  // Manual region paint state (M4)
  const [selectedPhaseIndex, setSelectedPhaseIndex] = useState(null);
  const [region, setRegion] = useState({ rowStart: 0, rowEnd: 0, colStart: 0, colEnd: 0 });
  const [assignBusy, setAssignBusy] = useState(false);

  // Paint mode: 'rectangle' uses drag-to-rect (M4-rect), 'polygon' uses
  // click-to-add-vertex with explicit close (added later — rectangle is
  // good for axis-aligned regions, polygon for irregular grain
  // outlines).
  const [paintMode, setPaintMode] = useState('rectangle');  // 'rectangle' | 'polygon' | 'wand'
  const wand = useEdsWand();
  const [replaceFrom, setReplaceFrom] = useState(null);
  // Regions first: the classification groups pixels by composition alone,
  // and naming them is a separate act. 'regions' | 'phases'.
  const [mapView, setMapView] = useState('regions');
  const [selectedRegionId, setSelectedRegionId] = useState(null);
  // Smoothing box width for the clustering. null = the backend default (5).
  // Without it KMeans returns confetti and the cluster count has to stay tiny
  // to hide that - measured at k=8 on SampleB: 3491 pieces, median size 1 px.
  const [scale, setScale] = useState(null);
  // Smoothing expressed as a LENGTH. `scale` alone is not portable: the same
  // "5" is a 2.5 um box at a 0.5 um step and a 3.75 um box at a 0.75 um step,
  // so a recipe carried between two scans changes the analysis by 50 % while
  // the control reads the same number. 'px' stays the default — users who
  // know their data think in pixels, and the pixel path must keep the exact
  // arithmetic every map this app ever made was built with.
  const [scaleUnit, setScaleUnit] = useState('px');   // 'px' | 'um'
  const [scaleUm, setScaleUm] = useState(null);
  // What the backend actually smoothed with, as it reported it. RESOLVED,
  // never requested: "asked for 2 um, used 4 px, i.e. 2.0 um" is the whole
  // point, and a request the file could not honour has to say so rather than
  // look like it worked.
  const [scaleReport, setScaleReport] = useState(null);
  // What the RUN said about its own trustworthiness. Held APART from
  // `phaseMap` deliberately: `phaseMap` is overwritten by every later
  // `GET /phase-map` answer — a colour click, a merge, a paint, a name — and
  // none of those carry `warnings`. A warning parked on the map object would
  // therefore vanish the first time the user touched anything, which is
  // precisely when they are least likely to look again. These describe the
  // CLASSIFICATION, not the current pixels, so they stand until the next
  // classification replaces them or the map is cleared.
  const [classifyWarnings, setClassifyWarnings] = useState([]);
  // The elements this scan measured, as plain symbols. Recorded with a saved
  // preset so `element_set_differs` has something to compare against — see
  // `elementSymbol.js` for why the Aztec line names ("Al Kα1") must not
  // travel.
  const [measuredElements, setMeasuredElements] = useState([]);
  const [regionBusy, setRegionBusy] = useState(false);
  // User-authored rules. They decide which phases may COMPETE for a region,
  // and they only take effect on the next classification - like the scale and
  // the region count, and for the same reason: changing them silently under
  // an existing map would leave the picture and its explanation disagreeing.
  const [rules, setRules] = useState(null);
  // Hand-declared regions and per-element clustering weights. Both are
  // opt-in: empty means the automatic grouping runs exactly as before.
  const [regionDefs, setRegionDefs] = useState([]);
  const [elementWeights, setElementWeights] = useState({});
  const [clusterRemainder, setClusterRemainder] = useState(true);
  // Bumped by anything that changes the grouping. Merging renumbers
  // ids, splitting adds them, a boundary move changes the pixels — the
  // inspector has to re-read or it describes a region that is gone.
  const [mapVersion, setMapVersion] = useState(0);
  // Region ids the user named by hand, and the pending "are you sure" that
  // guards them. Held here rather than in the controls so EVERY route to a
  // re-classification is covered — the button in the rail, the Apply in the
  // region-definition editor and the Apply in the rules editor all call
  // `handleAutoClassify`, and a guard on only the first is not a guard.
  const [handNamedRegionIds, setHandNamedRegionIds] = useState(() => new Set());
  const [pendingReclassify, setPendingReclassify] = useState(false);
  // The SAME store the EBSD phase map uses, so a phase keeps its colour on
  // both pages and the user's choice survives a reload.
  const colorOverrides = usePhaseColorStore((s) => s.overrides);
  const setPhaseColor = usePhaseColorStore((s) => s.setColor);
  const resetPhaseColor = usePhaseColorStore((s) => s.resetColor);
  const [polygonVertices, setPolygonVertices] = useState([]);  // [[col, row], ...]

  // Last clicked pixel (for the readout under the canvas)
  const [hoveredPixel, setHoveredPixel] = useState(null);

  // Grouping mode. 'cluster' groups the composition and matches each
  // group's mean; 'pixel' matches every pixel on its own. Cluster is the
  // default because per-pixel composition on real data carries several
  // at% of systematic error (see the 2026-08-19 design spec).
  const [mode, setMode] = useState('cluster');
  const [nClusters, setNClusters] = useState(null);   // null -> backend picks it

  // Which library phases take part. Empty set = "not loaded yet"; the
  // request only narrows when a STRICT subset is ticked, so a stale list
  // can never silently drop phases from a run.
  const [cifPhases, setCifPhases] = useState([]);
  const [selectedPhaseKeys, _setSelectedPhaseKeys] = useState(new Set());
  // True while `selectedPhaseKeys` is still exactly the collection-derived
  // auto-seed; false once the user hand-edits it (checkbox, select-all/none,
  // or "show all phases"). The collection-follow effect below only ever
  // overwrites the selection while this is still true — App.jsx keeps every
  // page mounted and merely hides inactive ones with `display: none`, so the
  // active collection can change on another page while this one sits in the
  // background, and an untouched selection should follow it back in without
  // ever clobbering a choice the user already made. Same pattern as
  // SinglePixelPhaseTestDialog's `isAutoSeedRef`.
  const isAutoSeedRef = useRef(true);
  // How many phases that appeared on the LAST refresh sit outside the active
  // collection (and so were left unticked). Purely informational; reset on
  // every fresh per-file load and on every refresh (including down to 0 when
  // nothing was left out).
  const [newOutsideCollection, setNewOutsideCollection] = useState(0);

  // The setter every consumer uses (checkbox toggle, select-all/none, "show
  // all phases" below). Marks the selection as hand-edited so the collection-
  // follow effect never overwrites a choice the user just made. Seeding code
  // in this hook uses the raw `_setSelectedPhaseKeys` instead, precisely
  // because seeding is not a hand edit.
  const setSelectedPhaseKeys = useCallback((updater) => {
    isAutoSeedRef.current = false;
    _setSelectedPhaseKeys(updater);
  }, []);

  const loadCifPhases = useCallback(async () => {
    try {
      const res = await edsApi.cifPhases();
      const phases = res.data?.phases || [];
      setCifPhases(phases);
      // Seed the selection from the active collection, if any — narrowed to
      // the E1 key space (CIF filenames) `selectedPhaseKeys` actually lives
      // in; see `cifNarrowSelection`. `cifPhases` itself STAYS the full
      // list: only `selectedPhaseKeys` is narrowed here. Shrinking
      // `cifPhases` instead would make `selectedPhaseKeys.size ===
      // cifPhases.length` true again for a fully-selected collection, the
      // `isSubset` check in `runClassify` would go false, and the run would
      // silently classify against the WHOLE library — see
      // phaseMapCollection.test.jsx for the pinned regression.
      //
      // Read the collection LIVE from the store here, not the
      // `collections`/`activeName` render-scope selectors: this callback's
      // closure over those was captured when the effect was SCHEDULED, but
      // the promise can resolve after the user switched the active
      // collection elsewhere — this page can sit mounted-but-hidden while
      // that happens (App.jsx never unmounts pages) — and the collection-
      // follow effect below only fires on an `activeName` CHANGE, so if that
      // change already happened before this `.then()` resolves, nothing
      // would ever correct a seed taken from the stale closed-over value.
      const live = useCollectionStore.getState().data;
      const colKeys = activeKeySet(live?.collections || [], live?.state?.active || null);
      const { keys } = cifNarrowSelection(phases, colKeys);
      _setSelectedPhaseKeys(new Set(keys));
      isAutoSeedRef.current = true;   // fresh seed: nothing hand-edited yet
      setNewOutsideCollection(0);     // a fresh load, not a refresh — nothing to report yet
      return phases;
    } catch {
      setCifPhases([]);
      return [];
    }
  }, []);

  // Load the phase list once per file so the picker has something to show
  // before the first classification.
  useEffect(() => { loadCifPhases(); }, [loadCifPhases, filePath]);

  // Follow the active collection while this page sits mounted-but-hidden —
  // AND follow a membership change to that SAME collection made elsewhere
  // while this page never reloaded its phase list (Task 10 added the one
  // place that can do that: the database browser's "move to collection", or
  // the Collection Manager, editing the collection this page is already
  // showing). `membersSignature` is a content fingerprint of exactly that
  // collection's effective members (`activeKeySignature`, see
  // `collectionFilter.js`), not `collections` itself — `collections` gets a
  // brand-new array reference from the store on every refresh anywhere in
  // the app (including an unrelated collection's own edit, or this page's
  // OWN periodic poll of something else), and depending on it directly would
  // reseed the picker on every one of those for no reason; the signature
  // only changes when THIS collection's members did.
  //
  // App.jsx keeps every page mounted and merely hides inactive ones with
  // `display: none` — the user can flip the toolbar's active collection (or
  // edit it) on a different, visible page and come back here without this
  // page ever having reloaded its phase list. Without this effect the count
  // line below would recompute against the NEW `activeName`/membership on
  // every render (it is derived directly, not cached) while
  // `selectedPhaseKeys` stayed pinned to the OLD selection — the count line
  // would then name a collection the checkboxes disagree with.
  //
  // Deliberately separate from `loadCifPhases` above (not merged into it, and
  // not added to its deps) so that function's own "seeds once per file"
  // intent stays true; this effect reseeds too, but ONLY while nothing has
  // been hand-edited since the last seed. Mirrors
  // SinglePixelPhaseTestDialog's own collection-follow effect.
  const membersSignature = activeKeySignature(collections, activeName);
  useEffect(() => {
    if (!isAutoSeedRef.current || cifPhases.length === 0) return;
    const colKeys = activeKeySet(collections, activeName);
    const { keys } = cifNarrowSelection(cifPhases, colKeys);
    _setSelectedPhaseKeys(new Set(keys));
    // still true: re-deriving from the collection is not a hand edit
    isAutoSeedRef.current = true;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeName, membersSignature]);

  // ...and again whenever the user opens the picker, because the library can
  // grow under it. Since 2026-09-12 the backend reads `Database/CIF_Library`
  // itself and a downloaded CIF is a phase immediately — but this list was
  // fetched once per file, so a CIF downloaded mid-session was missing from the
  // checkboxes until the file was reloaded. It never silently dropped a phase
  // from a run (`runClassify` sends `phase_keys` only for a STRICT subset, and
  // a stale all-ticked list is not one), so the damage was confined to the
  // picker: the user could not deselect around a phase they could not see.
  //
  // Opening the <details> is the moment the list is about to be read, so it is
  // the moment to refresh — no polling, and no request on a panel nobody opened.
  // Which keys the picker has already SHOWN the user. Declared before the
  // callback that reads it: safe either way today, but one refactor away from a
  // TDZ error if anything ever called the callback during this render pass.
  // Without it "new since last time" cannot be told from "deselected", and the
  // two need opposite treatment.
  const knownPhaseKeysRef = useRef(new Set());

  const refreshCifPhases = useCallback(async () => {
    let phases;
    try {
      phases = (await edsApi.cifPhases()).data?.phases || [];
    } catch {
      return;                     // keep what is on screen; this is a refresh
    }
    // Which keys are new since the picker last showed the list. Computed HERE,
    // from the ref, and not inside the updater below: React may invoke an
    // updater twice, and a ref written inside one would make the second pass
    // see the new keys as already known and drop them.
    const known = knownPhaseKeysRef.current;
    const appeared = new Set(phases.filter(p => !known.has(p.key)).map(p => p.key));
    setCifPhases(phases);
    // With a collection active, an appeared phase is only auto-ticked when it
    // is IN the collection — otherwise a CIF the user downloaded mid-session
    // for something unrelated would be silently ticked, and the run would be
    // wider than the collection named in the toolbar says. Read live for the
    // same race reason as `loadCifPhases` above: this function is triggered
    // by a synchronous user click (opening the <details>), but the fetch it
    // awaits can still outlive a collection switch made elsewhere on this
    // mounted-but-hidden page. Split into `appearedIn`/`appearedOutside`
    // BEFORE the updater below, not inside it — same double-invoke reason
    // `appeared` itself is computed from the ref rather than inline.
    const live = useCollectionStore.getState().data;
    const colKeys = activeKeySet(live?.collections || [], live?.state?.active || null);
    const appearedIn = new Set();
    const appearedOutside = [];
    for (const key of appeared) {
      // keyForPath maps the EDS key `Al.cif` onto the collection key `Al`.
      if (!colKeys || colKeys.has(keyForPath(key))) appearedIn.add(key);
      else appearedOutside.push(key);
    }
    setNewOutsideCollection(appearedOutside.length);
    // Merge, never reset. `loadCifPhases` ticks everything, which is right for a
    // first load and wrong here: it would throw away a deselection the user made
    // seconds ago. A phase that appeared since is ticked, because "all take
    // part" is the default; one that is gone stops being counted. An EMPTY
    // selection is left empty — at refresh time the list has been shown, so
    // empty almost always means the user pressed "Select none", and re-ticking
    // would overrule the clearest instruction they can give.
    //
    // "Almost always": the merge itself can empty the set, if every ticked
    // phase has left the library and nothing new arrived. The next refresh then
    // reads that as an explicit "none". It needs a mid-session deletion of
    // exactly the user's selection, the state is visible (`n/N` in the summary
    // plus the orange "none selected" line), and "Select all" recovers it — so
    // it is left as it stands rather than guessed at.
    _setSelectedPhaseKeys((prev) => {
      if (prev.size === 0) return prev;
      const next = new Set();
      for (const p of phases) {
        if (prev.has(p.key) || appearedIn.has(p.key)) next.add(p.key);
      }
      return next;
    });
  }, []);

  useEffect(() => { knownPhaseKeysRef.current = new Set(cifPhases.map(p => p.key)); },
            [cifPhases]);

  // The measured element list, per file, for what a saved preset records it
  // was authored against. Failure is silent and leaves the list EMPTY, which
  // makes the save omit the field — the honest outcome. Inventing a list
  // would hand `element_set_differs` a claim nobody made.
  useEffect(() => {
    let alive = true;
    // A scale report describes ONE run on ONE file. Carrying it across a file
    // change would quote the previous scan's step size beside this one's map.
    setScaleReport(null);
    (async () => {
      try {
        const res = await edsApi.elements();
        if (alive) setMeasuredElements(elementSymbols(res?.data?.elements || []));
      } catch {
        if (alive) setMeasuredElements([]);
      }
    })();
    return () => { alive = false; };
  }, [filePath]);

  /**
   * Take the region definitions and weights a stored map was built with.
   *
   * Only ever ADDS: a map that carries none leaves the editor as it is, so
   * arriving at a page whose backend map predates this feature cannot wipe
   * definitions the user is halfway through writing. Called only where the
   * map came from disk — after an action the user just took, the editor
   * already holds the truth and the round trip would only fight it.
   */
  const adoptMapProvenance = useCallback((data) => {
    if (!data) return;
    if (Array.isArray(data.region_defs) && data.region_defs.length) {
      setRegionDefs(data.region_defs);
    }
    if (data.element_weights && Object.keys(data.element_weights).length) {
      setElementWeights(data.element_weights);
    }
  }, []);

  // On mount or file change, sync from backend. The backend clears its
  // store on close_file(), so a 200-with-loaded:false response is the
  // expected "nothing to show" state, not an error.
  useEffect(() => {
    let cancelled = false;
    setError(null);
    // Windows are written in at% against ONE dataset's chemistry. Carrying
    // them to the next file leaves numbers that mean something else silently
    // armed, and the next Classify applies them. Whatever the new file's own
    // stored map carries is adopted below.
    setRegionDefs([]);
    setElementWeights({});
    setClusterRemainder(true);
    setHandNamedRegionIds(new Set());
    // They describe a run against ANOTHER scan's chemistry. Whatever this
    // file's own stored map carries, it carries no warnings — they are not
    // persisted — so the honest state on arrival is none.
    setClassifyWarnings([]);
    edsApi.getPhaseMap(true)
      .then((res) => {
        if (cancelled) return;
        setPhaseMap(res.data?.loaded ? res.data : null);
        if (res.data?.loaded) adoptMapProvenance(res.data);
      })
      .catch(() => { /* 4xx is fine — just means nothing classified yet */ });
    return () => { cancelled = true; };
  }, [filePath, adoptMapProvenance]);

  // Push the colour choices to the backend, which renders the PNG. Runs on
  // mount too: the store is persisted, so a returning user's colours must
  // reach a freshly started backend before the first render.
  useEffect(() => {
    let cancelled = false;
    edsApi.setPhaseColors(colorOverrides)
      .then((res) => {
        // Deliberately NOT adopting provenance here. This effect runs on
        // every colour change, and adopting would replace a definition the
        // user just deleted or a threshold they just retyped with the last
        // classified map's copy - an edit silently rolled back by clicking a
        // swatch. Adoption belongs where the map arrives from disk.
        if (!cancelled && res.data?.loaded) setPhaseMap(res.data);
      })
      .catch(() => { /* colours are a preference; never break the map */ });
    return () => { cancelled = true; };
  }, [colorOverrides]);

  // Every region tool answers with the whole refreshed map, so they share
  // one caller. Keeping them separate would mean five copies of the same
  // error handling and the same chance for one of them to drift.
  const runRegionOp = useCallback(async (fn, args) => {
    setRegionBusy(true);
    setError(null);
    try {
      const res = await fn(args);
      if (res.data?.loaded) setPhaseMap(res.data);
      setMapVersion((v) => v + 1);
      return res.data;
    } catch (e) {
      setError(e.response?.data?.detail || String(e));
      return null;
    } finally {
      setRegionBusy(false);
    }
  }, []);

  const handleAssignRegionPhase = useCallback(async (regionId, phaseIndex) => {
    const data = await runRegionOp(edsApi.assignRegionPhase, { regionId, phaseIndex });
    // Only on success, and only when a phase was actually given: clearing a
    // region back to unclassified is not a name to warn about losing.
    if (data) {
      setHandNamedRegionIds((prev) => {
        const next = new Set(prev);
        if (Number(phaseIndex) >= 0) next.add(Number(regionId));
        else next.delete(Number(regionId));
        return next;
      });
    }
    return data;
  }, [runRegionOp]);

  const handleMergeRegions = useCallback((keepId, dropId) =>
    runRegionOp(edsApi.mergeRegions, { keepId, dropId }),
  [runRegionOp]);

  const handleSplitRegion = useCallback((regionId, nParts = 2) =>
    runRegionOp(edsApi.splitRegion, { regionId, nParts }),
  [runRegionOp]);

  const handleGrowRegion = useCallback((regionId, nPixels) =>
    runRegionOp(edsApi.growRegion, { regionId, nPixels }),
  [runRegionOp]);

  const handleSnapEdges = useCallback((strength) =>
    runRegionOp(edsApi.snapRegionEdges, { strength }),
  [runRegionOp]);

  const runClassify = useCallback(async () => {
    // "None selected" must not run the whole library — that inverts the
    // clearest instruction the user can give. Refuse instead.
    if (cifPhases.length > 0 && selectedPhaseKeys.size === 0) {
      setError(t('phaseMap.noneSelected'));
      return;
    }
    setLoading(true); setError(null);
    try {
      // Only send phase_keys for a STRICT subset — an all-selected list
      // means "everything", and sending it would freeze the run against a
      // library the user may since have extended.
      const isSubset = cifPhases.length > 0
        && selectedPhaseKeys.size > 0
        && selectedPhaseKeys.size < cifPhases.length;
      const res = await edsApi.autoClassify({
        tolerance,
        minScore,
        mode,
        nClusters,
        // Exactly one statement about the smoothing box. In um the pixel
        // count is the backend's answer, not ours — it depends on this
        // file's step size, which is the entire reason the field exists.
        ...(scaleUnit === 'um' && Number.isFinite(Number(scaleUm)) && Number(scaleUm) > 0
          ? { scaleUm: Number(scaleUm) }
          : (scale != null ? { scale } : {})),
        ...(rules && (rules.rules?.length || rules.phase_keys) ? { rules } : {}),
        ...(isSubset ? { phaseKeys: [...selectedPhaseKeys] } : {}),
        elementWeights,
        // A definition with no clause in it claims nothing on the backend,
        // but sending it would still flip the run onto the manual path and
        // put every pixel in the leftover region. Drop the half-written
        // ones here so an in-progress edit cannot wipe the map.
        regionDefs: regionDefs.filter(
          (d) => d.elements?.length || d.ratios?.length || d.enrichment?.length),
        clusterRemainder,
      });
      setPhaseMap(res.data);
      // Only ever from a classify response, and always replaced — an answer
      // without the key clears them rather than leaving the previous run's
      // warnings standing beside a map they no longer describe.
      setClassifyWarnings(
        Array.isArray(res.data?.warnings) ? res.data.warnings : []);
      const report = readScaleReport(res.data);
      setScaleReport(report);
      // One number on screen for one box. Without this write-back the pixel
      // slider would keep showing the value it had while the run used the
      // resolved one, and the panel would contradict itself about what was
      // averaged over.
      if (report?.scale_source === 'um' && report.scale_px_used != null) {
        setScale(Number(report.scale_px_used));
      }
      setSelectedPhaseIndex(null);
      setSelectedRegionId(null);
      // The regions were rebuilt from scratch; whatever ids were named by
      // hand describe groups that no longer exist.
      setHandNamedRegionIds(new Set());
    } catch (err) {
      setError(err.response?.data?.detail || err.message || t('phaseMap.errorClassify'));
    } finally {
      setLoading(false);
    }
  }, [tolerance, minScore, mode, nClusters, scale, scaleUnit, scaleUm,
      rules, regionDefs,
      elementWeights, clusterRemainder, cifPhases, selectedPhaseKeys, t]);

  const handNamedCount = useMemo(
    () => countHandNamedRegions(phaseMap?.regions, handNamedRegionIds),
    [phaseMap, handNamedRegionIds],
  );

  /**
   * Classify, asking first when it would destroy work.
   *
   * Anything that can destroy work is a confirmation, not a tooltip. With
   * nothing named by hand there is nothing to lose, so the run starts
   * immediately — a dialog on every click would train the user to dismiss the
   * one that matters.
   */
  const handleAutoClassify = useCallback(async () => {
    if (handNamedCount > 0) { setPendingReclassify(true); return; }
    await runClassify();
  }, [handNamedCount, runClassify]);

  const confirmReclassify = useCallback(async () => {
    setPendingReclassify(false);
    await runClassify();
  }, [runClassify]);

  const cancelReclassify = useCallback(() => setPendingReclassify(false), []);

  const handleClearMap = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      await edsApi.clearPhaseMap();
      setPhaseMap(null);
      setClassifyWarnings([]);
      setSelectedPhaseIndex(null);
      setHandNamedRegionIds(new Set());
    } catch (err) {
      setError(err.response?.data?.detail || err.message || t('phaseMap.errorClear'));
    } finally {
      setLoading(false);
    }
  }, [t]);

  // Assign the single clicked pixel. Goes through the same endpoint as a
  // rectangle (r0==r1, c0==c1) so it inherits the grid-mismatch guard and
  // the locked_mask bookkeeping rather than opening a second write path.
  const handleAssignPixel = useCallback(async (row, col, phaseIndex) => {
    setAssignBusy(true); setError(null);
    try {
      const res = await edsApi.assignRegion(
        Number(row), Number(row), Number(col), Number(col), Number(phaseIndex));
      setPhaseMap(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || t('phaseMap.errorAssign'));
    } finally {
      setAssignBusy(false);
    }
  }, [t]);

  const handleUndo = useCallback(async () => {
    setError(null);
    try {
      const res = await edsApi.phaseMapUndo();
      setPhaseMap(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || t('phaseMap.errorAssign'));
    }
  }, [t]);

  // The correction a seeded selection cannot make: the recorded case is
  // "sd_0302719 won 55 % of my map and it should be Al". Lassoing 55 % of a
  // map by hand is not a workflow.
  const handleReplacePhase = useCallback(async (fromIdx, toIdx) => {
    if (fromIdx == null || toIdx == null || fromIdx === toIdx) return;
    setAssignBusy(true); setError(null);
    try {
      const res = await edsApi.replacePhase(Number(fromIdx), Number(toIdx));
      setPhaseMap(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || t('phaseMap.errorAssign'));
    } finally {
      setAssignBusy(false);
    }
  }, [t]);

  const handleAssignRegion = useCallback(async (phaseIndex) => {
    if (!phaseMap) return;
    const r = region;
    setAssignBusy(true); setError(null);
    try {
      const res = await edsApi.assignRegion(
        Number(r.rowStart), Number(r.rowEnd),
        Number(r.colStart), Number(r.colEnd),
        Number(phaseIndex),
      );
      setPhaseMap(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || t('phaseMap.errorAssign'));
    } finally {
      setAssignBusy(false);
    }
  }, [phaseMap, region, t]);

  const handleSendToIndexing = useCallback(() => {
    if (!phaseMap) return;
    onIndexingHandoff?.(phaseMap);
  }, [phaseMap, onIndexingHandoff]);

  const handleClosePolygon = useCallback(async (phaseIndex) => {
    if (polygonVertices.length < 3) return;
    setAssignBusy(true); setError(null);
    try {
      const res = await edsApi.assignPolygon(polygonVertices, Number(phaseIndex));
      setPhaseMap(res.data);
      setPolygonVertices([]);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || t('phaseMap.errorPolygon'));
    } finally {
      setAssignBusy(false);
    }
  }, [polygonVertices, t]);

  const handleCancelPolygon = useCallback(() => {
    setPolygonVertices([]);
  }, []);

  return {
    phaseMap, loading, error,
    // Which scan every one of these settings describes. The controls key
    // their preset attestation on it: a compatibility report is a statement
    // about ONE file, and carrying it to the next one is a false claim.
    filePath,
    tolerance, setTolerance, minScore, setMinScore,
    mode, setMode, nClusters, setNClusters,
    cifPhases, loadCifPhases, refreshCifPhases,
    // The active collection, its keys already narrowed to what this hook can
    // use — computed here so PhaseMapControls never has to know the E1/E2 key
    // split `cifNarrowSelection` exists to hide.
    collections, activeName, newOutsideCollection,
    colorOverrides, setPhaseColor, resetPhaseColor,
    mapView, setMapView,
    selectedRegionId, setSelectedRegionId,
    scale, setScale, regionBusy, mapVersion,
    scaleUnit, setScaleUnit, scaleUm, setScaleUm, scaleReport,
    classifyWarnings,
    // What a saved preset records it was authored against. `step_x_um` is
    // only known once a classification has answered — before that the field
    // is omitted rather than guessed.
    authoredElements: measuredElements,
    authoredStepUm: scaleReport?.step_x_um ?? null,
    rules, setRules,
    regionDefs, setRegionDefs,
    elementWeights, setElementWeights,
    clusterRemainder, setClusterRemainder,
    handleAssignRegionPhase, handleMergeRegions, handleSplitRegion,
    handleGrowRegion, handleSnapEdges,
    selectedPhaseKeys, setSelectedPhaseKeys,
    selectedPhaseIndex, setSelectedPhaseIndex,
    replaceFrom, setReplaceFrom,
    region, setRegion,
    assignBusy,
    hoveredPixel, setHoveredPixel,
    paintMode, setPaintMode,
    wand,
    // The wand's "Assign" handler writes the returned map straight back.
    setPhaseMap,
    polygonVertices, setPolygonVertices,
    handleAutoClassify, handleClearMap,
    handNamedCount, pendingReclassify, confirmReclassify, cancelReclassify,
    handleAssignRegion, handleAssignPixel, handleUndo, handleReplacePhase,
    handleClosePolygon, handleCancelPolygon,
    handleSendToIndexing,
  };
}


/**
 * Convert a click on the phase map to (row, col), through the zoom.
 *
 * `hostRect` must come from an element that is NOT itself transformed:
 * `getBoundingClientRect()` already includes a CSS transform, so measuring the
 * zoomed node would count the zoom twice. `zoomedRect` turns that untransformed
 * box into the virtual box the pixels are actually drawn in, and
 * `mapCoords.pointerToRowCol` does the letterbox arithmetic — the same routine
 * the element tiles use, rather than a second copy of it that can drift.
 */
export function pixelFromClickAt(e, hostRect, view, nRows, nCols) {
  if (!hostRect || !hostRect.width || !hostRect.height) return null;
  if (!nRows || !nCols) return null;
  return pointerToRowCol(e, zoomedRect(hostRect, view || IDENTITY_VIEW),
                         [nRows, nCols]);
}


/**
 * PhaseMapCanvas — renders the colored phase map and supports two
 * mouse interactions on it:
 *
 *   - **Click** = inspect pixel: store the (row, col) in the handle
 *     so the controls panel can render a per-pixel readout (which
 *     phase, which score) without a round-trip.
 *   - **Drag** = paint a rectangular region: write start/end into the
 *     ``region`` state. The overlay rectangle uses an SVG layer with
 *     ``viewBox="0 0 nCols nRows"`` and the same
 *     ``preserveAspectRatio`` as the img's ``objectFit: contain``, so
 *     the rectangle always lines up exactly with the underlying
 *     pixels — even when the image is letterboxed.
 *
 * No phase map → empty placeholder so the parent can still slot the
 * canvas into the layout without flicker.
 */
export function PhaseMapCanvas({ handle, onInspect, onAssignPixel, wand,
                                onPickRegion, view, onZoomAt, onPan,
                                claimOverlay,
                                onResetView, background }) {
  const { t } = useTranslation('eds');
  const {
    phaseMap, hoveredPixel, setHoveredPixel,
    region, setRegion,
    paintMode, polygonVertices, setPolygonVertices,
    mapView,
  } = handle;

  // The region view draws its own image. Falling back to the phase image
  // keeps a map made before regions existed (or a per-pixel run, which has
  // no groups at all) visible instead of blank.
  const shownImage = (mapView === 'regions' && phaseMap?.region_image)
    ? phaseMap.region_image
    : phaseMap?.image;
  const [drag, setDrag] = useState(null);  // { startRow, startCol, endRow, endCol } | null
  // Measured for the pointer maths; deliberately the UNtransformed box.
  const hostRef = useRef(null);
  const [panFrom, setPanFrom] = useState(null);
  const activeView = view || IDENTITY_VIEW;
  // The wheel listener is attached natively (it has to be non-passive to
  // preventDefault), so it reads the view through a ref rather than
  // closing over a stale one.
  const viewRef = useRef(activeView);
  viewRef.current = activeView;
  const zoomed = isZoomed(activeView);

  // Read the shape from the payload rather than from `nRows`/`nCols`: those
  // are declared after this component's early return, so referencing them here
  // would be a use-before-define at render time.
  // Ctrl+wheel zooms to the cursor, matching the element tiles. Attached
  // natively because React's synthetic wheel handler is passive and cannot
  // preventDefault, which the browser needs to not zoom the whole page.
  // A plain wheel is left alone so the panel can still be scrolled.
  useEffect(() => {
    const host = hostRef.current;
    if (!host || !onZoomAt) return undefined;
    const onWheel = (e) => {
      if (!e.ctrlKey) return;
      e.preventDefault();
      const rect = host.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      onZoomAt(
        wheelFactor(e.deltaY),
        (e.clientX - rect.left) / rect.width,
        (e.clientY - rect.top) / rect.height,
      );
    };
    host.addEventListener('wheel', onWheel, { passive: false });
    return () => host.removeEventListener('wheel', onWheel);
  }, [onZoomAt]);

  const pixelFromClick = useCallback((e) => pixelFromClickAt(
    e, hostRef.current?.getBoundingClientRect(), viewRef.current,
    phaseMap?.n_rows, phaseMap?.n_cols,
  ), [phaseMap?.n_rows, phaseMap?.n_cols]);

  const isPolygon = paintMode === 'polygon';
  const isWand = paintMode === 'wand';

  /** Drag with Ctrl (or in pan mode) moves a zoomed view instead of painting. */
  const beginPan = useCallback((e) => {
    setPanFrom({ x: e.clientX, y: e.clientY });
  }, []);

  const handleMouseDown = useCallback((e) => {
    if (onPan && zoomed && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      beginPan(e);
      return;
    }
    if (isPolygon) return;  // polygon mode uses click-to-add-vertex, not drag
    const px = pixelFromClick(e);
    if (!px || !phaseMap) return;
    e.preventDefault();
    setDrag({ startRow: px.row, startCol: px.col, endRow: px.row, endCol: px.col });
  }, [phaseMap, isPolygon, onPan, zoomed, beginPan]);

  const handleMouseMove = useCallback((e) => {
    if (panFrom) {
      const host = hostRef.current;
      if (host && onPan) {
        const rect = host.getBoundingClientRect();
        // Deltas relative to the LAST point, not the drag origin: the
        // controller accumulates, so sending the whole offset each move
        // would accelerate the pan quadratically.
        onPan(
          -(e.clientX - panFrom.x) / Math.max(1, rect.width),
          -(e.clientY - panFrom.y) / Math.max(1, rect.height),
        );
        setPanFrom({ ...panFrom, x: e.clientX, y: e.clientY });
      }
      return;
    }
    if (!drag) return;
    const px = pixelFromClick(e);
    if (!px) return;
    setDrag(d => d && { ...d, endRow: px.row, endCol: px.col });
  }, [drag, panFrom, onPan]);

  const handleMouseUp = useCallback((e) => {
    if (panFrom) { setPanFrom(null); return; }
    if (!drag) return;
    const px = pixelFromClick(e);
    const endRow = px ? px.row : drag.endRow;
    const endCol = px ? px.col : drag.endCol;
    const sameSpot = endRow === drag.startRow && endCol === drag.startCol;
    if (sameSpot) {
      // Treat as a click — don't create a 1-pixel "region", just inspect.
      setHoveredPixel({ row: endRow, col: endCol });
      // Drive the SAME per-pixel panels every other map on this page drives.
      // Without this the phase map was the one clickable surface that did
      // not move Pixel Quantification or Phase Suggestion, so the
      // suggestion silently kept describing a pixel picked somewhere else —
      // a user clicking a Cu-rich region got the candidate list for
      // wherever they last clicked a tile.
      onInspect?.(endRow, endCol);
      if (onPickRegion) {
        // Region view: the click selects the region under it. It must NOT
        // fall through to the single-pixel paint below — that is the phase
        // view's gesture, and here it would quietly hand-edit one pixel
        // instead of picking the region the user aimed at.
        onPickRegion(endRow, endCol);
      } else if (isWand) {
        // Wand mode: the click seeds a selection instead of assigning.
        wand?.seedAt(endRow, endCol);
      } else {
        // Armed phase -> the click also assigns that single pixel.
        onAssignPixel?.(endRow, endCol);
      }
    } else {
      const r0 = Math.min(drag.startRow, endRow);
      const r1 = Math.max(drag.startRow, endRow);
      const c0 = Math.min(drag.startCol, endCol);
      const c1 = Math.max(drag.startCol, endCol);
      setRegion({ rowStart: r0, rowEnd: r1, colStart: c0, colEnd: c1 });
      setHoveredPixel(null);
    }
    setDrag(null);
  }, [drag, setHoveredPixel, setRegion, onInspect, onAssignPixel, isWand, wand,
      onPickRegion]);

  const handleDoubleClick = useCallback(() => {
    if (onResetView) onResetView();
  }, [onResetView]);

  const handleMouseLeave = useCallback(() => {
    if (panFrom) setPanFrom(null);
    // Cancel the drag if the mouse leaves the image — otherwise a
    // mouseup outside the canvas would commit a stale rectangle.
    if (drag) setDrag(null);
  }, [drag]);

  const handleClick = useCallback((e) => {
    // Polygon mode only — rectangle mode handles selection via
    // mousedown/up. Each click adds a vertex; closing happens via
    // the explicit "Close polygon" button in the controls panel.
    if (!isPolygon) return;
    const px = pixelFromClick(e);
    if (!px) return;
    e.preventDefault();
    setPolygonVertices(prev => [...prev, [px.col, px.row]]);
  }, [isPolygon, setPolygonVertices]);

  if (!phaseMap?.image) {
    return (
      <div style={{
        flex: 1, background: C.bgSecondary, border: `1px solid ${C.border}`,
        borderRadius: 6, display: 'flex', alignItems: 'center', justifyContent: 'center',
        minHeight: 250, color: C.textSecondary, textAlign: 'center', padding: 20,
      }}>
        <div>
          <div style={{ fontSize: '20pt', opacity: 0.3, marginBottom: 8 }}>{'□'}</div>
          <div style={{ fontSize: '10pt' }}>{t('phaseMap.noMapTitle')}</div>
          <div style={{ fontSize: '9pt', marginTop: 4, opacity: 0.7 }}>
            {t('phaseMap.noMapHintPre')}<b>{t('phaseMap.noMapHintAction')}</b>{t('phaseMap.noMapHintPost')}
          </div>
        </div>
      </div>
    );
  }

  const nRows = phaseMap.n_rows;
  const nCols = phaseMap.n_cols;

  // Pick which rectangle to draw: the live drag (during mousedown/move)
  // overrides any committed region so the user gets immediate feedback.
  const liveRect = drag
    ? {
        x: Math.min(drag.startCol, drag.endCol),
        y: Math.min(drag.startRow, drag.endRow),
        w: Math.abs(drag.endCol - drag.startCol) + 1,
        h: Math.abs(drag.endRow - drag.startRow) + 1,
      }
    : (region && (region.rowEnd > region.rowStart || region.colEnd > region.colStart))
      ? {
          x: region.colStart,
          y: region.rowStart,
          w: region.colEnd - region.colStart + 1,
          h: region.rowEnd - region.rowStart + 1,
        }
      : null;

  let readout = null;
  if (hoveredPixel) {
    const { row, col } = hoveredPixel;
    readout = (
      <div style={{
        position: 'absolute', bottom: 4, left: 4, right: 4,
        background: alpha(C.bg, 80), border: `1px solid ${C.border}`,
        borderRadius: 4, padding: '4px 8px', fontSize: '9pt',
        color: C.text, pointerEvents: 'none',
      }}>
        {t('phaseMap.readout', { row, col })}
      </div>
    );
  }

  return (
    <div style={{
      flex: 1, background: C.bgSecondary, border: `1px solid ${C.border}`,
      borderRadius: 6, position: 'relative',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      overflow: 'hidden', minHeight: 250,
    }}>
      {/* Image + overlay both stretch to the same logical box, so the
          SVG's preserveAspectRatio aligns the rectangle to the pixel
          grid regardless of how the image is letterboxed. The wrapper
          fills the container in BOTH dimensions — using only
          ``maxWidth/maxHeight`` would let a small EBSD grid (e.g.
          270×120) sit as a postage-stamp in the middle while the
          panel has 1500×800 px of empty real estate around it.
          ``objectFit: contain`` on the img + matching
          ``preserveAspectRatio: xMidYMid meet`` on the SVG keeps both
          letterbox calculations identical, so the rectangle / polygon
          overlay never drifts off the underlying pixels. */}
      {/* Two nested boxes on purpose. The OUTER one is what the pointer
          maths measures and what the wheel listener hangs on - it never
          moves, so getBoundingClientRect() stays the UNtransformed box and
          the zoom is not counted twice. The INNER one carries the
          transform, and the image, the wand overlay and the
          rectangle/polygon SVG all live inside it so they move together
          and cannot drift apart. */}
      <div
        ref={hostRef}
        onDoubleClick={handleDoubleClick}
        style={{
          position: 'relative', width: '100%', height: '100%',
          overflow: 'hidden',
          cursor: panFrom ? 'grabbing' : undefined,
        }}
      >
      <div style={{
        position: 'relative', width: '100%', height: '100%',
        transform: viewToTransform(activeView),
        transformOrigin: '0 0',
      }}>
        {/* Background first, the map over it. Both use objectFit:contain in the
            same box, so they register as long as they cover the same field of
            view - measured 60.00 um against 60.39 um on a real file. The MAP is
            what fades, not the background: the map is the thing being placed,
            and fading the background instead would leave the categorical
            colours at full strength over a washed-out image. */}
        {background?.image && (
          <img
            src={`data:image/png;base64,${background.image}`}
            alt=""
            aria-hidden="true"
            draggable={false}
            style={{
              position: 'absolute', inset: 0,
              width: '100%', height: '100%', objectFit: 'contain',
              borderRadius: 4, display: 'block', pointerEvents: 'none',
              userSelect: 'none',
            }}
          />
        )}
        <img
          src={`data:image/png;base64,${shownImage}`}
          alt={t('phaseMap.canvasAltText')}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
          onClick={handleClick}
          draggable={false}
          title={isPolygon
            ? t('phaseMap.canvasPolygonTooltip')
            : t('phaseMap.canvasRectTooltip')}
          style={{
            width: '100%', height: '100%', objectFit: 'contain',
            borderRadius: 4, cursor: 'crosshair',
            opacity: background?.image ? background.opacity : 1,
            // Phase maps are categorical — interpolation across class
            // boundaries would render garbage colours.
            imageRendering: 'pixelated',
            display: 'block',
            userSelect: 'none',
          }}
        />
        {/* What the last "Try it" claimed, laid over the map. A count
            cannot tell you whether you caught the right pixels - 202 px of
            matrix and 202 px of particle read identically - and this is the
            same answer as a picture, before anything is committed. Pinned
            at 0.75 rather than full: it has to read as an ANSWER ABOUT the
            map, not as the map. */}
        {claimOverlay && (
          <img
            src={`data:image/png;base64,${claimOverlay}`}
            alt=""
            aria-hidden="true"
            draggable={false}
            style={{
              position: 'absolute', inset: 0,
              width: '100%', height: '100%', objectFit: 'contain',
              imageRendering: 'pixelated', opacity: 0.75,
              pointerEvents: 'none', userSelect: 'none', display: 'block',
            }}
          />
        )}
        {isWand && wand?.mask && (
          <WandOverlay mask={wand.mask} shape={wand.shape} seed={wand.seed} />
        )}
        {(liveRect || (isPolygon && polygonVertices.length > 0)) && (
          <svg
            viewBox={`0 0 ${nCols} ${nRows}`}
            preserveAspectRatio="xMidYMid meet"
            style={{
              position: 'absolute', inset: 0, pointerEvents: 'none',
              width: '100%', height: '100%',
            }}
          >
            {liveRect && !isPolygon && (
              /* Drawn as a dark casing under a white dashed line ("marching
                 ants") instead of one tinted stroke. A single cyan outline
                 was invisible on this very map: the phase colours are a
                 golden-ratio HSV walk, so a fixed accent colour lands on a
                 near-match sooner or later — cyan on the teal Al-Fe-Mn-Si
                 region had almost no contrast. Black-under-white reads on
                 any fill. The old width also collapsed to ~1 screen px,
                 because non-scaling-stroke makes strokeWidth a SCREEN
                 length while the value was computed in viewBox units. */
              <>
                <rect
                  x={liveRect.x} y={liveRect.y}
                  width={liveRect.w} height={liveRect.h}
                  fill="rgba(255, 255, 255, 0.16)"
                  stroke="rgba(0, 0, 0, 0.85)"
                  strokeWidth={4}
                  vectorEffect="non-scaling-stroke"
                />
                <rect
                  x={liveRect.x} y={liveRect.y}
                  width={liveRect.w} height={liveRect.h}
                  fill="none"
                  stroke="#ffffff"
                  strokeWidth={2}
                  strokeDasharray="6 4"
                  vectorEffect="non-scaling-stroke"
                />
              </>
            )}
            {isPolygon && polygonVertices.length > 0 && (
              <>
                {/* Filled polygon (only when closed) gives a preview of
                    what the assign will paint. While the polygon is
                    still open we show only the polyline so the user
                    can tell open vs. closed at a glance. */}
                {/* Same black-under-white treatment as the rectangle, for
                    the same reason: readable over any phase colour. */}
                {polygonVertices.length >= 3 && (
                  <>
                    <polygon
                      points={polygonVertices.map(([x, y]) => `${x + 0.5},${y + 0.5}`).join(' ')}
                      fill="rgba(255, 255, 255, 0.16)"
                      stroke="rgba(0, 0, 0, 0.85)"
                      strokeWidth={4}
                      vectorEffect="non-scaling-stroke"
                    />
                    <polygon
                      points={polygonVertices.map(([x, y]) => `${x + 0.5},${y + 0.5}`).join(' ')}
                      fill="none"
                      stroke="#ffffff"
                      strokeWidth={2}
                      strokeDasharray="6 4"
                      vectorEffect="non-scaling-stroke"
                    />
                  </>
                )}
                {polygonVertices.length < 3 && (
                  <>
                    <polyline
                      points={polygonVertices.map(([x, y]) => `${x + 0.5},${y + 0.5}`).join(' ')}
                      fill="none"
                      stroke="rgba(0, 0, 0, 0.85)"
                      strokeWidth={4}
                      vectorEffect="non-scaling-stroke"
                    />
                    <polyline
                      points={polygonVertices.map(([x, y]) => `${x + 0.5},${y + 0.5}`).join(' ')}
                      fill="none"
                      stroke="#ffffff"
                      strokeWidth={2}
                      strokeDasharray="6 4"
                      vectorEffect="non-scaling-stroke"
                    />
                  </>
                )}
                {polygonVertices.map(([x, y], i) => (
                  <circle
                    key={`v${i}`}
                    cx={x + 0.5} cy={y + 0.5}
                    r={Math.max(1.2, Math.min(nCols, nRows) / 90)}
                    fill="#ffffff"
                    stroke="rgba(0, 0, 0, 0.85)"
                    strokeWidth={2}
                    vectorEffect="non-scaling-stroke"
                  />
                ))}
              </>
            )}
          </svg>
        )}
      </div>
      </div>
      {readout}
    </div>
  );
}


/**
 * The smoothing box, said in either unit, plus what the last run resolved it
 * to.
 *
 * Module level, never nested inside `PhaseMapControls`: a component defined
 * during render is a NEW type on every render, so React unmounts and remounts
 * it — which drops the pointer capture on any control inside and kills a drag
 * after the first mouse move. That exact bug shipped here once already.
 *
 * The px slider itself stays where it always was (`RegionPanel`); this row
 * owns the unit and the um value, so there is never a second px control
 * disagreeing with the first. When um wins, the resolved px is written back
 * into `scale`, so both readouts quote the same box.
 */
export function SmoothingScale({
  unit, setUnit, scaleUm, setScaleUm, scale, report, t,
}) {
  const isUm = unit === 'um';
  const src = report?.scale_source || null;
  const box = report?.scale_box_um || null;
  // Both edges, because the box is square in PIXELS and need not be square
  // in microns — a scan with a different x and y step has a rectangular
  // smoothing box, and quoting one edge for it would be a wrong number.
  const anisotropic = box && box.x != null && box.y != null
    && Math.abs(Number(box.x) - Number(box.y)) > 0.005;

  let resolved = null;
  if (src === 'um' || src === 'pixels') {
    if (report?.scale_px_used != null && box?.x != null) {
      resolved = anisotropic
        ? t('phaseMap.scaleResolvedXY', {
            px: report.scale_px_used, x: um1(box.x), y: um1(box.y) })
        : t('phaseMap.scaleResolved', {
            px: report.scale_px_used, um: um1(box.x) });
    } else if (report?.scale_px_used != null) {
      // No step size in the file, and none was asked for: the pixel count is
      // the whole truth available, and saying "= ? um" would invent one.
      resolved = t('phaseMap.scaleResolvedPxOnly', { px: report.scale_px_used });
    }
  }

  return (
    <div
      data-smoothing-scale
      style={{ display: 'flex', flexDirection: 'column', gap: 3 }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
        {/* A plain span, not `Label`: `Label` renders only `children` and
            `style`, so a `title` handed to it is silently dropped — and the
            tooltip is where the 2.5-vs-3.75 um explanation lives. */}
        <span style={{ fontSize: '9pt', color: C.textSecondary }}
              title={t('phaseMap.scaleUnitTooltip')}>
          {t('phaseMap.scaleUnit')}
        </span>
        <div style={{ display: 'flex', flexShrink: 0 }} role="group"
             aria-label={t('phaseMap.scaleUnit')}>
          {[['px', t('phaseMap.scaleUnitPx')], ['um', t('phaseMap.scaleUnitUm')]].map(
            ([id, label]) => (
              <button
                key={id}
                type="button"
                onClick={() => setUnit(id)}
                aria-pressed={unit === id}
                title={t('phaseMap.scaleUnitTooltip')}
                style={{
                  padding: '2px 10px', fontSize: '8.5pt', cursor: 'pointer',
                  border: `1px solid ${alpha(C.purple, unit === id ? 70 : 25)}`,
                  background: unit === id ? alpha(C.purple, 30) : 'transparent',
                  color: unit === id ? C.text : C.textSecondary,
                  borderRadius: id === 'px' ? '3px 0 0 3px' : '0 3px 3px 0',
                }}
              >
                {label}
              </button>
            ))}
        </div>
        {isUm && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, minWidth: 0 }}>
            <input
              type="number" min={0} step={0.25}
              value={scaleUm ?? ''}
              placeholder={t('phaseMap.scaleUmPlaceholder')}
              aria-label={t('phaseMap.scaleUmLabel')}
              title={t('phaseMap.scaleUnitTooltip')}
              onChange={(e) => setScaleUm(
                e.target.value === '' ? null : Number(e.target.value))}
              style={{
                width: 62, minWidth: 0, fontSize: '8.5pt', padding: '1px 4px',
                background: 'transparent', color: C.text,
                border: `1px solid ${alpha(C.purple, 25)}`, borderRadius: 3,
              }}
            />
            <Label secondary small>{t('phaseMap.scaleUnitUm')}</Label>
          </div>
        )}
        {!isUm && scale != null && (
          <Label secondary small>{t('phaseMap.scalePxValue', { px: scale })}</Label>
        )}
      </div>

      {resolved && (
        <div data-smoothing-resolved
             style={{ fontSize: '8.5pt', color: C.textSecondary }}>
          {resolved}
        </div>
      )}

      {/* A physical width that quietly became a pixel count is the failure
          this whole field exists to prevent. Say it, in the panel, in orange
          — and keep the backend's own sentence as the fallback. */}
      {src === 'pixels_no_step' && (
        <div
          role="alert" data-smoothing-no-step
          style={{ fontSize: '8.5pt', color: C.orange || '#f0b429' }}
        >
          {t('phaseMap.scaleNoStep', {
            um: um1(report?.scale_um_requested),
            px: report?.scale_px_used ?? '?',
            // Only when there is prose to fall back TO. An empty
            // `defaultValue` would turn a missing translation into a blank
            // line, which is the one outcome worse than English.
            ...(report?.scale_note ? { defaultValue: report.scale_note } : {}),
          })}
        </div>
      )}

      {src === 'not_applicable' && (
        <div data-smoothing-na
             style={{ fontSize: '8.5pt', color: C.textSecondary }}>
          {t('phaseMap.scaleNotApplicable')}
        </div>
      )}
    </div>
  );
}


/**
 * Classify warnings whose consequence is a WRONG NAME on the map.
 *
 * The scorer drops carbon and oxygen before it compares anything
 * (`_CHEM_IGNORE` in `crystal_hint_phase_fit`). On a metal that is a
 * reasonable simplification; on an oxide or a carbide it is the whole
 * question. Al2O3 is scored on its aluminium alone and is therefore
 * indistinguishable from Al metal — the map will confidently name a metal
 * where the oxide is, with a high score, and nothing on screen used to say
 * so. These three are not caveats about precision, they are "the answer may
 * be the wrong substance", which is why they get `role="alert"`.
 *
 * A code NOT in this set still renders — with the backend's own prose and a
 * quieter `role="status"` — so a warning added on the backend reaches the
 * user before this build knows its name. Silence is the one outcome that
 * would put us back where we started.
 */
export const WRONG_ANSWER_WARNINGS = new Set([
  'scoring_ignores_c_and_o',
  'indistinguishable_after_excluding_c_and_o',
  'phase_has_no_scoreable_chemistry',
]);

/**
 * Interpolation values for one warning, read from its `detail`.
 *
 * The parameter is `n`, never `count`: i18next treats `count` as a plural
 * selector and would then look for `key_one`/`key_other` keys that do not
 * exist, silently falling back past the translation we wrote.
 *
 * Only the fields the messages actually use are lifted out. A `detail` that
 * is missing or shaped differently yields fewer parameters, and the
 * translation still renders — with a literal placeholder at worst, never a
 * crash on a page the user reached by pressing Classify.
 */
export function warningParams(w) {
  const d = (w && w.detail) || {};
  const p = {};
  if (Array.isArray(d.elements) && d.elements.length) {
    p.elements = d.elements.join(' + ');
  }
  if (Number.isFinite(Number(d.mean_at_pct))) {
    p.mean = Number(d.mean_at_pct).toFixed(1);
  }
  if (Array.isArray(d.phases) && d.phases.length) {
    p.n = d.phases.length;
    p.phases = d.phases.slice(0, 4).join(', ')
      + (d.phases.length > 4 ? ' +' + (d.phases.length - 4) : '');
  }
  if (Array.isArray(d.pairs) && d.pairs.length) {
    p.n = d.pairs.length;
    p.pairs = d.pairs.slice(0, 3).map((x) => `${x?.a} / ${x?.b}`).join('; ')
      + (d.pairs.length > 3 ? ' +' + (d.pairs.length - 3) : '');
  }
  return p;
}

/**
 * The warnings a classify run returned, under the result summary.
 *
 * Deliberately NOT a toast. Two testers have now said the same thing about
 * other things on this page: a message that disappears after six seconds is
 * not a warning. This one sits beside the numbers it qualifies for as long as
 * the map does, and goes away only when the run that produced it does.
 *
 * Module level, like `SmoothingScale`: a component declared inside another
 * component is a NEW component type on every render, so React unmounts and
 * remounts the whole subtree — which in this repo has already cost a slider
 * its pointer capture mid-drag.
 *
 * Renders nothing at all for an empty list. An empty box that appears after
 * every clean run trains the eye to skip the place the real warning will
 * appear.
 */
export function ClassifyWarnings({ warnings, t }) {
  const list = (Array.isArray(warnings) ? warnings : [])
    // A warning with neither a code to translate nor prose to fall back on
    // has nothing to say; rendering it would print a bare i18n key.
    .filter((w) => w && (w.code || w.message));
  if (list.length === 0) return null;

  const ORANGE = C.orange || '#f0b429';
  return (
    <div
      data-classify-warnings
      style={{ display: 'flex', flexDirection: 'column', gap: 4 }}
    >
      <div style={{ fontSize: '8.5pt', fontWeight: 600, color: ORANGE }}>
        {t('phaseMap.warnings.title')}
      </div>
      {list.map((w, i) => {
        const code = String(w.code || '');
        const severe = WRONG_ANSWER_WARNINGS.has(code);
        // The backend ships English prose beside every code. It is the
        // fallback for a code this build has no translation for — and it is
        // only passed when it exists, because an empty `defaultValue` turns a
        // missing translation into a blank line, which is worse than English.
        const prose = typeof w.message === 'string' && w.message.trim()
          ? w.message : '';
        return (
          <div
            key={code || i}
            data-classify-warning={code || 'unknown'}
            role={severe ? 'alert' : 'status'}
            style={{
              fontSize: '8.5pt', lineHeight: 1.35,
              color: severe ? ORANGE : C.textSecondary,
              background: severe ? alpha(ORANGE, 8) : 'transparent',
              border: `1px solid ${alpha(severe ? ORANGE : C.border, severe ? 30 : 40)}`,
              borderRadius: 4, padding: '4px 6px',
            }}
          >
            {t(`phaseMap.warnings.${code}`, {
              ...warningParams(w),
              ...(prose ? { defaultValue: prose } : {}),
            })}
          </div>
        );
      })}
    </div>
  );
}

/**
 * The hand-edit counters, in the order a reviewer reads them.
 *
 * Mirrors `_OP_COUNTER` in `phase_map_store.py`. A counter the backend adds
 * later is simply not shown here rather than shown unlabelled.
 */
export const HAND_EDIT_KEYS = Object.freeze([
  'regions_named', 'merges', 'splits', 'paints',
  'grows', 'edge_snaps', 'phase_replacements',
]);

/**
 * What was done to this map by hand — the question a sign-off has to answer.
 *
 * `n_locked` (pixels painted) was the only trace any of this left, and five
 * of the six ways to put a thumb on a phase map do not lock a pixel. So the
 * number on screen could not tell "merged nothing" from "merged forty times
 * until it looked right". The backend now records all of them; this is the
 * only place in the app they are visible.
 *
 * Three distinct states, and conflating any two of them would be a false
 * claim in somebody's provenance:
 *   - not tracked   — a map restored from a sidecar older than the edit log.
 *                     Its counts are `null`. "0" here would be an assertion
 *                     we cannot back.
 *   - tracked, zero — a measurement, and the one a reviewer wants: this map
 *                     is exactly as it was classified.
 *   - tracked, some — the counts, plus the caveat about renumbered ids when
 *                     a boundary moved.
 */
export function HandEditSummary({ handEdits, t }) {
  if (!handEdits || typeof handEdits !== 'object') return null;

  if (!handEdits.tracked) {
    return (
      <div data-hand-edits data-hand-edits-tracked="false"
           style={{ fontSize: '8pt', color: C.textSecondary }}>
        {t('phaseMap.handEdits.untracked')}
      </div>
    );
  }

  const counts = handEdits.counts || {};
  const parts = HAND_EDIT_KEYS
    .filter((k) => Number(counts[k]) > 0)
    .map((k) => t(`phaseMap.handEdits.${k}`, { n: Number(counts[k]) }));

  return (
    <div data-hand-edits data-hand-edits-tracked="true"
         style={{ fontSize: '8pt', color: C.textSecondary }}>
      {parts.length === 0
        ? t('phaseMap.handEdits.none')
        : t('phaseMap.handEdits.some', { list: parts.join(' · ') })}
      {/* Region-derived particle ids are ordered by (region_id, -n_px,
          centroid), so a merge pops an id and renumbers everything above it.
          An id burned onto a figure before that edit no longer points at the
          same particle — which is exactly what a reproducibility sign-off is
          for. */}
      {handEdits.region_grid_edited === true && (
        <div data-hand-edits-regrid role="status"
             style={{ color: C.orange || '#f0b429', marginTop: 2 }}>
          {t('phaseMap.handEdits.regionGridEdited')}
        </div>
      )}
    </div>
  );
}


/** What "no preset has been applied to this scan" looks like. */
export const NO_APPLIED_PRESET = Object.freeze({ name: '', compatibility: null });

/**
 * Stamp a compatibility report with the file it was checked against, and when.
 *
 * A report that says `ok: true` and nothing else is authoritative-looking and
 * unfalsifiable: nothing in it, or in the `provenance.json` it lands in, says
 * WHICH scan passed. Apply a preset to scan A, switch to scan B, classify and
 * export, and the record carries a clean bill of health for a file that was
 * never checked - a false "compatible" stamp, which is worse than none.
 *
 * Two halves, and this is the second. The first is that the attestation is
 * dropped outright when the file changes; this one makes a stale one
 * self-evident if it ever survives by another route.
 */
export function stampCompatibility(report, filePath, now = new Date()) {
  if (!report) return null;
  return {
    ...report,
    checked_file: filePath || null,
    checked_at: (now instanceof Date ? now : new Date(now)).toISOString(),
  };
}

/** PhaseMapControls — controls + legend + region painting in the right rail. */
export function PhaseMapControls({ handle }) {
  const { t } = useTranslation(['eds', 'collections']);
  const {
    phaseMap, setPhaseMap, loading, error,
    // `tolerance` is deliberately NOT taken here: the slider is gone and the
    // hook keeps sending the recorded value. Reading it in the controls would
    // leave a dead control one edit away from coming back.
    minScore, setMinScore,
    mode, setMode, nClusters, setNClusters,
    cifPhases, selectedPhaseKeys, setSelectedPhaseKeys, refreshCifPhases,
    collections, activeName, newOutsideCollection,
    selectedPhaseIndex, setSelectedPhaseIndex,
    replaceFrom, setReplaceFrom,
    region, setRegion,
    assignBusy,
    paintMode, setPaintMode,
    // Without this the first render in wand mode threw ReferenceError and the
    // ErrorBoundary replaced the entire EDS page — one click was enough.
    wand,
    polygonVertices,
    handleAutoClassify, handleClearMap,
    handNamedCount = 0, pendingReclassify, confirmReclassify, cancelReclassify,
    handleAssignRegion, handleAssignPixel, handleUndo, handleReplacePhase,
    handleClosePolygon, handleCancelPolygon,
    handleSendToIndexing,
    colorOverrides, setPhaseColor, resetPhaseColor,
    mapView, setMapView,
    scale, scaleUnit, setScaleUnit, scaleUm, setScaleUm, scaleReport,
    // Default so a handle assembled by a test (or an older caller) renders
    // the panel rather than throwing on `.length` of undefined.
    classifyWarnings = [],
    filePath,
  } = handle;

  // The preset last applied, plus the report it was applied under. Both go
  // into the export request so `provenance.json` can say which recipe was
  // used and whether it was forced over a refusal.
  const [appliedPreset, setAppliedPreset] = useState(NO_APPLIED_PRESET);
  const [exportOpen, setExportOpen] = useState(false);

  // Every applied report records WHICH file it was applied to, and is stamped
  // with the file it was checked against and the moment it was checked.
  const notePresetApplied = useCallback(({ name, compatibility } = {}) => {
    setAppliedPreset({
      name: name || '',
      file: filePath || '',
      compatibility: stampCompatibility(compatibility, filePath),
    });
  }, [filePath]);

  // A compatibility report describes ONE scan, so it is DERIVED against the
  // file currently loaded rather than cleared by an effect. Nothing used to
  // reset it at all: "apply to A (passes), switch to B, classify, export"
  // wrote `compatibility: {ok: true}` into B's provenance for a check B never
  // had, and nothing in the folder revealed it. Deriving beats an effect for
  // the same reason it beats a listener - there is no render in which the
  // stale record is still live, and no second copy to forget.
  const applied = (appliedPreset.file || '') === (filePath || '')
    ? appliedPreset : NO_APPLIED_PRESET;

  const hasMap = !!phaseMap?.loaded;
  const summary = phaseMap?.summary || [];
  const realPhases = summary.filter(s => !s.is_unclassified);
  const unclassifiedRow = summary.find(s => s.is_unclassified);
  const clusters = phaseMap?.clusters || [];
  const isWandMode = paintMode === 'wand';
  // The region view only exists when there ARE regions. A per-pixel run
  // has no groups and an old map predates them; without this the toggle is
  // hidden AND the phase legend is hidden, leaving no legend at all.
  const showRegions = mapView === 'regions'
    && (phaseMap?.regions || []).length > 0;
  // The legend doubles as the phase PICKER, and `summary` deliberately omits
  // phases with zero pixels. That made the one phase a manual correction is
  // usually FOR — "this particle is beta-AlFeSi, the classifier missed it" —
  // impossible to select. `all_phases` carries every candidate.
  const allPhases = phaseMap?.all_phases || [];
  const unplacedPhases = allPhases.filter(p => p.n_pixels === 0);

  const togglePhase = (key) => {
    const next = new Set(selectedPhaseKeys);
    if (next.has(key)) next.delete(key); else next.add(key);
    setSelectedPhaseKeys(next);
  };

  // Counts for the phase-picker header, shown only while a collection is
  // active. Computed with the SAME `cifNarrowSelection` used to seed the
  // selection in the hook above, so the number on screen can never disagree
  // with what actually gets sent to `/auto-classify`.
  const collectionInfo = activeName
    ? cifNarrowSelection(cifPhases, activeKeySet(collections, activeName))
    : null;
  // Collection members with no matching entry in the CIF library at all —
  // same concept `collections:counts.missingFromLibrary` already names
  // elsewhere in this app, not a per-method availability gap (EDS chemistry
  // matching needs nothing but the CIF itself, unlike the Phase Tester's
  // `.sht` requirement).
  const missingFromLibrary = collectionInfo
    ? collectionInfo.inCollection - collectionInfo.usableHere : 0;
  // The escape hatch: a collection filter with no one-click way back out is
  // the exact defect this feature must avoid. Restores every phase, not just
  // the collection's — same target set as the "All" button above. Counts as
  // a hand edit too (goes through the public `setSelectedPhaseKeys`), so a
  // later collection switch must not narrow it back down under the user.
  const showAllCifPhases = () => setSelectedPhaseKeys(new Set(cifPhases.map(p => p.key)));

  return (
    <GroupBox title={t('phaseMap.title')}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {/* --- The recipe, as a portable file. Above the controls it fills,
                because applying one changes every one of them. --- */}
        <PresetBar handle={handle} onApplied={notePresetApplied} />

        {/* --- Grouping mode. The rail is a fixed 280 px, so this row wraps
                rather than overflowing it — an earlier version pushed the
                cluster-count box 22 px past the edge. --- */}
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
          <Label secondary small title={t('phaseMap.modeTooltip')}>
            {t('phaseMap.grouping')}
          </Label>
          <div style={{ display: 'flex', gap: 0, flexShrink: 0 }} role="group" aria-label={t('phaseMap.grouping')}>
            {[['cluster', t('phaseMap.modeCluster')], ['pixel', t('phaseMap.modePixel')]].map(([id, label]) => (
              <button
                key={id}
                type="button"
                onClick={() => setMode(id)}
                aria-pressed={mode === id}
                title={t('phaseMap.modeTooltip')}
                style={{
                  padding: '2px 10px', fontSize: '8.5pt', cursor: 'pointer',
                  border: `1px solid ${alpha(C.purple, mode === id ? 70 : 25)}`,
                  background: mode === id ? alpha(C.purple, 30) : 'transparent',
                  color: mode === id ? C.text : C.textSecondary,
                  borderRadius: id === 'cluster' ? '3px 0 0 3px' : '0 3px 3px 0',
                }}
              >
                {label}
              </button>
            ))}
          </div>
          {mode === 'cluster' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 4, minWidth: 0 }}>
              <Label secondary small title={t('phaseMap.nClustersTooltip')}>
                {t('phaseMap.nClusterCount')}
              </Label>
              <input
                type="number" min={2} max={20}
                value={nClusters ?? ''}
                placeholder={t('phaseMap.nClustersAuto')}
                onChange={(e) => setNClusters(e.target.value === '' ? null : Number(e.target.value))}
                title={t('phaseMap.nClustersTooltip')}
                style={{
                  width: 58, minWidth: 0, flexShrink: 1,
                  fontSize: '8.5pt', padding: '1px 4px',
                  background: 'transparent', color: C.text,
                  border: `1px solid ${alpha(C.purple, 25)}`, borderRadius: 3,
                }}
              />
            </div>
          )}
        </div>

        {/* --- Smoothing, as a length or as pixels. Only in cluster mode:
                per-pixel classification smooths nothing, so a width would be
                a control with no effect. --- */}
        {mode === 'cluster' && (
          <SmoothingScale
            unit={scaleUnit}
            setUnit={setScaleUnit}
            scaleUm={scaleUm}
            setScaleUm={setScaleUm}
            scale={scale}
            report={scaleReport}
            t={t}
          />
        )}

        {/* --- Min score.
                The Tolerance slider that used to sit beside it is GONE. It
                was inert: `AutoClassifyRequest` accepts it and the sidecar
                stores it, but `auto_classify_pixels` never reads it and
                `eds_clustering.py` references it zero times. Saying so in a
                grey label under a live-looking slider was honest and still
                wrong - a control that cannot change the result has no claim
                on the panel. The value is still sent and still recorded in
                provenance.json as `tolerance_effective: false`.

                What replaces it is one sentence explaining the control that
                DOES decide what stays grey. It lived in the manual, which is
                not where the user is standing. --- */}
        <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
          <div style={{ flex: 1 }}>
            <Label secondary small style={{ display: 'block', marginBottom: 2 }} title={t('phaseMap.minScoreTooltip')}>
              {t('phaseMap.minScore', { value: minScore.toFixed(2) })}
            </Label>
            <input
              type="range" min={0} max={1} step={0.05}
              value={minScore}
              onChange={(e) => setMinScore(Number(e.target.value))}
              style={{ width: '100%', accentColor: C.purple }}
              title={t('hoverTips.minScoreSlider')}
            />
            <div data-min-score-note
                 style={{ fontSize: '9pt', color: C.textSecondary, lineHeight: 1.35 }}>
              {t('phaseMap.minScoreNote')}
            </div>
          </div>
        </div>

        {/* One line, for the user who read an older manual and would
            otherwise hunt the panel for a slider that is not coming back. */}
        {mode === 'pixel' && (
          <div data-tolerance-removed
               style={{ fontSize: '9pt', color: C.textSecondary, lineHeight: 1.35 }}>
            {t('phaseMap.toleranceRemoved')}
          </div>
        )}

        {/* --- Which phases take part --- */}
        {cifPhases.length > 0 && (
          <details
            style={{ marginTop: 2 }}
            data-testid="phase-selection"
            onToggle={(e) => { if (e.currentTarget.open) refreshCifPhases(); }}
          >
            <summary style={{
              cursor: 'pointer', fontSize: '8.5pt', color: C.textSecondary,
              userSelect: 'none',
            }} title={t('phaseMap.phaseSelectionTooltip')}>
              {t('phaseMap.phaseSelection')} ({selectedPhaseKeys.size}/{cifPhases.length})
            </summary>
            <div style={{ display: 'flex', gap: 6, margin: '4px 0' }}>
              <button
                type="button"
                onClick={() => setSelectedPhaseKeys(new Set(cifPhases.map(p => p.key)))}
                style={{ fontSize: '8pt', cursor: 'pointer', background: 'transparent',
                         border: `1px solid ${alpha(C.purple, 25)}`, borderRadius: 3,
                         color: C.textSecondary, padding: '1px 8px' }}
              >{t('phaseMap.selectAll')}</button>
              <button
                type="button"
                onClick={() => setSelectedPhaseKeys(new Set())}
                style={{ fontSize: '8pt', cursor: 'pointer', background: 'transparent',
                         border: `1px solid ${alpha(C.purple, 25)}`, borderRadius: 3,
                         color: C.textSecondary, padding: '1px 8px' }}
              >{t('phaseMap.selectNone')}</button>
            </div>
            {/* Collection-narrowed count + the one-click way back to the whole
                library — only shown while a collection is active, mirroring
                the Phase Tester's own count line (SinglePixelPhaseTestDialog). */}
            {activeName && collectionInfo && (
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                gap: 8, margin: '2px 0 6px', fontSize: '8pt', color: C.textSecondary }}>
                <span>
                  {t('collections:counts.usableHere', {
                    usable: collectionInfo.usableHere, total: collectionInfo.inCollection,
                    collection: activeName,
                  })}
                  {missingFromLibrary > 0 && (
                    <> · {t('collections:counts.missingFromLibrary', { count: missingFromLibrary })}</>
                  )}
                  {newOutsideCollection > 0 && (
                    <> · {t('collections:counts.newOutsideCollection', { count: newOutsideCollection })}</>
                  )}
                </span>
                <button
                  type="button"
                  onClick={showAllCifPhases}
                  style={{ background: 'transparent', border: 'none', color: C.purple, cursor: 'pointer',
                           fontSize: '8pt', padding: 0, textDecoration: 'underline', whiteSpace: 'nowrap' }}
                >
                  {t('collections:counts.showAll')}
                </button>
              </div>
            )}
            <div style={{ maxHeight: 160, overflowY: 'auto', paddingRight: 4 }}>
              {cifPhases.map((p) => (
                <label key={p.key} style={{
                  display: 'flex', alignItems: 'center', gap: 6, fontSize: '8.5pt',
                  padding: '1px 0', cursor: 'pointer', color: C.text,
                }} title={p.formula || p.cif_filename}>
                  <input
                    type="checkbox"
                    checked={selectedPhaseKeys.has(p.key)}
                    onChange={() => togglePhase(p.key)}
                    style={{ accentColor: C.purple }}
                  />
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {p.cif_filename}
                  </span>
                </label>
              ))}
            </div>
            {selectedPhaseKeys.size === 0 && (
              <div style={{ fontSize: '8pt', color: C.orange, marginTop: 2 }}>
                {t('phaseMap.noneSelected')}
              </div>
            )}
          </details>
        )}

        {/* --- Action buttons --- */}
        <div style={{ display: 'flex', gap: 6 }}>
          <Button
            variant="primary"
            onClick={handleAutoClassify}
            disabled={loading}
            style={{ flex: 1 }}
            title={t('phaseMap.autoClassifyTooltip')}
          >
            {loading ? t('phaseMap.classifying') : (hasMap ? t('phaseMap.reclassify') : t('phaseMap.autoClassify'))}
          </Button>
          {hasMap && (
            <Button
              variant="secondary"
              onClick={handleClearMap}
              disabled={loading}
              title={t('phaseMap.clearTooltip')}
            >
              {t('phaseMap.clear')}
            </Button>
          )}
        </div>

        {/* The numbers leave the app here. Disabled without a map, because
            there would be nothing to count. */}
        <Button
          onClick={() => setExportOpen(true)}
          disabled={!hasMap}
          title={hasMap ? t('export.buttonTooltip') : t('export.needMap')}
          style={{ width: '100%' }}
        >
          {t('export.button')}
        </Button>

        {error && (
          <div role="alert" style={{
            fontSize: '9pt', color: C.red,
            background: alpha(C.red, 8), border: `1px solid ${alpha(C.red, 30)}`,
            borderRadius: 4, padding: '4px 8px',
          }}>
            {error}
          </div>
        )}

        {/* --- Stats --- */}
        {hasMap && (
          <div style={{ fontSize: '9pt', color: C.textSecondary }}>
            {t('phaseMap.stats', { classified: phaseMap.n_classified, total: phaseMap.n_classified + phaseMap.n_unclassified })}
            {realPhases.length === 1
              ? t('phaseMap.statsPhasesOne', { count: realPhases.length })
              : t('phaseMap.statsPhasesOther', { count: realPhases.length })}
            {phaseMap.k_used > 0 && ` · ${t('phaseMap.kUsed', { k: phaseMap.k_used })}`}
          </div>
        )}

        {/* --- Library notes ---
                Same chips the suggestion panel shows, and this is where they
                matter most: those compositions were just used to score every
                pixel of the map above. A phase whose stored composition its own
                CIF contradicts produces a map that is confidently wrong, and
                until now the note only existed on the other panel. --- */}
        {hasMap && phaseMap.library_skipped?.length > 0 && (
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
            <SuggestLibrarySkipped skipped={phaseMap.library_skipped} />
          </div>
        )}
        {/* --- What this run does not know ---
            Directly under the numbers it qualifies, because that is where the
            eye lands after pressing Classify, and it stays there. A phase
            named on metal content alone where an oxide is sitting is a wrong
            answer, not a rounding error; the backend has been able to say so
            for a while and nothing on this page was listening. --- */}
        <ClassifyWarnings warnings={classifyWarnings} t={t} />

        {/* --- What a human changed after the classifier had its say ---
            The counts, not the pixels. A reviewer signing a map off has to be
            able to tell "merged nothing" from "merged until it looked
            right", and until now only painted pixels left any trace. --- */}
        {hasMap && <HandEditSummary handEdits={phaseMap?.hand_edits} t={t} />}

        {/* --- How good the numbers under the map are ---
                The same note the pixel table and the region panel carry, and
                it belongs here most of all: these at% are what every candidate
                was scored against. A window the quantification could not price
                is left out and the rest renormalise without it, so the note
                names the element the classifier had to treat as unmeasured. --- */}
        {hasMap && phaseMap.quantification && (
          <QuantificationNote provenance={phaseMap.quantification} />
        )}

        {/* --- Cluster report. The point of clustering is that ambiguity can
                be stated per region, which is impossible per pixel: "this
                group matches A at 0.98 and B at 0.95, chemistry cannot
                separate them". --- */}
        {clusters.length > 0 && (
          <details style={{ marginTop: 2 }}>
            <summary style={{
              cursor: 'pointer', fontSize: '8.5pt', color: C.textSecondary,
              userSelect: 'none',
            }}>
              {t('phaseMap.clusterReport')} ({clusters.length})
            </summary>
            <div style={{
              maxHeight: 240, overflowY: 'auto', marginTop: 4,
              border: `1px solid ${C.border}`, borderRadius: 4, padding: 4,
            }}>
              {clusters.map((c) => (
                <div key={c.cluster_id} style={{
                  padding: '4px 2px',
                  borderBottom: `1px solid ${alpha(C.border, 40)}`,
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 6 }}>
                    <span style={{
                      fontSize: '9pt', fontWeight: 600,
                      color: c.cif_filename ? C.green : C.orange,
                      overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    }}>
                      {c.cif_filename || t('phaseMap.clusterUnmatched')}
                    </span>
                    <span style={{ fontSize: '8.5pt', color: C.textSecondary, flexShrink: 0 }}>
                      {c.percentage}% · {c.score}
                    </span>
                  </div>
                  <div style={{ fontSize: '8pt', color: C.textSecondary, marginTop: 1 }}>
                    {Object.entries(c.mean_at_pct)
                      .map(([el, v]) => `${el} ${v}`).join(' · ')}
                  </div>
                  {c.ambiguous && (
                    <div style={{ fontSize: '8pt', color: C.orange, marginTop: 1 }}
                         title={t('phaseMap.ambiguousTooltip')}>
                      ⚠ {t('phaseMap.ambiguous')}
                      {c.runners_up?.length > 0 && `: ${c.runners_up
                        .map(r => `${r.cif_filename} (${r.score})`).join(', ')}`}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </details>
        )}

        {/* --- Regions vs phases ----------------------------------
            Regions come first because that is the order of the work: the
            data says which pixels belong together, the user says what they
            are. The phase view is the same map seen through those names. */}
        {hasMap && (phaseMap?.regions || []).length > 0 && (
          <div style={{ display: 'flex', gap: 4, marginTop: 4 }}>
            {[
              { id: 'regions', label: t('regions.viewRegions'),
                tip: t('regions.viewRegionsTooltip') },
              { id: 'phases', label: t('regions.viewPhases'),
                tip: t('regions.viewPhasesTooltip') },
            ].map((v) => {
              const active = mapView === v.id;
              return (
                <button
                  key={v.id}
                  type="button"
                  onClick={() => setMapView(v.id)}
                  title={v.tip}
                  style={{
                    flex: 1, fontSize: '8.5pt', padding: '3px 6px',
                    borderRadius: 3, cursor: 'pointer',
                    background: active ? alpha(C.cyan, 22) : 'transparent',
                    color: active ? C.text : C.textSecondary,
                    border: `1px solid ${active ? alpha(C.cyan, 50) : C.border}`,
                  }}
                >
                  {v.label}
                </button>
              );
            })}
          </div>
        )}

        {hasMap && showRegions && <RegionPanel handle={handle} />}

        {/* --- Legend --- */}
        {hasMap && !showRegions && realPhases.length > 0 && (
          <div style={{
            display: 'flex', flexDirection: 'column', gap: 2,
            maxHeight: 220, overflowY: 'auto',
            border: `1px solid ${C.border}`, borderRadius: 4, padding: 4,
          }}>
            {/* Without this the swatch reads as decoration and nobody
                discovers the picker. */}
            <div style={{ fontSize: '7.5pt', color: C.textSecondary, padding: '0 2px 2px' }}>
              {t('phaseMap.colorHint')}
            </div>
            {realPhases.map((s) => {
              const active = selectedPhaseIndex === s.phase_index;
              return (
                <div
                  key={s.phase_index}
                  role="button"
                  tabIndex={0}
                  onClick={() => setSelectedPhaseIndex(active ? null : s.phase_index)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      setSelectedPhaseIndex(active ? null : s.phase_index);
                    }
                  }}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 6,
                    padding: '3px 6px', borderRadius: 3,
                    background: active ? alpha(C.cyan, 18) : 'transparent',
                    border: active ? `1px solid ${alpha(C.cyan, 50)}` : '1px solid transparent',
                    cursor: 'pointer',
                  }}
                  title={t('phaseMap.legendTooltip', { cif: s.cif_filename, formula: s.formula, pct: s.percentage })}
                >
                  {/* Swatch doubles as the colour picker, exactly as on the
                      EBSD phase map: click to pick, right-click to reset.
                      Keyed on the phase name WITHOUT the .cif, which is what
                      the EBSD page uses, so one choice covers both pages. */}
                  <span
                    style={{
                      width: 14, height: 14, borderRadius: 3, flexShrink: 0,
                      display: 'inline-block', position: 'relative',
                      background: s.color || '#3c3c3c',
                      border: phaseNameKey(s.cif_filename) in colorOverrides
                        ? `1px solid ${C.text}` : `1px solid ${C.border}`,
                    }}
                    onClick={(e) => e.stopPropagation()}
                    onContextMenu={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      resetPhaseColor(phaseNameKey(s.cif_filename));
                    }}
                    title={t('phaseMap.colorTooltip', { name: s.cif_filename })}
                  >
                    <input
                      type="color"
                      value={s.color || '#3c3c3c'}
                      onChange={(e) => setPhaseColor(phaseNameKey(s.cif_filename), e.target.value)}
                      style={{
                        position: 'absolute', inset: 0, width: '100%', height: '100%',
                        opacity: 0, cursor: 'pointer', border: 'none', padding: 0,
                      }}
                      aria-label={t('phaseMap.colorAria', { name: s.cif_filename })}
                    />
                  </span>
                  <span style={{
                    flex: 1, minWidth: 0, fontSize: '9pt',
                    color: active ? C.text : C.textSecondary,
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  }}>
                    {s.cif_filename}
                  </span>
                  <span style={{ fontSize: '8pt', color: C.textSecondary, flexShrink: 0 }}>
                    {s.percentage}%
                  </span>
                </div>
              );
            })}
            {unclassifiedRow && unclassifiedRow.n_pixels > 0 && (
              <div style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '3px 6px', borderRadius: 3, opacity: 0.8,
              }}>
                <span style={{
                  width: 14, height: 14, borderRadius: 3, flexShrink: 0,
                  background: '#3c3c3c', border: `1px solid ${C.border}`,
                }} />
                <span style={{ flex: 1, fontSize: '9pt', color: C.textSecondary, fontStyle: 'italic' }}>
                  {t('phaseMap.unclassified')}
                </span>
                <span style={{ fontSize: '8pt', color: C.textSecondary }}>
                  {unclassifiedRow.percentage}%
                </span>
              </div>
            )}
          </div>
        )}

        {/* --- Hand tools. Collapsed while grouping, because they belong to
            the phase view and stacking both toolsets is what made this rail
            unreadable. --- */}
        {hasMap && showRegions && (
          <Label secondary small style={{ display: 'block', marginTop: 6 }}>
            {t('regions.handToolsHint')}
          </Label>
        )}
        <details open={!showRegions} style={{ marginTop: 2 }}>
          <summary style={{ cursor: 'pointer', fontSize: '8.5pt',
                            color: C.textSecondary, userSelect: 'none' }}
                   title={t('regions.handToolsTooltip')}>
            {t('regions.handTools')}
          </summary>
        {/* --- Region painting (M4) --- */}
        {hasMap && (
          <div style={{
            display: 'flex', flexDirection: 'column', gap: 4,
            border: `1px solid ${C.border}`, borderRadius: 4, padding: 6,
            background: alpha(C.bgSecondary, 50),
          }}>
            {/* Mode toggle: rectangle (drag) vs polygon (click vertices) */}
        {/* Undo — one level, and it redoes. Every mutation snapshots first, so
            this covers a re-classify as well as a paint. */}
        {hasMap && phaseMap.undo_label && (
          <Button
            variant="secondary"
            onClick={handleUndo}
            style={{ width: '100%', marginTop: 2 }}
            title={t('phaseMap.undoTooltip', { what: phaseMap.undo_label })}
          >
            {t('phaseMap.undo', { what: phaseMap.undo_label })}
          </Button>
        )}
        {/* Replace one phase with another, map-wide. The wand is a feature-scale
            instrument; this is the map-scale one. */}
        {hasMap && realPhases.length > 0 && (
          <details style={{ marginTop: 2 }}>
            <summary style={{ cursor: 'pointer', fontSize: '8.5pt',
                              color: C.textSecondary, userSelect: 'none' }}
                     title={t('phaseMap.replaceTooltip')}>
              {t('phaseMap.replaceTitle')}
            </summary>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4,
                          flexWrap: 'wrap' }}>
              <select
                value={replaceFrom ?? ''}
                onChange={(e) => setReplaceFrom(e.target.value === '' ? null : Number(e.target.value))}
                aria-label={t('phaseMap.replaceFrom')}
                style={{ flex: 1, minWidth: 0, fontSize: '8.5pt', padding: '2px 4px',
                         background: 'transparent', color: C.text,
                         border: `1px solid ${alpha(C.purple, 25)}`, borderRadius: 3 }}
              >
                <option value="">{t('phaseMap.replaceFrom')}</option>
                {realPhases.map((s) => (
                  <option key={s.phase_index} value={s.phase_index}>
                    {s.cif_filename} ({s.percentage}%)
                  </option>
                ))}
                <option value={-1}>{t('phaseMap.unclassified')}</option>
              </select>
              <span style={{ fontSize: '9pt', color: C.textSecondary }}>{'→'}</span>
              <Button
                variant="warning"
                disabled={replaceFrom == null || selectedPhaseIndex == null
                          || replaceFrom === selectedPhaseIndex || assignBusy}
                onClick={() => handleReplacePhase(replaceFrom, selectedPhaseIndex)}
                title={selectedPhaseIndex == null
                  ? t('phaseMap.replaceNeedTarget') : t('phaseMap.replaceGo')}
              >
                {t('phaseMap.replaceButton')}
              </Button>
            </div>
            <Label secondary small style={{ display: 'block', marginTop: 3 }}>
              {t('phaseMap.replaceHint')}
            </Label>
          </details>
        )}
        {/* --- Seeded selection (wand) --- */}
        {isWandMode && wand.seed && wand.growth.length > 0 && (
          <div style={{
            marginTop: 4, padding: '7px 8px', borderRadius: 4,
            background: alpha(C.cyan, 8),
            border: `1px solid ${alpha(C.cyan, 25)}`,
          }}>
            {/* A phase is rarely one blob. "Connected" grows from the seed;
                "everywhere" takes every pixel like it across the whole map — a
                chemistry-space selection rather than a spatial one. */}
            <div style={{ display: 'flex', marginBottom: 3 }} role="group" aria-label={t('wand.scope')}>
              {[['connected', t('wand.scopeConnected')], ['all', t('wand.scopeAll')]].map(([id, label], i) => (
                <button
                  key={id} type="button"
                  onClick={() => wand.setScope(id)}
                  aria-pressed={wand.scope === id}
                  title={id === 'all' ? t('wand.scopeAllTooltip') : t('wand.scopeConnectedTooltip')}
                  style={{
                    padding: '1px 8px', fontSize: '8pt', cursor: 'pointer',
                    border: `1px solid ${alpha(C.cyan, wand.scope === id ? 60 : 22)}`,
                    background: wand.scope === id ? alpha(C.cyan, 25) : 'transparent',
                    color: wand.scope === id ? C.text : C.textSecondary,
                    borderRadius: i === 0 ? '3px 0 0 3px' : '0 3px 3px 0',
                  }}
                >{label}</button>
              ))}
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 6 }}>
              <Label secondary small>{t('wand.selection')}</Label>
              <span style={{ fontSize: '9pt', fontWeight: 700, color: C.cyan,
                             fontVariantNumeric: 'tabular-nums' }}>
                {t('wand.pixels', { count: wand.nSelected })}
              </span>
            </div>
            {/* The slider walks a growth curve sampled where pixels actually are.
                Thresholding the chemistry directly has dead bands — 2 % through
                10 % of the range returned an identical selection on real data —
                so the axis is the pixel count instead. The count is also the leak
                detector: a jump from 300 to 48 000 announces itself here. */}
            <input
              type="range" min={0} max={wand.growth.length - 1} step={1}
              value={wand.step}
              onChange={(e) => wand.setStep(Number(e.target.value))}
              onMouseUp={wand.refreshStats}
              onKeyUp={wand.refreshStats}
              aria-label={t('wand.sliderAria')}
              title={t('wand.sliderTooltip')}
              style={{ width: '100%', accentColor: C.cyan, marginTop: 2 }}
            />
            {wand.stats && wand.stats.n_pixels > 0 && (
              <div style={{ fontSize: '8pt', color: C.textSecondary, marginTop: 2 }}>
                {Object.entries(wand.stats.mean_at_pct)
                  .sort((a, b) => b[1] - a[1]).slice(0, 5)
                  .map(([el, v]) => el + ' ' + v).join(' · ')}
              </div>
            )}
            {wand.stats && Object.keys(wand.stats.enrichment || {}).length > 0 && (
              <div style={{ fontSize: '8pt', color: C.textSecondary, marginTop: 1 }}
                   title={t('wand.enrichmentTooltip')}>
                {t('wand.enrichment')}
                {': '}
                {Object.entries(wand.stats.enrichment)
                  .filter(([, v]) => v >= 1.3)
                  .sort((a, b) => b[1] - a[1]).slice(0, 4)
                  .map(([el, v]) => el + ' ' + v + 'x').join(' · ') || t('wand.enrichmentNone')}
              </div>
            )}
            <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
              <Button
                variant="primary"
                disabled={selectedPhaseIndex == null || wand.loading}
                onClick={async () => {
                  const res = await wand.commit(selectedPhaseIndex);
                  if (res) setPhaseMap(res);
                }}
                style={{ flex: 1 }}
                title={selectedPhaseIndex == null
                  ? t('wand.assignTooltipNoPhase') : t('wand.assignTooltip')}
              >
                {t('wand.assign')}
              </Button>
              <Button variant="secondary" onClick={wand.clear} title={t('wand.cancelTooltip')}>
                {t('wand.cancel')}
              </Button>
            </div>
            {wand.error && (
              <div role="alert" style={{ fontSize: '8pt', color: C.red, marginTop: 3 }}>
                {wand.error}
              </div>
            )}
          </div>
        )}
        {isWandMode && !wand.seed && (
          <Label secondary small style={{ marginTop: 4 }}>{t('wand.hint')}</Label>
        )}
        {/* Phases the classifier placed nowhere. The legend cannot show
            them (zero pixels) but they are exactly what a manual
            correction usually needs to assign. */}
        {hasMap && unplacedPhases.length > 0 && (
          <details style={{ marginTop: 2 }}>
            <summary style={{ cursor: 'pointer', fontSize: '8.5pt',
                              color: C.textSecondary, userSelect: 'none' }}>
              {t('phaseMap.unplaced', { count: unplacedPhases.length })}
            </summary>
            <div style={{ maxHeight: 150, overflowY: 'auto', marginTop: 4 }}>
              {unplacedPhases.map((s) => {
                const active = selectedPhaseIndex === s.phase_index;
                return (
                  <div
                    key={s.phase_index}
                    onClick={() => setSelectedPhaseIndex(active ? null : s.phase_index)}
                    title={s.formula || s.cif_filename}
                    style={{
                      display: 'flex', alignItems: 'center', gap: 6,
                      padding: '3px 5px', borderRadius: 3, cursor: 'pointer',
                      fontSize: '8.5pt',
                      background: active ? alpha(C.purple, 25) : 'transparent',
                      border: `1px solid ${active ? alpha(C.purple, 60) : 'transparent'}`,
                    }}
                  >
                    <span style={{ width: 10, height: 10, borderRadius: 2,
                                   background: s.color, flexShrink: 0 }} />
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis',
                                   whiteSpace: 'nowrap' }}>{s.cif_filename}</span>
                  </div>
                );
              })}
            </div>
          </details>
        )}
            {/* Only the paint-mode label and its toggle belong on one
                row; everything above stacks. */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <Label secondary small>{t('phaseMap.paintMode')}</Label>
              <div style={{
                display: 'flex', border: `1px solid ${C.border}`, borderRadius: 4,
                overflow: 'hidden', flex: 1,
              }}>
                {[
                  { id: 'rectangle', label: t('phaseMap.rectangle'), tip: t('hoverTips.paintModeRectangle') },
                  { id: 'polygon', label: t('phaseMap.polygon'), tip: t('hoverTips.paintModePolygon') },
                  { id: 'wand', label: t('wand.mode'), tip: t('wand.modeTooltip') },
                ].map((m) => {
                  const active = paintMode === m.id;
                  return (
                    <button
                      key={m.id}
                      onClick={() => setPaintMode(m.id)}
                      aria-pressed={active}
                      title={m.tip}
                      style={{
                        flex: 1, padding: '4px 0',
                        background: active ? C.purple : 'transparent',
                        color: active ? C.bg : C.textSecondary,
                        fontWeight: active ? 700 : 400,
                        fontSize: '9pt', border: 'none',
                        cursor: 'pointer', transition: 'all 0.15s',
                      }}
                    >
                      {m.label}
                    </button>
                  );
                })}
              </div>
            </div>

            {paintMode === 'rectangle' ? (
              <>
                <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                  <NumberInput
                    value={region.rowStart}
                    onChange={(e) => setRegion(r => ({ ...r, rowStart: Number(e.target.value) }))}
                    min={0}
                    style={{ width: 56 }}
                    title={t('phaseMap.rowStartTooltip')}
                  />
                  <span style={{ alignSelf: 'center', fontSize: '8pt', color: C.textSecondary }}>–</span>
                  <NumberInput
                    value={region.rowEnd}
                    onChange={(e) => setRegion(r => ({ ...r, rowEnd: Number(e.target.value) }))}
                    min={0}
                    style={{ width: 56 }}
                    title={t('phaseMap.rowEndTooltip')}
                  />
                  <NumberInput
                    value={region.colStart}
                    onChange={(e) => setRegion(r => ({ ...r, colStart: Number(e.target.value) }))}
                    min={0}
                    style={{ width: 56 }}
                    title={t('phaseMap.colStartTooltip')}
                  />
                  <span style={{ alignSelf: 'center', fontSize: '8pt', color: C.textSecondary }}>–</span>
                  <NumberInput
                    value={region.colEnd}
                    onChange={(e) => setRegion(r => ({ ...r, colEnd: Number(e.target.value) }))}
                    min={0}
                    style={{ width: 56 }}
                    title={t('phaseMap.colEndTooltip')}
                  />
                </div>
                <div style={{ display: 'flex', gap: 4 }}>
                  <Button
                    variant="warning"
                    onClick={() => handleAssignRegion(selectedPhaseIndex ?? -1)}
                    disabled={assignBusy || selectedPhaseIndex == null}
                    style={{ flex: 1 }}
                    title={selectedPhaseIndex == null
                      ? t('phaseMap.assignRegionTooltipNoPhase')
                      : t('phaseMap.assignRegionTooltip', { rowStart: region.rowStart, rowEnd: region.rowEnd, colStart: region.colStart, colEnd: region.colEnd, phase: selectedPhaseIndex })}
                  >
                    {t('phaseMap.assignRegion')}
                  </Button>
                  <Button
                    variant="secondary"
                    onClick={() => handleAssignRegion(-1)}
                    disabled={assignBusy}
                    title={t('phaseMap.clearRegionTooltip')}
                  >
                    {t('phaseMap.clearRegion')}
                  </Button>
                </div>
              </>
            ) : (
              <>
                <Label secondary small>
                  {t('phaseMap.polygonHint', {
                    count: polygonVertices.length,
                    suffix: polygonVertices.length >= 3
                      ? t('phaseMap.polygonHintReady')
                      : t('phaseMap.polygonHintNeedMore'),
                  })}
                </Label>
                <div style={{ display: 'flex', gap: 4 }}>
                  <Button
                    variant="warning"
                    onClick={() => handleClosePolygon(selectedPhaseIndex ?? -1)}
                    disabled={assignBusy || selectedPhaseIndex == null || polygonVertices.length < 3}
                    style={{ flex: 1 }}
                    title={selectedPhaseIndex == null
                      ? t('phaseMap.closeAssignTooltipNoPhase')
                      : polygonVertices.length < 3
                        ? t('phaseMap.closeAssignTooltipNeedMore')
                        : t('phaseMap.closeAssignTooltip', { count: polygonVertices.length, phase: selectedPhaseIndex })}
                  >
                    {t('phaseMap.closeAssign')}
                  </Button>
                  <Button
                    variant="secondary"
                    onClick={() => handleClosePolygon(-1)}
                    disabled={assignBusy || polygonVertices.length < 3}
                    title={t('phaseMap.polygonClearTooltip')}
                  >
                    {t('phaseMap.clearRegion')}
                  </Button>
                  <Button
                    variant="secondary"
                    onClick={handleCancelPolygon}
                    disabled={polygonVertices.length === 0}
                    title={t('phaseMap.polygonCancelTooltip')}
                  >
                    ×
                  </Button>
                </div>
              </>
            )}
          </div>
        )}

        </details>

        {/* --- Send to indexing (M5 entry point) --- */}
        {hasMap && realPhases.length > 0 && (
          <Button
            variant="primary"
            onClick={handleSendToIndexing}
            style={{ width: '100%' }}
            title={t('phaseMap.sendToIndexingTooltip')}
          >
            {t('phaseMap.sendToIndexing')}
          </Button>
        )}
        <ExportDialog
          open={exportOpen}
          onClose={() => setExportOpen(false)}
          presetName={applied.name}
          compatibility={applied.compatibility}
        />

        {/* Anything that can destroy work is a confirmation, not a tooltip.
            The count is what makes it actionable: "6 names" is a decision,
            "some names" is a shrug. */}
        <ConfirmDialog
          open={!!pendingReclassify}
          title={t('phaseMap.confirmReclassify.title')}
          message={`${handNamedCount === 1
            ? t('phaseMap.confirmReclassify.bodyOne', { count: handNamedCount })
            : t('phaseMap.confirmReclassify.bodyOther', { count: handNamedCount })} ${
            t('phaseMap.confirmReclassify.kept')}`}
          confirmLabel={t('phaseMap.confirmReclassify.confirm')}
          cancelLabel={t('phaseMap.confirmReclassify.cancel')}
          variant="danger"
          onConfirm={() => confirmReclassify?.()}
          onCancel={() => cancelReclassify?.()}
        />
      </div>
    </GroupBox>
  );
}
