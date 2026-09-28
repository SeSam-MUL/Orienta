import { useState, useEffect, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { indexApi, ebsdApi, edsApi } from '../../services/api';
import useDataStore from '../../stores/useDataStore';
import useCollectionStore from '../../stores/useCollectionStore';
import { activeKeySet, activeKeySignature, narrowSelection } from '../PhaseCollections/collectionFilter';
import { omissions } from './phaseTestOmissions';
import LinkedPatternImage from '../PatternMatch/LinkedPatternImage';
import { useLinkedPatternMarkers } from '../PatternMatch/useLinkedPatternMarkers';
import PatternExportDialog from '../PatternMatch/PatternExportDialog';
import { useZoomViews, SYNC_SINGLE } from '../EDS/hooks/useZoomViews';
import { viewToTransform, zoomedRect, isZoomed } from '../EDS/zoomView';
import useWheelZoom from '../common/useWheelZoom';

const C = {
  bg: '#282a36', bgSecondary: '#21222c', border: '#44475a', text: '#f8f8f2',
  textSecondary: '#6272a4', accent: '#bd93f9', green: '#50fa7b',
  orange: '#ffb86c', red: '#ff5555',
};
const rColor = (r) => (r == null ? C.textSecondary : r >= 0.3 ? C.green : r >= 0.15 ? C.orange : C.red);
const img = (b64) => (b64 ? `data:image/png;base64,${b64}` : null);
// Distinct tints for stacked EDS element overlays on the scan picker.
const EDS_COLORS = ['#ff5555', '#50fa7b', '#8be9fd', '#ffb86c', '#bd93f9', '#f1fa8c', '#ff79c6', '#62d6e8'];
const edsSym = (el) => (typeof el === 'string' ? el : (el?.element || el?.symbol || el?.name || ''));

const SCAN_VIEW = 'scan';
const MEASURED_VIEW = 'measured';
const COMPARE_VIEW = 'compare';

export default function SinglePixelPhaseTestDialog({ open, onClose, currentMethod = 'hough', onUsePhase }) {
  const { t } = useTranslation(['indexing', 'phasemap', 'collections']);
  // One view per picture, not one shared view: the scan map, the measured
  // pattern and the comparison panels show different things at different
  // native sizes, so zooming one says nothing about the others. The three
  // comparison panels are the exception and share a view — they are the same
  // detector frame three ways, and comparing them means looking at one spot.
  const zoom = useZoomViews(SYNC_SINGLE);
  const currentIndex = useDataStore((s) => s.currentIndex) || 0;
  const gridShape = useDataStore((s) => s.gridShape) || [0, 0];
  const setPosition = useDataStore((s) => s.setPosition);
  const fileLoaded = useDataStore((s) => s.ebsdLoaded || s.isFileOpen);
  const filePath = useDataStore((s) => s.filePath);
  // The active phase collection, if any — narrows which phases get
  // pre-selected below. `data.state` is `{}` until the store's first load
  // resolves, so `?.active` (not `.active`) is required here: reading past
  // an undefined `state` would throw, not just read as "no collection".
  const collections = useCollectionStore((s) => s.data.collections);
  const activeName = useCollectionStore((s) => s.data.state?.active || null);

  const [nRows, nCols] = gridShape;

  const [pixelIndex, setPixelIndex] = useState(currentIndex);
  // Simulation bandwidth (SHT) — drives the actual compute. Higher = sharper
  // simulated bands (better discrimination), slower. Switching this builds a
  // new per-(phaseset, geometry, bandwidth) backend (cold on first use).
  const [bandwidth, setBandwidth] = useState(128);
  const [edsMode, setEdsMode] = useState('filter');
  // Dynamic-background-remove the MEASURED pattern before comparison. The
  // simulated dynamical pattern is already background-free, so removing the
  // measured pattern's background makes the NCC correlate band-vs-band (better
  // R + discrimination) and reveals the Kikuchi bands the phase-ID relies on.
  // Default ON — without it the comparison degrades and the phase is misranked.
  const [bgRemove, setBgRemove] = useState(true);
  // Geometry PC for the simulated pattern. Default OFF = stable map-mean PC, so
  // clicking through pixels reuses the cached backend (no ~1 s/phase rebuild per
  // click). ON = exact per-pixel PC (more accurate geometry, rebuilds each click).
  const [usePixelPc, setUsePixelPc] = useState(false);
  const [aperture, setAperture] = useState('circular');
  const [apertureRadius, setApertureRadius] = useState(1.0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [selected, setSelected] = useState(null);
  const [showExcluded, setShowExcluded] = useState(false);

  // Linked crosshair + numbered markers shared across the three comparison
  // panels, plus the publication export composer (same as the Pattern Match
  // dialogs in IndexingPage / PhaseMapPage).
  const markerCtl = useLinkedPatternMarkers();
  const [exportOpen, setExportOpen] = useState(false);

  // Transient confirmation shown under the "Use for indexing" buttons after a
  // phase is added, so the click gives visible feedback now that the dialog no
  // longer closes on add (the user keeps it open to collect phases across
  // multiple pixels). ``{ formula, method, status }``; auto-clears after ~2.6 s.
  const [justUsed, setJustUsed] = useState(null);
  const justUsedTimerRef = useRef(null);
  const markUsed = useCallback((formula, methodLabel, status) => {
    setJustUsed({ formula, method: methodLabel, status });
    if (justUsedTimerRef.current) clearTimeout(justUsedTimerRef.current);
    justUsedTimerRef.current = setTimeout(() => setJustUsed(null), 2600);
  }, []);

  // Phase picker: the library phases we *can* test, plus the user's
  // current selection. Defaults to ALL phases selected once they load.
  const [phases, setPhases] = useState([]);
  const [selectedKeys, setSelectedKeys] = useState(() => new Set());
  const [showPhases, setShowPhases] = useState(false);
  // The fetch of the testable phase list has three outcomes, but
  // `phases.length === 0` cannot distinguish any of them: still in flight,
  // resolved to an empty list, or rejected. That collapse is exactly the
  // hole this state exists to close — with a collection active, "still
  // loading" and "failed to load" both left `noneSelected` false (it
  // requires `phases.length > 0`) AND `allSelected` true (its own
  // `phases.length === 0` clause), so the button read "Auto-Run all
  // phases" and, uncaught, ran the entire library on click while the
  // toolbar named a specific collection. `phasesLoadState` tracks the
  // fetch itself, independently of what `phases` currently holds, so the
  // collection-aware guard below can tell those three states apart.
  // `retryTick` exists purely to give "Retry" something to bump — it is
  // read only as a dependency of the fetch effect, never for its value.
  const [phasesLoadState, setPhasesLoadState] = useState('idle'); // idle | loading | loaded | error
  const [phasesRetryTick, setPhasesRetryTick] = useState(0);
  const retryPhases = useCallback(() => setPhasesRetryTick((n) => n + 1), []);
  // True while `selectedKeys` is still exactly the collection-derived
  // auto-seed; false once the user hand-edits it (checkbox, select-all/none,
  // or "show all phases"). The collection-follow effect below only ever
  // overwrites the selection while this is still true — App.jsx keeps this
  // dialog mounted-but-hidden rather than unmounting it, so the active
  // collection can change while the dialog sits open in the background, and
  // an untouched selection should follow it back in without ever clobbering
  // a choice the user already made.
  const isAutoSeedRef = useRef(true);

  // Background-job state. ``progress`` mirrors the backend poll response.
  const [progress, setProgress] = useState(null);
  // Poll interval id + a monotonic run id so a stale poll from a previous
  // job can never write into a newer run's state.
  const pollRef = useRef(null);
  const runIdRef = useRef(0);
  const jobIdRef = useRef(null);
  // The id of the LAST COMPLETED run. Unlike jobIdRef (cleared on 'done'),
  // this stays set after a run finishes so /remask can reuse the cached
  // patterns to recompute masked R without a full re-render.
  const lastJobIdRef = useRef(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  }, []);

  // Real scan overview (Band Contrast) — fetched once when the dialog
  // opens with a file loaded. Used as the ScanPicker background. Uses the
  // 'bc' mode so this picker shows the SAME crisp band-contrast map as the
  // Indexing Navigation Map / EBSD Viewer overview (native Oxford BC, with a
  // computed FFT image-quality fallback), instead of a washed-out per-pixel
  // mean-intensity map.
  const [overviewB64, setOverviewB64] = useState(null);
  // Measured pattern at the active pixel — fetched whenever pixelIndex
  // changes so the user SEES the pattern before running the test.
  const [previewB64, setPreviewB64] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  // Monotonic request counter so a fast pixel change discards a stale
  // (slower) preview response instead of flashing the wrong pattern.
  const previewReqRef = useRef(0);
  // Surfaced fetch error for the measured-pattern preview, so a failed fetch
  // shows "preview failed" instead of silently reading as "no pattern".
  const [previewError, setPreviewError] = useState(null);

  // EDS element overlays on the scan picker — the user can toggle MULTIPLE
  // elements on at once (each tinted a distinct colour) to pick pixels by
  // chemistry. ``edsOverlaySel`` is the ordered list of active symbols;
  // ``edsOverlayMaps`` caches each element's fetched (colour-tinted) map.
  const [edsElements, setEdsElements] = useState([]);
  const [edsOverlaySel, setEdsOverlaySel] = useState([]);
  const [edsOverlayMaps, setEdsOverlayMaps] = useState({});
  const [edsOpacity, setEdsOpacity] = useState(0.6);

  // Live per-pixel EDS chemistry for the ACTIVE crosshair pixel — fetched on
  // every pixel change so the EDS readout (and the user's sense of what the
  // chemistry filter will keep/drop) follows the crosshair, instead of being
  // frozen on the pixel of the last completed run. ``null`` = no EDS / loading.
  const [livePixelAtPct, setLivePixelAtPct] = useState(null);
  const chemReqRef = useRef(0);

  useEffect(() => { if (typeof currentIndex === 'number') setPixelIndex(currentIndex); }, [currentIndex]);

  // Fetch the scan overview once when the dialog opens with a file loaded.
  useEffect(() => {
    if (!open || !fileLoaded) return;
    let cancelled = false;
    ebsdApi.overview('bc')
      .then((r) => { if (!cancelled) setOverviewB64(r?.data?.image || null); })
      .catch(() => { /* silently fall back to the empty picker */ });
    return () => { cancelled = true; };
  }, [open, fileLoaded, filePath]);

  // Fetch the testable library phases once when the dialog opens. Default
  // the selection to ALL of them (so the run tests every phase by default).
  useEffect(() => {
    if (!open || !fileLoaded) return;
    let cancelled = false;
    setPhasesLoadState('loading');
    indexApi.phaseTestPhases()
      .then((r) => {
        if (cancelled) return;
        const list = Array.isArray(r?.data) ? r.data : [];
        setPhases(list);
        setPhasesLoadState('loaded');
        // Seed the selection from the active collection, if any — narrowed to
        // the phases this dialog can actually run (each needs an .sht
        // master). `phases` itself STAYS the full list: only `selectedKeys`
        // is narrowed here. Shrinking `phases` instead would make
        // `selectedKeys.size === phases.length` true again for a fully-
        // selected collection, `subsetSelected` would go false, and the run
        // would silently test the WHOLE library — see
        // phaseTestCollection.test.jsx for the pinned regression.
        //
        // Read the collection LIVE from the store here, not the
        // `collections`/`activeName` render-scope selectors: this callback's
        // closure over those was captured when the effect was SCHEDULED, but
        // the promise can resolve after the user switched the active
        // collection elsewhere (the dialog can sit open-but-hidden while
        // that happens — see the collection-follow effect right below). If
        // this used the closed-over values, the seed would silently match
        // whatever collection was active when the request was SENT, not the
        // one active when the phases actually ARRIVED — the collection-
        // follow effect only fires on an `activeName` change, so if that
        // change already happened before this `.then()` runs, nothing would
        // ever correct the mismatch.
        const live = useCollectionStore.getState().data;
        const liveActiveName = live?.state?.active || null;
        const liveCollections = live?.collections || [];
        const colKeys = activeKeySet(liveCollections, liveActiveName);
        const { keys } = narrowSelection(
          list.map((p) => p.key), colKeys, list.map((p) => p.key));
        setSelectedKeys(new Set(keys));
        isAutoSeedRef.current = true;   // fresh seed: nothing hand-edited yet
      })
      .catch(() => {
        if (cancelled) return;
        setPhases([]);
        setPhasesLoadState('error');
      });
    return () => { cancelled = true; };
    // This effect itself fires ONCE per dialog-open/file-load, matching its
    // existing intent ("Default the selection to ALL of them" above) — the
    // `.then()` above no longer even reads the render-scope `collections`/
    // `activeName` (it reads the store live instead, for the reason
    // explained there). A later collection SWITCH while the dialog stays
    // open is handled by the separate effect right below, which only
    // touches the selection while it is still untouched (isAutoSeedRef);
    // "Show all phases" is the manual escape either way. `phasesRetryTick`
    // is in the deps purely so "Retry" (after a failed fetch) re-runs this
    // exact effect — it never appears anywhere else.
  }, [open, fileLoaded, filePath, phasesRetryTick]);

  // Follow the active collection while the dialog sits open — AND follow a
  // membership change to that SAME collection, made from somewhere else
  // while this dialog never closed (Task 10 added the one place that can
  // cause that: the database browser's "move to collection", or the
  // Collection Manager, editing the collection this dialog is already
  // showing). `membersSignature` is a content fingerprint of exactly that
  // collection's effective members (`activeKeySignature`, see
  // `collectionFilter.js`), not `collections` itself — `collections` gets a
  // brand-new array reference from the store on every refresh anywhere in
  // the app (including an unrelated collection's own edit), and depending on
  // it directly would reseed this dialog on every one of those for no
  // reason; the signature only changes when THIS collection's members did.
  //
  // App.jsx keeps every page mounted and merely hides inactive ones with
  // `display: none` rather than unmounting them — `showPhaseTestDialog` is
  // local state untouched by the `isActive` prop, so this dialog (and its
  // `open`/`fileLoaded`/`filePath` deps above) can stay exactly as they were
  // while the user navigates away, flips the toolbar's active collection (or
  // edits it), and navigates back. Without this effect the header would
  // recompute against the NEW `activeName`/membership on every render (it is
  // derived from `phases`/`collections`/`activeName` directly, not cached)
  // while `selectedKeys` stayed pinned to the OLD selection — the count line
  // would then name a collection the checkboxes disagree with.
  //
  // Deliberately separate from the fetch effect above (not merged into it,
  // and not added to its deps) so that effect's own "seeds once" comment
  // stays true; this one seeds again, but ONLY while nothing has been
  // hand-edited since the last seed.
  const membersSignature = activeKeySignature(collections, activeName);
  useEffect(() => {
    if (!isAutoSeedRef.current || phases.length === 0) return;
    const colKeys = activeKeySet(collections, activeName);
    const { keys } = narrowSelection(
      phases.map((p) => p.key), colKeys, phases.map((p) => p.key));
    setSelectedKeys(new Set(keys));
    // still true: re-deriving from the collection is not a hand edit
    isAutoSeedRef.current = true;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeName, membersSignature]);

  // Stop polling + invalidate the run when the dialog closes or unmounts so
  // a late poll response can't setState after close/unmount.
  useEffect(() => {
    if (!open) { runIdRef.current += 1; stopPolling(); }
  }, [open, stopPolling]);
  useEffect(() => () => {
    runIdRef.current += 1; stopPolling();
    if (justUsedTimerRef.current) clearTimeout(justUsedTimerRef.current);
  }, [stopPolling]);

  // Fetch the measured pattern for the active pixel whenever it changes.
  useEffect(() => {
    if (!open || !fileLoaded || nCols <= 0) { setPreviewB64(null); setPreviewError(null); return; }
    const row = Math.floor(pixelIndex / nCols);
    const col = pixelIndex % nCols;
    const myReq = ++previewReqRef.current;
    let cancelled = false;
    setPreviewLoading(true); setPreviewError(null);
    // When BG-removal is on, fetch the preview through the dynamic-BG display
    // filter so the previewed measured pattern matches what the NCC sees.
    const cfg = bgRemove ? { params: { display_filter: 'Dynamic BG' } } : undefined;
    ebsdApi.getPattern(row, col, cfg)
      .then((res) => {
        if (cancelled || myReq !== previewReqRef.current) return;  // stale
        setPreviewB64(res?.data?.image || null);
      })
      .catch((err) => {
        if (cancelled || myReq !== previewReqRef.current) return;
        setPreviewB64(null);
        setPreviewError(err?.response?.data?.detail || err?.message || 'fetch failed');
      })
      .finally(() => {
        if (!cancelled && myReq === previewReqRef.current) setPreviewLoading(false);
      });
    return () => { cancelled = true; };
  }, [open, fileLoaded, pixelIndex, nCols, bgRemove, filePath]);

  // Fetch the live EDS chemistry for the active pixel whenever it changes, so
  // the EDS readout tracks the crosshair (not the last run's pixel).
  useEffect(() => {
    if (!open || !fileLoaded) { setLivePixelAtPct(null); return; }
    const myReq = ++chemReqRef.current;
    let cancelled = false;
    indexApi.phaseTestPixelChemistry(pixelIndex)
      .then((res) => {
        if (cancelled || myReq !== chemReqRef.current) return;  // stale
        setLivePixelAtPct(res?.data?.pixel_at_pct || null);
      })
      .catch(() => {
        if (cancelled || myReq !== chemReqRef.current) return;
        setLivePixelAtPct(null);
      });
    return () => { cancelled = true; };
  }, [open, fileLoaded, pixelIndex, filePath]);

  // Fetch the list of EDS elements once when the dialog opens, to populate the
  // scan-picker overlay selector. Empty list → the overlay control hides.
  useEffect(() => {
    if (!open || !fileLoaded) { setEdsElements([]); return; }
    let cancelled = false;
    edsApi.elements()
      .then((r) => { if (!cancelled) setEdsElements(Array.isArray(r?.data?.elements) ? r.data.elements : []); })
      .catch(() => { if (!cancelled) setEdsElements([]); });
    return () => { cancelled = true; };
  }, [open, fileLoaded, filePath]);

  // Toggle an EDS element overlay on/off. On first activation, fetch its
  // colour-tinted element map (cached in edsOverlayMaps; tint is stable per
  // element so a colour always means the same element).
  const toggleEdsEl = useCallback((sym) => {
    setEdsOverlaySel((sel) => (sel.includes(sym) ? sel.filter((x) => x !== sym) : [...sel, sym]));
    if (!edsOverlayMaps[sym]) {
      const idx = Math.max(0, edsElements.findIndex((e) => edsSym(e) === sym));
      const color = EDS_COLORS[idx % EDS_COLORS.length];
      edsApi.getMap(sym, 'counts', 'hot', color)
        .then((r) => { if (r?.data?.image) setEdsOverlayMaps((m) => ({ ...m, [sym]: r.data.image })); })
        .catch(() => { /* overlay just won't show for this element */ });
    }
  }, [edsOverlayMaps, edsElements]);

  // Bug fix: when the ACTIVE FILE changes, drop everything cached from the
  // previous file (result/preview/overview/phases) so a stale pattern from the
  // old file can never linger; the per-file effects above then refetch fresh.
  useEffect(() => {
    runIdRef.current += 1; stopPolling();
    setResult(null); setSelected(null); setProgress(null); setError(null);
    setPreviewB64(null); setPreviewError(null); setOverviewB64(null);
    setEdsOverlaySel([]); setEdsOverlayMaps({}); setEdsElements([]);
    // Blank to empty here only. The phaseTestPhases fetch effect above also
    // has `filePath` in its deps and fires on the same render, but its
    // reseed happens inside an async `.then()` — so it lands after every
    // effect in this commit has run, this one included, regardless of which
    // is declared first. Collection-aware seeding lives in that ONE place,
    // not here.
    setPhases([]); setSelectedKeys(new Set());
    setPhasesLoadState('idle');   // the fetch effect below flips this to 'loading' again
    isAutoSeedRef.current = true;   // no phases yet: nothing to have hand-edited
    markerCtl.clearMarkers(); markerCtl.setHover(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filePath]);

  // Markers live in the active pixel's detector frame — clear them when the
  // pixel changes (a new measured pattern). They persist across phase changes.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { markerCtl.clearMarkers(); markerCtl.setHover(null); }, [pixelIndex]);

  const previewRow = nCols > 0 ? Math.floor(pixelIndex / nCols) : 0;
  const previewCol = nCols > 0 ? pixelIndex % nCols : 0;

  // A strict subset is selected → pass phase_keys; otherwise omit it so the
  // backend tests every phase (its default).
  const allSelected = phases.length === 0 || selectedKeys.size === phases.length;
  const subsetSelected =
    phases.length > 0 && selectedKeys.size > 0 && selectedKeys.size < phases.length;
  // Library phases exist but the (collection-narrowed) selection is empty.
  // The backend reads an ABSENT phase_keys as "test every phase" — sending
  // `phase_keys: []` is not the fix, omitting it is what triggers the whole
  // library, which is exactly the trap this guards against. "0 selected"
  // must refuse to run, not silently widen to everything. Mirrors
  // PhaseMapPanel.jsx's `noneSelected` (EDS "Suggest Phases" panel) so the
  // two screens agree.
  const noneSelected = phases.length > 0 && selectedKeys.size === 0;
  const running = progress?.status === 'running';

  // The same trap through a different door: `noneSelected` only fires once
  // the phase list has actually loaded (it requires `phases.length > 0`).
  // While a collection is active and the fetch is still in flight — or
  // failed outright — `phases` is `[]`, `noneSelected` is false, AND
  // `allSelected` is true (its own `phases.length === 0` clause), so the
  // button reads "Auto-Run all phases" and a click would run the whole
  // library while the toolbar names a specific collection. Gated on
  // `activeName`: with no collection active, an empty `phases` list
  // legitimately means "the backend will test everything" (the normal,
  // library-loading and no-collections case), and this must not touch that.
  const collectionPhasesBlocked = !!activeName && phasesLoadState !== 'loaded';
  const collectionBlockReason = !collectionPhasesBlocked ? null
    : phasesLoadState === 'error'
      ? t('phaseTest.collectionPhasesFailed', { collection: activeName })
      : t('phaseTest.collectionPhasesLoading', { collection: activeName });

  // Counts for the phase-picker header, shown only while a collection is
  // active. Computed from the SAME narrowSelection used to seed the
  // selection above, so the number on screen can never disagree with what
  // actually gets sent.
  const collectionInfo = activeName
    ? narrowSelection(
        phases.map((p) => p.key), activeKeySet(collections, activeName), phases.map((p) => p.key))
    : null;
  const notUsableHere = collectionInfo ? collectionInfo.inCollection - collectionInfo.usableHere : 0;

  const runAuto = useCallback(async () => {
    // "0 selected" must not run the whole library — the button is already
    // disabled for this, but the check is repeated here so a stray call
    // (e.g. a race between disabling and a queued click) cannot slip
    // through to `phaseTestStart` with an omitted `phase_keys`, which the
    // backend reads as "test everything".
    if (noneSelected) {
      setError(t('phaseTest.noneSelected'));
      return;
    }
    // Same reasoning, the "still loading"/"failed to load" door: refuse
    // rather than let an unresolved collection silently widen to the whole
    // library. See `collectionPhasesBlocked` above for why `noneSelected`
    // alone does not already cover this.
    if (collectionPhasesBlocked) {
      setError(collectionBlockReason);
      return;
    }
    // Invalidate any in-flight job, then start a fresh one.
    const myRun = ++runIdRef.current;
    stopPolling();
    setLoading(true); setError(null); setSelected(null); setProgress(null);

    const params = {
      pixel_index: pixelIndex, eds_weighting: edsMode,
      aperture, aperture_radius: apertureRadius, max_bandwidth: bandwidth,
      bg_remove: bgRemove, use_pixel_pc: usePixelPc,
    };
    if (subsetSelected) params.phase_keys = [...selectedKeys];

    try {
      const res = await indexApi.phaseTestStart(params);
      if (myRun !== runIdRef.current) return;  // superseded while awaiting
      const jobId = res.data.job_id;
      jobIdRef.current = jobId;
      lastJobIdRef.current = jobId;  // kept after 'done' for /remask reuse
      // Show the experimental preview + total immediately (partial result).
      setResult(res.data);
      setProgress({ done: 0, total: res.data.total, current: null, status: 'running' });

      const tick = async () => {
        if (myRun !== runIdRef.current) return;
        try {
          const pr = await indexApi.phaseTestProgress(jobId);
          if (myRun !== runIdRef.current) return;
          const d = pr.data || {};
          setProgress({ done: d.done, total: d.total, current: d.current, status: d.status });
          if (d.status === 'done') {
            stopPolling();
            const full = d.result || {};
            setResult(full);
            setSelected(full.candidates?.[0] || null);
            jobIdRef.current = null;
          } else if (d.status === 'error') {
            stopPolling();
            setError(d.error || t('phaseTest.failed'));
            jobIdRef.current = null;
          } else if (d.status === 'cancelled') {
            stopPolling();
            setError(t('phaseTest.cancelled'));
            jobIdRef.current = null;
          }
        } catch (err) {
          if (myRun !== runIdRef.current) return;
          stopPolling();
          setError(err?.response?.data?.detail || err?.message || t('phaseTest.failed'));
          jobIdRef.current = null;
        }
      };
      // Poll immediately, then every 500 ms (the immediate first poll keeps
      // tests deterministic without fake timers).
      tick();
      pollRef.current = setInterval(tick, 500);
    } catch (err) {
      if (myRun !== runIdRef.current) return;
      setError(err?.response?.data?.detail || err?.message || t('phaseTest.failed'));
      setResult(null); setSelected(null); setProgress(null);
    } finally {
      if (myRun === runIdRef.current) setLoading(false);
    }
  }, [pixelIndex, edsMode, aperture, apertureRadius, bandwidth, bgRemove, usePixelPc, subsetSelected, selectedKeys, noneSelected, collectionPhasesBlocked, collectionBlockReason, stopPolling, t]);

  const cancelRun = useCallback(async () => {
    const jobId = jobIdRef.current;
    if (!jobId) return;
    try { await indexApi.phaseTestCancel(jobId); } catch { /* poll will observe state */ }
  }, []);

  // Mask-only change (toggle or radius) → recompute masked R + diff/simulated
  // PNGs from the LAST run's cached patterns via /remask (cheap, no re-render).
  // Only valid after a run completed (lastJobIdRef + result set) and not mid-run.
  // On a 404 (cache expired) we fall back to a full re-run so the user isn't stuck.
  const remask = useCallback(async (apertureOverride, radiusOverride) => {
    const ap = apertureOverride ?? aperture;
    const rad = radiusOverride ?? apertureRadius;
    const jobId = lastJobIdRef.current;
    if (!jobId || !result || running || loading) return;  // need a completed run
    try {
      const res = await indexApi.phaseTestRemask({ job_id: jobId, aperture: ap, aperture_radius: rad });
      const full = res.data;
      setResult(full);
      // Keep the user's currently-selected phase selected if it still exists,
      // otherwise fall back to the top candidate.
      setSelected((prev) => {
        const keep = prev && full.candidates?.find((c) => c.phase_key === prev.phase_key);
        return keep || full.candidates?.[0] || null;
      });
    } catch (err) {
      if (err?.response?.status === 404) { runAuto(); }  // cache expired → full re-run
      else { setError(err?.response?.data?.detail || err?.message || t('phaseTest.remaskFailed')); }
    }
  }, [aperture, apertureRadius, result, running, loading, runAuto, t]);

  const toggleKey = useCallback((key) => {
    isAutoSeedRef.current = false;   // hand edit: the collection-follow effect must leave this alone now
    setSelectedKeys((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key); else next.add(key);
      return next;
    });
  }, []);

  const toggleAll = useCallback(() => {
    isAutoSeedRef.current = false;   // hand edit, even when the result happens to be "all"
    setSelectedKeys((prev) =>
      prev.size === phases.length ? new Set() : new Set(phases.map((p) => p.key)));
  }, [phases]);

  // The escape hatch: a collection filter with no one-click way back out is
  // the exact defect this feature must avoid. Restores every phase, not just
  // the collection's — same target set as toggleAll's "select all". Counts
  // as a hand edit too: once the user has explicitly asked for everything, a
  // later collection switch must not narrow it back down under them.
  const showAllPhases = useCallback(() => {
    isAutoSeedRef.current = false;
    setSelectedKeys(new Set(phases.map((p) => p.key)));
  }, [phases]);

  if (!open) return null;

  const cands = result?.candidates || [];
  // Two reasons a phase is missing from the table, kept apart. See
  // `phaseTestOmissions.js` for why they must not be merged.
  const { excluded, unsupported } = omissions(result);

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', display: 'flex',
      alignItems: 'center', justifyContent: 'center', zIndex: 999 }} onClick={onClose}>
      <div onClick={(e) => e.stopPropagation()} style={{ background: C.bgSecondary,
        border: `1px solid ${C.border}`, borderRadius: 8, padding: 20, width: 920,
        maxHeight: '92vh', overflow: 'auto' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
          <span style={{ color: C.accent, fontWeight: 700, fontSize: 15 }}>{t('phaseTest.title')}</span>
          <button onClick={onClose} title={t('hoverTips.phaseTestClose')} aria-label={t('hoverTips.phaseTestClose')} style={{ background: 'transparent', border: 'none',
            color: C.textSecondary, cursor: 'pointer', fontSize: 18 }}>&times;</button>
        </div>

        {/* 2-column body: LEFT = picker + pixel input + measured preview,
            RIGHT = EDS/mask/run controls + comparison + ranked list. */}
        <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>

        {/* LEFT column: scan picker + pixel index + measured pattern preview */}
        <div style={{ width: 300, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div style={{ background: C.bg, border: `1px solid ${C.border}`, borderRadius: 6, padding: 10 }}>
            <div style={{ fontSize: 11, color: C.textSecondary, marginBottom: 6 }}>
              {nRows > 0 && nCols > 0
                ? t('phaseTest.clickScan', { rows: nRows, cols: nCols })
                : t('phaseTest.scanUnknown')}
            </div>
            {nRows > 0 && nCols > 0 ? (
              <ScanPicker
              view={zoom.viewFor(SCAN_VIEW)}
              onZoomAt={(f, px, py) => zoom.zoomAtPointer(SCAN_VIEW, f, px, py)}
              onPan={(dx, dy) => zoom.pan(SCAN_VIEW, dx, dy)}
              onResetView={() => zoom.resetOne(SCAN_VIEW)}
                gridShape={gridShape}
                pixelIndex={pixelIndex}
                overviewB64={overviewB64}
                edsOverlays={edsOverlaySel.map((s) => edsOverlayMaps[s]).filter(Boolean)}
                edsOpacity={edsOpacity}
                t={t}
                onPick={(r, c) => {
                  const idx = r * nCols + c;
                  setPixelIndex(idx);
                  // Sync the global store so EBSD Viewer / Phase Map jump to
                  // the same pixel when the user switches tabs.
                  if (setPosition) setPosition(r, c, idx, null);
                }}
              />
            ) : (
              <div style={{ height: 120, display: 'flex', alignItems: 'center',
                justifyContent: 'center', color: C.textSecondary, fontSize: 10, opacity: 0.6 }}>
                {t('phaseTest.loadingOverview')}
              </div>
            )}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 8 }}>
              <span style={{ fontSize: 11, color: C.textSecondary }}>{t('phaseTest.pixelIndex')}</span>
              <input type="number" value={pixelIndex} min={0}
                onChange={(e) => setPixelIndex(parseInt(e.target.value, 10) || 0)}
                title={t('hoverTips.phaseTestPixelIndex')}
                style={{ width: 90, background: C.bg, color: C.text, border: `1px solid ${C.border}`,
                  borderRadius: 4, padding: '3px 8px', fontSize: 12 }} />
            </div>
            {edsElements.length > 0 && (
              <div style={{ marginTop: 8 }}>
                <div style={{ fontSize: 11, color: C.textSecondary, marginBottom: 4 }}>{t('phaseTest.edsOverlay')}</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }} title={t('hoverTips.phaseTestEdsOverlay')}>
                  {edsElements.map((el, i) => {
                    const sym = edsSym(el);
                    const on = edsOverlaySel.includes(sym);
                    const color = EDS_COLORS[i % EDS_COLORS.length];
                    return (
                      <button key={sym} type="button" onClick={() => toggleEdsEl(sym)} aria-pressed={on}
                        style={{ fontSize: 11, padding: '2px 8px', borderRadius: 10, cursor: 'pointer',
                          border: `1px solid ${on ? color : C.border}`,
                          background: on ? color : C.bg, color: on ? '#000' : C.text }}>
                        {sym}
                      </button>
                    );
                  })}
                </div>
                {edsOverlaySel.length > 0 && (
                  <input type="range" min={0} max={1} step={0.05} value={edsOpacity}
                    onChange={(e) => setEdsOpacity(parseFloat(e.target.value))}
                    title={t('hoverTips.phaseTestEdsOpacity')} style={{ width: 100, marginTop: 6 }} />
                )}
              </div>
            )}
          </div>

          {/* Measured pattern preview for the active pixel */}
          <div style={{ background: C.bg, border: `1px solid ${C.border}`, borderRadius: 6, padding: 10 }}>
            <div style={{ fontSize: 11, color: C.text, marginBottom: 6 }}>
              {t('phaseTest.measuredPattern', { row: previewRow, col: previewCol })}
            </div>
            {previewB64 ? (
              <LinkedPatternImage
                src={img(previewB64)}
                alt={t('phaseTest.measuredPattern', { row: previewRow, col: previewCol })}
                view={zoom.viewFor(MEASURED_VIEW)}
                onZoomAt={(f, px, py) => zoom.zoomAtPointer(MEASURED_VIEW, f, px, py)}
                onPan={(dx, dy) => zoom.pan(MEASURED_VIEW, dx, dy)}
                onResetView={() => zoom.resetOne(MEASURED_VIEW)}
                style={{ width: '100%', aspectRatio: '1', background: '#000',
                  border: `1px solid ${C.border}`, borderRadius: 4 }} />
            ) : (
              <div title={previewError || undefined} style={{ width: '100%', aspectRatio: '1', background: C.bgSecondary,
                border: `1px solid ${C.border}`, borderRadius: 4, display: 'flex',
                alignItems: 'center', justifyContent: 'center', color: previewError ? C.orange : C.textSecondary, fontSize: 10 }}>
                {previewLoading ? t('phaseTest.loadingShort')
                  : previewError ? t('phaseTest.previewFailed')
                  : (fileLoaded ? t('phaseTest.noPattern') : t('phaseTest.loadAFile'))}
              </div>
            )}
          </div>
        </div>

        {/* RIGHT column: all existing controls + results */}
        <div style={{ flex: 1, minWidth: 0 }}>

        {/* EDS + mask bar (pixel-index input lives in the left column) */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center',
          background: C.bg, border: `1px solid ${C.border}`, borderRadius: 6, padding: '8px 12px', marginBottom: 12 }}>
          {(() => {
            // Prefer the LIVE chemistry of the active crosshair pixel; fall
            // back to the last run's pixel only until the live fetch lands.
            const edsPct = livePixelAtPct || result?.pixel_at_pct;
            if (!edsPct) return null;
            // Only the elements actually present (>0.5 at%) to keep it readable.
            const shown = Object.entries(edsPct).filter(([, v]) => v >= 0.5);
            if (shown.length === 0) return null;
            return (
              <span style={{ fontSize: 10, color: C.textSecondary }}
                title={t('phaseTest.edsAtTooltip', { row: previewRow, col: previewCol })}>
                {t('phaseTest.edsAt', { row: previewRow, col: previewCol, values: shown.map(([k, v]) => `${k} ${v.toFixed(0)}%`).join(' · ') })}
              </span>
            );
          })()}
          <span style={{ marginLeft: 'auto', fontSize: 10, color: C.textSecondary }}>{t('phaseTest.edsLabel')}</span>
          {['off', 'soft', 'filter'].map((m) => (
            <button key={m} onClick={() => setEdsMode(m)} title={t(`hoverTips.phaseTestEds${m.charAt(0).toUpperCase() + m.slice(1)}`)} style={{ fontSize: 10,
              background: edsMode === m ? C.green : 'transparent', color: edsMode === m ? C.bg : C.textSecondary,
              border: `1px solid ${edsMode === m ? C.green : C.border}`, borderRadius: 3, padding: '2px 8px', cursor: 'pointer' }}>{t(`phaseTest.eds${m.charAt(0).toUpperCase() + m.slice(1)}`)}</button>
          ))}
          {/* Quality = simulation bandwidth (SHT). Drives the actual NCC/R
              compute, not just the preview. */}
          <span style={{ fontSize: 10, color: C.textSecondary }}>{t('phaseTest.quality')}</span>
          <select value={bandwidth} onChange={(e) => setBandwidth(parseInt(e.target.value, 10))}
            title={t('phaseTest.qualityTip')}
            style={{ fontSize: 10, background: C.bg, color: C.text, border: `1px solid ${C.border}`,
              borderRadius: 3, padding: '2px 6px', cursor: 'pointer' }}>
            <option value={88}>{t('phaseTest.qualityFast')}</option>
            <option value={128}>{t('phaseTest.qualityStandard')}</option>
          </select>
          <label style={{ fontSize: 10, color: C.textSecondary, display: 'flex', alignItems: 'center', gap: 4 }}
            title={t('phaseTest.bgRemoveTip')}>
            <input type="checkbox" checked={bgRemove}
              onChange={(e) => setBgRemove(e.target.checked)} /> {t('phaseTest.bgRemove')}
          </label>
          <label style={{ fontSize: 10, color: C.textSecondary, display: 'flex', alignItems: 'center', gap: 4 }}
            title={t('phaseTest.usePixelPcTip')}>
            <input type="checkbox" checked={usePixelPc}
              onChange={(e) => setUsePixelPc(e.target.checked)} /> {t('phaseTest.usePixelPc')}
          </label>
          <label style={{ fontSize: 10, color: C.textSecondary, display: 'flex', alignItems: 'center', gap: 4 }}
            title={t('hoverTips.phaseTestMask')}>
            <input type="checkbox" checked={aperture !== 'full'}
              onChange={(e) => {
                const ap = e.target.checked ? 'circular' : 'full';
                setAperture(ap);
                // Mask-only change → recompute from cache (no full re-render).
                if (result && !running && !loading) remask(ap, undefined);
              }} /> {t('phaseTest.mask')}
          </label>
          <span style={{ fontSize: 10, color: C.textSecondary }}>{t('phaseTest.maskRadius')}</span>
          <input type="range" min={0.3} max={1.0} step={0.05} value={apertureRadius}
            disabled={aperture === 'full'}
            title={t('hoverTips.phaseTestMaskRadius')}
            onChange={(e) => setApertureRadius(parseFloat(e.target.value))}
            onMouseUp={() => { if (result && !running && !loading) remask(undefined, apertureRadius); }}
            onTouchEnd={() => { if (result && !running && !loading) remask(undefined, apertureRadius); }}
            onKeyUp={() => { if (result && !running && !loading) remask(undefined, apertureRadius); }} />
          <span style={{ fontSize: 10, color: C.textSecondary, minWidth: 30 }}>
            {Math.round(apertureRadius * 100)}%
          </span>
        </div>

        {/* Phase picker — collapsible. Hidden if no library phases loaded
            (then the run just tests all, the backend default). */}
        {phases.length > 0 && (
          <div style={{ marginBottom: 10, background: C.bg, border: `1px solid ${C.border}`,
            borderRadius: 6 }}>
            <button onClick={() => setShowPhases((v) => !v)}
              title={t('hoverTips.phaseTestPhasesToggle')}
              style={{ background: 'transparent', border: 'none', color: C.text, cursor: 'pointer',
                fontSize: 11, padding: '8px 12px', width: '100%', textAlign: 'left' }}>
              {showPhases ? '▾' : '▸'} {t('phaseTest.phasesToggle', { selected: selectedKeys.size, total: phases.length })}
            </button>
            {showPhases && (
              <div style={{ borderTop: `1px solid ${C.border}` }}>
                <button onClick={toggleAll}
                  title={t('hoverTips.phaseTestSelectToggle')}
                  style={{ background: 'transparent', border: 'none', color: C.accent, cursor: 'pointer',
                    fontSize: 10, padding: '6px 12px', width: '100%', textAlign: 'left' }}>
                  {selectedKeys.size === phases.length ? t('phaseTest.selectNone') : t('phaseTest.selectAll')}
                </button>
                {activeName && collectionInfo && (
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    gap: 8, padding: '2px 12px 6px', fontSize: 9.5, color: C.textSecondary,
                    borderBottom: `1px solid ${C.border}` }}>
                    <span>
                      {t('collections:counts.usableHere', {
                        usable: collectionInfo.usableHere, total: collectionInfo.inCollection,
                        collection: activeName,
                      })}
                      {notUsableHere > 0 && (
                        <> · {t('collections:counts.notUsableHere', { count: notUsableHere })}</>
                      )}
                    </span>
                    <button onClick={showAllPhases}
                      style={{ background: 'transparent', border: 'none', color: C.accent, cursor: 'pointer',
                        fontSize: 9.5, padding: 0, textDecoration: 'underline', whiteSpace: 'nowrap' }}>
                      {t('collections:counts.showAll')}
                    </button>
                  </div>
                )}
                <div style={{ maxHeight: 180, overflowY: 'auto', padding: '0 12px 8px' }}>
                  {phases.map((p) => (
                    <label key={p.key} title={t('hoverTips.phaseTestPhaseCheckbox', { label: p.label })} style={{ display: 'flex', alignItems: 'center', gap: 6,
                      fontSize: 11, color: C.text, padding: '3px 0', cursor: 'pointer' }}>
                      <input type="checkbox" checked={selectedKeys.has(p.key)}
                        onChange={() => toggleKey(p.key)} />
                      <span>{p.label}</span>
                      {p.space_group && (
                        <span style={{ fontSize: 9, color: C.textSecondary }}>{p.space_group}</span>
                      )}
                    </label>
                  ))}
                </div>
                {noneSelected && (
                  <div style={{ fontSize: 9.5, color: C.orange, padding: '0 12px 8px' }}>
                    {t('phaseTest.noneSelected')}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* The same refusal as `noneSelected` above, reached through the
            OTHER door: a collection is active but the phase list hasn't
            resolved yet (or failed), so the checkbox list above doesn't even
            exist to show its own "select at least one phase" line. This is
            that line's counterpart for that case — same box, same inline
            styling — plus a retry when the fetch itself is what failed. */}
        {phases.length === 0 && collectionPhasesBlocked && (
          <div style={{ marginBottom: 10, background: C.bg, border: `1px solid ${C.border}`,
            borderRadius: 6, padding: '8px 12px', fontSize: 9.5, color: C.orange }}>
            {collectionBlockReason}
            {phasesLoadState === 'error' && (
              <button onClick={retryPhases}
                title={t('hoverTips.phaseTestPhasesRetry')}
                style={{ background: 'transparent', border: 'none', color: C.accent, cursor: 'pointer',
                  fontSize: 9.5, padding: 0, marginLeft: 6, textDecoration: 'underline' }}>
                {t('phaseTest.retry')}
              </button>
            )}
          </div>
        )}

        <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 12 }}>
          <button onClick={runAuto}
            disabled={!fileLoaded || running || loading || noneSelected || collectionPhasesBlocked}
            title={noneSelected ? t('phaseTest.noneSelected')
              : collectionPhasesBlocked ? collectionBlockReason
              : t('hoverTips.phaseTestRun')}
            style={{ background: C.accent, color: '#fff', border: 'none', borderRadius: 4,
              padding: '8px 16px', fontSize: 12, fontWeight: 600, cursor: 'pointer',
              opacity: (!fileLoaded || running || loading || noneSelected || collectionPhasesBlocked) ? 0.5 : 1 }}>
            {(running || loading) ? t('phaseTest.testing')
              : allSelected ? t('phaseTest.autoRunAll') : t('phaseTest.runSelected', { count: selectedKeys.size })}
          </button>
          {running && (
            <button onClick={cancelRun}
              title={t('hoverTips.phaseTestCancel')}
              style={{ background: 'transparent', color: C.red, border: `1px solid ${C.red}`,
                borderRadius: 4, padding: '8px 14px', fontSize: 12, fontWeight: 600, cursor: 'pointer' }}>
              {t('phaseTest.cancel')}
            </button>
          )}
        </div>

        {/* Live progress bar while the background job runs. During the cold
            backend build (done===0) the per-phase count is meaningless, so show
            the backend's "Building phase models…" message + an indeterminate
            pulsing bar instead of a frozen "Testing 0/N". */}
        {running && ((progress?.done ?? 0) === 0 ? (
          <div style={{ marginBottom: 12 }}>
            <div style={{ height: 7, background: '#44475a', borderRadius: 3, overflow: 'hidden' }}>
              <div className="sppt-indeterminate" style={{
                width: '35%', height: '100%', background: C.accent }} />
            </div>
            <div style={{ fontSize: 11, color: C.text, marginTop: 4 }}>
              ⏳ {progress?.current || t('phaseTest.preparing')}
            </div>
            <div style={{ fontSize: 10, color: C.textSecondary, marginTop: 2 }}>
              {t('phaseTest.buildingHint')}
            </div>
            <style>{`@keyframes sppt-indet{0%{margin-left:-35%}100%{margin-left:100%}}
              .sppt-indeterminate{animation:sppt-indet 1.2s ease-in-out infinite}`}</style>
          </div>
        ) : (
          <div style={{ marginBottom: 12 }}>
            <div style={{ height: 7, background: '#44475a', borderRadius: 3, overflow: 'hidden' }}>
              <div style={{
                width: `${progress?.total ? Math.round(((progress.done || 0) / progress.total) * 100) : 0}%`,
                height: '100%', background: C.accent, transition: 'width 0.2s' }} />
            </div>
            <div style={{ fontSize: 10, color: C.textSecondary, marginTop: 4 }}>
              {t('phaseTest.testingProgress', { done: progress?.done ?? 0, total: progress?.total ?? 0 })}
              {progress?.current ? ` — ${progress.current}` : ''}…
            </div>
          </div>
        ))}

        {error && <div style={{ color: C.red, fontSize: 11, marginBottom: 10 }}>{t('phaseTest.errorPrefix', { error })}</div>}

        {/* Comparison panel for the selected phase */}
        {selected && (
          <div style={{ marginBottom: 12 }}>
            <div style={{ display: 'flex', gap: 10 }}>
              {[[t('phaseTest.measured'), result?.experimental_png], [t('phaseTest.simulated', { formula: selected.display_formula || selected.formula }), selected.simulated_png],
                [t('phaseTest.nccDifference'), selected.ncc_diff_png]].map(([label, b64]) => (
                <div key={label} style={{ flex: 1, textAlign: 'center' }}>
                  <div style={{ fontSize: 10, color: C.text, marginBottom: 4 }}>{label}</div>
                  {b64 ? (
                    <LinkedPatternImage src={img(b64)} alt={label}
                      markers={markerCtl.markers} hover={markerCtl.hover}
                      onHover={markerCtl.setHover} onClick={markerCtl.handlePanelClick}
                      view={zoom.viewFor(COMPARE_VIEW)}
                      onZoomAt={(f, px, py) => zoom.zoomAtPointer(COMPARE_VIEW, f, px, py)}
                      onPan={(dx, dy) => zoom.pan(COMPARE_VIEW, dx, dy)}
                      onResetView={() => zoom.resetOne(COMPARE_VIEW)}
                      style={{ width: '100%', aspectRatio: '1', background: '#000',
                        border: `1px solid ${C.border}`, borderRadius: 4 }} />
                  ) : <div style={{ aspectRatio: '1', background: C.bg, borderRadius: 4 }} />}
                </div>
              ))}
            </div>
            <div style={{ marginTop: 6, display: 'flex', gap: 10, alignItems: 'center',
              justifyContent: 'center', flexWrap: 'wrap' }}>
              <span style={{ fontSize: 19, fontWeight: 700, color: rColor(selected.r_score) }}>
                {t('phaseTest.rScore', { value: selected.r_score?.toFixed(3) ?? '—' })}
              </span>
              {selected.orientation_source === 'hough' && (
                <span
                  style={{ fontSize: 10, color: '#ffb86c', border: '1px solid #ffb86c44',
                    borderRadius: 3, padding: '2px 6px', cursor: 'help' }}
                  title={t('phaseTest.orientationHoughTip')}
                >
                  ⬡ {t('phaseTest.orientationFromHough')}
                </span>
              )}
              {selected.chemistry_fit != null && (
                <span style={{ fontSize: 11, color: C.accent }}>
                  {t('phaseTest.chemistry', { value: selected.chemistry_fit.toFixed(2) })}
                </span>
              )}
              {markerCtl.markers.length > 0 && (
                <button onClick={markerCtl.clearMarkers}
                  title={t('hoverTips.phaseTestClearMarkers')}
                  style={{ fontSize: 10, background: C.bg, color: C.text, border: `1px solid ${C.border}`,
                    borderRadius: 3, padding: '3px 8px', cursor: 'pointer' }}>
                  {t('phaseTest.clearMarkers', { count: markerCtl.markers.length })}
                </button>
              )}
              <button onClick={() => setExportOpen(true)} title={t('phaseTest.exportTip')}
                style={{ fontSize: 10, fontWeight: 600, background: C.green, color: C.bg, border: 'none',
                  borderRadius: 3, padding: '3px 10px', cursor: 'pointer' }}>
                {t('phaseTest.export')}
              </button>
            </div>

            {/* Use this phase for indexing — pick the method; disabled methods
                have no file for this phase (✗). Adds the phase and switches the
                indexing method, but keeps this dialog OPEN so the user can click
                the next pixel and add another phase (close it with ×). */}
            {onUsePhase && (
              <>
              <div style={{ marginTop: 6, display: 'flex', gap: 6, alignItems: 'center', justifyContent: 'center', flexWrap: 'wrap' }}>
                <span style={{ fontSize: 9, color: '#6272a4' }}>{t('phaseTest.useForIndexing')}:</span>
                {[
                  { m: 'hough', path: selected.cif_path, label: 'Hough' },
                  { m: 'spherical', path: selected.sht_path, label: 'Spherical' },
                  { m: 'dictionary', path: selected.master_h5_path, label: 'Dictionary' },
                ].map(({ m, path, label }) => (
                  <button key={m} disabled={!path}
                    onClick={() => {
                      const status = onUsePhase(selected, m);
                      // Only confirm if the click actually did something (path
                      // present ⇒ button enabled ⇒ non-null status).
                      if (status) markUsed(selected.display_formula || selected.formula, label, status);
                    }}
                    title={path ? t('phaseTest.useForIndexingTip', { method: label }) : t('phaseTest.methodUnavailable', { method: label })}
                    style={{ fontSize: 9, padding: '3px 8px', borderRadius: 3, cursor: path ? 'pointer' : 'not-allowed',
                      background: path ? (m === currentMethod ? C.green : C.bg) : '#2a2a3a',
                      color: path ? (m === currentMethod ? C.bg : C.text) : '#6272a4',
                      border: `1px solid ${path ? (m === currentMethod ? C.green : C.border) : '#3a3a4a'}` }}>
                    {path ? '✓' : '✗'} {label}
                  </button>
                ))}
              </div>
              {justUsed && (
                <div role="status" aria-live="polite" style={{ marginTop: 5, textAlign: 'center', fontSize: 10,
                  color: justUsed.status === 'exists' ? C.orange : C.green }}>
                  {justUsed.status === 'exists'
                    ? t('phaseTest.alreadyInList', { formula: justUsed.formula, method: justUsed.method })
                    : t('phaseTest.addedToList', { formula: justUsed.formula, method: justUsed.method })}
                </div>
              )}
              </>
            )}
          </div>
        )}

        {/* Ranked list — own scroll area so clicking through the phases never
            pushes the measured/simulated/NCC patterns above out of view. */}
        <div style={{ maxHeight: 340, overflowY: 'auto', paddingRight: 4 }}>
        {cands.map((c) => (
          <div key={c.phase_key} onClick={() => setSelected(c)}
            style={{ display: 'grid', gridTemplateColumns: '130px 1fr 1fr', gap: 10, alignItems: 'center',
              background: C.bgSecondary, cursor: 'pointer', padding: '7px 10px', marginBottom: 5,
              border: `1px solid ${selected?.phase_key === c.phase_key ? C.green : C.border}`, borderRadius: 5 }}>
            <span style={{ fontSize: 11, color: c.rank === 1 ? C.green : C.text, fontWeight: c.rank === 1 ? 700 : 400 }}>
              {c.rank === 1 ? '🏆 ' : ''}<span>{c.display_formula || c.formula}</span>
            </span>
            <Bar value={c.r_score} label={c.r_score?.toFixed(3) ?? '—'} color={rColor(c.r_score)} />
            <Bar value={c.chemistry_fit} label={c.chemistry_fit?.toFixed(2) ?? '—'}
              color={c.chemistry_fit == null ? C.textSecondary
                : c.chemistry_fit >= 0.6 ? C.green : c.chemistry_fit >= 0.35 ? C.orange : C.red} />
          </div>
        ))}
        </div>

        {unsupported.length > 0 && (
          <div style={{ marginTop: 8 }}>
            <div style={{ fontSize: 10, color: C.textSecondary,
                          border: `1px dashed ${C.border}`, borderRadius: 5,
                          padding: '6px 10px' }}
                 data-testid="phase-test-unsupported">
              {t('phaseTest.unsupportedCount', { count: unsupported.length })}
              {unsupported.map((u) => (
                <div key={u.key} style={{ padding: '2px 0 0' }}>
                  {u.known
                    ? t(`phaseTest.unsupportedRow${u.reason === 'no_sht' ? 'NoSht' : ''}`,
                        { key: u.key })
                    : t('phaseTest.unsupportedRow',
                        { key: u.key, reason: u.reason })}
                </div>
              ))}
            </div>
          </div>
        )}

        {excluded.length > 0 && (
          <div style={{ marginTop: 8 }}>
            <button onClick={() => setShowExcluded((v) => !v)} title={t('hoverTips.phaseTestExcludedToggle')} style={{ background: 'transparent',
              border: `1px dashed ${C.border}`, color: C.textSecondary, borderRadius: 5, padding: '6px 10px',
              fontSize: 10, cursor: 'pointer', width: '100%', textAlign: 'left' }}>
              {t('phaseTest.excludedByEds', { count: excluded.length })}
            </button>
            {showExcluded && excluded.map((e) => (
              <div key={e.key} style={{ fontSize: 10, color: C.textSecondary, padding: '4px 10px' }}>
                {t('phaseTest.excludedRow', { formula: e.label, value: e.fit != null ? e.fit.toFixed(2) : '—' })}
              </div>
            ))}
          </div>
        )}
        </div>{/* end RIGHT column */}
        </div>{/* end 2-column body */}
        <PatternExportDialog
          open={exportOpen} onClose={() => setExportOpen(false)}
          sources={{
            experimental: result?.experimental_png || null,
            simulated: selected?.simulated_png || null,
            ncc: selected?.ncc_diff_png || null,
            heatmap: null,
          }}
          rNcc={{ r: selected?.r_score, ncc: null }}
          stepUm={null}
          mapCols={null}
          markers={markerCtl.markers}
        />
      </div>
    </div>
  );
}

function Bar({ value, label, color }) {
  const pct = value == null ? 0 : Math.max(0, Math.min(100, value * 100));
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <div style={{ flex: 1, height: 7, background: '#44475a', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color }} />
      </div>
      <span style={{ fontSize: 10, color, minWidth: 32, textAlign: 'right' }}>{label}</span>
    </div>
  );
}

/**
 * Click-to-pick scan overview with the real Band Contrast image as
 * background + an SVG crosshair at the active pixel. Mirrors the
 * CrystalHint SinglePixelMode picker. Falls back to an empty box (the
 * crosshair + click logic still works) until the overview loads.
 */
function ScanPicker({
  gridShape, pixelIndex, overviewB64, onPick, t, edsOverlays = [], edsOpacity = 0.6,
  view = null, onZoomAt, onPan, onResetView,
}) {
  const [nRows, nCols] = gridShape;
  const zoomView = view || { scale: 1, cx: 0.5, cy: 0.5 };
  const zoomed = isZoomed(zoomView);
  const bindWheel = useWheelZoom(onZoomAt);
  const panRef = useRef(null);
  const movedRef = useRef(false);
  const maxW = 280;
  const maxH = 240;
  const aspect = nCols / Math.max(nRows, 1);
  const w = aspect >= maxW / maxH ? maxW : Math.round(maxH * aspect);
  const h = aspect >= maxW / maxH ? Math.round(maxW / aspect) : maxH;
  const cy = Math.floor(pixelIndex / Math.max(nCols, 1));
  const cx = pixelIndex % Math.max(nCols, 1);
  const handleClick = (ev) => {
    // A drag that panned the view is not a pick — otherwise letting go after
    // moving the map would silently jump to another pixel.
    if (movedRef.current) { movedRef.current = false; return; }
    const box = ev.currentTarget.getBoundingClientRect();
    // The picture is drawn through a CSS transform, so the box is NOT where
    // the content is once zoomed. `zoomedRect` gives the rectangle the content
    // actually occupies; measuring the box instead would pick the wrong pixel
    // by exactly the pan offset.
    const rect = zoomedRect(box, zoomView);
    const x = ev.clientX - rect.left;
    const y = ev.clientY - rect.top;
    const c = Math.max(0, Math.min(nCols - 1, Math.floor((x / rect.width) * nCols)));
    const r = Math.max(0, Math.min(nRows - 1, Math.floor((y / rect.height) * nRows)));
    onPick(r, c);
  };

  const startPan = (ev) => {
    if (!zoomed || ev.button !== 0 || !onPan) return;
    panRef.current = { x: ev.clientX, y: ev.clientY, w: ev.currentTarget.clientWidth, h: ev.currentTarget.clientHeight };
    movedRef.current = false;
  };
  const movePan = (ev) => {
    const p = panRef.current;
    if (!p) return;
    const dx = ev.clientX - p.x;
    const dy = ev.clientY - p.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) movedRef.current = true;
    panRef.current = { ...p, x: ev.clientX, y: ev.clientY };
    onPan(-dx / (p.w * zoomView.scale), -dy / (p.h * zoomView.scale));
  };
  const endPan = () => { panRef.current = null; };
  const px = ((cx + 0.5) / nCols) * w;
  const py = ((cy + 0.5) / nRows) * h;
  return (
    <div
      ref={bindWheel}
      data-scan-picker
      onClick={handleClick}
      onMouseDown={startPan}
      onMouseMove={movePan}
      onMouseUp={endPan}
      onMouseLeave={endPan}
      style={{
        position: 'relative', width: w, height: h,
        background: C.bgSecondary, border: `1px solid ${C.border}`,
        borderRadius: 4, cursor: zoomed ? 'grab' : 'crosshair',
        // Non-negotiable: the zoomed picture stays inside its frame.
        overflow: 'hidden',
      }}
    >
      {zoomed && (
        <div
          data-scanpicker-zoom-badge
          onClick={(e) => { e.stopPropagation(); onResetView?.(); }}
          title={t('phasemap:hoverTips.zoomReset')}
          style={{
            position: 'absolute', top: 3, right: 3, zIndex: 3,
            padding: '1px 6px', borderRadius: 9, background: 'rgba(0,0,0,.6)',
            color: '#fff', fontSize: 9, lineHeight: 1.5, cursor: 'pointer', userSelect: 'none',
          }}
        >
          {zoomView.scale.toFixed(1)}×
        </div>
      )}
      <div style={{
        position: 'absolute', inset: 0,
        transform: viewToTransform(zoomView), transformOrigin: '50% 50%',
      }}>
      {overviewB64 ? (
        <img
          src={`data:image/png;base64,${overviewB64}`}
          alt={t('phaseTest.scanOverviewAlt')}
          style={{
            position: 'absolute', inset: 0, width: '100%', height: '100%',
            objectFit: 'fill', imageRendering: 'pixelated', pointerEvents: 'none',
          }}
        />
      ) : (
        <div style={{
          position: 'absolute', inset: 0, display: 'flex',
          alignItems: 'center', justifyContent: 'center',
          color: C.textSecondary, fontSize: 10, opacity: 0.5,
        }}>
          {t('phaseTest.loadingOverview')}
        </div>
      )}
      {edsOverlays.map((b64, i) => (
        <img
          key={i}
          src={`data:image/png;base64,${b64}`}
          alt={t('phaseTest.edsOverlayAlt')}
          style={{
            position: 'absolute', inset: 0, width: '100%', height: '100%',
            objectFit: 'fill', imageRendering: 'pixelated', pointerEvents: 'none',
            opacity: edsOpacity, mixBlendMode: 'screen',
          }}
        />
      ))}
      <svg width={w} height={h} style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
        <line x1={px} y1={0} x2={px} y2={h} stroke={C.accent} strokeWidth="1" opacity="0.7" />
        <line x1={0} y1={py} x2={w} y2={py} stroke={C.accent} strokeWidth="1" opacity="0.7" />
        <circle cx={px} cy={py} r={5} fill={C.accent} stroke="#fff" strokeWidth="1.5" />
        <text x={4} y={h - 6} fill="#fff" fontSize="10" style={{
          paintOrder: 'stroke', stroke: '#000', strokeWidth: 2,
        }}>
          ({cy}, {cx})
        </text>
      </svg>
      </div>
    </div>
  );
}
