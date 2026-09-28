/**
 * The phase library.
 *
 * WHAT THIS PAGE IS FOR. Sebastian, holding the shipped app: "ich bin
 * generell damit unzufrieden wie wir phasen wo anzeigen und nennen ... ich
 * musste trotzdem ewig suchen um das cif zu finden". A first-time-user agent
 * put the same thing in numbers: told to "use the Al phases", no search term
 * he tried found pure aluminium, and he said he would close the program and
 * ask a colleague for a list of filenames instead.
 *
 * This is the M0 scaffold: the search from `phaseSearch.js`, the four load
 * states, and the entry points. The facets (M1), the bands (M2) and the
 * profile card (M3) land in it; the shape below is what they land in.
 *
 * THE FOUR STATES, and why the fourth one is written down (spec §2.8):
 *
 *   loading         asked, waiting -- never an empty list
 *   error           could not ask; still asking, and the page says so
 *   empty           asked, answered, and the library really is empty
 *   filtered-empty  asked, answered, and the filters match nothing
 *
 * The first three are the lesson from `useAppVersion` and `phaseDiscovery`:
 * a failed request is not an answer, and "no phases found" for a library of
 * 36 sends the user off to rebuild what is already there. The fourth is the
 * one Fassung 1 of the spec forgot, and it is the nastiest, because when the
 * filters match nothing every facet count is 0 -- so the one way out of the
 * state is invisible from inside it. Hence: name the filter that is biting,
 * and put "clear filters" in the message itself.
 *
 * FACET COUNTS DO NOT APPEAR BEFORE THE LIST. In the mockups they sat there
 * complete while "loading" was still running, which says the numbers are
 * known and the phases are merely slow. They are the same request.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { phaseLibraryApi } from '../../services/api';
import { addBreadcrumb } from '../../services/breadcrumbs';
import { nextDelay } from '../../services/retryBackoff';
import { colors, spacing } from '../../theme/components';
import { buildIndex, search } from './phaseSearch';
import { createLibraryLoader, pageState } from './libraryLoad';
import {
  elementFacets, applyElementFacets, toggleElement,
  capabilityFacets, applyCapabilityFacets, toggleCapability,
} from './facets';
import FacetPanel from './FacetPanel';
import PhaseBands from './PhaseBands';
import PhaseRow from './PhaseRow';
import GlossaryLegend from './GlossaryLegend';
import useDatabaseReveal from '../../stores/useDatabaseReveal';
import IdentityLine from './IdentityLine';
import { systemBands } from './bands';
import { siblingsOf } from './siblings';
import GroupPanel from './GroupPanel';
import useGroups from './useGroups';
import { dragData } from './dragPayload';
import { flags } from './groupPersistence';
import useCollectionStore from '../../stores/useCollectionStore';

/** How long a wait has to be before the page explains itself. */
export const SLOW_AFTER_MS = 3000;

const S = {
  page: { height: '100%', display: 'flex', flexDirection: 'column',
          background: colors.bg, color: colors.text, overflow: 'hidden' },
  head: { display: 'flex', alignItems: 'center', gap: spacing.groupMargin,
          padding: `${spacing.groupMargin}px ${spacing.outerMargin}px`,
          borderBottom: `1px solid ${colors.border}` },
  title: { fontSize: '13pt', fontWeight: 600, margin: 0 },
  searchBox: { flex: 1, maxWidth: 480, padding: '6px 10px', fontSize: '10pt',
               background: colors.bgSecondary, color: colors.text,
               border: `1px solid ${colors.border}`, borderRadius: 3 },
  count: { fontSize: '9pt', color: colors.textSecondary, whiteSpace: 'nowrap' },
  views: { display: 'flex', gap: 2 },
  activeFilters: {
    display: 'flex', alignItems: 'baseline', gap: spacing.groupMargin,
    padding: `4px ${spacing.outerMargin}px`, fontSize: '9pt',
    background: `${colors.purple}18`, color: colors.textSecondary,
    borderBottom: `1px solid ${colors.border}`,
  },
  selection: {
    display: 'flex', alignItems: 'baseline', gap: spacing.innerSpacing,
    padding: `4px ${spacing.outerMargin}px`, fontSize: '9pt',
    color: colors.textSecondary, borderBottom: `1px solid ${colors.border}`,
  },
  clearAll: {
    background: 'transparent', border: 'none', color: colors.purple,
    cursor: 'pointer', fontSize: '9pt', padding: 0,
  },
  viewBtn: (on) => ({
    padding: '3px 10px', fontSize: '9pt', borderRadius: 3, cursor: 'pointer',
    border: `1px solid ${on ? colors.purple : colors.border}`,
    background: on ? `${colors.purple}33` : 'transparent',
    color: colors.text,
  }),
  body: { flex: 1, display: 'flex', minHeight: 0 },
  facets: { width: 240, minWidth: 240, borderRight: `1px solid ${colors.border}`,
            padding: spacing.groupMargin, overflowY: 'auto' },
  list: { flex: 1, padding: spacing.groupMargin, overflowY: 'auto' },
  message: { maxWidth: 520, margin: '48px auto', textAlign: 'center',
             fontSize: '10pt', lineHeight: 1.6, color: colors.textSecondary },
  row: { padding: '6px 8px', borderBottom: `1px solid ${colors.border}`,
         fontSize: '10pt', display: 'flex', gap: spacing.innerSpacing, alignItems: 'baseline' },
  why: { fontSize: '8pt', color: colors.textSecondary },
  groupHead: { fontSize: '9pt', fontWeight: 600, color: colors.textSecondary,
               margin: `${spacing.outerMargin}px 0 ${spacing.compactMargin}px` },
  button: { background: 'transparent', color: colors.purple, cursor: 'pointer',
            border: `1px solid ${colors.border}`, borderRadius: 3,
            padding: '4px 12px', fontSize: '9pt' },
};

export default function PhaseLibraryPage({
  isActive = true, loadIndex, onNavigate = null, loadCard = null,
}) {
  const { t } = useTranslation('phaselibrary');
  const [loadState, setLoadState] = useState('idle');
  const [payload, setPayload] = useState(null);
  const [query, setQuery] = useState('');
  const [elements, setElements] = useState([]);
  const [methods, setMethods] = useState([]);
  // 'systems' is the default view (spec §2), the flat list is the other one.
  // "My groups" joins them with the groups themselves.
  const [view, setView] = useState('systems');
  // MEASURED, which is why this exists: the index endpoint answers in 0.11 s
  // warm and 16.4 s cold -- 74 s the very first time on this machine, before
  // the OS had the CIFs in its file cache. It reads 36 structure files and
  // the headers of 65 HDF5 masters. So the first person to open this page
  // after Orienta starts waits a quarter of a minute in front of a sentence
  // that says nothing about why, and a silent wait that long reads as a
  // hang. After three seconds the page says what it is doing.
  const [slow, setSlow] = useState(false);
  // Keyed on the PHASE, so one phase ticked in three bands is ticked in all
  // three -- it is one phase in three systems, which is what the view says.
  const [selected, setSelected] = useState(() => new Set());
  const loaderRef = useRef(null);
  const revealInBrowser = useDatabaseReveal((st) => st.reveal);

  // The groups. Read once, when the page is first opened: they live outside
  // React (a store), so a second page visit must not lose what is there, and
  // re-reading on every visit would throw away an unsaved complaint.
  const groups = useGroups((st) => st.groups);
  const groupsLoaded = useGroups((st) => st.loaded);
  const groupUnsaved = useGroups((st) => st.unsaved);
  const groupStorageOut = useGroups((st) => st.unavailable);
  const canUndoGroups = useGroups((st) => st.past.length > 0);
  // KEEP ASKING WHEN THE FIRST READ FAILS. `load()` sets `loaded` in both
  // branches, so this effect fires once and, after a failure, never again:
  // groups were dead for the rest of the session with no retry and no
  // button (the "read again" button only appears once something is
  // unsaved, and after a failed read nothing is). The case is not
  // hypothetical and is named two files away in `libraryLoad.js` -- the
  // first start after a runtime update, window up, uvicorn not yet -- where
  // the phase index recovers by itself on the same screen where the groups
  // did not.
  useEffect(() => {
    if (!isActive || groupsLoaded) return undefined;
    let cancelled = false;
    let timer = null;
    let delay = 0;                       // nextDelay(0) is the first wait
    const attempt = () => {
      useGroups.getState().load().then((ok) => {
        if (cancelled || ok) return;
        delay = nextDelay(delay);
        timer = setTimeout(attempt, delay);
      });
    };
    attempt();
    return () => { cancelled = true; if (timer) clearTimeout(timer); };
  }, [isActive, groupsLoaded]);
  // "Show me this in the Database browser" leaves the page, so the page --
  // not the card -- owns the navigation.
  const onReveal = (category, name) => {
    if (revealInBrowser(category, name) && onNavigate) onNavigate('database');
  };

  // `loadIndex` is injectable so the tests -- and, until c1's endpoint lands,
  // a harness -- can drive the page without a backend. The default is the one
  // place the URL is named.
  const fetchIndex = useCallback(
    () => (loadIndex ? loadIndex() : phaseLibraryApi.getIndex().then((r) => r.data)),
    [loadIndex],
  );

  // Only while the page is open. App.jsx mounts EVERY page cell at once and
  // hides the inactive ones with a style, so the hooks in here run from app
  // start whether or not anyone has been to this page -- and the early
  // `return null` below happens after them, too late to stop a request. With
  // the backend down that would be a retry loop running from boot, on a page
  // nobody opened.
  useEffect(() => {
    if (!isActive) return undefined;
    const loader = createLibraryLoader({
      fetchIndex,
      onState: setLoadState,
      onData: setPayload,
      note: addBreadcrumb,
    });
    loaderRef.current = loader;
    loader.run();
    return () => { loader.dispose(); loaderRef.current = null; };
  }, [fetchIndex, isActive]);

  useEffect(() => {
    if (loadState !== 'loading') { setSlow(false); return undefined; }
    const t = setTimeout(() => setSlow(true), SLOW_AFTER_MS);
    return () => clearTimeout(t);
  }, [loadState]);

  const phases = payload?.phases;
  // Rebuilding the index is the expensive part, so it happens when the
  // library changes and not when the query does.
  const index = useMemo(() => (phases ? buildIndex(phases) : []), [phases]);
  const found = useMemo(() => search(index, query), [index, query]);

  // Search and elements are one conjunction: the chips narrow what the
  // search found, in every group, and the facet counts are measured on
  // what survives both -- otherwise a chip's number would not be what
  // clicking it gives you.
  const hits = useMemo(() => {
    if (!elements.length && !methods.length) return found;
    const all = found.identity.concat(found.prototype, found.text).map((e) => e.phase);
    const keep = new Set(
      applyCapabilityFacets(applyElementFacets(all, elements), methods)
        .map((p) => p.key));
    const f = (rows) => rows.filter((e) => keep.has(e.key));
    return { identity: f(found.identity), prototype: f(found.prototype),
             text: f(found.text) };
  }, [found, elements, methods]);

  const total = index.length;
  const shownRows = hits.identity.concat(hits.prototype, hits.text);
  const shown = shownRows.length;
  const shownPhases = shownRows.map((e) => e.phase);
  const elementRows = useMemo(
    () => (phases ? elementFacets(phases, shownPhases, elements) : []),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [phases, shown, elements, methods, query]);
  const capabilityRows = useMemo(
    () => (phases ? capabilityFacets(phases, shownPhases, methods) : []),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [phases, shown, elements, methods, query]);

  // BANDS HOLD IDENTITY HITS ONLY. A citation hit dropped into a band looks
  // exactly like a phase that IS what you typed, and §2.1 forbids precisely
  // that -- it is the mixing that produced `Al` -> 36 of 36. The other two
  // classes keep their own headings below the bands, in both views.
  const layout = useMemo(() => systemBands(hits.identity.map((e) => e.phase)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [phases, shown, elements, methods, query]);
  const whyByKey = useMemo(() => {
    const m = new Map();
    for (const e of shownRows) m.set(e.key, e.why);
    return m;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shown, elements, methods, query, phases]);

  const filtersActive = query.trim().length > 0
    || elements.length > 0 || methods.length > 0;
  const state = pageState({ loadState, total, shown, filtersActive });
  // All three, because any one of them can be what emptied the list and from
  // a screen with nothing on it the user cannot see which.
  //
  // EACH ONE SAYS WHICH IT IS. The three were joined raw and dropped into
  // a sentence that began "Search term:", so with only an element chip on
  // the page said `Search term: Al + Fe` -- naming the one filter that was
  // empty. The reader clears the (already empty) search box, nothing
  // happens, and the page has told them where to look and been wrong.
  const activeFilterText = [
    query.trim() && t('state.filterSearch', { value: query.trim() }),
    elements.length && t('state.filterElements', { value: elements.join(' + ') }),
    methods.length && t('state.filterMethods', {
      value: methods.map((m) => t(`facets.method.${m}`)).join(' + ') }),
  ].filter(Boolean).join(' · ');
  const clearFilters = () => { setQuery(''); setElements([]); setMethods([]); };
  // One function for both views, so a tick means the same thing in each.
  const toggleSelected = (key) => setSelected((cur) => {
    const next = new Set(cur);
    if (next.has(key)) next.delete(key); else next.add(key);
    return next;
  });

  // A NAME LANDS IN THE LIST WITHOUT ASKING AGAIN.
  //
  // The index costs 0.11 s warm and 74 s cold on this machine, so refetching
  // it after somebody types a name would put the page back into "reading the
  // library" for a quarter of a minute over one word. The endpoint's ANSWER
  // is patched into the row instead -- what was stored, not what was typed,
  // so the screen cannot show a name the library does not have. Everything
  // downstream rebuilds from `phases`: the index, the search, the labels.
  const onNamed = useCallback((stored) => {
    setPayload((cur) => (cur ? {
      ...cur,
      phases: cur.phases.map(
        (p) => (p.key === stored.key ? { ...p, ...stored } : p)),
    } : cur));
  }, []);

  /**
   * "Use this group for the next run."
   *
   * NOT a new path. Indexing and the phase tester already read the ACTIVE
   * collection through `useCollectionStore` and `collectionsApi.resolve`,
   * which is where the per-method availability (`Al -> no_master` for
   * Dictionary) is worked out. Sebastian's ask was "ich will sie aus jeder
   * Anzeige heraus benutzen", and the way to honour that is to feed the
   * mechanism that is already there, not to build a second one that can
   * disagree with it.
   *
   * `setActive` keeps the hidden list and re-reads afterwards, so every
   * other consumer sees the change -- which also settles the one real
   * hazard of having two stores over one endpoint while the old collections
   * UI is still standing.
   */
  const useGroupForRun = useCallback((g) => {
    // The id, not the name: `active` goes through `resolve_ref`, which takes
    // either and STORES the id, so a later rename does not break it.
    useCollectionStore.getState().setActive(g.id);
    if (onNavigate) onNavigate('indexing');
  }, [onNavigate]);

  // Anything the library changes about a group has to reach the pickers that
  // read the same folder -- Indexing, the phase tester, the database
  // browser. One re-read, rather than each of them polling and disagreeing
  // in between.
  const groupsRevision = useGroups((st) => st.journal.length);
  const firstRevision = useRef(null);
  useEffect(() => {
    if (firstRevision.current === null) { firstRevision.current = groupsRevision; return; }
    if (groupsRevision === firstRevision.current) return;
    useCollectionStore.getState().load();
  }, [groupsRevision]);

  // The name the list shows, for any key -- the group panel lists members
  // and a raw key there helps nobody.
  const labelByKey = useMemo(() => {
    const m = new Map();
    for (const e of index) m.set(e.key, e.display);
    return m;
  }, [index]);
  const labelForKey = useCallback((key) => labelByKey.get(key) || key,
    [labelByKey]);
  /**
   * Entries that look like this one -- same space group, same compound.
   *
   * Computed here because it needs the whole library, which the card does
   * not have. Two pairs on these 36 phases: the alpha twins a reader got
   * right only by accident, and two Mg17Al12 entries nobody had noticed.
   */
  const siblingsFor = useCallback((key) => (phases ? siblingsOf(phases, key) : []),
    [phases]);
  // The groups holding a phase, by name -- the question the tag model
  // creates and the card could not answer.
  const groupsFor = useCallback(
    (key) => groups.filter((g) => g.members.includes(key)).map((g) => g.name),
    [groups]);
  // Which group a run will use. It was readable only from a chip in the
  // toolbar, in a different part of the screen from the list of groups.
  const activeCollectionName = useCollectionStore(
    (s) => s.data?.state?.active || null);

  // A stable identity, so binding a row's draggable does not tear down and
  // rebind on every render -- rebinding mid-drag drops the drag.
  const selectedList = useMemo(() => Array.from(selected), [selected]);
  const dragPayload = useCallback(
    (key) => dragData(key, selectedList), [selectedList]);

  if (!isActive) return null;

  return (
    <div style={S.page} data-testid="phase-library-page">
      <div style={S.head}>
        <h2 style={S.title}>{t('title')}</h2>
        <input
          style={S.searchBox}
          type="search"
          autoFocus
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={t('searchPlaceholder')}
          aria-label={t('searchLabel')}
        />
        {/* "Entries", not "phases". The library holds 36 FILES, and a lab
            head counted at least four pairs that are one phase twice --
            `Mg17Al12_mp-2151_conventional_standard` against
            `Mg17Al12_MP-mp-2151`, two models of beta-AlFeSi, the three
            alpha approximants, the Mn0.5Fe0.5Al5Si0.68 twins -- and put the
            number of distinct physical phases "somewhere near 30". Saying
            36 phases states something the library has not established.

            Announced, because a count that only changes visually is
            invisible to anyone driving this from the keyboard. Silent until
            the library is actually here -- see below. */}
        <span style={S.count} role="status" aria-live="polite">
          {state === 'ready' || state === 'filtered-empty'
            ? (filtersActive
              // The announced string names the FILTERS, not just a number.
              // A keyboard user tabbing nine chips and pressing the wrong
              // one heard a number change and nothing else -- and the only
              // place the page ever named the active filters was the empty
              // state, the one moment it is already too late.
              ? t('entryCountFiltered', {
                shown, total, filters: activeFilterText,
              })
              : view === 'systems'
              // Both numbers, because 93 alone reads as a library three times
              // this size and 36 alone leaves the reader counting one card
              // three times.
                ? t('entryCountBands',
                    { shown, total, placements: layout.placementCount })
                : t('entryCount', { shown, total }))
            : ''}
        </span>
        <div style={S.views} role="group" aria-label={t('view.label')}>
          {['systems', 'list'].map((v) => (
            <button
              key={v}
              type="button"
              aria-pressed={view === v}
              style={S.viewBtn(view === v)}
              onClick={() => setView(v)}
            >
              {t(`view.${v}`)}
            </button>
          ))}
        </div>
      </div>

      {/* WHAT IS FILTERING, WHILE IT IS FILTERING. This line used to exist
          only in the empty state, and a user put the problem better than the
          spec did: the dangerous case is not zero results, it is seventeen
          plausible-looking results that are secretly the wrong seventeen.
          With two filters on and a list on screen, nothing said which two. */}
      {filtersActive && (state === 'ready') && (
        <div style={S.activeFilters} data-testid="active-filters">
          <span>{t('activeFilters', { filters: activeFilterText })}</span>
          <button type="button" style={S.clearAll} onClick={clearFilters}>
            {t('state.clearFilters')}
          </button>
        </div>
      )}

      {selected.size > 0 && (
        // A checkbox that does nothing beats a button that pretends, but
        // NOTHING said so: a keyboard user would have ticked five phases
        // across three bands and then tabbed to the bottom of a 93-row list
        // looking for the control that uses them. One line costs nothing.
        <div style={S.selection} data-testid="selection-note">
          {t('selection.count', { count: selected.size })}
          {' — '}
          {t('selection.forNow')}
          <button type="button" style={S.clearAll}
                  onClick={() => setSelected(new Set())}>
            {t('selection.clear')}
          </button>
        </div>
      )}

      <div style={S.body}>
        <div style={S.facets} data-testid="phase-library-facets">
          {/* GROUPS FIRST. They are the user's own organisation of the
              library; the facets are the library's. Someone who has made
              `Al-Fe intermetallics` looks for it before they look for a
              chip, and a drop target below the fold is a drop target
              nobody hits. It is shown even while the phases are still
              loading -- unlike the facet counts, a group's existence is
              not an answer to the request in flight. */}
          {groupsLoaded && (
            <GroupPanel
              groups={groups}
              visibleKeys={shownRows.map((e) => e.key)}
              selection={selectedList}
              transitional={!flags.backendReady}
              unsaved={groupUnsaved}
              onReread={() => useGroups.getState().load()}
              unavailable={groupStorageOut}
              canUndo={canUndoGroups}
              onCreate={(name) => useGroups.getState().create(name)}
              onAdd={(id, keys) => useGroups.getState().addMany(id, keys)}
              onMove={(id, keys) => useGroups.getState().moveMany(id, keys)}
              onRename={(id, name) => useGroups.getState().rename(id, name)}
              onDelete={(id) => useGroups.getState().remove(id)}
              onUndo={() => useGroups.getState().undo()}
              onUse={useGroupForRun}
              onNest={(id, parentId) => useGroups.getState().setParent(id, parentId)}
              onRemove={(id, keys) => useGroups.getState().dropMany(id, keys)}
              // An open group lists its phases by the name the list shows,
              // not by their key: `sd_1814127` in a group called "S-phase"
              // helps nobody.
              labelFor={labelForKey}
              libraryTotal={total}
              activeGroupName={activeCollectionName}
              // `setActive(null)` -- the only way back to the whole
              // library, and until now there was none anywhere in the app.
              onStopUsing={() => useCollectionStore.getState().setActive(null)}
              // The suggestion run writes straight to the folder, so the
              // screen has to go and look rather than guess what appeared.
              onSuggested={() => useGroups.getState().load()}
            />
          )}
          {/* Deliberately empty until the phases are here: the counts and the
              list come from one request, and showing the numbers first says
              they are known. M1 fills this. */}
          {state === 'ready' || state === 'filtered-empty' ? (
            <FacetPanel
              elementRows={elementRows}
              selectedElements={elements}
              onToggleElement={(sym) => setElements((cur) => toggleElement(cur, sym))}
              onResetElements={() => setElements([])}
              capabilityRows={capabilityRows}
              selectedCapabilities={methods}
              onToggleCapability={(c) => setMethods((cur) => toggleCapability(cur, c))}
              onResetCapabilities={() => setMethods([])}
              unassignedMasters={payload?.unassigned_masters || []}
            />
          ) : null}
          {/* Under the filters rather than beside a term: someone who has
              stopped at one word gets the tooltip, someone who wants the
              lot gets this, and it is reachable with the keyboard either
              way (§2.9). */}
          {(state === 'ready' || state === 'filtered-empty') && <GlossaryLegend />}
        </div>

        <div style={S.list}>
          {state === 'loading' && (
            <div style={S.message} data-testid="state-loading">
              <p>{t('state.loading')}</p>
              {slow && <p data-testid="state-loading-slow">{t('state.loadingSlow')}</p>}
            </div>
          )}

          {state === 'error' && (
            <div style={S.message} data-testid="state-error">
              <p>{t('state.errorTitle')}</p>
              <p>{t('state.errorRetrying')}</p>
              <button type="button" style={S.button}
                      onClick={() => loaderRef.current?.retry()}>
                {t('state.retryNow')}
              </button>
            </div>
          )}

          {state === 'empty' && (
            <div style={S.message} data-testid="state-empty">
              <p>{t('state.emptyTitle')}</p>
              <p>{t('state.emptyWhere')}</p>
            </div>
          )}

          {state === 'filtered-empty' && (
            <div style={S.message} data-testid="state-filtered-empty">
              {/* Naming the filter matters here more than anywhere: with
                  nothing shown, every facet count is 0, so the user cannot
                  see which one is biting. */}
              <p>{t('state.filteredEmptyTitle', { total })}</p>
              <p>{t('state.filteredEmptyBy', { filter: activeFilterText })}</p>
              <button type="button" style={S.button} onClick={clearFilters}>
                {t('state.clearFilters')}
              </button>
            </div>
          )}

          {state === 'ready' && view === 'systems' && (
            <div data-testid="phase-list">
              <PhaseBands
                bands={layout.bands}
                selected={selected}
                onToggleSelect={toggleSelected}
                whyFor={(key) => whyByKey.get(key)}
                onReveal={onReveal}
                loadCard={loadCard}
                dragPayload={dragPayload}
                    onNamed={onNamed}
                    siblingsFor={siblingsFor}
                    groupsFor={groupsFor}
              />
              {hits.prototype.length > 0 && (
                <>
                  <div style={S.groupHead}>
                    {t('group.prototype')} ({hits.prototype.length})
                  </div>
                  <HitGroup
                    rows={hits.prototype}
                    selected={selected}
                    onToggle={toggleSelected}
                    selectLabelFor={(name) => t('selectPhase', { name })}
                    whyText={(k) => t(`why.${k}`)}
                    onReveal={onReveal}
                    loadCard={loadCard}
                    dragPayload={dragPayload}
                    onNamed={onNamed}
                    siblingsFor={siblingsFor}
                    groupsFor={groupsFor}
                  />
                </>
              )}
              {hits.text.length > 0 && (
                <>
                  <div style={S.groupHead}>
                    {t('group.citation')} ({hits.text.length})
                  </div>
                  <HitGroup
                    rows={hits.text}
                    selected={selected}
                    onToggle={toggleSelected}
                    selectLabelFor={(name) => t('selectPhase', { name })}
                    whyText={(k) => t(`why.${k}`)}
                    onReveal={onReveal}
                    loadCard={loadCard}
                    dragPayload={dragPayload}
                    onNamed={onNamed}
                    siblingsFor={siblingsFor}
                    groupsFor={groupsFor}
                  />
                </>
              )}
            </div>
          )}

          {state === 'ready' && view === 'list' && (
            <div data-testid="phase-list">
              <HitGroup
                    rows={hits.identity}
                    selected={selected}
                    onToggle={toggleSelected}
                    selectLabelFor={(name) => t('selectPhase', { name })}
                    whyText={(k) => t(`why.${k}`)}
                    onReveal={onReveal}
                    loadCard={loadCard}
                    dragPayload={dragPayload}
                    onNamed={onNamed}
                    siblingsFor={siblingsFor}
                    groupsFor={groupsFor}
                  />
              {hits.prototype.length > 0 && (
                <>
                  <div style={S.groupHead}>{t('group.prototype')}</div>
                  <HitGroup
                    rows={hits.prototype}
                    selected={selected}
                    onToggle={toggleSelected}
                    selectLabelFor={(name) => t('selectPhase', { name })}
                    whyText={(k) => t(`why.${k}`)}
                    onReveal={onReveal}
                    loadCard={loadCard}
                    dragPayload={dragPayload}
                    onNamed={onNamed}
                    siblingsFor={siblingsFor}
                    groupsFor={groupsFor}
                  />
                </>
              )}
              {hits.text.length > 0 && (
                <>
                  <div style={S.groupHead}>{t('group.citation')}</div>
                  <HitGroup
                    rows={hits.text}
                    selected={selected}
                    onToggle={toggleSelected}
                    selectLabelFor={(name) => t('selectPhase', { name })}
                    whyText={(k) => t(`why.${k}`)}
                    onReveal={onReveal}
                    loadCard={loadCard}
                    dragPayload={dragPayload}
                    onNamed={onNamed}
                    siblingsFor={siblingsFor}
                    groupsFor={groupsFor}
                  />
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * One block of hits, in the same row component the bands use.
 *
 * It was a plain `<div>` until the keyboard pass (§2.9): not focusable, not
 * selectable, and therefore a phase you could tick in the band view and not
 * in the list -- with the selection meant to survive the switch between
 * them. One phase, two behaviours, decided by a button at the top of the
 * page.
 */
function HitGroup({
  rows, selected, onToggle, selectLabelFor, whyText, onReveal, loadCard,
  dragPayload, onNamed, siblingsFor, groupsFor,
}) {
  return rows.map((e) => (
    <PhaseRow
      key={e.key}
      phase={e.phase}
      label={e.display}
      why={e.why}
      checked={selected.has(e.key)}
      onToggle={onToggle}
      selectLabel={selectLabelFor(e.display)}
      whyText={whyText}
      detail={<IdentityLine phase={e.phase} />}
      withCard
      loadCard={loadCard}
      onReveal={onReveal}
      dragPayload={dragPayload}
                    onNamed={onNamed}
                    siblingsFor={siblingsFor}
                    groupsFor={groupsFor}
    />
  ));
}
