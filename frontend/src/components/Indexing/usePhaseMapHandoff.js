import { useEffect, useRef } from 'react';
import { edsApi } from '../../services/api';
import useDataStore from '../../stores/useDataStore';
import { pickHandoffFiles, handoffFileKindKey, handoffMethodLabelKey } from './phaseMapHandoff';

/**
 * The hand-off from the EDS page ("→ Send to Indexing") to the Indexing page.
 *
 * It lives here and not in IndexingPage because the page is ~4000 lines behind
 * dozens of API calls and nothing mounts it in a test; this is the part of it
 * that decides which files a run gets, and that is worth being able to test.
 *
 * Once, when the page opens with `pendingPhaseMapIndexing` set:
 *   1) for the method that is selected NOW (Hough, Spherical or Dictionary --
 *      the method is not changed) select the discovered file of every phase of
 *      the map, ONE per phase: the CIF itself for Hough, the SHT / the fitting
 *      dictionary whose linked CIF it is for the others (see phaseMapHandoff.js);
 *   2) install the classified-pixel mask and switch routing on (`onInstall`) --
 *      also when the list holds no file at all: every phase is then reported
 *      as missing, and the run is not silently turned into something else;
 *   3) consume the flag, so going back and forth does not re-apply it.
 * The wait-gate is `discoveredFor === method`: files cannot be matched against
 * a list that has not arrived, nor against the previous method's list, which
 * stays on screen until the new one arrives.
 *
 * Afterwards the payload is kept. When the user changes the method with routing
 * still on, the same phases are re-selected from the NEW method's list instead
 * of leaving an empty list with routing switched on.
 *
 * @param {object} o
 * @param {boolean} o.isActive         the Indexing page is the visible page
 * @param {string|null} o.filePath     the loaded file; a new one forgets the hand-off
 * @param {string} o.method            'hough' | 'spherical' | 'dictionary'
 * @param {object[]} o.discoveredFiles the current list
 * @param {string|null} o.discoveredFor the method that list was fetched for
 * @param {boolean} o.routingOn        per-phase routing is switched on
 * @param {() => boolean} o.hasSelection  a phase is already selected for this method
 * @param {() => object} [o.getDictContext]  `{currentPc, detectorShape, geom}`: what a
 *   Dictionary has to fit, read when the selection is made
 * @param {boolean} [o.dictContextReady]  that context belongs to the loaded file and
 *   has arrived; a Dictionary hand-off waits for it
 * @param {(files: object[], extra: {dictPaths: object}) => void} o.onSelect
 *   select these files as the phases (`extra.dictPaths`: Dictionary, path -> dictionary to run)
 * @param {(cfg: object, files: object[]) => void} o.onInstall  mask + routing on
 * @param {(msg: string) => void} o.log
 * @param {(key: string, params?: object) => string} o.t
 */
export default function usePhaseMapHandoff({
  isActive, filePath, method, discoveredFiles, discoveredFor, routingOn,
  hasSelection, getDictContext, dictContextReady = true, onSelect, onInstall, log, t,
}) {
  const payloadRef = useRef(null);       // the stored hand-off payload
  const selectedForRef = useRef(null);   // the method the selection was made for
  // The callbacks are re-created every render; the effect must see the latest
  // without re-running for them.
  const live = useRef(null);
  live.current = { hasSelection, getDictContext, onSelect, onInstall, log, t };

  function applySelection(cfg, m, files) {
    const { log: say, t: tr, onSelect: select, getDictContext: ctxOf } = live.current;
    const picked = pickHandoffFiles(files, m, cfg.cif_filenames, ctxOf ? ctxOf() : {});
    const { matched, missing, needsDictionary, several, dictPaths } = picked;
    if (matched.length > 0) select(matched, { dictPaths });
    if (missing.length > 0) {
      say(m === 'hough'
        ? tr('messages.handoffMissing', { count: missing.length, list: missing.join(', ') })
        : tr('messages.handoffMissingForMethod', {
          count: missing.length, list: missing.join(', '),
          kind: tr(handoffFileKindKey(m)), method: tr(handoffMethodLabelKey(m)),
        }));
    }
    if (needsDictionary.length > 0) {
      say(tr('messages.handoffNeedsDictionary',
        { count: needsDictionary.length, list: needsDictionary.join(', ') }));
    }
    for (const s of several) {
      say(tr('messages.handoffSeveralFiles', {
        name: s.name, kind: tr(handoffFileKindKey(m)),
        taken: s.taken, others: s.others.join(', '),
      }));
    }
    return matched;
  }

  useEffect(() => {
    if (!isActive) return undefined;
    // A Dictionary is chosen by what fits the detector and the pattern centre of
    // the loaded file. Until the page has heard that file's detector context the
    // values behind `getDictContext` are defaults (no shape: anything fits) or
    // the previous file's, and the choice, once written, outranks the page's own
    // live check -- so wait, the same way as for the method's list.
    if (method === 'dictionary' && !dictContextReady) return undefined;

    if (!useDataStore.getState().pendingPhaseMapIndexing) {
      // Method switched after the hand-off, routing still on: redo the
      // pre-selection for the new method from the stored payload.
      const stored = payloadRef.current;
      if (!stored || !routingOn) return undefined;
      if (selectedForRef.current === method) return undefined;
      if (discoveredFor !== method) return undefined;
      selectedForRef.current = method;
      // Something is already chosen for this method (a phase sent over from
      // the Phase Test, say): leave the user's choice alone.
      if (live.current.hasSelection()) return undefined;
      const matched = applySelection(stored, method, discoveredFiles);
      live.current.log(live.current.t('messages.handoffApplied',
        { phases: matched.length, pixels: stored.n_classified }));
      return undefined;
    }

    // The list for THIS method has arrived (it may hold nothing: then every
    // phase is reported missing below, and the toggle and the mask still come).
    if (discoveredFor !== method) return undefined;

    let cancelled = false;
    (async () => {
      let consume = true;
      try {
        const res = await edsApi.phaseMapIndexingConfig();
        if (cancelled) { consume = false; return; }
        const cfg = res.data;
        if (!cfg?.loaded) {
          live.current.log(live.current.t('messages.handoffSkipped'));
          return;
        }
        payloadRef.current = cfg;
        selectedForRef.current = method;
        const matched = applySelection(cfg, method, discoveredFiles);
        live.current.onInstall(cfg, matched);
        live.current.log(live.current.t('messages.handoffApplied',
          { phases: matched.length, pixels: cfg.n_classified }));
      } catch (err) {
        live.current.log(live.current.t('messages.handoffError',
          { error: err.response?.data?.detail || err.message }));
      } finally {
        // A run that was cancelled (the list changed, the page was left) leaves
        // the flag for the run that replaces it.
        if (consume) useDataStore.getState().setPendingPhaseMapIndexing(false);
      }
    })();
    return () => { cancelled = true; };
  }, [isActive, method, discoveredFiles, discoveredFor, routingOn, dictContextReady]);

  // After the effect above, so that in the commit that changes the method it
  // still sees the old selection state and waits for the new list. The page
  // clears its phase list in an effect of its own on the method (declared after
  // this hook), so a selection made in that same commit would be wiped: the
  // next list that arrives redoes it, which is what resetting here achieves.
  useEffect(() => {
    selectedForRef.current = null;
  }, [method]);
  useEffect(() => {
    payloadRef.current = null;
    selectedForRef.current = null;
  }, [filePath]);
}
