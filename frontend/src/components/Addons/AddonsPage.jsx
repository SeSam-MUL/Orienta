/**
 * The add-ons page: what is installed, where Orienta looked, and consent.
 *
 * The first thing most people will meet here is an EMPTY list, so the empty
 * state is not an afterthought: it names the folders that were searched and
 * offers a rescan. Before this, nothing in the app stated where an add-on has
 * to be put — the answer lived in a README outside it, and an interface whose
 * first act is to send you elsewhere has not started.
 *
 * A rescan is just a re-fetch. Discovery reads the disk on every call, so
 * there is nothing to invalidate and no backend work to add.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { addonsApi } from '../../services/addonsApi';
import { indexApi } from '../../services/api';
import useResultStore from '../../stores/useResultStore';
import { describeFailure } from './addonReasons';
import AddonList from './AddonList';
import ConsentDialog from './ConsentDialog';
import SchemaForm, { defaultsFor } from './SchemaForm';
import OutputPanels, { toDelimited } from './OutputPanels';
import { consentIsStale } from './addonIdentity';
import useAddonLayerRequests from '../../stores/useAddonLayerRequests';
import { Button, colors, spacing } from '../../theme/components';

//: ONE object, forever. `selected?.analysis?.params_schema || {type:'object'}`
//: built a NEW literal on every render whenever the field was absent, and
//: SchemaForm keys its onUnsupported effect on the schema's identity — so the
//: page re-rendered itself in a loop and the window stopped responding.
//: Measured: with params_schema absent the test worker was killed at 120 s;
//: with `{}` (truthy, so the fallback is never reached) the same test ran in
//: 1.9 s. The defensive default was the trap.
const EMPTY_SCHEMA = { type: 'object' };

/** A step size as a person writes it, not as float32 widens it.
 *
 * Measured in the running app: a 0.2 µm scan printed
 * "Schrittweite: 0.20000000298023224 µm". The file stores the step as
 * float32, JSON hands it over widened to double, and the raw number went
 * into the sentence — seventeen digits of which one is a measurement.
 *
 * Four decimals in µm is a tenth of a nanometre, finer than any EBSD step,
 * and `Number()` drops the trailing zeros so 0.5 stays "0.5". Deliberately
 * not `toLocaleString`: this is a number inside a translated sentence and a
 * locale would put a comma in it for de, which is right for prose and wrong
 * for a value someone may retype into a field.
 *
 * A step below 5e-5 µm prints as "0 µm". No EBSD scan has one -- that is
 * half an ångström -- and a made-up precision would be worse than a zero
 * that is visibly wrong.
 */
export function formatStepUm(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return value;
  return Number(value.toFixed(4));
}


export default function AddonsPage({ isActive = true }) {
  const { t } = useTranslation('addons');
  const [data, setData] = useState(null);
  const [failure, setFailure] = useState(null);
  const [busyName, setBusyName] = useState(null);
  const [pendingConsent, setPendingConsent] = useState(null);
  const [selected, setSelected] = useState(null);      // {addon, analysis}
  // TWO pieces of state, and the split is the whole point. `values` is what
  // the form shows — the add-on's declared defaults with the user's edits on
  // top. `touched` is the set of names the user actually changed, and only
  // those are sent.
  //
  // One object cannot do both. The form merges each edit onto the value it
  // was given, so handing it the merged object and keeping the result as the
  // request body promoted every declared default into the request the moment
  // any one field was edited — measured: typing a path sent
  // {"n_components":3,"report_dir":"D:/typed"}. The runner records what it is
  // given as a chosen value, so a methods paragraph would have attributed a
  // choice nobody made, and an author changing a default between page load
  // and run would have been silently overruled.
  const [values, setValues] = useState({});
  const [touched, setTouched] = useState(() => new Set());
  const [unsupported, setUnsupported] = useState([]);
  // SEPARATE from busyName, which means "this add-on's switch is in flight".
  // One flag for both let a run on add-on A be started twice: toggling add-on
  // B moved busyName to B, A's Run button re-enabled, and a second POST went
  // out with the first still unresolved. Measured: run-job POSTs = 2.
  const [runBusy, setRunBusy] = useState(false);
  const [job, setJob] = useState(null);        // {id, state, message, fraction}
  // Minted per SUCCESSFUL run and put in the layer id. Without it a second
  // run produces the same id, the layer reducer dedupes it and the bitmap
  // cache hits — so "show as layer" after re-running is a no-op and the user
  // goes on reading the first run's numbers under a label saying otherwise.
  const [runToken, setRunToken] = useState(null);
  const [outcome, setOutcome] = useState(null); // {outputs, ignored_params}
  const pollRef = useRef(null);
  const [backendResultId, setBackendResultId] = useState(null);
  const [backendResult, setBackendResult] = useState(null);

  const indexingResult = useResultStore((s) => s.indexingResult);

  // The same precedence the Phase Maps page uses: this session's own run
  // first, else what the backend is showing — the active result, else the
  // last stored one, which is what get_last_indexing_result draws.
  //
  // NEVER by activating one. /run-job takes the id explicitly, and activation
  // switches the loaded FILE — the hazard autoAdoptGuard exists for.
  useEffect(() => {
    let cancelled = false;
    indexApi.listResults()
      .then((res) => {
        if (cancelled) return;
        const rs = res.data?.results || [];
        const shown = rs.find((r) => r.is_active) ?? rs[rs.length - 1] ?? null;
        setBackendResultId(shown?.id ?? null);
        setBackendResult(shown);
      })
      .catch(() => { if (!cancelled) { setBackendResultId(null); setBackendResult(null); } });
    return () => { cancelled = true; };
    // `isActive` too, and it is the only page that lacked it: pages are
    // hidden, not unmounted, so a panel opened before the user went to
    // Indexing and activated a different result would still name — and write
    // into — the old one. The same convention every other page follows.
  }, [selected, isActive]);

  const resultId = indexingResult?.result_id ?? backendResultId;
  const resultFile = indexingResult?.result_id
    ? (indexingResult.source_file || indexingResult.file_path || '')
    : (backendResult?.source_file || '');

  // `clearFailure` is false for the re-read that FOLLOWS an action, and that
  // is not a detail: the re-read succeeds, so a load that always cleared would
  // wipe the refusal the user is meant to read, milliseconds after they caused
  // it. Measured — the message never appeared at all. An explicit rescan does
  // clear it, because then the user is asking for the current state.
  const load = useCallback(async ({ clearFailure = true } = {}) => {
    try {
      const res = await addonsApi.list();
      setData(res.data);
      if (clearFailure) setFailure(null);
    } catch (err) {
      // The server's own sentence survives; describeFailure only adds a title
      // when it recognises the code.
      //
      // No fallback DETAIL is passed in: describeFailure would then have no
      // way to tell the server's sentence from ours, and a bodyless failure
      // -- the backend not running, the likeliest first failure of all --
      // printed the same sentence twice. The headline for a listing is always
      // ours, because this route has no refusal codes to translate.
      const described = describeFailure(
        { reason: err?.response?.data?.reason,
          detail: err?.response?.data?.detail }, t);
      const serverSaid = Boolean(err?.response?.data?.detail);
      setFailure({ title: t('list.loadFailed'),
                   detail: serverSaid ? described.title : '' });
    }
  }, [t]);

  useEffect(() => { load(); }, [load]);

  const setEnabled = useCallback(async (addon, enabled) => {
    setBusyName(addon.name);
    try {
      await addonsApi.setEnabled(addon.name, enabled);
      setFailure(null);
    } catch (err) {
      setFailure(describeFailure(
        { reason: err?.response?.data?.reason,
          detail: err?.response?.data?.detail }, t));
    } finally {
      setBusyName(null);
      // Always re-read: a refused activation CHANGES the row (the runtime
      // records the refusal), so the list after a failure is not the list
      // before it. Without clearing the message, for the reason in `load`.
      load({ clearFailure: false });
    }
  }, [load, t]);

  const onToggle = useCallback((addon) => {
    if (addon.enabled) { setEnabled(addon, false); return; }
    // Consent is asked once per DECISION, and a name is not a decision.
    // `known` only says a row exists under this name: delete an add-on and
    // drop a different one with the same name in its place — a colleague's
    // zip, a fork — and the old row still says known, while the duplicate
    // guard stays silent because only one is installed. Consent given for one
    // author's code would then cover another's, unasked. So the version and
    // the DOI the decision was made ABOUT are compared with the ones in front
    // of us, and any difference asks again.
    if (addon.known && !consentIsStale(addon)) { setEnabled(addon, true); return; }
    setPendingConsent(addon);
  }, [setEnabled]);

  const onRunAnalysis = useCallback((addon, analysis) => {
    setSelected({ addon, analysis });
    const schema = analysis?.params_schema || EMPTY_SCHEMA;
    setValues(defaultsFor(schema));   // shown from the start, chosen by nobody
    setTouched(new Set());
    setUnsupported([]);
    // EVERYTHING the previous analysis produced goes too. Without this the
    // panel re-labelled run A's numbers with analysis B's name and B's
    // add-on, and three things then lied at once: a CSV saved as
    // "<B>-<A's output>.csv"; a "Show as layer" id built from B's name with
    // A's run token, which the server answers from ITS OWN store — B's
    // earlier map under A's label — or 404s with a remedy that cannot work;
    // and "this analysis is already running" about an analysis that is not.
    setOutcome(null);
    setRunToken(null);
    setJob(null);
  }, []);

  // Re-resolved from the CURRENT listing on every render, not held from the
  // click. `selected` is a snapshot, and a rescan (or a disable) leaves it
  // describing an add-on that has moved on: measured, the panel went on
  // offering a field the edited manifest no longer declared, and would have
  // sent it. The names are the identity; the objects are not.
  const liveAddon = selected
    ? (data?.addons || []).find((a) => a.name === selected.addon.name) || null
    : null;
  const liveAnalysis = liveAddon
    ? (liveAddon.analyses || []).find((a) => a.key === selected.analysis.key) || null
    : null;
  const schema = liveAnalysis?.params_schema || EMPTY_SCHEMA;

  // Which names the user actually changed. Computed by comparing the form's
  // next object with the current one rather than trusting a per-field
  // callback, because the form's contract is "here is the whole next object"
  // — and a key that DISAPPEARS (an emptied number box) is a change too.
  const onFormChange = useCallback((next) => {
    setTouched((prev) => {
      const out = new Set(prev);
      for (const name of new Set([...Object.keys(next), ...Object.keys(values)])) {
        if (next[name] !== values[name]) out.add(name);
      }
      return out;
    });
    setValues(next);
  }, [values]);

  const params = useMemo(() => {
    const out = {};
    const declared = new Set(Object.keys(schema.properties || {}));
    // `name in values` and not `values[name] !== undefined`: a cleared number
    // box removes the key, and "the user cleared it" must send nothing at all
    // rather than send the default back.
    //
    // Filtered by what the CURRENT schema declares, so an edit made before a
    // rescan is not sent for a parameter the manifest has since dropped. The
    // runner would discard it and report it in ignored_params; not sending it
    // is the same answer one step earlier.
    for (const name of touched) {
      if (declared.has(name) && name in values) out[name] = values[name];
    }
    return out;
  }, [touched, values, schema]);

  // Polls exactly as ComputeDiagnosticsPanel does — the job lives in the
  // backend and is fetched by id, so this is a timer and a cleanup, nothing
  // more. Stops on done, on failed, and on unmount: a poll that outlives its
  // job is a request every second for the rest of the session, and the health
  // check waits behind each one.
  useEffect(() => {
    if (!job?.id || job.state !== 'running') return undefined;
    const jobId = job.id;
    // The first look is IMMEDIATE, then every 500 ms. Waiting half a second
    // before the first poll leaves the panel saying nothing about a run that
    // has already started — and for a short analysis that is the only window
    // there is.
    const tick = async () => {
      try {
        const res = await addonsApi.job(jobId);
        const next = res.data || {};
        setJob((cur) => (cur && cur.id === jobId ? { ...cur, ...next } : cur));
        if (next.state === 'done') {
          const got = await addonsApi.jobResult(jobId);
          setOutcome(got.data || null);
          // Whatever this analysis put on the Phase Maps stack before is now
          // describing a run that no longer exists in the server's map store:
          // it is keyed without the run token, so this run overwrote it.
          for (const o of (got.data?.outputs || [])) {
            if (o.kind !== 'map') continue;
            useAddonLayerRequests.getState().supersede(
              `addon:${selected.addon.name}/${resultId}/`
              + `${selected.analysis.key}/${o.key}`);
          }
          setRunToken(jobId);      // unique per run, and short enough for an id
          // The citation panel lives on another page and keys on the
          // result id, which this run did not change. Without this it
          // never re-reads, and the add-on's sentence never appears.
          useAddonLayerRequests.getState().bumpCitations();
        } else if (next.state === 'failed') {
          // The add-on's own message is in `error`; /result would only answer
          // with the same failure, so it is not asked.
          setFailure(describeFailure(
            { reason: next.reason, detail: next.error }, t));
        }
      } catch (err) {
        setJob((cur) => (cur && cur.id === jobId
          ? { ...cur, state: 'failed' } : cur));
        setFailure(describeFailure(
          { reason: err?.response?.data?.reason,
            detail: err?.response?.data?.detail }, t));
      }
    };
    tick();
    pollRef.current = setInterval(tick, 500);
    return () => {
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    };
  }, [job?.id, job?.state, t]);

  const exportTable = useCallback(async (how, output) => {
    // EVERY row, not the rows on screen: the table is capped for rendering,
    // and a file that stopped where the display stops would be
    // complete-looking and short.
    const columns = output.columns || [];
    const rows = output.rows || [];
    if (how === 'copy') {
      await navigator.clipboard?.writeText?.(toDelimited(columns, rows, '	'));
      return;
    }
    // The BOM is decided HERE, once, for both ways out. Excel on Windows
    // reads a BOM-less UTF-8 CSV as the system code page and turns a unit
    // like µm into mojibake, and this file exists to be opened in a
    // spreadsheet. It used to be prepended by the Electron main process
    // instead, which meant the browser download — the same table, saved from
    // a different build — silently came out in a different encoding. The
    // clipboard copy above gets none: it hands text to another application
    // directly, and a BOM there is a stray character in the first cell.
    const csv = '﻿' + toDelimited(columns, rows, ',');
    const name = `${selected?.addon?.name || 'addon'}-${output.key}.csv`;
    if (window.electronAPI?.saveText) {
      // saveText and not saveFile: saveFile only hands back a path for the
      // BACKEND to write to, and these numbers never reach the backend.
      await window.electronAPI.saveText({ defaultPath: name, text: csv });
      return;
    }
    // In a browser there is no Electron dialog, and a download is the honest
    // equivalent rather than a disabled button.
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = name; a.click();
    URL.revokeObjectURL(url);
  }, [selected]);

  const showAsLayer = useCallback((output) => {
    if (!resultId || !selected || !runToken) return;
    // addon:<name>/<result_id>/<analysis_key>/<key>#<run_token>. The RESULT is
    // in the id because /api/phasemap/layer takes none — every other layer is
    // implicitly the active result, and an add-on map is bound to one.
    const id = `addon:${selected.addon.name}/${resultId}/`
      + `${selected.analysis.key}/${output.key}#${runToken}`;
    useAddonLayerRequests.getState().request(
      id, output.label || output.key);
    // `detail: { page }`, the shape every other sender in the app uses and
    // the only one App.jsx reads. A bare string was silently ignored: the
    // layer arrived on the stack and the user stayed on this page with
    // nothing to show for the click — found in the acceptance run, not by a
    // test, because both halves were valid on their own.
    window.dispatchEvent(new CustomEvent('navigate-to',
                                         { detail: { page: 'phasemap' } }));
  }, [resultId, selected, runToken]);

  const browseFor = useCallback(async (name) => {
    // openFolder, not openFile: `format = "path"` says only that the value is
    // a path and stays out of the provenance trail, so the app chooses — and
    // the picker that can create a directory fits a report folder. In a
    // browser there is no picker and the field stays typeable, which is the
    // whole point of not hiding it behind the button.
    const picked = await window.electronAPI?.openFolder?.();
    if (!picked) return;
    setValues((v) => ({ ...v, [name]: picked }));
    setTouched((prev) => new Set(prev).add(name));
  }, []);

  const startRun = useCallback(async () => {
    // The handler refuses on its own, not only through a disabled button: the
    // button can be re-enabled by unrelated state (it was, by toggling
    // another add-on), and a guard that lives only in an attribute is a guard
    // a second click walks past.
    if (!selected || !resultId || unsupported.length || runBusy) return;
    if (job?.state === 'running') return;
    if (!liveAnalysis || !liveAddon?.enabled) return;
    setRunBusy(true);
    try {
      // A second run REPLACES the first: the runtime replaces the provenance
      // step and the stored map for a re-run, and a page that accumulated
      // panels would be the only component with a different idea of which run
      // is current.
      setOutcome(null);
      const res = await addonsApi.runJob(selected.addon.name, {
        result_id: resultId,
        analysis_key: selected.analysis.key,
        params,
      });
      setJob({ id: res.data?.job_id, state: 'running', message: '',
               fraction: null });
      setFailure(null);
    } catch (err) {
      setFailure(describeFailure(
        { reason: err?.response?.data?.reason,
          detail: err?.response?.data?.detail }, t));
    } finally {
      setRunBusy(false);
    }
  }, [selected, resultId, unsupported, params, runBusy, liveAnalysis,
      liveAddon, job, t]);

  const addons = data?.addons || [];
  const searched = data?.searched_paths || [];

  return (
    <div style={{ padding: spacing.panelPadding || 16, color: colors.text }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 12 }}>
        <h2 style={{ margin: 0 }}>{t('page.title')}</h2>
        <span style={{ flex: 1 }} />
        {/* Wrapped, not passed directly: onClick would hand `load` the click
            event as its options object. */}
        <Button small onClick={() => load()}>{t('list.rescan')}</Button>
      </div>
      <p style={{ color: colors.textSecondary, fontSize: '10pt' }}>
        {t('page.subtitle')}
      </p>

      {failure && (
        <div role="alert" style={{
          border: `1px solid ${colors.warning || colors.accent}`,
          borderRadius: 4, padding: 10, marginBottom: 12,
        }}>
          <strong>{failure.title}</strong>
          {failure.detail && (
            <div style={{ fontSize: '9pt' }}>{failure.detail}</div>
          )}
        </div>
      )}

      {!data && !failure && (
        // Five minutes of a blank page is what axios's timeout allows, and a
        // page with a title and nothing under it reads as a rendering bug.
        <div role="status">{t('list.loading')}</div>
      )}

      {data && addons.length === 0 ? (
        <div data-testid="addons-empty">
          <h3 style={{ marginBottom: 4 }}>{t('list.emptyTitle')}</h3>
          {searched.length > 0 ? (
            <>
              <div style={{ fontSize: '10pt' }}>{t('list.emptySearched')}</div>
              <ul style={{ fontSize: '9pt', wordBreak: 'break-all' }}>
                {searched.map((p) => <li key={p}>{p}</li>)}
              </ul>
              <div style={{ color: colors.textSecondary, fontSize: '9pt' }}>
                {t('list.emptyHint')}
              </div>
            </>
          ) : (
            // "Folders searched:" above an empty list, followed by "put one
            // of them in there", points at nothing — and it is reachable
            // whenever a reloaded frontend meets a backend that was not
            // restarted. Not knowing is a different fact, and it is said.
            <div style={{ fontSize: '10pt' }}>{t('list.searchedUnknown')}</div>
          )}
        </div>
      ) : (
        <AddonList addons={addons} onToggle={onToggle}
                   onRunAnalysis={onRunAnalysis} busyName={busyName} />
      )}

      {selected && (
        <div data-testid="addon-run-panel" style={{
          border: `1px solid ${colors.border}`, borderRadius: 6,
          padding: 12, marginTop: 12,
        }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
            <strong>{selected.analysis.label || selected.analysis.key}</strong>
            <span style={{ color: colors.textSecondary, fontSize: '9pt' }}>
              {selected.addon.display_name}
            </span>
          </div>

          {/* WHICH result this writes into. The citation and the provenance
              are written into it, so a page that runs against one without
              naming it repeats a mistake this repo already has a rule
              against: a per-pixel panel must name its pixel. */}
          {resultId ? (
            <div style={{ fontSize: '9pt', marginTop: 6 }}>
              {t('run.into', { id: resultId })}
              {resultFile && (
                <div style={{ color: colors.textSecondary, wordBreak: 'break-all' }}>
                  {resultFile}
                </div>
              )}
            </div>
          ) : (
            <div role="status" style={{ color: colors.warning || colors.accent,
                                        fontSize: '9pt', marginTop: 6 }}>
              {t('run.noResult')}
            </div>
          )}

          <div style={{ marginTop: 10 }}>
            <SchemaForm
              schema={schema}
              value={values}
              onChange={onFormChange}
              onUnsupported={setUnsupported}
              onBrowse={browseFor}
              disabled={runBusy}
            />
          </div>

          {selected && !liveAnalysis && (
            // The analysis the panel was opened for is not in the current
            // listing any more — the manifest was edited, or the add-on was
            // removed. Said, because a Run button that is simply off is read
            // as a broken page.
            <div role="status" style={{ color: colors.warning || colors.accent,
                                        fontSize: '9pt' }}>
              {t('run.gone')}
            </div>
          )}
          {liveAnalysis && liveAddon && !liveAddon.enabled && (
            <div role="status" style={{ color: colors.warning || colors.accent,
                                        fontSize: '9pt' }}>
              {t('run.addonOff')}
            </div>
          )}
          {unsupported.length > 0 && (
            // Running anyway would use the add-on's own default for a setting
            // the user was never shown: a run they did not configure, recorded
            // as though they had.
            <div role="status" style={{ color: colors.warning || colors.accent,
                                        fontSize: '9pt' }}>
              {t('run.unsupported', { names: unsupported.join(', ') })}
            </div>
          )}

          {job?.state === 'running' && (
            <div role="status" style={{ fontSize: '9pt', marginTop: 8 }}>
              <div>
                {t('run.running')}
                {job.message ? ` — ${job.message}` : ''}
                {typeof job.fraction === 'number'
                  ? ` — ${Math.round(job.fraction * 100)} %` : ''}
              </div>
              {/* Known limit, said rather than hidden: the job id lives in
                  this page's state and there is no listing endpoint to find
                  a running job again after a reload. */}
              {/* WHY the Run button is off. A disabled control with no
                  reason is read as a broken page — and "start it again" is
                  the first thing anyone tries. */}
              <div>{t('run.alreadyRunning')}</div>
              {/* Known limit, said rather than hidden: the job id lives in
                  this page's state and there is no listing endpoint to find
                  a running job again after a reload. */}
              <div style={{ color: colors.textSecondary }}>
                {t('run.reloadWarning')}
              </div>
            </div>
          )}

          {outcome?.ignored_params?.length > 0 && (
            // Reachable exactly when a manifest was edited while the app was
            // open: the run discarded a parameter the user believes they set.
            <div role="status" style={{ color: colors.warning || colors.accent,
                                        fontSize: '9pt', marginTop: 8 }}>
              {t('run.ignored', { names: outcome.ignored_params.join(', ') })}
            </div>
          )}

          {outcome?.context && (
            // The two numbers the run was given, beside the numbers it
            // produced. Both can be ABSENT on a real result, and absent is
            // the case worth printing: an add-on that reports a length gets
            // no step size and reports pixels, with nothing in its own output
            // to say so. The quality string is the map's LABEL, from the
            // backend, printed verbatim — it names the measurement ("Band
            // Contrast (native)" vs "Pattern Quality (computed)"), and those
            // are different numbers from the same scan.
            <div data-testid="run-context"
                 style={{ fontSize: '8.5pt', marginTop: 8,
                          color: colors.textSecondary }}>
              <div>{t('run.contextHeading')}</div>
              <div>
                {typeof outcome.context.step_size_um === 'number'
                  ? t('run.contextStep',
                      { value: formatStepUm(outcome.context.step_size_um) })
                  : t('run.contextStepMissing')}
              </div>
              <div>
                {outcome.context.quality_source
                  ? t('run.contextQuality',
                      { value: outcome.context.quality_source })
                  : t('run.contextQualityMissing')}
              </div>
            </div>
          )}

          {outcome?.outputs?.length > 0 && (
            <OutputPanels
              outputs={outcome.outputs}
              onExport={exportTable}
              mapSlot={(o) => (
                <div style={{ display: 'flex', gap: 8, alignItems: 'baseline' }}>
                  <span>{o.label || o.key}</span>
                  <Button small onClick={() => showAsLayer(o)}>
                    {t('outputs.showAsLayer')}
                  </Button>
                </div>
              )}
            />
          )}

          <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
            <Button onClick={() => setSelected(null)}>{t('run.close')}</Button>
            <Button
              variant="primary"
              disabled={!resultId || unsupported.length > 0 || runBusy
                        || job?.state === 'running'
                        || !liveAnalysis || !liveAddon?.enabled}
              onClick={startRun}
            >
              {t('run.start')}
            </Button>
          </div>
        </div>
      )}

      {addons.length > 0 && (
        <div style={{ color: colors.textSecondary, fontSize: '9pt', marginTop: 8 }}>
          {/* What the runtime actually does: the step and library registries
              are append-only for the process, on purpose, because clearing
              them would retroactively change what an already-stored result
              claims about itself. */}
          {t('list.restartHint')}
        </div>
      )}

      <ConsentDialog
        addon={pendingConsent}
        onCancel={() => setPendingConsent(null)}
        onAccept={() => {
          const addon = pendingConsent;
          setPendingConsent(null);
          setEnabled(addon, true);
        }}
      />
    </div>
  );
}
