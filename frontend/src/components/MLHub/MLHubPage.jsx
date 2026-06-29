/**
 * ML Hub — React port of gui/ml_hub_gui.py (MLHubWidget).
 *
 * Layout: 3 tabs
 *   Status   — model availability, loaded model info, store stats
 *   Training — data path, epochs, batch size, learning rate, progress, log
 *   Prediction — pattern indices input, predict button, results table
 */

import { useState, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { mlApi } from '../../services/api';
import {
  colors, alpha, spacing,
  Button, Input, NumberInput, Tabs, TabPanel, GroupBox,
  FormRow, Label, ProgressBar, Separator, Card, StatusDot,
  usePrompt, PromptDialog,
} from '../../theme/components';

// ---------------------------------------------------------------------------
// Status pill matching PyQt5 styling
// ---------------------------------------------------------------------------
function StatusPill({ active, label }) {
  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 5,
      padding: '3px 10px',
      borderRadius: 20,
      fontSize: '9pt',
      fontWeight: 600,
      background: active ? alpha(colors.green, 12) : alpha(colors.red, 12),
      color: active ? colors.green : colors.red,
      border: `1px solid ${active ? alpha(colors.green, 30) : alpha(colors.red, 30)}`,
      transition: 'background 0.3s, color 0.3s, border-color 0.3s',
    }}>
      <span style={{
        width: 6, height: 6, borderRadius: '50%',
        background: active ? colors.green : colors.red,
        display: 'inline-block', flexShrink: 0,
      }} />
      {label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Model card list item
// ---------------------------------------------------------------------------
function ModelCard({ model, isSelected, onClick }) {
  const { t } = useTranslation('mlhub');
  const [hovered, setHovered] = useState(false);
  const name = model.name || model.model_name || t('mlhub:modelCard.unknownModel');
  const loaded = model.loaded || model.is_loaded || false;
  const type = model.type || model.model_type || '';
  const accuracy = model.accuracy != null ? model.accuracy : null;
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      title={t('mlhub:modelCard.tooltip')}
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        width: '100%',
        padding: '10px 12px',
        borderRadius: 5,
        border: `1px solid ${isSelected ? colors.purple : hovered ? alpha(colors.purple, 50) : colors.border}`,
        borderLeft: isSelected ? `3px solid ${colors.purple}` : `1px solid ${isSelected ? colors.purple : hovered ? alpha(colors.purple, 50) : colors.border}`,
        background: isSelected ? alpha(colors.purple, 10) : hovered ? alpha(colors.purple, 5) : 'transparent',
        cursor: 'pointer',
        marginBottom: 5,
        textAlign: 'left',
        transition: 'all 0.15s',
        color: colors.text,
        transform: hovered ? 'translateX(2px)' : 'none',
      }}
    >
      <div>
        <div style={{ fontSize: '10pt', fontWeight: 600, color: isSelected ? colors.purple : colors.text, marginBottom: 2 }}>
          {name}
        </div>
        {type && <div style={{ fontSize: '9pt', color: colors.textSecondary }}>{type}</div>}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 4 }}>
        {loaded && <StatusPill active label={t('mlhub:modelCard.loaded')} />}
        {accuracy != null && (
          <span style={{ fontSize: '9pt', color: colors.cyan }}>
            {t('mlhub:modelCard.accuracy', { value: (accuracy * 100).toFixed(1) })}
          </span>
        )}
      </div>
    </button>
  );
}

// ---------------------------------------------------------------------------
// Prediction results table
// ---------------------------------------------------------------------------
function PredictResultTable({ results }) {
  const { t } = useTranslation('mlhub');
  if (!results || results.length === 0) return null;

  const thStyle = {
    padding: '6px 10px',
    textAlign: 'left',
    color: colors.textSecondary,
    fontWeight: 600,
    fontSize: '9pt',
    borderBottom: `1px solid ${colors.border}`,
    whiteSpace: 'nowrap',
  };
  const tdStyle = (i) => ({
    padding: '5px 10px',
    fontSize: '9pt',
    borderBottom: `1px solid ${alpha(colors.border, 13)}`,
    background: i % 2 === 0 ? 'transparent' : alpha(colors.border, 20),
    transition: 'background 0.1s',
  });

  return (
    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
      <thead>
        <tr>
          {[
            t('mlhub:results.columns.index'),
            t('mlhub:results.columns.phase'),
            t('mlhub:results.columns.confidence'),
            t('mlhub:results.columns.score'),
          ].map((h) => (
            <th key={h} style={thStyle}>{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {results.map((row, i) => {
          const idx = row.index ?? row.pattern_index ?? i;
          const phase = row.phase || row.phase_name || t('mlhub:results.empty');
          const conf = row.confidence != null ? `${(row.confidence * 100).toFixed(1)}%` : t('mlhub:results.empty');
          const score = row.score != null ? row.score.toFixed(4) : t('mlhub:results.empty');
          return (
            <tr key={i} className="table-row-hover">
              <td style={{ ...tdStyle(i), color: colors.textSecondary }}>{idx}</td>
              <td style={{ ...tdStyle(i), color: colors.cyan, fontWeight: 600 }}>{phase}</td>
              <td style={{ ...tdStyle(i), color: colors.green }}>{conf}</td>
              <td style={{ ...tdStyle(i), color: colors.text }}>{score}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------------------
// Training log textarea
// ---------------------------------------------------------------------------
function TrainingLog({ lines }) {
  const { t } = useTranslation('mlhub');
  const ref = useRef(null);
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [lines]);
  const handleCopy = () => {
    if (lines.length === 0) return;
    navigator.clipboard.writeText(lines.join('\n')).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    }).catch(() => {});
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '2px 8px', background: colors.bgSecondary,
        border: `1px solid ${colors.border}`, borderBottom: 'none',
        borderRadius: '4px 4px 0 0',
      }}>
        <span style={{ fontSize: '8pt', color: colors.textSecondary, opacity: 0.7 }}>
          {lines.length > 0
            ? t('mlhub:log.titleWithCount', { count: lines.length })
            : t('mlhub:log.title')}
        </span>
        <button
          onClick={handleCopy}
          disabled={lines.length === 0}
          title={copied ? t('mlhub:log.copiedTooltip') : t('mlhub:log.copyTooltip')}
          aria-label={t('mlhub:log.copyAria')}
          style={{
            background: 'none', border: 'none', cursor: lines.length ? 'pointer' : 'default',
            color: copied ? colors.green : colors.textSecondary, fontSize: '9pt', padding: '1px 4px',
            opacity: lines.length ? 0.7 : 0.3,
            transition: 'color 0.2s',
          }}
        >
          {copied ? '\u2713' : '\uD83D\uDCCB'}
        </button>
      </div>
      <div
        ref={ref}
        className="thin-scrollbar"
        style={{
          background: colors.bgSecondary,
          border: `1px solid ${colors.border}`,
          borderRadius: '0 0 4px 4px',
          height: 160,
          overflowY: 'auto',
          padding: '6px 8px',
          fontFamily: "'Courier New', monospace",
          fontSize: '9pt',
          color: colors.text,
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-all',
        }}
      >
        {lines.length === 0
          ? <span style={{ color: colors.textSecondary }}>{t('mlhub:log.placeholder')}</span>
          : lines.join('\n')}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
// Launch gate: the denoiser / FAISS-embedding training sections have no backend
// endpoint wired, and the classifier "Training data path" field is cosmetic
// (handleTrain always trains on the last indexing result). Hide them for launch.
const SHOW_UNWIRED_ML = false;

export default function MLHubPage({ onNavigate, isActive = false }) {
  const { t } = useTranslation(['mlhub', 'common']);
  const [activeTab, setActiveTab] = useState('status');

  // Status tab state
  const [status, setStatus] = useState(null);
  const [statusLoading, setStatusLoading] = useState(true);
  const [models, setModels] = useState([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState(null);
  const [storeStats, setStoreStats] = useState(null);

  // Training tab state
  // (training data path field removed — handleTrain always uses the last
  // indexing result, so the path was cosmetic)
  const [epochs, setEpochs] = useState(50);
  const [batchSize, setBatchSize] = useState(32);
  const [learningRate, setLearningRate] = useState(0.001);
  const [ciThreshold, setCiThreshold] = useState(0.3);
  const [training, setTraining] = useState(false);
  const [trainProgress, setTrainProgress] = useState(0);
  const [trainTotal, setTrainTotal] = useState(50);
  const [trainLog, setTrainLog] = useState([]);
  const [trainMsg, setTrainMsg] = useState(null);
  const [trainErr, setTrainErr] = useState(false);

  // Denoising training
  const [dnEpochs, setDnEpochs] = useState(30);
  const [dnBatchSize, setDnBatchSize] = useState(16);

  // Embedding training
  const [embDim, setEmbDim] = useState(128);
  const [embEpochs, setEmbEpochs] = useState(50);

  // Prediction tab state
  const [patternInput, setPatternInput] = useState('0, 1, 2');
  const [predictLoading, setPredictLoading] = useState(false);
  const [predictResults, setPredictResults] = useState(null);
  const [predictError, setPredictError] = useState(null);

  const [askPrompt, promptProps] = usePrompt();

  // ---------------------------------------------------------------------------
  // Load status and models
  // ---------------------------------------------------------------------------
  const refreshStatus = () => {
    setStatusLoading(true);
    mlApi.status()
      .then((res) => {
        setStatus(res.data);
        // store_stats is part of the status payload — wire it so the
        // Training Store panel reflects reality (it never did before).
        setStoreStats(res.data?.store_stats || null);
      })
      .catch(() => setStatus(null))
      .finally(() => setStatusLoading(false));
  };

  useEffect(() => {
    if (!isActive) return;
    refreshStatus();

    setModelsLoading(true);
    mlApi.models()
      .then((res) => {
        const list = res.data?.models || res.data || [];
        const arr = Array.isArray(list) ? list : [];
        setModels(arr);
        if (arr.length > 0) setSelectedModel(arr[0]);
      })
      .catch(() => setModels([]))
      .finally(() => setModelsLoading(false));
  }, [isActive]);

  const mlAvailable = status?.available || status?.ml_available || false;
  const modelLoaded = status?.model_loaded || status?.loaded || false;

  // ---------------------------------------------------------------------------
  // Training
  // ---------------------------------------------------------------------------
  const appendLog = (msg) => setTrainLog((prev) => [...prev, msg]);

  const trainAbort = useRef(false);

  const handleTrain = async () => {
    setTraining(true);
    trainAbort.current = false;
    setTrainProgress(0);
    setTrainTotal(epochs);
    setTrainLog([]);
    setTrainMsg(null);
    setTrainErr(false);
    appendLog(t('mlhub:trainLog.starting', { epochs, batch: batchSize, lr: learningRate, ci: ciThreshold.toFixed(2) }));
    appendLog(t('mlhub:trainLog.addingSamples'));
    try {
      const { data } = await mlApi.train({
        trainingDataPath: '__last_indexing__',
        epochs, batchSize, ciThreshold,
      });
      const taskId = data.task_id;
      let lastLogLen = 0;
      // Poll the real background task.
      // eslint-disable-next-line no-constant-condition
      while (true) {
        await new Promise((r) => setTimeout(r, 1000));
        if (trainAbort.current) { appendLog(t('mlhub:trainLog.stoppedWatching')); break; }
        let st;
        try {
          ({ data: st } = await mlApi.trainStatus(taskId));
        } catch (e) {
          if (e.response?.status === 404) { appendLog(t('mlhub:trainLog.taskExpired')); break; }
          throw e;
        }
        if (Array.isArray(st.log) && st.log.length > lastLogLen) {
          st.log.slice(lastLogLen).forEach(appendLog);
          lastLogLen = st.log.length;
        }
        if (typeof st.epoch === 'number') setTrainProgress(st.epoch);
        if (typeof st.total === 'number') setTrainTotal(st.total);
        if (st.status === 'completed') {
          const r = st.result || {};
          setTrainMsg(t('mlhub:trainMsg.complete', {
            count: r.phase_names?.length || 0,
            names: (r.phase_names || []).join(', '),
            loss: r.best_val_loss != null ? r.best_val_loss.toFixed(4) : '?',
          }));
          setTrainErr(false);
          refreshStatus();
          break;
        }
        if (st.status === 'failed') {
          setTrainMsg(st.error || t('mlhub:trainMsg.failed'));
          setTrainErr(true);
          break;
        }
      }
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || t('mlhub:trainMsg.failed');
      appendLog(t('mlhub:trainLog.error', { message: msg }));
      setTrainMsg(msg);
      setTrainErr(true);
    } finally {
      setTraining(false);
    }
  };

  const handleClearStore = async () => {
    const ok = await askPrompt({
      message: t('mlhub:clearStorePrompt'),
    });
    if (ok?.trim().toLowerCase() !== 'clear') { appendLog(t('mlhub:trainLog.clearCancelled')); return; }
    try {
      const { data } = await mlApi.clearStore();
      appendLog(t('mlhub:trainLog.storeCleared', { count: data.samples_removed ?? 0 }));
      setTrainMsg(t('mlhub:trainMsg.storeCleared'));
      setTrainErr(false);
      refreshStatus();
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || t('mlhub:trainMsg.clearFailed');
      appendLog(t('mlhub:trainLog.error', { message: msg }));
      setTrainMsg(msg);
      setTrainErr(true);
    }
  };

  // ---------------------------------------------------------------------------
  // Prediction
  // ---------------------------------------------------------------------------
  const parseIndices = (raw) =>
    raw.split(/[\s,]+/).map((s) => s.trim()).filter(Boolean)
      .map(Number).filter((n) => !isNaN(n));

  const handlePredict = async () => {
    const indices = parseIndices(patternInput);
    if (indices.length === 0) {
      setPredictError(t('mlhub:prediction.noValidIndex'));
      return;
    }
    setPredictLoading(true);
    setPredictError(null);
    setPredictResults(null);
    try {
      const res = await mlApi.predict(indices);
      const rows = res.data?.results || res.data?.predictions || res.data || [];
      setPredictResults(Array.isArray(rows) ? rows : [rows]);
    } catch (err) {
      setPredictError(err.response?.data?.detail || err.message || t('mlhub:prediction.predictionFailed'));
    } finally {
      setPredictLoading(false);
    }
  };

  // ---------------------------------------------------------------------------
  // Tabs definition
  // ---------------------------------------------------------------------------
  const tabs = [
    { id: 'status', label: t('mlhub:tabs.status'), tip: t('mlhub:tabs.statusTip') },
    { id: 'training', label: t('mlhub:tabs.training'), tip: t('mlhub:tabs.trainingTip') },
    { id: 'prediction', label: t('mlhub:tabs.prediction'), tip: t('mlhub:tabs.predictionTip') },
  ];

  // ---------------------------------------------------------------------------
  // Status tab
  // ---------------------------------------------------------------------------
  const statusTab = (
    <div className="thin-scrollbar" style={{ display: 'flex', flexDirection: 'column', gap: spacing.outerSpacing, height: '100%', overflow: 'auto' }}>

      {/* Model Status */}
      <GroupBox title={t('mlhub:status.modelStatus')}>
        {statusLoading ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {[0.7, 0.5, 0.6].map((w, i) => (
              <div key={i} className="skeleton-shimmer" style={{
                height: 14, borderRadius: 3,
                width: `${w * 100}%`,
                background: alpha(colors.border, 19),
              }} />
            ))}
          </div>
        ) : status ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: spacing.innerSpacing }}>
            <div style={{
              padding: '8px 10px',
              borderRadius: 4,
              border: `1px solid ${colors.border}`,
              fontSize: '11pt',
              fontWeight: 'bold',
              color: modelLoaded ? colors.green : colors.red,
            }}>
              {modelLoaded
                ? (status.phase_names
                    ? t('mlhub:status.modelLoadedPhases', { count: status.phase_names.length, names: status.phase_names.join(', ') })
                    : t('mlhub:status.modelLoaded'))
                : t('mlhub:status.noModelLoaded')}
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <div className="table-row-hover" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '2px 4px', borderRadius: 3 }}>
                <Label secondary>{t('mlhub:status.mlModule')}</Label>
                <StatusPill active={mlAvailable} label={mlAvailable ? t('mlhub:status.available') : t('mlhub:status.unavailable')} />
              </div>
              <div className="table-row-hover" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '2px 4px', borderRadius: 3 }}>
                <Label secondary>{t('mlhub:status.model')}</Label>
                <StatusPill active={modelLoaded} label={modelLoaded ? t('mlhub:status.loaded') : t('mlhub:status.notLoaded')} />
              </div>
              {status.backend && (
                <div className="table-row-hover" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '2px 4px', borderRadius: 3 }}>
                  <Label secondary>{t('mlhub:status.backend')}</Label>
                  <span style={{ fontSize: '9pt', color: colors.cyan }}>{status.backend}</span>
                </div>
              )}
              {status.device && (
                <div className="table-row-hover" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '2px 4px', borderRadius: 3 }}>
                  <Label secondary>{t('mlhub:status.device')}</Label>
                  <span style={{ fontSize: '9pt', color: colors.orange }}>{status.device}</span>
                </div>
              )}
            </div>
          </div>
        ) : (
          <Label style={{ color: colors.red }}>{'\u26A0'} {t('mlhub:status.failedToLoad')}</Label>
        )}
      </GroupBox>

      {/* Training Store Stats */}
      <GroupBox title={t('mlhub:status.trainingStore')}>
        {storeStats ? (
          <div>
            <Label secondary small>
              {t('mlhub:status.storeTotal', { count: storeStats.total_samples ?? 0 })}
              {storeStats.samples_per_phase
                ? t('mlhub:status.storePhases', { phases: Object.entries(storeStats.samples_per_phase).map(([k, v]) => `${k}: ${v}`).join(', ') })
                : ''}
              {storeStats.mean_confidence != null ? t('mlhub:status.storeMeanCi', { value: storeStats.mean_confidence.toFixed(3) }) : ''}
            </Label>
          </div>
        ) : (
          <Label secondary small>{t('mlhub:status.noTrainingData')}</Label>
        )}
        <Separator />
        <div style={{ display: 'flex', gap: spacing.innerSpacing, marginTop: spacing.innerSpacing }}>
          <Button small onClick={() => setActiveTab('training')} title={t('mlhub:status.goToTrainingTooltip')}>
            {t('mlhub:status.goToTraining')}
          </Button>
        </div>
      </GroupBox>

      {/* Models List */}
      <GroupBox title={modelsLoading ? t('mlhub:status.modelsLoading') : t('mlhub:status.modelsCount', { count: models.length })} style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <div style={{ flex: 1, overflowY: 'auto' }}>
          {modelsLoading ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, padding: 4 }}>
              {[0.85, 0.7, 0.6].map((w, i) => (
                <div key={i} className="skeleton-shimmer" style={{
                  height: 42, borderRadius: 5,
                  width: `${w * 100}%`,
                  background: alpha(colors.border, 15),
                }} />
              ))}
            </div>
          ) : models.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '16px 8px' }}>
              <div style={{ fontSize: '20pt', opacity: 0.3, marginBottom: 6 }}>{'\u2699'}</div>
              <Label secondary small style={{ display: 'block' }}>{t('mlhub:status.noModels')}</Label>
              <div style={{ fontSize: '8pt', color: colors.textSecondary, marginTop: 4, opacity: 0.7 }}>
                {t('mlhub:status.noModelsHint')}
              </div>
              <Button small style={{ marginTop: 8 }} onClick={() => setActiveTab('training')} title={t('mlhub:status.goToTrainingEmptyTooltip')}>
                {t('mlhub:status.goToTraining')}
              </Button>
            </div>
          ) : models.map((model, i) => (
            <ModelCard
              key={model.name || model.model_name || i}
              model={model}
              isSelected={selectedModel === model}
              onClick={() => setSelectedModel(model)}
            />
          ))}
        </div>
      </GroupBox>
    </div>
  );

  // ---------------------------------------------------------------------------
  // Training tab
  // ---------------------------------------------------------------------------
  const trainingTab = (
    <div className="thin-scrollbar" style={{ display: 'flex', flexDirection: 'column', gap: spacing.outerSpacing, height: '100%', overflow: 'auto' }}>

      {/* Phase Classifier Training */}
      <GroupBox title={t('mlhub:training.classifierTitle')}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: spacing.innerSpacing }}>
          <Label secondary small>
            {t('mlhub:training.classifierDesc')}
          </Label>

          <div style={{ fontSize: '9pt', color: colors.textSecondary, fontStyle: 'italic' }}>
            {t('mlhub:training.usesLastResult')}
          </div>

          <div style={{ display: 'flex', gap: spacing.innerSpacing, flexWrap: 'wrap' }}>
            <FormRow label={t('mlhub:training.epochs')} title={t('mlhub:training.epochsTooltip')} style={{ marginBottom: 0, flex: '0 0 auto' }}>
              <NumberInput
                value={epochs}
                onChange={(e) => setEpochs(Number(e.target.value))}
                min={1}
                max={1000}
                step={10}
                style={{ width: 72 }}
                title={t('mlhub:hoverTips.epochs')}
              />
            </FormRow>
            <FormRow label={t('mlhub:training.batch')} title={t('mlhub:training.batchTooltip')} style={{ marginBottom: 0, flex: '0 0 auto' }}>
              <NumberInput
                value={batchSize}
                onChange={(e) => setBatchSize(Number(e.target.value))}
                min={1}
                max={256}
                step={8}
                style={{ width: 72 }}
                title={t('mlhub:hoverTips.batch')}
              />
            </FormRow>
            <FormRow label={t('mlhub:training.lr')} title={t('mlhub:training.lrTooltip')} style={{ marginBottom: 0, flex: '0 0 auto' }}>
              <NumberInput
                value={learningRate}
                onChange={(e) => setLearningRate(Number(e.target.value))}
                min={0.0001}
                max={0.1}
                step={0.0001}
                style={{ width: 90 }}
                title={t('mlhub:hoverTips.learningRate')}
              />
            </FormRow>
          </div>

          <FormRow label={t('mlhub:training.ciThreshold')} title={t('mlhub:training.ciThresholdTooltip')}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <input
                type="range"
                min={0}
                max={100}
                value={Math.round(ciThreshold * 100)}
                onChange={(e) => setCiThreshold(Number(e.target.value) / 100)}
                style={{ flex: 1 }}
                title={t('mlhub:hoverTips.ciThreshold')}
                aria-label={t('mlhub:hoverTips.ciThreshold')}
              />
              <span style={{
                fontSize: '10pt',
                fontWeight: 'bold',
                color: ciThreshold >= 0.3 ? colors.green : ciThreshold >= 0.15 ? colors.orange : colors.red,
                minWidth: 36,
              }}>
                {ciThreshold.toFixed(2)}
              </span>
            </div>
          </FormRow>

          <div style={{ display: 'flex', gap: spacing.innerSpacing, flexWrap: 'wrap' }}>
            <Button variant="primary" onClick={handleTrain} disabled={training}
              title={t('mlhub:training.startTrainingTooltip')}>
              {training ? t('mlhub:training.trainingInProgress') : t('mlhub:training.startTraining')}
            </Button>
            <Button onClick={() => { trainAbort.current = true; }} disabled={!training}
              title={t('mlhub:training.stopTooltip')}>
              {t('mlhub:training.stop')}
            </Button>
            <Button onClick={handleClearStore} disabled={training}
              title={t('mlhub:training.clearStoreTooltip')}>
              {t('mlhub:training.clearStore')}
            </Button>
          </div>

          {training && (
            <ProgressBar
              value={trainProgress}
              max={trainTotal}
              label={t('mlhub:training.epochProgress', { current: trainProgress, total: trainTotal })}
              color={colors.purple}
            />
          )}

          {trainMsg && (
            <div style={{
              fontSize: '9pt',
              color: trainErr ? colors.red : colors.green,
              padding: '4px 6px',
              border: `1px solid ${trainErr ? colors.red : colors.green}`,
              borderRadius: 4,
            }}>
              {trainMsg}
            </div>
          )}
        </div>
      </GroupBox>

      {/* Denoising Training — not wired to a backend endpoint yet */}
      {SHOW_UNWIRED_ML && (
      <GroupBox title={t('mlhub:training.denoiserTitle')}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: spacing.innerSpacing }}>
          <Label secondary small>
            {t('mlhub:training.denoiserDesc')}
          </Label>
          <div style={{
            fontSize: '8.5pt', color: colors.orange, padding: '3px 6px',
            border: `1px dashed ${alpha(colors.orange, 50)}`, borderRadius: 4,
          }}>
            {t('mlhub:training.notWired')}
          </div>
          <div style={{ display: 'flex', gap: spacing.innerSpacing, flexWrap: 'wrap', opacity: 0.5 }}>
            <FormRow label={t('mlhub:training.epochs')} title={t('mlhub:training.denoiserEpochsTooltip')} style={{ marginBottom: 0 }}>
              <NumberInput value={dnEpochs} onChange={(e) => setDnEpochs(Number(e.target.value))} min={5} max={500} step={5} style={{ width: 72 }} disabled />
            </FormRow>
            <FormRow label={t('mlhub:training.batch')} title={t('mlhub:training.batchTooltip')} style={{ marginBottom: 0 }}>
              <NumberInput value={dnBatchSize} onChange={(e) => setDnBatchSize(Number(e.target.value))} min={4} max={128} step={4} style={{ width: 72 }} disabled />
            </FormRow>
          </div>
          <div style={{ display: 'flex', gap: spacing.innerSpacing }}>
            <Button disabled title={t('mlhub:training.notWiredTooltip')}>{t('mlhub:training.trainDenoiser')}</Button>
            <Button disabled title={t('mlhub:training.notWiredTooltip')}>{t('mlhub:training.loadDenoiser')}</Button>
          </div>
        </div>
      </GroupBox>
      )}

      {/* Embedding Indexing */}
      {SHOW_UNWIRED_ML && (
      <GroupBox title={t('mlhub:training.embeddingTitle')}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: spacing.innerSpacing }}>
          <Label secondary small>
            {t('mlhub:training.embeddingDesc')}
          </Label>
          <div style={{
            fontSize: '8.5pt', color: colors.orange, padding: '3px 6px',
            border: `1px dashed ${alpha(colors.orange, 50)}`, borderRadius: 4,
          }}>
            {t('mlhub:training.embeddingNotWired')}
          </div>
          <div style={{ display: 'flex', gap: spacing.innerSpacing, flexWrap: 'wrap', opacity: 0.5 }}>
            <FormRow label={t('mlhub:training.dim')} title={t('mlhub:training.embeddingDimTooltip')} style={{ marginBottom: 0 }}>
              <NumberInput value={embDim} onChange={(e) => setEmbDim(Number(e.target.value))} min={32} max={512} step={32} style={{ width: 72 }} disabled />
            </FormRow>
            <FormRow label={t('mlhub:training.epochs')} title={t('mlhub:training.epochsTooltip')} style={{ marginBottom: 0 }}>
              <NumberInput value={embEpochs} onChange={(e) => setEmbEpochs(Number(e.target.value))} min={5} max={200} step={5} style={{ width: 72 }} disabled />
            </FormRow>
          </div>
          <div style={{ display: 'flex', gap: spacing.innerSpacing }}>
            <Button disabled title={t('mlhub:training.notWiredTooltip')}>{t('mlhub:training.trainEncoder')}</Button>
            <Button disabled title={t('mlhub:training.notWiredTooltip')}>{t('mlhub:training.buildFaissIndex')}</Button>
          </div>
        </div>
      </GroupBox>
      )}

      {/* Training Log */}
      <GroupBox title={t('mlhub:training.logTitle')}>
        <TrainingLog lines={trainLog} />
      </GroupBox>
    </div>
  );

  // ---------------------------------------------------------------------------
  // Prediction tab
  // ---------------------------------------------------------------------------
  const predictionTab = (
    <div className="thin-scrollbar" style={{ display: 'flex', flexDirection: 'column', gap: spacing.outerSpacing, height: '100%', overflow: 'auto' }}>

      <GroupBox title={t('mlhub:prediction.title')}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: spacing.innerSpacing }}>
          <Label secondary small>
            {t('mlhub:prediction.desc')}
          </Label>

          {selectedModel && (
            <div style={{ fontSize: '10pt', color: colors.textSecondary }}>
              {t('mlhub:prediction.model')} <span style={{ color: colors.purple, fontWeight: 600 }}>
                {selectedModel.name || selectedModel.model_name || t('mlhub:prediction.unknownModel')}
              </span>
            </div>
          )}

          <FormRow label={t('mlhub:prediction.patternIndices')} title={t('mlhub:prediction.patternIndicesTooltip')}>
            <Input
              value={patternInput}
              onChange={(e) => setPatternInput(e.target.value)}
              placeholder={t('mlhub:prediction.patternIndicesPlaceholder')}
              style={{ fontFamily: "'Courier New', monospace" }}
              title={t('mlhub:hoverTips.patternIndices')}
            />
          </FormRow>

          <Button
            onClick={handlePredict}
            disabled={predictLoading || !mlAvailable}
            title={t('mlhub:prediction.predictTooltip')}
            style={{ alignSelf: 'flex-start' }}
          >
            {predictLoading ? t('mlhub:prediction.predicting') : t('mlhub:prediction.predict')}
          </Button>

          {!mlAvailable && (
            <div style={{ fontSize: '9pt', color: colors.orange }}>
              {t('mlhub:prediction.mlUnavailable')}
            </div>
          )}

          {predictError && (
            <div role="alert" style={{
              padding: '6px 10px',
              borderRadius: 4,
              fontSize: '9pt',
              background: alpha(colors.red, 8),
              border: `1px solid ${colors.red}`,
              color: colors.red,
              animation: 'fadeSlideIn 0.2s ease-out',
            }}>
              {predictError}
            </div>
          )}
        </div>
      </GroupBox>

      {predictResults ? (
        <GroupBox title={t('mlhub:prediction.resultsTitle', { count: predictResults.length })} style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column', animation: 'fadeSlideIn 0.2s ease-out' }}>
          <div style={{ flex: 1, overflowY: 'auto' }}>
            <PredictResultTable results={predictResults} />
          </div>
        </GroupBox>
      ) : !predictError ? (
        <div style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: colors.bgSecondary,
          border: `1px solid ${colors.border}`,
          borderRadius: 6,
        }}>
          <div style={{ textAlign: 'center', color: colors.textSecondary }}>
            <div style={{ fontSize: '28pt', marginBottom: 8, opacity: 0.3 }}>&#9671;</div>
            <div style={{ fontSize: '10pt' }}>{t('mlhub:prediction.noPredictions')}</div>
            <div style={{ fontSize: '9pt', marginTop: 4 }}>{t('mlhub:prediction.noPredictionsHint')}</div>
          </div>
        </div>
      ) : null}
    </div>
  );

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      color: colors.text,
      fontFamily: "'Segoe UI', system-ui, sans-serif",
      gap: spacing.outerSpacing,
    }}>
      {/* Header */}
      <div>
        <h1 style={{ margin: 0, fontSize: '18pt', fontWeight: 700, color: colors.accent }}>{t('mlhub:header.title')}</h1>
        <div style={{ fontSize: '10pt', color: colors.textSecondary, marginTop: 3 }}>
          {t('mlhub:header.subtitle')}
        </div>
      </div>

      <Tabs tabs={tabs} activeTab={activeTab} onTabChange={setActiveTab} />

      <TabPanel visible={activeTab === 'status'} style={{ padding: 0 }}>
        {statusTab}
      </TabPanel>
      <TabPanel visible={activeTab === 'training'} style={{ padding: 0 }}>
        {trainingTab}
      </TabPanel>
      <TabPanel visible={activeTab === 'prediction'} style={{ padding: 0 }}>
        {predictionTab}
      </TabPanel>
      <PromptDialog {...promptProps} />
    </div>
  );
}
