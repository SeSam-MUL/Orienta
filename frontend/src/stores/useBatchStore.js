import { create } from 'zustand';
import { batchV2Api } from '../services/api';

// Persist batchId across app restarts so the user can resume a running/queued
// batch instead of losing track of it. Only the id is persisted — full state
// (jobs, memory) is rehydrated from the backend on next polling tick.
const LS_KEY = 'kikuchipy_active_batch_id';
const loadStoredBatchId = () => {
  try { return localStorage.getItem(LS_KEY) || null; }
  catch { return null; }
};
const storeBatchId = (id) => {
  try {
    if (id) localStorage.setItem(LS_KEY, id);
    else localStorage.removeItem(LS_KEY);
  } catch { /* localStorage may be unavailable */ }
};

const useBatchStore = create((set, get) => ({
  // State
  batchId: loadStoredBatchId(),
  status: null,      // 'pending' | 'running' | 'paused' | 'completed' | 'failed'
  jobs: [],
  memory: null,
  preflight: null,
  preprocessing: {},  // { frame_averaging: ['3x3'], background_removal: ['dynamic'] }
  config: {},         // parsed config_json from backend (incl. export_dir)
  totalJobs: 0,
  completed: 0,
  failed: 0,
  pollInterval: null,
  connectionError: false,  // surfaced when polling hits a network error
  consecutivePollFailures: 0,

  // Actions
  createBatch: async (files, phases, config) => {
    const res = await batchV2Api.create(files, phases, config);
    const data = res.data;
    storeBatchId(data.batch_id);
    set({
      batchId: data.batch_id,
      preflight: data.preflight,
      totalJobs: data.total_jobs,
      status: 'pending',
      connectionError: false,
      consecutivePollFailures: 0,
    });
    return data;
  },

  startBatch: async () => {
    const { batchId } = get();
    if (!batchId) return;
    try {
      await batchV2Api.start(batchId);
      set({ status: 'running' });
      get().startPolling();
    } catch (err) {
      // 409 = already running; treat as success and start polling anyway
      if (err.response?.status === 409) {
        set({ status: 'running' });
        get().startPolling();
      } else {
        throw err;
      }
    }
  },

  pauseBatch: async () => {
    const { batchId } = get();
    if (!batchId) return;
    await batchV2Api.pause(batchId);
  },

  resumeBatch: async () => {
    const { batchId } = get();
    if (!batchId) return;
    await batchV2Api.resume(batchId);
    set({ status: 'running' });
    get().startPolling();
  },

  stopBatch: async () => {
    const { batchId } = get();
    if (!batchId) return;
    await batchV2Api.stop(batchId);
  },

  fetchStatus: async () => {
    const { batchId } = get();
    if (!batchId) return;
    try {
      const [statusRes, jobsRes] = await Promise.all([
        batchV2Api.status(batchId),
        batchV2Api.jobs(batchId),
      ]);
      const s = statusRes.data;
      // Parse the backend's config_json once so the Dashboard knows where
      // exports landed (export_dir may differ from the source directory).
      let parsedConfig = {};
      if (s.config_json) {
        try { parsedConfig = JSON.parse(s.config_json); } catch { /* keep {} */ }
      }
      set({
        status: s.status,
        completed: s.completed,
        failed: s.failed,
        memory: s.memory,
        preprocessing: s.preprocessing_applied || {},
        config: parsedConfig,
        jobs: jobsRes.data,
        connectionError: false,
        consecutivePollFailures: 0,
      });
      // Stop polling on any terminal OR quiescent status.
      // "paused" is what the backend writes after a user-initiated Stop
      // (batch_manager.run_batch_sync sets status=paused when self._stopped
      // and skips the "completed" update). Without this, polling continued
      // indefinitely after a Stop, wasting network round-trips.
      if (s.status === 'completed'
          || s.status === 'failed'
          || s.status === 'paused') {
        get().stopPolling();
      }
    } catch (err) {
      // 404 = batch not found (server lost it); stop polling and clear id
      if (err.response?.status === 404) {
        storeBatchId(null);
        get().stopPolling();
        set({ batchId: null, status: null, connectionError: false });
        return;
      }
      // Network/timeout: increment failures, surface after 3 strikes
      // (single failures are common and self-heal on the next tick)
      const failures = get().consecutivePollFailures + 1;
      set({ consecutivePollFailures: failures });
      if (failures >= 3) {
        set({ connectionError: true });
        get().stopPolling();
      }
    }
  },

  /**
   * Restore polling on page load if we have a stored batchId and it's still
   * an active batch on the backend. Call from BatchDashboard mount.
   */
  rehydrate: async () => {
    const { batchId } = get();
    if (!batchId) return;
    try {
      const res = await batchV2Api.status(batchId);
      const s = res.data;
      set({
        status: s.status,
        completed: s.completed,
        failed: s.failed,
        memory: s.memory,
        totalJobs: s.total_jobs,
      });
      if (s.status === 'running' || s.status === 'pending') {
        get().startPolling();
      } else {
        // Fetch one more time to get jobs, then stop
        get().fetchStatus();
      }
    } catch (err) {
      if (err.response?.status === 404) {
        // Stored batch no longer exists
        storeBatchId(null);
        set({ batchId: null });
      }
    }
  },

  startPolling: () => {
    get().stopPolling();
    const id = setInterval(() => get().fetchStatus(), 2000);
    set({ pollInterval: id });
  },

  stopPolling: () => {
    const { pollInterval } = get();
    if (pollInterval) {
      clearInterval(pollInterval);
      set({ pollInterval: null });
    }
  },

  reset: () => {
    get().stopPolling();
    storeBatchId(null);
    set({
      batchId: null, status: null, jobs: [], memory: null,
      preflight: null, totalJobs: 0, completed: 0, failed: 0,
      connectionError: false, consecutivePollFailures: 0,
    });
  },
}));

export default useBatchStore;
