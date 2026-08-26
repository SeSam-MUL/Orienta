/**
 * Toast notification store — lightweight notification system.
 * Types: success, error, info, warning
 */
import { create } from 'zustand';
import { reportUiError } from '../services/errorReporter';
import { addBreadcrumb } from '../services/breadcrumbs';

let nextId = 0;

const useToastStore = create((set, get) => ({
  toasts: [],

  addToast: (message, type = 'info', duration = 4000) => {
    const id = ++nextId;
    // A toast disappears after a few seconds and used to leave no trace at
    // all — yet an error toast is exactly what the user screenshots. Errors
    // go to the backend log; warnings only into the breadcrumb trail, where
    // they give context to whatever fails next.
    if (type === 'error') {
      reportUiError(message, 'toast');
    } else if (type === 'warning') {
      addBreadcrumb('ui-warning', String(message ?? ''));
    }
    set((s) => ({ toasts: [...s.toasts, { id, message, type }] }));
    if (duration > 0) {
      setTimeout(() => get().removeToast(id), duration);
    }
    return id;
  },

  removeToast: (id) => set((s) => ({
    toasts: s.toasts.filter((t) => t.id !== id),
  })),
}));

// Convenience helpers
export const toast = {
  success: (msg, dur) => useToastStore.getState().addToast(msg, 'success', dur),
  error:   (msg, dur) => useToastStore.getState().addToast(msg, 'error', dur ?? 6000),
  info:    (msg, dur) => useToastStore.getState().addToast(msg, 'info', dur),
  warning: (msg, dur) => useToastStore.getState().addToast(msg, 'warning', dur),
};

export default useToastStore;
