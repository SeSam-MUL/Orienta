// @vitest-environment jsdom
import { describe, it, expect, beforeEach, vi } from 'vitest';

const reportUiError = vi.fn();
const addBreadcrumb = vi.fn();
vi.mock('../services/errorReporter', () => ({
  reportUiError: (...a) => reportUiError(...a),
}));
vi.mock('../services/breadcrumbs', () => ({
  addBreadcrumb: (...a) => addBreadcrumb(...a),
}));

import useToastStore, { toast } from './useToastStore';

beforeEach(() => {
  reportUiError.mockClear();
  addBreadcrumb.mockClear();
  useToastStore.setState({ toasts: [] });
});

describe('useToastStore error reporting', () => {
  it('reports error toasts — they vanish after 6 s but are what users screenshot', () => {
    toast.error('Indexing failed: phase_list required');
    expect(reportUiError).toHaveBeenCalledWith(
      'Indexing failed: phase_list required',
      'toast',
    );
  });

  it('records warnings as breadcrumbs, not as reports', () => {
    toast.warning('Pattern quality is low');
    expect(reportUiError).not.toHaveBeenCalled();
    expect(addBreadcrumb).toHaveBeenCalledWith('ui-warning', 'Pattern quality is low');
  });

  it('leaves success and info alone', () => {
    toast.success('Saved');
    toast.info('Loading finished');
    expect(reportUiError).not.toHaveBeenCalled();
    expect(addBreadcrumb).not.toHaveBeenCalled();
  });

  it('still shows the toast as before', () => {
    const id = toast.error('boom', 0); // duration 0 = no auto-dismiss
    const { toasts } = useToastStore.getState();
    expect(toasts).toHaveLength(1);
    expect(toasts[0]).toMatchObject({ id, message: 'boom', type: 'error' });
  });

  it('removeToast still works', () => {
    const id = toast.error('boom', 0);
    useToastStore.getState().removeToast(id);
    expect(useToastStore.getState().toasts).toHaveLength(0);
  });
});
