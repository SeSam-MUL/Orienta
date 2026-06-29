/**
 * Progress store - tracks background task status.
 */

import { create } from 'zustand';

const useProgressStore = create((set, get) => ({
  tasks: {}, // { taskId: { status, progress, message, result, error } }

  setTask: (taskId, data) => set((state) => ({
    tasks: { ...state.tasks, [taskId]: { ...state.tasks[taskId], ...data } },
  })),

  removeTask: (taskId) => set((state) => {
    const { [taskId]: _, ...rest } = state.tasks;
    return { tasks: rest };
  }),

  getTask: (taskId) => get().tasks[taskId] || null,

  clearCompleted: () => set((state) => {
    const active = {};
    for (const [id, task] of Object.entries(state.tasks)) {
      if (task.status === 'running') {
        active[id] = task;
      }
    }
    return { tasks: active };
  }),
}));

export default useProgressStore;
