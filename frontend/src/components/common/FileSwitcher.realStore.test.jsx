// @vitest-environment jsdom
//
// The header file switcher against the REAL data store, the REAL loaded-files store and the REAL
// axios instance (only the transport is replaced), so what is pinned is what a user's click does
// to the state the other pages read, not what a mock was told to answer.
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { render, fireEvent, waitFor, cleanup } from '@testing-library/react';
import FileSwitcher from './FileSwitcher';
import api, { ebsdApi } from '../../services/api';
import useDataStore from '../../stores/useDataStore';
import useLoadedFilesStore from '../../stores/useLoadedFilesStore';

const A = '/data/fileA.h5oina';
const B = '/data/fileB.h5oina';
const C = '/data/fileC.h5oina';

const original = api.defaults.adapter;
let backendActive;
let pending;          // path -> release()
let answers;          // path -> switch answer body
let failing;          // paths whose switch is answered with an error
let writes;           // every file_path the viewer copy of the load answer (ebsdInfo) was set to
let unsubscribe;

function respond(config, data) {
  return { status: 200, statusText: 'OK', headers: {}, config, data };
}

function installBackend() {
  backendActive = A;
  pending = {};
  failing = new Set();
  answers = {
    [B]: { success: true, file_path: B, grid_shape: [3, 4], scan_geometry: null },
    [C]: { success: true, file_path: C, grid_shape: [5, 6], scan_geometry: null },
  };
  api.defaults.adapter = (config) => {
    const url = String(config.url);
    if (url.includes('/api/ebsd/switch-file')) {
      const path = JSON.parse(config.data).path;
      return new Promise((resolve, reject) => {
        pending[path] = () => {
          if (failing.has(path)) {
            reject(Object.assign(new Error('Request failed with status code 500'), {
              isAxiosError: true, config, response: { status: 500, data: { detail: 'load failed' } },
            }));
            return;
          }
          backendActive = path;
          resolve(respond(config, answers[path]));
        };
      });
    }
    if (url.includes('/api/ebsd/loaded-files')) {
      return Promise.resolve(respond(config, {
        files: [A, B, C].map((p) => ({ path: p, name: p.split('/').pop(), active: p === backendActive })),
      }));
    }
    if (url.includes('/api/ebsd/info')) {
      return Promise.resolve(respond(config, {
        loaded: true, file_path: backendActive, data_shape: backendActive === A ? [2, 2, 8, 8] : [3, 4, 8, 8],
        scan_geometry: null,
      }));
    }
    return Promise.resolve(respond(config, {}));
  };
}

const prevInfo = { tag: 'previous file', grid_shape: [2, 2], scan_geometry: null };

beforeEach(() => {
  cleanup();
  installBackend();
  useLoadedFilesStore.setState({ files: [] });
  useDataStore.setState({
    isFileOpen: true, filePath: A, gridShape: [2, 2], ebsdLoaded: false, ebsdInfo: null,
  });
  writes = [];
  unsubscribe = useDataStore.subscribe((st, prev) => {
    if (st.ebsdInfo !== prev.ebsdInfo) writes.push(st.ebsdInfo?.file_path);
  });
});
afterEach(() => { unsubscribe(); api.defaults.adapter = original; });

async function startHeaderSwitch(to, onSwitched) {
  const view = render(<FileSwitcher onSwitched={onSwitched} />);
  await view.findByText('fileA.h5oina');
  fireEvent.change(view.container.querySelector('select'), { target: { value: to } });
  await waitFor(() => expect(pending[to]).toBeTypeOf('function'));
  return view;
}

describe('header switch after a renderer reload (the viewer has not been entered yet)', () => {
  it('leaves ebsdLoaded false and ebsdInfo empty, so the viewer still hydrates itself on its first visit', async () => {
    // syncFromBackend fills the file fields after a reload but never sets `ebsdLoaded`; the viewer's
    // hydration effect returns early when it is true.
    const view = await startHeaderSwitch(B);
    pending[B]();
    await waitFor(() => expect(useDataStore.getState().filePath).toBe(B));
    await waitFor(() => expect(view.queryByText('switching…')).toBeNull());
    const st = useDataStore.getState();
    expect(st.ebsdLoaded).toBe(false);
    expect(st.ebsdInfo).toBeNull();
    // the store's own grid still follows the file
    expect(st.gridShape).toEqual([3, 4]);
  });
});

describe('header switch with a viewer that is already hydrated', () => {
  it('replaces the viewer copy of the load answer with the new file\'s', async () => {
    useDataStore.setState({ ebsdLoaded: true, ebsdInfo: prevInfo });
    const view = await startHeaderSwitch(B);
    pending[B]();
    await waitFor(() => expect(useDataStore.getState().ebsdInfo?.grid_shape).toEqual([3, 4]));
    await waitFor(() => expect(view.queryByText('switching…')).toBeNull());
    expect(useDataStore.getState().ebsdInfo.file_path).toBe(B);
    expect(useDataStore.getState().ebsdLoaded).toBe(true);
  });
});

describe('two switches in flight (header and another place), answers in either order', () => {
  beforeEach(() => { useDataStore.setState({ ebsdLoaded: true, ebsdInfo: prevInfo }); writes = []; });

  it('the older header answer arriving AFTER the newer one is never written', async () => {
    const onSwitched = [];
    const view = await startHeaderSwitch(B, (p) => onSwitched.push(p));
    const viewerSwitch = ebsdApi.switchFile(C);            // started later, from the viewer panel
    await waitFor(() => expect(pending[C]).toBeTypeOf('function'));
    pending[C]();
    const res = await viewerSwitch;
    useDataStore.getState().setEBSDLoaded(res.data);        // what the viewer panel does with its answer
    pending[B]();
    await waitFor(() => expect(view.queryByText('switching…')).toBeNull());
    expect(useDataStore.getState().ebsdInfo.file_path).toBe(C);
    expect(useDataStore.getState().ebsdInfo.grid_shape).toEqual([5, 6]);
    expect(writes).toEqual([C]);
    expect(onSwitched).toEqual([]);
  });

  it('the older header answer arriving BEFORE the newer one waits for it and is never written', async () => {
    const onSwitched = [];
    const view = await startHeaderSwitch(B, (p) => onSwitched.push(p));
    const viewerSwitch = ebsdApi.switchFile(C);
    await waitFor(() => expect(pending[C]).toBeTypeOf('function'));
    pending[B]();
    await new Promise((r) => setTimeout(r, 20));
    expect(view.queryByText('switching…')).not.toBeNull();  // still deciding: the newer one is pending
    expect(useDataStore.getState().ebsdInfo).toBe(prevInfo);
    pending[C]();
    const res = await viewerSwitch;
    useDataStore.getState().setEBSDLoaded(res.data);
    await waitFor(() => expect(view.queryByText('switching…')).toBeNull());
    expect(useDataStore.getState().ebsdInfo.file_path).toBe(C);
    expect(writes).toEqual([C]);
    expect(onSwitched).toEqual([]);
  });

  it.each([['the newer one answered with its error first', true], ['the older one answered first', false]])(
    'a newer switch that FAILS does not supersede the older: its answer is written (%s)', async (_, newerFirst) => {
      failing.add(C);
      const onSwitched = [];
      const view = await startHeaderSwitch(B, (p) => onSwitched.push(p));
      const viewerSwitch = ebsdApi.switchFile(C).catch((e) => e);
      await waitFor(() => expect(pending[C]).toBeTypeOf('function'));
      if (newerFirst) { pending[C](); await viewerSwitch; pending[B](); } else { pending[B](); pending[C](); }
      await viewerSwitch;
      await waitFor(() => expect(onSwitched).toEqual([B]));
      await waitFor(() => expect(view.queryByText('switching…')).toBeNull());
      expect(useDataStore.getState().ebsdInfo.file_path).toBe(B);
      expect(writes).toEqual([B]);
    });

  it.each([['remove'], ['clear'], ['clear-all']])(
    'a header switch pending while the list is edited (%s) is written and reported', async (op) => {
      const onSwitched = [];
      const view = await startHeaderSwitch(B, (p) => onSwitched.push(p));
      if (op === 'remove') await ebsdApi.removeLoadedFile(C);
      else if (op === 'clear') await ebsdApi.clearLoadedFiles();
      else await ebsdApi.clearAllLoadedFiles();
      pending[B]();
      await waitFor(() => expect(onSwitched).toEqual([B]));
      await waitFor(() => expect(view.queryByText('switching…')).toBeNull());
      expect(useDataStore.getState().ebsdInfo.file_path).toBe(B);
      expect(useDataStore.getState().ebsdInfo.grid_shape).toEqual([3, 4]);
    });

  it('a lone header switch is written and reported as before', async () => {
    const onSwitched = [];
    const view = await startHeaderSwitch(B, (p) => onSwitched.push(p));
    pending[B]();
    await waitFor(() => expect(onSwitched).toEqual([B]));
    await waitFor(() => expect(view.queryByText('switching…')).toBeNull());
    expect(useDataStore.getState().ebsdInfo.file_path).toBe(B);
  });
});
