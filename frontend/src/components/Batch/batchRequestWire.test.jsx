// @vitest-environment node
/**
 * What the Batch wizard actually puts on the wire for a spherical run.
 *
 * Kept apart from the component tests on purpose: those mock
 * `../../services/api` wholesale, so they can only ever prove what the wizard
 * asked the client for — never what the client sent. The defect pinned here was
 * exactly that shape, three times over:
 *
 *   * the Step 5 form wrote `bandwidth` / `nregions` / `refine`;
 *   * `batch_manager.run_single_indexing_job` reads `spherical_bandwidth` /
 *     `spherical_nregions` / `spherical_refine`;
 *   * and `createBatch` sent neither spelling.
 *
 * So every batch spherical run was L=88 / 10 regions / refine=True whatever the
 * user typed, and the engine choice the Indexing page offers did not exist here
 * at all. Nothing was red.
 *
 * The contract these tests hold is "one key set, no translation table": the
 * names below are the backend's, written out literally so that renaming either
 * side breaks this file rather than a user's run.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('axios', () => {
  const inst = {
    get: vi.fn(() => Promise.resolve({ data: {} })),
    put: vi.fn(() => Promise.resolve({ data: {} })),
    post: vi.fn(() => Promise.resolve({ data: {} })),
    delete: vi.fn(() => Promise.resolve({ data: {} })),
    interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } },
  };
  return { default: { create: () => inst, __inst: inst }, __inst: inst };
});

import axios from 'axios';
import { batchV2Api } from '../../services/api';
import {
  buildBatchConfig,
  SPHERICAL_DEFAULTS,
  SPHERICAL_CONFIG_KEYS,
} from './BatchWorkflow';

/**
 * The keys `backend/api/services/batch_manager.py` reads out of `job_config`
 * for a spherical job. Transcribed by hand; the backend test
 * `tests/test_batch_spherical_options.py` holds the same statement from the
 * other end.
 */
const BACKEND_KEYS = [
  'backend',
  'spherical_bandwidth',
  'spherical_nregions',
  'spherical_refine',
];

const lastPost = (url) => {
  const call = [...axios.__inst.post.mock.calls].reverse().find((c) => c[0] === url);
  return call ? call[1] : null;
};

beforeEach(() => { vi.clearAllMocks(); });

describe('the spherical key set is shared, not translated', () => {
  it('the wizard ships exactly the names the backend reads', () => {
    expect([...SPHERICAL_CONFIG_KEYS].sort()).toEqual([...BACKEND_KEYS].sort());
  });

  it('defaults to the same engine the Indexing page defaults to', () => {
    expect(SPHERICAL_DEFAULTS.backend).toBe('spherical_gpu');
  });
});

describe('buildBatchConfig', () => {
  const formValues = {
    spherical_bandwidth: 64,
    spherical_nregions: 5,
    spherical_refine: false,
    backend: 'emsphinx',
    auto_export: true,
    export_dir: 'D:/out',
  };

  it('carries every spherical setting the user typed, unrenamed', () => {
    const cfg = buildBatchConfig('spherical', formValues);
    expect(cfg.backend).toBe('emsphinx');
    expect(cfg.spherical_bandwidth).toBe(64);
    expect(cfg.spherical_nregions).toBe(5);
    expect(cfg.spherical_refine).toBe(false);
    // And not under the old names, which the backend would ignore.
    expect('bandwidth' in cfg).toBe(false);
    expect('nregions' in cfg).toBe(false);
    expect('refine' in cfg).toBe(false);
  });

  it('fills the defaults when the user touched nothing', () => {
    const cfg = buildBatchConfig('spherical', { auto_export: false });
    for (const k of BACKEND_KEYS) {
      expect(cfg[k]).toBe(SPHERICAL_DEFAULTS[k]);
    }
  });

  it('adds no spherical keys to a Hough or Dictionary batch', () => {
    for (const method of ['hough', 'dictionary']) {
      const cfg = buildBatchConfig(method, formValues);
      for (const k of BACKEND_KEYS) {
        expect({ method, key: k, present: k in cfg })
          .toEqual({ method, key: k, present: false });
      }
    }
  });

  it('leaves the export and preprocessing block as it was', () => {
    const cfg = buildBatchConfig('hough', {
      auto_export: true, export_dir: 'D:/out', export_formats: ['h5'],
      include_eds_in_export: true, frame_averaging: true,
      postproc_ci_threshold: 0.2,
    });
    expect(cfg.auto_export).toBe(true);
    expect(cfg.export_dir).toBe('D:/out');
    expect(cfg.export_formats).toEqual(['h5']);
    expect(cfg.include_eds).toBe(true);
    expect(cfg.preprocessing.frame_averaging).toBe(true);
    expect(cfg.postprocessing.ci_threshold).toBe(0.2);
  });
});

describe('POST /api/batch-v2/create', () => {
  it('puts the spherical settings on the wire verbatim', async () => {
    const cfg = buildBatchConfig('spherical', {
      spherical_bandwidth: 128, spherical_nregions: 8,
      spherical_refine: true, backend: 'spherical_gpu',
    });
    await batchV2Api.create(['a.h5oina'], ['Al.sht'], cfg);

    const body = lastPost('/api/batch-v2/create');
    expect(body).not.toBeNull();
    expect(body.config.backend).toBe('spherical_gpu');
    expect(body.config.spherical_bandwidth).toBe(128);
    expect(body.config.spherical_nregions).toBe(8);
    expect(body.config.spherical_refine).toBe(true);
  });
});
