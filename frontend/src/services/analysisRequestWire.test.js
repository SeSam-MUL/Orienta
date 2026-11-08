/**
 * The statistics phase, the batch parameters and the Excel download reach the backend as the routes read them:
 * the real axios bodies, with the field names of the pydantic models (tests/test_analysis_frontend_contract.py
 * holds the names against the models; this holds the values).
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('axios', () => {
  const post = vi.fn(() => Promise.resolve({ data: {} }));
  const get = vi.fn(() => Promise.resolve({ data: {} }));
  const inst = {
    post, get, put: vi.fn(), delete: vi.fn(),
    interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } },
    defaults: { headers: { common: {} } },
  };
  return { default: { create: () => inst, ...inst } };
});

const load = async () => {
  const mod = await import('./api');
  const axios = (await import('axios')).default;
  return { analysisApi: mod.analysisApi, post: axios.post, get: axios.get };
};

describe('analysisApi: the statistics phase', () => {
  beforeEach(() => vi.resetModules());

  it('grainSize always sends stats_phase: null means all phases, a missing key would keep the last phase', async () => {
    const { analysisApi, post } = await load();
    post.mockClear();
    await analysisApi.grainSize('mtex', 'cell_area');
    expect(post.mock.calls.at(-1)).toEqual(['/api/analysis/grain-size', { edge_rule: 'mtex', ecd_basis: 'cell_area', stats_phase: null }]);
    await analysisApi.grainSize('mtex', 'cell_area', 2);
    expect(post.mock.calls.at(-1)[1].stats_phase).toBe(2);
    await analysisApi.grainSize('keep_all', 'measured_pixels', 'most_measured');
    expect(post.mock.calls.at(-1)[1].stats_phase).toBe('most_measured');
  });

  it('reconstructGrains sends no min intercept when it is not given, and one when it is (0 included)', async () => {
    const { analysisApi, post } = await load();
    post.mockClear();
    await analysisApi.reconstructGrains(4, 3, undefined, 3.1, true);
    expect(JSON.parse(JSON.stringify(post.mock.calls.at(-1)[1]))).not.toHaveProperty('min_intercept_um');
    await analysisApi.reconstructGrains(4, 3, null, 3.1, true);
    expect(JSON.parse(JSON.stringify(post.mock.calls.at(-1)[1]))).not.toHaveProperty('min_intercept_um');
    await analysisApi.reconstructGrains(4, 3, 0, 3.1, true);
    expect(post.mock.calls.at(-1)[1].min_intercept_um).toBe(0);
  });

  it('reconstructGrains sends the phase with the other settings', async () => {
    const { analysisApi, post } = await load();
    post.mockClear();
    await analysisApi.reconstructGrains(4, 3, 1.5, 3.1, true, 'matlab_script', 'cell_area', 2);
    expect(post.mock.calls.at(-1)).toEqual(['/api/analysis/grains/reconstruct', {
      misorientation_threshold: 4, min_grain_size: 3, min_intercept_um: 1.5, alpha: 3.1, fill: true,
      edge_rule: 'matlab_script', ecd_basis: 'cell_area', stats_phase: 2,
    }]);
    await analysisApi.reconstructGrains();
    expect(post.mock.calls.at(-1)[1].stats_phase).toBeNull();
  });
});

describe('analysisApi: batch and download', () => {
  beforeEach(() => vi.resetModules());

  it('batch sends the phase choice and the reconstruction of the Grain tab', async () => {
    const { analysisApi, post } = await load();
    post.mockClear();
    await analysisApi.batch('F', 'O.xlsx', 'outer_row', 'measured_pixels', 'most_measured',
      { threshold: 6.5, minPixels: 5, minIntercept: 2, alpha: 0, fill: false });
    expect(post.mock.calls.at(-1)).toEqual(['/api/analysis/batch', {
      folder_path: 'F', output_path: 'O.xlsx', grain_edge_rule: 'outer_row', grain_ecd_basis: 'measured_pixels',
      grain_stats_phase: 'most_measured',
      misorientation_threshold: 6.5, min_grain_size: 5, alpha: 0, fill: false, min_intercept_um: 2,
    }]);
  });

  it('batch without the parameters leaves them out of the body (the batch\'s own defaults)', async () => {
    const { analysisApi, post } = await load();
    post.mockClear();
    await analysisApi.batch('F', 'O.xlsx', 'outer_row', 'cell_area');
    const body = JSON.parse(JSON.stringify(post.mock.calls.at(-1)[1]));
    expect(body).toEqual({
      folder_path: 'F', output_path: 'O.xlsx', grain_edge_rule: 'outer_row', grain_ecd_basis: 'cell_area', grain_stats_phase: null,
    });
  });

  it('batch sends the RX criteria of the Recrystallization tab, each under its own name', async () => {
    const { analysisApi, post } = await load();
    post.mockClear();
    await analysisApi.batch('F', 'O.xlsx', 'outer_row', 'cell_area', null, null,
      { gosThreshold: 2, gbcFraction: 0.5, gkamThreshold: 0.8, minRadius: 1.5 });
    const body = JSON.parse(JSON.stringify(post.mock.calls.at(-1)[1]));
    expect(body).toMatchObject({ rx_gos_threshold: 2, rx_gbc_fraction: 0.5, rx_gkam_threshold: 0.8, rx_min_radius_um: 1.5 });
  });

  it('batch without the RX criteria leaves them out of the body', async () => {
    const { analysisApi, post } = await load();
    post.mockClear();
    await analysisApi.batch('F', 'O.xlsx', 'outer_row', 'cell_area');
    const body = JSON.parse(JSON.stringify(post.mock.calls.at(-1)[1]));
    expect(Object.keys(body).filter((k) => k.startsWith('rx_'))).toEqual([]);
  });

  it('downloadExcel asks for a blob, so the header X-Orienta-Warnings can be read', async () => {
    const { analysisApi, get } = await load();
    get.mockClear();
    await analysisApi.downloadExcel('Run 3');
    expect(get.mock.calls.at(-1)).toEqual(['/api/analysis/export/excel-download', { params: { name: 'Run 3' }, responseType: 'blob' }]);
  });
});

describe('analysisApi: the RX criteria', () => {
  beforeEach(() => vi.resetModules());

  it('rxClassify posts the four numbers, each under its own name (a swapped pair would classify with the wrong one)', async () => {
    const { analysisApi, post } = await load();
    post.mockClear();
    await analysisApi.rxClassify({ gosThreshold: 1, gbcFraction: 2, gkamThreshold: 3, minRadius: 4 });
    expect(post.mock.calls.at(-1)).toEqual(['/api/analysis/rx', {
      rx_gos_threshold: 1, rx_gbc_fraction: 2, rx_gkam_threshold: 3, rx_min_radius_um: 4,
    }]);
  });
});

describe('analysisApi: the texture phase', () => {
  beforeEach(() => vi.resetModules());

  it('texture sends the phase choice, or nothing at all', async () => {
    const { analysisApi, post } = await load();
    post.mockClear();
    await analysisApi.texture('most_measured');
    expect(post.mock.calls.at(-1)).toEqual(['/api/analysis/texture', { phase: 'most_measured' }]);
    await analysisApi.texture(2);
    expect(post.mock.calls.at(-1)).toEqual(['/api/analysis/texture', { phase: 2 }]);
    await analysisApi.texture(0);
    expect(post.mock.calls.at(-1)[1]).toEqual({ phase: 0 });
    await analysisApi.texture();
    expect(post.mock.calls.at(-1)).toEqual(['/api/analysis/texture', undefined]);
  });

  it('texture sends the preset and the tolerance angle under the backend\'s names, when given', async () => {
    const { analysisApi, post } = await load();
    post.mockClear();
    await analysisApi.texture('most_measured', 'BCC_Rolling', 4.5);
    expect(post.mock.calls.at(-1)).toEqual(['/api/analysis/texture', { phase: 'most_measured', preset: 'BCC_Rolling', tolerance_deg: 4.5 }]);
    await analysisApi.texture(1, 'FCC_Rolling', undefined);
    expect(post.mock.calls.at(-1)[1]).toEqual({ phase: 1, preset: 'FCC_Rolling' });
  });

  it('batch sends the Texture tab\'s preset and tolerance, and nothing when it has none', async () => {
    const { analysisApi, post } = await load();
    post.mockClear();
    await analysisApi.batch('F', 'O.xlsx', 'outer_row', 'cell_area', null, null, null, { preset: 'BCC_Rolling', toleranceDeg: 6 });
    expect(JSON.parse(JSON.stringify(post.mock.calls.at(-1)[1]))).toMatchObject({ texture_preset: 'BCC_Rolling', texture_tolerance_deg: 6 });
    await analysisApi.batch('F', 'O.xlsx', 'outer_row', 'cell_area');
    const body = JSON.parse(JSON.stringify(post.mock.calls.at(-1)[1]));
    expect(body).not.toHaveProperty('texture_preset');
    expect(body).not.toHaveProperty('texture_tolerance_deg');
  });
});
