// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from 'vitest';

const mockPost = vi.fn();
const mockAxiosInstance = {
  post: mockPost,
  get: vi.fn(),
  interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } },
};
vi.mock('axios', () => ({ default: { create: vi.fn(() => mockAxiosInstance) } }));
const { analysisApi } = await import('../api');

describe('analysisApi.reconstructGrains', () => {
  beforeEach(() => mockPost.mockReset());

  it('sends alpha and fill with MTEX defaults', async () => {
    mockPost.mockResolvedValue({ data: {} });
    await analysisApi.reconstructGrains(5.0, 3, 1.5);
    expect(mockPost).toHaveBeenCalledWith('/api/analysis/grains/reconstruct', {
      misorientation_threshold: 5.0, min_grain_size: 3, min_intercept_um: 1.5, alpha: 3.1, fill: true,
      edge_rule: 'matlab_script', ecd_basis: 'cell_area', stats_phase: null,
    });
  });

  it('passes explicit values through', async () => {
    mockPost.mockResolvedValue({ data: {} });
    await analysisApi.reconstructGrains(5.0, 3, 1.5, 0, false);
    expect(mockPost.mock.calls[0][1]).toMatchObject({ alpha: 0, fill: false });
  });
});
