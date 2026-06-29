// @vitest-environment jsdom
import { describe, it, expect, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';

const image = vi.fn(() => Promise.resolve({ data: { image: 'BASE64', phase_name: 'Al' } }));
vi.mock('../../services/api', () => ({ poleFigureApi: { image: (a) => image(a) } }));

import { usePoleFigure } from './usePoleFigure';

describe('usePoleFigure', () => {
  it('fetches and exposes the image', async () => {
    const { result } = renderHook(() => usePoleFigure({ phaseId: 0, hkl: '100,110,111', mode: 'both', refreshKey: 1 }));
    await waitFor(() => expect(result.current.image).toBe('BASE64'));
    expect(image).toHaveBeenCalledWith({ phaseId: 0, hkl: '100,110,111', mode: 'both', subsample: 20000 });
  });
});
