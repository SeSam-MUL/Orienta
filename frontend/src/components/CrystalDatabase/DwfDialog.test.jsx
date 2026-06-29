// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach } from 'vitest';
import { cleanup, render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

const getDwf = vi.fn();
const updateDwf = vi.fn();
vi.mock('../../services/api', () => ({
  dbApi: {
    getDwf: (...a) => getDwf(...a),
    updateDwf: (...a) => updateDwf(...a),
  },
}));

import DwfDialog from './DwfDialog';

beforeEach(() => {
  getDwf.mockResolvedValue({ data: { entries: [
    { 'Element': 'Al', 'DWB 300 K': 0.007993, 'Reference': 'unverified (legacy)' },
    { 'Element': 'Fe (BCC)', 'DWB 300 K': 0.005771, 'Reference': '' },
  ] } });
  updateDwf.mockResolvedValue({ data: { updated: [], added: ['Cu'] } });
});
afterEach(() => { cleanup(); vi.clearAllMocks(); });

describe('DwfDialog', () => {
  it('loads entries with values and references', async () => {
    render(<DwfDialog onClose={() => {}} />);
    expect(await screen.findByDisplayValue('0.007993')).toBeInTheDocument();
    expect(screen.getByDisplayValue('unverified (legacy)')).toBeInTheDocument();
    expect(screen.getByText('Al')).toBeInTheDocument();
    expect(screen.getByText('Fe (BCC)')).toBeInTheDocument();
  });

  it('saves an edited value with its reference (upsert)', async () => {
    render(<DwfDialog onClose={() => {}} />);
    const dwb = await screen.findByDisplayValue('0.007993');
    fireEvent.change(dwb, { target: { value: '0.0081' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save' }));
    await waitFor(() => expect(updateDwf).toHaveBeenCalledTimes(1));
    const updates = updateDwf.mock.calls[0][0];
    const al = updates.find(u => u.element === 'Al');
    expect(al).toMatchObject({ element: 'Al', dwb_300k: 0.0081, reference: 'unverified (legacy)' });
  });

  it('adds a new element (Cu) and saves it', async () => {
    render(<DwfDialog onClose={() => {}} />);
    await screen.findByDisplayValue('0.007993');
    fireEvent.change(screen.getByPlaceholderText('Element (Cu)'), { target: { value: 'Cu' } });
    fireEvent.change(screen.getByPlaceholderText('B [nm²]'), { target: { value: '0.0055' } });
    fireEvent.change(screen.getByPlaceholderText('Reference'), { target: { value: 'Peng 1996' } });
    fireEvent.click(screen.getByRole('button', { name: /\+ Add/ }));
    // new editable element input shows up
    expect(screen.getByDisplayValue('Cu')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Save' }));
    await waitFor(() => expect(updateDwf).toHaveBeenCalledTimes(1));
    const updates = updateDwf.mock.calls[0][0];
    expect(updates).toContainEqual({ element: 'Cu', dwb_300k: 0.0055, reference: 'Peng 1996' });
  });

  it('refuses to add a duplicate element', async () => {
    render(<DwfDialog onClose={() => {}} />);
    await screen.findByDisplayValue('0.007993');
    fireEvent.change(screen.getByPlaceholderText('Element (Cu)'), { target: { value: 'Al' } });
    fireEvent.change(screen.getByPlaceholderText('B [nm²]'), { target: { value: '0.008' } });
    fireEvent.click(screen.getByRole('button', { name: /\+ Add/ }));
    expect(screen.getByText(/already in the table/)).toBeInTheDocument();
    expect(updateDwf).not.toHaveBeenCalled();
  });

  it('Save is disabled with no changes', async () => {
    render(<DwfDialog onClose={() => {}} />);
    await screen.findByDisplayValue('0.007993');
    expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled();
  });
});
