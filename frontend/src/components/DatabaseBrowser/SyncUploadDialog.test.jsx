// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { afterEach } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import SyncUploadDialog from './SyncUploadDialog';

const CATEGORIES = [
  { id: 'sht',        label: 'SHT',        uploadable: 27, both: 3 },
  { id: 'h5',         label: 'MC h5',      uploadable: 43, both: 9 },
  { id: 'master',     label: 'Master H5',  uploadable: 0,  both: 0 },
  { id: 'cif',        label: 'CIF',        uploadable: 0,  both: 30 },
  { id: 'xtal',       label: 'XTAL',       uploadable: 0,  both: 30 },
  { id: 'dictionary', label: 'Dictionary', uploadable: 6,  both: 0, defaultOff: true },
];

afterEach(() => { cleanup(); vi.clearAllMocks(); });

describe('SyncUploadDialog', () => {
  it('renders nothing when closed', () => {
    const { container } = render(<SyncUploadDialog open={false} categories={CATEGORIES} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('lists categories with upload/on-server counts', () => {
    render(<SyncUploadDialog open categories={CATEGORIES} onStart={() => {}} onCancel={() => {}} onClose={() => {}} />);
    expect(screen.getByText('SHT')).toBeInTheDocument();
    expect(screen.getByText('Master H5')).toBeInTheDocument();
    // SHT row: 27 to upload · 3 on server
    expect(screen.getByText(/27 to upload/)).toBeInTheDocument();
    expect(screen.getByText(/43 to upload/)).toBeInTheDocument();
  });

  it('default-selects categories with uploadable>0 and Start shows their total', () => {
    render(<SyncUploadDialog open categories={CATEGORIES} onStart={() => {}} onCancel={() => {}} onClose={() => {}} />);
    // 27 (sht) + 43 (h5) = 70; master/cif/xtal have 0 uploadable
    expect(screen.getByRole('button', { name: /Upload 70 file/ })).toBeInTheDocument();
  });

  it('does NOT default-select a defaultOff (dictionary) category', () => {
    const onStart = vi.fn();
    render(<SyncUploadDialog open categories={CATEGORIES} onStart={onStart} onCancel={() => {}} onClose={() => {}} />);
    // Dictionary (6 uploadable) is defaultOff → excluded from the 70 default.
    fireEvent.click(screen.getByRole('button', { name: /Upload 70 file/ }));
    expect(new Set(onStart.mock.calls[0][0])).toEqual(new Set(['sht', 'h5']));
  });

  it('can opt-in the dictionary category, adding it to the upload set', () => {
    const onStart = vi.fn();
    render(<SyncUploadDialog open categories={CATEGORIES} onStart={onStart} onCancel={() => {}} onClose={() => {}} />);
    // The dictionary row checkbox is the 6th category checkbox.
    const checkboxes = screen.getAllByRole('checkbox');
    fireEvent.click(checkboxes[5]); // dictionary row
    fireEvent.click(screen.getByRole('button', { name: /Upload 76 file/ }));
    expect(new Set(onStart.mock.calls[0][0])).toEqual(new Set(['sht', 'h5', 'dictionary']));
  });

  it('calls onStart with the selected category ids and overwrite flag', () => {
    const onStart = vi.fn();
    render(<SyncUploadDialog open categories={CATEGORIES} onStart={onStart} onCancel={() => {}} onClose={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: /Upload 70 file/ }));
    expect(onStart).toHaveBeenCalledTimes(1);
    const [ids, overwrite] = onStart.mock.calls[0];
    expect(new Set(ids)).toEqual(new Set(['sht', 'h5']));
    expect(overwrite).toBe(false);
  });

  it('passes overwrite=true when the overwrite checkbox is ticked', () => {
    const onStart = vi.fn();
    render(<SyncUploadDialog open categories={CATEGORIES} onStart={onStart} onCancel={() => {}} onClose={() => {}} />);
    // The overwrite checkbox is the last checkbox (after the 5 category rows)
    const checkboxes = screen.getAllByRole('checkbox');
    fireEvent.click(checkboxes[checkboxes.length - 1]);
    fireEvent.click(screen.getByRole('button', { name: /Upload 70 file/ }));
    expect(onStart.mock.calls[0][1]).toBe(true);
  });

  it('shows a progress bar + Cancel while running', () => {
    const onCancel = vi.fn();
    render(
      <SyncUploadDialog
        open categories={CATEGORIES}
        running progress={{ current: 12, total: 70, name: 'Al.sht' }}
        onStart={() => {}} onCancel={onCancel} onClose={() => {}}
      />,
    );
    expect(screen.getByText(/Uploading 12\/70/)).toBeInTheDocument();
    const cancel = screen.getByRole('button', { name: /Cancel/ });
    fireEvent.click(cancel);
    expect(onCancel).toHaveBeenCalled();
  });

  it('shows a result summary when done', () => {
    render(
      <SyncUploadDialog
        open categories={CATEGORIES}
        result={{ uploaded: 70, upToDate: 0, conflicts: 0, errors: 0 }}
        onStart={() => {}} onCancel={() => {}} onClose={() => {}}
      />,
    );
    expect(screen.getByText('70')).toBeInTheDocument();
    expect(screen.getByText(/uploaded/)).toBeInTheDocument();
  });
});
