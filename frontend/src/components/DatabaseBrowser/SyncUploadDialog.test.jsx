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
        result={{ transferred: 70, upToDate: 0, conflicts: 0, errors: 0 }}
        onStart={() => {}} onCancel={() => {}} onClose={() => {}}
      />,
    );
    expect(screen.getByText('70')).toBeInTheDocument();
    expect(screen.getByText(/uploaded/)).toBeInTheDocument();
  });
});

// --- Download mode (server -> local) -------------------------------------
const DL_CATEGORIES = [
  { id: 'sht',        label: 'SHT',        downloadable: 10, both: 2, uploadable: 0 },
  { id: 'h5',         label: 'MC h5',      downloadable: 5,  both: 0, uploadable: 0, downloadOff: true },
  { id: 'master',     label: 'Master H5',  downloadable: 8,  both: 1, uploadable: 0 },
  { id: 'cif',        label: 'CIF',        downloadable: 30, both: 0, uploadable: 0 },
  { id: 'xtal',       label: 'XTAL',       downloadable: 30, both: 0, uploadable: 0 },
  { id: 'dictionary', label: 'Dictionary', downloadable: 6,  both: 0, uploadable: 0, defaultOff: true },
];

describe('SyncUploadDialog (download mode)', () => {
  it('lists categories with download/already-local counts', () => {
    render(<SyncUploadDialog open mode="download" categories={DL_CATEGORIES} onStart={() => {}} onCancel={() => {}} onClose={() => {}} />);
    // SHT row: 10 to download · 2 already local
    expect(screen.getByText(/10 to download/)).toBeInTheDocument();
    expect(screen.getByText(/2 already local/)).toBeInTheDocument();
  });

  it('default-selects SHT+Master+CIF+XTAL but NOT MC h5 (downloadOff) or Dictionary (defaultOff)', () => {
    const onStart = vi.fn();
    render(<SyncUploadDialog open mode="download" categories={DL_CATEGORIES} onStart={onStart} onCancel={() => {}} onClose={() => {}} />);
    // 10 (sht) + 8 (master) + 30 (cif) + 30 (xtal) = 78; h5 & dictionary excluded
    fireEvent.click(screen.getByRole('button', { name: /Download 78 file/ }));
    expect(new Set(onStart.mock.calls[0][0])).toEqual(new Set(['sht', 'master', 'cif', 'xtal']));
  });

  it('can opt-in MC h5, adding it to the download set', () => {
    const onStart = vi.fn();
    render(<SyncUploadDialog open mode="download" categories={DL_CATEGORIES} onStart={onStart} onCancel={() => {}} onClose={() => {}} />);
    // MC h5 is the 2nd category checkbox
    const checkboxes = screen.getAllByRole('checkbox');
    fireEvent.click(checkboxes[1]);
    fireEvent.click(screen.getByRole('button', { name: /Download 83 file/ }));
    expect(new Set(onStart.mock.calls[0][0])).toEqual(new Set(['sht', 'master', 'cif', 'xtal', 'h5']));
  });

  it('shows a download progress label + download result summary', () => {
    const { rerender } = render(
      <SyncUploadDialog open mode="download" categories={DL_CATEGORIES}
        running progress={{ current: 3, total: 78, name: 'Al.sht' }}
        onStart={() => {}} onCancel={() => {}} onClose={() => {}} />,
    );
    expect(screen.getByText(/Downloading 3\/78/)).toBeInTheDocument();
    rerender(
      <SyncUploadDialog open mode="download" categories={DL_CATEGORIES}
        result={{ transferred: 78, upToDate: 0, conflicts: 0, errors: 0 }}
        onStart={() => {}} onCancel={() => {}} onClose={() => {}} />,
    );
    const countEl = screen.getByText('78');
    expect(countEl).toBeInTheDocument();
    // "downloaded" also appears in the subtitle, so scope to the result label.
    expect(countEl.parentElement).toHaveTextContent('downloaded');
  });
});
