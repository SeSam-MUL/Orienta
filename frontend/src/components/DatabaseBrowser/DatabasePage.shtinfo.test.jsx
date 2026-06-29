// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ShtInfoBlocks } from './DatabasePage';

vi.mock('../../services/api', () => ({
  dbApi: { shtInfo: vi.fn().mockResolvedValue({ data: {
    provenance: {
      source_xtal: { name: 'Ni.xtal', found: true },
      source_cif: { name: 'Ni.cif', found: true },
      reference: '10.17188/1316752', origin: 'recovered',
    },
    parameters: { dmin: 0.05, npx: 500, voltage_kV: 20, source: 'sht_binary' },
  } }) },
}));

describe('ShtInfoBlocks', () => {
  it('renders parent xtal/cif and a parameter', async () => {
    render(<ShtInfoBlocks filename="Ni (Ni) [cF4] {20kV}.sht" isLocal />);
    expect(await screen.findByText('Ni.xtal')).toBeInTheDocument();
    expect(screen.getByText('Ni.cif')).toBeInTheDocument();
  });

  it('shows the unknown label when source not found', async () => {
    const { dbApi } = await import('../../services/api');
    dbApi.shtInfo.mockResolvedValueOnce({ data: {
      provenance: { source_xtal: { name: null, found: false },
                    source_cif: { name: null, found: false }, reference: '', origin: 'recovered' },
      parameters: { source: 'unknown' },
    }});
    render(<ShtInfoBlocks filename="x.sht" isLocal />);
    // UI now defaults to English via i18n (was a hardcoded German "unbekannt").
    expect(await screen.findAllByText('unknown')).toBeTruthy();
  });
});
