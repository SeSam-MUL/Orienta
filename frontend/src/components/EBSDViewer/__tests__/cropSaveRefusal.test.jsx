// @vitest-environment jsdom
/**
 * The viewer, mounted: saving a crop whose file layout cannot name the pattern source the viewer
 * holds is refused by the backend with a load code (HTTP 400 and the `X-Orienta-Code` header).
 * The log of the viewer shows the translated sentence of that code, not the server's English prose
 * and not "Request failed with status code 400".
 *
 * `variantCodeSurfaces.test.jsx` pins the wording and, for this page, only the call in the source;
 * this mounts the page with the API stubbed (a loaded dataset that is a crop, the native save dialog
 * answering a path) and clicks "Save crop".
 */
import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import i18n from '../../../i18n';
import useDataStore from '../../../stores/useDataStore';

const PROSE = 'export_crop cannot read the processed patterns; English prose from the server';
const refused = (code) => ({
  message: 'Request failed with status code 400',
  response: { status: 400, headers: { 'x-orienta-code': code }, data: { detail: PROSE } },
});

const mocks = vi.hoisted(() => ({ exportCrop: vi.fn() }));

vi.mock('../../../services/api', async (importOriginal) => {
  const actual = await importOriginal();
  const empty = () => Promise.resolve({ data: {} });
  const group = (overrides = {}) => new Proxy({}, {
    get: (_t, name) => (name in overrides ? overrides[name] : empty),
  });
  return {
    ...actual,
    ebsdApi: group({
      exportCrop: (...a) => mocks.exportCrop(...a),
      // the active dataset IS a crop: the Save button is offered for it
      getCrop: () => Promise.resolve({ data: { window: {
        source_file: 'scan.h5oina', row0: 0, col0: 0, rows: 2, cols: 3, original_shape: [4, 6] } } }),
    }),
    edsApi: group(),
    pcApi: group(),
    scanGeometryApi: group(),
  };
});

import EBSDViewer from '../EBSDViewer';

const COMMON = {
  en: (await import('../../../locales/en/common.json')).default,
  de: (await import('../../../locales/de/common.json')).default,
};

beforeEach(async () => {
  globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
  window.electronAPI = { saveFile: vi.fn().mockResolvedValue('C:/out/crop.h5oina') };
  mocks.exportCrop.mockReset();
  useDataStore.setState({
    ebsdLoaded: true,
    ebsdInfo: { navigation_shape: [3, 2], signal_shape: [8, 8], pattern_shape: [8, 8] },
    gridShape: [2, 3],
  });
  await i18n.changeLanguage('en');
});
afterEach(() => {
  cleanup();
  delete window.electronAPI;
  useDataStore.setState({ ebsdLoaded: false, ebsdInfo: null });
});

async function save(lang, code) {
  await i18n.changeLanguage(lang);
  mocks.exportCrop.mockRejectedValue(refused(code));
  render(<EBSDViewer isActive onNavigate={() => {}} />);
  const button = await screen.findByTestId('crop-export', {}, { timeout: 4000 });
  fireEvent.click(button);
  await waitFor(() => expect(mocks.exportCrop).toHaveBeenCalledWith('C:/out/crop.h5oina'),
    { timeout: 4000 });
}

describe('EBSD viewer, mounted', () => {
  for (const lang of ['en', 'de']) {
    it(`${lang}: a refused crop save shows the sentence of its code in the log`, async () => {
      await save(lang, 'variantNotSupported');
      await waitFor(() => expect(document.body.textContent).toContain(COMMON[lang].errors.variantNotSupported),
        { timeout: 4000 });
      expect(document.body.textContent).not.toContain('English prose');
      expect(document.body.textContent).not.toContain('Request failed with status code');
    });
  }
});

// --- a crop of a hexagonal scan is saved as an Orienta .h5 ------------------------------------

const EBSDVIEWER = {
  en: (await import('../../../locales/en/ebsdviewer.json')).default,
  de: (await import('../../../locales/de/ebsdviewer.json')).default,
};

const HEX_GEOMETRY = {
  kind: 'hex', n_rows: 2, n_cols: 3, grid_shape: [2, 3], step_x: 1, step_y: 0.8660254, steps_known: true,
  odd_row_shift: 0.5, shift_confirmed: true, has_valid_mask: false, n_valid: 6,
};

/** Mount the viewer on a crop of a hexagonal scan and press "Save crop"; the save dialog answers `path`. */
async function saveHex(lang, settle, path = 'C:/out/crop.h5') {
  await i18n.changeLanguage(lang);
  useDataStore.setState({ ebsdInfo: { ...useDataStore.getState().ebsdInfo, scan_geometry: HEX_GEOMETRY } });
  window.electronAPI.saveFile.mockResolvedValue(path);
  settle();
  render(<EBSDViewer isActive onNavigate={() => {}} />);
  fireEvent.click(await screen.findByTestId('crop-export', {}, { timeout: 4000 }));
  await waitFor(() => expect(mocks.exportCrop).toHaveBeenCalledWith(path), { timeout: 4000 });
}

describe('Save crop of a hexagonal scan', () => {
  it('suggests an .h5 name and offers the .h5 filter, not .h5oina', async () => {
    await saveHex('en', () => mocks.exportCrop.mockResolvedValue({ data: { hex: true, n_points: 6 } }));
    const args = window.electronAPI.saveFile.mock.calls[0][0];
    expect(args.defaultPath).toMatch(/\.h5$/);
    expect(args.filters[0].extensions).toEqual(['h5']);
  });

  it('a square crop still suggests .h5oina', async () => {
    mocks.exportCrop.mockResolvedValue({ data: { n_points: 6 } });
    render(<EBSDViewer isActive onNavigate={() => {}} />);
    fireEvent.click(await screen.findByTestId('crop-export', {}, { timeout: 4000 }));
    await waitFor(() => expect(mocks.exportCrop).toHaveBeenCalled(), { timeout: 4000 });
    const args = window.electronAPI.saveFile.mock.calls[0][0];
    expect(args.defaultPath).toMatch(/\.h5oina$/);
    expect(args.filters[0].extensions).toEqual(['h5oina']);
  });

  it('hints at the .h5 file, not at EDS maps and electron images, on the Save crop button', async () => {
    await saveHex('en', () => mocks.exportCrop.mockResolvedValue({ data: { hex: true, n_points: 6 } }));
    const hint = screen.getByTestId('crop-export').title;
    expect(hint).toBe(EBSDVIEWER.en.crop.saveCropHintHex);
    expect(hint).not.toContain('cut to match');
  });

  for (const [kind, geometry, placeholder] of [
    ['hexagonal', HEX_GEOMETRY, '/path/to/crop.h5'],
    ['square', undefined, '/path/to/crop.h5oina'],
  ]) {
    it(`the typed-path prompt of a ${kind} crop shows ${placeholder}`, async () => {
      window.electronAPI = {};                                       // browser mode: no native dialog
      if (geometry) useDataStore.setState({ ebsdInfo: { ...useDataStore.getState().ebsdInfo, scan_geometry: geometry } });
      render(<EBSDViewer isActive onNavigate={() => {}} />);
      fireEvent.click(await screen.findByTestId('crop-export', {}, { timeout: 4000 }));
      expect(await screen.findByPlaceholderText(placeholder, {}, { timeout: 4000 })).toBeTruthy();
    });
  }

  for (const lang of ['en', 'de']) {
    it(`${lang}: the log says what was kept and what was not carried`, async () => {
      await saveHex(lang, () => mocks.exportCrop.mockResolvedValue({ data: {
        hex: true, path: 'C:/out/crop.h5', n_points: 9, n_selected: 7, grid_shape: [3, 3],
        odd_row_shift: -0.5, pc_map_dropped: true, patterns_converted: false,
        note_items: [{ code: 'cropEdsNotCarried' }, { code: 'cropPcMapDropped' }, { code: 'cropMaskRecorded' }],
      } }));
      const crop = EBSDVIEWER[lang].crop;
      const fill = (text, values) => text.replace(/\{\{(\w+)\}\}/g, (_m, k) => values[k]);
      // The collapsed log shows its last line only: read all of it through the copy button.
      const copied = [];
      Object.defineProperty(navigator, 'clipboard', {
        configurable: true,
        value: { writeText: (text) => { copied.push(text); return Promise.resolve(); } },
      });
      await waitFor(() => {
        fireEvent.click(screen.getByLabelText(EBSDVIEWER[lang].log.copyAriaLabel));
        expect(copied.at(-1)).toContain(fill(crop.saveCropHexDetail, { rows: 3, cols: 3, shift: '-0.5' }));
      }, { timeout: 4000 });
      const body = copied.at(-1);
      expect(body).toContain(crop.saveCropHexNotCarried);
      expect(body).toContain(crop.saveCropHexPcMapDropped);
      expect(body).toContain(fill(crop.saveCropHexMaskNote, { points: 9 }));
      expect(body).not.toContain(crop.saveCropHexConverted);
      // the square wording would claim electron images were cut
      expect(body).not.toContain(fill(crop.saveCropDetail, { cut: 0, copied: 0, images: 0 }));
    });

    it(`${lang}: a path that is not .h5 is refused with the sentence of its code`, async () => {
      await saveHex(lang, () => mocks.exportCrop.mockRejectedValue(refused('cropHexExtension')),
        'C:/out/crop.h5oina');
      await waitFor(() => expect(document.body.textContent).toContain(EBSDVIEWER[lang].errors.cropHexExtension),
        { timeout: 4000 });
      expect(document.body.textContent).not.toContain('English prose');
    });

    for (const code of ['cropHexValidMask', 'cropHexNoDetector']) {
      it(`${lang}: a hexagonal crop refused with ${code} shows its sentence, not the server's prose`, async () => {
        await saveHex(lang, () => mocks.exportCrop.mockRejectedValue(refused(code)));
        await waitFor(() => expect(document.body.textContent).toContain(EBSDVIEWER[lang].errors[code]),
          { timeout: 4000 });
        expect(document.body.textContent).not.toContain('English prose');
      });
    }
  }

  it('words both refusals of a hexagonal crop in all four languages', async () => {
    for (const lang of ['en', 'de', 'ja', 'zh']) {
      const errors = (await import(`../../../locales/${lang}/ebsdviewer.json`)).default.errors;
      for (const code of ['cropHexValidMask', 'cropHexNoDetector']) {
        expect(typeof errors[code], `${lang} ${code}`).toBe('string');
        expect(errors[code].length, `${lang} ${code}`).toBeGreaterThan(30);
      }
    }
  });
});
