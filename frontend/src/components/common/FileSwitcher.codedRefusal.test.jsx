// @vitest-environment jsdom
// A switch reloads the file, so the backend can refuse it with a code: a measurement or pattern source
// the file does not hold any more, an APEX pairing that conflicts with the
// pattern file. The header switcher shows the refusal in the interface's language, not
// in the backend's English. Real FileSwitcher, real stores, real i18n; the only seam is the API.
import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup, act } from '@testing-library/react';

const refusal = (code, prose) => Object.assign(new Error('Request failed with status code 400'), {
  response: { status: 400, headers: code ? { 'x-orienta-code': code } : {}, data: { detail: prose } },
});

vi.mock('../../services/api', () => ({
  ebsdApi: { loadedFiles: vi.fn(), switchFile: vi.fn() },
  isSupersededLoad: () => false,
}));

import i18n from '../../i18n';
import { ebsdApi } from '../../services/api';
import useDataStore from '../../stores/useDataStore';
import useLoadedFilesStore from '../../stores/useLoadedFilesStore';
import FileSwitcher from './FileSwitcher';
import commonEn from '../../locales/en/common.json';
import commonDe from '../../locales/de/common.json';
import commonJa from '../../locales/ja/common.json';
import commonZh from '../../locales/zh/common.json';
import viewerEn from '../../locales/en/ebsdviewer.json';
import viewerDe from '../../locales/de/ebsdviewer.json';
import viewerJa from '../../locales/ja/ebsdviewer.json';
import viewerZh from '../../locales/zh/ebsdviewer.json';

const PROSE = 'BACKEND ENGLISH SENTENCE';
// the file's own text of each code, per language: one from the shared sentences, one from the viewer's page
const TEXT = {
  en: { loadVariantUnknown: commonEn.errors.loadVariantUnknown, apexGuidMismatch: viewerEn.errors.apexGuidMismatch },
  de: { loadVariantUnknown: commonDe.errors.loadVariantUnknown, apexGuidMismatch: viewerDe.errors.apexGuidMismatch },
  ja: { loadVariantUnknown: commonJa.errors.loadVariantUnknown, apexGuidMismatch: viewerJa.errors.apexGuidMismatch },
  zh: { loadVariantUnknown: commonZh.errors.loadVariantUnknown, apexGuidMismatch: viewerZh.errors.apexGuidMismatch },
};

async function switchAndFail(code) {
  ebsdApi.loadedFiles.mockResolvedValue({ data: { files: [
    { path: '/data/fileA.h5oina', name: 'fileA', active: true },
    { path: '/data/fileB.h5oina', name: 'fileB', active: false },
  ] } });
  ebsdApi.switchFile.mockRejectedValue(refusal(code, PROSE));
  const { container } = render(<FileSwitcher />);
  await screen.findByText('fileA');
  fireEvent.change(container.querySelector('select'), { target: { value: '/data/fileB.h5oina' } });
  await waitFor(() => expect(ebsdApi.switchFile).toHaveBeenCalled());
  return container;
}

beforeEach(() => {
  vi.clearAllMocks();
  useLoadedFilesStore.setState({ files: [] });
  useDataStore.setState({ filePath: '/data/fileA.h5oina', isFileOpen: true });
});

afterEach(async () => {
  cleanup();
  await act(async () => { await i18n.changeLanguage('en'); });
});

describe('a refused switch in the header file switcher', () => {
  it.each(Object.keys(TEXT))('says the refusal in %s and not in the backend English', async (lang) => {
    await act(async () => { await i18n.changeLanguage(lang); });
    for (const code of ['loadVariantUnknown', 'apexGuidMismatch']) {
      cleanup();
      const container = await switchAndFail(code);
      await waitFor(() => expect(container.textContent).toContain(TEXT[lang][code]));
      expect(container.textContent).not.toContain(PROSE);
    }
  });

  it('shows the backend sentence when the refusal carries no code', async () => {
    const container = await switchAndFail(null);
    await waitFor(() => expect(container.textContent).toContain(PROSE));
  });
});
