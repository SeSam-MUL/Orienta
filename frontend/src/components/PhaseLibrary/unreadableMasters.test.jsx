// @vitest-environment jsdom
/**
 * A master the library refuses to offer is said out loud.
 *
 * After the Dictionary gate started refusing masters
 * that kikuchipy cannot load (EMsoft's GPU master program writes them), the
 * "unassigned masters" list was built from the readable ones only, so a master
 * the user had just simulated was neither offered nor reported -- the phase
 * simply looked as if it had none. The backend now reports them under
 * `unreadable_masters` (code `masterUnreadableByKikuchipy`); this is the page
 * side of it.
 */
import { describe, it, expect, afterEach } from 'vitest';
import { render, screen, cleanup, act } from '@testing-library/react';
import i18n from '../../i18n';
import fixture from './__fixtures__/library.json';
import PhaseLibraryPage from './PhaseLibraryPage';

afterEach(cleanup);

const refused = (name) => ({
  name, rel: `Database/EBSD_H5_Cache/Mg/${name}`, bytes: 210061320, date: '2026-10-12',
  program: 'EMEBSDmasterOpenCL.f90', code: 'masterUnreadableByKikuchipy', phase_key: 'Mg_COD-9008506',
});

async function show(unreadable, lang = 'en') {
  await act(async () => { await i18n.changeLanguage(lang); });
  const payload = { phases: fixture.phases, unassigned_masters: [], unreadable_masters: unreadable };
  render(<PhaseLibraryPage loadIndex={() => Promise.resolve(payload)} />);
  await act(async () => { await Promise.resolve(); });
}

describe('unreadable masters', () => {
  it('names the file and says what happens to it', async () => {
    await show([refused('Mg_COD-9008506_master_E20kV_npx500.h5')]);
    const note = await screen.findByTestId('unreadable-masters');
    expect(note.textContent).toContain('Mg_COD-9008506_master_E20kV_npx500.h5');
    expect(note.textContent).toMatch(/cannot be used for Dictionary indexing/);
    expect(note.textContent).toMatch(/not offered/);
  });

  it('counts several files and lists every name', async () => {
    await show([refused('a_master.h5'), refused('b_master.h5')]);
    const note = await screen.findByTestId('unreadable-masters');
    expect(note.textContent).toMatch(/^2 simulated masters/);
    expect(note.textContent).toContain('a_master.h5, b_master.h5');
  });

  it('is silent when every master can be read', async () => {
    await show([]);
    await screen.findByTestId('facet-counts');
    expect(screen.queryByTestId('unreadable-masters')).toBeNull();
  });

  it('speaks German to a German user, in the polite form', async () => {
    await show([refused('x_master.h5')], 'de');
    const note = await screen.findByTestId('unreadable-masters');
    expect(note.textContent).toMatch(/Simulieren Sie/);
    expect(note.textContent).toContain('x_master.h5');
  });
});
