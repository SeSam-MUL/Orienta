// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor, within }
  from '@testing-library/react';

// Mocked at the module boundary rather than with a fake axios: what is under
// test is the page's behaviour against the six named endpoints, and a test
// that also had to model axios would pass or fail for reasons of its own.
vi.mock('../../services/addonsApi', () => ({
  addonsApi: {
    list: vi.fn(),
    setEnabled: vi.fn(),
    runJob: vi.fn(),
    job: vi.fn(),
    jobResult: vi.fn(),
    mapImage: vi.fn(),
  },
}));

import { addonsApi } from '../../services/addonsApi';
import AddonsPage from './AddonsPage';

afterEach(cleanup);

const GOOD = {
  name: 'bc-gmm', display_name: 'Band contrast mixture model',
  version: '0.1.0', authors: ['A Researcher <a@example.org>'],
  doi: '10.5281/zenodo.1234567', origin: 'directory',
  source_path: 'C:/Users/x/.orienta/addons/bc-gmm',
  enabled: false, known: false, compatible: true, compatibility: '',
  conflict: '', error: '', analyses: [{ key: 'addon.bc_gmm', label: 'BC GMM' }],
  crashes: 0, crash_limit: 3, disabled_by: null,
};

function listing(addons, searched = ['C:/Users/x/.orienta/addons']) {
  return { data: { api_version: 0, orienta_version: '0.4.4',
                   searched_paths: searched, addons } };
}

beforeEach(() => {
  vi.clearAllMocks();
  addonsApi.list.mockResolvedValue(listing([GOOD]));
  addonsApi.setEnabled.mockResolvedValue({ data: {} });
});

describe('the empty state', () => {
  it('names the folder it searched and offers a rescan', async () => {
    // The first thing most people meet. Before this the app stated nowhere
    // where an add-on has to be put.
    addonsApi.list.mockResolvedValue(listing([], ['D:/shared/addons']));
    render(<AddonsPage />);
    expect(await screen.findByTestId('addons-empty')).toBeTruthy();
    expect(screen.getByText('D:/shared/addons')).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: /rescan/i }));
    await waitFor(() => expect(addonsApi.list).toHaveBeenCalledTimes(2));
  });

  it('shows an add-on that appears after a rescan', async () => {
    addonsApi.list.mockResolvedValueOnce(listing([]));
    addonsApi.list.mockResolvedValueOnce(listing([GOOD]));
    render(<AddonsPage />);
    await screen.findByTestId('addons-empty');
    fireEvent.click(screen.getByRole('button', { name: /rescan/i }));
    expect(await screen.findByText('Band contrast mixture model')).toBeTruthy();
  });
});

describe('what a row says', () => {
  it('renders the author’s display name and the third-party disclosure', async () => {
    render(<AddonsPage />);
    expect(await screen.findByText('Band contrast mixture model')).toBeTruthy();
    expect(screen.getByText(/did not write it/i)).toBeTruthy();
  });

  it('says "no DOI" in words, because "" and missing look the same', async () => {
    addonsApi.list.mockResolvedValue(listing([{ ...GOOD, doi: '' }]));
    render(<AddonsPage />);
    expect(await screen.findByText(/No DOI declared/i)).toBeTruthy();
  });

  it('shows a rejected manifest AS rejected, with its error and path', async () => {
    addonsApi.list.mockResolvedValue(listing([{
      ...GOOD, name: '', display_name: 'C:/addons/broken',
      error: 'declares no analyses', analyses: [],
      source_path: 'C:/addons/broken',
    }]));
    render(<AddonsPage />);
    // Shown, not dropped: an installed add-on that simply does not appear is
    // indistinguishable from one that was never installed.
    expect(await screen.findByText(/rejected/i)).toBeTruthy();
    // getAllBy: a rejected row has no name, so the backend puts its PATH in
    // display_name — the path is on the row twice, deliberately.
    expect(screen.getAllByText(/C:\/addons\/broken/).length).toBeGreaterThan(0);
    expect(screen.queryByRole('button', { name: /^enable$/i })).toBeNull();
  });

  it('shows a conflict and refuses to enable', async () => {
    addonsApi.list.mockResolvedValue(listing([{
      ...GOOD, conflict: 'another add-on also declares addon.bc_gmm',
    }]));
    render(<AddonsPage />);
    expect(await screen.findByText(/also declares/)).toBeTruthy();
    expect(screen.getByRole('button', { name: /enable/i }).disabled).toBe(true);
  });

  it('shows an incompatible add-on’s requirement and refuses to enable', async () => {
    addonsApi.list.mockResolvedValue(listing([{
      ...GOOD, compatible: false, compatibility: 'needs Orienta >=99',
    }]));
    render(<AddonsPage />);
    expect(await screen.findByText(/>=99/)).toBeTruthy();
    expect(screen.getByRole('button', { name: /enable/i }).disabled).toBe(true);
  });

  it('says when ORIENTA switched it off, and how often it failed', async () => {
    // Otherwise the switch is simply off for no visible reason, and the user
    // assumes it was their own doing.
    addonsApi.list.mockResolvedValue(listing([{
      ...GOOD, enabled: false, known: true, crashes: 3, disabled_by: 'runtime',
    }]));
    render(<AddonsPage />);
    expect(await screen.findByText(/Switched off by Orienta/i)).toBeTruthy();
    expect(screen.getByText(/3 of 3/)).toBeTruthy();
  });

  it('does not claim Orienta switched off one the user did', async () => {
    addonsApi.list.mockResolvedValue(listing([{
      ...GOOD, enabled: false, known: true, crashes: 0, disabled_by: 'user',
    }]));
    render(<AddonsPage />);
    await screen.findByText('Band contrast mixture model');
    expect(screen.queryByText(/Switched off by Orienta/i)).toBeNull();
  });

  it('says that a changed add-on takes effect after a restart', async () => {
    // What the runtime actually does: the step and library registries are
    // append-only for the process, on purpose.
    render(<AddonsPage />);
    expect(await screen.findByText(/after a restart/i)).toBeTruthy();
  });
});

describe('consent', () => {
  it('asks before the first enable, and only enables on acceptance', async () => {
    render(<AddonsPage />);
    fireEvent.click(await screen.findByRole('button', { name: /enable/i }));

    const dialog = await screen.findByRole('dialog');
    expect(dialog.textContent).toContain('Band contrast mixture model');
    expect(dialog.textContent).toContain('0.1.0');
    expect(dialog.textContent).toContain('A Researcher <a@example.org>');
    expect(dialog.textContent).toContain('10.5281/zenodo.1234567');
    expect(dialog.textContent).toContain('C:/Users/x/.orienta/addons/bc-gmm');
    // The switch has NOT been thrown yet.
    expect(addonsApi.setEnabled).not.toHaveBeenCalled();

    // Scoped to the dialog: the row behind it has an Enable button too, and
    // an unscoped query cannot tell which one the user would be pressing.
    fireEvent.click(within(dialog).getByRole('button', { name: /^enable$/i }));
    await waitFor(() =>
      expect(addonsApi.setEnabled).toHaveBeenCalledWith('bc-gmm', true));
  });

  it('cancelling enables nothing', async () => {
    render(<AddonsPage />);
    fireEvent.click(await screen.findByRole('button', { name: /enable/i }));
    await screen.findByRole('dialog');
    fireEvent.click(screen.getByRole('button', { name: /cancel/i }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(addonsApi.setEnabled).not.toHaveBeenCalled();
  });

  it('ASKS AGAIN when the add-on is not the one that was allowed', async () => {
    // A name is not an identity. Delete grain-stats, drop a colleague's
    // grain-stats in its place: only one is installed, so the duplicate guard
    // is silent, and the old trust row still says `known`. Consent given for
    // one author's code would cover another's, unasked.
    addonsApi.list.mockResolvedValue(listing([{
      ...GOOD, known: true, known_version: '0.0.9',
      known_doi: '10.5281/zenodo.OTHER',
    }]));
    render(<AddonsPage />);
    fireEvent.click(await screen.findByRole('button', { name: /enable/i }));
    expect(await screen.findByRole('dialog')).toBeTruthy();
    expect(addonsApi.setEnabled).not.toHaveBeenCalled();
  });

  it('says on the row that it changed since consent', async () => {
    addonsApi.list.mockResolvedValue(listing([{
      ...GOOD, known: true, known_version: '0.0.9', known_doi: GOOD.doi,
    }]));
    render(<AddonsPage />);
    expect(await screen.findByText(/changed since you allowed it/i)).toBeTruthy();
    expect(screen.getByText(/0\.0\.9/)).toBeTruthy();
  });

  it('a trust file from before those fields does not re-ask', async () => {
    // Those files predate the check; turning every one of them into a fresh
    // prompt would teach the dialog to be clicked away.
    addonsApi.list.mockResolvedValue(listing([{
      ...GOOD, known: true, known_version: '', known_doi: '',
    }]));
    render(<AddonsPage />);
    fireEvent.click(await screen.findByRole('button', { name: /enable/i }));
    await waitFor(() =>
      expect(addonsApi.setEnabled).toHaveBeenCalledWith('bc-gmm', true));
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('opens with focus on CANCEL and closes on Escape', async () => {
    // aria-modal was declared and nothing honoured it: focus stayed on BODY
    // and Escape did nothing. Focus lands on cancel, not accept — a Return
    // pressed out of habit must not be the decision to run a stranger's code.
    render(<AddonsPage />);
    fireEvent.click(await screen.findByRole('button', { name: /enable/i }));
    const dialog = await screen.findByRole('dialog');
    await waitFor(() => expect(document.activeElement.textContent)
      .toMatch(/cancel/i));
    fireEvent.keyDown(document, { key: 'Escape' });
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(addonsApi.setEnabled).not.toHaveBeenCalled();
    expect(dialog).toBeTruthy();
  });

  it('does not ask again for an add-on already decided about', async () => {
    // `known` means a decision is on file. Asking every time would train
    // people to click it away.
    addonsApi.list.mockResolvedValue(listing([{ ...GOOD, known: true }]));
    render(<AddonsPage />);
    fireEvent.click(await screen.findByRole('button', { name: /enable/i }));
    await waitFor(() =>
      expect(addonsApi.setEnabled).toHaveBeenCalledWith('bc-gmm', true));
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('disabling never asks', async () => {
    addonsApi.list.mockResolvedValue(listing([
      { ...GOOD, enabled: true, known: true }]));
    render(<AddonsPage />);
    fireEvent.click(await screen.findByRole('button', { name: /disable/i }));
    await waitFor(() =>
      expect(addonsApi.setEnabled).toHaveBeenCalledWith('bc-gmm', false));
    expect(screen.queryByRole('dialog')).toBeNull();
  });
});

describe('failures', () => {
  it('shows the server’s sentence when a refusal comes back', async () => {
    addonsApi.list.mockResolvedValue(listing([{ ...GOOD, known: true }]));
    addonsApi.setEnabled.mockRejectedValue({
      response: { data: { reason: 'addon_import_failed',
                          detail: 'bc-gmm: ModuleNotFoundError: sklearn' } },
    });
    render(<AddonsPage />);
    fireEvent.click(await screen.findByRole('button', { name: /enable/i }));
    const alert = await screen.findByRole('alert');
    // The add-on's own words survive: they are the only text that says what
    // actually went wrong.
    expect(alert.textContent).toContain('ModuleNotFoundError: sklearn');
    // AND the code was turned into our sentence. Without this the reason is
    // simply unused — the thirteen codes exist for this one purpose, and a
    // German user would get the English server text as the headline too.
    expect(alert.textContent).toContain('This add-on could not be loaded');
  });

  it('a successful enable re-reads the list', async () => {
    // The dominant defect shape in this repo: the row would keep saying
    // "Enable", the "Enabled" badge would never appear, and the user would
    // click again. The happy path is the most-used one.
    addonsApi.list
      .mockResolvedValueOnce(listing([{ ...GOOD, known: true }]))
      .mockResolvedValueOnce(listing([{ ...GOOD, known: true, enabled: true }]));
    render(<AddonsPage />);
    fireEvent.click(await screen.findByRole('button', { name: /enable/i }));
    await waitFor(() => expect(addonsApi.list).toHaveBeenCalledTimes(2));
    expect(await screen.findByText(/^Enabled$/)).toBeTruthy();
    expect(screen.getByRole('button', { name: /disable/i })).toBeTruthy();
  });

  it('a backend that did not say where it looked says THAT, not nothing', async () => {
    // Reachable whenever a reloaded frontend meets a backend that was not
    // restarted — which is most of this project's working days. "Folders
    // searched:" above an empty list, then "put one in there", points at
    // nothing.
    addonsApi.list.mockResolvedValue(
      { data: { api_version: 0, orienta_version: '0.4.4', addons: [] } });
    render(<AddonsPage />);
    expect(await screen.findByText(/did not say which folders/i)).toBeTruthy();
    expect(screen.queryByText(/Folders searched/i)).toBeNull();
  });

  it('a bodyless failure does not print the same sentence twice', async () => {
    // The likeliest first failure of all: the backend is not running.
    addonsApi.list.mockRejectedValue(new Error('Network Error'));
    render(<AddonsPage />);
    const alert = await screen.findByRole('alert');
    const once = alert.textContent.indexOf('could not be loaded');
    expect(once).toBeGreaterThanOrEqual(0);
    expect(alert.textContent.indexOf('could not be loaded', once + 1)).toBe(-1);
  });

  it('says it is looking while the first listing is in flight', async () => {
    let resolve;
    addonsApi.list.mockReturnValue(new Promise((r) => { resolve = r; }));
    render(<AddonsPage />);
    expect(screen.getByRole('status').textContent).toMatch(/looking for add-ons/i);
    resolve(listing([GOOD]));
    await screen.findByText('Band contrast mixture model');
  });

  it('re-reads the list after a refusal, because the row has changed', async () => {
    // The runtime records a failed activation, so the row afterwards is not
    // the row before — a page that kept the old one would show a switch the
    // backend no longer agrees with.
    addonsApi.list.mockResolvedValue(listing([{ ...GOOD, known: true }]));
    addonsApi.setEnabled.mockRejectedValue({
      response: { data: { reason: 'addon_import_failed', detail: 'boom' } },
    });
    render(<AddonsPage />);
    fireEvent.click(await screen.findByRole('button', { name: /enable/i }));
    await waitFor(() => expect(addonsApi.list).toHaveBeenCalledTimes(2));
  });

  it('the message survives the re-read that follows the failure', async () => {
    // Found by this test failing: the re-read SUCCEEDS, so a load that
    // always cleared the message wiped the refusal milliseconds after the
    // user caused it — the alert never appeared at all.
    addonsApi.list.mockResolvedValue(listing([{ ...GOOD, known: true }]));
    addonsApi.setEnabled.mockRejectedValue({
      response: { data: { reason: 'addon_import_failed', detail: 'boom' } },
    });
    render(<AddonsPage />);
    fireEvent.click(await screen.findByRole('button', { name: /enable/i }));
    await waitFor(() => expect(addonsApi.list).toHaveBeenCalledTimes(2));
    expect(screen.getByRole('alert').textContent).toContain('boom');
  });

  it('an explicit rescan clears the message, because that asks for the state', async () => {
    addonsApi.list.mockResolvedValue(listing([{ ...GOOD, known: true }]));
    addonsApi.setEnabled.mockRejectedValue({
      response: { data: { reason: 'addon_import_failed', detail: 'boom' } },
    });
    render(<AddonsPage />);
    fireEvent.click(await screen.findByRole('button', { name: /enable/i }));
    await screen.findByRole('alert');
    fireEvent.click(screen.getByRole('button', { name: /rescan/i }));
    await waitFor(() => expect(screen.queryByRole('alert')).toBeNull());
  });

  it('a failed listing says so and does not render an empty state', async () => {
    // "Nothing installed" and "we could not ask" are different facts, and the
    // empty state would send the user to create folders for no reason.
    addonsApi.list.mockRejectedValue({
      response: { data: { detail: 'Connection refused' } } });
    render(<AddonsPage />);
    const alert = await screen.findByRole('alert');
    expect(alert.textContent).toContain('Connection refused');
    expect(screen.queryByTestId('addons-empty')).toBeNull();
  });
});
