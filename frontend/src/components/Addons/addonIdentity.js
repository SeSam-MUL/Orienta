/**
 * Is the add-on in front of us the one the recorded decision was about?
 *
 * `known` only says a row exists under this NAME, and a name is not an
 * identity. Delete `grain-stats` and drop a different `grain-stats` in its
 * place — a colleague's zip, a fork, a newer build — and the old row still
 * says known. Only one is installed, so the duplicate-source guard stays
 * silent, and consent given for one author's code would cover another's
 * without anyone being asked. That is the runner's module collision one level
 * up, and it is answered the same way: named, not guessed at.
 *
 * Its own module, imported by both the page and the list, because the page
 * imports the list — the two would otherwise import each other.
 */

/**
 * @param {{known?: boolean, version?: string, doi?: string,
 *          known_version?: string, known_doi?: string}} addon
 */
export function consentIsStale(addon) {
  if (!addon || !addon.known) return false;
  const changed = (recorded, current) =>
    Boolean(recorded) && String(recorded) !== String(current || '');
  return changed(addon.known_version, addon.version)
    || changed(addon.known_doi, addon.doi);
}

export default consentIsStale;
