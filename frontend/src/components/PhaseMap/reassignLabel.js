/**
 * What the Reassign button should say, given the last phase-check summary.
 *
 * The check has two stages and they repair different things: whole grains
 * stolen by a chemically degenerate phase (stage 1), and the one- to
 * four-pixel islands sitting INSIDE a grain that stage 1 skips because they
 * fall under MIN_GRAIN_PX (stage 2). The button applies both, so it has to
 * name both — a label that says "Reassign 3 grains" while it is also about to
 * repair 19 pixels is describing less than it does.
 *
 * Pure on purpose: the button's wording and its enabled-ness are the same
 * decision, and a test can hold them to it without a backend.
 */
export function reassignWork(info) {
  const grains = Number(info?.n_reassign) || 0;
  const islands = Number(info?.n_islands_reassign) || 0;
  return { grains, islands, has: grains > 0 || islands > 0 };
}

/** `{ key, params }` for i18n. Never returns a key for "nothing to do" — the
 *  button keeps its ordinary label and is simply disabled. */
export function reassignLabelKey(info) {
  const { grains, islands } = reassignWork(info);
  if (grains > 0 && islands > 0) {
    return { key: 'phasemap:phaseCheck.reassignButtonBoth', params: { n: grains, islands } };
  }
  if (islands > 0) {
    return { key: 'phasemap:phaseCheck.reassignButtonIslands', params: { islands } };
  }
  return { key: 'phasemap:phaseCheck.reassignButton', params: { n: grains } };
}
