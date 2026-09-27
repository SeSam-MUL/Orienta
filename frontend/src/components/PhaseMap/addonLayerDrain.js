/**
 * Reading an add-on layer id, and deciding what a hand-over does to the stack.
 *
 * `planAddonDrain`: what to add and what could not be added, decided BEFORE
 * anything is dispatched.
 *
 * That order is the whole point. `layerStackReducer`'s ADD returns the state
 * unchanged for a FULL stack and for a DUPLICATE, so a drain that inferred
 * "refused" from "the state did not change" would report every duplicate as
 * "stack full" — and in development StrictMode runs the drain effect twice,
 * so the second, deduplicated pass would say the stack is full on every run.
 *
 * A duplicate is not a refusal: the layer the user asked for is on the stack,
 * which is what they wanted.
 */
/**
 * The four parts of an add-on layer id, or null if this is not one.
 *
 * `addon:<name>/<result_id>/<analysis_key>/<key>#<run_token>`
 *
 * The RESULT is in the id because /api/phasemap/layer takes no result id —
 * every other layer in the stack is implicitly the active result, while an
 * add-on map is bound to one explicit result. Without it this function could
 * not build the URL at all: it is module level and has no result in scope.
 *
 * The TOKEN is minted per successful run and stripped here. Without it a
 * second run produces the same id, the reducer dedupes it and the bitmap
 * cache hits, so "show as layer" after re-running is a no-op and the user
 * goes on reading the FIRST run's numbers under a label that says otherwise.
 * Stripping is written as `split('#')[0]`, literally and in one place: a
 * forgotten strip is invisible, because `#` starts a URL fragment — the
 * request still reaches the right path and every test stays green.
 */
export function parseAddonLayerId(id) {
  if (typeof id !== 'string' || !id.startsWith('addon:')) return null;
  const [withoutToken] = id.split('#');
  const parts = withoutToken.slice('addon:'.length).split('/');
  if (parts.length !== 4 || parts.some((p) => !p)) return null;
  const [name, resultId, analysisKey, key] = parts;
  return { name, resultId, analysisKey, key };
}

/** Everything of an add-on layer id except the run token. */
export function baseOf(id) {
  return typeof id === 'string' ? id.split('#')[0] : id;
}

export function planAddonDrain(pending, layers, max) {
  const current = layers || [];
  const present = new Set(current.map((l) => l.id));
  const toAdd = [];
  const refused = [];
  // A NEWER run of the same analysis replaces its own earlier layer rather
  // than stacking beside it. Measured in the acceptance run: two rows with
  // the identical author label, the lower one drawing the previous run's map,
  // and nothing on screen to tell them apart — the "two claims about one
  // dataset" defect in the place it does the most damage, a figure. The ids
  // differ only in the run token, so the base id is what identifies them.
  const toRemove = [];
  for (const req of pending || []) {
    if (present.has(req.id)) continue;
    for (const l of current) {
      if (l.id !== req.id && baseOf(l.id) === baseOf(req.id)
          && !toRemove.includes(l.id)) {
        toRemove.push(l.id);
      }
    }
  }
  // Room counted AFTER the replacements: a re-run must not be refused for a
  // full stack when it is taking its own predecessor's place.
  let room = max - current.length + toRemove.length;
  for (const req of pending || []) {
    if (present.has(req.id)) continue;
    if (room <= 0) { refused.push(req); continue; }
    room -= 1;
    present.add(req.id);
    toAdd.push(req);
  }
  return { toAdd, toRemove, refused };
}

export default planAddonDrain;
